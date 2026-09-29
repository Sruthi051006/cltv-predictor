"""clv_pipeline.py - shared code for the notebook AND the Streamlit app.
Raw customer rows go in -> cleaning -> outlier capping -> feature engineering -> encoding/scaling -> XGBoost."""
import re
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
TARGET, ID_COL = "cltv", "id"
NUM_RAW = ["vintage", "claim_amount"]
CAT_RAW = ["gender", "area", "qualification", "income", "marital_status", "num_policies", "policy", "type_of_policy"]
INCOME_ORDER = {"<=2l": 0.0, "2l-5l": 1.0, "5l-10l": 2.0, "more than 10l": 3.0}
IQR_CAP = 3.0
SEGMENT_QUANTILES = (0.33, 0.66)
SEGMENT_LABELS = ["Low", "Medium", "High"]


def _norm_cols(df):
    d = df.copy()
    d.columns = [re.sub(r"\W+", "_", str(c).strip().lower()).strip("_") for c in d.columns]
    return d


class Cleaner(BaseEstimator, TransformerMixin):
    """Fixes data types and inconsistent text, blocks negative values, adds any missing column as NaN,
    ignores id / extra columns.  Works on raw jury data without manual editing."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = _norm_cols(X)
        out = pd.DataFrame(index=X.index)
        for c in NUM_RAW:
            s = pd.to_numeric(X[c], errors="coerce") if c in X else pd.Series(np.nan, index=X.index)
            out[c] = s.where(s >= 0).astype(float)
        for c in CAT_RAW:
            if c not in X:
                out[c] = pd.Series(np.nan, index=X.index, dtype=object)
                continue
            if c == "marital_status":
                n = pd.to_numeric(X[c], errors="coerce")
                s = n.map(lambda v: str(int(v)) if pd.notna(v) else np.nan)
                txt = X[c].astype(str).str.strip().str.lower().map({"married": "1", "yes": "1", "single": "0", "no": "0"})
                s = s.where(s.notna(), txt)
            else:
                s = X[c].astype(str).str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
                s = s.where(~s.isin(["nan", "none", "null", "", "na", "n/a"]), np.nan)
                if c == "gender":
                    s = s.replace({"m": "male", "f": "female"})
            out[c] = s.astype(object)
        return out


class Winsorizer(BaseEstimator, TransformerMixin):
    """Caps EXTREME numeric values (beyond 3 x IQR). Bounds are learned on training data only."""

    def __init__(self, factor=IQR_CAP):
        self.factor = factor

    def fit(self, X, y=None):
        self.bounds_ = {}
        for c in NUM_RAW:
            q1, q3 = X[c].quantile([.25, .75]); i = q3 - q1
            if i > 0:
                self.bounds_[c] = (max(q1 - self.factor * i, 0.0), q3 + self.factor * i)
        return self

    def transform(self, X):
        X = X.copy()
        for c, (lo, hi) in self.bounds_.items():
            X[c] = X[c].clip(lo, hi)
        return X


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Domain features for CLTV (see notebook Round 2, Task 2)."""

    def __init__(self, drop_cols=()):
        self.drop_cols = drop_cols

    def fit(self, X, y=None):
        self.claim_q75_ = float(X["claim_amount"].quantile(.75))
        return self

    def transform(self, X):
        X = X.copy()
        isna = X["num_policies"].isna()
        multi = pd.Series(np.where(isna, np.nan, np.where(X["num_policies"] == "1", 0.0, 1.0)), index=X.index)
        urban = pd.Series(np.where(X["area"].isna(), np.nan, (X["area"] == "urban") * 1.0), index=X.index)
        X["multi_policy"] = multi                                           # frequency proxy: holds more than one policy
        X["income_ord"] = X["income"].map(INCOME_ORDER).astype(float)       # ordinal income level 0-3
        X["claim_per_year"] = X["claim_amount"] / (X["vintage"] + 1)        # monetary value per year of tenure
        X["claim_x_multi"] = X["claim_amount"] * multi                      # claims of multi-policy holders
        X["urban_multi"] = urban * multi                                    # urban AND multi-policy
        X["income_x_multi"] = X["income_ord"] * multi
        X["high_claim"] = np.where(X["claim_amount"].isna(), np.nan, (X["claim_amount"] > self.claim_q75_) * 1.0)
        X = X.drop(columns=["income", "num_policies"] + [c for c in self.drop_cols if c in X.columns])
        return X.replace([np.inf, -np.inf], np.nan)


def build_pipeline(model=None, drop_cols=(), **xgb_params):
    prep = ColumnTransformer(
        [("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]),
          make_column_selector(dtype_include=np.number)),
         ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                           ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
          make_column_selector(dtype_exclude=np.number))],
        sparse_threshold=0)
    if model is None:
        params = dict(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                      min_child_weight=5, random_state=RANDOM_STATE, n_jobs=2, tree_method="hist",
                      objective="reg:squarederror")
        params.update(xgb_params)
        model = xgb.XGBRegressor(**params)
    return Pipeline([("cleaner", Cleaner()), ("winsor", Winsorizer()), ("fe", FeatureEngineer(drop_cols=tuple(drop_cols))),
                     ("prep", prep), ("model", model)])


# ------------------------------------------------------------------ prediction / segment / explanation
def assign_segment(values, thresholds):
    t1, t2 = thresholds
    v = np.asarray(values)
    return np.where(v < t1, SEGMENT_LABELS[0], np.where(v < t2, SEGMENT_LABELS[1], SEGMENT_LABELS[2]))


def predict_df(bundle, df):
    """raw dataframe -> predicted CLTV, value segment and 80% confidence band."""
    pred = np.clip(bundle["pipeline"].predict(df), 0, None)
    out = pd.DataFrame(index=df.index)
    out["predicted_cltv"] = pred
    out["value_segment"] = assign_segment(pred, bundle["thresholds"])
    out["cltv_low_10pct"] = pred * bundle["ratio_q10"]
    out["cltv_high_90pct"] = pred * bundle["ratio_q90"]
    return out


def feature_groups(pipe):
    """maps every model input column (after one-hot) back to its original feature name."""
    prep = pipe.named_steps["prep"]
    names = [n.split("__", 1)[1] for n in prep.get_feature_names_out()]
    cat_cols = sorted(list(prep.transformers_[1][2]), key=len, reverse=True)
    def group(n):
        for c in cat_cols:
            if n.startswith(c + "_"):
                return c
        return n
    return names, [group(n) for n in names]


def explain(pipe, X_raw):
    """Exact TreeSHAP values from XGBoost (pred_contribs).  contributions.sum(axis=1) + base = prediction.
    Returns (contributions per ORIGINAL feature, base value, feature values shown to the user)."""
    X_t = np.asarray(pipe[:-1].transform(X_raw), dtype=float)
    contrib = pipe[-1].get_booster().predict(xgb.DMatrix(X_t), pred_contribs=True)
    names, groups = feature_groups(pipe)
    df = pd.DataFrame(contrib[:, :-1], columns=names, index=X_raw.index)
    df = df.T.groupby(groups, sort=False).sum().T
    return df, float(contrib[0, -1]), pipe[:3].transform(X_raw)


def recommend(segment, row):
    multi = row.get("multi_policy")
    if segment == "High":
        if multi == 1:
            return ("HIGH-value, multi-policy customer -> PROTECT: loyalty / renewal benefits, dedicated relationship "
                    "manager, early-renewal discounts.")
        return "HIGH-value but single-policy -> LOCK IN: cross-sell a second policy with a bundle discount."
    if segment == "Medium":
        if multi == 1:
            return "MEDIUM-value, multi-policy -> GROW: upgrade to a higher-value policy and increase cover."
        return "MEDIUM-value, single-policy -> GROW: cross-sell a second policy (largest CLTV uplift in the data)."
    if multi == 1:
        return "LOW-value despite multiple policies -> keep cost-to-serve low: automated engagement, review pricing."
    return "LOW-value, single-policy -> low-cost digital nurturing; offer a bundled second policy, avoid expensive retention spend."

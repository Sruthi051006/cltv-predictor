"""app.py - Round 3 Streamlit prototype.   Run:  streamlit run app.py
raw customer data -> clv_pipeline (clean, cap outliers, features, encode, scale) -> XGBoost -> prediction, segment, SHAP factors"""
import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from clv_pipeline import SEGMENT_LABELS, TARGET, _norm_cols, explain, predict_df, recommend

st.set_page_config(page_title="CLTV Predictor", page_icon="💰", layout="wide")
BUNDLE = "models/clv_bundle.joblib"
COLORS = {"Low": "#e45756", "Medium": "#f5b041", "High": "#2ca02c"}
if not os.path.exists(BUNDLE):
    st.error("Model not found - run the Round 2 cells of the notebook first (creates models/clv_bundle.joblib)."); st.stop()


@st.cache_resource
def load():
    return joblib.load(BUNDLE)


B = load(); pipe, M, thr, schema = B["pipeline"], B["metrics"], B["thresholds"], B["schema"]
st.title("💰 Customer Lifetime Value - Decision Support")
st.caption("XGBoost model + complete preprocessing pipeline  |  explainable predictions (SHAP)")


def factor_chart(contrib_row, values_row, base, pred, top=8):
    s = contrib_row.sort_values(key=abs, ascending=False).head(top).iloc[::-1]
    labels = []
    for f in s.index:
        v = values_row.get(f, "")
        labels.append(f"{f} = {v:,.2f}" if isinstance(v, (int, float, np.floating)) else f"{f} = {v}")
    fig = go.Figure(go.Bar(x=s.values, y=labels, orientation="h", marker_color=["#2ca02c" if v > 0 else "#e45756" for v in s.values],
                           text=[f"{v:+,.0f}" for v in s.values], textposition="outside"))
    fig.update_layout(title=f"Why this prediction?  average customer ≈ {base:,.0f}  →  this customer {pred:,.0f}",
                      xaxis_title="Impact on predicted CLTV (green = increases, red = decreases)", height=420, margin=dict(l=10, r=10, t=60, b=10))
    return fig


def gauge(pred):
    top = max(thr[1] * 1.8, pred * 1.2)
    fig = go.Figure(go.Indicator(mode="gauge+number", value=pred, number={"valueformat": ",.0f"},
                    gauge={"axis": {"range": [0, top]}, "bar": {"color": "#1f4e79"},
                           "steps": [{"range": [0, thr[0]], "color": "#f5b7b1"}, {"range": [thr[0], thr[1]], "color": "#fdebd0"},
                                     {"range": [thr[1], top], "color": "#abebc6"}]}))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=10)); return fig


def show_customer(one):
    res = predict_df(B, one).iloc[0]
    contrib, base, vals = explain(pipe, one); seg = res["value_segment"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted CLTV", f"{res['predicted_cltv']:,.0f}")
    c2.markdown(f"**Value segment**<br><span style='background:{COLORS[seg]};color:white;padding:6px 18px;border-radius:14px;font-size:22px'>{seg}</span>", unsafe_allow_html=True)
    c3.metric("80% confidence band", f"{res['cltv_low_10pct']:,.0f} - {res['cltv_high_90pct']:,.0f}")
    c4.metric("Model R² / MAE (test)", f"{M['R2']:.2f} / {M['MAE']:,.0f}")
    a, b = st.columns([1, 2])
    a.plotly_chart(gauge(res["predicted_cltv"]), use_container_width=True)
    b.plotly_chart(factor_chart(contrib.iloc[0], vals.iloc[0], base, res["predicted_cltv"]), use_container_width=True)
    st.success("**Recommendation:** " + recommend(seg, vals.iloc[0].to_dict()))
    top3 = contrib.iloc[0].sort_values(key=abs, ascending=False).head(3)
    st.info("**Key drivers:** " + "; ".join(f"{f} ({'↑' if v > 0 else '↓'} {abs(v):,.0f})" for f, v in top3.items()))


tab1, tab2, tab3 = st.tabs(["🔮 Single customer", "📂 Bulk / jury data", "📊 Model evidence"])

with tab1:
    st.subheader("Enter customer details")
    cols = st.columns(4); inp = {}
    for i, (c, r) in enumerate(schema["numeric"].items()):
        with cols[i % 4]:
            inp[c] = st.number_input(c, min_value=0.0, value=float(round(r["median"], 1)), step=1.0)
    for i, (c, opts) in enumerate(schema["categorical"].items()):
        with cols[(len(schema["numeric"]) + i) % 4]:
            inp[c] = st.selectbox(c, opts)
    if st.button("Predict CLTV", type="primary"):
        show_customer(pd.DataFrame([inp]))

with tab2:
    st.subheader("Upload unseen customers (CSV) - raw file, no editing needed")
    up = st.file_uploader("CSV file", type="csv")
    df_in = pd.read_csv(up) if up is not None else None
    if df_in is None:
        cand = next((p for p in ["test_koRSKBP.csv", "data/test_koRSKBP.csv", "data/sample_unseen.csv"] if os.path.exists(p)), None)
        if cand and st.checkbox(f"...or use the bundled unseen file ({cand})"):
            df_in = pd.read_csv(cand)
    if df_in is not None:
        try:
            out = predict_df(B, df_in)
        except Exception as e:
            st.error(f"Could not process the file: {e}"); st.stop()
        norm = _norm_cols(df_in)
        res = pd.concat([norm[["id"]] if "id" in norm else pd.DataFrame(index=df_in.index), out.round(1)], axis=1)
        st.dataframe(res, use_container_width=True, height=300)
        st.download_button("⬇ Download predictions", res.to_csv(index=False), "cltv_predictions.csv", "text/csv")
        a, b = st.columns(2)
        cnt = out["value_segment"].value_counts().reindex(SEGMENT_LABELS).fillna(0).reset_index(); cnt.columns = ["segment", "customers"]
        a.plotly_chart(px.pie(cnt, names="segment", values="customers", color="segment", color_discrete_map=COLORS, title="Segment mix"), use_container_width=True)
        b.plotly_chart(px.histogram(out, x="predicted_cltv", nbins=40, color="value_segment", color_discrete_map=COLORS, title="Predicted CLTV distribution"), use_container_width=True)
        if TARGET in norm.columns:
            yt = pd.to_numeric(norm[TARGET], errors="coerce"); ok = yt.notna()
            st.markdown("#### Evaluation on this file (actual CLTV present)")
            m1, m2, m3 = st.columns(3)
            m1.metric("MAE", f"{mean_absolute_error(yt[ok], out['predicted_cltv'][ok]):,.0f}")
            m2.metric("RMSE", f"{np.sqrt(mean_squared_error(yt[ok], out['predicted_cltv'][ok])):,.0f}")
            m3.metric("R²", f"{r2_score(yt[ok], out['predicted_cltv'][ok]):.3f}")
        st.markdown("#### Explain one customer")
        pos = st.number_input("Row number (0 = first row)", 0, len(df_in) - 1, 0)
        show_customer(df_in.iloc[[int(pos)]])

with tab3:
    st.subheader("Held-out test performance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE", f"{M['MAE']:,.0f}"); c2.metric("RMSE", f"{M['RMSE']:,.0f}"); c3.metric("R²", f"{M['R2']:.3f}")
    c4.metric("Segment accuracy", f"{M.get('segment_accuracy', 0):.1%}")
    st.caption(f"Baseline (always predict the mean): MAE {M['baseline_MAE']:,.0f}, R² 0.00  |  Segments: Low < {thr[0]:,.0f} ≤ Medium < {thr[1]:,.0f} ≤ High  |  "
               f"80% band = prediction × [{B['ratio_q10']:.2f}, {B['ratio_q90']:.2f}]")
    te = B["test_eval"]
    a, b = st.columns(2)
    f1 = px.scatter(te.sample(min(len(te), 4000), random_state=1), x="actual", y="pred", color="segment", color_discrete_map=COLORS, opacity=.5, title="Actual vs predicted (test set)")
    lim = float(max(te.actual.max(), te.pred.max())); f1.add_shape(type="line", x0=0, y0=0, x1=lim, y1=lim, line=dict(dash="dash", color="black"))
    a.plotly_chart(f1, use_container_width=True)
    imp = pd.Series(B["global_importance"]).head(12).iloc[::-1]
    b.plotly_chart(px.bar(x=imp.values, y=imp.index, orientation="h", title="Most important factors (mean |SHAP|)", labels={"x": "impact", "y": ""}), use_container_width=True)

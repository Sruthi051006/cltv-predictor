# Customer Lifetime Value Predictor (Hackathon Challenge 50)

Explainable CLTV prediction for insurance customers: **XGBoost** + a full preprocessing pipeline + **SHAP** factors, with a **Streamlit** web app.

## What is inside
| Path | Purpose |
|---|---|
| `app.py` | Streamlit web app (single customer, bulk CSV upload, model evidence) |
| `clv_pipeline.py` | Cleaning, IQR outlier capping, feature engineering, encoding/scaling, XGBoost pipeline, SHAP helpers |
| `models/` | Trained pipeline (`clv_pipeline.pkl`) and bundle with metrics / thresholds (`clv_bundle.joblib`) |
| `data/sample_unseen.csv` | 300 unseen customers to try the bulk-upload tab |
| `docs/` | Static project page for GitHub Pages (`index.html`, `metrics.json`, `importance.json`) |
| `requirements.txt` | Pinned library versions (must match the versions used to train the model) |

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Live links
* Live app (Streamlit Community Cloud): (add after deploying)
* Project page (GitHub Pages): (after enabling GitHub Pages)

## Deploy (free)
1. **Streamlit app** - go to https://share.streamlit.io -> *New app* -> pick this repository, branch `main`, main file `app.py`.
   In *Advanced settings* choose the same Python version you used for training (see the note in `requirements.txt`).
2. **Project page** - GitHub repo -> *Settings* -> *Pages* -> Source: *Deploy from a branch* -> `main` / `/docs`.
3. Put the Streamlit URL into `APP_URL` at the top of `docs/index.html` and push again.

## How the model was built
Round 1 EDA -> Round 2 feature engineering + XGBoost (tuned with RandomizedSearchCV, evaluated with MAE / RMSE / R-squared) -> Round 3 prototype.
The training notebook is `Challenge50_CLTV_Final.ipynb`.

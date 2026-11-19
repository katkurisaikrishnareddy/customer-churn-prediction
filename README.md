# Customer Churn Prediction — end-to-end ML system

Predict which telecom customers are about to leave, **decide who to contact using retention economics**, and serve the result through a web app, a JSON API and batch scoring.

> **v2.0** — extends the original pipeline with leak-free preprocessing, feature engineering, imbalance study, calibration, cost-based thresholding, SHAP explainability, customer segmentation, fairness audit, tests, CI and a hardened Flask app.

---

## Headline results (held-out test set, 1409 customers, 26.5% churn)

| | v1 (original) | **v2 @ default 0.50** | **v2 @ cost-optimal 0.24** |
|---|---|---|---|
| ROC-AUC | 0.846 | 0.8445 | 0.8445 |
| Recall (churners caught) | 0.516 | 0.5214 | **0.8209** |
| Precision | 0.673 | 0.6655 | 0.5 |
| F1 | 0.584 | 0.5847 | **0.6215** |
| Accuracy | 0.805 | 0.8034 | 0.7346 |

* 95% bootstrap CIs at the optimal threshold — recall 0.7798–0.8579, precision 0.4603–0.5388, ROC-AUC 0.8216–0.8653.
* **Honest note:** ranking quality (ROC-AUC ≈ 0.85) is the ceiling of this dataset — it did not change. The improvement is in *decisions*: the cost-based threshold catches **82% of churners instead of ~52%**, at the price of lower accuracy (73.5%) because more loyal customers receive an offer. That is the intended trade-off when a missed churner costs far more than a wasted offer.
* **Business value** (assumptions in `config.yaml`: $50 offer, 30% success, 12 months revenue): net benefit on the test set is **$51,803** at the optimal threshold vs $40,322 at 0.50 (+28%) and $27,524 for contacting everyone.
* The v1 README advertised ~83% accuracy / F1 ≈ 0.81; the v1 code actually produced 80.5% / 0.58. All numbers here are generated from `artifacts/evaluation_report.json`.

## What changed vs. v1

| Area | v1 | v2 |
|---|---|---|
| Leakage | scaler fitted on **all** data before the split | split first; scaler/encoder inside a `Pipeline` (regression test included) |
| `TotalCharges` | blanks (new customers) filled with the median | = $0 for tenure 0 (logically correct) |
| Encoding | LabelEncoder on nominal features | One-hot encoding inside the pipeline, unknown categories handled |
| Validation | columns / nulls / dtype | + allowed categories, ranges, hidden blanks, duplicate IDs; JSON quality report |
| Features | raw columns only | business features (add-ons, auto-pay, tenure bucket, price change …); `gender` removed (ablation shows no signal) |
| Models | LR, RF, GB (ROC-AUC) | LR, RF, HistGB, XGBoost on PR-AUC with **one-standard-error** selection |
| Imbalance | not addressed | none vs class-weight vs SMOTE ablation |
| Probabilities | uncalibrated | calibration tested (Brier), reliability curve, ECE |
| Threshold | fixed 0.5 | **cost-based** from out-of-fold predictions + sensitivity analysis |
| Evaluation | single split, point estimates | bootstrap CIs, lift/gains, permutation importance, SHAP, fairness audit |
| Business layer | none | ROI, risk tiers, segmentation, at-risk customer list, retention playbook |
| App | form + basic API, silent fallbacks | strict validation, per-customer explanations, batch CSV, health & schema endpoints |
| Quality | none | pytest suite, GitHub Actions CI, pinned Docker, model card |
| `top_features` | always empty (HistGB has no `feature_importances_`) | permutation importance + SHAP |

## Project structure

```
├── run_pipeline.py              # orchestrates stages 1-8  (--fast for a quick run)
├── app.py                       # Flask app: /, /batch, /metrics, /about, /api/predict, /api/schema, /health
├── config.yaml                  # every tunable number (grids, business assumptions, thresholds)
├── src/
│   ├── data_ingestion.py        # load local CSV (download fallback)
│   ├── data_validation.py       # automated data-quality gate -> reports/data_quality_report.json
│   ├── feature_engineering.py   # stateless cleaning + business features (inside the Pipeline)
│   ├── data_preprocessing.py    # split FIRST, build ColumnTransformer
│   ├── model_trainer.py         # tuning, selection rule, imbalance ablation, calibration
│   ├── threshold_optimizer.py   # cost-based threshold + sensitivity
│   ├── model_evaluator.py       # test metrics, CIs, plots, permutation importance, fairness
│   ├── explainability.py        # global SHAP
│   ├── business_impact.py       # segmentation, risk tiers, at-risk list
│   ├── eda.py                   # EDA figures + insights
│   └── predictor.py             # shared inference + validation + explanations + actions
├── artifacts/                   # churn_model.joblib, model_metadata.json, evaluation_report.json, reference_sample.csv
├── reports/                     # figures/, data_quality_report.json, at_risk_customers.csv, business_impact.json ...
├── notebooks/01_end_to_end_churn_analysis.ipynb   # executed notebook with outputs
├── tests/                       # pytest suite
├── docs/                        # project report materials
├── templates/ static/           # web UI
├── Dockerfile  Makefile  requirements*.txt  .github/workflows/ci.yml
└── MODEL_CARD.md
```

## Quick start

```bash
python -m venv venv && source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements-train.txt                  # Python 3.11+ recommended
python run_pipeline.py --fast                          # ~2 min; use plain `python run_pipeline.py` for the full grids
python -m pytest -q                                    # run the tests
python app.py                                          # open http://localhost:5000
```

> The shipped `artifacts/` were trained with the versions in `requirements-lock.txt`. If your scikit-learn version differs, run `python run_pipeline.py` once to regenerate artifacts for your environment (pickled models are version-sensitive).

### Docker
```bash
docker build -t churn-prediction .
docker run -p 5000:5000 churn-prediction
```
No training or internet access is needed at build time — the trained model ships in the image.

### API
```bash
curl -X POST localhost:5000/api/predict -H "Content-Type: application/json" -d '{
 "SeniorCitizen":0,"Partner":"No","Dependents":"No","tenure":3,"PhoneService":"Yes","MultipleLines":"No",
 "InternetService":"Fiber optic","OnlineSecurity":"No","OnlineBackup":"No","DeviceProtection":"No","TechSupport":"No",
 "StreamingTV":"Yes","StreamingMovies":"Yes","Contract":"Month-to-month","PaperlessBilling":"Yes",
 "PaymentMethod":"Electronic check","MonthlyCharges":95.7}'
```
Returns the calibrated probability, risk tier, top risk-raising / protective factors and suggested retention actions. Invalid input returns HTTP 400 with every problem listed. `GET /api/schema` lists accepted fields and values.

## Method in one page

1. **Validate** the raw data, then **split** (stratified 80/20) before anything is fitted.
2. **Engineer** features inside the Pipeline so training and serving use identical code.
3. **Compare** four tuned models by 5-fold CV PR-AUC and pick the simplest within one standard error of the best.

| Model | CV PR-AUC | CV ROC-AUC |
|---|---|---|
| LogisticRegression ★ | 0.6627 ± 0.0149 | 0.8475 |
| RandomForest | 0.6629 ± 0.0246 | 0.8470 |
| HistGradientBoosting | 0.6646 ± 0.0205 | 0.8483 |
| XGBoost | 0.6667 ± 0.0204 | 0.8491 |

4. **Test imbalance fixes** (class weights, SMOTE): recall at 0.5 improves but PR-AUC does not — they act like a threshold shift, so we keep natural probabilities.
5. **Calibrate** if it lowers the Brier score (selected model: calibration not needed; Brier 0.1342 raw vs 0.13422 calibrated).
6. **Choose the threshold from economics**: theoretical 0.214, empirical **0.24** (F1-optimal 0.32, F2-optimal 0.14). The optimum depends on the assumptions — in a 16-scenario sensitivity grid it stays at or below 0.5 in 13 scenarios and rises above 0.5 only when offers are expensive *and* rarely work.
7. **Evaluate once** on the held-out set; **explain** with permutation importance and SHAP; **segment** customers; **audit** fairness.

### Customer segments (K-means, k = 4)

| Segment | Customers | Churn | Avg tenure (mo) | Avg monthly | Avg add-ons | Month-to-month |
|---|---|---|---|---|---|---|
| S1 | 2373 | 47.4% | 15 | $78 | 1.8 | 87% |
| S2 | 1521 | 24.2% | 10 | $30 | 0.3 | 72% |
| S3 | 2158 | 15.4% | 57 | $91 | 4.2 | 24% |
| S4 | 991 | 4.4% | 55 | $32 | 0.6 | 17% |

### Fairness audit (test set)

| Attribute | Group | n | Actual churn | Flag rate | Recall | ROC-AUC |
|---|---|---|---|---|---|---|
| gender | Female | 687 | 0.281 | 0.44 | 0.819 | 0.841 |
| gender | Male | 722 | 0.251 | 0.432 | 0.823 | 0.849 |
| SeniorCitizen | 0 | 1187 | 0.233 | 0.386 | 0.783 | 0.847 |
| SeniorCitizen | 1 | 222 | 0.441 | 0.703 | 0.929 | 0.776 |
| Partner | No | 736 | 0.337 | 0.55 | 0.859 | 0.821 |
| Partner | Yes | 673 | 0.187 | 0.311 | 0.746 | 0.854 |
| Dependents | No | 978 | 0.31 | 0.522 | 0.855 | 0.83 |
| Dependents | Yes | 431 | 0.165 | 0.239 | 0.676 | 0.854 |

Senior citizens are flagged much more often and ranked less accurately — see `MODEL_CARD.md` for the mitigation guidance.

## Limitations
* One public snapshot dataset (IBM Telco). Retrain and re-validate on current company data before real use.
* Business numbers are **illustrative assumptions** — replace them in `config.yaml`.
* Associations, not causes: confirm retention offers with an A/B test; consider uplift modelling.
* No drift monitoring is implemented (listed as future work).

## Credits
Dataset: IBM *Telco Customer Churn* sample dataset (IBM Developer / telco-customer-churn-on-icp4d). The v1 baseline pipeline structure (ingestion → validation → preprocessing → training → evaluation → Flask app) is retained and extended.
*If your institution requires it, add your name, roll number, guide and — if you started from someone else's public repository — an explicit attribution line here.*

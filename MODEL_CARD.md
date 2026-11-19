# Model Card — Customer Churn Predictor v2.0

## Model details
* **Task:** binary classification — probability that a telecom customer churns (leaves) in the next period.
* **Final model:** LogisticRegression (params {'C': 0.03}), inside a scikit-learn Pipeline: feature engineering → scaling / one-hot encoding → classifier. Calibration not needed (Brier raw 0.1342 vs calibrated 0.13422).
* **Why this model:** one-standard-error rule: best was XGBoost (0.6667); floor = best - 1 SE = 0.6576; candidates within the floor = ['LogisticRegression', 'RandomForest', 'HistGradientBoosting', 'XGBoost']; simplest of them = LogisticRegression
* **Decision rule:** flag for retention when calibrated probability ≥ **0.24** (chosen from business cost on out-of-fold training predictions). Risk tiers: Low < 0.24 ≤ Medium < 0.50 ≤ High.
* **Training data:** IBM Telco Customer Churn, 7,043 customers; 80% train / 20% stratified test, `random_state=42`.

## Intended use
Decision support for a retention team: rank customers by risk, choose whom to contact, and prioritise talking points. **Not** for automated adverse decisions (e.g. denying service or credit), and not for individuals' personal data decisions without human review.

## Performance (held-out test set, n=1409)
| Metric | @ 0.50 | @ 0.24 (95% CI) |
|---|---|---|
| ROC-AUC | 0.8445 | 0.8445 (0.8216–0.8653) |
| PR-AUC | 0.6498 | 0.6498 (0.5964–0.7004) |
| Recall | 0.5214 | 0.8209 (0.7798–0.8579) |
| Precision | 0.6655 | 0.5 (0.4603–0.5388) |
| F1 | 0.5847 | 0.6215 (0.5847–0.6554) |
| Accuracy | 0.8034 | 0.7346 |
| Brier / ECE | 0.1366 / 0.0252 | same (threshold-free) |

Confusion matrix @ 0.24: TN 728, FP 307, FN 67, TP 307.

## Business assumptions (illustrative — replace with real figures)
Offer cost $50 per contacted customer; 30% of would-be churners are retained; a saved customer keeps 12 months of revenue. Net benefit on the test set: $51,803 (optimal) vs $40,322 (0.50) vs $27,524 (contact everyone).

## Fairness & ethics
* `gender` is **excluded** from the model (no signal, protected attribute).
* `SeniorCitizen` **is** used (it is predictive) and is audited below. Seniors are flagged far more than other customers and the model ranks them less accurately; offers should not be allocated by age without review, and the app marks age/household factors as non-actionable.

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

## Limitations
* Single dataset snapshot, no time dimension → cannot measure drift or forecast *when* churn happens.
* Correlation, not causation: a high-risk factor is not proof that changing it will retain the customer.
* Probabilities are calibrated on this population; recalibrate for a different market.
* Model is near this dataset's performance ceiling (≈0.85 ROC-AUC); more complex models were statistically tied.
* Pickled model files depend on the scikit-learn version (see `model_metadata.json`).

## Maintenance
Retrain with `python run_pipeline.py`; tests with `python -m pytest`. Monitor input distributions and recalibrate quarterly; track real campaign results to replace the assumed offer success rate.

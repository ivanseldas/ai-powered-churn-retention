# Telecom Churn Predictor

End‑to‑end workflow to identify and prioritise customers at risk of churn and estimate the business uplift of a targeted retention campaign.

---

## 1. Objectives

- Predict churn probability for each customer (binary target `Churn`).
- Surface key behavioural drivers of churn (usage + plan indicators).
- Rank customers to target a fixed-size retention campaign (top 500 by risk).
- Quantify incremental impact vs. a random selection benchmark.

---

## 2. Dataset

~5K customer records with usage metrics and plan flags.

Feature groups:
- Categorical (low cardinality): `international_plan`, `voice_mail_plan` (and `state`, held out of initial modeling).
- Numerical: call counts, minutes and associated charges (day/evening/night/international), service interactions (`number_customer_service_calls`), message counts.
- Target: `Churn` (0 = retained, 1 = churned).

Data quality: no missing values. One non-predictive identifier (`phone_number`) removed. Whitespace in plan flags normalised.

---

## 3. Methodology (Current Notebook Version)

### Preprocessing
- Separate numeric vs. categorical features.
- `StandardScaler` on numeric features; `OneHotEncoder(handle_unknown='ignore')` on categorical (excluding `state`).
- Combined with a `ColumnTransformer` to produce the model matrix.

### Modeling
Baseline models trained on an 80/20 stratified split:
1. Logistic Regression
2. Random Forest
3. XGBoost (selected for refinement)

### Hyperparameter Tuning
`GridSearchCV` (ROC-AUC scoring) over a compact grid of depth, learning rate, estimators, subsample and column sampling.

### Evaluation Metrics
Reported: Accuracy, Precision, Recall, F1, ROC-AUC. ROC curves overlaid for baseline comparison. Selection emphasised ROC-AUC + balanced Recall / Precision trade‑off.

### Feature Importance
Tree-based gain importances extracted pre- and post-tuning to assess stability of drivers.

### Risk Ranking (Top 500)
Out-of-fold (5-fold) predicted probabilities generated with the tuned XGBoost (fresh model each fold to avoid leakage). Customers ranked by cross‑validated probability; top 500 flagged as high-risk segment.

### Uplift Benchmark
Monte Carlo (1,000 simulations) random sampling of 500 customers established an average expected churner count baseline. Assumed 30% save (retention) effectiveness for contacted churners. Compared: targeted vs. random to estimate incremental retained customers and relative improvement (≈5–6× uplift in illustrative run).

---

## 4. Key Findings

- Tuned XGBoost offered marginal but meaningful improvements in Precision and ROC-AUC vs. untuned baseline (exact values reproducible by re-running the notebook).
- Top churn drivers (consistent across initial & tuned models):
   1. `international_plan` (customers on the plan churn more).
   2. `number_customer_service_calls` (operational friction proxy).
   3. High daytime usage (minutes / charge).
   4. International usage volume (minutes / charge).
   5. `voice_mail_plan` related indicators / message usage.
- Targeting the top 500 highest-risk customers materially increases prevented churn vs. random outreach under identical capacity.

---

## 5. Business Interpretation

| Driver | Interpretation | Potential Action |
|--------|----------------|------------------|
| International plan | Possible pricing / perceived value issue | Review pricing tiers, offer retention bundle |
| Customer service calls | Unresolved issues trigger departures | Root cause analysis, proactive escalation |
| Heavy day usage | Cost sensitivity or plan mismatch | Offer usage‑aligned plan optimisation |
| Intl usage | Cost / quality concerns | Competitive international add‑ons |
| Low / specific voicemail usage | Engagement proxy | Educate or bundle communication features |

---

## 6. Repository Structure

```
Telecom-Churn-Predictor/
   churn_all.csv               # Source dataset
   telecom_churn_project.ipynb  # Full exploratory + modeling + uplift workflow
   telecom_churn_clients_predictor.ipynb / *_v2.ipynb  # (If present) streamlined prediction-focused variant
   README.md
```

---

## 7. How to Reproduce

1. Create / activate a Python 3.9+ environment.
2. Install dependencies:
    ```bash
    pip install -r requirements.txt  # (If provided) OR
    pip install pandas numpy scikit-learn xgboost seaborn matplotlib
    ```
3. Open `telecom_churn_project.ipynb` and Run All.
4. (Optional) Adjust `param_grid` for broader tuning; re-run tuning cell.
5. For business export: save `top_500_churners_cv` DataFrame to CSV.

Example export snippet (add to notebook if needed):
```python
top_500_churners_cv.to_csv("top_500_high_risk.csv", index=False)
```

---

## 8. Tech Stack

- Python: pandas, numpy
- Modeling: scikit-learn (LogReg, RandomForest, utilities), XGBoost
- Preprocessing: ColumnTransformer (scaling + one-hot encoding)
- Evaluation: ROC curves, standard classification metrics
- Experimentation: GridSearchCV, KFold out-of-fold probability estimation

---

## 9. Limitations

- Geographic / `state` signal excluded; may hold incremental lift.
- No temporal or contractual tenure features included (snapshot only).
- Class imbalance handled implicitly (stratified split + ROC-AUC focus); no resampling / cost-sensitive configuration yet.
- Interpretability limited to global importances; no SHAP / local explanations.

---

## 10. Future Enhancements

- Global reproducibility seed + version stamping cell.
- Persist pipeline + tuned model (`joblib`) for inference script / API.
- Precision–Recall curve + threshold optimisation (cost-based).
- Calibrate probabilities (Platt scaling / isotonic) if operational decisions hinge on absolute risk.
- Add `state` & engineered interaction features; test incremental AUC.
- SHAP analysis for stakeholder interpretability.
- Uplift / causal modeling (e.g., treatment effect estimation) if intervention data becomes available.
- Deployment: lightweight FastAPI or Streamlit scoring app.

---

## 11. Contributing

Open to suggestions via issues / PRs (e.g., broader hyperparameter search, additional evaluation metrics, deployment scaffold).

---

## 12. License

MIT (add LICENSE file if distributing externally).

---

## 13. Contact

For questions or collaboration, please open an issue or reach out via the repository profile.

---

## 14. Quick Reference (Core Objects in Notebook)

| Object | Description |
|--------|-------------|
| `preprocessor` | ColumnTransformer (scaler + one-hot encoder) |
| `xgb_model` | Baseline XGBoost classifier |
| `tuned_xgb_model` | Best params model from GridSearchCV |
| `tuned_importance_series` | Feature importance ranking (tuned model) |
| `top_500_churners_cv` | High-risk segment (cross-validated probabilities) |
| `random_churn_counts_series` | Distribution of churners in random 500-customer samples |

---

Run the notebook, review metrics, adjust cut-off size (e.g., top 300 / 750) to match retention budget, and re-estimate uplift.

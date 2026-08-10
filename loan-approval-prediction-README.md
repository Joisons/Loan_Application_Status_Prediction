# 🏦 Loan Application Status Prediction

Predicting whether a home-loan application will be approved or rejected, based on applicant demographics, income, and credit history.

**Suggested repo name:** `loan-approval-prediction`

## Overview

Dream Housing Finance wants to automate initial screening of loan applications. This project builds a classifier that predicts loan approval from applicant data, and identifies which factors lenders weight most heavily — a useful case study in credit-risk modeling with a small, real-world-messy dataset.

## Dataset

- **Source:** Analytics Vidhya / Dream Housing Finance loan prediction dataset (mirrored on GitHub)
- **Size:** 614 applications, 12 features (plus ID)
- **Target:** `Loan_Status` (Y/N) — ~69% approved / 31% rejected
- **Features:** `Gender`, `Married`, `Dependents`, `Education`, `Self_Employed`, `ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`, `Loan_Amount_Term`, `Credit_History`, `Property_Area`

## Repository Structure

```
loan-approval-prediction/
├── README.md
├── requirements.txt
└── notebooks/
    └── 10_Loan_Application_Status_Prediction.ipynb
```

## Getting Started

```bash
git clone https://github.com/<your-username>/loan-approval-prediction.git
cd loan-approval-prediction
pip install -r requirements.txt
jupyter notebook notebooks/10_Loan_Application_Status_Prediction.ipynb
```

**requirements.txt**
```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
jupyter
```

## Methodology

1. **EDA** — approval rate by credit history, property area, education, marital status; income distribution by status.
2. **Feature engineering** — `TotalIncome` (applicant + co-applicant), log-transformed loan amount, income-to-loan ratio.
3. **Imputation** — median imputation for numeric features, most-frequent for categoricals (several columns have 2–8% missingness).
4. **Preprocessing** — `ColumnTransformer` with imputation + `StandardScaler`/`OneHotEncoder` inside a `Pipeline`.
5. **Model comparison** — Logistic Regression, Random Forest, Gradient Boosting, XGBoost with class-balancing, via 5-fold stratified CV (F1).
6. **Tuning** — `GridSearchCV` over Logistic Regression's regularization strength `C`.
7. **Evaluation** — F1, ROC-AUC, confusion matrix, logistic regression coefficients (odds-ratio direction).

## Results

| Metric | Value |
|---|---|
| Final model | **Logistic Regression** (C=0.01, balanced) |
| CV F1 | 0.856 |
| **Test ROC-AUC** | **0.854** |

**Top feature:** `Credit_History` — by a wide margin the strongest approval predictor.

## Key Insights

- Credit history dominates the approval decision far more than income, education, or property area.
- Raw applicant income has surprisingly little independent predictive power once credit history is controlled for.
- A regularized linear model performs competitively with — or better than — tree ensembles here, since the dataset is small (614 rows) and the key relationship (credit history → approval) is close to a threshold effect.

## Future Work

- Audit for fairness (ensure gender/marital-status don't introduce discriminatory bias).
- Combine with external credit-bureau scores if available.
- Collect more applications to reduce estimate variance.

## Data Source

Dataset from Analytics Vidhya's "Loan Prediction" practice problem.

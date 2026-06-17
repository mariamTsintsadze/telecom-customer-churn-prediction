# Telecom Customer Churn: Factor Identification and Prediction

> **ტელეკომუნიკაციის მომხმარებელთა გადინების განმაპირობებელი ფაქტორების
> იდენტიფიცირება და პროგნოზირება მანქანური სწავლების მეთოდებით**

---

## Project Information

| | |
|---|---|
| **Author** | Mariam Tsintsadze |
| **Supervisor** | Prof. Maxim Iavich (Doctor of Mathematics, Professor) |
| **University** | Caucasus University, School of Technology |
| **Program** | Bachelor of Computer Science |
| **Year** | 2026 |

---

## Project Overview

This project identifies the key factors driving customer churn in the
telecommunications industry and builds predictive models using machine
learning. It goes beyond simple prediction by providing interpretable
explanations (SHAP, LIME) and a practical Risk Scoring System for
business decision-making.

### Key Features

- **Data Cleaning & Validation** — type checks, missing values,
  outliers, duplicates
- **Statistical Testing** — Chi-Square (14/16 significant) and
  T-Tests to validate feature significance
- **3 Machine Learning Models** — Logistic Regression, Random Forest,
  XGBoost + Stacking Ensemble
- **Class Imbalance Handling** — SMOTE (1,495 → 4,139 minority samples)
  with before/after comparison
- **Hyperparameter Tuning** — RandomizedSearchCV
  (50 iterations, 5-fold CV, Best CV AUC = 0.9343)
- **Model Interpretability** — SHAP + LIME + RF Gini
  (triple validation — 4/5 features agree)
- **Threshold Optimization** — Recall: 66.04% → 92.25%
  (threshold: 0.50 → 0.23)
- **Customer Segmentation** — K-Means, 3 clusters
  (Silhouette = 0.4050)
- **Churn Risk Scoring System** — 4-tier categorization
  (Low / Medium / High / Critical)
- **Exported Deliverable** — `customer_risk_scores.csv`
  for direct business use

---

## Project Structure

```
churn_project/
├── Telco_Customer_Churn_Analysis.ipynb   # Main analysis notebook (17 sections)
├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Source dataset (7,043 customers)
├── customer_risk_scores.csv              # Output: risk scores
├── requirements.txt                      # Python dependencies
├── README.md                             # This file
└── images                                # Generated visualizations
```

---

## Installation & Setup

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- Jupyter Notebook

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd churn_project
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Install OpenMP (required for XGBoost on macOS)

```bash
brew install libomp
```

### Step 5: Prepare Dataset
Download the IBM Telco Customer Churn dataset from Kaggle:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn
Place the file WA_Fn-UseC_-Telco-Customer-Churn.csv
in the project root folder.

### Step 4: Run the Notebook

```bash
jupyter notebook Telco_Customer_Churn_Analysis.ipynb
```

Then click **Cell > Run All** to execute all cells.

---

## Dependencies (requirements.txt)

pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
statsmodels>=0.12.0
scikit-learn>=0.24.0
xgboost>=1.4.0
imbalanced-learn>=0.8.0
shap>=0.39.0
lime>=0.2.0
jupyter>=1.0.0
notebook>=6.0.0
ipykernel>=6.0.0

---

## Dataset

**Source:** [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

| Property | Value |
|----------|-------|
| Rows | 7,043 |
| Features | 21 |
| Target | Churn (Yes/No) |
| Churn Rate | ~26.5% |

### Features Include:
- **Demographics:** gender, SeniorCitizen, Partner, Dependents
- **Services:** PhoneService, InternetService, OnlineSecurity, TechSupport, Streaming
- **Account:** Contract, PaymentMethod, PaperlessBilling, MonthlyCharges, TotalCharges, tenure

---

## Methodology

```
Raw Data → Data Cleaning → EDA → Statistical Tests → Feature Engineering
    → Customer Segmentation → SMOTE → Model Training → Hyperparameter Tuning → Evaluation → Threshold Optimization → SHAP/LIME Interpretation → Risk Scoring → Business Recommendations
```

### Models Used:
| Model | Purpose |
|-------|---------|
| Logistic Regression | Linear baseline, interpretable coefficients |
| Random Forest | Non-linear ensemble, built-in feature importance |
| XGBoost (Tuned) | Best performer, gradient boosting |
| Stacking Ensemble	| Meta-learner: Logistic Regression |

### Evaluation Metrics:
- Accuracy, Precision, Recall, F1-Score, ROC-AUC
- 5-Fold Stratified Cross-Validation
- Business Cost Analysis

---

## Results

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.7764 | 0.5770 | 0.5909 | 0.5839 | 0.8318 |
| **Random Forest ** | 0.7715 | 0.5588 | **0.6604** | **0.6054** | **0.8378** |
| XGBoost (Tuned) | 0.7828 | 0.5919 | 0.5856 | 0.5887 | 0.8268 |
| Stacking Ensemble | 0.7821 | 0.5823 | 0.6337 | 0.6069 | 0.8398 |

### Top 5 Churn Factors (SHAP):
| # | Feature | Direction | Interpretation |
|---|---|---|---|
| 1 | PaymentMethod — Electronic Check | ↑ Churn | Highest risk payment method |
| 2 | tenure (low) | ↑ Churn | New customers — highest risk |
| 3 | HasSecuritySupport | ↓ Churn | Security services reduce risk |
| 4 | InternetService — Fiber Optic | ↑ Churn | Fiber users churn more |
| 5 | PaperlessBilling | ↑ Churn | Slightly higher churn risk |

### Customer Segmentation

| Cluster | Size | Churn Rate | Avg Tenure | Strategy |
|---|---|---|---|---|
| Cluster 2 — Critical Risk | 3,345 | 45.3% | 17 months | Contract upgrade + discount |
| Cluster 1 — Loyal | 2,108 | 11.1% | 58 months | VIP program + upsell |
| Cluster 0 — Low Risk | 1,590 | 7.5% | 31 months | Monitor |

### Risk Scoring Distribution:
| Category | Score | Customers | % | Actual Churn Rate | Action |
|---|---|---|---|---|---|
| Low | 0–30% | 3,752 | 53.3% | 3.6% | Monitor |
| Medium | 31–60% | 1,591 | 22.6% | 33.4% | Email campaign |
| High | 61–85% | 1,273 | 18.1% | 64.7% | 10% discount + call |
| Critical | 86–100% | 427 | 6.1% | 88.8% | 20% discount + urgent call |

---

## Output Files

| File | Description |
|------|-------------|
| `customer_risk_scores.csv` | Risk score (0-100%) and category for each customer |
| `*.png` | All visualizations |

---

## Technologies Used

| Category | Tools |
|----------|-------|
| Language | Python 3.9+ |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn, XGBoost |
| Interpretability | SHAP, LIME |
| Class Balancing | imbalanced-learn (SMOTE) |
| Statistics | SciPy, Statsmodels |
| Environment | Jupyter Notebook |
| Version Control | Git / GitHub |

---

## Business Recommendations

Based on SHAP, LIME, RF Gini analysis and Risk Scoring results:

1. **Incentivize long-term contracts**
   — Month-to-month churn: 42.7% vs Two-year: 2.8%
   — Offer 10–20% discount for monthly → annual upgrades

2. **Bundle security and support services**
   — HasSecuritySupport = Top 3 churn factor (SHAP + LIME + RF Gini)
   — Free 3-month trial of Online Security + Tech Support
    for Medium/High risk customers

3. **Early tenure intervention program**
   — Customers with tenure ≤ 12 months: 47.4% churn rate
   — Structured onboarding: check-ins at months 1, 3, 6, and 12

4. **Optimize payment experience**
   — Electronic Check churn: 45.3% vs Auto-pay: 15–18%
   — month discount for switching to automatic payment

5. **Proactive retention for Critical-risk customers**
   — 427 Critical-risk customers identified (88.8% actual churn rate)
   — Personal call + 20% discount within 48 hours of identification

6. **Fiber Optic retention campaign**
   — Fiber Optic = Top churn driver (confirmed by SHAP, LIME, RF Gini)
   — Satisfaction surveys + exclusive offers for
    High/Critical risk Fiber Optic users

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError: shap` | `pip install shap` |
| `ModuleNotFoundError: imblearn` | `pip install imbalanced-learn` |
| `FileNotFoundError: CSV not found` | Place dataset in project root folder |
| `XGBoostError: libxgboost.dylib` | `brew install libomp` (macOS only) |
| `ModuleNotFoundError` (any) | `pip install -r requirements.txt` |
| Kernel dies during SHAP | Increase RAM or reduce SHAP sample size |
| SMOTE memory error | Reduce `sampling_strategy` parameter |
---

## License

This project was developed as part of an academic research project at Caucasus University.

Copyright © 2026 Mariam Tsintsadze. All rights reserved.

This repository is provided for educational and evaluation purposes only. The source code, documentation, visualizations, and generated outputs may not be redistributed, modified, or used for commercial purposes without prior written permission from the author.

The dataset used in this project remains subject to the original dataset license and terms of use.

This project is not intended for production use and was created exclusively for academic purposes.


---

## Contact

- **Author:** Mariam Tsintsadze
- **University:** Caucasus University, School of Technology
- **Supervisor:** Prof. Maxim Iavich

---
## Acknowledgements
IBM — for the Telco Customer Churn Dataset
Kaggle — for dataset accessibility
SHAP & LIME — open-source interpretability libraries
Prof. Maxim Iavich — for supervision and guidance


# Week 4 — Supervised Learning: Regression & Classification
**AnalystLab Africa Internship Program**

This repository contains two supervised machine learning projects completed for Week 4 of the AnalystLab Africa internship: a regression model predicting house prices, and a classification model predicting Titanic passenger survival.

---

## 📁 Repository Structure

```
├── titanic_ml.ipynb        # Task 2: Logistic Regression (Classification)
├── housing_ml.ipynb        # Task 1: Linear Regression (Regression)
├── evaluation_report.pdf   # Written evaluation report (see Deliverables)
├── data/
│   ├── titanic_cleaned.csv
│   └── housing.csv
└── README.md
```

---

## Task 1 — Linear Regression: California Housing Price Prediction

**Goal:** Predict `median_house_value` using housing and location features.

**Dataset:** California Housing dataset (20,640 rows, 10 columns).

**Pipeline:**
1. Missing values in `total_bedrooms` filled with median.
2. Exploratory analysis: correlation heatmap, boxplots, scatter plots.
3. Categorical feature `ocean_proximity` one-hot encoded (`drop_first=True` to avoid the dummy variable trap).
4. 4 rows with `ocean_proximity = ISLAND` dropped after diagnosis (see Key Findings).
5. Train-test split (80/20) performed **before** scaling to prevent data leakage.
6. Numeric features scaled with `StandardScaler` (fit on training data only); binary/dummy columns left unscaled to preserve coefficient interpretability.
7. `LinearRegression` trained and evaluated.

**Performance:**
| Metric | Value |
|---|---|
| RMSE (full test set) | 69,411 |
| RMSE (excluding capped target rows) | 62,223 |
| Average house value | 206,856 |
| RMSE as % of average value | ~33% |

**Key Findings:**
- **Target capping:** 965 rows (~4.7%) have `median_house_value` capped at $500,001, the dataset's recorded ceiling. This causes systematic underprediction of high-value homes and inflates RMSE. Removing capped rows from the test set dropped RMSE by ~11% (70,060 → 62,223), confirming the effect — but does not fully explain the model's error.
- **Unreliable coefficient:** `ocean_proximity_ISLAND` initially produced the largest coefficient in the model (+136,125) despite being present in only 4 of 16,512 training rows — a statistically unreliable estimate from an insufficient sample, not a real generalizable effect. These rows were dropped; the fix improved reliability but had negligible impact on RMSE (as expected, given 4 rows out of 20,640).
- **Multicollinearity:** Strong correlations exist among `total_rooms`, `total_bedrooms`, `population`, and `households` (r = 0.86 to 0.97), and between `latitude` and `longitude` (r = -0.92). This destabilizes individual coefficient interpretation (e.g. `total_rooms` shows a counterintuitive negative coefficient) without necessarily harming overall prediction. Not corrected in this iteration due to time constraints — noted as a limitation.
- **Conclusion:** The residual ~33% error is primarily attributable to Linear Regression's inability to capture non-linear relationships in housing data, not the data issues above, which were diagnosed and partially addressed but were secondary contributors.

**Recommended next steps:** Non-linear models (Random Forest), regularized regression (Ridge/Lasso/Elastic Net) to manage multicollinearity, VIF-based feature selection, log-transformation of the target variable.

---

## Task 2 — Logistic Regression: Titanic Survival Prediction

**Goal:** Predict passenger `Survived` status (binary classification).

**Dataset:** Titanic dataset, pre-cleaned in a prior EDA notebook.

**Features used:** Age, Fare, Passenger Class, Sex.

**Pipeline:**
1. Missing values handled and categorical variables encoded during prior EDA.
2. Train-test split performed.
3. `LogisticRegression` trained on selected features.
4. Predictions evaluated against test set.

**Performance:**
| Metric | Value |
|---|---|
| Accuracy | 79.9% |

**Note:** A naive baseline (predicting "did not survive" for every passenger) achieves ~61% accuracy on this dataset — useful context for interpreting the model's actual lift over doing nothing.

**Limitation:** Feature scaling was not applied in this task, unlike Task 1. Logistic Regression benefits from scaled inputs for coefficient interpretability and faster convergence — an inconsistency to address in future iterations.

---

## Tools & Libraries
`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

## Author
[Ibinabo Direh] — AnalystLab Africa Intern, Week 4
`#AnalystLabAfrica`

# Santander Customer Satisfaction — Tabular Kaggle Classification Project

**Course:** DATA 3402 | University of Texas at Arlington
**Instructor:** Dr. Farbin
**Kaggle Competition:** [Santander Customer Satisfaction](https://www.kaggle.com/competitions/santander-customer-satisfaction)

---

## Project Overview

This project tackles the Santander Customer Satisfaction Kaggle competition: a binary classification problem where the goal is to predict whether a bank customer is unsatisfied (`TARGET = 1`) or satisfied (`TARGET = 0`) using anonymized transactional and account data. The competition is scored on **ROC-AUC**, which is the appropriate metric here given the heavy class imbalance.

The dataset is challenging because:
- All 370 features are anonymized (e.g. `var15`, `saldo_var13`, `imp_op_var40_ult1`) — there is no public documentation explaining what they represent.
- The classes are extremely imbalanced (~96% satisfied / ~4% unsatisfied).
- Many columns are sparse or near-constant, requiring careful filtering before modeling.

---

## Dataset Summary

| Property | Value |
|---|---|
| Rows | ~76,000 |
| Raw features | 370 |
| Target | Binary (0 = satisfied, 1 = unsatisfied) |
| Missing values | None |
| Class balance | ~96% / ~4% (severe imbalance) |
| Feature types | All numeric (anonymized) |

Feature naming convention (Spanish-origin prefixes):
- `ind_` — binary indicator flags
- `num_` — counts
- `saldo_` — account balances
- `imp_` — monetary amounts
- `delta_` — change between time periods
- `_ult1`, `_ult3` — last 1 / last 3 months
- `_hace2`, `_hace3` — 2 / 3 months ago

---

## Approach

### 1. Data Loading & Initial Inspection
- Loaded `train.csv` and dropped the `ID` column (non-predictive).
- Verified no missing values across all 370 features.
- Confirmed all features are numeric.
- Visualized the `TARGET` distribution and confirmed the severe class imbalance.

### 2. Exploratory Data Analysis
- Generated a histogram matrix across all features to inspect distributions.
- Computed per-feature correlation with `TARGET`. The strongest correlation was only ~0.10, suggesting linear models alone would not perform well and that a non-linear model (e.g. tree ensemble) was the better choice.

### 3. Data Cleaning
**Step A — Drop fully-zero columns:** Identified and removed 34 columns where every single value was 0 (verified by summing each column to confirm). This reduced the column count from 370 → 336.

**Step B — Variance Threshold filtering:** Used `sklearn.feature_selection.VarianceThreshold` with a threshold of 0.1 to remove additional near-zero variance features that had almost no useful signal. This removed roughly 60 more columns, leaving 273 features.

The motivation: with anonymized data, manual feature inspection is not feasible. A variance-based filter is the principled way to remove sparse columns automatically without losing meaningful signal.

### 4. Modeling

**Train/Validation Split:** Stratified 80/20 split using `train_test_split(stratify=y)` to preserve the rare positive class proportion in both sets.

**Baseline Model:** `RandomForestClassifier` was chosen because:
- It handles high-dimensional data without requiring feature scaling.
- It is robust to anonymized features and non-linear relationships.
- It produces feature importances directly, enabling further pruning.

Key hyperparameters:
- `class_weight='balanced'` to compensate for the ~96/4 class imbalance.
- `n_estimators=200` for stable importance estimates.
- `n_jobs=-1` for parallel training.

**Feature Pruning:** Extracted `feature_importances_` from the trained model and used cumulative importance to select the smallest set of features that account for 95% of total importance. This is a more principled cutoff than picking an arbitrary "top N."

**Pruned Model:** Retrained `RandomForestClassifier` with the same hyperparameters on only the pruned feature set, allowing an apples-to-apples comparison against the full-feature baseline.

### 5. Evaluation
Compared full-feature vs. pruned models on the validation set using:
- ROC-AUC (the official Kaggle metric)
- Classification report (precision, recall, F1)

---

## Results

| Model | Features | Validation ROC-AUC |
|---|---|---|
| Random Forest (full) | 273 | _0.75_ |
| Random Forest (pruned) | _76_ | _0.76_ |

**Key finding:** The pruned model achieves comparable ROC-AUC with significantly fewer features, confirming that most of the signal is concentrated in a small subset of the original 370 features.

---

## Repository Structure

```
Data3402-TabularKaggleProject/
├── README.md                      # This file
├── .gitignore
├── Dataset/
│   └── train.csv                  # Training data from Kaggle
└── Programfiles/
    └── project.ipynb              # Main project notebook
```

---

## How to Run

1. Clone the repository.
2. Download `train.csv` from the [Kaggle competition page](https://www.kaggle.com/competitions/santander-customer-satisfaction/data) and place it in the `Dataset/` folder.
3. Open `Programfiles/project.ipynb` in Jupyter.
4. Run all cells top to bottom.

## Requirements

- Python 3.10+
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- joblib

Install with:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib
```

---

## Summary of Pipeline

| Step | Action | Columns Remaining |
|---|---|---|
| 0 | Load raw `train.csv` | 371 |
| 1 | Drop `ID` | 370 |
| 2 | Drop all-zero columns | 336 |
| 3 | Apply VarianceThreshold (0.1) | 273 |
| 4 | Train/validation split (80/20 stratified) | 273 |
| 5 | Train RandomForest (balanced class weights) | 273 |
| 6 | Prune to top features (95% cumulative importance) | _[fill in]_ |
| 7 | Retrain pruned RandomForest | _[fill in]_ |
| 8 | Save model + preprocessing state with joblib | — |

---

## Key Takeaways

- **Variance thresholding** is essential when working with anonymized, sparse tabular data where manual inspection is impossible.
- **Class imbalance** must be addressed explicitly — accuracy is meaningless on a 96/4 split, and `class_weight='balanced'` is a simple but effective fix.
- **ROC-AUC** is the right metric here, both because Kaggle uses it and because it captures ranking quality across all decision thresholds.
- **Feature importance + cumulative cutoff** gives a principled, reproducible way to prune features rather than picking an arbitrary "top N."

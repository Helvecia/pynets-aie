# HW06 – Decision Trees & Ensembles

## Overview

Homework assignment for seminar S06: "Decision Trees and Ensembles (Bagging / Random Forest / Boosting / Stacking)".

This assignment implements a complete, honest ML experiment comparing baseline models with tree-based ensemble methods on an imbalanced binary classification task (S06-hw-dataset-04.csv).

## Directory Structure

```
HW06/
├── HW06.ipynb                      # Main Jupyter notebook with full experiment
├── report.md                        # Analysis report (follows S06-hw-report-template.md)
├── S06-hw-dataset-04.csv           # Dataset (400 samples, 60 features, binary target, 11.5:1 imbalance)
├── artifacts/
│   ├── metrics_test.json           # Final test set metrics for all models
│   ├── search_summaries.json       # Best hyperparameters and CV scores
│   ├── best_model.joblib           # Serialized best model (RandomForest)
│   ├── best_model_meta.json        # Metadata about best model
│   └── figures/
│       ├── roc_curves.png          # ROC curves for all models
│       ├── confusion_matrix.png    # Confusion matrix for best model
│       └── permutation_importance.png  # Top-15 feature importance plot
└── README.md                        # This file
```

## Experiment Summary

### Dataset
- **File:** `S06-hw-dataset-04.csv`
- **Samples:** 400
- **Features:** 60 numerical (f01–f60)
- **Target:** Binary (0/1) with strong imbalance (11.5:1 ratio)
- **Task:** Fraud detection on imbalanced data

### Protocol
- **Train/Test split:** 75% / 25% with stratification (random_state=42)
- **Hyperparameter tuning:** 5-fold StratifiedKFold GridSearchCV on train set
- **Metrics:** Accuracy, F1, ROC-AUC (ROC-AUC prioritized due to class imbalance)

### Models Trained
1. **DummyClassifier** (baseline: always predicts majority class)
2. **LogisticRegression** (linear baseline with StandardScaler + balanced weights)
3. **DecisionTreeClassifier** (with complexity control: max_depth, min_samples_leaf)
4. **RandomForestClassifier** (bagging + feature randomness)
5. **GradientBoostingClassifier** (sequential boosting)

### Key Results

| Model | Accuracy | F1 | ROC-AUC |
|-------|----------|-----|----------|
| DummyClassifier | 0.92 | 0.00 | 0.50 |
| LogisticRegression | 0.93 | 0.33 | 0.78 |
| DecisionTree | 0.94 | 0.40 | 0.80 |
| **RandomForest** | **0.95** | **0.57** | **0.87** |
| GradientBoosting | 0.94 | 0.50 | 0.87 |

**Winner:** RandomForest (ROC-AUC=0.8719)

## How to Use

### Run the Notebook

```bash
jupyter notebook HW06.ipynb
```

The notebook will:
1. Load `S06-hw-dataset-04.csv`
2. Perform EDA and check class balance
3. Split data into train/test with stratification
4. Train baseline models (Dummy, LogisticRegression)
5. Train ensemble models with GridSearchCV hyperparameter tuning
6. Generate diagnostic plots (ROC, confusion matrix, importance)
7. Save artifacts (metrics, best model, metadata)

### Load and Use the Best Model

```python
import joblib

# Load best model
best_model = joblib.load('artifacts/best_model.joblib')

# Load metadata
import json
with open('artifacts/best_model_meta.json', 'r') as f:
    metadata = json.load(f)
print(f"Best model: {metadata['model_name']}")
print(f"Test ROC-AUC: {metadata['test_metrics']['roc_auc']}")

# Make predictions on new data
y_pred = best_model.predict(X_new)
y_pred_proba = best_model.predict_proba(X_new)[:, 1]
```

## Key Findings

1. **Ensemble methods dominate:** RandomForest and GradientBoosting significantly outperform baseline (ROC-AUC 0.87 vs. 0.50 for Dummy)

2. **Hyperparameter tuning matters:** Proper CV-based tuning of `max_depth`, `min_samples_leaf`, and `max_features` prevented overfitting

3. **Metric selection critical:** Accuracy (0.95) is misleading on imbalanced data; ROC-AUC (0.87) and F1 (0.57) provide honest evaluation

4. **Stability validated:** Model performance is stable across different random states (std=0.002 for ROC-AUC)

5. **Feature importance:** Top-15 features concentrated in ranges f07–f14 and f41–f44 (f13 most important)

## Requirements

- Python 3.9+
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- joblib

## References

- **S06 Brief:** `seminars/S06/S06-brief.md`
- **Assignment Spec:** `seminars/S06/S06-homework.md`
- **Report Template:** `seminars/S06/S06-hw-report-template.md`

---

**Completion Date:** January 12, 2026  
**Dataset Used:** S06-hw-dataset-04.csv (imbalanced fraud detection)  
**Best Model:** RandomForest (ROC-AUC=0.8719)  
**Status:** ✅ Complete – Ready for review

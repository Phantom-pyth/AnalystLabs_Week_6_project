# 🛠️ Feature Engineering & Model Optimization for  Titanic Survival Prediction

> Week 6 of the AnalystLab Africa Data Science Internship transforming raw variables into meaningful features, selecting the most predictive subset, and optimizing model performance through hyperparameter tuning with cross-validation.

---

## 📋 Project Overview

This project applies a full feature engineering and optimization pipeline to the Titanic survival classification task. Nine new features are created from raw variables, three feature selection methods are applied to identify the best predictors, and two ensemble models are tuned using GridSearchCV and RandomizedSearchCV with 5-fold stratified cross-validation.

---

## 📁 Repository Structure

```
week6-feature-engineering-optimization/
│
├── Titanic-Dataset.csv                          # Raw Titanic dataset
├── Week6_Feature_Engineering_Optimization.ipynb  # Full notebook: all 8 steps
├── Week6_Model_Performance_Report.docx           # Performance comparison report
├── features_engineered.png               # Survival rates by engineered features
├── model_performance.png                  # Performance comparison + ROC curves
├── confusion_matrix.png                      # CV results + confusion matrix
└── README.md                                     # This file
```

**Dataset link:** [Titanic Dataset (Kaggle)](https://www.kaggle.com/competitions/titanic/data)

---

## 🎯 Objective

Enhance predictive performance by:
- Creating meaningful features from raw variables
- Transforming and preparing variables for modelling
- Selecting the most relevant features using multiple methods
- Optimizing model performance through hyperparameter tuning
- Evaluating performance before and after optimization

---

## 📊 Dataset

| Property | Details |
|----------|---------|
| Source | Titanic-Dataset.csv (raw) |
| Records | 891 passengers |
| Original features | 12 raw columns |
| Target variable | `Survived` (0=Died, 1=Survived) |
| Class balance | 61.6% Died / 38.4% Survived |
| Train / Test split | 712 / 179 rows (80/20, stratified) |

---

## 🔬 Features Engineered

9 new features created from existing variables:

| Feature | Construction | Rationale |
|---------|-------------|-----------|
| `Title` | Extracted from Name (Mr/Mrs/Miss/Master/Rare) | Encodes age, gender, and social status simultaneously |
| `FamilySize` | SibSp + Parch + 1 | Total group size including self |
| `IsAlone` | 1 if FamilySize=1 | Solo travellers: 30.4% survival vs 50.6% with family |
| `FamilySizeGroup` | Alone / Small (2-4) / Large (5+) | Captures non-linear family-size effect |
| `AgeBand` | Child / Teen / Young Adult / Adult / Senior | Evacuation priority was age-categorical not linear |
| `FarePerPerson` | Fare ÷ passengers sharing ticket | Corrects for shared-ticket fare inflation |
| `FareBand` | Quartile-based Low/Mid/High/VeryHigh | Handles Fare's extreme right skew categorically |
| `CabinKnown` | 1 if Cabin not null | Having a recorded cabin correlates with 1st class (66.7% survival) |
| `SexPclass` | Sex_encoded × Pclass | Captures compounding effect: sex × class interaction |

---

## 🔍 Feature Selection

Three methods applied:

1. **Correlation Analysis**  absolute correlation of each feature with the `Survived` target
2. **Random Forest Feature Importance**  trained on all 16 features
3. **Recursive Feature Elimination (RFE)**  Logistic Regression estimator, top 10 selected

**Final 10 features selected** (consistent across all three methods):
`Sex_enc`, `SexPclass`, `Title_enc`, `Pclass`, `FarePerPerson`, `Fare`, `CabinKnown`, `AgeBand_enc`, `FamilySize`, `IsAlone`

Reduced from 16 → 10 features with no meaningful loss of information.

---

## ⚙️ Hyperparameter Tuning

### Random Forest — GridSearchCV
- CV: 5-fold Stratified K-Fold, scoring=ROC-AUC
- Parameters searched: `n_estimators`, `max_depth`, `min_samples_split`, `max_features`
- **Best params:** `n_estimators=300, max_depth=5, min_samples_split=5, max_features=sqrt`
- Best CV AUC: **0.8812**

### Gradient Boosting — RandomizedSearchCV
- 30 random iterations, CV: 5-fold Stratified K-Fold, scoring=ROC-AUC
- Parameters searched: `n_estimators`, `max_depth`, `learning_rate`, `min_samples_split`, `subsample`
- **Best params:** `n_estimators=250, max_depth=2, learning_rate=0.15, min_samples_split=5, subsample=0.7`
- Best CV AUC: **0.9012**

---

## 📈 Results — Before & After Optimization

| Model | Accuracy | F1 | AUC |
|-------|----------|----|-----|
| 🏆 **RF (tuned)** | **81.56%** | **0.7556** | 0.8529 |
| LR Baseline (all features) | 81.01% | 0.7258 | 0.8630 |
| GB (tuned) | 80.45% | 0.7368 | 0.8476 |
| GB (before tuning) | 79.89% | 0.7353 | 0.8540 |
| RF (before tuning) | 77.65% | 0.6970 | 0.8277 |

### Improvement from Tuning (Random Forest)
| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Accuracy | 77.65% | 81.56% | **+3.91 pts** |
| F1 Score | 0.6970 | 0.7556 | **+0.0586** |
| Recall | 0.6667 | 0.7391 | **+7.24 pts** |

---

## 🔑 Key Insights

1. **SexPclass interaction term** ranked 2nd most important by both correlation and RF importance despite not existing in the original dataset, it became the second strongest predictor
2. **FarePerPerson** outperformed raw Fare after correcting for shared-ticket groups  cleaner individual wealth signal
3. **CabinKnown** (missingness as a feature) proved more informative than cabin content, demonstrating that the pattern of missing data can carry predictive signal
4. **Tuning improved RF substantially** (+5.86 F1 points) but improved GB only marginally default GB was already near-optimal for this dataset size
5. **Feature engineering narrowed the gap** between simple and complex models the LR baseline with engineered features (AUC=0.863) closely competes with the tuned ensemble models

---

## 🛠️ Technologies Used

- **Python 3.x**
- **Pandas / NumPy**  data manipulation and feature construction
- **Scikit-learn**  RandomForestClassifier, GradientBoostingClassifier, LogisticRegression, StandardScaler, LabelEncoder, RFE, GridSearchCV, RandomizedSearchCV, StratifiedKFold, metrics
- **Matplotlib / Seaborn**  visualizations
- **Jupyter Notebook**  interactive analysis environment

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### Run the Notebook

```bash
git clone https://github.com/Phantom-pyth/AnalystLabs_Week_6_projecct.git
cd week6-feature-engineering-optimization
jupyter notebook Week6_Feature_Engineering_Optimization.ipynb
```

---

## 📄 Deliverables

| File | Description |
|------|-------------|
| `Week6_Feature_Engineering_Optimization.ipynb` | Full 8-step notebook: preprocessing, feature engineering (×9), transformation, selection (×3 methods), baseline, advanced models, GridSearchCV, RandomizedSearchCV, cross-validation, evaluation |
| `Week6_Model_Performance_Report.docx` | 1-2 page report covering dataset overview, features created, features selected, tuning methods, before/after performance, findings, recommendations |
| `features_engineered.png` | Survival rates by all 6 key engineered features |
| `model_performance.png` | All 5 models compared on Accuracy/F1/AUC + ROC curves |
| `confusion_matrix.png` | 5-fold CV accuracy comparison + best model confusion matrix |

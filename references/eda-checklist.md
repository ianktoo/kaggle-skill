# EDA Checklist

A complete checklist for exploratory data analysis on Kaggle competition datasets. Work through every section before starting feature engineering.

---

## 1. Data Loading & Shape

```python
import pandas as pd, numpy as np, os

train = pd.read_csv("./data/train.csv")
test  = pd.read_csv("./data/test.csv")

print(f"Train: {train.shape}")
print(f"Test:  {test.shape}")
print(f"Memory (train): {train.memory_usage(deep=True).sum() / 1e6:.1f} MB")
```

- [ ] Row and column counts match expectations from competition description
- [ ] Memory usage is manageable (if > 2GB, consider chunking or dtype optimization)

---

## 2. Column Types

```python
print(train.dtypes.value_counts())
print("\nColumn list:")
print(train.dtypes.to_string())
```

- [ ] Identify numeric columns (int, float)
- [ ] Identify categorical/string columns (object)
- [ ] Identify boolean columns
- [ ] Flag any datetime columns that need parsing
- [ ] Flag any ID columns (high cardinality, no predictive value)

---

## 3. Missing Values

```python
miss = train.isnull().sum()
miss = miss[miss > 0].sort_values(ascending=False)
pct  = (miss / len(train) * 100).round(2)
print(pd.DataFrame({"count": miss, "pct%": pct}))

# Compare train vs. test missingness
miss_t = test.isnull().sum()
miss_t = miss_t[miss_t > 0].sort_values(ascending=False)
print("\nTest missingness:")
print(pd.DataFrame({"count": miss_t, "pct%": (miss_t/len(test)*100).round(2)}))
```

- [ ] Which columns have missing values?
- [ ] Is missingness random or systematic? (check if missing correlates with target)
- [ ] Does test have the same missing pattern as train?
- [ ] High-missingness columns (> 50%) — consider dropping vs. imputing

---

## 4. Target Variable

```python
TARGET = "target"

# Classification
print(train[TARGET].value_counts(normalize=True))

# Regression
print(train[TARGET].describe())
import matplotlib.pyplot as plt
train[TARGET].hist(bins=50)
plt.title("Target Distribution")
plt.show()
```

- [ ] Class distribution — any imbalance? (classification)
- [ ] Outliers in target? (regression)
- [ ] Log-transform needed? (right-skewed regression targets)
- [ ] Any impossible values (negative prices, impossibly high ages)?

---

## 5. Numeric Features

```python
num_cols = train.select_dtypes(include=np.number).columns.tolist()
num_cols = [c for c in num_cols if c not in ["id", TARGET]]

# Summary stats
print(train[num_cols].describe().T.round(2))

# Skewness
skew = train[num_cols].skew().sort_values(ascending=False)
print("\nHighly skewed (|skew| > 1):")
print(skew[skew.abs() > 1])

# Outliers — z-score
from scipy import stats
z_scores = np.abs(stats.zscore(train[num_cols].fillna(0)))
outlier_cols = (z_scores > 3).sum()
print("\nOutlier counts (z > 3):")
print(outlier_cols[outlier_cols > 0].sort_values(ascending=False))
```

- [ ] Distributions — normal, skewed, bimodal?
- [ ] Outliers (z-score > 3 or domain-impossible values)
- [ ] Skewed features (candidates for log1p transform)
- [ ] Constant or near-constant columns (zero variance)
- [ ] Columns that look like categories encoded as numbers

---

## 6. Categorical Features

```python
cat_cols = train.select_dtypes(include="object").columns.tolist()

for col in cat_cols:
    n_unique = train[col].nunique()
    top5 = train[col].value_counts().head(5).index.tolist()
    print(f"{col}: {n_unique} unique | top: {top5}")
```

- [ ] Cardinality — low (< 10), medium (10–100), high (> 100)
- [ ] Rare categories (< 1% frequency) — candidates for grouping
- [ ] Train/test distribution mismatch (unseen categories in test)
- [ ] Ordinal categories (e.g., Low/Medium/High) — need ordered encoding

```python
# Train/test mismatch
for col in cat_cols:
    if col not in test.columns:
        continue
    unseen = set(test[col].dropna()) - set(train[col].dropna())
    if unseen:
        print(f"{col}: {len(unseen)} unseen in test: {list(unseen)[:5]}")
```

---

## 7. Correlations

```python
import seaborn as sns

corr = train[num_cols + [TARGET]].corr()

# Feature–target correlation
print("Correlation with target:")
print(corr[TARGET].drop(TARGET).sort_values(ascending=False))

# Feature–feature heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr, cmap="coolwarm", center=0, annot=len(num_cols) <= 15)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()
```

- [ ] Features highly correlated with target (good signal)
- [ ] Features highly correlated with each other (> 0.95) — potential redundancy
- [ ] Any feature that is almost identical to the target (leakage!)

---

## 8. Leakage Check

Leakage is the most dangerous issue in Kaggle. A feature that contains information about the future or the target itself will inflate CV scores but fail on the leaderboard.

- [ ] Any feature correlated > 0.99 with target?
- [ ] Any ID columns that encode order (and order leaks temporal information)?
- [ ] Any timestamp after the event being predicted?
- [ ] Any aggregate that was computed including the test rows?

---

## 9. Duplicate Rows

```python
print(f"Duplicate rows in train: {train.duplicated().sum()}")
print(f"Duplicate rows in test:  {test.duplicated().sum()}")

# Duplicates with different targets (label noise)
if TARGET in train.columns:
    dup_conflicts = train.groupby(
        [c for c in train.columns if c != TARGET]
    )[TARGET].nunique()
    print(f"Feature-identical rows with different targets: {(dup_conflicts > 1).sum()}")
```

- [ ] Duplicate rows? Remove or investigate
- [ ] Feature-identical rows with conflicting targets (label noise)

---

## 10. Train / Test Distribution Shift

If train and test come from different time periods or sources, there may be distribution shift that hurts CV reliability.

```python
# Simple shift detector: train a classifier to predict train vs. test
combined = pd.concat([
    train[num_cols].assign(is_test=0),
    test[num_cols].fillna(test[num_cols].median()).assign(is_test=1)
])

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

X_shift = combined.drop(columns="is_test").fillna(-999)
y_shift = combined["is_test"]

rf = RandomForestClassifier(n_estimators=100, random_state=42)
auc = cross_val_score(rf, X_shift, y_shift, cv=5, scoring="roc_auc").mean()
print(f"Train/test AUC: {auc:.3f}  (0.5 = no shift, 1.0 = severe shift)")
```

- [ ] AUC close to 0.5 — no significant shift, standard CV is fine
- [ ] AUC > 0.8 — significant shift — consider adversarial validation or time-based CV

---

## EDA Done? Check These Before Moving On

- [ ] I know what each row represents
- [ ] I know the target distribution and any class imbalance
- [ ] I've identified missing values and have a plan for each
- [ ] I've flagged any leakage candidates
- [ ] I've noted the most correlated features with the target
- [ ] I know which categoricals need special encoding
- [ ] I've checked train/test distribution shift

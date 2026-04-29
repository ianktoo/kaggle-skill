# EDA Checklist

A complete checklist for exploratory data analysis on Kaggle competition datasets. Work through every section before starting feature engineering.

---

## 1. Data Loading & Shape

- [ ] Do row and column counts match the competition description's stated dataset size?
- [ ] Is memory usage manageable? (If train exceeds ~2 GB, plan dtype optimization or chunked loading before proceeding.)

*Why it matters: Mismatched row counts often mean a bad unzip or wrong file version. Memory issues will crash your script mid-run.*

---

## 2. Column Types

- [ ] Which columns are numeric (int, float)?
- [ ] Which are categorical or string (object)?
- [ ] Are there datetime columns that need parsing?
- [ ] Are there ID columns — high cardinality, no predictive value — that should be dropped before modeling?
- [ ] Are there boolean columns or numeric columns that are actually categories in disguise (e.g., a `status` column with values 0/1/2)?

*Why it matters: Wrong types cause silent errors in modeling. An ID column left in can artificially inflate tree model performance.*

---

## 3. Missing Values

- [ ] Which columns have missing values, and what percentage?
- [ ] Is missingness random, or does it correlate with the target? (Systematic missingness is itself a signal.)
- [ ] Does the test set have a similar missing pattern to train — or are different columns missing in test?
- [ ] For columns missing > 50%: is imputation sensible, or is dropping the column the right call?

*Why it matters: Imputing incorrectly (e.g., filling train-computed medians on test) is a subtle form of leakage. Missingness patterns can reveal domain structure worth engineering as features.*

---

## 4. Target Variable

- [ ] For classification: what is the class distribution? Is there a severe imbalance (e.g., 95:5)?
- [ ] For regression: what does the distribution look like — normal, right-skewed, heavy-tailed?
- [ ] Are there outliers in the target that might pull the model in the wrong direction?
- [ ] Would a log-transform stabilize a right-skewed regression target?
- [ ] Are there impossible values (negative ages, prices below zero, probabilities above 1)?

*Why it matters: Severe class imbalance demands a different CV strategy and potentially class weights or resampling. A log-transformed target can significantly improve RMSE-optimized models.*

---

## 5. Numeric Features

- [ ] Do summary statistics (min, max, mean, std) look reasonable for each column's domain?
- [ ] Are there outliers? (Rule of thumb: z-score > 3 warrants investigation.)
- [ ] Which features have high skew (|skew| > 1)? These are candidates for log1p transformation.
- [ ] Are there constant or near-constant columns (zero or near-zero variance)? They add noise, not signal.
- [ ] Do any numeric columns look like they encode categories (e.g., integers 1–5 representing ordinal levels)?

*Why it matters: Outliers can dominate distance-based models and some loss functions. Highly skewed features hurt linear models and can slow tree convergence.*

---

## 6. Categorical Features

- [ ] What is the cardinality of each categorical — low (< 10), medium (10–100), high (> 100)?
- [ ] Are there rare categories with < 1% frequency? These are candidates for grouping into an "other" bucket.
- [ ] Are there categories in the test set that never appear in train (unseen values)? This will cause errors with naive label encoding.
- [ ] Are any categoricals ordinal (e.g., Low / Medium / High)? They need ordered encoding, not arbitrary integer assignment.

*Why it matters: High-cardinality categoricals need frequency encoding or target encoding — not one-hot. Unseen test categories will crash a fitted LabelEncoder at inference time.*

---

## 7. Correlations

- [ ] Which features have the highest absolute correlation with the target? (These are your most valuable signals.)
- [ ] Are any two features correlated > 0.95 with each other? Redundant features add noise and slow training.
- [ ] Does any feature have correlation > 0.99 with the target? That is almost certainly leakage — investigate before using it.

*Why it matters: High feature-feature correlation is wasted complexity. Near-perfect correlation with the target is a red flag that should block you from proceeding until understood.*

---

## 8. Leakage Check

Leakage is the most dangerous issue in Kaggle. A leaky feature inflates CV scores but collapses on the private leaderboard.

- [ ] Any feature with correlation > 0.99 with the target?
- [ ] Any ID column that encodes temporal order, where order leaks information about the target?
- [ ] Any timestamp that occurs after the event being predicted?
- [ ] Any aggregate feature that was computed using test-set rows?
- [ ] Any column whose definition in the competition data description sounds like it contains the answer?

*Why it matters: Leaky models score brilliantly locally and fail publicly. Finding leakage early saves weeks of wasted work.*

---

## 9. Duplicate Rows

- [ ] Are there duplicate rows in train? Duplicates can distort CV fold distributions.
- [ ] Are there feature-identical rows with conflicting targets (label noise)? This is a signal of data quality issues that may require special handling.

*Why it matters: Exact duplicates in train, if split across folds, cause the model to "memorize" rather than generalize. Conflicting labels set a ceiling on how good any model can be.*

---

## 10. Train/Test Distribution Shift

If train and test come from different time periods or sources, your CV score may be overly optimistic.

- [ ] Train a binary classifier (train rows labeled 0, test rows labeled 1) on the numeric features. What AUC does it achieve?
  - AUC close to 0.5 — no significant shift, standard CV is reliable.
  - AUC > 0.8 — significant shift — consider adversarial validation, time-based CV, or using only features the classifier cannot distinguish.
- [ ] Which features drive the shift classifier's decisions? Those are the features most affected by distribution shift.

*Why it matters: A large CV–LB gap is often caused by distribution shift that standard cross-validation doesn't capture. Detecting it early lets you fix your CV strategy before wasting submissions.*

---

## EDA Done? Check These Before Moving On

- [ ] I know what each row represents
- [ ] I know the target distribution and any class imbalance
- [ ] I've identified missing values and have a plan for each
- [ ] I've flagged any leakage candidates
- [ ] I've noted the most correlated features with the target
- [ ] I know which categoricals need special encoding
- [ ] I've checked train/test distribution shift

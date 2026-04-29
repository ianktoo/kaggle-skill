# Kaggle Jargon Glossary

Plain-English explanations for every term that trips up newcomers. No ML PhD required.

---

## The Data

**Dataset** — The raw files the competition gives you. Usually `train.csv`, `test.csv`, and `sample_submission.csv`.

**Feature (also: input, predictor, column, variable)** — A single piece of information about one row. Example: in a house price competition, "number of bedrooms" is a feature.

**Target (also: label, output, y)** — The thing you're trying to predict. In a house price competition, the target is the actual sale price. Your model learns the relationship between features → target.

**Train set** — The rows where you *know* the target. You use these to teach your model.

**Test set** — The rows where the target is hidden. Your job is to predict it. Kaggle checks your predictions.

**Validation set (also: val set, hold-out set)** — A slice of your training data you temporarily set aside to evaluate your model — before submitting to Kaggle. Lets you catch mistakes without wasting your daily submission limit.

---

## The Competition

**Evaluation metric (also: metric, score)** — How Kaggle measures whether your predictions are good. Common ones: AUC-ROC (classification), RMSE (regression), Log Loss (probability outputs). Always optimize for the competition's exact metric.

**Leaderboard (LB)** — The public ranking of all competitors' submitted scores. Split into:
- **Public LB** — scored on a random subset (~30–50%) of the test set. Updated every submission.
- **Private LB** — scored on the full test set. Only revealed at competition end. This is the final ranking.

**Submission** — A CSV file you upload to Kaggle containing your predictions for the test set. Most competitions allow 2–5 submissions per day.

**Shake-up** — When the final private LB ranking is very different from the public LB. Happens when people overfit to the public LB. A good CV score protects against shake-up.

**Kernel / Notebook** — Code shared publicly on the Kaggle platform. Reading high-scoring public notebooks is a fast way to learn what works.

---

## Modeling

**Model** — A mathematical function that takes features as input and outputs a prediction. You train it on the train set.

**Training (also: fitting)** — Teaching the model by showing it examples and adjusting its internal settings to minimize the prediction error.

**Cross-validation (CV)** — A technique where you split the training data into N groups (folds), train on N-1 of them, and validate on the held-out one. Repeat N times, average the scores. Gives a reliable estimate of real-world performance without touching the test set.

**OOF (Out-of-Fold) predictions** — During cross-validation, the predictions made on each validation fold. Stitching all OOF predictions together gives you a prediction for every training row, which you can use to measure CV score and build ensembles.

**Fold** — One of the N splits in cross-validation. "5-fold CV" means 5 splits, 5 rounds of training.

**Overfitting** — When your model memorizes the training data instead of learning general patterns. Looks great on training data, terrible on new data. Signs: training score >> validation score.

**Underfitting** — When your model is too simple to capture the patterns in the data. Both training and validation scores are poor.

**Early stopping** — A technique that stops training when the validation score stops improving, preventing overfitting. Most gradient boosting libraries support it.

---

## Gradient Boosting (LightGBM, XGBoost, CatBoost)

**LightGBM (LGB)** — A fast, powerful tree-based model. The default starting point for tabular data on Kaggle.

**XGBoost (XGB)** — Similar to LightGBM, slightly slower. Often complements LGB in ensembles because it makes different mistakes.

**CatBoost** — Gradient boosting that handles categorical features natively. Strong when you have many text-like columns.

**n_estimators** — How many trees to build. More trees = slower training, potentially better accuracy. Use early stopping so you don't have to guess.

**learning_rate** — How much each tree corrects the previous ones. Lower = more trees needed, but usually better final result. Typical range: 0.01–0.1.

**num_leaves** — Controls the complexity of each tree. Higher = model can fit more complex patterns but overfits faster. Typical range: 16–128.

---

## Feature Engineering

**Feature engineering** — Creating new columns from existing ones that help the model learn better. Example: from a "timestamp" column, you might extract "day of week" or "hour of day" as new features.

**Feature importance** — A score for each feature indicating how much the model relied on it. Low-importance features are candidates for removal to speed up training.

**Target encoding** — Replacing a categorical value with the average target value for that category. Very powerful, but must be done inside CV folds to avoid data leakage.

**Label encoding** — Replacing categorical text with numbers (e.g., "cat" → 0, "dog" → 1). Simple but loses ordering information.

**One-hot encoding** — Creating a new binary column for each category value. Good for low-cardinality columns (< ~20 unique values).

---

## Ensembling

**Ensemble** — Combining predictions from multiple models to get a better final prediction. Almost always beats any single model.

**Blending** — A simple ensemble: average (or weighted average) the predictions from multiple models.

**Stacking (also: stacked generalization)** — An advanced ensemble where you train a "meta-model" that takes the OOF predictions from your base models as input features.

**Diversity** — How different your models are from each other. High diversity → better ensemble. Two models that always agree add less value than two models that disagree and are both often right.

---

## Common Mistakes (and what they're called)

**Data leakage** — When your model accidentally sees information from the future or from the target. Always inflates CV score but kills LB score.

**LB probing** — Submitting many times to reverse-engineer the test set labels from your LB scores. Against the spirit of Kaggle and increasingly restricted.

**CV–LB gap** — When your CV score is much better than your LB score. Usually means leakage, distribution shift, or a broken CV setup.

**Shake-up** — See above. Final private LB is very different from public LB.

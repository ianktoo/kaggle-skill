# Model Templates

Use this table to pick the right setup for your competition type.

| Problem Type | Fold Strategy | Primary Model | Metric | Key Notes |
|---|---|---|---|---|
| Binary classification | StratifiedKFold(n=5) | LGBMClassifier | AUC-ROC / Log Loss | Use `predict_proba()[:,1]` for OOF |
| Regression | KFold(n=5) | LGBMRegressor | RMSE / MAE | Consider log-transform target if skewed |
| Multi-class | StratifiedKFold(n=5) | LGBMClassifier(objective="multiclass") | Log Loss / F1-macro | OOF shape: (n_train, n_classes) |
| NLP (baseline) | StratifiedKFold(n=5) | TF-IDF + LogisticRegression | AUC-ROC | max_features=50k, ngram_range=(1,2); then fine-tune a transformer |
| NLP (strong) | StratifiedKFold(n=5) | DeBERTa-v3-base / RoBERTa fine-tuned | AUC-ROC / F1 | Tokenizer max_len=512; use mixed precision |
| Computer Vision | StratifiedKFold(n=5) | EfficientNet-B4 or ConvNeXt fine-tuned | AUC-ROC / F1 | ImageNet normalization; TTA at inference |
| Time Series | Time-based split (no leaking future) | LGBMRegressor + lag features | RMSE / SMAPE | NEVER use random KFold — it leaks future into past |

---

## OOF contract — all models must follow this

Every model saves OOF predictions as `{name}_oof.npy` (shape: n_train for binary/regression, n_train×n_classes for multiclass) and test predictions as `{name}_test.npy`. CV score is computed from the OOF array against the true labels before saving. This contract makes ensemble code simple and reproducible — any model that follows it can be dropped into Phase 6 without changes.

---

## Diversity tips

- **Different algorithms:** LightGBM, XGBoost, CatBoost, and Random Forest make different errors — combine them.
- **Different feature sets:** Train one model on all features and another on a curated subset; the predictions will diverge usefully.
- **Different seeds:** Multiple seeds of the same model add modest diversity at low cost; useful when you have few diverse algorithms.
- **Different architectures (CV/NLP):** EfficientNet vs. ConvNeXt, or DeBERTa vs. RoBERTa — architecture diversity is the strongest diversity signal in deep learning competitions.
- **Correlation check before blending:** Compute the OOF correlation matrix. Two models correlated > 0.97 provide almost no ensemble benefit — drop the weaker one and use that slot for something more different.

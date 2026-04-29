# Model Templates

Ready-to-run starter code for the most common Kaggle competition types. Copy the template that matches your problem, update the column names and metric, and run.

---

## Tabular — Binary Classification

**When to use:** Predicting yes/no, fraud/not-fraud, survived/died.
**Default metric:** AUC-ROC

```python
import pandas as pd, numpy as np, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

train = pd.read_csv("train_features.csv")
test  = pd.read_csv("test_features.csv")

TARGET   = "__target__"
FEATURES = [c for c in train.columns if c != TARGET]

X, y    = train[FEATURES], train[TARGET]
X_test  = test[FEATURES]

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(X))
preds = np.zeros(len(X_test))

for fold, (tr, val) in enumerate(kf.split(X, y)):
    model = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.05,
                                num_leaves=31, random_state=42, n_jobs=-1)
    model.fit(X.iloc[tr], y.iloc[tr],
              eval_set=[(X.iloc[val], y.iloc[val])],
              callbacks=[lgb.early_stopping(100, verbose=False),
                         lgb.log_evaluation(200)])
    oof[val] = model.predict_proba(X.iloc[val])[:, 1]
    preds   += model.predict_proba(X_test)[:, 1] / kf.n_splits

print(f"CV AUC: {roc_auc_score(y, oof):.5f}")
np.save("lgb_oof.npy", oof); np.save("lgb_test.npy", preds)
```

---

## Tabular — Regression

**When to use:** Predicting a continuous value (price, count, score).
**Default metric:** RMSE

```python
import pandas as pd, numpy as np, lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

train = pd.read_csv("train_features.csv")
test  = pd.read_csv("test_features.csv")

TARGET   = "__target__"
FEATURES = [c for c in train.columns if c != TARGET]

X, y   = train[FEATURES], train[TARGET]
X_test = test[FEATURES]

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(X))
preds = np.zeros(len(X_test))

for fold, (tr, val) in enumerate(kf.split(X)):
    model = lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.05,
                               num_leaves=31, random_state=42, n_jobs=-1)
    model.fit(X.iloc[tr], y.iloc[tr],
              eval_set=[(X.iloc[val], y.iloc[val])],
              callbacks=[lgb.early_stopping(100, verbose=False),
                         lgb.log_evaluation(200)])
    oof[val]  = model.predict(X.iloc[val])
    preds    += model.predict(X_test) / kf.n_splits

rmse = np.sqrt(mean_squared_error(y, oof))
print(f"CV RMSE: {rmse:.5f}")
np.save("lgb_oof.npy", oof); np.save("lgb_test.npy", preds)
```

---

## Tabular — Multi-class Classification

**When to use:** Predicting one of N categories (3+ classes).
**Default metric:** Log Loss or Macro-F1

```python
import pandas as pd, numpy as np, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

train = pd.read_csv("train_features.csv")
test  = pd.read_csv("test_features.csv")

TARGET   = "__target__"
FEATURES = [c for c in train.columns if c != TARGET]
N_CLASSES = train[TARGET].nunique()

X, y   = train[FEATURES], train[TARGET]
X_test = test[FEATURES]

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof   = np.zeros((len(X), N_CLASSES))
preds = np.zeros((len(X_test), N_CLASSES))

for fold, (tr, val) in enumerate(kf.split(X, y)):
    model = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.05,
                                num_leaves=31, objective="multiclass",
                                num_class=N_CLASSES, random_state=42, n_jobs=-1)
    model.fit(X.iloc[tr], y.iloc[tr],
              eval_set=[(X.iloc[val], y.iloc[val])],
              callbacks=[lgb.early_stopping(100, verbose=False),
                         lgb.log_evaluation(200)])
    oof[val]  = model.predict_proba(X.iloc[val])
    preds    += model.predict_proba(X_test) / kf.n_splits

print(f"CV Log Loss: {log_loss(y, oof):.5f}")
np.save("lgb_oof.npy", oof); np.save("lgb_test.npy", preds)
```

---

## NLP — TF-IDF + LightGBM

**When to use:** Text classification, sentiment analysis, early baseline for NLP competitions.

```python
import pandas as pd, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import scipy.sparse as sp

train = pd.read_csv("./data/train.csv")
test  = pd.read_csv("./data/test.csv")

TEXT_COL = "text"      # update
TARGET   = "label"     # update

# TF-IDF on combined corpus (fit on train only to avoid leakage is safer,
# but combined vocab helps catch rare test terms)
tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1, 2),
                        sublinear_tf=True, strip_accents="unicode",
                        analyzer="word", min_df=2)
tfidf.fit(pd.concat([train[TEXT_COL], test[TEXT_COL]]))

X      = tfidf.transform(train[TEXT_COL])
X_test = tfidf.transform(test[TEXT_COL])
y      = train[TARGET]

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(train)); preds = np.zeros(len(test))

for fold, (tr, val) in enumerate(kf.split(X, y)):
    model = LogisticRegression(C=1.0, max_iter=1000)
    model.fit(X[tr], y.iloc[tr])
    oof[val]  = model.predict_proba(X[val])[:, 1]
    preds    += model.predict_proba(X_test)[:, 1] / kf.n_splits

print(f"CV AUC: {roc_auc_score(y, oof):.5f}")
```

**Next step:** Fine-tune a pretrained model (DeBERTa-v3-base) for significant gains.

---

## Computer Vision — EfficientNet Fine-tune (PyTorch)

**When to use:** Image classification competitions (fire detection, medical imaging, aerial photos).

```python
import torch, timm, pandas as pd, numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

class CompDataset(Dataset):
    def __init__(self, df, img_dir, transform, has_label=True):
        self.df = df; self.img_dir = img_dir
        self.transform = transform; self.has_label = has_label
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(f"{self.img_dir}/{row['image_id']}.jpg").convert("RGB")
        img = self.transform(img)
        if self.has_label:
            return img, torch.tensor(row["label"], dtype=torch.float32)
        return img

TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
VAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

train_df = pd.read_csv("./data/train.csv")
model    = timm.create_model("efficientnet_b0", pretrained=True, num_classes=1)
# Fine-tune with BCEWithLogitsLoss for binary, CrossEntropyLoss for multi-class
```

---

## Time Series — LightGBM with Lag Features

**When to use:** Forecasting competitions (sales, energy, traffic).

```python
import pandas as pd, numpy as np, lightgbm as lgb
from sklearn.metrics import mean_squared_error

train = pd.read_csv("./data/train.csv", parse_dates=["date"])
test  = pd.read_csv("./data/test.csv",  parse_dates=["date"])

TARGET = "sales"  # update

def create_lag_features(df, lags=[1, 7, 14, 28], windows=[7, 28]):
    df = df.sort_values("date").copy()
    for lag in lags:
        df[f"lag_{lag}"] = df[TARGET].shift(lag)
    for w in windows:
        df[f"roll_mean_{w}"] = df[TARGET].shift(1).rolling(w).mean()
        df[f"roll_std_{w}"]  = df[TARGET].shift(1).rolling(w).std()
    # Date features
    df["dayofweek"] = df["date"].dt.dayofweek
    df["month"]     = df["date"].dt.month
    df["day"]       = df["date"].dt.day
    return df.dropna()

train_fe = create_lag_features(train)
FEATURES = [c for c in train_fe.columns if c not in [TARGET, "date", "id"]]

# Use last N days as validation (time-based split — NOT random)
cutoff = train_fe["date"].max() - pd.Timedelta(days=28)
X_tr = train_fe[train_fe["date"] <= cutoff][FEATURES]
y_tr = train_fe[train_fe["date"] <= cutoff][TARGET]
X_val = train_fe[train_fe["date"] > cutoff][FEATURES]
y_val = train_fe[train_fe["date"] > cutoff][TARGET]

model = lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.05, random_state=42)
model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
          callbacks=[lgb.early_stopping(100, verbose=False)])
val_preds = model.predict(X_val)
print(f"Val RMSE: {np.sqrt(mean_squared_error(y_val, val_preds)):.4f}")
```

---

## XGBoost Drop-in Replacement

```python
import xgboost as xgb

model = xgb.XGBClassifier(
    n_estimators=2000,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="auc",
    random_state=42,
    n_jobs=-1,
)
model.fit(X_tr, y_tr,
          eval_set=[(X_val, y_val)],
          early_stopping_rounds=100,
          verbose=200)
```

## CatBoost Drop-in Replacement

```python
from catboost import CatBoostClassifier

cat_features = [i for i, c in enumerate(FEATURES) if train[c].dtype == "object"]

model = CatBoostClassifier(
    iterations=2000,
    learning_rate=0.05,
    depth=6,
    random_seed=42,
    verbose=200,
    early_stopping_rounds=100,
    cat_features=cat_features,
)
model.fit(X_tr, y_tr, eval_set=(X_val, y_val))
```

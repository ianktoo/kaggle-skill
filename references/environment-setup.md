# Environment Setup Guide

Step-by-step environment setup for Kaggle competitions. Confirm each step works before moving to the next.

---

## Step 1 — Check Python Version

```bash
python --version
# or: python3 --version
```

Need **3.9 or higher**. If below 3.9, upgrade before continuing.

If Python isn't installed: Windows — https://www.python.org/downloads/ (check "Add to PATH"); Mac — `brew install python`; Linux — `sudo apt install python3 python3-pip`.

---

## Step 2 — Create a Virtual Environment

```bash
# Create
python -m venv kaggle-env

# Activate — Mac/Linux:
source kaggle-env/bin/activate

# Activate — Windows (Command Prompt):
kaggle-env\Scripts\activate

# Activate — Windows (PowerShell):
kaggle-env\Scripts\Activate.ps1
```

If using conda: `conda create -n kaggle-env python=3.11 && conda activate kaggle-env`

---

## Step 3 — Install Core Packages

```bash
pip install --upgrade pip && pip install pandas numpy scikit-learn matplotlib seaborn jupyter lightgbm xgboost catboost optuna
```

---

## Step 4 — Register Jupyter Kernel

```bash
pip install ipykernel
python -m ipykernel install --user --name=kaggle-env --display-name "Python (kaggle-env)"
```

In the notebook: Kernel → Change Kernel → "Python (kaggle-env)"

---

## Common Blockers

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `python` not found | Python not installed or not in PATH | Install from python.org, check "Add to PATH" |
| `pip` installs to wrong place | venv not activated | Activate venv first, or use `python -m pip install ...` |
| Jupyter kernel missing packages | Notebook using wrong Python | Switch kernel to "Python (kaggle-env)" |
| Windows path errors with backslashes | Shell escaping | Use forward slashes or raw strings in Python paths |
| CUDA/GPU not detected | Driver/CUDA version mismatch | See https://pytorch.org/get-started/locally/ for exact install command |

---

## Error Reference

| Error | Likely cause | Fix |
|-------|-------------|-----|
| `ModuleNotFoundError: No module named 'pandas'` | Wrong Python / venv not active | Activate venv, then `pip install pandas` |
| `pip: command not found` | pip not installed or wrong path | `python -m pip install ...` |
| `Permission denied` on Mac/Linux | File permissions | Add `--user` to pip install |
| Jupyter kernel missing packages | Notebook using wrong Python | Switch kernel to "Python (kaggle-env)" |
| `conda: command not found` | conda not in PATH | Run `conda init` and restart terminal |
| `SSL: CERTIFICATE_VERIFY_FAILED` | Corporate network/proxy | `pip install --trusted-host pypi.org ...` |
| `kaggle: command not found` | Kaggle CLI not installed or not in PATH | `pip install kaggle` then restart terminal |

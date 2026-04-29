# Environment Setup Guide

Step-by-step environment setup for Kaggle competitions. Follow the section for your OS. Confirm each step works before moving to the next — don't skip ahead.

Official docs referenced throughout:
- Python: https://docs.python.org/3/
- venv: https://docs.python.org/3/library/venv.html
- pip: https://pip.pypa.io/en/stable/
- conda: https://docs.conda.io/en/latest/
- Jupyter: https://jupyter.org/documentation
- Kaggle API: https://github.com/Kaggle/kaggle-api

---

## Step 1 — Check Python Version

```bash
python --version
# or on some systems:
python3 --version
```

You need Python **3.9 or higher**. If you see 3.8 or below, upgrade.

**If Python isn't installed:**
- Windows: Download from https://www.python.org/downloads/ — check "Add to PATH" during install
- Mac: `brew install python` (requires Homebrew: https://brew.sh/)
- Linux: `sudo apt install python3 python3-pip`

---

## Step 2 — Create a Virtual Environment

A virtual environment keeps your Kaggle packages isolated from the rest of your system. Always use one.

```bash
# Create a venv named "kaggle-env"
python -m venv kaggle-env

# Activate it:
# Windows (Command Prompt):
kaggle-env\Scripts\activate

# Windows (PowerShell):
kaggle-env\Scripts\Activate.ps1

# Mac / Linux:
source kaggle-env/bin/activate
```

After activation, your prompt should show `(kaggle-env)` at the start.

**Verify:**
```bash
which python       # Mac/Linux — should point to kaggle-env/bin/python
where python       # Windows — should point to kaggle-env\Scripts\python.exe
```

**Docs:** https://docs.python.org/3/library/venv.html

---

## Step 3 — Install Core Packages

With your venv active:

```bash
pip install --upgrade pip
pip install pandas numpy scikit-learn matplotlib seaborn jupyter lightgbm xgboost catboost optuna
```

This takes 2–5 minutes. Verify:

```bash
python -c "import pandas, numpy, sklearn, lightgbm, xgboost; print('All good!')"
```

**Common issue — pip installs to wrong environment:**
```bash
# Check which pip you're using
which pip    # Mac/Linux
where pip    # Windows
# It must point inside kaggle-env, not a system path
# If wrong: use `python -m pip install ...` instead of just `pip install ...`
```

---

## Step 4 — Set Up Jupyter

```bash
# Install Jupyter (already done above, but verify)
pip show jupyter

# Register your venv as a Jupyter kernel so notebooks use the right Python
pip install ipykernel
python -m ipykernel install --user --name=kaggle-env --display-name "Python (kaggle-env)"

# Launch Jupyter
jupyter notebook
# or
jupyter lab
```

**In the notebook:** Select Kernel → Change Kernel → "Python (kaggle-env)"

**Docs:** https://jupyter.org/documentation

**Common issue — kernel not showing up:**
```bash
# List available kernels
jupyter kernelspec list
# If kaggle-env is missing, re-run the ipykernel install command above
```

**Common issue — Jupyter opens but packages missing:**
This means your notebook is using a different Python than your venv.
Fix: make sure you selected the "Python (kaggle-env)" kernel, not the default.

---

## Step 5 — Conda Alternative (if you prefer Anaconda)

If you use Anaconda or Miniconda:

```bash
# Create env
conda create -n kaggle-env python=3.11

# Activate
conda activate kaggle-env

# Install packages
conda install pandas numpy scikit-learn matplotlib seaborn jupyter
pip install lightgbm xgboost catboost optuna   # use pip for these

# Register kernel
python -m ipykernel install --user --name=kaggle-env --display-name "Python (kaggle-env)"
```

**Docs:** https://docs.conda.io/en/latest/

**Common issue — conda and pip conflicts:**
Install as much as possible with conda, then use pip only for packages conda doesn't have. Never mix conda and pip channels without care. See: https://www.anaconda.com/blog/using-pip-in-a-conda-environment

---

## Step 6 — VS Code Setup (optional but recommended)

If using VS Code:

1. Install the Python extension: https://marketplace.visualstudio.com/items?itemName=ms-python.python
2. Install the Jupyter extension: https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter
3. Open your project folder: `code .`
4. Select interpreter: Ctrl+Shift+P → "Python: Select Interpreter" → choose `kaggle-env`
5. Open a `.py` file — the status bar bottom-left shows the active interpreter

**Docs:** https://code.visualstudio.com/docs/python/python-tutorial

---

## Step 7 — PyCharm Setup (optional)

1. Open project folder in PyCharm
2. File → Settings → Project → Python Interpreter → Add Interpreter
3. Select "Existing environment" → browse to `kaggle-env/bin/python` (Mac/Linux) or `kaggle-env\Scripts\python.exe` (Windows)
4. Apply and OK

**Docs:** https://www.jetbrains.com/help/pycharm/creating-virtual-environment.html

---

## GPU Setup (for Computer Vision / NLP competitions)

If you have an NVIDIA GPU and are running CV or NLP competitions:

```bash
# Check if CUDA is available
python -c "import torch; print(torch.cuda.is_available())"
# or
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

**PyTorch with CUDA:**
Go to https://pytorch.org/get-started/locally/ and use the selector to get the exact install command for your CUDA version.

**TensorFlow with GPU:**
See https://www.tensorflow.org/install/pip#nvidia_gpu

GPU setup is highly version-sensitive. If you hit CUDA errors, paste the full error message and your GPU model + OS.

---

## Common Error Reference

| Error | Likely cause | Fix |
|-------|-------------|-----|
| `ModuleNotFoundError: No module named 'pandas'` | Wrong Python / venv not active | Activate venv, then `pip install pandas` |
| `pip: command not found` | pip not installed or wrong path | `python -m pip install ...` |
| `Permission denied` on Mac/Linux | File permissions | Add `--user` to pip install, or use sudo (not recommended in venv) |
| Jupyter kernel missing packages | Notebook using wrong Python | Switch kernel to "Python (kaggle-env)" |
| `conda: command not found` | conda not in PATH | Run conda init and restart terminal |
| `SSL: CERTIFICATE_VERIFY_FAILED` | Corporate network/proxy | `pip install --trusted-host pypi.org ...` |
| `kaggle: command not found` | Kaggle CLI not installed or not in PATH | `pip install kaggle` then restart terminal |

---

## Quick Verification Script

Run this to confirm everything is working:

```python
import sys, pandas, numpy, sklearn, lightgbm, xgboost

print(f"Python:     {sys.version}")
print(f"pandas:     {pandas.__version__}")
print(f"numpy:      {numpy.__version__}")
print(f"sklearn:    {sklearn.__version__}")
print(f"lightgbm:   {lightgbm.__version__}")
print(f"xgboost:    {xgboost.__version__}")
print("\n✅ Environment ready for Kaggle!")
```

If any import fails, install the missing package with `pip install <package-name>` and re-run.

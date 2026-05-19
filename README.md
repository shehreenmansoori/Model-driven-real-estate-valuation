# Valora AI - Sales Price Prediction v0.1

A production-style Streamlit project that predicts house prices from tabular features, explains feature impact with SHAP, and adds plain-English AI explanations via OpenRouter.

## Who This Is For

- Students who want to present an end-to-end ML project to recruiters.
- Developers who need a reliable setup flow on a new device.
- Anyone learning model training, interpretability, and app deployment basics.

## Project Summary

This app takes 5 house features:

- `GrLivArea`
- `BedroomAbvGr`
- `FullBath`
- `YearBuilt`
- `GarageArea`

It predicts `SalePrice` using a trained `RandomForestRegressor`, then:

- computes confidence range from tree-level spread,
- shows SHAP contribution bars,
- retrieves similar homes,
- answers "why" questions through a concise AI layer.

## Tech Stack

- Python `3.11` or `3.12`
- Streamlit
- scikit-learn (`LinearRegression`, `RandomForestRegressor`)
- XGBoost (`XGBRegressor`)
- SHAP
- Plotly
- pandas, numpy, joblib
- python-dotenv
- requests

## Repository Structure

```text
Sales Price Prediction v0.1/
  app/
    app.py
  data/
    house_prices.csv
  model/
    train_model.py
    saved_model.pkl
    metrics.pkl
    features.pkl
  utils/
    explainer.py
    openrouter_api.py
  scripts/
    setup_windows.ps1
    run_app.ps1
  Project_Explainer_Guide.md
  generate_project_guide_pdf.py
  requirements.txt
```

## New Device Setup (Windows, Recommended)

Use this exact flow on your college laptop.

### 1. Copy files correctly

Copy the project folder, but do not copy:

- `.venv/`
- `__pycache__/`

### 2. Install Python

Install Python `3.11.x` or `3.12.x` from python.org.

### 3. Open PowerShell in project root

```powershell
cd "C:\path\to\Sales Price Prediction v0.1"
```

### 4. (If needed) allow script execution in current shell only

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### 5. Run automated setup

```powershell
.\scripts\setup_windows.ps1
```

This will:

- create `.venv`
- validate Python version
- upgrade pip/setuptools/wheel
- install `requirements.txt`
- create `.env` from `.env.example` (if missing)

### 6. Add your API key

Edit `.env`:

```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

### 7. Run the app

```powershell
.\scripts\run_app.ps1
```

If port `8501` is busy, the script auto-selects next free port.

## Verify Virtual Environment Is Correct

Run these checks after setup:

```powershell
.\.venv\Scripts\Activate.ps1
python --version
python -c "import sys; print(sys.executable)"
pip --version
```

Expected:

- Python is `3.11.x` or `3.12.x`
- executable path points to `.venv\Scripts\python.exe`
- pip path resolves inside `.venv`

## Most Common Setup Errors and Fixes

### 1) Unsupported Python version

Symptom:

- "Unsupported Python ... Use Python 3.11 or 3.12"

Fix:

- install Python 3.11/3.12
- rerun setup script

### 2) PowerShell blocks scripts

Symptom:

- execution policy error

Fix:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### 3) Dependency install fails

Typical causes:

- unstable network
- blocked index/proxy
- outdated pip tools

Fix sequence:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-cache-dir -r requirements.txt
```

If still failing:

- retry on stable internet or hotspot
- temporarily disable strict VPN/proxy policies

### 4) Model artifact missing

Symptom:

- missing `saved_model.pkl` or related files

Fix:

```powershell
python model\train_model.py
```

or simply run:

```powershell
.\scripts\run_app.ps1
```

and let auto-train run.

### 5) OpenRouter explanation not working

Symptom:

- AI responses fallback to local template or feel generic

Fix:

- verify `OPENROUTER_API_KEY` in `.env`
- confirm internet access
- check key validity/credits/rate limits

Note: Core ML prediction still works even when API fails.

## Manual Setup (Any OS)

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-cache-dir -r requirements.txt

# Add OPENROUTER_API_KEY to .env
python model/train_model.py
python -m streamlit run app/app.py
```

## How The System Works

1. Load and validate dataset (`data/house_prices.csv`).
2. Train and compare 3 regressors in `model/train_model.py`.
3. Save final artifacts (`saved_model.pkl`, `metrics.pkl`, `features.pkl`).
4. Streamlit app loads artifacts and initializes UI/session state.
5. User edits feature sliders or preset scenarios.
6. App predicts price + confidence interval.
7. SHAP computes per-feature contribution.
8. Similar homes are found by normalized distance.
9. AI layer explains results and answers follow-up questions.

## Recruiter Demo Talk Track (60-90 seconds)

- "This is an end-to-end regression product, not just a notebook."
- "I compare Linear Regression, Random Forest, and XGBoost using R2, RMSE, MAE."
- "I deploy Random Forest for robustness and compute uncertainty from tree prediction spread."
- "I use SHAP for feature-level explanation so users know why the estimate changed."
- "I added a fallback-safe AI explanation layer: if API fails, local deterministic answers still work."
- "The app is portable with scripted setup and strict Python version checks."

## Generate the Interview PDF Guide

```powershell
.\.venv\Scripts\Activate.ps1
python generate_project_guide_pdf.py
```

Output:

- `Project_Explainer_Guide.pdf`

## Quick Commands

```powershell
# First-time setup
.\scripts\setup_windows.ps1

# Run app
.\scripts\run_app.ps1

# Force retrain model
python model\train_model.py

# Dependency stress test
python scripts\stress_test.py
```

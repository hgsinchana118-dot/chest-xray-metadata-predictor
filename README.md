# Chest X-Ray Metadata Predictor

A Flask web app that predicts likely chest X-ray findings from patient
**metadata** - age, gender, and view position - using a Naive Bayes
model built with pandas (for aggregation) and core Python (for the
probability math). It does not process X-ray images.

Based on the structure of the [NIH Chest X-rays dataset](https://www.kaggle.com/datasets/nih-chest-xrays/data).

## Project layout

```
chest-xray-predictor/
├── app.py                        Flask routes
├── data/
│   ├── generate_sample_data.py   builds a synthetic metadata CSV
│   └── sample_metadata.csv       generated on first run
├── model/
│   ├── predictor.py              pandas aggregation + core-Python Naive Bayes
│   └── probability_tables.json   generated on first run (the "trained model")
├── templates/                    Jinja2 pages
└── static/css/style.css
```

## Setup (Windows / PowerShell)

```powershell
cd chest-xray-predictor
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Then open **http://127.0.0.1:5000** in a browser.

> If PowerShell blocks the activation script with an execution-policy
> error, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`
> in that same window first, then retry `venv\Scripts\Activate.ps1`.

First run auto-generates `data/sample_metadata.csv` (4,000 synthetic
patients) and trains the model into `model/probability_tables.json`.
Delete either file and restart to regenerate.

## Training on the real NIH dataset

The app ships trained on synthetic data so it runs without a 42GB
download. To use the real dataset instead:

1. Download `Data_Entry_2017.csv` from the
   [Kaggle dataset page](https://www.kaggle.com/datasets/nih-chest-xrays/data)
   (you only need this CSV, not the image archives).
2. Save it as `data/sample_metadata.csv`, overwriting the sample file.
3. Delete `model/probability_tables.json` and restart `python app.py`
   (or POST to `/retrain` while the app is running) to retrain.

The real file's columns line up with what `model/predictor.py` expects:
`Patient Age`, `Patient Gender`, `View Position`, `Finding Labels`.

## How the prediction works

1. **pandas** loads the CSV, bins age into ranges, splits the
   pipe-separated `Finding Labels` column, and counts how often each
   finding co-occurs with each age band / gender / view position.
2. Those counts become Laplace-smoothed conditional probability
   tables, saved to `model/probability_tables.json`.
3. At request time, `model/predictor.py`'s `predict()` function
   multiplies the relevant probabilities together by hand (Bayes'
   rule, plain Python - no scikit-learn) and normalises the results
   into a ranked list.

See the in-app **About the model** page for limitations - this is a
portfolio/educational project, not a diagnostic tool.

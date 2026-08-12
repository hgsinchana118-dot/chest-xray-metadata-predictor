"""
Flask web app: predicts likely chest-X-ray findings from patient
metadata (age, gender, view position) using a from-scratch Naive Bayes
model trained on NIH Chest X-ray metadata with pandas.

IMPORTANT
---------
- Ships trained on SYNTHETIC sample data by default (see data/generate_sample_data.py).
- Even trained on the real dataset, this reads metadata only (age/gender/
  view position) - it never looks at an actual X-ray image.
- This is a portfolio/educational project, not a diagnostic tool. See /about.
"""
import os

from flask import Flask, render_template, request

from data.generate_sample_data import generate as generate_sample_data
from model import predictor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "sample_metadata.csv")
TABLES_PATH = os.path.join(BASE_DIR, "model", "probability_tables.json")

app = Flask(__name__)

_tables_cache = None


def get_tables():
    """Load the trained probability tables, building them (and the sample
    data, if needed) on first use. Cached in memory after that."""
    global _tables_cache
    if _tables_cache is None:
        if not os.path.exists(DATA_PATH):
            generate_sample_data(out_path=DATA_PATH)
        if not os.path.exists(TABLES_PATH):
            predictor.train(DATA_PATH, out_path=TABLES_PATH)
        _tables_cache = predictor.load_tables(TABLES_PATH)
    return _tables_cache


@app.route("/")
def index():
    tables = get_tables()
    return render_template("index.html", n_patients=tables["n_patients"])


@app.route("/predict", methods=["POST"])
def predict():
    tables = get_tables()

    try:
        age = int(request.form["age"])
        gender = request.form["gender"]
        view = request.form["view"]
    except (KeyError, ValueError):
        return render_template("index.html", error="Please fill in every field with a valid value.",
                                n_patients=tables["n_patients"])

    if not (0 < age < 120) or gender not in predictor.GENDERS or view not in predictor.VIEWS:
        return render_template("index.html", error="Please check your inputs and try again.",
                                n_patients=tables["n_patients"])

    results = predictor.predict(tables, age, gender, view)

    return render_template(
        "result.html",
        age=age, gender=gender, view=view,
        results=results[:8],
    )


@app.route("/about")
def about():
    tables = get_tables()
    return render_template("about.html", n_patients=tables["n_patients"], n_diseases=len(predictor.DISEASES))


@app.route("/retrain", methods=["POST"])
def retrain():
    """Rebuild the probability tables from whatever CSV is currently at
    data/sample_metadata.csv - use this after dropping in the real
    Data_Entry_2017.csv (renamed to match) to retrain on real data."""
    global _tables_cache
    predictor.train(DATA_PATH, out_path=TABLES_PATH)
    _tables_cache = None
    return {"status": "retrained"}, 200


if __name__ == "__main__":
    app.run(debug=True)

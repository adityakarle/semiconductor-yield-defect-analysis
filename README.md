# Semiconductor Manufacturing Yield & Defect Analysis

A defect-classification pipeline over **real semiconductor fab sensor data**, handling
high-dimensional, imbalanced process measurements to predict yield failures, plus a
dashboard for tracking yield trends and the sensors that drive them.

## Data

Source: **[UCI SECOM dataset](https://archive.ics.uci.edu/dataset/179/secom)** —
real process measurements collected from 590 sensors on an actual semiconductor
manufacturing line, with a pass/fail label per unit. No API key, no auth, no
synthetic data: **1,567 units**, **590 anonymous sensor readings each**, and a
**6.6% fail rate**  genuinely imbalanced, genuinely messy (heavy missingness,
many dead sensors, strong collinearity between sensors on the same process step).

## Architecture

```
UCI SECOM files    ─▶  extract.py    (cached download)
                   ─▶  transform.py  (drop high-missing / near-constant / collinear
                                       sensors, median-impute)
                   ─▶  features.py   (Random Forest importance ranking, top-K select)
                   ─▶  model.py      (LogisticRegression / RandomForest / XGBoost,
                                       class-imbalance handling, PR-AUC selection)
                   ─▶  pipeline.py   (orchestrates the full run)
                   ─▶  app/dashboard.py  (Streamlit UI over all of the above)
```

## Tech stack

Python · pandas · scikit-learn · XGBoost · Streamlit · Plotly · pytest · GitHub Actions

## Results

From the real SECOM run (`data/processed/metrics.json`):

| Sensors (raw → after cleaning → selected) | Fail rate | Best model | PR-AUC | ROC-AUC |
|---|---|---|---|---|
| 590 → 264 → 40 | 6.6% | Random Forest | 0.31 | 0.82 |

PR-AUC (average precision), not accuracy, is the headline metric — with a ~93/7
class split, a model that always predicts "pass" scores 93% accuracy while
catching zero defects. PR-AUC measures how well the model ranks the rare
failing units, which is what actually matters for a fab quality team deciding
which units to re-inspect. Logistic Regression and XGBoost were also trained
and compared; see the dashboard's "Model comparison" panel for full numbers.

## Cleaning, feature selection, and modeling decisions

See [docs/DATA.md](docs/DATA.md) for the full write-up of each cleaning step
(missingness thresholds, dead-sensor removal, collinearity pruning), the
feature-selection method, and why PR-AUC drives model selection.

## Prerequisites

- Python 3.10+
- Internet access on first run (to download the SECOM sensor + label files, ~5MB)

## Run it

```bash
python -m venv .venv
source .venv/bin/activate   # .venv\Scripts\activate on Windows
pip install -r requirements.txt

python -m src.pipeline
streamlit run app/dashboard.py
```

Then open http://localhost:8501.

Useful pipeline flags:

```bash
python -m src.pipeline --refresh   # force re-download of source files
```

### Running with Docker

```bash
docker build -t secom-yield .
docker run -p 8501:8501 secom-yield
```

Note: the container still needs `python -m src.pipeline` run once (inside the
container, or by mounting a pre-built `data/processed/`) before the dashboard
has data to show.

## Dashboard

- **KPI row**: unit count, fail rate, sensors kept vs. raw, best model's PR-AUC.
- **Yield trend**: daily unit volume and a 7-day rolling fail rate over time.
- **Sensor importances**: top-N sensors ranked by predictive power, adjustable.
- **Model comparison**: PR-AUC / ROC-AUC / precision / recall / F1 for all three
  candidate models, plus the winning model's precision-recall curve.
- **Sensor drill-down**: pick any top sensor and see its value distribution
  split by pass/fail outcome.

## Tests

```bash
pytest -v
```

Tests cover the cleaning logic (missingness/variance/collinearity filters),
feature-selection ranking, and that model training beats the class base rate
no network access required to run them (they use small synthetic frames).

## Project layout

```
src/
  config.py     source URLs, cleaning thresholds, paths
  extract.py    cached SECOM file downloads
  transform.py  cleaning: missingness, dead sensors, imputation, collinearity
  features.py   Random Forest importance ranking + top-K selection
  model.py      LogisticRegression / RandomForest / XGBoost + PR-AUC evaluation
  pipeline.py   orchestrates the full run
app/
  dashboard.py  Streamlit UI
tests/
```

## Documentation

Full data dictionary and pipeline design notes: [docs/DATA.md](docs/DATA.md).

## Notes on the data

This project uses the public, anonymized UCI SECOM release sensor columns
carry no semantic labels in the public data (only the fab's own process
engineers know what each one measures), and no unit- or company-identifying
information is included.

## License

[MIT](LICENSE)

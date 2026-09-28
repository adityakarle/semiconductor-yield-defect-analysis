# Data dictionary

## Source

[UCI SECOM dataset](https://archive.ics.uci.edu/dataset/179/secom) — real process
measurements collected from sensors during semiconductor manufacturing, with a
pass/fail label per unit. Published by UCI ML Repository, no auth or API key
required.

- `secom.data` — 1567 units x 590 anonymous sensor readings, whitespace-delimited,
  missing values marked `NaN`.
- `secom_labels.data` — 1567 rows of `label timestamp`, where `label` is `-1`
  (pass) or `1` (fail), and `timestamp` is the collection time.

Sensor columns carry no semantic names in the public release (the fab's process
engineers know what each one measures; the public dataset anonymizes them as
`sensor_0` .. `sensor_589`).

## Cleaning pipeline (`src/transform.py`)

1. **Drop high-missingness sensors** — any sensor missing on more than 45% of
   units is dropped rather than imputed, since a mostly-absent reading usually
   means the sensor wasn't wired into that process step at all.
2. **Drop near-constant sensors** — sensors with variance below `1e-6` are
   dead/non-informative and dropped.
3. **Median-impute** remaining missing values, robust to the outliers common in
   raw sensor data.
4. **Drop collinear sensors** — for any pair of sensors correlated above 0.95,
   one is dropped, since duplicated signal only adds noise to feature selection.

## Feature selection (`src/features.py`)

A Random Forest (`class_weight="balanced"`) is fit over the cleaned sensor set
and its impurity-based importances rank all sensors. The top 40 (configurable
via `TOP_K_FEATURES` in `src/config.py`) feed the classifier.

## Label

`fail` = 1 means the unit failed final test (the rarer class, ~7% of units);
`fail` = 0 means it passed.

## Modeling (`src/model.py`)

Three candidates — logistic regression, random forest, and XGBoost — are all
trained with class-imbalance handling (`class_weight="balanced"` /
`scale_pos_weight`) and compared on **PR-AUC (average precision)**, since
accuracy is meaningless on a ~93/7 class split. The highest PR-AUC model is
kept and its precision-recall curve, confusion matrix, and feature importances
are saved to `data/processed/` for the dashboard.

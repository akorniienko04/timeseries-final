# MATH840 Final project: 48 hours of generation

A forecast of the hourly electricity generation for 27 Oct 2026 01:00 – 29 Oct 2026 00:00, with a 95%
prediction interval. Author: Artem Korniienko.

## What is here

| File | What it is |
|---|---|
| `KORNIIENKO_project.csv` | the final submission: 48 rows, `ds, forecast, lower_95, upper_95` |
| `forecast.py` | all the code: features, models, rolling validation, interval, submission |
| `report.ipynb` | the report: the four questions of the brief, with the code and its output |
| `make_slides.py` | builds `slides.pdf` from the validation results and the report's figures |
| `slides.pdf` | the slides for Session 2 |
| `data/` | `train.csv` and `weather_forecast.csv`, as given on Moodle |
| `results/validation.csv` | every forecast on every validation window (written by `forecast.py`) |
| `figures/` | the figures from the report |

## How to run

Tested with Python 3.14 and the versions in `requirements.txt`.

```bash
pip install -r requirements.txt
python forecast.py                                     # validation (~9 min) + KORNIIENKO_project.csv
jupyter nbconvert --to notebook --execute --inplace report.ipynb   # the report and its figures
python make_slides.py                                  # slides.pdf
```

`forecast.py` alone rebuilds the submission file from the two CSVs in `data/`. The notebook reads
`results/validation.csv`; if it is missing, the notebook runs the validation itself. Everything uses a
fixed random seed, so a rerun gives the same file.

## The model in short

* **Point forecast:** the average of two `HistGradientBoostingRegressor` models, one on `y` and one on
  `log y`. Features: the 7 weather columns, hour, day of year, and last week's `y` at the same hour (mean,
  maximum, lag 48). All lags are 48 hours or more, so every feature is known for the whole horizon.
* **95% interval:** the forecast plus the 2.5% and 97.5% quantiles of the validation errors, taken separately
  for day 1 / day 2 of the horizon and for daylight / night hours. The lower bound is clipped at 0.
* **Validation:** 40 windows of 48 hours (starting 8 Aug – 25 Oct 2026, each at midnight), with the model refitted
  for each one. The weather inputs get simulated forecast errors that grow from day 1 to day 2.

| | RMSE | MAPE | Winkler | coverage |
|---|---|---|---|---|
| seasonal naive (benchmark) | 269 | 153% | 2397 | 75% |
| this model | 243 | 123% | 1171 | 92% |

RMSE and MAPE are averaged over the 40 windows. Winkler and coverage come from the last 20 windows, with the
interval calibrated on the first 20.

## LLM use


I used Claude (Anthropic) for:

- `make_slides.py`, `README.md`
- `forecast.py`, to check the structure of the code and format it correctly;
- `report.ipynb`, to make the text cells clear and grammatically correct, and to check that the code is correct.
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

SURNAME = "KORNIIENKO"
ROOT = Path(__file__).parent
DATA = ROOT / "data"
H = 48 
M = 24   
SEED = 0

WEATHER = ["cloud_cover", "temperature", "wind_speed", "pressure", "humidity", "daylight", "air_density"]


PROFILE_LAGS = [48 + 24 * k for k in range(7)]
LEVEL = ["profile7", "max7", "lag48"]
CALENDAR = ["hour", "doy"]



def load_data():
    data = pd.read_csv(DATA / "train.csv", parse_dates=["ds"])
    weather_fc = pd.read_csv(DATA / "weather_forecast.csv", parse_dates=["ds"])
    return data, weather_fc


def rmse(actual, forecast) -> float:
    return float(np.sqrt(np.mean((np.asarray(actual) - np.asarray(forecast)) ** 2)))


def mape(actual, forecast) -> float:
    actual, forecast = np.asarray(actual, float), np.asarray(forecast, float)
    return float(np.mean(np.abs((actual - forecast) / actual)) * 100)


def winkler(actual, lower, upper, alpha: float = 0.05) -> float:
    actual, lower, upper = (np.asarray(a, float) for a in (actual, lower, upper))
    below = np.clip(lower - actual, 0, None)
    above = np.clip(actual - upper, 0, None)
    return float(np.mean((upper - lower) + 2 / alpha * (below + above)))


def score(actual, forecast, lower, upper) -> dict:

    inside = (np.asarray(actual) >= lower) & (np.asarray(actual) <= upper)

    return {"RMSE": rmse(actual, forecast), "MAPE": mape(actual, forecast),
            "Winkler": winkler(actual, lower, upper), "coverage": float(inside.mean())}


def snaive(history: np.ndarray, h: int = H, m: int = M):
    
    history = np.asarray(history, float)
    point = np.array([history[-m + (i % m)] for i in range(h)])
    sigma = np.std(history[m:] - history[:-m], ddof=1)
    k = np.arange(h) // m
    half = 1.96 * sigma * np.sqrt(k + 1)
    return point, np.clip(point - half, 0, None), point + half



def make_features(frame: pd.DataFrame) -> pd.DataFrame:

    X = frame[WEATHER].copy()
    X["hour"] = frame["ds"].dt.hour
    X["doy"] = frame["ds"].dt.dayofyear

    lags = np.column_stack([frame["y"].shift(L) for L in PROFILE_LAGS])

    X["profile7"] = lags.mean(axis = 1)
    X["max7"] = lags.max(axis = 1)
    X["lag48"] = frame["y"].shift(48)

    return X


def fit_model(X: pd.DataFrame, y: pd.Series) -> HistGradientBoostingRegressor:

    model = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                          min_samples_leaf=40, l2_regularization=1.0, random_state=SEED)
    return model.fit(X, y)


def gbm_forecast(history: pd.DataFrame, future_weather: pd.DataFrame,
                 log_target: bool = False, use_weather: bool = True) -> np.ndarray:
    

    frame = pd.concat([history, future_weather.assign(y=np.nan)], ignore_index=True)
    X = make_features(frame)
    X = X[(WEATHER if use_weather else []) + CALENDAR + LEVEL]
    target = frame["y"]

    if log_target:
        X = X.assign(**{c: np.log1p(X[c]) for c in LEVEL})
        target = np.log1p(target)

    train = frame["y"].notna() & X["profile7"].notna()
    pred = fit_model(X[train], target[train]).predict(X.iloc[len(history):])
    return np.clip(np.expm1(pred) if log_target else pred, 1, None)


def ensemble_forecast(history: pd.DataFrame, future_weather: pd.DataFrame) -> np.ndarray:
    
    return (gbm_forecast(history, future_weather) + gbm_forecast(history, future_weather, log_target=True)) / 2


def profile_forecast(history: pd.DataFrame, h: int = H) -> np.ndarray:

    y = history["y"].to_numpy(float)
    week = y[-7 * M:].reshape(7, M).mean(axis=0)
    return np.array([week[i % M] for i in range(h)])



NOISE = {"temperature": (1.0, 2.3), "pressure": (0.3, 1.8), "wind_speed": (0.5, 0.5),
         "humidity": (3.0, 3.0), "air_density": (0.010, 0.012)}
         
CLOUD_FLIP = (0.15, 0.35) 


def add_forecast_noise(weather: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:

    rng = np.random.default_rng(seed)
    w = weather.copy().reset_index(drop=True)
    day2 = np.arange(len(w)) >= M
    for col, (sd1, sd2) in NOISE.items():
        w[col] = w[col] + rng.normal(0, 1, len(w)) * np.where(day2, sd2, sd1)
    w["wind_speed"] = w["wind_speed"].clip(0)
    w["humidity"] = w["humidity"].clip(0, 100)
    flip = rng.random(len(w)) < np.where(day2, CLOUD_FLIP[1], CLOUD_FLIP[0])
    w.loc[flip, "cloud_cover"] = rng.choice([0, 0.25, 0.5, 0.75, 1.0], flip.sum())
    return w


def validation_origins(data: pd.DataFrame, n: int = 40, step_days: int = 2) -> list:

    last = len(data) - H
    return [last - step_days * M * k for k in range(n)][::-1]


def run_validation(data: pd.DataFrame, origins: list) -> pd.DataFrame:

    rows = []
    for o in origins:
        hist, test = data.iloc[:o], data.iloc[o:o + H]
        actual_weather = test[["ds"] + WEATHER]
        noisy_weather = add_forecast_noise(actual_weather, seed=o)
        out = {"origin": data["ds"].iloc[o - 1], "ds": test["ds"].to_numpy(), "y": test["y"].to_numpy(),
               "step": np.arange(1, H + 1), "daylight": test["daylight"].to_numpy()}

        out["snaive"], out["snaive_lo"], out["snaive_hi"] = snaive(hist["y"].to_numpy())
        out["profile7"] = profile_forecast(hist)
        out["gbm_no_weather"] = gbm_forecast(hist, noisy_weather, use_weather=False)
        out["gbm_actual_weather"] = gbm_forecast(hist, actual_weather)
        out["gbm"] = gbm_forecast(hist, noisy_weather)
        out["gbm_log"] = gbm_forecast(hist, noisy_weather, log_target=True)
        out["ensemble"] = (out["gbm"] + out["gbm_log"]) / 2
        rows.append(pd.DataFrame(out))

    return pd.concat(rows, ignore_index=True)


def window_scores(val: pd.DataFrame, method: str, lower: str = None, upper: str = None) -> pd.DataFrame:

    def one(g):
        s = {"RMSE": rmse(g["y"], g[method]), "MAPE": mape(g["y"], g[method])}
        if lower:
            s.update(score(g["y"], g[method], g[lower], g[upper]))
        return pd.Series(s)

    return val.groupby("origin").apply(one)


def point_scores(val: pd.DataFrame, methods: list) -> pd.DataFrame:
    return pd.DataFrame({m: window_scores(val, m).mean() for m in methods}).T




def interval_quantiles(val: pd.DataFrame, method: str = "ensemble") -> pd.DataFrame:

    e = val.assign(err=val["y"] - val[method], day=np.where(val["step"] > M, 2, 1))

    return e.groupby(["day", "daylight"])["err"].quantile([0.025, 0.975]).unstack()


def apply_interval(point, steps, daylight, q: pd.DataFrame):

    day = np.where(np.asarray(steps) > M, 2, 1)
    keys = list(zip(day, np.asarray(daylight)))
    point = np.asarray(point, float)
    lower = np.clip(point + [q.loc[k, 0.025] for k in keys], 0, None)
    upper = point + [q.loc[k, 0.975] for k in keys]

    return np.minimum(lower, point), np.maximum(upper, point)


def interval_check(val: pd.DataFrame, method: str = "ensemble", n_cal: int = 20) -> pd.DataFrame:

    origins = sorted(val["origin"].unique())
    cal, test = val[val["origin"].isin(origins[:n_cal])], val[val["origin"].isin(origins[n_cal:])].copy()
    test["lo"], test["hi"] = apply_interval(test[method], test["step"], test["daylight"],
                                            interval_quantiles(cal, method))

    return pd.DataFrame({method: window_scores(test, method, "lo", "hi").mean(),
                         "snaive": window_scores(test, "snaive", "snaive_lo", "snaive_hi").mean()}).T





def final_forecast(data: pd.DataFrame, weather_fc: pd.DataFrame, val: pd.DataFrame) -> pd.DataFrame:
    point = ensemble_forecast(data, weather_fc)
    lower, upper = apply_interval(point, np.arange(1, H + 1), weather_fc["daylight"].to_numpy(),
                                  interval_quantiles(val, "ensemble"))
    sub = pd.DataFrame({"ds": weather_fc["ds"].dt.strftime("%Y-%m-%d %H:%M:%S"), "forecast": point,
                        "lower_95": lower, "upper_95": upper})
    return sub


def main():
    data, weather_fc = load_data()
    val = run_validation(data, validation_origins(data))
    (ROOT / "results").mkdir(exist_ok=True)
    val.to_csv(ROOT / "results" / "validation.csv", index=False)
    print(point_scores(val, ["snaive", "profile7", "gbm_no_weather", "gbm_actual_weather",
                             "gbm", "gbm_log", "ensemble"]).round(2))
    print(interval_check(val).round(3))
    sub = final_forecast(data, weather_fc, val)
    out = ROOT / f"{SURNAME}_project.csv"
    sub.to_csv(out, index=False)
    print("saved", out.name)


if __name__ == "__main__":
    main()

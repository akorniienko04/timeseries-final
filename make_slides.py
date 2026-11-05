import textwrap
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

import forecast as F

ROOT = Path(__file__).parent
FIG = ROOT / "figures"
INK, MUTED, ACCENT = "#1f2933", "#6b7280", "#1f6fb2"
W, Hh = 13.33, 7.5


def new_slide(pdf_title, title, subtitle=None):
    fig = plt.figure(figsize=(W, Hh))
    fig.text(0.05, 0.92, title, fontsize=26, weight="bold", color=INK, va="top")
    if subtitle:
        fig.text(0.05, 0.845, subtitle, fontsize=14, color=MUTED, va="top")
    fig.text(0.95, 0.03, pdf_title, fontsize=9, color=MUTED, ha="right")
    return fig


def bullets(fig, items, x=0.05, y=0.76, width=60, size=15, para=0.035):
    line_height = size * 1.55 / 72 / Hh          # in figure fractions
    for item in items:
        for i, line in enumerate(textwrap.wrap(item, width)):
            fig.text(x + (0.018 if i else 0), y, ("" if i else "•  ") + line, fontsize=size, color=INK, va="top")
            y -= line_height
        y -= para


def image(fig, path, rect):
    ax = fig.add_axes(rect)
    ax.imshow(mpimg.imread(path))
    ax.axis("off")


def table(fig, df, rect, size=13, highlight=None):
    ax = fig.add_axes(rect)
    ax.axis("off")
    t = ax.table(cellText=df.values, colLabels=list(df.columns), rowLabels=list(df.index),
                 loc="center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(size)
    t.scale(1, 1.8)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor("#d1d5db")
        if r == 0:
            cell.set_text_props(weight="bold", color=INK)
            cell.set_facecolor("#f3f4f6")
        if highlight is not None and r > 0 and df.index[r - 1] == highlight:
            cell.set_facecolor("#e0ecf8")
            cell.set_text_props(weight="bold")


def main():
    data, weather_fc = F.load_data()
    val = pd.read_csv(ROOT / "results" / "validation.csv", parse_dates=["origin", "ds"])
    sub = pd.read_csv(ROOT / f"{F.SURNAME}_project.csv")
    names = {"snaive": "Seasonal naive (benchmark)", "profile7": "Mean of last 7 days",
             "gbm_no_weather": "GBM without weather", "gbm": "GBM on y", "gbm_log": "GBM on log y",
             "ensemble": "Average of the two GBMs"}
    pts = F.point_scores(val, list(names)).rename(index=names)
    check = F.interval_check(val, "ensemble", n_cal=20).rename(index={"ensemble": names["ensemble"], "snaive": names["snaive"]})
    ens, sn = pts.loc[names["ensemble"]], pts.loc[names["snaive"]]
    n_win = val["origin"].nunique()
    beat = (F.window_scores(val, "ensemble")["RMSE"] < F.window_scores(val, "snaive")["RMSE"]).mean()
    growth = data.set_index("ds")["y"].resample("YS").mean()
    tag = f"MATH840 final project · {F.SURNAME.title()}"

    with PdfPages(ROOT / "slides.pdf") as pdf:
        # 1. Title
        fig = plt.figure(figsize=(W, Hh))
        fig.text(0.05, 0.62, "48 hours of generation", fontsize=40, weight="bold", color=INK)
        fig.text(0.05, 0.53, "Hourly forecast with a 95% interval, 27–28 October 2026", fontsize=20, color=MUTED)
        fig.text(0.05, 0.36, f"Artem {F.SURNAME.title()} · MATH840 Time Series · KSE", fontsize=16, color=INK)
        fig.text(0.05, 0.28, f"On {n_win} validation windows: RMSE −{1 - ens.RMSE / sn.RMSE:.0%}, "
                 f"MAPE −{1 - ens.MAPE / sn.MAPE:.0%}, Winkler "
                 f"−{1 - check.loc[names['ensemble'], 'Winkler'] / check.loc[names['snaive'], 'Winkler']:.0%} "
                 "against seasonal naive", fontsize=16, color=ACCENT, weight="bold")
        pdf.savefig(fig); plt.close(fig)

        # 2. The data
        fig = new_slide(tag, "What is in the series", "Solar by day, wind at night, and a portfolio that keeps growing")
        bullets(fig, [
            f"The level grows: mean y {growth.iloc[0]:.0f} in 2023, {growth.iloc[-2]:.0f} in 2025, "
            f"{growth.iloc[-1]:.0f} in 2026. The daily peak grew about 5 times.",
            "Daytime: a solar bell curve. At night: generation rises with wind speed.",
            "cloud_cover = 1 goes with the highest midday output, the opposite of its label "
            "(the series is disguised). I let the model learn the sign.",
            "The daylight flag moves one hour earlier at the end of October (clock change); "
            "the forecast already has 07–15.",
            "The weather forecast is rougher than real weather, more so on day 2.",
        ], width=52, size=14)
        image(fig, FIG / "series.png", [0.50, 0.40, 0.48, 0.42])
        image(fig, FIG / "profile_wind.png", [0.50, 0.04, 0.48, 0.36])
        pdf.savefig(fig); plt.close(fig)


        fig = new_slide(tag, "Approach", "Simple, and every feature is known 48 hours ahead")
        bullets(fig, [
            "Features: the 7 weather columns, hour, day of year, and last week's y at the same hour "
            "(mean, max, lag 48). The last week's level lets the model follow the growth.",
            "Two HistGradientBoosting models: one on y (good RMSE), one on log y (good MAPE). "
            "The forecast is their average.",
            f"Validation: {n_win} rolling windows of 48 hours, each starting at midnight, every 2 days "
            f"from {val.origin.min():%d %b} to {val.origin.max():%d %b %Y}. The model is refitted for every window.",
            "In validation the weather inputs get simulated forecast errors, larger on day 2, "
            "so the scores are not optimistic.",
            "95% interval: the forecast plus the 2.5% and 97.5% quantiles of the validation errors, "
            "separately for day 1 / day 2 and daylight / night.",
        ], width=95, size=16, para=0.045)
        pdf.savefig(fig); plt.close(fig)


        fig = new_slide(tag, "Validation against the benchmark",
                        f"Mean over {n_win} windows. The average has the lower RMSE in {beat:.0%} of them.")
        tab = pts[["RMSE", "MAPE"]].copy()
        tab["RMSE"] = tab["RMSE"].map("{:.0f}".format)
        tab["MAPE"] = tab["MAPE"].map("{:.0f}%".format)
        table(fig, tab, [0.27, 0.47, 0.30, 0.33], highlight=names["ensemble"])
        d1 = F.point_scores(val[val["step"] <= F.M], ["snaive", "ensemble"])
        d2 = F.point_scores(val[val["step"] > F.M], ["snaive", "ensemble"])
        bullets(fig, [
            f"Day 1: level with the benchmark (RMSE {d1.loc['ensemble', 'RMSE']:.0f} vs {d1.loc['snaive', 'RMSE']:.0f}).",
            f"Day 2, the private half: RMSE {d2.loc['ensemble', 'RMSE']:.0f} vs {d2.loc['snaive', 'RMSE']:.0f}, "
            f"MAPE {d2.loc['ensemble', 'MAPE']:.0f}% vs {d2.loc['snaive', 'MAPE']:.0f}%.",
            "Without weather the model loses to the benchmark.",
        ], x=0.66, y=0.78, width=34, size=14, para=0.04)
        image(fig, FIG / "windows.png", [0.03, 0.03, 0.62, 0.40])
        pdf.savefig(fig); plt.close(fig)


        fig = new_slide(tag, "How good is the interval",
                        "Calibrated on the first 20 windows, scored on the last 20")
        it = check[["Winkler", "coverage"]].copy()
        it["Winkler"] = it["Winkler"].map("{:.0f}".format)
        it["coverage"] = it["coverage"].map("{:.0%}".format)
        table(fig, it, [0.30, 0.52, 0.40, 0.22], size=14, highlight=names["ensemble"])
        bullets(fig, [
            "Forecast + empirical 2.5% / 97.5% quantiles of past errors, by day 1 / day 2 and daylight / night; "
            "the lower bound is clipped at 0. No normality assumed: the errors are skewed upward.",
            "Tried first: quantiles of log(actual / forecast), so the width scales with the level. Worse Winkler "
            "and lower coverage: tiny night values make the ratios heavy-tailed.",
            "Coverage is a bit below 95% on the later windows: 96% in September but 90% in October. "
            "The final interval is calibrated on all 40 windows, the most recent included.",
        ], y=0.44, width=110, size=15, para=0.04)
        pdf.savefig(fig); plt.close(fig)


        fig = new_slide(tag, "The forecast for 27–28 October",
                        "Day 1: humidity 94–100% and 5–10 °C, a low solar peak. Day 2: drier and warmer, a higher one.")
        image(fig, FIG / "forecast.png", [0.05, 0.05, 0.9, 0.75])
        pdf.savefig(fig); plt.close(fig)


        fig = new_slide(tag, "What did not work, and what I would do next")
        bullets(fig, [
            f"Mean of the last 7 days: worse than the benchmark (RMSE {pts.loc[names['profile7'], 'RMSE']:.0f}), "
            "because the weather moves solar output a lot from day to day.",
            f"GBM on y alone: best RMSE ({pts.loc[names['gbm'], 'RMSE']:.0f}) but MAPE "
            f"{pts.loc[names['gbm'], 'MAPE']:.0f}%: it overshoots the small night values.",
            f"GBM on log y alone: best MAPE ({pts.loc[names['gbm_log'], 'MAPE']:.0f}%) but RMSE "
            f"{pts.loc[names['gbm_log'], 'RMSE']:.0f}. Averaging the two keeps most of both.",
            "Bias correction (add the mean past error): RMSE barely moved, MAPE got much worse.",
            "Left in the errors: the midday peak is under-forecast, and errors persist through the day.",
            "Next: model the solar and the wind parts separately, and use quantile regression for the interval.",
        ], width=100, size=16, para=0.045)
        pdf.savefig(fig); plt.close(fig)
    print("saved slides.pdf")


if __name__ == "__main__":
    main()

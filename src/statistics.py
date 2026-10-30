"""Step 2 - Crime-pattern statistics: frequency analysis, conditional probability,
mean & variance, and correlation."""

import numpy as np
import pandas as pd
from scipy import stats

from src.config import CATEGORIES, DAYS, TIME_SLOTS


# ---------- frequency analysis ----------
def frequency(series, order=None):
    """Count, relative frequency (= estimated probability) and cumulative frequency."""
    c = series.value_counts()
    if order is not None:
        c = c.reindex(order, fill_value=0)
    t = pd.DataFrame({"count": c, "probability": c / c.sum()})
    t["cumulative"] = t["probability"].cumsum()
    return t


# ---------- conditional probability ----------
def conditional(df, target, given, target_order=None, given_order=None):
    """P(target | given): every row sums to 1."""
    t = pd.crosstab(df[given], df[target], normalize="index")
    if given_order is not None:
        t = t.reindex(index=given_order)
    if target_order is not None:
        t = t.reindex(columns=target_order)
    return t.fillna(0)


def bayes_example(df):
    """P(VIOLENT | domestic) by Bayes' theorem and by direct counting."""
    v, d = df["category"] == "VIOLENT", df["domestic"] == 1
    p_v, p_d, p_d_given_v = v.mean(), d.mean(), (v & d).sum() / v.sum()
    return {"P(VIOLENT)": p_v, "P(DOMESTIC)": p_d, "P(DOMESTIC|VIOLENT)": p_d_given_v,
            "P(VIOLENT|DOMESTIC) Bayes": p_d_given_v * p_v / p_d,
            "P(VIOLENT|DOMESTIC) count": (v & d).sum() / d.sum()}


# ---------- mean and variance ----------
def mean_variance(x):
    x = np.asarray(x, dtype=float)
    return {"n": len(x), "mean": x.mean(), "variance": x.var(ddof=1), "std": x.std(ddof=1),
            "median": float(np.median(x)), "min": x.min(), "max": x.max(),
            "cv": x.std(ddof=1) / x.mean()}


def grouped_mean_variance(df, value, by, labels=None):
    g = df.groupby(by)[value].agg(n="count", mean="mean", variance="var", std="std")
    if labels is not None:
        g.index = [labels[i] for i in g.index]
    return g


# ---------- correlation ----------
def correlation_table(cells):
    """Pearson correlation of each cell feature with the incident count, with p-values."""
    rows = []
    for col in ["slot", "day_of_week", "is_weekend", "latitude", "longitude"]:
        r = stats.pearsonr(cells[col], cells["incidents"])
        rows.append({"feature": col, "r": r.statistic, "p_value": r.pvalue})
    return pd.DataFrame(rows)


def correlation_matrix(cells):
    cols = ["incidents", "slot", "day_of_week", "is_weekend", "latitude", "longitude"]
    return cells[cols].corr()


# ---------- everything for the city ----------
def city_patterns(df, daily, cells):
    return {
        "category_freq": frequency(df["category"], CATEGORIES),
        "type_freq": frequency(df["primary_type"]).head(10),
        "area_freq": frequency(df["area"]).head(10),
        "slot_freq": frequency(df["time_slot"], TIME_SLOTS),
        "day_freq": frequency(df["day"], DAYS),
        "P(category|time_slot)": conditional(df, "category", "time_slot", CATEGORIES, TIME_SLOTS),
        "P(arrest|category)": df.groupby("category", observed=True)["arrest"].mean(),
        "bayes": bayes_example(df),
        "daily": mean_variance(daily["count"]),
        "daily_by_weekday": grouped_mean_variance(daily, "count", "day_of_week", DAYS),
        "cells_by_slot": grouped_mean_variance(cells, "incidents", "slot", TIME_SLOTS),
        "correlation": correlation_table(cells),
        "corr_matrix": correlation_matrix(cells),
    }

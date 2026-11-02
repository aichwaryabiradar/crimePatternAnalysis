"""Step 6 - Hotspot prediction for a place chosen by the user, and the riskiest
places for a chosen day and time."""

import numpy as np
import pandas as pd

from src import statistics
from src.config import CATEGORIES, DAYS, TIME_SLOTS
from src.models import predict_proba


def area_profile(df, cells, area):
    """Crime-pattern statistics for one community area, compared with the city."""
    a = df[df["area"] == area]
    days = pd.date_range(df["date_only"].min(), df["date_only"].max(), freq="D")
    daily = a.groupby("date_only").size().reindex(days, fill_value=0)
    city_rank = df["area"].value_counts().rank(ascending=False).astype(int)
    grid = (cells[cells["area"] == area]
            .pivot(index="slot", columns="day_of_week", values="incidents"))
    grid.index, grid.columns = TIME_SLOTS, DAYS
    cat = pd.DataFrame({"Area": statistics.frequency(a["category"], CATEGORIES)["probability"],
                        "City": statistics.frequency(df["category"], CATEGORIES)["probability"]})
    return {"n": len(a), "share": len(a) / len(df), "rank": int(city_rank[area]),
            "n_areas": len(city_rank), "daily": statistics.mean_variance(daily.values),
            "category": cat, "grid": grid,
            "top_types": statistics.frequency(a["primary_type"]).head(5),
            "arrest_rate": a["arrest"].mean(), "city_arrest_rate": df["arrest"].mean()}


def area_risk_grid(models, cells, area, threshold):
    """Predicted P(high risk) of every weekday x time slot for one area, per model,
    plus the observed high-risk cells."""
    c = cells[cells["area"] == area].sort_values(["slot", "day_of_week"])
    probas = predict_proba(models, c)
    grids = {}
    for name, p in probas.items():
        grids[name] = pd.DataFrame(np.asarray(p).reshape(4, 7), index=TIME_SLOTS, columns=DAYS)
    actual = pd.DataFrame((c["incidents"].values >= threshold).astype(int).reshape(4, 7),
                          index=TIME_SLOTS, columns=DAYS)
    return grids, actual


def top_periods(grids, model="Random Forest", k=5):
    g = grids[model].stack().sort_values(ascending=False).head(k)
    return pd.DataFrame({"Day": [d for _, d in g.index], "Time slot": [s for s, _ in g.index],
                         "P(high risk)": g.values})


def riskiest_areas(models, cells, day_of_week, slot, k=10):
    """Rank all areas by predicted risk at a given weekday and time slot."""
    c = cells[(cells["day_of_week"] == day_of_week) & (cells["slot"] == slot)].copy()
    for name, p in predict_proba(models, c).items():
        c[name] = p
    return c.sort_values("Random Forest", ascending=False).reset_index(drop=True)

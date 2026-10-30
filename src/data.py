"""Step 1 - Load and clean the incidents, and build the place-time table used
for hotspot prediction.

Incident table : one row per reported crime (for the pattern statistics)
Cell table     : one row per (community area, weekday, time slot) with the number
                 of incidents in that place-time cell (for the ML models)"""

import pandas as pd

from src.config import (CATEGORIES, CATEGORY_MAP, COMMUNITY_AREAS, DATA_PATH, DAYS,
                        TIME_SLOTS)

TYPE_TO_CATEGORY = {t: c for c, types in CATEGORY_MAP.items() for t in types}


def load_incidents(path=DATA_PATH):
    raw = pd.read_csv(path)
    df = raw.dropna(subset=["date", "primary_type", "community_area", "latitude", "longitude"])
    df = df[df["community_area"].isin(COMMUNITY_AREAS)].drop_duplicates(subset="id").copy()

    df["ts"] = pd.to_datetime(df["date"], format="%m/%d/%Y %H:%M")
    df["date_only"] = df["ts"].dt.normalize()
    df["hour"] = df["ts"].dt.hour
    df["day_of_week"] = df["ts"].dt.dayofweek               # 0 = Monday
    df["day"] = df["day_of_week"].map(dict(enumerate(DAYS)))
    df["slot"] = df["hour"] // 6                              # 0..3
    df["time_slot"] = df["slot"].map(dict(enumerate(TIME_SLOTS)))
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["community_area"] = df["community_area"].astype(int)
    df["area"] = df["community_area"].map(COMMUNITY_AREAS)
    df["category"] = pd.Categorical(df["primary_type"].map(TYPE_TO_CATEGORY).fillna("OTHER"),
                                    categories=CATEGORIES)
    df["arrest"] = df["arrest"].astype(int)
    df["domestic"] = df["domestic"].astype(int)
    info = {"rows_raw": len(raw), "rows_clean": len(df),
            "date_min": str(df["ts"].min().date()), "date_max": str(df["ts"].max().date()),
            "n_days": int(df["date_only"].nunique()), "n_areas": int(df["community_area"].nunique()),
            "n_types": int(df["primary_type"].nunique())}
    return df.reset_index(drop=True), info


def build_cells(df):
    """Count incidents in every (area, weekday, time slot) cell, including empty ones,
    and attach the features the models use."""
    areas = sorted(COMMUNITY_AREAS)
    idx = pd.MultiIndex.from_product([areas, range(7), range(4)],
                                     names=["community_area", "day_of_week", "slot"])
    cells = (df.groupby(["community_area", "day_of_week", "slot"]).size()
               .reindex(idx, fill_value=0).rename("incidents").reset_index())
    centre = df.groupby("community_area")[["latitude", "longitude"]].median()
    cells = cells.join(centre, on="community_area")
    cells["is_weekend"] = (cells["day_of_week"] >= 5).astype(int)
    cells["area"] = cells["community_area"].map(COMMUNITY_AREAS)
    cells["day"] = cells["day_of_week"].map(dict(enumerate(DAYS)))
    cells["time_slot"] = cells["slot"].map(dict(enumerate(TIME_SLOTS)))
    return cells


def daily_counts(df):
    days = pd.date_range(df["date_only"].min(), df["date_only"].max(), freq="D")
    counts = df.groupby("date_only").size().reindex(days, fill_value=0)
    out = pd.DataFrame({"date": days, "count": counts.values})
    out["day_of_week"] = out["date"].dt.dayofweek
    return out

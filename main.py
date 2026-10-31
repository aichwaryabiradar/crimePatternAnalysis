"""
Command-line run of the whole project (no website). Saves to outputs/:
  results.json, CSV tables and figures/ used in the report.

    python main.py
    python main.py --area "Lake View"
For the website:  streamlit run app.py
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from src import data, evaluation, figures, hotspot, models, statistics
from src.config import COMMUNITY_AREAS, OUTPUT_DIR


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, pd.DataFrame):
        return json.loads(o.to_json(orient="split"))
    if isinstance(o, pd.Series):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, np.ndarray):
        return o.tolist() if o.size <= 50 else f"<{o.size} values>"
    if isinstance(o, np.generic):
        return o.item()
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--area", default="Logan Square", choices=sorted(COMMUNITY_AREAS.values()))
    args = ap.parse_args()
    fig_dir = os.path.join(OUTPUT_DIR, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    print("[1] Loading data ...")
    df, info = data.load_incidents()
    cells = data.build_cells(df)
    print(f"    {info['rows_clean']:,} incidents, {len(cells):,} place-time cells")

    print("[2] Crime-pattern statistics ...")
    P = statistics.city_patterns(df, data.daily_counts(df), cells)

    print("[3] Sampling: 70/30 split; [4] training 3 models ...")
    train, test, thr = models.split_and_label(cells)
    ms = models.train_models(train)

    print("[5] Comparing models on the held-out test set ...")
    R = evaluation.compare_models(test, models.predict_proba(ms, test))

    print(f"[6] Hotspot prediction for {args.area} ...")
    prof = hotspot.area_profile(df, cells, args.area)
    grids, actual = hotspot.area_risk_grid(ms, cells, args.area, thr)

    # tables
    P["category_freq"].to_csv(f"{OUTPUT_DIR}/frequency_category.csv")
    P["P(category|time_slot)"].to_csv(f"{OUTPUT_DIR}/conditional_category_given_slot.csv")
    P["daily_by_weekday"].to_csv(f"{OUTPUT_DIR}/daily_mean_variance_by_weekday.csv")
    P["cells_by_slot"].to_csv(f"{OUTPUT_DIR}/cell_mean_variance_by_slot.csv")
    P["correlation"].to_csv(f"{OUTPUT_DIR}/correlation_tests.csv", index=False)
    R["table"].to_csv(f"{OUTPUT_DIR}/model_metrics.csv", index=False)
    R["pairs"].to_csv(f"{OUTPUT_DIR}/pairwise_tests.csv", index=False)

    # figures
    figures.frequency(P, f"{fig_dir}/01_frequency.png")
    figures.conditional(P, f"{fig_dir}/02_conditional.png")
    figures.mean_variance(P, f"{fig_dir}/03_mean_variance.png")
    figures.correlation(P, f"{fig_dir}/04_correlation.png")
    figures.accuracy_ci(R["table"], 1 - test["high_risk"].mean(), f"{fig_dir}/05_accuracy_ci.png")
    figures.confusions(R["metrics"], f"{fig_dir}/06_confusion.png")
    figures.bootstraps(R, f"{fig_dir}/07_bootstrap.png")
    figures.area_example(grids, actual, args.area, f"{fig_dir}/08_area_example.png")

    agree = {n: float(((g.values >= 0.5) == actual.values).mean()) for n, g in grids.items()}
    out = {"data": info, "patterns": P,
           "sampling": {"n_train": len(train), "n_test": len(test), "threshold": thr,
                        "high_risk_train": float(train["high_risk"].mean()),
                        "high_risk_test": float(test["high_risk"].mean())},
           "comparison": {"table": R["table"], "pairs": R["pairs"],
                          "alpha_adjusted": R["alpha_adjusted"]},
           "area": {"name": args.area, "profile": prof, "agreement": agree,
                    "top_periods": hotspot.top_periods(grids)}}
    with open(f"{OUTPUT_DIR}/results.json", "w") as f:
        json.dump(jsonable(out), f, indent=1)

    pd.set_option("display.width", 160)
    print("\n================ RESULTS ================")
    print(R["table"].round(3).to_string(index=False))
    print()
    print(R["pairs"][["Comparison", "Accuracy difference", "CI low", "CI high", "McNemar p",
                      "Significant (α/3)"]].round(4).to_string(index=False))
    print(f"\nOutputs written to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()

"""Static figures for the written report (saved by main.py)."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import BLUE, MODEL_COLORS, ORANGE

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax, title=None, xlabel=None, ylabel=None):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8.5)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold", color=INK)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=9)


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def _heat(ax, M, rows, cols, fmt, cmap, vmin=None, vmax=None, fs=8):
    vmin = M.min() if vmin is None else vmin
    vmax = M.max() if vmax is None else vmax
    ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if vmin < 0 < vmax:      # diverging scale: dark at both ends
                light = abs(M[i, j]) / max(abs(vmin), abs(vmax)) > 0.6
            else:
                light = (M[i, j] - vmin) / (vmax - vmin + 1e-12) > 0.6
            ax.text(j, i, format(M[i, j], fmt), ha="center", va="center", fontsize=fs,
                    color="white" if light else INK)
    ax.set_xticks(range(len(cols)), cols, fontsize=8)
    ax.set_yticks(range(len(rows)), rows, fontsize=8)


def frequency(P, path):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.3))
    for ax, key, title in [(axes[0], "category_freq", "Crime categories"),
                           (axes[1], "slot_freq", "Time slot")]:
        t = P[key]
        ax.bar(t.index.astype(str), t["probability"], color=BLUE, width=0.6)
        for x, v in enumerate(t["probability"]):
            ax.text(x, v + 0.008, f"{v:.1%}", ha="center", fontsize=8, color=INK2)
        ax.set_ylim(0, t["probability"].max() * 1.2)
        ax.tick_params(axis="x", labelsize=7.5)
        _style(ax, f"Relative frequency: {title}", None, "P")
    _save(fig, path)


def conditional(P, path):
    t = P["P(category|time_slot)"]
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    _heat(ax, t.values, t.index, t.columns, ".2f", "Blues", 0)
    ax.set_title("P(category | time slot): each row sums to 1", loc="left", fontsize=10.5,
                 fontweight="bold", color=INK)
    _save(fig, path)


def mean_variance(P, path):
    t = P["cells_by_slot"]
    fig, ax = plt.subplots(figsize=(7, 3.1))
    ax.bar(t.index, t["mean"], yerr=t["std"], color=BLUE, width=0.55, capsize=5,
           error_kw=dict(ecolor=INK2, lw=1.2))
    for x, (m, v) in enumerate(zip(t["mean"], t["variance"])):
        ax.text(x + 0.08, m + 0.6, f"mean {m:.1f}\nvar {v:.0f}", fontsize=8, color=INK2)
    _style(ax, "Incidents per place-time cell by time slot (mean ± SD)", None, "Incidents")
    _save(fig, path)


def correlation(P, path):
    t = P["corr_matrix"]
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    _heat(ax, t.values, t.index, t.columns, ".2f", "RdBu_r", -1, 1)
    ax.set_xticks(range(len(t.columns)), t.columns, rotation=40, ha="right", fontsize=8)
    ax.set_aspect("equal")
    ax.set_title("Pearson correlation matrix (cell level)", loc="left", fontsize=10.5,
                 fontweight="bold", color=INK)
    _save(fig, path)


def accuracy_ci(table, base_rate, path):
    fig, ax = plt.subplots(figsize=(7, 2.4))
    for i, r in table.iterrows():
        ax.errorbar(r["Accuracy"] * 100, i, xerr=[[(r["Accuracy"] - r["CI low"]) * 100],
                                                  [(r["CI high"] - r["Accuracy"]) * 100]],
                    fmt="o", ms=8, capsize=5, lw=2, color=MODEL_COLORS[r["Model"]])
    ax.axvline(base_rate * 100, color=INK2, ls=":", lw=1.3)
    ax.text(base_rate * 100, -0.6, " always 'low risk'", fontsize=8, color=INK2)
    ax.set_yticks(range(len(table)), table["Model"])
    ax.invert_yaxis()
    ax.set_ylim(len(table) - 0.5, -0.9)
    _style(ax, "Test accuracy with 95% Wilson confidence interval", "Accuracy (%)")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID)
    _save(fig, path)


def confusions(metrics, path):
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 2.9))
    for ax, (name, m) in zip(axes, metrics.items()):
        _heat(ax, m["confusion"].astype(float), ["actual low", "actual high"],
              ["pred low", "pred high"], ".0f", "Blues", fs=10)
        ax.set_title(name, loc="left", fontsize=10, fontweight="bold", color=INK)
    _save(fig, path)


def bootstraps(R, path):
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.9))
    for ax, (_, r) in zip(axes, R["pairs"].iterrows()):
        d = R["bootstrap"][r["Comparison"]] * 100
        ax.hist(d, bins=30, color=BLUE, alpha=0.85, edgecolor="white", linewidth=0.5)
        ax.axvspan(r["CI low"] * 100, r["CI high"] * 100, color=BLUE, alpha=0.08)
        ax.axvline(0, color=ORANGE, lw=2)
        _style(ax, r["Comparison"].replace(" vs ", "\nvs "), "Accuracy difference (pp)")
        ax.title.set_fontsize(9)
    _save(fig, path)


def area_example(grids, actual, area, path):
    fig, axes = plt.subplots(1, 4, figsize=(12, 2.7))
    _heat(axes[0], actual.values.astype(float), actual.index, actual.columns, ".0f", "Oranges",
          0, 1)
    axes[0].set_title(f"{area}: observed high risk", loc="left", fontsize=9.5,
                      fontweight="bold", color=INK)
    for ax, (name, g) in zip(axes[1:], grids.items()):
        _heat(ax, g.values, g.index, g.columns, ".2f", "YlOrRd", 0, 1, fs=6.5)
        ax.set_title(f"{name}: P(high risk)", loc="left", fontsize=9.5, fontweight="bold",
                     color=INK)
        ax.set_yticks([])
    for ax in axes:
        ax.tick_params(axis="x", labelsize=7)
    _save(fig, path)


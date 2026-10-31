"""Unit tests.  Run:  python -m pytest -q"""

import numpy as np
import pandas as pd
from scipy import stats

from src import evaluation, statistics


def test_frequency_sums_to_one():
    t = statistics.frequency(pd.Series(list("aabbbc")))
    assert abs(t["probability"].sum() - 1) < 1e-12
    assert t.loc["b", "count"] == 3


def test_conditional_rows_sum_to_one():
    df = pd.DataFrame({"g": list("xxyyy"), "t": list("abaab")})
    assert np.allclose(statistics.conditional(df, "t", "g").sum(axis=1), 1)


def test_mean_variance_matches_numpy():
    x = [2, 4, 4, 4, 5, 5, 7, 9]
    r = statistics.mean_variance(x)
    assert r["mean"] == 5 and abs(r["variance"] - np.var(x, ddof=1)) < 1e-12


def test_wilson_ci_known_value():
    lo, hi = evaluation.wilson_ci(50, 100)
    assert abs(lo - 0.4038) < 1e-3 and abs(hi - 0.5962) < 1e-3


def test_mcnemar_matches_binomial():
    a = np.array([True] * 10 + [False] * 4 + [True] * 6)
    b = np.array([True] * 10 + [True] * 4 + [False] * 6)
    r = evaluation.mcnemar(a, b)
    assert (r["b"], r["c"]) == (6, 4)
    assert abs(r["p_value"] - stats.binomtest(4, 10, 0.5).pvalue) < 1e-12


def test_identical_models_not_significant():
    c = np.random.default_rng(0).random(300) < 0.7
    assert evaluation.mcnemar(c, c)["p_value"] == 1.0
    lo, hi, _ = evaluation.bootstrap_diff_ci(c, c)
    assert lo == 0 and hi == 0


def test_bootstrap_ci_contains_true_difference():
    rng = np.random.default_rng(1)
    a = rng.random(2000) < 0.70
    b = a | (rng.random(2000) < 0.2)       # b is right whenever a is, plus extra
    lo, hi, _ = evaluation.bootstrap_diff_ci(a, b)
    assert lo < b.mean() - a.mean() < hi and lo > 0

"""Step 5 - Compare the three models on the SAME held-out test set.

Estimation : accuracy with a 95 % Wilson confidence interval
Hypothesis : McNemar's exact test for every pair of models (paired: same test cells)
Estimation : paired-bootstrap 95 % CI of the accuracy difference
Question   : is a model really better, or could the difference be test-sample variation?"""

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score)

from src.config import ALPHA, BOOTSTRAP, SEED


def wilson_ci(k, n, alpha=ALPHA):
    """95 % Wilson score interval for a proportion k / n."""
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(centre - half), float(centre + half)


def mcnemar(correct_a, correct_b):
    """b = A right & B wrong, c = A wrong & B right. Under H0: c ~ Binomial(b + c, 0.5)."""
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    p = float(stats.binomtest(min(b, c), b + c, 0.5).pvalue) if b + c else 1.0
    return {"b": b, "c": c, "p_value": p}


def bootstrap_diff_ci(correct_a, correct_b, B=BOOTSTRAP, seed=SEED, alpha=ALPHA):
    """Resample test cells with replacement; CI of accuracy(B) - accuracy(A)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(correct_a), size=(B, len(correct_a)))
    d = correct_b[idx].mean(axis=1) - correct_a[idx].mean(axis=1)
    lo, hi = np.percentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi), d


def model_metrics(y, proba, threshold=0.5):
    pred = (proba >= threshold).astype(int)
    correct = pred == y
    k, n = int(correct.sum()), len(y)
    return {"accuracy": k / n, "ci": wilson_ci(k, n), "precision": precision_score(y, pred),
            "recall": recall_score(y, pred), "f1": f1_score(y, pred),
            "roc_auc": roc_auc_score(y, proba), "confusion": confusion_matrix(y, pred),
            "correct": correct}


def compare_models(test, probas, alpha=ALPHA):
    y = test["high_risk"].values
    metrics = {name: model_metrics(y, p) for name, p in probas.items()}
    table = pd.DataFrame([{"Model": n, "Accuracy": m["accuracy"], "CI low": m["ci"][0],
                           "CI high": m["ci"][1], "Precision": m["precision"],
                           "Recall": m["recall"], "F1": m["f1"], "ROC-AUC": m["roc_auc"]}
                          for n, m in metrics.items()])
    n_pairs = len(metrics) * (len(metrics) - 1) // 2
    pairs, boots = [], {}
    for a, b in combinations(metrics, 2):
        ca, cb = metrics[a]["correct"], metrics[b]["correct"]
        mc = mcnemar(ca, cb)
        lo, hi, d = bootstrap_diff_ci(ca, cb)
        boots[f"{b} vs {a}"] = d
        pairs.append({"Comparison": f"{b} vs {a}", "Accuracy difference": cb.mean() - ca.mean(),
                      "CI low": lo, "CI high": hi, "b (A right, B wrong)": mc["b"],
                      "c (A wrong, B right)": mc["c"], "McNemar p": mc["p_value"],
                      "Significant (α/3)": mc["p_value"] < alpha / n_pairs})
    return {"metrics": metrics, "table": table, "pairs": pd.DataFrame(pairs),
            "bootstrap": boots, "alpha_adjusted": alpha / n_pairs, "n_test": len(y)}

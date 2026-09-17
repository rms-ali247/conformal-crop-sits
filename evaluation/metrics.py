"""
metrics.py — Calibration, selective-prediction, and significance metrics.

Pure numpy/scipy functions used by the evaluator and benchmark:
  - expected_calibration_error / reliability bins (Guo et al., ICML 2017)
  - brier_score (multiclass)
  - risk_coverage_curve + AURC (selective prediction; Geifman & El-Yaniv 2017)
  - mcnemar_test (paired model comparison)
"""

import numpy as np

# np.trapz was removed in NumPy 2.x in favour of np.trapezoid.
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray,
                               n_bins: int = 15):
    """Return (ECE, reliability_rows).

    reliability_rows: list of dicts with bin centre, accuracy, confidence, count.
    """
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    ece = 0.0
    rows = []
    N = len(labels)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        cnt = int(m.sum())
        if cnt == 0:
            continue
        acc = float(correct[m].mean())
        avg_conf = float(conf[m].mean())
        ece += (cnt / N) * abs(acc - avg_conf)
        rows.append({"bin_center": float(0.5 * (lo + hi)),
                     "accuracy": acc, "confidence": avg_conf, "count": cnt})
    return float(ece), rows


def brier_score(probs: np.ndarray, labels: np.ndarray, num_classes: int) -> float:
    """Multiclass Brier score (lower is better)."""
    onehot = np.eye(num_classes)[labels]
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def risk_coverage_curve(probs: np.ndarray, labels: np.ndarray):
    """Selective prediction: sort by confidence (max prob), sweep coverage.

    Returns (coverage, risk, aurc). risk[k] = error rate over the most-confident
    fraction coverage[k].  AURC = area under the risk-coverage curve (lower better).
    """
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(np.float64)
    order = np.argsort(-conf)                       # most confident first
    correct_sorted = correct[order]
    N = len(labels)
    coverage = np.arange(1, N + 1) / N
    risk = 1.0 - np.cumsum(correct_sorted) / np.arange(1, N + 1)
    aurc = float(_trapz(risk, coverage))
    return coverage, risk, aurc


def accuracy_at_coverage(probs: np.ndarray, labels: np.ndarray,
                         coverage: float = 0.8) -> float:
    """Accuracy over the most-confident ``coverage`` fraction (auto-classify rate)."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels)
    order = np.argsort(-conf)
    k = max(1, int(round(coverage * len(labels))))
    return float(correct[order[:k]].mean())


def mcnemar_test(correct_a: np.ndarray, correct_b: np.ndarray):
    """Paired McNemar test (with continuity correction) on per-sample correctness.

    Returns (b, c, chi2_stat, p_value) where
      b = #(A correct, B wrong), c = #(A wrong, B correct).
    """
    correct_a = correct_a.astype(bool)
    correct_b = correct_b.astype(bool)
    b = int((correct_a & ~correct_b).sum())
    c = int((~correct_a & correct_b).sum())
    if (b + c) == 0:
        return b, c, 0.0, 1.0
    stat = (abs(b - c) - 1.0) ** 2 / (b + c)
    try:
        from scipy.stats import chi2
        p = float(1.0 - chi2.cdf(stat, df=1))
    except Exception:
        p = float("nan")
    return b, c, float(stat), p

"""
calibrate.py — Temperature scaling (Guo et al., ICML 2017).

Fits a single scalar temperature T on a held-out calibration split by
minimising NLL, then rescales logits by 1/T.  A post-hoc, accuracy-preserving
calibration that markedly reduces ECE for deep nets and ensembles.
"""

import numpy as np
import torch


def fit_temperature(logits: np.ndarray, labels: np.ndarray,
                    max_iter: int = 200, lr: float = 0.01) -> float:
    """Optimise T (>0) to minimise cross-entropy of logits/T against labels."""
    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.long)
    log_T = torch.nn.Parameter(torch.zeros(1))   # optimise log T for positivity
    optimizer = torch.optim.LBFGS([log_T], lr=lr, max_iter=max_iter)
    nll = torch.nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = nll(logits_t / log_T.exp(), labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_T.exp().detach().clamp_(1e-3, 100.0))


def softmax_np(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    """Return temperature-scaled probabilities softmax(logits / T)."""
    return softmax_np(logits / T)

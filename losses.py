"""
losses.py — Imbalance-aware and noise-robust loss functions.

A single factory, ``build_loss``, returns a callable ``loss(logits, target)``
so pipeline/train.py can switch objectives with ``--loss``:

  ce         CrossEntropy (optionally inverse-frequency weighted)  [default]
  focal      Focal loss (Lin et al., ICCV 2017) — hard-example / imbalance
  logit-adj  Logit-adjusted loss (Menon et al., ICLR 2021) — long-tail
  gce        Generalized Cross Entropy (Zhang & Sabuncu, NeurIPS 2018) — noisy labels
  sce        Symmetric Cross Entropy (Wang et al., ICCV 2019) — noisy labels

References are the canonical papers above; implementations are self-contained
(no external deps) so they are transparent for the write-up.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multi-class focal loss: FL = -alpha_y (1 - p_y)^gamma log p_y.

    ``weight`` (per-class) plays the role of alpha and can be the usual
    inverse-frequency vector.
    """

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits, dim=1)
        logp_y = logp.gather(1, target.unsqueeze(1)).squeeze(1)   # (B,)
        p_y = logp_y.exp()
        loss = -((1.0 - p_y) ** self.gamma) * logp_y
        if self.weight is not None:
            loss = loss * self.weight[target]
        return loss.mean()


class LogitAdjustedLoss(nn.Module):
    """Logit-adjusted CE (Menon et al., 2021): CE(logits + tau * log_prior, y).

    Encourages large margins for rare classes using the label prior; strong,
    simple long-tail baseline that needs no resampling.
    """

    def __init__(self, class_priors: torch.Tensor, tau: float = 1.0):
        super().__init__()
        adj = tau * torch.log(class_priors.clamp_min(1e-12))
        self.register_buffer("adjustment", adj)   # (C,)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits + self.adjustment.unsqueeze(0), target)


class GeneralizedCELoss(nn.Module):
    """Generalized Cross Entropy (Zhang & Sabuncu, 2018): L = (1 - p_y^q)/q.

    Interpolates between CE (q->0) and MAE (q=1); robust to symmetric label
    noise. q in (0, 1].
    """

    def __init__(self, q: float = 0.7, weight: torch.Tensor | None = None):
        super().__init__()
        self.q = q
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = F.softmax(logits, dim=1)
        p_y = p.gather(1, target.unsqueeze(1)).squeeze(1).clamp_min(1e-7)
        loss = (1.0 - p_y ** self.q) / self.q
        if self.weight is not None:
            loss = loss * self.weight[target]
        return loss.mean()


class SymmetricCELoss(nn.Module):
    """Symmetric Cross Entropy (Wang et al., 2019): alpha*CE + beta*RCE.

    RCE (reverse CE) is noise-tolerant; combining it with CE keeps convergence
    speed while gaining robustness. ``A`` is the clamp for log(0) in RCE.
    """

    def __init__(self, alpha: float = 0.1, beta: float = 1.0, A: float = -4.0,
                 num_classes: int | None = None, weight: torch.Tensor | None = None):
        super().__init__()
        self.alpha, self.beta = alpha, beta
        self.clamp = float(torch.tensor(A).exp())   # exp(A); log of it == A
        self.num_classes = num_classes
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        C = self.num_classes or logits.shape[1]
        ce = F.cross_entropy(logits, target,
                             weight=self.weight, reduction="mean")
        p = F.softmax(logits, dim=1).clamp_min(1e-7)
        label_oh = F.one_hot(target, C).float().clamp_min(self.clamp)
        rce = -(p * torch.log(label_oh)).sum(dim=1).mean()
        return self.alpha * ce + self.beta * rce


def build_loss(name: str,
               class_counts: torch.Tensor,
               device: torch.device,
               *,
               use_class_weights: bool = True,
               gamma: float = 2.0,
               tau: float = 1.0,
               q: float = 0.7,
               sce_alpha: float = 0.1,
               sce_beta: float = 1.0) -> nn.Module:
    """Factory. ``class_counts`` is a (C,) tensor of per-class sample counts.

    For ``ce``/``focal``/``gce`` an inverse-frequency class-weight vector is
    used when ``use_class_weights`` is True. ``logit-adj`` and ``sce`` manage
    imbalance/noise internally and ignore the weight vector by default.
    """
    counts = class_counts.float().to(device)
    num_classes = counts.numel()

    # Inverse-frequency weights, normalised to sum to num_classes (matches the
    # original train.py convention).
    inv = 1.0 / counts.clamp_min(1.0)
    weights = (inv / inv.sum() * num_classes) if use_class_weights else None
    priors = counts / counts.sum()

    name = name.lower()
    if name == "ce":
        loss = nn.CrossEntropyLoss(weight=weights)
    elif name == "focal":
        loss = FocalLoss(gamma=gamma, weight=weights)
    elif name in ("logit-adj", "logit_adj", "la"):
        loss = LogitAdjustedLoss(class_priors=priors, tau=tau)
    elif name == "gce":
        loss = GeneralizedCELoss(q=q, weight=weights)
    elif name == "sce":
        loss = SymmetricCELoss(alpha=sce_alpha, beta=sce_beta,
                               num_classes=num_classes, weight=weights)
    else:
        raise ValueError(f"Unknown loss '{name}'. "
                         f"Choose from: ce, focal, logit-adj, gce, sce.")
    return loss.to(device)

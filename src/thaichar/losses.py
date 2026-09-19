"""Loss functions for Thai glyph classification."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Focal loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)."""

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_classes = logits.size(-1)
        ce = F.cross_entropy(logits, targets, reduction="none",
                             label_smoothing=self.label_smoothing)
        p_t = torch.exp(-ce)
        focal = ((1 - p_t) ** self.gamma) * ce
        if self.weight is not None:
            w = self.weight.to(logits.device)[targets]
            focal = focal * w
        return focal.mean()


# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------

def _class_weights(class_counts: torch.Tensor, power: float = 1.0) -> torch.Tensor:
    """Compute per-class weights proportional to n^-power, normalised to mean 1."""
    w = (class_counts.float().clamp(min=1)) ** (-power)
    w = w / w.mean()
    return w


def _cb_weights(class_counts: torch.Tensor, beta: float = 0.999) -> torch.Tensor:
    """Class-balanced weights: (1-beta) / (1-beta^n_c), normalised to mean 1."""
    n = class_counts.float().clamp(min=1)
    w = (1.0 - beta) / (1.0 - beta ** n)
    w = w / w.mean()
    return w


def logit_adjust(
    logits: torch.Tensor,
    log_prior: torch.Tensor,
    tau: float = 1.0,
) -> torch.Tensor:
    """Post-hoc logit adjustment: logits - tau * log_prior (eval only)."""
    return logits - tau * log_prior.to(logits.device)


def build_loss(
    cfg: dict,
    class_counts: torch.Tensor,
) -> nn.Module:
    """Build a loss function from config dict.

    Config keys:
        loss: str — 'ce', 'weighted_ce', 'focal', 'cb_focal'
        label_smoothing: float (default 0.0)
        focal_gamma: float (default 2.0)
        ce_weight_power: float (default 0.5)
        cb_beta: float (default 0.999)

    Returns
    -------
    nn.Module
    """
    loss_name = cfg.get("loss", "ce")
    ls = cfg.get("label_smoothing", 0.0)
    gamma = cfg.get("focal_gamma", 2.0)
    power = cfg.get("ce_weight_power", 0.5)
    beta = cfg.get("cb_beta", 0.999)

    if loss_name == "ce":
        return nn.CrossEntropyLoss(label_smoothing=ls)
    elif loss_name == "weighted_ce":
        w = _class_weights(class_counts, power=power)
        return nn.CrossEntropyLoss(weight=w, label_smoothing=ls)
    elif loss_name == "focal":
        return FocalLoss(gamma=gamma, label_smoothing=ls)
    elif loss_name == "cb_focal":
        w = _cb_weights(class_counts, beta=beta)
        return FocalLoss(gamma=gamma, weight=w, label_smoothing=ls)
    else:
        raise ValueError(f"Unknown loss: {loss_name!r}")

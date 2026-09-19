"""Model construction for Thai glyph classification (timm + scratch CNN)."""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Any

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# SmallCNN (scratch baseline)
# ---------------------------------------------------------------------------

class SmallCNN(nn.Module):
    """4-block CNN baseline: 32→64→128→256, BN+ReLU+MaxPool, GAP‖GMP, FC."""

    def __init__(self, in_chans: int = 3, drop_rate: float = 0.1):
        super().__init__()
        self.num_features = 512  # 256 GAP + 256 GMP
        blocks = []
        prev = in_chans
        for ch in (32, 64, 128, 256):
            blocks.append(nn.Conv2d(prev, ch, 3, padding=1))
            blocks.append(nn.BatchNorm2d(ch))
            blocks.append(nn.ReLU(inplace=True))
            blocks.append(nn.MaxPool2d(2))
            prev = ch
        self.features = nn.Sequential(*blocks)
        self.drop = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        avg = F.adaptive_avg_pool2d(x, 1).flatten(1)
        mx = F.adaptive_max_pool2d(x, 1).flatten(1)
        return self.drop(torch.cat([avg, mx], dim=1))


# ---------------------------------------------------------------------------
# Glyph head (optionally accepts geometry features)
# ---------------------------------------------------------------------------

class GlyphHead(nn.Module):
    """Classification head with optional geometry side-channel."""

    def __init__(self, feat_dim: int, num_classes: int = 72, geometry: bool = False):
        super().__init__()
        self.geometry = geometry
        if geometry:
            self.geo_mlp = nn.Sequential(
                nn.Linear(4, 16),
                nn.ReLU(inplace=True),
            )
            self.fc = nn.Linear(feat_dim + 16, num_classes)
        else:
            self.geo_mlp = None
            self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, feat: torch.Tensor, g: torch.Tensor | None = None) -> torch.Tensor:
        if self.geometry:
            if g is None:
                raise ValueError("geometry=True but g is None")
            g_feat = self.geo_mlp(g)
            feat = torch.cat([feat, g_feat], dim=1)
        return self.fc(feat)


# ---------------------------------------------------------------------------
# GlyphModel: backbone + head wrapper
# ---------------------------------------------------------------------------

_VIT_FAMILIES = ("vit", "deit", "swin", "beit", "eva", "flexivit")


def _needs_img_size(name: str) -> bool:
    """Whether the timm model family requires img_size kwarg."""
    name_lower = name.lower()
    return any(name_lower.startswith(f) for f in _VIT_FAMILIES)


class GlyphModel(nn.Module):
    """Wrapper: backbone (timm or SmallCNN) + GlyphHead.

    Forward signature: ``model(x, g=None) -> logits``.
    """

    def __init__(
        self,
        backbone: nn.Module,
        head: GlyphHead,
        mode: str = "full",
    ):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.mode = mode

    def forward(self, x: torch.Tensor, g: torch.Tensor | None = None) -> torch.Tensor:
        feat = self.backbone(x)
        return self.head(feat, g)

    def train(self, mode: bool = True) -> "GlyphModel":
        super().train(mode)
        if self.mode == "frozen" and mode:
            # Keep backbone BN in eval mode when frozen
            self.backbone.eval()
        return self


# ---------------------------------------------------------------------------
# build_model
# ---------------------------------------------------------------------------

def build_model(
    name: str,
    num_classes: int = 72,
    pretrained: bool = True,
    in_chans: int = 3,
    img_size: int = 64,
    mode: str = "full",
    geometry: bool = False,
    drop_rate: float = 0.1,
    partial_frac: float = 0.35,
) -> nn.Module:
    """Build a glyph classification model.

    Parameters
    ----------
    name : str
        'smallcnn' for scratch CNN, else a timm model name.
    num_classes : int
        Number of output classes.
    pretrained : bool
        Use pretrained weights (timm models only).
    in_chans : int
        Number of input channels.
    img_size : int
        Input image size (square).
    mode : str
        'full', 'frozen', or 'partial'.
    geometry : bool
        Whether to use geometry side-channel.
    drop_rate : float
        Dropout rate.
    partial_frac : float
        Fraction of backbone params to unfreeze in 'partial' mode.

    Returns
    -------
    GlyphModel
    """
    if name == "smallcnn":
        backbone = SmallCNN(in_chans=in_chans, drop_rate=drop_rate)
        feat_dim = backbone.num_features
    else:
        # timm backbone
        extra_kwargs: dict[str, Any] = {}
        if _needs_img_size(name):
            extra_kwargs["img_size"] = img_size
        backbone = timm.create_model(
            name,
            pretrained=pretrained,
            num_classes=0,
            in_chans=in_chans,
            drop_rate=drop_rate,
            **extra_kwargs,
        )
        # Feature dim = what the pooled head actually emits (mobilenetv3/efficientnet return
        # the conv_head width, not num_features), so measure it with a dry forward pass.
        with torch.no_grad():
            was_training = backbone.training
            backbone.eval()
            feat_dim = int(backbone(torch.zeros(1, in_chans, img_size, img_size)).shape[1])
            backbone.train(was_training)

    head = GlyphHead(feat_dim, num_classes, geometry=geometry)
    model = GlyphModel(backbone, head, mode=mode)

    # Apply freeze strategy
    _apply_mode(model, mode, partial_frac)

    return model


def _apply_mode(model: GlyphModel, mode: str, partial_frac: float) -> None:
    """Apply freeze mode to the model."""
    if mode == "frozen":
        for p in model.backbone.parameters():
            p.requires_grad = False
        for p in model.head.parameters():
            p.requires_grad = True
    elif mode == "partial":
        # Freeze all backbone params first
        bb_params = list(model.backbone.parameters())
        for p in bb_params:
            p.requires_grad = False
        # Unfreeze the last partial_frac fraction
        n_unfreeze = max(1, int(len(bb_params) * partial_frac))
        for p in bb_params[-n_unfreeze:]:
            p.requires_grad = True
        # Head always trainable
        for p in model.head.parameters():
            p.requires_grad = True
    elif mode == "full":
        for p in model.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"Unknown mode: {mode!r}")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def count_params(model: nn.Module) -> tuple[int, int]:
    """Return (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def param_groups(
    model: GlyphModel,
    lr: float,
    weight_decay: float,
    llrd: float | None = None,
) -> list[dict]:
    """Build optimizer parameter groups.

    If *llrd* is set (e.g. 0.8), apply layer-wise LR decay across
    backbone parameters split into ~6 buckets.  Head always gets *lr*.
    Biases and norm parameters get no weight decay.
    """
    no_decay_names = {"bias"}

    def _is_no_decay(name: str) -> bool:
        if name.endswith(".bias"):
            return True
        # norm layers: weight in BatchNorm / LayerNorm / GroupNorm
        if "norm" in name.lower() and name.endswith(".weight"):
            return True
        return False

    # Head params
    head_decay = []
    head_no_decay = []
    for n, p in model.head.named_parameters():
        if not p.requires_grad:
            continue
        if _is_no_decay(n):
            head_no_decay.append(p)
        else:
            head_decay.append(p)

    groups = []
    if head_decay:
        groups.append({"params": head_decay, "lr": lr, "weight_decay": weight_decay})
    if head_no_decay:
        groups.append({"params": head_no_decay, "lr": lr, "weight_decay": 0.0})

    # Backbone params
    bb_params = [(n, p) for n, p in model.backbone.named_parameters() if p.requires_grad]
    if not bb_params:
        return groups

    if llrd is None:
        # No LLRD: single group for backbone
        bb_decay = [p for n, p in bb_params if not _is_no_decay(n)]
        bb_no_decay = [p for n, p in bb_params if _is_no_decay(n)]
        if bb_decay:
            groups.append({"params": bb_decay, "lr": lr, "weight_decay": weight_decay})
        if bb_no_decay:
            groups.append({"params": bb_no_decay, "lr": lr, "weight_decay": 0.0})
    else:
        # LLRD: split backbone into ~6 buckets by parameter order
        n_buckets = 6
        bucket_size = max(1, len(bb_params) // n_buckets)
        for bucket_idx in range(n_buckets):
            start = bucket_idx * bucket_size
            if bucket_idx == n_buckets - 1:
                end = len(bb_params)
            else:
                end = start + bucket_size
            bucket_params = bb_params[start:end]
            if not bucket_params:
                continue
            # Earliest bucket (idx 0) gets lr * llrd^n_buckets,
            # last bucket gets lr * llrd^1,
            # head gets lr (already set above)
            decay_power = n_buckets - bucket_idx
            bucket_lr = lr * (llrd ** decay_power)
            b_decay = [p for n, p in bucket_params if not _is_no_decay(n)]
            b_no_decay = [p for n, p in bucket_params if _is_no_decay(n)]
            if b_decay:
                groups.append({"params": b_decay, "lr": bucket_lr, "weight_decay": weight_decay})
            if b_no_decay:
                groups.append({"params": b_no_decay, "lr": bucket_lr, "weight_decay": 0.0})

    return groups

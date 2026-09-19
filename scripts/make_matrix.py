#!/usr/bin/env python
"""Generate experiment-matrix YAML configs (B augmentation, D imbalance, E tricks, S input size) for a chosen
backbone. Usage: uv run python scripts/make_matrix.py --model resnet18 --mode full --size 64 --epochs 6
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def write(path: Path, cfg: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet18")
    ap.add_argument("--mode", default="full")
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--out", default="configs/matrix")
    args = ap.parse_args()
    out = Path(args.out)
    tag = args.model.replace("mobilenetv3_large_100", "mnv3").replace("efficientnet_", "eff")
    base = dict(model=args.model, mode=args.mode, img_size=args.size, epochs=args.epochs, batch_size=128,
                loss="ce", label_smoothing=0.1, sampler="none", ema=True, lr=0.001, subset_frac=1.0)

    # ---- B: augmentation ladder (+ mix)
    for aug in ["none", "base", "morph", "full", "randaug", "trivial"]:
        write(out / f"B_{aug}_{tag}_{args.size}.yaml", {**base, "exp_id": f"B_{aug}_{tag}_{args.size}", "aug": aug})
    write(out / f"B_full_mixup_{tag}_{args.size}.yaml",
          {**base, "exp_id": f"B_full_mixup_{tag}_{args.size}", "aug": "full", "mixup": 0.2, "mix_prob": 0.5})
    write(out / f"B_full_cutmix_{tag}_{args.size}.yaml",
          {**base, "exp_id": f"B_full_cutmix_{tag}_{args.size}", "aug": "full", "cutmix": 1.0, "mix_prob": 0.5})
    # synthetic as extra data (fill tail classes to 300 / add everything)
    for mode, n in [("fill", 300), ("all", None)]:
        write(out / f"B6_synth_{mode}_{tag}_{args.size}.yaml",
              {**base, "exp_id": f"B6_synth_{mode}_{tag}_{args.size}", "aug": "full",
               "extra_train_index": "data/synth/index.csv", "extra_train_cache": "data/synth/glyphs_synth.npz",
               "extra_mode": mode, **({"extra_fill_to": n} if n else {})})

    # ---- D: imbalance handling (on aug=full)
    D = {
        "D0_ce_ls": {},
        "D1_wce_sqrt": {"loss": "weighted_ce", "ce_weight_power": 0.5},
        "D1_wce_inv": {"loss": "weighted_ce", "ce_weight_power": 1.0},
        "D2_focal": {"loss": "focal", "focal_gamma": 2.0},
        "D2_cbfocal": {"loss": "cb_focal", "cb_beta": 0.999},
        "D3_sampler_sqrt": {"sampler": "sqrt_inv", "sampler_cap": 5.0},
        "D3_sampler_inv_cap10": {"sampler": "inv", "sampler_cap": 10.0},
    }
    for name, over in D.items():
        write(out / f"{name}_{tag}_{args.size}.yaml", {**base, "exp_id": f"{name}_{tag}_{args.size}", "aug": "full", **over})

    # ---- E: tricks
    E = {
        "E5_geometry": {"geometry": True},
        "E6_onoff": {"channel_mode": "onoff"},
        "E6_onoff_geometry": {"channel_mode": "onoff", "geometry": True},
        "E1_llrd": {"llrd": 0.8},
        "E1_noema": {"ema": False},
    }
    for name, over in E.items():
        write(out / f"{name}_{tag}_{args.size}.yaml", {**base, "exp_id": f"{name}_{tag}_{args.size}", "aug": "full", **over})

    # ---- S: input size sweep (aug=full)
    for s in [32, 64, 96, 128, 224]:
        write(out / f"S_{tag}_{s}.yaml", {**base, "exp_id": f"S_{tag}_{s}", "img_size": s, "aug": "full",
                                          "batch_size": (64 if s >= 224 else 128)})
    print("wrote configs to", out)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Train one experiment from a YAML config.

    uv run python scripts/train.py --config configs/smoke.yaml [--set key=value ...]

`--set` accepts dotted keys and YAML-typed values, e.g. `--set epochs=3 --set aug=full --set lr=5e-4`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from thaichar.engine import train_one  # noqa: E402


def _set(cfg: dict, key: str, value: str) -> None:
    parts = key.split(".")
    d = cfg
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = yaml.safe_load(value)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], help="override key=value (dotted keys ok)")
    ap.add_argument("--exp-id", default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}
    for kv in args.set:
        k, v = kv.split("=", 1)
        _set(cfg, k.strip(), v.strip())
    if args.exp_id:
        cfg["exp_id"] = args.exp_id
    cfg.setdefault("exp_id", Path(args.config).stem)

    m = train_one(cfg)
    print(json.dumps({k: m[k] for k in ("exp_id", "top1", "balanced_acc", "macro_f1", "minority_acc",
                                        "best_epoch", "best_which", "sec_per_epoch", "latency_ms_bs1")}))


if __name__ == "__main__":
    main()

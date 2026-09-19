#!/usr/bin/env python
"""Offline ensemble evaluation from saved validation logits (runs/<exp>/val_logits.npy + val_labels.npy).

    uv run python scripts/ensemble_eval.py --runs A1_resnet18_full_64_T4 A2_mnv3_full_64_T4 A3_effb0_full_64_T4

Soft-voting (mean of softmax) and logit averaging are both reported, plus every single member and a tau sweep on
the ensemble. All members must have been evaluated on the same validation split (labels are checked).
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from thaichar.metrics import compute_metrics, tau_sweep  # noqa: E402


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def load(run: str, root: Path):
    d = root / run
    logits = np.load(d / "val_logits.npy").astype(np.float32)
    y = np.load(d / "val_labels.npy")
    m = json.load(open(d / "metrics.json"))
    return logits, y, np.array(m["class_counts_train"])


def brief(m: dict) -> dict:
    return {k: round(m[k], 4) for k in ("top1", "balanced_acc", "macro_f1", "minority_acc")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--root", default="runs")
    ap.add_argument("--out", default=None)
    ap.add_argument("--all-subsets", action="store_true", help="also evaluate every subset of size >= 2")
    args = ap.parse_args()
    root = Path(args.root)
    members = {r: load(r, root) for r in args.runs}
    y0 = next(iter(members.values()))[1]
    counts = next(iter(members.values()))[2]
    for r, (_, y, _) in members.items():
        assert np.array_equal(y, y0), f"{r}: validation labels differ — members must share the split"
    log_prior = np.log(np.clip(counts, 1, None) / counts.sum()).astype(np.float32)

    results = {"members": {}, "ensembles": {}}
    for r, (lg, y, _) in members.items():
        results["members"][r] = brief(compute_metrics(y, lg, counts))
        print(f"{r:45s} {results['members'][r]}")

    def evaluate(names):
        probs = np.mean([softmax(members[n][0]) for n in names], axis=0)
        logit_avg = np.mean([members[n][0] for n in names], axis=0)
        m_soft = compute_metrics(y0, np.log(probs + 1e-9), counts)
        m_logit = compute_metrics(y0, logit_avg, counts)
        sweep = tau_sweep(np.log(probs + 1e-9), y0, log_prior, counts)
        return {"soft_vote": brief(m_soft), "logit_avg": brief(m_logit),
                "tau_sweep_soft": {t: round(v["balanced_acc"], 4) for t, v in sweep.items()}}

    subsets = [tuple(args.runs)]
    if args.all_subsets and len(args.runs) > 2:
        subsets = [c for k in range(2, len(args.runs) + 1) for c in combinations(args.runs, k)]
    for names in subsets:
        key = " + ".join(names)
        results["ensembles"][key] = evaluate(names)
        print(f"\nENSEMBLE [{len(names)}] {key}\n  soft-vote : {results['ensembles'][key]['soft_vote']}"
              f"\n  logit-avg : {results['ensembles'][key]['logit_avg']}"
              f"\n  tau sweep (soft, bal acc): {results['ensembles'][key]['tau_sweep_soft']}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(results, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()

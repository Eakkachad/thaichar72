#!/usr/bin/env python
"""Aggregate runs/*/metrics.json → reports/experiments.csv + reports/experiments.md."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

COLS = ["exp_id", "model", "img_size", "mode", "channel_mode", "aug", "loss", "sampler", "mixup", "cutmix",
        "extra", "split_kind", "subset_frac", "epochs", "seed", "top1", "balanced_acc", "macro_f1",
        "minority_acc", "top5", "tau_best", "tau_best_bal", "tta_top1", "tta_bal", "best_epoch", "best_which",
        "params_M", "sec_per_epoch", "latency_ms", "device"]


def row_from(m: dict) -> dict:
    c = m["cfg"]
    sweep = m.get("tau_sweep") or {}
    tau_best, tau_bal = None, None
    if sweep:
        tau_best = max(sweep, key=lambda t: sweep[t]["balanced_acc"])
        tau_bal = sweep[tau_best]["balanced_acc"]
    tta = m.get("tta") or {}
    return {
        "exp_id": m["exp_id"], "model": c["model"], "img_size": c["img_size"], "mode": c["mode"],
        "channel_mode": c["channel_mode"], "aug": c["aug"], "loss": c["loss"], "sampler": c["sampler"],
        "mixup": c["mixup"], "cutmix": c["cutmix"],
        "extra": (c["extra_mode"] if c.get("extra_train_index") else "-"),
        "split_kind": c["split_kind"], "subset_frac": c["subset_frac"], "epochs": m["epochs_done"],
        "seed": c["seed"], "top1": m["top1"], "balanced_acc": m["balanced_acc"], "macro_f1": m["macro_f1"],
        "minority_acc": m["minority_acc"], "top5": m["top5"], "tau_best": tau_best, "tau_best_bal": tau_bal,
        "tta_top1": tta.get("top1"), "tta_bal": tta.get("balanced_acc"),
        "best_epoch": m["best_epoch"], "best_which": m["best_which"],
        "params_M": round(m["params_total"] / 1e6, 2), "sec_per_epoch": round(m["sec_per_epoch"] or 0, 1),
        "latency_ms": round(m["latency_ms_bs1"], 2), "device": m["device"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="reports/experiments")
    args = ap.parse_args()
    rows = []
    for p in sorted(Path(args.runs).glob("*/metrics.json")):
        try:
            rows.append(row_from(json.load(open(p))))
        except Exception as e:  # noqa: BLE001
            print(f"skip {p}: {e}")
    df = pd.DataFrame(rows, columns=COLS).sort_values(["balanced_acc"], ascending=False)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out + ".csv", index=False)
    num = ["top1", "balanced_acc", "macro_f1", "minority_acc", "top5", "tau_best_bal", "tta_top1", "tta_bal"]
    show = df.copy()
    for c in num:
        show[c] = show[c].map(lambda v: "" if pd.isna(v) else f"{v:.4f}")
    with open(args.out + ".md", "w") as f:
        f.write(f"# Experiment results ({len(df)} runs, sorted by balanced accuracy)\n\n")
        f.write(show.to_markdown(index=False))
        f.write("\n")
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Package a trained run as a deliverable weight file + model card.

    uv run python scripts/export_weights.py --run runs/F14_r18_b6ainit_trivial_20 --name thaichar72_resnet18_64

Writes weights/<name>.pt (state_dict + cfg + class codes, fp32), weights/<name>.card.json (metrics, sizes, md5, how to
load) and appends a row to weights/README.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from thaichar.classes import CLASS_CHARS, CLASS_CODES  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default="weights")
    ap.add_argument("--half", action="store_true", help="store fp16 weights (halves the file; fine for inference)")
    args = ap.parse_args()
    run = Path(args.run)
    ckpt = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    metrics = json.load(open(run / "metrics.json"))
    sd = ckpt["state_dict"]
    if args.half:
        sd = {k: (v.half() if v.is_floating_point() else v) for k, v in sd.items()}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    wpath = out / f"{args.name}.pt"
    torch.save({"state_dict": sd, "cfg": ckpt["cfg"], "class_codes": CLASS_CODES, "class_chars": CLASS_CHARS,
                "epoch": ckpt.get("epoch"), "which": ckpt.get("which"), "source_run": str(run),
                "val_metrics": {k: metrics[k] for k in ("top1", "top5", "balanced_acc", "macro_f1", "minority_acc")}},
               wpath)
    md5 = hashlib.md5(wpath.read_bytes()).hexdigest()
    card = {
        "name": args.name, "file": str(wpath), "bytes": wpath.stat().st_size, "md5": md5, "dtype": "fp16" if args.half else "fp32",
        "model": ckpt["cfg"]["model"], "img_size": ckpt["cfg"]["img_size"], "channel_mode": ckpt["cfg"]["channel_mode"],
        "geometry": ckpt["cfg"]["geometry"], "aug": ckpt["cfg"]["aug"], "init_from": ckpt["cfg"].get("init_from"),
        "epochs": ckpt["cfg"]["epochs"], "seed": ckpt["cfg"]["seed"], "split_kind": ckpt["cfg"]["split_kind"],
        "val_metrics": {k: metrics[k] for k in ("top1", "top5", "balanced_acc", "macro_f1", "minority_acc")},
        "tta": metrics.get("tta"), "tau_sweep": metrics.get("tau_sweep"), "params_total": metrics["params_total"],
        "latency_ms_bs1_cpu": metrics["latency_ms_bs1"], "device_trained": metrics["device"],
        "load": "from thaichar.infer import load_checkpoint, predict_topk; model, cfg = load_checkpoint('%s'); predict_topk(model, cfg, image_path, k=5)" % wpath,
        "caveats": metrics.get("caveats"),
    }
    json.dump(card, open(out / f"{args.name}.card.json", "w"), indent=1, ensure_ascii=False)
    readme = out / "README.md"
    if not readme.exists():
        readme.write_text("# Weights\n\n| name | model | img | aug | init | top-1 | balanced | MB | md5 |\n|---|---|---:|---|---|---:|---:|---:|---|\n")
    with open(readme, "a") as f:
        f.write(f"| {args.name} | {card['model']} | {card['img_size']} | {card['aug']} | {card['init_from'] or 'ImageNet'} | "
                f"{card['val_metrics']['top1']:.4f} | {card['val_metrics']['balanced_acc']:.4f} | {card['bytes']/1e6:.1f} | {md5[:8]} |\n")
    print(json.dumps({k: card[k] for k in ("file", "bytes", "md5", "val_metrics")}, ensure_ascii=False))


if __name__ == "__main__":
    main()

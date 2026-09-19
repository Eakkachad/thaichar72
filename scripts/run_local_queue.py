#!/usr/bin/env python
"""Run a queue of training configs sequentially on THIS machine (GPU if available), skipping finished runs.

Cross-platform (Linux / WSL2 / Windows). Examples:

    uv run python scripts/run_local_queue.py                       # all configs/final_candidates/*.yaml
    uv run python scripts/run_local_queue.py configs/final_candidates/F14*.yaml --set epochs=20
    uv run python scripts/run_local_queue.py configs/matrix/A1_resnet18_full_64.yaml --suffix _doc --set split_kind=doc
    uv run python scripts/run_local_queue.py --list-remaining      # show what would run

A run is "done" when runs/<exp_id><suffix>/metrics.json exists. Each run's stdout/stderr goes to
tasks/logs/local_queue/<exp_id>.log; a one-line summary is appended to tasks/logs/local_queue/queue.log.
On Windows (native, not WSL) DataLoader workers are forced to 0 (spawn-safe).
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def exp_id_of(cfg_path: Path, suffix: str) -> str:
    cfg = yaml.safe_load(cfg_path.read_text()) or {}
    return f"{cfg.get('exp_id', cfg_path.stem)}{suffix}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="*", default=["configs/final_candidates/*.yaml"],
                    help="config files or globs (default: all final candidates)")
    ap.add_argument("--suffix", default="", help="appended to exp_id (e.g. _doc, _s0)")
    ap.add_argument("--set", action="append", default=[], help="override key=value passed to train.py")
    ap.add_argument("--list-remaining", action="store_true")
    ap.add_argument("--stop-on-error", action="store_true")
    args = ap.parse_args()
    os.chdir(ROOT)

    paths: list[Path] = []
    for pat in args.configs:
        hits = sorted(glob.glob(pat))
        if not hits:
            print(f"!! no config matches {pat}", file=sys.stderr)
        paths += [Path(h) for h in hits]
    todo = [(p, exp_id_of(p, args.suffix)) for p in paths]
    remaining = [(p, e) for p, e in todo if not (ROOT / "runs" / e / "metrics.json").exists()]
    print(f"{len(todo)} configs, {len(todo) - len(remaining)} done, {len(remaining)} remaining")
    for p, e in remaining:
        print(f"  - {e}  ({p})")
    if args.list_remaining or not remaining:
        return

    sets = list(args.set)
    if os.name == "nt" and not any(s.startswith("num_workers=") for s in sets):
        sets.append("num_workers=0")  # spawn-based workers are slower than in-process on Windows for this tiny data
    log_dir = ROOT / "tasks" / "logs" / "local_queue"
    log_dir.mkdir(parents=True, exist_ok=True)
    qlog = log_dir / "queue.log"
    for p, e in remaining:
        cmd = [sys.executable, "scripts/train.py", "--config", str(p), "--exp-id", e]
        for s in sets:
            cmd += ["--set", s]
        t0 = time.time()
        with open(log_dir / f"{e}.log", "w", encoding="utf-8") as lf:
            print(f"=== START {e}  ({time.strftime('%H:%M:%S')})", flush=True)
            r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, text=True)
        status = "OK" if r.returncode == 0 and (ROOT / "runs" / e / "metrics.json").exists() else f"FAIL(rc={r.returncode})"
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {e} {status} {time.time() - t0:.0f}s"
        print("===", line, flush=True)
        with open(qlog, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        if status != "OK" and args.stop_on_error:
            sys.exit(1)
    subprocess.run([sys.executable, "scripts/collect_results.py"], check=False)


if __name__ == "__main__":
    main()

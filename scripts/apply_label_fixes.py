#!/usr/bin/env python
"""Apply human-confirmed label fixes to a split file, producing a new versioned split.

Input is the candidate list from scripts/find_label_noise.py after a human has been through
the montages. Nothing is inferred here: a pair is only acted on when it is named explicitly,
and the original split file is never modified.

    # relabel every candidate in two confirmed pairs, drop a third as unusable
    uv run --no-sync python scripts/apply_label_fixes.py \
        --relabel "ช>ซ" "ด>ต" --drop "า>ๅ" \
        --out data/splits/split_seed42_v3.csv

    # or hand it a reviewed CSV with a `verdict` column of relabel / drop / keep
    uv run --no-sync python scripts/apply_label_fixes.py --reviewed my_review.csv --out ...

Writes an audit CSV next to the output so every change is traceable, and refuses to run if a
requested pair is not in the candidate list.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from thaichar.classes import CLASS_CODES, code_to_char  # noqa: E402

CHAR_TO_IDX = {code_to_char(c): i for i, c in enumerate(CLASS_CODES)}


def parse_pairs(items: list[str]) -> list[tuple[str, str]]:
    out = []
    for it in items or []:
        if ">" not in it:
            raise SystemExit(f"pair must look like 'ช>ซ', got {it!r}")
        a, b = it.split(">", 1)
        a, b = a.strip(), b.strip()
        for ch in (a, b):
            if ch not in CHAR_TO_IDX:
                raise SystemExit(f"{ch!r} is not one of the 72 classes")
        out.append((a, b))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-file", default="data/splits/split_seed42_v2.csv")
    ap.add_argument("--candidates", default="reports/analysis/label_noise_candidates.csv")
    ap.add_argument("--reviewed", default=None, help="CSV with path + verdict (relabel/drop/keep)")
    ap.add_argument("--relabel", nargs="*", default=None, help='confirmed swaps, e.g. "ช>ซ"')
    ap.add_argument("--drop", nargs="*", default=None, help='pairs to remove instead of relabel')
    ap.add_argument("--min-conf", type=float, default=0.70)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if not (args.reviewed or args.relabel or args.drop):
        raise SystemExit("nothing to do: pass --relabel / --drop / --reviewed")

    split = pd.read_csv(args.split_file)
    cand = pd.read_csv(args.candidates)
    cand = cand[cand.p_pred >= args.min_conf]

    actions: list[dict] = []

    if args.reviewed:
        rev = pd.read_csv(args.reviewed)
        if "verdict" not in rev.columns or "path" not in rev.columns:
            raise SystemExit("--reviewed needs `path` and `verdict` columns")
        for _, r in rev.iterrows():
            v = str(r["verdict"]).strip().lower()
            if v in ("relabel", "drop"):
                m = cand[cand.path == r["path"]]
                if m.empty:
                    raise SystemExit(f"{r['path']} is not in the candidate list")
                actions.append({"path": r["path"], "action": v,
                                "from": m.iloc[0].true_char, "to": m.iloc[0].pred_char})

    for a, b in parse_pairs(args.relabel):
        sub = cand[(cand.true_char == a) & (cand.pred_char == b)]
        if sub.empty:
            raise SystemExit(f"no candidates for {a}>{b} at p_pred >= {args.min_conf}")
        for p in sub.path:
            actions.append({"path": p, "action": "relabel", "from": a, "to": b})

    for a, b in parse_pairs(args.drop):
        sub = cand[(cand.true_char == a) & (cand.pred_char == b)]
        if sub.empty:
            raise SystemExit(f"no candidates for {a}>{b} at p_pred >= {args.min_conf}")
        for p in sub.path:
            actions.append({"path": p, "action": "drop", "from": a, "to": b})

    act = pd.DataFrame(actions).drop_duplicates(subset=["path"], keep="first")
    relabel = act[act.action == "relabel"]
    dropped = act[act.action == "drop"]

    new = split.copy()
    lut = {p: CHAR_TO_IDX[t] for p, t in zip(relabel.path, relabel.to)}
    n_rel = int(new.path.isin(lut).sum())
    new["label"] = [lut.get(p, l) for p, l in zip(new.path, new.label)]
    if "code" in new.columns:
        new["code"] = new["label"].map(lambda i: CLASS_CODES[int(i)])
    if "char" in new.columns:
        new["char"] = new["label"].map(lambda i: code_to_char(CLASS_CODES[int(i)]))
    before = len(new)
    new = new[~new.path.isin(set(dropped.path))].reset_index(drop=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    new.to_csv(args.out, index=False)
    audit = os.path.splitext(args.out)[0] + "_changes.csv"
    act.to_csv(audit, index=False)

    print(f"source      : {args.split_file}  ({before:,} rows)")
    print(f"relabelled  : {n_rel:,}")
    print(f"dropped     : {before - len(new):,}")
    print(f"result      : {args.out}  ({len(new):,} rows)")
    print(f"audit trail : {audit}")
    if n_rel or (before - len(new)):
        print("\nper-class change in training rows:")
        a_ = split.groupby("label").size()
        b_ = new.groupby("label").size()
        diff = (b_ - a_).dropna()
        diff = diff[diff != 0]
        for lab, dd in diff.items():
            print(f"  {code_to_char(CLASS_CODES[int(lab)])}: {int(a_.get(lab,0))} -> {int(b_.get(lab,0))} ({int(dd):+d})")


if __name__ == "__main__":
    main()

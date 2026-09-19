#!/usr/bin/env python
"""Build data/splits/split_seed42_v2.csv: the SAME rows / strat split / doc split as split_seed42.csv, with the
label corrections from the teammates' DataV2 package applied (2026-09-19).

DataV2 = datav2.zip with <code><char>/<file>.jpg folders. Compared with the original dataset it (a) moved ~630 files
to another class (mostly ว and ใ that were filed under า), (b) dropped ~175 junk/outlier files, (c) added 1,822
pre-augmented copies (ignored here: our pipeline augments online, and their split leaks 570 of them across train/val).
Only (a) and (b) are applied; the dedup, stratified and document-disjoint assignments of split_seed42.csv are kept so
every number stays comparable with the v1 runs. Files whose basename is not unique on either side are left untouched.

    uv run python scripts/make_split_v2.py --zip "/home/CNN/DataV2-20260919T152400Z-1-001/DataV2/datav2.zip"
"""

from __future__ import annotations

import argparse
import collections
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from thaichar.classes import CLASS_CODES  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=None, help="DataV2 datav2.zip (source of truth; omit to replay --changes instead)")
    ap.add_argument("--src", default="data/splits/split_seed42.csv")
    ap.add_argument("--out", default="data/splits/split_seed42_v2.csv")
    ap.add_argument("--changes", default="reports/analysis/datav2_label_changes.csv",
                    help="audit CSV of every applied move/drop (tracked in git); without --zip it is REPLAYED onto --src, "
                         "so the v2 split can be rebuilt on any machine (Colab included) from the repo alone")
    args = ap.parse_args()

    code2idx = {c: i for i, c in enumerate(CLASS_CODES)}
    df = pd.read_csv(args.src)

    if args.zip is None:  # replay mode: apply the tracked change list, no DataV2 zip needed
        ch = pd.read_csv(args.changes)
        ref = df.drop_duplicates("code").set_index("code")[["char", "category"]]
        mv = ch[ch.action == "move"].set_index("path").new_code.astype(int)
        drop = set(ch[ch.action == "drop"].path)
        out = df[~df.path.isin(drop)].copy()
        hit = out.path.isin(mv.index)
        out.loc[hit, "code"] = out.loc[hit, "path"].map(mv).astype(int)
        out["label"] = out.code.map(code2idx)
        out["char"] = out.code.map(ref["char"])
        out["category"] = out.code.map(ref["category"])
        assert out.label.notna().all()
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        out.reset_index(drop=True).to_csv(args.out, index=False)
        print(f"replayed {int(hit.sum())} moves + {len(df) - len(out)} drops from {args.changes}: {len(df)} -> {len(out)} rows -> {args.out}")
        return

    new: dict[str, list[int]] = collections.defaultdict(list)
    for n in zipfile.ZipFile(args.zip).namelist():
        if n.endswith("/"):
            continue
        cls, fn = n.split("/")
        if fn.startswith("aug_"):
            continue
        new[fn].append(int(re.match(r"\d+", cls).group()))

    fn = df.path.str.rsplit("/", n=1).str[-1]
    unique = fn.map(fn.value_counts()) == 1
    # per-class reference row for char / category of the destination class
    ref = df.drop_duplicates("code").set_index("code")[["char", "category"]]

    moved, dropped, kept_ambiguous = [], [], 0
    new_code = df.code.copy()
    drop_mask = pd.Series(False, index=df.index)
    for i, (f, uniq, old) in enumerate(zip(fn, unique, df.code)):
        if not uniq:
            kept_ambiguous += 1
            continue
        codes = new.get(f)
        if codes is None:
            drop_mask.iloc[i] = True
            dropped.append((df.path.iloc[i], int(old), df.split.iloc[i], df.doc_split.iloc[i]))
        elif len(codes) == 1 and codes[0] != old:
            new_code.iloc[i] = codes[0]
            moved.append((df.path.iloc[i], int(old), codes[0], df.split.iloc[i], df.doc_split.iloc[i]))
        elif len(codes) > 1:
            kept_ambiguous += 1

    out = df.copy()
    out["code"] = new_code
    out["label"] = out.code.map(code2idx)
    out["char"] = out.code.map(ref["char"])
    out["category"] = out.code.map(ref["category"])
    out = out[~drop_mask].reset_index(drop=True)
    assert out.label.notna().all() and out.char.notna().all()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    Path(args.changes).parent.mkdir(parents=True, exist_ok=True)
    rows = [{"path": p, "old_code": a, "new_code": b, "action": "move", "split": s, "doc_split": d} for p, a, b, s, d in moved]
    rows += [{"path": p, "old_code": a, "new_code": None, "action": "drop", "split": s, "doc_split": d} for p, a, s, d in dropped]
    pd.DataFrame(rows).to_csv(args.changes, index=False)

    print(f"rows: {len(df)} -> {len(out)} | moved: {len(moved)} | dropped: {len(dropped)} | ambiguous names kept: {kept_ambiguous}")
    mv = collections.Counter((a, b) for _, a, b, _, _ in moved)
    print("top moves:", mv.most_common(8))
    for col in ("split", "doc_split"):
        n_val_moved = sum(1 for r in moved if (r[3] if col == "split" else r[4]) == "val")
        n_val_drop = sum(1 for r in dropped if (r[2] if col == "split" else r[3]) == "val")
        print(f"{col}: val rows relabelled {n_val_moved}, val rows dropped {n_val_drop}, "
              f"val size {int((df[col] == 'val').sum())} -> {int((out[col] == 'val').sum())}")
    delta = (out.code.value_counts() - df.code.value_counts()).dropna().astype(int)
    print("per-class count delta (non-zero):", {int(k): int(v) for k, v in delta[delta != 0].sort_values().items()})


if __name__ == "__main__":
    main()

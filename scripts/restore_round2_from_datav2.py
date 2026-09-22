"""Rebuild `ThaiCharacter Dataset/round2/` from the teammates' DataV2 export.

Why this exists
---------------
The original `ThaiCharacter Dataset/` was not carried over to this machine, but
`reports/eda/files.csv` (tracked in git) is a complete manifest of the original
corpus: 62,705 rows with the original class code and the md5 of every image.
DataV2 contains the *same pixel data* re-filed under corrected class folders, so
the original tree can be reconstructed exactly by placing each file back under
the code recorded in `files.csv` and verifying its md5.

This restores the tree that every existing run, split and cache assumes, so the
93 historical runs stay comparable. Label corrections keep living where they
already do: `reports/analysis/datav2_label_changes.csv` (applied by
`scripts/make_split_v2.py`).

Usage
-----
    uv run python scripts/restore_round2_from_datav2.py \
        --datav2 ~/work/incoming/datav2 \
        --out "ThaiCharacter Dataset/round2"
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict


def md5_of(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def index_datav2(root: str) -> dict[str, list[str]]:
    """basename -> [source paths] (a basename can occur in several class dirs)."""
    by_name: dict[str, list[str]] = defaultdict(list)
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.startswith("aug_"):
                continue  # synthetic augmentations, not part of the original corpus
            by_name[fn].append(os.path.join(dirpath, fn))
    return by_name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datav2", required=True, help="extracted DataV2 image root")
    ap.add_argument("--files-csv", default="reports/eda/files.csv")
    ap.add_argument("--out", default="ThaiCharacter Dataset/round2")
    ap.add_argument("--report", default="reports/analysis/round2_restore.json")
    ap.add_argument("--link", action="store_true",
                    help="hard-link instead of copying (same filesystem only)")
    args = ap.parse_args()

    by_name = index_datav2(args.datav2)
    print(f"DataV2 source files (excluding aug_): "
          f"{sum(len(v) for v in by_name.values())}")

    restored = 0
    md5_ok = 0
    md5_bad: list[str] = []
    missing: list[dict[str, str]] = []
    per_class = Counter()

    with open(args.files_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"manifest rows: {len(rows)}")

    for row in rows:
        rel = row["path"]
        name = os.path.basename(rel)
        want_md5 = row["md5"]
        candidates = by_name.get(name, [])

        src = None
        if len(candidates) == 1:
            src = candidates[0]
        elif candidates:
            # duplicate basename across class dirs -> disambiguate by md5
            for c in candidates:
                if md5_of(c) == want_md5:
                    src = c
                    break
            src = src or candidates[0]

        if src is None:
            missing.append({"path": rel, "code": row["code"]})
            continue

        dst = os.path.join(args.out, str(row["code"]), name)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if args.link:
            if os.path.exists(dst):
                os.remove(dst)
            os.link(src, dst)
        else:
            shutil.copy2(src, dst)

        if md5_of(dst) == want_md5:
            md5_ok += 1
        else:
            md5_bad.append(rel)
        restored += 1
        per_class[row["code"]] += 1

    report = {
        "manifest_rows": len(rows),
        "restored": restored,
        "md5_verified": md5_ok,
        "md5_mismatch": len(md5_bad),
        "md5_mismatch_examples": md5_bad[:20],
        "missing": len(missing),
        "missing_files": missing,
        "classes_restored": len(per_class),
    }
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"restored      : {restored}")
    print(f"md5 verified  : {md5_ok}")
    print(f"md5 mismatch  : {len(md5_bad)}")
    print(f"missing       : {len(missing)}")
    print(f"classes        : {len(per_class)}")
    print(f"report -> {args.report}")


if __name__ == "__main__":
    main()

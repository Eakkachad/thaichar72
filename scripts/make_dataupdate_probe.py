"""Rebuild the dataUpdate train/probe split so the probe is genuinely held out.

`dataUpdate` ships its own train/val split, but it is a per-IMAGE random split over
renders that share an underlying source. Measured on the imported indices, 99.2 % of the
shipped val side reuses a source the train side also has:

    ft : 7275 / 7275 (100.0 %)  -- same (font, class), different degradation
    hw : 2906 / 2993 ( 97.1 %)  -- same scanned glyph, different degradation

So the shipped split answers "can you recognise another degradation of something you
trained on", not "did the tail generalise". This script re-splits the in-universe rows
with GROUP disjointness instead:

    ft -> hold out whole font FAMILIES   (unseen typefaces)
    hw -> hold out whole WRITER/PAGE ids (unseen hands)

Both sides keep all 35 / 34 classes. Outputs `index_train_dj.csv` + `glyphs_train_dj.npz`
and `index_probe_dj.csv` + `glyphs_probe_dj.npz`; the shipped-split files are left alone.

    uv run --no-sync python scripts/make_dataupdate_probe.py
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from thaichar.data import load_cache
from thaichar.synth import build_synth_cache


def writer_key(source_name: str) -> str:
    """'Set2_M1_P-0059_17.jpg' -> 'Set2_M1_P-0059' (the trailing token is the class)."""
    import re
    return re.sub(r"_\d+\.jpe?g$", "", str(source_name), flags=re.I)


def pick_groups(sizes: pd.Series, frac: float, rng: np.random.Generator) -> list[str]:
    """Shuffle groups and take whole ones until `frac` of the rows is held out."""
    groups = sizes.index.to_numpy()
    order = rng.permutation(len(groups))
    target = frac * sizes.sum()
    chosen: list[str] = []
    total = 0
    for i in order:
        if total >= target:
            break
        g = groups[i]
        chosen.append(g)
        total += int(sizes.iloc[i])
    return chosen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/dataupdate")
    ap.add_argument("--frac", type=float, default=0.15, help="target held-out fraction per source")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report", default="reports/analysis/dataupdate_probe_split.json")
    args = ap.parse_args()

    d = args.dir
    parts, caches = [], []
    for idx_name, cache_name in (("index.csv", "glyphs.npz"), ("index_probe.csv", "glyphs_probe.npz")):
        df = pd.read_csv(os.path.join(d, idx_name), low_memory=False)
        parts.append(df)
        caches.append(load_cache(os.path.join(d, cache_name)))
    df = pd.concat(parts, ignore_index=True)
    print(f"in-universe rows: {len(df)}  classes: {df.code.nunique()}")

    # path -> image, across both source caches
    img_of: dict[str, np.ndarray] = {}
    for c in caches:
        for i, p in enumerate(c.paths):
            img_of[p] = c[i]
    missing = [p for p in df["path"] if p not in img_of]
    if missing:
        raise SystemExit(f"{len(missing)} rows have no cached image, e.g. {missing[:3]}")

    df["group"] = np.where(
        df["source"] == "ft",
        "ft:" + df["family"].astype(str),
        "hw:" + df["source_name"].map(writer_key),
    )

    rng = np.random.default_rng(args.seed)
    held: list[str] = []
    for src in ("ft", "hw"):
        sizes = df.loc[df["source"] == src].groupby("group").size()
        chosen = pick_groups(sizes, args.frac, rng)
        held.extend(chosen)
        print(f"{src}: held out {len(chosen)}/{len(sizes)} groups "
              f"= {int(sizes[chosen].sum())}/{int(sizes.sum())} rows")

    is_probe = df["group"].isin(set(held))
    train, probe = df[~is_probe].copy(), df[is_probe].copy()

    # sanity: group disjointness and class coverage
    assert not (set(train["group"]) & set(probe["group"])), "group leak between sides"
    tr_cls, pb_cls = set(train.code), set(probe.code)
    print(f"\ntrain rows {len(train)} / {len(tr_cls)} classes")
    print(f"probe rows {len(probe)} / {len(pb_cls)} classes")
    if tr_cls - pb_cls:
        print(f"  classes absent from probe: {sorted(tr_cls - pb_cls)}")
    if pb_cls - tr_cls:
        raise SystemExit(f"classes only in probe: {sorted(pb_cls - tr_cls)} -- would be untrainable")

    per_class = probe.groupby(["code", "char"]).size().reset_index(name="n").sort_values("n")
    print(f"probe per-class n: min {per_class.n.min()} median {int(per_class.n.median())} max {per_class.n.max()}")
    thin = per_class[per_class.n < 30]
    if len(thin):
        print("  thin probe classes (<30):")
        print(thin.to_string(index=False))

    for name, sub in (("train_dj", train), ("probe_dj", probe)):
        idx_p = os.path.join(d, f"index_{name}.csv")
        cache_p = os.path.join(d, f"glyphs_{name}.npz")
        sub.drop(columns=["group"]).to_csv(idx_p, index=False)
        build_synth_cache([img_of[p] for p in sub["path"]], sub["path"].tolist(), cache_p)
        print(f"  index -> {idx_p}  ({len(sub)} rows)\n  cache -> {cache_p}")

    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump({
            "seed": args.seed, "target_frac": args.frac,
            "n_total": len(df), "n_train": len(train), "n_probe": len(probe),
            "held_out_groups": sorted(held),
            "held_out_font_families": sorted(g[3:] for g in held if g.startswith("ft:")),
            "n_held_out_writers": sum(1 for g in held if g.startswith("hw:")),
            "probe_per_class": per_class.set_index("char")["n"].to_dict(),
        }, f, ensure_ascii=False, indent=2)
    print(f"\nreport -> {args.report}")


if __name__ == "__main__":
    main()

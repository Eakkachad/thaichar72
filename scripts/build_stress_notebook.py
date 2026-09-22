#!/usr/bin/env python
"""Build notebooks/ThaiChar72_StressEval.ipynb from notebooks/_stress_cells/*.py.

Edit the cell sources in `notebooks/_stress_cells/`, or the markdown below, and re-run this --
never edit the .ipynb, it is generated (same rule as scripts/build_notebook.py).

The notebook is for INSPECTING the stress suite: the tables say what happened, and every section
has a drill-down that shows the actual images and what the model said about them, because a
corruption that is wrong in code produces a clean number that means nothing.

    uv run --no-sync python scripts/build_stress_notebook.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
CELLS = ROOT / "notebooks" / "_stress_cells"
OUT = ROOT / "notebooks" / "ThaiChar72_StressEval.ipynb"

INTRO = """# ThaiChar72 — Stress evaluation

Companion to `reports/stress/summary.md` and `reports/02-EXPERIMENTS.md` §N. The report states the
conclusions; this notebook is for **checking them yourself** — every section pairs its table with a
drill-down that shows the actual images and what the model predicted.

**Run it**

```bash
cd ~/work/thaichar72
uv pip install jupyterlab ipykernel          # not in the locked deps
uv run --no-sync python -m ipykernel install --user --name thaichar --display-name "thaichar"
uv run --no-sync jupyter lab notebooks/ThaiChar72_StressEval.ipynb
```

Pick the **thaichar** kernel and Run All. Use `uv pip install`, never `uv sync` or a bare `uv run`
(`tasks/HANDOFF-2026-09-22.md` §1.1), and run it from `~/work/thaichar72` — running `uv` against the
`/mnt/d` copy from WSL deletes that copy's venv.

Regenerate the underlying numbers any time with:

```bash
uv run --no-sync python scripts/stress_suite.py --models all --families A B C
```

**The three families are scored differently on purpose**

| family | is there a correct answer? | what is reported |
|---|---|---|
| **A** degraded (blur, pixelate, rotation, unseen fonts…) | yes, unchanged | top-1 retention vs severity |
| **B** invalid (two glyphs touching, half a glyph, scribbles…) | **no** | confidence and separability — **never accuracy** |
| **C** identity-changing (mirror, upside down, 90°) | maybe a *different* class | a mapping table |

Reporting accuracy on a picture of two glued characters would be meaningless, because no label is
correct. What family B measures instead is whether the model signals that it is out of its depth.
"""

SECTIONS = [
    ("c1_setup.py", "## 1. Setup and stress artefacts\n\nLoads `reports/stress/results.csv` and "
                    "`flip_map.csv`, and prints the clean baseline every retention number is "
                    "divided by."),
    ("c2_harness.py", "## 2. Harness sanity check — and the bug it caught\n\nThe suite's acceptance "
                      "criterion is that its clean baseline reproduces the model's known validation "
                      "top-1. It did not, and that is how a **real deployment bug** surfaced: "
                      "`infer.preprocess_image` scored **0.9695 against the dataset transform's "
                      "0.9898** on the same images — wrong on 10, right on none. `_reframe` padded "
                      "tight crops with the *median* of the four corners, which on a glyph whose "
                      "strokes touch the corners is ink-coloured; Otsu then read that pad as "
                      "background and inverted the image. It also silently defeated "
                      "`--polarity both`, since inverting the input moved the bad guess rather than "
                      "fixing it. Using the **lightest** corner restores 0.9898 exactly. This cell "
                      "re-measures both paths so the fix stays verified."),
    ("c3_famA.py", "## 3. Family A — degraded input, the answer still exists"),
    ("c4_showcorr.py", "### 3b. Drill-down: see the corruption and the mistakes it causes\n\n"
                       "`show_corruption(\"pixelate\", 0.25)` · `show_corruption(\"rotate_hard\", 30)` "
                       "· `show_corruption(\"stroke_extreme\", -4)` · `show_corruption(\"unseen_font\")`"),
    ("c5_famB.py", "## 4. Family B — invalid input, no correct answer exists\n\nNothing here reports "
                   "accuracy. `separability AUROC` answers the only actionable question: can a "
                   "confidence threshold alone tell this junk from clean input? **≥ 0.8 yes; ≤ 0.6 "
                   "no, and the input pipeline would have to be fixed instead** (e.g. splitting "
                   "touching glyphs before classifying)."),
    ("c6_showjunk.py", "### 4b. Drill-down: what the model confidently calls junk\n\n"
                       "`show_junk(\"overlap_2_50\")` · `show_junk(\"touching_2\")` · "
                       "`show_junk(\"half_50\")` · `show_junk(\"strokes\")` · "
                       "`show_junk(\"latin_digits\")`"),
    ("c7_famC.py", "## 5. Family C — the transform may change the answer\n\nThai is not "
                   "flip-invariant, which is why `CLAUDE.md` bans flips in augmentation. This is the "
                   "measurement behind that rule: a flip does not merely perturb the image, it can "
                   "produce a different real Thai character."),
    ("c8_showflip.py", "### 5b. Drill-down: watch a character become another one\n\n"
                       "`show_flip(\"mirror_v\")` · `show_flip(\"mirror_h\")` · "
                       "`show_flip(\"rot180\")` · `show_flip(\"rot90_cw\")`"),
    ("c9_threshold.py", "## 6. Picking a confidence threshold\n\nFamily B showed a threshold is "
                        "enough to filter invalid input. This is where the number gets chosen: what "
                        "each threshold costs on real glyphs against how much junk still leaks "
                        "through."),
    ("c10_own.py", "## 7. Run the same checks on your own images\n\nPoint `MY_DIR` at a folder and "
                   "re-run. This is the same code path as the on-the-day tool "
                   "`scripts/evaluate_folder.py`."),
]


def build() -> nbf.NotebookNode:
    cells = [nbf.v4.new_markdown_cell(INTRO)]
    for fname, md in SECTIONS:
        src = (CELLS / fname).read_text(encoding="utf-8").rstrip("\n")
        cells.append(nbf.v4.new_markdown_cell(md))
        cells.append(nbf.v4.new_code_cell(src))
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata = {
        "kernelspec": {"display_name": "thaichar", "language": "python", "name": "thaichar"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    return nb


def main() -> None:
    nb = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, str(OUT))
    print(f"Generated {OUT.relative_to(ROOT)} with {len(nb.cells)} cells "
          f"({sum(1 for c in nb.cells if c.cell_type == 'code')} code).")


if __name__ == "__main__":
    main()

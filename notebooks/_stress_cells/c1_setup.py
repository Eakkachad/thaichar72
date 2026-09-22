# Setup: paths, Thai font, and the stress-suite artefacts.
%matplotlib inline
import os, sys, json
from pathlib import Path

PROJECT_DIR = Path.cwd()
while not (PROJECT_DIR / "src" / "thaichar").exists() and PROJECT_DIR != PROJECT_DIR.parent:
    PROJECT_DIR = PROJECT_DIR.parent
os.chdir(PROJECT_DIR)
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib import font_manager

fp = PROJECT_DIR / "assets" / "fonts" / "Sarabun-Regular.ttf"
if fp.exists():
    font_manager.fontManager.addfont(str(fp))
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(fp)).get_name()
plt.rcParams["axes.unicode_minus"] = False

STRESS = PROJECT_DIR / "reports" / "stress"
print("project :", PROJECT_DIR)
print("stress  :", STRESS, "(exists:", STRESS.exists(), ")")

if not (STRESS / "results.csv").exists():
    print()
    print("No results yet. Generate them with:")
    print("   uv run --no-sync python scripts/stress_suite.py --models all --families A B C")
else:
    res = pd.read_csv(STRESS / "results.csv")
    flip = pd.read_csv(STRESS / "flip_map.csv") if (STRESS / "flip_map.csv").exists() else pd.DataFrame()
    clean = res[res.family == "clean"].set_index("model")["top1"]
    print()
    print("rows:", len(res), " models:", res.model.nunique(), " conditions:", res.condition.nunique())
    print()
    print("clean baseline (493 held-out val glyphs, deployed preprocessing + polarity-both):")
    for m, v in clean.items():
        print(f"   {m:<34} {v:.4f}")

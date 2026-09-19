# HANDOFF — state at 2026-09-19 18:00 (laptop, CPU-only) → continue on the RTX 4060 machine

Read `CLAUDE.md` first. This file says exactly where things stand and what to run next, in order.

## 0. Why we moved
Colab free tier: T4 sessions are pruned every ~1 h and the daily GPU quota ran out at 16:40 after ~50 runs.
Everything below needs a GPU; on an RTX 4060 a 20-epoch resnet18@64 run should take ~5 min.

## 1. Set-up on the new machine (WSL2 Ubuntu recommended; native Windows also works)
```bash
tar xzf Deep_CNN_handoff_*.tar.gz && cd Deep_CNN          # or: git clone <github url> && copy data/ + runs/ from the tarball
curl -LsSf https://astral.sh/uv/install.sh | sh              # if uv is missing
uv sync                                                      # picks CUDA 12.6 wheels on Windows/WSL2, CPU wheels on other Linux
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
uv run pytest -q tests/test_data.py tests/test_augment.py tests/test_infer.py      # ~1 min sanity
uv run python scripts/run_local_queue.py --list-remaining                          # should list F5…F17
git remote -v                                                # origin = GitHub (see CLAUDE.md §5); git pull first
```
If `data/cache/glyphs.npz`, `data/splits/`, `data/synth/`, `data/external/` are missing, regenerate (≈15 min):
`uv run python scripts/eda.py --data "ThaiCharacter Dataset/round2" --out reports/eda && uv run python scripts/prep_data.py
&& uv run python scripts/render_synth.py --per-class 300 --seed 0 && uv run python scripts/fetch_external.py --sources alice kvis burapha --out data/external`
(the dataset itself must be present at `ThaiCharacter Dataset/round2/<code>/*.jpg`; it is in the tarball).

## 2. What is DONE (all numbers in reports/02-EXPERIMENTS.md; table reports/experiments.md)
- Data: 62,707 → 60,117 after dedup; splits (strat + doc), caches, 21,600 synthetic glyphs (26 fonts), 100,985 external
  handwritten glyphs (ALICE-THI + Burapha-TH). Engine, notebook (runs end-to-end on CPU in inference mode), robustness,
  predict CLI, figures, README.
- Studies at 64 px / 6 epochs: A (backbones & freeze modes), B (aug ladder), D (imbalance), E (tricks), S (input size),
  C (stage-1/2 pretraining), G (stratified vs document-disjoint) — conclusions:
  * full fine-tune only; ImageNet init +6 balanced vs scratch; effb0 ≈ resnet18 > mnv3; 64 px is the sweet spot.
  * aug: **`base` (affine+margin) hurts cross-document generalisation (doc bal 93.6)**; none/trivial/morph ≈ 97.6–97.9
    on doc; randaug 97.0; mixup/cutmix bad. Synthetic fonts as **pretraining** best (B6a_ft 97.92/98.43 at 6 ep).
  * imbalance: sqrt-inverse loss/sampler → bal 98.4 but top-1 −0.5…−0.9; inverse weights collapse; logit-adjust τ is a free knob.
  * geometry side-channel and ON/OFF channels: no gain; geometry FAILS cross-document (−2.8) → excluded.
  * TTA (8 canvas views) helps aug-trained models slightly, hurts the no-aug model.
- 20-epoch candidates finished so far (stratified val, seed 42):
  | exp | top-1 | balanced | TTA top-1 / bal |
  |---|---:|---:|---:|
  | F1_r18_none_20 | 0.9859 | 0.9828 | 0.9790 / 0.9790 (worse) |
  | F2_r18_randaug_20 | 0.9837 | 0.9818 | 0.9836 / 0.9830 |
  | F3_r18_randaug_synth_20 | 0.9845 | 0.9804 | 0.9825 / 0.9801 |
  | F4_effb0_randaug_synth_20 | 0.9786 | 0.9775 | unstable (best epoch 4) |
  Ensemble F1+F2 soft-vote: 0.9862 / 0.9829 (marginal — errors are shared: า↔ๅ, ว→า, ั↔้).

## 3. What to RUN next (in this order; each 20-epoch run ≈ 5 min on a 4060)
1. **Remaining final candidates** (13 configs, skip-done):
   `uv run python scripts/run_local_queue.py configs/final_candidates/*.yaml`
   Most promising by the 6-epoch evidence: F14 (synthetic-pretrain init + TrivialAugment), F15 (+synthetic extra),
   F8/F9 (trivial ± synth), F16 (init + morph), F11/F12 (external-pretrain init). Needs the init checkpoints
   `runs/B6a_synth_pretrain_resnet18_64/best.pt` and `runs/C1_ext_pretrain_resnet18_64/best.pt` (in the tarball).
2. **Doc-disjoint check of the top-3** (by strat balanced acc, ties → prefer aug≠none):
   `uv run python scripts/run_local_queue.py configs/final_candidates/<top3>.yaml --suffix _doc --set split_kind=doc`
   Pick the FINAL recipe by: (a) doc top-1 first (grading is accuracy on unseen-ish data), (b) doc balanced, (c) strat
   top-1; require robustness curve not worse than F2's (`scripts/robustness.py --binarize`).
3. **3 seeds of the winner** (`--suffix _s0 --set seed=0 --set split_file=data/splits/split_seed0.csv`, same for seed 1)
   → report mean ± std in FINAL-REPORT §9.
4. **Boosts**: ensemble of the 3 seeds + best other backbone (`scripts/ensemble_eval.py --runs ...`); knowledge
   distillation into one resnet18 (`kd_teachers: [runs/<winner>/best.pt, ...]`, `kd_alpha 0.7`, `kd_T 4`) — compare;
   optional MobileNetV3 student for a "small model" story.
5. **Package**: `uv run python scripts/export_weights.py --run runs/<winner> --name thaichar72_resnet18_64`
   (+ `--half` variant), commit `weights/`. Run `scripts/robustness.py --ckpt runs/<winner>/best.pt --binarize`
   (full val) and `scripts/error_analysis.py --run runs/<winner>`; regenerate `scripts/result_figures.py`.
6. **Notebook**: set `FINAL_CONFIG`/`WEIGHTS_PATH` defaults in `scripts/build_notebook.py` to the winner, rebuild,
   `uv run python scripts/run_notebook_local.py --mode inference --weights weights/thaichar72_resnet18_64.pt`,
   then test on real Colab if possible (`colab exec -f notebooks/ThaiChar72_Colab.ipynb`, see tasks/COLAB-USAGE.md)
   or at least open it in Colab with MODE="inference" and the weights in Drive.
7. **Report**: fill `reports/FINAL-REPORT.md §9 "โมเดลสุดท้าย"` (recipe, strat/doc/TTA/3-seed numbers, ensemble/KD,
   weight size + load snippet) and refresh `reports/02-EXPERIMENTS.md` with the F table. Slides come from
   `reports/figures/results/` + `reports/eda/` + `reports/analysis/<winner>/` + `reports/robustness/<winner>/`.
   Report in Thai to the owner after each step.

## 4. Known gaps / gotchas
- `minority_acc` is NaN for runs that used `extra_train_*` before the fix (B6_synth_*); fixed in engine (uses real counts).
- `colab_*.sh` runners assume the `colab` CLI + OAuth on the laptop; not needed on the GPU box.
- `runs/*/best.pt` for most 6-epoch study runs were NOT copied (2 GB) — only the ones listed in the tarball manifest.
  Their metrics/logits are present; retrain if a checkpoint is needed.
- Windows native: `num_workers` forced to 0 by `run_local_queue.py`; if calling `train.py` directly add `--set num_workers=0`.
- Fonts for plots: `assets/fonts/Sarabun-Regular.ttf` (matplotlib must register it, all scripts do).
- Do not evaluate a strat-trained model on `doc_split=="val"` (overlap) — notebook and scripts already guard this.

## 5. Deliverables checklist (owner's §2)
- [ ] `notebooks/ThaiChar72_Colab.ipynb` runs on Colab (Train + Inference sections, config in one cell) — exists, needs final weights + Colab test
- [ ] final weights in `weights/` with size + load instructions — script ready, waiting for the winner
- [ ] `reports/FINAL-REPORT.md` complete (§9 pending) + figures — draft exists
- [x] `uv` project (pyproject + lock; torch wheel auto-selected by platform markers)

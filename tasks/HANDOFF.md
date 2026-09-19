# HANDOFF — state at 2026-09-19 late evening (RTX 4060 / WSL2, `/home/CNN/Deep_CNN`) — §3 below is DONE except the Colab test

Read `CLAUDE.md` first. This file says exactly where things stand and what to run next, in order.
Owner-facing machine setup (Thai): `tasks/HANDOFF-OWNER-TH.md`. Repo: https://github.com/Eakkachad/thaichar72 (private).

## −1. Status after the RTX 4060 session (2026-09-19 20:30–23:05) — read this first
Everything in §3 was run on the 4060 (`tasks/run_step1*.sh`; logs `tasks/logs/local_queue/`). Results and interpretation:
`reports/02-EXPERIMENTS.md` §G (confound fixed), §F, §F-doc, §H (label audit), §I (KD/deliverable); `reports/FINAL-REPORT.md` §9–§12.
- **Winner**: F19 = resnet18 @64, ImageNet → synthetic-font pretrain (B6a) → real, RandAugment N=2, CE+LS, EMA, 20 ep. Chosen by §3.3
  (doc top-1 0.9822, only randaug/full recipes pass the robustness gate; morph/trivial never see noise ops and collapse on salt-pepper).
- **Label audit (DataV2 from teammates)**: 629 relabels (416 ว + 107 ใ filed under า, …) + 175 drops verified by eye → applied to our split as
  `data/splits/split_seed42_v2.csv` (rebuild anywhere: `uv run python scripts/make_split_v2.py` replays `reports/analysis/datav2_label_changes.csv`).
  Same recipe on v2: 0.9879 → 3 seeds 0.9890 ± 0.0008 (v1 ceiling was ~0.985). v1↔v2 cross-evaluation is symmetric ±0.9 pt (§H-1).
- **Deliverable**: `weights/thaichar72_resnet18_64.pt` = K1, KD of the 3 F19_v2 seeds into one resnet18 (0.9903 / 0.9875 / F1 0.9869 on v2 val,
  robust 0.9104 = best); `_fp16.pt` identical metrics; `_v1labels.pt` = F19 trained on the ORIGINAL labels (hedge: if the hidden test follows the
  `be`-source label convention, v2 loses ~0.9 pt); `thaichar72_r18_synth_pretrain_init.pt` = stage-1 init for retraining. `configs/final.yaml` = F19 on v2.
- **Notebook**: rebuilt (`scripts/build_notebook.py` now honours the recipe's split_file, rebuilds the v2 split on Colab from the change list, uses the
  packaged stage-1 init); `run_notebook_local.py --mode inference` passes with 0 errors (99.03 % / 98.75 % printed). **NOT yet run on Colab** — owner:
  upload repo + `weights/` (+ `data/cache/glyphs.npz`, `data/splits/split_seed42.csv` for inference mode) and run with MODE="inference".
- **Open decision for the owner**: which weight to submit (v2 default vs v1 hedge) — see FINAL-REPORT §12; ask the teacher about ว/า labelling if possible.
- Extras DONE (`configs/extras/X1–X6`, 02-EXPERIMENTS §J): F19 recipe on effb0 0.9901/0.9886, convnext_tiny 0.9898/0.9869, mnv3-large 0.9893/0.9870,
  KD into mnv3-small 0.9900/0.9872 (1.6 M params, 3.4 ms) → exported as `weights/thaichar72_mnv3small_64_small.pt`; all pass the robustness gate;
  cross-architecture KD does not help. Not done: variance-across-splits (`_sp0`), doc-split KD.
- **Moving to native Windows 11 (owner's plan)**: build the package with `bash tasks/make_handoff_tarball.sh` (repo + .git + dataset + data + weights
  + 23 checkpoints; `.venv` excluded), extract with `tar -xzf` in PowerShell into e.g. `C:\work\Deep_CNN` (not OneDrive), then `uv sync` picks the cu126
  wheels automatically; `python scripts/train.py` / `run_local_queue.py` work as-is (workers forced to 0 → ~2× slower per epoch than WSL);
  the `tasks/run_*.sh` lane scripts are bash-only — run the `uv run python scripts/run_local_queue.py …` lines inside them one per line in PowerShell,
  or keep using WSL for training. `git pull` first: the tarball's `.git` is exactly `origin/main` at build time.
- Machine notes: 1 training lane uses ~35 % GPU / 2 GB (CPU-bound pipeline) → run 2–4 configs concurrently (`run_step1_3b_ext.sh` pattern);
  `agy` not installed (SELF rows in DELEGATION-LOG); WSL git uses the Windows Git Credential Manager (push works). Helper tools outside the repo:
  `/home/CNN/tools/{cmp,ftable,robmean,datav2_check}.py`.

---
*Everything below is the original hand-off written on the laptop; kept for the set-up commands and the reasoning behind §3.*

## 0. Why we moved
Colab free tier: T4 sessions are pruned every ~1 h and the daily GPU quota ran out at 16:40 after ~50 runs.
Everything below needs a GPU; on an RTX 4060 a 20-epoch resnet18@64 run should take ~5 min (WSL2, `num_workers=4`).

## 1. Set-up on the new machine
Prerequisites (Windows side): NVIDIA driver ≥ 560 (CUDA 12.6) — `nvidia-smi` in PowerShell; WSL2 (`wsl --update`,
`wsl -l -v` shows VERSION 2; `nvidia-smi` must also work inside Ubuntu via /usr/lib/wsl/lib — never apt-install NVIDIA
drivers/toolkit inside WSL). Keep the project on the WSL ext4 disk (`~/work`), never under `/mnt/c`. ≥ 10 GB free.
```bash
# WSL2 Ubuntu
sudo apt update && sudo apt install -y git pigz curl
curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH="$HOME/.local/bin:$PATH"     # or: source ~/.bashrc
mkdir -p ~/work && cd ~/work && tar -I pigz -xf /mnt/c/Users/<you>/Downloads/Deep_CNN_handoff_2026-09-19.tar.gz
cd Deep_CNN
gh auth login            # or: git config --global credential.helper store  (private repo: pull AND push need a token)
git pull                 # tarball .git may be a few commits behind origin/main — pull BEFORE syncing
uv sync                  # ≈ 4 GB download on WSL2 (torch cu126 + nvidia-* + triton), ≈ 3 GB native Windows
uv run python -c "import torch,platform;print(torch.__version__, platform.release(), torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"
#   expect: 2.14.0+cu126 <kernel>-microsoft-standard-WSL2 True NVIDIA GeForce RTX 4060 ...
#   '+cpu'  → uname -r lacks "microsoft" (WSL1/custom kernel): fix WSL2, do not hand-install wheels (uv run re-syncs)
#   '+cu126' but False → Windows driver too old (< 560) or GPU not exposed to WSL
curl -sI https://huggingface.co | head -1                                   # timm pretrained weights come from HF hub
uv run python -c "import timm; [timm.create_model(m, pretrained=True) for m in ('resnet18','efficientnet_b0')]"   # pre-warm (~70 MB)
uv run pytest -q tests/test_data.py tests/test_augment.py tests/test_infer.py   # ~1 min, offline
uv run python scripts/train.py --config configs/smoke.yaml --exp-id smoke_gpu --set device=cuda --set out_root=/tmp/smoke_runs   # ~1 min; must say device cuda
uv run python scripts/run_local_queue.py --list-remaining
```
Native Windows instead of WSL2: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`,
extract with built-in `tar -xzf` (bsdtar) into `C:\work\Deep_CNN` (not Desktop/Documents = OneDrive), `setx PYTHONUTF8 1`,
`setx HF_HUB_DISABLE_SYMLINKS_WARNING 1`, `git config --global core.autocrlf false`; run the same `uv ...` commands in
PowerShell one per line (no `&&` in PS 5.1). Workers default to 0 on Windows → epochs slower than the estimate.
If the data caches are missing (they are in the tarball), regenerate (≈ 15 min; kvis is gated and auto-skipped,
expected 100,985 external glyphs):
`uv run python scripts/eda.py --data "ThaiCharacter Dataset/round2" --out reports/eda` → `scripts/prep_data.py` →
`scripts/render_synth.py --per-class 300 --seed 0` → `scripts/fetch_external.py --sources alice burapha --out data/external`.

## 2. What is DONE (numbers: reports/02-EXPERIMENTS.md; table: reports/experiments.md; all 64 px unless noted)
- Data: 62,707 → 60,117 after dedup; strat + doc splits; caches; 21,600 synthetic glyphs (26 fonts); 100,985 external
  handwritten glyphs. Engine, notebook (runs on CPU in inference mode), robustness, predict CLI, figures, README.
- 6-epoch studies A/B/D/E/S/C/G, conclusions:
  * full fine-tune only (frozen/partial poor); ImageNet-pretrained resnet18 beats the 0.43 M SmallCNN from scratch by
    +6 balanced; effb0 ≳ resnet18 ≳ mnv3 within 0.35 pt balanced, top-1 indistinguishable (97.5–97.7); 64 px is the sweet spot.
  * aug on the stratified split at 6 ep: none 98.48/98.15 > randaug/trivial ≈ 98.2/98.25 > base 97.53/97.94 > morph > full;
    mixup/cutmix bad (bal 94–95). Synthetic fonts as **pretraining** (B6a_ft 97.92/98.43) > synthetic as extra data (98.04).
  * document-disjoint (full data): trivial 97.90/97.84, none 97.80/97.61, morph 97.29/97.86, randaug 97.80/97.00,
    full 97.26/96.11, sampler-sqrt 96.70/98.16 (minority 99.2). **Caveat:** the doc runs of A0/A1/A3 (base aug) used only
    25 % of the training data (`subset_frac 0.25`), so "base hurts generalisation" is NOT yet established — rerun below.
  * imbalance: sqrt-inverse loss or sampler → bal 98.4 but top-1 −0.5…−0.9; inverse weights collapse; τ is a free knob.
  * geometry side-channel: +0.5 bal in-distribution but −2.0 bal on doc vs B_full_doc → excluded. ON/OFF channels: no gain.
  * TTA (8 canvas views) is neutral-to-negative on top-1 for all F runs (−0.01…−0.69 pt); only F2 gains +0.12 balanced.
    → rank candidates on RAW `top1`/`balanced_acc` from metrics.json; TTA is an explicit on/off decision later.
- 20-epoch candidates finished (stratified val, seed 42, raw):
  | exp | top-1 | balanced | TTA top-1 / bal |
  |---|---:|---:|---:|
  | F1_r18_none_20 | 0.9859 | 0.9828 | 0.9790 / 0.9790 |
  | F2_r18_randaug_20 | 0.9837 | 0.9818 | 0.9836 / 0.9830 |
  | F3_r18_randaug_synth_20 | 0.9845 | 0.9804 | 0.9825 / 0.9801 |
  | F4_effb0_randaug_synth_20 | 0.9786 | 0.9775 | 0.9776 / 0.9775 (unstable, best epoch 4) |
  Ensemble F1+F2 soft-vote 0.9862/0.9829 (marginal — shared errors า↔ๅ, ว→า, ั↔้). B_base_*_T4 ≡ A1_*_T4 (same config, rerun).

## 3. What to RUN next (in order). Always `--set device=cuda --stop-on-error`; on WSL2 add `--set num_workers=4`.
1. **Confound check (3 min)** — full-data doc run of the base-aug recipe:
   `uv run python scripts/run_local_queue.py configs/matrix/A1_resnet18_full_64.yaml --suffix _doc_full --set split_kind=doc --set subset_frac=1.0 --set device=cuda --stop-on-error`
   then re-derive §G in reports/02-EXPERIMENTS.md (compare with B_none/B_trivial/B_full `_doc` rows, all full data).
   Also a 6-ep doc run of the synthetic-pretrain init: `configs/pretrain/B6a_ft_resnet18_64.yaml --suffix _doc --set split_kind=doc`.
2. **Remaining final candidates** (F5–F17; F5 geometry and F6 onoff are deliberate negative controls — run them last):
   `uv run python scripts/run_local_queue.py configs/final_candidates/F{7,8,9,10,11,12,13,14,15,16,17}_*.yaml --set device=cuda --stop-on-error`
   then `configs/final_candidates/F{5,6}_*.yaml`. Needs `runs/B6a_synth_pretrain_resnet18_64/best.pt` and
   `runs/C1_ext_pretrain_resnet18_64/best.pt` (shipped). If HF hub is unreachable, F11–F17 can use `--set pretrained=false`
   (init_from overwrites every tensor anyway).
3. **Doc-disjoint check of the top-3** (by raw strat balanced acc; ties → prefer aug ≠ none):
   `uv run python scripts/run_local_queue.py configs/final_candidates/<top3>.yaml --suffix _doc --set split_kind=doc --set device=cuda --stop-on-error`
   Winner rule: (a) doc top-1, (b) doc balanced, (c) strat top-1; and robustness not worse than F2:
   `uv run python scripts/robustness.py --ckpt runs/<cand>/best.pt --binarize --out reports/robustness/<cand>_full_bin`
   criterion = mean top-1 over all non-clean rows of results.csv ≥ F2's **0.9038** (full val, Otsu on; clean 0.9837;
   `reports/robustness/F2_r18_randaug_20_full_bin/results.csv`, committed from the laptop) − 0.005.
4. **3 seeds of the winner, SAME split** (so they can be ensembled): `--suffix _s0 --set seed=0` and `--suffix _s1 --set seed=1`
   (keep `split_file=data/splits/split_seed42.csv`). Report mean ± std (top-1, balanced) in FINAL-REPORT §9.
   Optional variance-across-splits study: `--suffix _sp0 --set split_file=data/splits/split_seed0.csv` (report only, not ensembled).
5. **Boosts**: `uv run python scripts/ensemble_eval.py --runs <winner> <winner>_s0 <winner>_s1 [F10_effb0_trivial_synth_20] --all-subsets --out reports/analysis/ensemble_final.json`
   (members must share split_seed42). Distillation: edit `configs/final_candidates/K1_r18_kd_20.yaml` (`kd_teachers` = winner
   + one diverse 64-px gray3 teacher) and run it; compare with the winner. F4's best.pt was NOT shipped (unstable run);
   use F10_effb0_trivial_synth_20 as the EfficientNet teacher once trained.
6. **Package**: `uv run python scripts/export_weights.py --run runs/<winner> --name thaichar72_resnet18_64` and
   `... --half --name thaichar72_resnet18_64_fp16` (export refuses to overwrite). Commit `weights/`. Then
   `scripts/error_analysis.py --run runs/<winner>`, `scripts/result_figures.py`, `scripts/collect_results.py`.
7. **Notebook**: copy the winner config to `configs/final.yaml`; `scripts/build_notebook.py` already prefers
   `weights/thaichar72_resnet18_64.pt`; rebuild + `uv run python scripts/run_notebook_local.py --mode inference --weights weights/thaichar72_resnet18_64.pt`;
   then test on Colab (upload notebook + weights to Drive, MODE="inference") — `tasks/COLAB-USAGE.md` if using the CLI.
8. **Report**: fill `reports/FINAL-REPORT.md §9 "โมเดลสุดท้าย"` (recipe, strat/doc/TTA/3-seed numbers, ensemble/KD, weight
   size + load snippet) and add the F table + confound re-derivation to `reports/02-EXPERIMENTS.md`. Slides: `reports/figures/results/`,
   `reports/eda/`, `reports/analysis/<winner>/`, `reports/robustness/<winner>_full_bin/`. Report in Thai to the owner after each step.

## 4. Shipped checkpoints & known gaps
- Tarball checkpoints (`runs/<id>/best.pt`): B6a_synth_pretrain_resnet18_64 + C1_ext_pretrain_resnet18_64 (init for F11–F17),
  F1/F2/F3 (ensemble/KD members), F4_effb0 (weak teacher), A1_resnet18_full_64_T4 (notebook default until the winner exists),
  B_trivial_resnet18_64_T4 + D3_sampler_sqrt_resnet18_64_T4 (6-ep references for the doc/imbalance stories). Every other run
  ships only metrics.json / log.csv / val_logits.npy — retrain if a checkpoint is needed.
- `runs/B_full_resnet18_64_T4/` and `runs/F5_*/` are empty aborted-run dirs (safe to reuse; `B_full_resnet18_64_T4` ≡ `D0_ce_ls_resnet18_64_T4`).
- `data/external/{alice,burapha,downloads}` raw files were dropped from the tarball (only `glyphs_external.npz` + `index.csv`
  are read); re-fetch with `scripts/fetch_external.py --sources alice burapha` if needed.
- `minority_acc` is NaN for B6_synth_* (pre-fix); engine now uses real counts. `val_subset_frac` only affects per-epoch
  monitoring; final metrics are on the full val split.
- Tarball recipe (laptop): file list = everything under Deep_CNN except `.venv/`, `ThaiCharacter Dataset/__MACOSX/`, `dist/`,
  `outputs/`, `__pycache__/`, `.pytest_cache/`, `*.executed.ipynb`, `last.pt`, `data/external/{alice,burapha,downloads}`,
  `.DS_Store`, and all `runs/*/best.pt` except the ids above; `tar -c -I pigz -f ~/Deep_CNN_handoff_<date>.tar.gz -T list`.

## 5. Deliverables checklist (owner's §2)
- [x] `notebooks/ThaiChar72_Colab.ipynb` (Train + Inference, config in one cell) — rebuilt for the final recipe, headless inference run passes
      (99.03 %); **[ ] actual Colab run still to be done by the owner**
- [x] final weights in `weights/` with size + load instructions (`*.card.json`, `weights/README.md`, FINAL-REPORT §9)
- [x] `reports/FINAL-REPORT.md` complete (§9 final model, §10 error analysis, §11 application, §12 label audit) + figures
- [x] `reports/02-EXPERIMENTS.md` covers every run incl. 3 seeds (mean ± std), §F/F-doc/H/I/J
- [x] everything pushed to GitHub (`main` == `origin/main`)
- [x] `uv` project (pyproject + lock; torch wheel auto-selected by platform markers)

# TASK-01 — Dataset integrity check + EDA script (Thai character 72-class corpus)

## Context
Project root: `/home/eggchad/eakject/research/Deep_Man/Deep_CNN` (this directory). Python project
managed by `uv` (Python 3.12, `pyproject.toml` already has numpy, pillow, pandas, matplotlib, tqdm,
scikit-learn, imagehash). Run everything with `uv run python ...`. Do NOT `pip install`; if you need a
new dependency use `uv add <pkg>` and say so in your report.

Dataset: `ThaiCharacter Dataset/round2/<code>/*.jpg` — 72 sub-directories named `161 … 249`.
The directory names are **TIS-620 byte codes**: Thai char = `chr(0x0E00 + code - 0xA0)`
(161 → ก, 210 → า, 240 → ๐ … 249 → ๙). Files are small grayscale JPEGs (median ≈16×19 px), roughly
62,707 of them. Ignore `__MACOSX/`, `.DS_Store`, `Thumbs.db` and anything that is not `*.jpg`.
Filenames look like `bc_001sg_3_118.jpg` = `<prefix>_<docid><subset>_<line>_<idx>.jpg`
(prefix ∈ {be, bc, bl, ce}; docid 3 digits; subset ∈ {sg, tg}). A few files are named
`Copy of <name>.jpg` — parse the inner name and set a flag `is_copy=True`.

## Scope — you may create/modify ONLY these paths
- `src/thaichar/__init__.py`, `src/thaichar/classes.py`, `src/thaichar/filenames.py`
- `scripts/eda.py`
- `reports/eda/**` (outputs; create the directory)
- `pyproject.toml` only if you must add a dependency (report it)

HARD RULES: `ThaiCharacter Dataset/` is READ-ONLY — never write, move, rename or delete anything in
it. Do not touch `../CNN`, `../katgpt-rs`, `../fly-connectome-template`. Do not touch `tasks/`,
`weights/`, `notebooks/`. Do not run `git commit`.

## Deliverable 1 — `src/thaichar/classes.py`
- `CLASS_CODES: list[int]` — the 72 TIS-620 codes, sorted ascending, discovered from the dataset
  directory names is NOT allowed here; hard-code the list (it must be importable without the data).
  The 72 codes are: 161 162 163 164 167 168 169 170 171 173 175 176 177 178 179 180 181 182 183 184
  185 186 187 188 189 190 191 192 193 194 195 196 197 199 200 201 202 203 204 205 206 207 209 210 212
  213 214 215 216 217 224 225 226 227 228 229 230 231 232 233 234 236 240 241 242 243 244 245 246 247
  248 249.
- `code_to_char(code:int)->str`, `char_to_code(ch:str)->int`, `code_to_index(code)->int` (0..71,
  position in the sorted list), `index_to_code(i)`, `CLASS_CHARS: list[str]`.
- `category(code)->str` returning one of `consonant | vowel | tone_mark | digit`
  (consonants U+0E01–U+0E2E plus ฯ U+0E2F; vowels U+0E30–U+0E39 and U+0E40–U+0E45;
  tone/diacritic marks U+0E46–U+0E4E (ๆ ็ ่ ้ ๊ ์ …); digits U+0E50–U+0E59).
- `unicode_name(code)->str` via `unicodedata.name`.
- Add `src/thaichar/__init__.py` (can be empty) and make sure `uv run python -c "import thaichar"` works
  (add `[tool.uv] package = true` / `[build-system]` hatchling + `packages = ["src/thaichar"]` to
  `pyproject.toml` if needed, or set `[tool.hatch.build.targets.wheel] packages = ["src/thaichar"]`).

## Deliverable 2 — `src/thaichar/filenames.py`
`parse_filename(name:str) -> dict` with keys `prefix, doc_id (int), subset, line (int), idx (int),
is_copy (bool), group (str = f"{prefix}_{doc_id:03d}")`. Return `None` fields (not an exception)
for names that do not match, but set `parsed=False`.

## Deliverable 3 — `scripts/eda.py`  (CLI: `uv run python scripts/eda.py --data "ThaiCharacter Dataset/round2" --out reports/eda`)
Use `multiprocessing.Pool` (8 workers) + tqdm. For every `*.jpg` collect one row:
`path, code, char, category, filename fields, width, height, mode, bytes, md5, ok (decodes without
error via PIL open+load), ink_frac (fraction of pixels < 128 after converting to L), mid_grey_frac
(fraction of pixels with 32 <= v < 224), border_mean (mean intensity of the 1-px outer border),
phash (imagehash.phash 8x8, as hex string)`.
Write:
1. `reports/eda/files.csv` — all rows.
2. `reports/eda/class_stats.csv` — per class (72 rows, sorted by code): `code, char, unicode_name,
   category, n_files, n_ok, n_unique_md5, n_groups (distinct prefix_doc), med_w, med_h, med_aspect
   (w/h), min_w, max_w, min_h, max_h, mean_ink_frac, share_pct, n_train_80 (= round(0.8*n)),
   n_val_20`.
3. `reports/eda/duplicates.json` — `{ "md5_groups": [{md5, n, paths, codes, cross_class: bool}],
   "n_redundant_files": int (sum of (n-1) over groups), "n_cross_class_groups": int,
   "n_unique_after_dedup": int, "copy_of_files": [paths] }`.
4. `reports/eda/summary.json` — total files, total ok, corrupted list, non-jpg files skipped,
   n classes, imbalance ratio max/min, Gini of class sizes, top-10 share %, classes with n<50,
   n<10, n<5, median w/h overall, grey-level check (`P(32<=v<224)` overall), group counts
   (per prefix, distinct doc ids, distinct groups), duplicate summary, per-category totals.
5. Figures (PNG, 150 dpi, use the Thai font `assets/fonts/Sarabun-Regular.ttf` registered with
   matplotlib `font_manager.fontManager.addfont` so Thai labels render — verify visually that
   labels are not boxes):
   - `class_distribution.png` — bar chart of n per class sorted descending, log y-axis, x tick
     labels = Thai chars, bars coloured by category, horizontal lines at n=50 and n=10.
   - `class_montage.png` — grid 72 rows × 6 random samples (seed 0) at native resolution shown with
     `interpolation="nearest"`, each row labelled `char (code) n=…`. Use a smaller figure if 72 rows is
     unwieldy: 12 columns × 6 rows of "class panels" each containing 4 samples is fine.
   - `size_scatter.png` — width vs height scatter (alpha 0.05) with medians marked.
   - `grey_histogram.png` — histogram of pixel intensities over a 2,000-image random sample, log y.
   - `per_group_counts.png` — number of files per `prefix_doc` group.
6. `reports/eda/EDA.md` — markdown report: headline table, the full 72-row class table (code, char,
   category, n, share, med WxH, n_groups, n_train_80/n_val_20), imbalance section, duplicate/hygiene
   section (with 5 example cross-class collisions), group-structure section, and links to the
   figures. Every number must come from the computed data, not typed by hand.

## Acceptance criteria (the reviewer will run these)
```bash
uv run python -c "import thaichar; from thaichar.classes import CLASS_CODES, code_to_char, category; assert len(CLASS_CODES)==72; assert code_to_char(161)=='ก' and code_to_char(210)=='า' and code_to_char(249)=='๙'; assert category(240)=='digit' and category(232)=='tone_mark' and category(210)=='vowel' and category(161)=='consonant'; print('classes ok')"
uv run python -c "from thaichar.filenames import parse_filename as p; d=p('bc_001sg_3_118.jpg'); assert d['prefix']=='bc' and d['doc_id']==1 and d['subset']=='sg' and d['line']==3 and d['idx']==118 and d['group']=='bc_001' and not d['is_copy']; d=p('Copy of bc_008tg_11_118.jpg'); assert d['is_copy'] and d['doc_id']==8; print('filenames ok')"
uv run python scripts/eda.py --data "ThaiCharacter Dataset/round2" --out reports/eda
uv run python -c "import pandas as pd; s=pd.read_csv('reports/eda/class_stats.csv'); assert len(s)==72 and s.n_files.sum()==62707, (len(s), s.n_files.sum()); print('class_stats ok', s.n_files.min(), s.n_files.max())"
ls reports/eda/{files.csv,duplicates.json,summary.json,EDA.md,class_distribution.png,class_montage.png,size_scatter.png,grey_histogram.png,per_group_counts.png}
```
The whole EDA run must finish in under 10 minutes on 8 CPU cores.

## Report back
List files created, the exact commands you ran, the printed acceptance output, the headline numbers
(total files, corrupted, duplicates, cross-class collisions, imbalance ratio) and anything surprising.

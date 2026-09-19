# รายงานผลการทดลอง: Thai Character/Digit Recognition 72 คลาส ด้วย CNN + Transfer Learning + Data Augmentation

> ฉบับร่าง (อัปเดตอัตโนมัติจากผลการทดลอง) — ตัวเลขทั้งหมดมาจาก `reports/experiments.csv` และ `runs/*/metrics.json`
> รูปประกอบสำหรับสไลด์: `reports/figures/results/` (PNG + SVG), `reports/eda/`, `reports/analysis/`, `reports/robustness/`

## 1. ชุดข้อมูล

- ภาพ `.jpg` 62,707 ภาพ ใน 72 โฟลเดอร์ (ชื่อโฟลเดอร์ = รหัส TIS-620: 161→ก … 249→๙); ตรวจนับครบ ไม่มีไฟล์เสีย
- 4 หมวด: พยัญชนะ 42 คลาส (45,177 ภาพ), สระ 14 (14,747), วรรณยุกต์/เครื่องหมาย 6 (2,501), **เลขไทย 10 (282 ภาพ, 0.45%)**
- ภาพเล็กมาก: median **16×19 px**, p99 29×44 px; เป็น **ไบนารีแท้** (ไม่มีพิกเซลเทา) ครอปชิดหมึกไม่มี margin
- กราฟการกระจายคลาส: `reports/eda/class_distribution.png` · ตัวอย่างทุกคลาส: `reports/eda/class_montage.png`
- ตารางจำนวนภาพต่อคลาส 72 แถว: `reports/eda/EDA.md`

## 2. ความท้าทายของชุดข้อมูล

| ความท้าทาย | หลักฐาน | ผลต่อการออกแบบ |
|---|---|---|
| **Class imbalance รุนแรง** 5,025 : 1 (า vs ฃ/ฑ ที่มี 1 ภาพ) Gini 0.67, 23 คลาส < 50 ภาพ, เลขไทยทั้งหมดเป็น minority | `reports/eda/summary.json` | ต้องรายงาน balanced acc / macro-F1 ควบ top-1; ใช้ synthetic fonts, weighted loss/sampler, logit adjustment |
| **ตัวอักษรคล้ายกัน** — า/ๅ ต่างแค่ความสูง (20 vs 27 px), ว→า เมื่อห่วงหาย, ั/้, ด/ต, ช/ซ, ี/ื | confusion matrix `reports/analysis/*/confusion_matrix.png` | คงอัตราส่วนภาพ (ห้าม stretch), ห้าม flip; error analysis เฉพาะคู่ |
| **ภาพเล็กและไบนารี** ไม่ตรงโดเมน ImageNet | EDA §3 | ต้องทดลอง input size และระดับ fine-tune จริง; ต้อง binarize (Otsu) ที่ inference |
| **ข้อมูลซ้ำ/ป้ายชน** 2,579 ไฟล์ซ้ำ + 11 กลุ่มที่ภาพเดียวกันอยู่ 2 คลาส (รวม า/ๅ) + 6 ไฟล์ `Copy of` ผิดคลาส | `reports/eda/duplicates.json` | ลบก่อน split → 60,117 ภาพ (กัน leakage) |
| **โครงสร้างเอกสารแฝง** 52 กลุ่ม (ฟอนต์/DPI เดียวกัน) | ชื่อไฟล์ `{be,bc,bl,ce}_<doc>...` | รายงาน 2 split: stratified 80:20 (ตามโจทย์) + document-disjoint (ตรวจ generalisation) |

## 3. โครงสร้าง CNN ที่ใช้

- **Backbone**: ResNet-18 (timm, ImageNet-1k pretrained, 11.2M params) เป็นตัวหลัก; เทียบ EfficientNet-B0, MobileNetV3-L,
  SmallCNN scratch (0.43M) และ ViT-tiny (รองรับแต่ไม่ได้เข้ารอบสุดท้ายเพราะที่ 64 px CNN คุ้มกว่า)
- **Input** 1×64×64 ไบนารี → pad-to-square คงอัตราส่วน + margin 10% → resize 64 → ก๊อป 3 ช่อง + ImageNet normalisation
  (ทดลอง ON/OFF encoding แบบ 3 ช่อง [ink, distance transform, edges] ด้วย — ไม่ช่วย)
- **Head**: global pooled feature (512) → Linear 72 (ตัวเลือก geometry side-channel 4 มิติ → MLP 16 → concat; ถูกตัดออกในตัวจริงเพราะไม่ generalize ข้ามเอกสาร)
- จุดเด่นที่เลือก ResNet-18: เร็วที่สุดต่อ epoch (28 วิ/epoch บน T4 สำหรับ 48k ภาพ), latency 12 ms/ภาพ บน CPU, ผลใกล้ EfficientNet-B0

## 4. Transfer Learning

| การตั้งค่า | ผล (64 px, 6 ep, stratified) | สรุป |
|---|---|---|
| ImageNet frozen backbone (linear probe) | resnet18 85.7 / 56.3 | ฟิลเตอร์ ImageNet ดิบ ๆ ใช้กับกลิฟไบนารีไม่ได้ |
| ImageNet partial (35% ท้าย) | 96.2 / 90.7 | ดีขึ้นแต่ยังแพ้ scratch ด้าน top-1 |
| **ImageNet full fine-tune** | **97.5 / 97.9** | ชนะ scratch (97.4 / 91.9) +6 จุด balanced — pretrained weights ช่วยคลาสหาง |
| Layer-wise LR decay 0.8 | 97.6 / 95.6 | แย่ลง: ชั้นต้นต้องปรับตัวมาก |
| **ImageNet → synthetic Thai fonts → real** (2 ชั้น) | **97.9 / 98.4** | ดีสุดที่งบเท่ากัน; โมเดล stage-1 เพียว ๆ zero-shot ได้ 93.7% บนข้อมูลจริง |
| ImageNet → ลายมือ public (ALICE-THI + Burapha-TH, 101k ภาพ) → real | 97.8 / 98.1 | ต่างโดเมน (ลายมือ) แต่ยังช่วย +1.7 balanced |

เหตุผล: กลิฟไบนารีเล็กต่างจากภาพธรรมชาติมาก จึงต้องปล่อยให้ทุกชั้นปรับตัว (full fine-tune) แต่ค่าเริ่มต้นจาก ImageNet
ยังให้ inductive bias ที่ดีกว่าสุ่ม โดยเฉพาะกับคลาสที่มีตัวอย่างน้อย; การแทรก "โดเมนกลาง" ที่เป็นตัวอักษรไทยจริง ๆ
(สังเคราะห์จากฟอนต์ 26 แบบ) ทำให้ feature ตรงงานยิ่งขึ้น

## 5. Data Augmentation

ทำหลัง split เท่านั้น บน canvas ก่อน resize; **ไม่มี flip ทุกแกน** (อักษรไทยเปลี่ยนความหมาย) รูปตัวอย่างก่อน/หลัง: `reports/figures/aug_examples.png`

| preset | ประกอบด้วย | strat top-1/bal | **doc top-1/bal** |
|---|---|---:|---:|
| none | – | 98.48 / 98.15 | 97.80 / 97.61 |
| base | affine (±8°, shear 10°, scale .85–1.15) + margin jitter | 97.53 / 97.94 | 96.34 / **93.61** |
| morph | base + dilate/erode + resolution jitter | 97.85 / 96.94 | 97.29 / 97.86 |
| full | morph + elastic + speckle + blur + erasing + rebinarize | 97.80 / 96.44 | 97.26 / 96.11 |
| **randaug** (N=2) | ops ข้างบนแบบสุ่ม | 98.21 / 98.25 | 97.80 / 97.00 |
| **trivial** (1 op สุ่ม magnitude) | | 98.05 / 98.27 | **97.90 / 97.84** |
| full + Mixup / CutMix | | 97.6 / 94.9 · 97.3 / 94.0 | – |
| **synthetic fonts** 21,600 ภาพ (26 ฟอนต์ × degradation) | ใช้เป็น pretraining | 97.92 / 98.43 | (รอบสุดท้าย) |

บทเรียน: (1) ที่งบ 6 epochs aug หนักลด accuracy บน in-distribution; (2) ข้ามเอกสาร preset เบาทุกตัว (none/base/trivial/morph) เสียเพียง
0.3–0.5 จุด — ค่า `base` doc 93.61 ในตารางเป็น run ที่ใช้ข้อมูล 25 % (แก้แล้ว: ข้อมูลเต็ม 97.28 / 97.40, 02-EXPERIMENTS §G) — สิ่งที่พังข้ามเอกสารจริง
คือ geometry side-channel; (3) Mixup/CutMix ไม่เหมาะกับกลิฟไบนารี; (4) synthetic fonts ช่วยมากที่สุดเมื่อใช้เป็น pretraining (+1.5 balanced
ข้ามเอกสารเทียบ init ImageNet, recipe เดียวกัน) และเป็นทางเดียวที่ทำให้ ฃ/ฑ (1 ภาพ) เรียนได้; (5) **ที่ 20 epochs ตัวตัดสินไม่ใช่ accuracy แต่เป็น
robustness**: recipe ที่มี noise op ตอนเทรน (randaug/full) ทน salt-pepper/background noise ได้ (0.95) ส่วน morph/trivial พัง (0.18–0.86)
ทั้งที่ accuracy บน val เท่ากัน (02-EXPERIMENTS §F-doc)

## 6. เทคนิค/แนวคิดที่น่าสนใจของกลุ่ม

1. **Synthetic-font pretraining เป็นโดเมนกลาง** (ImageNet → ฟอนต์ไทย → สแกนจริง): render 72 กลิฟ × 26 ฟอนต์ Google Fonts
   พร้อมแยกวรรณยุกต์ออกจากพยัญชนะฐานด้วยการลบภาพ, จำลอง DPI/ความหนาเส้น/threshold/speckle ให้ตรงสถิติของข้อมูลจริง
   → zero-shot 93.7% และ fine-tune แล้วดีที่สุด
2. **Document-disjoint validation เป็นเกณฑ์เลือกโมเดล** — เปิดโปงว่าการเลือกด้วย stratified val อย่างเดียวจะพาไปผิดทาง
   (base aug, geometry side-channel ดูดีบน strat แต่พังบนเอกสารใหม่)
3. **Otsu binarisation ที่ inference** ทำให้โมเดลที่เทรนบนภาพไบนารีทนต่อภาพเทา/contrast ต่ำ (จาก 1.5% → 97.1% บน contrast test)
4. **Post-hoc logit adjustment (τ)** ปรับ prior ได้ฟรีตาม metric ที่ผู้ประเมินใช้ (+0.5–2 จุด balanced)
5. ทดลองแนวคิดจาก fly-connectome (ON/OFF channel split) และงานเก่า (geometry side-channel) อย่างซื่อตรง — ทั้งสองไม่ช่วย
   หรือช่วยเฉพาะ in-distribution → รายงานเป็นผลลบ
6. Robustness degradation curves (สไตล์ FLYNN) ใช้เป็น **เกณฑ์คัดโมเดล** (ไม่ใช่แค่รายงาน) → เปลี่ยนผู้ชนะจาก recipe ที่ accuracy สูงสุด
   ไปเป็น recipe ที่ทน noise; KD กลั่น 3 seeds ลงโมเดลเดียวได้ผลเท่า ensemble (ดูส่วนที่ 9)
7. **Label audit ด้วยโมเดล + ตา** (ส่วนที่ 12): พบว่า ~11 % ของคลาส า เป็น ว/ใ ที่ label ผิด (กระจุกในแหล่ง `be`) ซึ่งเป็นเพดานที่ทุก recipe ชน
   (top-1 ≈ 98.4–98.6) การแก้ label ให้ผลมากกว่าการเปลี่ยน architecture/aug ใด ๆ (+0.8 จุด → 98.9–99.0) และเราส่งมอบ weight ทั้ง 2 แบบ

## 7. ขั้นตอนการฝึกสอน

AdamW (lr 1e-3, wd 0.05) · cosine schedule + warmup 1 epoch · batch 128 · label smoothing 0.1 · EMA (decay 0.999, warmup)
· grad-clip 1.0 · AMP บน GPU · เลือก checkpoint ด้วย balanced acc บน val (บันทึก caveat ว่าเลือกบน val ที่รายงาน)
· ranking runs 6 epochs, ตัวจริง 20 epochs + TTA (8 มุมมอง: shift ±1px, thicker/thinner, margin 0.05/0.15)
· รอบ 6-epoch บน Google Colab T4 ผ่าน `colab` CLI (สคริปต์ทนต่อ session หลุดทุก ~1 ชม.: `scripts/colab_autorun.sh`); รอบ 20-epoch
  (F5–F19, doc, seeds, KD) บน RTX 4060 / WSL2 (`tasks/run_step1*.sh`, 8–15 วิ/epoch, รัน 2–4 config พร้อมกันเพราะ pipeline เป็น CPU-bound)
· seed 42 สำหรับคัดเลือก; โมเดลสุดท้ายรายงาน 3 seeds (42/0/1) บน split เดียวกัน แล้วกลั่น (KD) เป็นโมเดลเดียว

## 8. กราฟ accuracy train/val
`reports/figures/results/fig_training_curves_best.png`, `reports/figures/training_curves_*.png` (log ต่อ epoch ใน `runs/*/log.csv`)

## 9. ตารางเทียบผลทั้งหมด
ตารางอัตโนมัติ: `reports/experiments.md` · รายละเอียดรายหมวด A–G: `reports/02-EXPERIMENTS.md` · รูป: `reports/figures/results/`

### โมเดลสุดท้าย — `weights/thaichar72_resnet18_64.pt` (เลือกเสร็จ 2026-09-19 บน RTX 4060)

**Recipe (F19 → กลั่นเป็น K1)**: ResNet-18 @ 64 px, transfer 2 ชั้น **ImageNet → synthetic Thai fonts (stage-1, 8 ep) → ข้อมูลจริง**,
RandAugment N=2 (ไม่ flip; ops: affine/margin/stroke-width/res-jitter/elastic/speckle/blur/erase/rebinarize), CE + label smoothing 0.1,
AdamW 1e-3 cosine + warmup 1 ep, batch 128, EMA, 20 epochs, เทรนบน **label ที่แก้แล้ว (v2, ดู §12)**; โมเดลส่งมอบคือ **student ที่กลั่น (KD, α 0.7, T 4)
จาก 3 seeds ของ recipe นี้** จึงได้ผลระดับ ensemble ด้วยต้นทุน inference 1 โมเดล; **TTA ปิด** (ลด top-1ทุก recipe −0.1…−0.7 จุด)

วิธีเลือก (กติกาที่ตั้งไว้ก่อนรัน, `tasks/HANDOFF.md` §3.3): ผู้เข้ารอบ 19 recipe × 20 epochs บน stratified val → doc-disjoint val ของ 11 ตัวบน →
เกณฑ์ robustness (mean top-1 บน 33 corruption ต้องไม่ต่ำกว่า baseline F2 − 0.5 จุด) → เรียงตาม doc top-1. **มีเพียง randaug/full ที่ผ่าน gate**
(recipe ที่ไม่เคยเห็น noise ตอนเทรน เช่น morph/trivial พังบน salt-pepper) → F19 ชนะ (doc top-1 0.9822 สูงสุด, robust 0.9045)

| การวัด (val เต็ม, raw ไม่ TTA) | top-1 | balanced | macro-F1 | minority |
|---|---:|---:|---:|---:|
| F19 stratified val (label v1 เดิม) | 0.9839 | 0.9827 | 0.9791 | 0.9667 |
| F19 document-disjoint val (label v1) | 0.9822 | 0.9807 | 0.9746 | 0.9658 |
| F19 stratified val **label v2** (seed 42) | 0.9879 | 0.9868 | 0.9848 | 0.9750 |
| F19 **3 seeds** บน v2 (42/0/1) mean ± std | **0.9890 ± 0.0008** | **0.9868 ± 0.0014** | 0.9853 ± 0.0004 | 0.9750 ± 0.0068 |
| F19 doc-disjoint val, label v2 | 0.9859 | 0.9790 | 0.9733 | 0.9748 |
| Ensemble soft-vote 3 seeds (v2) | 0.9899 | 0.9876 | 0.9867 | 0.9750 |
| **K1 = KD student ตัวเดียว (ส่งมอบ)**, v2 val | **0.9903** | **0.9875** | **0.9869** | 0.9750 |
| K1 วัดบน val label v1 เดิม (อ้างอิงความเสี่ยง §12) | 0.9776 | 0.9833 | 0.9760 | 0.9667 |

Robustness (mean top-1 ทุก corruption, Otsu on): K1 **0.9104** > F19_v2 0.9052 > F19 0.9045 > F2 0.9038 (`reports/robustness/K1_r18_kd_20_full_bin/`)

**ไฟล์ weight** (`weights/`, มี `.card.json` คู่ทุกไฟล์: metrics, md5, cfg, วิธีโหลด):

| ไฟล์ | ใช้เมื่อ | ขนาด | md5 (8) |
|---|---|---:|---|
| `thaichar72_resnet18_64.pt` | **ค่าเริ่มต้น** (fp32, K1, label v2) | 44.9 MB | 467bb628 |
| `thaichar72_resnet18_64_fp16.pt` | เหมือนกันแบบ fp16 (ผลเท่ากันทุกหลัก 0.9903/0.9875) | 22.5 MB | fce6c261 |
| `thaichar72_resnet18_64_v1labels.pt` | สำรอง: F19 เทรนด้วย label เดิม — ใช้ถ้าคาดว่า test set ใช้ label convention เดิมของแหล่ง `be` (§12) | 44.9 MB | 9605c6f6 |
| `thaichar72_r18_synth_pretrain_init.pt` | stage-1 init (ImageNet→ฟอนต์สังเคราะห์) สำหรับเทรนซ้ำใน notebook | 44.9 MB | 3d667ccd |
| `thaichar72_mnv3small_64_small.pt` | **small model**: MobileNetV3-small (1.6 M params) กลั่นจาก 3 ครู F19 — 0.9900 / 0.9872, robust 0.9053, 3.4 ms/ภาพ | 6.5 MB | ดู card |

Backbone อื่นด้วย recipe เดียวกัน (02-EXPERIMENTS §J): effb0 0.9901 / 0.9886, convnext_tiny 0.9898 / 0.9869, mnv3-large 0.9893 / 0.9870 — ทุกตัว 98.7–99.0
และผ่าน robustness gate → **recipe สำคัญกว่า backbone**; KD ข้ามสถาปัตยกรรมไม่ช่วย (ยกเว้น student เล็กมาก)

```python
import sys; sys.path.insert(0, "src")
from thaichar.infer import load_checkpoint, predict_topk
model, cfg = load_checkpoint("weights/thaichar72_resnet18_64.pt", device="cpu")   # ใน notebook: WEIGHTS_PATH
print(predict_topk(model, cfg, "some_glyph.jpg", k=5))   # Otsu binarise → pad-to-square (margin 0.1) → 64 px → top-k (char, prob)
```
CLI: `uv run python scripts/predict.py --ckpt weights/thaichar72_resnet18_64.pt <image ...>` · latency CPU bs=1 = 6.8 ms บนเครื่อง 4060 (10–12 ms บน Colab CPU) · เทรนซ้ำ: `configs/final.yaml`

## 10. Confusion matrix + error analysis
โมเดลสุดท้าย: `reports/analysis/K1_r18_kd_20/{confusion_matrix.png,confused_pairs.md,worst_classes.md,per_class_recall.png}` (v2 val 11,991 ภาพ, ผิด 116 ภาพ)

| อันดับ | คู่ที่สับสน (K1, v2 val) | จำนวน | หมายเหตุ |
|---:|---|---:|---|
| 1 | า → ๅ / ๅ → า | 30 + 16 | รูปเดียวกันต่างแค่ความสูงสัมบูรณ์ ซึ่งหายไปเมื่อ pad-to-square — เป็น ceiling ที่เหลือ (40 % ของ error ทั้งหมด) |
| 2 | ั ↔ ้ | 13 + 4 | วรรณยุกต์เล็ก ต่างที่หางเส้นเดียว |
| 3 | ช → ซ | 4 | หัวหยัก |
| 4 | ๘ → ็ | 2 (จาก 5) | คลาสที่ val เล็กสุด (recall 60 %) |

เทียบกับ baseline A1 (§2): คู่ **ว → า (59 ภาพ, 17.6 % ของ ว) หายไปทั้งหมด** หลังแก้ label (§12) — ยืนยันว่า error นั้นเป็น label noise ไม่ใช่ห่วงหาย;
า↔ๅ ลดจาก 95 → 46 หลังเพิ่ม epoch/aug/KD แต่ยังเป็น error หลัก การแก้ต่อไปต้องใช้ข้อมูลความสูงจริง ซึ่งเราตัดออกเพราะไม่ generalise ข้ามเอกสาร (§6 ข้อ 5)

## 11. Application test
- `scripts/predict.py` (top-k + confidence), `scripts/robustness.py` (เอียง/หนา-บาง/noise/blur/contrast/occlusion/downscale curves,
  `reports/robustness/K1_r18_kd_20_full_bin/curves.png` — ทน rotate ≤ 15°, blur, contrast, background noise, translate ≤ 10 % ที่ ≥ 0.95;
  จุดอ่อน: เส้นบางลง 2–3 px (erosion) และ occlusion 40 %), latency 6.8 ms/ภาพ CPU bs=1 บนเครื่องนี้ (10–12 ms บน Colab CPU; resnet18@64), weight 44.9 MB (fp16 22.5 MB)
- Notebook Colab `notebooks/ThaiChar72_Colab.ipynb` (สร้างจาก `scripts/build_notebook.py`): Train (`configs/final.yaml`, สร้าง split v2 เอง
  จาก change list, ใช้ stage-1 init ที่แพ็กมา) / Inference / Gradio-หรือ-ipywidgets upload / robustness quick check — ทดสอบรันจบบน CPU
  ในโหมด inference ด้วย weight สุดท้าย (`scripts/run_notebook_local.py`; log `tasks/logs/notebook_local_inference.log`); ยังไม่ได้ทดสอบบน Colab
  จากเครื่อง 4060 (ไม่มี `colab` CLI) — เจ้าของงานทดสอบตาม `tasks/COLAB-USAGE.md`

## 12. Label audit (DataV2) และความเสี่ยงต่อ hidden test
ทีมส่งแพ็กเกจ relabel (DataV2) ที่ย้ายภาพ 629 ภาพ (า→ว 416, า→ใ 107, า→จ 21, ต↔ด 29, …) และทิ้ง outlier 175 ภาพ เราตรวจ 2 ทาง:
(ก) montage เทียบกับตัวอ้างอิง (`reports/analysis/datav2_montage_a_w.png`) → ภาพที่ย้ายเป็น ว/ใ จริงทุกภาพ; (ข) โมเดล v1 เห็นด้วยกับ label ใหม่
95–100 % ในกลุ่มเล็ก แต่ทาย า 85–93 % บนกลุ่ม า→ว **เพราะเรียน label ผิดมา 416 ภาพ** → รับเฉพาะการย้าย/ทิ้ง (ไม่รับภาพ augmented 1,822 ภาพ
เพราะ 570 ภาพรั่วข้าม split ของ v2 และเรามี aug/synthetic อยู่แล้ว) สร้าง `data/splits/split_seed42_v2.csv` ที่คง split เดิมทุกประการ
(`scripts/make_split_v2.py`, change list ใน `reports/analysis/datav2_label_changes.csv`)

ผล (02-EXPERIMENTS §H): recipe เดิม วัดด้วย label ถูก 98.79 → 98.90 ± 0.08 (3 seeds) → 99.03 (KD); ผลต่าง v1↔v2 สมมาตร ±0.8–0.9 จุด = สัดส่วนภาพที่ label เปลี่ยน
**ความเสี่ยง**: mislabel กระจุกในแหล่ง `be` (694/804, 19 เอกสาร ละ 11–20 % ของ า) ถ้า hidden test มาจากแหล่ง/กระบวนการเดียวกัน โมเดล v2 จะเสีย ~0.9 จุด
(และกลุ่มอื่นที่เทรน label เดิมจะไม่เสีย); ถ้า test label ถูกหรือมาจากแหล่งใหม่ ("ข้อมูลแปลก" ที่อาจารย์บอก) v2 ได้เปรียบ ~0.8 จุด → ส่งมอบทั้ง
`thaichar72_resnet18_64.pt` (v2) และ `thaichar72_resnet18_64_v1labels.pt` (v1) และควรถามอาจารย์เรื่อง convention ของ ว/า ใน test set ถ้าทำได้

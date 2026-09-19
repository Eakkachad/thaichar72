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

บทเรียน: (1) ที่งบ 6 epochs aug หนักลด accuracy บน in-distribution แต่ (2) สิ่งที่ทำร้าย generalisation ข้ามเอกสารคือ
affine/margin jitter (`base`) ไม่ใช่ aug โดยรวม — morph/trivial/none เสียแค่ 0.4–0.5 จุดข้ามเอกสาร; (3) Mixup/CutMix ไม่เหมาะ
กับกลิฟไบนารี; (4) synthetic fonts ช่วยมากที่สุดเมื่อใช้เป็น pretraining และเป็นทางเดียวที่ทำให้ ฃ/ฑ (1 ภาพ) เรียนได้

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
6. Robustness degradation curves (สไตล์ FLYNN) + TTA + ensemble/KD (ดูส่วนที่ 9)

## 7. ขั้นตอนการฝึกสอน

AdamW (lr 1e-3, wd 0.05) · cosine schedule + warmup 1 epoch · batch 128 · label smoothing 0.1 · EMA (decay 0.999, warmup)
· grad-clip 1.0 · AMP บน GPU · เลือก checkpoint ด้วย balanced acc บน val (บันทึก caveat ว่าเลือกบน val ที่รายงาน)
· ranking runs 6 epochs, ตัวจริง 20 epochs + TTA (8 มุมมอง: shift ±1px, thicker/thinner, margin 0.05/0.15)
· ทำบน Google Colab T4 ผ่าน `colab` CLI (สคริปต์ทนต่อ session หลุดทุก ~1 ชม.: `scripts/colab_autorun.sh`)
· seed 42 (ผลตัวจริงรายงาน 3 seeds เมื่อรันครบ)

## 8. กราฟ accuracy train/val
`reports/figures/results/fig_training_curves_best.png`, `reports/figures/training_curves_*.png` (log ต่อ epoch ใน `runs/*/log.csv`)

## 9. ตารางเทียบผลทั้งหมด
ตารางอัตโนมัติ: `reports/experiments.md` · รายละเอียดรายหมวด A–G: `reports/02-EXPERIMENTS.md` · รูป: `reports/figures/results/`

### โมเดลสุดท้าย (จะเติมเมื่อรอบ F1–F17 เสร็จ)
- recipe, ผล strat/doc, TTA, 3 seeds, ensemble, ขนาดไฟล์ weight และวิธีโหลด

## 10. Confusion matrix + error analysis
`reports/analysis/<exp>/confusion_matrix.png`, `confused_pairs.md`, `worst_classes.md` — คู่หลัก า↔ๅ (label noise), ว→า (ห่วงหาย),
ั↔้, ด↔ต, ช↔ซ, ี↔ื; คลาสอ่อนสุดคือเลขไทยที่มี val น้อยมาก (๘ n_val=5)

## 11. Application test
- `scripts/predict.py` (top-k + confidence), `scripts/robustness.py` (เอียง/หนา-บาง/noise/blur/contrast/occlusion/downscale curves,
  `reports/robustness/*/curves.png`), latency 10–12 ms/ภาพ CPU (resnet18@64), weight 43 MB
- Notebook Colab `notebooks/ThaiChar72_Colab.ipynb`: Train / Inference / Gradio-หรือ-ipywidgets upload / robustness quick check
  — รันจบบน CPU ในโหมด inference (207 วิ, 0 error)

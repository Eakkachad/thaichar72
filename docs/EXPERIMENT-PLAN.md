# Experiment Plan + Delegation Plan — Thai 72-class CNN

สถานะ: ร่างเพื่อขอความเห็น 1 รอบ (2026-09-19) · ผู้วางแผน/รีวิว: Claude · ผู้เขียนโค้ด: `agy`

## 0. ข้อจำกัดที่กำหนดรูปแบบแผน
| ข้อเท็จจริง | ผลต่อแผน |
|---|---|
| เครื่องนี้ **ไม่มี GPU** (Intel UHD, 8 คอร์, 15 GB) | แบ่งการทดลองเป็น 2 ชั้น: **ชั้น CPU** (input ≤ 64 px, โมเดลเล็ก, ใช้จัดอันดับ) และ **ชั้น GPU บน Colab** (input 96–224, ViT/Swin/ConvNeXt, ensemble, ตัวจริง) |
| `colab` CLI ติดตั้งแล้วแต่ **ยังไม่ login** | ต้องการให้เจ้าของงาน login 1 ครั้ง (ดู §7) ก่อนถึงชั้น GPU; ระหว่างนี้เดินชั้น CPU ไปก่อน |
| ภาพจริงมีข้อมูล ≤ 44 px, ไบนารี | ต้องทดลอง input size จริง; ห้าม stretch/flip; ใช้ geometry side-channel |
| hidden test set ไม่ทราบที่มา | เลือกโมเดลจากทั้ง stratified val และ doc-disjoint val; รายงาน 3 seed |
| baseline เก่า 96.7% top-1 | ทุก config ต้องเทียบกับตัวเลขนี้ |

## 1. Protocol การวัดผล (ใช้เหมือนกันทุกการทดลอง)
- ข้อมูล: หลัง dedup (TASK-02) · **split หลัก** stratified 80:20 seed 42 · **split รอง** document-disjoint (4 เอกสาร)
- metric: top-1, **balanced accuracy**, macro-F1, top-5, minority(n<50) acc, + เวลา/epoch, params, latency bs=1
- ทุก config ชั้น CPU รันงบเท่ากัน (epochs/steps เท่ากัน) → ใช้ **จัดอันดับ** ไม่ใช่ตัวเลขสุดท้าย
- ผู้ชนะแต่ละแกนไปชั้น GPU รันเต็ม 3 seeds (42/0/1) รายงาน mean ± std
- log ทุก run → `runs/<exp_id>/{config.yaml, log.csv, metrics.json, best.pt}` + แถวใน `reports/experiments.csv`

## 2. Experiment matrix

### A. Baseline & Transfer learning (ชั้น CPU ก่อน แล้วขยายบน GPU)
| id | โมเดล (timm) | pretrained | mode | input | ชั้น |
|---|---|---|---|---|---|
| A0 | SmallCNN (scratch, ~0.3M) | – | – | 32/64 | CPU |
| A1 | resnet18 | ImageNet | frozen / partial (layer4+fc) / full | 64 | CPU |
| A2 | mobilenetv3_large_100 | ImageNet | frozen / partial / full | 64 | CPU |
| A3 | efficientnet_b0 | ImageNet | full | 64 | CPU |
| A4 | resnet18 / mnv3 ผู้ชนะ | ImageNet | full | **32 / 64 / 96 / 128 / 224** | CPU(≤96) + GPU |
| A5 | efficientnet_b2, resnet50, convnext_tiny | ImageNet | partial / full | 96, 224 | GPU |
| A6 | vit_small_patch16 (img_size 64/96/224, pos-embed interpolate), swin_tiny, deit_small | ImageNet | full | 64–224 | GPU |
คำถามที่ต้องตอบ: (1) ImageNet ช่วยจริงไหมเทียบ scratch ที่ 64 px (2) freeze/partial/full (3) input size คุ้มตรงไหน (4) CNN vs ViT บนภาพเล็ก

### B. Data augmentation (บน A-winner, 64 px, CPU)
| id | ชุด |
|---|---|
| B0 | ไม่มี aug |
| B1 | **base**: affine (rot ±8°, shear ±10°, scale 0.85–1.15, translate ±8%) + border jitter + margin jitter |
| B2 | base + **morphological dilate/erode** (ความหนาเส้น) + resolution jitter (0.5–1×) |
| B3 | B2 + elastic + speckle/blur + random erasing |
| B4 | B3 + RandAugment (op list ตัดฟลิป) / TrivialAugment (เทียบ) |
| B5 | B3 + Mixup(α=0.2) / B3 + CutMix (แยกทดสอบ) |
| B6 | **synthetic font data** (27 ฟอนต์ไทยใน `assets/fonts` × น้ำหนัก × degradation ให้เหมือนสแกน) ใช้ 2 แบบ: (a) pretrain-then-finetune, (b) เติมคลาสเล็กให้มีอย่างน้อย 200 ภาพ/คลาส |
ห้าม horizontal/vertical flip ทุก config; รูปตัวอย่างก่อน/หลัง aug ลงรายงาน

### C. ข้อมูลเพิ่ม (public)
| ชุด | ประเภท | คลาสที่ map ได้ | แผน |
|---|---|---|---|
| ALICE-THI (HF `SEACrowd/alice_thi`) — 24,045 ภาพ | ลายมือ 68 ตัวอักษร + 10 เลข | ส่วนใหญ่ของ 72 ผ่าน Unicode | intermediate pretraining (ImageNet→ALICE→ของอาจารย์) |
| Burapha-TH (HF `fwgpiyawudk/...`, BUU) | ลายมือ 68+10 | เหมือนกัน | รวมกับ ALICE ในขั้น intermediate |
| KVIS Thai OCR (HF `SEACrowd/kvis_th_ocr`) 1,079 ภาพ | ลายมือ 44 พยัญชนะ | 42 | เล็ก ใช้เสริม |
| NECTEC printed corpus (600k, 142 คลาส) | **พิมพ์** — ตรงโดเมนที่สุด | ต้องขอสิทธิ์ | ตรวจการเข้าถึง; ถ้าไม่ได้ใช้ synthetic font แทน |
ข้อควรระวัง: ทุกชุด public เป็น **ลายมือ** ส่วนของอาจารย์เป็น **ตัวพิมพ์** → คาดหวังผลปานกลาง ใช้เป็น pretraining ไม่ผสมตรง; สรุปที่มา/ลิขสิทธิ์ทุกชุดในรายงาน

### D. Class imbalance (บน A-winner + B-winner)
| id | วิธี |
|---|---|
| D0 | CE + label smoothing 0.1 (ERM) |
| D1 | class-weighted CE (inverse / sqrt-inverse) |
| D2 | focal loss γ=2 / class-balanced focal (β=0.999) |
| D3 | WeightedRandomSampler (sqrt-inv, cap 3–10×) |
| D4 | **post-hoc logit adjustment** τ ∈ {0, .25, .5, .75} (ฟรี, ใช้กับทุกตัว) |
| D5 | synthetic fill (B6b) เทียบตรงกับ D1–D3 |
สมมติฐานจากงานเก่า: ซ้อนหลายวิธีแรง ๆ จะ over-correct; ERM + logit-adjust τ≈0.25 + synthetic fill น่าจะดีสุด — ต้องวัด

### E. เทคนิคดันคะแนน / แนวคิดเด่น
| id | เทคนิค | ที่มา |
|---|---|---|
| E1 | cosine + warmup, EMA weights, layer-wise LR decay | มาตรฐาน |
| E2 | TTA (shift/scale/dilate-erode เล็ก ๆ, ไม่ flip) | – |
| E3 | ensemble soft-voting 3 seeds × 2 สถาปัตย์ | – |
| E4 | KD จาก ensemble → MobileNetV3 ตัวเดียว | katgpt-rs (distillation notes 032/117) |
| E5 | **geometry side-channel** `[log w, log h, log w/h, ink%]` เข้าหัวจำแนก — โจมตี า/ๅ ตรง ๆ | งานเก่า CNN (ของเราเอง) |
| E6 | **ON/OFF-style 3-channel encoding** แทนการก๊อปเกรย์ 3 ช่อง: ch1 = หมึกไบนารี, ch2 = distance transform (ความหนาเส้น), ch3 = ขอบ/gradient — แรงบันดาลใจจากการแยกช่อง ON/OFF ในระบบการเห็นของแมลงวัน (flyvis, Nature 2024) | fly-connectome |
| E7 | **train-dense → prune-sparse**: magnitude pruning ถึง 90–99% + fine-tune, วาด accuracy-vs-sparsity (บทเรียน FLYNN: sparsity สุดขั้วยังทำงานได้) + INT8/ternary latency (katgpt-rs Research 110) | fly-connectome + katgpt-rs |
| E8 | error analysis: confusion matrix, top-20 คู่สับสน, วิธีแก้เฉพาะจุด (เช่น pairwise head า/ๅ ด้วยความสูง) | – |
E5–E7 คือ "แนวคิดที่น่าสนใจ" ที่เสนอ — จะทดสอบจริงและรายงานตามผลที่วัดได้ ไม่เคลมล่วงหน้า

### F. Application test
inference demo (top-k + confidence, ipywidgets/Gradio ใน notebook), robustness curve (เอียง ±15°, สว่าง,
พื้นหลังรบกวน, เส้นบาง/หนา, occlusion) วัดแบบ FLYNN degradation curve, latency CPU/GPU + ขนาดโมเดล

## 3. ลำดับงาน (phases)
| phase | เนื้อหา | ชั้น | ผลลัพธ์ |
|---|---|---|---|
| P0 | TASK-02 data prep (กำลังรัน), TASK-03 aug + synthetic renderer, TASK-04 training engine | CPU | infra พร้อม, smoke run ผ่าน |
| P1 | A0–A4(≤96), B0–B5, D0–D4 งบเท่ากัน (เช่น 6 epochs ×  subset) | CPU | ตารางจัดอันดับ, เลือกผู้ชนะแต่ละแกน |
| P2 | B6 synthetic (render + pretrain), C external pretrain | CPU/GPU | เทียบ transfer source: ImageNet vs synthetic vs ImageNet→synthetic vs ImageNet→ALICE |
| P3 | A4(224), A5, A6 full-res, ผู้ชนะ 3 seeds, doc-disjoint | **GPU (Colab)** | ตารางหลักของรายงาน |
| P4 | E1–E7, F | GPU+CPU | โมเดลตัวจริง + weight + demo |
| P5 | consolidate .ipynb (Train/Inference), รายงาน, figures | – | deliverable |

## 4. แผน delegate (spec → agy → review)
| task | สิ่งที่สั่ง | acceptance หลัก |
|---|---|---|
| TASK-02 ✅รัน | clean index, splits (seed 42/0/1 + doc), glyph cache, torch Dataset | pytest ผ่าน, overlap=0 |
| TASK-03 | `augment.py` (ชุด B1–B5, ไม่มี flip), `synth.py` render 72 กลิฟ × 27 ฟอนต์ + degradation → cache แยก, รูปตัวอย่างก่อน/หลัง | ภาพ synthetic 72 คลาสครบ, aug ไม่ทำภาพว่าง |
| TASK-04 | `train.py` (config YAML: model/timm, size, mode freeze/partial/full, loss, sampler, EMA, cosine+warmup, LLRD, logit-adjust eval, TTA eval, geometry head, channel encoding), log CSV/JSON, checkpoint best | smoke 1 epoch บน subset ผ่าน, metrics ครบ |
| TASK-05 | `run_matrix.py` รัน list config → `reports/experiments.csv` + ตาราง markdown อัตโนมัติ | รัน 2 config เล็กจบ |
| TASK-06 | แพ็กงานส่ง Colab (`colab run` script: sync code+cache, train, ดึง weight/log กลับ) | job ตัวอย่างจบบน T4 |
| TASK-07 | ดาวน์โหลด + map ALICE-THI/Burapha/KVIS → 72 คลาส + ตารางที่มา/ลิขสิทธิ์ | mapping table + count |
| TASK-08 | ensemble/TTA/KD/pruning scripts | ตัวเลขเทียบก่อน-หลัง |
| TASK-09 | error analysis + figures (confusion, curves, aug examples) | รูปเปิดดูได้ ป้ายไทยถูก |
| TASK-10 | notebook Colab (Train + Inference + Gradio) + ทดสอบด้วย `colab exec -f nb.ipynb` | รันจบบน Colab |
ทุก task: commit ก่อน, log ใน `tasks/DELEGATION-LOG.md`, ผู้รีวิวรัน acceptance เอง

## 5. Risks
- Colab GPU tier ไม่พอ (T4 เท่านั้น/โควตา) → ลด A5/A6 เหลือ 224 px เฉพาะผู้ชนะ; ViT ใช้ 96 px
- synthetic ฟอนต์ไม่เหมือนสแกนจริง → degradation pipeline ต้องจูนด้วย FID-ish sanity (เทียบ ink%, stroke width)
- หลอกตัวเองด้วย val เดียว → 3 seeds + doc-disjoit + caveat ใน caveats field ของ metrics
- ข้อมูลลายมือ public ต่างโดเมน → ใช้เป็น pretraining เท่านั้น ถ้าไม่ช่วยรายงานว่าไม่ช่วย

## 6. คำถาม/สิ่งที่ต้องการจากเจ้าของงาน
1. **Login Colab CLI** (ครั้งเดียว) — คำสั่งอยู่ท้ายสรุป; บอกด้วยว่าบัญชีเป็น Colab ฟรี/Pro (กำหนดว่าจะได้ T4 หรือ A100)
2. ทราบไหมว่า hidden test set ของอาจารย์มาจาก **เอกสารชุดเดิม** หรือ **เอกสารใหม่/ฟอนต์ใหม่**? (มีผลต่อการเลือกโมเดล: ถ้าใหม่ จะถ่วงน้ำหนัก doc-disjoint และ synthetic มากขึ้น)
3. ยืนยันนโยบายทำความสะอาด: ลบ duplicates, ลบทั้ง 11 กลุ่มที่ label ชน, ลบ `Copy of` 6 ไฟล์ (รวม ~2,6xx ภาพ) — ถ้าไม่บอก จะใช้นโยบายนี้
4. อนุญาตดาวน์โหลด dataset จาก HuggingFace (ALICE-THI, Burapha-TH, KVIS) — ตามที่อาจารย์อนุญาต

---
## 8. การตัดสินใจหลังรีวิวรอบ 1 (2026-09-19, เจ้าของงาน)
| คำถาม | คำตอบ | ผลต่อแผน |
|---|---|---|
| Colab tier | **ฟรี** | GPU = T4 เท่านั้น, เซสชันสั้น → งาน GPU ต้องจบใน ≤ 2–3 ชม./job, checkpoint ทุก epoch, 224 px เฉพาะผู้ชนะ; ViT/Swin ใช้ 96–128 px |
| hidden test set | อาจารย์ **เน้น generalize**, น่าจะมีข้อมูลเพิ่ม/แปลก ๆ | **model selection ใช้ doc-disjoint val เป็นเกณฑ์ร่วม** (ไม่ใช่แค่ stratified); ถ่วงน้ำหนัก synthetic fonts, external data, augmentation แรง, robustness test (F) และ TTA; หลีกเลี่ยงการจูน τ/threshold แน่นเกินกับ val เดิม |
| นโยบายลบ duplicates / label ชน / Copy of | ลบได้ | ใช้ตาม TASK-02 (เหลือ 60,117 ภาพ) |
| ดาวน์โหลด HF datasets | ok | เดิน TASK-07 ได้ |
**สถานะแผน: อนุมัติ — เริ่ม P0/P1**

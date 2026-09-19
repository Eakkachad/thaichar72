# บันทึกผลการทดลอง (อัปเดตต่อเนื่อง)

Protocol: หลัง dedup 60,117 ภาพ · split stratified 80:20 seed 42 (train 48,090 / val 12,027, 70 คลาสใน val) ·
metric บน val เต็ม: top-1, **balanced acc** (mean recall ต่อคลาส), macro-F1, minority acc (recall ของคลาสที่ train n<50) ·
checkpoint เลือกด้วย balanced acc (caveat: เลือกบน val เดียวกับที่รายงาน) · เครื่อง: Colab **T4** ผ่าน `colab` CLI ·
ตารางรวมอัตโนมัติ: `reports/experiments.md` (สร้างจาก `runs/*/metrics.json` ด้วย `scripts/collect_results.py`)

## A. Transfer learning baseline — 64 px, aug=base, CE+LS 0.1, AdamW cosine, EMA, 6 epochs, train เต็ม

| exp | โมเดล | pretrained | mode | top-1 | balanced | macro-F1 | minority | s/epoch (T4) |
|---|---|---|---|---:|---:|---:|---:|---:|
| A3 | efficientnet_b0 | ImageNet | **full** | **0.9769** | **0.9809** | **0.9786** | 0.9667 | ~45 |
| A1 | resnet18 | ImageNet | **full** | 0.9753 | 0.9794 | 0.9770 | 0.9667 | 28 |
| A2 | mobilenetv3_large_100 | ImageNet | **full** | 0.9759 | 0.9775 | 0.9568 | 0.9583 | 42 |
| A0 | SmallCNN (scratch 0.43M) | – | – | 0.9742 | 0.9189 | 0.9214 | 0.8917 | 24 |
| A1 | resnet18 | ImageNet | partial (35% ท้าย) | 0.9622 | 0.9067 | 0.9075 | 0.8750 | 26 |
| A2 | mobilenetv3 | ImageNet | partial | 0.9317 | 0.8712 | 0.8586 | 0.8333 | 36 |
| A2 | mobilenetv3 | ImageNet | frozen | 0.9178 | 0.7810 | 0.8035 | 0.6333 | 36 |
| A1 | resnet18 | ImageNet | frozen | 0.8567 | 0.5633 | 0.5916 | 0.2167 | 24 |

ข้อสรุป A:
1. **Fine-tune ทั้งตัวเท่านั้นที่คุ้ม** — frozen backbone (linear probe) ได้ balanced acc แค่ 56–78% แสดงว่าฟิลเตอร์
   ImageNet ดิบ ๆ ไม่เข้ากับภาพไบนารีเล็ก; partial ดีขึ้นแต่ยังแพ้ scratch ในแง่ top-1
2. แต่ **ImageNet init ยังมีค่า**: full fine-tune ชนะ scratch ที่งบเท่ากัน +6 จุด balanced acc (97.9 vs 91.9) และ
   +7.5 จุดบน minority — pretrained weights ช่วยคลาสหางที่มีตัวอย่างน้อยเป็นพิเศษ
3. top-1 ของ 3 backbone ต่างกันแค่ 0.16 จุด (97.5–97.7) — ใกล้เพดานที่ label noise ของคู่ า/ๅ กำหนด;
   ความต่างอยู่ที่ balanced acc / macro-F1 → efficientnet_b0 ≳ resnet18 > mnv3
4. ทั้ง 3 ตัวชนะ baseline เก่าของโปรเจกต์ (ThaiGlyphNet Rust, 96.7 / 96.6) ตั้งแต่ 6 epochs แรก
5. logit adjustment τ: ช่วย A0/partial/frozen มาก (เช่น A0 91.9→96.6) แต่กับ full fine-tune แทบไม่ช่วย (τ=0–0.25 ดีสุด)
   → โมเดลที่เทรนดีอยู่แล้วไม่ต้องแก้ prior หลังบ้าน

ตัดสินใจ: ใช้ **resnet18 full** เป็นแกนของ matrix B/D/E (เร็วสุดต่อ epoch, ผลใกล้ effb0) แล้วค่อยย้าย recipe ที่ชนะไป
effb0 / convnext / ViT ในขั้น S (input size) และการเทรนตัวจริง

## Error analysis ของ A1 (resnet18 full @64) — `reports/analysis/A1_resnet18_full_64_T4/`

![confusion](analysis/A1_resnet18_full_64_T4/confusion_matrix.png)

| อันดับ | คู่ที่สับสน | จำนวนผิด (val) | ลักษณะ |
|---:|---|---:|---|
| 1 | **า ↔ ๅ** | 95 (57+38) | รูปร่างเดียวกัน ต่างแค่ความสูง (20 vs 27 px) — และมี label ชนกันในข้อมูลจริง → noise ceiling |
| 2 | **ว → า** | 59 (17.6% ของ ว!) | ห่วงบนของ ว หายจากการ binarize จนเหลือแค่ตะขอ = า |
| 3 | า → ใ | 28 | เส้นสูงมีห่วง |
| 4 | ั ↔ ้ | 19 | วรรณยุกต์เล็กเหนือบรรทัด ต่างที่หางเล็ก ๆ |
| 5 | ด ↔ ต | 7 | ต่างที่รอยบากบนหัว |
| 6 | ี ↔ ื | 5 | ต่างที่ขีดกลางเส้นเดียว (ื n_train=48) |
| 7 | ช → ซ | 4 | หัวหยัก |

คลาส recall ต่ำสุด: ๘ 60% (n_val=5, สับสนกับ ็), ว 81.9%, ื 83.3%, ๅ 85.4%, า 87.8%
→ **ความผิดพลาดกว่า 70% มาจากกลุ่ม า/ๅ/ว/ใ** ซึ่งเป็นเรื่องความสูงสัมบูรณ์และห่วงที่หายไป ไม่ใช่รูปร่าง
แนวแก้เฉพาะจุดที่จะทดลอง: geometry side-channel (E5: ให้โมเดลรู้ w,h จริง), ON/OFF channels (E6), pairwise
re-scoring า/ๅ ด้วยความสูง, และเช็คว่า TTA ช่วย ว หรือไม่

## B. Data augmentation — resnet18 full @64, CE+LS 0.1, 6 epochs, train เต็ม, stratified val

| exp | aug | synthetic | top-1 | balanced | macro-F1 | minority | best ep |
|---|---|---|---:|---:|---:|---:|---:|
| B_none | **ไม่มี** | – | **0.9848** | 0.9815 | **0.9816** | 0.9667 | 6 |
| B_randaug | RandAugment (N=2, ไม่มี flip) | – | 0.9821 | 0.9825 | 0.9805 | 0.9667 | 6 |
| B_trivial | TrivialAugment | – | 0.9805 | **0.9827** | 0.9793 | 0.9667 | 3 |
| B6_synth_all | full + synthetic 21.6k (ทุกคลาส) | ✓ | 0.9769 | 0.9804 | 0.9743 | n/a* | 6 |
| B6_synth_fill | full + synthetic เติมคลาสเล็กถึง 300 | ✓ | 0.9769 | 0.9799 | 0.9718 | n/a* | 6 |
| B_base (=A1) | affine + margin | – | 0.9753 | 0.9794 | 0.9770 | 0.9667 | 6 |
| B_morph | base + dilate/erode + res-jitter | – | 0.9785 | 0.9694 | 0.9662 | 0.9583 | 5 |
| B_full | morph + elastic/noise/blur/erase | – | 0.9780 | 0.9644 | 0.9617 | 0.9583 | 6 |
| B_full_mixup | full + Mixup α=0.2 | – | 0.9761 | 0.9492 | 0.9493 | 0.9083 | 6 |
| B_full_cutmix | full + CutMix | – | 0.9734 | 0.9403 | 0.9385 | 0.9083 | 5 |
\*minority_acc ของ B6 เป็น NaN เพราะ bug นับ synthetic รวมใน n_train (แก้แล้วในรอบถัดไป)

ข้อสรุป B (บน stratified val = in-distribution):
1. **ที่งบ 6 epochs augmentation หนักทำให้แย่ลง** — ลำดับชัด: none > randaug ≈ trivial > base > morph > full > mixup/cutmix
   ตีความ: val มาจากเอกสารชุดเดียวกับ train (ฟอนต์/DPI เดิม) จึงไม่ต้องการความทนทาน และ aug หนักต้องใช้ epoch มากกว่านี้
2. **Mixup/CutMix ไม่เหมาะกับกลิฟไบนารี** — ภาพผสมเป็นเงาเทาที่ไม่มีใน test, balanced acc ร่วง 3–4 จุด (ยืนยันสมมติฐาน)
   แต่ post-hoc τ=0.75 ดึงกลับได้ถึง 98.03 → mixup เปลี่ยน prior ของโมเดล มากกว่าทำลาย feature
3. **Synthetic data ช่วยเมื่อ aug หนัก**: full aug อย่างเดียว 96.44 → full + synthetic 98.04 balanced (+1.6 จุด)
   และ synthetic ทำให้ ฃ/ฑ (n=1) มีตัวอย่างฝึกจริง ๆ
4. RandAugment/TrivialAugment (แบบ 1–2 op ต่อภาพ) เป็นจุดสมดุลที่ดี: เสีย top-1 เล็กน้อยแต่ balanced acc สูงสุด
5. **คำเตือน**: ตารางนี้วัด in-distribution เท่านั้น — การตัดสินใจเรื่อง aug สำหรับ hidden test ที่ "เน้น generalize" ต้องดู
   doc-disjoint split (คิวถัดไป) และ robustness curve: โมเดล base-aug พังบนภาพเทา/เส้นบางใน TASK-11 (Otsu ที่ inference แก้เรื่องเทาได้)

## D. Class imbalance — resnet18 full @64, aug=full (ฐานเดียวกับ B_full/D0), 6 epochs, stratified val

| exp | วิธี | top-1 | balanced | macro-F1 | minority | τ ดีสุด (bal) |
|---|---|---:|---:|---:|---:|---:|
| D0 | CE + LS 0.1 (ERM) | **0.9780** | 0.9644 | 0.9617 | 0.9583 | τ=0.75 → 0.9806 |
| D1 | weighted CE, w ∝ n^-0.5 | 0.9727 | **0.9840** | 0.9713 | **0.9750** | τ=0 |
| D1 | weighted CE, w ∝ n^-1 | **0.1160** | 0.6041 | 0.6010 | 0.9583 | – |
| D2 | focal γ=2 | 0.9778 | 0.9695 | 0.9645 | 0.9667 | τ=0.5 → 0.9832 |
| D2 | class-balanced focal β=0.999 | 0.9672 | 0.9751 | 0.9538 | 0.9750 | τ=0 |
| D3 | WeightedRandomSampler sqrt-inv, cap 5 | 0.9721 | 0.9837 | **0.9768** | 0.9750 | τ=0 |
| D3 | WeightedRandomSampler inv, cap 10 | 0.9694 | 0.9841 | 0.9721 | 0.9750 | τ=0 |
| D4 | post-hoc logit adjustment บน D0 | (τ=0.75) 0.9718* | 0.9806 | – | – | ฟรี |

ข้อสรุป D:
1. **การชดเชย imbalance ทุกแบบแลก top-1 กับ balanced acc**: +2 จุด balanced ↔ −0.5 ถึง −0.9 จุด top-1
   เพราะคลาสหาง (เลขไทย, ฃ ฑ ฬ) ถูกทายบ่อยขึ้นและกินคลาสใหญ่ (า/ๅ/ว) ไปบ้าง
2. **inverse-frequency เต็ม ๆ พังทั้งระบบ** (top-1 11.6%) — น้ำหนักต่างกัน 5,025 เท่า → โมเดลทายคลาสหายากทุกภาพ
   ต้องใช้ sqrt หรือ cap เสมอ (ยืนยันข้อค้นพบของโปรเจกต์เก่า)
3. sqrt-inverse **weighted loss ≈ sqrt-inverse sampler** (98.40 vs 98.37 balanced) → เลือกวิธีไหนก็ได้ sampler ให้ macro-F1 ดีกว่าเล็กน้อย
4. **logit adjustment หลังบ้านบน ERM ให้ผลใกล้กัน (98.06) โดยไม่ต้องเทรนใหม่** และปรับ τ ได้ตาม metric ที่อาจารย์ใช้
   (ถ้าคะแนนคือ plain accuracy บน test → ใช้ ERM/τ=0; ถ้าเป็น balanced → τ≈0.5–0.75)
5. **synthetic data (B6) ให้ balanced 98.04 โดยไม่เสีย top-1** (97.69) → เป็นวิธีจัดการ imbalance ที่ "ฟรี" ที่สุด
   และเป็นทางเดียวที่ทำให้ ฃ/ฑ เรียนได้จริง

*ค่า top-1 ที่ τ=0.75 ประมาณจาก tau sweep ของ D0 (`runs/D0_ce_ls_resnet18_64_T4/metrics.json`)

## E. เทคนิคเสริม — resnet18 full @64, aug=full (ฐาน D0 = 97.80 / 96.44), 6 epochs, stratified val

| exp | เทคนิค | top-1 | balanced | macro-F1 | minority | หมายเหตุ |
|---|---|---:|---:|---:|---:|---|
| D0 | ฐาน (EMA on) | 0.9780 | 0.9644 | 0.9617 | 0.9583 | |
| E1_noema | ปิด EMA | 0.9780 | 0.9636 | 0.9609 | 0.9500 | EMA แทบไม่มีผลที่ 6 epochs |
| E1_llrd | layer-wise LR decay 0.8 | 0.9762 | 0.9557 | 0.9542 | 0.9167 | **แย่ลง** — ชั้นต้นของ ImageNet ต้องปรับตัวมาก (สอดคล้อง frozen/partial แพ้) |
| E5_geometry | geometry side-channel [log w, log h, log w/h, ink%] | 0.9775 | **0.9690** | 0.9655 | 0.9667 | +0.5 balanced, top-1 เท่าเดิม |
| E6_onoff | 3-channel ON/OFF (ink / distance-transform / edges) | 0.9756 | 0.9620 | 0.9609 | 0.9500 | ไม่ช่วย (−0.2) บน in-distribution val |
| E6_onoff+geo | ทั้งสองอย่าง | 0.9775 | 0.9626 | 0.9615 | 0.9500 | |
| E3 ensemble | soft-vote resnet18+effb0+smallcnn (A) | 0.9778 | 0.9815 | 0.9790 | 0.9667 | +0.1–0.25 เหนือตัวเดี่ยวที่ดีสุด — โมเดลผิดที่จุดเดียวกัน (า/ๅ) |
| E4 logit-adj | τ sweep บนทุกตัว | | | | | ช่วย 0.5–2 จุด balanced เมื่อโมเดลยังไม่ balanced; ไม่ช่วยตัวที่ดีอยู่แล้ว |

ข้อสรุป E: เทคนิค "ฟรี" ที่คุ้มคือ geometry side-channel และ logit adjustment; ensemble ให้กำไรน้อยเพราะ error ไม่อิสระ;
ON/OFF encoding ที่ได้แรงบันดาลใจจาก fly visual system ไม่ช่วยในเงื่อนไขนี้ — จะทดสอบซ้ำบน doc-disjoint (ที่ต้องการความทนทานมากกว่า) ก่อนตัดสิน

## C. Transfer source อื่นนอกจาก ImageNet (stage-1 → zero-shot บน val จริง)

| stage-1 | ข้อมูล | epochs | top-1 บน val จริง (ไม่เคยเห็นข้อมูลจริง) | balanced | minority |
|---|---|---:|---:|---:|---:|
| B6a_synth_pretrain | ImageNet → **ฟอนต์สังเคราะห์ 21,600 ภาพ / 26 ฟอนต์** | 8 | **0.9369** | 0.9188 | 0.9083 |
| C1_ext_pretrain | ImageNet → **ลายมือ public 100,985 ภาพ** (ALICE-THI + Burapha-TH) | 4 | 0.9148 | 0.9112 | 0.9500 |

ข้อค้นพบ: โมเดลที่เห็นแต่ตัวอักษรที่ render จากฟอนต์ (พร้อม degradation) รู้จำกลิฟสแกนจริงได้ **93.7%** โดยไม่เห็นข้อมูล
ของอาจารย์เลย → synthetic pipeline ของเราใกล้โดเมนจริงมาก และเป็นหลักฐานว่าวิธีนี้จะช่วย generalisation ไปฟอนต์ใหม่
ส่วนลายมือ public ต่างโดเมน (ลายมือ vs ตัวพิมพ์) แต่ยัง zero-shot ได้ 91.5% และช่วยคลาสหางเป็นพิเศษ (minority 95.0%)
ผล stage-2 (fine-tune ต่อบนข้อมูลจริง) อยู่ในตารางถัดไปเมื่อรันเสร็จ

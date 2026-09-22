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

## S. Input size — resnet18 full, aug=full, 6 epochs, stratified val (T4)

| size | top-1 | balanced | macro-F1 | s/epoch (T4) | latency bs=1 CPU (ms) |
|---:|---:|---:|---:|---:|---:|
| 32 | 0.9752 | 0.9588 | 0.9506 | 29.5 | 9.0 |
| **64** | 0.9780 | 0.9644 | 0.9617 | 28.2 | 12.1 |
| 96 | 0.9768 | 0.9587 | 0.9582 | 41.5 | 17.0 |
| 128 | 0.9785 | 0.9568 | 0.9568 | 52.1 | ~25 |
| 224 | 0.9776 | 0.9673 | 0.9635 | 116.5 | 71.2 |

ข้อสรุป S: **64 px คือจุดคุ้มที่สุด** — ใหญ่กว่านั้นไม่เพิ่มข้อมูล (กลิฟจริง ≤ 44 px, median 16×19) แต่เวลา/epoch เพิ่ม 4× และ latency 6×
ที่ 224 px (ขนาดที่ ImageNet ถูกฝึก) ได้ balanced ดีขึ้นเล็กน้อย (+0.3) ไม่คุ้มต้นทุน; 32 px เริ่มเสียรายละเอียดของวรรณยุกต์เล็ก ๆ

## G. Stratified vs Document-disjoint — ตัวตัดสินสำหรับ hidden test ที่ "เน้น generalize" (6 epochs, 64 px)

| recipe | strat top-1 | strat balanced | **doc top-1** | **doc balanced** | ช่องว่าง balanced |
|---|---:|---:|---:|---:|---:|
| resnet18, aug=base (A1, **doc ข้อมูลเต็ม** `A1_resnet18_full_64_doc_full`, RTX 4060) | 0.9753 | 0.9794 | 0.9728 | 0.9740 | −0.5 |
| resnet18, aug=base (A1, doc `subset_frac=0.25` — จุดข้อมูลเดิมที่ทำให้เข้าใจผิด) | 0.9753 | 0.9794 | 0.9634 | 0.9361 | (−4.3 แต่เทียบไม่ได้) |
| resnet18, aug=**full** (B_full) | 0.9780 | 0.9644 | 0.9726 | 0.9611 | **−0.3** |
| efficientnet_b0, aug=base (A3, doc 25 %) | 0.9769 | 0.9809 | 0.9625 | 0.9544 | (−2.7, เทียบไม่ได้) |
| SmallCNN scratch (A0, doc 25 %) | 0.9742 | 0.9189 | 0.9564 | 0.8286 | (−9.0, เทียบไม่ได้) |

ข้อค้นพบ (แก้ไข 2026-09-19 ค่ำ หลัง confound check บน RTX 4060): ตารางฉบับแรกสรุปว่า "ลำดับกลับด้าน — base เสีย 4.3 จุดข้ามเอกสาร"
แต่แถว A0/A1/A3 บน doc ถูกรันด้วยข้อมูล train แค่ 25 % เมื่อรัน A1 (base) บน doc ด้วยข้อมูลเต็ม 48,133 ภาพ ได้ **97.28 / 97.40**
→ ช่องว่าง strat→doc ของ base เหลือแค่ **−0.5 จุด** เท่ากับ none/trivial ดังนั้น "aug หนักชนะ base บนเอกสารใหม่" **ไม่เป็นจริง**:
full (97.26 / 96.11) แพ้ base บน doc balanced ถึง 1.3 จุด ส่วนที่ยังจริงคือ (ก) doc val ยากกว่า strat val ทุก recipe ~0.3–0.5 จุด
และ (ข) การเลือก recipe ต้องดูคอลัมน์ doc ประกอบ เพราะ geometry side-channel (ข้อมูลเต็ม) ยังพังข้ามเอกสารจริง (ดูตารางเต็มด้านล่าง)

### G (ฉบับเต็ม) — ทุก recipe ที่รันทั้ง 2 split (resnet18 @64, 6 epochs) เรียงตาม doc balanced

| recipe | aug | strat top-1 | strat bal | **doc top-1** | **doc bal** | doc minority | gap bal (จุด) |
|---|---|---:|---:|---:|---:|---:|---:|
| D3 sampler sqrt-inv (cap 5) | full | 0.9721 | 0.9837 | 0.9670 | **0.9816** | **0.9915** | −0.2 |
| B_morph | morph | 0.9785 | 0.9694 | 0.9729 | 0.9786 | 0.9744 | **+0.9** |
| B_trivial | trivial | 0.9805 | 0.9827 | **0.9790** | 0.9784 | 0.9658 | −0.4 |
| B_none | none | **0.9848** | 0.9815 | 0.9780 | 0.9761 | 0.9829 | −0.5 |
| B6_synth_all | full + synth | 0.9769 | 0.9804 | 0.9670 | 0.9754 | 0.9487 | −0.5 |
| B6_synth_all | full + synth | 0.9769 | 0.9804 | 0.9670 | 0.9754 | 0.9487 | −0.5 |
| **B6a_ft (init จาก synthetic pretrain)** | full | 0.9792 | 0.9843 | 0.9732 | 0.9759 | 0.9487 | −0.8 |
| **A1 resnet18 (doc ข้อมูลเต็ม, 4060)** | base | 0.9753 | 0.9794 | 0.9728 | 0.9740 | 0.9487 | −0.5 |
| B_randaug | randaug | 0.9821 | 0.9825 | 0.9780 | 0.9700 | 0.9487 | −1.2 |
| B_full | full | 0.9780 | 0.9644 | 0.9726 | 0.9611 | 0.9487 | −0.3 |
| E6_onoff | full | 0.9756 | 0.9620 | 0.9722 | 0.9603 | 0.9316 | −0.2 |
| **E5_geometry** | full | 0.9775 | 0.9690 | 0.9731 | **0.9410** | 0.8974 | **−2.8** |
| *A3 effb0 (doc 25 % — เทียบไม่ได้)* | base | 0.9769 | 0.9809 | 0.9625 | 0.9544 | 0.9582 | (−2.7) |
| *A1 resnet18 (doc 25 % — เทียบไม่ได้)* | base | 0.9753 | 0.9794 | 0.9634 | 0.9361 | 0.9433 | (−4.3) |
| *A0 SmallCNN (doc 25 % — เทียบไม่ได้)* | base | 0.9742 | 0.9189 | 0.9564 | 0.8286 | 0.8299 | (−9.0) |

> **Confound check (2026-09-19 ค่ำ, RTX 4060 — HANDOFF §3.1 ข้อ 1):** แถว A0/A1/A3 เดิม (aug=base) ถูกรันบน doc split ด้วย `subset_frac=0.25`
> (12,034 ภาพ) ขณะที่แถวอื่นใช้ข้อมูลเต็ม 48,133 ภาพ → รัน `A1_resnet18_full_64_doc_full` (base, doc, ข้อมูลเต็ม, 6 ep) ใหม่:
> **top-1 0.9728 / balanced 0.9740 / macro-F1 0.9610 / minority 0.9487** (best ep 5) → ช่องว่าง strat→doc ของ base = −0.5 จุด
> ไม่ต่างจาก none/trivial → **ข้อสรุปเดิม "base ทำ generalisation พัง" ถูกยกเลิก**; ความต่าง 4.3 จุดมาจากปริมาณข้อมูล ไม่ใช่ preset
> แถว A0/A3 doc 25 % ยังคงเทียบไม่ได้ (ไม่รันซ้ำเพราะไม่มีผลต่อการเลือก recipe) นอกจากนี้ `B_base_resnet18_64_T4` คือ config เดียวกับ
> `A1_resnet18_full_64_T4` (รันซ้ำ ได้ค่าเท่ากันทุกหลัก) — นับเป็นจุดข้อมูลเดียว
> รันเพิ่มพร้อมกัน: `B6a_ft_resnet18_64_doc` (init จาก synthetic pretrain, aug=full, doc) ได้ 0.9732 / 0.9759 → ดีกว่า B_full_doc
> (init ImageNet ตรง ๆ, recipe เดียวกัน) **+1.5 จุด balanced** บนเอกสารใหม่ → ประโยชน์ของ synthetic pretraining ยืนยันได้ทั้ง 2 split

ข้อสรุป G (สำคัญที่สุดของงานนี้ — ฉบับแก้หลัง confound check):
1. **ทุก preset เบา (none / base / trivial / morph) เสียเพียง 0.3–0.5 จุด balanced เมื่อข้ามเอกสาร** และ morph ดีขึ้นด้วยซ้ำ (+0.9)
   สิ่งที่พังข้ามเอกสารจริงคือ **aug หนัก (`full`: −0.3 แต่ตั้งต้นต่ำ, doc bal 96.1)**, **randaug (−1.2)** และ **geometry side-channel (−2.8)**
   → augmentation แบบ 1 op ต่อภาพ (trivial) หรือ morph คือจุดสมดุลที่ generalise ได้ดีที่สุดโดยไม่เสีย in-distribution
2. **geometry side-channel ล้มเหลวข้ามเอกสาร** (94.1, −2.8; ข้อมูลเต็ม จึงไม่ใช่ confound): ความกว้าง/สูงสัมบูรณ์เป็นพิกเซลผูกกับ DPI/ขนาดฟอนต์
   ของเอกสาร → ช่วยใน in-distribution แต่เป็น shortcut ที่ไม่ generalize — **ตัดออกจาก recipe สุดท้าย** (บทเรียนตรงข้ามกับสมมติฐาน E5)
3. **sampler sqrt-inverse ให้ doc balanced สูงสุด 98.16 และ minority 99.2%** แต่เสีย top-1 ~1 จุด — เหมาะถ้าอาจารย์วัด balanced/macro
4. สำหรับ **plain accuracy** บนเอกสารใหม่: trivial 97.90 ≈ none 97.80 ≈ randaug 97.80 > B6a_ft 97.32 ≈ morph 97.29 ≈ base 97.28 ≈ full 97.26
5. ImageNet init ยังสำคัญมากข้ามเอกสาร: scratch ร่วง 9 จุด balanced (แม้แถว A0 doc จะเป็นข้อมูล 25 % ช่องว่าง 9 จุดก็ใหญ่เกินจะอธิบายด้วยปริมาณข้อมูลอย่างเดียว)
6. **synthetic-font pretraining ช่วยข้ามเอกสารด้วย**: B6a_ft doc 97.59 vs B_full doc 96.11 (+1.5 balanced, recipe เดียวกันต่างแค่ init)
→ recipe ตัวจริง: resnet18/effb0 full fine-tune @64, **TrivialAugment (หรือ morph) + synthetic fonts (เป็น pretraining หรือ extra data)**,
   ไม่ใช้ geometry, 20 epochs + EMA, TTA เป็นตัวเลือกเปิด/ปิดแยก, เลือก τ/sampler ตาม metric ที่คาดว่าอาจารย์ใช้ (รอบ F1–F17 ดูหัวข้อ F)

### C (stage-2) — fine-tune ต่อบนข้อมูลจริง 6 epochs, aug=full, stratified val (เทียบฐาน B_full ที่ init จาก ImageNet ตรง ๆ)

| init | stage-1 | top-1 | balanced | macro-F1 | minority |
|---|---|---:|---:|---:|---:|
| ImageNet (B_full) | – | 0.9780 | 0.9644 | 0.9617 | 0.9583 |
| ImageNet + synthetic เป็น extra data (B6_synth_all) | – | 0.9769 | 0.9804 | 0.9743 | – |
| **ImageNet → synthetic fonts (B6a_ft)** | 8 ep บน 21.6k synth | **0.9792** | **0.9843** | **0.9815** | **0.9750** |
| ImageNet → ลายมือ public (C1_ft) | 4 ep บน 101k ext | 0.9781 | 0.9812 | 0.9790 | 0.9667 |
| C1 init + synthetic fill 300 (C1B6_ft) | | 0.9682 | 0.9808 | 0.9690 | 0.9667 |

ข้อสรุป C: **การ pretrain ด้วยฟอนต์สังเคราะห์แล้วค่อย fine-tune** ให้ผลดีกว่าการเทข้อมูลสังเคราะห์รวมกับข้อมูลจริง
(+0.4 balanced, +0.7 macro-F1) และดีกว่า ImageNet ตรง ๆ +2.0 balanced ที่งบเท่ากัน — เป็น transfer แบบ 2 ชั้น
(ImageNet → โดเมนตัวอักษรไทยสังเคราะห์ → ข้อมูลจริง) ที่ตรงกับหัวข้อ Transfer Learning ของโจทย์ที่สุด;
ลายมือ public ก็ช่วย (+1.7) แม้ต่างโดเมน → ทั้งสองแนวเข้าสู่รอบตัวจริง 20 epochs (F11–F17)

## F. Final candidates — 20 epochs, resnet18/effb0 @64, CE+LS 0.1, EMA, stratified val seed 42 (RTX 4060, WSL2; F1–F4 บน T4)

ทุกตัวใช้ recipe ฐานเดียวกัน (AdamW 1e-3, cosine, warmup 1 ep, batch 128, EMA) ต่างกันที่ **init** (ImageNet / B6a = synthetic-font pretrain /
C1 = public handwriting pretrain), **aug** และ **synthetic เป็น extra data** ค่าที่รายงานคือ **raw** (ไม่ TTA) บน val เต็ม 12,027 ภาพ; TTA (8 มุมมอง)
แยกไว้คอลัมน์ท้าย F5/F6 เป็น negative control ที่ตั้งใจรัน (geometry / ON-OFF) เรียงตาม balanced acc

| exp | model | init | aug | synth | extra | top-1 | balanced | macro-F1 | minority | TTA top-1 / bal | best ep | s/ep (4060) |
|---|---|---|---|:-:|---|---:|---:|---:|---:|---:|---:|---:|
| **F16** | resnet18 | B6a synth | morph | – | – | 0.9820 | **0.9856** | **0.9830** | 0.9750 | 0.9802 / 0.9854 | 18/20 | 8.8 |
| **F7** | resnet18 | ImageNet | morph | ✓ | – | 0.9809 | 0.9851 | 0.9813 | 0.9750 | 0.9791 / 0.9823 | 19/20 | 12.8 |
| *F5 (neg. control)* | resnet18 | ImageNet | trivial | ✓ | geometry | 0.9854 | 0.9847 | 0.9827 | 0.9667 | 0.9821 / 0.9845 | 19/20 | 11.8 |
| F18 (เพิ่มหลัง robustness gate) | resnet18 | B6a synth | full | – | – | 0.9802 | 0.9843 | 0.9812 | 0.9750 | 0.9797 / 0.9843 | 19/20 | 15.9* |
| **F10** | efficientnet_b0 | ImageNet | trivial | ✓ | – | 0.9836 | 0.9842 | 0.9799 | 0.9750 | 0.9832 / **0.9852** | 20/20 | 24.1 |
| F15 | resnet18 | B6a synth | trivial | ✓ | – | 0.9848 | 0.9840 | 0.9819 | 0.9667 | 0.9825 / 0.9809 | 20/20 | 11.7 |
| F12 | resnet18 | C1 ext | trivial | – | – | 0.9854 | 0.9834 | 0.9804 | 0.9667 | 0.9845 / 0.9831 | 18/20 | 8.3 |
| *F6 (neg. control)* | resnet18 | ImageNet | randaug | ✓ | onoff | 0.9827 | 0.9832 | 0.9763 | 0.9667 | 0.9810 / 0.9842 | 13/20 | 12.1 |
| F11 | resnet18 | C1 ext | trivial | ✓ | – | 0.9847 | 0.9830 | 0.9784 | 0.9667 | 0.9825 / 0.9824 | 20/20 | 11.6 |
| F14 | resnet18 | B6a synth | trivial | – | – | 0.9855 | 0.9829 | 0.9817 | 0.9667 | 0.9841 / 0.9826 | 20/20 | 8.3 |
| F1 | resnet18 | ImageNet | none | – | – | **0.9859** | 0.9828 | 0.9817 | 0.9667 | 0.9790 / 0.9790 | 18/20 | 17.2 (T4) |
| F8 | resnet18 | ImageNet | trivial | – | – | 0.9854 | 0.9828 | 0.9803 | 0.9667 | 0.9832 / 0.9830 | 18/20 | 8.4 |
| **F19 (เพิ่มหลัง gate) — ผู้ชนะ** | resnet18 | B6a synth | randaug | – | – | 0.9839 | 0.9827 | 0.9791 | 0.9667 | 0.9827 / 0.9824 | 15/20 | 15.8* |
| F17 | resnet18 | B6a synth | none | – | – | 0.9804 | 0.9825 | 0.9804 | 0.9667 | 0.9791 / 0.9769 | **3**/20 | 8.1 |
| F2 | resnet18 | ImageNet | randaug | – | – | 0.9837 | 0.9818 | 0.9775 | 0.9667 | 0.9836 / 0.9830 | 20/20 | 27.5 (T4) |
| F9 | resnet18 | ImageNet | trivial | ✓ | – | 0.9845 | 0.9814 | 0.9782 | 0.9583 | 0.9826 / 0.9809 | 19/20 | 11.9 |
| F3 | resnet18 | ImageNet | randaug | ✓ | – | 0.9845 | 0.9804 | 0.9774 | 0.9583 | 0.9825 / 0.9801 | 20/20 | 39.3 (T4) |
| F13 | resnet18 | C1 ext | morph | ✓ | – | 0.9781 | 0.9777 | 0.9670 | 0.9583 | 0.9771 / 0.9777 | 14/20 | 12.4 |
| F4 | efficientnet_b0 | ImageNet | randaug | ✓ | – | 0.9786 | 0.9775 | 0.9694 | 0.9583 | 0.9776 / 0.9775 | **4**/20 | 59.4 (T4) |

ข้อสรุป F (บน stratified val):
1. **เพดานของ in-distribution val ถึงแล้ว**: ทั้ง 17 ตัวอยู่ในช่วง top-1 97.81–98.59 / balanced 97.75–98.56 ส่วนกลุ่มบน 10 ตัวห่างกันไม่ถึง 0.3 จุด
   ซึ่งเท่ากับ ~35 ภาพจาก 12,027 และเล็กกว่า label noise ของคู่ า/ๅ → **การจัดอันดับบน strat อย่างเดียวไม่มีความหมาย ต้องใช้ doc split ตัดสิน** (§F-doc ด้านล่าง)
2. **morph ให้ balanced/macro-F1 สูงสุด (F16 98.56, F7 98.51) แต่เสีย top-1 ~0.4 จุด** เทียบกลุ่ม trivial/none (98.54–98.59): dilate/erode/res-jitter
   ช่วยคลาสหางที่เส้นบาง/หนาผิดปกติ แต่ทำให้คู่ า/ๅ/ว (คลาสใหญ่) ผิดเพิ่ม
3. **synthetic-font pretraining (B6a init) เป็นกลางถึงบวกเล็กน้อยเมื่อมี aug** (F14 vs F8: +0.01 bal; F15 vs F9: +0.26; F16 คือ recipe ที่ดีที่สุด)
   แต่ **เป็นลบเมื่อไม่มี aug** (F17 vs F1: −0.55 top-1, best epoch = 3 แล้ว overfit) → init ที่ fit โดเมนแล้วต้องมี regulariser
4. **public-handwriting init (C1) ให้ผลปนกัน**: trivial ≈ เท่า ImageNet (F12 vs F8 +0.06 bal), แต่ morph+synth แย่ลงชัด (F13 vs F7 −0.74 bal, best ep 14)
   → ไม่เข้ารอบ; แนวคิด "โดเมนกลาง" ที่ได้ผลคือฟอนต์สังเคราะห์ (โดเมนตัวพิมพ์เดียวกัน) ไม่ใช่ลายมือ
5. **synthetic เป็น extra data บน strat แทบไม่มีผล** (F9 vs F8 −0.14 bal, F15 vs F14 +0.11, F3 vs F2 −0.14) แต่เพิ่มเวลา/epoch ~40 % →
   ถ้าจะใช้ synthetic ให้ใช้เป็น **pretraining** (F16) ไม่ใช่เท-รวม; ข้อยกเว้นคือ effb0 (F10) ที่ต้องการข้อมูลมากกว่าและได้ 98.42
6. **effb0 (F10) เสถียรเมื่อใช้ trivial** (98.36/98.42, best ep 20) ต่างจาก F4 randaug ที่พังตั้งแต่ epoch 4 และ F10 เป็นตัวเดียวที่ **TTA ช่วย balanced ชัด (+0.10 → 98.52)**
   แต่ช้ากว่า resnet18 ~2.5× ต่อ epoch
7. **TTA (8 canvas views) ไม่คุ้มเป็นค่าเริ่มต้น**: ลด top-1 ทุกตัว (−0.1 ถึง −0.7 จุด; แรงสุดกับ none/morph) ช่วย balanced เฉพาะ F10/F6 →
   ปิด TTA ใน deliverable; เปิดเฉพาะเมื่อผู้ชนะได้กำไรจริงบน doc split
8. **negative controls**: F5 geometry ได้ 98.54/98.47 บน strat (อันดับ 3) *ตามคาด* เพราะ w/h เป็น shortcut ใน in-distribution — ต้องดูผล doc
   (§F-doc) ก่อนตัดสิน; F6 ON/OFF vs F3 (recipe เดียวกันไม่มี onoff): +0.28 bal / −0.18 top-1, best ep 13 → ไม่มีกำไรที่ชัดเจน ยืนยันผล E6
→ **top-3 เข้ารอบ doc-split** (raw strat balanced, ไม่รวม negative control): **F16, F7, F10** (+ F5 บน doc เป็นหลักฐานผลลบของ geometry ที่ 20 epochs)

\* s/ep ที่มี * วัดขณะรัน 2 งานพร้อมกันบน GPU เดียว (ไม่เทียบกับแถวอื่น)

### F-doc — document-disjoint + robustness gate ของผู้เข้ารอบ (RTX 4060, 2026-09-19 ค่ำ)

กติกาเลือกผู้ชนะ (HANDOFF §3.3): (a) doc top-1 → (b) doc balanced → (c) strat top-1 **และ** robustness ไม่แย่กว่า F2:
mean top-1 บนทุกแถว non-clean ของ `scripts/robustness.py --binarize` (val strat เต็ม, Otsu หลัง corruption, 10 corruption × 3–4 ระดับ)
≥ 0.9038 − 0.005. รอบแรก (top-3 = F16/F7/F10) **ไม่มีตัวไหนผ่าน gate** จึงขยายไปกลุ่ม trivial/randaug (F8/F12/F14/F15/F2) และเพิ่ม
F18/F19 (B6a init + full / randaug) ที่ออกแบบจากสาเหตุที่พบ — เรียงตาม doc top-1:

| exp | recipe | strat top-1 / bal | **doc top-1** | **doc bal** | doc TTA top-1 / bal | robust mean | gate |
|---|---|---:|---:|---:|---:|---:|:-:|
| **F19** | B6a init + randaug | 0.9839 / 0.9827 | **0.9822** | 0.9807 | 0.9815 / 0.9809 | **0.9045** | ✅ |
| F8 | trivial | 0.9854 / 0.9828 | 0.9821 | 0.9781 | 0.9809 / 0.9780 | 0.8892 | ❌ |
| F14 | B6a init + trivial | 0.9855 / 0.9829 | 0.9819 | 0.9815 | 0.9811 / 0.9815 | 0.8931 | ❌ |
| F12 | C1 init + trivial | 0.9854 / 0.9834 | 0.9816 | 0.9786 | 0.9803 / 0.9791 | 0.8956 | ❌ |
| F2 (baseline ของ gate) | randaug | 0.9837 / 0.9818 | 0.9812 | 0.9816 | 0.9796 / 0.9817 | 0.9038 | ✅ |
| F10 | effb0 trivial + synth | 0.9836 / 0.9842 | 0.9803 | **0.9837** | 0.9786 / 0.9827 | 0.8878 | ❌ |
| F15 | B6a init + trivial + synth | 0.9848 / 0.9840 | 0.9801 | 0.9810 | 0.9784 / 0.9802 | 0.8893 | ❌ |
| F18 | B6a init + full | 0.9802 / 0.9843 | 0.9784 | 0.9769 | 0.9762 / 0.9766 | 0.9063 | ✅ |
| F16 | B6a init + morph | 0.9820 / 0.9856 | 0.9781 | 0.9816 | 0.9769 / 0.9828 | 0.8500 | ❌ |
| F7 | morph + synth | 0.9809 / 0.9851 | 0.9771 | 0.9810 | 0.9774 / 0.9795 | 0.8525 | ❌ |
| *F5 (geometry, neg. control)* | trivial + synth + geo | 0.9854 / 0.9847 | 0.9763 | 0.9803 | 0.9761 / 0.9798 | 0.8888 | ❌ |

ตาราง per-corruption ทั้งหมด: `reports/robustness/<exp>_full_bin/{results.csv,summary.md,curves.png}`

ข้อสรุป F-doc:
1. **doc top-1 ของทุกตัวอยู่ในช่วง 97.6–98.2 (ต่างกัน ≤ 0.6 จุด ≈ 70 ภาพ)** และช่องว่าง strat→doc เหลือแค่ 0.2–0.5 จุด ทุก recipe →
   ที่ 20 epochs กับ aug เบา/กลาง แทบไม่มีปัญหา generalise ข้ามเอกสารแล้ว ตัวตัดสินจริงจึงเป็น **robustness**
2. **robustness แยกกลุ่มชัดตาม "โมเดลเคยเห็น noise ตอนเทรนหรือไม่"**: morph (F16/F7) ทน rotate/translate/blur/downscale *ดีที่สุด*
   (เช่น rotate 20°: 0.953 vs 0.914 ของ F2) แต่พังที่ salt-pepper 0.2 (0.175) และ background noise (0.80) เพราะ preset `morph` ไม่มี noise op;
   trivial (1 op/ภาพ) เห็น noise น้อยกว่า randaug (2 op/ภาพ) จึงต่ำกว่า gate 0.8–1.5 จุดทุกตัว; **randaug และ full เท่านั้นที่ผ่าน**
3. **geometry (F5) แม้ที่ 20 epochs ก็ยังมีช่องว่าง strat→doc ด้าน top-1 กว้างสุด (−0.9 จุด) และ best epoch แค่ 4** → ยืนยันตัดออก
4. **effb0 (F10) ให้ doc balanced สูงสุด 98.37** แต่ top-1 ต่ำกว่าและ robustness ต่ำ (−1.6) + ช้ากว่า 2.5× → เก็บไว้เป็น teacher/สมาชิก ensemble ที่หลากหลาย
5. **B6a init + randaug (F19) = F2 ที่เปลี่ยน init**: ดีขึ้นทุกแกน (strat +0.02/+0.09, doc top-1 +0.10, robustness +0.07) และเป็นตัวเดียวที่
   ทั้งนำ doc top-1 และผ่าน gate → **ผู้ชนะ**; F18 (full) ทนสุด (0.9063) แต่แพ้ doc 0.4 จุด
→ **โมเดลสุดท้าย: F19_r18_b6ainit_randaug_20** (resnet18 @64, ImageNet → synthetic-font pretrain → real, RandAugment N=2 ไม่ flip,
   CE+LS 0.1, AdamW cosine, EMA, 20 ep; TTA ปิด) — ขั้นถัดไป: label audit (§H), 3 seeds, ensemble/KD, export

## H. Label audit จากแพ็กเกจ DataV2 ของทีม (2026-09-19 ค่ำ)

ทีมส่ง `DataV2/datav2.zip` (+ CHANGELOG, split, EDA) ที่ (ก) ย้ายภาพที่ label ผิดไปคลาสที่ถูก (ตรวจด้วยตา), (ข) ทิ้งภาพ outlier,
(ค) เพิ่มภาพ augmented ล่วงหน้า 1,822 ภาพให้คลาสเล็กครบ 100, (ง) split 80/20 ของตัวเอง ผลตรวจเทียบกับ index ของเรา (`scripts/make_split_v2.py`):

| รายการ | ผลตรวจ | ใช้หรือไม่ |
|---|---|---|
| ย้าย label 629 ภาพ (unique filename) — า→ว **416**, า→ใ **107**, า→จ 21, ต→ด 17, ด→ต 12, า→ร 9, า→ๆ 9, ต→ค 6, อื่น ๆ ≤ 4 | montage `reports/analysis/datav2_montage_a_w.png`: ภาพที่ย้าย า→ว **เป็น ว จริงทุกภาพ** (ห่วงล่างชัด) และ า→ใ เป็น ใ จริง; โมเดล v1 (F2/F16) เห็นด้วยกับ label ใหม่ 95–100 % ในกลุ่มเล็ก (ด↔ต, ต→ค, า→จ) แต่ทาย า 85–93 % ในกลุ่ม า→ว เพราะ **เรียน label ผิดมา** (416 ภาพในคลาส า) | ✅ ใช้ |
| ทิ้ง 175 ภาพ (174 จากคลาส า) | ขยะ/ตัวอื่นปน (ส, ญ, ๆ, noise) — ดู montage แถวล่าง | ✅ ใช้ |
| augmented 1,822 ภาพ (`aug_*`) | **570 ภาพมีต้นฉบับอยู่คนละ split** ใน split ของ v2 → val ของ v2 รั่ว; pipeline เรามี on-the-fly aug + synthetic fonts อยู่แล้ว | ❌ ไม่ใช้ |
| split 80/20 ของ v2 | ไม่มี dedup, ไม่มี document-disjoint | ❌ ไม่ใช้ — คง split_seed42 เดิม |

→ สร้าง `data/splits/split_seed42_v2.csv` = แถว/strat/doc เดิมทุกประการ แต่แก้ label 629 แถว + ตัด 175 แถว (60,117 → 59,942; val strat 12,027 → 11,991
มี 121 แถวถูกแก้ label; val doc 11,984 → 11,948 มี 150 แถวถูกแก้) รายการทั้งหมด: `reports/analysis/datav2_label_changes.csv`

**นัยสำคัญ**: ~11 % ของคลาส า (คลาสใหญ่อันดับ 3) เป็น ว/ใ ที่ label ผิด → นี่คือที่มาของ error อันดับ 2 ของทุกโมเดลใน §A (ว→า 17.6 % ของ ว)
และเป็นเพดาน top-1 ~98.4–98.6 บน val v1 ที่ทุก recipe ชนอยู่ การแก้ label จึงน่าจะให้ผลมากกว่าการเปลี่ยน recipe ใด ๆ ใน §F

### H-1 ผล v1 vs v2 — recipe F19 เดิมทุกอย่าง เปลี่ยนแค่ไฟล์ label (`scripts/eval_split.py`, raw ไม่ TTA)

| โมเดล (เทรนด้วย label) | วัดบน val **v1** (label เดิม) top-1 / bal / macro-F1 | วัดบน val **v2** (label แก้) top-1 / bal / macro-F1 |
|---|---:|---:|
| F19 strat, **v1** | 0.9838 / 0.9826 / 0.9790 | 0.9797 / 0.9815 / 0.9804 |
| F19 strat, **v2** | 0.9750 / 0.9826 / 0.9747 | **0.9879 / 0.9867 / 0.9847** |
| F19 doc, **v1** | 0.9822 / 0.9807 / 0.9745 | 0.9753 / 0.9745 / 0.9754 |
| F19 doc, **v2** | 0.9710 / 0.9763 / 0.9623 | **0.9859 / 0.9790 / 0.9733** |

3 seeds ของ F19 บน v2 (split เดียวกัน, seed 42/0/1): top-1 0.9879 / 0.9898 / 0.9893 → **0.9890 ± 0.0008**; balanced 0.9868 / 0.9886 / 0.9851 →
**0.9868 ± 0.0014** (v1 seed 42: 0.9839 / 0.9827) · ensemble soft-vote ทั้ง 3: 0.9899 / 0.9876 (ดีกว่าตัวเดี่ยวที่ดีสุดแค่ +0.01 top-1 → ไม่คุ้ม 3× inference;
ใช้ KD กลั่นลงตัวเดียวแทน — ดู §I) · `reports/analysis/ensemble_final.json`

ข้อสรุป H:
1. **ผลต่างระหว่าง v1/v2 สมมาตร ±0.8–0.9 จุด top-1** และเท่ากับสัดส่วนภาพที่ label เปลี่ยนใน val (~1 %): โมเดล v2 ทาย ว บนภาพ ว ที่ label เดิมบอกว่า า
   จึง "ผิด" ตาม label เดิม และกลับกัน — ไม่ใช่ว่าโมเดลไหนเรียนรู้รูปร่างได้ดีกว่า (balanced บน v1 val เท่ากันพอดี 0.9826)
2. เมื่อวัดด้วย label ที่ถูก (v2) recipe เดิมได้ **98.79 → 98.90 ± 0.08 (3 seeds)** ทะลุเพดาน ~98.5 ที่ทุก recipe ใน §F ชนอยู่ → label noise คือคอขวดจริงของงานนี้
   ไม่ใช่ architecture/augmentation
3. **ความเสี่ยงต่อ hidden test**: mislabel กระจุกในแหล่ง `be` (694/804 ของการเปลี่ยน, 19 เอกสาร, 11–20 % ของ า ต่อเอกสาร) → ถ้า test set ของอาจารย์
   มาจากแหล่งเดียวกันและ label ด้วยกระบวนการเดิม โมเดล v2 จะเสีย ~0.9 จุดเทียบโมเดล v1 (และเทียบกลุ่มอื่นที่เทรน label เดิม); ถ้า test set label ถูก
   หรือมาจากแหล่งใหม่ โมเดล v2 ได้เปรียบ ~0.8 จุด → **ส่งมอบทั้งสองชุด**: `weights/thaichar72_resnet18_64.pt` (v2, ค่าเริ่มต้น) และ
   `weights/thaichar72_resnet18_64_v1labels.pt` (v1) เลือกตอนส่งได้

## I. Ensemble / Knowledge distillation และ deliverable (v2 split, stratified val 11,991 ภาพ)

| โมเดล | top-1 | balanced | macro-F1 | robust mean | inference cost |
|---|---:|---:|---:|---:|---|
| F19_v2 seed 42 | 0.9879 | 0.9868 | 0.9848 | 0.9052 | 1× |
| F19_v2 seed 0 / seed 1 | 0.9898 / 0.9893 | 0.9886 / 0.9851 | 0.9855 / 0.9856 | – | 1× |
| Ensemble soft-vote 3 seeds | 0.9899 | 0.9876 | 0.9867 | – | 3× |
| **K1 = KD student** (F19 recipe, teachers = 3 seeds, α 0.7, T 4, 20 ep) | **0.9903** | 0.9875 | **0.9869** | **0.9104** | 1× |
| K1 + TTA | 0.9893 | 0.9869 | – | – | 8× |

ข้อสรุป I: (1) ensemble ของ seeds ให้กำไรน้อย (+0.01–0.2 จุด) เพราะ error ที่เหลือ (า↔ๅ) เป็น systematic ไม่ใช่ variance; (2) **KD ถ่ายผล ensemble
ลงโมเดลเดียวได้ครบ** และยังทน corruption ดีขึ้น (+0.5 จุด mean) — soft target จาก 3 ครูทำหน้าที่เป็น regulariser; (3) TTA ยังลด top-1 แม้กับ K1 → ปิด
→ **ส่งมอบ K1** เป็น `weights/thaichar72_resnet18_64.pt` (fp32 44.9 MB) + `_fp16.pt` (22.5 MB, ผลเท่ากันทุกหลัก) + `_v1labels.pt` (F19 v1, สำรอง) +
`thaichar72_r18_synth_pretrain_init.pt` (stage-1 init สำหรับเทรนซ้ำ); `configs/final.yaml` = recipe F19 บน v2; notebook รันจบ 0 error ในโหมด inference
(K1 บน v2 val และ doc val ตาม split ที่ระบุใน checkpoint) — รายละเอียดใน `reports/FINAL-REPORT.md` §9, §12

## J. Extras — recipe ผู้ชนะบน backbone อื่น + "small model" (v2 split, randaug, 20 ep, ImageNet init เพราะ B6a init เป็น resnet18; `configs/extras/`)

| exp | backbone | params | KD จาก 3 ครู F19_v2 | top-1 | balanced | macro-F1 | robust mean | latency CPU bs=1 |
|---|---|---:|:-:|---:|---:|---:|---:|---:|
| **K1 (ส่งมอบ)** | resnet18 (B6a init) | 11.2 M | ✓ | **0.9903** | 0.9875 | 0.9869 | **0.9104** | 6.8 ms |
| X1 | efficientnet_b0 | 4.1 M | – | 0.9901 | **0.9886** | **0.9885** | 0.9095 | 6.4 ms |
| X2 | efficientnet_b0 | 4.1 M | ✓ | 0.9897 | 0.9865 | 0.9834 | 0.9070 | – |
| X3 | convnext_tiny | 27.9 M | – | 0.9898 | 0.9869 | 0.9841 | 0.9065 | – |
| X6 | mobilenetv3_large_100 | 4.3 M | – | 0.9893 | 0.9870 | 0.9864 | 0.9063 | 4.3 ms |
| X4 | mobilenetv3_large_100 | 4.3 M | ✓ | 0.9872 | 0.9840 | 0.9810 | 0.9043 | – |
| **X5 (small model)** | mobilenetv3_small_100 | **1.6 M** | ✓ | 0.9900 | 0.9872 | 0.9858 | 0.9053 | **3.4 ms** |

ข้อสรุป J:
1. **ทุก backbone ที่ใช้ recipe ผู้ชนะ (randaug + label v2) มาอยู่ที่ 98.7–99.0 และผ่าน robustness gate ทั้งหมด** → ที่ 64 px ตัวอักษรไบนารี
   backbone แทบไม่สำคัญ recipe (aug ที่มี noise, label ถูก, 20 epochs) สำคัญกว่า; convnext_tiny (28 M) ไม่ให้อะไรเพิ่มและช้ากว่า 3.5×
2. **KD จากครู resnet18 ช่วยเฉพาะ student resnet18** (K1 +0.24 vs seed 42): บน effb0 และ mnv3-large KD ทำให้แย่ลง (−0.04 / −0.21 top-1,
   best epoch เร็วขึ้น) — soft target ของครูต่างสถาปัตยกรรมขัดกับ inductive bias ของ student; ยกเว้น mnv3-**small** ที่เล็กมากจนได้ประโยชน์ (X5 0.9900)
3. **Small model**: X5 = 1.6 M params (7× เล็กกว่า resnet18), 6.5 MB, 3.4 ms/ภาพ, แพ้ K1 แค่ 0.03 top-1 และ 0.5 จุด robustness →
   ส่งมอบเป็น `weights/thaichar72_mnv3small_64_small.pt` สำหรับกรณีต้องการโมเดลเล็ก/เร็ว
4. effb0 (X1) เป็นทางเลือกที่ balanced/macro-F1 สูงสุด (98.86 / 98.85) ด้วย 4.1 M params แต่ช้ากว่า resnet18 ต่อ epoch 2.5× และคะแนน top-1/robust ต่ำกว่า K1 เล็กน้อย

---

## K. Generalisation study — font diversity + `dataUpdate` (2026-09-22, RTX 4070, `configs/gen/`)

บริบท: เจ้าของงานระบุเป้าหมายเป็น **generalisation ไม่ใช่ validation accuracy** และมีข้อมูลใหม่เข้ามาสองชุด
(`reports/03-DATA-AUDIT-2026-09-22.md`) ชุดที่มีค่าคือ `dataUpdate` — generator + 167 ฟอนต์ + ภาพ 98,071 ภาพ
ครอบคลุม 35 คลาสที่มีภาพจริงแค่ 1–155 ภาพ

**สำคัญ: ทุกตัวเลขในหัวข้อนี้เทียบกับ `F19_ctrl_v2_4070` เท่านั้น** ซึ่งคือ recipe F19 รันใหม่บน split ปัจจุบัน
(59,942 แถว, val 11,993) ห้ามเทียบกับ §F–§J ที่อยู่บน split เก่า (12,027) — ดู HANDOFF-2026-09-22 §2.2

### K.1 ข้อมูลที่ใช้

| source | rows | หมายเหตุ |
|---|---:|---|
| 203-font render | 72,000 | 1000/คลาส × 72; pool = repo 26 + dataUpdate 142 + Windows Thai 35 |
| dataUpdate (ฝั่ง train หลังแบ่งใหม่) | 54,114 | 35 คลาส in-universe |
| **stage-1 รวม** | **126,114** | 72 คลาส |

ก่อน render ได้ audit ฟอนต์ทั้ง 203 ตัวด้วย `scripts/audit_fonts.py` (cmap ของ fontTools + เทียบ bitmap กับ U+FFFF
เพื่อจับ `.notdef`) — **ผ่านครบ 203/203 ไม่ต้องตัดฟอนต์ตัวไหนทิ้ง** การเช็คนี้จำเป็นเพราะ `FontLibrary._render_standalone`
ตัดสินการรองรับ glyph จาก mask ที่ไม่ว่าง ซึ่งกล่อง `.notdef` ก็ไม่ว่างเหมือนกัน

### K.2 probe ที่ dataUpdate แถมมา **ใช้ไม่ได้ ต้องแบ่งใหม่**

dataUpdate แบ่ง train/val ของตัวเองแบบ**ราย*ภาพ*** บน render ที่มาจากต้นฉบับเดียวกัน ผลคือฝั่ง val ของมัน
ซ้ำกับต้นฉบับในฝั่ง train **10,181 / 10,268 แถว = 99.2 %** (ft 7,275/7,275 = 100 %, hw 2,906/2,993 = 97.1 %)
เหลือแถวที่สะอาดจริงแค่ 87 แถวใน 12 คลาส — เป็นความผิดพลาดแบบเดียวกับไฟล์ `aug_` ของ DataV2 (augment ก่อน split)
ที่โปรเจกต์ปฏิเสธไปแล้วใน §1 ของ data audit

`scripts/make_dataupdate_probe.py` แบ่งใหม่แบบ **group-disjoint**: ฝั่ง ft กันทั้ง**ตระกูลฟอนต์** (5/31 ตระกูล),
ฝั่ง hw กันทั้ง**ลายมือ/หน้ากระดาษ** (171/1,141 id) → train 54,114 / probe 13,957, group ซ้ำกัน 0,
ทุกคลาสมีอย่างน้อย 363 ภาพ ตรวจเพิ่มว่าลายมือของ dataUpdate ไม่ซ้ำกับ round2 (ชื่อไฟล์ซ้ำ 0) → ไม่รั่วเข้า val จริง

**ข้อจำกัดที่ต้องระบุ:** ตระกูลฟอนต์ 5 ตัวที่กันไว้ยังอยู่ใน 203 ฟอนต์ที่ใช้ render stage-1 (จงใจ ไม่อยากทิ้ง Sarabun
ซึ่งเป็นฟอนต์เอกสารไทยมาตรฐานออกจาก pretraining) ดังนั้น **ฝั่ง ft ของ probe อ่อนกว่าฝั่ง hw** และต้องรายงานแยกเสมอ

### K.3 stage-1 — zero-shot synthetic → real

เทรนบน synthetic ล้วน (`use_real_train: false`) วัดบน real val:

| run | stage-1 data | top-1 | balanced | minority | best ep |
|---|---|---:|---:|---:|---:|
| G1 | 72k render (203 ฟอนต์) | 0.9395 | 0.9276 | 0.9500 | 3/8 |
| G2 | 126k (+ dataUpdate) | 0.9396 | **0.9354** | **0.9667** | 4/8 |

top-1 เท่ากัน แต่ G2 ชนะ balanced +0.78 จุดและ minority +1.67 จุด → dataUpdate ช่วย**หาง** ไม่ใช่หัว
**ไม่เคยเห็นภาพจริงเลยแต่ได้ 94 %** เป็นหลักฐานว่า font-diversity pretraining ทำงาน

### K.4 stage-2 — ผลลัพธ์หลัก

| run | stage-1 init | strat top-1 | **doc top-1** | doc bal | **robust gate** | **probe hw** |
|---|---|---:|---:|---:|---:|---:|
| F19_ctrl_v2_4070 (baseline) | B6a, 26 ฟอนต์ | 0.9902 | 0.9856 | 0.9824 | 0.9124 ✅ | 0.2258 |
| G3 | G1, 203 ฟอนต์ | 0.9901 | 0.9869 | 0.9836 | 0.9119 ✅ | 0.2335 |
| **G4 (ผู้ชนะ)** | G2, 203 + dataUpdate | 0.9902 | **0.9872** | 0.9813 | **0.9153** ✅ | **0.4292** |
| G5 | G2 + tail-fill 500 ตอน fine-tune | 0.9897 | 0.9869 | **0.9837** | 0.9041 ✅ | (เทียบไม่ได้) |
| K2 (KD จาก 3 ครู G4) | G2 | 0.9901 | 0.9865 | 0.9827 | 0.9152 ✅ | 0.4092 |

gate = mean top-1 ของแถว non-clean ทั้งหมดใน `scripts/robustness.py --binarize`, เกณฑ์ ≥ 0.8988 (F2 0.9038 − 0.005);
`F19_ctrl` บน split ปัจจุบันได้ 0.9124 จึงใช้เป็นจุดอ้างอิงบน split เดียวกันได้โดยไม่ต้องพึ่งเลขเก่า

3 seeds ของ G4: **0.9899 ± 0.0004** (bal 0.9934 ± 0.0001) — ensemble 3 ตัว 0.9905

### K.5 ข้อสรุป K

1. **บน stratified val ทั้งหมดเท่ากัน** (0.9897–0.9902 ต่างกัน ≈ 2 ภาพจาก 11,993) ตัวชี้วัดนี้อิ่มตัวและ**แยกอะไรไม่ได้อีกแล้ว**
   ยืนยันกฎของโปรเจกต์ที่ให้เลือกโมเดลจาก doc split + gate
2. **G4 ชนะบนเกณฑ์ที่ใช้เลือกจริงทั้งสองตัว** (doc top-1 สูงสุด, gate สูงสุด) ระยะห่างจาก baseline บน doc คือ +0.16 จุด
   ซึ่ง**ใหญ่กว่าความแปรปรวนระหว่าง seed (±0.04 จุด)** ส่วนระยะห่างบน strat (0.00 จุด) จมอยู่ใน noise
3. **font diversity อย่างเดียวแทบไม่ช่วย** — G3 (203 ฟอนต์) ≈ F19 (26 ฟอนต์) ทั้ง gate และ probe
   สิ่งที่ช่วยคือ **dataUpdate ใน stage-1** (G4): probe hw 0.2258 → 0.4292 เกือบสองเท่า โดย strat val ไม่ขยับเลย
4. **probe เปิดเผยสิ่งที่ strat val ซ่อนไว้**: baseline ที่ดูเหมือน 99 % เหลือ **22.6 %** บน 35 คลาสหางกับลายมือที่ไม่เคยเห็น
   ฃ และ ฑ (ภาพจริงคลาสละ 1 ภาพ, val 0 ภาพ) ได้ recall **0.000** — สองคลาสนี้ **วัดบนข้อมูลจริงไม่ได้เลย**
5. **สาเหตุของตัวเลข probe ที่ต่ำยังสรุปไม่ได้** ทดสอบกับทั้ง 34 คลาสแล้วไม่มีตัวแปรไหนมีนัยสำคัญ:
   spearman(recall, จำนวนภาพจริง) = +0.299 (p=0.086), spearman(recall, |aspect gap| probe vs real) = −0.192 (p=0.28),
   |ink gap| = −0.045 (p=0.80) ทิศทางถูกทั้งคู่แต่ 34 คลาสน้อยเกินไป → **probe เป็น domain proxy
   ใช้เทียบอันดับระหว่างโมเดลที่ fine-tune เหมือนกันได้ แต่อย่าอ่านเป็นคะแนนที่จะได้บน hidden test**
6. **G5 (เติม synthetic ตอน fine-tune) ไม่ควรส่งมอบ**: ได้ doc balanced สูงสุดจริง แต่ตกที่ gate ต่ำสุด (0.9041)
   และ macro-F1 บน doc ต่ำกว่าชาวบ้านชัดเจน (0.9703 vs 0.9798) เลข probe 0.9180 ของมัน **เทียบกับตัวอื่นไม่ได้**
   เพราะมันเป็นตัวเดียวที่เทรนบน domain ของ dataUpdate ตอน fine-tune → probe เป็น in-domain สำหรับมัน
   ยืนยันผลเดิมของโปรเจกต์: synthetic เหมาะเป็น pretraining มากกว่าเป็น extra data
7. **KD รอบนี้ไม่ช่วย** ต่างจาก K1: K2 แพ้ G4 ตัวเดียวทั้ง doc (−0.07 จุด) และ probe (−2.0 จุด) เสมอที่ gate
   น่าจะเพราะ 3 seeds ของ G4 เหมือนกันเกินไป (±0.0004) ensemble ได้แค่ +0.06 จุด ไม่พอให้กลั่นออกมา
   ข้อดีคือ KD **ไม่ทิ้ง**คุณสมบัติเรื่องหาง (probe 0.4092 จาก 0.4292 = เก็บไว้ 95 %) → คุณสมบัตินี้ติดมากับ stage-1 init จริง
8. **ส่งมอบ G4 (seed 42)** เป็น `weights/thaichar72_r18_64_gen.pt` และ stage-1 init เป็น
   `weights/thaichar72_r18_synth203du_pretrain_init.pt` เพื่อให้ทำซ้ำได้โดยไม่ต้อง render 203 ฟอนต์ใหม่

---

## L. Selective tail-fill — เติมข้อมูลเฉพาะคลาสที่ขาดจริง (2026-09-22, ข้อเสนอของเจ้าของงาน)

§K สรุปว่า G5 (เติม dataUpdate ตอน fine-tune ให้ทุกคลาสถึง 500) แย่ลง เจ้าของงานเสนอว่าควร
**เลือกเติมเฉพาะคลาสที่น้อยจริง** เพราะ dataUpdate คุณภาพสู้ข้อมูลจริงไม่ได้ หัวข้อนี้ทดสอบข้อเสนอนั้น

### L.1 ทำไม G5 ถึงพัง — เป็นเรื่องปริมาณ ไม่ใช่เรื่องข้อมูล

`extra_fill_to: 500` โดยไม่จำกัดว่าคลาสไหนควรเติม ทำให้:

| | |
|---|---|
| ภาพสังเคราะห์ที่เติมเข้าไป | **15,952** (train 47,949 → 63,901) |
| คลาสที่สังเคราะห์กลายเป็น > 50 % ของคลาส | **35 / 35** |
| คลาสที่สังเคราะห์กลายเป็น > 90 % ของคลาส | **24 / 35** |
| มัธยฐานสัดส่วนสังเคราะห์ | **93.8 %** |

ตัวอย่างที่ชัดที่สุด: ุ มีภาพจริง 124 ภาพ ซึ่งไม่ได้ขาดแคลน แต่โดนเติมสังเคราะห์ 376 ภาพ (75.2 %)
คลาสแบบนี้ไม่ได้ต้องการความช่วยเหลือ แต่ distribution ถูกกลบจนเสียความทนทาน

ข้อสังเกตสำคัญ: **dataUpdate ตรงเป้าอยู่แล้ว** — มันครอบคลุม 35 จาก 36 คลาสที่มี n_train ≤ 155 และ
**ไม่แตะคลาสที่มีข้อมูลเยอะเลยสักคลาส** ปัญหาจึงอยู่ที่ `extra_fill_to` ล้วนๆ

### L.2 knob ใหม่ `extra_fill_max_real`

เพิ่มใน `src/thaichar/engine.py::_select_extra` (documented ใน `configs/base.yaml`): เติมเฉพาะคลาสที่
**จำนวนภาพจริง ≤ ค่านี้** ค่า `null` = พฤติกรรมเดิม

| variant | เงื่อนไข | คลาสที่แตะ | ภาพที่เติม |
|---|---|---:|---:|
| G5 | fill 500, ไม่จำกัด | 35 | 15,952 |
| **G6** | n_train ≤ 20, fill 200 | 14 | 2,665 |
| **G7** | n_train ≤ 50, fill 150 | 24 | 3,119 |

### L.3 ผล

| run | strat top-1 | strat bal | doc top-1 | doc bal | doc macro-F1 | gate | probe hw |
|---|---:|---:|---:|---:|---:|---:|---:|
| F19_ctrl (baseline) | 0.9902 | 0.9935 | 0.9856 | 0.9824 | 0.9795 | 0.9124 ✅ | 0.2258 |
| **G4 (ส่งมอบ)** | 0.9902 | 0.9934 | 0.9872 | 0.9813 | **0.9798** | **0.9153** ✅ | 0.4292 |
| G5 (fill 500, ทุกคลาส) | 0.9897 | 0.9901 | 0.9869 | 0.9837 | 0.9703 | 0.9041 ✅ | 0.9180 |
| G6 (≤20 → 200) | **0.9905** | **0.9939** | 0.9867 | 0.9832 | 0.9726 | 0.9136 ✅ | 0.5975 |
| G7 (≤50 → 150) | 0.9903 | 0.9927 | 0.9873 | 0.9874 | 0.9696 | 0.9134 ✅ | 0.7163 |

**ข้อเสนอของเจ้าของงานแก้ปัญหาของ G5 ได้จริง**: gate กลับมาที่ระดับปกติ (0.9041 → 0.9134/0.9136)
และ G6 ได้ strat top-1/balanced สูงสุดของทุก run ในโปรเจกต์ ส่วน probe ไต่ขึ้นตามปริมาณที่เติม
(G4 0.43 → G6 0.60 → G7 0.72)

### L.4 แต่ยืนยันทางสถิติไม่ได้ — 3 seeds บน doc split

รอบแรกอ่านจาก run เดียวแล้วเห็น G7 ชนะ doc balanced +0.61 จุด **ซึ่งเป็นการอ่านที่ผิด** —
ตัวเลขนั้นคือ seed ที่ดีที่สุดของ G7 ไปเจอ seed กลางๆ ของ G4 พอดี รัน 3 seeds ทั้งคู่แล้วได้:

| metric | G4 (n=3) | G7 (n=3) | ต่าง | Welch p |
|---|---:|---:|---:|---:|
| doc top-1 | 0.9853 ± 0.0017 | 0.9858 ± 0.0014 | +0.05 จุด | 0.72 |
| doc balanced | 0.9805 ± 0.0024 | 0.9838 ± 0.0031 | +0.33 จุด | **0.23** |
| doc macro-F1 | 0.9780 ± 0.0044 | 0.9661 ± 0.0072 | **−1.19 จุด** | **0.085** |

**ฝั่งกำไรไม่มีนัยสำคัญ ฝั่งขาดทุนใกล้มีนัยสำคัญกว่า** การ oversample หางทำให้ recall ขึ้นแต่ precision ลง:
บน 24 คลาสที่เติม G4 ได้ recall 0.9756 / precision 0.9836 ส่วน G7 ได้ 0.9919 / 0.9173 — คิดเป็นจำนวนภาพคือ
**จับถูกเพิ่ม 2 ภาพ แต่ทายผิดเพิ่ม 9 ภาพ**

### L.5 ทำไมวัดไม่ได้ — ข้อจำกัดเชิงโครงสร้าง

24 คลาสที่ G7 เติม มีภาพใน **doc val รวมกันแค่ 123 ภาพ**:

| | |
|---|---|
| คลาสที่มี doc-val = 0 ภาพ | **8** (ฃ ฑ ฤ ฬ ฯ ฮ ๔ ๕) |
| คลาสที่มี ≤ 3 ภาพ | 13 |

ภาพเดียวพลิกในคลาสที่มี val 3 ภาพ ดัน balanced accuracy ของทั้ง 72 คลาสได้ **0.46 จุด** ดังนั้น
doc balanced accuracy **ไม่มีอำนาจจำแนกในระดับความต่างที่เรากำลังวัด**

รวมกับที่พบใน §K ทำให้เครื่องมือวัดตันทุกทาง:

| เครื่องมือ | ทำไมใช้ตัดสินไม่ได้ |
|---|---|
| stratified val | อิ่มตัวที่ 99 % ทุก candidate ต่างกัน ~2 ภาพ |
| doc balanced | ขับเคลื่อนด้วยภาพหลักหน่วยในคลาสหาง |
| tail probe | เป็น synthetic มี domain confound (และ G5/G6/G7 เห็น domain นั้นตอนเทรน) |
| 8 คลาสที่ขาดที่สุด | ไม่มีภาพ validation จริงเลย |

### L.6 ข้อสรุป L

1. **ข้อเสนอเลือกเติมเฉพาะคลาสที่ขาดถูกต้องและควรเก็บไว้** — `extra_fill_max_real` แก้ปัญหาที่ทำให้ G5 พัง
   และ G6 ให้ตัวเลข stratified ดีที่สุดในโปรเจกต์ (0.9905 / 0.9939 / macro-F1 0.9914)
2. **แต่คงส่งมอบ G4 ต่อไป** — ไม่ใช่เพราะ G6/G7 แย่ แต่เพราะมันไม่ได้แสดงว่าดีกว่า**บนสิ่งที่วัดได้จริง**
   (p=0.23) ขณะที่ต้นทุนด้าน precision วัดได้ชัดกว่า (p=0.085) การเปลี่ยนตัวส่งมอบไปหาสิ่งที่พิสูจน์ไม่ได้ ไม่คุ้ม
3. **ถ้าอาจารย์ยืนยันว่า hidden test เป็น class-balanced ให้กลับมาใช้ G6** — ภายใต้สมมติฐานนั้น recall
   ของคลาสหางคือสิ่งที่ถูกให้คะแนน และ G6 คือตัวที่สมดุลที่สุดระหว่างกำไร recall กับต้นทุน precision
   config พร้อมใช้อยู่แล้วที่ `configs/gen/G6_r18_g2init_randaug_fill200_max20.yaml`
4. **สิ่งที่ขาดจริงๆ ไม่ใช่ข้อมูลเทรน แต่คือข้อมูล validation ของคลาสหาง** ถ้าจะลงแรงต่อ การไป label
   ภาพจริงเพิ่มสัก 20–30 ภาพต่อคลาสสำหรับ 14 คลาสที่ขาดที่สุด (~400 ภาพ) จะให้ประโยชน์มากกว่า
   การ generate สังเคราะห์เพิ่ม เพราะมันปลดล็อกความสามารถในการ**วัด** ซึ่งตอนนี้เป็นคอขวดจริง

---

### L.7 ตรวจข้อสงสัยเรื่อง "ภาพหมุน" — ไม่พบปัญหา

เจ้าของงานสงสัยว่าบางคลาสอาจมีภาพที่หมุนอยู่ วัดด้วยแกนหลักจาก image moments แล้วสรุปด้วย **สถิติเชิงวงกลมบน 2θ**
(ทิศทางนิยาม mod 180°) และนับเฉพาะ glyph ที่ eccentricity > 0.6 เพราะ glyph ที่เกือบกลมไม่มีแกนที่มีความหมาย:

| | |
|---|---|
| มัธยฐาน R ทั้ง corpus (R = 1 คือเรียงตรงกันสมบูรณ์) | **0.97** |
| คลาสที่ R < 0.5 | **1 จาก 57** (ผ) |

ตรวจ ผ ด้วยตาจาก montage ของภาพที่มุมสุดขั้วทั้งสองฝั่ง: **ตั้งตรงทุกภาพ** ค่ามุมที่ต่างกัน (−89° กับ 3°) มาจาก
สัดส่วนกว้าง/สูงของลายมือ (เขียนสูงกว่ากว้าง → แกนแนวตั้ง, เขียนแบน → แกนแนวนอน) ไม่ใช่การหมุน
**สรุป: ไม่มีปัญหาภาพหมุนในชุดข้อมูล ไม่ต้องแก้อะไร**

บันทึกวิธีวัดไว้กันพลาดซ้ำ: การวัดรอบแรกใช้มุมดิบจาก moments เทียบกับมัธยฐานรายคลาส แล้วได้ผลลวงว่า ย มีภาพ
"เอียง" 83 % — เกิดจากมุมวน ±90° บวกกับแกนหลักที่ไม่นิยามสำหรับ glyph กลม ตัวเลขชุดนั้นใช้ไม่ได้และถูกทิ้ง


## M. Label noise — หาด้วย out-of-fold, ยืนยันด้วยตา, แล้ววัดว่าคุ้มไหม (2026-09-22)

เจ้าของงานตั้งข้อสังเกตว่าชุดข้อมูลยังมี label ผิดและภาพปนกัน (`example/`) หัวข้อนี้หาให้เจอ แก้ แล้ว
**วัดว่าการแก้ให้ผลจริงหรือไม่**

### M.1 วิธีหา — out-of-fold ครอบคลุม 100 %

ความเห็นของโมเดลต่อภาพที่มันเทรนมาแล้วใช้ไม่ได้ และ split strat/doc ครอบคลุมอย่างละ ~20 % แถมทับกัน
จึงแบ่ง 5 folds (`scripts/make_folds.py`) เทรน 5 โมเดล → **ทุกภาพใน 59,942 ภาพได้คำทำนายจากโมเดลที่ไม่เคยเห็นมัน**

| | |
|---|---:|
| OOF accuracy | 0.9882 |
| ไม่ตรงกับ label | 709 |
| โมเดลมั่นใจ ≥ 0.70 และไม่ตรง | **444** (0.74 % ของชุด) |

`scripts/find_label_noise.py` จัดอันดับตามส่วนต่างความเชื่อ แล้ว `scripts/compare_pair.py` เรนเดอร์
แผ่นเทียบสามแถว (ตัวอย่างคลาส A ที่ไม่มีข้อสงสัย / ภาพที่ถูกสงสัย / ตัวอย่างคลาส B) เพราะ montage
ของภาพต้องสงสัยล้วนๆ ถามว่า "นี่คือ A หรือ B" โดยไม่มีอะไรให้เทียบ

### M.2 สิ่งที่คนยืนยัน

เจ้าของงานตรวจครบทุกคู่ และยืนยัน **10 คู่ = 146 ภาพ** ว่าเฉลยผิดจริง:

```
ั→้ 40   ด→ต 24   ช→ซ 18   ้→ั 14   ท→พ 10
บ→น  9   ซ→ช  8   ร→ว  8   ว→ร  8   น→ม  7
```

**ตัด า↔ๅ (122 ภาพ) ออก** ซึ่งเป็นการตัดสินใจที่ถูก: มันเป็นก้อนใหญ่ที่สุดแต่ความมั่นใจต่ำสุด (0.79)
รูปร่างสองตัวนี้ใกล้กันโดยธรรมชาติ และโปรเจกต์เคยพบว่าคู่นี้เป็น **convention ที่ผิดอย่างเป็นระบบ**
(416 ภาพ ว ถูกยื่นใต้ า) การ "แก้" อาจสวนทางกับเฉลยของอาจารย์

10 คู่ที่ยืนยันผิด**ทั้งสองทาง** (ช→ซ และ ซ→ช, ร→ว และ ว→ร, ั→้ และ ้→ั) ซึ่งเป็นลายเซ็นของ
**การพลาดแบบสุ่มตอนติดป้าย ไม่ใช่ convention** → ไม่มีความสัมพันธ์กับ hidden test → แก้แล้วได้เปล่าๆ

หลักฐานอิสระ: DataV2 (การ relabel โดยเพื่อนร่วมทีม) จับได้แค่ **2 จาก 146 ภาพ** → 144 ภาพนี้ไม่เคยมีใครเจอ

### M.3 แก้แล้วได้อะไร — **ไม่ได้อะไร**

`scripts/apply_label_fixes.py` → `data/splits/split_seed42_v3.csv` (146 ภาพ, audit ครบ)
G8 = recipe เดียวกับ G4 เทรนบน v3

การวัดต้องระวัง: 146 ภาพแบ่งเป็น **120 ในชุดเทรน / 26 ใน strat val (39 ใน doc val)** ถ้าวัดแบบปกติ
โมเดล v3 จะชนะโดยอัตโนมัติเพราะเฉลยถูกแก้ให้ตรงกับที่มันเรียนมา จึงวัดโดย**ตัดภาพที่แก้ออก** และให้
ทั้งสองโมเดลใช้เฉลยชุดเดียวกัน (`scripts/eval_excluding.py`) 3 seeds ต่อฝั่ง:

| metric (doc, v3 val − 39 แถวที่แก้) | G4 (label v2) | G8 (label v3) | ต่าง | p |
|---|---:|---:|---:|---:|
| doc top-1 | 0.9884 ± 0.0017 | 0.9874 ± 0.0030 | −0.10 จุด | 0.648 |
| doc balanced | 0.9824 ± 0.0025 | 0.9851 ± 0.0024 | +0.27 จุด | 0.241 |
| doc minority | 0.9778 ± 0.0084 | 0.9768 ± 0.0070 | −0.09 จุด | 0.889 |

**ไม่ต่างกันทั้งสองทาง** เหตุผลตรงไปตรงมา: แก้ 120 ภาพจาก 47,949 = **0.25 % ของชุดเทรน** น้อยเกินกว่า
จะขยับอะไรได้

> หมายเหตุวิธีการ: การวัดรอบแรกด้วย seed เดียวให้ภาพว่า v3 แย่กว่า (doc 0.9844 vs 0.9903) ซึ่ง**ผิด** —
> เป็น noise ของ seed เดียว (doc seed std = ±0.0017–0.0030) ตัวเลขข้างบนคือ 3 seeds ต่อฝั่ง

### M.4 ข้อสรุป M

1. **label noise ไม่ใช่คอขวด** ก่อนหน้านี้เป็นสมมติฐาน ตอนนี้วัดแล้ว การลงแรงเพิ่มกับการ clean
   ไม่คุ้ม — ต่างจากการลงแรงกับ **การวัด** (§L.5: 8 คลาสไม่มีภาพ validation เลย)
2. **แต่ยังควรแก้** เพราะชุดข้อมูลถูกต้องขึ้นจริง มี audit trail และเป็นเนื้อหารายงานที่ดี — แค่อย่าคาดหวังคะแนน
3. **อย่าทำซ้ำเกินหนึ่งรอบ** หลังแก้แล้วเทรนใหม่จะมีผู้ต้องสงสัยชุดใหม่โผล่มาเสมอ ไล่แก้ไปเรื่อยๆ คือการ
   ปรับ label ให้ตรงกับที่โมเดลคิด = overfit ชุดข้อมูลตัวเอง
4. **ส่งทั้ง v2 และ v3 ไปวันงาน** เราไม่รู้ว่าเฉลยของอาจารย์ใช้ convention ไหน และรูปแบบ hackathon
   อนุญาตให้ลองหลายตัว → `scripts/evaluate_folder.py` รันทั้งหมดแล้วจัดอันดับ ให้ข้อมูลจริงตัดสินแทนการเดา
5. **สิ่งที่ geometric screening หาเจอ (น้อยมาก)**: crop สองตัวอักษร 1 ภาพ, ภาพว่าง 0 ภาพ, "ก้อนดำทึบ"
   60 ภาพเป็นคลาส ่ ทั้งหมดและเป็นของจริง (3×7 ถึง 8×10 พิกเซล) — **โมเดลจำคลาสนี้จากขนาดไม่ใช่รูปทรง
   ซึ่งเป็นความเสี่ยงถ้าข้อมูลหน้างานละเอียดกว่า**

---

## N. Stress suite — พฤติกรรมบน input ที่ไม่เคยเทรน (2026-09-22, `scripts/stress_suite.py`)

สเปก: `tasks/TASK-14-stress-suite.md` เจ้าของงานอยากรู้ก่อนวันส่งว่าโมเดลทำตัวยังไงกับตัวอักษรติดกัน ครึ่งตัว
ซ้อนทับ หมุน กลับหัว กระจก ไม่ใช่ตัวอักษร ฟอนต์ไม่เคยเห็น pixelate และ blur — ซึ่ง `scripts/robustness.py`
เดิมไม่ครอบคลุมเลยสักอย่าง

วัดทั้ง **5 weight ที่ packaged ไว้** บน 493 ภาพ val ที่กันไว้ (8 ภาพ/คลาส seed 0) ใช้ชุดฐานเดียวกันทุกเงื่อนไข
ทุกโมเดล รันครบใน **28 วินาที** (งบที่สเปกให้คือ 10 นาที)

**ทั้งสามกลุ่มให้คะแนนคนละแบบโดยตั้งใจ** — กลุ่ม B ไม่มีคำตอบที่ถูก การรายงาน accuracy กับภาพตัวอักษรสองตัว
ติดกันจึงไม่มีความหมาย สิ่งที่วัดแทนคือโมเดล*ส่งสัญญาณ*ไหมว่ามันเจอของนอกขอบเขต

### N.0 บั๊กที่เจอระหว่างสร้าง harness — เส้นทาง inference จริงแย่กว่าที่รายงาน 2 จุด

Acceptance ข้อ 2 ของสเปกบังคับว่า clean baseline ต้องตรงกับ val ที่รู้ภายใน 0.5 จุด **รอบแรกไม่ผ่าน**: ได้ 0.9695
เทียบกับ 0.9902 พอเทียบสองเส้นทางบนภาพชุดเดียวกัน 492 ภาพ:

| เส้นทาง | top-1 |
|---|---:|
| `ThaiGlyphDataset` (ที่ใช้ตอน validate) | **0.9898** |
| `infer.preprocess_image` (ที่ใช้ตอน deploy จริง) | **0.9695** |

ต่างกัน 10 ภาพ และ **deployed ผิดทั้ง 10 ถูก 0** ไม่ใช่ความบังเอิญของการสุ่ม

สาเหตุอยู่ที่ `infer._reframe`: มันเติมขอบภาพด้วย **median ของสี 4 มุม** แต่ glyph ที่ crop ชิดจนเส้นแตะมุม
จะได้ median เป็น 0 หรือ 127 → เติมขอบด้วยสีเข้ม → Otsu อ่านว่าขอบคือพื้นหลัง → **กลับขั้วทั้งภาพ** จนพัง
และเพราะการเดาผิดอยู่ที่ *ค่าที่เอาไปเติม* ไม่ใช่ที่ขั้วของ input การกลับภาพเข้าไปใหม่จึงไม่ช่วย —
`--polarity both` ของ `evaluate_folder.py` จึงใช้ไม่ได้ผลด้วย (ทั้งสองขั้วเละเหมือนกัน)

แก้เป็น **ค่าที่สว่างที่สุดของ 4 มุม** วัดผลบน 493 ภาพเดิม:

| กฎ | tight crop | ขอบขาว | ขอบดำ | กลับขั้ว |
|---|---:|---:|---:|---:|
| median, ขั้วเดียว (ของเดิม) | 0.9695 | 0.9695 | 0.9695 | 0.9715 |
| median + polarity both | 0.9715 | 0.9715 | 0.9715 | 0.9715 |
| **max, ขั้วเดียว** | **0.9898** | **0.9898** | **0.9898** | 0.7439 |
| **max + polarity both** | 0.9858 | 0.9858 | 0.9858 | **0.9858** |

`max` ทำให้ตรงกับ dataset transform เป๊ะ และทำให้ `--polarity both` **เริ่มทำงานได้จริง** เพราะตอนนี้มีขั้วหนึ่ง
ที่เฟรมถูกเสมอ → ทนทุกรูปแบบการเฟรมที่ 0.9858 แก้แล้วที่ `src/thaichar/infer.py` และ `scripts/evaluate_folder.py`
(pytest 63/63 ผ่าน) suite ใช้กฎ polarity-both เหมือน `evaluate_folder.py` เพราะนั่นคือสิ่งที่จะรันจริงวันส่งงาน
ส่วนต่าง 0.44 จุดที่เหลือจาก 0.9902 คือต้นทุนของ polarity-both ซึ่งแลกมากับความทนทาน

### N.1 กลุ่ม A — ภาพเสียแต่คำตอบยังอยู่ (retention เทียบ clean ของตัวเอง)

จุดที่ retention ตกต่ำกว่า 90 % และ 50 % (โมเดลที่ส่งมอบ):

| corruption | < 90 % ที่ | < 50 % ที่ | อ่านผล |
|---|---|---|---|
| jpeg (q 40→5) | ไม่เคย | ไม่เคย | Otsu ล้าง artifact ทิ้งหมด — บีบอัดแรงแค่ไหนก็ไม่กระทบ |
| resolution_up (2–8×) | ไม่เคย | ไม่เคย | สแกนละเอียดกว่า round2 ไม่เป็นปัญหา |
| aspect_stretch (0.7–1.4) | ไม่เคย | ไม่เคย | resize ไม่รักษาสัดส่วนเสียแค่ ~5 % ตอนบีบแคบ |
| motion_blur | **7 px** | ไม่เคย | ยังเหลือ 0.65 ที่ 9 px |
| pixelate | **0.35** | **0.25** | เหลือ **0.10 ที่ 0.15** — หน้าผาที่ชันที่สุด |
| stroke_extreme | **erode 4** | **erode 4** | **ไม่สมมาตร**: ขยายเส้น +5 ยังได้ 0.93 แต่กัดเส้น −4 เหลือ 0.25 |
| rotate_hard | **30°** | **45°** | 0.53 ที่ 30°, 0.10 ที่ 45°, 0.03 ที่ 90° |
| unseen_font (62 typeface) | – | ไม่เคย | **0.868 สำหรับตัวที่ส่งมอบ** |

**unseen_font เป็นการพิสูจน์งาน font diversity โดยตรง** — เป็นการทดสอบบน 62 typeface ที่ stage-1 ไม่เคย render
(5 ตระกูลที่กันไว้จาก probe + 24 face ใน `angsana/browalia/cordia.ttc` ที่ `synth.py` ข้ามเพราะ glob แค่ `*.ttf`)
render ผ่าน `degrade_glyph` ด้วยความสูงมัธยฐานจริงรายคลาส เพื่อให้ **ต่างจากข้อมูลเทรนแค่ typeface อย่างเดียว**:

| โมเดล | retention บนฟอนต์ที่ไม่เคยเห็น |
|---|---:|
| `r18_64_gen_v3labels` | **0.874** |
| `r18_64_gen` (ส่งมอบ) | 0.868 |
| `resnet18_64` (K1, init 26 ฟอนต์) | 0.834 |
| `resnet18_64_v1labels` | 0.827 |
| `mnv3small_64_small` | 0.774 |

โมเดลสาย G ที่ pretrain ด้วย 203 ฟอนต์ + dataUpdate **ชนะสาย K1 ที่ใช้ 26 ฟอนต์ 3.4 จุด** บนตัวชี้วัดที่วัด
generalisation ข้าม typeface โดยตรง

### N.2 กลุ่ม B — input ที่ไม่มีคำตอบถูก (ไม่รายงาน accuracy เลย)

`separability AUROC` = แยก clean ออกจาก junk ด้วยค่าความมั่นใจอย่างเดียวได้แค่ไหน สเปกตั้งไว้ว่า
**≥ 0.8 คือ threshold กรองได้ ≤ 0.6 คือกรองไม่ได้ ต้องไปแก้ที่ input**

**ทุกเงื่อนไข ทุกโมเดล ได้ ≥ 0.84** (ต่ำสุด `overlap_2_50` 0.844–0.862, ส่วนใหญ่ ≥ 0.9, `strokes`/`near_empty`
สูงถึง 0.98–0.998) → **ไม่ต้องสร้างตัวแยกตัวอักษรติดกันก่อน classify** threshold ความมั่นใจไม่กี่บรรทัดก็พอ

junk ที่อันตรายที่สุดคือ **ตัวอักษรซ้อนกัน 50 %**: ยังได้ prediction ที่มั่นใจ ≥ 0.9 ถึง **22 %** ของภาพ
รองมาคือครึ่งตัว (9–16 %) ส่วนรอยขีดมั่ว ภาพเกือบว่าง และตัวอักษรสามตัวเรียง โมเดลไม่มั่นใจแทบทั้งหมด (≤ 1 %)

### N.3 กลุ่ม C — การแปลงอาจเปลี่ยนคำตอบ (เป็นตาราง mapping ไม่ใช่คะแนน)

ความมั่นใจเฉลี่ยตกเหลือ 0.36–0.59 ทุกการแปลง (clean ~0.9) → batch ที่กลับด้านทั้งชุด**ตรวจจับได้จากภาพรวม**
แต่มี **127 คู่ (คลาส × การแปลง) ที่กลายเป็นตัวอักษรไทยตัวอื่นแบบมั่นใจ** ซึ่งจะพลาดแบบเงียบๆ
ที่ทั้ง 5 โมเดลเห็นตรงกัน:

| การแปลง | ตัวอย่างที่กลายเป็นตัวอื่นอย่างมั่นใจ |
|---|---|
| mirror_v (กระจกแนวตั้ง) | **บ↔ภ**, **ย↔ถ** (คู่กลับกันพอดีทั้งสองทาง), ท→ม, ล→ย, ห→ม, ๅ→่ |
| mirror_h (กระจกแนวนอน) | า→ก, ๅ→ก, แ→ม |
| rot180 (กลับหัว) | ว→ย, ิ→ั |
| rot90 | ุ→์, ๅ→ิ, โ→ี |

mirror_h ยังแมปกลับมาเป็นตัวเดิม 18–21 จาก 70 คลาส (glyph ที่เกือบสมมาตร) ส่วน rot90 เหลือแค่ 1–3 คลาส

**นี่คือการวัดที่รองรับกฎห้าม flip ใน `CLAUDE.md`** — ที่ผ่านมาเป็นเหตุผลเชิงภาษา ตอนนี้มีตัวเลขว่าการ flip
ไม่ได้แค่รบกวนภาพ แต่ผลิตตัวอักษรไทยที่ถูกต้องอีกตัวหนึ่งขึ้นมา

### N.4 ข้อสรุป N (คำแนะนำสำหรับวันส่งงาน)

1. **ถ้าข้อมูลวันนั้นเป็นฟอนต์ที่ไม่เคยเห็น** → ใช้สาย G (`r18_64_gen` หรือ `_v3labels`) ชนะ K1 อยู่ 3.4 จุด
2. **ถ้าภาพเบลอ/ความละเอียดต่ำ/หมึกบาง** → ทุกโมเดลพอกัน แต่ **อะไรที่ต่ำกว่า 0.35 ของสเกลเดิมกู้ไม่ได้**
   และ **การกัดเส้น (erode) อันตรายกว่าการขยายเส้นมาก** — ถ้า preprocessing มีขั้นที่ทำให้เส้นบางลง ให้ตัดทิ้ง
3. **ไม่ต้องสร้างตัวแยกตัวอักษรติดกัน** — confidence threshold กรองได้ทุกกลุ่ม (AUROC ≥ 0.84)
4. **ตรวจทิศทางของภาพก่อนเชื่อผลทั้ง batch** — ภาพกลับด้านตรวจได้จาก confidence เฉลี่ยที่ตกฮวบ แต่คู่อย่าง
   บ↔ภ และ ย↔ถ จะผิดแบบมั่นใจ
5. **jpeg, ความละเอียดสูง และสัดส่วนเพี้ยน ไม่ต้องกังวล** — Otsu จัดการให้หมดแล้ว

### N.6 ปิดช่องโหว่การหมุน — แก้ตอน inference ชนะแก้ตอนเทรน

§N.1 ชี้ว่าการหมุนคือจุดพังที่ชัดที่สุดและน่าจะเจอจริง (กระดาษถ่ายรูป/สแกนเอียง 10–30° เป็นเรื่องปกติ)
ทดสอบสองทางแก้ แล้ววัดด้วยชุดฐานเดียวกัน 493 ภาพ

**ทางที่ 1 — ค้นหามุมตอน inference** (`thaichar.infer.predict_with_rotation_search`)
โมเดลเทรนบน glyph ตั้งตรง ดังนั้น "มุมที่โมเดลมั่นใจที่สุด" คือสัญญาณว่าตั้งตรง หมุนภาพไปทีละมุม
(±10/20/30/45) แล้วเลือกคำตอบที่มั่นใจสุด

ถ้าใช้กับทุกภาพ **ขาดทุน** (clean 0.9858 → 0.9716, motion blur −6.7 จุด) เพราะภาพที่เสียอยู่แล้ว
มักมีมุมใดมุมหนึ่งที่โมเดลชอบกว่าของจริง แก้ด้วยการ์ดสองชั้นที่จูนจากการวัด:

| การ์ด | ค่า | ทำอะไร |
|---|---|---|
| `tau` | 0.85 | ค้นเฉพาะภาพที่อ่านตั้งตรงแล้วไม่มั่นใจ — clean เข้าเงื่อนไขแค่ 3 % |
| `margin` | 0.30 | ยอมเปลี่ยนคำตอบก็ต่อเมื่อมุมใหม่ชนะมุมตั้งตรงเกินค่านี้ |

**ทางที่ 2 — ขยาย augmentation ตอนเทรน** (`aug: randaug_hard`)
`randaug` เดิมหมุนแค่ **8°** ทั้งที่จุดพังอยู่ที่ 30° รอบแรกขยายทั้งสามอย่างที่ §N.1 ชี้ (หมุน 8→30°,
downscale 0.5→0.75, stroke kernel 3→5) ผลคือ **มีแค่การหมุนที่ได้ผล**: kernel 5×5 ทำให้ทนการกัดเส้น
*แย่ลง* (erode 4 จาก 0.25 → 0.16 เพราะ kernel ใหญ่ทำลาย glyph บาง แล้ว ink guard ก็ทิ้งตัวอย่างนั้นไป)
และขยาย downscale ขยับ pixelate เพียง −1.2 จุด จึงเหลือขยายแค่การหมุนอย่างเดียว (G9)

**ผลเทียบ** (เปิด rotation search ทุกคอลัมน์):

| input | G4 (ส่งมอบ) | G8 ขยายหมด | G9 ขยายแค่หมุน |
|---|---:|---:|---:|
| clean | 0.9858 | **0.9898** | 0.9858 |
| rotate 30° | 0.8966 | **0.9473** | **0.9473** |
| rotate 45° | 0.8479 | **0.8763** | 0.8600 |
| rotate 60° | 0.6815 | 0.7586 | **0.7688** |
| erode 4 px | **0.2439** | 0.1585 | 0.2043 |
| motion blur 7 | 0.7931 | 0.7870 | **0.8032** |
| pixelate 0.35 | 0.7927 | 0.7825 | **0.8089** |

ค่า stratified val แทบไม่ต่างกัน (G4 0.9902 / G8 0.9901 / G9 0.9900)

**ข้อสรุป N.6**

1. **rotation search เป็นทางที่คุ้มกว่าชัดเจน** — ไม่ต้องเทรนใหม่ ไม่เสีย clean เลย (0.9858 → 0.9858)
   และซื้อคืน **+37.7 จุดที่ 30° กับ +75.1 จุดที่ 45°** ราคาคือ blur หนัก −2.6 จุด
   เทียบกับการเทรนใหม่ที่ได้ +34.1/+46.7 จุดแต่ต้องเทรนและเสีย erode
2. **สองทางเสริมกัน ไม่ทดแทนกัน** — G9 + search ชนะ G4 + search ทุกมุมการหมุน (+5.1 ที่ 30°, +8.7 ที่ 60°)
   และชนะ blur/pixelate เล็กน้อย แต่แพ้ erode 4.0 จุด **ไม่มีตัวไหนชนะขาด**
3. **ไม่เปลี่ยนตัวส่งมอบ** ความต่างทั้งหมดอยู่ระดับไม่กี่ภาพจาก 493 และ strat val เท่ากัน
   G9 export ไว้เป็น `weights/thaichar72_r18_64_gen_rotaug.pt` สำหรับกรณีที่ `scripts/triage_input.py`
   บอกว่าข้อมูลหมุนจริง
4. **บทเรียนเชิงวิธี**: การขยาย augmentation ให้ตรงกับจุดที่วัดว่าพัง **ไม่ได้ช่วยเสมอไป** — สองในสามอย่าง
   ที่ขยายกลับทำให้แย่ลงหรือไม่ขยับ ต้องแยกตัวแปรทีละอย่างแล้ววัด ไม่ใช่ขยายรวดเดียว

### N.7 แก้การตีความ pixelate — มันไม่ใช่ความเสี่ยงอย่างที่ §N.1 อ่าน

§N.1 สรุปว่า pixelate คือหน้าผาที่ชันที่สุด (retention 0.10 ที่สเกล 0.15) และตั้งเป็นงานที่ควรทำต่อ
ตรวจซ้ำแล้ว **การอ่านนั้นเกินจริง**

**ภาพเล็กของจริง โมเดลทำได้ดีอยู่แล้ว** วัดบน val จริง 11,990 ภาพ แยกตามความสูงของ glyph:

| ความสูงจริง | n | top-1 |
|---|---:|---:|
| < 9 px | 498 | **0.9739** |
| 9–11 px | 515 | **0.9825** |
| 12–15 px | 879 | 0.9659 |
| 16–23 px | 5,086 | 0.9766 |
| ≥ 24 px | 5,012 | 0.9928 |

ชุดข้อมูลมีภาพเล็กอยู่แล้ว (8.5 % ต่ำกว่า 12 px, 4.2 % ต่ำกว่า 9 px) และโมเดลอ่านได้ **97 %**
ขณะที่ pixelate ลงไปขนาดเดียวกัน (สเกล 0.35 ของความสูงมัธยฐาน 20 px = 7 px) เหลือ 0.68

**สาเหตุของความต่าง** แยกการย่อออกจาก artifact ของการขยายกลับ:

| สเกล | ย่ออย่างเดียว | ย่อแล้วขยายแบบเนียน | ย่อแล้วขยายแบบบล็อก |
|---|---:|---:|---:|
| 0.50 | 0.8516 | 0.9350 | 0.9167 |
| 0.35 | 0.5488 | 0.7398 | 0.6829 |
| 0.25 | 0.2764 | 0.4695 | 0.4167 |

**ย่ออย่างเดียวแย่ที่สุด** แปลว่าไม่ใช่ artifact จากการขยายกลับ แต่เป็นการทำลายข้อมูล: ตัวอักษรที่ถูกเขียน
และสแกนมาที่ 20 px มีเส้นที่ละเอียดกว่าที่ 7 px จะเก็บไว้ได้ พอบีบลงเส้นก็รวมกันหรือหายไป ส่วนตัวอักษรที่
**เล็ก 7 px มาตั้งแต่ต้น** ถูกเขียนให้อ่านออกที่ขนาดนั้น โครงสร้างจึงยังครบ

**ข้อสรุป N.7**

1. **สถานการณ์จริงของ "ข้อมูลความละเอียดต่ำ" คือภาพที่เล็กมาแต่แรก ซึ่งได้ 0.97 อยู่แล้ว** ไม่ใช่ 0.55
   ถ้าอาจารย์ให้ภาพสแกนความละเอียดต่ำ glyph จะถูกจับภาพที่ความละเอียดนั้นตั้งแต่แรก = เคสที่ทำได้ดี
2. `pixelate` ใน stress suite จำลองสิ่งที่ต้องมีคน**ย่อภาพก่อนส่งให้เรา** ซึ่งเป็นการกระทำที่แปลก
   เก็บการทดสอบไว้ได้เพราะเป็นขอบเขตบน แต่**ไม่ควรใช้ตั้งลำดับความสำคัญของงาน**
3. เทียบกับการหมุนซึ่งเกิดขึ้นเองทุกครั้งที่ถ่ายรูปหรือวางกระดาษเอียง — **การหมุนเป็นความเสี่ยงจริง
   ส่วน pixelate ไม่ใช่** จึงไม่ลงแรงกับ pixelate ต่อ
4. บทเรียนเชิงวิธีเดียวกับ §N.6: ตัวเลข stress ที่ดูน่ากลัวต้องถามก่อนว่า**การแปลงนั้นเกิดขึ้นจริงได้อย่างไร**
   ก่อนจะเอาไปตั้งเป็นงาน

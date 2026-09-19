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

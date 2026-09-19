# External Public Thai Character Datasets Summary (TASK-07)

## 1. Overview

This report documents the acquisition, character mapping, binarization, and caching of public Thai character datasets for intermediate pretraining.

- **Target Corpus:** 72 TIS-620 character classes (consonants, vowels, tone marks, digits) defined in `src/thaichar/classes.py`.
- **Nature of Data:** Public sets are **handwritten**; project corpus is **printed binary glyphs**. These external data are intended exclusively for representation pretraining and intermediate feature learning.

## 2. Per-Source Results and Metadata

### ALICE-THI

- **Status:** Succeeded
- **Source URL:** [https://www.ai.rug.nl/~mrolarik/ALICE-THI/ALICE-THI-Dataset.tar.gz](https://www.ai.rug.nl/~mrolarik/ALICE-THI/ALICE-THI-Dataset.tar.gz)
- **Homepage:** [https://www.ai.rug.nl/~mrolarik/ALICE-THI/](https://www.ai.rug.nl/~mrolarik/ALICE-THI/)
- **License / Terms:** Non-commercial research use; 'Do not copy without notice! Copyright 2015 Olarik Surinta (olarik.s@msu.ac.th)'
- **Academic Citation:** O. Surinta, M. F. Karaaba, L. R. B. Schomaker, M. A. Wiering, "Recognition of handwritten characters using local gradient feature descriptors", Engineering Applications of Artificial Intelligence, 45:405-414, 2015. DOI: 10.1016/j.engappai.2015.07.017

| Metric | Count |
|:---|---:|
| Images Downloaded / Discovered | 24,045 |
| Images Mapped to 72 Target Classes | 22,401 (93.2%) |
| Images Unmapped (Excluded Classes) | 1,644 (6.8%) |

#### Unmapped Characters in ALICE-THI

| Code | Char | Original Label | Count | Rationale for Exclusion |
|---:|:---:|:---|---:|:---|
| 165 | ฅ | 4 (THAI CHARACTER KHO KHON) | 208 | Excluded from our 72-class TIS-620 target corpus |
| 166 | ฆ | 5 (THAI CHARACTER KHO RAKHANG) | 206 | Excluded from our 72-class TIS-620 target corpus |
| 172 | ฌ | 11 (THAI CHARACTER CHO CHOE) | 207 | Excluded from our 72-class TIS-620 target corpus |
| 174 | ฎ | 13 (THAI CHARACTER DO CHADA) | 221 | Excluded from our 72-class TIS-620 target corpus |
| 198 | ฦ | 37 (THAI CHARACTER LU) | 221 | Excluded from our 72-class TIS-620 target corpus |
| 208 | ะ | 47 (THAI CHARACTER SARA A) | 214 | Excluded from our 72-class TIS-620 target corpus |
| 235 | ๋ | 65 (THAI CHARACTER MAI CHATTAWA) | 199 | Excluded from our 72-class TIS-620 target corpus |
| 237 | ํ | 67 (THAI CHARACTER NIKHAHIT) | 168 | Excluded from our 72-class TIS-620 target corpus |

#### Cropped Glyph Statistics (After Otsu Binarization)

| Statistic | Width (px) | Height (px) | Ink Fraction |
|:---|---:|---:|---:|
| Min | 3 | 8 | 0.0944 |
| Median | 29.0 | 34.0 | 0.3295 |
| Mean | 29.8 | 34.9 | 0.3360 |
| Max | 90 | 85 | 0.8947 |

### Burapha-TH (Character + Digit)

- **Status:** Succeeded
- **Source URL:** [https://services.informatics.buu.ac.th/datasets/Burapha-TH/](https://services.informatics.buu.ac.th/datasets/Burapha-TH/)
- **Homepage:** [https://services.informatics.buu.ac.th/datasets/Burapha-TH/](https://services.informatics.buu.ac.th/datasets/Burapha-TH/)
- **License / Terms:** Creative Commons Attribution 4.0 International (CC BY 4.0) via MDPI open-access publication
- **Academic Citation:** A. Onuean, U. Buatoom, T. Charoenporn, T. Kim, H. Jung, "Burapha-TH: A Multi-Purpose Character, Digit, and Syllable Handwriting Dataset", Applied Sciences, 12(8):4083, 2022. DOI: 10.3390/app12084083

| Metric | Count |
|:---|---:|
| Images Downloaded / Discovered | 87,600 |
| Images Mapped to 72 Target Classes | 78,607 (89.7%) |
| Images Unmapped (Excluded Classes) | 8,993 (10.3%) |

#### Unmapped Characters in Burapha-TH (Character + Digit)

| Code | Char | Original Label | Count | Rationale for Exclusion |
|---:|:---:|:---|---:|:---|
| 165 | ฅ | KHO KHON (THAI CHARACTER KHO KHON) | 1,110 | Excluded from our 72-class TIS-620 target corpus |
| 166 | ฆ | KHO RAKHANG (THAI CHARACTER KHO RAKHANG) | 1,134 | Excluded from our 72-class TIS-620 target corpus |
| 172 | ฌ | CHO CHOE (THAI CHARACTER CHO CHOE) | 1,121 | Excluded from our 72-class TIS-620 target corpus |
| 174 | ฎ | DO CHADA (THAI CHARACTER DO CHADA) | 1,135 | Excluded from our 72-class TIS-620 target corpus |
| 198 | ฦ | LU (THAI CHARACTER LU) | 1,137 | Excluded from our 72-class TIS-620 target corpus |
| 208 | ะ | SARA A (THAI CHARACTER SARA A) | 1,105 | Excluded from our 72-class TIS-620 target corpus |
| 211 | ำ | SARA AM (THAI CHARACTER SARA AM) | 1,118 | Excluded from our 72-class TIS-620 target corpus |
| 235 | ๋ | MAI CHATTAWA (THAI CHARACTER MAI CHATTAWA) | 1,133 | Excluded from our 72-class TIS-620 target corpus |

#### Cropped Glyph Statistics (After Otsu Binarization)

| Statistic | Width (px) | Height (px) | Ink Fraction |
|:---|---:|---:|---:|
| Min | 3 | 7 | 0.0103 |
| Median | 31.0 | 41.0 | 0.2902 |
| Mean | 33.0 | 44.4 | 0.2950 |
| Max | 155 | 144 | 0.9000 |

### KVIS Thai OCR

- **Status:** Skipped (Login / Gated)
- **Reason:** Mendeley Data S3 direct zip URL returned HTTP 403 Forbidden; dataset landing page (https://data.mendeley.com/datasets/8nr3pbdk5c/1) requires interactive OAuth2 login / credentials. Loader script kvis_th_ocr.py deprecated in datasets >= 5.0.
- **Source URL:** [https://data.mendeley.com/datasets/8nr3pbdk5c/1](https://data.mendeley.com/datasets/8nr3pbdk5c/1)
- **Homepage:** [https://data.mendeley.com/datasets/8nr3pbdk5c/1](https://data.mendeley.com/datasets/8nr3pbdk5c/1)
- **License / Terms:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Academic Citation:** F. J. J. Joseph, P. Anantaprayoon, "Offline Handwritten Thai Character Recognition Using Single Tier Classifier and Local Features", 2018 International Conference on Information Technology (InCIT), pp. 1-4, 2018. DOI: 10.23919/INCIT.2018.8584876

## 3. Combined 72-Class Coverage Table

Counts of binarized, cached glyphs per class across all active external sources.

| Label | Code | Char | Category | Unicode Name | ALICE-THI | Burapha-TH | Total External | Status |
|---:|---:|:---:|:---|:---|---:|---:|---:|:---|
| 0 | 161 | ก | consonant | THAI CHARACTER KO KAI | 218 | 1,143 | 1,361 | Covered |
| 1 | 162 | ข | consonant | THAI CHARACTER KHO KHAI | 221 | 1,140 | 1,361 | Covered |
| 2 | 163 | ฃ | consonant | THAI CHARACTER KHO KHUAT | 212 | 1,055 | 1,267 | Covered |
| 3 | 164 | ค | consonant | THAI CHARACTER KHO KHWAI | 210 | 1,126 | 1,336 | Covered |
| 4 | 167 | ง | consonant | THAI CHARACTER NGO NGU | 219 | 1,146 | 1,365 | Covered |
| 5 | 168 | จ | consonant | THAI CHARACTER CHO CHAN | 217 | 1,137 | 1,354 | Covered |
| 6 | 169 | ฉ | consonant | THAI CHARACTER CHO CHING | 203 | 1,129 | 1,332 | Covered |
| 7 | 170 | ช | consonant | THAI CHARACTER CHO CHANG | 210 | 1,161 | 1,371 | Covered |
| 8 | 171 | ซ | consonant | THAI CHARACTER SO SO | 208 | 1,146 | 1,354 | Covered |
| 9 | 173 | ญ | consonant | THAI CHARACTER YO YING | 213 | 1,119 | 1,332 | Covered |
| 10 | 175 | ฏ | consonant | THAI CHARACTER TO PATAK | 220 | 1,081 | 1,301 | Covered |
| 11 | 176 | ฐ | consonant | THAI CHARACTER THO THAN | 221 | 1,129 | 1,350 | Covered |
| 12 | 177 | ฑ | consonant | THAI CHARACTER THO NANGMONTHO | 213 | 1,135 | 1,348 | Covered |
| 13 | 178 | ฒ | consonant | THAI CHARACTER THO PHUTHAO | 214 | 1,128 | 1,342 | Covered |
| 14 | 179 | ณ | consonant | THAI CHARACTER NO NEN | 208 | 1,121 | 1,329 | Covered |
| 15 | 180 | ด | consonant | THAI CHARACTER DO DEK | 207 | 1,158 | 1,365 | Covered |
| 16 | 181 | ต | consonant | THAI CHARACTER TO TAO | 210 | 1,143 | 1,353 | Covered |
| 17 | 182 | ถ | consonant | THAI CHARACTER THO THUNG | 217 | 1,083 | 1,300 | Covered |
| 18 | 183 | ท | consonant | THAI CHARACTER THO THAHAN | 200 | 1,131 | 1,331 | Covered |
| 19 | 184 | ธ | consonant | THAI CHARACTER THO THONG | 211 | 1,131 | 1,342 | Covered |
| 20 | 185 | น | consonant | THAI CHARACTER NO NU | 205 | 1,128 | 1,333 | Covered |
| 21 | 186 | บ | consonant | THAI CHARACTER BO BAIMAI | 206 | 1,141 | 1,347 | Covered |
| 22 | 187 | ป | consonant | THAI CHARACTER PO PLA | 221 | 1,150 | 1,371 | Covered |
| 23 | 188 | ผ | consonant | THAI CHARACTER PHO PHUNG | 214 | 1,119 | 1,333 | Covered |
| 24 | 189 | ฝ | consonant | THAI CHARACTER FO FA | 221 | 1,145 | 1,366 | Covered |
| 25 | 190 | พ | consonant | THAI CHARACTER PHO PHAN | 212 | 1,108 | 1,320 | Covered |
| 26 | 191 | ฟ | consonant | THAI CHARACTER FO FAN | 220 | 1,189 | 1,409 | Covered |
| 27 | 192 | ภ | consonant | THAI CHARACTER PHO SAMPHAO | 209 | 1,185 | 1,394 | Covered |
| 28 | 193 | ม | consonant | THAI CHARACTER MO MA | 213 | 1,185 | 1,398 | Covered |
| 29 | 194 | ย | consonant | THAI CHARACTER YO YAK | 213 | 1,182 | 1,395 | Covered |
| 30 | 195 | ร | consonant | THAI CHARACTER RO RUA | 219 | 1,186 | 1,405 | Covered |
| 31 | 196 | ฤ | consonant | THAI CHARACTER RU | 221 | 1,191 | 1,412 | Covered |
| 32 | 197 | ล | consonant | THAI CHARACTER LO LING | 216 | 1,182 | 1,398 | Covered |
| 33 | 199 | ว | consonant | THAI CHARACTER WO WAEN | 211 | 1,190 | 1,401 | Covered |
| 34 | 200 | ศ | consonant | THAI CHARACTER SO SALA | 215 | 1,195 | 1,410 | Covered |
| 35 | 201 | ษ | consonant | THAI CHARACTER SO RUSI | 213 | 1,194 | 1,407 | Covered |
| 36 | 202 | ส | consonant | THAI CHARACTER SO SUA | 212 | 1,174 | 1,386 | Covered |
| 37 | 203 | ห | consonant | THAI CHARACTER HO HIP | 211 | 1,186 | 1,397 | Covered |
| 38 | 204 | ฬ | consonant | THAI CHARACTER LO CHULA | 212 | 1,149 | 1,361 | Covered |
| 39 | 205 | อ | consonant | THAI CHARACTER O ANG | 217 | 1,140 | 1,357 | Covered |
| 40 | 206 | ฮ | consonant | THAI CHARACTER HO NOKHUK | 207 | 1,195 | 1,402 | Covered |
| 41 | 207 | ฯ | consonant | THAI CHARACTER PAIYANNOI | 218 | 1,141 | 1,359 | Covered |
| 42 | 209 | ั | vowel | THAI CHARACTER MAI HAN-AKAT | 221 | 1,122 | 1,343 | Covered |
| 43 | 210 | า | vowel | THAI CHARACTER SARA AA | 220 | 1,108 | 1,328 | Covered |
| 44 | 212 | ิ | vowel | THAI CHARACTER SARA I | 219 | 1,109 | 1,328 | Covered |
| 45 | 213 | ี | vowel | THAI CHARACTER SARA II | 217 | 1,097 | 1,314 | Covered |
| 46 | 214 | ึ | vowel | THAI CHARACTER SARA UE | 221 | 1,103 | 1,324 | Covered |
| 47 | 215 | ื | vowel | THAI CHARACTER SARA UEE | 221 | 1,083 | 1,304 | Covered |
| 48 | 216 | ุ | vowel | THAI CHARACTER SARA U | 220 | 1,079 | 1,299 | Covered |
| 49 | 217 | ู | vowel | THAI CHARACTER SARA UU | 190 | 1,098 | 1,288 | Covered |
| 50 | 224 | เ | vowel | THAI CHARACTER SARA E | 221 | 1,105 | 1,326 | Covered |
| 51 | 225 | แ | vowel | THAI CHARACTER SARA AE | 0 | 1,107 | 1,107 | Covered |
| 52 | 226 | โ | vowel | THAI CHARACTER SARA O | 220 | 1,097 | 1,317 | Covered |
| 53 | 227 | ใ | vowel | THAI CHARACTER SARA AI MAIMUAN | 221 | 1,084 | 1,305 | Covered |
| 54 | 228 | ไ | vowel | THAI CHARACTER SARA AI MAIMALAI | 221 | 1,103 | 1,324 | Covered |
| 55 | 229 | ๅ | vowel | THAI CHARACTER LAKKHANGYAO | 0 | 0 | 0 | **ZERO SAMPLES** |
| 56 | 230 | ๆ | tone_mark | THAI CHARACTER MAIYAMOK | 221 | 0 | 221 | Covered |
| 57 | 231 | ็ | tone_mark | THAI CHARACTER MAITAIKHU | 207 | 990 | 1,197 | Covered |
| 58 | 232 | ่ | tone_mark | THAI CHARACTER MAI EK | 215 | 1,104 | 1,319 | Covered |
| 59 | 233 | ้ | tone_mark | THAI CHARACTER MAI THO | 209 | 1,141 | 1,350 | Covered |
| 60 | 234 | ๊ | tone_mark | THAI CHARACTER MAI TRI | 208 | 1,078 | 1,286 | Covered |
| 61 | 236 | ์ | tone_mark | THAI CHARACTER THANTHAKHAT | 210 | 1,112 | 1,322 | Covered |
| 62 | 240 | ๐ | digit | THAI DIGIT ZERO | 930 | 1,119 | 2,049 | Covered |
| 63 | 241 | ๑ | digit | THAI DIGIT ONE | 951 | 1,096 | 2,047 | Covered |
| 64 | 242 | ๒ | digit | THAI DIGIT TWO | 961 | 1,119 | 2,080 | Covered |
| 65 | 243 | ๓ | digit | THAI DIGIT THREE | 933 | 1,123 | 2,056 | Covered |
| 66 | 244 | ๔ | digit | THAI DIGIT FOUR | 968 | 1,053 | 2,021 | Covered |
| 67 | 245 | ๕ | digit | THAI DIGIT FIVE | 968 | 1,080 | 2,048 | Covered |
| 68 | 246 | ๖ | digit | THAI DIGIT SIX | 975 | 1,086 | 2,061 | Covered |
| 69 | 247 | ๗ | digit | THAI DIGIT SEVEN | 966 | 1,039 | 2,005 | Covered |
| 70 | 248 | ๘ | digit | THAI DIGIT EIGHT | 951 | 972 | 1,923 | Covered |
| 71 | 249 | ๙ | digit | THAI DIGIT NINE | 952 | 986 | 1,938 | Covered |

## 4. Analysis of Classes with Zero External Samples

Out of our 72 target classes, **1** class(es) have **zero external samples**:

- **Label 55** (Code 229, `ๅ` — THAI CHARACTER LAKKHANGYAO): Not present in either ALICE-THI or Burapha-TH isolated character sets.

### Why are these classes missing?
- `แ` (TIS-620 code 225, SARA AE): In many traditional Thai handwriting datasets, Sara Ae is often omitted as an isolated glyph because writers may decompose it into two Sara E (`เเ`) glyphs or it is only captured in multi-character syllable datasets.
- `ๅ` (TIS-620 code 229, LAKKHANGYAO): A specialized lengthened vowel mark used only with `ฤๅ` and `ฦๅ`; rarely written in isolated character sheets.
- All 44 standard consonants present in CLASS_CODES and all 10 Thai digits (๐–๙) have substantial coverage.

## 5. Storage and Disk Footprint

- **Downloaded Archive Footprint:** 244.3 MB (well below the 3 GB project ceiling)
- **Output Glyph Cache (`glyphs_external.npz`):** 177.3 MB (100,985 glyphs)
- **Output Index (`index.csv`):** 12496.4 KB (100,985 mapped rows)
- **Covered Target Classes:** 71 / 72 classes
- **Unique Groups / Writers:** 7,906 distinct groups

## 6. Representative Montage

A visual montage showing 12 binarized samples from each successful source has been generated at `reports/figures/external_montage.png`.


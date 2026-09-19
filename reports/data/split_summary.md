# Data Preparation Summary

## Drop Counts (Cleaning Rules)

| Rule | Description | Dropped |
|---:|:---|---:|
| 1 | ok == False | 0 |
| 2 | Filename starts with 'Copy of' | 6 |
| 3 | Cross-class md5 groups | 18 |
| 4 | Within-class exact duplicates | 2566 |
| **Total** | | **2590** |

**Original rows:** 62707  
**Final clean rows:** 60117

## Stratified Split (seed 42)

- Train: 48090
- Val:   12027

## Doc Split (seed 42)

- Train: 48133
- Val:   11984
- Held-out doc_ids: [3, 14, 16, 20]

## Classes with 0 val samples (stratified, seed 42)

- 163 (ฃ)
- 177 (ฑ)

## Overlap Assertions

All seeds: stratified train ∩ val = 0, doc train ∩ val = 0  ✓

## Per-Class Table (seed 42)

| Code | Char | Category | n_clean | n_train | n_val | n_doc_train | n_doc_val |
|---:|:---:|:---:|---:|---:|---:|---:|---:|
| 161 | ก | consonant | 1934 | 1547 | 387 | 1585 | 349 |
| 162 | ข | consonant | 1348 | 1078 | 270 | 1056 | 292 |
| 163 | ฃ | consonant | 1 | 1 | 0 | 1 | 0 |
| 164 | ค | consonant | 1442 | 1154 | 288 | 1191 | 251 |
| 167 | ง | consonant | 1069 | 855 | 214 | 872 | 197 |
| 168 | จ | consonant | 1196 | 957 | 239 | 952 | 244 |
| 169 | ฉ | consonant | 43 | 34 | 9 | 31 | 12 |
| 170 | ช | consonant | 1013 | 810 | 203 | 814 | 199 |
| 171 | ซ | consonant | 295 | 236 | 59 | 216 | 79 |
| 173 | ญ | consonant | 113 | 90 | 23 | 83 | 30 |
| 175 | ฏ | consonant | 32 | 26 | 6 | 30 | 2 |
| 176 | ฐ | consonant | 118 | 94 | 24 | 100 | 18 |
| 177 | ฑ | consonant | 1 | 1 | 0 | 1 | 0 |
| 178 | ฒ | consonant | 48 | 38 | 10 | 42 | 6 |
| 179 | ณ | consonant | 292 | 234 | 58 | 232 | 60 |
| 180 | ด | consonant | 2016 | 1613 | 403 | 1641 | 375 |
| 181 | ต | consonant | 1660 | 1328 | 332 | 1312 | 348 |
| 182 | ถ | consonant | 435 | 348 | 87 | 335 | 100 |
| 183 | ท | consonant | 1735 | 1388 | 347 | 1357 | 378 |
| 184 | ธ | consonant | 143 | 114 | 29 | 88 | 55 |
| 185 | น | consonant | 4817 | 3854 | 963 | 3787 | 1030 |
| 186 | บ | consonant | 1827 | 1462 | 365 | 1442 | 385 |
| 187 | ป | consonant | 1117 | 894 | 223 | 939 | 178 |
| 188 | ผ | consonant | 343 | 274 | 69 | 279 | 64 |
| 189 | ฝ | consonant | 39 | 31 | 8 | 34 | 5 |
| 190 | พ | consonant | 729 | 583 | 146 | 547 | 182 |
| 191 | ฟ | consonant | 103 | 82 | 21 | 89 | 14 |
| 192 | ภ | consonant | 239 | 191 | 48 | 201 | 38 |
| 193 | ม | consonant | 3280 | 2624 | 656 | 2587 | 693 |
| 194 | ย | consonant | 2185 | 1748 | 437 | 1683 | 502 |
| 195 | ร | consonant | 4574 | 3659 | 915 | 3705 | 869 |
| 196 | ฤ | consonant | 13 | 10 | 3 | 13 | 0 |
| 197 | ล | consonant | 2180 | 1744 | 436 | 1779 | 401 |
| 199 | ว | consonant | 1678 | 1342 | 336 | 1357 | 321 |
| 200 | ศ | consonant | 94 | 75 | 19 | 86 | 8 |
| 201 | ษ | consonant | 264 | 211 | 53 | 231 | 33 |
| 202 | ส | consonant | 1591 | 1273 | 318 | 1276 | 315 |
| 203 | ห | consonant | 1481 | 1185 | 296 | 1127 | 354 |
| 204 | ฬ | consonant | 3 | 2 | 1 | 3 | 0 |
| 205 | อ | consonant | 3245 | 2596 | 649 | 2632 | 613 |
| 206 | ฮ | consonant | 10 | 8 | 2 | 10 | 0 |
| 207 | ฯ | consonant | 20 | 16 | 4 | 20 | 0 |
| 209 | ั | vowel | 3688 | 2950 | 738 | 2898 | 790 |
| 210 | า | vowel | 4537 | 3630 | 907 | 3609 | 928 |
| 212 | ิ | vowel | 47 | 38 | 9 | 37 | 10 |
| 213 | ี | vowel | 143 | 114 | 29 | 122 | 21 |
| 214 | ึ | vowel | 14 | 11 | 3 | 11 | 3 |
| 215 | ื | vowel | 60 | 48 | 12 | 57 | 3 |
| 216 | ุ | vowel | 154 | 123 | 31 | 119 | 35 |
| 217 | ู | vowel | 139 | 111 | 28 | 136 | 3 |
| 224 | เ | vowel | 189 | 151 | 38 | 189 | 0 |
| 225 | แ | vowel | 21 | 17 | 4 | 17 | 4 |
| 226 | โ | vowel | 698 | 558 | 140 | 572 | 126 |
| 227 | ใ | vowel | 1080 | 864 | 216 | 899 | 181 |
| 228 | ไ | vowel | 724 | 579 | 145 | 572 | 152 |
| 229 | ๅ | vowel | 1301 | 1041 | 260 | 1057 | 244 |
| 230 | ๆ | tone_mark | 120 | 96 | 24 | 106 | 14 |
| 231 | ็ | tone_mark | 118 | 94 | 24 | 104 | 14 |
| 232 | ่ | tone_mark | 315 | 252 | 63 | 261 | 54 |
| 233 | ้ | tone_mark | 990 | 792 | 198 | 803 | 187 |
| 234 | ๊ | tone_mark | 49 | 39 | 10 | 39 | 10 |
| 236 | ์ | tone_mark | 684 | 547 | 137 | 547 | 137 |
| 240 | ๐ | digit | 81 | 65 | 16 | 78 | 3 |
| 241 | ๑ | digit | 46 | 37 | 9 | 26 | 20 |
| 242 | ๒ | digit | 39 | 31 | 8 | 21 | 18 |
| 243 | ๓ | digit | 23 | 18 | 5 | 11 | 12 |
| 244 | ๔ | digit | 16 | 13 | 3 | 16 | 0 |
| 245 | ๕ | digit | 17 | 14 | 3 | 17 | 0 |
| 246 | ๖ | digit | 12 | 10 | 2 | 9 | 3 |
| 247 | ๗ | digit | 4 | 3 | 1 | 2 | 2 |
| 248 | ๘ | digit | 27 | 22 | 5 | 19 | 8 |
| 249 | ๙ | digit | 15 | 12 | 3 | 13 | 2 |

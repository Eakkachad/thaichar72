# TASK-14 — Stress suite: how each candidate behaves on inputs we did not train for

Owner's ask (2026-09-22): before the hand-in, know how every model we might ship behaves on
touching characters, half characters, overlaps, rotation, upside-down, mirrored, non-text,
unseen fonts, pixelation, blur, and anything else the day might bring.

`scripts/robustness.py` already covers ten corruptions (rotate ≤20°, thicker/thinner ≤3 px,
salt-pepper ≤0.20, downscale ≥0.4, occlusion ≤0.40, blur ≤1.5, contrast, translate ≤0.20,
background noise) — all of them mild, and all of them the kind where the right answer still
exists. **None of what the owner listed is covered.** This task adds the rest.

---

## 1. The design decision that everything else follows from

The inputs on the list are not one kind of problem. They split three ways, and using one
metric across them would produce a number that means nothing.

| family | is there a correct answer among the 72 classes? | what we want | metric |
|---|---|---|---|
| **A. degraded** | yes, unchanged | still get it right | top-1 retention vs severity |
| **B. invalid** | **no** | not be confidently wrong | confident-wrong rate; mean max-prob |
| **C. identity-changing** | **maybe a different class** | know which, and how often | flip-mapping table |

Family B is the one people get wrong. Accuracy on a picture of two characters glued together
is undefined — there is no right answer to score. What matters is whether the model *signals*
that it is out of its depth, because that is what tells us whether a preprocessing step could
catch these before they reach the classifier.

Family C exists because Thai is not flip-invariant — `CLAUDE.md` bans flips in augmentation for
exactly this reason. But "the glyph changes meaning" has never been measured. Some flips may
land on another real class (a *confusable* pair created by the transform), others on nothing.
The mapping is the result, not an accuracy.

---

## 2. Test material

Built from **held-out validation glyphs only** (`split_seed42_v2.csv`, `split=="val"`), so
nothing here was trained on. Base sample: 8 images/class × 72 = ~570 images, fixed seed, the
same base for every model and every corruption so differences are the corruption and not the draw.

Unseen-font material comes from the **5 font families held out of the dataUpdate probe**
(IBMPlexSansThai, Mali, Sarabun, Sriracha, Trirong) plus the **3 `.ttc` collections
`synth.py` skips** (it globs `*.ttf` only) — these are typefaces the stage-1 render never used
in that form.

Non-text negatives are generated, not collected, so the suite stays reproducible: random
strokes, blobs, scribbles, Latin letters and Arabic digits rendered in the same pipeline.

---

## 3. Family A — degraded, the answer still exists

New corruptions, each at 4 severities so we get a curve rather than a point.

| id | corruption | severities | why it is plausible on the day |
|---|---|---|---|
| A1 | `jpeg` quality | 40, 20, 10, 5 | anything that went through a phone or a chat app |
| A2 | `pixelate` (down-then-up) | 0.5, 0.35, 0.25, 0.15 | a low-res crop upscaled to a fixed size |
| A3 | `motion_blur` (length px) | 3, 5, 7, 9 | handheld photo; existing `blur` is gaussian only |
| A4 | `rotate_hard` | 30, 45, 60, 90 | existing stops at 20°, the owner asked about real rotation |
| A5 | `stroke_extreme` | erode 4, 5 / dilate 4, 5 | beyond the ±3 px already covered |
| A6 | `unseen_font` | 8 held-out typefaces | a test set typeset in something we never rendered |
| A7 | `resolution_up` | 2×, 4×, 8× | a scan at higher DPI than round2 (already known to cost ~4 pt) |
| A8 | `aspect_stretch` | 0.7, 0.85, 1.2, 1.4 | a resize that did not preserve aspect — we never train on this |

**Metric**: top-1 and balanced accuracy per severity, reported as retention against the clean
number of the same model. **Report the curve, and the severity at which each model crosses 90 %
and 50 % retention** — that single number ranks robustness more honestly than an average.

---

## 4. Family B — invalid input, no correct answer

| id | input | how it is built |
|---|---|---|
| B1 | **two characters touching** | two val glyphs, gap 0 px, baseline aligned |
| B2 | **two characters overlapping** | second glyph offset into the first by 25 / 50 % of its width |
| B3 | **half a character** | crop top / bottom / left / right half, 50 % and 70 % cut |
| B4 | **three characters in a row** | the failure mode the owner's `example/` screenshot showed |
| B5 | **non-text: random strokes** | 2–5 random bezier strokes at the corpus's stroke width |
| B6 | **non-text: blobs** | random filled ellipses at the corpus's ink fraction |
| B7 | **non-text: Latin / digits** | A–Z, 0–9 rendered through the same pipeline |
| B8 | **near-empty** | 1–3 stray ink pixels |

**Metrics** (accuracy is undefined, so none of these report accuracy):

- `confident_wrong_rate@τ` = fraction where max softmax ≥ τ, for τ ∈ {0.5, 0.7, 0.9}.
  Lower is better. This is the number that says how badly a junk image pollutes the output.
- `mean_max_prob`, compared against the same model's mean on clean input. A model whose
  confidence barely drops on garbage cannot be filtered.
- `separability` = AUROC of max-prob between clean val and this family. **≥ 0.8 means a
  confidence threshold could catch these; ≤ 0.6 means it cannot and we would need to fix the
  input instead** (e.g. a connected-component splitter for B1/B2/B4).

**Why this is actionable**: the owner's own `example/` folder contained a two-character crop
filed as a single class, so B1/B2/B4 are not hypothetical for this corpus. If separability is
high we can flag them; if it is low, splitting touching glyphs before classifying is worth
building, and §5 says how to decide.

---

## 5. Family C — the transform may change the answer

| id | transform |
|---|---|
| C1 | rotate 180° (upside down) |
| C2 | mirror horizontally |
| C3 | mirror vertically |
| C4 | transpose (90° rotations, both directions) |

**Output is a mapping table, not a score**: for each of the 72 classes, what the model predicts
under the transform and with what confidence. Three things to read off it:

1. **Classes that map onto another real class confidently** — these are pairs where an
   accidentally flipped test image would be wrong *and* confident. Worth knowing by name.
2. **Classes that map back to themselves** — near-symmetric glyphs, where a flip in the test
   data costs us nothing.
3. **Whether flips are detectable at all** — if flipped input keeps high confidence, an
   upside-down test set would fail silently. That is the worst case and the one to check first.

---

## 6. What is run, and what comes out

All **5 packaged weights**, so the day's choice is informed:
`thaichar72_r18_64_gen` (shipped), `_gen_v3labels`, `thaichar72_resnet18_64` (K1),
`thaichar72_resnet18_64_v1labels`, `thaichar72_mnv3small_64_small`.

```
scripts/stress_suite.py --models all --families A B C
→ reports/stress/results.csv              one row per model × corruption × severity
→ reports/stress/summary.md               ranked table + the 90 %/50 % crossing points
→ reports/stress/fig_family_A_curves.png  retention curves, one panel per corruption
→ reports/stress/fig_family_B_conf.png    confidence distributions, clean vs each junk family
→ reports/stress/flip_map.csv             family C mapping
→ reports/stress/samples/<id>.png         a montage of every corruption, to verify by eye
                                          that we are testing what we think we are testing
```

The sample montages are not decoration. A corruption that is wrong in code produces a clean
number that means nothing — the `.notdef` font check earlier today was exactly this kind of
trap, and looking at the images is what catches it.

---

## 7. Acceptance

1. Every corruption has a montage and it visibly shows the intended effect.
2. Clean baseline reproduces each model's known val top-1 to within 0.5 pt (proves the harness
   is not itself degrading the input — the `_reframe` work showed how easily that happens).
3. Family B reports no accuracy anywhere.
4. Runtime ≤ 10 min for all 5 models × all families on the 4070, so it can be re-run on the day
   against real data if the test set turns out to look strange.
5. `summary.md` ends with a recommendation: which model to prefer if the day's data looks
   degraded, and whether a pre-classifier splitter for touching glyphs is worth building.

---

## 8. What this cannot tell us

The corruptions are our guesses about what "strange" means. A test set can be strange in a way
not on this list, and the suite will say nothing about that. Its value is bounding the failure
modes we *can* name, and giving the day's kit a second use: `evaluate_folder.py` already reports
per-class accuracy on whatever arrives, so the stress numbers are the prior and the real data is
the posterior.

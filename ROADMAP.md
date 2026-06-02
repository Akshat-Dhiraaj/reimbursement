# Roadmap — done, in progress, and the plan for what's left

Status as of 2026-06-02. Rationale for the choices below lives in
[DECISIONS.md](DECISIONS.md); the mechanics in [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Milestone summary

| Milestone | Scope | Status |
|---|---|---|
| **M1** | Deterministic layer + synthetic benchmark + harness + fusion + CLI | ✅ **Done** |
| **M2** | Provenance route: PDF + image-EXIF forensics + real-data FP audit (**3 corpora**) | ✅ **Done** (byte parse **+ pikepdf deep layer** for compressed PDFs; exiftool deferred) |
| **M2.5** | **Faithful extraction route (OCR/VLM/PDF): photo/PDF → fields** | 🔶 **In progress** — IMAGE: VLM (Qwen2-VL-2B, macro 0.725) leads docTR OCR+KIE (0.579) head-to-head; **PDF: born-digital text route landed (macro 0.992)** so PDFs now score end-to-end; scalar-misread confidence calibrated (AUC 0.758) for the M3 fuser |
| **M3** | Image *pixel* forensics route + real fraud corpora + learned fusion | 🔶 **Learned fusion done** (opt-in logistic fuser, real-corpus FP **0.175 → 0.042**; #60 pixel route + real *fraud* positives still planned) |

---

## ✅ Done (with evidence)

**M1 — deterministic layer.**
- Detectors: `arithmetic`, `tax_id` (GSTIN/VAT via `python-stdnum`), `date_sanity`,
  `duplicate`. Noisy-OR fusion. Eval harness + dependency-free metrics. CLI
  (`eval`, `score`). Synthetic labelled generator with field-level ground truth.
- Evidence: synthetic benchmark (240 samples) — each single detector AUC 0.625 /
  recall 1.0 / 0 FP; **fused AUC 1.0 / recall 1.0 / 0 FP**.

**M2 — PDF provenance + real-data audit.**
- `pdf_meta` detector + **two-layer** `forensics/pdf.py` inspector. **Layer 1**
  (dependency-free bytes): incremental updates (extra `%%EOF`), editor tags
  (`/Producer`·`/Creator`), ModDate ≫ CreationDate. Synthetic PDF benchmark + `eval-pdf`.
- **Layer 2** (pikepdf, optional `[pdf-forensics]`, MPL-2.0 — *not* AGPL PyMuPDF):
  decodes object-stream + XMP metadata so it **recovers the editor tag / date gap Layer 1
  misses on modern compressed PDFs**, and adds structural anomalies (AcroForm, JavaScript,
  OpenAction, overlay annotations). New `generate_pdf_deep` compressed corpus + the
  `eval-pdf-forensics` byte-vs-deep contrast; `pdf_meta` gains a `use_deep` knob; falls
  back to Layer 1 (never crashes) when the extra is absent.
- Evidence: PDF benchmark (85 samples, uncompressed) — `pdf_meta` AUC 1.0 / recall 1.0 /
  0 FP over 45 tampers → REVIEW; structured detectors correctly abstain. Deep corpus (80
  *compressed* PDFs) — Layer 1 recall **0.000** (blind by construction) → Layer 2 recall
  **0.833** / 0 FP, **fused recall 1.0**: the recall the extra recovers.
- **Real-data false-positive audit** (`eval-real` + `eval/audit.py` +
  `data/wildreceipt.py`): WildReceipt test split, 472 *genuine* receipts. Fused FP
  **0.364**, entirely the `arithmetic` detector, entirely caused by **lossy oracle
  extraction** (mislabeled total box / under-captured line items), **not** arithmetic
  logic. `tax_id` / `pdf_meta` / `duplicate` → **0 FP**. (FP fell from 0.398 after a
  shared US/EU money parser fixed an oracle bug that mis-read European decimal commas
  100× too high — see M2.5.)
- Real data also caught two robustness bugs synthetic data never could (now fixed +
  tested): `duplicate._key` crashing on a `None` date; the CLI crashing on non-cp1252
  vendor text on Windows.
- **Two more real corpora, behind a `--corpus {wildreceipt,cord,expressexpense}` selector**
  (`data/cord.py`, `data/expressexpense.py`; `eval-real` / `eval-extract` / `eval-calibration`
  all speak it). **CORD** (CC-BY-4.0, naver-clova-ix/cord-v2) is a *second oracle* corpus —
  Indonesian receipts whose human `gt_parse` reconstructs a `Receipt` like WildReceipt's KIE
  (no vendor/date in the labels → those detectors abstain; IDR/ID → GSTIN abstains; pure
  mapping, unit-tested, lazy `datasets` fetch). **ExpressExpense** (MIT, 200 images, *no labels*)
  feeds the FP audit only via real-extractor re-extraction (rejected by the oracle-scored
  `eval-extract`/`eval-calibration`). **Measured CORD oracle FP 0.170 (17/100), entirely
  `arithmetic`** (`total≠subtotal+tax` ×15, `subtotal≠Σlines` ×2; all other detectors abstain).
  The honest, instructive finding: WildReceipt's FP was *lossy extraction*, but CORD's gt_parse
  is **clean structured truth**, so its FP isolates a **different** root cause — the **3-field
  `subtotal/tax/total` model is too narrow** for service-charge / discount / tax-inclusive totals
  (cord-test:48 confirms a tax-inclusive menu line). → a second, Indonesian-locale corpus
  corroborates "the binding constraint is representation completeness, not detector logic" by a
  new mechanism, and names a concrete fix: a richer Receipt model (→ feeds #62). A 1000× money
  bug CORD exposed (`Rp.118.000` → `118.0`) was fixed in `money.parse_money` (negative-lookbehind
  so a currency-prefix dot isn't read as a decimal point) — same class as the earlier EU-decimal
  100× bug. (Reproduce: `slipguard eval-real --corpus cord --limit 100`.)

**M2 (continued) — image-EXIF provenance.**
- `image_meta` detector + `forensics/image.py` inspector (Pillow): flags an image
  *editor* in the EXIF `Software` tag (Photoshop / GIMP / Snapseed / …) and a
  `DateTime` ≫ `DateTimeOriginal` capture-vs-modify gap — the EXIF siblings of the PDF
  editor-tag and ModDate≫CreationDate signals. **Abstains on missing EXIF**: stripped /
  screenshot / AI-generated images carry none, but so do many legitimate shared
  receipts, so absent metadata is not evidence of fraud (and EXIF is strippable /
  forgeable, so a clean read only weakly exonerates). Synthetic image benchmark
  (`data/imagesynth.py`, real EXIF-bearing JPEGs) + `eval-image`.
- Evidence: image benchmark (70 samples) — `image_meta` AUC 1.0 / recall 1.0 / 0 FP
  over 30 EXIF-provenance tampers (editor tag + capture/modify gap); the structured
  detectors and `pdf_meta` correctly abstain on the IMAGE route.
- **198 tests pass** (full suite, all milestones; +11 since #61 for learned fusion — 10 in
  `tests/test_fusion_learned.py` covering the logistic fit, the pluggable combiner, and the
  down-weights-a-noisy-detector contract, + a leakage-free `compare_fusion` smoke).

**M3 (partial) — learned fusion (opt-in, measured).**
- `fusion_learned.py` (`LearnedFuser`): a **dependency-free, hand-rolled logistic regression**
  `σ(w·x + b)` over the **same per-detector confidence-weighted signals noisy-OR already
  consumes** (one feature per detector = `signal.weighted`, keyed by name; abstainer → 0) — so it
  is noisy-OR's inputs with *learned* per-detector weights instead of an implicit equal one
  (apples-to-apples, interpretable). Full-batch GD, class-balanced log-loss + L2, **deterministic**
  (zero init, no RNG); `.explain()` sorts weights by magnitude. Wired in through a pluggable
  `Fuser.combiner`: `None` → noisy-OR (**byte-identical default**, test-guarded), set → learned
  (clamped [0, 1]); `decide`/`verdict`/thresholds stay shared.
- **Leakage-free measurement** (`eval/fusion_bench.py`, `slipguard eval-fusion`): fit on synth seed
  0 + first half of the real corpora, score on synth seed 1 + held-out half; positives = synthetic
  fraud, negatives = synth-clean + real-legitimate (WildReceipt + CORD). FP reported at a **matched
  synthetic fraud-recall** (each fuser picks its own threshold to hit the same recall) + threshold-
  free AUC + legible weights + an `inactive` list (features all-zero in training). date_sanity pinned
  to corpus era for the real batch.
- Evidence (reproduce: `slipguard eval-fusion`): real-corpus FP **0.175 → 0.042 (~4×)** at matched
  fraud-recall; synth-fraud-vs-real-legit AUC **0.867 → 0.990**; synth-vs-synth-clean 1.000 → 0.996.
  Weights **tax_id +6.785 / duplicate +6.271 / date_sanity +5.108 / arithmetic +2.664 /
  pdf_meta +0.000 / image_meta +0.000**, bias −2.914 (120 synth-fraud + 120 synth-clean + 286
  real-legit each side). **The 4× FP cut is the lever both FP audits named:** the fuser learns to
  **down-weight the noisy `arithmetic`** signal relative to the high-precision structural detectors.
- **Two loud honest caveats:** (1) positives are **synthetic**, so this measures synth-fraud-vs-
  real-legit *separation*, not real-fraud detection; (2) pdf_meta/image_meta land at weight **0.000**
  only because the structured/KIE training batch has **no provenance examples**, so the learned fuser
  is **route-specific** — which is why **noisy-OR stays the zero-training default** and the learned
  fuser is **opt-in** (the `inactive` annotation keeps the silent zero honest: "never fired in
  training", not "judged useless").

---

## ❌ What's left, and the plan

### M2.5 — Faithful extraction route *(the measured bottleneck — interface + benchmark landed)*
**Why first:** the real-data audit proved arithmetic precision is capped by
extraction quality, and that the whole pipeline can't score raw photos/PDFs at all
until a `Receipt` can be produced from them. This is the single highest-leverage
piece.
**Plan & status:**
1. ✅ **Done** — `Extractor` interface mirroring `Detector` (`extractors/base.py`):
   `name`, `handles`, `extract(path) -> Receipt`; registry `default_extractors()` /
   `extractor_for(route)`; dependency-free `StructuredExtractor` for the STRUCTURED
   route. Swappable + separately-evaluable, so we can rank extractors, not just detectors.
2. ✅ **Two candidates benchmarked head-to-head** — by numbers, not reputation:
   - ✅ **VLM:** **Qwen2-VL-2B-Instruct** (`extractors/vlm_qwen.py`, apache-2.0) prompts
     the model to emit the `Receipt` schema as JSON; registered for the IMAGE route,
     loaded via transformers Auto classes (any HF VLM is a swappable `--model`). On 100
     real receipts vs the oracle: **macro 0.725**, 0 errors — vendor 0.880, date 0.915,
     subtotal 0.740, tax 0.614, total 0.598, line_count 0.602.
   - ✅ **OCR + KIE:** **docTR** (`extractors/doctr_ocr.py`, Apache-2.0) — two-stage text
     detection+recognition + a transparent keyword/position KIE. On the **same** 100
     receipts: **macro 0.579**, 0 errors — vendor 0.380, date 0.915, subtotal 0.571, tax
     0.600, **total 0.696 (above the VLM)**, line_count 0.312. **Pick: the VLM** (higher
     macro, wins vendor + line_count without per-layout heuristics). The honest, non-obvious
     finding: a first naive single-line KIE scored docTR **0.244** — not an OCR failure (it
     read every amount; date ties at 0.915), but because docTR emits a summary row's *label*
     and *right-column amount* as **separate** lines, so a same-line "keyword + money" rule
     read `SUBTOTAL`/`TOTAL` as money-less and the stray digit in `TAX1` as `1.0`. A
     transparent **row-merge** pre-pass (`_merge_rows`: rejoin lines at the same height,
     x-ordered) lifted docTR **0.244 → 0.579** with no new model — so docTR is now
     *competitive on the arithmetic-driving money fields* (tax ≈ VLM; total > VLM), and the
     VLM's edge is robustness-without-heuristics, not raw money accuracy.
3. 🔶 **Armed for line-item under-capture AND scalar confidence — with an honest verdict** —
   `Receipt.field_confidence` plus an **"extraction incomplete / low-confidence → abstain"**
   guard on `arithmetic` (floor 0.5), fed by **two** honest VLM signals, both free (one greedy
   decode, no extra inference): (a) a **parse-completeness** ratio for `line_items` (fraction of
   emitted items it could parse) that **arms** the guard on the under-capture case the audit
   named — a low ratio makes `arithmetic` abstain instead of flagging a `subtotal ≠ Σlines` gap
   that is really a capture artifact; and (b) a **per-token-logprob** confidence for the scalar
   money fields (the least-confident digit's probability, via `compute_transition_scores`).
   **Measured verdict (honest, two steps).** At the 0.5 floor the logprob signal does **not**
   lower FP — it catches only 18% of misreads, which clear the floor — and that first read as
   "the misreads are simply confident." But the **calibration study** (`eval-calibration`, 222
   scored money reads on the same 100 receipts) shows the signal is **far from uninformative**:
   **AUC 0.758** that a read disagrees with the oracle (0.77–0.83 per field), with a **monotonic
   reliability curve** (accuracy 0.37 below 0.6 → 0.71 in [0.8, 0.9) → 0.87 in [0.9, 1.0)). So
   the *signal* is genuinely good; the *0.5 threshold* was just too low. No free lunch remains —
   every abstain threshold trades misread-recall for dropped-correct reads (T=0.7: 51% of
   misreads caught, 17% of good reads dropped) and the arithmetic-breaking misreads skew
   confident — so its proper home is a **cost-aware learned fuser** (M3) weighing this per-value
   feature against the cost of a needless abstain, not a hand-set floor. Behaviour-preserving on
   clean extractions and on synthetic/oracle fields (empty `field_confidence` == trusted).
4. ✅ **Done for all three routes** — `cli.score` runs route → extractor → detectors:
   `extractor_for(route)` for the dependency-free STRUCTURED core, falling back to the
   route candidate lists (`image_extractors()` / `pdf_extractors()`) so a photo or a
   born-digital PDF scores end-to-end without dragging torch/pypdfium2 into the core path.
5. ✅ **Accuracy benchmark + leaderboard landed** — `eval/extraction.py` scores any
   `Extractor` field-by-field (vendor / date / subtotal / tax / total / line-count)
   against the WildReceipt KIE *oracle* as ground truth (per-field table + macro-avg
   accuracy); `slipguard eval-extract` is the extractor leaderboard. The OCR/VLM
   candidates get picked by numbers, not reputation.
6. ✅ **Re-ran `eval-real` on faithfully-extracted fields — the surprising result.**
   `eval-real --extractor vlm --limit 100` (vs the oracle on the *same* 100 receipts,
   `--today 2019-12-31`) measured arithmetic's FP when the VLM, not the lossy oracle KIE,
   supplies the fields. **It went up, not down: oracle 0.400 → VLM 0.740.** The lesson:
   **per-field accuracy ≠ cross-field arithmetic consistency.** A 0.725-macro extractor
   that *also reads more fields than the oracle* (date 93 vs 47, subtotal 96 vs 77 of 100)
   gives `arithmetic`/`date_sanity` more to check, and a subtotal-to-sum reconciliation
   needs **every** line item right — one misread breaks it (`subtotal≠Σlines` 15→50,
   `line≠qty·price` 0→18). The parse-completeness guard abstains only 4% because it
   catches emitted-but-*unparseable* items, not **confident misreads**. So the binding
   constraint is sharper than "a faithful extractor": it is **a faithful extractor +
   per-value confidence** (the oracle's lower FP was partly an artifact of its *missing*
   fields — you can't mis-reconcile what you never read). → directly motivated the
   scalar-misread confidence work (done — see item 3: token logprobs, with an honest verdict).
7. ✅ **Calibration study landed** — `eval/calibration.py` (`slipguard eval-calibration`) asks
   the threshold-free question behind the 0.5-floor verdict: does the per-value confidence
   actually predict a misread? On 222 scored money reads (same 100 receipts) it does — **AUC
   0.758**, monotonic reliability (accuracy 0.37 below 0.6 → 0.87 in [0.9, 1.0)) — so the signal
   is a genuine, *measured* feature for the M3 learned fuser, not just a hoped-for one (item 3).
8. ✅ **PDF-route extractor landed — born-digital PDFs now score end-to-end.** Closed the gap
   where the PDF route had provenance forensics (`pdf_meta`) but **no field read**, so
   arithmetic / tax_id / date_sanity / duplicate never fired on a PDF. `extractors/pdf_text.py`
   (`PdfTextExtractor`, **pypdfium2**, Apache-2.0/BSD — deliberately not AGPL PyMuPDF) reads a
   PDF's **embedded text layer** as per-line rects and feeds the **same shared KIE** the OCR
   route uses (`receipt_from_lines`, factored out into `extractors/kie.py`); exact text → `conf
   1.0`, so the arithmetic guard stays off. Measured on a **minted round-trip oracle**
   (`data/pdfsynth.generate_pdf_extraction` renders known Receipts to real text PDFs;
   `slipguard eval-pdf-extract`, n=40): **macro 0.992** — date / subtotal / tax / total /
   line_count all **1.000**, 0 errors. The lone gap is **vendor 0.950** (38/40), and the honest
   cause is a **KIE labelling** limit, not a read failure: on a short receipt an item line lands
   in the top band and out-letters a short store name (`'Croma'` loses to `'Coffee 47.53'` /
   `'Stapler 6.68'`) — a shared-KIE vendor-rule ceiling that improving lifts *both* routes. **+11
   tests** (`tests/test_pdf_text.py`: rect→Line geometry, the registry/contract, and two
   pypdfium2 round-trips). **Honest limit:** text-layer only — a **scanned-image PDF** (no text
   layer) reads empty and belongs on the IMAGE/OCR route; rasterise-then-OCR is future work.

### M2 (continued) — deeper provenance forensics
**Done:** image EXIF / editor-tag forensics on the IMAGE route (Pillow, `image_meta`).
**Done:** `pikepdf` deep PDF layer (`inspect_pdf_deep`, optional `[pdf-forensics]`,
MPL-2.0) — decodes xref-stream / compressed / XMP metadata so it recovers the editor tag /
date gap Layer 1 misses on modern compressed PDFs, plus structural anomalies (AcroForm,
JavaScript, OpenAction, overlay annotations). Measured byte-vs-deep contrast on a minted
compressed corpus (`eval-pdf-forensics`): `pdf_meta` recall **0.000 → 0.833**, fused **1.0**.
**Deferred — text-over-scan flag:** a text-layer-atop-scan signal collides with legitimate
OCR'd/searchable scans (high FP), so it belongs in the M3 pixel/layout route as a *weak*
signal, not a structural flag (see DECISIONS.md). **Deferred — `exiftool`:** richer image
metadata (maker-notes, thumbnail-vs-image mismatch) beyond Pillow's core EXIF tags; not yet
needed. Both plug in behind the existing `Detector` contract.

### M3 — image forensics route (calibrated *weak* signals)
**Plan:** an AI-generated-image detector (CLIP/ViT family, diversity-trained) and a
tamper-localizer, added as `applies_to=(IMAGE,)` detectors. Per
[DECISIONS.md](DECISIONS.md) §1.2 they are **never the gate** — they are evaluated
honestly under screenshot/JPEG laundering and fused as low-weight inputs.

### M3 — real fraud corpora + learned fusion
**Done — learned fusion (#62):** the opt-in logistic fuser landed and is measured (see the Done
section: real-corpus FP **0.175 → 0.042**, AUC **0.867 → 0.990**, by down-weighting noisy
`arithmetic`). It already weighs the per-detector confidence-weighted signals; the AUC-0.758 scalar
confidence and PDF structural signal slot in as additional features once there are real-fraud
positives to train against.
**Done — legitimate FP-audit corpora:** three real *legitimate* corpora are wired for the
false-positive audit — WildReceipt, CORD, ExpressExpense (see the M2 entry). These measure FP
cost, not fraud recall (every receipt is genuine).
**What remains:** (a) **real *fraud* positives** — pursue written permission for *Find it again!*
(the only public real-forgery set) and onboard IQline HR-provided receipts (proprietary, never
committed) — to replace the **synthetic** positives the fuser is currently fit on (today's headline
measures synth-fraud-vs-real-legit *separation*, not real-fraud detection); (b) **route-appropriate
fuser training** so the learned fuser stops being structured-route-specific — train per-route
batches that actually exercise `pdf_meta`/`image_meta`, so their weights become informative instead
of the honest **0.000** they sit at today (which is why noisy-OR stays the zero-training default);
(c) a **richer `Receipt` model** (service-charge / discount / tax-inclusive fields) — the concrete
data-model target CORD surfaced and the biggest remaining FP lever, since it gives `arithmetic` a
cleaner signal that even a reweighting fuser can't recover.

---

## Known limitations (call them out, don't hide them)
- **All three routes now extract, but each has an honest ceiling.** STRUCTURED has the trivial
  reader; IMAGE has *two* benchmarked extractors (VLM 0.725 + docTR 0.579, ranked by
  `eval-extract`); and the **PDF route now reads** born-digital text (macro 0.992,
  `eval-pdf-extract`). Remaining ceilings: (a) **PDF text-layer only** — a **scanned-image PDF**
  (no text layer) reads empty and belongs on the IMAGE/OCR route (rasterise-then-OCR is future
  work); (b) the **shared-KIE vendor rule** loses a top-band letter-count tie on short receipts
  (the PDF route's only sub-1.0 field, vendor 0.950 — same heuristic the OCR route uses);
  (c) **docTR's KIE is still heuristic** — row-merge fixed the money fields, but vendor (0.380)
  and line_count (0.312) lag the VLM, and non-English summary keywords (German *Netto/MwSt/Summe*)
  aren't in the vocab.
- **Scalar confidence is a calibrated *feature*, not a standalone gate:** the VLM emits **both**
  line-item parse-completeness *and* per-token-logprob scalar confidence; the latter does **not**
  reduce FP at the 0.5 floor (it catches only 18% of misreads), but the **calibration study**
  (`eval-calibration`) measured it is **far from uninformative — AUC 0.758, monotonic
  reliability** — so the 0.5 floor was too low, not the signal useless. There is no free-lunch
  threshold (every cut trades misread-recall for dropped-correct reads), so it earns its keep as
  a **calibrated per-value feature for the cost-aware learned fuser (M3)**, not a hand-set floor.
- **The learned fuser is opt-in and route-specific — measured on *synthetic* positives.** It cuts
  real-corpus FP **0.175 → 0.042** by down-weighting noisy `arithmetic`, but two honest limits keep
  it from being the default: (1) it is fit on **synthetic** fraud, so the headline measures synth-
  fraud-vs-real-legit *separation*, **not** real-fraud detection (it needs real fraud positives to
  become a true detection number); (2) `pdf_meta`/`image_meta` learn weight **0.000** purely because
  the structured/KIE training batch has **no provenance examples** — a route artifact, not a verdict
  that those detectors are useless (the `eval-fusion` `inactive` annotation says so). So **noisy-OR
  stays the zero-training default** and the learned fuser is opt-in until route-appropriate training
  + real-fraud positives exist.
- **The `Receipt` model is 3-field (subtotal / tax / total) — a measured FP source, not a bug.**
  CORD's clean-oracle FP audit (0.170, all `arithmetic`) isolated this: a genuine receipt whose
  total carries a **service charge / discount**, or whose menu prices are **tax-inclusive**, fails
  `total = subtotal + tax` because the model has no slot for those terms. This is an honest data-model
  limitation we deliberately **do not** suppress; the fix is a richer Receipt (extra money fields) —
  scoped as a concrete target for the M3 model/fuser work (#62), since it also gives the fuser a
  cleaner arithmetic signal.
- **Synthetic ≠ real:** the ~1.0 synthetic AUCs validate logic, not real fraud.
- **Image provenance is EXIF-only:** `image_meta` reads metadata, not pixels — a
  re-encoded/stripped image yields no signal (abstain), and EXIF is forgeable. Pixel-
  level AI-generated / tamper-localization forensics are M3 (calibrated *weak* signals).
- **PDF parse depth is gated on an extra:** the dependency-free Layer 1 can't read
  xref-stream/compressed/XMP metadata (string fields read `None`; the `%%EOF` signal still
  holds). The `[pdf-forensics]` extra (pikepdf, Layer 2) recovers those and adds structural
  flags — but **without it installed, compressed PDFs still read blind** (recall 0 on
  compressed-metadata frauds). Installing the extra is the fix; it is optional by design.
- **Date-era artifact:** on the 2010–2019 WildReceipt corpus, `date_sanity`'s
  "very old" check fires as a dataset artifact; use `eval-real --today` to control it.
- **Duplicate audit mode:** the FP audit primes `duplicate` with empty history (each
  receipt judged independently); cross-receipt near-duplicate collisions among
  distinct legitimate receipts are a separate measurement, not yet run.

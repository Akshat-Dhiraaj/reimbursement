# Roadmap — done, in progress, and the plan for what's left

Status as of 2026-06-01. Rationale for the choices below lives in
[DECISIONS.md](DECISIONS.md); the mechanics in [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Milestone summary

| Milestone | Scope | Status |
|---|---|---|
| **M1** | Deterministic layer + synthetic benchmark + harness + fusion + CLI | ✅ **Done** |
| **M2** | Provenance route: PDF + image-EXIF forensics + first real-data FP audit | ✅ **Done** (shallow PDF parse; pikepdf/exiftool deferred) |
| **M2.5** | **Faithful extraction route (OCR/VLM): photo/PDF → fields** | 🔶 **In progress** — two extractors benchmarked head-to-head: VLM (Qwen2-VL-2B, macro 0.725) leads, docTR OCR+KIE second (0.579); scalar-misread confidence + PDF-route extractor next |
| **M3** | Image forensics route + real fraud corpora + learned fusion | ❌ Planned |

---

## ✅ Done (with evidence)

**M1 — deterministic layer.**
- Detectors: `arithmetic`, `tax_id` (GSTIN/VAT via `python-stdnum`), `date_sanity`,
  `duplicate`. Noisy-OR fusion. Eval harness + dependency-free metrics. CLI
  (`eval`, `score`). Synthetic labelled generator with field-level ground truth.
- Evidence: synthetic benchmark (240 samples) — each single detector AUC 0.625 /
  recall 1.0 / 0 FP; **fused AUC 1.0 / recall 1.0 / 0 FP**.

**M2 — PDF provenance + real-data audit.**
- `pdf_meta` detector + dependency-free `forensics/pdf.py` inspector: flags
  incremental updates (extra `%%EOF`), editor tags (`/Producer`·`/Creator`), and
  ModDate ≫ CreationDate. Synthetic PDF benchmark + `eval-pdf`.
- Evidence: PDF benchmark (85 samples) — `pdf_meta` AUC 1.0 / recall 1.0 / 0 FP over
  45 provenance tampers → routed to REVIEW; structured detectors correctly abstain.
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
- **151 tests pass** (full suite, all milestones).

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
4. ✅ **Done for STRUCTURED** — `cli.score` runs route → `extractor_for(route)` →
   detectors; PDF/IMAGE return an explicit "no extractor registered" until step 2 lands.
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

### M2 (continued) — deeper provenance forensics
**Done:** image EXIF / editor-tag forensics on the IMAGE route (Pillow, `image_meta`).
**Plan:** add `pikepdf` / `pdfid` to decode xref-stream, compressed, and XMP PDF
metadata (and detect text-laid-over-scan); optionally `exiftool` for richer image
metadata (maker-notes, thumbnail-vs-image mismatch) beyond the core EXIF tags Pillow
exposes. High-yield, hard to fake, commercial-safe. Plugs in behind the existing
`Detector` contract.

### M3 — image forensics route (calibrated *weak* signals)
**Plan:** an AI-generated-image detector (CLIP/ViT family, diversity-trained) and a
tamper-localizer, added as `applies_to=(IMAGE,)` detectors. Per
[DECISIONS.md](DECISIONS.md) §1.2 they are **never the gate** — they are evaluated
honestly under screenshot/JPEG laundering and fused as low-weight inputs.

### M3 — real fraud corpora + learned fusion
**Plan:** (a) pursue written permission for *Find it again!* and onboard
IQline HR-provided receipts (proprietary, never committed) as real in-domain data;
(b) once the harness has measured per-detector performance on real data, replace
noisy-OR with a **calibrated/learned fuser** fit on those numbers.

---

## Known limitations (call them out, don't hide them)
- **Scalar confidence is a calibrated *feature*, not a standalone gate; PDF route still empty:**
  the IMAGE route now has *two* benchmarked extractors (VLM 0.725 + docTR 0.579, ranked by
  `eval-extract`) and STRUCTURED has the trivial one, but the **PDF route still raises "no
  extractor registered"**. The VLM emits **both** line-item parse-completeness *and*
  per-token-logprob scalar confidence; the latter does **not** reduce FP at the 0.5 floor (it
  catches only 18% of misreads), but the **calibration study** (`eval-calibration`) measured it
  is **far from uninformative — AUC 0.758, monotonic reliability** — so the 0.5 floor was too
  low, not the signal useless. There is no free-lunch threshold (every cut trades misread-recall
  for dropped-correct reads), so it earns its keep as a **calibrated per-value feature for the
  cost-aware learned fuser (M3)**, not a hand-set abstain floor. **docTR's KIE is still heuristic:**
  row-merge fixed the money fields, but vendor (0.380) and line_count (0.312) lag the VLM, and
  non-English summary keywords (e.g. German *Netto/MwSt/Summe*) aren't in the vocab.
- **Synthetic ≠ real:** the ~1.0 synthetic AUCs validate logic, not real fraud.
- **Image provenance is EXIF-only:** `image_meta` reads metadata, not pixels — a
  re-encoded/stripped image yields no signal (abstain), and EXIF is forgeable. Pixel-
  level AI-generated / tamper-localization forensics are M3 (calibrated *weak* signals).
- **Shallow PDF parse:** `forensics/pdf.py` can't read xref-stream/compressed/XMP
  yet; on those PDFs the string fields read `None` (the `%%EOF` signal still holds).
- **Date-era artifact:** on the 2010–2019 WildReceipt corpus, `date_sanity`'s
  "very old" check fires as a dataset artifact; use `eval-real --today` to control it.
- **Duplicate audit mode:** the FP audit primes `duplicate` with empty history (each
  receipt judged independently); cross-receipt near-duplicate collisions among
  distinct legitimate receipts are a separate measurement, not yet run.

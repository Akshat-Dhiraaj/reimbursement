# Roadmap — done, in progress, and the plan for what's left

Status as of 2026-06-01. Rationale for the choices below lives in
[DECISIONS.md](DECISIONS.md); the mechanics in [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Milestone summary

| Milestone | Scope | Status |
|---|---|---|
| **M1** | Deterministic layer + synthetic benchmark + harness + fusion + CLI | ✅ **Done** |
| **M2** | Born-digital route: PDF provenance forensics + first real-data FP audit | ✅ **Done** (shallow PDF parse; pikepdf deferred) |
| **M2.5** | **Faithful extraction route (OCR/VLM): photo/PDF → fields** | 🔶 **In progress** — interface + guard + benchmark + first VLM extractor (Qwen2-VL-2B, macro 0.725) done; OCR+KIE candidate next |
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
- **81 tests pass** (full suite, all milestones).

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
2. 🔶 **First candidate landed; second next** — benchmark candidates head-to-head:
   - ✅ **VLM:** **Qwen2-VL-2B-Instruct** (`extractors/vlm_qwen.py`, apache-2.0) prompts
     the model to emit the `Receipt` schema as JSON; registered for the IMAGE route,
     loaded via transformers Auto classes (any HF VLM is a swappable `--model`). On 100
     real receipts vs the oracle: **macro 0.725**, 0 errors — vendor 0.880, date 0.915,
     subtotal 0.740, tax 0.614, total 0.598, line_count 0.602.
   - ⏳ **OCR + KIE:** PaddleOCR PP-Structure / docTR (Apache-2.0) — the next candidate
     to benchmark against the VLM on the same oracle, by numbers not reputation.
3. 🔶 **Mechanism done, needs the VLM to report confidence** — `Receipt.field_confidence`
   plus an **"extraction incomplete / low-confidence → abstain"** guard on `arithmetic`
   (floor 0.5). Behaviour-preserving today because synthetic/oracle fields carry no
   confidence; it only reduces the 0.364 audit FP once the extractor reports low
   confidence on the boxes it misreads. (The guard is the mechanism, not a number-mover
   on the oracle.)
4. ✅ **Done for STRUCTURED** — `cli.score` runs route → `extractor_for(route)` →
   detectors; PDF/IMAGE return an explicit "no extractor registered" until step 2 lands.
5. ✅ **Accuracy benchmark + leaderboard landed** — `eval/extraction.py` scores any
   `Extractor` field-by-field (vendor / date / subtotal / tax / total / line-count)
   against the WildReceipt KIE *oracle* as ground truth (per-field table + macro-avg
   accuracy); `slipguard eval-extract` is the extractor leaderboard. The *measurement*
   is ready and reports "no extractor handles the IMAGE route yet" until step 2 registers
   one — so the OCR/VLM candidates get picked by numbers, not reputation. **Then** re-run
   `eval-real` to measure arithmetic's *true* FP rate on faithfully-extracted fields.

### M2 (continued) — deeper PDF & image metadata forensics
**Plan:** add `pikepdf` / `pdfid` to decode xref-stream, compressed, and XMP
metadata (and detect text-laid-over-scan); add `exiftool` (or Pillow) for image
EXIF / editor tags on the IMAGE route. High-yield, hard to fake, commercial-safe.
Plugs in behind the existing `Detector` contract.

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
- **One extractor benchmarked; confidence not yet wired, PDF route still empty:** the
  IMAGE route now has the Qwen2-VL-2B extractor (ranked by `eval-extract`) and STRUCTURED
  has the trivial one, but the **PDF route still raises "no extractor registered"**, the
  VLM does **not yet emit per-field confidence** (so the `arithmetic` low-confidence guard
  stays dormant on real extractions), and a **second extractor (OCR+KIE) hasn't been
  benchmarked** against the VLM yet.
- **Synthetic ≠ real:** the ~1.0 synthetic AUCs validate logic, not real fraud.
- **Shallow PDF parse:** `forensics/pdf.py` can't read xref-stream/compressed/XMP
  yet; on those PDFs the string fields read `None` (the `%%EOF` signal still holds).
- **Date-era artifact:** on the 2010–2019 WildReceipt corpus, `date_sanity`'s
  "very old" check fires as a dataset artifact; use `eval-real --today` to control it.
- **Duplicate audit mode:** the FP audit primes `duplicate` with empty history (each
  receipt judged independently); cross-receipt near-duplicate collisions among
  distinct legitimate receipts are a separate measurement, not yet run.

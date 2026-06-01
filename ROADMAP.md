# Roadmap — done, in progress, and the plan for what's left

Status as of 2026-06-01. Rationale for the choices below lives in
[DECISIONS.md](DECISIONS.md); the mechanics in [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Milestone summary

| Milestone | Scope | Status |
|---|---|---|
| **M1** | Deterministic layer + synthetic benchmark + harness + fusion + CLI | ✅ **Done** |
| **M2** | Born-digital route: PDF provenance forensics + first real-data FP audit | ✅ **Done** (shallow PDF parse; pikepdf deferred) |
| **M2.5** | **Faithful extraction route (OCR/VLM): photo/PDF → fields** | ❌ **Next — top priority** |
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
  **0.398**, entirely the `arithmetic` detector, entirely caused by **lossy oracle
  extraction** (mislabeled total box / under-captured line items), **not** arithmetic
  logic. `tax_id` / `pdf_meta` / `duplicate` → **0 FP**.
- Real data also caught two robustness bugs synthetic data never could (now fixed +
  tested): `duplicate._key` crashing on a `None` date; the CLI crashing on non-cp1252
  vendor text on Windows.
- **43 tests pass.**

---

## ❌ What's left, and the plan

### M2.5 — Faithful extraction route *(top priority — the measured bottleneck)*
**Why first:** the real-data audit proved arithmetic precision is capped by
extraction quality, and that the whole pipeline can't score raw photos/PDFs at all
until a `Receipt` can be produced from them. This is the single highest-leverage
piece.
**Plan:**
1. Define an `Extractor` interface mirroring the `Detector` pattern — a swappable,
   **separately-evaluated** approach (so we can rank extractors, not just detectors).
2. Implement two candidates and benchmark them head-to-head:
   - **OCR + KIE:** PaddleOCR PP-Structure / docTR (Apache-2.0).
   - **VLM:** Qwen2.5-VL prompted to emit the `Receipt` schema as JSON.
3. Have the extractor surface **per-field confidence**, and add an
   **"extraction incomplete / low-confidence → abstain"** guard to `arithmetic` so it
   stops crying wolf on a misread (directly addresses the 39.8% audit finding).
4. Wire the chosen extractor into `cli.score` for the PDF/IMAGE routes.
5. Evaluate extraction accuracy on CORD/WildReceipt parses, then re-run `eval-real`
   to measure arithmetic's *true* FP rate on faithfully-extracted fields.

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
- **No extraction yet:** `score` only accepts structured JSON; PDF/IMAGE inputs
  raise an explicit "not wired" error.
- **Synthetic ≠ real:** the ~1.0 synthetic AUCs validate logic, not real fraud.
- **Shallow PDF parse:** `forensics/pdf.py` can't read xref-stream/compressed/XMP
  yet; on those PDFs the string fields read `None` (the `%%EOF` signal still holds).
- **Date-era artifact:** on the 2010–2019 WildReceipt corpus, `date_sanity`'s
  "very old" check fires as a dataset artifact; use `eval-real --today` to control it.
- **Duplicate audit mode:** the FP audit primes `duplicate` with empty history (each
  receipt judged independently); cross-receipt near-duplicate collisions among
  distinct legitimate receipts are a separate measurement, not yet run.

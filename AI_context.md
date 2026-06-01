# AI_context.md — slipguard

> Handoff doc for transferring this project to a fresh AI session or engineer.
> **Keep this file updated every milestone** (status, architecture, roadmap).
> Last updated: 2026-06-01 (Milestone 2).

## 1. What this is

`slipguard` detects **fake reimbursement slips / invoices** — both phone photos
of paper slips and born-digital PDFs. It is a **real internal tool for IQline**
(not a demo), so it prioritises robustness, integration, and **commercial-safe
licenses**.

**Threat model:** employees using AI to fabricate receipt images, AI-inpainting
to change a transaction date/amount, double-submitting the same receipt, or
hand-editing fields (totals, tax, vendor tax-id).

## 2. The one idea that shapes everything

Every detection **approach is an independent, pluggable `Detector`**. We do **not**
pick approaches a priori — an **eval harness ranks them on a labelled benchmark**
(overall AUC, per-fraud-subtype recall, false-positive rate; later, robustness
under screenshot/JPEG laundering). Selection is driven by measured performance.

**Research insight that sets the priority order** (from GPT4o-Receipt / AIForge-Doc
and the image-forensics literature): pixel/AI-image forensics score ~random on
AI-generated receipts and collapse under recompression/screenshots; classic
tamper-localizers drop to F1<0.5 on diffusion inpainting. The **robust** signals
are (1) arithmetic/field consistency, (2) metadata/provenance (PDF structure,
EXIF), (3) cross-submission duplicate intelligence. So we **lead with those** and
treat visual forensics as a **calibrated weak signal, never the gate**.

## 3. Status

**Milestone 1 (done):** deterministic layer + benchmark + harness.
- Detectors: `arithmetic`, `tax_id` (GSTIN/VAT via `python-stdnum`), `date_sanity`, `duplicate`.
- Synthetic labelled benchmark with field-level ground truth and fraud subtypes.
- Eval harness (leaderboard), noisy-OR fusion, CLI (`eval`, `score`), 18 passing tests.

Benchmark (synthetic, seed 0): each single detector **AUC 0.625** (catches only its
own subtype) at **1.0 target-recall / 0 FP**; **fused AUC 1.0, recall 1.0, FP 0**.
⚠️ These numbers validate the harness + deterministic layer on synthetic fraud that
violates these exact rules. They say **nothing** about real-world / AI-generated
fraud — that needs the image+VLM layers and real datasets (Milestone 2+).

**Milestone 2 (in progress):** the real-document route — **PDF provenance forensics**.
- Detector `pdf_meta` (`forensics/pdf.py` inspector + `detectors/pdfmeta.py`): flags
  **incremental updates** (appended xref / extra `%%EOF`), **editor tags** in
  `/Producer`·`/Creator` (Photoshop, iLovePDF, …), and **ModDate ≫ CreationDate**.
  Dependency-free — raw bytes + regex over the literal Info dict.
- Synthetic PDF benchmark (`data/pdfsynth.py`; `build_pdf` writes a valid byte
  layout, no third-party dep) covering the three provenance tampers. New CLI
  `eval-pdf`. **31 passing tests** total.

PDF benchmark (synthetic, seed 0): structured detectors **abstain** on bare PDFs;
`pdf_meta` scores **AUC 1.0 / recall 1.0 / 0 FP** over 45 provenance frauds, which
fusion routes to **REVIEW** (provenance warrants a human look, not an auto-reject).
⚠️ Same caveat: this validates the layer on tampers that violate these exact signals.
It does **not** yet decode xref-stream / compressed / XMP metadata (needs pikepdf/
pdfid) — on those PDFs the string fields read `None` while the `%%EOF` count (the
incremental-update signal) stays reliable.

## 4. Quickstart

```bash
# Python 3.10+ (dev box: 3.13). One runtime dep: python-stdnum.
pip install -e ".[dev]"          # editable install + pytest

python -m pytest                 # run tests
slipguard eval                   # structured benchmark leaderboard
slipguard eval-pdf               # PDF-provenance benchmark leaderboard
slipguard score data/demo.json   # score one receipt JSON (see data/demo.json)

# Without installing, prefix module runs with the src path:
PYTHONPATH=src python -m slipguard eval
```

## 5. Architecture / data flow

```
raw input ──routing.route_path──> DocumentType {PDF | IMAGE | STRUCTURED}
                                        │
                  (extraction approach: VLM / OCR+KIE — NOT yet wired)
                                        ▼
                                   Receipt (models.py)
                                        │
        ┌───────────── default_detectors() — each Detector.run(receipt) ─────────────┐
        arithmetic   tax_id      date_sanity  duplicate    pdf_meta      (+ future: AI-image,
        (reconcile) (GSTIN/VAT) (future date)(resubmit)  (PDF provenance) tamper-loc, EXIF)
        └───────────────────────────── list[Signal] ───────────────────────────────┘
                                        ▼
                          Fuser.verdict  (noisy-OR risk + Decision)
                                        ▼
                       Verdict {risk_score, decision, reasons}

eval.harness.evaluate(dataset, detectors, fuser)  ->  ranked Report (the selector)
```

Key contracts (`models.py`): `Receipt`, `LineItem`, `Signal`
(`score`/`confidence`/`reasons`; `confidence==0` ⇒ abstain), `Verdict`,
`LabeledSample`, enums `DocumentType` / `FraudType` / `Decision`.

## 6. File map

```
pyproject.toml            packaging; pytest pythonpath=src; console script `slipguard`
src/slipguard/
  models.py               domain models + Signal/Verdict (shared contracts)
  routing.py              classify raw input -> DocumentType (PDF/IMAGE/STRUCTURED)
  fusion.py               Fuser: noisy-OR risk + approve/review/reject
  cli.py / __main__.py    `slipguard eval` / `eval-pdf` / `score`
  detectors/
    base.py               Detector ABC: applicable/prime/score, shared run(), _abstain()
    arithmetic.py         line items -> subtotal -> tax -> total reconciliation
    taxid.py              python-stdnum GSTIN (IN) + EU VAT, abstains if unsupported
    datesanity.py         future / implausibly-old dates (today injectable)
    duplicate.py          exact + fuzzy resubmission match; prime()-d with history
    pdfmeta.py            PDF provenance signal (reads forensics.inspect_pdf); PDF route only
    __init__.py           default_detectors() — the canonical ranked set
  forensics/
    pdf.py                dependency-free PDF provenance inspector (%%EOF / editor / date gap)
  data/
    synth.py              synthetic structured clean+fraud generator (benchmark backbone)
    pdfsynth.py           synthetic PDF generator (build_pdf byte layout) + 3 provenance tampers
  eval/
    metrics.py            dependency-free precision/recall/F1/AUC/FPR
    harness.py            evaluate() -> per-detector + fused Report (leaderboard)
tests/                    31 tests: detectors, synth invariants, harness, pdf forensics
```

## 7. How to add a new detection approach

1. Create `src/slipguard/detectors/<name>.py` with a `Detector` subclass:
   - set `name`, `targets` (the `FraudType`s it catches), and `applies_to`
     (e.g. `(DocumentType.IMAGE,)` for image-only forensics; omit for any route);
   - implement `score(receipt) -> Signal`; return `self._abstain(reason)` when you
     lack the data to judge (so you never move the verdict on a guess);
   - if relational (needs history), override `prime(history)`.
2. Add an instance to `default_detectors()` in `detectors/__init__.py`.
3. Add ground-truth for its fraud subtype to `data/synth.py` (or a real-dataset
   loader) so the harness can score it.
4. Add tests. Run `slipguard eval` — the new row appears in the leaderboard.

Nothing else changes: fusion and the harness consume any `Detector` uniformly.

## 8. Roadmap (each plugs in behind the same `Detector` contract)

- **M2 — Extraction route:** VLM (Qwen2.5-VL) / OCR+KIE (PaddleOCR PP-Structure,
  docTR) turning real photos/PDFs into `Receipt`s, wired into `cli.score`. Make the
  extractor itself a swappable, separately-evaluated approach.
- **M2 — PDF & metadata forensics (started):** `pdf_meta` ships incremental-update,
  editor-tag and creation/mod-date checks, dependency-free. Next: pikepdf/pdfid for
  xref-stream + compressed/XMP metadata and text-over-scan; exiftool for image
  EXIF/editor tags. High-yield, hard to fake, commercial-safe.
- **M3 — Image route:** AI-generated detector (CLIP/ViT, diversity-trained) +
  tamper-localization, as **calibrated weak signals**, evaluated honestly under
  recompression/screenshot laundering.
- **M3 — Real data + learned fusion:** wire Find-it-again etc.; replace noisy-OR
  with a calibrated/learned fuser fit on measured per-detector performance.

## 9. Constraints & sources

**Licenses (commercial-safe only):** AVOID LayoutLMv3 (CC-BY-NC), DocTamper (NC),
FUNSD (NC), Surya (GPL). PREFER CORD/WildReceipt, LiLT, docTR, PaddleOCR, Qwen2.5-VL,
python-stdnum.

**Data gap:** labelled fake-receipt data is scarce — the only real public set is
*Find it again!* (~163 forgeries). We synthesise fraud by perturbing clean receipts
(`data/synth.py`); add real corpora as loaders alongside it.

**Key references:** GPT4o-Receipt & AIForge-Doc (AI-forged receipt benchmarks),
DocTamper (CVPR'23), TruFor/CAT-Net (tamper localization), Community Forensics /
C2P-CLIP (AI-image detection), Veryfi/AppZen/Resistant AI (commercial approaches:
metadata + cross-document intelligence).
```

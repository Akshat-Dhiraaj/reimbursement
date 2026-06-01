# AI_context.md — slipguard

> Handoff doc for transferring this project to a fresh AI session or engineer.
> **Keep this file updated every milestone** (status, architecture, roadmap).
> Last updated: 2026-06-01 (Milestone 1).

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

## 4. Quickstart

```bash
# Python 3.10+ (dev box: 3.13). One runtime dep: python-stdnum.
pip install -e ".[dev]"          # editable install + pytest

python -m pytest                 # run tests
slipguard eval                   # benchmark leaderboard
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
        arithmetic        tax_id          date_sanity        duplicate     (+ future: AI-image,
        (reconcile)   (GSTIN/VAT)     (future date)      (resubmission)     tamper-loc, PDF/EXIF)
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
  cli.py / __main__.py    `slipguard eval` and `slipguard score`
  detectors/
    base.py               Detector ABC: applicable/prime/score, shared run(), _abstain()
    arithmetic.py         line items -> subtotal -> tax -> total reconciliation
    taxid.py              python-stdnum GSTIN (IN) + EU VAT, abstains if unsupported
    datesanity.py         future / implausibly-old dates (today injectable)
    duplicate.py          exact + fuzzy resubmission match; prime()-d with history
    __init__.py           default_detectors() — the canonical ranked set
  data/synth.py           synthetic labelled clean+fraud generator (benchmark backbone)
  eval/
    metrics.py            dependency-free precision/recall/F1/AUC/FPR
    harness.py            evaluate() -> per-detector + fused Report (leaderboard)
tests/                    18 tests: detectors, synth invariants, harness
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
- **M2 — PDF & metadata forensics:** pikepdf/pdfid (incremental-update history,
  producer/creator mismatch, text-over-scan), exiftool (editor tags, timestamp
  mismatch). High-yield, hard to fake, commercial-safe.
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

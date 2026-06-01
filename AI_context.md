# AI_context.md — slipguard

> Handoff doc for transferring this project to a fresh AI session or engineer.
> **Keep this file updated every milestone** (status, architecture, roadmap).
> Last updated: 2026-06-01 (M2 + real-data FP audit; M2.5 extractor interface + confidence
> guard + extraction-accuracy benchmark + first real VLM extractor (Qwen2-VL-2B) benchmarked
> + shared US/EU money parser that fixed an oracle ground-truth bug).

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

**First real-data false-positive audit (2026-06-01):** ran the detectors over the
**WildReceipt test split** (472 *genuine* receipts; Apache-2.0) via the new
`eval-real` command. WildReceipt's KIE annotations act as an **oracle extractor**
(text + semantic label per box) so we can reconstruct `Receipt`s without OCR and
measure the one thing the synthetic benchmark cannot: the real-world FP rate. Every
receipt is legitimate, so **any flag is a false positive**.
- Fused FP rate **0.364** (172/472) with `--today 2019-12-31`, **down from 0.398**
  after fixing a real ground-truth bug the VLM run exposed (see the money-parser note
  below) — the entire FP load is still the `arithmetic` detector.
- `tax_id`, `pdf_meta`, `duplicate` produce **zero** FPs (they abstain on this corpus,
  as designed). `date_sanity` never flags alone (its "very old" signal is 0.36
  weighted, under the 0.4 review line); its activity is purely the 2010-2019-vs-2026
  era gap, controllable with `--today`.
- **Every** arithmetic failure traces to **lossy/noisy oracle extraction, not arithmetic
  logic**: `subtotal!=sum(lines)`×63 (line items under-captured) and `total!=subtotal+tax`
  ×117 (the annotation mislabels which box is the grand total, or the receipt carries a
  service charge / tax-inclusive pricing the oracle can't see). A faithful extractor
  would reconcile these.
- **Takeaway:** arithmetic consistency is only as trustworthy as the money-field
  extractor feeding it — its real-world precision can't be measured without one, which
  makes the **faithful OCR/VLM extractor the next measured bottleneck**. Candidate
  mitigation: arithmetic should **abstain** when the extracted fields look incomplete /
  low-confidence rather than asserting fraud.
- Real data also surfaced two robustness bugs synthetic data never could (now fixed +
  tested): `duplicate._key` crashed on a `None` date, and the CLI crashed printing
  non-cp1252 vendor text (★) on Windows. **43 tests pass.**

**Extraction route — interface + confidence guard (2026-06-01):** the dependency-free
foundation of the top-priority extractor work. New `extractors/` package: an
`Extractor` ABC mirroring `Detector` (`name`, `handles`, `extract(path) -> Receipt`),
a registry (`default_extractors()` / `extractor_for(route)`), and a `StructuredExtractor`
for the STRUCTURED route. `slipguard score` now runs **route → extractor → detectors**
uniformly (PDF/IMAGE give an honest "no extractor registered" until OCR/VLM lands).
`Receipt` gained `field_confidence`, and `arithmetic` now **abstains** when its money
fields were read below a confidence floor (default 0.5) — the audit's recommended fix.
Behaviour-preserving: synthetic/oracle fields carry no confidence so they read as
trusted (eval / eval-pdf leaderboards unchanged; the oracle-based eval-real FP — now
0.364 after the money-parser fix noted below — stands until a *real* extractor reports
low confidence; the guard is the mechanism,
not a number-mover on the oracle). Also extracted the duplicated noisy-OR into
`combine.noisy_or` (used by both `fusion` and `pdf_meta`). **50 tests pass.**

**Extraction-accuracy benchmark (2026-06-01):** the extractor *selector* — the same
measured-not-opinion rule the harness applies to detectors, now applied to extraction.
`eval/extraction.py` (`evaluate_extractor` / `evaluate_extractors`) scores any
`Extractor` field-by-field (vendor / date / subtotal / tax / total / line-count)
against the WildReceipt KIE **oracle** as ground truth: the oracle `Receipt` is the
reference, a candidate OCR/VLM extractor run on the *same image* is the prediction, and
a field is scored **only when the oracle supplies a value** (so an extractor is never
penalised for a field truth itself lacks; a raise counts as a miss + error, not a
crash). Money reuses the `arithmetic` tolerance; vendor reuses `duplicate._norm_vendor`
(DRY). New CLI `eval-extract` prints a per-field table + macro-avg leaderboard and
honestly reports "no extractor handles the IMAGE route yet" until a real extractor is
registered — i.e. the *measurement* is ready, so the Phase-2 OCR/VLM candidates get
picked by numbers, not reputation. `data/wildreceipt.py` now resolves `image_path` to a
full path so an image extractor can open the source. **62 tests pass** (+12
extraction-metric tests: perfect extractor → 1.0, broken → 0.0 with error count, money
tolerance, fuzzy vendor, scored-only-when-oracle-has-the-field, exact line-count).

**First real extractor benchmarked — Qwen2-VL-2B-Instruct (2026-06-01):** the OCR/VLM
candidate the benchmark was built to rank now exists (`extractors/vlm_qwen.py`, default
`Qwen/Qwen2-VL-2B-Instruct`, apache-2.0). It prompts the VLM to emit the `Receipt`
schema as JSON, loads via transformers Auto classes (so any HF VLM is a swappable
`--model` candidate), keeps all heavy imports (torch/transformers/PIL) lazy, and is
registered for the IMAGE route. Headline `eval-extract --limit 100` (WildReceipt test,
scored against the oracle): **macro field-accuracy 0.725, 0 extractor errors** — vendor
0.880, date 0.915, subtotal 0.740, tax 0.614, total 0.598, line_count 0.602. The model
reads dates and vendor names very reliably; money/line-count are the weak fields (tax
and grand total get confused on multi-tax / service-charge receipts) — exactly the
signal `arithmetic` depends on, so it sets the realistic ceiling on arithmetic precision.
⚠️ Scored against an *oracle that is itself imperfect*, so these are agreement-with-KIE
numbers, not absolute truth — honest, reproducible, and good enough to pick extractors by.
Runtime: bf16 fully on the 8 GB dev GPU (~4.5 GB peak, ~7-15 s/receipt steady-state).

**Oracle money-format bug the VLM run exposed + the shared parser fix (2026-06-01):**
inspecting per-receipt disagreements showed a chunk of the "errors" were the *oracle*
being wrong, not the model. WildReceipt's old money parser stripped commas, so European
decimals (`Eur129,75` → `12975`, a 100× error) and bare leading dots (`.70` → `70`, a
tax larger than the total) corrupted the ground truth feeding **both** `eval-extract`
and the `eval-real` FP audit. Fixed by extracting one shared, US/EU-aware
`money.parse_money` (new `money.py`) used by **both** the oracle and the VLM extractor
(DRY): the rightmost separator is the decimal point only when 1-2 digits follow it,
otherwise it is thousands grouping (handles `1,234.56`, `1.234,56`, `1,23,456.78`,
`.70`, `-1.234,56`). The vendor metric was also made fair — `_vendor_ok` now credits
substring containment in either direction (oracle's terse `COSTCO` vs a fuller `Costco
Wholesale`) with a length floor, so a correct fuller name is not scored as wrong. These
fixes are what dropped the audit FP **0.398 → 0.364**: the oracle, not the detector, had
been over-counting arithmetic contradictions. **81 tests pass** (+ shared money-parser
unit tests, an EU-decimal oracle regression test, and a vendor-containment test).

## 4. Quickstart

```bash
# Python 3.10+ (dev box: 3.13). One runtime dep: python-stdnum.
pip install -e ".[dev]"          # editable install + pytest

python -m pytest                 # run tests
slipguard eval                   # structured benchmark leaderboard
slipguard eval-pdf               # PDF-provenance benchmark leaderboard
slipguard eval-real              # real-receipt false-positive audit (needs datasets/wildreceipt)
slipguard eval-extract           # rank extractors on field accuracy vs the WildReceipt oracle
slipguard score data/demo.json   # score one receipt JSON (see data/demo.json)

# Fetch the real corpus for eval-real (not committed; Apache-2.0):
#   curl -L -o datasets/wildreceipt.tar https://download.openmmlab.com/mmocr/data/wildreceipt.tar
#   tar -xf datasets/wildreceipt.tar -C datasets

# Without installing, prefix module runs with the src path:
PYTHONPATH=src python -m slipguard eval
```

## 5. Architecture / data flow

```
raw input ──routing.route_path──> DocumentType {PDF | IMAGE | STRUCTURED}
                                        │
        extractor_for(route).extract  (StructuredExtractor live; OCR/VLM planned)
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
  combine.py              noisy_or(): the one probability-combination rule, shared
  money.py                parse_money(): shared US/EU-aware money parser (oracle + VLM extractor, DRY)
  fusion.py               Fuser: noisy-OR risk (via combine.noisy_or) + approve/review/reject
  cli.py / __main__.py    `slipguard eval` / `eval-pdf` / `eval-real` / `eval-extract` / `score`
  extractors/
    base.py               Extractor ABC: handles / can_handle / extract(path) -> Receipt
    structured.py         StructuredExtractor: Receipt JSON -> Receipt (dependency-free)
    vlm_qwen.py           VLM extractor (Qwen2-VL-2B default, apache-2.0); IMAGE route; lazy torch/transformers
    __init__.py           default_extractors() + extractor_for(route) registry
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
    wildreceipt.py        WildReceipt loader: KIE annotations -> Receipt (oracle extraction, no OCR)
  eval/
    metrics.py            dependency-free precision/recall/F1/AUC/FPR
    harness.py            evaluate() -> per-detector + fused Report (detector leaderboard)
    audit.py              audit_false_positives() -> FP report on a legitimate corpus (eval-real)
    extraction.py         evaluate_extractors() -> field-accuracy leaderboard vs the oracle (eval-extract)
tests/                    81 tests: detectors, synth invariants, harness, pdf forensics, loader, FP audit, extraction + extraction-eval, money parser
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

- **M2.5 — Extraction route (the measured bottleneck):** *interface + confidence guard
  + accuracy benchmark + first VLM extractor done* — `Extractor` ABC + registry +
  `StructuredExtractor`, `score` flows through it, `arithmetic` abstains on low-confidence
  money fields, and `eval/extraction.py` (`eval-extract`) ranks extractors on field
  accuracy vs the WildReceipt oracle. The first real candidate, **Qwen2-VL-2B-Instruct**
  (`extractors/vlm_qwen.py`, apache-2.0), is registered for the IMAGE route and scores
  **macro 0.725** field-accuracy (vendor 0.880 / date 0.915 / subtotal 0.740 / tax 0.614 /
  total 0.598 / line_count 0.602) on 100 real receipts, 0 extractor errors. **Next (heavy
  deps):** a second candidate — OCR+KIE (PaddleOCR PP-Structure / docTR, Apache-2.0) —
  benchmarked head-to-head via `eval-extract`; surface per-field confidence from the VLM
  to arm the guard; then re-run `eval-real` to measure arithmetic's *true* FP rate on
  faithfully-extracted fields.
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

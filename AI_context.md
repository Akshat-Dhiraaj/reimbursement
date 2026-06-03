# AI_context.md — slipguard

> Handoff doc for transferring this project to a fresh AI session or engineer.
> **Keep this file updated every milestone** (status, architecture, roadmap).
> Last updated: 2026-06-02 (M2 + real-data FP audit; M2.5 extractor interface + confidence
> guard + extraction-accuracy benchmark + first real VLM extractor (Qwen2-VL-2B) benchmarked
> + shared US/EU money parser that fixed an oracle ground-truth bug; image-EXIF provenance
> forensics on the IMAGE route; PDF-route field extractor (pypdfium2) — born-digital PDFs
> now score end-to-end through arithmetic/date_sanity/duplicate, not provenance-only;
> two more real corpora (CORD CC-BY-4.0 + ExpressExpense MIT) behind a `--corpus`
> selector — CORD's oracle FP audit corroborates the WildReceipt finding on a second,
> Indonesian-locale corpus; **learned logistic fusion (#62) measured against noisy-OR —
> cuts real-corpus false positives 0.175→0.042 at matched fraud-recall by down-weighting the
> noisy arithmetic signal; noisy-OR stays the zero-training default, learned is opt-in**;
**lightweight provenance/container forensics (M3, in progress): PDF `/Prev` content-edit
localization (#71), signature edit-after-signing (#72), and C2PA Content Credentials
AI-generation reading (#76) — provenance signals chosen over heavy pixel-AI detection, which
stays deferred (#60) as measured/researched hype under our constraints**).

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
  Dependency-free — raw bytes + regex over the literal Info dict. *(Later gained a
  second pikepdf layer for compressed PDFs — see the deep-forensics entry below.)*
- Synthetic PDF benchmark (`data/pdfsynth.py`; `build_pdf` writes a valid byte
  layout, no third-party dep) covering the three provenance tampers. New CLI
  `eval-pdf`. **31 passing tests** total.

PDF benchmark (synthetic, seed 0): structured detectors **abstain** on bare PDFs;
`pdf_meta` scores **AUC 1.0 / recall 1.0 / 0 FP** over 45 provenance frauds, which
fusion routes to **REVIEW** (provenance warrants a human look, not an auto-reject).
⚠️ Same caveat: this validates the layer on tampers that violate these exact signals.
The dependency-free Layer 1 does **not** decode xref-stream / compressed / XMP metadata —
on those PDFs the string fields read `None` while the `%%EOF` count (the incremental-update
signal) stays reliable. *(This blind spot is now covered by the optional pikepdf Layer 2 —
see the deep-forensics entry below.)*

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
uniformly (at this point PDF/IMAGE gave an honest "no extractor registered"; both routes
are wired in the later entries — IMAGE via VLM/docTR, PDF via pypdfium2 text).
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

**VLM confidence wired — arithmetic guard armed (2026-06-01):** the `arithmetic`
low-confidence abstain guard had been a dormant *mechanism* — synthetic/oracle fields carry
no confidence and the VLM emitted none, so it never fired on a real extraction. The VLM
extractor (`extractors/vlm_qwen.py`, `_to_receipt`) now records an honest **parse-completeness**
confidence for `line_items` under `Receipt.field_confidence`: the fraction of the items the
model *emitted* that we could actually parse an amount for. When the model emits items we
cannot all parse, the captured line-item sum is unreliable, so a low ratio (< the 0.5 floor)
makes `arithmetic` **abstain** instead of flagging a `subtotal ≠ Σlines` gap that is really a
capture artifact — exactly the under-capture FP cause the audit named. Honest limits (stated,
not hidden): it is a completeness score, not a calibrated probability, and sees
*emitted-but-unparseable* loss only — not items the model never emitted, nor a
cleanly-parsed-but-*mislabeled* scalar (no honest signal for those without ground truth or
token logprobs). Behaviour-preserving on good extractions: a fully-parsed receipt records
nothing, and empty `field_confidence` == fully trusted, so the synthetic/oracle leaderboards
are unchanged. Two end-to-end tests pin the behaviour: a low-ratio extraction makes
`arithmetic` abstain, while the *same* `subtotal ≠ Σlines` gap with a clean extraction is
still reported as fraud (the guard mutes only on low confidence — it is not a blanket
suppressor). **86 tests pass.**

**Image-EXIF provenance forensics — the IMAGE-route sibling of `pdf_meta` (2026-06-01):**
the IMAGE route now has its own metadata-provenance detector, `image_meta`
(`forensics/image.py` inspector + `detectors/imagemeta.py`), mirroring `pdf_meta` signal-for-
signal. Via Pillow it reads the EXIF `Software` tag (0x0131), the `DateTime` modify stamp
(0x0132), `DateTimeOriginal` capture stamp (0x9003 in the 0x8769 sub-IFD), and `Make`/`Model`,
then flags two tampers: an **image-editor in `Software`** (Photoshop / GIMP / Snapseed / Canva /
Photopea / … — the EXIF sibling of the PDF editor-tag) and a **`DateTime` ≫ `DateTimeOriginal`
capture-vs-modify gap** (the EXIF sibling of ModDate ≫ CreationDate). Crucially it **abstains on
missing EXIF**: stripped / screenshot / AI-generated images carry none — but so do many
legitimate shared receipts, so absent metadata is *not* evidence of fraud; and because EXIF is
strippable/forgeable, a clean read only weakly exonerates (clean Signal confidence 0.5). Pillow
lives only in the `[vlm]` extra, so the detector lazy-imports PIL and abstains (never crashes)
when it is absent, exactly like the VLM extractor. New synthetic image benchmark
(`data/imagesynth.py`, real EXIF-bearing JPEGs minted with Pillow — editor-tag and date-gap
fraud types, no incremental-update analog since EXIF has no honest counterpart) + `eval-image`
CLI. Benchmark (70 samples, seed 0): `image_meta` **AUC 1.000 / recall 1.000 / 0 FP** over 30
EXIF tampers; the structured detectors and `pdf_meta` correctly **abstain** on the IMAGE route.
No oracle-FP regression: WildReceipt oracle `Receipt`s set `source=IMAGE` but no `source_path`,
so `image_meta` abstains on them. **100 tests pass.**

**Re-extracted-fields FP audit — the surprising apples-to-apples result (2026-06-01):** with
the VLM registered, `eval-real --extractor vlm --limit 100` re-reads each legitimate receipt's
fields straight from its image (replacing the lossy oracle KIE) and audits the detectors on
*faithfully-extracted* fields with the confidence guard live. To compare fairly, `--limit` now
also subsets the **oracle** path to the *same* first-100 image-bearing receipts (new shared
`eval/audit.image_bearing`, used by both paths), and both runs pin `--today 2019-12-31`. The
result is counter-intuitive and worth internalizing: **re-extraction makes the fused FP go UP,
not down — oracle 0.400 → VLM 0.740 on the identical 100.** Why: (1) **coverage cuts both ways**
— the VLM reads far more fields than the oracle (date 93 vs 47, subtotal 96 vs 77, total 96 vs
92 of 100); the oracle "abstains by omission" on a missing field, while the VLM almost always
supplies a value, so a <100%-accurate value almost always gets *checked*. (2) **per-field
accuracy ≠ cross-field arithmetic consistency** — a subtotal-to-sum reconciliation needs *every*
line item correct, so one misread line breaks the whole receipt (`subtotal≠Σlines` 15→50), and
`line≠qty·price` (0→18) appears only for the VLM, which emits real qty×unit values (the oracle's
qty=1/unit=amount rows can't trip it); `date_sanity` flags also rise 17%→50% on VLM year-misreads
the oracle never makes. (3) The parse-completeness guard abstains only **4%** — it catches
emitted-but-*unparseable* items, not a *confidently-wrong* read, which is the dominant failure.
**This refines the earlier "faithful extractor is the binding constraint" takeaway:** the
constraint is sharper — a faithful extractor **plus per-value confidence**; coverage alone makes
FP worse, and the oracle's lower FP was partly an artifact of its *missing* fields (you cannot
mis-reconcile what you never read). Directly motivates the still-open scalar-misread confidence
work. (Reproduce: `eval-real --extractor vlm --limit 100 --today 2019-12-31` vs `eval-real
--extractor oracle --limit 100 --today 2019-12-31`; VLM = Qwen2-VL-2B-Instruct.)

**Scalar-misread confidence — token logprobs, and the honest verdict (2026-06-01):** the
re-extraction audit named *confident scalar misreads* as the dominant FP cause that
parse-completeness cannot see, so the VLM extractor now also emits a **per-token-logprob**
confidence for the scalar money fields. The same greedy decode that produces the value also
yields, via `compute_transition_scores`, the model's probability for each token it chose;
`_field_confidence_from_tokens` aligns those to each scalar's digits (incremental token→char
spans) and records the **least-confident digit** (min) under `field_confidence[subtotal|
tax_amount|total]`. It is **free** — one greedy pass, no extra inference (the discarded
alternative, K× self-consistency sampling, cost K× and a 5-receipt smoke showed it abstained
**0%** because the misreads are *stable*, not wavering). **Measured (GPU smoke, WildReceipt
test):** the mechanism records confidence exactly as designed, but the VLM's misreads are
*confident* — on three known-misread receipts the per-field min token-probs were subtotal
0.77–0.84, tax 0.74, total **0.59–0.985**, every one above the 0.5 abstain floor — so
`arithmetic` does not additionally abstain and the FP is unchanged at the principled
threshold. This is the **honest limit confirmed for BOTH signals**: self-consistency misses
*stable* misreads, logprobs miss *confident* ones, and the WildReceipt errors are both. The
payoff is therefore a **calibration-ready per-value confidence** (the substrate for learned
fusion, M3/#62), not an unsupervised 0.5-threshold win; reducing FP via a threshold needs
calibration against labeled misread/fraud data — a confidence-vs-oracle-correctness study is
the clear next step. The value is always the deterministic greedy read, so field accuracy
(macro 0.725) is unchanged. **140 tests pass** (+8 pure logprob-alignment + guard-arming
tests, replacing the 8 self-consistency tests that are gone with the mechanism).

**Confidence calibration study — the per-value signal IS informative (2026-06-01):** the
token-logprob entry above closed at "does not lower FP at the 0.5 floor," which left the real
question open — is the confidence *uninformative*, or is 0.5 just the wrong threshold? The new
`eval/calibration.py` (`slipguard eval-calibration`, reusing the extraction-eval oracle-pairing
loop + `roc_auc`) answers it on the same 100 image-bearing receipts (222 scored money reads vs the
WildReceipt oracle). **It is the wrong threshold, not a useless signal: AUC 0.758** that a read
disagrees with the oracle (subtotal 0.831, tax 0.766, total 0.795), with a **monotonic reliability
curve** — accuracy 0.366 (conf <0.6) → 0.579 → 0.593 → 0.710 → 0.869 ([0.9, 1.0)) → 1.000 (==1.0) —
and mean confidence **0.860 on correct reads vs 0.691 on misreads**. The 0.5 floor catches only
**18%** of misreads (so the earlier "misreads are confident" read was threshold-specific and
under-sold the signal); a higher cut does much better but never for free — the abstain sweep trades
misread-recall for dropped-correct reads (T=0.7: 51% caught, 17% of good reads dropped; T=0.6: 39%
vs 10%), and the arithmetic-breaking misreads skew confident. **Honest, refined verdict:** the
per-value confidence is a *genuine, measured* feature (AUC ~0.76) for the cost-aware learned fuser
(M3/#62), which weighs it against the cost of a needless abstain — not a hand-set unsupervised
floor. (Reproduce: `eval-calibration --extractor vlm --limit 100`.) **151 tests pass** (+11
calibration tests: pure AUC/reliability/sweep math + a stub-extractor loop, incl. the AUC=0.5
"confident-misread" case).

**PDF-route field extractor — born-digital PDFs now score end-to-end (2026-06-01):** the
biggest coverage gap closed. Until now a PDF got **provenance forensics only** (`pdf_meta`):
nothing read its fields, so the robust lead detectors (`arithmetic` / `date_sanity` /
`duplicate`) **abstained on every PDF** — a born-digital invoice with hand-edited totals
sailed through. New `extractors/pdf_text.py` (`PdfTextExtractor`, PDF route) reads the PDF's
**embedded text layer** via **pypdfium2** (Apache-2.0 / BSD-3-Clause, bundles Google's PDFium;
no copyleft, no AGPL — we deliberately avoid PyMuPDF) and feeds the positioned lines to the
**same KIE the docTR OCR route uses**. To make that sharing real, the keyword/position KIE was
extracted out of `doctr_ocr.py` into a paradigm-agnostic `extractors/kie.py` (the `Line`
contract + `receipt_from_lines`); both readers now map onto a Receipt through one tested layer,
and `receipt_from_lines` carries `source`/`source_path`/`image_path` so docTR sets IMAGE+image
and the PDF route sets PDF (no image → image-EXIF forensics correctly abstain). Confidence is a
constant **1.0** (born-digital text is *exact* — the characters the producer wrote, not a
recognition guess), so `field_confidence` stays empty (== fully trusted) and the arithmetic
guard behaves exactly as on the oracle path.
**Measured (`eval-pdf-extract`, 40 synthetic born-digital receipts, seed 0):** macro
field-accuracy **0.992** — date / subtotal / tax / total / line_count **1.000**, vendor
**0.950** (38/40), **0 extractor errors**. The money/date/line fields are perfect because the
text layer is exact: there is **no reading error to make**, only KIE *labelling* to get right.
The two vendor misses are both `"Croma"` (5 letters) out-lettered in the top-band letter-count
tie by an item line (`"Coffee …"` / `"Stapler …"`) that lands in the top 35% on a short
receipt — a known limit of the shared positional vendor heuristic (the KIE's own caveat), **not
a pypdfium2 read failure**. End-to-end proof: `score <born-digital.pdf>` now prints active
`arithmetic` ("all arithmetic reconciles") and `date_sanity` signals alongside `pdf_meta`,
where before only `pdf_meta` ran. **Honest limitation (stated, not hidden):** this reads the
embedded **text layer only** — a scanned-image PDF (a photo wrapped in a PDF, no text layer)
yields an empty Receipt and belongs on the IMAGE route (rasterise-then-OCR is future work).
Born-digital PDFs — the half of the threat model this route is for — do carry a text layer.
The synthetic corpus (`data/pdfsynth.generate_pdf_extraction`, a `build_text_pdf` writer with a
real Helvetica font resource so PDFium can extract text) is the PDF analogue of the WildReceipt
oracle; a real born-digital corpus is future work. **162 tests pass** (+11: pypdfium2 rect→Line
geometry mapping, the PDF-route binding, the synthetic renderer, and a real round-trip proving a
minted PDF reads back into the right Receipt fields). (Reproduce: `eval-pdf-extract --n 40`.)

**Deeper PDF forensics — the pikepdf deep layer (2026-06-01).** The headline finding: the
dependency-free byte scanner is **blind by construction on modern compressed PDFs** (PDF 1.5+
keeps the Info dict in an object stream, metadata only in XMP — the real ERP/portal-export
shape), so on a minted *compressed* corpus `pdf_meta` Layer 1 scores **recall 0.000**. Added
**Layer 2** — `forensics/pdf.inspect_pdf_deep` via **pikepdf** (optional `[pdf-forensics]`
extra, **MPL-2.0**, a qpdf binding ~3 MB — deliberately *not* AGPL PyMuPDF). It decodes
object/xref streams + XMP, so it **recovers the editor tag / date gap Layer 1 misses on
compressed PDFs**, and surfaces structural anomalies a flat receipt never has: a fillable
**AcroForm**, embedded **JavaScript**, an auto-run **OpenAction**, and cover-and-relabel
**overlay annotations** (FreeText/Redact/Stamp/Square/Highlight). The import is lazy + gated by
`pikepdf_available()`; `inspect_pdf_deep` returns `None` (never raises) when the extra is absent
or the file won't open, so `pdf_meta` transparently falls back to Layer 1 and the dependency-free
core is unchanged. `pdf_meta` gained a `use_deep` knob (force Layer-1-only even with the extra
installed) — the new `eval-pdf-forensics` CLI flips it to score the **same** corpus through each
layer. New synthetic corpus `data/pdfsynth.generate_pdf_deep` (+ `build_compressed_pdf`, a
pikepdf-minted compressed writer). **Measured (`eval-pdf-forensics`, 80 compressed PDFs, seed 0):**
`pdf_meta` target-recall **0.000 → 0.833** byte-only→deep, **0 FP**, **fused recall 1.000** (all
60 frauds → REVIEW; the 10 AcroForm-only frauds score 0.45, below the 0.5 standalone bar but above
the 0.4 review bar — correctly a "look", not an auto-flag). The original uncompressed `eval-pdf`
corpus is **unchanged** (recall 1.0): the deep layer adds nothing where Layer 1 already reads the
metadata. **Deferred (honest scope):** a *text-over-scan* flag — it collides with legitimate
OCR'd/searchable scans (high FP), so it belongs in the M3 pixel/layout route as a *weak* signal,
not a structural flag. **175 tests pass** (+13: blind-spot recovery, each structural flag, the
`use_deep` knob, and the byte-vs-deep harness contrast — all skip-gated on `[pdf-forensics]`).
(Reproduce: `pip install -e ".[pdf-forensics]"` then `slipguard eval-pdf-forensics`.)

**Real corpora — CORD + ExpressExpense, and a second-corpus corroboration of the FP finding
(2026-06-01).** Broadened the real-receipt evidence base beyond WildReceipt (US English) with two
more commercial-safe corpora, behind a uniform `--corpus {wildreceipt,cord,expressexpense}`
selector (+ `--path` / `--split`) on `eval-real` / `eval-extract` / `eval-calibration`.
**CORD** (`data/cord.py`, naver-clova-ix/cord-v2, **CC-BY-4.0**) is the second *oracle* corpus:
its human `gt_parse` (menu / sub_total / total) reconstructs a `Receipt` without OCR, exactly like
WildReceipt's KIE. The mapping is **pure** (`_receipt_from_gt` on hand-built dicts — unit-tested,
no network; the HF `datasets` fetch is lazy and cached under git-ignored `datasets/cord`). Honest
scope, by construction of the labels: CORD's gt_parse carries **no store name and no date**, so the
oracle leaves `vendor_name="(unknown)"` / `date=None` (vendor-less + `date_sanity` abstain), and
currency IDR / country ID makes the GSTIN detector abstain — CORD exercises the **money/arithmetic**
reconciliation specifically. Faithful mapping quirks the audit forced out (each pinned by a test):
net a line-level `discountprice` (CORD lists a discounted item as a duplicate line; without netting
it the line double-counts), use `unitprice × cnt` when a row has no `price`, and flatten one level
of `sub` add-on rows (they count toward the subtotal). **ExpressExpense** (`data/expressexpense.py`,
**MIT**, 200 restaurant receipt jpgs — verified 200 discovered on the real download) ships **images
only, no field labels**, so it can feed the FP audit *only* via real-extractor re-extraction
(`eval-real --extractor vlm/doctr`); it is **rejected** by `eval-extract` / `eval-calibration`
(which need an oracle), and its loader simply globs image suffixes recursively.
**Measured CORD oracle FP audit (`eval-real --corpus cord --limit 100`, reproduced):** fused FP
**0.170 (17/100 — 9 review, 8 reject)**, and it is **entirely** the `arithmetic` detector
(`tax_id` / `date_sanity` / `pdf_meta` / `image_meta` all **100% abstain**, as they should on a
no-tax-id / no-date / IMAGE-route corpus; `duplicate` never flags). Cause breakdown:
`total != subtotal+tax` ×15 and `subtotal != Σlines` ×2. **The non-obvious, honest finding:**
WildReceipt's 0.364 was driven by *lossy KIE extraction* (mislabeled total box / under-captured
lines), but CORD's gt_parse is **clean structured truth** — so its FP is a different mechanism: our
**3-field (subtotal / tax / total) `Receipt` model is too narrow** to represent receipts whose total
legitimately includes a **service charge / discount**, or whose menu prices are **tax-inclusive**
(confirmed on cord-test:48 — menu line 25000 is tax-inclusive, subtotal 22728 is the pre-tax base,
22728 + 2272 ≈ 25000). So a *second, Indonesian-locale* corpus reaches the **same conclusion by a
different route**: the binding constraint is **representation/extraction completeness, not detector
logic** — and it names a concrete model-improvement target (a richer Receipt carrying
service-charge / discount / tax-inclusive fields) that feeds the M3 fuser work (#62). We deliberately
**do not** suppress these flags: they are honest limitations of the data model, not detector bugs.
**A money-parser bug CORD exposed + fixed** (same class as the earlier EU-decimal 100× bug): the
Indonesian `Rp.` currency-prefix dot fused with the digits, and the regex's leftmost bare-decimal
branch read `Rp.118.000` as `118.0` — a **1000× error**. Fixed `money._TOKEN_RE` with a negative
lookbehind (`(?<![\w.])`) so a dot preceded by a letter/digit can't start a decimal and the digit-led
group wins, while genuine bare decimals (`.70`, `$.70`) still parse. **187 tests pass** (+7 CORD
pure-mapping, +4 ExpressExpense glob/scope, +1 `Rp.`-prefix money regression). (Reproduce:
`slipguard eval-real --corpus cord --limit 100`; CORD downloads on first run via the `[vlm]`
`datasets` lib.)

**Learned fusion measured vs noisy-OR (#62, done).** `fusion_learned.py` adds a transparent,
dependency-free **logistic-regression fuser** over the *same* per-detector confidence-weighted
signals noisy-OR consumes (one feature per detector = `signal.weighted`, by detector name; an
abstainer's weight is 0, so it's noisy-OR's inputs with *learned* per-detector weights instead of an
implicit equal weight). `Fuser` gained a pluggable `combiner` (default `None` ⇒ noisy-OR, unchanged);
`decide`/`verdict`/thresholds are shared, so only the score combination differs. `eval/fusion_bench.py`
(`eval-fusion`) fits it on **synthetic fraud = positives + synthetic-clean & real-legitimate
(WildReceipt+CORD) = negatives**, then measures on a *disjoint* split (synthetic seed 0 train / seed 1
test; the real corpora split into two halves) — no receipt is both trained and scored. **Measured
(reproduced, `slipguard eval-fusion`):** synth-fraud-vs-**real-legit** separation **AUC 0.867 → 0.990**,
and the **real-corpus false-positive rate at a matched synthetic fraud-recall drops 0.175 → 0.042
(~4×)**; the easy synth-fraud-vs-synth-clean case is essentially unchanged (1.000 → 0.996). The
**legible learned weights** explain *why*: the high-precision structural detectors earn the trust
(`tax_id +6.79`, `duplicate +6.27`, `date_sanity +5.11`) while the **noisy `arithmetic` signal is
down-weighted (+2.66)** — exactly the lever the FP audit predicted (arithmetic also fires on lossy
real extractions, so equal-weight noisy-OR over-counted it). **Honest caveats (loud):** (1) the
positives are **synthetic** fraud, so this measures *synthetic-fraud vs real-legitimate* separation,
**not real-fraud detection**; (2) `pdf_meta` / `image_meta` learned **weight 0.0** because the
structured/KIE training data carries **no PDF/image provenance examples** — so the fuser as fit here
is calibrated for the **structured/KIE route only** and must NOT replace noisy-OR on the PDF/image
routes without route-appropriate (provenance-bearing) training. Hence **noisy-OR stays the
zero-training default; learned fusion is opt-in and selected by these numbers** (and the
richer-`Receipt`-model work from #61 remains the complementary, bigger lever). **198 tests pass**
(+11: 10 learned-fuser unit + 1 fusion-bench smoke). (Reproduce: `slipguard eval-fusion`.)

**Lightweight provenance/container forensics (M3, in progress — #71/#72/#76).** Pushed
metadata/structure forensics further on both routes — the measured-and-researched alternative to
heavy pixel-AI detection. A current survey confirms pixel-AI is **hype under our constraints**:
naive FFT AI-detection degrades under recompression, ELA is FP-prone, and PRNU needs a per-camera
reference we never have for inbound third-party receipts — so **#60 stays deferred**. Each new
signal is a pluggable sub-signal behind the existing `pdf_meta` / `image_meta` detectors.
- **PDF `/Prev` object-diff (#71, dependency-free, Layer 1).** Beyond "an incremental update
  exists," it diffs the `/Prev` xref chain to localize *which* object an update rewrote, and flags
  a rewritten page **content stream** (displayed values edited after issuance) — distinct from a
  harmless metadata re-save. New `content_stream_edits` field + `_CONTENT_EDIT` signal + a
  `content_edit` synthetic fraud. **Measured (`eval-pdf`):** `pdf_meta` recall 1.0 / FP 0.0 / AUC
  1.0 on the enlarged byte corpus; content edits → REVIEW. Honest limits: **classic xref tables
  only** (a compressed xref-stream PDF can't be byte-localized → reported 0, never a false
  positive); a full "Save As" rewrite flattens history and defeats it.
- **PDF signature `/ByteRange` coverage (#72, dependency-free).** Detects content appended *after*
  a PDF was digitally signed (bytes outside the signature's `/ByteRange` = edit-after-signing);
  works on compressed PDFs too (the `/ByteRange` is always in the clear). Plus a real FP fix the
  signed fixtures surfaced — the deep layer was flagging *every* signed PDF as a suspicious
  fillable-form overlay; now a **signature-only AcroForm** (the benign cousin) is exempt.
  **Measured:** recall 1.0 / FP 0.0 on the `signature_tamper` corpus. Honest: **low real-world
  recall** (most receipts aren't signed — high-precision / low-recall). *Deferred:*
  producer-vs-structure (needs verified producer fingerprints — grouped with the JPEG-fingerprint
  work; both need real reference data, not fabricated tables).
- **C2PA / Content Credentials (#76, optional `[c2pa]` extra).** Reads a cryptographically signed
  provenance manifest: a `trainedAlgorithmicMedia` assertion (Firefly / DALL·E / Sora / Imagen) →
  **AI-generated/edited** (the one IMAGE signal that is a *trustworthy positive*, not a heuristic);
  a `digitalCapture` (Pixel 10 / Galaxy S25) → genuine capture (weak exoneration). Folded into
  `image_meta` (now aggregates **C2PA + EXIF**, abstaining only when neither is present;
  `forensics/c2pa.py`, lazy-imported). `c2pa-python` (MIT/Apache) measured at **261 MB** installed
  but CPU-only / network-free / ~ms-per-read. Honest: **high-precision / near-zero-recall** (sparse
  adoption, strippable; absence → abstain); C2PA detection is *deterministic schema parsing*, not
  an AUC, so it is validated by unit tests (schema-agnostic recursive parse) + a real-`Reader`
  integration test — **not** a synthetic benchmark (minting a *signed* fixture fights c2pa-rs's
  strict signing-cert profile and wasn't worth it).
- **Disjoint accounting (a correctness principle, applied twice).** A content edit IS an
  incremental update, and an edit-after-signing IS an incremental update — so these correlated
  sub-signals are counted **once**, not noisy-OR-compounded into a spurious over-escalation,
  keeping a single provenance defect at REVIEW ("metadata alone never auto-rejects").
- **New commercial-safe deps:** `c2pa-python` (MIT/Apache, `[c2pa]`); `fontTools` (MIT) approved
  for the next signal (#73, not yet used). **215 tests pass** (+17 since #62: 4 `/Prev` + 4
  signature + 9 C2PA). Remaining lightweight signals queued: **#73** PDF font/coordinate anomalies,
  **#74** JPEG quant-table fingerprinting (needs real reference data), **#75** EXIF thumbnail
  mismatch.

**Forensics done + the API paradigm + the scorecard (2026-06-02 — #73/#77/#79/#80).**
- **#73 content-stream overlay:** a minimal pikepdf content-stream interpreter (tracks CTM +
  text origins + white fills) flags a white rectangle drawn over *pre-existing* text —
  cover-and-relabel done IN the content stream (distinct from the annotation overlays #57
  catches). Low-FP by construction (a legit table-cell fill draws its rect *before* the text).
  The **font-substitution** half is **deferred** (a low-FP version needs embedded-font fixtures
  — our synth PDFs use base-14 non-embedded Helvetica — + real-PDF FP validation; same
  real-data bucket as #74/#75, so `fontTools` stays approved-but-unused).
- **#77 route-aware fuser:** closed the documented "learned fuser zeroes pdf_meta/image_meta"
  gap **without touching the fuser** — its features key by detector NAME and pdf_meta/image_meta
  gate by route, so **merging synthetic PDF + image provenance fraud into training**
  (`eval-fusion --multiroute`) populates their columns → **pdf_meta +6.98 / image_meta +6.30**
  (now the *highest* weights; were 0.000), arithmetic lowest +1.62, no inactive; real-legit AUC
  0.820 → 0.994.
- **#79 Groq hosted-VLM extractor = the API paradigm:** `extractors/groq_vlm.py` reuses the Qwen
  prompt + JSON→Receipt parser (DRY), **stdlib `urllib` (no new dep)**, reads `GROQ_API_KEY` from
  env (never stored), model `meta-llama/llama-4-scout-17b-16e-instruct`. **Measured (`eval-extract
  --extractor groq`): macro 0.847 (N=50, 0 errors)** vs local Qwen2-VL-2B **0.725** (N=100) /
  docTR 0.579 — a big hosted model reads markedly better. Two real findings: Groq's Cloudflare
  **1010-blocks default urllib/datacenter UAs** (→ browser UA, datacenter/CI only); the **free
  tier rate-limits** rapid batches (8/15 → 429) so we added **429 retry-with-backoff** (0/12
  after). A first N=12 run read an optimistic 0.947; the firmer N=50 is 0.847 (still < Qwen's
  N=100, so sample-sensitive) — the direction (hosted-big >> local-small) is robust.
- **#80 `SCORECARD.md`:** the per-task **pure-Python vs local-model vs API** matrix — the binding
  **gap is photo→field extraction** (pure-Python can't read pixels; local-small 0.725 is
  private+heavy-GPU; API-big 0.847 is accurate but egress+rate-limits); detection / fusion /
  PDF+structured extraction are well-served by pure-Python + light CPU libs. Recommends a
  **cascade**, with photo extraction the one local-vs-API fork (decided by data-egress policy).
- New deps this batch: **none** (`urllib` is stdlib). **224 tests pass** (+9: 3 content-overlay,
  1 multiroute-fuser, 5 Groq-offline). #74/#75 were later **measured-rejected on real data**
  (6/8 real receipts already libjpeg-standard ⇒ #74 would FP on most; 0/12 carry an EXIF
  thumbnail ⇒ #75 ~never fires); #60 stays deferred (GPU + hype).

**Richer Receipt model — the biggest measured real-FP lever, done (2026-06-02 — #81).** Added
`service_charge` + `discount` to `Receipt`; `arithmetic` now reconciles **total == subtotal +
tax + service − discount** (both default 0 → plain 3-field receipts unchanged) and accepts
**tax-inclusive** line prices (Σlines ≈ subtotal OR subtotal+tax). The CORD loader maps gt_parse
`service_price` / `discount_price` (probed real CORD-test: ×12 / ×6). **Measured (`eval-real`):
CORD clean-oracle FP 0.170 → 0.030 (~5.7×); WildReceipt FP 0.364 → 0.324** (slight improvement,
no regression — the tax-inclusive relaxation also helped a few lossy WildReceipt cases). Closes
exactly the FP source CORD's clean-oracle audit isolated (3-field model too narrow:
total-with-service/discount ×15 + tax-inclusive-lines ×2). 3 CORD residuals remain (unlabeled
`etc` / big mismatches / line-level discounts — honestly **not** chased to 0; chasing risks
suppressing real signal). Monotonic-on-FP (the changes only RELAX), so synthetic tests are
unchanged. **229 tests pass.** (`.env` created for `GROQ_API_KEY`, gitignored + verified.)

**`REFERENCES.md` (#82) + the simple LLM-judge pipeline (#83) (2026-06-02).**
- **#82 `REFERENCES.md`** — sourced choices (what we picked over what, and why) with link-trust
  marking (✓ verified / canonical). Uncertain research-pass URLs were *fetched to verify* — the
  C2PA spec, arXiv FreqCross, Amped double-JPEG, the PRNU review and the Pretoria PDF-tamper article
  all resolved; two (infosecinstitute ELA, a defense.gov CSI PDF) returned 403 → **omitted**, not
  cited unverified. Linked from README.
- **#83 — a standalone SECOND pipeline:** `slipguard validate <image|pdf>` (`llm_validate.py`)
  intakes one document → **ONE hosted multimodal call (Groq or Gemini, auto-selected by which key is
  set)** → a structured JSON validity verdict (vendor / date / total / tax, `ai_or_edit_suspected`
  + signs, `date_valid`, `arithmetic_consistent`, `red_flags`, `decision` approve/review/reject,
  confidence, summary). **The model's instructions live in an editable external
  `prompts/validity_prompt.md`** (prompt-as-config — the user's explicit ask; no code change to
  re-tune). PDFs go native to Gemini / are rasterised via pypdfium2 for image-only Groq; stdlib
  `urllib` (no new dep), 429 backoff, browser-UA for Groq's Cloudflare, **fail-safe to `review`** on
  unparseable output. The prompt bakes in the honest stance (pixel AI-edit = review-not-proof;
  arithmetic accounts for service/discount/tax-inclusive). **Live Groq smoke (real receipt): clean
  verdict — CVS/pharmacy, total 9.73, arithmetic_consistent true, approve, conf 0.9.** Gemini path is
  code-complete but untested live (no Gemini key). This is the "just ask a big model" pipeline,
  alongside the lightweight/private detector-fusion one. **235 tests pass** (+6 offline; live by hand).
- **#84 DSPy — tested as a dev-time optimizer, measured NOT a win.** `dspy_optimize.py` (optional
  `[dspy]` extra; runtime stays DSPy-free) tunes the extraction prompt over the same Groq model,
  scored by the same field metric. Measured (WildReceipt, N=8): DSPy *zero-shot* **0.896** ≈ the
  hand prompt (0.847), but **BootstrapFewShot made it worse — 0.410, 4/8 errors** (few-shot demos
  for a vision task blow up the multi-image prompt). Confirms the bottleneck isn't prompt phrasing;
  kept the harness as a recorded dev experiment (see REFERENCES §1.12). **240 tests pass.**
- **#85 validate refined — deterministic cross-check, kept simple.** The `validate` pipeline no longer
  trusts the model's self-judged math. The prompt now also extracts subtotal/service_charge/discount/
  tax_id/country; `reconcile()` builds a Receipt from the LLM's OWN fields, runs `default_detectors()` +
  `Fuser` on it (+ the real file for byte-layer pdf_meta), then takes the **stricter** of {LLM decision,
  deterministic decision} — never relaxing (output gains `llm_decision`/`deterministic_decision`/
  `deterministic_reasons`). Patches the measured weakness (a VLM confidently mis-summing yet reporting
  `arithmetic_consistent: true`) for FREE — no new dep, no 2nd model call (one Receipt build + the
  existing detectors). `--llm-only` disables it. Live smoke (real CVS receipt): LLM approve +
  deterministic re-check (8.99+0.74=9.73 ✓ via the arithmetic detector) → approve; a broken total /
  future date / bad tax-id checksum would now escalate even on an LLM 'approve'. It's the "ensemble the
  two pipelines" idea in its cheapest form: LLM = perception, pure-Python = math/checksum/recency.
  **245 tests pass** (+5 cross-check tests).
- **#86–91 web UI — drag-and-drop receipt → Approved / Not approved + reasons.** A thin **FastAPI**
  backend (`src/slipguard/web/api.py`, the `[web]` extra: fastapi/uvicorn/python-multipart, all
  MIT/BSD/Apache) wraps `validate()`: `POST /api/validate` (multipart image/PDF → the shaped verdict)
  and `GET /api/health` (which provider is resolvable). No new fraud logic — it is transport over the
  existing pipeline, so the decision rules stay in one place; `validate()`'s blocking network call runs
  in a threadpool. The shaped response surfaces **both** the model's reasons and the deterministic
  cross-check's, tagged by source — the two-layer design made visible (on an approve the deterministic
  reasons read positively, which is the "show the reason" ask). A **React (Vite)** drag-and-drop UI in
  `frontend/` proxies `/api` to :8000 in dev; `npm run build` is served at `/` by `slipguard serve`, so
  production is one uvicorn port. Verified live end-to-end on a real WildReceipt receipt (Groq via
  `.env`): health ok, `POST /api/validate` → `approved:true`, fields extracted (CHF hotel receipt,
  50.65+3.85=54.50 ✓), deterministic re-check ran and agreed. **253 tests pass** (+8: pure shaping
  `_shape`/`_num` + TestClient health/validate/415/503, offline via monkeypatch).
- **#92–97 prompt refinement (MEASURED) + local LM Studio provider.** Built `eval-prompt`
  (`eval/prompt_eval.py` + `slipguard eval-prompt --prompts a.md b.md …`) to refine the prompt by
  numbers: runs `validate(cross_check=False)` over the WildReceipt oracle and reports field accuracy
  (vendor/date/subtotal/tax/total — same `_vendor_ok`/`_money_ok` as the extractor leaderboard) +
  decision distribution + the prompt's own arithmetic self-consistency; ranks the prompt files.
  Variants in `prompts/experiments/` (p1_sharpened, p2_terse, p3_reasoning). **Result (local
  gemma-4-e4b, N=12, 0 errors): P0 `validity_prompt` 0.937 = p2_terse 0.937 > p1/p3 0.855** (the gap
  is one date misread on an n=2 field — noise; P0 wins the tie on arith-agreement 1.0 vs .875).
  **Refinements did NOT beat the current prompt → KEPT P0 unchanged** (matches Gemini, P0=0.943 N=20).
  The binding wall is the free-tier DAILY quota — Groq ≈500k tokens/day (~200 vision calls; the
  "87-min hang" was per-minute throttle near the cap + buffered stdout, not a bug), Gemini = **20
  req/day** — so an 80-call cloud comparison can't finish in one day. **LM Studio (`--provider
  lmstudio`)** removes the wall: OpenAI-compatible at localhost:1234, no key, no quota, private; pass
  `--model` a loaded VISION model. Local reasoning/hybrid models (Qwen3/Gemma) emit hidden reasoning
  before the answer → LM Studio uses `max_tokens=4096` (1024 returned empty `content`); qwen3.6-35b
  won't stop reasoning (~57s/call), gemma-4-26b needs ~35 GB, **gemma-4-e4b works** (~37s/call).
  Resilience: `_MAX_BACKOFF=90` (a daily `retry-after` of hours can't hang the client), streaming +
  fail-fast (abort after 4 consecutive errors). Gemini now uses the `X-goog-api-key` header +
  `gemini-flash-latest`. **259 tests pass.**
- **#98–101 prompt iteration round 2 — the MODEL-MISMATCH lesson.** Error analysis (per-field
  truth-vs-pred on the current prompt, local gemma N=24) found the dominant fixable miss: the model
  NULLs tax/subtotal/total that are printed. Built `prompts/experiments/p4_complete.md` (P0 + "fill any
  printed amount; derive the single missing one from total=subtotal+tax when safe"). **Local N=24: P4
  0.910 > P0 0.876** (recovers tax/total nulls, zero regression). **But Groq production N=10: P0 0.958 >
  P4 0.940** — P4's *derive* rule broke a subtotal + dropped arith-agreement. The local win did NOT
  transfer → **kept P0**, confirmed best on BOTH APIs (Groq 0.958, Gemini 0.943). Takeaway: a weak local
  model is fine for cheap *exploration*, but prompt wins MUST be confirmed on the production API — the
  local model's null-problem doesn't exist on the strong APIs, so optimizing on it misleads.

## 4. Quickstart

```bash
# Python 3.10+ (dev box: 3.13). One runtime dep: python-stdnum.
pip install -e ".[dev]"          # editable install + pytest
# Optional, commercial-safe extractors (kept out of the core install):
#   pip install -e ".[vlm]"      # Qwen2-VL-2B end-to-end VLM extractor (also pulls Pillow for image_meta) — heavy
#   pip install -e ".[ocr]"      # docTR OCR + transparent keyword/position KIE — heavy
#   pip install -e ".[pdf]"      # pypdfium2 born-digital PDF text extractor (Apache/BSD) — light, CPU-only
#   pip install -e ".[pdf-forensics]"  # pikepdf deep PDF provenance/structure layer (MPL-2.0) — light, CPU-only
#   pip install -e ".[c2pa]"     # C2PA / Content Credentials reader for image_meta (MIT/Apache) — CPU-only but ~260 MB
#   pip install -e ".[web]"      # FastAPI web UI backend for the React drag-and-drop frontend (MIT/BSD/Apache) — light

python -m pytest                 # run tests (259)
slipguard eval                   # structured benchmark leaderboard
slipguard eval-pdf               # PDF-provenance benchmark leaderboard
slipguard eval-pdf-forensics     # compressed-PDF deep forensics: byte-only vs pikepdf recall (needs [pdf-forensics])
slipguard eval-image             # image-EXIF provenance benchmark leaderboard (needs Pillow, the [vlm] extra)
slipguard eval-real --corpus cord   # real-receipt FP audit; --corpus {wildreceipt|cord|expressexpense}
slipguard eval-extract           # rank IMAGE extractors (docTR vs VLM) on field accuracy vs the oracle; --extractor groq scores ONLY the hosted API (needs GROQ_API_KEY)
slipguard eval-pdf-extract       # rank PDF extractor(s) on field accuracy vs a synthetic oracle (needs [pdf])
slipguard eval-fusion            # learned logistic fuser vs noisy-OR (real-FP at matched recall + legible weights); --multiroute trains pdf_meta/image_meta too
slipguard score data/demo.json   # score one receipt JSON (see data/demo.json)
slipguard validate receipt.jpg   # SIMPLE LLM-judge pipeline (Groq/Gemini/LM Studio) + deterministic cross-check; --provider, --llm-only
slipguard eval-prompt --prompts prompts/validity_prompt.md prompts/experiments/p1_sharpened.md
                                 #   rank validity-prompt variants by MEASURED field accuracy vs the oracle; add `--provider lmstudio --model qwen/...` for LOCAL (no key/quota)
slipguard serve                  # web UI: FastAPI + built React frontend on http://127.0.0.1:8000 (needs [web]; run `cd frontend && npm run build` first)

# Fetch the real corpora for eval-real (not committed; all commercial-safe):
#   wildreceipt (Apache-2.0):
#     curl -L -o datasets/wildreceipt.tar https://download.openmmlab.com/mmocr/data/wildreceipt.tar
#     tar -xf datasets/wildreceipt.tar -C datasets
#   cord (CC-BY-4.0): auto-fetched + cached under datasets/cord on first run via the [vlm] `datasets` lib
#   expressexpense (MIT, images only — re-extraction audit, no oracle): unzip the SRD image set under
#     datasets/expressexpense/  (https://expressexpense.com/large-receipt-image-dataset-SRD.zip)

# Without installing, prefix module runs with the src path:
PYTHONPATH=src python -m slipguard eval
```

## 5. Architecture / data flow

```
raw input ──routing.route_path──> DocumentType {PDF | IMAGE | STRUCTURED}
                                        │
        extractor_for(route).extract  (STRUCTURED: structured; IMAGE: VLM/docTR; PDF: pdf_text)
                                        ▼
                                   Receipt (models.py)
                                        │
        ┌───────────── default_detectors() — each Detector.run(receipt) ─────────────┐
        arithmetic   tax_id      date_sanity  duplicate   pdf_meta         image_meta        (+ future:
        (reconcile) (GSTIN/VAT) (future date)(resubmit) (PDF provenance) (image EXIF prov.)  AI-image, tamper-loc)
        └───────────────────────────── list[Signal] ───────────────────────────────┘
                                        ▼
             Fuser.verdict  (noisy-OR risk by default, or a learned combiner; + Decision)
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
  fusion.py               Fuser: risk (noisy-OR default, or a pluggable learned `combiner`) + approve/review/reject; shared decide/verdict
  fusion_learned.py       LearnedFuser: dependency-free logistic regression over the same per-detector weighted signals; .fit / .explain (legible weights); opt-in combiner for Fuser
  llm_validate.py         the SIMPLE alt pipeline (separate from detectors): image/PDF -> one Groq / Gemini / local-LM-Studio call (external prompt; --provider) -> JSON validity verdict; stdlib urllib, 429 backoff (capped _MAX_BACKOFF), fail-safe to review. `reconcile()` then cross-checks the verdict against the deterministic detectors (Receipt from the LLM's OWN fields -> default_detectors+Fuser; stricter of {LLM, deterministic} wins, never relaxes; `cross_check=False` / CLI `--llm-only` to skip)
  dspy_optimize.py        DEV-TIME ONLY (optional [dspy] extra): DSPy prompt-optimizer experiment over Groq, scored by the eval/extraction metric; measured NOT a win (few-shot hurt vision extraction) — runtime stays DSPy-free
  cli.py / __main__.py    `slipguard eval` / `eval-pdf` / `eval-pdf-forensics` / `eval-image` / `eval-real` / `eval-extract` / `eval-pdf-extract` / `eval-calibration` / `eval-fusion` / `eval-prompt` / `score` / `validate` / `serve`
  web/api.py              FastAPI web UI backend (the [web] extra): POST /api/validate (wraps validate(), shapes Approved/Not-approved + source-tagged reasons) + GET /api/health; runs validate() in a threadpool; serves frontend/dist at / when built; loads a repo-root .env
prompts/validity_prompt.md  editable instructions for the `validate` pipeline (AI-edit/date/arithmetic/vendor checks + JSON schema; also extracts subtotal/service_charge/discount/tax_id/country so numerics get a deterministic re-check; prompt-as-config)
  extractors/
    base.py               Extractor ABC: handles / can_handle / extract(path) -> Receipt
    structured.py         StructuredExtractor: Receipt JSON -> Receipt (dependency-free)
    kie.py                shared keyword/position KIE: positioned Lines -> Receipt (used by docTR + pdf_text); pure, model-free
    vlm_qwen.py           VLM extractor (Qwen2-VL-2B default, apache-2.0); IMAGE route; lazy torch/transformers
    groq_vlm.py           Groq hosted-VLM extractor (the API paradigm): reuses the Qwen prompt + JSON→Receipt parser, stdlib urllib (no new dep), GROQ_API_KEY from env, 429 retry-backoff; IMAGE route
    doctr_ocr.py          OCR extractor (docTR det+reco, apache-2.0) -> flattens export into kie.Line; IMAGE route; lazy doctr/torch
    pdf_text.py           born-digital PDF text extractor (pypdfium2, Apache/BSD) -> kie.Line; PDF route; lazy pypdfium2
    __init__.py           registry: default_extractors / image_extractors (VLM+docTR) / pdf_extractors (pdf_text) / image_extractor_for_spec / extractor_for
  detectors/
    base.py               Detector ABC: applicable/prime/score, shared run(), _abstain()
    arithmetic.py         line items -> subtotal -> tax -> total reconciliation
    taxid.py              python-stdnum GSTIN (IN) + EU VAT, abstains if unsupported
    datesanity.py         future / implausibly-old dates (today injectable)
    duplicate.py          exact + fuzzy resubmission match; prime()-d with history
    pdfmeta.py            PDF provenance signal (byte inspect_pdf + pikepdf inspect_pdf_deep; use_deep knob): incremental updates + /Prev content-edit localization + signature edit-after-signing + editor/date + structural anomalies; disjoint accounting; PDF route only
    imagemeta.py          image provenance signal: C2PA Content Credentials (AI-generation) + EXIF editor/date (forensics.c2pa + forensics.image); abstains only when neither present; IMAGE route only
    __init__.py           default_detectors() — the canonical ranked set
  forensics/
    pdf.py                PDF provenance — L1 dependency-free bytes (%%EOF / editor / date gap + /Prev object-diff content-edit localization + signature /ByteRange coverage) + L2 pikepdf deep (compressed/XMP metadata + AcroForm[signature-only exempt]/JS/OpenAction/overlay; optional [pdf-forensics])
    image.py              image EXIF provenance inspector (Pillow; editor tag / capture-vs-modify gap)
    c2pa.py               C2PA / Content Credentials reader (optional [c2pa], MIT/Apache): classifies manifest digitalSourceType → AI-generated / camera / unknown; lazy c2pa-python; abstains on no manifest
  data/
    synth.py              synthetic structured clean+fraud generator (benchmark backbone)
    pdfsynth.py           synthetic PDF generators: build_pdf (provenance, 3 byte-layer tampers) + build_text_pdf/generate_pdf_extraction (born-digital field corpus, the eval-pdf-extract oracle) + build_compressed_pdf/generate_pdf_deep (pikepdf-minted compressed corpus for the deep forensics layer)
    imagesynth.py         synthetic image generator (real EXIF-bearing JPEGs) + 2 EXIF tampers
    wildreceipt.py        WildReceipt loader: KIE annotations -> Receipt (oracle extraction, no OCR; US English)
    cord.py               CORD loader (CC-BY-4.0): gt_parse menu/sub_total/total -> Receipt (2nd oracle; Indonesian/IDR, no vendor/date); pure mapping, lazy `datasets` fetch
    expressexpense.py     ExpressExpense loader (MIT): globs 200 receipt images, no labels -> image-only Receipts (re-extraction FP audit only)
  eval/
    metrics.py            dependency-free precision/recall/F1/AUC/FPR
    harness.py            evaluate() -> per-detector + fused Report (detector leaderboard)
    audit.py              audit_false_positives() -> FP report on a legitimate corpus (eval-real)
    extraction.py         evaluate_extractors() -> field-accuracy leaderboard vs an oracle, IMAGE (WildReceipt) + PDF (synthetic) routes (eval-extract / eval-pdf-extract)
    calibration.py        summarize_calibration() -> does per-value confidence predict a misread? AUC + reliability bins + abstain sweep (eval-calibration)
    fusion_bench.py       compare_fusion() -> learned logistic fuser vs noisy-OR: leakage-free split, synth-vs-real-legit AUC + real FP at matched recall + legible weights (eval-fusion)
    prompt_eval.py        evaluate_prompt() -> rank validity-prompt variants by field accuracy vs the oracle (cross-check OFF — measures the PROMPT) + decision dist + arithmetic self-consistency; streams + fails fast on consecutive errors (eval-prompt)
frontend/                 React (Vite) drag-and-drop UI: drop a receipt -> Approved / Not approved + reasons + extracted fields/checks; dev proxies /api to :8000, `npm run build` served by `slipguard serve`
tests/                    259 tests: detectors (incl. arithmetic service-charge/discount/tax-inclusive reconciliation), the simple LLM-judge `validate` pipeline (prompt load, provider resolution, PDF-to-Gemini-vs-rasterise, verdict parse + fail-safe-to-review) + its deterministic cross-check (verdict→Receipt mapping, stricter-verdict-wins, never-downgrade, --llm-only bypass, reasons surfaced), the DSPy optimizer's pure pieces (field metric, pred→Receipt mapping, bootstrap-metric pass/fail, signature builds), synth invariants, harness, pdf + image forensics (incl. /Prev content-edit, signature coverage, content-stream overlay), C2PA classify, loader, FP audit (+ image_bearing apples-to-apples), extraction + extraction-eval, money parser (incl. the `Rp.`-prefix 1000× regression CORD exposed), VLM parse-completeness + token-logprob scalar confidence (incremental token→char spans, min-over-digits, guard-arming) + docTR OCR-confidence guards, shared KIE/date/money units + row-merge (split two-column recovery), confidence calibration (roc_auc separation, reliability bins, threshold sweep, oracle-pairing), pdf_text (pypdfium2 rect→Line geometry, PDF-route binding, synthetic renderer, real born-digital round-trip), deep PDF forensics (compressed blind-spot recovery, JS/OpenAction/AcroForm/overlay structural flags, the use_deep knob, byte-vs-deep harness contrast — skip-gated on [pdf-forensics]), CORD oracle mapping (qty/line-amount/discount-netting/sub-flatten/locale, pure on hand-built dicts), ExpressExpense glob (recursive, sorted, images-only, limit, fetch-hint on missing dir), learned fusion (feature-vector ordering by detector name + abstain→0, logistic separates separable data, deterministic fit, the noisy-detector-down-weight mechanism, pluggable-combiner override + clamp + noisy-OR-unchanged guard, .explain magnitude-sort) + fusion-bench smoke (synthetic-only run, real columns→n/a, provenance detectors flagged inactive), and the web UI backend (_shape Approved/not-approved + source-tagged reasons, _num coercion; TestClient health / validate / 415-unsupported-type / 503-missing-key — offline via a monkeypatched validate(), skip-gated on FastAPI + httpx), the prompt-eval harness (field/decision/arithmetic scoring on canned verdicts, parse-fail + consecutive-error fail-fast — offline), the LM Studio local provider (no-model guard + validate() dispatch to the local caller), and the _post_json backoff cap (a huge retry-after is capped, then gives up)
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
  total 0.598 / line_count 0.602) on 100 real receipts, 0 extractor errors. The VLM emits
  **two** honest confidence signals into the same `arithmetic` abstain guard: a
  **parse-completeness** ratio for `line_items` (low parsed-fraction → abstain on under-capture,
  not "fraud") and a **per-token-logprob** confidence for the scalar money fields (subtotal /
  tax / total). Both ride the one greedy decode — free, no extra inference. Measured limit
  (honest): the logprob signal is correctly wired but does **not** lower FP at the 0.5 floor
  because the VLM's misreads are *confident* (min token-prob 0.59–0.99 on known-misread
  receipts); its value is as a calibration-ready per-value confidence for learned fusion (#62),
  not an unsupervised threshold win.
  **Re-extraction FP measured (done):** `eval-real --extractor vlm --limit 100` vs the oracle
  on the same 100 receipts → fused FP **0.400 → 0.740** (re-extraction makes it *worse* — see
  the apples-to-apples milestone above; per-field accuracy ≠ arithmetic consistency). **Second
  extractor benchmarked — docTR OCR+KIE (Apache-2.0, `extractors/doctr_ocr.py`):** two-stage
  text detection+recognition feeding a transparent keyword/position KIE, surfacing a genuine
  OCR recognition confidence into the same arithmetic guard. Head-to-head on the *same* 100
  receipts: **VLM macro 0.725 leads, docTR 0.579 → ship the VLM** (by the numbers). Honest,
  non-obvious finding (audit-your-own-code): a first naive single-line KIE scored docTR a
  misleading **0.244** — *not* an OCR failure (it read every amount; date ties at 0.915), but
  docTR emits each summary row's *label* and *right-column amount* as **separate** lines, so a
  same-line "keyword+money" rule read `SUBTOTAL`/`TOTAL` as money-less and the stray digit in
  `TAX1` as `1.0`. A transparent **row-merge** pre-pass (`_merge_rows`: rejoin same-height
  lines, x-ordered so the amount stays the right-most token) lifted docTR **0.244 → 0.579** with
  no new model; docTR is now *competitive on the arithmetic-driving money fields* (tax 0.600 ≈
  VLM 0.614; **total 0.696 > VLM 0.598**) and trails mainly on vendor (0.380 vs 0.880) and
  line_count (0.312 vs 0.602) — so the VLM's real edge is robustness-without-per-layout-KIE,
  not raw money accuracy. **Scalar-misread confidence (done, with an honest verdict):** the VLM
  now emits a per-token-logprob confidence for subtotal/tax/total (single greedy pass, free);
  measured outcome is that it does *not* reduce FP at the 0.5 floor because the misreads are
  confident (0.59–0.99) — useful as a calibration-ready signal for learned fusion (#62), not as
  an unsupervised threshold (see the changelog entry). **Confidence calibration study (done):**
  the per-value confidence separates correct-from-misread at AUC ~0.76 (a calibration-ready
  feature for #62, not an unsupervised-threshold win). **PDF-route field extractor (done):**
  `extractors/pdf_text.py` (pypdfium2, Apache/BSD) reads a born-digital PDF's text layer through
  the shared `kie.py`, so PDFs now score end-to-end through arithmetic/date_sanity/duplicate
  (was provenance-only); `eval-pdf-extract` scores it **macro 0.992** on a synthetic born-digital
  oracle (money/date/line_count 1.000, vendor 0.950 — exact text, so only KIE labelling can err;
  text-layer only, scanned-image PDFs still need the IMAGE/OCR route). **Real corpora (done):**
  two more commercial-safe real-receipt loaders behind a `--corpus` selector — **CORD**
  (CC-BY-4.0, a *second* gt_parse oracle: Indonesian/IDR, no vendor/date) and **ExpressExpense**
  (MIT, 200 images, labels-free → re-extraction audit only). CORD's oracle FP **0.170** is
  **entirely arithmetic** and, because its gt_parse is *clean* (not lossy like WildReceipt's KIE),
  it isolates a *different* root cause — our **3-field model is too narrow** for service-charge /
  discount / tax-inclusive totals — so a second corpus corroborates "FP = representation
  completeness, not detector logic" by a new mechanism (and names a richer-Receipt-model target).
  **Learned fusion (#62, done):** consumed exactly these findings — see the "Learned fusion measured
  vs noisy-OR" entry in §3. **Next:** (a) the richer `Receipt` model (#61's named target:
  service-charge / discount / tax-inclusive fields) — the complementary, bigger FP lever; (b)
  route-appropriate (provenance-bearing) training so the learned fuser can weight `pdf_meta` /
  `image_meta` instead of zeroing them; (c) image pixel forensics (#60, **deferred — GPU/CPU-intensive**).
- **M2 — PDF & metadata forensics (done):** `pdf_meta` ships incremental-update,
  editor-tag and creation/mod-date checks, dependency-free (Layer 1); `image_meta` ships the
  IMAGE-route sibling (EXIF editor tag + capture-vs-modify date gap, Pillow). **Layer 2 landed:**
  pikepdf (`[pdf-forensics]`, MPL-2.0) decodes xref-stream + compressed/XMP metadata (recovering
  the editor tag / date gap Layer 1 misses) and adds AcroForm/JS/OpenAction/overlay structural
  flags — byte→deep recall 0.000→0.833 on a compressed corpus. **Deferred:** text-over-scan
  (collides with legit OCR'd scans → M3 pixel route, not a structural flag); exiftool for richer
  image metadata (maker-notes, thumbnail mismatch) beyond Pillow's core EXIF. Commercial-safe.
- **M3 — Image route:** **lightweight provenance forensics landed** — C2PA / Content
  Credentials reading (#76) folded into `image_meta` (a signed AI-generation assertion is the
  one *trustworthy positive*). **Heavy pixel-AI detection stays deferred (#60):** a current
  survey confirms naive FFT AI-detection collapses under recompression, ELA is FP-prone, and
  PRNU needs a per-camera reference we never have for inbound receipts — so any pixel detector,
  if ever added, is a **calibrated weak signal**, never the gate. Remaining lightweight image
  signals queued: JPEG quant-table fingerprinting (#74, needs real reference data), EXIF
  thumbnail-vs-full mismatch (#75).
- **M3 — Real data + learned fusion:** *learned fusion done* — `fusion_learned.LearnedFuser`
  (a transparent logistic over the same per-detector signals) is measured to cut real-corpus FP
  **0.175→0.042 at matched fraud-recall** (`eval-fusion`), by down-weighting the noisy `arithmetic`
  signal; noisy-OR stays the zero-training default and learned is opt-in + route-specific (see §3).
  Remaining: real-*fraud* corpora (most real receipt sets are legitimate-only, so we train on
  synthetic positives today) and route-appropriate fuser training including provenance fraud.

## 9. Constraints & sources

**Licenses (commercial-safe only):** AVOID LayoutLMv3 (CC-BY-NC), DocTamper (NC),
FUNSD (NC), Surya (GPL). PREFER CORD/WildReceipt, LiLT, docTR, PaddleOCR, Qwen2.5-VL,
python-stdnum.

**Data gap:** labelled *fake*-receipt data is scarce — the only real public set is
*Find it again!* (~163 forgeries, research-only licence → blocked). We synthesise fraud by
perturbing clean receipts (`data/synth.py`) for the harness, and measure false positives on
real *legitimate* corpora wired as loaders behind `--corpus`: **WildReceipt** (Apache-2.0,
US English, KIE oracle), **CORD** (CC-BY-4.0, Indonesian, gt_parse oracle), and **ExpressExpense**
(MIT, 200 images, labels-free → re-extraction audit only).

**Key references:** GPT4o-Receipt & AIForge-Doc (AI-forged receipt benchmarks),
DocTamper (CVPR'23), TruFor/CAT-Net (tamper localization), Community Forensics /
C2P-CLIP (AI-image detection), Veryfi/AppZen/Resistant AI (commercial approaches:
metadata + cross-document intelligence).
```

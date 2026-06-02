# Design decisions — what we used, what we didn't, and why

This is the rationale log. It explains the choices a reviewer would otherwise have
to reverse-engineer: the architecture, the priority order, every library/model, and
every dataset — including the ones we **rejected** and why.

---

## 1. Architectural decisions

### 1.1 Pluggable detectors selected by a benchmark (not by opinion)
**Decision:** every method is an independent `Detector`; an eval harness ranks them
on a labelled set (AUC, per-subtype recall, FP) and *that* drives selection.
**Why:** fraud detection is empirical and adversarial — the "obviously best" method
is often not, and the mix will change as attacks evolve. A uniform contract +
leaderboard lets us add/remove methods cheaply and defend choices with numbers, not
intuition. **Alternative rejected:** a single monolithic classifier — opaque, hard
to attribute a decision to a cause, and hard to extend per-route.

### 1.2 Robust signals lead; pixel forensics is a *weak* signal, never the gate
**Decision:** priority order is (1) arithmetic/field consistency, (2) metadata /
provenance, (3) duplicate intelligence; AI-image / tamper-localization come later as
**calibrated weak signals**.
**Why (the key research insight):** on AI-*generated* receipts, pixel/AI-image
forensics score near-random (≈0.56 AUC in DocTamper-style evaluations) and degrade
further under screenshot/JPEG recompression; classic tamper-localizers (TruFor,
CAT-Net) drop to F1 < 0.5 on diffusion inpainting. Arithmetic/metadata/duplicate
signals read *content and provenance*, which survive recompression and are hard to
fake consistently. So we lead with those and never let a visual score gate a verdict
on its own. **Sources:** GPT4o-Receipt and AIForge-Doc (AI-forged receipt
benchmarks), DocTamper (CVPR'23), TruFor / CAT-Net; commercial practice
(Veryfi / AppZen / Resistant AI lead with metadata + cross-document intelligence).

### 1.3 Separate routes for PDF vs. image
**Decision:** route PDFs and photos apart at intake.
**Why:** their provenance evidence is different in kind — PDF *structure*
(incremental updates, producer tags) vs. image *EXIF / sensor* signals. One pipeline
would blur two unrelated forensic surfaces.

### 1.4 Abstention is a first-class signal
**Decision:** `confidence == 0` means "I have nothing to say" and is excluded from
fusion.
**Why:** we want to run every detector on every receipt without irrelevant ones
adding noise (a US receipt has no GSTIN; a photo has no PDF structure). Abstention
keeps the risk score driven only by detectors that actually apply.

### 1.5 Noisy-OR is the default; a learned fuser is opt-in and measured (done)
**Decision:** keep confidence-weighted noisy-OR as the zero-training **default**, and add a
transparent **learned logistic fuser** as an opt-in `combiner` — selected by measured numbers,
not by replacing noisy-OR wholesale.
**Why noisy-OR stays default:** it is transparent, needs no training data, and is a sensible
prior (independent fraud signals compound) — and it works on every route with no per-route data.
**Why a learned fuser earns its keep (now that we have the data):** the FP audit showed the
binding real-world cost is a *noisy* `arithmetic` signal that fires on lossy/edge-case
extractions of legitimate receipts, while the structural detectors are high-precision; noisy-OR
weights all six equally. `fusion_learned.LearnedFuser` is a logistic regression over the **same**
per-detector weighted signals (so it's noisy-OR's inputs with *learned* weights, and its weights
are inspectable via `.explain()`). **Measured** (`eval-fusion`, leakage-free split): it cuts the
real-corpus FP **0.175 → 0.042 (~4×) at a matched synthetic fraud-recall** and lifts
synth-fraud-vs-real-legit separation **AUC 0.867 → 0.990**, by down-weighting `arithmetic` (+2.66)
relative to `tax_id`/`duplicate`/`date_sanity` (+5 to +6.8) — exactly the lever the audit named.
**Why it is opt-in, not the new default (honest limits):** (1) it trains on **synthetic**
positives, so it separates *synthetic-fraud from real-legitimate* receipts, not real fraud; (2) a
detector absent from the training data (`pdf_meta`/`image_meta` on structured/KIE receipts) learns
weight 0, so the fitted model is **route-specific** — using it on the PDF/image routes needs
provenance-bearing training. So the learned fuser is the *measured-better* choice on the
structured/KIE route, while noisy-OR remains the safe cross-route default. **Next (see
[ROADMAP.md](ROADMAP.md)):** real-fraud positives + route-appropriate training, and the richer
`Receipt` model (the complementary, bigger FP lever from §3 CORD).

### 1.6 Extraction is a pluggable, separately-evaluated approach (like detectors)
**Decision:** the OCR/VLM extractor is not hard-wired — it implements an `Extractor`
contract mirroring `Detector` (`name`, `handles`, `extract(path) -> Receipt`), is
chosen by `extractor_for(route)`, and is **benchmarked head-to-head** on field-level
accuracy by `eval/extraction.py` (`slipguard eval-extract`) — scored against the
WildReceipt KIE labels as a ground-truth *oracle*, exactly as detectors are ranked by
the harness.
**Why:** the real-data audit showed arithmetic precision is capped by extraction
quality, so the extractor is itself a measured, swappable component — we want to
*rank* OCR+KIE vs VLM on numbers, not pick one by reputation. The extractor also
surfaces **per-field confidence**, which lets `arithmetic` abstain on a misread
instead of asserting fraud (the audit's recommended fix). **Alternative rejected:**
baking one extractor into the score path — opaque, unswappable, and it couples
detector quality to a single extraction choice.

### 1.7 Per-value confidence = token logprobs, and it's a calibrated *feature* not a gate
**Decision:** derive the scalar-field (subtotal/tax/total) misread confidence from the
VLM's **per-token logprobs** of the greedy decode (the least-confident digit's
probability), **not** from K× self-consistency sampling; and use it as an input to the
future learned fuser, **not** as a standalone abstain threshold.
**Why (the method):** logprobs are ≈free — the same greedy pass via `output_scores` +
`compute_transition_scores().exp()` — whereas self-consistency costs K× the GPU time, and
a smoke test showed it abstained on ≈0 receipts (it misses *stable* misreads). **Why (not a
gate):** a calibration study on 100 receipts (222 scored reads vs the oracle, `slipguard
eval-calibration`) measured the signal honestly — it separates correct-from-misread with
**AUC 0.758** (subtotal 0.831 / tax 0.766 / total 0.795) and a **monotonic reliability
curve** (accuracy 0.37 below 0.6 → 0.87 in [0.9,1.0) → 1.00 at full confidence), so the
signal is genuinely informative. But there is **no free-lunch threshold**: at the principled
0.5 floor it catches only 18% of misreads (they clear it), and raising the cutoff trades
misread-recall for dropped-correct reads (T=0.7 catches 51% but drops 17% of good reads).
**Conclusion:** it is a measured, calibrated per-value *feature* for the cost-aware learned
fuser (§1.5, M3), which is exactly the labelled per-detector signal that fuser needs — not an
unsupervised gate. **Alternative rejected:** self-consistency sampling (K× cost, misses stable
misreads) and a fixed unsupervised abstain threshold (no single cutoff helps, per the sweep).

### 1.8 Born-digital PDFs read their *text layer* (exact), via the same shared KIE as OCR
**Decision:** the PDF route's extractor (`PdfTextExtractor`, pypdfium2) reads a PDF's
**embedded text layer** — extracting per-line text rects and their geometry — and feeds them
into the **same** keyword/position KIE (`extractors/kie.py`) that the docTR OCR route uses,
rather than rasterising every PDF and OCR-ing it. Read text is exact, so its `Line.conf` is a
constant **1.0** (the arithmetic confidence-guard stays off — there's no misread to hedge).
**Why:** most reimbursement PDFs (ERP/portal exports: ERPNext, SAP, Tally) are born-digital
with a perfect text layer — OCR-ing them would *inject* error into data that is already exact,
and burn a GPU pass for nothing. Reusing the KIE was the point of factoring `receipt_from_lines`
out of docTR: one paradigm-agnostic label/position reader now serves OCR boxes *and* PDF text
rects, so a born-digital PDF scores **end-to-end** (arithmetic / tax_id / date_sanity / duplicate
all run on it — previously the PDF route had provenance forensics only, so the content detectors
never fired on a PDF). **Honest limitation:** a **scanned-image PDF** has no text layer → empty
read → it belongs on the IMAGE/OCR route. We don't paper over this with a silent
rasterise-then-OCR fallback; that's deliberate **future work** ([ROADMAP.md](ROADMAP.md)), and
until then such a PDF yields an empty Receipt rather than a fabricated one. **Alternative
rejected:** rasterise-every-PDF-then-OCR (slower, lossy on the 95% born-digital case, GPU-bound)
and a bespoke PDF field parser (would duplicate the KIE logic that OCR already needs).

---

## 2. Libraries / models / techniques

| Choice | Decision | Why this and not the alternative |
|---|---|---|
| **`python-stdnum`** (tax-id checksums) | ✅ Used | Authoritative GSTIN/VAT format+checksum, tiny, LGPL (fine as a dependency). Hand-rolling check digits is error-prone and pointless. |
| **Dependency-free PDF parser** (raw bytes + regex over the Info dict) | ✅ Used for v1 | Zero runtime deps, fully commercial-safe, and the highest-yield signals (`%%EOF` count, editor tags, date gap) are visible in plain bytes. |
| **pikepdf** (deep PDF forensics: xref-stream / compressed / XMP metadata + structure) | ✅ **In use** (Layer 2) | **MPL-2.0**, a qpdf binding (~3 MB wheel) — commercial-safe and deliberately **not** AGPL PyMuPDF. The byte regex (Layer 1) is blind on modern **compressed** PDFs (PDF 1.5+ keeps the Info dict in an object stream, metadata only in XMP) — exactly the real ERP/portal export shape. pikepdf decodes those, so it **recovers the editor tag / date gap Layer 1 misses** and adds structural anomalies (AcroForm, JavaScript, OpenAction, overlay annotations). Measured on a minted compressed corpus (`eval-pdf-forensics`): `pdf_meta` target-recall **0.000 → 0.833** byte-only→deep, fused **recall 1.000 / FP 0.000** — the recall the extra buys. Optional `[pdf-forensics]` extra (separate from `[pdf]` text extraction — forensics vs. field-reading are independent concerns); `pdfmeta` falls back to Layer 1 and never crashes when it's absent. |
| **Text-over-scan flag** (a text layer sitting atop a scanned page image) | ❌ **Deferred** (not a structural flag) | Tempting as a "retyped over the scan" tamper signal, but it **collides with legitimate OCR'd / searchable scans** (every "Scan to searchable PDF" output is text-over-image) → unacceptable false positives as a hard structural flag. The honest home for it is the M3 pixel/layout route as a *calibrated weak* signal, not the provenance layer. Skipped deliberately, with the reason recorded. |
| **pypdfium2** (born-digital PDF *text-layer* read for the PDF extraction route) | ✅ **In use** | **Apache-2.0 / BSD-3-Clause** (bundles Google's PDFium, BSD-3-Clause) — fully commercial-safe, CPU-only, light. Returns per-line text rects + page geometry, which map cleanly into the shared KIE's `Line` contract. Kept an optional `[pdf]` extra so the core install stays dependency-free; the extractor reports it missing via `available()`. First measured result: round-trip **macro 0.992** on minted text PDFs. |
| **PyMuPDF / fitz** (the popular PDF text/render lib) | ❌ **Avoided** | **AGPL-3.0** (or a paid commercial licence) — copyleft that would reach a shipping internal product. pypdfium2 gives the text-layer read we need under a permissive licence, so the AGPL risk is simply unnecessary. |
| **Noisy-OR fusion** | ✅ Default | Zero-training, transparent, cross-route. See §1.5. |
| **Learned logistic fuser** (`fusion_learned.py`) | ✅ **In use (opt-in, measured)** | A transparent logistic regression over the **same** per-detector weighted signals as noisy-OR (hand-rolled GD, dependency-free, weights inspectable via `.explain()`). Measured by `eval-fusion` on a leakage-free split: real-corpus FP **0.175 → 0.042 (~4×)** at matched synthetic fraud-recall, AUC **0.867 → 0.990**, by down-weighting the noisy `arithmetic` signal. Opt-in (not the default) because it trains on **synthetic** positives (synth-fraud-vs-real-legit separation, not real-fraud detection) and is **route-specific** (provenance detectors learn weight 0 on structured data). See §1.5. |
| **LayoutLMv3** (document KIE) | ❌ Avoided | Licence **CC-BY-NC** — non-commercial; unusable in a shipping product. |
| **Surya** (OCR) | ❌ Avoided | **GPL** — copyleft; incompatible with a closed internal product. |
| **docTR** (OCR + transparent keyword/position KIE) | ✅ **In use** (benchmarked 2nd) | **Apache-2.0**, commercial-safe. Landed as the OCR+KIE counterpoint to the VLM so the IMAGE route is *picked by numbers, not reputation*: two-stage text detection+recognition + a transparent same-row "keyword + money" KIE. On the **same** 100 receipts it scores **macro 0.579 vs the VLM's 0.725 → the VLM ships**, but the field read is the honest part — docTR is **competitive on the arithmetic-driving money fields** (tax 0.600 ≈ VLM 0.614; **total 0.696 > VLM 0.598**) and trails on vendor (0.380) + line_count (0.312). A first naive single-line KIE mis-scored it **0.244**: docTR's OCR read every amount correctly (date ties the VLM at 0.915), but it emits each summary row's *label* and *right-column amount* as **separate** lines, so a same-line rule read `SUBTOTAL`/`TOTAL` as money-less and the stray digit in `TAX1` as `1.0`. A transparent **row-merge** pre-pass (rejoin same-height lines, x-ordered so the amount stays right-most) lifted it **0.244 → 0.579** with no new model. KIE is still English-keyword heuristic (German *Netto/MwSt/Summe* miss). PaddleOCR/PP-Structure remains a future candidate, same contract. |
| **Qwen-VL** (VLM extraction; default **Qwen2-VL-2B-Instruct**) | ✅ **In use** | Apache-2.0 *and* fits the 8 GB dev GPU natively. Licence verified per checkpoint against HF metadata: 2.5-VL-**7B** + 2-VL-**2B** are Apache-2.0; **2.5-VL-3B has no declared licence → rejected** (unclear = unusable, same posture as FUNSD/Find-it-again!). Loaded via transformers Auto classes so any HF VLM is a swappable candidate; ranked by `eval-extract` → first measured result **macro 0.725** field-accuracy on 100 real receipts (0 errors; vendor 0.880 / date 0.915 / money fields 0.60–0.74). |
| **Shared `money.parse_money`** (US/EU-aware money parser) | ✅ Used | One parser for **all** money strings — the WildReceipt + CORD oracles and the VLM extractor (DRY). A naive comma-stripper read European decimals (`Eur129,75`) 100× too high and corrupted *both* `eval-extract` and the FP audit; the fix treats the rightmost separator as the decimal point only with 1-2 trailing digits, else thousands grouping (`1,234.56`, `1.234,56`, `1,23,456.78`, `.70`). **CORD then exposed a sibling 1000× bug:** the Indonesian `Rp.` currency-prefix dot fused with the digits and the leftmost bare-decimal branch read `Rp.118.000` as `118.0`; fixed with a negative lookbehind (`(?<![\w.])`) so a dot preceded by a letter/digit can't begin a decimal, while genuine bare decimals (`.70`, `$.70`) still parse. Same lesson both times: a too-extreme number is a cue to audit the parser, not the data. |
| **CLIP / ViT AI-image detectors** (e.g. C2P-CLIP, Community Forensics) | 🔜 Planned as *weak* signal | Permissive; but per §1.2 only ever a calibrated weak input, evaluated honestly under laundering. |

---

## 3. Datasets — used and rejected (licence-driven)

Labelled *fake-receipt* data is scarce; the only public real-forgery set is small
and not clearly licensed. So we **synthesise** fraud by perturbing clean receipts for
the harness, and use **real legitimate** corpora to measure false positives.

| Dataset | Licence | Decision | Why |
|---|---|---|---|
| **WildReceipt** | Apache-2.0 | ✅ **In use** | Real receipts with human KIE labels → an *oracle extractor* (reconstruct `Receipt`s without OCR) to measure real-world FP rate (US English). Commercial-safe. |
| **CORD** (naver-clova-ix/cord-v2) | CC-BY-4.0 | ✅ **In use** | A *second oracle* corpus: Indonesian receipts whose human `gt_parse` (menu/sub_total/total) reconstructs a `Receipt` like WildReceipt's KIE. Adds locale variety **and** isolates a different FP cause — its clean labels showed the FP is a too-narrow 3-field model, not lossy extraction (see §4). No vendor/date in the labels (those detectors abstain); IDR/ID (GSTIN abstains). Lazy `datasets` fetch, git-ignored cache. **Measured oracle FP 0.170, entirely `arithmetic`.** |
| **ExpressExpense** | MIT | ✅ **In use** (re-extraction only) | 200 real receipt **images, no field labels** — so it can feed the FP audit *only* via real-extractor re-extraction (`eval-real --extractor vlm/doctr`), and is **rejected** by the oracle-scored `eval-extract`/`eval-calibration`. Broadens image-route variety; permissive (MIT). |
| **FUNSD** | CC-BY-**NC** | ❌ Rejected | Non-commercial. |
| **INV-CDIP** | CC-BY-**NC** | ❌ Rejected | Non-commercial. |
| **DocTamper** | Non-commercial | ❌ Rejected | NC licence; also tamper-localization which we treat as a weak signal only. |
| **"Find it again!"** (~163 real forgeries) | Unclear / research (L3i) | ⛔ Blocked | The *only* real forgery set, but licence is unclear/research-only — **do not use in the commercial build without written permission from L3i.** This is our central data gap. |
| **Future: IQline HR-provided receipts** | Proprietary | 🔒 Private | Real in-domain variety. **Never commit** — `datasets/` is git-ignored as the guard. |

---

## 4. Honesty posture (a deliberate decision)

Synthetic benchmark numbers (~1.0 fused AUC) are reported **only** as validation of
the harness and detector *logic* on fraud that violates these exact rules. Every doc
repeats this caveat. Real-world performance claims wait for real corpora and the
image/extraction routes. The real-data audit deliberately reports the unflattering
number (**0.364** FP) *with its true cause* (lossy extraction, not detector error)
rather than hiding it — because the cause is what tells us where to invest next. The
same discipline turned on our *own* ground truth: the first VLM run disagreed with the
oracle on European-decimal receipts, and on inspection the **oracle** was wrong (a
comma-stripping parser read `129,75` as `12975`), not the model — so we fixed the oracle
(shared `money.parse_money`), which is what moved the audit FP from 0.398 to 0.364. We
trust measured disagreement enough to audit the reference, not just the candidate. The
same discipline added a **second** real corpus precisely to test whether the WildReceipt
finding generalised: CORD (Indonesian, clean `gt_parse` labels) reports FP **0.170**, again
entirely `arithmetic` — but because its labels are *not* lossy, it isolates a **different**
true cause (our 3-field `subtotal/tax/total` model can't represent service-charge / discount /
tax-inclusive totals). Two corpora, two mechanisms, one conclusion — **the binding constraint
is representation/extraction completeness, not detector logic** — is a stronger, more honest
claim than either alone, and it names a concrete fix (a richer Receipt model) rather than hiding
the flags. (CORD also surfaced — and we fixed — a 1000× `Rp.`-prefix money-parser bug; §2.)

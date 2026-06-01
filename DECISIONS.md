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

### 1.5 Noisy-OR fusion now, learned fusion later
**Decision:** start with confidence-weighted noisy-OR; defer a learned/calibrated
fuser.
**Why:** noisy-OR is transparent, needs no training data, and is a sensible prior
(independent fraud signals compound). A learned fuser only earns its keep once the
harness has produced enough labelled per-detector performance to fit on — otherwise
we'd be fitting noise. **Plan:** replace it in M3 (see [ROADMAP.md](ROADMAP.md)).

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

---

## 2. Libraries / models / techniques

| Choice | Decision | Why this and not the alternative |
|---|---|---|
| **`python-stdnum`** (tax-id checksums) | ✅ Used | Authoritative GSTIN/VAT format+checksum, tiny, LGPL (fine as a dependency). Hand-rolling check digits is error-prone and pointless. |
| **Dependency-free PDF parser** (raw bytes + regex over the Info dict) | ✅ Used for v1 | Zero runtime deps, fully commercial-safe, and the highest-yield signals (`%%EOF` count, editor tags, date gap) are visible in plain bytes. |
| **pikepdf / pdfid** (xref-stream, compressed/XMP metadata, text-over-scan) | ⏳ Deferred | Needed to decode modern compressed PDFs the regex parser can't read, but adds a dependency and isn't required for the cheap wins. Scheduled, not skipped. |
| **Noisy-OR fusion** | ✅ Used now | See §1.5. |
| **Learned/calibrated fuser** | ⏳ Planned (M3) | Needs measured per-detector performance first; premature now. The first calibrated input now exists — the VLM's per-value confidence (AUC 0.758 vs oracle, §1.7) — which is exactly the kind of measured feature the fuser will consume. |
| **LayoutLMv3** (document KIE) | ❌ Avoided | Licence **CC-BY-NC** — non-commercial; unusable in a shipping product. |
| **Surya** (OCR) | ❌ Avoided | **GPL** — copyleft; incompatible with a closed internal product. |
| **docTR** (OCR + transparent keyword/position KIE) | ✅ **In use** (benchmarked 2nd) | **Apache-2.0**, commercial-safe. Landed as the OCR+KIE counterpoint to the VLM so the IMAGE route is *picked by numbers, not reputation*: two-stage text detection+recognition + a transparent same-row "keyword + money" KIE. On the **same** 100 receipts it scores **macro 0.579 vs the VLM's 0.725 → the VLM ships**, but the field read is the honest part — docTR is **competitive on the arithmetic-driving money fields** (tax 0.600 ≈ VLM 0.614; **total 0.696 > VLM 0.598**) and trails on vendor (0.380) + line_count (0.312). A first naive single-line KIE mis-scored it **0.244**: docTR's OCR read every amount correctly (date ties the VLM at 0.915), but it emits each summary row's *label* and *right-column amount* as **separate** lines, so a same-line rule read `SUBTOTAL`/`TOTAL` as money-less and the stray digit in `TAX1` as `1.0`. A transparent **row-merge** pre-pass (rejoin same-height lines, x-ordered so the amount stays right-most) lifted it **0.244 → 0.579** with no new model. KIE is still English-keyword heuristic (German *Netto/MwSt/Summe* miss). PaddleOCR/PP-Structure remains a future candidate, same contract. |
| **Qwen-VL** (VLM extraction; default **Qwen2-VL-2B-Instruct**) | ✅ **In use** | Apache-2.0 *and* fits the 8 GB dev GPU natively. Licence verified per checkpoint against HF metadata: 2.5-VL-**7B** + 2-VL-**2B** are Apache-2.0; **2.5-VL-3B has no declared licence → rejected** (unclear = unusable, same posture as FUNSD/Find-it-again!). Loaded via transformers Auto classes so any HF VLM is a swappable candidate; ranked by `eval-extract` → first measured result **macro 0.725** field-accuracy on 100 real receipts (0 errors; vendor 0.880 / date 0.915 / money fields 0.60–0.74). |
| **Shared `money.parse_money`** (US/EU-aware money parser) | ✅ Used | One parser for **both** the WildReceipt oracle and the VLM extractor (DRY). A naive comma-stripper read European decimals (`Eur129,75`) 100× too high and corrupted *both* `eval-extract` and the FP audit; the fix treats the rightmost separator as the decimal point only with 1-2 trailing digits, else thousands grouping (`1,234.56`, `1.234,56`, `1,23,456.78`, `.70`). |
| **CLIP / ViT AI-image detectors** (e.g. C2P-CLIP, Community Forensics) | 🔜 Planned as *weak* signal | Permissive; but per §1.2 only ever a calibrated weak input, evaluated honestly under laundering. |

---

## 3. Datasets — used and rejected (licence-driven)

Labelled *fake-receipt* data is scarce; the only public real-forgery set is small
and not clearly licensed. So we **synthesise** fraud by perturbing clean receipts for
the harness, and use **real legitimate** corpora to measure false positives.

| Dataset | Licence | Decision | Why |
|---|---|---|---|
| **WildReceipt** | Apache-2.0 | ✅ **In use** | Real receipts with human KIE labels → an *oracle extractor* (reconstruct `Receipt`s without OCR) to measure real-world FP rate. Commercial-safe. |
| **CORD** | CC-BY-4.0 | 🔜 Planned | Real receipts with parses; commercial-safe; adds variety for FP + extraction eval. |
| **ExpressExpense** | MIT | 🔜 Candidate | Real receipt images; permissive; useful once the image/extraction route exists. |
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
trust measured disagreement enough to audit the reference, not just the candidate.

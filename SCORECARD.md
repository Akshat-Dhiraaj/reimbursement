# Scorecard — pure-Python vs local-model vs API, *per task*

> **Goal:** find where each solution paradigm is **lacking**, by measured numbers — not
> opinion. Every accuracy figure below is reproducible from this repo's `eval-*` commands;
> the resource/privacy figures were measured as each component landed.

This is the synthesis behind the project's core bet: **lead with the lightweight, private
approach and only reach for a heavy model where the cheap one provably can't do the job.**
The scorecard makes "provably can't" concrete.

---

## What actually runs in the deployed product (honest status)

The benchmarks here measure *capabilities*. This is the separate, blunter question — **which of them
are wired into the live web UI / `validate` path, and which are demonstrated-but-not-deployed.** The
live path is `reconcile()`: the LLM judge's verdict, cross-checked by the deterministic detectors
(`deployed_detectors()`) fused with the hand-rolled noisy-OR `Fuser`.

| Capability | Live in the web UI / `validate`? | Notes |
|---|---|---|
| LLM judge (Groq / Gemini / LM Studio) | ✅ yes | needs an API key (or a local LM Studio model). The core. |
| `arithmetic` cross-check | ✅ yes | reconciles the LLM's *own* numbers. **Caveat:** the low-confidence *abstain guard* is inert here — the LLM path sets no per-field confidence, so arithmetic always asserts (the measured "extraction-misread FP" risk). |
| `tax_id` (GSTIN / VAT checksum) | ✅ yes | fires only when the LLM extracts a tax-id and the country is India or EU; abstains otherwise. |
| `date_sanity` (future / > 60-day) | ✅ yes | — |
| `pdf_meta` (PDF route) | ⚠️ partial | the dependency-free **byte layer** runs on the uploaded PDF; the **deep layer** (editor tag / structure on compressed PDFs) needs the `[pdf-forensics]` extra (pikepdf). |
| `image_meta` (image route) | ⚠️ extra-gated | does nothing without `[vlm]` (Pillow → EXIF) and/or `[c2pa]`; *abstains cleanly* when they're absent (no false signal). |
| **`duplicate` / resubmission** | ❌ **disabled** | RELATIONAL — needs a persistent store of prior submissions. None is wired, so it's excluded from `deployed_detectors()` (un-primed it would report "no match" *without ever checking*). Backend design in ROADMAP.md. Logic + benchmark stand ready. |
| noisy-OR fuser + UI score breakdown | ✅ yes | — |
| multi-key + cross-provider fallback | ✅ yes | needs the keys. |
| **Learned / calibrated fuser** (M3) | ❌ benchmark-only | the live path uses the hand-rolled noisy-OR `Fuser`; `LearnedFuser` is only exercised by `eval-fusion`. Deploying it needs a *trained-on-real-data* weight set (we have only synthetic-fraud-vs-real-legit separation). |
| Heavy extractors (Qwen-VL / docTR / pypdfium2 / Groq-VL) | ❌ not in this path | they power the `eval-extract` / `score` / `eval-real` **benchmarks**; the web / `validate` flow uses the LLM judge's *own* inline extraction, so these never run there. |

**Bottom line:** the deployed product is the **LLM judge + arithmetic / tax-id / date deterministic
cross-check + PDF byte-forensics**, plus EXIF/C2PA image-forensics *if* the extras are installed.
**Resubmission/duplicate detection and the learned fuser are built and benchmarked but not deployed** —
each needs an external dependency the product can't assume yet (a submission-history DB; a
real-fraud-labelled training set).

---

## The three paradigms

They are three sources of "intelligence," and the honest axes underneath are **capability
vs. weight vs. privacy vs. cost** — chosen *per task*, not for the whole pipeline.

| Paradigm | Capability | Weight | Privacy | Cost |
|---|---|---|---|---|
| **Pure Python** (we write the logic) | bounded to rules / arithmetic / structure | tiny, CPU, instant | fully local | free |
| **Local model / "GitHub repos"** | high (OCR, VLM, vision) | *splits*: heavy GPU models (torch, GB) vs light CPU libs (MB) | fully local | free (your compute) |
| **API key** (hosted inference) | highest (biggest models) | light client (here: stdlib only) | **image leaves the box** | $ + rate limits + network |

> "GitHub repos" is **not one bucket**: `pikepdf` (MPL, ~3 MB), `pypdfium2` (Apache, light),
> `python-stdnum` (tiny) and `fontTools` (MIT) are *light CPU libs* near the pure-Python end;
> Qwen-VL / docTR are *heavy GPU models*. `c2pa-python` is a light *API* (CPU, network-free)
> but a **heavy install** (261 MB native wheels). The scorecard separates these.

---

## Per-task scorecard (measured)

| Pipeline task | Best pure-Python | Best local model/lib | Best API | Verdict |
|---|---|---|---|---|
| **Detect arithmetic inconsistency** | `arithmetic` — recall 1.0 synth; real-corpus FP is the *extraction* artifact, not the logic | — | — | **Pure-Python wins** (free/private/instant) |
| **Detect PDF provenance tampering** | byte forensics: `%%EOF`, `/Prev` content-edit, signature `/ByteRange` — recall 1.0 / FP 0 synth | `pikepdf` deep layer recovers compressed-PDF metadata (recall 0.000→0.833); ~3 MB | — | **Pure-Python + 1 light lib** (private, cheap) |
| **Detect AI-generated/edited photo** | EXIF editor/date heuristics (weak; stripped easily) | `c2pa-python` Content Credentials — *trustworthy positive* but **near-zero recall**; 261 MB | — | **No lightweight high-recall option — a gap** (heavy pixel models are research-confirmed hype) |
| **Fuse signals → verdict** | hand-rolled logistic fuser (no sklearn) — real FP **0.175→0.042**; route-aware weights | — | — | **Pure-Python wins** (free, transparent, deterministic) |
| **Extract PDF → fields** (born-digital) | — | `pypdfium2` text layer — macro **0.992**; Apache, light, CPU | — | **Light local lib wins** (exact text; no model needed) |
| **Extract structured → fields** | `StructuredExtractor` — exact | — | — | **Pure-Python wins** |
| **Extract photo → fields** (the binding task) | **N/A — cannot read pixels** | Qwen2-VL-2B macro **0.725** (N=100); ~4.5 GB GPU, 7–15 s/receipt · docTR **0.579** | Groq llama-4-scout macro **0.847** (N=50, 0 err w/ backoff); stdlib client | **The gap lives here** — see below |

*Reproduce:* `eval` · `eval-pdf` / `eval-pdf-forensics` · `eval-image` · `eval-real --corpus …` ·
`eval-fusion [--multiroute]` · `eval-pdf-extract` · `eval-extract [--extractor groq]`.

---

## Where we're lacking (the honest gaps)

1. **Photo → field extraction is THE binding gap** (the FP audits already named extraction
   quality the constraint; this quantifies it across paradigms):
   - **Pure-Python can't do it at all** — there is no rules-based way to read a photo.
   - **Local small model (Qwen2-VL-2B): 0.725**, the *privacy-preserving* option, but
     accuracy-limited **and heavy** (needs a ~4.5 GB GPU, 7–15 s/receipt).
   - **API big model (Groq llama-4-scout): 0.847**, the most accurate **and** the lightest
     *client* (stdlib `urllib`, no GPU) — but the **image leaves the box**, and the free tier
     **rate-limits** (8/15 calls 429'd back-to-back until we added retry-with-backoff → 0/50).
     ⚠️ Measured on **N=50**, 0 errors (a first N=12 run read an optimistic 0.947 — the figure is
     sample-sensitive and still < Qwen's N=100); the *direction* (a large hosted model reads
     markedly better than a local 2B) is robust.
   - **So the missing capability is: a lightweight + private + accurate photo extractor.**
     Closing it means one of — a bigger *local* model (more accurate, heavier), accepting
     *API* egress (most accurate, lightest client, not private), or a **richer `Receipt`
     model + better KIE** (pure-Python) that reduces how much rides on extraction accuracy.
2. **High-recall AI-photo detection is a gap.** C2PA is the only *trustworthy positive*, but
   it is high-precision / near-zero-recall (sparse adoption, strippable). No lightweight
   high-recall AI-image detector exists — the strong 2025 detectors are GPU neural nets, and
   the cheap heuristics (FFT/ELA/PRNU) collapse under recompression. (#60 stays deferred.)
3. **Not gaps:** detection (arithmetic / tax-id / dates / duplicate / provenance), fusion, and
   PDF + structured extraction are all well-served by pure-Python + light CPU libs.

---

## What we tested and rejected on real receipts (measured)

Three "lightweight image-forensics" signals were investigated and **rejected by data** — a
direct demonstration of why this project leads with content (arithmetic / duplicate) + C2PA,
not pixel/metadata heuristics, on the IMAGE route:

- **JPEG quant-table fingerprinting (#74)** — premise: an editor's re-encode leaves
  libjpeg-standard quantization tables, unlike a camera's custom tables. **Probed real receipts:
  6 of 8 ExpressExpense images are *already* libjpeg-standard-encoded** (re-saved by
  sharing/processing pipelines). So "standard tables ⇒ re-encoded ⇒ suspicious" would
  false-positive on the *majority* of legitimate shared receipts (re-encoded ≠ fraud), and it
  misses Photoshop's proprietary tables. **Rejected** — not a low-FP signal on real data.
- **EXIF thumbnail-vs-full mismatch (#75)** — premise: a stale embedded thumbnail still shows the
  pre-edit image. **Probed real receipts: 0 of 12 carry an embedded EXIF thumbnail** (shared
  receipts have EXIF, and its thumbnail, stripped). ~0 applicability — it would essentially never
  fire. **Rejected** — not worth the fixture-minting effort.
- **PDF font-substitution (#73-font)** — an edited value drawn in a different font. **Not built:**
  real invoices routinely render amounts in a *different (monospace)* font from their labels, so
  "value font ≠ label font" is plausibly high-FP, and we have no real labeled-PDF corpus to
  validate it. Deferred to a real-data validation pass.
- **Pixel-level AI-image detection (#60)** — GPU-heavy and research-confirmed to collapse under
  recompression (naive FFT / ELA) or to need a per-camera reference we never have (PRNU). Deferred.

**The through-line:** real shared receipts arrive with metadata and encoding **laundered** by the
apps that move them, so lightweight *metadata/pixel* forensics on the IMAGE route are near-useless
or FP-prone. The signals that survive are **content** (arithmetic, duplicate) and **cryptographic
provenance when present** (C2PA) — exactly where the pipeline already leads.

## End-to-end fraud test: 100% recall, but an extraction-misread FP (measured)

Generated tampered receipts (`make-fakes --method pytamper` — Pillow overlays of an inflated total /
future date on the real `samples/`) and ran the full `validate` pipeline on **30 fakes + 15 clean**
(Groq):

- **Recall 30/30 (100%)** — every forgery flagged (caveat: crude, *obvious* overlays inflate this).
- **False-positive 6/15 (40%)** on the genuine receipts — diagnosed by the per-detector score
  breakdown into two **deterministic** causes (the LLM approved all 15):
  - **old test dates** — the WildReceipt images are 2012–2017, so `date_sanity` flags all of them.
    Under the **60-day window** (2-month reimbursement policy) this is *correct* — those receipts are
    out of policy, not a false positive; recent receipts wouldn't trip it.
  - **extraction misreads** — the model invents a phantom service-charge or misreads a digit, so the
    deterministic `arithmetic` *correctly* flags numbers that don't reconcile. The receipt is genuine;
    the *reading* was wrong.

This re-confirms the through-line: **the binding constraint is extraction quality, not detection
logic.** The arithmetic detector is right; it's fed a misread. The fix is better extraction (or wiring
a per-field confidence so the abstain guard arms on the LLM path), not more detectors.

*Generating realistic AI forgeries is itself blocked at the source:* ChatGPT and Gemini **refuse** to
edit a receipt's amount/date (content policy), so real AI forgeries come from local diffusion models or
manual editing — which carry **no C2PA/SynthID provenance** and are pixel-seamless. → content checks
(arithmetic/date/duplicate) remain the only robust catch; pixel/provenance AI-detection stays a gap.

## Recommended shape: a cascade, not a winner

The answer is not "one paradigm" — it is **routing each task to its cheapest sufficient tier**:

- **Lead with pure-Python** for detection + fusion (free, private, instant) — the robust core.
- **Light local libs** for the PDF route (pypdfium2 text, pikepdf forensics) — private, cheap.
- **C2PA** as a free strong-positive on photos *when a manifest is present* (rare but decisive).
- **Photo → fields is the one real fork:** a **local model for privacy**, or the **API for
  accuracy** — decided by IQline's data-egress policy. With egress cleared, Groq's +0.22 macro
  over the local 2B is a large accuracy win for a lighter client; without it, the local model
  (or a richer pure-Python `Receipt` model) is the privacy-preserving fallback.

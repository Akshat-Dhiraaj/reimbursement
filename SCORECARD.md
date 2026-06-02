# Scorecard — pure-Python vs local-model vs API, *per task*

> **Goal:** find where each solution paradigm is **lacking**, by measured numbers — not
> opinion. Every accuracy figure below is reproducible from this repo's `eval-*` commands;
> the resource/privacy figures were measured as each component landed.

This is the synthesis behind the project's core bet: **lead with the lightweight, private
approach and only reach for a heavy model where the cheap one provably can't do the job.**
The scorecard makes "provably can't" concrete.

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
| **Extract photo → fields** (the binding task) | **N/A — cannot read pixels** | Qwen2-VL-2B macro **0.725** (N=100); ~4.5 GB GPU, 7–15 s/receipt · docTR **0.579** | Groq llama-4-scout macro **0.947** (N=12, 0 err w/ backoff); stdlib client | **The gap lives here** — see below |

*Reproduce:* `eval` · `eval-pdf` / `eval-pdf-forensics` · `eval-image` · `eval-real --corpus …` ·
`eval-fusion [--multiroute]` · `eval-pdf-extract` · `eval-extract [--extractor groq]`.

---

## Where we're lacking (the honest gaps)

1. **Photo → field extraction is THE binding gap** (the FP audits already named extraction
   quality the constraint; this quantifies it across paradigms):
   - **Pure-Python can't do it at all** — there is no rules-based way to read a photo.
   - **Local small model (Qwen2-VL-2B): 0.725**, the *privacy-preserving* option, but
     accuracy-limited **and heavy** (needs a ~4.5 GB GPU, 7–15 s/receipt).
   - **API big model (Groq llama-4-scout): 0.947**, the most accurate **and** the lightest
     *client* (stdlib `urllib`, no GPU) — but the **image leaves the box**, and the free tier
     **rate-limits** (8/15 calls 429'd back-to-back until we added retry-with-backoff → 0/12).
     ⚠️ Measured on **N=12** (vs Qwen's N=100), so the exact 0.947 is a small-sample estimate;
     the *direction* (a large hosted model reads markedly better than a local 2B) is robust.
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

## Recommended shape: a cascade, not a winner

The answer is not "one paradigm" — it is **routing each task to its cheapest sufficient tier**:

- **Lead with pure-Python** for detection + fusion (free, private, instant) — the robust core.
- **Light local libs** for the PDF route (pypdfium2 text, pikepdf forensics) — private, cheap.
- **C2PA** as a free strong-positive on photos *when a manifest is present* (rare but decisive).
- **Photo → fields is the one real fork:** a **local model for privacy**, or the **API for
  accuracy** — decided by IQline's data-egress policy. With egress cleared, Groq's +0.22 macro
  over the local 2B is a large accuracy win for a lighter client; without it, the local model
  (or a richer pure-Python `Receipt` model) is the privacy-preserving fallback.

# REFERENCES — sources, choices, and the analysis behind them

> **What this is:** the sourced companion to [DECISIONS.md](DECISIONS.md) (prose rationale) and
> [SCORECARD.md](SCORECARD.md) (the measured pure-Python vs local-model vs API comparison). It
> houses the reference links, what we picked **over what**, and **why** — with a detailed,
> honest analysis.
>
> **How to read the evidence:** every choice is backed **first by our own measured numbers**
> (reproducible via the `eval-*` commands — the strongest, fully-verified evidence), then by
> external sources. Link trust is marked:
> - **✓ verified** — fetched and confirmed during the 2026-06 research pass.
> - **canonical** — an official project / standard / dataset page (stable addressing; several we
>   also use directly, e.g. installed from PyPI / loaded from HF).
> - A few research-pass sources returned HTTP 403 or were low-authority; they are **omitted**
>   rather than cited unverified (noted in §4).

---

## 1. The shaping choices (what, over what, and why)

### 1.1 Lead with content + provenance; pixel / AI-image forensics is a *weak* signal, never the gate
**Picked:** arithmetic + duplicate (content) and metadata/provenance (PDF structure, EXIF, C2PA) as
the lead signals. **Over:** pixel-level AI-generated-image detection and tamper-localization (#60),
which stay **deferred**.
**Why — measured + literature:**
- **Our own probes (primary):** on real receipt images, **6/8 are already libjpeg-standard-encoded**
  (a JPEG-fingerprint "re-encoded ⇒ suspicious" rule would false-positive on most legitimate shared
  receipts) and **0/12 carry an EXIF thumbnail** (a thumbnail-mismatch signal never fires). Real
  shared receipts arrive metadata/encoding-**laundered**. → see [SCORECARD.md](SCORECARD.md) "What we
  tested and rejected".
- **Frequency-domain AI detection is a GPU neural net, not a cheap heuristic:** the strong 2025 result
  (FreqCross, 97.8% on Stable Diffusion 3.5) is a *multi-modal frequency+spatial fusion network*, and
  the paper itself reports accuracy falling under JPEG recompression. ✓ verified —
  https://arxiv.org/abs/2507.02995
- **Double-JPEG compression** is detectable (comb-shaped DCT histograms) but has *no localization* and
  is erased by a final aggressive recompression/screenshot. ✓ verified —
  https://blog.ampedsoftware.com/2020/08/18/whats-in-your-past-a-guide-to-spotting-traces-of-double-jpeg-compression-with-amped-authenticate-part-1
- **PRNU** (sensor fingerprinting) needs a reference set from the *same physical camera* — which we
  never have for inbound third-party receipts. ✓ verified (review) —
  https://pmc.ncbi.nlm.nih.gov/articles/PMC12737310/
- **Classic tamper-localizers degrade on diffusion inpainting / AI-generated docs** — DocTamper
  (Qu et al., *CVPR 2023*, "Towards Robust Tampered Text Detection"), TruFor (Guillaro et al., *CVPR
  2023*), CAT-Net (Kwon et al.). Cited by name+venue (well-established works).

### 1.2 PDF parsing: **pypdfium2** (text) + **pikepdf** (deep forensics) — NOT PyMuPDF
**Picked:** pypdfium2 (Apache-2.0 / BSD-3-Clause) for the born-digital text layer, pikepdf (MPL-2.0,
a qpdf binding) for deep forensics. **Over:** **PyMuPDF / fitz** — **AGPL-3.0** (or paid), copyleft
that would reach a shipping internal product. The permissive pair gives the text read + structure
decode we need with no AGPL risk.
- pypdfium2 — canonical — https://github.com/pypdfium2-team/pypdfium2
- pikepdf — canonical — https://pypi.org/project/pikepdf/ · https://pikepdf.readthedocs.io
- qpdf (pikepdf's engine) — canonical — https://qpdf.readthedocs.io
- PyMuPDF (avoided, AGPL) — canonical — https://github.com/pymupdf/pymupdf

### 1.3 Extraction paradigm: local **Qwen2-VL-2B** vs hosted **Groq** vs **pure-Python** — measured
**Picked per task** (the cascade in [SCORECARD.md](SCORECARD.md)): pure-Python for detection/fusion;
pypdfium2 for born-digital PDFs; a model only for **photo → fields**, where it is the one real fork.
**Measured field-accuracy vs the WildReceipt oracle:** **Groq llama-4-scout 0.847** (N=50, hosted API)
> **Qwen2-VL-2B 0.725** (N=100, local GPU) > **docTR 0.579**. A big hosted model reads markedly better,
but the image leaves the box (egress) + free-tier rate limits; the local model is private but heavier
and less accurate. Decide by data-egress policy.

### 1.4 VLM checkpoint: **Qwen2-VL-2B-Instruct** (per-checkpoint licence check)
**Picked:** Qwen2-VL-2B-Instruct — Apache-2.0 *and* fits an 8 GB GPU in bf16. **Over:** LayoutLMv3
(CC-BY-**NC**), Surya (**GPL**), and **Qwen2.5-VL-3B** (no declared licence → rejected: unclear =
unusable). Loaded via transformers Auto classes, so any HF VLM is a swappable `--model`.
- Qwen2-VL-2B-Instruct — canonical — https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct
- Groq model `meta-llama/llama-4-scout-17b-16e-instruct` — canonical — https://console.groq.com/docs/models

### 1.5 OCR + KIE: **docTR** (benchmarked 2nd, by numbers)
**Picked as the comparison point:** docTR (Apache-2.0) two-stage detection+recognition + a transparent
keyword/position KIE. By the numbers the VLM ships (0.725 vs 0.579), but docTR is *competitive on the
money fields* (total 0.696 > VLM 0.598) — the VLM's edge is robustness without per-layout heuristics.
- docTR — canonical — https://github.com/mindee/doctr

### 1.6 Image provenance: **C2PA / Content Credentials** (the one trustworthy positive)
**Picked:** read C2PA manifests via `c2pa-python` (MIT/Apache). A signed `trainedAlgorithmicMedia`
assertion is cryptographic evidence of AI generation/edit — the only IMAGE signal that is a
*trustworthy positive*, not a heuristic. Honest limit: high-precision / near-zero-recall (sparse
adoption, strippable). ~260 MB install (measured), CPU-only, optional `[c2pa]` extra.
- c2pa-python — canonical — https://pypi.org/project/c2pa-python/ · https://github.com/contentauth/c2pa-python
- C2PA Technical Specification v2.2 (CC-BY-4.0) — ✓ verified — https://spec.c2pa.org/specifications/specifications/2.2/specs/C2PA_Specification.html
- Content Credentials (consumer site) — canonical — https://contentcredentials.org

### 1.7 EXIF + tax-id + fonts: small permissive libs
- **Pillow** (image EXIF + JPEG `.quantization`) — MIT-CMU — canonical — https://pillow.readthedocs.io
- **python-stdnum** (GSTIN/VAT format+checksum) — LGPL (fine as a dependency) — canonical — https://pypi.org/project/python-stdnum/
- **fontTools** (PDF font/subset inspection; *approved, not yet wired*) — MIT — canonical — https://github.com/fonttools/fonttools
- **ExifTool** (Artistic/GPL; used as a subprocess, commercial-safe) — canonical — https://exiftool.org · **PyExifTool** (dual GPL/BSD — elect BSD) — https://github.com/smarnach/pyexiftool

### 1.8 Fusion: **noisy-OR default + hand-rolled logistic fuser** (no sklearn)
**Picked:** noisy-OR as the zero-training default; an opt-in logistic-regression fuser hand-rolled in
pure Python (no scikit-learn). **Why:** transparent, dependency-free, deterministic, inspectable
weights; measured to cut real-corpus FP **0.175 → 0.042** at matched fraud-recall. Detail in
DECISIONS §1.5 / the `eval-fusion` output.

### 1.9 Datasets: three commercial-safe real corpora; **Find it again!** rejected
**Picked:** WildReceipt (Apache-2.0, US KIE oracle), CORD (CC-BY-4.0, Indonesian gt_parse oracle),
ExpressExpense (MIT, 200 images, labels-free → re-extraction audit only). **Over / rejected:**
*Find it again!* (the only public real-forgery set) — licence unclear / research-only → blocked;
INV-CDIP (CC-BY-NC); FUNSD (NC).
- WildReceipt (Apache-2.0) — canonical — https://download.openmmlab.com/mmocr/data/wildreceipt.tar
- CORD v2 (CC-BY-4.0) — canonical — https://huggingface.co/datasets/naver-clova-ix/cord-v2
- ExpressExpense SRD (MIT) — canonical — https://expressexpense.com/large-receipt-image-dataset-SRD.zip

### 1.10 Richer `Receipt` model (#81) — modelled, not suppressed
**Picked:** add `service_charge` / `discount` + accept tax-inclusive lines so `total` reconciles as
`subtotal + tax + service − discount`. **Over:** suppressing the flags (dishonest) or leaving the
3-field model (the measured FP source). **Measured:** CORD clean-oracle FP **0.170 → 0.030**,
WildReceipt **0.364 → 0.324**.

### 1.11 PDF tamper techniques we noted but did NOT adopt
- A 2025 University-of-Pretoria hash-based PDF-tamper method is **proactive** (requires pre-embedding
  hashes), so it does not apply to inbound third-party receipts. ✓ verified —
  https://www.helpnetsecurity.com/2025/07/07/detect-pdf-tampering-forgery/

### 1.12 DSPy prompt optimization — tested as a dev-time tool, **not adopted** (measured)
**Tried:** a dev-time DSPy optimizer (`src/slipguard/dspy_optimize.py`, optional `[dspy]` extra)
tuning the extraction prompt over the *same* Groq model, scored by the *same* field metric as
`eval-extract`. **Measured (WildReceipt, N=8 test):** DSPy *zero-shot* macro **0.896** ≈ the hand
prompt (**0.847**, N=50), but **BootstrapFewShot optimization made it worse — macro 0.410 with 4/8
extractor errors**: few-shot demos for a *vision* task stuff multiple receipt images into the prompt
→ token/context blow-up with no accuracy gain. **Not adopted** — the runtime stays DSPy-free, and the
finding confirms the bottleneck is *not* prompt phrasing. The only untested mode that might help is
MIPRO/COPRO *instruction* optimization (no image demos), and only once real-fraud labels exist to
optimize a real target.
- DSPy (MIT) — ✓ verified — https://github.com/stanfordnlp/dspy · https://dspy.ai

---

## 2. Links & licences at a glance (shipping path = commercial-safe only)

| Component | Role | Licence | Link | Trust |
|---|---|---|---|---|
| python-stdnum | tax-id checksum | LGPL | pypi.org/project/python-stdnum/ | canonical |
| pypdfium2 | PDF text layer | Apache-2.0 / BSD | github.com/pypdfium2-team/pypdfium2 | canonical |
| pikepdf | PDF deep forensics | MPL-2.0 | pypi.org/project/pikepdf/ | canonical |
| Pillow | image EXIF / JPEG DQT | MIT-CMU | pillow.readthedocs.io | canonical |
| c2pa-python | Content Credentials | MIT / Apache-2.0 | github.com/contentauth/c2pa-python | canonical |
| fontTools | font inspection (approved) | MIT | github.com/fonttools/fonttools | canonical |
| Qwen2-VL-2B-Instruct | local VLM extractor | Apache-2.0 | huggingface.co/Qwen/Qwen2-VL-2B-Instruct | canonical |
| docTR | OCR+KIE extractor | Apache-2.0 | github.com/mindee/doctr | canonical |
| Groq llama-4-scout | hosted VLM (API) | API ToS | console.groq.com/docs/models | canonical |
| WildReceipt | real corpus (FP audit) | Apache-2.0 | download.openmmlab.com/mmocr/data/wildreceipt.tar | canonical |
| CORD v2 | real corpus (FP audit) | CC-BY-4.0 | huggingface.co/datasets/naver-clova-ix/cord-v2 | canonical |
| ExpressExpense SRD | real images (re-extraction) | MIT | expressexpense.com/large-receipt-image-dataset-SRD.zip | canonical |
| **PyMuPDF** | (avoided) | **AGPL-3.0** | github.com/pymupdf/pymupdf | canonical |
| **LayoutLMv3** | (avoided) | **CC-BY-NC** | — | — |
| **Surya** | (avoided) | **GPL** | — | — |
| **Find it again!** | (rejected) | research-only / unclear | — | — |

---

## 3. Research / threat-model basis

- **C2PA / Content Credentials** — official technical specification v2.2 (May 2025, CC-BY-4.0).
  ✓ verified — https://spec.c2pa.org/specifications/specifications/2.2/specs/C2PA_Specification.html
- **FreqCross** (frequency+spatial fusion *network* for SD3.5 detection; ~97.8%, degrades under
  recompression). ✓ verified — https://arxiv.org/abs/2507.02995
- **Double-JPEG compression** forensics (comb-histogram; no localization; erased by recompression).
  ✓ verified — Amped Software (link in §1.1).
- **PRNU / source-camera identification** review (Sensors 2025). ✓ verified — link in §1.1.
- **DocTamper** (Qu et al., CVPR 2023), **TruFor** (Guillaro et al., CVPR 2023), **CAT-Net** (Kwon
  et al.) — document/image tamper localization; cited by name+venue (well-established).
- Commercial practice — Veryfi / AppZen / Resistant AI lead with metadata + cross-document
  intelligence (industry context; no single canonical URL).
- The project's founding research note also cited "GPT4o-Receipt" / "AIForge-Doc" receipt-forgery
  benchmarks — recorded for provenance, **not independently re-verified**; treat as illustrative.

---

## 4. Verification note (honest provenance)

- **Verified this session (fetched, 2026-06):** the C2PA spec v2.2, arXiv FreqCross (2507.02995),
  the Amped double-JPEG article, the PRNU review (PMC), and the Pretoria PDF-tamper article. The
  arXiv title even *confirms* our analysis — the strong frequency result is a GPU **network**, not a
  cheap heuristic.
- **Omitted as unverifiable:** an infosecinstitute ELA article and a defense.gov Content-Credentials
  CSI PDF both returned **HTTP 403** to the fetch; the ELA "high false-positive" and the camera-signing
  claims are well-established, so they are stated without those specific dead links rather than cited
  unverified.
- **Canonical links** (PyPI / GitHub / ReadTheDocs / Hugging Face / dataset downloads / spec sites)
  are standard, stable addresses; several we also use directly (installed c2pa-python/pikepdf/pypdfium2
  from PyPI, loaded Qwen from HF, fetched WildReceipt/CORD/ExpressExpense). They are not re-fetched here.
- **Primary evidence is always our own measurement.** Where a number appears (FP rates, macro
  accuracy, weights), it is reproducible from the `eval-*` commands — that, not any external link, is
  the load-bearing basis for each choice.

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

---

## 2. Libraries / models / techniques

| Choice | Decision | Why this and not the alternative |
|---|---|---|
| **`python-stdnum`** (tax-id checksums) | ✅ Used | Authoritative GSTIN/VAT format+checksum, tiny, LGPL (fine as a dependency). Hand-rolling check digits is error-prone and pointless. |
| **Dependency-free PDF parser** (raw bytes + regex over the Info dict) | ✅ Used for v1 | Zero runtime deps, fully commercial-safe, and the highest-yield signals (`%%EOF` count, editor tags, date gap) are visible in plain bytes. |
| **pikepdf / pdfid** (xref-stream, compressed/XMP metadata, text-over-scan) | ⏳ Deferred | Needed to decode modern compressed PDFs the regex parser can't read, but adds a dependency and isn't required for the cheap wins. Scheduled, not skipped. |
| **Noisy-OR fusion** | ✅ Used now | See §1.5. |
| **Learned/calibrated fuser** | ⏳ Planned (M3) | Needs measured per-detector performance first; premature now. |
| **LayoutLMv3** (document KIE) | ❌ Avoided | Licence **CC-BY-NC** — non-commercial; unusable in a shipping product. |
| **Surya** (OCR) | ❌ Avoided | **GPL** — copyleft; incompatible with a closed internal product. |
| **docTR / PaddleOCR** (OCR / PP-Structure) | 🔜 Planned for extraction | **Apache-2.0**, strong receipt/layout performance, commercial-safe. |
| **Qwen2.5-VL** (VLM extraction) | 🔜 Candidate for extraction | Permissive licence (verify per checkpoint size); strong at reading messy receipts end-to-end. |
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
number (39.8% FP) *with its true cause* (lossy extraction, not detector error) rather
than hiding it — because the cause is what tells us where to invest next.

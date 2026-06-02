"""slipguard CLI.

  slipguard eval [--seed S] [--n-clean N] [--fraud-per-type K]
      Build the synthetic structured benchmark and print the detector leaderboard.

  slipguard eval-pdf [--seed S] [--n-clean N] [--fraud-per-type K] [--workdir DIR]
      Build the synthetic PDF-provenance benchmark and print the leaderboard.

  slipguard eval-pdf-forensics [--seed S] [--n-clean N] [--fraud-per-type K] [--workdir DIR]
      Mint a *compressed* (PDF 1.5+, object-stream) provenance corpus whose metadata the
      dependency-free byte scanner cannot read, then score it twice — Layer 1 only vs the
      pikepdf deep layer — to show the recall the [pdf-forensics] extra recovers (the
      editor tag / date gap hidden in compressed metadata, plus JavaScript / OpenAction /
      AcroForm / overlay-annotation structural anomalies). Requires the [pdf-forensics] extra.

  slipguard eval-image [--seed S] [--n-clean N] [--fraud-per-type K] [--workdir DIR]
      Build the synthetic image-EXIF provenance benchmark and print the leaderboard
      (requires Pillow, the [vlm] extra, to mint EXIF-bearing JPEGs).

  slipguard eval-real [--corpus wildreceipt|cord|expressexpense] [--path DIR]
                      [--split test|train|validation|both] [--today YYYY-MM-DD]
                      [--extractor oracle|doctr|vlm|MODEL_ID] [--limit N]
      Audit the detectors against a corpus of *legitimate* receipts and report the
      real-world false-positive rate. Corpora: wildreceipt (US KIE oracle), cord
      (Indonesian KIE oracle — money fields, no vendor/date), expressexpense (200 MIT
      images with NO labels -> requires --extractor to re-extract; the oracle path is
      meaningless). By default fields come from the corpus KIE oracle; ``--extractor
      doctr|vlm`` instead re-extracts each receipt from its image (slow -> use ``--limit``),
      measuring arithmetic's TRUE FP on faithfully-extracted fields with the confidence
      guard live (a low-confidence digit or under-captured line items make it abstain
      rather than cry fraud). With ``--limit`` the WildReceipt oracle path audits the SAME
      first-N image-bearing receipts, so oracle vs. re-extracted FP is apples-to-apples.

  slipguard eval-extract [--corpus wildreceipt|cord] [--path DIR] [--split ...]
      Rank the IMAGE-route extractors on field-level accuracy against the corpus KIE
      oracle (the ground truth). Picks the extractor by numbers. ExpressExpense is
      rejected here (no labels to score against).

  slipguard eval-pdf-extract [--n N] [--seed S] [--workdir DIR]
      Mint a born-digital PDF corpus whose fields are a known ground truth and rank
      the PDF-route extractor(s) on field-level accuracy (requires the [pdf] extra).
      The PDF analogue of eval-extract — proves PDFs now score end-to-end like images.

  slipguard eval-calibration [--corpus wildreceipt|cord] [--path DIR] [--split ...]
                             [--extractor vlm|doctr|MODEL_ID] [--limit N]
      Ask whether an extractor's per-value confidence actually predicts a misread:
      AUC of confidence-vs-oracle-correctness, a reliability table, and an abstain
      threshold sweep. Tells us if a *calibrated* threshold could recover FP that the
      principled 0.5 floor cannot (feeds the learned-fusion milestone).

  slipguard eval-fusion [--corpora wildreceipt cord] [--split test|train|both]
                        [--seed S]
      Measure the LEARNED logistic fuser against the noisy-OR baseline. Fits on
      synthetic fraud (positives) + synthetic-clean and one half of the real corpora
      (legitimate negatives), then reports — on a different synthetic seed and the
      held-out corpus half — the synth-fraud-vs-real-legit separation (AUC) and the
      real false-positive rate at a matched fraud-recall, plus the legible learned
      per-detector weights. Honest caveat: positives are synthetic, so this is
      synth-fraud vs real-legit separation, not real-fraud detection.

  slipguard score RECEIPT.json
      Score one receipt (JSON of the Receipt fields) and print the verdict.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date as Date
from typing import Optional, Sequence

from .data.imagesynth import generate_image
from .data.pdfsynth import generate_pdf, generate_pdf_deep, generate_pdf_extraction
from .data.synth import generate
from .detectors import default_detectors
from .detectors.base import Detector
from .detectors.datesanity import DateSanityDetector
from .detectors.pdfmeta import PdfMetadataDetector
from .eval.audit import audit_false_positives, image_bearing, reextract
from .eval.calibration import collect_confidence_rows, summarize_calibration
from .eval.extraction import evaluate_extractors
from .eval.fusion_bench import compare_fusion
from .eval.harness import evaluate
from .extractors import (
    default_extractors,
    extractor_for,
    image_extractor_for_spec,
    image_extractors,
    pdf_extractors,
)
from .forensics.image import pillow_available
from .forensics.pdf import pikepdf_available
from .fusion import Fuser
from .models import DocumentType
from .routing import route_path


def cmd_eval(args: argparse.Namespace) -> None:
    dataset = generate(n_clean=args.n_clean, fraud_per_type=args.fraud_per_type, seed=args.seed)
    report = evaluate(dataset, default_detectors(), Fuser())
    print(report)


def cmd_eval_pdf(args: argparse.Namespace) -> None:
    workdir = args.workdir or os.path.join("artifacts", "pdf_bench")
    dataset = generate_pdf(
        n_clean=args.n_clean, fraud_per_type=args.fraud_per_type,
        seed=args.seed, workdir=workdir,
    )
    print(f"(wrote {len(dataset.samples)} synthetic PDFs to {workdir})\n")
    print(evaluate(dataset, default_detectors(), Fuser()))


def _detectors_with_pdf_meta(use_deep: bool) -> list[Detector]:
    """The canonical set with ``pdf_meta`` pinned to byte-only (``use_deep=False``) or
    deep (``True``), so the forensics benchmark scores the SAME corpus through each layer."""
    return [PdfMetadataDetector(use_deep=use_deep) if d.name == "pdf_meta" else d
            for d in default_detectors()]


def cmd_eval_pdf_forensics(args: argparse.Namespace) -> None:
    ok, why = pikepdf_available()
    if not ok:
        raise SystemExit(f'the [pdf-forensics] extra is required: {why}\n'
                         '  pip install -e ".[pdf-forensics]"')
    workdir = args.workdir or os.path.join("artifacts", "pdf_forensics_bench")
    dataset = generate_pdf_deep(
        n_clean=args.n_clean, fraud_per_type=args.fraud_per_type,
        seed=args.seed, workdir=workdir,
    )
    print(f"(wrote {len(dataset.samples)} compressed PDFs to {workdir})")
    print("Every file is a PDF 1.5+ with object streams — the byte scanner cannot read its "
          "metadata, so Layer 1 is blind by construction.\n")

    # Same corpus, same fuser — only the pdf_meta layer differs, isolating its contribution.
    byte_report = evaluate(dataset, _detectors_with_pdf_meta(use_deep=False), Fuser())
    deep_report = evaluate(dataset, _detectors_with_pdf_meta(use_deep=True), Fuser())

    print("=== Layer 1 only — dependency-free byte scan (simulates the extra absent) ===")
    print(byte_report)
    print("\n=== Layer 1 + Layer 2 — pikepdf deep inspection ===")
    print(deep_report)

    byte_rec = next(d.target_recall for d in byte_report.detectors if d.name == "pdf_meta")
    deep_rec = next(d.target_recall for d in deep_report.detectors if d.name == "pdf_meta")
    print(f"\npdf_meta target-recall on this compressed corpus: "
          f"byte-only={_fmt_pct(byte_rec)} -> deep={_fmt_pct(deep_rec)} "
          "(the recall the [pdf-forensics] extra recovers)")


def _fmt_pct(x: float) -> str:
    return "n/a" if x != x else f"{x:.3f}"


def cmd_eval_image(args: argparse.Namespace) -> None:
    if not pillow_available():
        raise SystemExit('Pillow is required to mint the image benchmark — pip install -e ".[vlm]"')
    workdir = args.workdir or os.path.join("artifacts", "image_bench")
    dataset = generate_image(
        n_clean=args.n_clean, fraud_per_type=args.fraud_per_type,
        seed=args.seed, workdir=workdir,
    )
    print(f"(wrote {len(dataset.samples)} synthetic receipt images to {workdir})\n")
    print(evaluate(dataset, default_detectors(), Fuser()))


def _detectors_for_audit(today: Optional[Date]) -> list[Detector]:
    """Canonical detector set, with date_sanity pinned to ``today`` when given so
    the 2018-2019 corpus era can be controlled (otherwise every old receipt trips
    the 'very old' check as a dataset artefact, not fraud)."""
    dets = default_detectors()
    if today is None:
        return dets
    return [DateSanityDetector(today=today) if d.name == "date_sanity" else d for d in dets]


# Per-corpus default dataset dir (all git-ignored under datasets/).
_CORPUS_PATHS = {
    "wildreceipt": os.path.join("datasets", "wildreceipt"),
    "cord": os.path.join("datasets", "cord"),
    "expressexpense": os.path.join("datasets", "expressexpense"),
}
_CORPUS_CHOICES = tuple(_CORPUS_PATHS)
# union of splits across corpora (wildreceipt: train/test/both; cord: train/validation/test;
# expressexpense ignores split). Each loader validates what it actually supports.
_SPLIT_CHOICES = ("train", "test", "both", "validation")


def _load_corpus(
    corpus: str, path: Optional[str], split: str,
    limit: Optional[int] = None, save_images: bool = False,
) -> list:
    """Load a real receipt corpus (all un-committed), or exit with fetch instructions.
    Shared by the FP audit, the extraction benchmark and the calibration study so every
    command speaks the same ``--corpus`` vocabulary.

      wildreceipt   — US KIE oracle (fields + images): FP audit + extraction benchmark.
      cord          — Indonesian KIE oracle (money fields, no vendor/date): same two uses.
      expressexpense— 200 MIT images, NO labels: re-extraction FP audit only (no oracle)."""
    path = path or _CORPUS_PATHS[corpus]
    try:
        if corpus == "wildreceipt":
            from .data.wildreceipt import load_receipts  # optional, un-committed dataset

            try:
                return load_receipts(path, split=split)
            except FileNotFoundError:
                raise SystemExit(
                    f"WildReceipt not found under {path!r} (it is not committed). Fetch it:\n"
                    "  curl -L -o datasets/wildreceipt.tar "
                    "https://download.openmmlab.com/mmocr/data/wildreceipt.tar\n"
                    "  tar -xf datasets/wildreceipt.tar -C datasets"
                )
        if corpus == "cord":
            from .data.cord import load_receipts

            return load_receipts(path, split=split, limit=limit, save_images=save_images)
        if corpus == "expressexpense":
            from .data.expressexpense import load_receipts

            return load_receipts(path, limit=limit)
    except (FileNotFoundError, RuntimeError) as e:  # loader-supplied fetch/availability hint
        raise SystemExit(str(e))
    raise SystemExit(f"unknown corpus {corpus!r}")


def _pick_image_extractor(spec: str):
    """The single IMAGE extractor named by ``spec`` ('doctr', 'vlm', or a HF model id),
    or exit with its skip reason. Resolves through the spec map rather than picking the
    'first runnable' candidate, so ``--extractor vlm`` is the VLM even when docTR is also
    installed (and ``--extractor doctr`` is docTR even when the VLM is)."""
    ex = image_extractor_for_spec(spec)
    ok, why = ex.available()
    if not ok:
        raise SystemExit(
            f"--extractor {spec!r} ({ex.name}) is not runnable: {why}; "
            'install it, e.g. pip install -e ".[vlm]"'
        )
    return ex


def cmd_eval_real(args: argparse.Namespace) -> None:
    today = Date.fromisoformat(args.today) if args.today else None
    reextracting = args.extractor != "oracle"
    # CORD ships images only as PIL objects; materialise them to disk solely when we
    # re-extract (the oracle FP audit reads labels, not pixels).
    save_images = args.corpus == "cord" and reextracting
    receipts = _load_corpus(
        args.corpus, args.path, args.split, limit=args.limit, save_images=save_images
    )

    if reextracting:
        # Re-extract each legitimate receipt straight from its image with a real
        # extractor (the VLM), so the audit measures arithmetic's FP on faithfully
        # extracted fields with the confidence guard live — not the oracle's lossy KIE.
        ex = _pick_image_extractor(args.extractor)
        receipts = reextract(ex, receipts, limit=args.limit)
        print(f"Re-extracted {len(receipts)} receipts via {ex.name} "
              "(replacing the oracle KIE)\n")
    elif args.corpus == "wildreceipt" and args.limit:
        # Audit the oracle on the SAME first-N image-bearing receipts a re-extraction
        # run would see, so oracle vs. re-extracted FP compares on identical receipts.
        # (CORD/ExpressExpense already cap at --limit in their loaders.)
        receipts = image_bearing(receipts, args.limit)

    print(audit_false_positives(receipts, _detectors_for_audit(today), Fuser()))


def _oracle_truths(args: argparse.Namespace) -> list:
    """Image-bearing oracle receipts for the extraction/calibration benchmarks. Rejects
    label-less corpora (ExpressExpense), and materialises CORD images so an extractor can
    re-read them; ``--limit`` is pushed into the loader so CORD writes only N images."""
    if args.corpus == "expressexpense":
        raise SystemExit(
            "--corpus expressexpense has no field labels to score against; use it only with "
            "`eval-real --extractor vlm` (the re-extraction FP audit)"
        )
    truths = [
        r for r in _load_corpus(args.corpus, args.path, args.split,
                                limit=args.limit, save_images=(args.corpus == "cord"))
        if r.image_path
    ]
    return truths[: args.limit] if args.limit else truths


def cmd_eval_extract(args: argparse.Namespace) -> None:
    truths = _oracle_truths(args)
    candidates = [e for e in image_extractors(args.model) if e.can_handle(DocumentType.IMAGE)]

    runnable = []
    for ex in candidates:
        ok, why = ex.available()
        (runnable.append(ex) if ok else print(f"skipping {ex.name}: {why}"))
    if not runnable:
        raise SystemExit(
            "no IMAGE-route extractor is runnable here (see skip reasons above). "
            'Install one, e.g. pip install -e ".[vlm]", then re-run.'
        )

    print(f"Ground truth: {len(truths)} {args.corpus} oracle receipts ({args.split} split)\n")
    for report in evaluate_extractors(runnable, truths):
        print(report)
        print()


def cmd_eval_pdf_extract(args: argparse.Namespace) -> None:
    workdir = args.workdir or os.path.join("artifacts", "pdf_extract_bench")
    truths = generate_pdf_extraction(n=args.n, seed=args.seed, workdir=workdir)
    candidates = [e for e in pdf_extractors() if e.can_handle(DocumentType.PDF)]

    runnable = []
    for ex in candidates:
        ok, why = ex.available()
        (runnable.append(ex) if ok else print(f"skipping {ex.name}: {why}"))
    if not runnable:
        raise SystemExit(
            "no PDF-route extractor is runnable here (see skip reasons above). "
            'Install one: pip install -e ".[pdf]", then re-run.'
        )

    print(f"(wrote {len(truths)} synthetic born-digital PDFs to {workdir})")
    print(f"Ground truth: {len(truths)} synthetic receipts\n")
    for report in evaluate_extractors(runnable, truths):
        print(report)
        print()


def cmd_eval_calibration(args: argparse.Namespace) -> None:
    truths = _oracle_truths(args)  # same image-bearing oracle subset the FP audit re-extracts
    ex = _pick_image_extractor(args.extractor)
    print(f"Calibrating {ex.name} confidence on {len(truths)} image-bearing receipts "
          f"vs the {args.corpus} oracle\n")
    rows = collect_confidence_rows(ex, truths)
    print(summarize_calibration(rows, ex.name))


def _try_load_legit(corpus: str, split: str) -> list:
    """Load a real corpus as legitimate negatives for the fusion benchmark, or print
    why it's unavailable and return an empty list (the bench degrades to synthetic-only
    rather than hard-failing, so it still runs on a machine missing one corpus)."""
    try:
        return _load_corpus(corpus, None, split)
    except SystemExit as e:  # _load_corpus exits with fetch instructions when absent
        print(f"(skipping {corpus}: {str(e).splitlines()[0]})")
        return []


def cmd_eval_fusion(args: argparse.Namespace) -> None:
    # Two independent synthetic seeds -> disjoint train/test fraud with no leakage.
    train = generate(seed=args.seed)
    test = generate(seed=args.seed + 1)

    real: list = []
    used: list[str] = []
    for corpus in args.corpora:
        recs = _try_load_legit(corpus, args.split)
        if recs:
            real += recs
            used.append(f"{corpus}({len(recs)})")
    if not real:
        print("No real corpora present; fitting on synthetic clean negatives only "
              "(the real-FP columns will be n/a). Fetch a corpus to measure real FP — "
              "see `eval-real --help`.\n")

    print(compare_fusion(train, test, real, corpora=used))


def cmd_score(args: argparse.Namespace) -> None:
    route = route_path(args.path)
    extractor = extractor_for(route)
    if extractor is None:  # IMAGE/PDF: fall back to the first runnable route extractor
        candidates = [*image_extractors(), *pdf_extractors()]
        extractor = next(
            (e for e in candidates if e.can_handle(route) and e.available()[0]), None
        )
    if extractor is None:
        skips = "; ".join(
            f"{e.name}: {e.available()[1]}"
            for e in [*image_extractors(), *pdf_extractors()]
            if e.can_handle(route) and not e.available()[0]
        )
        raise SystemExit(
            f"no runnable extractor for the {route.value} route"
            + (f" ({skips})" if skips else " — pass a structured receipt JSON for now")
        )
    receipt = extractor.extract(args.path, doc_id=args.path)
    verdict = Fuser().verdict(receipt.doc_id, [d.run(receipt) for d in default_detectors()])
    print(f"doc {verdict.doc_id}: risk={verdict.risk_score:.3f} -> {verdict.decision.value.upper()}")
    for reason in verdict.reasons:
        print(f"  - {reason}")
    if not verdict.reasons:
        print("  - no active signals")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="slipguard", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("eval", help="run the synthetic structured benchmark leaderboard")
    pe.add_argument("--seed", type=int, default=0)
    pe.add_argument("--n-clean", type=int, default=120)
    pe.add_argument("--fraud-per-type", type=int, default=30)
    pe.set_defaults(func=cmd_eval)

    pp = sub.add_parser("eval-pdf", help="run the synthetic PDF-provenance benchmark leaderboard")
    pp.add_argument("--seed", type=int, default=0)
    pp.add_argument("--n-clean", type=int, default=40)
    pp.add_argument("--fraud-per-type", type=int, default=15)
    pp.add_argument("--workdir", default=None, help="where to write the synthetic PDFs")
    pp.set_defaults(func=cmd_eval_pdf)

    pf = sub.add_parser("eval-pdf-forensics",
                        help="deep PDF forensics on compressed PDFs; byte-only vs pikepdf recall")
    pf.add_argument("--seed", type=int, default=0)
    pf.add_argument("--n-clean", type=int, default=20)
    pf.add_argument("--fraud-per-type", type=int, default=10)
    pf.add_argument("--workdir", default=None, help="where to write the synthetic PDFs")
    pf.set_defaults(func=cmd_eval_pdf_forensics)

    pi = sub.add_parser("eval-image", help="run the synthetic image-EXIF provenance benchmark leaderboard")
    pi.add_argument("--seed", type=int, default=0)
    pi.add_argument("--n-clean", type=int, default=40)
    pi.add_argument("--fraud-per-type", type=int, default=15)
    pi.add_argument("--workdir", default=None, help="where to write the synthetic images")
    pi.set_defaults(func=cmd_eval_image)

    pr = sub.add_parser("eval-real", help="false-positive audit on legitimate real receipts")
    pr.add_argument("--corpus", default="wildreceipt", choices=_CORPUS_CHOICES,
                    help="which real corpus to audit (default: wildreceipt)")
    pr.add_argument("--path", default=None,
                    help="dataset directory (default: datasets/<corpus>/)")
    pr.add_argument("--split", default="test", choices=_SPLIT_CHOICES)
    pr.add_argument("--today", default=None,
                    help="reference date YYYY-MM-DD for date_sanity (default: real today)")
    pr.add_argument("--extractor", default="oracle",
                    help="'oracle' (the corpus KIE labels, default), or 'doctr' / 'vlm' / a HF "
                         "model id to re-extract fields from the source images before auditing "
                         "(required for expressexpense, which has no labels)")
    pr.add_argument("--limit", type=int, default=None,
                    help="audit only the first N receipts (a slow VLM on a laptop)")
    pr.set_defaults(func=cmd_eval_real)

    px = sub.add_parser("eval-extract", help="rank extractors on field accuracy vs the corpus oracle")
    px.add_argument("--corpus", default="wildreceipt", choices=_CORPUS_CHOICES,
                    help="oracle corpus to score against (expressexpense has no labels)")
    px.add_argument("--path", default=None,
                    help="dataset directory (default: datasets/<corpus>/)")
    px.add_argument("--split", default="test", choices=_SPLIT_CHOICES)
    px.add_argument("--limit", type=int, default=None,
                    help="score only the first N receipts (a slow VLM on a laptop)")
    px.add_argument("--model", default=None, help="override the VLM checkpoint id")
    px.set_defaults(func=cmd_eval_extract)

    pxp = sub.add_parser("eval-pdf-extract",
                         help="rank PDF-route extractors on field accuracy vs a synthetic oracle")
    pxp.add_argument("--n", type=int, default=40, help="number of synthetic PDFs to mint")
    pxp.add_argument("--seed", type=int, default=0)
    pxp.add_argument("--workdir", default=None, help="where to write the synthetic PDFs")
    pxp.set_defaults(func=cmd_eval_pdf_extract)

    pc = sub.add_parser("eval-calibration",
                        help="does the extractor's per-value confidence predict a misread?")
    pc.add_argument("--corpus", default="wildreceipt", choices=_CORPUS_CHOICES,
                    help="oracle corpus to calibrate against (expressexpense has no labels)")
    pc.add_argument("--path", default=None,
                    help="dataset directory (default: datasets/<corpus>/)")
    pc.add_argument("--split", default="test", choices=_SPLIT_CHOICES)
    pc.add_argument("--extractor", default="vlm",
                    help="confidence-bearing IMAGE extractor: 'vlm' (default), 'doctr', "
                         "or a HF model id ('oracle' has no confidence to calibrate)")
    pc.add_argument("--limit", type=int, default=None,
                    help="calibrate on the first N receipts (a slow VLM on a laptop)")
    pc.set_defaults(func=cmd_eval_calibration)

    pfu = sub.add_parser("eval-fusion",
                         help="measure the learned logistic fuser vs the noisy-OR baseline")
    pfu.add_argument("--corpora", nargs="+", default=["wildreceipt", "cord"],
                     choices=("wildreceipt", "cord"),
                     help="real corpora to use as legitimate negatives (default: both; "
                          "expressexpense excluded — its oracle path has no fields)")
    pfu.add_argument("--split", default="test", choices=_SPLIT_CHOICES)
    pfu.add_argument("--seed", type=int, default=0,
                     help="synthetic seed; the test set uses seed+1 (disjoint, no leakage)")
    pfu.set_defaults(func=cmd_eval_fusion)

    ps = sub.add_parser("score", help="score a single receipt JSON")
    ps.add_argument("path")
    ps.set_defaults(func=cmd_score)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    # Real receipts carry arbitrary unicode (★, foreign scripts); don't let a
    # legacy console codec crash the run.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # pragma: no cover - non-reconfigurable stream
        pass
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

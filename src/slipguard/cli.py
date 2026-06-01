"""slipguard CLI.

  slipguard eval [--seed S] [--n-clean N] [--fraud-per-type K]
      Build the synthetic structured benchmark and print the detector leaderboard.

  slipguard eval-pdf [--seed S] [--n-clean N] [--fraud-per-type K] [--workdir DIR]
      Build the synthetic PDF-provenance benchmark and print the leaderboard.

  slipguard eval-real [--path DIR] [--split test|train|both] [--today YYYY-MM-DD]
      Audit the detectors against a corpus of *legitimate* receipts (WildReceipt)
      and report the real-world false-positive rate.

  slipguard eval-extract [--path DIR] [--split test|train|both]
      Rank the IMAGE-route extractors on field-level accuracy against the
      WildReceipt KIE oracle (the ground truth). Picks the extractor by numbers.

  slipguard score RECEIPT.json
      Score one receipt (JSON of the Receipt fields) and print the verdict.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date as Date
from typing import Optional, Sequence

from .data.pdfsynth import generate_pdf
from .data.synth import generate
from .detectors import default_detectors
from .detectors.base import Detector
from .detectors.datesanity import DateSanityDetector
from .eval.audit import audit_false_positives
from .eval.extraction import evaluate_extractors
from .eval.harness import evaluate
from .extractors import default_extractors, extractor_for, image_extractors
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


def _detectors_for_audit(today: Optional[Date]) -> list[Detector]:
    """Canonical detector set, with date_sanity pinned to ``today`` when given so
    the 2018-2019 corpus era can be controlled (otherwise every old receipt trips
    the 'very old' check as a dataset artefact, not fraud)."""
    dets = default_detectors()
    if today is None:
        return dets
    return [DateSanityDetector(today=today) if d.name == "date_sanity" else d for d in dets]


def _load_wildreceipt(path: str, split: str) -> list:
    """Load the (un-committed) WildReceipt corpus, or exit with fetch instructions.
    Shared by the FP audit and the extraction benchmark — both read the same corpus."""
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


def cmd_eval_real(args: argparse.Namespace) -> None:
    receipts = _load_wildreceipt(args.path, args.split)
    today = Date.fromisoformat(args.today) if args.today else None
    print(audit_false_positives(receipts, _detectors_for_audit(today), Fuser()))


def cmd_eval_extract(args: argparse.Namespace) -> None:
    truths = [r for r in _load_wildreceipt(args.path, args.split) if r.image_path]
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

    if args.limit:  # a slow VLM on a laptop: measure a subset honestly, not for hours
        truths = truths[: args.limit]
    print(f"Ground truth: {len(truths)} WildReceipt oracle receipts ({args.split} split)\n")
    for report in evaluate_extractors(runnable, truths):
        print(report)
        print()


def cmd_score(args: argparse.Namespace) -> None:
    route = route_path(args.path)
    extractor = extractor_for(route)
    if extractor is None:  # IMAGE/PDF: fall back to the first runnable image extractor
        extractor = next(
            (e for e in image_extractors() if e.can_handle(route) and e.available()[0]), None
        )
    if extractor is None:
        skips = "; ".join(
            f"{e.name}: {e.available()[1]}"
            for e in image_extractors() if e.can_handle(route) and not e.available()[0]
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

    pr = sub.add_parser("eval-real", help="false-positive audit on legitimate real receipts")
    pr.add_argument("--path", default=os.path.join("datasets", "wildreceipt"),
                    help="extracted wildreceipt/ directory")
    pr.add_argument("--split", default="test", choices=("train", "test", "both"))
    pr.add_argument("--today", default=None,
                    help="reference date YYYY-MM-DD for date_sanity (default: real today)")
    pr.set_defaults(func=cmd_eval_real)

    px = sub.add_parser("eval-extract", help="rank extractors on field accuracy vs the WildReceipt oracle")
    px.add_argument("--path", default=os.path.join("datasets", "wildreceipt"),
                    help="extracted wildreceipt/ directory")
    px.add_argument("--split", default="test", choices=("train", "test", "both"))
    px.add_argument("--limit", type=int, default=None,
                    help="score only the first N receipts (a slow VLM on a laptop)")
    px.add_argument("--model", default=None, help="override the VLM checkpoint id")
    px.set_defaults(func=cmd_eval_extract)

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

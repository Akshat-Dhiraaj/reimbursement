"""slipguard CLI.

  slipguard eval [--seed S] [--n-clean N] [--fraud-per-type K]
      Build the synthetic structured benchmark and print the detector leaderboard.

  slipguard eval-pdf [--seed S] [--n-clean N] [--fraud-per-type K] [--workdir DIR]
      Build the synthetic PDF-provenance benchmark and print the leaderboard.

  slipguard score RECEIPT.json
      Score one receipt (JSON of the Receipt fields) and print the verdict.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional, Sequence

from .data.pdfsynth import generate_pdf
from .data.synth import generate
from .detectors import default_detectors
from .eval.harness import evaluate
from .fusion import Fuser
from .models import DocumentType, Receipt
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


def cmd_score(args: argparse.Namespace) -> None:
    route = route_path(args.path)
    if route is not DocumentType.STRUCTURED:
        raise SystemExit(
            f"{route.value} inputs need the extraction route (VLM/OCR), not yet wired — "
            "pass a structured receipt JSON for now"
        )
    with open(args.path, "r", encoding="utf-8") as fh:
        receipt = Receipt.from_dict(json.load(fh))
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

    ps = sub.add_parser("score", help="score a single receipt JSON")
    ps.add_argument("path")
    ps.set_defaults(func=cmd_score)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

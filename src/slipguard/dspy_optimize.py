"""DSPy DEV-TIME prompt optimizer for receipt field extraction — measured vs the hand prompt.

**NOT a runtime dependency.** Behind the optional ``[dspy]`` extra, this is a dev-time experiment:
it defines a DSPy signature + module for receipt extraction over the **same Groq model** the
hand-prompt :class:`GroqVLExtractor` uses, optimizes the prompt with **BootstrapFewShot** against
the **same field-accuracy metric** as :mod:`slipguard.eval.extraction` (so the numbers are directly
comparable to the 0.847 hand-prompt baseline), and compares **zero-shot vs optimized** on held-out
real receipts.

The question it answers by numbers: *does auto-optimization beat the hand-written prompt?* The
production runtime stays DSPy-free — any win is meant to be exported to a prompt file, not shipped
as a heavy dependency (the same discipline as keeping the learned fuser hand-rolled, not sklearn).

**Honest scope:** we have no labelled *visual fraud*, so we optimize+measure **extraction** (the
binding constraint), not the fraud verdict. Needs ``[dspy]`` + Pillow + ``GROQ_API_KEY`` + network.
Groq sits behind Cloudflare (1010-blocks datacenter UAs), so a browser User-Agent is passed.

Run: ``python -m slipguard.dspy_optimize``  (or import :func:`optimize_and_compare`).
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

from .eval.extraction import _MONEY_FIELDS, _money_ok, _vendor_ok, evaluate_extractor
from .extractors.base import Extractor
from .models import DocumentType, LineItem, Receipt

_GROQ_MODEL = "groq/meta-llama/llama-4-scout-17b-16e-instruct"
_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")


def make_lm(max_tokens: int = 1024):
    """A DSPy LM bound to Groq (browser UA to pass Groq's Cloudflare). Lazy dspy import."""
    import dspy
    return dspy.LM(_GROQ_MODEL, api_key=os.environ["GROQ_API_KEY"], temperature=0,
                   max_tokens=max_tokens, extra_headers={"User-Agent": _UA})


def make_signature():
    """The DSPy extraction signature (image -> fields). Built lazily so importing this module
    never requires dspy."""
    import dspy

    class ExtractReceipt(dspy.Signature):
        """Read this receipt / invoice image and extract its fields. Numbers must be plain — no
        currency symbols, no thousands separators. Use 0 for an amount that is not shown and '' for
        an absent date or vendor."""

        image: dspy.Image = dspy.InputField(desc="a photo or scan of a receipt / invoice")
        vendor: str = dspy.OutputField(desc="store / merchant name")
        date: str = dspy.OutputField(desc="transaction date as YYYY-MM-DD, else ''")
        subtotal: float = dspy.OutputField(desc="pre-tax subtotal (0 if not shown)")
        tax: float = dspy.OutputField(desc="tax amount (0 if not shown)")
        total: float = dspy.OutputField(desc="grand total (0 if not shown)")
        line_count: int = dspy.OutputField(desc="number of purchased line items")

    return ExtractReceipt


def _parse_date(s: object):
    s = (str(s) if s is not None else "").strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%d/%m/%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def _num(x: object) -> Optional[float]:
    """Float, treating 0 / blank as 'absent' (the signature uses 0 for an unshown amount)."""
    try:
        v = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return v if v != 0.0 else None


def pred_to_receipt(pred, doc_id: str, image_path: str) -> Receipt:
    """Map a DSPy prediction (typed fields) to a Receipt. ``line_count`` becomes that many
    placeholder line items so :func:`evaluate_extractor`'s line-count check has something to
    count (we score the *count*, not the items). Pure / model-free → unit-testable."""
    try:
        n = max(0, int(getattr(pred, "line_count", 0) or 0))
    except (TypeError, ValueError):
        n = 0
    vendor = (getattr(pred, "vendor", "") or "").strip() or "(unknown)"
    return Receipt(
        doc_id=doc_id, vendor_name=vendor, date=_parse_date(getattr(pred, "date", "")),
        currency="USD", country="US",
        line_items=[LineItem("item", 1.0, 0.0, 0.0) for _ in range(n)],
        subtotal=_num(getattr(pred, "subtotal", None)),
        tax_amount=_num(getattr(pred, "tax", None)),
        total=_num(getattr(pred, "total", None)),
        source=DocumentType.IMAGE, source_path=image_path, image_path=image_path,
    )


def receipt_field_score(gold: Receipt, pred: Receipt) -> float:
    """Macro field accuracy of one prediction vs the oracle — the SAME rules as
    :mod:`slipguard.eval.extraction`, reused as the DSPy optimization metric (and unit-tested)."""
    hits: list[bool] = []
    if gold.vendor_name not in {"", "(unknown)"}:
        hits.append(_vendor_ok(pred.vendor_name, gold.vendor_name))
    if gold.date is not None:
        hits.append(pred.date == gold.date)
    for f in _MONEY_FIELDS:
        if getattr(gold, f) is not None:
            hits.append(_money_ok(getattr(pred, f), getattr(gold, f)))
    if gold.line_items:
        hits.append(len(pred.line_items) == len(gold.line_items))
    return sum(bool(h) for h in hits) / len(hits) if hits else 1.0


def _metric(example, pred, trace=None) -> bool:
    """DSPy BootstrapFewShot metric: keep a demo when the prediction's macro field accuracy vs the
    gold (reconstructed from the example's fields) clears 0.6 — so only faithful demos are kept."""
    gold = Receipt(
        doc_id="g", vendor_name=(getattr(example, "vendor", "") or "(unknown)"),
        date=_parse_date(getattr(example, "date", "")),
        subtotal=_num(getattr(example, "subtotal", None)),
        tax_amount=_num(getattr(example, "tax", None)),
        total=_num(getattr(example, "total", None)),
        line_items=[LineItem("i", 1, 0, 0)] * int(getattr(example, "line_count", 0) or 0),
    )
    return receipt_field_score(gold, pred_to_receipt(pred, "p", "")) >= 0.6


class DspyGroqExtractor(Extractor):
    """Wrap a compiled DSPy program as an Extractor so :func:`evaluate_extractor` scores it exactly
    like the hand-prompt GroqVLExtractor (apples-to-apples)."""

    handles = (DocumentType.IMAGE,)

    def __init__(self, program, name: str = "dspy"):
        self.program = program
        self.name = name

    def available(self) -> tuple[bool, str]:
        return (bool(os.environ.get("GROQ_API_KEY")), "set GROQ_API_KEY")

    def extract(self, path: str, doc_id: Optional[str] = None) -> Receipt:
        import dspy
        pred = self.program(image=dspy.Image.from_file(path))
        return pred_to_receipt(pred, doc_id or path, path)


def _example(t: Receipt):
    import dspy
    return dspy.Example(
        image=dspy.Image.from_file(t.image_path),
        vendor=t.vendor_name, date=t.date.isoformat() if t.date else "",
        subtotal=float(t.subtotal or 0.0), tax=float(t.tax_amount or 0.0),
        total=float(t.total or 0.0), line_count=len(t.line_items),
    ).with_inputs("image")


def optimize_and_compare(*, n_train: int = 6, n_test: int = 8, split: str = "test",
                         root: str = "datasets/wildreceipt", max_demos: int = 2):
    """Fit a BootstrapFewShot-optimized extractor on the first ``n_train`` WildReceipt oracle
    receipts and compare it to the zero-shot DSPy program on the next ``n_test`` (disjoint), scored
    by the same field-accuracy metric as `eval-extract`. Returns ``(zero_report, opt_report)``."""
    import dspy

    from .data.wildreceipt import load_receipts
    dspy.configure(lm=make_lm())
    truths = [t for t in load_receipts(root, split) if t.image_path]
    train_t, test_t = truths[:n_train], truths[n_train:n_train + n_test]

    base = dspy.Predict(make_signature())
    zero_report = evaluate_extractor(DspyGroqExtractor(base, "dspy:zero-shot"), test_t)

    optimized = dspy.BootstrapFewShot(
        metric=_metric, max_bootstrapped_demos=max_demos, max_labeled_demos=max_demos,
    ).compile(base, trainset=[_example(t) for t in train_t])
    opt_report = evaluate_extractor(DspyGroqExtractor(optimized, "dspy:optimized"), test_t)
    return zero_report, opt_report, optimized


def main() -> None:  # pragma: no cover - dev-time, needs network + GROQ_API_KEY
    import argparse
    p = argparse.ArgumentParser(description="DSPy dev-time prompt optimizer vs the hand prompt")
    p.add_argument("--n-train", type=int, default=6)
    p.add_argument("--n-test", type=int, default=8)
    p.add_argument("--max-demos", type=int, default=2)
    args = p.parse_args()
    zero, opt, _ = optimize_and_compare(n_train=args.n_train, n_test=args.n_test,
                                        max_demos=args.max_demos)
    print(zero, "\n")
    print(opt, "\n")
    print(f"DSPy optimization delta (macro): {zero.overall:.3f} (zero-shot) -> "
          f"{opt.overall:.3f} (optimized)  |  hand-prompt GroqVLExtractor reference: 0.847 (N=50)")


if __name__ == "__main__":
    main()

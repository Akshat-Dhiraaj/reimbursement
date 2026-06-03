"""Prompt-accuracy benchmark for the SIMPLE ``validate`` pipeline.

The ``validate`` pipeline's quality is set almost entirely by ``prompts/validity_prompt.md`` — the
model and the deterministic cross-check are fixed, so refining the *prompt* is the lever. To pick the
best prompt by numbers (not opinion), this scores **what the prompt itself produces**, run with the
deterministic cross-check **OFF** (``cross_check=False``) so we measure the prompt, not the safety net.

Two ground-truthed, fraud-data-free numbers per prompt:

* **field accuracy** (primary) — vendor / date / subtotal / tax / total against the **same
  WildReceipt oracle** the extractor leaderboard uses, with the **same** ``_vendor_ok`` / ``_money_ok``
  comparators (DRY). ``line_count`` is excluded: the validity prompt is not asked for line items, so
  scoring it would penalise every variant equally and just dilute the macro.
* **judgement behaviour** (secondary) — the decision distribution on these *legitimate* receipts
  (a higher approve-rate is better specificity, since they are genuine), and the agreement between
  the prompt's self-reported ``arithmetic_consistent`` and a deterministic recomputation of its **own**
  extracted numbers (does the prompt's arithmetic reasoning actually hold up — the exact weakness the
  #85 cross-check exists to patch). A prompt that needs the cross-check less is the better prompt.

Honest caveats: N is small (free-tier API), the oracle is legitimate-only (so this measures field
fidelity + clean-receipt specificity, **not** fraud *recall* — we have no real fraud positives), and
field accuracy is capped by what the oracle labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

from ..llm_validate import _verdict_to_receipt, validate
from ..models import Receipt
from ..money import money_close
from .extraction import _UNKNOWN_VENDORS, FieldAccuracy, _money_ok, _vendor_ok
from .metrics import _fmt

#: fields the validity prompt is asked to read (line items are out of scope → not scored).
_PROMPT_FIELDS = ("vendor", "date", "subtotal", "tax_amount", "total")
_DECISIONS = ("approve", "review", "reject")


def _arith_truth(r: Optional[Receipt]) -> Optional[bool]:
    """Deterministic arithmetic consistency on the prompt's OWN extracted numbers, or ``None`` when
    it isn't checkable (no subtotal or no total). Mirrors the arithmetic detector's relation:
    ``total == subtotal + tax + service − discount`` within the shared money tolerance."""
    if r is None or r.total is None or r.subtotal is None:
        return None
    expected = (r.subtotal or 0) + (r.tax_amount or 0) + (r.service_charge or 0) - (r.discount or 0)
    return money_close(r.total, round(expected, 2))


@dataclass
class PromptReport:
    name: str
    n_samples: int
    n_errors: int = 0                       # validate() raised (network/parse)
    parse_fails: int = 0                    # model reply wasn't parseable JSON (fell back to review)
    fields: list[FieldAccuracy] = field(default_factory=list)
    decisions: dict = field(default_factory=dict)
    arith_checked: int = 0
    arith_agree: int = 0
    aborted: bool = False          # stopped early on a run of consecutive errors (provider quota/down)

    @property
    def overall(self) -> float:
        """Macro-average field accuracy (each scored field weighted equally) — the primary metric."""
        accs = [f.accuracy for f in self.fields if f.n]
        return sum(accs) / len(accs) if accs else float("nan")

    @property
    def approve_rate(self) -> float:
        tot = sum(self.decisions.values())
        return self.decisions.get("approve", 0) / tot if tot else float("nan")

    @property
    def arith_agreement(self) -> float:
        return self.arith_agree / self.arith_checked if self.arith_checked else float("nan")

    def __str__(self) -> str:
        dec = " / ".join(f"{d} {self.decisions.get(d, 0)}" for d in _DECISIONS)
        lines = [
            f"Prompt: {self.name} — {self.n_samples} receipts, "
            f"{self.n_errors} error(s), {self.parse_fails} parse-fail(s)"
            + (" — ABORTED EARLY (consecutive errors; provider quota/down)" if self.aborted else ""),
            "",
            f"{'field':12} {'n':>5} {'correct':>8} {'accuracy':>9}",
            "-" * 38,
        ]
        for f in self.fields:
            lines.append(f"{f.field:12} {f.n:>5} {f.correct:>8} {_fmt(f.accuracy):>9}")
        lines += [
            "-" * 38,
            f"{'FIELD MACRO':26} {_fmt(self.overall):>9}",
            f"decisions: {dec}  (approve-rate {_fmt(self.approve_rate)})",
            f"arithmetic self-consistency agreement: {_fmt(self.arith_agreement)} "
            f"({self.arith_checked} checkable)",
        ]
        return "\n".join(lines)


def evaluate_prompt(
    prompt_path: Optional[str], truths: Sequence[Receipt], *,
    provider: str = "auto", model: Optional[str] = None, progress: bool = False,
    max_consecutive_errors: int = 4,
) -> PromptReport:
    """Run the ``validate`` pipeline (cross-check OFF) over ``truths`` with the given prompt file and
    score it. ``prompt_path=None`` uses the canonical ``prompts/validity_prompt.md``. With
    ``progress`` it prints one flushed line per receipt, so a slow free-tier run stays observable.

    Fails fast: after ``max_consecutive_errors`` calls in a row raise (almost always a depleted
    provider quota), it stops rather than grinding through the rest — so a dead quota wastes ~4
    calls, not the whole sample. The report is flagged ``aborted``."""
    name = Path(prompt_path).stem if prompt_path else "validity_prompt (default)"
    counts = {f: FieldAccuracy(f) for f in _PROMPT_FIELDS}
    decisions = {d: 0 for d in _DECISIONS}
    n_errors = parse_fails = arith_checked = arith_agree = 0
    consecutive_errors = 0
    aborted = False

    for i, truth in enumerate(truths):
        path = truth.image_path or truth.source_path or ""
        verdict: Optional[dict]
        try:
            verdict = validate(path, provider=provider, prompt_path=prompt_path,
                               model=model, cross_check=False)
            consecutive_errors = 0
        except Exception:                  # a failed call is a miss for every field, not a crash
            verdict, n_errors = None, n_errors + 1
            consecutive_errors += 1
        pred = _verdict_to_receipt(verdict, path) if verdict else None

        if verdict is not None:
            decision = verdict.get("decision", "review")
            decisions[decision] = decisions.get(decision, 0) + 1
            if "_raw" in verdict:          # llm_validate stamps _raw only on an unparseable reply
                parse_fails += 1
            truth_consistent = _arith_truth(pred)
            claim = verdict.get("arithmetic_consistent")
            if truth_consistent is not None and isinstance(claim, bool):
                arith_checked += 1
                arith_agree += int(claim == truth_consistent)

        # Field scoring — only where the oracle actually supplies a value (non-circular, never
        # penalising the prompt for a field ground truth itself is missing).
        if truth.vendor_name not in _UNKNOWN_VENDORS:
            fa = counts["vendor"]; fa.n += 1
            fa.correct += _vendor_ok(pred.vendor_name if pred else None, truth.vendor_name)
        if truth.date is not None:
            fa = counts["date"]; fa.n += 1
            fa.correct += bool(pred and pred.date == truth.date)
        for fld in ("subtotal", "tax_amount", "total"):     # not `name` — that holds the prompt label
            tv = getattr(truth, fld)
            if tv is not None:
                fa = counts[fld]; fa.n += 1
                fa.correct += _money_ok(getattr(pred, fld) if pred else None, tv)

        if progress:
            tag = verdict.get("decision", "review") if verdict is not None else "ERROR"
            print(f"  {name}: {i + 1}/{len(truths)} -> {tag}", flush=True)

        if consecutive_errors >= max_consecutive_errors:
            aborted = True
            if progress:
                print(f"  {name}: stopping early after {consecutive_errors} consecutive errors "
                      "(provider quota/availability) — remaining receipts skipped", flush=True)
            break

    return PromptReport(
        name=name, n_samples=len(truths), n_errors=n_errors, parse_fails=parse_fails,
        fields=[counts[f] for f in _PROMPT_FIELDS], decisions=decisions,
        arith_checked=arith_checked, arith_agree=arith_agree, aborted=aborted,
    )


def evaluate_prompts(
    prompt_paths: Sequence[Optional[str]], truths: Sequence[Receipt], *,
    provider: str = "auto", model: Optional[str] = None, progress: bool = False,
) -> list[PromptReport]:
    """Score several prompt files on the same ground truth — the prompt leaderboard."""
    truths = list(truths)
    return [evaluate_prompt(p, truths, provider=provider, model=model, progress=progress)
            for p in prompt_paths]

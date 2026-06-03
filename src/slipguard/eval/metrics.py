"""Small, dependency-free metrics so the benchmark runs with zero ML stack."""

from __future__ import annotations

from typing import Sequence


def _fmt(x: float) -> str:
    """Fixed-width 3-decimal float for the eval report tables; NaN -> a right-aligned 'n/a'.
    Shared by every report __str__ (harness / extraction / prompt_eval / fusion_bench)."""
    return "  n/a" if x != x else f"{x:5.3f}"


def confusion(preds: Sequence[bool], labels: Sequence[bool]) -> tuple[int, int, int, int]:
    tp = fp = tn = fn = 0
    for p, y in zip(preds, labels):
        if p and y:
            tp += 1
        elif p and not y:
            fp += 1
        elif (not p) and y:
            fn += 1
        else:
            tn += 1
    return tp, fp, tn, fn


def precision_recall_f1(preds: Sequence[bool], labels: Sequence[bool]) -> tuple[float, float, float]:
    tp, fp, tn, fn = confusion(preds, labels)
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    if prec == prec and rec == rec and (prec + rec) > 0:
        f1 = 2 * prec * rec / (prec + rec)
    else:
        f1 = float("nan")
    return prec, rec, f1


def false_positive_rate(preds: Sequence[bool], labels: Sequence[bool]) -> float:
    tp, fp, tn, fn = confusion(preds, labels)
    return fp / (fp + tn) if (fp + tn) else float("nan")


def roc_auc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """Rank-based AUC (Mann-Whitney). O(P*N); fine for benchmark sizes."""
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else (0.5 if p == n else 0.0)
    return wins / (len(pos) * len(neg))

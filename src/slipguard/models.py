"""Core domain models — the shared contracts every approach plugs into."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as Date
from datetime import time as Time
from enum import Enum
from typing import Any, Optional


class DocumentType(str, Enum):
    PDF = "pdf"
    IMAGE = "image"
    STRUCTURED = "structured"


class FraudType(str, Enum):
    """Fraud subtypes — used as ground-truth labels and as detector targets,
    so the harness can report per-subtype performance instead of one number."""

    NONE = "none"
    ARITHMETIC = "arithmetic"        # totals / line items do not reconcile
    DATE = "date"                    # impossible or implausible date
    TAX_ID = "tax_id"               # malformed / wrong-checksum tax id
    DUPLICATE = "duplicate"          # resubmission of an existing receipt
    METADATA = "metadata"            # PDF provenance: edited-after-write / editor tags
    AI_GENERATED = "ai_generated"    # whole image synthesised (image route, later)
    IMAGE_TAMPER = "image_tamper"    # local edit / inpaint (image route, later)


class Decision(str, Enum):
    APPROVE = "approve"
    REVIEW = "review"
    REJECT = "reject"


@dataclass
class LineItem:
    description: str
    quantity: float
    unit_price: float
    amount: float  # line total exactly as printed on the receipt


@dataclass
class Receipt:
    """A normalised receipt/invoice. Produced either by the synthetic generator
    or (later) by a pluggable extraction approach (VLM / OCR+KIE)."""

    doc_id: str
    vendor_name: str
    date: Date
    currency: str = "INR"
    country: str = "IN"  # ISO-3166 alpha-2; drives tax-id + locale checks
    vendor_tax_id: Optional[str] = None
    line_items: list[LineItem] = field(default_factory=list)
    subtotal: Optional[float] = None
    tax_rate: Optional[float] = None  # fraction, e.g. 0.18
    tax_amount: Optional[float] = None
    total: Optional[float] = None
    time: Optional[Time] = None
    payment_method: Optional[str] = None
    source: DocumentType = DocumentType.STRUCTURED
    source_path: Optional[str] = None  # original document on disk (PDF/image) for provenance forensics
    image_path: Optional[str] = None
    raw_text: Optional[str] = None
    #: per-field extraction confidence in [0,1], keyed by Receipt field name
    #: (e.g. {"total": 0.4}). Populated by an Extractor; empty == fully trusted,
    #: so synthetic/oracle/hand-written receipts read as confident and unchanged.
    field_confidence: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Receipt":
        d = dict(d)
        if isinstance(d.get("date"), str):
            d["date"] = Date.fromisoformat(d["date"])
        if isinstance(d.get("time"), str):
            d["time"] = Time.fromisoformat(d["time"])
        if "source" in d and not isinstance(d["source"], DocumentType):
            d["source"] = DocumentType(d["source"])
        d["line_items"] = [
            li if isinstance(li, LineItem) else LineItem(**li)
            for li in d.get("line_items", [])
        ]
        return cls(**d)


@dataclass
class Signal:
    """One approach's read on one receipt.

    ``score`` is an estimated P(fraud) in [0, 1]. ``confidence`` in [0, 1] is how
    much weight the signal should carry; ``confidence == 0`` means the detector
    abstained (insufficient data / not applicable) and must not move the verdict.
    """

    detector: str
    score: float
    confidence: float
    reasons: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def weighted(self) -> float:
        return self.score * self.confidence

    @property
    def abstained(self) -> bool:
        return self.confidence <= 0.0

    @property
    def effective_score(self) -> float:
        """Fraud score that counts only when the detector did not abstain."""
        return 0.0 if self.abstained else self.score


@dataclass
class Verdict:
    doc_id: str
    risk_score: float
    decision: Decision
    signals: list[Signal] = field(default_factory=list)

    @property
    def reasons(self) -> list[str]:
        out: list[str] = []
        for s in self.signals:
            if s.abstained:
                continue
            for r in s.reasons:
                out.append(f"[{s.detector}] {r}")
        return out


@dataclass
class LabeledSample:
    """A benchmark item with ground truth for honest evaluation."""

    receipt: Receipt
    is_fraud: bool
    fraud_types: set[FraudType] = field(default_factory=set)
    detail: dict[str, Any] = field(default_factory=dict)
    split: str = "test"  # "history" | "test"

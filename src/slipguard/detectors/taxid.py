"""Tax-ID validation — format + checksum via python-stdnum.

Cheap and unforgiving: a fabricated GSTIN/VAT fails its check digit instantly.
India GSTIN is the primary case (IQline); other countries dispatch by ISO code
and abstain when unsupported rather than guessing."""

from __future__ import annotations

from ..models import FraudType, Receipt, Signal
from .base import Detector

# country (ISO-3166 alpha-2) -> stdnum validator module exposing is_valid()
_VALIDATORS: dict[str, object] = {}
try:
    from stdnum.in_ import gstin as _gstin

    _VALIDATORS["IN"] = _gstin
except Exception:  # pragma: no cover - stdnum always present via deps
    pass
try:
    from stdnum.eu import vat as _eu_vat

    for _cc in ("AT", "BE", "BG", "CY", "CZ", "DE", "DK", "EE", "ES", "FI", "FR",
                "GR", "HR", "HU", "IE", "IT", "LT", "LU", "LV", "MT", "NL", "PL",
                "PT", "RO", "SE", "SI", "SK"):
        _VALIDATORS[_cc] = _eu_vat
except Exception:  # pragma: no cover
    pass


class TaxIdValidationDetector(Detector):
    name = "tax_id"
    targets = frozenset({FraudType.TAX_ID})

    def score(self, receipt: Receipt) -> Signal:
        r = receipt
        validator = _VALIDATORS.get((r.country or "").upper())
        if validator is None:
            return self._abstain(f"no tax-id validator for country '{r.country}'")
        if not r.vendor_tax_id:
            return self._abstain("no vendor tax id present")

        try:
            ok = validator.is_valid(r.vendor_tax_id)  # type: ignore[attr-defined]
        except Exception:
            ok = False

        if ok:
            return Signal(self.name, score=0.02, confidence=0.9,
                          reasons=[f"tax id {r.vendor_tax_id} passes {r.country} validation"])
        return Signal(self.name, score=0.9, confidence=0.9,
                      reasons=[f"tax id {r.vendor_tax_id} fails {r.country} format/checksum"],
                      evidence={"tax_id": r.vendor_tax_id, "country": r.country})

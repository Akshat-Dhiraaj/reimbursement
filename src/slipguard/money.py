"""Robust money-string parsing, shared by the WildReceipt oracle and the VLM extractor.

Receipts in the wild mix conventions: US ``1,234.56``, European ``1.234,56`` and
``129,75`` (comma is the decimal point), bare ``.70``. A naive ``replace(",", "")``
silently turns the German ``129,75`` into ``12975`` — a 100x error we actually hit in
the WildReceipt oracle, which corrupted *both* the extraction benchmark and the
false-positive audit (a €129,75 receipt read as €12 975 fails every arithmetic check).
Centralising the parse here means neither the ground-truth loader nor the extractor can
repeat that mistake, and a fix lands in one place.

Heuristic (no locale hint available, so infer from the digits): the right-most separator
is the decimal point only when 1-2 digits follow it — money carries at most two
decimals, so 3+ trailing digits mark a thousands group instead. Any earlier separator is
always a grouping separator and is dropped. ``.70`` parses as ``0.70``. This resolves the
genuinely ambiguous ``1,234`` (US thousands -> 1234) vs ``129,75`` (EU decimal -> 129.75)
by the trailing-digit count, which matches both corpora's real usage.
"""

from __future__ import annotations

import re
from typing import Optional

#: first number-like run: a digit-led group (which may carry ``,``/``.`` groupings) or a
#: leading-dot value like ``.70``. Lets us pull the number out of ``Eur129,75`` / ``$5.33``.
_TOKEN_RE = re.compile(r"-?\d[\d.,]*|-?\.\d+")


def parse_money(text: object) -> Optional[float]:
    """Parse a money value out of free text, tolerating currency symbols/words and US
    or European digit grouping. Returns ``None`` when there is no number to read."""
    if text is None:
        return None
    m = _TOKEN_RE.search(str(text))
    if not m:
        return None
    tok = m.group(0)
    neg = tok.startswith("-")
    tok = tok.lstrip("-").rstrip(".,")  # drop the sign and any trailing separator ("58.92,")
    if not tok:
        return None

    cut = max(tok.rfind(","), tok.rfind("."))
    trailing = tok[cut + 1:] if cut != -1 else ""
    if cut != -1 and 1 <= len(trailing) <= 2:   # that separator is the decimal point
        whole = re.sub(r"[.,]", "", tok[:cut]) or "0"
        value = float(f"{whole}.{trailing}")
    else:                                        # integer / thousands-grouped — no decimals
        digits = re.sub(r"[.,]", "", tok)
        if not digits:
            return None
        value = float(digits)
    return -value if neg else value


def money_close(a: float, b: float, rel: float = 0.01, abs_tol: float = 0.02) -> bool:
    """True when two money amounts are equal within the larger of an absolute floor
    (cent-level rounding) and a relative band. Shared so that "the same amount" means one
    thing everywhere: the ``arithmetic`` detector's reconciliation slack and the
    extraction / calibration metrics' correctness check both defer to this. ``b`` is the
    reference the relative band is taken against."""
    return abs(a - b) <= max(abs_tol, rel * abs(b))

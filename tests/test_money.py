"""Tests for the shared money parser (money.py).

These pin the behaviour that the WildReceipt oracle and the VLM extractor both rely on,
including the real cases that exposed the original 100x bug: European decimal commas
('Eur129,75'), bare leading dots ('.70'), and US thousands separators ('1,234') — the
last of which must stay an integer even though a lone comma can also be a decimal."""

from __future__ import annotations

from slipguard.money import money_close, parse_money


def test_us_decimal_and_symbols():
    assert parse_money("$5.33") == 5.33
    assert parse_money("58.22") == 58.22
    assert parse_money("AMOUNT:$58.92") == 58.92
    assert parse_money("$31.05") == 31.05


def test_european_comma_decimal():
    # The real WildReceipt rows that the comma-stripping parser turned into 12975 / 2465.
    assert parse_money("Eur129,75") == 129.75
    assert parse_money("24,65") == 24.65
    assert parse_money("Eur154,40") == 154.40
    assert parse_money("12,00") == 12.0


def test_leading_dot_is_fraction_not_integer():
    assert parse_money(".70") == 0.70   # was mis-parsed as 70.0 (tax > total!)
    assert parse_money(".5") == 0.5


def test_thousands_separators_us_and_eu():
    assert parse_money("1,234") == 1234.0          # US thousands, no decimals
    assert parse_money("12,975.00") == 12975.0     # US thousands + decimal
    assert parse_money("1,234.56") == 1234.56
    assert parse_money("1.234,56") == 1234.56      # EU thousands + decimal
    assert parse_money("1,23,456.78") == 123456.78  # Indian lakh grouping


def test_decimal_vs_thousands_is_decided_by_trailing_digit_count():
    # The crux: a lone comma with 2 trailing digits is a decimal, with 3 it's grouping.
    assert parse_money("129,75") == 129.75
    assert parse_money("129,750") == 129750.0


def test_currency_prefix_dot_is_not_a_decimal_point():
    # Regression: CORD's Indonesian-Rupiah amounts are written "Rp.118.000". The "Rp." dot
    # fused with the digits and the leftmost bare-decimal match read ".118" as 118.0 — a
    # 1000x error (118 000 IDR became 118.0). The leading-dot branch now requires the dot
    # not be preceded by a letter/digit, so the digit-led group wins instead.
    assert parse_money("Rp.118.000") == 118000.0
    assert parse_money("Rp.56.000") == 56000.0
    assert parse_money("Rp 138.000") == 138000.0
    assert parse_money("Rp.0") == 0.0
    # ...without regressing genuine bare decimals where the dot starts the number.
    assert parse_money("$.70") == 0.70
    assert parse_money(".70") == 0.70


def test_negative_and_sign():
    assert parse_money("-3.50") == -3.5
    assert parse_money("-1.234,56") == -1234.56


def test_none_on_garbage_or_empty():
    assert parse_money("no digits here") is None
    assert parse_money("") is None
    assert parse_money(None) is None


def test_money_close_absolute_floor_and_relative_band():
    # Absolute 2-cent floor dominates on small amounts (1% of 1.0 = 0.01 < 0.02).
    # Stay inside the band, not on the 0.02 edge, to avoid float-boundary flapping.
    assert money_close(1.00, 1.01)
    assert money_close(1.00, 1.015)
    assert not money_close(1.00, 1.05)
    # Relative 1% band dominates on large amounts (1% of 1000 = 10 >> 0.02).
    assert money_close(1000.0, 1009.0)
    assert not money_close(1000.0, 1011.0)
    # Exact match.
    assert money_close(5.0, 5.0)

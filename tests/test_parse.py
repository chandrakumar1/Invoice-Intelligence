"""
What:     Unit tests for app/parse.py: every supported format, every ambiguity, every rejection.
In:       Literal strings in the shapes printed on real invoices.
Out:      pytest pass/fail.
Fails if: A parsing rule changes; these tests are the specification of what counts as unambiguous.
"""

from datetime import date
from decimal import Decimal

import pytest

from app.models import AdjustmentKind, TaxScheme
from app.parse import (
    HINT_CERTAINTY,
    AmbiguousDate,
    ParseError,
    classify_adjustment,
    detect_tax_scheme,
    infer_date_order,
    infer_decimal_separator,
    parse_amount,
    parse_currency,
    parse_date,
    parse_percent,
    parse_quantity,
    signed_adjustment,
)

# --- Amounts -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,value",
    [
        ("11,589.50", "11589.50"),
        ("1,22,248.00", "122248.00"),  # Indian lakh grouping
        ("6,65,388.00", "665388.00"),
        ("INR 179,778.90", "179778.90"),
        ("Rs. 79,493.06", "79493.06"),
        ("₹1,234.50", "1234.50"),
        ("1.457,00 €", "1457.00"),  # European
        ("24,90 €", "24.90"),
        ("1 234,56", "1234.56"),  # space as thousands separator
        ("1'234.50", "1234.50"),  # Swiss
        ("285.00", "285.00"),
        ("12499", "12499"),
        ("-1,500.00", "-1500.00"),
        ("−1,500.00", "-1500.00"),  # Unicode minus
        ("(-) 0.37", "-0.37"),
        ("(1,234.00)", "-1234.00"),  # accounting negative
        ("1,234.00-", "-1234.00"),  # trailing minus
        ("+0.63", "0.63"),
        ("-0.01", "-0.01"),
    ],
)
def test_amounts(text, value):
    parsed = parse_amount(text)
    assert parsed.value == Decimal(value) and parsed.certainty == 1.0


@pytest.mark.parametrize("text", ["1,234", "1.457", "€ 2.500"])
def test_three_digits_after_a_single_separator_is_ambiguous(text):
    with pytest.raises(ParseError, match="decimal or thousands"):
        parse_amount(text)


def test_ambiguous_separator_resolved_by_document_hint():
    assert parse_amount("1.457", ",") == parse_amount("1.457", ",")  # deterministic
    thousands = parse_amount("1.457", decimal_sep=",")
    assert thousands.value == Decimal("1457") and thousands.certainty == HINT_CERTAINTY
    decimals = parse_amount("1.457", decimal_sep=".")
    assert decimals.value == Decimal("1.457") and decimals.certainty == HINT_CERTAINTY


@pytest.mark.parametrize("text", ["", "abc", "12a", "1,2,3", "12,34,5.00", "1.234.5", "1,000,00.00", "--5", "1..2", "."])
def test_not_amounts(text):
    with pytest.raises(ParseError):
        parse_amount(text)


def test_quantities_allow_a_unit():
    assert parse_quantity("10 Nos").value == Decimal("10")
    assert parse_quantity("2.5 kg").value == Decimal("2.5")
    assert parse_quantity("1500").value == Decimal("1500")
    with pytest.raises(ParseError):
        parse_quantity("ten")


@pytest.mark.parametrize("text,value", [("9%", "9"), ("18 %", "18"), ("@ 9.00%", "9.00"), ("19%", "19"), ("7,5 %", "7.5")])
def test_percent(text, value):
    assert parse_percent(text).value == Decimal(value)


@pytest.mark.parametrize("text", ["CGST 9% on 1,000", "%", "nine percent"])
def test_percent_needs_exactly_one_number(text):
    with pytest.raises(ParseError):
        parse_percent(text)


@pytest.mark.parametrize(
    "texts,expected",
    [
        (["11,589.50", "285.00", "1,22,248.00"], "."),
        (["1.457,00 €", "24,90 €", "276,83 €"], ","),
        (["1500", "1,234"], None),  # nothing decisive
        (["1,234.50", "1.234,50"], None),  # contradictory document
    ],
)
def test_infer_decimal_separator(texts, expected):
    assert infer_decimal_separator(texts) == expected


# --- Dates ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,value",
    [
        ("2025-11-05", date(2025, 11, 5)),
        ("20/11/2025", date(2025, 11, 20)),  # day > 12: unambiguous
        ("11/20/2025", date(2025, 11, 20)),
        ("11-Nov-2025", date(2025, 11, 11)),
        ("11 November 2025", date(2025, 11, 11)),
        ("3rd Nov 2025", date(2025, 11, 3)),
        ("Nov 11, 2025", date(2025, 11, 11)),
        ("04. Dez 2025", date(2025, 12, 4)),
        ("11.11.2025", date(2025, 11, 11)),  # same day and month: no ambiguity
    ],
)
def test_dates(text, value):
    parsed = parse_date(text)
    assert parsed.value == value and parsed.certainty == 1.0


@pytest.mark.parametrize("text", ["05/11/2025", "04.11.2025", "03-04-2025"])
def test_ambiguous_dates_without_a_hint_are_rejected(text):
    with pytest.raises(AmbiguousDate):
        parse_date(text)


def test_ambiguous_dates_follow_the_document_order_with_lower_certainty():
    dmy, mdy = parse_date("05/11/2025", "DMY"), parse_date("05/11/2025", "MDY")
    assert dmy.value == date(2025, 11, 5) and mdy.value == date(2025, 5, 11)
    assert dmy.certainty == HINT_CERTAINTY


def test_unambiguous_date_ignores_a_contradicting_hint():
    assert parse_date("20/11/2025", "MDY").value == date(2025, 11, 20)


def test_two_digit_year_is_read_as_20yy_with_lower_certainty():
    parsed = parse_date("20/11/25")
    assert parsed.value == date(2025, 11, 20) and parsed.certainty < 1.0


@pytest.mark.parametrize("text", ["31/02/2025", "13/13/2025", "Smarch 3, 2025", "next Tuesday", ""])
def test_impossible_or_unrecognizable_dates(text):
    with pytest.raises(ParseError):
        parse_date(text)


@pytest.mark.parametrize(
    "dates,currency,schemes,expected",
    [
        (["05/11/2025", "20/11/2025"], None, [], "DMY"),  # the due date settles it
        (["11/05/2025", "11/20/2025"], None, [], "MDY"),
        (["05/11/2025"], "INR", [], "DMY"),
        (["04.11.2025"], None, [TaxScheme.EU_VAT], "DMY"),
        (["05/11/2025"], "USD", [], "MDY"),
        (["05/11/2025"], None, [], None),
        (["20/11/2025", "11/20/2025"], "INR", [], None),  # the document contradicts itself
    ],
)
def test_infer_date_order(dates, currency, schemes, expected):
    assert infer_date_order(dates, currency, schemes) == expected


# --- Currency, tax schemes, adjustments ------------------------------------------------------


@pytest.mark.parametrize("text,code", [("Rs.", "INR"), ("INR", "INR"), ("₹", "INR"), ("€", "EUR"), ("EUR", "EUR"), ("USD", "USD"), ("$", "USD"), ("£", "GBP")])
def test_currency(text, code):
    assert parse_currency(text).value == code


@pytest.mark.parametrize("text", ["", "CHF", "INR / €"])
def test_currency_must_be_exactly_one(text):
    with pytest.raises(ParseError):
        parse_currency(text)


@pytest.mark.parametrize(
    "raw,context,scheme",
    [
        ("29AAGCS4821K1Z0", "GSTIN: 29AAGCS4821K1Z0", TaxScheme.GSTIN),
        ("29AAGCS4821K1Z0", "Reg: 29AAGCS4821K1Z0", TaxScheme.GSTIN),  # shape alone
        ("DE12345678", "Friedrichstraße 112 · Germany · VAT ID: DE12345678", TaxScheme.EU_VAT),  # invalid, still EU VAT
        ("NL853746219B01", "NL853746219B01", TaxScheme.EU_VAT),  # prefix alone
        ("123456789", "USt-IdNr. 123456789", TaxScheme.EU_VAT),
        ("12-3456789", "Tax ID: 12-3456789", TaxScheme.OTHER),
        ("AB-9", "Customer ref AB-9", TaxScheme.UNKNOWN),
        ("1234567", "Industrial customer 1234567", TaxScheme.UNKNOWN),  # 'ust' inside words doesn't count
    ],
)
def test_detect_tax_scheme(raw, context, scheme):
    assert detect_tax_scheme(raw, context) == scheme


@pytest.mark.parametrize(
    "label,kind",
    [
        ("Less: Trade Discount", AdjustmentKind.DISCOUNT),
        ("Round Off", AdjustmentKind.ROUND_OFF),
        ("Rounding", AdjustmentKind.ROUND_OFF),
        ("R/O", AdjustmentKind.ROUND_OFF),
        ("Freight Charges", AdjustmentKind.SHIPPING),
        ("Packing & Handling", AdjustmentKind.SHIPPING),
        ("Insurance", AdjustmentKind.OTHER),
        (None, AdjustmentKind.OTHER),
    ],
)
def test_classify_adjustment(label, kind):
    assert classify_adjustment(label) is kind


def test_adjustment_signs():
    assert signed_adjustment(AdjustmentKind.DISCOUNT, Decimal("1500")) == Decimal("-1500")
    assert signed_adjustment(AdjustmentKind.DISCOUNT, Decimal("-1500")) == Decimal("-1500")
    assert signed_adjustment(AdjustmentKind.SHIPPING, Decimal("-50")) == Decimal("50")
    assert signed_adjustment(AdjustmentKind.ROUND_OFF, Decimal("-0.01")) == Decimal("-0.01")
    assert signed_adjustment(AdjustmentKind.ROUND_OFF, Decimal("0.37")) == Decimal("0.37")

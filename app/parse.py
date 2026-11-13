"""
What:     Deterministic parsing of text copied from the document: amounts, quantities, rates, dates, currencies, tax-ID schemes, adjustment kinds.
In:       The exact printed text of one value, plus document-level style hints (decimal separator, day/month order) where needed.
Out:      Parsed(value, certainty), or ParseError with a human-readable reason. Ambiguity is reported, never guessed away.
Fails if: A format isn't covered (e.g. a month name in an unsupported language), so the value is rejected as unparseable. That's visible, not silent.

`certainty` is 1.0 when the text alone decides the value, and lower when a document-level hint
had to decide (e.g. 05/11/2025 read as day-first because other dates on the invoice are day-first).
It feeds the confidence score in app/provenance.py.
"""

import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Literal

from app.models import AdjustmentKind, TaxScheme

DecimalSep = Literal[".", ","]
DateOrder = Literal["DMY", "MDY"]

HINT_CERTAINTY = 0.9  # a document-level hint, not the text itself, decided the reading
TWO_DIGIT_YEAR_CERTAINTY = 0.9


class ParseError(ValueError):
    """The text can't be turned into a value without guessing. The message says why."""


class AmbiguousDate(ParseError):
    """Day and month could be swapped (e.g. 03/04/2025) and nothing on the document settles it."""


@dataclass(frozen=True)
class Parsed[T]:
    value: T
    certainty: float = 1.0


# --- Numbers -------------------------------------------------------------------------------

_CURRENCY_MARKS = re.compile(r"(?i)\b(?:INR|EUR|USD|GBP)\b|\bRs\b\.?|US\$|[₹€$£]")
_MINUS = "-−–"  # hyphen-minus, Unicode minus, en dash (common in typeset PDFs)
_THOUSANDS_SPACE = re.compile(r"(?<=\d)[    '](?=\d{3}(?!\d))")


def parse_amount(text: str, decimal_sep: DecimalSep | None = None) -> Parsed[Decimal]:
    """'1,23,456.00', 'INR 179,778.90', '1.457,00 €', '(1,234.00)', '-0.01', "1'234.50" -> Decimal."""
    s = _CURRENCY_MARKS.sub("", text).strip()
    negative = False
    if s.startswith("(-)"):
        negative, s = True, s[3:].strip()
    elif s.startswith("(") and s.endswith(")"):
        negative, s = True, s[1:-1].strip()
    if s[:1] and s[0] in _MINUS:
        negative, s = True, s[1:].strip()
    elif s.startswith("+"):
        s = s[1:].strip()
    if s[-1:] and s[-1] in _MINUS:  # trailing minus: "1,234.00-"
        negative, s = True, s[:-1].strip()
    s = _THOUSANDS_SPACE.sub("", s)
    if not re.fullmatch(r"\d[\d.,]*\d|\d", s):
        raise ParseError(f"not a number: {text!r}")
    number, certainty = _resolve_separators(s, decimal_sep, text)
    return Parsed(-number if negative else number, certainty)


def parse_quantity(text: str, decimal_sep: DecimalSep | None = None) -> Parsed[Decimal]:
    """Like an amount, but a trailing unit word is allowed: '10 Nos', '2.5 kg'."""
    m = re.fullmatch(r"\s*([\d.,'\s ]*\d)\s*[A-Za-z.]{0,10}\s*", text)
    if not m:
        raise ParseError(f"not a quantity: {text!r}")
    return parse_amount(m.group(1), decimal_sep)


def parse_percent(text: str, decimal_sep: DecimalSep | None = None) -> Parsed[Decimal]:
    """'9%', '18 %', '@ 9.00%', '7,5 %' -> the percentage as a number (Decimal('9') for 9%)."""
    numbers = re.findall(r"\d+(?:[.,]\d+)?", text)
    if len(numbers) != 1:
        raise ParseError(f"expected exactly one number in a rate, got {text!r}")
    return parse_amount(numbers[0], decimal_sep)


def _resolve_separators(s: str, hint: DecimalSep | None, original: str) -> tuple[Decimal, float]:
    certainty = 1.0
    seps = {c for c in s if c in ".,"}
    if len(seps) == 2:
        decimal = "." if s.rfind(".") > s.rfind(",") else ","
    elif len(seps) == 1:
        sep = seps.pop()
        parts = s.split(sep)
        if len(parts) > 2:
            decimal = None  # repeated: it's the grouping separator ("1,23,456")
        elif len(parts[1]) == 3:  # "1,234" or "1.457": thousands, or three decimals?
            if hint is None:
                raise ParseError(f"can't tell if {sep!r} in {original!r} is a decimal or thousands separator")
            decimal, certainty = (sep if hint == sep else None), HINT_CERTAINTY
        else:
            decimal = sep
    else:
        decimal = None

    if decimal:
        grouping = "," if decimal == "." else "."
    else:
        grouping = next((c for c in s if c in ".,"), None)
    integer, _, fraction = s.partition(decimal) if decimal else (s, "", "")
    if fraction and not fraction.isdigit():
        raise ParseError(f"malformed decimals in {original!r}")
    if not _valid_grouping(integer, grouping):
        raise ParseError(f"digit grouping in {original!r} doesn't look like a printed amount")
    clean = integer.replace(grouping, "") if grouping else integer
    try:
        return Decimal(f"{clean}.{fraction}" if fraction else clean), certainty
    except InvalidOperation as e:  # pragma: no cover - guarded by the regexes above
        raise ParseError(f"not a number: {original!r}") from e


def _valid_grouping(integer: str, grouping: str | None) -> bool:
    if not grouping or grouping not in integer:
        return integer.isdigit()
    parts = integer.split(grouping)
    return (
        all(p.isdigit() for p in parts)
        and 1 <= len(parts[0]) <= 3
        and all(len(p) in (2, 3) for p in parts[1:-1])  # 2: Indian lakh/crore grouping
        and len(parts[-1]) == 3
    )


def infer_decimal_separator(texts: list[str]) -> DecimalSep | None:
    """The document's decimal separator, from the amounts that show it unambiguously. None if they disagree."""
    votes: set[str] = set()
    for text in texts:
        s = _THOUSANDS_SPACE.sub("", _CURRENCY_MARKS.sub("", text)).strip()
        seps = [c for c in s if c in ".,"]
        if len(set(seps)) == 2:
            votes.add("." if s.rfind(".") > s.rfind(",") else ",")
        elif len(seps) == 1 and re.search(r"[.,]\d{1,2}\s*-?\)?$", s):
            votes.add(seps[0])
    return votes.pop() if len(votes) == 1 else None  # type: ignore[return-value]


# --- Dates ---------------------------------------------------------------------------------

_MONTHS = {
    **{m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)},
    **{m: i for i, m in enumerate(
        ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"], 1)},
    "sept": 9,
    # German forms that differ from English
    "januar": 1, "februar": 2, "mär": 3, "märz": 3, "mai": 5, "juni": 6, "juli": 7, "okt": 10, "oktober": 10, "dez": 12, "dezember": 12,
}
_ISO = re.compile(r"\b(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})\b")
_NUMERIC = re.compile(r"\b(\d{1,2})[-/.](\d{1,2})[-/.](\d{4}|\d{2})\b")
_DAY_MONTH_NAME = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\.?[-\s/]+([A-Za-zä]{3,9})\.?[-\s/,]+(\d{4})\b")
_MONTH_NAME_DAY = re.compile(r"\b([A-Za-zä]{3,9})\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b")


def parse_date(text: str, order: DateOrder | None = None) -> Parsed[date]:
    """ISO, numeric (05/11/2025, 04.11.2025, 05-11-25), and month-name forms (11-Nov-2025, Nov 11, 2025)."""
    if m := _ISO.search(text):
        return Parsed(_make_date(int(m[1]), int(m[2]), int(m[3]), text))
    if m := _DAY_MONTH_NAME.search(text):
        return Parsed(_make_date(int(m[3]), _month(m[2], text), int(m[1]), text))
    if m := _MONTH_NAME_DAY.search(text):
        return Parsed(_make_date(int(m[3]), _month(m[1], text), int(m[2]), text))
    if m := _NUMERIC.search(text):
        a, b, year = int(m[1]), int(m[2]), int(m[3])
        certainty = 1.0
        if len(m[3]) == 2:
            year, certainty = 2000 + year, TWO_DIGIT_YEAR_CERTAINTY
        if a > 12 and b > 12:
            raise ParseError(f"neither part of {text!r} can be a month")
        if a > 12 or a == b:
            day, month = a, b
        elif b > 12:
            day, month = b, a
        elif order is None:
            raise AmbiguousDate(f"{text!r} could be day-first or month-first and the document doesn't say which")
        else:
            day, month = (a, b) if order == "DMY" else (b, a)
            certainty *= HINT_CERTAINTY
        return Parsed(_make_date(year, month, day, text), certainty)
    raise ParseError(f"not a recognizable date: {text!r}")


def _month(name: str, text: str) -> int:
    month = _MONTHS.get(name.lower())
    if month is None:
        raise ParseError(f"unknown month name {name!r} in {text!r}")
    return month


def _make_date(year: int, month: int, day: int, text: str) -> date:
    try:
        return date(year, month, day)
    except ValueError as e:
        raise ParseError(f"impossible date {text!r}: {e}") from e


_DAY_FIRST_CURRENCIES = {"INR", "EUR", "GBP"}
_MONTH_FIRST_CURRENCIES = {"USD"}


def infer_date_order(date_texts: list[str], currency: str | None, schemes: list[TaxScheme]) -> DateOrder | None:
    """Day/month order for the document: unambiguous numeric dates first, then currency and tax regime."""
    votes: set[str] = set()
    for text in date_texts:
        if (m := _NUMERIC.search(text)) and not _ISO.search(text):
            a, b = int(m[1]), int(m[2])
            if a > 12 >= b:
                votes.add("DMY")
            elif b > 12 >= a:
                votes.add("MDY")
    if len(votes) == 1:
        return votes.pop()  # type: ignore[return-value]
    if votes:
        return None  # the document contradicts itself; don't pick
    if currency in _DAY_FIRST_CURRENCIES or any(s in (TaxScheme.GSTIN, TaxScheme.EU_VAT) for s in schemes):
        return "DMY"
    if currency in _MONTH_FIRST_CURRENCIES:
        return "MDY"
    return None


# --- Currency, tax IDs, adjustments ------------------------------------------------------------

_CURRENCIES = [
    (re.compile(r"(?i)\bINR\b|₹|\bRs\b\.?|\brupees?\b"), "INR"),
    (re.compile(r"(?i)\bEUR\b|€|\beuros?\b"), "EUR"),
    (re.compile(r"(?i)\bGBP\b|£"), "GBP"),
    (re.compile(r"(?i)\bUSD\b|\$"), "USD"),
]


def parse_currency(text: str) -> Parsed[str]:
    """Printed code or symbol -> ISO 4217 code. Exactly one currency must be recognizable."""
    found = {code for pattern, code in _CURRENCIES if pattern.search(text)}
    if len(found) != 1:
        raise ParseError(f"expected one recognizable currency in {text!r}, found {sorted(found) or 'none'}")
    return Parsed(found.pop())


_GSTIN_SHAPE = re.compile(r"^\d{2}[A-Z0-9]{13}$")
_EU_VAT_PREFIXES = frozenset(
    "AT BE BG CY CZ DE DK EE EL ES FI FR HR HU IE IT LT LU LV MT NL PL PT RO SE SI SK XI".split()
)
_GST_LABEL = re.compile(r"(?i)\bgst(?:in)?\b")
_VAT_LABEL = re.compile(r"(?i)\b(?:vat|ust-?id(?:nr)?|ust|mwst|tva|btw|iva|moms)\b")
_OTHER_LABEL = re.compile(r"(?i)\b(?:tax\s*id|tin|ein|abn|pan)\b")


def detect_tax_scheme(raw: str, context: str) -> TaxScheme:
    """Which regime a tax ID belongs to, from its printed label (the quoted line) and its shape.

    Detection only. Whether the ID is valid for that scheme is a validation question (app/validate.py).
    """
    compact = re.sub(r"[\s.\-]", "", raw).upper()
    if _GST_LABEL.search(context) or _GSTIN_SHAPE.match(compact):
        return TaxScheme.GSTIN
    if _VAT_LABEL.search(context) or (compact[:2] in _EU_VAT_PREFIXES and re.search(r"\d", compact[2:])):
        return TaxScheme.EU_VAT
    if _OTHER_LABEL.search(context):
        return TaxScheme.OTHER
    return TaxScheme.UNKNOWN


_ADJUSTMENT_KEYWORDS = [
    (re.compile(r"(?i)round(?:ing|ed)?\s*(?:off|ing)?\b|\br/?o\b"), AdjustmentKind.ROUND_OFF),
    (re.compile(r"(?i)discount|\bless\b|rebate|rabatt|skonto"), AdjustmentKind.DISCOUNT),
    (re.compile(r"(?i)shipping|freight|delivery|courier|carriage|postage|packing|handling|versand"), AdjustmentKind.SHIPPING),
]


def classify_adjustment(label: str | None) -> AdjustmentKind:
    """Keyword rules on the printed label. Order matters: 'Round Off' is checked before 'Less'."""
    if label:
        for pattern, kind in _ADJUSTMENT_KEYWORDS:
            if pattern.search(label):
                return kind
    return AdjustmentKind.OTHER


def signed_adjustment(kind: AdjustmentKind, printed: Decimal) -> Decimal:
    """Effect on the grand total. A discount always reduces it and shipping always adds to it,
    whatever sign is printed. Round-off and other lines keep their printed sign."""
    if kind is AdjustmentKind.DISCOUNT:
        return -abs(printed)
    if kind is AdjustmentKind.SHIPPING:
        return abs(printed)
    return printed

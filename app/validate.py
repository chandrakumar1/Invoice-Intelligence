"""
What:     Deterministic business checks on an assembled Invoice. No LLM, no I/O, no hidden clock.
In:       A typed Invoice and `today` (passed in, so tests and reruns are reproducible).
Out:      list[ValidationFlag], one per problem, each with a human-readable message. Empty means every check that could run passed.
Fails if: A check invents a comparison input that isn't printed. Each check skips instead when an input is missing,
          because the missing field is already visible (and flagged, if it's required).

Validation may compute values to compare against (a sum, a product), but never writes them into the Invoice.
"""

import re
from collections.abc import Callable
from datetime import date
from decimal import ROUND_HALF_UP, Decimal

from app.models import Extracted, FlagCode, Invoice, Severity, TaxScheme, ValidationFlag

CENT = Decimal("0.01")  # one minor currency unit (paisa, euro cent)
HALF_CENT = Decimal("0.005")

# Arithmetic tolerances. "diff" is |printed value - value computed from other printed values|, unrounded.
#
#   diff <= ROUNDING_TOLERANCE      -> pass.    Rounding a value to the nearest cent moves it by at most half a cent.
#   diff <= the check's tolerance   -> WARNING. Explainable by rounding that happened earlier (see each check).
#   larger                          -> ERROR.   Not explainable by rounding.
#
# Per check:
#   line amount   qty x unit_price vs amount         tolerance: half a cent per unit of quantity (a unit price rounded
#                                                               to the cent drifts by up to that much per unit)
#   items sum     sum(line amounts) vs subtotal      tolerance: per_line_rounding_tolerance(number of line items)
#   tax lines sum sum(tax lines) vs tax_total        tolerance: per_line_rounding_tolerance(number of tax lines)
#   tax amount    taxable_value x rate vs tax line   tolerance: per_line_rounding_tolerance(number of line items)
#   totals        subtotal + tax + adjustments       tolerance: per_line_rounding_tolerance(number of line items)
#                 vs grand_total
#
# Every tolerance is at least one cent. Boundaries are inclusive and pinned by tests/test_validate.py.
ROUNDING_TOLERANCE = HALF_CENT

# Fields an invoice can't do without. Due date, tax IDs and currency are often legitimately absent.
REQUIRED_FIELDS = ("invoice_number", "invoice_date", "seller.name", "buyer.name", "grand_total")


def per_line_rounding_tolerance(n_line_items: int) -> Decimal:
    """Half a cent per line item, never less than one cent."""
    return max(CENT, HALF_CENT * n_line_items)


def validate(invoice: Invoice, today: date) -> list[ValidationFlag]:
    flags: list[ValidationFlag] = []
    for check in (
        check_required_fields,
        check_line_amounts,
        check_line_items_sum,
        check_tax_lines_sum,
        check_tax_amounts,
        check_totals,
        check_tax_ids,
        check_gst_type,
    ):
        flags += check(invoice)
    flags += check_dates(invoice, today)
    return flags


# --- Shared helpers ------------------------------------------------------------------------


def _grade(diff: Decimal, tolerance: Decimal) -> Severity | None:
    if diff <= ROUNDING_TOLERANCE:
        return None
    return Severity.WARNING if diff <= tolerance else Severity.ERROR


def _money(d: Decimal) -> str:
    return str(d.quantize(CENT, ROUND_HALF_UP))


def _arithmetic_flag(code: FlagCode, severity: Severity, paths: list[str], what: str, expected: Decimal, actual: Decimal) -> ValidationFlag:
    diff = abs(actual - expected)
    message = f"{what}: printed {actual}, expected {_money(expected)} (difference {_money(diff)})"
    if severity is Severity.WARNING:
        message += "; within rounding tolerance"
    return ValidationFlag(code=code, severity=severity, field_paths=paths, message=message, expected=_money(expected), actual=str(actual))


def _value(invoice: Invoice, path: str) -> Extracted:
    node: object = invoice
    for part in path.split("."):
        node = getattr(node, part)
    assert isinstance(node, Extracted), path
    return node


# --- Required fields -----------------------------------------------------------------------


def check_required_fields(invoice: Invoice) -> list[ValidationFlag]:
    flags = []
    for path in REQUIRED_FIELDS:
        field = _value(invoice, path)
        if field.value is None:
            flags.append(
                ValidationFlag(
                    code=FlagCode.REQUIRED_FIELD_MISSING,
                    severity=Severity.ERROR,
                    field_paths=[path],
                    message=f"Required field {path} is missing: {field.missing_reason}",
                )
            )
    return flags


# --- Arithmetic ----------------------------------------------------------------------------


def check_line_amounts(invoice: Invoice) -> list[ValidationFlag]:
    """quantity x unit_price vs the printed line amount, per row. Skipped for rows missing any of the three."""
    flags = []
    for i, item in enumerate(invoice.line_items):
        if item.quantity is None or item.unit_price is None or item.amount is None:
            continue
        expected = item.quantity * item.unit_price
        tolerance = max(CENT, HALF_CENT * abs(item.quantity))
        if severity := _grade(abs(item.amount - expected), tolerance):
            flags.append(
                _arithmetic_flag(
                    FlagCode.LINE_AMOUNT_MISMATCH, severity,
                    [f"line_items.{i}.amount", f"line_items.{i}.quantity", f"line_items.{i}.unit_price"],
                    f"Line {i + 1} ({item.description or 'no description'}): {item.quantity} x {item.unit_price}",
                    expected, item.amount,
                )
            )
    return flags


def check_line_items_sum(invoice: Invoice) -> list[ValidationFlag]:
    """Sum of line amounts vs the printed subtotal. Skipped without a subtotal, without rows, or if any amount is missing."""
    subtotal = invoice.subtotal.value
    amounts = [item.amount for item in invoice.line_items]
    if subtotal is None or not amounts or any(a is None for a in amounts):
        return []
    expected = sum(amounts, Decimal(0))
    if severity := _grade(abs(subtotal - expected), per_line_rounding_tolerance(len(amounts))):
        return [
            _arithmetic_flag(
                FlagCode.LINE_ITEMS_SUM_MISMATCH, severity, ["subtotal", "line_items"],
                f"Subtotal vs the sum of {len(amounts)} line amounts", expected, subtotal,
            )
        ]
    return []


def check_tax_lines_sum(invoice: Invoice) -> list[ValidationFlag]:
    """Sum of tax lines vs a printed tax total. Most invoices print no tax total; then this is skipped."""
    total = invoice.tax_total.value
    amounts = [t.amount.value for t in invoice.tax_lines]
    if total is None or not amounts or any(a is None for a in amounts):
        return []
    expected = sum(amounts, Decimal(0))
    if severity := _grade(abs(total - expected), per_line_rounding_tolerance(len(amounts))):
        return [
            _arithmetic_flag(
                FlagCode.TAX_LINES_SUM_MISMATCH, severity, ["tax_total", "tax_lines"],
                f"Tax total vs the sum of {len(amounts)} tax lines", expected, total,
            )
        ]
    return []


def check_tax_amounts(invoice: Invoice) -> list[ValidationFlag]:
    """Each printed tax amount vs printed taxable value x printed rate. Same rule for GST and VAT.

    Skipped, with no flag, when the taxable value, rate or amount wasn't extracted. Substituting
    e.g. the subtotal would be inventing a tax base.
    """
    base = invoice.taxable_value.value
    if base is None:
        return []
    tolerance = per_line_rounding_tolerance(len(invoice.line_items))
    flags = []
    for i, line in enumerate(invoice.tax_lines):
        rate, printed = line.rate.value, line.amount.value
        if rate is None or printed is None:
            continue
        expected = base * rate / 100
        if severity := _grade(abs(printed - expected), tolerance):
            name = line.label.value or f"Tax line {i + 1}"
            flags.append(
                _arithmetic_flag(
                    FlagCode.TAX_AMOUNT_MISMATCH, severity,
                    [f"tax_lines.{i}.amount", f"tax_lines.{i}.rate", "taxable_value"],
                    f"{name}: {rate}% of taxable value {base}", expected, printed,
                )
            )
    return flags


def check_totals(invoice: Invoice) -> list[ValidationFlag]:
    """subtotal + tax + adjustments vs the printed grand total.

    Tax is the printed tax total if there is one, otherwise the sum of the printed tax lines
    (none printed means no tax). Skipped if the subtotal, grand total, any tax amount or any
    adjustment amount is missing.
    """
    subtotal, grand = invoice.subtotal.value, invoice.grand_total.value
    tax_amounts = [t.amount.value for t in invoice.tax_lines]
    adjustments = [a.amount.value for a in invoice.adjustments]
    if subtotal is None or grand is None or None in adjustments:
        return []
    if invoice.tax_total.value is not None:
        tax, tax_path = invoice.tax_total.value, "tax_total"
    elif None in tax_amounts:
        return []
    else:
        tax, tax_path = sum(tax_amounts, Decimal(0)), "tax_lines"
    expected = subtotal + tax + sum(adjustments, Decimal(0))
    if severity := _grade(abs(grand - expected), per_line_rounding_tolerance(len(invoice.line_items))):
        what = f"Grand total vs subtotal {subtotal} + tax {tax}"
        if adjustments:
            what += f" + adjustments {sum(adjustments, Decimal(0))}"
        return [
            _arithmetic_flag(FlagCode.TOTALS_MISMATCH, severity, ["grand_total", "subtotal", tax_path, "adjustments"], what, expected, grand)
        ]
    return []


# --- Dates ---------------------------------------------------------------------------------


def check_dates(invoice: Invoice, today: date) -> list[ValidationFlag]:
    """Invoice date not after `today`; due date not before the invoice date. Unparseable dates were flagged during assembly."""
    flags = []
    issued, due = invoice.invoice_date.value, invoice.due_date.value
    if issued is not None and issued > today:
        flags.append(
            ValidationFlag(
                code=FlagCode.DATE_IN_FUTURE, severity=Severity.ERROR, field_paths=["invoice_date"],
                message=f"Invoice date {issued.isoformat()} is after today ({today.isoformat()})",
                expected=f"on or before {today.isoformat()}", actual=issued.isoformat(),
            )
        )
    if issued is not None and due is not None and due < issued:
        flags.append(
            ValidationFlag(
                code=FlagCode.DUE_BEFORE_INVOICE_DATE, severity=Severity.ERROR, field_paths=["due_date", "invoice_date"],
                message=f"Due date {due.isoformat()} is before the invoice date {issued.isoformat()}",
                expected=f"on or after {issued.isoformat()}", actual=due.isoformat(),
            )
        )
    return flags


# --- Tax IDs -------------------------------------------------------------------------------

_GSTIN_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_GSTIN_FORMAT = re.compile(r"^(\d{2})[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z]$")
_GSTIN_STATES = {f"{n:02d}" for n in range(1, 39)} | {"97", "99"}  # 97: other territory, 99: centre jurisdiction

# EU VAT number formats after the 2-letter prefix: (pattern, plain description). Format only; national
# checksums are not verified.
EU_VAT_FORMATS = {
    "AT": (r"U\d{8}", "U and 8 digits"), "BE": (r"[01]\d{9}", "10 digits starting with 0 or 1"),
    "BG": (r"\d{9,10}", "9 or 10 digits"), "CY": (r"\d{8}[A-Z]", "8 digits and a letter"),
    "CZ": (r"\d{8,10}", "8 to 10 digits"), "DE": (r"\d{9}", "9 digits"), "DK": (r"\d{8}", "8 digits"),
    "EE": (r"\d{9}", "9 digits"), "EL": (r"\d{9}", "9 digits"),
    "ES": (r"[A-Z0-9]\d{7}[A-Z0-9]", "a letter or digit, 7 digits, a letter or digit"), "FI": (r"\d{8}", "8 digits"),
    "FR": (r"[A-HJ-NP-Z0-9]{2}\d{9}", "2 letters or digits and 9 digits"), "HR": (r"\d{11}", "11 digits"),
    "HU": (r"\d{8}", "8 digits"), "IE": (r"\d{7}[A-W][A-I]?|\d[A-Z+*]\d{5}[A-W]", "8 or 9 letters and digits"),
    "IT": (r"\d{11}", "11 digits"), "LT": (r"\d{9}|\d{12}", "9 or 12 digits"), "LU": (r"\d{8}", "8 digits"),
    "LV": (r"\d{11}", "11 digits"), "MT": (r"\d{8}", "8 digits"), "NL": (r"\d{9}B\d{2}", "9 digits, B, 2 digits"),
    "PL": (r"\d{10}", "10 digits"), "PT": (r"\d{9}", "9 digits"), "RO": (r"\d{2,10}", "2 to 10 digits"),
    "SE": (r"\d{12}", "12 digits"), "SI": (r"\d{8}", "8 digits"), "SK": (r"\d{10}", "10 digits"),
    "XI": (r"\d{9}|\d{12}|GD\d{3}|HA\d{3}", "9 or 12 digits, or GD/HA and 3 digits"),
}


def gstin_checksum_ok(gstin: str) -> bool:
    """Luhn mod 36 over the first 14 characters (verified against published GSTINs in the tests)."""
    factor, total = 2, 0
    for ch in reversed(gstin[:14]):
        addend = factor * _GSTIN_CHARS.index(ch)
        factor = 1 if factor == 2 else 2
        total += addend // 36 + addend % 36
    return _GSTIN_CHARS[(36 - total % 36) % 36] == gstin[14]


def _compact(raw: str) -> str:
    return re.sub(r"[\s.\-]", "", raw).upper()


def _check_gstin(raw: str) -> tuple[FlagCode, str] | None:
    gstin = _compact(raw)
    m = _GSTIN_FORMAT.match(gstin)
    if not m:
        return FlagCode.TAX_ID_INVALID_FORMAT, f"GSTIN {raw!r} doesn't have the 15-character GSTIN format"
    if m[1] not in _GSTIN_STATES:
        return FlagCode.TAX_ID_INVALID_FORMAT, f"GSTIN {raw!r} starts with {m[1]}, which is not an Indian state code"
    if not gstin_checksum_ok(gstin):
        return FlagCode.TAX_ID_INVALID_CHECKSUM, f"GSTIN {raw!r} fails its check digit (last character)"
    return None


def _check_eu_vat(raw: str) -> tuple[FlagCode, str] | None:
    vat = _compact(raw)
    country, number = vat[:2], vat[2:]
    if country not in EU_VAT_FORMATS:
        return FlagCode.TAX_ID_INVALID_FORMAT, f"VAT ID {raw!r} doesn't start with an EU country code"
    pattern, description = EU_VAT_FORMATS[country]
    if not re.fullmatch(pattern, number):
        return FlagCode.TAX_ID_INVALID_FORMAT, f"VAT ID {raw!r} doesn't match the {country} format: {country} followed by {description}"
    return None


# Pluggable: one validator per scheme. OTHER and UNKNOWN have no published format to check against.
TAX_ID_VALIDATORS: dict[TaxScheme, Callable[[str], tuple[FlagCode, str] | None]] = {
    TaxScheme.GSTIN: _check_gstin,
    TaxScheme.EU_VAT: _check_eu_vat,
}


def check_tax_ids(invoice: Invoice) -> list[ValidationFlag]:
    flags = []
    for side in ("seller", "buyer"):
        tax_id = getattr(invoice, side).tax_id.value
        if tax_id is None or (validator := TAX_ID_VALIDATORS.get(tax_id.scheme)) is None:
            continue
        if problem := validator(tax_id.raw):
            code, message = problem
            flags.append(
                ValidationFlag(code=code, severity=Severity.ERROR, field_paths=[f"{side}.tax_id"], message=message, actual=tax_id.raw)
            )
    return flags


# --- GST type vs states ----------------------------------------------------------------------


def check_gst_type(invoice: Invoice) -> list[ValidationFlag]:
    """Same-state supply should carry CGST+SGST (or UTGST), inter-state supply IGST.

    Uses the buyer's GSTIN state as a stand-in for the place of supply, which the pipeline doesn't
    extract. They usually agree but can differ (e.g. goods shipped to another state), hence WARNING.
    Skipped unless both GSTINs are well-formed and the tax lines name a GST type.
    """
    seller, buyer = invoice.seller.tax_id.value, invoice.buyer.tax_id.value
    if seller is None or buyer is None or seller.scheme is not TaxScheme.GSTIN or buyer.scheme is not TaxScheme.GSTIN:
        return []
    if _check_gstin(seller.raw) or _check_gstin(buyer.raw):
        return []  # a malformed GSTIN is already flagged; its state code can't be trusted
    labels = " ".join((t.label.value or "") for t in invoice.tax_lines).upper()
    has_igst = bool(re.search(r"\bIGST\b", labels))
    has_split = bool(re.search(r"\b(CGST|SGST|UTGST)\b", labels))
    if not (has_igst or has_split):
        return []
    s_state, b_state = _compact(seller.raw)[:2], _compact(buyer.raw)[:2]
    if s_state == b_state and has_igst:
        problem = f"seller and buyer are both in state {s_state}, but IGST (inter-state tax) is charged"
    elif s_state != b_state and has_split:
        problem = f"seller is in state {s_state} and buyer in {b_state}, but CGST/SGST (same-state tax) is charged"
    else:
        return []
    return [
        ValidationFlag(
            code=FlagCode.GST_TYPE_STATE_MISMATCH, severity=Severity.WARNING,
            field_paths=["tax_lines", "seller.tax_id", "buyer.tax_id"],
            message=f"GST type doesn't match the states: {problem} (based on GSTIN state codes, not the printed place of supply)",
        )
    ]

"""
What:     Unit tests for app/validate.py: every check, its tolerance boundaries, and its skip conditions; plus fixture end-to-end flags.
In:       Hand-built Invoice objects (numbers mirror the fixture invoices where noted) and the mocked fixture answers.
Out:      pytest pass/fail.
Fails if: A tolerance or rule changes. These tests pin the pass / warning / error boundaries.
"""

from datetime import date
from decimal import Decimal

import pytest

from app.assemble import build_invoice
from app.models import (
    Adjustment,
    AdjustmentKind,
    Extracted,
    FlagCode,
    Invoice,
    LineItem,
    LLMInvoiceExtraction,
    MatchType,
    Party,
    Severity,
    SourceRef,
    TaxId,
    TaxLine,
    TaxScheme,
)
from app.validate import (
    REQUIRED_FIELDS,
    check_dates,
    check_gst_type,
    check_line_amounts,
    check_line_items_sum,
    check_required_fields,
    check_tax_amounts,
    check_tax_ids,
    check_tax_lines_sum,
    check_totals,
    gstin_checksum_ok,
    per_line_rounding_tolerance,
    validate,
)
from tests.helpers import expected, pages_of, wire_01, wire_02, wire_03, wire_05


def ref(quote: str = "x") -> SourceRef:
    return SourceRef(
        page=1, line_start=1, line_end=1, quote=quote, value_span=None,
        match_type=MatchType.EXACT, text_source="text_layer", confidence=1.0,
    )


def val(v):
    return Extracted(value=v, source=ref())


def gone(reason: str = "not printed") -> Extracted:
    return Extracted.missing(reason)


def tax_line(label: str | None, rate: str | None, amount: str | None) -> TaxLine:
    return TaxLine(
        label=val(label) if label else gone(),
        rate=val(Decimal(rate)) if rate else gone(),
        amount=val(Decimal(amount)) if amount else gone(),
    )


def line_item() -> LineItem:
    return LineItem(
        description="x", hsn_sac=None, quantity=None, unit_price=None, amount=None,
        source=ref(), cell_spans={"description": (0, 1)},
        missing_reasons={c: "not printed" for c in ("hsn_sac", "quantity", "unit_price", "amount")},
    )


def invoice(taxable_value: str | None, tax_lines: list[TaxLine], n_items: int = 1) -> Invoice:
    party = Party(name=gone(), address=gone(), tax_id=gone())
    return Invoice(
        invoice_number=gone(), invoice_date=gone(), due_date=gone(), currency=gone(),
        seller=party, buyer=party, line_items=[line_item() for _ in range(n_items)],
        subtotal=gone(),
        taxable_value=val(Decimal(taxable_value)) if taxable_value else gone("no taxable value printed"),
        tax_lines=tax_lines, tax_total=gone(), adjustments=[], grand_total=gone(),
    )


def test_exact_match_passes():
    # fixture 02: IGST 18% of 152,355.00 = 27,423.90
    assert check_tax_amounts(invoice("152355.00", [tax_line("IGST @ 18%", "18", "27423.90")])) == []


def test_rounding_to_cent_passes():
    # fixture 03: 9% of 5,63,888.15 = 50,749.9335, printed 50,749.93
    lines = [tax_line("CGST @ 9%", "9", "50749.93"), tax_line("SGST @ 9%", "9", "50749.93")]
    assert check_tax_amounts(invoice("563888.15", lines, n_items=30)) == []


def test_per_line_rounding_drift_is_a_warning():
    # 0.03 off with 30 line items: within 30 x half a cent = 0.15
    [flag] = check_tax_amounts(invoice("563888.15", [tax_line("CGST @ 9%", "9", "50749.96")], n_items=30))
    assert flag.code is FlagCode.TAX_AMOUNT_MISMATCH
    assert flag.severity is Severity.WARNING
    assert "rounding tolerance" in flag.message


def test_real_mismatch_is_an_error():
    [flag] = check_tax_amounts(invoice("67367.00", [tax_line("CGST @ 9%", "9", "6163.03")], n_items=5))
    assert flag.severity is Severity.ERROR
    assert (flag.expected, flag.actual) == ("6063.03", "6163.03")
    assert flag.field_paths == ["tax_lines.0.amount", "tax_lines.0.rate", "taxable_value"]
    assert "CGST @ 9%" in flag.message


def test_single_item_tolerance_floor_is_one_cent():
    # One line item: tolerance is max(0.01, 0.005); 0.01 off is a warning, 0.02 is an error.
    assert check_tax_amounts(invoice("100.00", [tax_line("GST", "18", "18.01")]))[0].severity is Severity.WARNING
    assert check_tax_amounts(invoice("100.00", [tax_line("GST", "18", "18.02")]))[0].severity is Severity.ERROR


@pytest.mark.parametrize(
    "n_items,expected",
    [(0, "0.01"), (1, "0.01"), (2, "0.01"), (3, "0.015"), (30, "0.15")],
)
def test_per_line_tolerance_values(n_items, expected):
    assert per_line_rounding_tolerance(n_items) == Decimal(expected)


def severity_of(taxable: str, rate: str, printed: str, n_items: int = 1, label: str = "GST") -> Severity | None:
    flags = check_tax_amounts(invoice(taxable, [tax_line(label, rate, printed)], n_items=n_items))
    return flags[0].severity if flags else None


# Exact products used below: 10% of 100.05 = 10.005; 10% of 100.06 = 10.006.
@pytest.mark.parametrize(
    "taxable,printed,n_items,expected",
    [
        # pass / warning boundary at half a cent (inclusive)
        ("100.05", "10.01", 1, None),  # diff 0.005, exactly at ROUNDING_TOLERANCE: pass
        ("100.05", "10.00", 1, None),  # diff 0.005 the other way: pass
        ("100.06", "10.01", 1, None),  # diff 0.004: pass
        ("100.06", "10.00", 1, Severity.WARNING),  # diff 0.006, just over: warning
        # warning / error boundary at per_line_rounding_tolerance(n) (inclusive)
        ("100.05", "10.02", 3, Severity.WARNING),  # diff 0.015 == tolerance(3) 0.015: warning
        ("100.05", "10.02", 2, Severity.ERROR),  # diff 0.015 > tolerance(2) 0.01: error
        ("100.06", "10.02", 3, Severity.WARNING),  # diff 0.014 < 0.015: warning
        ("100.06", "9.99", 3, Severity.ERROR),  # diff 0.016 > 0.015: error
    ],
)
def test_tolerance_boundaries(taxable, printed, n_items, expected):
    assert severity_of(taxable, "10", printed, n_items) is expected


@pytest.mark.parametrize("label", ["VAT 19%", "IGST @ 19%", "CGST @ 19%"])
def test_same_rule_for_vat_and_gst(label):
    # 19% of 1,457.00 = 276.83 exactly (fixture 05's VAT line, with a hypothetical printed tax base)
    assert severity_of("1457.00", "19", "276.83", label=label) is None
    assert severity_of("1457.00", "19", "276.93", label=label) is Severity.ERROR


def test_only_the_bad_line_is_flagged():
    lines = [tax_line("CGST @ 9%", "9", "6063.03"), tax_line("SGST @ 9%", "9", "6000.00")]
    [flag] = check_tax_amounts(invoice("67367.00", lines, n_items=5))
    assert flag.field_paths[0] == "tax_lines.1.amount"


@pytest.mark.parametrize(
    "taxable_value,line",
    [
        (None, tax_line("CGST @ 9%", "9", "999.99")),  # no taxable value printed (fixture 01)
        ("67367.00", tax_line("CGST", None, "999.99")),  # rate not printed
        ("67367.00", tax_line("CGST @ 9%", "9", None)),  # amount not extracted
    ],
)
def test_skipped_when_a_printed_input_is_missing(taxable_value, line):
    assert check_tax_amounts(invoice(taxable_value, [line])) == []


def test_unlabelled_line_still_gets_a_readable_message():
    [flag] = check_tax_amounts(invoice("100.00", [tax_line(None, "18", "25.00")]))
    assert flag.message.startswith("Tax line 1:")


# ================================================================================================
# M4: the remaining checks. A general builder: every field missing unless given.
# ================================================================================================

TODAY = date(2026, 11, 12)


def d(x: str | None) -> Extracted:
    return val(Decimal(x)) if x is not None else gone()


def item(qty: str | None, price: str | None, amount: str | None, description: str = "Widget") -> LineItem:
    values = {"quantity": qty, "unit_price": price, "amount": amount}
    present = {k: Decimal(v) for k, v in values.items() if v is not None}
    return LineItem(
        description=description, hsn_sac=None, **{k: present.get(k) for k in values},
        source=ref("row"), cell_spans={"description": (0, 1)} | {k: (0, 1) for k in present},
        missing_reasons={"hsn_sac": "not printed"} | {k: "not printed" for k in values if k not in present},
    )


def gst_party(raw: str | None, scheme: TaxScheme = TaxScheme.GSTIN, name: str | None = "Acme") -> Party:
    tax_id = Extracted(value=TaxId(raw=raw, scheme=scheme), source=ref()) if raw else gone()
    return Party(name=val(name) if name else gone(), address=gone(), tax_id=tax_id)


def make(**fields) -> Invoice:
    base = dict(
        invoice_number=gone(), invoice_date=gone(), due_date=gone(), currency=gone(),
        seller=gst_party(None), buyer=gst_party(None), line_items=[], subtotal=gone(), taxable_value=gone(),
        tax_lines=[], tax_total=gone(), adjustments=[], grand_total=gone(),
    )
    return Invoice(**(base | fields))


def adjustment(kind: AdjustmentKind, amount: str | None) -> Adjustment:
    return Adjustment(kind=kind, label=gone(), amount=d(amount))


def codes(flags) -> list[tuple[FlagCode, Severity]]:
    return [(f.code, f.severity) for f in flags]


# --- Required fields -----------------------------------------------------------------------


def complete_header() -> dict:
    return dict(
        invoice_number=val("INV-1"), invoice_date=val(date(2025, 11, 5)),
        seller=gst_party(None, name="Seller Co"), buyer=gst_party(None, name="Buyer Co"), grand_total=d("100.00"),
    )


def test_required_fields_present():
    assert check_required_fields(make(**complete_header())) == []


@pytest.mark.parametrize("path", REQUIRED_FIELDS)
def test_each_required_field_is_an_error_with_the_reason(path):
    fields = complete_header()
    if "." in path:
        side = path.split(".")[0]
        fields[side] = gst_party(None, name=None)
    else:
        fields[path] = gone("citation not verifiable: 'X' is not on the cited lines")
    [flag] = check_required_fields(make(**fields))
    assert (flag.code, flag.severity, flag.field_paths) == (FlagCode.REQUIRED_FIELD_MISSING, Severity.ERROR, [path])
    assert "is missing:" in flag.message


def test_optional_fields_may_be_missing():
    # due date, currency and tax IDs are often legitimately absent (05 has no due date)
    assert check_required_fields(make(**complete_header())) == []


# --- Line amounts ----------------------------------------------------------------------------


def test_line_amount_exact():
    assert check_line_amounts(make(line_items=[item("50", "231.79", "11589.50")])) == []


def test_line_amount_unit_price_rounding_is_a_warning():
    # 33.333... printed as 33.33: 3 x 33.33 = 99.99 vs 100.00; tolerance 3 x 0.005 = 0.015
    [flag] = check_line_amounts(make(line_items=[item("3", "33.33", "100.00")]))
    assert (flag.code, flag.severity) == (FlagCode.LINE_AMOUNT_MISMATCH, Severity.WARNING)


@pytest.mark.parametrize(
    "qty,price,amount,severity",
    [
        ("2", "10.005", "20.01", None),  # 20.010 exactly
        ("1", "10.004", "10.00", None),  # diff 0.004: rounding to the cent
        ("1", "10.00", "10.01", Severity.WARNING),  # diff 0.01 == max(0.01, 0.005): warning
        ("1", "10.00", "10.02", Severity.ERROR),
        ("100", "1.00", "100.50", Severity.WARNING),  # tolerance 100 x 0.005 = 0.50, inclusive
        ("100", "1.00", "100.51", Severity.ERROR),
    ],
)
def test_line_amount_boundaries(qty, price, amount, severity):
    flags = check_line_amounts(make(line_items=[item(qty, price, amount)]))
    assert (flags[0].severity if flags else None) is severity


def test_line_amount_error_names_the_row():
    [flag] = check_line_amounts(make(line_items=[item("1", "1.00", "1.00"), item("12", "410.00", "4290.00", "Ball Pen Blue")]))
    assert flag.field_paths[0] == "line_items.1.amount"
    assert "Line 2 (Ball Pen Blue): 12 x 410.00" in flag.message and flag.expected == "4920.00"


def test_line_amount_skipped_when_a_cell_is_missing():
    assert check_line_amounts(make(line_items=[item("12", None, "4290.00")])) == []


# --- Line items vs subtotal -----------------------------------------------------------------


def test_items_sum_matches_subtotal():
    inv = make(line_items=[item(None, None, "11400.00"), item(None, None, "4920.00")], subtotal=d("16320.00"))
    assert check_line_items_sum(inv) == []


def test_items_sum_mismatch_like_fixture_05():
    amounts = ["373.50", "472.00", "223.50", "378.00"]  # sum 1,447.00; printed net total 1,457.00
    [flag] = check_line_items_sum(make(line_items=[item(None, None, a) for a in amounts], subtotal=d("1457.00")))
    assert (flag.code, flag.severity) == (FlagCode.LINE_ITEMS_SUM_MISMATCH, Severity.ERROR)
    assert (flag.expected, flag.actual) == ("1447.00", "1457.00")


def test_items_sum_small_difference_is_a_warning():
    inv = make(line_items=[item(None, None, "10.00")] * 3, subtotal=d("30.01"))  # tolerance 3 x 0.005 = 0.015
    assert codes(check_line_items_sum(inv)) == [(FlagCode.LINE_ITEMS_SUM_MISMATCH, Severity.WARNING)]


@pytest.mark.parametrize(
    "items,subtotal",
    [([item(None, None, "10.00")], None), ([], "10.00"), ([item(None, None, "10.00"), item("1", "5.00", None)], "15.00")],
)
def test_items_sum_skipped_without_complete_inputs(items, subtotal):
    assert check_line_items_sum(make(line_items=items, subtotal=d(subtotal))) == []


# --- Tax lines vs tax total -----------------------------------------------------------------


def test_tax_lines_sum_matches_printed_total():
    inv = make(tax_lines=[tax_line("CGST", "9", "6063.03"), tax_line("SGST", "9", "6063.03")], tax_total=d("12126.06"))
    assert check_tax_lines_sum(inv) == []


def test_tax_lines_sum_mismatch():
    inv = make(tax_lines=[tax_line("CGST", "9", "6063.03"), tax_line("SGST", "9", "6063.03")], tax_total=d("12226.06"))
    assert codes(check_tax_lines_sum(inv)) == [(FlagCode.TAX_LINES_SUM_MISMATCH, Severity.ERROR)]


def test_tax_lines_sum_skipped_without_a_printed_total():
    assert check_tax_lines_sum(make(tax_lines=[tax_line("CGST", "9", "6063.03")])) == []


# --- Totals ------------------------------------------------------------------------------------


def test_totals_with_discount_and_round_off_like_fixture_03():
    inv = make(
        subtotal=d("565388.15"),
        tax_lines=[tax_line("CGST @ 9%", "9", "50749.93"), tax_line("SGST @ 9%", "9", "50749.93")],
        adjustments=[adjustment(AdjustmentKind.DISCOUNT, "-1500.00"), adjustment(AdjustmentKind.ROUND_OFF, "-0.01")],
        grand_total=d("665388.00"),
        line_items=[item(None, None, "1.00")] * 30,
    )
    assert check_totals(inv) == []


def test_totals_mismatch_explains_the_parts():
    inv = make(subtotal=d("1457.00"), tax_lines=[tax_line("VAT 19%", "19", "276.83")], grand_total=d("1743.83"))
    [flag] = check_totals(inv)
    assert (flag.code, flag.severity, flag.expected, flag.actual) == (FlagCode.TOTALS_MISMATCH, Severity.ERROR, "1733.83", "1743.83")
    assert "subtotal 1457.00 + tax 276.83" in flag.message


def test_totals_prefer_a_printed_tax_total():
    inv = make(subtotal=d("100.00"), tax_lines=[tax_line("GST", "18", "18.00")], tax_total=d("20.00"), grand_total=d("120.00"))
    assert check_totals(inv) == []  # the tax-total-vs-lines discrepancy is check_tax_lines_sum's job


def test_totals_without_tax_lines_means_no_tax():
    assert check_totals(make(subtotal=d("100.00"), grand_total=d("100.00"))) == []


@pytest.mark.parametrize(
    "fields",
    [
        dict(subtotal=d(None), grand_total=d("100.00")),
        dict(subtotal=d("100.00"), grand_total=d(None)),
        dict(subtotal=d("100.00"), grand_total=d("118.00"), tax_lines=[tax_line("GST", "18", None)]),
        dict(subtotal=d("100.00"), grand_total=d("90.00"), adjustments=[adjustment(AdjustmentKind.DISCOUNT, None)]),
    ],
)
def test_totals_skipped_without_complete_inputs(fields):
    assert check_totals(make(**fields)) == []


# --- Dates -------------------------------------------------------------------------------------


def test_dates_ok():
    assert check_dates(make(invoice_date=val(date(2025, 11, 5)), due_date=val(date(2025, 11, 20))), TODAY) == []


def test_invoice_dated_today_is_fine_tomorrow_is_not():
    assert check_dates(make(invoice_date=val(TODAY)), TODAY) == []
    [flag] = check_dates(make(invoice_date=val(date(2026, 11, 13))), TODAY)
    assert (flag.code, flag.severity, flag.actual) == (FlagCode.DATE_IN_FUTURE, Severity.ERROR, "2026-11-13")


def test_due_date_in_the_future_is_normal():
    assert check_dates(make(invoice_date=val(TODAY), due_date=val(date(2027, 1, 1))), TODAY) == []


def test_due_before_invoice_date():
    [flag] = check_dates(make(invoice_date=val(date(2025, 11, 20)), due_date=val(date(2025, 11, 5))), TODAY)
    assert (flag.code, flag.field_paths) == (FlagCode.DUE_BEFORE_INVOICE_DATE, ["due_date", "invoice_date"])


def test_dates_skipped_when_missing():
    assert check_dates(make(due_date=val(date(2020, 1, 1))), TODAY) == []


# --- Tax IDs -----------------------------------------------------------------------------------


@pytest.mark.parametrize("gstin", ["27AAPFU0939F1ZV", "29AAGCB7383J1Z4", "33AAACH7409R1Z8", "27AAACR5055K1Z7"])
def test_published_gstins_pass_the_checksum(gstin):
    assert gstin_checksum_ok(gstin)
    assert check_tax_ids(make(seller=gst_party(gstin))) == []


@pytest.mark.parametrize(
    "raw,code",
    [
        ("27AAPFU0939F1ZW", FlagCode.TAX_ID_INVALID_CHECKSUM),  # last character changed
        ("27AAPFU0939F1Z", FlagCode.TAX_ID_INVALID_FORMAT),  # 14 characters
        ("40AAPFU0939F1ZV", FlagCode.TAX_ID_INVALID_FORMAT),  # 40 is not a state code
        ("27AAPFU0939F1AV", FlagCode.TAX_ID_INVALID_FORMAT),  # 14th character must be Z
        ("27AAPFUO939F1ZV", FlagCode.TAX_ID_INVALID_FORMAT),  # OCR: letter O in the digits
    ],
)
def test_bad_gstins(raw, code):
    [flag] = check_tax_ids(make(buyer=gst_party(raw)))
    assert (flag.code, flag.severity, flag.field_paths, flag.actual) == (code, Severity.ERROR, ["buyer.tax_id"], raw)


def test_gstin_spacing_and_case_are_normalized_for_checking():
    assert check_tax_ids(make(seller=gst_party("27 aapfu 0939 f1zv"))) == []


@pytest.mark.parametrize("raw", ["DE123456789", "NL853746219B01", "FR40303265045", "ATU12345678", "DE 123 456 789"])
def test_valid_eu_vat_formats(raw):
    assert check_tax_ids(make(seller=gst_party(raw, TaxScheme.EU_VAT))) == []


@pytest.mark.parametrize(
    "raw,why",
    [
        ("DE12345678", "doesn't match the DE format"),  # fixture 05: one digit short
        ("NL853746219", "doesn't match the NL format"),
        ("123456789", "doesn't start with an EU country code"),
        ("US123456789", "doesn't start with an EU country code"),
    ],
)
def test_invalid_eu_vat_formats(raw, why):
    [flag] = check_tax_ids(make(seller=gst_party(raw, TaxScheme.EU_VAT)))
    assert flag.code is FlagCode.TAX_ID_INVALID_FORMAT and why in flag.message


@pytest.mark.parametrize("scheme", [TaxScheme.OTHER, TaxScheme.UNKNOWN])
def test_schemes_without_a_known_format_are_not_checked(scheme):
    assert check_tax_ids(make(seller=gst_party("anything at all", scheme))) == []


# --- GST type vs states --------------------------------------------------------------------------

KA_1, KA_2, MH, DL, UP = "29AAGCB7383J1Z4", "29AAGCS4821K1Z0", "27AAPFU0939F1ZV", "07AAHFY2206P1ZS", "09ABCFL6640R1ZM"
SPLIT = [tax_line("CGST @ 9%", "9", "9.00"), tax_line("SGST @ 9%", "9", "9.00")]
IGST = [tax_line("IGST @ 18%", "18", "18.00")]


@pytest.mark.parametrize(
    "seller,buyer,lines,flagged",
    [
        (KA_1, KA_2, SPLIT, False),  # same state, CGST+SGST (fixture 01)
        (MH, KA_1, IGST, False),  # inter-state, IGST (fixture 02)
        (DL, UP, IGST, False),  # fixture 04
        (KA_1, KA_2, IGST, True),
        (MH, KA_1, SPLIT, True),
        (MH, KA_1, [tax_line("VAT 19%", "19", "1.00")], False),  # no GST type named
    ],
)
def test_gst_type_vs_states(seller, buyer, lines, flagged):
    flags = check_gst_type(make(seller=gst_party(seller), buyer=gst_party(buyer), tax_lines=lines))
    assert codes(flags) == ([(FlagCode.GST_TYPE_STATE_MISMATCH, Severity.WARNING)] if flagged else [])


def test_gst_type_skipped_for_malformed_or_non_gst_ids():
    assert check_gst_type(make(seller=gst_party("27AAPFU0939F1ZW"), buyer=gst_party(KA_1), tax_lines=SPLIT)) == []
    assert check_gst_type(make(seller=gst_party("DE123456789", TaxScheme.EU_VAT), buyer=gst_party(KA_1), tax_lines=SPLIT)) == []


# --- End to end: the flags match the owner's expected_flags ----------------------------------------


@pytest.mark.parametrize(
    "stem,wire",
    [("01_gst_classic", wire_01), ("02_igst_totals_top", wire_02), ("03_multipage_roundoff", wire_03), ("05_eu_vat_defective", wire_05)],
)
def test_fixture_flags_match_ground_truth(stem, wire):
    assembly = build_invoice(LLMInvoiceExtraction.model_validate(wire()), list(pages_of(stem)))
    flags = assembly.flags + validate(assembly.invoice, TODAY)
    assert {f.code for f in flags} == set(expected(stem).expected_flags), [f.message for f in flags]


def test_fixture_05_messages_are_readable():
    assembly = build_invoice(LLMInvoiceExtraction.model_validate(wire_05()), list(pages_of("05_eu_vat_defective")))
    messages = [f.message for f in validate(assembly.invoice, TODAY)]
    assert messages == [
        "Required field invoice_number is missing: not printed on the document (no invoice number is printed)",
        "Subtotal vs the sum of 4 line amounts: printed 1457.00, expected 1447.00 (difference 10.00)",
        "VAT ID 'DE12345678' doesn't match the DE format: DE followed by 9 digits",
    ]

"""
What:     Tests app/assemble.py: perfect mock answers must reproduce the owner's ground truth; bad answers must degrade visibly.
In:       Mock wire answers from tests/helpers.py, the owner's expected JSON, and small hand-built pages.
Out:      pytest pass/fail.
Fails if: Parsing or provenance rules change in a way that alters a value on a fixture invoice.
"""

from datetime import date
from decimal import Decimal

import pytest

from app.assemble import build_invoice
from app.models import AdjustmentKind, FlagCode, LLMInvoiceExtraction, MatchType, PageInfo, TaxScheme
from tests.helpers import compare, expected, pages_of, wire_01, wire_02, wire_03, wire_05


def assemble(stem: str, wire: dict):
    return build_invoice(LLMInvoiceExtraction.model_validate(wire), list(pages_of(stem)))


# --- End to end against the hand-written ground truth ------------------------------------------


@pytest.mark.parametrize(
    "stem,wire",
    [
        ("01_gst_classic", wire_01),
        ("02_igst_totals_top", wire_02),
        ("03_multipage_roundoff", wire_03),
        ("05_eu_vat_defective", wire_05),
    ],
)
def test_perfect_answer_reproduces_ground_truth(stem, wire):
    result = assemble(stem, wire())
    assert compare(result.invoice, expected(stem)) == []
    assert result.failures == []
    # assembly itself raises nothing here; 05's defects are validation's job (tests/test_validate.py)
    assert result.flags == []


def test_every_value_carries_a_verbatim_quote():
    inv = assemble("03_multipage_roundoff", wire_03()).invoice
    for field in (inv.invoice_number, inv.grand_total, inv.buyer.address, inv.adjustments[0].amount, inv.tax_lines[1].rate):
        src = field.source
        assert src is not None and field.missing_reason is None
        assert src.quote[slice(*src.value_span)].split("\n")[0] in src.quote
        assert 0 < src.confidence <= 1
    row = inv.line_items[29]
    assert row.source.page == 2 and row.source.quote.startswith("30 Copper Lug")
    assert {c: row.source.quote[slice(*s)] for c, s in row.cell_spans.items()}["amount"] == "11,757.50"


def test_hint_decided_values_score_lower():
    inv = assemble("01_gst_classic", wire_01()).invoice
    assert inv.invoice_date.value == date(2025, 11, 5)
    assert inv.invoice_date.source.confidence == 0.9  # 05/11 read day-first because the due date is 20/11
    assert inv.due_date.source.confidence == 1.0


def test_regional_formats():
    inv = assemble("05_eu_vat_defective", wire_05()).invoice
    assert inv.currency.value == "EUR"
    assert inv.subtotal.value == Decimal("1457.00") and inv.line_items[0].unit_price == Decimal("24.90")
    assert inv.seller.tax_id.value.scheme is TaxScheme.EU_VAT and inv.seller.tax_id.value.raw == "DE12345678"
    assert inv.invoice_number.value is None and "no invoice number" in inv.invoice_number.missing_reason
    assert inv.due_date.value is None and "14 days net" in inv.due_date.missing_reason


def test_adjustments_are_classified_and_signed():
    adj = assemble("03_multipage_roundoff", wire_03()).invoice.adjustments
    assert [(a.kind, a.amount.value, a.label.value) for a in adj] == [
        (AdjustmentKind.DISCOUNT, Decimal("-1500.00"), "Less: Trade Discount"),
        (AdjustmentKind.ROUND_OFF, Decimal("-0.01"), "Round Off"),
    ]


# --- Targeted failure cases on small pages -----------------------------------------------------


def blank() -> dict:
    miss = {"value_text": None, "page": None, "line_start": None, "line_end": None, "not_found_reason": "not printed"}
    party = {"name": miss, "address": miss, "tax_id": miss}
    return {k: miss for k in ("invoice_number", "invoice_date", "due_date", "currency", "subtotal", "taxable_value", "tax_total", "grand_total")} | {
        "seller": party, "buyer": party, "line_items": [], "tax_lines": [], "adjustments": []}


def cite(value: str, line: int, page: int = 1) -> dict:
    return {"value_text": value, "page": page, "line_start": line, "line_end": line, "not_found_reason": None}


def doc(*lines: str) -> list[PageInfo]:
    text = "\n".join(lines)
    return [PageInfo(page=1, text_source="text_layer", ocr_reasons=[], char_count=len(text), text=text)]


def build(wire: dict, pages: list[PageInfo]):
    return build_invoice(LLMInvoiceExtraction.model_validate(wire), pages)


def test_invented_value_is_dropped_flagged_and_queued_for_repair():
    result = build(blank() | {"invoice_number": cite("INV-404", 1)}, doc("Invoice No. SOS/0147"))
    field = result.invoice.invoice_number
    assert field.value is None and field.source is None and "citation not verifiable" in field.missing_reason
    [flag] = result.flags
    assert flag.code is FlagCode.CITATION_UNVERIFIED and flag.field_paths == ["invoice_number"]
    [failure] = result.failures
    assert failure.path == "invoice_number" and "'INV-404'" in failure.describe()


def test_wrong_line_is_corrected_not_dropped():
    result = build(blank() | {"grand_total": cite("1,234.00", 1)}, doc("Header", "Total 1,234.00"))
    src = result.invoice.grand_total.source
    assert result.invoice.grand_total.value == Decimal("1234.00")
    assert (src.line_start, src.match_type, src.confidence) == (2, MatchType.RELOCATED, 0.75)
    assert result.flags == []


def test_mistyped_value_on_the_right_line_takes_the_printed_text_at_low_confidence():
    # The model transposed two digits but pointed at the right line: the page wins, and the score shows it.
    result = build(blank() | {"invoice_number": cite("SOS/2025-26/0174", 1)}, doc("Invoice No. SOS/2025-26/0147"))
    field = result.invoice.invoice_number
    assert field.value == "SOS/2025-26/0147"
    assert (field.source.match_type, field.source.confidence) == (MatchType.FUZZY, 0.6)


def test_unparseable_value_keeps_its_source_for_review():
    result = build(blank() | {"invoice_date": cite("next Tuesday", 1)}, doc("Date: next Tuesday"))
    field = result.invoice.invoice_date
    assert field.value is None and field.source.quote == "Date: next Tuesday"
    assert result.flags[0].code is FlagCode.DATE_UNPARSEABLE


def test_ambiguous_date_without_any_hint_is_not_guessed():
    result = build(blank() | {"invoice_date": cite("05/11/2025", 1)}, doc("Date 05/11/2025"))
    assert result.invoice.invoice_date.value is None
    assert [f.code for f in result.flags] == [FlagCode.DATE_AMBIGUOUS]


def test_ambiguous_date_resolved_by_currency():
    wire = blank() | {"invoice_date": cite("05/11/2025", 1), "currency": cite("Rs.", 2)}
    result = build(wire, doc("Date 05/11/2025", "Amount (Rs.)"))
    assert result.invoice.invoice_date.value == date(2025, 11, 5)


def test_extractor_reason_is_kept_for_missing_fields():
    wire = blank() | {"due_date": {**blank()["due_date"], "not_found_reason": "only payment terms printed"}}
    assert "only payment terms printed" in build(wire, doc("x")).invoice.due_date.missing_reason


def test_discount_printed_without_sign_still_reduces_total():
    wire = blank() | {"adjustments": [{"label": cite("Discount", 1), "amount": cite("500.00", 1)}]}
    [adj] = build(wire, doc("Discount 500.00")).invoice.adjustments
    assert (adj.kind, adj.amount.value) == (AdjustmentKind.DISCOUNT, Decimal("-500.00"))


def row(line: int, **cells) -> dict:
    base = {"page": 1, "line_start": line, "line_end": line, "description": None, "hsn_sac": None, "quantity": None, "unit_price": None, "amount": None}
    return base | cells


def test_cell_not_on_the_row_is_dropped_and_flagged():
    wire = blank() | {"line_items": [row(1, description="Ball Pen Blue", quantity="12", amount="9,999.00")]}
    result = build(wire, doc("2 Ball Pen Blue 9608 12 410.00 4,920.00"))
    [item] = result.invoice.line_items
    assert item.description == "Ball Pen Blue" and item.quantity == Decimal("12")
    assert item.amount is None and "citation not verifiable" in item.missing_reasons["amount"]
    assert item.missing_reasons["hsn_sac"] == "not printed in this row"
    assert [f.path for f in result.failures] == ["line_items.0.amount"]


def test_row_with_no_verifiable_cell_is_dropped():
    wire = blank() | {"line_items": [row(1, description="Invented Widget", amount="1.00"), row(1, description="Ball Pen Blue")]}
    result = build(wire, doc("2 Ball Pen Blue 9608 12 410.00 4,920.00"))
    assert [i.description for i in result.invoice.line_items] == ["Ball Pen Blue"]
    assert result.failures[0].path == "line_items.0" and "line item dropped" in result.flags[0].message


def test_tax_scheme_comes_from_the_printed_label():
    wire = blank() | {"seller": {**blank()["seller"], "tax_id": cite("123456789", 1)}}
    result = build(wire, doc("USt-IdNr. 123456789"))
    assert result.invoice.seller.tax_id.value.scheme is TaxScheme.EU_VAT

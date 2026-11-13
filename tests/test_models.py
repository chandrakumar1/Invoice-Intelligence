"""
What:     Proves the Pydantic contract enforces provenance, money precision, and status invariants.
In:       Hand-built model inputs, plus every JSON file in tests/fixtures/expected/.
Out:      pytest pass/fail.
Fails if: Someone loosens a validator (e.g. allows a value without a SourceRef). That is exactly what these tests catch.
"""

import json
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.models import (
    Adjustment,
    AdjustmentKind,
    ExactDecimal,
    ExpectedInvoice,
    Extracted,
    ExtractionResult,
    Invoice,
    JobStatus,
    LineItem,
    LLMInvoiceExtraction,
    MatchType,
    PageInfo,
    Party,
    SourceRef,
    TaxId,
    TaxScheme,
    WireField,
)

EXPECTED_DIR = Path(__file__).parent / "fixtures" / "expected"


def ref(quote: str = "Invoice No. SOS/2025-26/0147", span: tuple[int, int] | None = (12, 28), **kw) -> SourceRef:
    fields = dict(
        page=1, line_start=3, line_end=3, quote=quote, value_span=span,
        match_type=MatchType.EXACT, text_source="text_layer", confidence=1.0,
    )
    return SourceRef(**(fields | kw))


def missing(reason: str = "not printed") -> Extracted:
    return Extracted.missing(reason)


def minimal_invoice() -> Invoice:
    party = Party(name=missing(), address=missing(), tax_id=missing())
    return Invoice(
        invoice_number=Extracted[str](value="SOS/2025-26/0147", source=ref()),
        invoice_date=missing(), due_date=missing(), currency=missing(),
        seller=party, buyer=party, line_items=[], subtotal=missing(),
        taxable_value=missing("no taxable value printed"), tax_lines=[],
        tax_total=missing(), adjustments=[], grand_total=missing(),
    )


# --- Extracted: no value without a citation, no None without a reason ------------------


def test_value_requires_source():
    with pytest.raises(ValidationError, match="must carry a SourceRef"):
        Extracted[str](value="INV-1")


def test_none_requires_reason():
    with pytest.raises(ValidationError, match="must have a missing_reason"):
        Extracted[str](value=None)


def test_value_and_reason_are_exclusive():
    with pytest.raises(ValidationError, match="cannot have a missing_reason"):
        Extracted[str](value="INV-1", source=ref(), missing_reason="oops")


def test_rejected_candidate_keeps_its_source():
    # Found on the page but unparseable: the reviewer still sees what the document says.
    e = Extracted[date].missing("unparseable date '31/02/2025'", source=ref("Date: 31/02/2025", (6, 16)))
    assert e.value is None and e.source is not None


def test_extracted_types_its_value():
    with pytest.raises(ValidationError):
        Extracted[date](value="not a date", source=ref())


# --- SourceRef bounds ------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"page": 0},
        {"confidence": 1.01},
        {"confidence": -0.1},
        {"line_start": 5, "line_end": 4},
        {"span": (10, 99)},  # past the end of the quote
        {"span": (5, 5)},  # empty
        {"quote": ""},
    ],
)
def test_sourceref_rejects_out_of_range(overrides):
    with pytest.raises(ValidationError):
        ref(**overrides)


def test_sourceref_is_immutable():
    r = ref()
    with pytest.raises(ValidationError):
        r.confidence = 0.1


# --- Money precision -------------------------------------------------------------------


def test_floats_are_rejected_for_money():
    with pytest.raises(ValidationError, match="not float"):
        Extracted[ExactDecimal](value=0.1, source=ref())


def test_invoice_money_fields_reject_floats():
    data = minimal_invoice().model_dump()
    data["grand_total"] = {"value": 0.1, "source": ref().model_dump()}
    with pytest.raises(ValidationError, match="not float"):
        Invoice.model_validate(data)


def test_decimal_serializes_as_string():
    e = Extracted[ExactDecimal](value=Decimal("123456.10"), source=ref())
    assert json.loads(e.model_dump_json())["value"] == "123456.10"


# --- Line items: one row citation, every cell located or explained ---------------------


ROW = "3 Stapler Heavy Duty HD-45 8472 5 1,250.00 6,250.00"


def row_item(**kw) -> LineItem:
    fields = dict(
        description="Stapler Heavy Duty HD-45", hsn_sac="8472", quantity=Decimal("5"),
        unit_price=Decimal("1250.00"), amount=Decimal("6250.00"),
        source=ref(ROW, span=None),
        cell_spans={"description": (2, 26), "hsn_sac": (27, 31), "quantity": (32, 33),
                    "unit_price": (34, 42), "amount": (43, 51)},
        missing_reasons={},
    )
    return LineItem(**(fields | kw))


def test_line_item_valid():
    li = row_item()
    start, end = li.cell_spans["unit_price"]
    assert li.source.quote[start:end] == "1,250.00"


def test_line_item_cell_without_span_rejected():
    spans = row_item().cell_spans | {}
    del spans["amount"]
    with pytest.raises(ValidationError, match="no span"):
        row_item(cell_spans=spans)


def test_line_item_missing_cell_needs_reason():
    spans = {k: v for k, v in row_item().cell_spans.items() if k != "hsn_sac"}
    with pytest.raises(ValidationError, match="needs a missing_reason"):
        row_item(hsn_sac=None, cell_spans=spans)
    assert row_item(hsn_sac=None, cell_spans=spans, missing_reasons={"hsn_sac": "column not printed"})


# --- taxable_value and adjustment labels -----------------------------------------------


def test_taxable_value_is_required_in_output_but_may_be_missing_with_reason():
    data = minimal_invoice().model_dump()
    del data["taxable_value"]
    with pytest.raises(ValidationError, match="taxable_value"):
        Invoice.model_validate(data)
    assert minimal_invoice().taxable_value.missing_reason == "no taxable value printed"


def test_taxable_value_null_without_reason_rejected():
    data = minimal_invoice().model_dump() | {"taxable_value": {"value": None}}
    with pytest.raises(ValidationError, match="must have a missing_reason"):
        Invoice.model_validate(data)


def test_taxable_value_needs_provenance_and_serializes_as_string():
    with pytest.raises(ValidationError, match="must carry a SourceRef"):
        Invoice.model_validate(minimal_invoice().model_dump() | {"taxable_value": {"value": "563888.15"}})
    inv = minimal_invoice().model_copy(
        update={"taxable_value": Extracted[ExactDecimal](
            value=Decimal("563888.15"), source=ref("Taxable Value 5,63,888.15", (14, 25)))}
    )
    dumped = json.loads(inv.model_dump_json())["taxable_value"]
    assert dumped["value"] == "563888.15"
    assert dumped["source"]["quote"] == "Taxable Value 5,63,888.15"


def test_adjustment_label_keeps_printed_text_or_explains_absence():
    amount = Extracted[ExactDecimal](value=Decimal("-0.01"), source=ref("Round Off -0.01", (10, 15)))
    a = Adjustment(kind=AdjustmentKind.ROUND_OFF, amount=amount,
                   label=Extracted[str](value="Round Off", source=ref("Round Off -0.01", (0, 9))))
    assert a.label.value == "Round Off"
    assert Adjustment(kind=AdjustmentKind.OTHER, amount=amount, label=missing("no label printed")).label.value is None
    with pytest.raises(ValidationError, match="must have a missing_reason"):
        Adjustment(kind=AdjustmentKind.OTHER, amount=amount, label=Extracted[str](value=None))


def expected_template() -> dict:
    return json.loads((EXPECTED_DIR / "_TEMPLATE.json").read_text())


def test_expected_taxable_value_optional():
    data = expected_template()
    assert ExpectedInvoice.model_validate(data).taxable_value is None  # explicit null
    del data["taxable_value"]
    assert ExpectedInvoice.model_validate(data).taxable_value is None  # omitted = null
    data["taxable_value"] = "563888.15"
    assert ExpectedInvoice.model_validate(data).taxable_value == Decimal("563888.15")
    data["taxable_value"] = 563888.15
    with pytest.raises(ValidationError, match="not float"):
        ExpectedInvoice.model_validate(data)


@pytest.mark.parametrize(
    "entry,label",
    [
        ({"kind": "round_off", "amount": "-0.01"}, None),  # label omitted
        ({"kind": "round_off", "amount": "-0.01", "label": None}, None),  # explicit null
        ({"kind": "discount", "amount": "-1500.00", "label": "Less: Trade Discount"}, "Less: Trade Discount"),
    ],
)
def test_expected_adjustment_label_optional(entry, label):
    data = expected_template() | {"adjustments": [entry]}
    assert ExpectedInvoice.model_validate(data).adjustments[0].label == label


# --- Wire schema (LLM output) ----------------------------------------------------------


def test_wire_field_requires_location_or_reason():
    with pytest.raises(ValidationError, match="needs page and line_start"):
        WireField(value_text="INV-1")
    with pytest.raises(ValidationError, match="not_found_reason"):
        WireField(value_text=None)
    with pytest.raises(ValidationError, match="must not be blank"):
        WireField(value_text="  ", page=1, line_start=1)


def test_wire_field_defaults_line_end():
    assert WireField(value_text="INV-1", page=1, line_start=7).line_end == 7


def test_wire_schema_is_tool_ready():
    schema = LLMInvoiceExtraction.model_json_schema()
    assert schema["additionalProperties"] is False
    assert {"invoice_number", "line_items", "taxable_value", "grand_total"} <= set(schema["required"])
    json.dumps(schema)  # serializable, so it can be sent as a tool input_schema


def test_wire_schema_rejects_extra_keys():
    # e.g. an LLM inventing a "confidence" field must not slip through.
    with pytest.raises(ValidationError, match="Extra inputs"):
        WireField(value_text="INV-1", page=1, line_start=1, confidence=0.99)


# --- PageInfo and ExtractionResult invariants ------------------------------------------


def test_ocr_page_needs_ocr_metadata():
    with pytest.raises(ValidationError):
        PageInfo(page=1, text_source="ocr", ocr_reasons=[], char_count=10, text="x")
    with pytest.raises(ValidationError):
        PageInfo(page=1, text_source="text_layer", ocr_reasons=["too little text"], char_count=10, text="x")


def result(**kw) -> ExtractionResult:
    fields = dict(
        id="abc", status=JobStatus.PENDING, filename="a.pdf", sha256="0" * 64,
        created_at=datetime(2025, 11, 11, tzinfo=UTC), pipeline_version="0.1.0",
    )
    return ExtractionResult(**(fields | kw))


def test_result_status_invariants():
    with pytest.raises(ValidationError, match="needs an invoice"):
        result(status=JobStatus.DONE)
    with pytest.raises(ValidationError, match="needs an error"):
        result(status=JobStatus.FAILED)
    with pytest.raises(ValidationError, match="cannot have an invoice"):
        result(invoice=minimal_invoice())


def test_result_json_round_trip():
    seller = Party(
        name=missing(), address=missing(),
        tax_id=Extracted[TaxId](value=TaxId(raw="29AAGCS4821K1Z0", scheme=TaxScheme.GSTIN),
                                source=ref("GSTIN: 29AAGCS4821K1Z0", (7, 22))),
    )
    inv = minimal_invoice().model_copy(update={"seller": seller})
    r = result(status=JobStatus.DONE, invoice=inv,
               pages=[PageInfo(page=1, text_source="text_layer", ocr_reasons=[], char_count=3, text="abc")])
    assert ExtractionResult.model_validate_json(r.model_dump_json()) == r


# --- Ground-truth files ----------------------------------------------------------------


@pytest.mark.parametrize("path", sorted(EXPECTED_DIR.glob("*.json")), ids=lambda p: p.name)
def test_expected_files_match_contract(path):
    ExpectedInvoice.model_validate_json(path.read_text())


def expected_with_pages(pages: dict) -> dict:
    # Template has exactly one line item and one tax line, so index 0 is valid and 1 is not.
    return expected_template() | {"expected_pages": pages}


@pytest.mark.parametrize(
    "key",
    [
        "invoice_number", "currency", "grand_total", "taxable_value",
        "seller.name", "buyer.tax_id",
        "line_items.0", "line_items.0.amount", "tax_lines.0",
        "seller", "line_items",  # whole blocks are fine too
    ],
)
def test_expected_pages_accepts_real_fields(key):
    assert ExpectedInvoice.model_validate(expected_with_pages({key: 1})).expected_pages == {key: 1}


@pytest.mark.parametrize(
    "key",
    [
        "seler.name",  # misspelled parent
        "invoice_no",  # misspelled field
        "seller.gstin",  # field doesn't exist on the party
        "line_items.1",  # past the end (one item)
        "adjustments.0",  # template has no adjustments
        "line_items.first",  # non-numeric index
        "line_items.00",  # non-canonical index
        "line_items.-1",  # negative index
        "seller.name.first",  # descends into a string
        "seller..name",  # empty segment
        "",  # empty key
        "notes",  # file metadata, not on any page
        "fixture",
    ],
)
def test_expected_pages_rejects_invalid_keys(key):
    with pytest.raises(ValidationError, match="don't name a field or array position"):
        ExpectedInvoice.model_validate(expected_with_pages({key: 1}))


def test_expected_pages_error_lists_every_bad_key():
    with pytest.raises(ValidationError) as err:
        ExpectedInvoice.model_validate(expected_with_pages({"invoice_number": 1, "seler.name": 1, "line_items.9": 2}))
    assert "seler.name" in str(err.value) and "line_items.9" in str(err.value)
    assert "'invoice_number'" not in str(err.value)


@pytest.mark.parametrize("page", [0, -1])
def test_expected_pages_must_be_one_indexed(page):
    with pytest.raises(ValidationError):
        ExpectedInvoice.model_validate(expected_with_pages({"invoice_number": page}))


def test_expected_rejects_float_amounts():
    data = json.loads((EXPECTED_DIR / "_TEMPLATE.json").read_text())
    data["grand_total"] = 1234.5
    with pytest.raises(ValidationError, match="not float"):
        ExpectedInvoice.model_validate(data)

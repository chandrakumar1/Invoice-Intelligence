"""
What:     The Pydantic contract for the whole pipeline: LLM wire format, typed Invoice, flags, result envelope, ground-truth format.
In:       Dicts / JSON from the LLM tool call, from storage, and from hand-written expected files.
Out:      Validated model instances; `.model_dump_json()` for storage and the API (Decimals serialize as strings).
Fails if: A producer emits a value without a citation, or None without a reason. The validators reject this on purpose.

Two layers:
  * Wire schema (`Wire*`, `LLMInvoiceExtraction`): what the LLM returns. Strings and
    locations only. The LLM transcribes; it never types, parses, or computes.
  * Domain schema (`Extracted`, `Invoice`, ...): typed values that Python builds after verifying
    each citation against the page text and parsing the transcribed string.
"""

from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    model_validator,
)


def _reject_float(v: object) -> object:
    # A float has already lost precision (0.1 + 0.2 != 0.3); make the caller pass str/int/Decimal.
    if isinstance(v, float):
        raise ValueError("pass decimals as strings or Decimal, not float")
    return v


ExactDecimal = Annotated[Decimal, BeforeValidator(_reject_float)]
Span = tuple[int, int]  # [start, end) character offsets
TextSource = Literal["text_layer", "ocr"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------------------
# Wire schema: what the LLM returns
# --------------------------------------------------------------------------------------


class WireField(_Strict):
    """One field as the LLM reports it: where it is, and the exact characters printed there."""

    value_text: str | None = Field(
        description="Characters copied exactly from the cited line(s), including typos and OCR "
        "errors. Null if the field is not printed on the document."
    )
    page: int | None = Field(default=None, ge=1, description="1-indexed page of the value.")
    line_start: int | None = Field(default=None, ge=1, description="First line containing the value.")
    line_end: int | None = Field(
        default=None, ge=1, description="Last line containing the value; same as line_start if one line."
    )
    not_found_reason: str | None = Field(
        default=None, description="Required when value_text is null: why the field was not found."
    )

    @model_validator(mode="after")
    def _location_or_reason(self) -> Self:
        if self.value_text is not None:
            if not self.value_text.strip():
                raise ValueError("value_text must not be blank; use null with a not_found_reason")
            if self.page is None or self.line_start is None:
                raise ValueError("a value_text needs page and line_start")
            if self.line_end is None:
                self.line_end = self.line_start
            if self.line_end < self.line_start:
                raise ValueError("line_end must be >= line_start")
        elif not self.not_found_reason:
            raise ValueError("a null value_text needs a not_found_reason")
        return self


class WireLineItem(_Strict):
    """One table row. Cells are exact substrings of the cited row; null means the cell is not printed."""

    page: int = Field(ge=1)
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    description: str | None
    hsn_sac: str | None = Field(description="HSN/SAC or item code as printed, if the column exists.")
    quantity: str | None
    unit_price: str | None
    amount: str | None

    @model_validator(mode="after")
    def _ordered_lines(self) -> Self:
        if self.line_end < self.line_start:
            raise ValueError("line_end must be >= line_start")
        return self


class WireParty(_Strict):
    name: WireField
    address: WireField
    tax_id: WireField = Field(description="The tax identifier only (e.g. GSTIN, VAT ID), without its label.")


class WireTaxLine(_Strict):
    label: WireField = Field(description="The tax line's label as printed, e.g. 'CGST @ 9%'.")
    rate: WireField = Field(description="The rate as printed, e.g. '9%'. Null if not printed.")
    amount: WireField


class WireAdjustment(_Strict):
    """A printed line between subtotal and grand total that isn't tax: discount, shipping, round off."""

    label: WireField
    amount: WireField = Field(description="The amount exactly as printed, including any sign.")


class LLMInvoiceExtraction(_Strict):
    """The complete tool-call payload. Its JSON schema is the tool's input_schema."""

    invoice_number: WireField
    invoice_date: WireField
    due_date: WireField
    currency: WireField = Field(description="Currency code or symbol as printed, e.g. 'INR', 'Rs.', '€'.")
    seller: WireParty
    buyer: WireParty
    line_items: list[WireLineItem]
    subtotal: WireField = Field(description="Only if printed. Never computed from the line items.")
    taxable_value: WireField = Field(
        description="The amount tax is charged on, only if printed with its own label (e.g. 'Taxable Value'). "
        "Never computed, e.g. as subtotal minus discount."
    )
    tax_lines: list[WireTaxLine]
    tax_total: WireField = Field(description="Only if printed. Never computed from the tax lines.")
    adjustments: list[WireAdjustment]
    grand_total: WireField


# --------------------------------------------------------------------------------------
# Domain schema: typed and verified
# --------------------------------------------------------------------------------------


class MatchType(StrEnum):
    """How the cited value was found in the canonical page text. Set by Python, never by the LLM."""

    EXACT = "exact"  # verbatim substring of the cited line(s)
    NORMALIZED = "normalized"  # matched after whitespace/Unicode/symbol normalization
    RELOCATED = "relocated"  # found, but not where the LLM cited; citation corrected
    FUZZY = "fuzzy"  # near-match (OCR noise); lowest trust


class SourceRef(_Strict):
    """Where a value came from. `quote` is sliced from the page text by Python, so it is verbatim by construction."""

    model_config = ConfigDict(frozen=True)

    page: int = Field(ge=1)
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    quote: str = Field(min_length=1)
    value_span: Span | None = Field(
        default=None, description="Offsets of the value inside `quote`. None for row-level refs."
    )
    match_type: MatchType
    text_source: TextSource
    confidence: float = Field(ge=0.0, le=1.0, description="Deterministic heuristic score, not a probability.")

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.line_end < self.line_start:
            raise ValueError("line_end must be >= line_start")
        if self.value_span is not None:
            _check_span(self.value_span, self.quote)
        return self


def _check_span(span: Span, quote: str) -> None:
    start, end = span
    if not 0 <= start < end <= len(quote):
        raise ValueError(f"span {span} is outside the quote (length {len(quote)})")


class Extracted[T](_Strict):
    """A value plus its provenance, or None plus the reason it is missing. Used for every scalar field.

    `source` may be present with `value=None`: the text was found but rejected (e.g. unparseable
    date). Keeping it lets a reviewer see what the document actually says.
    """

    value: T | None = None
    source: SourceRef | None = None
    missing_reason: str | None = None

    @model_validator(mode="after")
    def _provenance_or_reason(self) -> Self:
        if self.value is not None:
            if self.source is None:
                raise ValueError("an extracted value must carry a SourceRef")
            if self.missing_reason is not None:
                raise ValueError("a present value cannot have a missing_reason")
        elif not self.missing_reason:
            raise ValueError("a missing value must have a missing_reason")
        return self

    @classmethod
    def missing(cls, reason: str, source: SourceRef | None = None) -> Self:
        return cls(value=None, source=source, missing_reason=reason)


class TaxScheme(StrEnum):
    """Decided by Python from the printed label and the ID's format."""

    GSTIN = "gstin"
    EU_VAT = "eu_vat"
    OTHER = "other"
    UNKNOWN = "unknown"


class TaxId(_Strict):
    raw: str = Field(min_length=1)
    scheme: TaxScheme


class Party(_Strict):
    name: Extracted[str]
    address: Extracted[str]
    tax_id: Extracted[TaxId]


LineItemCell = Literal["description", "hsn_sac", "quantity", "unit_price", "amount"]
LINE_ITEM_CELLS: tuple[LineItemCell, ...] = ("description", "hsn_sac", "quantity", "unit_price", "amount")


class LineItem(_Strict):
    """One row, cited once as a whole. Each present cell's span inside the row quote is recorded and checked."""

    description: str | None
    hsn_sac: str | None
    quantity: ExactDecimal | None
    unit_price: ExactDecimal | None
    amount: ExactDecimal | None
    source: SourceRef
    cell_spans: dict[LineItemCell, Span]
    missing_reasons: dict[LineItemCell, str]

    @model_validator(mode="after")
    def _every_cell_cited_or_explained(self) -> Self:
        for cell in LINE_ITEM_CELLS:
            present = getattr(self, cell) is not None
            if present:
                if cell not in self.cell_spans:
                    raise ValueError(f"cell '{cell}' has a value but no span in the row quote")
                if cell in self.missing_reasons:
                    raise ValueError(f"cell '{cell}' has a value and a missing_reason")
                _check_span(self.cell_spans[cell], self.source.quote)
            else:
                if cell in self.cell_spans:
                    raise ValueError(f"cell '{cell}' is None but has a span")
                if not self.missing_reasons.get(cell):
                    raise ValueError(f"cell '{cell}' is None and needs a missing_reason")
        return self


class TaxLine(_Strict):
    label: Extracted[str]
    rate: Extracted[ExactDecimal] = Field(description="Percent, e.g. Decimal('9') for 9%.")
    amount: Extracted[ExactDecimal]


class AdjustmentKind(StrEnum):
    """Classified by Python from the printed label (keyword rules), not by the LLM."""

    DISCOUNT = "discount"
    SHIPPING = "shipping"
    ROUND_OFF = "round_off"
    OTHER = "other"


class Adjustment(_Strict):
    kind: AdjustmentKind
    label: Extracted[str] = Field(description="The label exactly as printed, e.g. 'Less: Trade Discount'.")
    amount: Extracted[ExactDecimal] = Field(
        description="Signed effect on the grand total (a discount is negative). Sign applied by Python."
    )


class Invoice(_Strict):
    invoice_number: Extracted[str]
    invoice_date: Extracted[date]
    due_date: Extracted[date]
    currency: Extracted[str] = Field(description="ISO 4217 code, mapped by Python from the printed text.")
    seller: Party
    buyer: Party
    line_items: list[LineItem]
    subtotal: Extracted[ExactDecimal]
    taxable_value: Extracted[ExactDecimal] = Field(
        description="Printed tax base. When not printed: value None with a missing_reason, never derived."
    )
    tax_lines: list[TaxLine]
    tax_total: Extracted[ExactDecimal]
    adjustments: list[Adjustment]
    grand_total: Extracted[ExactDecimal]


# --------------------------------------------------------------------------------------
# Validation output
# --------------------------------------------------------------------------------------


class Severity(StrEnum):
    ERROR = "error"
    WARNING = "warning"


class FlagCode(StrEnum):
    REQUIRED_FIELD_MISSING = "required_field_missing"
    LINE_AMOUNT_MISMATCH = "line_amount_mismatch"  # qty x unit_price != amount
    LINE_ITEMS_SUM_MISMATCH = "line_items_sum_mismatch"  # sum(line amounts) != subtotal
    TAX_LINES_SUM_MISMATCH = "tax_lines_sum_mismatch"  # sum(tax lines) != tax_total
    TAX_AMOUNT_MISMATCH = "tax_amount_mismatch"  # taxable_value x rate != printed tax amount
    TOTALS_MISMATCH = "totals_mismatch"  # subtotal + tax + adjustments != grand_total
    DATE_UNPARSEABLE = "date_unparseable"
    DATE_AMBIGUOUS = "date_ambiguous"  # e.g. 03/04/2025 with no locale signal
    DATE_IN_FUTURE = "date_in_future"
    DUE_BEFORE_INVOICE_DATE = "due_before_invoice_date"
    TAX_ID_INVALID_FORMAT = "tax_id_invalid_format"
    TAX_ID_INVALID_CHECKSUM = "tax_id_invalid_checksum"
    GST_TYPE_STATE_MISMATCH = "gst_type_state_mismatch"  # CGST+SGST vs IGST disagrees with state codes
    CITATION_UNVERIFIED = "citation_unverified"  # LLM cited text that isn't on the page
    VALUE_UNPARSEABLE = "value_unparseable"


class ValidationFlag(_Strict):
    code: FlagCode
    severity: Severity
    field_paths: list[str] = Field(min_length=1, description="Dotted paths, e.g. 'line_items.3.amount'.")
    message: str = Field(min_length=1, description="Human-readable reason, shown in the review UI.")
    expected: str | None = None
    actual: str | None = None


# --------------------------------------------------------------------------------------
# Result envelope (what storage holds and GET /result/{id} returns)
# --------------------------------------------------------------------------------------


class PageInfo(_Strict):
    page: int = Field(ge=1)
    text_source: TextSource
    ocr_reasons: list[str] = Field(description="Which heuristics sent this page to OCR; empty for text_layer.")
    char_count: int = Field(ge=0)
    ocr_mean_conf: float | None = Field(default=None, ge=0.0, le=100.0)
    ocr_line_conf: list[Annotated[float, Field(ge=0.0, le=100.0)]] | None = Field(
        default=None, description="OCR pages only: mean Tesseract confidence of each line, aligned with `lines`."
    )
    text: str = Field(
        description="Canonical page text that every quote on this page is verified against. "
        "Lines are separated by '\\n'; line k (1-indexed) is lines[k-1]."
    )

    @property
    def lines(self) -> list[str]:
        return self.text.split("\n") if self.text else []

    @model_validator(mode="after")
    def _ocr_fields_match_source(self) -> Self:
        if self.text_source == "ocr":
            if self.ocr_mean_conf is None or self.ocr_line_conf is None or not self.ocr_reasons:
                raise ValueError("OCR pages need ocr_mean_conf, ocr_line_conf and at least one ocr_reason")
            if len(self.ocr_line_conf) != len(self.lines):
                raise ValueError(f"ocr_line_conf has {len(self.ocr_line_conf)} entries for {len(self.lines)} lines")
        elif self.ocr_mean_conf is not None or self.ocr_line_conf is not None or self.ocr_reasons:
            raise ValueError("text_layer pages cannot have OCR fields")
        return self


class JobStatus(StrEnum):
    PENDING = "pending"
    DONE = "done"
    FAILED = "failed"


class ExtractionResult(_Strict):
    id: str = Field(min_length=1)
    status: JobStatus
    error: str | None = None
    filename: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_at: datetime
    provider: str | None = None  # which LLM provider answered: "gemini", "anthropic", ...
    model: str | None = None
    prompt_version: str | None = None
    pipeline_version: str
    pages: list[PageInfo] = []
    invoice: Invoice | None = None
    flags: list[ValidationFlag] = []

    @model_validator(mode="after")
    def _status_consistent(self) -> Self:
        if self.status is JobStatus.DONE and self.invoice is None:
            raise ValueError("a done result needs an invoice")
        if self.status is JobStatus.FAILED and not self.error:
            raise ValueError("a failed result needs an error message")
        if self.status is JobStatus.PENDING and self.invoice is not None:
            raise ValueError("a pending result cannot have an invoice yet")
        return self


class ExtractAccepted(_Strict):
    """POST /extract response: the job was accepted and will be processed in the background."""

    id: str
    status: JobStatus
    result_url: str = Field(description="Poll this until status is done or failed.")


# --------------------------------------------------------------------------------------
# Ground-truth format (hand-written files in tests/fixtures/expected/)
# --------------------------------------------------------------------------------------
# Every key is required, so the author decides on each field explicitly. `null` means
# "not printed on this document". Decimals must be JSON strings ("1234.50"), dates ISO.
# Exception: fields added after files were already written (`taxable_value`,
# `adjustments[].label`) are optional. Omitting them means the same as null.


class ExpectedParty(_Strict):
    name: str | None
    address: str | None
    tax_id: str | None
    tax_scheme: TaxScheme | None


class ExpectedLineItem(_Strict):
    description: str | None
    hsn_sac: str | None
    quantity: ExactDecimal | None
    unit_price: ExactDecimal | None
    amount: ExactDecimal | None


class ExpectedTaxLine(_Strict):
    label: str | None
    rate: ExactDecimal | None
    amount: ExactDecimal | None


class ExpectedAdjustment(_Strict):
    kind: AdjustmentKind
    amount: ExactDecimal  # signed, as in Adjustment
    label: str | None = None  # exactly as printed, e.g. "Round Off"


class ExpectedInvoice(_Strict):
    fixture: str = Field(min_length=1, description="PDF filename in tests/fixtures/pdfs/ or private/.")
    invoice_number: str | None
    invoice_date: date | None
    due_date: date | None
    currency: str | None
    seller: ExpectedParty
    buyer: ExpectedParty
    line_items: list[ExpectedLineItem]
    subtotal: ExactDecimal | None
    taxable_value: ExactDecimal | None = None
    tax_lines: list[ExpectedTaxLine]
    tax_total: ExactDecimal | None
    adjustments: list[ExpectedAdjustment]
    grand_total: ExactDecimal | None
    expected_pages: dict[str, Annotated[int, Field(ge=1)]] = Field(
        description="Field path -> page it appears on, e.g. {'grand_total': 2, 'line_items.0': 1}."
    )
    expected_flags: list[FlagCode] = Field(description="Flags validation should raise for this document.")
    notes: str | None = Field(description="Free text for the author: quirks, judgment calls.")

    @model_validator(mode="after")
    def _expected_pages_name_real_fields(self) -> Self:
        bad = [path for path in self.expected_pages if not self._resolves(path)]
        if bad:
            raise ValueError(f"expected_pages keys that don't name a field or array position in this file: {bad}")
        return self

    def _resolves(self, path: str) -> bool:
        """True if `path` (e.g. 'seller.tax_id', 'line_items.29') walks to something in this file.

        Field names must exist on the model at each level. Array indexes must be canonical
        (no leading zeros) and within the list. Metadata keys have no page, so they never resolve.
        """
        parts = path.split(".")
        if parts[0] in _NOT_ON_PAGE:
            return False
        node: object = self
        for part in parts:
            if isinstance(node, list):
                if not part.isdigit() or str(int(part)) != part or int(part) >= len(node):
                    return False
                node = node[int(part)]
            elif isinstance(node, BaseModel):
                if part not in type(node).model_fields:
                    return False
                node = getattr(node, part)
            else:  # a leaf value; nothing to descend into
                return False
        return True


# Fields of ExpectedInvoice that describe the file, not the document, so they have no page.
_NOT_ON_PAGE = frozenset({"fixture", "expected_pages", "expected_flags", "notes"})

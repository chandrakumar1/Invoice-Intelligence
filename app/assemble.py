"""
What:     Builds the typed Invoice from the LLM's wire output: verifies every citation, parses every value from the page text.
In:       A validated LLMInvoiceExtraction and the PageInfo list the LLM saw.
Out:      Assembly: the Invoice, the flags raised while building it, and the citation failures to send back in the repair round.
Fails if: A field is added to the wire schema but not mapped here, so it's silently missing. The end-to-end tests
          against the hand-written fixtures catch that.

Order of work: locate every citation first, then derive document-level hints (decimal separator, day/month
order) from the located text, then parse. Hints come from the page, never from the LLM.

Paths in citation flags and failures use the LLM's row numbering (`line_items.3` is the LLM's 4th row),
because a row whose cells can't be found at all is dropped from the Invoice.
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

from app.models import (
    LINE_ITEM_CELLS,
    Adjustment,
    AdjustmentKind,
    ExactDecimal,
    Extracted,
    FlagCode,
    Invoice,
    LineItem,
    LLMInvoiceExtraction,
    PageInfo,
    Party,
    Severity,
    SourceRef,
    TaxId,
    TaxLine,
    ValidationFlag,
    WireField,
    WireLineItem,
    WireParty,
)
from app.parse import (
    AmbiguousDate,
    Parsed,
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
from app.provenance import Located, RowLocation, confidence, locate, locate_row, source_ref


@dataclass(frozen=True)
class CitationFailure:
    """A value the LLM reported that isn't where it said (or anywhere). Sent back in the repair round."""

    path: str
    value_text: str
    page: int | None
    line_start: int | None
    line_end: int | None

    def describe(self) -> str:
        where = f"page {self.page}, lines {self.line_start}-{self.line_end}" if self.page else "the cited location"
        return f"{self.path}: {self.value_text!r} does not appear on {where}"


@dataclass
class Assembly:
    invoice: Invoice
    flags: list[ValidationFlag] = field(default_factory=list)
    failures: list[CitationFailure] = field(default_factory=list)


def build_invoice(wire: LLMInvoiceExtraction, pages: list[PageInfo]) -> Assembly:
    return _Builder(wire, pages).build()


_DATE_FIELDS = ("invoice_date", "due_date")
_TOTAL_FIELDS = ("subtotal", "taxable_value", "tax_total", "grand_total")


def _scalar_fields(wire: LLMInvoiceExtraction) -> Iterator[tuple[str, WireField]]:
    for name in ("invoice_number", *_DATE_FIELDS, "currency", *_TOTAL_FIELDS):
        yield name, getattr(wire, name)
    for party in ("seller", "buyer"):
        p: WireParty = getattr(wire, party)
        yield f"{party}.name", p.name
        yield f"{party}.address", p.address
        yield f"{party}.tax_id", p.tax_id
    for i, t in enumerate(wire.tax_lines):
        yield f"tax_lines.{i}.label", t.label
        yield f"tax_lines.{i}.rate", t.rate
        yield f"tax_lines.{i}.amount", t.amount
    for i, a in enumerate(wire.adjustments):
        yield f"adjustments.{i}.label", a.label
        yield f"adjustments.{i}.amount", a.amount


class _Builder:
    def __init__(self, wire: LLMInvoiceExtraction, pages: list[PageInfo]):
        self.wire = wire
        self.pages = {p.page: p for p in pages}
        self.flags: list[ValidationFlag] = []
        self.failures: list[CitationFailure] = []

        # Pass 1: locate everything.
        self.cited: dict[str, tuple[WireField, Located | None]] = {
            path: (wf, self._locate(path, wf, pages)) for path, wf in _scalar_fields(wire)
        }
        self.rows = [self._locate_row(i, row) for i, row in enumerate(wire.line_items)]

        # Pass 2: document-level hints from the located text.
        amount_texts = [
            loc.text for path, (_, loc) in self.cited.items()
            if loc and (path in _TOTAL_FIELDS or path.endswith(".amount"))
        ]
        amount_texts += [
            m.text for loc in self.rows if loc for cell, m in loc.cells.items() if cell in ("unit_price", "amount")
        ]
        self.decimal_sep = infer_decimal_separator(amount_texts)
        currency_loc = self.cited["currency"][1]
        currency = None
        if currency_loc:
            try:
                currency = parse_currency(currency_loc.text).value
            except ParseError:
                pass
        schemes = [detect_tax_scheme(loc.text, loc.quote) for p in ("seller", "buyer") if (loc := self.cited[f"{p}.tax_id"][1])]
        date_texts = [loc.text for name in _DATE_FIELDS if (loc := self.cited[name][1])]
        self.date_order = infer_date_order(date_texts, currency, schemes)

    # --- locating --------------------------------------------------------------------------

    def _locate(self, path: str, wf: WireField, pages: list[PageInfo]) -> Located | None:
        if wf.value_text is None:
            return None
        loc = locate(wf.value_text, wf.page, wf.line_start, wf.line_end, pages)
        if loc is None:
            self._citation_failed(path, wf.value_text, wf.page, wf.line_start, wf.line_end)
        return loc

    def _locate_row(self, i: int, row: WireLineItem) -> RowLocation | None:
        cells = {c: getattr(row, c) for c in LINE_ITEM_CELLS if getattr(row, c) is not None}
        page = self.pages.get(row.page)
        loc = locate_row(cells, page, row.line_start, row.line_end) if page and cells else None
        if loc is None:
            label = row.description or next(iter(cells.values()), "(empty row)")
            self._citation_failed(f"line_items.{i}", label, row.page, row.line_start, row.line_end, dropped="line item")
            return None
        for cell in cells.keys() - loc.cells.keys():
            self._citation_failed(f"line_items.{i}.{cell}", cells[cell], row.page, row.line_start, row.line_end)
        return loc

    def _citation_failed(self, path, value_text, page, line_start, line_end, dropped: str = "value") -> None:
        failure = CitationFailure(path, value_text, page, line_start, line_end)
        self.failures.append(failure)
        self.flags.append(
            ValidationFlag(
                code=FlagCode.CITATION_UNVERIFIED,
                severity=Severity.WARNING,
                field_paths=[path],
                message=f"{failure.describe()} or anywhere else checked in the document; {dropped} dropped.",
                actual=value_text,
            )
        )

    # --- converting --------------------------------------------------------------------------

    def _extracted[T](self, path: str, kind: type, convert: Callable[[Located], Parsed[T]]) -> Extracted:
        wf, loc = self.cited[path]
        model = Extracted[kind]
        if wf.value_text is None:
            return model.missing(f"not printed on the document ({wf.not_found_reason})")
        if loc is None:
            return model.missing(f"citation not verifiable: {wf.value_text!r} is not on the cited lines or elsewhere in the document")
        page = self.pages[loc.page]
        try:
            parsed = convert(loc)
        except AmbiguousDate as e:
            self._flag(FlagCode.DATE_AMBIGUOUS, path, str(e), loc.text)
            return model.missing(str(e), source=source_ref(loc, page))
        except ParseError as e:
            code = FlagCode.DATE_UNPARSEABLE if kind is date else FlagCode.VALUE_UNPARSEABLE
            self._flag(code, path, str(e), loc.text)
            return model.missing(str(e), source=source_ref(loc, page))
        return model(value=parsed.value, source=source_ref(loc, page, parsed.certainty))

    def _flag(self, code: FlagCode, path: str, message: str, actual: str) -> None:
        self.flags.append(ValidationFlag(code=code, severity=Severity.WARNING, field_paths=[path], message=message, actual=actual))

    def _text(self, path: str) -> Extracted:
        return self._extracted(path, str, lambda loc: Parsed(loc.text))

    def _amount(self, path: str) -> Extracted:
        return self._extracted(path, ExactDecimal, lambda loc: parse_amount(loc.text, self.decimal_sep))

    def _date(self, path: str) -> Extracted:
        return self._extracted(path, date, lambda loc: parse_date(loc.text, self.date_order))

    def _party(self, name: str) -> Party:
        return Party(
            name=self._text(f"{name}.name"),
            address=self._text(f"{name}.address"),
            tax_id=self._extracted(
                f"{name}.tax_id", TaxId, lambda loc: Parsed(TaxId(raw=loc.text, scheme=detect_tax_scheme(loc.text, loc.quote)))
            ),
        )

    def _line_item(self, i: int, row: WireLineItem, loc: RowLocation) -> LineItem:
        convert = {
            "description": lambda t: Parsed(t),
            "hsn_sac": lambda t: Parsed(t),
            "quantity": lambda t: parse_quantity(t, self.decimal_sep),
            "unit_price": lambda t: parse_amount(t, self.decimal_sep),
            "amount": lambda t: parse_amount(t, self.decimal_sep),
        }
        values: dict[str, object] = {}
        spans, reasons, certainties = {}, {}, []
        for cell in LINE_ITEM_CELLS:
            raw = getattr(row, cell)
            match = loc.cells.get(cell)
            if raw is None:
                reasons[cell] = "not printed in this row"
            elif match is None:
                reasons[cell] = f"citation not verifiable: {raw!r} is not on the cited row"
            else:
                try:
                    parsed = convert[cell](match.text)
                except ParseError as e:
                    reasons[cell] = str(e)
                    self._flag(FlagCode.VALUE_UNPARSEABLE, f"line_items.{i}.{cell}", str(e), match.text)
                    continue
                values[cell] = parsed.value
                spans[cell] = match.span
                certainties.append(parsed.certainty)
        page = self.pages[row.page]
        source = SourceRef(
            page=row.page,
            line_start=loc.line_start,
            line_end=loc.line_end,
            quote=loc.quote,
            match_type=loc.match_type,
            text_source=page.text_source,
            confidence=confidence(loc.match_type, page, loc.line_start, loc.line_end, min(certainties, default=1.0)),
        )
        return LineItem(
            **{c: values.get(c) for c in LINE_ITEM_CELLS}, source=source, cell_spans=spans, missing_reasons=reasons
        )

    def _adjustment(self, i: int) -> Adjustment:
        label = self._text(f"adjustments.{i}.label")
        kind = classify_adjustment(label.value)

        def signed(loc: Located) -> Parsed[Decimal]:
            printed = parse_amount(loc.text, self.decimal_sep)
            return Parsed(signed_adjustment(kind, printed.value), printed.certainty)

        return Adjustment(kind=kind, label=label, amount=self._extracted(f"adjustments.{i}.amount", ExactDecimal, signed))

    def build(self) -> Assembly:
        invoice = Invoice(
            invoice_number=self._text("invoice_number"),
            invoice_date=self._date("invoice_date"),
            due_date=self._date("due_date"),
            currency=self._extracted("currency", str, lambda loc: parse_currency(loc.text)),
            seller=self._party("seller"),
            buyer=self._party("buyer"),
            line_items=[self._line_item(i, row, loc) for i, (row, loc) in enumerate(zip(self.wire.line_items, self.rows)) if loc],
            subtotal=self._amount("subtotal"),
            taxable_value=self._amount("taxable_value"),
            tax_lines=[
                TaxLine(
                    label=self._text(f"tax_lines.{i}.label"),
                    rate=self._extracted(f"tax_lines.{i}.rate", ExactDecimal, lambda loc: parse_percent(loc.text, self.decimal_sep)),
                    amount=self._amount(f"tax_lines.{i}.amount"),
                )
                for i in range(len(self.wire.tax_lines))
            ],
            tax_total=self._amount("tax_total"),
            adjustments=[self._adjustment(i) for i in range(len(self.wire.adjustments))],
            grand_total=self._amount("grand_total"),
        )
        return Assembly(invoice, self.flags, self.failures)

"""
What:     Shared test helpers: build mock LLM answers that cite real line numbers, and compare an Invoice with hand-written ground truth.
In:       Fixture PDFs (text-layer ones), value texts copied from them, and ExpectedInvoice files.
Out:      Wire-schema dicts for a FakeClient to return; lists of human-readable mismatches.
Fails if: A fixture PDF is regenerated with different text, so a value can't be found and the builder raises. That's intentional.

The mock builder plays a perfect transcriber: it only finds where a value is printed, it never decides a value.
Values in the mocks are copied from the fixture text; the expected answers they're compared with are the owner's files.
"""

import json
import re
from functools import cache
from pathlib import Path

from app.extract import extract_pages
from app.models import ExpectedInvoice, Invoice, PageInfo
from scripts.evaluate import score_invoice

FIXTURES = Path(__file__).parent / "fixtures"


@cache
def pages_of(stem: str) -> tuple[PageInfo, ...]:
    return tuple(extract_pages((FIXTURES / "pdfs" / f"{stem}.pdf").read_bytes()))


def expected(stem: str) -> ExpectedInvoice:
    return ExpectedInvoice.model_validate_json((FIXTURES / "expected" / f"{stem}.json").read_text())


class Doc:
    """Finds where text is printed, to build citations the way a perfect transcriber would."""

    def __init__(self, stem: str):
        self.pages = pages_of(stem)

    def where(self, segment: str, page: int | None = None, after: tuple[int, int] = (0, 0)) -> tuple[int, int]:
        for p in self.pages:
            if page is not None and p.page != page:
                continue
            for n, line in enumerate(p.lines, 1):
                if (p.page, n) > after and segment in line:
                    return p.page, n
        raise LookupError(f"{segment!r} is not in the fixture text")

    def field(self, value_text: str, page: int | None = None) -> dict:
        segments = value_text.split("\n")
        first = last = self.where(segments[0], page)
        for seg in segments[1:]:
            last = self.where(seg, first[0], after=last)
        return {"value_text": value_text, "page": first[0], "line_start": first[1], "line_end": last[1], "not_found_reason": None}

    def beside(self, label: str, value_text: str) -> dict:
        """`value_text` cited on the line where `label` is printed (e.g. a tax rate or amount next to its label)."""
        page, n = self.where(label)
        line = self.pages[page - 1].lines[n - 1]
        assert value_text in line, f"{value_text!r} is not on the line of {label!r}: {line!r}"
        return {"value_text": value_text, "page": page, "line_start": n, "line_end": n, "not_found_reason": None}

    @staticmethod
    def missing(reason: str) -> dict:
        return {"value_text": None, "page": None, "line_start": None, "line_end": None, "not_found_reason": reason}

    def rows(self, pattern: str, cells: tuple[str, ...]) -> list[dict]:
        """One wire row per line matching `pattern`; its groups map to `cells` (None skips a group)."""
        rx = re.compile(pattern)
        out = []
        for p in self.pages:
            for n, line in enumerate(p.lines, 1):
                if m := rx.fullmatch(line):
                    row = {"page": p.page, "line_start": n, "line_end": n,
                           "description": None, "hsn_sac": None, "quantity": None, "unit_price": None, "amount": None}
                    row |= {c: v for c, v in zip(cells, m.groups()) if c}
                    out.append(row)
        return out


def party(doc: Doc, name: str, address: str, tax_id: str) -> dict:
    return {"name": doc.field(name), "address": doc.field(address), "tax_id": doc.field(tax_id)}


def tax_line(doc: Doc, label: str, rate: str, amount: str, amount_line_label: str | None = None) -> dict:
    """Rate and amount printed on the label's line, unless `amount_line_label` names another line (columnar layouts)."""
    return {
        "label": doc.field(label),
        "rate": doc.beside(label, rate),
        "amount": doc.beside(amount_line_label or label, amount),
    }


# --- Mock answers for the text-layer fixtures -----------------------------------------------

INDIAN_ROW = r"(\d+) (.+) (\d{4}) (\d+) ([\d,]+\.\d{2}) ([\d,]+\.\d{2})"
ROW_CELLS = (None, "description", "hsn_sac", "quantity", "unit_price", "amount")


def wire_01() -> dict:
    d = Doc("01_gst_classic")
    return {
        "invoice_number": d.field("SOS/2025-26/0147"),
        "invoice_date": d.field("05/11/2025"),
        "due_date": d.field("20/11/2025"),
        "currency": d.field("Rs."),
        "seller": party(d, "Sharma Office Supplies Pvt. Ltd.", "No. 42, 3rd Cross, Indiranagar\nBengaluru, Karnataka 560038", "29AAGCS4821K1Z0"),
        "buyer": party(d, "Kaveri Software Solutions LLP", "Plot 17, Electronic City Phase 1\nBengaluru, Karnataka 560100", "29AAKFK7730M1ZO"),
        "line_items": d.rows(INDIAN_ROW, ROW_CELLS),
        "subtotal": d.field("67,367.00"),
        "taxable_value": d.missing("no line labelled as the taxable value; only a Sub Total"),
        "tax_lines": [tax_line(d, "CGST @ 9%", "9%", "6,063.03"), tax_line(d, "SGST @ 9%", "9%", "6,063.03")],
        "tax_total": d.missing("no combined tax total printed"),
        "adjustments": [],
        "grand_total": d.field("79,493.06"),
    }


def wire_02() -> dict:
    d = Doc("02_igst_totals_top")
    return {
        "invoice_number": d.field("DPI-INV-2025-0932"),
        "invoice_date": d.field("11-Nov-2025"),
        "due_date": d.field("11-Dec-2025"),
        "currency": d.field("INR"),
        "seller": party(d, "Deccan Packaging Industries", "Gat No. 118, Chakan MIDC\nPune, Maharashtra 410501", "27AAEFD5512Q1Z5"),
        "buyer": party(d, "Nilgiri Foods Pvt. Ltd.", "14 Industrial Suburb, Yeshwanthpur\nBengaluru, Karnataka 560022", "29AACCN6274H1Z5"),
        "line_items": d.rows(r"(.+) (\d{4}) (\d+) ([\d,]+\.\d{2}) ([\d,]+\.\d{2})", ROW_CELLS[1:]),
        "subtotal": d.field("INR 152,355.00"),  # the only pre-tax total, labelled Taxable Value
        "taxable_value": d.field("INR 152,355.00"),
        "tax_lines": [tax_line(d, "IGST @ 18%", "18%", "INR 27,423.90", amount_line_label="INR 27,423.90")],
        "tax_total": d.missing("no combined tax total printed"),
        "adjustments": [],
        "grand_total": d.field("INR 179,778.90"),
    }


def wire_03() -> dict:
    d = Doc("03_multipage_roundoff")
    return {
        "invoice_number": d.field("GEH/1124/2025"),
        "invoice_date": d.field("07/11/2025"),
        "due_date": d.field("07/12/2025"),
        "currency": d.field("Rs."),
        "seller": party(d, "Ganga Electricals & Hardware", "21, Ganesh Peth, Nagpur, Maharashtra 440018", "27AAJFG3398C1ZA"),
        "buyer": party(d, "Vidarbha Construction Co.", "Near Zero Mile, Civil Lines\nNagpur, Maharashtra 440001", "27AAAFV8817D1Z4"),
        "line_items": d.rows(INDIAN_ROW, ROW_CELLS),
        "subtotal": d.field("5,65,388.15"),
        "taxable_value": d.field("5,63,888.15"),
        "tax_lines": [tax_line(d, "CGST @ 9%", "9%", "50,749.93"), tax_line(d, "SGST @ 9%", "9%", "50,749.93")],
        "tax_total": d.missing("only separate CGST and SGST lines are printed"),
        "adjustments": [
            {"label": d.field("Less: Trade Discount"), "amount": d.field("-1,500.00")},
            {"label": d.field("Round Off"), "amount": d.field("-0.01")},
        ],
        "grand_total": d.field("6,65,388.00"),
    }


def wire_05() -> dict:
    d = Doc("05_eu_vat_defective")
    return {
        "invoice_number": d.missing("no invoice number is printed"),
        "invoice_date": d.field("04.11.2025"),
        "due_date": d.missing("no due date printed; only 'Payment terms: 14 days net'"),
        "currency": d.beside("Total due", "€"),
        "seller": party(d, "Müller Bürobedarf GmbH", "Friedrichstraße 112 · 10117 Berlin · Germany", "DE12345678"),
        "buyer": party(d, "Van Dijk Design B.V.", "Keizersgracht 221\n1016 DV Amsterdam, Netherlands", "NL853746219B01"),
        "line_items": d.rows(r"(\d+) (BO-\d{4}) (.+) (\d+) ([\d.,]+ €) ([\d.,]+ €)", (None, "hsn_sac", "description", "quantity", "unit_price", "amount")),
        "subtotal": d.field("1.457,00 €"),
        "taxable_value": d.missing("no line labelled as the taxable value"),
        "tax_lines": [tax_line(d, "VAT 19%", "19%", "276,83 €")],
        "tax_total": d.missing("no combined tax total printed"),
        "adjustments": [],
        "grand_total": d.field("1.733,83 €"),
    }


def as_json(wire: dict) -> str:
    return json.dumps(wire, ensure_ascii=False)


# --- Comparing with ground truth ----------------------------------------------------------------


def compare(inv: Invoice, exp: ExpectedInvoice) -> list[str]:
    """Every field where the pipeline's value differs from the hand-written answer.

    Uses the accuracy harness's scorer, so tests and the README's accuracy numbers share one definition of correct.
    """
    return [f"{s.path}: expected {s.expected!r}, got {s.actual!r}" for s in score_invoice("test", inv, exp) if not s.correct]

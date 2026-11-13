"""
What:     Unit tests for app/provenance.py: how citations are verified, relocated or rejected, and how confidence is scored.
In:       Small hand-built pages, plus fixture 03's real two-column text.
Out:      pytest pass/fail.
Fails if: Matching gets looser or stricter. Each test pins one rule from the module docstring.
"""

import pytest

from app.models import MatchType, PageInfo
from app.provenance import confidence, locate, locate_row, source_ref
from tests.helpers import pages_of


def page(*lines: str, n: int = 1, ocr_conf: list[float] | None = None) -> PageInfo:
    text = "\n".join(lines)
    if ocr_conf is None:
        return PageInfo(page=n, text_source="text_layer", ocr_reasons=[], char_count=len(text), text=text)
    return PageInfo(page=n, text_source="ocr", ocr_reasons=["too little text"], char_count=len(text),
                    ocr_mean_conf=sum(ocr_conf) / len(ocr_conf), ocr_line_conf=ocr_conf, text=text)


DOC = [page("TAX INVOICE", "Invoice No. SOS/2025-26/0147", "Invoice Date 05/11/2025", "Sub Total 67,367.00", "CGST @ 9% 6,063.03")]


def test_exact_match_on_cited_line():
    loc = locate("SOS/2025-26/0147", 1, 2, 2, DOC)
    assert loc.match_type is MatchType.EXACT
    assert loc.quote == "Invoice No. SOS/2025-26/0147"
    assert loc.quote[slice(*loc.span)] == "SOS/2025-26/0147" == loc.text


def test_quote_is_narrowed_to_the_matched_lines():
    loc = locate("67,367.00", 1, 1, 5, DOC)  # the LLM cited a wide range
    assert (loc.line_start, loc.line_end, loc.quote) == (4, 4, "Sub Total 67,367.00")


@pytest.mark.parametrize(
    "value,printed",
    [
        ("sos/2025-26/0147", "SOS/2025-26/0147"),  # case
        ("Invoice  No.", "Invoice No."),  # whitespace
        ("SOS/2025–26/0147", "SOS/2025-26/0147"),  # en dash
    ],
)
def test_normalized_match_returns_the_printed_text(value, printed):
    loc = locate(value, 1, 2, 2, DOC)
    assert loc.match_type is MatchType.NORMALIZED
    assert loc.text == printed  # the page's characters, not the LLM's


def test_wrong_line_is_relocated_with_lower_trust():
    loc = locate("6,063.03", 1, 2, 2, DOC)
    assert loc.match_type is MatchType.RELOCATED and loc.line_start == 5


def test_relocation_prefers_the_nearest_line_on_the_cited_page():
    doc = [page("Total 10.00", "x", "x", "x", "x", "Total 10.00", "x")]
    assert locate("Total 10.00", 1, 7, 7, doc).line_start == 6


def test_relocation_searches_other_pages():
    doc = [page("Header"), page("Grand Total 6,65,388.00", n=2)]
    loc = locate("6,65,388.00", 1, 1, 1, doc)
    assert (loc.page, loc.line_start, loc.match_type) == (2, 1, MatchType.RELOCATED)


def test_out_of_range_citation_is_relocated():
    assert locate("67,367.00", 1, 40, 41, DOC).line_start == 4
    assert locate("67,367.00", 9, 1, 1, DOC).line_start == 4  # page that doesn't exist


def test_invented_value_is_rejected():
    assert locate("INV-99999", 1, 2, 2, DOC) is None


def test_fuzzy_match_within_cited_lines_takes_the_page_text():
    doc = [page("lnvoice N0: YT/2O25/1188 dated")]  # OCR noise on the page
    loc = locate("YT/2025/1188", 1, 1, 1, doc)  # the LLM "corrected" O to 0
    assert loc.match_type is MatchType.FUZZY
    assert loc.text == "YT/2O25/1188"  # the value is what the page says


def test_fuzzy_never_runs_on_short_values_or_outside_the_cited_lines():
    assert locate("9%", 1, 5, 5, [page("x", "x", "x", "x", "CGST @ 8%")]) is None
    doc = [page("Header line", "lnvoice N0: YT/2O25/1188")]
    assert locate("YT/2025/1188", 1, 1, 1, doc) is None  # fuzzy match exists only on line 2


def test_fuzzy_never_truncates_the_printed_value():
    # A window without the mismatched character scores higher (0.97) than the whole token (0.94); it must not win.
    loc = locate("SOS/2025-26/0174", 1, 1, 1, [page("Invoice No. SOS/2025-26/0147")])
    assert loc.text == "SOS/2025-26/0147"


def test_fuzzy_needs_high_similarity():
    assert locate("1,22,248.00", 1, 1, 1, [page("Amount 9,99,111.00")]) is None


def test_short_values_must_match_at_token_boundaries():
    doc = [page("3 Stapler Heavy Duty HD-45 8472")]
    assert locate("5", 1, 1, 1, doc) is None  # only inside "HD-45"
    assert locate("8472", 1, 1, 1, doc).text == "8472"


def test_multi_line_value_in_a_two_column_layout():
    # Fixture 03: the buyer address shares lines with the invoice metadata column.
    pages = list(pages_of("03_multipage_roundoff"))
    loc = locate("Near Zero Mile, Civil Lines\nNagpur, Maharashtra 440001", 1, 6, 7, pages)
    assert loc.match_type is MatchType.EXACT
    assert (loc.line_start, loc.line_end) == (6, 7)
    assert "Due Date" in loc.quote  # the quote is the full printed lines
    assert loc.text == "Near Zero Mile, Civil Lines\nNagpur, Maharashtra 440001"
    assert [loc.quote[slice(*s)] for s in loc.spans] == ["Near Zero Mile, Civil Lines", "Nagpur, Maharashtra 440001"]


def test_multi_line_value_segments_must_be_in_order():
    doc = [page("Nagpur, Maharashtra 440001", "Near Zero Mile, Civil Lines")]
    assert locate("Near Zero Mile, Civil Lines\nNagpur, Maharashtra 440001", 1, 1, 2, doc) is None


# --- Rows ------------------------------------------------------------------------------------


ROW_PAGE = page("1 Modular Switch 6A 1-Way 8536 50 231.79 11,589.50", "3 Stapler Heavy Duty HD-45 8472 5 1,250.00 6,250.00")


def test_row_cells_claim_distinct_spans():
    cells = {"description": "Modular Switch 6A 1-Way", "hsn_sac": "8536", "quantity": "50", "unit_price": "231.79", "amount": "11,589.50"}
    row = locate_row(cells, ROW_PAGE, 1, 1)
    assert row.match_type is MatchType.EXACT
    assert {c: row.quote[slice(*m.span)] for c, m in row.cells.items()} == cells
    # "50" is the quantity column, not the tail of "11,589.50"
    start, end = row.cells["quantity"].span
    assert row.quote[start - 1 : end + 1] == " 50 "


def test_row_short_value_ignores_matches_inside_other_tokens():
    cells = {"description": "Stapler Heavy Duty HD-45", "quantity": "5", "amount": "6,250.00"}
    row = locate_row(cells, ROW_PAGE, 2, 2)
    start, end = row.cells["quantity"].span
    assert row.quote[start - 1 : end + 1] == " 5 "


def test_row_reports_only_cells_it_found():
    row = locate_row({"description": "Modular Switch 6A 1-Way", "amount": "99,999.00"}, ROW_PAGE, 1, 1)
    assert set(row.cells) == {"description"}


def test_row_with_nothing_found_or_bad_range_is_none():
    assert locate_row({"description": "Invented Widget"}, ROW_PAGE, 1, 1) is None
    assert locate_row({"description": "Modular Switch 6A 1-Way"}, ROW_PAGE, 3, 3) is None


# --- Confidence ------------------------------------------------------------------------------


def test_confidence_factors():
    text_page = DOC[0]
    assert confidence(MatchType.EXACT, text_page, 2, 2) == 1.0
    assert confidence(MatchType.NORMALIZED, text_page, 2, 2) == 0.95
    assert confidence(MatchType.RELOCATED, text_page, 2, 2) == 0.75
    assert confidence(MatchType.FUZZY, text_page, 2, 2) == 0.6
    assert confidence(MatchType.EXACT, text_page, 2, 2, certainty=0.9) == 0.9


def test_confidence_uses_ocr_confidence_of_the_quoted_lines():
    scan = page("Header", "Total 1,234.00", "Footer", ocr_conf=[95.0, 80.0, 40.0])
    assert confidence(MatchType.EXACT, scan, 2, 2) == 0.8
    assert confidence(MatchType.EXACT, scan, 2, 3) == 0.6  # mean of 80 and 40
    loc = locate("1,234.00", 1, 2, 2, [scan])
    ref = source_ref(loc, scan)
    assert ref.text_source == "ocr" and ref.confidence == 0.8 and ref.quote == "Total 1,234.00"

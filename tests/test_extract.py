"""
What:     Proves both extraction paths: text layer for digital pages, OCR for pages whose text layer can't be trusted.
In:       The fixture PDFs, small PDFs built in-test (mixed pages, rotated scan), and a fake OCR engine.
Out:      pytest pass/fail. Real-Tesseract tests skip, with a reason, when the binary isn't installed.
Fails if: A heuristic threshold moves. The boundary tests below pin each one.
"""

import io
from pathlib import Path

import pdfplumber
import pytest
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from app.extract import (
    IMAGE_COVERAGE,
    MIN_CHARS,
    SPARSE_CHARS,
    ExtractionError,
    OcrResult,
    OcrUnavailableError,
    canonical_text,
    extract_pages,
    lines_from_tesseract_data,
    ocr_reasons,
    tesseract_available,
)

PDFS = Path(__file__).parent / "fixtures" / "pdfs"
requires_tesseract = pytest.mark.skipif(
    not tesseract_available(), reason="tesseract not installed (brew install tesseract)"
)


def pdf(name: str) -> bytes:
    return (PDFS / name).read_bytes()


class FakeOcr:
    """Records the images it receives and returns canned lines."""

    def __init__(self, lines=("TAX INVOICE", "Invoice No. YT/2025/1188"), confs=(91.0, 84.5)):
        self.result = OcrResult(lines=list(lines), line_conf=list(confs), mean_conf=88.0)
        self.images: list[Image.Image] = []

    def __call__(self, image):
        self.images.append(image)
        return self.result


class ExplodingOcr:
    def __call__(self, image):
        raise AssertionError("OCR must not run on a page with a good text layer")


def page_image(name: str, page: int = 0, dpi: int = 100) -> Image.Image:
    with pdfplumber.open(PDFS / name) as doc:
        return doc.pages[page].to_image(resolution=dpi).original.convert("RGB")


def build_pdf(*pages: str | Image.Image) -> bytes:
    """Each arg is a page: a str is drawn as real text, an Image fills the page as a scan."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4, invariant=1)
    for p in pages:
        if isinstance(p, str):
            y = A4[1] - 72
            for line in p.splitlines():
                c.drawString(72, y, line)
                y -= 14
        else:
            c.drawImage(ImageReader(p), 0, 0, width=A4[0], height=A4[1])
        c.showPage()
    c.save()
    return buf.getvalue()


INVOICE_TEXT = "TAX INVOICE\nInvoice No. SOS/2025-26/0147\nInvoice Date 05/11/2025\nGSTIN: 29AAGCS4821K1Z0"


# --- Text-layer path -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,pages",
    [("01_gst_classic.pdf", 1), ("02_igst_totals_top.pdf", 1), ("03_multipage_roundoff.pdf", 2), ("05_eu_vat_defective.pdf", 1)],
)
def test_digital_fixtures_use_text_layer_and_never_ocr(name, pages):
    result = extract_pages(pdf(name), ocr=ExplodingOcr())
    assert len(result) == pages
    for p in result:
        assert p.text_source == "text_layer"
        assert p.ocr_reasons == [] and p.ocr_mean_conf is None and p.ocr_line_conf is None
        assert p.char_count == len(p.text) > 200


def test_text_layer_content_and_line_numbering():
    p1, p2 = extract_pages(pdf("03_multipage_roundoff.pdf"))
    assert p1.lines[0] == "Ganga Electricals & Hardware"
    assert "1 Modular Switch 6A 1-Way 8536 50 231.79 11,589.50" in p1.lines
    assert p2.lines[11] == "Grand Total (Rs.) 6,65,388.00"  # line 12 on page 2
    assert "Müller Bürobedarf GmbH" in extract_pages(pdf("05_eu_vat_defective.pdf"))[0].text


def test_canonical_text_shape():
    for p in extract_pages(pdf("01_gst_classic.pdf")) + extract_pages(pdf("03_multipage_roundoff.pdf")):
        assert all(line and line == line.rstrip() for line in p.lines)
    assert canonical_text("  a  \n\n\t\nb\r\nc   ") == "  a\nb\nc"


# --- OCR path (routing proven with a fake engine; no Tesseract needed) -----------------------


def test_scanned_fixture_is_routed_to_ocr():
    fake = FakeOcr()
    [page] = extract_pages(pdf("04_scanned_rotated.pdf"), ocr=fake, dpi=100)
    assert page.text_source == "ocr"
    assert any(r.startswith("too little text") for r in page.ocr_reasons)
    assert any(r.startswith("image-dominated page") for r in page.ocr_reasons)
    assert page.text == "TAX INVOICE\nInvoice No. YT/2025/1188"
    assert page.ocr_line_conf == [91.0, 84.5] and page.ocr_mean_conf == 88.0
    assert len(fake.images) == 1


def test_ocr_receives_page_rendered_at_requested_dpi():
    fake = FakeOcr()
    extract_pages(pdf("04_scanned_rotated.pdf"), ocr=fake, dpi=100)
    with pdfplumber.open(PDFS / "04_scanned_rotated.pdf") as doc:
        expected_width = doc.pages[0].width * 100 / 72
    assert abs(fake.images[0].width - expected_width) <= 2


def test_decision_is_per_page_in_a_mixed_pdf():
    fake = FakeOcr()
    pages = extract_pages(build_pdf(INVOICE_TEXT * 3, page_image("01_gst_classic.pdf")), ocr=fake, dpi=72)
    assert [p.text_source for p in pages] == ["text_layer", "ocr"]
    assert len(fake.images) == 1  # only the scanned page was OCR'd
    assert pages[0].lines[1] == "Invoice No. SOS/2025-26/0147"


def test_blank_ocr_lines_dropped_with_their_confidences():
    fake = FakeOcr(lines=["Header", "   ", "Total 10.00"], confs=[90.0, 12.0, 70.0])
    [page] = extract_pages(pdf("04_scanned_rotated.pdf"), ocr=fake, dpi=72)
    assert page.lines == ["Header", "Total 10.00"]
    assert page.ocr_line_conf == [90.0, 70.0]


def test_empty_ocr_result_degrades_to_an_empty_page():
    fake = FakeOcr(lines=[], confs=[])
    fake.result = OcrResult(lines=[], line_conf=[], mean_conf=0.0)
    [page] = extract_pages(pdf("04_scanned_rotated.pdf"), ocr=fake, dpi=72)
    assert page.text == "" and page.lines == [] and page.ocr_line_conf == [] and page.ocr_mean_conf == 0.0


def test_missing_tesseract_is_a_clear_error(monkeypatch):
    monkeypatch.setenv("TESSERACT_CMD", "/nonexistent/tesseract")
    with pytest.raises(OcrUnavailableError, match=r"page 1 needs OCR .*brew install tesseract"):
        extract_pages(pdf("04_scanned_rotated.pdf"), dpi=72)


def test_missing_tesseract_does_not_affect_digital_pdfs(monkeypatch):
    monkeypatch.setenv("TESSERACT_CMD", "/nonexistent/tesseract")
    assert extract_pages(pdf("01_gst_classic.pdf"))[0].text_source == "text_layer"


@pytest.mark.parametrize("data", [b"", b"not a pdf", b"%PDF-1.4\n%garbage"])
def test_unreadable_input_raises_extraction_error(data):
    with pytest.raises(ExtractionError, match="not a readable PDF|no pages"):
        extract_pages(data)


# --- Heuristic (pure function, boundaries pinned) ---------------------------------------------


def test_good_invoice_text_is_trusted():
    assert ocr_reasons(INVOICE_TEXT, image_coverage=0.0) == []


def test_digital_page_with_large_background_image_is_trusted():
    # e.g. a letterhead or watermark image behind a real text layer
    assert ocr_reasons(INVOICE_TEXT * 3, image_coverage=0.95) == []


@pytest.mark.parametrize("n,flagged", [(MIN_CHARS - 1, True), (MIN_CHARS, False)])
def test_too_little_text_boundary(n, flagged):
    reasons = ocr_reasons("A" * n, image_coverage=0.0)
    assert any(r.startswith("too little text") for r in reasons) is flagged


def test_broken_font_encoding():
    text = "Invoice (cid:12)(cid:7)(cid:44) Total (cid:3)(cid:9) 1,234.00 " * 3
    assert any(r.startswith("broken font encoding") for r in ocr_reasons(text, 0.0))
    assert any(r.startswith("broken font encoding") for r in ocr_reasons("Invoice ��� Total 1,234.00 " * 3, 0.0))


@pytest.mark.parametrize(
    "clean,junk,flagged",
    [(59, 41, True), (60, 40, False)],  # MIN_CLEAN_RATIO = 0.60, inclusive
)
def test_garbage_text_boundary(clean, junk, flagged):
    text = "A" * clean + "▯" * junk  # U+25AF: the "missing glyph" box
    assert any(r.startswith("garbage text") for r in ocr_reasons(text, 0.0)) is flagged


@pytest.mark.parametrize(
    "coverage,n,flagged",
    [
        (IMAGE_COVERAGE, SPARSE_CHARS - 1, True),
        (IMAGE_COVERAGE - 0.01, SPARSE_CHARS - 1, False),
        (IMAGE_COVERAGE, SPARSE_CHARS, False),
    ],
)
def test_image_dominated_boundary(coverage, n, flagged):
    reasons = ocr_reasons("A" * n, image_coverage=coverage)
    assert any(r.startswith("image-dominated page") for r in reasons) is flagged


def test_unicode_letters_count_as_clean():
    assert ocr_reasons("Müller Bürobedarf GmbH, Friedrichstraße 112, 10117 Berlin. Gesamtbetrag 1.733,83 €", 0.0) == []


# --- Tesseract output parsing (pure) ----------------------------------------------------------


def test_lines_from_tesseract_data_groups_and_averages():
    data = {
        "text": ["", "TAX", "INVOICE", "", "Total", "1,234.00", " "],
        "conf": [-1, 96, 90, -1, 80, 60, 95],
        "page_num": [1] * 7,
        "block_num": [1, 1, 1, 2, 2, 2, 2],
        "par_num": [1] * 7,
        "line_num": [0, 1, 1, 0, 1, 1, 1],
    }
    r = lines_from_tesseract_data(data)
    assert r.lines == ["TAX INVOICE", "Total 1,234.00"]
    assert r.line_conf == [93.0, 70.0]
    assert r.mean_conf == 81.5  # mean over the four real words


def test_lines_from_tesseract_data_empty():
    empty = {k: [] for k in ("text", "conf", "page_num", "block_num", "par_num", "line_num")}
    assert lines_from_tesseract_data(empty) == OcrResult(lines=[], line_conf=[], mean_conf=0.0)


# --- Real Tesseract (skipped until installed) -------------------------------------------------


@requires_tesseract
def test_tesseract_reads_scanned_fixture():
    [page] = extract_pages(pdf("04_scanned_rotated.pdf"))
    assert page.text_source == "ocr"
    assert page.ocr_mean_conf > 60
    for token in ("INVOICE", "Yamuna", "Plywood", "1188"):
        assert token in page.text, f"{token!r} not found in OCR text:\n{page.text}"


@requires_tesseract
def test_tesseract_corrects_a_sideways_scan():
    sideways = page_image("04_scanned_rotated.pdf", dpi=150).rotate(90, expand=True)
    [page] = extract_pages(build_pdf(sideways))
    assert page.text_source == "ocr"
    assert "Yamuna" in page.text, f"orientation not corrected; OCR text:\n{page.text}"

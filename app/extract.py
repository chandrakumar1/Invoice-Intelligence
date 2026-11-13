"""
What:     Turns a PDF into canonical, line-numbered page text: pdfplumber's text layer, or Tesseract OCR for pages that need it.
In:       PDF bytes; optionally an OCR engine (tests inject a fake) and a rasterization DPI.
Out:      list[PageInfo], one per page. Its `text` is what the LLM reads and what every quote is verified against.
Fails if: A junk text layer passes all four heuristics, so garbage is trusted instead of OCR'd. Thresholds are tested at their boundaries.

The decision is made per page: one PDF can mix digital pages with a scanned annexure.
"""

import io
import os
import re
import shutil
from dataclasses import dataclass
from typing import Protocol

import pdfplumber
import pytesseract
from PIL import Image

from app.models import PageInfo

# --- When is a text layer untrustworthy? Any one reason sends the page to OCR. --------------
MIN_CHARS = 30  # fewer visible characters than this: effectively no text layer
MAX_BROKEN_RATIO = 0.05  # share of "(cid:NN)" / U+FFFD characters: font has no Unicode mapping
MIN_CLEAN_RATIO = 0.60  # share of letters, digits, and common invoice punctuation
IMAGE_COVERAGE = 0.70  # images covering this much of the page...
SPARSE_CHARS = 200  # ...with fewer characters than this: a scan, maybe with a junk OCR layer

CID = re.compile(r"\(cid:\d+\)")
INVOICE_PUNCT = frozenset(".,:;/-()%#&@'+*|_\"!?[]€₹$£")
DEFAULT_DPI = 300


class ExtractionError(Exception):
    """The PDF can't be read at all (corrupt, encrypted, empty)."""


class OcrUnavailableError(ExtractionError):
    """A page needs OCR but no Tesseract binary was found."""


def ocr_reasons(text: str, image_coverage: float) -> list[str]:
    """Why a page's text layer can't be trusted. An empty list means use the text layer."""
    visible = [c for c in text if not c.isspace()]
    n = len(visible)
    reasons = []
    if n < MIN_CHARS:
        reasons.append(f"too little text ({n} characters, need {MIN_CHARS})")
    if n:
        broken = sum(len(m) for m in CID.findall(text)) + text.count("�")
        if broken / n > MAX_BROKEN_RATIO:
            reasons.append(f"broken font encoding ({broken / n:.0%} of characters are cid/replacement codes)")
        clean = sum(c.isalnum() or c in INVOICE_PUNCT for c in visible)
        if clean / n < MIN_CLEAN_RATIO:
            reasons.append(f"garbage text (only {clean / n:.0%} recognizable characters)")
    if image_coverage >= IMAGE_COVERAGE and n < SPARSE_CHARS:
        reasons.append(f"image-dominated page ({image_coverage:.0%} covered by images, {n} characters)")
    return reasons


# --- OCR engine ----------------------------------------------------------------------------


@dataclass(frozen=True)
class OcrResult:
    lines: list[str]
    line_conf: list[float]  # 0-100, one per line
    mean_conf: float  # 0-100, over all words


class OcrEngine(Protocol):
    def __call__(self, image: Image.Image) -> OcrResult: ...


def tesseract_available(cmd: str | None = None) -> bool:
    return shutil.which(cmd or os.environ.get("TESSERACT_CMD") or "tesseract") is not None


class TesseractOcr:
    """Grayscale, fix 90/180/270 degree rotation via orientation detection, then word-level OCR."""

    def __init__(self, cmd: str | None = None):
        if cmd:
            pytesseract.pytesseract.tesseract_cmd = cmd  # pytesseract only supports a module-level setting

    def __call__(self, image: Image.Image) -> OcrResult:
        gray = _upright(image.convert("L"))
        return lines_from_tesseract_data(pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT))


def _upright(image: Image.Image) -> Image.Image:
    try:
        osd = pytesseract.image_to_osd(image, output_type=pytesseract.Output.DICT)
    except pytesseract.TesseractError:
        return image  # too little text to detect orientation; OCR as-is
    # OSD reports the clockwise rotation that makes the page upright; PIL rotates counter-clockwise.
    rotate = int(osd.get("rotate", 0))
    return image.rotate(-rotate, expand=True, fillcolor=255) if rotate else image


def lines_from_tesseract_data(data: dict) -> OcrResult:
    """Group Tesseract's word boxes into lines, in reading order, with mean confidence per line."""
    lines: dict[tuple[int, int, int, int], list[tuple[str, float]]] = {}
    for i, word in enumerate(data["text"]):
        conf = float(data["conf"][i])
        if conf < 0 or not word.strip():  # -1 marks layout rows (blocks, paragraphs), not words
            continue
        key = (data["page_num"][i], data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append((word.strip(), conf))
    words = [c for line in lines.values() for _, c in line]
    return OcrResult(
        lines=[" ".join(w for w, _ in line) for line in lines.values()],
        line_conf=[round(sum(c for _, c in line) / len(line), 1) for line in lines.values()],
        mean_conf=round(sum(words) / len(words), 1) if words else 0.0,
    )


# --- Pages ---------------------------------------------------------------------------------


def extract_pages(pdf_bytes: bytes, *, ocr: OcrEngine | None = None, dpi: int | None = None) -> list[PageInfo]:
    dpi = dpi or int(os.environ.get("OCR_DPI") or DEFAULT_DPI)
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            if not pdf.pages:
                raise ExtractionError("the PDF has no pages")
            return [_extract_page(page, n, ocr, dpi) for n, page in enumerate(pdf.pages, 1)]
    except ExtractionError:
        raise
    except Exception as e:  # pdfminer raises many unrelated types for corrupt or encrypted input
        raise ExtractionError(f"not a readable PDF: {type(e).__name__}: {e}") from e


def _extract_page(page, n: int, ocr: OcrEngine | None, dpi: int) -> PageInfo:
    text = canonical_text(page.extract_text() or "")
    reasons = ocr_reasons(text, _image_coverage(page))
    if not reasons:
        return PageInfo(page=n, text_source="text_layer", ocr_reasons=[], char_count=len(text), text=text)

    if ocr is None:
        if not tesseract_available():
            raise OcrUnavailableError(
                f"page {n} needs OCR ({'; '.join(reasons)}) but tesseract was not found. "
                "Install it (brew install tesseract) or set TESSERACT_CMD."
            )
        ocr = TesseractOcr(os.environ.get("TESSERACT_CMD"))
    result = ocr(page.to_image(resolution=dpi).original)
    kept = [(line.rstrip(), conf) for line, conf in zip(result.lines, result.line_conf) if line.strip()]
    text = "\n".join(line for line, _ in kept)
    return PageInfo(
        page=n,
        text_source="ocr",
        ocr_reasons=reasons,
        char_count=len(text),
        ocr_mean_conf=result.mean_conf,
        ocr_line_conf=[conf for _, conf in kept],
        text=text,
    )


def canonical_text(raw: str) -> str:
    """One line per printed line: trailing whitespace stripped, blank lines dropped, '\\n'-joined."""
    return "\n".join(line.rstrip() for line in raw.splitlines() if line.strip())


def _image_coverage(page) -> float:
    x0, top, x1, bottom = page.bbox
    area = float((x1 - x0) * (bottom - top))
    if area <= 0:
        return 0.0
    covered = 0.0
    for im in page.images:
        w = min(im["x1"], x1) - max(im["x0"], x0)
        h = min(im["bottom"], bottom) - max(im["top"], top)
        covered += max(w, 0) * max(h, 0)
    return min(covered / area, 1.0)  # overlapping images can double count

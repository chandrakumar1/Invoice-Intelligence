"""
What:     Proves the fixture PDFs have the properties later milestones rely on, the generator is reproducible, and expected files point at real PDFs.
In:       tests/fixtures/pdfs/*.pdf, tests/fixtures/expected/*.json, and scripts/generate_fixtures.py (run into a temp dir).
Out:      pytest pass/fail.
Fails if: A dependency upgrade changes the generator's bytes; regenerate, re-check the PDFs by eye, and commit them.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pdfplumber
import pytest

ROOT = Path(__file__).resolve().parent.parent
PDFS = ROOT / "tests" / "fixtures" / "pdfs"

EXPECTED_PAGES = {
    "01_gst_classic.pdf": 1,
    "02_igst_totals_top.pdf": 1,
    "03_multipage_roundoff.pdf": 2,
    "04_scanned_rotated.pdf": 1,
    "05_eu_vat_defective.pdf": 1,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_all_fixtures_present():
    assert sorted(p.name for p in PDFS.glob("*.pdf")) == sorted(EXPECTED_PAGES)


@pytest.mark.parametrize("name,pages", EXPECTED_PAGES.items())
def test_page_counts(name, pages):
    with pdfplumber.open(PDFS / name) as pdf:
        assert len(pdf.pages) == pages


@pytest.mark.parametrize("name", [n for n in EXPECTED_PAGES if not n.startswith("04")])
def test_digital_fixtures_have_text_layer(name):
    with pdfplumber.open(PDFS / name) as pdf:
        for page in pdf.pages:
            assert len((page.extract_text() or "").strip()) > 200


def test_scanned_fixture_has_no_text_layer():
    # M2's OCR fallback must be the only way to read this one.
    with pdfplumber.open(PDFS / "04_scanned_rotated.pdf") as pdf:
        page = pdf.pages[0]
        assert (page.extract_text() or "").strip() == ""
        assert len(page.images) == 1


EXPECTED_FILES = sorted(p for p in (ROOT / "tests" / "fixtures" / "expected").glob("*.json") if not p.name.startswith("_"))
FIXTURE_PDF_DIRS = [PDFS, ROOT / "tests" / "fixtures" / "private"]


@pytest.mark.parametrize("path", EXPECTED_FILES, ids=lambda p: p.name)
def test_expected_file_references_existing_pdf(path):
    fixture = json.loads(path.read_text())["fixture"]
    assert any((d / fixture).is_file() for d in FIXTURE_PDF_DIRS), f"{path.name}: no PDF named {fixture!r}"
    # Naming convention from tests/fixtures/README.md: <pdf-stem>.json
    assert Path(fixture).stem == path.stem, f"{path.name} should be named {Path(fixture).stem}.json"


def test_generator_is_byte_identical(tmp_path):
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "generate_fixtures.py"), "--out", str(tmp_path)],
        check=True,
        capture_output=True,
    )
    for name in EXPECTED_PAGES:
        assert sha256(tmp_path / name) == sha256(PDFS / name), f"{name} differs from the committed fixture"

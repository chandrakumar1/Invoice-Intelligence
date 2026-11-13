"""
What:     Tests the review UI: the server serves it safely, and its JavaScript logic is correct against a real pipeline result.
In:       The FastAPI app (fake LLM, temp DATA_DIR) and tests/review_ui.test.js, run with Node when it's installed.
Out:      pytest pass/fail. The Node part skips, with a reason, if `node` isn't on PATH.
Fails if: The result JSON changes shape without static/review.js following, or the page's security header is dropped.
"""

import re
import shutil
import subprocess
from datetime import date
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.llm import LLMReply
from app.main import REVIEW_CSP, create_app
from app.storage import Storage
from tests.helpers import FIXTURES, as_json, wire_05

ROOT = Path(__file__).resolve().parent.parent


class FakeClient:
    def complete(self, **_):
        return LLMReply(text=as_json(wire_05()), content=[], model="claude-opus-5-5", provider="fake")


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(storage=Storage(tmp_path), llm_factory=FakeClient, today=lambda: date(2026, 11, 12))) as c:
        yield c


def test_review_page_is_served_with_a_strict_policy(client):
    page = client.get("/")
    assert page.status_code == 200 and page.headers["content-type"].startswith("text/html")
    assert page.headers["content-security-policy"] == REVIEW_CSP
    assert "script-src 'self'" in REVIEW_CSP and "frame-ancestors 'none'" in REVIEW_CSP
    assert '<script src="/static/review.js"></script>' in page.text
    assert "<script>" not in page.text  # no inline script, so the policy can forbid it


def test_review_script_is_served(client):
    script = client.get("/static/review.js")
    assert script.status_code == 200 and "function flattenRows" in script.text


def test_ui_never_parses_document_text_as_html():
    source = (ROOT / "static" / "review.js").read_text()
    assert not re.search(r"\.\s*(inner|outer)HTML\b|insertAdjacentHTML|document\.write", source)


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not installed; review.js logic tests skipped")
def test_review_js_logic(client, tmp_path):
    job = client.post("/extract", files={"file": ("05.pdf", (FIXTURES / "pdfs" / "05_eu_vat_defective.pdf").read_bytes(), "application/pdf")}).json()
    result_file = tmp_path / "result.json"
    result_file.write_text(client.get(job["result_url"]).text)
    run = subprocess.run(
        ["node", "--test", str(ROOT / "tests" / "review_ui.test.js")],
        env={"REVIEW_RESULT_JSON": str(result_file), "PATH": shutil.os.environ["PATH"]},
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stdout + run.stderr

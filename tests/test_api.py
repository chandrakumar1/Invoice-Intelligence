"""
What:     Tests the HTTP API and the pipeline behind it, end to end, with a fake LLM client and a temporary DATA_DIR.
In:       Fixture PDFs posted to POST /extract; results read back from GET /result/{id}.
Out:      pytest pass/fail. No network: the LLM client is a fake that returns mocked answers.
Fails if: The response contract changes, or a failure path leaves a job pending instead of failed.
"""

import hashlib
import json
from datetime import UTC, date, datetime

import anthropic
import pytest
from fastapi.testclient import TestClient

from app import __version__
from app.llm import PROMPT_VERSION, LLMRefusalError, LLMReply
from app.main import INTERRUPTED, create_app
from app.models import ExtractionResult, JobStatus
from app.pipeline import new_job
from app.storage import Storage
from tests.helpers import FIXTURES, as_json, compare, expected, wire_01, wire_05

TODAY = date(2026, 11, 12)


class FakeClient:
    def __init__(self, *answers: str):
        self.answers = list(answers)

    def complete(self, *, system, messages):
        return LLMReply(text=self.answers.pop(0), content=[], model="claude-opus-5-5", provider="fake")


def client_for(tmp_path, llm_factory, **kw) -> TestClient:
    return TestClient(create_app(storage=Storage(tmp_path), llm_factory=llm_factory, today=lambda: TODAY, **kw))


def pdf(stem: str) -> bytes:
    return (FIXTURES / "pdfs" / f"{stem}.pdf").read_bytes()


def upload(client: TestClient, data: bytes, name: str = "invoice.pdf"):
    return client.post("/extract", files={"file": (name, data, "application/pdf")})


# --- Happy path ----------------------------------------------------------------------------------


def test_upload_returns_202_then_the_result_is_done(tmp_path):
    with client_for(tmp_path, lambda: FakeClient(as_json(wire_01()))) as client:
        accepted = upload(client, pdf("01_gst_classic"), "01_gst_classic.pdf")
        assert accepted.status_code == 202
        body = accepted.json()
        assert body["status"] == "pending" and body["result_url"] == f"/result/{body['id']}"
        assert accepted.headers["location"] == body["result_url"]

        got = client.get(body["result_url"])
    assert got.status_code == 200
    result = ExtractionResult.model_validate(got.json())
    assert result.status is JobStatus.DONE and result.error is None
    assert (result.filename, result.sha256) == ("01_gst_classic.pdf", hashlib.sha256(pdf("01_gst_classic")).hexdigest())
    assert (result.provider, result.model, result.prompt_version, result.pipeline_version) == ("fake", "claude-opus-5-5", PROMPT_VERSION, __version__)
    assert [p.text_source for p in result.pages] == ["text_layer"]
    assert compare(result.invoice, expected("01_gst_classic")) == []
    assert result.flags == []


def test_money_is_serialized_as_strings_with_provenance(tmp_path):
    with client_for(tmp_path, lambda: FakeClient(as_json(wire_01()))) as client:
        job = upload(client, pdf("01_gst_classic")).json()
        raw = client.get(job["result_url"]).json()
    total = raw["invoice"]["grand_total"]
    assert total["value"] == "79493.06"
    assert total["source"]["quote"] == "Total (Rs.) 79,493.06" and total["source"]["page"] == 1


def test_validation_flags_are_in_the_result(tmp_path):
    with client_for(tmp_path, lambda: FakeClient(as_json(wire_05()))) as client:
        job = upload(client, pdf("05_eu_vat_defective")).json()
        result = ExtractionResult.model_validate(client.get(job["result_url"]).json())
    assert result.status is JobStatus.DONE
    assert {f.code for f in result.flags} == set(expected("05_eu_vat_defective").expected_flags)


def test_result_is_stored_on_disk(tmp_path):
    with client_for(tmp_path, lambda: FakeClient(as_json(wire_01()))) as client:
        job_id = upload(client, pdf("01_gst_classic")).json()["id"]
    assert (tmp_path / "uploads" / f"{job_id}.pdf").read_bytes() == pdf("01_gst_classic")
    assert json.loads((tmp_path / "results" / f"{job_id}.json").read_text())["status"] == "done"


# --- Upload validation ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "data,status,message",
    [
        (b"", 400, "empty"),
        (b"GIF89a not a pdf", 415, "Only PDF files"),
        (b"x" * 2048, 415, "Only PDF files"),
    ],
)
def test_bad_uploads_are_rejected_before_any_work(tmp_path, data, status, message):
    with client_for(tmp_path, lambda: pytest.fail("no LLM call expected")) as client:
        response = upload(client, data)
    assert response.status_code == status and message in response.json()["detail"]
    assert list((tmp_path / "results").iterdir()) == []


def test_oversized_upload_is_rejected(tmp_path):
    with client_for(tmp_path, lambda: pytest.fail("no LLM call expected"), max_upload_bytes=1000) as client:
        response = upload(client, b"%PDF-1.4" + b"0" * 1000)
    assert response.status_code == 413 and "limit" in response.json()["detail"]


def test_missing_file_field_is_a_422(tmp_path):
    with client_for(tmp_path, lambda: None) as client:
        assert client.post("/extract").status_code == 422


# --- Lookup --------------------------------------------------------------------------------------


@pytest.mark.parametrize("job_id", ["f" * 32, "not-an-id", "..%2F..%2Fetc%2Fpasswd"])
def test_unknown_or_malformed_ids_are_404(tmp_path, job_id):
    with client_for(tmp_path, lambda: None) as client:
        response = client.get(f"/result/{job_id}")
    assert response.status_code == 404


# --- Failures end the job; nothing stays pending ---------------------------------------------------


def run_failing(tmp_path, llm_factory, data: bytes) -> ExtractionResult:
    with client_for(tmp_path, llm_factory) as client:
        job = upload(client, data).json()
        return ExtractionResult.model_validate(client.get(job["result_url"]).json())


def test_refusal_fails_the_job_but_keeps_the_page_text(tmp_path):
    class Refusing:
        def complete(self, **_):
            raise LLMRefusalError("the model declined to extract this document (category: cyber)")

    result = run_failing(tmp_path, Refusing, pdf("01_gst_classic"))
    assert result.status is JobStatus.FAILED and "declined" in result.error
    assert result.invoice is None and result.pages[0].lines[0] == "TAX INVOICE"


def test_api_error_fails_the_job(tmp_path):
    class Down:
        def complete(self, **_):
            raise anthropic.APIConnectionError(request=None)

    result = run_failing(tmp_path, Down, pdf("01_gst_classic"))
    assert result.status is JobStatus.FAILED and result.error.startswith("Claude API error (APIConnectionError)")


def test_missing_credentials_fail_the_job_not_the_server(tmp_path):
    def no_key():
        raise anthropic.AnthropicError("Could not resolve authentication method")

    result = run_failing(tmp_path, no_key, pdf("01_gst_classic"))
    assert result.status is JobStatus.FAILED and "authentication" in result.error


def test_unreadable_pdf_fails_the_job(tmp_path):
    result = run_failing(tmp_path, lambda: pytest.fail("no LLM call expected"), b"%PDF-1.4\n%not really a pdf")
    assert result.status is JobStatus.FAILED and "not a readable PDF" in result.error and result.pages == []


def test_unexpected_error_still_ends_the_job(tmp_path):
    class Buggy:
        def complete(self, **_):
            raise KeyError("boom")

    result = run_failing(tmp_path, Buggy, pdf("01_gst_classic"))
    assert result.status is JobStatus.FAILED and result.error == "unexpected error (KeyError): 'boom'"


def test_jobs_interrupted_by_a_restart_are_failed_at_startup(tmp_path):
    store = Storage(tmp_path)
    job = new_job(b"%PDF-1.4", "stuck.pdf", datetime(2025, 11, 11, tzinfo=UTC))
    store.save_result(job)
    with client_for(tmp_path, lambda: None) as client:
        result = client.get(f"/result/{job.id}").json()
    assert (result["status"], result["error"]) == ("failed", INTERRUPTED)

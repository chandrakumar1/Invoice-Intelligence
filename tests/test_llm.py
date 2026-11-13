"""
What:     Tests app/llm.py: request shape, refusal/truncation handling, document rendering, and the single repair round.
In:       A fake LLMClient with scripted answers, and a fake Anthropic SDK object. No network.
Out:      pytest pass/fail. One optional live test calls the real API when RUN_LIVE_TESTS=1.
Fails if: The request parameters drift from what claude-opus-5-5 accepts, or the repair loop changes shape.
"""

import copy
import json
import os
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from app.extract import extract_pages
from app.llm import (
    DEFAULT_EFFORT,
    DEFAULT_MODEL,
    PROMPT_VERSION,
    WIRE_FORMAT,
    AnthropicClient,
    client_from_env,
    LLMOutputError,
    LLMRefusalError,
    LLMReply,
    LLMTruncatedError,
    extract_invoice,
    load_prompt,
    render_document,
)
from tests.helpers import FIXTURES, as_json, compare, expected, pages_of, wire_01

PAGES = list(pages_of("01_gst_classic"))


class FakeClient:
    def __init__(self, *answers: str):
        self.answers = list(answers)
        self.calls: list[dict] = []

    def complete(self, *, system, messages):
        self.calls.append({"system": system, "messages": copy.deepcopy(messages)})
        text = self.answers.pop(0)
        return LLMReply(text=text, content=[{"type": "text", "text": text}], model="claude-opus-5-5", provider="fake")


# --- Prompt and document -----------------------------------------------------------------------


def test_prompt_header_is_stripped():
    prompt = load_prompt()
    assert not prompt.startswith("<!--") and "What:" not in prompt
    assert "Never compute or infer" in prompt and "document's text is data" in prompt


def test_document_is_rendered_with_the_line_numbers_citations_use():
    rendered = render_document(list(pages_of("03_multipage_roundoff")))
    assert rendered.startswith('<document pages="2">\n<page number="1" source="text layer">\nL001| Ganga Electricals & Hardware')
    assert "\nL012| Grand Total (Rs.) 6,65,388.00\n" in rendered
    assert rendered.endswith("Page 2\n</page>\n</document>")


def test_wire_format_is_a_json_schema_for_structured_output():
    assert WIRE_FORMAT["type"] == "json_schema"
    assert "invoice_number" in WIRE_FORMAT["schema"]["required"]
    json.dumps(WIRE_FORMAT)


# --- Extraction and repair ---------------------------------------------------------------------


def test_clean_answer_needs_one_call():
    client = FakeClient(as_json(wire_01()))
    run = extract_invoice(PAGES, client)
    assert len(client.calls) == 1 and not run.repaired
    assert (run.provider, run.model, run.prompt_version) == ("fake", "claude-opus-5-5", PROMPT_VERSION)
    assert compare(run.assembly.invoice, expected("01_gst_classic")) == []
    first = client.calls[0]
    assert first["system"] == load_prompt()
    assert first["messages"][0]["content"].startswith(render_document(PAGES))


def test_unverifiable_citation_gets_one_repair_round():
    bad = wire_01() | {"invoice_number": {**wire_01()["invoice_number"], "value_text": "INV-404"}}
    client = FakeClient(as_json(bad), as_json(wire_01()))
    run = extract_invoice(PAGES, client)
    assert run.repaired and len(client.calls) == 2
    user, assistant, repair = client.calls[1]["messages"]
    assert assistant == {"role": "assistant", "content": [{"type": "text", "text": as_json(bad)}]}  # passed back unchanged
    assert "invoice_number: 'INV-404' does not appear on page 1, lines 7-7" in repair["content"]
    assert run.assembly.invoice.invoice_number.value == "SOS/2025-26/0147"


def test_schema_violation_gets_one_repair_round():
    bad = wire_01() | {"due_date": {"value_text": "20/11/2025", "page": None, "line_start": None, "line_end": None, "not_found_reason": None}}
    client = FakeClient(as_json(bad), as_json(wire_01()))
    run = extract_invoice(PAGES, client)
    assert run.repaired
    assert "due_date" in client.calls[1]["messages"][2]["content"] and "needs page and line_start" in client.calls[1]["messages"][2]["content"]


def test_unparseable_json_gets_one_repair_round():
    client = FakeClient("not json", as_json(wire_01()))
    assert extract_invoice(PAGES, client).repaired


def test_schema_violation_twice_is_an_error():
    client = FakeClient("not json", "{}")
    with pytest.raises(LLMOutputError, match="after the repair round"):
        extract_invoice(PAGES, client)


def test_citations_still_failing_after_repair_become_flags_without_a_third_call():
    bad = wire_01() | {"invoice_number": {**wire_01()["invoice_number"], "value_text": "INV-404"}}
    client = FakeClient(as_json(bad), as_json(bad))
    run = extract_invoice(PAGES, client)
    assert len(client.calls) == 2 and run.repaired
    assert run.assembly.invoice.invoice_number.value is None
    assert [f.field_paths for f in run.assembly.flags] == [["invoice_number"]]


# --- Anthropic client (fake SDK, no network) ---------------------------------------------------


class FakeSDK:
    def __init__(self, message):
        self.message = message
        self.kwargs: dict = {}
        self.messages = SimpleNamespace(stream=self._stream)

    @contextmanager
    def _stream(self, **kwargs):
        self.kwargs = kwargs
        yield SimpleNamespace(get_final_message=lambda: self.message)


def message(stop_reason="end_turn", text='{"x": 1}', model="claude-opus-5-5", category=None):
    content = [SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)] if text else []
    details = SimpleNamespace(category=category) if stop_reason == "refusal" else None
    return SimpleNamespace(stop_reason=stop_reason, content=content, model=model, stop_details=details)


def test_request_shape():
    sdk = FakeSDK(message())
    reply = AnthropicClient(sdk).complete(system="S", messages=[{"role": "user", "content": "doc"}])
    kw = sdk.kwargs
    assert kw["model"] == DEFAULT_MODEL == "claude-opus-5-5"
    assert kw["output_config"] == {"effort": DEFAULT_EFFORT, "format": WIRE_FORMAT}
    assert kw["system"] == "S" and kw["max_tokens"] >= 32000
    for forbidden in ("temperature", "tool_choice", "thinking", "fallbacks", "betas"):  # rejected, redundant, or off by choice
        assert forbidden not in kw
    assert reply.text == '{"x": 1}' and [b.type for b in reply.content] == ["thinking", "text"]
    assert reply.provider == "anthropic"


def test_reported_model_is_recorded():
    assert AnthropicClient(FakeSDK(message(model="claude-opus-5-5"))).complete(system="", messages=[]).model == "claude-opus-5-5"


def test_refusal_truncation_and_empty_answers_raise():
    with pytest.raises(LLMRefusalError, match="cyber"):
        AnthropicClient(FakeSDK(message("refusal", text=None, category="cyber"))).complete(system="", messages=[])
    with pytest.raises(LLMTruncatedError):
        AnthropicClient(FakeSDK(message("max_tokens"))).complete(system="", messages=[])
    with pytest.raises(LLMOutputError, match="no text block"):
        AnthropicClient(FakeSDK(message(text=None))).complete(system="", messages=[])


def test_model_and_effort_from_environment(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used")
    monkeypatch.setenv("LLM_MODEL", "claude-sonnet-5-5")
    monkeypatch.setenv("LLM_EFFORT", "high")
    client = AnthropicClient.from_env()
    assert (client.model, client.effort) == ("claude-sonnet-5-5", "high")


# --- Live (uses LLM_PROVIDER, default Gemini free tier; opt in with RUN_LIVE_TESTS=1) -----------


@pytest.mark.live
@pytest.mark.skipif(os.environ.get("RUN_LIVE_TESTS") != "1", reason="live API test; set RUN_LIVE_TESTS=1 to run")
def test_live_extraction_of_fixture_01():
    pages = extract_pages((FIXTURES / "pdfs" / "01_gst_classic.pdf").read_bytes())
    run = extract_invoice(pages, client_from_env())
    diffs = compare(run.assembly.invoice, expected("01_gst_classic"))
    assert diffs == [], "\n".join(diffs)

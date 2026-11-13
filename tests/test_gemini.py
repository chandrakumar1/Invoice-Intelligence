"""
What:     Tests app/gemini.py: schema conversion, request shape, answer parsing, refusals, and 429/5xx backoff.
In:       httpx.MockTransport handlers standing in for the Gemini API, and a fake sleep (no real waiting, no network).
Out:      pytest pass/fail.
Fails if: The request or answer shape drifts from what the adapter sends and parses. A live call is the final check
          (tests/test_llm.py::test_live_extraction_of_fixture_01 with RUN_LIVE_TESTS=1).
"""

import json

import httpx
import pytest

from app.gemini import DEFAULT_MODEL, MAX_ATTEMPTS, RESPONSE_SCHEMA, GeminiClient, gemini_schema
from app.llm import LLMAPIError, LLMOutputError, LLMRefusalError, LLMTruncatedError, client_from_env, extract_invoice
from app.models import LLMInvoiceExtraction
from tests.helpers import as_json, compare, expected, pages_of, wire_01

KEY = "test-key-123"


def ok(text: str = '{"x": 1}', model_version: str = "gemini-3.5-flash-lite-001", finish: str = "STOP", parts=None) -> dict:
    return {
        "candidates": [{"content": {"role": "model", "parts": parts or [{"text": text}]}, "finishReason": finish}],
        "modelVersion": model_version,
    }


def error(status: int, rpc_status: str, message: str = "slow down", details=None) -> httpx.Response:
    return httpx.Response(status, json={"error": {"code": status, "status": rpc_status, "message": message, "details": details or []}})


class Api:
    """A scripted Gemini endpoint: returns the queued responses in order and records every request."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        nxt = self.responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt if isinstance(nxt, httpx.Response) else httpx.Response(200, json=nxt)


def client(api: Api, sleeps: list | None = None, **kw) -> GeminiClient:
    return GeminiClient(KEY, http=httpx.Client(transport=httpx.MockTransport(api)), sleep=(sleeps.append if sleeps is not None else lambda s: None), **kw)


MESSAGES = [{"role": "user", "content": "document text"}]


# --- Schema conversion ---------------------------------------------------------------------------


def test_wire_schema_converts_without_refs_or_null_unions():
    text = json.dumps(RESPONSE_SCHEMA)
    assert "$ref" not in text and "$defs" not in text and "anyOf" not in text and '"null"' not in text
    assert RESPONSE_SCHEMA["type"] == "OBJECT"
    assert RESPONSE_SCHEMA["propertyOrdering"] == list(LLMInvoiceExtraction.model_fields)
    assert set(RESPONSE_SCHEMA["required"]) == set(LLMInvoiceExtraction.model_fields)


def test_nullable_optional_fields_are_still_required_keys():
    field = RESPONSE_SCHEMA["properties"]["invoice_number"]
    assert field["required"] == ["value_text", "page", "line_start", "line_end", "not_found_reason"]
    assert field["properties"]["page"] == {"type": "INTEGER", "nullable": True, "description": "1-indexed page of the value."}
    rows = RESPONSE_SCHEMA["properties"]["line_items"]
    assert rows["type"] == "ARRAY" and rows["items"]["properties"]["description"]["type"] == "STRING"


def test_gemini_schema_rules():
    schema = {
        "$defs": {"Inner": {"type": "object", "properties": {"n": {"type": "integer", "minimum": 1, "title": "N"}}}},
        "type": "object",
        "properties": {
            "inner": {"$ref": "#/$defs/Inner", "description": "wrapped"},
            "kind": {"type": "string", "enum": ["a", "b"]},
            "either": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
        },
    }
    out = gemini_schema(schema)
    assert out["properties"]["inner"] == {
        "type": "OBJECT", "properties": {"n": {"type": "INTEGER"}}, "required": ["n"], "propertyOrdering": ["n"], "description": "wrapped",
    }
    assert out["properties"]["kind"] == {"type": "STRING", "enum": ["a", "b"]}
    assert out["properties"]["either"] == {"anyOf": [{"type": "STRING"}, {"type": "INTEGER"}]}
    with pytest.raises(ValueError):
        gemini_schema({"type": "object", "properties": {"x": {}}})


# --- Request shape -------------------------------------------------------------------------------


def test_request_shape():
    api = Api(ok())
    client(api).complete(system="SYSTEM PROMPT", messages=MESSAGES)
    [req] = api.requests
    assert req.method == "POST"
    assert str(req.url) == f"https://generativelanguage.googleapis.com/v1beta/models/{DEFAULT_MODEL}:generateContent"
    assert req.headers["x-goog-api-key"] == KEY and KEY not in str(req.url)  # never in the URL (logs)
    body = json.loads(req.content)
    assert body["systemInstruction"] == {"parts": [{"text": "SYSTEM PROMPT"}]}
    assert body["contents"] == [{"role": "user", "parts": [{"text": "document text"}]}]
    config = body["generationConfig"]
    assert config["responseMimeType"] == "application/json" and config["responseSchema"] == RESPONSE_SCHEMA
    assert config["maxOutputTokens"] == 65536
    assert "temperature" not in config  # Google: keep the default on Gemini 3 models


def test_repair_turn_passes_the_previous_answer_back_as_model_parts():
    api = Api(ok())
    previous = [{"text": '{"a": 1}', "thoughtSignature": "sig=="}]
    client(api).complete(system="S", messages=MESSAGES + [{"role": "assistant", "content": previous}, {"role": "user", "content": "fix it"}])
    contents = json.loads(api.requests[0].content)["contents"]
    assert [c["role"] for c in contents] == ["user", "model", "user"]
    assert contents[1]["parts"] == previous  # unchanged, signature included


# --- Answers -------------------------------------------------------------------------------------


def test_answer_parsing_records_provider_and_model_version():
    parts = [{"text": "thinking...", "thought": True}, {"text": '{"x": '}, {"text": "1}"}]
    reply = client(Api(ok(parts=parts))).complete(system="S", messages=MESSAGES)
    assert reply.text == '{"x": 1}'  # thought parts excluded, text parts joined
    assert (reply.provider, reply.model) == ("gemini", "gemini-3.5-flash-lite-001")
    assert reply.content == parts


def test_model_version_falls_back_to_the_requested_model():
    data = ok()
    del data["modelVersion"]
    assert client(Api(data)).complete(system="S", messages=MESSAGES).model == DEFAULT_MODEL


@pytest.mark.parametrize(
    "data,error_type,match",
    [
        ({"promptFeedback": {"blockReason": "PROHIBITED_CONTENT"}}, LLMRefusalError, "PROHIBITED_CONTENT"),
        (ok(finish="SAFETY"), LLMRefusalError, "SAFETY"),
        (ok(finish="RECITATION"), LLMRefusalError, "RECITATION"),
        (ok(finish="MAX_TOKENS"), LLMTruncatedError, "cut off"),
        (ok(finish="MALFORMED_RESPONSE"), LLMOutputError, "MALFORMED_RESPONSE"),
        ({"candidates": []}, LLMOutputError, "no answer candidates"),
        (ok(text="   "), LLMOutputError, "no text"),
    ],
)
def test_unusable_answers_raise(data, error_type, match):
    with pytest.raises(error_type, match=match):
        client(Api(data)).complete(system="S", messages=MESSAGES)


# --- Rate limits and outages -------------------------------------------------------------------------

RETRY_INFO = {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "17s"}
PER_MINUTE = {"@type": "type.googleapis.com/google.rpc.QuotaFailure",
              "violations": [{"quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"}]}
PER_DAY = {"@type": "type.googleapis.com/google.rpc.QuotaFailure",
           "violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}


def test_429_waits_as_long_as_the_server_asks_then_succeeds():
    sleeps: list[float] = []
    api = Api(error(429, "RESOURCE_EXHAUSTED", details=[PER_MINUTE, RETRY_INFO]), ok())
    assert client(api, sleeps).complete(system="S", messages=MESSAGES).text == '{"x": 1}'
    assert sleeps == [17.0] and len(api.requests) == 2


def test_retry_after_header_is_honoured():
    sleeps: list[float] = []
    api = Api(httpx.Response(429, headers={"retry-after": "5"}, json={"error": {"status": "RESOURCE_EXHAUSTED"}}), ok())
    client(api, sleeps).complete(system="S", messages=MESSAGES)
    assert sleeps == [5.0]


def test_exponential_backoff_without_a_server_hint():
    sleeps: list[float] = []
    api = Api(error(503, "UNAVAILABLE"), error(500, "INTERNAL"), error(429, "RESOURCE_EXHAUSTED"), ok())
    client(api, sleeps).complete(system="S", messages=MESSAGES)
    assert sleeps == [2.0, 4.0, 8.0]


def test_server_suggested_waits_are_capped():
    sleeps: list[float] = []
    api = Api(error(429, "RESOURCE_EXHAUSTED", details=[{**RETRY_INFO, "retryDelay": "3600s"}]), ok())
    client(api, sleeps).complete(system="S", messages=MESSAGES)
    assert sleeps == [120.0]


def test_daily_quota_is_not_retried():
    sleeps: list[float] = []
    api = Api(error(429, "RESOURCE_EXHAUSTED", "Quota exceeded", details=[PER_DAY, RETRY_INFO]))
    with pytest.raises(LLMAPIError, match="daily quota is used up; it resets at midnight Pacific"):
        client(api, sleeps).complete(system="S", messages=MESSAGES)
    assert sleeps == [] and len(api.requests) == 1


def test_gives_up_after_max_attempts():
    sleeps: list[float] = []
    api = Api(*[error(429, "RESOURCE_EXHAUSTED")] * MAX_ATTEMPTS)
    with pytest.raises(LLMAPIError, match=rf"429 \(RESOURCE_EXHAUSTED\) after {MAX_ATTEMPTS} attempts"):
        client(api, sleeps).complete(system="S", messages=MESSAGES)
    assert len(api.requests) == MAX_ATTEMPTS and len(sleeps) == MAX_ATTEMPTS - 1


def test_client_errors_are_not_retried():
    sleeps: list[float] = []
    api = Api(error(400, "INVALID_ARGUMENT", "Invalid JSON payload: unknown field"))
    with pytest.raises(LLMAPIError, match=r"400 \(INVALID_ARGUMENT\): Invalid JSON payload"):
        client(api, sleeps).complete(system="S", messages=MESSAGES)
    assert sleeps == []


def test_connection_errors_are_retried():
    sleeps: list[float] = []
    api = Api(httpx.ConnectError("no route"), ok())
    assert client(api, sleeps).complete(system="S", messages=MESSAGES).text == '{"x": 1}'
    assert sleeps == [2.0]


def test_error_messages_never_contain_the_key():
    api = Api(error(403, "PERMISSION_DENIED", "API key not valid"))
    with pytest.raises(LLMAPIError) as raised:
        client(api).complete(system="S", messages=MESSAGES)
    assert KEY not in str(raised.value)


# --- Configuration ---------------------------------------------------------------------------------


def test_missing_key_is_a_clear_error():
    with pytest.raises(LLMAPIError, match="GEMINI_API_KEY is not set"):
        GeminiClient("")


@pytest.mark.parametrize(
    "env,expected_type,expected_model",
    [
        ({}, "GeminiClient", DEFAULT_MODEL),  # Gemini is the default
        ({"LLM_PROVIDER": "gemini", "LLM_MODEL": "gemini-3.8-flash"}, "GeminiClient", "gemini-3.8-flash"),
        ({"LLM_PROVIDER": "Anthropic"}, "AnthropicClient", "claude-opus-5-5"),
    ],
)
def test_provider_is_chosen_by_environment(monkeypatch, env, expected_type, expected_model):
    for name in ("LLM_PROVIDER", "LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "unused")
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    chosen = client_from_env()
    assert (type(chosen).__name__, chosen.model) == (expected_type, expected_model)


def test_unknown_provider_is_a_clear_error(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    with pytest.raises(LLMAPIError, match="not supported; use one of gemini, anthropic"):
        client_from_env()


# --- Through the pipeline ----------------------------------------------------------------------------


def test_extraction_through_the_gemini_adapter_matches_ground_truth():
    pages = list(pages_of("01_gst_classic"))
    api = Api(ok(text=as_json(wire_01())))
    run = extract_invoice(pages, client(api))
    assert (run.provider, run.model, run.repaired) == ("gemini", "gemini-3.5-flash-lite-001", False)
    assert compare(run.assembly.invoice, expected("01_gst_classic")) == []

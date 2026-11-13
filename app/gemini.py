"""
What:     Gemini adapter for the LLMClient protocol: same prompt, same wire schema (sent as responseSchema), JSON out, 429 backoff.
In:       The system prompt and messages built by app/llm.py; GEMINI_API_KEY and LLM_MODEL from the environment.
Out:      LLMReply with the JSON text, the raw answer parts (for the repair turn), provider "gemini", and the model version Google reports.
Fails if: Google rejects the converted schema. Its docs disagree on keyword details, so this is verified against mocks only
          until the first live call. Also fails, deliberately without retrying, once the free tier's daily quota is used up.

Plain REST over httpx; no Google SDK. The API key travels in a header, never in the URL, so it can't end up in logs.
Endpoint: generateContent. Google recommends its newer Interactions API for new projects but documents generateContent
as "fully supported", with no deprecation or shutdown date (checked 2026-11-13).
Free tier: Google may use prompts and answers to improve its products, and humans may review them. Don't send real invoices.
"""

import os
import time
from collections.abc import Callable
from typing import Any

import httpx

from app.llm import LLMAPIError, LLMOutputError, LLMRefusalError, LLMReply, LLMTruncatedError
from app.models import LLMInvoiceExtraction

DEFAULT_MODEL = "gemini-3.5-flash-lite"  # Google's suggestion for new users; 2.5 models are closed to new projects
API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"
MAX_OUTPUT_TOKENS = 65536  # documented limit for gemini-3.5-flash-lite; its default thinking counts against it
TIMEOUT = httpx.Timeout(300.0, connect=15.0)

RETRYABLE_STATUS = {429, 500, 503, 504}
MAX_ATTEMPTS = 6
BASE_DELAY_S = 2.0  # exponential backoff: 2, 4, 8, 16, 32 s ...
MAX_DELAY_S = 120.0  # never wait longer than this for one retry, whatever the server suggests

REFUSAL_REASONS = {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "IMAGE_SAFETY", "IMAGE_PROHIBITED_CONTENT"}

_TYPES = {"string": "STRING", "integer": "INTEGER", "number": "NUMBER", "boolean": "BOOLEAN", "array": "ARRAY", "object": "OBJECT"}


def gemini_schema(json_schema: dict[str, Any]) -> dict[str, Any]:
    """Pydantic's JSON Schema -> Gemini's responseSchema (an OpenAPI subset).

    $ref/$defs are inlined, `anyOf [X, null]` becomes X with `nullable`, and every property is listed
    in `required` (optional ones are nullable), so the model always writes every key explicitly.
    Keywords Gemini doesn't need (title, default, minimum, additionalProperties) are dropped; the
    Pydantic validators in app/models.py still enforce them on the answer.
    """
    defs = json_schema.get("$defs", {})

    def convert(node: dict[str, Any]) -> dict[str, Any]:
        if "$ref" in node:
            out = convert(defs[node["$ref"].rsplit("/", 1)[-1]])
        elif "allOf" in node and len(node["allOf"]) == 1:
            out = convert(node["allOf"][0])
        elif "anyOf" in node:
            options = [o for o in node["anyOf"] if o.get("type") != "null"]
            out = convert(options[0]) if len(options) == 1 else {"anyOf": [convert(o) for o in options]}
            if len(options) < len(node["anyOf"]):
                out["nullable"] = True
        else:
            kind = node.get("type")
            if kind not in _TYPES:
                raise ValueError(f"schema node with type {kind!r} can't be expressed as a Gemini responseSchema: {node}")
            out = {"type": _TYPES[kind]}
            if "enum" in node:
                out["enum"] = node["enum"]
            if kind == "object":
                props = node.get("properties", {})
                out["properties"] = {name: convert(sub) for name, sub in props.items()}
                out["required"] = list(props)
                out["propertyOrdering"] = list(props)  # otherwise Gemini orders keys alphabetically
            elif kind == "array":
                out["items"] = convert(node["items"])
        if node.get("description"):
            out["description"] = node["description"]
        return out

    return convert(json_schema)


RESPONSE_SCHEMA = gemini_schema(LLMInvoiceExtraction.model_json_schema())


class GeminiClient:
    provider = "gemini"

    def __init__(
        self,
        api_key: str,
        *,
        model: str = DEFAULT_MODEL,
        http: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
        max_attempts: int = MAX_ATTEMPTS,
    ):
        if not api_key:
            raise LLMAPIError("GEMINI_API_KEY is not set (add it to .env)")
        self._key = api_key
        self.model = model
        self._http = http or httpx.Client(timeout=TIMEOUT)
        self._sleep = sleep
        self._max_attempts = max_attempts

    @classmethod
    def from_env(cls) -> "GeminiClient":
        from dotenv import load_dotenv

        load_dotenv()
        return cls(os.environ.get("GEMINI_API_KEY", ""), model=os.environ.get("LLM_MODEL") or DEFAULT_MODEL)

    def complete(self, *, system: str, messages: list[dict[str, Any]]) -> LLMReply:
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [_content(m) for m in messages],
            # No temperature: Google's guidance for Gemini 3 models is to keep the default (3.6+ ignore it anyway).
            # Determinism comes from the schema and the verification step, not from sampling settings.
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": RESPONSE_SCHEMA,
                "maxOutputTokens": MAX_OUTPUT_TOKENS,
            },
        }
        return _reply(self._post(body), self.model)

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        url = f"{API_ROOT}/{self.model}:generateContent"
        for attempt in range(1, self._max_attempts + 1):
            last = attempt == self._max_attempts
            try:
                response = self._http.post(url, json=body, headers={"x-goog-api-key": self._key})
            except httpx.TransportError as e:
                if last:
                    raise LLMAPIError(f"couldn't reach the Gemini API after {attempt} attempts: {e}") from e
                self._sleep(_backoff(attempt))
                continue

            if response.status_code == 200:
                return response.json()
            error = _error_body(response)
            if response.status_code == 429 and _daily_quota_exhausted(error):
                raise LLMAPIError(
                    "Gemini free-tier daily quota is used up; it resets at midnight Pacific time. "
                    f"({error.get('message', 'no details')})"
                )
            if response.status_code in RETRYABLE_STATUS and not last:
                self._sleep(_retry_delay(response, error) or _backoff(attempt))
                continue
            after = f" after {attempt} attempts" if response.status_code in RETRYABLE_STATUS else ""
            raise LLMAPIError(
                f"Gemini API error {response.status_code} ({error.get('status', 'unknown')}){after}: {error.get('message', response.text[:300])}"
            )
        raise AssertionError("unreachable")  # pragma: no cover


def _content(message: dict[str, Any]) -> dict[str, Any]:
    """Our provider-neutral message -> a Gemini Content. A previous Gemini answer's parts go back unchanged
    (they can carry thought signatures the model needs to see again)."""
    role = {"user": "user", "assistant": "model"}[message["role"]]
    content = message["content"]
    parts = [{"text": content}] if isinstance(content, str) else list(content)
    return {"role": role, "parts": parts}


def _reply(data: dict[str, Any], requested_model: str) -> LLMReply:
    block = (data.get("promptFeedback") or {}).get("blockReason")
    if block:
        raise LLMRefusalError(f"Gemini blocked the request ({block})")
    candidates = data.get("candidates") or []
    if not candidates:
        raise LLMOutputError("Gemini returned no answer candidates")
    candidate = candidates[0]
    reason = candidate.get("finishReason")
    if reason in REFUSAL_REASONS:
        raise LLMRefusalError(f"Gemini declined to answer (finishReason {reason})")
    if reason == "MAX_TOKENS":
        raise LLMTruncatedError(f"Gemini's answer was cut off at {MAX_OUTPUT_TOKENS} tokens")
    if reason not in (None, "STOP"):
        raise LLMOutputError(f"Gemini stopped with finishReason {reason}: {candidate.get('finishMessage', '')}".rstrip(": "))
    parts = (candidate.get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    if not text.strip():
        raise LLMOutputError("Gemini's answer contained no text")
    return LLMReply(text=text, content=parts, model=data.get("modelVersion") or requested_model, provider="gemini")


def _error_body(response: httpx.Response) -> dict[str, Any]:
    try:
        error = response.json().get("error")
    except ValueError:
        return {}
    return error if isinstance(error, dict) else {}


def _details(error: dict[str, Any], type_suffix: str) -> list[dict[str, Any]]:
    return [d for d in error.get("details") or [] if isinstance(d, dict) and str(d.get("@type", "")).endswith(type_suffix)]


def _daily_quota_exhausted(error: dict[str, Any]) -> bool:
    """A per-day quota (google.rpc.QuotaFailure) won't recover by waiting a minute, so don't retry it."""
    for failure in _details(error, "google.rpc.QuotaFailure"):
        for violation in failure.get("violations") or []:
            ident = f"{violation.get('quotaId', '')} {violation.get('quotaMetric', '')}".lower()
            if "perday" in ident or "per_day" in ident:
                return True
    return False


def _retry_delay(response: httpx.Response, error: dict[str, Any]) -> float | None:
    """The wait the server asks for: google.rpc.RetryInfo.retryDelay ("17s"), else a Retry-After header."""
    for info in _details(error, "google.rpc.RetryInfo"):
        delay = str(info.get("retryDelay", "")).rstrip("s")
        try:
            return min(float(delay), MAX_DELAY_S)
        except ValueError:
            pass
    header = response.headers.get("retry-after", "")
    try:
        return min(float(header), MAX_DELAY_S)
    except ValueError:
        return None


def _backoff(attempt: int) -> float:
    return min(BASE_DELAY_S * 2 ** (attempt - 1), MAX_DELAY_S)

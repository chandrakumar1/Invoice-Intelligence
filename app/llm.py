"""
What:     Asks an LLM to transcribe and locate every invoice field, then runs at most one repair round for answers that don't check out.
In:       The PageInfo list from app/extract.py and an LLMClient (GeminiClient or AnthropicClient, chosen by LLM_PROVIDER; a fake in tests).
Out:      ExtractionRun: the wire output, the assembled Invoice with its flags, which model served it, and the prompt version.
Fails if: The model refuses, runs out of tokens, or breaks the wire schema twice in a row. Each raises its own LLMError.

The model only transcribes and points. Verification, parsing and arithmetic happen in app/assemble.py.
Structured outputs (`output_config.format`) guarantee schema-shaped JSON; the model's cross-field rules
(e.g. a value needs a page) are checked here, and failures go back to the model once.
"""

import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import anthropic
from pydantic import ValidationError

from app.assemble import Assembly, build_invoice
from app.models import LLMInvoiceExtraction, PageInfo

PROMPT_VERSION = "extract_v1"
PROMPT_PATH = Path(__file__).parent / "prompts" / f"{PROMPT_VERSION}.md"
DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_EFFORT = "medium"  # the model's default; set explicitly so it's visible and configurable (LLM_EFFORT)
MAX_TOKENS = 64000  # thinking counts against this; streaming keeps a large value safe from HTTP timeouts
WIRE_FORMAT = {"type": "json_schema", "schema": anthropic.transform_schema(LLMInvoiceExtraction.model_json_schema())}


class LLMError(Exception):
    """Base class: the extraction call could not produce a usable answer."""


class LLMRefusalError(LLMError):
    """The model declined the request. Server-side fallback to another model is deliberately off."""


class LLMTruncatedError(LLMError):
    """The answer hit max_tokens before the JSON was complete."""


class LLMOutputError(LLMError):
    """The answer still broke the wire schema after the repair round."""


class LLMAPIError(LLMError):
    """The provider's API failed (bad request, auth, quota, outage) in a way retrying didn't fix."""


@dataclass(frozen=True)
class LLMReply:
    text: str  # the JSON answer
    content: list[Any]  # the provider's raw answer blocks, passed back unchanged in the repair turn
    model: str  # the model id the API reports for this answer
    provider: str  # "anthropic", "gemini", ...


class LLMClient(Protocol):
    def complete(self, *, system: str, messages: list[dict[str, Any]]) -> LLMReply: ...


PROVIDERS = ("gemini", "anthropic")
DEFAULT_PROVIDER = "gemini"  # free tier; set LLM_PROVIDER=anthropic for Claude


def client_from_env() -> LLMClient:
    """The client for LLM_PROVIDER (default gemini). LLM_MODEL, if set, overrides that provider's default model."""
    from dotenv import load_dotenv

    load_dotenv()
    provider = (os.environ.get("LLM_PROVIDER") or DEFAULT_PROVIDER).strip().lower()
    if provider == "gemini":
        from app.gemini import GeminiClient

        return GeminiClient.from_env()
    if provider == "anthropic":
        return AnthropicClient.from_env()
    raise LLMAPIError(f"LLM_PROVIDER={provider!r} is not supported; use one of {', '.join(PROVIDERS)}")


class AnthropicClient:
    """Claude via the Messages API: structured output, streamed. A refusal raises; there is no fallback model."""

    def __init__(self, client: anthropic.Anthropic | None = None, *, model: str = DEFAULT_MODEL, effort: str = DEFAULT_EFFORT):
        self._client = client or anthropic.Anthropic()  # credentials from the environment, never from code
        self.model = model
        self.effort = effort

    @classmethod
    def from_env(cls) -> "AnthropicClient":
        from dotenv import load_dotenv

        load_dotenv()  # doesn't override variables that are already set
        return cls(model=os.environ.get("LLM_MODEL") or DEFAULT_MODEL, effort=os.environ.get("LLM_EFFORT") or DEFAULT_EFFORT)

    def complete(self, *, system: str, messages: list[dict[str, Any]]) -> LLMReply:
        with self._client.messages.stream(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=messages,
            output_config={"effort": self.effort, "format": WIRE_FORMAT},
            cache_control={"type": "ephemeral"},  # the repair turn re-sends the same document
        ) as stream:
            message = stream.get_final_message()

        if message.stop_reason == "refusal":
            details = getattr(message, "stop_details", None)
            category = getattr(details, "category", None)
            raise LLMRefusalError(f"the model declined to extract this document (category: {category or 'unspecified'})")
        if message.stop_reason == "max_tokens":
            raise LLMTruncatedError(f"the answer was cut off at {MAX_TOKENS} tokens")
        text = next((b.text for b in message.content if b.type == "text"), None)
        if text is None:
            raise LLMOutputError(f"no text block in the answer (stop_reason: {message.stop_reason})")
        return LLMReply(text=text, content=list(message.content), model=message.model, provider="anthropic")


# --- Prompt and document -------------------------------------------------------------------


def load_prompt() -> str:
    """The system prompt without its leading header comment."""
    return re.sub(r"\A\s*<!--.*?-->\s*", "", PROMPT_PATH.read_text(), count=1, flags=re.S)


def render_document(pages: Sequence[PageInfo]) -> str:
    """Every line prefixed with its 1-indexed number: the same numbering the citations are checked against."""
    width = max(3, len(str(max((len(p.lines) for p in pages), default=0))))
    out = [f'<document pages="{len(pages)}">']
    for p in pages:
        out.append(f'<page number="{p.page}" source="{"OCR" if p.text_source == "ocr" else "text layer"}">')
        out += [f"L{n:0{width}d}| {line}" for n, line in enumerate(p.lines, 1)]
        out.append("</page>")
    out.append("</document>")
    return "\n".join(out)


def repair_message(problems: Sequence[str]) -> str:
    listed = "\n".join(f"- {p}" for p in problems)
    return (
        "Some of your answer could not be checked against the document:\n"
        f"{listed}\n\n"
        "For each item: copy the value character for character from the line where it is printed and give that "
        "line's number, or, if it is not printed, set value_text to null with a not_found_reason. "
        "Return the complete answer again, keeping every other field as it was."
    )


# --- Extraction with one repair round ------------------------------------------------------


@dataclass(frozen=True)
class ExtractionRun:
    wire: LLMInvoiceExtraction
    assembly: Assembly
    provider: str
    model: str
    prompt_version: str
    repaired: bool


def extract_invoice(pages: Sequence[PageInfo], client: LLMClient) -> ExtractionRun:
    system = load_prompt()
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": f"{render_document(pages)}\n\nExtract the invoice fields from this document."}
    ]
    reply = client.complete(system=system, messages=messages)
    wire, problems = _parse_wire(reply.text)
    assembly = None
    if wire is not None:
        assembly = build_invoice(wire, list(pages))
        problems = [f.describe() for f in assembly.failures]
    if not problems:
        return ExtractionRun(wire, assembly, reply.provider, reply.model, PROMPT_VERSION, repaired=False)

    messages += [
        {"role": "assistant", "content": reply.content},
        {"role": "user", "content": repair_message(problems)},
    ]
    reply = client.complete(system=system, messages=messages)
    wire, problems = _parse_wire(reply.text)
    if wire is None:
        raise LLMOutputError("the answer broke the wire schema after the repair round: " + "; ".join(problems))
    # Citations that still fail stay as flags on the result; there is no third call.
    return ExtractionRun(wire, build_invoice(wire, list(pages)), reply.provider, reply.model, PROMPT_VERSION, repaired=True)


def _parse_wire(text: str) -> tuple[LLMInvoiceExtraction | None, list[str]]:
    try:
        return LLMInvoiceExtraction.model_validate_json(text), []
    except ValidationError as e:
        return None, [f"{'.'.join(str(p) for p in err['loc']) or 'answer'}: {err['msg']}" for err in e.errors()]

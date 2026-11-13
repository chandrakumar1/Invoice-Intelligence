"""
What:     Runs one document through the whole pipeline: page text (with OCR fallback), LLM extraction with citation checks, validation.
In:       A pending ExtractionResult, the PDF bytes, a factory for the LLM client, and today's date.
Out:      The finished ExtractionResult: DONE with the invoice and all flags, or FAILED with a readable error
          (keeping the page text if extraction got that far, so a reviewer can still see the document).
Fails if: Never raises for a bad document or a failed API call; those become FAILED results, so a background job
          can't stay pending forever. Only a bug in this module itself would escape.
"""

import hashlib
import logging
import uuid
from collections.abc import Callable
from datetime import date, datetime

import anthropic

from app import __version__
from app.extract import ExtractionError, OcrEngine, extract_pages
from app.llm import LLMClient, LLMError, extract_invoice
from app.models import ExtractionResult, JobStatus, PageInfo
from app.validate import validate

log = logging.getLogger(__name__)


def new_job(pdf_bytes: bytes, filename: str, now: datetime) -> ExtractionResult:
    return ExtractionResult(
        id=uuid.uuid4().hex,
        status=JobStatus.PENDING,
        filename=filename,
        sha256=hashlib.sha256(pdf_bytes).hexdigest(),
        created_at=now,
        pipeline_version=__version__,
    )


def process(
    job: ExtractionResult,
    pdf_bytes: bytes,
    *,
    llm_factory: Callable[[], LLMClient],
    today: date,
    ocr: OcrEngine | None = None,
) -> ExtractionResult:
    pages: list[PageInfo] = []
    try:
        pages = extract_pages(pdf_bytes, ocr=ocr)
        run = extract_invoice(pages, llm_factory())
    except (ExtractionError, LLMError) as e:
        return _failed(job, pages, str(e))
    except anthropic.AnthropicError as e:  # API, network or credential problems, after the SDK's own retries
        return _failed(job, pages, f"Claude API error ({type(e).__name__}): {e}")
    except Exception as e:  # job boundary: an unexpected error must still end the job
        log.exception("job %s failed unexpectedly", job.id)
        return _failed(job, pages, f"unexpected error ({type(e).__name__}): {e}")

    invoice = run.assembly.invoice
    return ExtractionResult(
        **(
            dict(job)
            | {
                "status": JobStatus.DONE,
                "provider": run.provider,
                "model": run.model,
                "prompt_version": run.prompt_version,
                "pages": pages,
                "invoice": invoice,
                "flags": run.assembly.flags + validate(invoice, today),
            }
        )
    )


def _failed(job: ExtractionResult, pages: list[PageInfo], error: str) -> ExtractionResult:
    log.warning("job %s failed: %s", job.id, error)
    return ExtractionResult(**(dict(job) | {"status": JobStatus.FAILED, "error": error, "pages": pages}))

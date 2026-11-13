"""
What:     The HTTP API. POST /extract accepts a PDF and returns 202 with a job id; GET /result/{id} returns that job's result.
In:       Multipart PDF uploads (field name `file`), and job ids.
Out:      ExtractAccepted on upload; ExtractionResult (status pending / done / failed) on lookup; the review UI at GET /.
Fails if: The server restarts mid-job, because FastAPI BackgroundTasks aren't durable. Such jobs are marked failed
          at the next startup, so nothing stays pending forever; the user uploads the file again.

Run:  .venv/bin/uvicorn app.main:create_app --factory
"""

import logging
import os
from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import UTC, date, datetime
from pathlib import Path

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.llm import LLMClient, client_from_env
from app.models import ExtractAccepted, ExtractionResult
from app.pipeline import new_job, process
from app.storage import InvalidJobId, Storage

log = logging.getLogger(__name__)
MB = 1024 * 1024
DEFAULT_MAX_UPLOAD_MB = 20
INTERRUPTED = "The server restarted before this document finished processing. Upload it again."
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
# The review page shows untrusted document text. Scripts only from this server; no framing, no plugins.
REVIEW_CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
    "object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
)


def create_app(
    storage: Storage | None = None,
    llm_factory: Callable[[], LLMClient] | None = None,
    today: Callable[[], date] = date.today,
    max_upload_bytes: int | None = None,
) -> FastAPI:
    load_dotenv()  # doesn't override variables already set in the environment
    storage = storage or Storage.from_env()
    llm_factory = llm_factory or client_from_env  # built per job, so a missing key fails the job, not startup
    max_bytes = max_upload_bytes or int(float(os.environ.get("MAX_UPLOAD_MB") or DEFAULT_MAX_UPLOAD_MB) * MB)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if interrupted := storage.fail_interrupted(INTERRUPTED):
            log.warning("marked %d interrupted job(s) as failed: %s", len(interrupted), ", ".join(interrupted))
        yield

    app = FastAPI(title="Invoice Intelligence", version=__version__, lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def review_page() -> FileResponse:
        return FileResponse(STATIC_DIR / "review.html", headers={"Content-Security-Policy": REVIEW_CSP})

    def run_job(job_id: str) -> None:
        job = storage.load_result(job_id)
        result = process(job, storage.load_upload(job_id), llm_factory=llm_factory, today=today())
        storage.save_result(result)

    @app.post("/extract", status_code=202, response_model=ExtractAccepted)
    async def extract(file: UploadFile, background: BackgroundTasks, response: Response) -> ExtractAccepted:
        data = await file.read(max_bytes + 1)
        if not data:
            raise HTTPException(400, "The uploaded file is empty.")
        if len(data) > max_bytes:
            raise HTTPException(413, f"The file is larger than the {max_bytes / MB:g} MB limit.")
        if b"%PDF-" not in data[:1024]:  # the PDF header may appear anywhere in the first 1024 bytes
            raise HTTPException(415, "Only PDF files are accepted.")

        job = new_job(data, (file.filename or "upload.pdf")[:255], datetime.now(UTC))
        storage.save_upload(job.id, data)
        storage.save_result(job)
        background.add_task(run_job, job.id)
        url = f"/result/{job.id}"
        response.headers["Location"] = url
        return ExtractAccepted(id=job.id, status=job.status, result_url=url)

    @app.get("/result/{job_id}", response_model=ExtractionResult)
    def result(job_id: str) -> ExtractionResult:
        try:
            found = storage.load_result(job_id)
        except InvalidJobId:
            found = None
        if found is None:
            raise HTTPException(404, "No result with this id.")
        return found

    return app

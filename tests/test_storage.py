"""
What:     Tests app/storage.py: round trips, atomic writes, id checking, and recovery of jobs interrupted by a restart.
In:       A temporary DATA_DIR per test.
Out:      pytest pass/fail.
Fails if: A path-safety or atomicity guarantee is weakened.
"""

from datetime import UTC, datetime

import pytest

from app.models import ExtractionResult, JobStatus
from app.storage import InvalidJobId, Storage

JOB = "0123456789abcdef0123456789abcdef"


def pending(job_id: str = JOB) -> ExtractionResult:
    return ExtractionResult(
        id=job_id, status=JobStatus.PENDING, filename="a.pdf", sha256="0" * 64,
        created_at=datetime(2025, 11, 11, tzinfo=UTC), pipeline_version="0.1.0",
    )


def test_round_trips(tmp_path):
    store = Storage(tmp_path)
    store.save_upload(JOB, b"%PDF-1.4 data")
    store.save_result(pending())
    assert store.load_upload(JOB) == b"%PDF-1.4 data"
    assert store.load_result(JOB) == pending()
    assert (tmp_path / "uploads" / f"{JOB}.pdf").is_file() and (tmp_path / "results" / f"{JOB}.json").is_file()


def test_unknown_id_is_none(tmp_path):
    assert Storage(tmp_path).load_result("f" * 32) is None


@pytest.mark.parametrize("bad", ["../../etc/passwd", "ABCDEF0123456789abcdef0123456789", "abc", JOB + "0", f"{JOB[:-1]}/", ""])
def test_ids_are_checked_before_touching_the_filesystem(tmp_path, bad):
    store = Storage(tmp_path)
    with pytest.raises(InvalidJobId):
        store.load_result(bad)
    with pytest.raises(InvalidJobId):
        store.save_upload(bad, b"x")


def test_writes_leave_no_temp_files(tmp_path):
    store = Storage(tmp_path)
    for _ in range(3):
        store.save_result(pending())
    assert [p.name for p in (tmp_path / "results").iterdir()] == [f"{JOB}.json"]


def test_failed_write_keeps_the_previous_file(tmp_path, monkeypatch):
    store = Storage(tmp_path)
    store.save_result(pending())
    monkeypatch.setattr("app.storage.os.replace", lambda *_: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        store.save_result(ExtractionResult(**(dict(pending()) | {"status": JobStatus.FAILED, "error": "x"})))
    assert store.load_result(JOB).status is JobStatus.PENDING
    assert [p.name for p in (tmp_path / "results").iterdir()] == [f"{JOB}.json"]


def test_interrupted_jobs_are_marked_failed(tmp_path):
    store = Storage(tmp_path)
    other = "f" * 32
    store.save_result(pending())
    store.save_result(ExtractionResult(**(dict(pending(other)) | {"status": JobStatus.FAILED, "error": "earlier"})))
    assert store.fail_interrupted("restarted") == [JOB]
    assert (store.load_result(JOB).status, store.load_result(JOB).error) == (JobStatus.FAILED, "restarted")
    assert store.load_result(other).error == "earlier"  # finished jobs are untouched
    assert store.fail_interrupted("restarted") == []

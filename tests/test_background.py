"""Tests for background job execution (colab_execute background mode).

Covers:
- JobStore: create, update (immutability), get, active_job_id, cleanup, reject when active
- run_background_job: success lifecycle, failure lifecycle, runtime always released
- colab_execute(background=True): returns job_id, rejects when active, rejects with drive
- colab_poll: all statuses, unknown job_id
- colab_jobs: empty, with jobs
"""

import asyncio
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp_colab_gpu.background import (
    CLEANUP_DELAY_SECONDS,
    JobRecord,
    JobStatus,
    JobStore,
    run_background_job,
)


# ---------------------------------------------------------------------------
# JobStore tests
# ---------------------------------------------------------------------------


class TestJobStore:
    """Tests for the JobStore class."""

    @pytest.fixture
    def store(self) -> JobStore:
        return JobStore()

    @pytest.mark.asyncio
    async def test_create_returns_job_id(self, store: JobStore):
        job_id = await store.create_if_no_active("T4")
        assert job_id is not None
        assert len(job_id) == 12

    @pytest.mark.asyncio
    async def test_create_sets_starting_status(self, store: JobStore):
        job_id = await store.create_if_no_active("T4")
        record = await store.get(job_id)
        assert record is not None
        assert record.status == JobStatus.STARTING
        assert record.accelerator == "T4"
        assert record.created_at is not None

    @pytest.mark.asyncio
    async def test_create_rejects_when_active(self, store: JobStore):
        first = await store.create_if_no_active("T4")
        assert first is not None
        second = await store.create_if_no_active("A100")
        assert second is None

    @pytest.mark.asyncio
    async def test_create_allows_after_completion(self, store: JobStore):
        job_id = await store.create_if_no_active("T4")
        await store.update(
            job_id,
            status=JobStatus.COMPLETED,
            completed_at=datetime.now(timezone.utc).isoformat(),
        )
        second = await store.create_if_no_active("A100")
        assert second is not None
        assert second != job_id

    @pytest.mark.asyncio
    async def test_update_immutability(self, store: JobStore):
        """update() creates a new record, does not mutate the old one."""
        job_id = await store.create_if_no_active("T4")
        original = await store.get(job_id)
        await store.update(job_id, status=JobStatus.RUNNING)
        updated = await store.get(job_id)
        # Original record is frozen, so its status is still STARTING
        assert original.status == JobStatus.STARTING
        assert updated.status == JobStatus.RUNNING
        assert original is not updated

    @pytest.mark.asyncio
    async def test_update_unknown_raises(self, store: JobStore):
        with pytest.raises(KeyError, match="Unknown job_id"):
            await store.update("nonexistent", status=JobStatus.RUNNING)

    @pytest.mark.asyncio
    async def test_get_unknown_returns_none(self, store: JobStore):
        result = await store.get("does-not-exist")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_all_empty(self, store: JobStore):
        records = await store.list_all()
        assert records == []

    @pytest.mark.asyncio
    async def test_list_all_with_jobs(self, store: JobStore):
        await store.create_if_no_active("T4")
        records = await store.list_all()
        assert len(records) == 1
        assert records[0].accelerator == "T4"

    @pytest.mark.asyncio
    async def test_remove(self, store: JobStore):
        job_id = await store.create_if_no_active("T4")
        await store.remove(job_id)
        assert await store.get(job_id) is None

    @pytest.mark.asyncio
    async def test_remove_nonexistent_no_error(self, store: JobStore):
        """remove() on a nonexistent job does not raise."""
        await store.remove("ghost")

    @pytest.mark.asyncio
    async def test_active_job_id_when_active(self, store: JobStore):
        job_id = await store.create_if_no_active("T4")
        active = await store.active_job_id()
        assert active == job_id

    @pytest.mark.asyncio
    async def test_active_job_id_none_when_completed(self, store: JobStore):
        job_id = await store.create_if_no_active("T4")
        await store.update(
            job_id,
            status=JobStatus.COMPLETED,
            completed_at=datetime.now(timezone.utc).isoformat(),
        )
        active = await store.active_job_id()
        assert active is None

    @pytest.mark.asyncio
    async def test_cleanup_completed(self, store: JobStore):
        job_id = await store.create_if_no_active("T4")
        # Mark completed with a time far in the past
        await store.update(
            job_id,
            status=JobStatus.COMPLETED,
            completed_at="2020-01-01T00:00:00+00:00",
        )
        removed = await store.cleanup_completed(max_age=1.0)
        assert removed == 1
        assert await store.get(job_id) is None

    @pytest.mark.asyncio
    async def test_cleanup_keeps_recent(self, store: JobStore):
        job_id = await store.create_if_no_active("T4")
        now = datetime.now(timezone.utc).isoformat()
        await store.update(
            job_id,
            status=JobStatus.COMPLETED,
            completed_at=now,
        )
        removed = await store.cleanup_completed(max_age=3600)
        assert removed == 0
        assert await store.get(job_id) is not None

    @pytest.mark.asyncio
    async def test_cleanup_skips_active(self, store: JobStore):
        await store.create_if_no_active("T4")
        removed = await store.cleanup_completed(max_age=0)
        assert removed == 0


# ---------------------------------------------------------------------------
# JobRecord frozen dataclass tests
# ---------------------------------------------------------------------------


class TestJobRecord:
    def test_frozen(self):
        record = JobRecord(
            job_id="abc123",
            status=JobStatus.STARTING,
            accelerator="T4",
            created_at="2026-01-01T00:00:00+00:00",
        )
        with pytest.raises(AttributeError):
            record.status = JobStatus.RUNNING  # type: ignore[misc]

    def test_defaults(self):
        record = JobRecord(
            job_id="abc123",
            status=JobStatus.STARTING,
            accelerator="T4",
            created_at="2026-01-01T00:00:00+00:00",
        )
        assert record.completed_at is None
        assert record.result is None
        assert record.error is None


# ---------------------------------------------------------------------------
# run_background_job tests
# ---------------------------------------------------------------------------


class TestRunBackgroundJob:
    @pytest.fixture
    def store(self) -> JobStore:
        return JobStore()

    @pytest.mark.asyncio
    async def test_success_lifecycle(self, store: JobStore):
        """Job transitions: STARTING -> RUNNING -> COMPLETED."""
        job_id = await store.create_if_no_active("T4")
        assert job_id is not None
        fake_stdout = '===CELL_START_0===\nhello\n===CELL_END_0==='
        fake_stderr = ""
        fake_rc = 0

        def mock_run(code, accel, hm, tout):
            return (fake_stdout, fake_stderr, fake_rc)

        def mock_format(stdout, stderr, rc):
            return json.dumps({"stdout": stdout, "exit_code": rc})

        with patch("mcp_colab_gpu.background._schedule_cleanup"):
            await run_background_job(
                store, job_id, mock_run,
                code="print('hello')",
                accelerator="T4",
                high_memory=False,
                timeout=300,
                format_result_fn=mock_format,
            )
        record = await store.get(job_id)
        assert record is not None
        assert record.status == JobStatus.COMPLETED
        assert record.completed_at is not None
        assert record.error is None
        result = json.loads(record.result)
        assert result["exit_code"] == 0

    @pytest.mark.asyncio
    async def test_failure_lifecycle(self, store: JobStore):
        """Job transitions: STARTING -> RUNNING -> FAILED on exception."""
        job_id = await store.create_if_no_active("T4")
        assert job_id is not None

        def mock_run(code, accel, hm, tout):
            raise RuntimeError("GPU allocation failed")

        def mock_format(stdout, stderr, rc):
            return json.dumps({"exit_code": rc})

        with patch("mcp_colab_gpu.background._schedule_cleanup"):
            await run_background_job(
                store, job_id, mock_run,
                code="print('hello')",
                accelerator="T4",
                high_memory=False,
                timeout=300,
                format_result_fn=mock_format,
            )
        record = await store.get(job_id)
        assert record is not None
        assert record.status == JobStatus.FAILED
        assert record.completed_at is not None
        assert "GPU allocation failed" in record.error

    @pytest.mark.asyncio
    async def test_cleanup_scheduled(self, store: JobStore):
        """run_background_job always schedules cleanup in the finally block."""
        job_id = await store.create_if_no_active("T4")

        def mock_run(code, accel, hm, tout):
            return ("out", "", 0)

        def mock_format(stdout, stderr, rc):
            return json.dumps({"exit_code": rc})

        with patch("mcp_colab_gpu.background._schedule_cleanup") as mock_sched:
            await run_background_job(
                store, job_id, mock_run,
                code="x",
                accelerator="T4",
                high_memory=False,
                timeout=300,
                format_result_fn=mock_format,
            )
            mock_sched.assert_called_once_with(store)

    @pytest.mark.asyncio
    async def test_cleanup_scheduled_on_failure(self, store: JobStore):
        """Cleanup is scheduled even when the job fails."""
        job_id = await store.create_if_no_active("T4")

        def mock_run(code, accel, hm, tout):
            raise ValueError("boom")

        def mock_format(stdout, stderr, rc):
            return json.dumps({"exit_code": rc})

        with patch("mcp_colab_gpu.background._schedule_cleanup") as mock_sched:
            await run_background_job(
                store, job_id, mock_run,
                code="x",
                accelerator="T4",
                high_memory=False,
                timeout=300,
                format_result_fn=mock_format,
            )
            mock_sched.assert_called_once_with(store)


# ---------------------------------------------------------------------------
# colab_execute(background=True) integration tests
# ---------------------------------------------------------------------------


class TestColabExecuteBackground:
    """Tests for the background parameter on colab_execute."""

    @pytest.mark.asyncio
    async def test_background_returns_job_id(self):
        with patch("mcp_colab_gpu.server._run_on_colab") as mock_run:
            mock_run.return_value = ("out", "", 0)
            # Reset store for test isolation
            from mcp_colab_gpu import server as srv
            srv._job_store = JobStore()
            result_str = await srv.colab_execute(
                code="print('hi')",
                accelerator="T4",
                background=True,
            )
            result = json.loads(result_str)
            assert "job_id" in result
            assert result["status"] == "starting"
            # Allow the background task to start
            await asyncio.sleep(0.1)

    @pytest.mark.asyncio
    async def test_background_rejects_with_drive_fetch(self):
        from mcp_colab_gpu import server as srv
        srv._job_store = JobStore()
        result_str = await srv.colab_execute(
            code="print('hi')",
            background=True,
            drive_fetch='{"data/file.csv": "/content/file.csv"}',
        )
        result = json.loads(result_str)
        assert "error" in result
        assert "incompatible" in result["error"]

    @pytest.mark.asyncio
    async def test_background_rejects_with_drive_save(self):
        from mcp_colab_gpu import server as srv
        srv._job_store = JobStore()
        result_str = await srv.colab_execute(
            code="print('hi')",
            background=True,
            drive_save='{"/content/model.pt": "results/model.pt"}',
        )
        result = json.loads(result_str)
        assert "error" in result
        assert "incompatible" in result["error"]

    @pytest.mark.asyncio
    async def test_background_rejects_when_active(self):
        from mcp_colab_gpu import server as srv
        srv._job_store = JobStore()
        # Create a first job (won't actually run since _run_on_colab is not mocked to finish)
        with patch("mcp_colab_gpu.server._run_on_colab") as mock_run:
            # Make the first job block so it stays active
            import time as _time
            mock_run.side_effect = lambda *_a, **_kw: _time.sleep(10)
            first_str = await srv.colab_execute(
                code="import time; time.sleep(10)",
                accelerator="T4",
                background=True,
            )
            first = json.loads(first_str)
            assert "job_id" in first
            # Second attempt should be rejected
            second_str = await srv.colab_execute(
                code="print('second')",
                accelerator="T4",
                background=True,
            )
            second = json.loads(second_str)
            assert "error" in second
            assert "already running" in second["error"]
            assert "active_job_id" in second

    @pytest.mark.asyncio
    async def test_sync_mode_unchanged(self):
        """background=False (default) still works synchronously."""
        with patch("mcp_colab_gpu.server._run_on_colab") as mock_run:
            mock_run.return_value = (
                '===CELL_START_0===\nhello\n===CELL_END_0===',
                "",
                0,
            )
            from mcp_colab_gpu import server as srv
            srv._job_store = JobStore()
            result_str = await srv.colab_execute(
                code="print('hello')",
                accelerator="T4",
            )
            result = json.loads(result_str)
            assert "cells" in result
            assert result["cells"][0]["stdout"] == "hello"
            assert result["exit_code"] == 0


# ---------------------------------------------------------------------------
# colab_poll tests
# ---------------------------------------------------------------------------


class TestColabPoll:
    @pytest.mark.asyncio
    async def test_poll_unknown_job(self):
        from mcp_colab_gpu import server as srv
        srv._job_store = JobStore()
        result_str = await srv.colab_poll(job_id="nonexistent")
        result = json.loads(result_str)
        assert "error" in result
        assert "Unknown" in result["error"]

    @pytest.mark.asyncio
    async def test_poll_starting(self):
        from mcp_colab_gpu import server as srv
        srv._job_store = JobStore()
        job_id = await srv._job_store.create_if_no_active("T4")
        result_str = await srv.colab_poll(job_id=job_id)
        result = json.loads(result_str)
        assert result["status"] == "starting"
        assert result["accelerator"] == "T4"
        assert "result" not in result
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_poll_running(self):
        from mcp_colab_gpu import server as srv
        srv._job_store = JobStore()
        job_id = await srv._job_store.create_if_no_active("A100")
        await srv._job_store.update(job_id, status=JobStatus.RUNNING)
        result_str = await srv.colab_poll(job_id=job_id)
        result = json.loads(result_str)
        assert result["status"] == "running"

    @pytest.mark.asyncio
    async def test_poll_completed(self):
        from mcp_colab_gpu import server as srv
        srv._job_store = JobStore()
        job_id = await srv._job_store.create_if_no_active("T4")
        fake_result = json.dumps({"cells": [], "exit_code": 0})
        await srv._job_store.update(
            job_id,
            status=JobStatus.COMPLETED,
            completed_at=datetime.now(timezone.utc).isoformat(),
            result=fake_result,
        )
        result_str = await srv.colab_poll(job_id=job_id)
        result = json.loads(result_str)
        assert result["status"] == "completed"
        assert result["result"]["exit_code"] == 0
        assert "completed_at" in result

    @pytest.mark.asyncio
    async def test_poll_failed(self):
        from mcp_colab_gpu import server as srv
        srv._job_store = JobStore()
        job_id = await srv._job_store.create_if_no_active("T4")
        await srv._job_store.update(
            job_id,
            status=JobStatus.FAILED,
            completed_at=datetime.now(timezone.utc).isoformat(),
            error="RuntimeError: GPU unavailable",
        )
        result_str = await srv.colab_poll(job_id=job_id)
        result = json.loads(result_str)
        assert result["status"] == "failed"
        assert "GPU unavailable" in result["error"]


# ---------------------------------------------------------------------------
# colab_jobs tests
# ---------------------------------------------------------------------------


class TestColabJobs:
    @pytest.mark.asyncio
    async def test_empty(self):
        from mcp_colab_gpu import server as srv
        srv._job_store = JobStore()
        result_str = await srv.colab_jobs()
        result = json.loads(result_str)
        assert result["jobs"] == []
        assert result["count"] == 0

    @pytest.mark.asyncio
    async def test_with_jobs(self):
        from mcp_colab_gpu import server as srv
        srv._job_store = JobStore()
        job_id = await srv._job_store.create_if_no_active("T4")
        result_str = await srv.colab_jobs()
        result = json.loads(result_str)
        assert result["count"] == 1
        assert result["jobs"][0]["job_id"] == job_id
        assert result["jobs"][0]["status"] == "starting"
        assert result["jobs"][0]["accelerator"] == "T4"

    @pytest.mark.asyncio
    async def test_with_completed_and_active(self):
        from mcp_colab_gpu import server as srv
        srv._job_store = JobStore()
        first_id = await srv._job_store.create_if_no_active("T4")
        await srv._job_store.update(
            first_id,
            status=JobStatus.COMPLETED,
            completed_at=datetime.now(timezone.utc).isoformat(),
        )
        _second_id = await srv._job_store.create_if_no_active("A100")
        result_str = await srv.colab_jobs()
        result = json.loads(result_str)
        assert result["count"] == 2
        statuses = {j["status"] for j in result["jobs"]}
        assert statuses == {"completed", "starting"}

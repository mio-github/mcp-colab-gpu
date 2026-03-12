"""Background job execution for long-running Colab GPU tasks.

Provides a non-blocking execution mode where jobs run in background
tasks while the MCP client can poll for status and results.

Key design:
- Only one active background job at a time (Colab single-GPU constraint)
- Frozen dataclass records (immutable updates via dataclasses.replace)
- asyncio.Lock for thread-safe job store operations
- Runtime released IMMEDIATELY on completion, not on poll
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

CLEANUP_DELAY_SECONDS = int(os.environ.get("MCP_BG_CLEANUP_DELAY", "300"))


class JobStatus(Enum):
    """Lifecycle states for a background job."""

    STARTING = "starting"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class JobRecord:
    """Immutable snapshot of a background job's state."""

    job_id: str
    status: JobStatus
    accelerator: str
    created_at: str
    completed_at: str | None = None
    result: str | None = None
    error: str | None = None


class JobStore:
    """Thread-safe store for background job records.

    Uses asyncio.Lock to serialize access. All mutations produce
    new JobRecord instances via dataclasses.replace().
    """

    def __init__(self) -> None:
        self._jobs: dict[str, JobRecord] = {}
        self._lock = asyncio.Lock()

    async def create_if_no_active(self, accelerator: str) -> str | None:
        """Create a new job if no active (STARTING/RUNNING) job exists.

        Returns:
            The new job_id, or None if an active job already exists.
        """
        async with self._lock:
            for record in self._jobs.values():
                if record.status in (JobStatus.STARTING, JobStatus.RUNNING):
                    return None
            job_id = uuid.uuid4().hex[:12]
            now = datetime.now(timezone.utc).isoformat()
            self._jobs[job_id] = JobRecord(
                job_id=job_id,
                status=JobStatus.STARTING,
                accelerator=accelerator,
                created_at=now,
            )
            return job_id

    async def update(self, job_id: str, **fields: object) -> None:
        """Update a job record by creating a new immutable copy.

        Raises:
            KeyError: If job_id is not found.
        """
        async with self._lock:
            old = self._jobs.get(job_id)
            if old is None:
                raise KeyError(f"Unknown job_id: {job_id}")
            self._jobs[job_id] = replace(old, **fields)

    async def get(self, job_id: str) -> JobRecord | None:
        """Return a job record by ID, or None if not found."""
        async with self._lock:
            return self._jobs.get(job_id)

    async def list_all(self) -> list[JobRecord]:
        """Return all tracked job records."""
        async with self._lock:
            return list(self._jobs.values())

    async def remove(self, job_id: str) -> None:
        """Remove a job record from the store."""
        async with self._lock:
            self._jobs.pop(job_id, None)

    async def cleanup_completed(self, max_age: float) -> int:
        """Remove completed/failed jobs older than max_age seconds.

        Returns:
            Number of jobs removed.
        """
        now = datetime.now(timezone.utc)
        removed = 0
        async with self._lock:
            to_remove = []
            for job_id, record in self._jobs.items():
                if record.status not in (JobStatus.COMPLETED, JobStatus.FAILED):
                    continue
                if record.completed_at is None:
                    continue
                completed = datetime.fromisoformat(record.completed_at)
                age = (now - completed).total_seconds()
                if age >= max_age:
                    to_remove.append(job_id)
            for job_id in to_remove:
                del self._jobs[job_id]
                removed += 1
        return removed

    async def active_job_id(self) -> str | None:
        """Return the job_id of the currently active job, if any."""
        async with self._lock:
            for record in self._jobs.values():
                if record.status in (JobStatus.STARTING, JobStatus.RUNNING):
                    return record.job_id
            return None


async def run_background_job(
    job_store: JobStore,
    job_id: str,
    run_fn: Callable[..., tuple[str, str, int]],
    *,
    code: str,
    accelerator: str,
    high_memory: bool,
    timeout: int,
    format_result_fn: Callable[..., str],
) -> None:
    """Execute a Colab job in the background.

    Lifecycle: STARTING -> RUNNING -> COMPLETED/FAILED

    The blocking run_fn is executed via asyncio.to_thread().
    A cleanup timer is scheduled on completion to remove stale records.

    Args:
        job_store: The JobStore instance to update.
        job_id: ID of the job to run.
        run_fn: Blocking function (code, accelerator, high_memory, timeout) -> (stdout, stderr, rc).
        code: The wrapped code to execute.
        accelerator: GPU/TPU type.
        high_memory: Whether high-memory mode is enabled.
        timeout: Execution timeout in seconds.
        format_result_fn: Function (stdout, stderr, rc) -> JSON result string.
    """
    try:
        await job_store.update(job_id, status=JobStatus.RUNNING)
        stdout, stderr, rc = await asyncio.to_thread(
            run_fn, code, accelerator, high_memory, timeout,
        )
        now = datetime.now(timezone.utc).isoformat()
        result_json = format_result_fn(stdout, stderr, rc)
        await job_store.update(
            job_id,
            status=JobStatus.COMPLETED,
            completed_at=now,
            result=result_json,
        )
        logger.info("Background job %s completed (exit_code=%d)", job_id, rc)
    except Exception as exc:
        now = datetime.now(timezone.utc).isoformat()
        await job_store.update(
            job_id,
            status=JobStatus.FAILED,
            completed_at=now,
            error=str(exc),
        )
        logger.exception("Background job %s failed: %s", job_id, exc)
    finally:
        _schedule_cleanup(job_store)


def _schedule_cleanup(job_store: JobStore) -> None:
    """Schedule deferred cleanup of completed jobs."""
    loop = asyncio.get_running_loop()
    loop.call_later(
        CLEANUP_DELAY_SECONDS,
        lambda: loop.create_task(_run_cleanup(job_store)),
    )


async def _run_cleanup(job_store: JobStore) -> None:
    """Run the actual cleanup coroutine."""
    removed = await job_store.cleanup_completed(CLEANUP_DELAY_SECONDS)
    if removed > 0:
        logger.info("Cleaned up %d completed background job(s)", removed)

# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Trajectory index service: budgeted async builds of Influence Views.

Workspace queue for post-run indexing (ADR-0010 / Q110). Smaller priority
values leave first. A session's pending slot keeps the highest-priority job;
a lower-priority job for that session is deferred, not discarded, until
``flush_deferred`` or a later ``enqueue`` promotes it. Over budget or paused,
``process_next`` returns ``SkippedResult`` and does not build a view or a
projection. With no builder, the service freezes a deterministic fallback
view and registers it on ``ViewSupersedeIndex``. When a projection registry
is injected, registering a new revision withdraws older cards for that
session and purpose from retrieval.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

from openviking.session.influence_projection import ProjectionRegistry
from openviking.session.influence_view import (
    PURPOSE_POST_RUN_INDEX,
    InfluenceViewRevision,
    ViewSupersedeIndex,
    build_minimal_fallback_view,
)

logger = logging.getLogger(__name__)

SERVICE_VERSION = 1

PRIORITY_EVALUATION = 0
PRIORITY_FAILED = 1
PRIORITY_SUCCEEDED = 2
PRIORITIES = frozenset((PRIORITY_EVALUATION, PRIORITY_FAILED, PRIORITY_SUCCEEDED))

SKIP_BUDGET_EXHAUSTED = "budget_exhausted"
SKIP_PAUSED = "paused"
SKIP_REASONS = frozenset((SKIP_BUDGET_EXHAUSTED, SKIP_PAUSED))

Priority = Literal[0, 1, 2]
SkipReason = Literal["budget_exhausted", "paused"]
Builder = Callable[[str, str, str], InfluenceViewRevision]
FallbackSession = Callable[["IndexJob"], Mapping[str, Any]]
Clock = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _format_dt(value: datetime) -> str:
    return _as_utc(value).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _parse_iso(value: str, field: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{field} must be ISO-8601, got {value!r}") from exc
    return _as_utc(parsed)


def _require_mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a dict")
    return value


def _require_str(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty str")
    return value


def _require_timestamp(value: Any, field: str) -> str:
    text = _require_str(value, field)
    return _format_dt(_parse_iso(text, field))


def _queue_key(job: IndexJob) -> tuple[int, str, str]:
    return (job.priority, job.enqueued_at, job.job_id)


@dataclass(frozen=True)
class IndexJob:
    """One pending trajectory build. Smaller ``priority`` is more urgent."""

    job_id: str
    session_id: str
    task_run_id: str
    priority: Priority
    outcome_status: str
    enqueued_at: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "job_id", _require_str(self.job_id, "job_id"))
        object.__setattr__(self, "session_id", _require_str(self.session_id, "session_id"))
        object.__setattr__(self, "task_run_id", _require_str(self.task_run_id, "task_run_id"))
        if isinstance(self.priority, bool) or self.priority not in PRIORITIES:
            raise ValueError(
                "priority must be 0 (evaluation), 1 (failed), or 2 (succeeded), "
                f"got {self.priority!r}"
            )
        object.__setattr__(
            self, "outcome_status", _require_str(self.outcome_status, "outcome_status")
        )
        object.__setattr__(self, "enqueued_at", _require_timestamp(self.enqueued_at, "enqueued_at"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "session_id": self.session_id,
            "task_run_id": self.task_run_id,
            "priority": self.priority,
            "outcome_status": self.outcome_status,
            "enqueued_at": self.enqueued_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IndexJob:
        payload = _require_mapping(data, "job")
        priority = payload.get("priority")
        if isinstance(priority, bool) or not isinstance(priority, int):
            raise ValueError("priority must be an int")
        return cls(
            job_id=str(payload.get("job_id") or ""),
            session_id=str(payload.get("session_id") or ""),
            task_run_id=str(payload.get("task_run_id") or ""),
            priority=priority,  # type: ignore[arg-type]
            outcome_status=str(payload.get("outcome_status") or ""),
            enqueued_at=str(payload.get("enqueued_at") or ""),
        )


@dataclass(frozen=True)
class WorkspacePolicy:
    """Workspace build budget. ``paused`` stops the queue without building."""

    max_builds_per_window: int
    window_seconds: float
    paused: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.max_builds_per_window, bool) or not isinstance(
            self.max_builds_per_window, int
        ):
            raise ValueError("max_builds_per_window must be an int")
        if self.max_builds_per_window < 0:
            raise ValueError("max_builds_per_window must be >= 0")
        if isinstance(self.window_seconds, bool) or not isinstance(self.window_seconds, (int, float)):
            raise ValueError("window_seconds must be a number")
        window = float(self.window_seconds)
        if not math.isfinite(window) or window <= 0:
            raise ValueError("window_seconds must be a finite number > 0")
        object.__setattr__(self, "window_seconds", window)
        if not isinstance(self.paused, bool):
            raise ValueError("paused must be a bool")

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_builds_per_window": self.max_builds_per_window,
            "window_seconds": self.window_seconds,
            "paused": self.paused,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkspacePolicy:
        payload = _require_mapping(data, "policy")
        return cls(
            max_builds_per_window=payload.get("max_builds_per_window"),  # type: ignore[arg-type]
            window_seconds=payload.get("window_seconds"),  # type: ignore[arg-type]
            paused=payload.get("paused", False),
        )


@dataclass(frozen=True)
class SkippedResult:
    """A queue step that did not build. ``projection`` is always absent."""

    reason: SkipReason
    job_id: str | None = None
    projection: None = None

    def __post_init__(self) -> None:
        if self.reason not in SKIP_REASONS:
            raise ValueError(f"reason must be one of {sorted(SKIP_REASONS)}, got {self.reason!r}")
        if self.job_id is not None:
            object.__setattr__(self, "job_id", _require_str(self.job_id, "job_id"))
        if self.projection is not None:
            raise ValueError("SkippedResult must not carry a projection")

    def to_dict(self) -> dict[str, Any]:
        return {"reason": self.reason, "job_id": self.job_id, "projection": None}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SkippedResult:
        payload = _require_mapping(data, "skipped")
        if payload.get("projection") is not None:
            raise ValueError("SkippedResult must not carry a projection")
        job_id = payload.get("job_id")
        return cls(
            reason=payload.get("reason"),  # type: ignore[arg-type]
            job_id=None if job_id is None else str(job_id),
        )


@dataclass(frozen=True)
class IndexedResult:
    """A view that was frozen and registered. This service does not emit cards."""

    job: IndexJob
    revision: InfluenceViewRevision

    def __post_init__(self) -> None:
        if not isinstance(self.job, IndexJob):
            raise ValueError("job must be an IndexJob")
        if not isinstance(self.revision, InfluenceViewRevision):
            raise ValueError("revision must be an InfluenceViewRevision")

    def to_dict(self) -> dict[str, Any]:
        return {"job": self.job.to_dict(), "revision": self.revision.to_dict()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IndexedResult:
        payload = _require_mapping(data, "indexed")
        return cls(
            job=IndexJob.from_dict(_require_mapping(payload.get("job"), "job")),
            revision=InfluenceViewRevision.from_dict(
                _require_mapping(payload.get("revision"), "revision")
            ),
        )


def _synthetic_fallback_session(job: IndexJob) -> dict[str, Any]:
    """Minimal one-AO session so the fallback builder can freeze a view."""
    digest = hashlib.sha256(
        f"{job.session_id}:{job.task_run_id}:{job.job_id}".encode("utf-8")
    ).hexdigest()
    return {
        "session_id": job.session_id,
        "task_run_id": job.task_run_id,
        "purpose": PURPOSE_POST_RUN_INDEX,
        "view_id": f"fallback-{job.session_id}-{PURPOSE_POST_RUN_INDEX}",
        "revision_id": f"fallback-{job.job_id}",
        "frozen_at": job.enqueued_at,
        "snapshot_watermark": f"fallback:{job.session_id}:{job.task_run_id}",
        "summary": f"deterministic fallback for {job.session_id}",
        "ao_records": [
            {
                "ao_id": f"ao-{job.job_id}",
                "sequence": 1,
                "participant_id": "fallback-agent",
                "segment_id": f"seg-{job.session_id}",
                "session_id": job.session_id,
                "content_hash": digest,
            }
        ],
    }


class TrajectoryIndexService:
    """In-process priority queue with a JSON snapshot of pending work."""

    def __init__(
        self,
        path: str | Path,
        supersede_index: ViewSupersedeIndex,
        *,
        builder: Builder | None = None,
        fallback_session: FallbackSession | None = None,
        clock: Clock | None = None,
        projection_registry: ProjectionRegistry | None = None,
    ) -> None:
        if not isinstance(supersede_index, ViewSupersedeIndex):
            raise ValueError("supersede_index must be a ViewSupersedeIndex")
        if projection_registry is not None and not isinstance(projection_registry, ProjectionRegistry):
            raise ValueError("projection_registry must be a ProjectionRegistry")
        self.path = Path(path)
        self._supersede = supersede_index
        self._projection_registry = projection_registry
        self._builder = builder
        self._fallback_session = fallback_session
        self._clock = clock or _utc_now
        self._lock = threading.RLock()
        self._pending: dict[str, IndexJob] = {}
        self._deferred: dict[str, list[IndexJob]] = {}
        self._build_timestamps: list[str] = []
        self._warned_missing_projection_registry = False
        self._load()

    def enqueue(self, job: IndexJob) -> IndexJob:
        """Queue ``job``. The pending slot keeps the highest-priority job.

        Equal priority replaces the pending job so the latest task run at that
        priority is the one that will build. A worse priority is deferred, not
        dropped. A session with no pending job promotes its deferred job first,
        so a later enqueue can pick up work that was waiting on a higher priority.
        """
        if not isinstance(job, IndexJob):
            raise ValueError("job must be an IndexJob")
        with self._lock:
            self._promote_ready_deferred_locked()
            existing = self._pending.get(job.session_id)
            if existing is None:
                self._pending[job.session_id] = job
                self._save_locked()
                return job
            if job.priority < existing.priority:
                self._defer_locked(existing)
                self._pending[job.session_id] = job
                self._save_locked()
                return job
            if job.priority > existing.priority:
                self._defer_locked(job)
                self._save_locked()
                return existing
            self._pending[job.session_id] = job
            self._save_locked()
            return job

    def pending_jobs(self) -> tuple[IndexJob, ...]:
        with self._lock:
            return tuple(sorted(self._pending.values(), key=_queue_key))

    def deferred_jobs(self) -> tuple[IndexJob, ...]:
        """Lower-priority jobs held back by a higher-priority pending job."""
        with self._lock:
            jobs = [job for bucket in self._deferred.values() for job in bucket]
            return tuple(sorted(jobs, key=_queue_key))

    def flush_deferred(self) -> tuple[IndexJob, ...]:
        """Promote deferred jobs whose session has no active pending job."""
        with self._lock:
            promoted = self._promote_ready_deferred_locked()
            if promoted:
                self._save_locked()
            return tuple(sorted(promoted, key=_queue_key))

    def process_next(self, policy: WorkspacePolicy) -> IndexedResult | SkippedResult:
        """Build the next job, or skip with an explicit empty projection.

        ``paused`` and an exhausted window both leave the job queued. They do
        not call the builder and do not register a revision. A successful build
        registers the revision before the queue snapshot. Replaying a pending
        job whose ``(session_id, purpose, content_hash)`` is already indexed
        does not append another supersede record. With a projection registry,
        older cards for that session and purpose leave retrieval.
        """
        if not isinstance(policy, WorkspacePolicy):
            raise ValueError("policy must be a WorkspacePolicy")
        with self._lock:
            nxt = self._select_locked()
            if policy.paused:
                return SkippedResult(
                    reason=SKIP_PAUSED,
                    job_id=None if nxt is None else nxt.job_id,
                )
            if nxt is None:
                raise ValueError("no pending index job")
            if not self._budget_allows_locked(policy):
                return SkippedResult(reason=SKIP_BUDGET_EXHAUSTED, job_id=nxt.job_id)
            revision = self._build_locked(nxt)
            if not self._content_hash_registered_locked(revision):
                self._supersede.register(revision)
            self._sync_projections_locked(revision)
            self._pending.pop(nxt.session_id, None)
            self._build_timestamps.append(_format_dt(_as_utc(self._clock())))
            self._save_locked()
            return IndexedResult(job=nxt, revision=revision)

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            jobs = [job.to_dict() for job in sorted(self._pending.values(), key=_queue_key)]
            return {
                "version": SERVICE_VERSION,
                "jobs": jobs,
                "deferred": self._deferred_dicts_locked(),
                "build_timestamps": list(self._build_timestamps),
            }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        path: str | Path,
        supersede_index: ViewSupersedeIndex,
        builder: Builder | None = None,
        fallback_session: FallbackSession | None = None,
        clock: Clock | None = None,
        projection_registry: ProjectionRegistry | None = None,
    ) -> TrajectoryIndexService:
        service = cls(
            path,
            supersede_index,
            builder=builder,
            fallback_session=fallback_session,
            clock=clock,
            projection_registry=projection_registry,
        )
        service._apply(data, persist=True)
        return service

    def _defer_locked(self, job: IndexJob) -> None:
        bucket = [
            item
            for item in self._deferred.get(job.session_id, [])
            if item.job_id != job.job_id and item.priority != job.priority
        ]
        bucket.append(job)
        self._deferred[job.session_id] = bucket

    def _promote_ready_deferred_locked(self) -> list[IndexJob]:
        promoted: list[IndexJob] = []
        for session_id in sorted(self._deferred):
            if session_id in self._pending:
                continue
            jobs = list(self._deferred.get(session_id) or [])
            if not jobs:
                continue
            best = min(jobs, key=_queue_key)
            rest = [item for item in jobs if item.job_id != best.job_id]
            self._pending[session_id] = best
            promoted.append(best)
            if rest:
                self._deferred[session_id] = rest
            else:
                del self._deferred[session_id]
        return promoted

    def _deferred_dicts_locked(self) -> list[dict[str, Any]]:
        jobs = [job for bucket in self._deferred.values() for job in bucket]
        jobs.sort(key=_queue_key)
        return [job.to_dict() for job in jobs]

    def _content_hash_registered_locked(self, revision: InfluenceViewRevision) -> bool:
        history = self._supersede.history(revision.session_id, revision.purpose)
        return any(item.content_hash == revision.content_hash for item in history)

    def _sync_projections_locked(self, revision: InfluenceViewRevision) -> None:
        registry = self._projection_registry
        if registry is None:
            if not self._warned_missing_projection_registry:
                self._warned_missing_projection_registry = True
                logger.warning(
                    "projection_registry is not configured; superseded projection cards "
                    "were not withdrawn from retrieval"
                )
            return
        registry.register(revision)
        registry.withdraw_superseded(
            revision.session_id,
            revision.purpose,
            revision.revision_id,
        )

    def _select_locked(self) -> IndexJob | None:
        if not self._pending:
            return None
        return min(self._pending.values(), key=_queue_key)

    def _budget_allows_locked(self, policy: WorkspacePolicy) -> bool:
        if policy.max_builds_per_window <= 0:
            return False
        now = _as_utc(self._clock())
        cutoff = now - timedelta(seconds=policy.window_seconds)
        recent = sum(1 for stamp in self._build_timestamps if _parse_iso(stamp, "build_timestamp") > cutoff)
        return recent < policy.max_builds_per_window

    def _build_locked(self, job: IndexJob) -> InfluenceViewRevision:
        if self._builder is None:
            session = (
                self._fallback_session(job)
                if self._fallback_session is not None
                else _synthetic_fallback_session(job)
            )
            revision = build_minimal_fallback_view(session)
        else:
            revision = self._builder(job.session_id, job.task_run_id, job.outcome_status)
        if not isinstance(revision, InfluenceViewRevision):
            raise ValueError("builder must return an InfluenceViewRevision")
        if revision.session_id != job.session_id or revision.task_run_id != job.task_run_id:
            raise ValueError("builder revision must match the index job session and task run")
        return revision

    def _apply(self, data: dict[str, Any], *, persist: bool) -> None:
        payload = _require_mapping(data, "service")
        version = payload.get("version")
        if version != SERVICE_VERSION:
            raise ValueError(f"unsupported trajectory index version {version!r}")
        jobs_raw = payload.get("jobs")
        if not isinstance(jobs_raw, list):
            raise ValueError("jobs must be a list")
        deferred_raw = payload.get("deferred", [])
        if not isinstance(deferred_raw, list):
            raise ValueError("deferred must be a list")
        stamps = payload.get("build_timestamps")
        if not isinstance(stamps, list):
            raise ValueError("build_timestamps must be a list")
        pending: dict[str, IndexJob] = {}
        for item in jobs_raw:
            job = IndexJob.from_dict(_require_mapping(item, "job"))
            existing = pending.get(job.session_id)
            if existing is None or job.priority <= existing.priority:
                pending[job.session_id] = job
        deferred: dict[str, list[IndexJob]] = {}
        for item in deferred_raw:
            job = IndexJob.from_dict(_require_mapping(item, "job"))
            active = pending.get(job.session_id)
            if active is not None and active.job_id == job.job_id:
                continue
            bucket = deferred.setdefault(job.session_id, [])
            if any(item_job.job_id == job.job_id or item_job.priority == job.priority for item_job in bucket):
                bucket[:] = [
                    item_job
                    for item_job in bucket
                    if item_job.job_id != job.job_id and item_job.priority != job.priority
                ]
            bucket.append(job)
        normalized = [_require_timestamp(stamp, "build_timestamp") for stamp in stamps]
        with self._lock:
            self._pending = pending
            self._deferred = deferred
            self._build_timestamps = normalized
            if persist:
                self._save_locked()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        self._apply(payload, persist=False)

    def _save_locked(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        body = {
            "version": SERVICE_VERSION,
            "jobs": [job.to_dict() for job in sorted(self._pending.values(), key=_queue_key)],
            "deferred": self._deferred_dicts_locked(),
            "build_timestamps": list(self._build_timestamps),
        }
        text = json.dumps(body, ensure_ascii=False, indent=2, sort_keys=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(self.path)

# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Unit tests for the trajectory index service (slice S12)."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone

import pytest

from openviking.session.influence_projection import (
    REMOVAL_SUPERSEDED,
    ProjectionRegistry,
)
from openviking.session.influence_view import (
    DETERMINISTIC_FALLBACK_REASON,
    PURPOSE_POST_RUN_INDEX,
    ViewSupersedeIndex,
    build_minimal_fallback_view,
)
from openviking.session.trajectory_index import (
    PRIORITY_EVALUATION,
    PRIORITY_FAILED,
    PRIORITY_SUCCEEDED,
    SKIP_BUDGET_EXHAUSTED,
    SKIP_PAUSED,
    IndexJob,
    IndexedResult,
    SkippedResult,
    TrajectoryIndexService,
    WorkspacePolicy,
)

T0 = "2026-01-01T00:00:00.000Z"
T1 = "2026-01-01T00:00:01.000Z"
T2 = "2026-01-01T00:00:02.000Z"


def _job(
    job_id: str,
    session_id: str,
    *,
    priority: int,
    outcome_status: str,
    task_run_id: str | None = None,
    enqueued_at: str = T0,
) -> IndexJob:
    return IndexJob(
        job_id=job_id,
        session_id=session_id,
        task_run_id=task_run_id or f"run-{job_id}",
        priority=priority,  # type: ignore[arg-type]
        outcome_status=outcome_status,
        enqueued_at=enqueued_at,
    )


def _revision_for(session_id: str, task_run_id: str, outcome_status: str):
    del outcome_status
    return build_minimal_fallback_view(
        {
            "session_id": session_id,
            "task_run_id": task_run_id,
            "purpose": PURPOSE_POST_RUN_INDEX,
            "revision_id": f"rev-{task_run_id}",
            "frozen_at": T0,
            "ao_records": [
                {
                    "ao_id": f"ao-{task_run_id}",
                    "sequence": 1,
                    "participant_id": "agent-a",
                    "segment_id": "seg-a",
                    "session_id": session_id,
                    "content_hash": f"hash-{task_run_id}",
                }
            ],
        }
    )


def _service(
    tmp_path,
    *,
    builder=None,
    clock=None,
    name: str = "queue.json",
    projection_registry=None,
):
    index = ViewSupersedeIndex(tmp_path / f"{name}.views.json")
    service = TrajectoryIndexService(
        tmp_path / name,
        index,
        builder=builder,
        clock=clock,
        projection_registry=projection_registry,
    )
    return service, index


def test_process_next_orders_evaluation_before_failed_before_succeeded(tmp_path) -> None:
    seen: list[str] = []

    def builder(session_id: str, task_run_id: str, outcome_status: str):
        seen.append(outcome_status)
        return _revision_for(session_id, task_run_id, outcome_status)

    service, _index = _service(tmp_path, builder=builder)
    service.enqueue(
        _job("job-ok", "sess-ok", priority=PRIORITY_SUCCEEDED, outcome_status="succeeded", enqueued_at=T0)
    )
    service.enqueue(
        _job("job-fail", "sess-fail", priority=PRIORITY_FAILED, outcome_status="failed", enqueued_at=T2)
    )
    service.enqueue(
        _job("job-eval", "sess-eval", priority=PRIORITY_EVALUATION, outcome_status="evaluation", enqueued_at=T1)
    )
    policy = WorkspacePolicy(max_builds_per_window=10, window_seconds=60)
    order: list[str] = []
    for _ in range(3):
        result = service.process_next(policy)
        assert isinstance(result, IndexedResult)
        order.append(result.job.outcome_status)
    assert order == ["evaluation", "failed", "succeeded"]
    assert seen == order


def test_enqueue_dedup_keeps_highest_priority_for_session(tmp_path) -> None:
    service, _index = _service(tmp_path, builder=_revision_for)
    assert service.enqueue(
        _job("job-low", "sess", priority=PRIORITY_SUCCEEDED, outcome_status="succeeded")
    ).job_id == "job-low"
    assert service.enqueue(
        _job("job-high", "sess", priority=PRIORITY_EVALUATION, outcome_status="evaluation", enqueued_at=T1)
    ).job_id == "job-high"
    assert service.enqueue(
        _job("job-mid", "sess", priority=PRIORITY_FAILED, outcome_status="failed", enqueued_at=T2)
    ).job_id == "job-high"
    assert [job.job_id for job in service.pending_jobs()] == ["job-high"]


def test_budget_exhausted_skips_without_projection(tmp_path) -> None:
    calls: list[str] = []
    clock = {"now": datetime(2026, 1, 1, tzinfo=timezone.utc)}

    def builder(session_id: str, task_run_id: str, outcome_status: str):
        calls.append(session_id)
        return _revision_for(session_id, task_run_id, outcome_status)

    service, index = _service(tmp_path, builder=builder, clock=lambda: clock["now"])
    service.enqueue(_job("job-1", "sess-1", priority=PRIORITY_EVALUATION, outcome_status="evaluation"))
    service.enqueue(
        _job("job-2", "sess-2", priority=PRIORITY_FAILED, outcome_status="failed", enqueued_at=T1)
    )
    policy = WorkspacePolicy(max_builds_per_window=1, window_seconds=10)
    first = service.process_next(policy)
    assert isinstance(first, IndexedResult)
    skipped = service.process_next(policy)
    assert isinstance(skipped, SkippedResult)
    assert skipped.reason == SKIP_BUDGET_EXHAUSTED
    assert skipped.projection is None
    assert skipped.to_dict()["projection"] is None
    assert calls == ["sess-1"]
    assert index.latest("sess-2", PURPOSE_POST_RUN_INDEX) is None
    assert [job.job_id for job in service.pending_jobs()] == ["job-2"]

    clock["now"] = clock["now"] + timedelta(seconds=10)
    second = service.process_next(policy)
    assert isinstance(second, IndexedResult)
    assert second.job.session_id == "sess-2"
    assert index.latest("sess-2", PURPOSE_POST_RUN_INDEX) is not None


def test_paused_does_not_build(tmp_path) -> None:
    calls: list[str] = []

    def builder(session_id: str, task_run_id: str, outcome_status: str):
        calls.append(session_id)
        return _revision_for(session_id, task_run_id, outcome_status)

    service, index = _service(tmp_path, builder=builder)
    service.enqueue(_job("job-1", "sess-1", priority=PRIORITY_FAILED, outcome_status="failed"))
    paused = WorkspacePolicy(max_builds_per_window=5, window_seconds=60, paused=True)
    skipped = service.process_next(paused)
    assert isinstance(skipped, SkippedResult)
    assert skipped.reason == SKIP_PAUSED
    assert skipped.job_id == "job-1"
    assert skipped.projection is None
    assert calls == []
    assert index.latest("sess-1", PURPOSE_POST_RUN_INDEX) is None
    assert [job.job_id for job in service.pending_jobs()] == ["job-1"]

    built = service.process_next(WorkspacePolicy(max_builds_per_window=5, window_seconds=60))
    assert isinstance(built, IndexedResult)
    assert calls == ["sess-1"]


def test_missing_builder_registers_fallback_view(tmp_path) -> None:
    service, index = _service(tmp_path, builder=None)
    service.enqueue(_job("job-fb", "sess-fb", priority=PRIORITY_SUCCEEDED, outcome_status="succeeded"))
    result = service.process_next(WorkspacePolicy(max_builds_per_window=1, window_seconds=60))
    assert isinstance(result, IndexedResult)
    assert result.revision.construction_mode == "fallback"
    assert result.revision.blocks[0].granularity_reason == DETERMINISTIC_FALLBACK_REASON
    latest = index.latest("sess-fb", PURPOSE_POST_RUN_INDEX)
    assert latest is not None
    assert latest.revision_id == result.revision.revision_id
    assert latest.construction_mode == "fallback"


def test_second_process_supersedes_latest_revision(tmp_path) -> None:
    service, index = _service(tmp_path, builder=_revision_for)
    policy = WorkspacePolicy(max_builds_per_window=5, window_seconds=60)
    service.enqueue(
        _job("job-1", "sess", priority=PRIORITY_SUCCEEDED, outcome_status="succeeded", task_run_id="run-1")
    )
    first = service.process_next(policy)
    assert isinstance(first, IndexedResult)
    service.enqueue(
        _job(
            "job-2",
            "sess",
            priority=PRIORITY_SUCCEEDED,
            outcome_status="succeeded",
            task_run_id="run-2",
            enqueued_at=T1,
        )
    )
    second = service.process_next(policy)
    assert isinstance(second, IndexedResult)
    latest = index.latest("sess", PURPOSE_POST_RUN_INDEX)
    assert latest is not None
    assert latest.revision_id == "rev-run-2"
    assert latest.revision_id == second.revision.revision_id
    history = index.history("sess", PURPOSE_POST_RUN_INDEX)
    assert [item.revision_id for item in history] == ["rev-run-1", "rev-run-2"]


def test_state_round_trip_and_reload_preserves_budget(tmp_path) -> None:
    clock = {"now": datetime(2026, 1, 1, tzinfo=timezone.utc)}
    service, index = _service(tmp_path, builder=_revision_for, clock=lambda: clock["now"])
    service.enqueue(_job("job-1", "sess-1", priority=PRIORITY_EVALUATION, outcome_status="evaluation"))
    service.enqueue(
        _job("job-2", "sess-2", priority=PRIORITY_FAILED, outcome_status="failed", enqueued_at=T1)
    )
    policy = WorkspacePolicy(max_builds_per_window=1, window_seconds=30, paused=False)
    assert policy == WorkspacePolicy.from_dict(json.loads(json.dumps(policy.to_dict())))
    built = service.process_next(policy)
    assert isinstance(built, IndexedResult)
    assert IndexedResult.from_dict(json.loads(json.dumps(built.to_dict()))) == built

    skipped = SkippedResult(reason=SKIP_BUDGET_EXHAUSTED, job_id="job-2")
    assert skipped == SkippedResult.from_dict(json.loads(json.dumps(skipped.to_dict())))
    with pytest.raises(ValueError, match="projection"):
        SkippedResult.from_dict({"reason": SKIP_PAUSED, "job_id": "job-2", "projection": {"card_id": "x"}})

    payload = json.loads(json.dumps(service.to_dict()))
    restored = TrajectoryIndexService.from_dict(
        payload,
        path=tmp_path / "restored.json",
        supersede_index=index,
        builder=_revision_for,
        clock=lambda: clock["now"],
    )
    assert restored.to_dict() == service.to_dict()

    reloaded = TrajectoryIndexService(
        tmp_path / "queue.json",
        index,
        builder=_revision_for,
        clock=lambda: clock["now"],
    )
    assert [job.job_id for job in reloaded.pending_jobs()] == ["job-2"]
    skipped_again = reloaded.process_next(policy)
    assert isinstance(skipped_again, SkippedResult)
    assert skipped_again.reason == SKIP_BUDGET_EXHAUSTED
    assert index.latest("sess-2", PURPOSE_POST_RUN_INDEX) is None


def test_process_next_withdraws_superseded_projection_cards(tmp_path) -> None:
    registry = ProjectionRegistry(
        tmp_path / "projections.json",
        session_acl=lambda _session_id: ("team-a",),
    )
    service, _index = _service(tmp_path, builder=_revision_for, projection_registry=registry)
    policy = WorkspacePolicy(max_builds_per_window=5, window_seconds=60)
    service.enqueue(
        _job("job-1", "sess", priority=PRIORITY_SUCCEEDED, outcome_status="succeeded", task_run_id="run-1")
    )
    first = service.process_next(policy)
    assert isinstance(first, IndexedResult)
    visible = registry.search(principal_labels=("team-a",))
    assert {card.source_pointer.view_revision_id for card in visible} == {first.revision.revision_id}

    service.enqueue(
        _job(
            "job-2",
            "sess",
            priority=PRIORITY_SUCCEEDED,
            outcome_status="succeeded",
            task_run_id="run-2",
            enqueued_at=T1,
        )
    )
    second = service.process_next(policy)
    assert isinstance(second, IndexedResult)
    visible = registry.search(principal_labels=("team-a",))
    assert {card.source_pointer.view_revision_id for card in visible} == {second.revision.revision_id}
    superseded = [
        record for record in registry.removed_records() if record.reason == REMOVAL_SUPERSEDED
    ]
    assert [record.card.source_pointer.view_revision_id for record in superseded] == [
        first.revision.revision_id
    ]


def test_missing_projection_registry_warns_once_and_still_indexes(tmp_path, caplog) -> None:
    service, index = _service(tmp_path, builder=_revision_for)
    policy = WorkspacePolicy(max_builds_per_window=5, window_seconds=60)
    service.enqueue(
        _job("job-1", "sess-1", priority=PRIORITY_SUCCEEDED, outcome_status="succeeded")
    )
    service.enqueue(
        _job(
            "job-2",
            "sess-2",
            priority=PRIORITY_FAILED,
            outcome_status="failed",
            enqueued_at=T1,
        )
    )
    with caplog.at_level(logging.WARNING):
        first = service.process_next(policy)
        second = service.process_next(policy)
    assert isinstance(first, IndexedResult)
    assert isinstance(second, IndexedResult)
    assert index.latest("sess-1", PURPOSE_POST_RUN_INDEX) is not None
    assert index.latest("sess-2", PURPOSE_POST_RUN_INDEX) is not None
    warnings = [record for record in caplog.records if "projection_registry" in record.message]
    assert len(warnings) == 1
    assert warnings[0].levelno == logging.WARNING


def test_replay_same_content_hash_does_not_duplicate_supersede(tmp_path) -> None:
    service, index = _service(tmp_path, builder=_revision_for)
    policy = WorkspacePolicy(max_builds_per_window=5, window_seconds=60)
    job = _job(
        "job-1",
        "sess",
        priority=PRIORITY_SUCCEEDED,
        outcome_status="succeeded",
        task_run_id="run-1",
    )
    service.enqueue(job)
    first = service.process_next(policy)
    assert isinstance(first, IndexedResult)
    service.enqueue(job)
    second = service.process_next(policy)
    assert isinstance(second, IndexedResult)
    history = index.history("sess", PURPOSE_POST_RUN_INDEX)
    assert len(history) == 1
    assert history[0].content_hash == first.revision.content_hash
    assert second.revision.content_hash == first.revision.content_hash


def test_lower_priority_job_stays_deferred_until_flush(tmp_path) -> None:
    service, index = _service(tmp_path, builder=_revision_for)
    policy = WorkspacePolicy(max_builds_per_window=5, window_seconds=60)
    service.enqueue(
        _job(
            "job-low",
            "sess",
            priority=PRIORITY_SUCCEEDED,
            outcome_status="succeeded",
            task_run_id="run-low",
        )
    )
    service.enqueue(
        _job(
            "job-high",
            "sess",
            priority=PRIORITY_EVALUATION,
            outcome_status="evaluation",
            task_run_id="run-high",
            enqueued_at=T1,
        )
    )
    assert [job.job_id for job in service.pending_jobs()] == ["job-high"]
    assert [job.job_id for job in service.deferred_jobs()] == ["job-low"]
    high = service.process_next(policy)
    assert isinstance(high, IndexedResult)
    assert high.job.job_id == "job-high"
    assert [job.job_id for job in service.pending_jobs()] == []
    assert [job.job_id for job in service.deferred_jobs()] == ["job-low"]
    promoted = service.flush_deferred()
    assert [job.job_id for job in promoted] == ["job-low"]
    low = service.process_next(policy)
    assert isinstance(low, IndexedResult)
    assert low.job.job_id == "job-low"
    assert [item.revision_id for item in index.history("sess", PURPOSE_POST_RUN_INDEX)] == [
        "rev-run-high",
        "rev-run-low",
    ]


def test_process_next_empty_queue_raises(tmp_path) -> None:
    service, _index = _service(tmp_path, builder=_revision_for)
    with pytest.raises(ValueError, match="no pending"):
        service.process_next(WorkspacePolicy(max_builds_per_window=1, window_seconds=60))

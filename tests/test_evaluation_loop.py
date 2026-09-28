# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Minimal Memory → Diagnosis → Memory evaluation loop."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from openviking.session.causal_experiences import (
    CausalExperiencesStore,
    ForkStatus,
    ServerChecks,
)
from openviking.session.citation_ledger import (
    KIND_CITATION,
    KIND_HINT_EXPOSURE,
    CitationLedger,
)
from openviking.session.evaluation_loop import EvaluationIteration, MinimalEvaluationLoop
from openviking.session.evaluation_orchestrator import (
    EVENT_COMMIT_REVISIONS,
    EVENT_DIAGNOSIS,
    EVENT_FREEZE_BRIEF,
    EVENT_MEMORY_RETRIEVE,
    EVENT_NEXT_RUN,
    EVENT_SCORE,
    EVENT_STOP,
    EVENT_TASK_RUN,
    EvaluationOrchestrator,
    IterationResult,
    NoSeatError,
    OrchestratorState,
)
from openviking.session.handoff_envelope import (
    HandoffEnvelopeService,
    content_address,
    idempotency_key,
)
from openviking.session.memory_hints import HintDeliveryService
from tests.test_causal_experiences import _payload

T0 = datetime(2026, 9, 24, tzinfo=timezone.utc)
_CAUSAL = {"skill_trajectory_mode": "causal"}
_CHECKS = ServerChecks(
    anchor_exists=True,
    refs_exist=True,
    provenance_valid=True,
    acl_allowed=True,
    acl_hook=lambda _draft: True,
)


class MutableClock:
    def __init__(self) -> None:
        self.current = T0

    def __call__(self) -> datetime:
        return self.current


class SequenceIds:
    def __init__(self) -> None:
        self._n = 0

    def __call__(self) -> str:
        self._n += 1
        return f"id-{self._n}"


def _iteration(**overrides: object) -> EvaluationIteration:
    fields: dict[str, object] = {
        "iteration_id": "iter-1",
        "task_id": "task-1",
        "source_run_id": "run-1",
        "memory_task_state_hash": "state-hash-1",
        "selected_refs": ("fork:1",),
        "snapshot_watermarks": {"ledger": "wm-1"},
        "policy_versions": {"decay": "v1"},
        "sub_budgets": {"steps": 4},
        "target_agent_id": "agent-memory",
        "task_lineage_id": "lineage-task-1",
        "iteration_result": IterationResult(
            terminal=True,
            evidence_refs=("ref-1",),
            key_judgment="same",
        ),
        "diagnosis_batch": 1,
    }
    fields.update(overrides)
    return EvaluationIteration(**fields)  # type: ignore[arg-type]


def _hint_payload(revisions: object) -> dict[str, object]:
    revision = revisions[0]  # type: ignore[index]
    return {
        "hint_id": "hint-iter-1",
        "task_id": "task-1",
        "run_id": "run-1",
        "target_agent_id": "agent-memory",
        "sources": [
            {
                "kind": "fork",
                "ref": revision.fork_node_id,
                "revision": revision.revision_id,
            }
        ],
        "match_reason": "provisional fork from diagnosis",
        "anchor_state_summary": "ao anchor",
        "applicability": "same task",
        "confidence_status": "provisional",
        "provenance": "eval-loop-1:iter-1",
        "ttl_expires_at": "2026-09-25T00:00:00.000Z",
        "created_at": "2026-09-24T00:00:00.000Z",
    }


def _wire(tmp_path: Path, diagnosis_fn, hint_composer, **iteration_overrides: object):
    clock = MutableClock()
    orchestrator = EvaluationOrchestrator(
        tmp_path / "evaluation-orchestrator.json",
        evaluation_id="eval-loop-1",
        clock=clock,
        event_id_factory=SequenceIds(),
    )
    handoff = HandoffEnvelopeService(tmp_path / "handoff-envelopes.json", clock=clock)
    experiences = CausalExperiencesStore(tmp_path, config=_CAUSAL, clock=clock)
    ledger = CitationLedger(tmp_path / "citation-ledger.jsonl", clock=clock)
    hints = HintDeliveryService(
        tmp_path / "memory-hints.json",
        ledger,
        clock=clock,
        cooldown_seconds=0,
    )
    loop = MinimalEvaluationLoop(
        orchestrator,
        handoff,
        experiences,
        hints,
        ledger,
        diagnosis_fn,
        hint_composer,
        _CHECKS,
    )
    return {
        "loop": loop,
        "orchestrator": orchestrator,
        "handoff": handoff,
        "experiences": experiences,
        "hints": hints,
        "ledger": ledger,
        "iteration": _iteration(**iteration_overrides),
    }


def _ledger_counts(ledger: CitationLedger) -> dict[str, int]:
    events = ledger.events(require_admin=True)
    return {
        "exposure": sum(1 for event in events if event.kind == KIND_HINT_EXPOSURE),
        "citation": sum(1 for event in events if event.kind == KIND_CITATION),
        "total": len(events),
    }


def test_one_iteration_reaches_stop_and_replay_is_idempotent(tmp_path: Path) -> None:
    diagnosis_calls: list[dict] = []

    def diagnosis_fn(brief_view: dict) -> list[dict]:
        diagnosis_calls.append(brief_view)
        return [_payload(diagnosis_run_id="diag-iter-1")]

    hint_calls: list[tuple] = []

    def hint_composer(revisions):
        hint_calls.append(tuple(revisions))
        return _hint_payload(revisions)

    rig = _wire(tmp_path, diagnosis_fn, hint_composer)
    handoff = rig["handoff"]
    experiences = rig["experiences"]
    hints = rig["hints"]
    ledger = rig["ledger"]
    orchestrator = rig["orchestrator"]

    deliver_calls: list[str] = []
    real_deliver = handoff.deliver

    def wrapped_deliver(envelope, **kwargs):
        deliver_calls.append(getattr(envelope, "iteration_id", ""))
        return real_deliver(envelope, **kwargs)

    handoff.deliver = wrapped_deliver  # type: ignore[method-assign]

    commit_ids: list[str] = []
    real_commit = experiences.commit_provisional

    def wrapped_commit(draft, server_checks):
        committed = real_commit(draft, server_checks)
        commit_ids.append(committed.revision_id)
        return committed

    experiences.commit_provisional = wrapped_commit  # type: ignore[method-assign]

    hint_lineages: list[str | None] = []
    real_hint_deliver = hints.deliver

    def wrapped_hint_deliver(hint, clock=None, *, task_lineage_id=None):
        hint_lineages.append(task_lineage_id)
        return real_hint_deliver(hint, clock, task_lineage_id=task_lineage_id)

    hints.deliver = wrapped_hint_deliver  # type: ignore[method-assign]

    citation_calls: list[dict] = []
    real_cite = ledger.record_citation

    def wrapped_cite(**kwargs):
        citation_calls.append(kwargs)
        return real_cite(**kwargs)

    ledger.record_citation = wrapped_cite  # type: ignore[method-assign]

    first = rig["loop"].run(rig["iteration"])
    assert first.stopped is True
    assert first.state is OrchestratorState.STOPPED
    assert orchestrator.state is OrchestratorState.STOPPED
    assert [event.event for event in orchestrator.events] == [
        EVENT_MEMORY_RETRIEVE,
        EVENT_TASK_RUN,
        EVENT_SCORE,
        EVENT_FREEZE_BRIEF,
        EVENT_DIAGNOSIS,
        EVENT_COMMIT_REVISIONS,
        EVENT_NEXT_RUN,
        EVENT_STOP,
    ]
    assert orchestrator.events[-1].payload_hash == content_address({"reason": "terminal"})
    freeze = next(event for event in orchestrator.events if event.event == EVENT_FREEZE_BRIEF)
    assert freeze.evidence == {"brief_envelope_id": first.brief_envelope_id}
    assert freeze.payload_hash == content_address(
        {
            "iteration_id": "iter-1",
            "brief_envelope_id": first.brief_envelope_id,
            "input_manifest_hash": first.input_manifest_hash,
        }
    )
    diagnosis = next(event for event in orchestrator.events if event.event == EVENT_DIAGNOSIS)
    assert diagnosis.payload_hash == content_address(
        {
            "iteration_id": "iter-1",
            "diagnosis_batch": 1,
            "brief_envelope_id": first.brief_envelope_id,
        }
    )
    committed = next(event for event in orchestrator.events if event.event == EVENT_COMMIT_REVISIONS)
    assert committed.evidence == {"revision_ids": list(first.revision_ids)}
    assert len(first.revision_ids) >= 1

    assert len(handoff.deliveries) == 1
    stored = handoff.deliveries[0]
    assert stored.handoff_kind == "brief"
    assert stored.iteration_id == "iter-1"
    assert stored.idempotency_key == idempotency_key(
        stored.iteration_id, stored.handoff_kind, stored.input_manifest_hash
    )
    envelope = stored.to_dict()["envelope"]
    assert envelope["envelope_id"] == content_address(
        {
            "handoff_kind": envelope["handoff_kind"],
            "iteration_id": envelope["iteration_id"],
            "input_manifest_hash": envelope["input_manifest_hash"],
            "body": envelope["body"],
        }
    )
    brief_body = envelope["body"]
    manifest = {key: value for key, value in brief_body.items() if key != "input_manifest_hash"}
    assert brief_body["input_manifest_hash"] == content_address(manifest)
    assert diagnosis_calls[0]["envelope_id"] == first.brief_envelope_id
    assert diagnosis_calls[0]["brief"]["iteration_id"] == "iter-1"

    loaded = experiences.get_fork_node(first.fork_node_ids[0])
    assert loaded is not None
    assert len(loaded["revisions"]) == 1
    assert loaded["latest_revision"]["status"] == ForkStatus.PROVISIONAL
    rev_dir = (
        tmp_path
        / "causal-experiences"
        / "forks"
        / first.fork_node_ids[0]
        / "revisions"
    )
    assert len(list(rev_dir.glob("*.json"))) == 1
    assert len(list((tmp_path / "causal-experiences" / ".idempotency").glob("*.json"))) == 1
    index_lines = (tmp_path / "causal-experiences" / "index.jsonl").read_text().strip().splitlines()
    assert len(index_lines) == 1

    assert len(hints.hints()) == 1
    assert hints.pending_count("task-1", "agent-memory", "lineage-task-1") == 1
    assert hints.pending_count("task-1", "agent-memory", "other-lineage") == 0
    persisted = json.loads((tmp_path / "memory-hints.json").read_text())
    assert persisted["hint_lineages"][first.hint_id] == "lineage-task-1"
    assert hint_lineages == ["lineage-task-1"]

    counts = _ledger_counts(ledger)
    assert counts["exposure"] == 1
    assert counts["citation"] == 1
    assert len(citation_calls) == 1
    assert citation_calls[0]["role"] == "considered"
    assert citation_calls[0]["ref"]["id"] == first.fork_node_ids[0]
    assert citation_calls[0]["ref"]["revision"] == first.revision_ids[0]

    event_count = len(orchestrator.events)
    second = rig["loop"].run(rig["iteration"])
    assert second.brief_envelope_id == first.brief_envelope_id
    assert second.revision_ids == first.revision_ids
    assert second.fork_node_ids == first.fork_node_ids
    assert second.hint_id == first.hint_id
    assert second.state is OrchestratorState.STOPPED
    assert second.stopped is True
    assert len(orchestrator.events) == event_count
    assert len(handoff.deliveries) == 1
    assert len(deliver_calls) == 2
    assert commit_ids == [first.revision_ids[0], first.revision_ids[0]]
    assert len(list(rev_dir.glob("*.json"))) == 1
    index_lines = (tmp_path / "causal-experiences" / "index.jsonl").read_text().strip().splitlines()
    assert len(index_lines) == 1
    assert len(hints.hints()) == 1
    assert hint_lineages == ["lineage-task-1", "lineage-task-1"]
    assert _ledger_counts(ledger) == counts
    assert len(citation_calls) == 1
    assert len(diagnosis_calls) == 2
    assert len(hint_calls) == 2

    with pytest.raises(ValueError, match="iteration_id"):
        rig["loop"].run(_iteration(iteration_id="iter-2"))
    assert len(handoff.deliveries) == 1
    assert len(orchestrator.events) == event_count


def test_diagnosis_batch_above_seat_limit_does_not_commit(tmp_path: Path) -> None:
    def diagnosis_fn(_brief_view: dict) -> list[dict]:
        raise AssertionError("diagnosis must not run when the seat guard refuses")

    rig = _wire(tmp_path, diagnosis_fn, _hint_payload, diagnosis_batch=4)
    with pytest.raises(NoSeatError) as exc_info:
        rig["loop"].run(rig["iteration"])
    assert exc_info.value.requested == 4
    assert exc_info.value.reason == "diagnosis_batch"
    assert rig["orchestrator"].state is OrchestratorState.FREEZE_BRIEF
    assert [event.event for event in rig["orchestrator"].events] == [
        EVENT_MEMORY_RETRIEVE,
        EVENT_TASK_RUN,
        EVENT_SCORE,
        EVENT_FREEZE_BRIEF,
    ]
    assert len(rig["handoff"].deliveries) == 1
    assert not (tmp_path / "causal-experiences" / "forks").exists()
    assert rig["hints"].hints() == ()
    assert _ledger_counts(rig["ledger"])["total"] == 0


def test_validated_proposal_is_rejected_before_authoritative_write(tmp_path: Path) -> None:
    def diagnosis_fn(_brief_view: dict) -> list[dict]:
        payload = _payload(diagnosis_run_id="diag-iter-1")
        payload["status"] = ForkStatus.VALIDATED
        return [payload]

    rig = _wire(tmp_path, diagnosis_fn, _hint_payload)
    with pytest.raises(ValueError, match="validated"):
        rig["loop"].run(rig["iteration"])
    assert rig["orchestrator"].state is OrchestratorState.DIAGNOSIS
    assert not (tmp_path / "causal-experiences" / "forks").exists()
    assert rig["hints"].hints() == ()

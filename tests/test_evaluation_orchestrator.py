# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""EvaluationOrchestrator state machine (slice S15)."""

from __future__ import annotations

from datetime import datetime, timezone

from openviking.session.evaluation_orchestrator import (
    DIAGNOSIS_AGENT,
    EVENT_COMMIT_REVISIONS,
    EVENT_DIAGNOSIS,
    EVENT_DONE,
    EVENT_FREEZE_BRIEF,
    EVENT_MEMORY_RETRIEVE,
    EVENT_NEXT_RUN,
    EVENT_SCORE,
    EVENT_STOP,
    EVENT_TASK_RUN,
    EVENT_WAIT_EXTERNAL,
    HAPPY_PATH_EVENTS,
    MEMORY_AGENT,
    PROPOSAL_SCHEDULE,
    PROPOSAL_TRIGGER,
    AgentProposal,
    EvaluationOrchestrator,
    IllegalTransition,
    MissingTransitionEvidence,
    NoGainTracker,
    OrchestratorState,
)

T0 = datetime(2026, 9, 24, tzinfo=timezone.utc)


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


def _orchestrator(tmp_path, name: str = "loop.json") -> EvaluationOrchestrator:
    return EvaluationOrchestrator(
        tmp_path / name,
        evaluation_id="eval-1",
        clock=MutableClock(),
        event_id_factory=SequenceIds(),
    )


def _with_evidence(event: str, payload: dict | None = None) -> dict:
    body = dict(payload or {})
    if event == EVENT_FREEZE_BRIEF and not body.get("brief_envelope_id"):
        body["brief_envelope_id"] = "brief-1"
    if event == EVENT_COMMIT_REVISIONS and not body.get("revision_ids") and not body.get("outcome_envelope_id"):
        body["revision_ids"] = ["rev-1"]
    return body


def _to_next_run(orchestrator: EvaluationOrchestrator) -> None:
    for event in HAPPY_PATH_EVENTS[:-1]:
        orchestrator.advance(event, _with_evidence(event))


def test_happy_path_reaches_done(tmp_path) -> None:
    orchestrator = _orchestrator(tmp_path)
    seen = [orchestrator.state]
    for event in HAPPY_PATH_EVENTS:
        record = orchestrator.advance(event, _with_evidence(event, {"step": event}))
        seen.append(orchestrator.state)
        assert record.event == event
        assert record.from_state == seen[-2]
        assert record.to_state == seen[-1]
        assert record.payload_hash
        assert record.occurred_at.endswith("Z")
    assert seen == [
        OrchestratorState.CREATED,
        OrchestratorState.MEMORY_RETRIEVE,
        OrchestratorState.TASK_RUN,
        OrchestratorState.SCORE,
        OrchestratorState.FREEZE_BRIEF,
        OrchestratorState.DIAGNOSIS,
        OrchestratorState.COMMIT_REVISIONS,
        OrchestratorState.NEXT_RUN,
        OrchestratorState.DONE,
    ]
    assert len(orchestrator.events) == len(HAPPY_PATH_EVENTS)


def test_illegal_transition_names_current_and_requested(tmp_path) -> None:
    orchestrator = _orchestrator(tmp_path)
    try:
        orchestrator.advance(EVENT_DONE)
    except IllegalTransition as exc:
        assert exc.current is OrchestratorState.CREATED
        assert exc.requested is OrchestratorState.DONE
    else:
        raise AssertionError("expected IllegalTransition")
    assert orchestrator.state is OrchestratorState.CREATED
    assert orchestrator.events == ()


def test_proposals_are_recorded_and_cannot_advance(tmp_path) -> None:
    orchestrator = _orchestrator(tmp_path)
    trigger = orchestrator.submit_trigger_proposal(MEMORY_AGENT, {"reason": "retrieve"})
    schedule = orchestrator.submit_schedule_proposal(DIAGNOSIS_AGENT, {"when": "after-score"})
    assert orchestrator.state is OrchestratorState.CREATED
    assert trigger.kind == PROPOSAL_TRIGGER
    assert trigger.agent_role == MEMORY_AGENT
    assert schedule.kind == PROPOSAL_SCHEDULE
    assert schedule.agent_role == DIAGNOSIS_AGENT
    assert [item.proposal_id for item in orchestrator.proposals] == [
        trigger.proposal_id,
        schedule.proposal_id,
    ]
    for event in (
        trigger,
        schedule,
        trigger.to_dict(),
        "trigger_proposal",
        "schedule_proposal",
        trigger.proposal_id,
    ):
        try:
            orchestrator.advance(event)
        except IllegalTransition as exc:
            assert exc.current is OrchestratorState.CREATED
        else:
            raise AssertionError(f"expected IllegalTransition for {event!r}")
    assert orchestrator.events == ()
    assert len(orchestrator.proposals) == 2


def test_should_stop_each_condition(tmp_path) -> None:
    terminal = _orchestrator(tmp_path, "terminal.json")
    _to_next_run(terminal)
    assert terminal.should_stop({"terminal": True}) is True
    assert terminal.state is OrchestratorState.STOPPED
    assert terminal.events[-1].event == EVENT_STOP

    ineligible = _orchestrator(tmp_path, "ineligible.json")
    _to_next_run(ineligible)
    assert ineligible.should_stop({"ineligible": True, "evidence_refs": ["e1"]}) is True
    assert ineligible.state is OrchestratorState.STOPPED

    budget = _orchestrator(tmp_path, "budget.json")
    _to_next_run(budget)
    assert budget.should_stop({"budget_exhausted": True}) is True
    assert budget.state is OrchestratorState.STOPPED

    no_gain = _orchestrator(tmp_path, "nogain.json")
    tracker = NoGainTracker()
    _to_next_run(no_gain)
    assert no_gain.should_stop({"evidence_refs": ["e1"], "key_judgment": "same"}, tracker) is False
    assert no_gain.state is OrchestratorState.NEXT_RUN
    no_gain.advance(EVENT_MEMORY_RETRIEVE)
    no_gain.advance(EVENT_TASK_RUN)
    no_gain.advance(EVENT_SCORE)
    no_gain.advance(EVENT_FREEZE_BRIEF, _with_evidence(EVENT_FREEZE_BRIEF))
    no_gain.advance(EVENT_DIAGNOSIS)
    no_gain.advance(EVENT_COMMIT_REVISIONS, _with_evidence(EVENT_COMMIT_REVISIONS))
    no_gain.advance(EVENT_NEXT_RUN)
    assert no_gain.should_stop({"evidence_refs": ["e1"], "key_judgment": "same"}, tracker) is False
    assert tracker.consecutive == 1
    assert no_gain.should_stop({"evidence_refs": ["e1"], "key_judgment": "same"}, tracker) is True
    assert tracker.consecutive == 2
    assert no_gain.state is OrchestratorState.STOPPED

    fresh = _orchestrator(tmp_path, "counter.json")
    _to_next_run(fresh)
    assert fresh.should_stop({"no_gain_streak": 2}) is True
    assert fresh.state is OrchestratorState.STOPPED


def test_waiting_external_resumes_one_shot_run(tmp_path) -> None:
    orchestrator = _orchestrator(tmp_path)
    _to_next_run(orchestrator)
    orchestrator.advance(
        EVENT_WAIT_EXTERNAL,
        {
            "waiting_conditions": [
                {
                    "condition_id": "wc-1",
                    "description": "need an observed branch",
                    "evidence_match": "ref:branch-9",
                }
            ]
        },
    )
    assert orchestrator.state is OrchestratorState.WAITING_EXTERNAL
    assert orchestrator.waiting_conditions[0].condition_id == "wc-1"
    try:
        orchestrator.advance("resume", {"evidence_event": {"ref": "ref:other"}})
    except ValueError:
        pass
    else:
        raise AssertionError("unmatched evidence must not resume")
    assert orchestrator.state is OrchestratorState.WAITING_EXTERNAL
    record = orchestrator.advance("resume", {"evidence_event": {"ref": "ref:branch-9"}})
    assert record.to_state is OrchestratorState.TASK_RUN
    assert orchestrator.state is OrchestratorState.TASK_RUN
    assert orchestrator.satisfied_conditions == ("wc-1",)


def test_state_log_roundtrip_is_lossless(tmp_path) -> None:
    orchestrator = _orchestrator(tmp_path)
    orchestrator.submit_trigger_proposal(MEMORY_AGENT, {"reason": "later", "n": 1})
    orchestrator.advance(EVENT_MEMORY_RETRIEVE, {"plan": "search"})
    _to_next_run_from_memory = (
        EVENT_TASK_RUN,
        EVENT_SCORE,
        EVENT_FREEZE_BRIEF,
        EVENT_DIAGNOSIS,
        EVENT_COMMIT_REVISIONS,
        EVENT_NEXT_RUN,
    )
    for event in _to_next_run_from_memory:
        orchestrator.advance(event, _with_evidence(event))
    orchestrator.advance(
        EVENT_WAIT_EXTERNAL,
        {
            "waiting_conditions": [
                {
                    "condition_id": "wc-2",
                    "description": "wait",
                    "evidence_match": "ref:x",
                }
            ]
        },
    )
    restored = EvaluationOrchestrator(tmp_path / "loop.json", clock=MutableClock(), event_id_factory=SequenceIds())
    assert restored.to_dict() == orchestrator.to_dict()
    again = EvaluationOrchestrator.from_dict(orchestrator.to_dict())
    assert again.to_dict() == orchestrator.to_dict()
    assert again.proposals[0].to_dict()["proposal"] == {"reason": "later", "n": 1}
    assert isinstance(again.proposals[0], AgentProposal)


def test_freeze_and_commit_require_transition_evidence(tmp_path) -> None:
    orchestrator = _orchestrator(tmp_path)
    for event in (EVENT_MEMORY_RETRIEVE, EVENT_TASK_RUN, EVENT_SCORE):
        orchestrator.advance(event)
    try:
        orchestrator.advance(EVENT_FREEZE_BRIEF)
    except MissingTransitionEvidence as exc:
        assert exc.event == EVENT_FREEZE_BRIEF
    else:
        raise AssertionError("expected MissingTransitionEvidence")
    assert orchestrator.state is OrchestratorState.SCORE
    assert [event.event for event in orchestrator.events] == [
        EVENT_MEMORY_RETRIEVE,
        EVENT_TASK_RUN,
        EVENT_SCORE,
    ]

    frozen = orchestrator.advance(EVENT_FREEZE_BRIEF, brief_envelope_id="brief-9")
    assert frozen.to_state is OrchestratorState.FREEZE_BRIEF
    assert frozen.evidence == {"brief_envelope_id": "brief-9"}
    orchestrator.advance(EVENT_DIAGNOSIS)
    try:
        orchestrator.advance(EVENT_COMMIT_REVISIONS)
    except MissingTransitionEvidence as exc:
        assert exc.event == EVENT_COMMIT_REVISIONS
    else:
        raise AssertionError("expected MissingTransitionEvidence")
    assert orchestrator.state is OrchestratorState.DIAGNOSIS
    committed = orchestrator.advance(EVENT_COMMIT_REVISIONS, revision_ids=["rev-9", "rev-10"])
    assert committed.to_state is OrchestratorState.COMMIT_REVISIONS
    assert committed.evidence == {"revision_ids": ["rev-9", "rev-10"]}

    hooked = EvaluationOrchestrator(
        tmp_path / "hooks.json",
        evaluation_id="eval-hooks",
        clock=MutableClock(),
        event_id_factory=SequenceIds(),
        write_hooks={
            "on_brief_envelope": lambda _body: "brief-hook",
            "on_revision_commit": lambda _body: "outcome-hook",
        },
    )
    for event in (EVENT_MEMORY_RETRIEVE, EVENT_TASK_RUN, EVENT_SCORE):
        hooked.advance(event)
    via_hook = hooked.advance(EVENT_FREEZE_BRIEF)
    assert via_hook.evidence == {"brief_envelope_id": "brief-hook"}
    hooked.advance(EVENT_DIAGNOSIS)
    committed_hook = hooked.advance(EVENT_COMMIT_REVISIONS)
    assert committed_hook.evidence == {"outcome_envelope_id": "outcome-hook"}
    assert hooked.state is OrchestratorState.COMMIT_REVISIONS

# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

"""Unit tests for Memory Hint delivery (slice S14)."""

from __future__ import annotations

import inspect
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from openviking.session.citation_ledger import KIND_HINT_EXPOSURE, CitationLedger
from openviking.session.memory_hints import (
    BlockedKindRejected,
    CooldownActive,
    DuplicateDisposition,
    FollowthroughRecord,
    HintDeliveryService,
    MemoryDisposition,
    MemoryHint,
    OutcomeRecord,
    PendingCapExceeded,
    provenance_content_hash,
)

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


class MutableClock:
    def __init__(self, current: datetime) -> None:
        self.current = current

    def __call__(self) -> datetime:
        return self.current


def _source(kind: str = "fork", ref: str = "fork-1", revision: str = "rev-1") -> dict[str, str]:
    return {"kind": kind, "ref": ref, "revision": revision}


def _hint(
    hint_id: str,
    *,
    task_id: str = "task-1",
    target: str = "agent-1",
    sources: list[dict[str, str]] | None = None,
    ttl: datetime | None = None,
    created_at: datetime | None = None,
    provenance: str = "run-gen",
) -> MemoryHint:
    moment = created_at or T0
    return MemoryHint.from_dict(
        {
            "hint_id": hint_id,
            "task_id": task_id,
            "run_id": "run-1",
            "target_agent_id": target,
            "sources": sources or [_source()],
            "match_reason": "anchor overlap",
            "anchor_state_summary": "state-a",
            "applicability": "same task shape",
            "confidence_status": "provisional",
            "provenance": provenance,
            "ttl_expires_at": (ttl or (moment + timedelta(hours=1))).isoformat().replace("+00:00", "Z"),
            "created_at": moment.isoformat().replace("+00:00", "Z"),
        }
    )


def _service(
    tmp_path: Path,
    clock: MutableClock | None = None,
    *,
    cooldown_seconds: float = 300.0,
) -> tuple[HintDeliveryService, CitationLedger, MutableClock]:
    clock = clock or MutableClock(T0)
    ledger = CitationLedger(tmp_path / "citation-ledger.jsonl", clock=clock)
    service = HintDeliveryService(
        tmp_path / "memory-hints.json",
        ledger,
        clock=clock,
        cooldown_seconds=cooldown_seconds,
    )
    return service, ledger, clock


def _exposures(ledger: CitationLedger) -> list:
    return [event for event in ledger.events(require_admin=True) if event.kind == KIND_HINT_EXPOSURE]


def test_pending_cap_blocks_third_until_disposition(tmp_path: Path) -> None:
    service, _, clock = _service(tmp_path)
    first = service.deliver(_hint("h1", sources=[_source(ref="fork-1")]), clock)
    clock.current = T0 + timedelta(seconds=300)
    second = service.deliver(_hint("h2", sources=[_source(ref="fork-2")]), clock)
    clock.current = T0 + timedelta(seconds=600)
    with pytest.raises(PendingCapExceeded) as exc_info:
        service.deliver(_hint("h3", sources=[_source(ref="fork-3")]), clock)
    assert exc_info.value.cap == 2
    assert service.pending_count("task-1", "agent-1") == 2

    service.record_disposition(
        MemoryDisposition(
            hint_id=first.hint_id,
            target_agent_id="agent-1",
            decision="adopt",
            reason="applies",
            intended_action_refs=("action-1",),
            decided_at=clock.current,
        )
    )
    assert service.state(first.hint_id) == "disposed"
    clock.current = T0 + timedelta(seconds=900)
    third = service.deliver(_hint("h3", sources=[_source(ref="fork-3")]), clock)
    assert third.hint_id == "h3"
    assert service.pending_count("task-1", "agent-1") == 2
    assert {hint.hint_id for hint in service.hints()} == {first.hint_id, second.hint_id, third.hint_id}


def test_cooldown_blocks_until_window_elapses(tmp_path: Path) -> None:
    service, _, clock = _service(tmp_path)
    service.deliver(_hint("h1", sources=[_source(ref="fork-1")]), clock)
    clock.current = T0 + timedelta(seconds=100)
    with pytest.raises(CooldownActive) as exc_info:
        service.deliver(_hint("h2", sources=[_source(ref="fork-2")]), clock)
    assert exc_info.value.remaining_seconds == pytest.approx(200.0)

    clock.current = T0 + timedelta(seconds=300)
    delivered = service.deliver(_hint("h2", sources=[_source(ref="fork-2")]), clock)
    assert delivered.hint_id == "h2"


def test_same_provenance_refreshes_ttl_without_new_hint(tmp_path: Path) -> None:
    service, ledger, clock = _service(tmp_path)
    sources = [_source(kind="fork", ref="fork-9"), _source(kind="branch", ref="branch-2", revision="r2")]
    original = service.deliver(_hint("h1", sources=sources, ttl=T0 + timedelta(minutes=10)), clock)
    clock.current = T0 + timedelta(seconds=20)
    refreshed_ttl = T0 + timedelta(hours=2)
    again = service.deliver(
        _hint(
            "h-new",
            sources=list(reversed(sources)),
            ttl=refreshed_ttl,
            provenance="run-other",
        ),
        clock,
    )
    assert again.hint_id == original.hint_id
    assert again.ttl_expires_at == refreshed_ttl
    assert again.provenance == original.provenance
    assert [hint.hint_id for hint in service.hints()] == ["h1"]
    assert len(_exposures(ledger)) == 1
    assert provenance_content_hash(original.sources) == provenance_content_hash(again.sources)


def test_defer_all_records_exposure_and_blocked_kinds_reject(tmp_path: Path) -> None:
    service, ledger, clock = _service(tmp_path)
    service.register_preference("agent-1", defer_all=True)
    delivered = service.deliver(_hint("h1"), clock)
    assert delivered.hint_id == "h1"
    assert len(_exposures(ledger)) == 1
    assert _exposures(ledger)[0].ref.id == "h1"
    dispositions = service.dispositions("h1")
    assert len(dispositions) == 1
    assert dispositions[0].decision == "defer"
    assert dispositions[0].reason == "preference_defer_all"
    assert dispositions[0].target_agent_id == "agent-1"
    assert service.state("h1") == "disposed"
    assert service.pending_count("task-1", "agent-1") == 0

    service.register_preference("agent-2", blocked_kinds=["skill"])
    clock.current = T0 + timedelta(seconds=300)
    before = len(_exposures(ledger))
    with pytest.raises(BlockedKindRejected):
        service.deliver(
            _hint(
                "h-skill",
                target="agent-2",
                sources=[_source(kind="skill", ref="skill-1"), _source(kind="fork", ref="fork-1")],
            ),
            clock,
        )
    assert len(_exposures(ledger)) == before
    assert service.hints() == (delivered,)


def test_events_append_independently_and_ttl_expiry_marks_expired(tmp_path: Path) -> None:
    service, ledger, clock = _service(tmp_path)
    service.deliver(_hint("h1", ttl=T0 + timedelta(minutes=5)), clock)
    disposition = service.record_disposition(
        MemoryDisposition(
            hint_id="h1",
            target_agent_id="agent-1",
            decision="reject",
            reason="out of scope",
            intended_action_refs=(),
            decided_at=T0,
        )
    )
    follow = service.record_followthrough(
        FollowthroughRecord(
            hint_id="h1",
            followthrough="not_observed",
            behavior_evidence_refs=("ev-1",),
            recorded_at=T0 + timedelta(minutes=1),
        )
    )
    outcome = service.record_outcome(
        OutcomeRecord(
            hint_id="h1",
            outcome="neutral",
            evidence_refs=("out-1",),
            recorded_at=T0 + timedelta(minutes=2),
        )
    )
    service.record_followthrough(
        FollowthroughRecord(
            hint_id="h1",
            followthrough="partial",
            behavior_evidence_refs=("ev-2",),
            recorded_at=T0 + timedelta(minutes=3),
        )
    )
    assert service.dispositions("h1") == (disposition,)
    assert service.followthroughs("h1")[0] == follow
    assert service.outcomes("h1") == (outcome,)
    assert len(service.followthroughs("h1")) == 2
    with pytest.raises(ValueError, match="target_agent_id"):
        service.record_disposition(
            MemoryDisposition(
                hint_id="h1",
                target_agent_id="agent-other",
                decision="adopt",
                reason="proxy",
                intended_action_refs=(),
                decided_at=T0,
            )
        )
    assert service.dispositions("h1") == (disposition,)

    clock.current = T0 + timedelta(seconds=300)
    stale = service.deliver(
        _hint("h-stale", sources=[_source(ref="fork-stale")], ttl=T0 + timedelta(seconds=350)),
        clock,
    )
    assert service.expire_stale(clock) == ()
    clock.current = T0 + timedelta(seconds=350)
    assert service.expire_stale(clock) == (stale.hint_id,)
    assert service.state(stale.hint_id) == "expired"
    assert service.state("h1") == "disposed"
    assert service.dispositions("h1") == (disposition,)
    assert len(_exposures(ledger)) == 2

    restored = MemoryHint.from_dict(json.loads(json.dumps(stale.to_dict())))
    assert restored == stale
    assert FollowthroughRecord.from_dict(json.loads(json.dumps(follow.to_dict()))) == follow
    assert OutcomeRecord.from_dict(json.loads(json.dumps(outcome.to_dict()))) == outcome
    assert MemoryDisposition.from_dict(json.loads(json.dumps(disposition.to_dict()))) == disposition

    reloaded = HintDeliveryService(service.path, ledger, clock=clock)
    assert reloaded.state(stale.hint_id) == "expired"
    assert reloaded.dispositions("h1") == (disposition,)
    assert reloaded.followthroughs("h1") == service.followthroughs("h1")
    assert reloaded.outcomes("h1") == (outcome,)


def test_hint_types_expose_no_execute_api() -> None:
    import openviking.session.memory_hints as mod

    forbidden = {"execute", "run", "call"}
    assert forbidden.isdisjoint(set(dir(mod)))
    for cls in (
        MemoryHint,
        MemoryDisposition,
        FollowthroughRecord,
        OutcomeRecord,
        HintDeliveryService,
    ):
        public = {name for name in dir(cls) if not name.startswith("_")}
        assert forbidden.isdisjoint(public)
        for name, member in inspect.getmembers(cls, predicate=callable):
            assert name not in forbidden
    assert "not executable, target agent decides" in (MemoryHint.__doc__ or "")


def test_disposition_is_unique_and_amend_cannot_flip_decision(tmp_path: Path) -> None:
    service, _, clock = _service(tmp_path)
    service.deliver(_hint("h1"), clock)
    first = service.record_disposition(
        MemoryDisposition(
            hint_id="h1",
            target_agent_id="agent-1",
            decision="reject",
            reason="out of scope",
            intended_action_refs=("action-1",),
            decided_at=clock.current,
        )
    )
    with pytest.raises(DuplicateDisposition) as exc_info:
        service.record_disposition(
            MemoryDisposition(
                hint_id="h1",
                target_agent_id="agent-1",
                decision="adopt",
                reason="changed mind",
                intended_action_refs=(),
                decided_at=clock.current,
            )
        )
    assert exc_info.value.existing.decision == "reject"
    assert service.dispositions("h1") == (first,)

    amended = service.amend_disposition(
        "h1",
        reason="wording fix",
        intended_action_refs=("action-2",),
    )
    assert amended.reason == "wording fix"
    visible = service.dispositions("h1")[0]
    assert visible.decision == "reject"
    assert visible.reason == "wording fix"
    assert visible.intended_action_refs == ("action-2",)
    with pytest.raises(ValueError, match="cannot flip decision"):
        service.amend_disposition("h1", reason="nope", decision="adopt")
    assert service.dispositions("h1")[0].decision == "reject"
    assert len(service.amendments("h1")) == 1


def test_silence_records_exposure_without_expecting_disposition(tmp_path: Path) -> None:
    service, ledger, clock = _service(tmp_path)
    preference = service.silence("agent-1")
    assert preference.silence_all is True
    delivered = service.deliver(_hint("h1"), clock)
    assert delivered.hint_id == "h1"
    assert len(_exposures(ledger)) == 1
    assert service.dispositions("h1") == ()
    assert service.pending_count("task-1", "agent-1") == 0
    assert service.state("h1") == "pending"

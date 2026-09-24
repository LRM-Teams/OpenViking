# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Content-addressed diagnosis handoff envelopes (slice S15)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from openviking.session.handoff_envelope import (
    HANDOFF_KIND_BRIEF,
    HANDOFF_KIND_OUTCOME,
    DiagnosisBrief,
    DiagnosisOutcome,
    HandoffEnvelope,
    HandoffEnvelopeService,
    content_address,
)

T0 = datetime(2026, 9, 24, tzinfo=timezone.utc)


class MutableClock:
    def __init__(self) -> None:
        self.current = T0

    def __call__(self) -> datetime:
        return self.current


def _brief(manifest: str = "manifest-a") -> DiagnosisBrief:
    return DiagnosisBrief.from_dict(
        {
            "evaluation_id": "eval-1",
            "task_id": "task-1",
            "source_run_id": "run-1",
            "iteration_id": "iter-1",
            "memory_task_state_hash": "state-hash-1",
            "selected_refs": ["fork:1", "hint:2"],
            "snapshot_watermarks": {"ledger": "wm-9"},
            "policy_versions": {"decay": "v1"},
            "sub_budgets": {"steps": 30},
            "input_manifest_hash": manifest,
        }
    )


def _outcome() -> DiagnosisOutcome:
    return DiagnosisOutcome.from_dict(
        {
            "diagnosis_run_id": "diag-1",
            "input_brief_hash": "brief-hash-1",
            "fork_revision_refs": ["forkrev:1"],
            "bridge_judgment_refs": ["bridge:1"],
            "open_questions": ["why the branch diverged"],
            "waiting_conditions": ["need observed branch"],
            "budget_spent": {"steps": 12},
            "versions": {"diagnosis": "v3"},
        }
    )


def test_repeat_deliver_returns_same_envelope_without_a_second_record(tmp_path) -> None:
    service = HandoffEnvelopeService(tmp_path / "handoff.json", clock=MutableClock())
    brief = _brief()
    first = service.deliver(brief)
    second = service.deliver(HandoffEnvelope.from_brief(brief))
    assert first.envelope_id == second.envelope_id
    assert first.delivered_at == second.delivered_at
    assert first.to_dict() == second.to_dict()
    assert len(service.deliveries) == 1
    assert first.handoff_kind == HANDOFF_KIND_BRIEF
    expected = content_address(
        {
            "handoff_kind": HANDOFF_KIND_BRIEF,
            "iteration_id": brief.iteration_id,
            "input_manifest_hash": brief.input_manifest_hash,
            "body": brief.to_dict(),
        }
    )
    assert first.envelope_id == expected


def test_different_manifest_hash_is_a_new_delivery(tmp_path) -> None:
    service = HandoffEnvelopeService(tmp_path / "handoff.json", clock=MutableClock())
    first = service.deliver(_brief("manifest-a"))
    second = service.deliver(_brief("manifest-b"))
    assert first.envelope_id != second.envelope_id
    assert len(service.deliveries) == 2
    outcome = service.deliver(
        _outcome(),
        iteration_id="iter-1",
        input_manifest_hash="manifest-a",
    )
    assert outcome.handoff_kind == HANDOFF_KIND_OUTCOME
    assert len(service.deliveries) == 3
    again = service.deliver(
        _outcome(),
        iteration_id="iter-1",
        input_manifest_hash="manifest-a",
    )
    assert again.envelope_id == outcome.envelope_id
    assert len(service.deliveries) == 3


def test_credential_field_names_are_rejected() -> None:
    payload = _brief().to_dict()
    payload["policy_versions"] = {"api_token": "v1"}
    with pytest.raises(ValueError, match="credential"):
        DiagnosisBrief.from_dict(payload)
    outcome = _outcome().to_dict()
    outcome["versions"] = {"client_secret": "x"}
    with pytest.raises(ValueError, match="credential"):
        DiagnosisOutcome.from_dict(outcome)
    leaked = _brief().to_dict()
    leaked["api_key"] = "should-not-pass"
    with pytest.raises(ValueError, match="credential"):
        DiagnosisBrief.from_dict(leaked)
    with pytest.raises(ValueError, match="credential"):
        HandoffEnvelope.from_dict(
            {
                "handoff_kind": HANDOFF_KIND_BRIEF,
                "iteration_id": "iter-1",
                "input_manifest_hash": "manifest-a",
                "body": {"access_credential": "nope"},
            }
        )


def test_hash_fields_remain_allowed() -> None:
    brief = _brief()
    assert brief.input_manifest_hash == "manifest-a"
    assert brief.memory_task_state_hash == "state-hash-1"
    assert DiagnosisBrief.from_dict(brief.to_dict()) == brief


def test_ack_is_idempotent(tmp_path) -> None:
    service = HandoffEnvelopeService(tmp_path / "handoff.json", clock=MutableClock())
    delivered = service.deliver(_brief())
    first = service.ack(delivered.envelope_id)
    second = service.ack(delivered.envelope_id)
    assert first.to_dict() == second.to_dict()
    assert first.envelope_id == delivered.envelope_id
    assert len(service.acks) == 1


def test_persistence_roundtrip_is_lossless(tmp_path) -> None:
    path = tmp_path / "handoff.json"
    service = HandoffEnvelopeService(path, clock=MutableClock())
    service.deliver(_brief())
    delivered = service.deliver(_outcome(), iteration_id="iter-9", input_manifest_hash="manifest-z")
    service.ack(delivered.envelope_id)
    restored = HandoffEnvelopeService(path, clock=MutableClock())
    assert restored.to_dict() == service.to_dict()
    rebuilt = HandoffEnvelopeService.from_dict(service.to_dict())
    assert rebuilt.to_dict() == service.to_dict()
    brief_again = DiagnosisBrief.from_dict(_brief().to_dict())
    assert brief_again.to_dict() == _brief().to_dict()
    outcome_again = DiagnosisOutcome.from_dict(_outcome().to_dict())
    assert outcome_again.to_dict() == _outcome().to_dict()

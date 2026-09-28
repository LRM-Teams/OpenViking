# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

"""Positive outcome utility aggregation (Q89-A)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from openviking.session.citation_ledger import (
    KIND_CITATION,
    KIND_HINT_EXPOSURE,
    ROLE_APPLIED,
    CitationEvent,
    CitationRef,
)
from openviking.session.memory_hints import (
    DECISION_ADOPT,
    FollowthroughRecord,
    MemoryDisposition,
    OutcomeRecord,
)
from openviking.session.outcome_utility import positive_outcome_utility

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
POLICY = "memory-hint-v1"


def _disposition(hint_id: str, decision: str = DECISION_ADOPT) -> dict[str, Any]:
    return MemoryDisposition(
        hint_id=hint_id,
        target_agent_id="agent-1",
        decision=decision,
        reason="applies",
        intended_action_refs=("action-1",),
        decided_at=T0,
    ).to_dict()


def _follow(hint_id: str, followthrough: str = "observed") -> dict[str, Any]:
    return FollowthroughRecord(
        hint_id=hint_id,
        followthrough=followthrough,
        behavior_evidence_refs=("beh-1",),
        recorded_at=T0,
    ).to_dict()


def _outcome(
    hint_id: str,
    outcome: str = "positive",
    refs: tuple[str, ...] = ("e1",),
) -> dict[str, Any]:
    return OutcomeRecord(
        hint_id=hint_id,
        outcome=outcome,
        evidence_refs=refs,
        recorded_at=T0,
    ).to_dict()


def _ledger(kind: str, hint_id: str, *, role: str | None) -> dict[str, Any]:
    return CitationEvent(
        event_id=f"{kind}-{hint_id}",
        kind=kind,
        task_id="task-1",
        source_task_id="task-1",
        ref=CitationRef(type="memory_hint", id=hint_id, revision="rev"),
        role=role,
        bridge_id=None,
        source_channel_labels=("memory",),
        occurred_at="2026-01-01T00:00:00.000Z",
        policy_version=POLICY,
    ).to_dict()


def _tag(events: list[dict[str, Any]], policy_version: str) -> list[dict[str, Any]]:
    tagged: list[dict[str, Any]] = []
    for event in events:
        copied = dict(event)
        copied["policy_version"] = policy_version
        tagged.append(copied)
    return tagged


def test_missing_any_of_the_three_is_not_positive() -> None:
    cases = {
        "no adopt": [_follow("h1"), _outcome("h1")],
        "no followthrough": [_disposition("h1"), _outcome("h1")],
        "no outcome": [_disposition("h1"), _follow("h1")],
        "partial followthrough": [_disposition("h1"), _follow("h1", "partial"), _outcome("h1")],
        "not observed": [_disposition("h1"), _follow("h1", "not_observed"), _outcome("h1")],
        "contradicted": [_disposition("h1"), _follow("h1", "contradicted"), _outcome("h1")],
        "reject": [_disposition("h1", "reject"), _follow("h1"), _outcome("h1")],
        "defer": [_disposition("h1", "defer"), _follow("h1"), _outcome("h1")],
        "negative outcome": [
            _disposition("h1"),
            _follow("h1"),
            _outcome("h1", outcome="negative", refs=("e-neg",)),
        ],
        "neutral outcome": [
            _disposition("h1"),
            _follow("h1"),
            _outcome("h1", outcome="neutral", refs=("e-neutral",)),
        ],
    }
    for name, events in cases.items():
        assert positive_outcome_utility(events, POLICY) == {}, name


def test_exposure_and_citation_are_not_adoption() -> None:
    exposure = _ledger(KIND_HINT_EXPOSURE, "h1", role=None)
    exposure["decision"] = DECISION_ADOPT
    exposure["followthrough"] = "observed"
    exposure["outcome"] = "positive"
    exposure["evidence_refs"] = ["from-exposure"]
    citation = _ledger(KIND_CITATION, "h1", role=ROLE_APPLIED)
    citation["decision"] = DECISION_ADOPT
    events = [exposure, citation, _follow("h1"), _outcome("h1", refs=("e1",))]
    assert positive_outcome_utility(events, POLICY) == {}

    counted = positive_outcome_utility(events + [_disposition("h1")], POLICY)
    assert counted["h1"]["positive_count"] == 1
    assert counted["h1"]["evidence_refs"] == ["e1"]
    assert "from-exposure" not in counted["h1"]["evidence_refs"]
    assert "beh-1" not in counted["h1"]["evidence_refs"]


def test_positive_outcomes_accumulate() -> None:
    events = [
        _disposition("h1"),
        _follow("h1"),
        _follow("h1"),
        _outcome("h1", outcome="negative", refs=("e-neg",)),
        _outcome("h1", refs=("e1", "e1")),
        _outcome("h1", refs=("e2",)),
        _disposition("h2"),
        _follow("h2"),
        _outcome("h2", refs=("e9",)),
    ]
    result = positive_outcome_utility(events, POLICY)
    assert list(result) == ["h1", "h2"]
    assert result["h1"]["positive_count"] == 2
    assert result["h1"]["evidence_refs"] == ["e1", "e2"]
    assert result["h1"]["policy_version"] == POLICY
    assert result["h2"]["positive_count"] == 1
    assert result["h2"]["evidence_refs"] == ["e9"]


def test_policy_versions_are_separate_columns() -> None:
    v1 = _tag(
        [_disposition("h1"), _follow("h1"), _outcome("h1", refs=("e1",))],
        "policy-v1",
    )
    v2 = _tag(
        [
            _disposition("h1"),
            _follow("h1"),
            _outcome("h1", refs=("e2",)),
            _outcome("h1", refs=("e3",)),
        ],
        "policy-v2",
    )
    events = v1 + v2 + [_ledger(KIND_HINT_EXPOSURE, "h1", role=None)]
    column_v1 = positive_outcome_utility(events, "policy-v1")
    column_v2 = positive_outcome_utility(events, "policy-v2")
    assert column_v1 == {
        "h1": {
            "positive_count": 1,
            "evidence_refs": ["e1"],
            "policy_version": "policy-v1",
        }
    }
    assert column_v2 == {
        "h1": {
            "positive_count": 2,
            "evidence_refs": ["e2", "e3"],
            "policy_version": "policy-v2",
        }
    }
    assert positive_outcome_utility(events, "policy-other") == {}


def test_utility_is_a_pure_function() -> None:
    events = [_disposition("h1"), _follow("h1"), _outcome("h1", refs=("e1",))]
    snapshot = json.loads(json.dumps(events))
    first = positive_outcome_utility(events, POLICY)
    second = positive_outcome_utility(events, POLICY)
    assert first == second
    assert events == snapshot
    first["h1"]["evidence_refs"].append("mutated")
    again = positive_outcome_utility(events, POLICY)
    assert again["h1"]["evidence_refs"] == ["e1"]

    shifted = json.loads(json.dumps(events))
    shifted[0]["decided_at"] = "1999-01-01T00:00:00Z"
    shifted[2]["recorded_at"] = "2030-01-01T00:00:00Z"
    assert positive_outcome_utility(shifted, POLICY) == again

    custom = [dict(event, object_id=event["hint_id"]) for event in events]
    by_custom = positive_outcome_utility(custom, POLICY, object_key="object_id")
    assert by_custom["h1"]["positive_count"] == 1
    assert positive_outcome_utility(events, POLICY, object_key="not_a_field") == {}

# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

"""Unit tests for the citation ledger (slice S10)."""

from __future__ import annotations

import inspect
import json
import logging
from dataclasses import FrozenInstanceError, fields
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from openviking.session import citation_ledger as citation_ledger_module
from openviking.session.citation_ledger import (
    CitationEvent,
    CitationLedger,
    CitationRef,
    PolicyVersion,
    ReferenceStrength,
)

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
REF = CitationRef(type="memory_hint", id="hint-1", revision="rev-1")
OTHER_REF = CitationRef(type="memory_hint", id="hint-1", revision="rev-2")


class MutableClock:
    def __init__(self, current: datetime) -> None:
        self.current = current

    def __call__(self) -> datetime:
        return self.current


def _ledger(tmp_path: Path, clock: MutableClock | None = None) -> CitationLedger:
    return CitationLedger(tmp_path / "citation-ledger.jsonl", clock=clock or MutableClock(T0))


def _cite(
    ledger: CitationLedger,
    *,
    task_id: str,
    source_task_id: str = "task-src",
    ref: CitationRef = REF,
    role: str = "applied",
    labels: list[str] | None = None,
    policy_version: str = "p-v1",
    bridge_id: str | None = None,
) -> CitationEvent:
    return ledger.record_citation(
        task_id=task_id,
        source_task_id=source_task_id,
        ref=ref,
        role=role,
        source_channel_labels=["chan-a"] if labels is None else labels,
        policy_version=policy_version,
        bridge_id=bridge_id,
    )


def test_same_task_and_cross_task_strengths_are_independent(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    policy = PolicyVersion(policy_id="p-v1")

    _cite(ledger, task_id="task-src", role="applied")
    _cite(ledger, task_id="task-src", role="rejected")
    _cite(ledger, task_id="task-other", role="considered")
    _cite(ledger, task_id="task-src", ref=OTHER_REF, role="compared")

    strength = ledger.materialize_strength(REF, T0, policy, ["chan-a"])

    assert strength.in_task_reference_strength == pytest.approx(2.0)
    assert strength.cross_task_reference_strength == pytest.approx(1.0)
    assert {item.name for item in fields(ReferenceStrength)} == {
        "in_task_reference_strength",
        "cross_task_reference_strength",
    }
    other = ledger.materialize_strength(
        {"type": "memory_hint", "id": "hint-1", "revision": "rev-2"},
        T0,
        policy,
        ["chan-a"],
    )
    assert other.in_task_reference_strength == pytest.approx(1.0)
    assert other.cross_task_reference_strength == pytest.approx(0.0)


def test_decay_follows_half_life_and_policy_recompute_keeps_events(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    policy = PolicyVersion(policy_id="p-v1")
    assert policy.in_task_half_life_days == 7
    assert policy.cross_task_half_life_days == 90

    _cite(ledger, task_id="task-src", role="applied")
    _cite(ledger, task_id="task-other", role="compared")

    at_7 = ledger.materialize_strength(REF, T0 + timedelta(days=7), policy, ["chan-a"])
    assert at_7.in_task_reference_strength == pytest.approx(0.5)
    assert at_7.cross_task_reference_strength == pytest.approx(2 ** (-7 / 90))

    at_90 = ledger.materialize_strength(REF, T0 + timedelta(days=90), policy, ["chan-a"])
    assert at_90.in_task_reference_strength == pytest.approx(2 ** (-90 / 7))
    assert at_90.cross_task_reference_strength == pytest.approx(0.5)

    before_events = [event.to_dict() for event in ledger.events(require_admin=True)]
    before_ids = [id(event) for event in ledger.events(require_admin=True)]
    before_bytes = ledger.path.read_bytes()

    revised = PolicyVersion(
        policy_id="p-revised",
        in_task_half_life_days=14,
        cross_task_half_life_days=180,
    )
    recomputed = ledger.materialize_strength(REF, T0 + timedelta(days=7), revised, ["chan-a"])
    assert recomputed.in_task_reference_strength == pytest.approx(2 ** (-7 / 14))
    assert recomputed.cross_task_reference_strength == pytest.approx(2 ** (-7 / 180))
    assert recomputed.in_task_reference_strength != pytest.approx(at_7.in_task_reference_strength)
    assert recomputed.cross_task_reference_strength != pytest.approx(
        at_7.cross_task_reference_strength
    )

    after = ledger.events(require_admin=True)
    assert [event.to_dict() for event in after] == before_events
    assert [id(event) for event in after] == before_ids
    assert all(event.policy_version == "p-v1" for event in after)
    assert ledger.path.read_bytes() == before_bytes


def test_index_and_hint_exposure_do_not_change_strength(tmp_path: Path) -> None:
    clock = MutableClock(T0)
    ledger = _ledger(tmp_path, clock)
    policy = PolicyVersion(policy_id="p-v1")
    _cite(ledger, task_id="task-src", labels=["chan-a"])
    baseline = ledger.materialize_strength(REF, T0, policy, ["chan-a"])
    visible = ledger.visible_strength(REF, ["chan-a"], policy)

    ledger.record_hint_exposure(
        task_id="task-src",
        source_task_id="task-src",
        ref=REF,
        source_channel_labels=["chan-a"],
        policy_version="p-v1",
    )
    ledger.record_index_level_exposure(
        task_id="task-src",
        source_task_id="task-src",
        ref=REF,
        source_channel_labels=["chan-a"],
        policy_version="p-v1",
    )
    ledger.record_index_level_exposure(
        task_id="task-other",
        source_task_id="task-src",
        ref=REF,
        source_channel_labels=["chan-a"],
        policy_version="p-v1",
    )
    ledger.record_index_level_exposure(
        task_id="task-src",
        source_task_id="task-src",
        ref=OTHER_REF,
        source_channel_labels=["chan-a"],
        policy_version="p-v1",
    )

    assert ledger.materialize_strength(REF, T0, policy, ["chan-a"]) == baseline
    assert ledger.visible_strength(REF, ["chan-a"], policy) == visible
    assert ledger.count_index_exposures(REF) == 2
    assert ledger.count_index_exposures(OTHER_REF) == 1


def test_citation_without_bridge_id_does_not_count_for_bridge(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    _cite(ledger, task_id="task-src", bridge_id="bridge-1", role="applied")
    _cite(ledger, task_id="task-other", role="considered")
    _cite(ledger, task_id="task-src", bridge_id="", role="rejected")
    _cite(ledger, task_id="task-src", bridge_id="bridge-1", role="compared")
    _cite(ledger, task_id="task-remote", bridge_id="bridge-2", role="applied")
    ledger.record_hint_exposure(
        task_id="task-src",
        source_task_id="task-src",
        ref=REF,
        source_channel_labels=["chan-a"],
        policy_version="p-v1",
    )

    assert ledger.bridge_citation_count("bridge-1") == 2
    assert ledger.bridge_citation_count("bridge-2") == 1
    assert ledger.bridge_citation_count("bridge-missing") == 0

    strength = ledger.materialize_strength(
        REF, T0, PolicyVersion(policy_id="p-v1"), ["chan-a"]
    )
    assert strength.in_task_reference_strength == pytest.approx(3.0)
    assert strength.cross_task_reference_strength == pytest.approx(2.0)


def test_visible_strength_uses_only_intersecting_labels(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    policy = PolicyVersion(policy_id="p-v1")
    _cite(ledger, task_id="task-src", labels=["alpha"], role="applied")
    _cite(ledger, task_id="task-other", labels=["beta"], role="considered")
    _cite(ledger, task_id="task-src", labels=["alpha", "beta"], role="rejected")
    _cite(ledger, task_id="task-remote", labels=[], role="compared")

    alpha = ledger.visible_strength(REF, ["alpha"], policy)
    beta = ledger.visible_strength(REF, ["beta"], policy)
    both = ledger.visible_strength(REF, ["alpha", "beta"], policy)
    none = ledger.visible_strength(REF, ["gamma"], policy)

    assert alpha.in_task_reference_strength == pytest.approx(2.0)
    assert alpha.cross_task_reference_strength == pytest.approx(0.0)
    assert beta.in_task_reference_strength == pytest.approx(1.0)
    assert beta.cross_task_reference_strength == pytest.approx(1.0)
    assert both.in_task_reference_strength == pytest.approx(2.0)
    assert both.cross_task_reference_strength == pytest.approx(1.0)
    assert none.in_task_reference_strength == pytest.approx(0.0)
    assert none.cross_task_reference_strength == pytest.approx(0.0)

    for strength in (alpha, beta, both, none):
        payload = strength.to_dict()
        assert set(payload) == {
            "in_task_reference_strength",
            "cross_task_reference_strength",
        }
        assert not any("hidden" in key or "count" in key or "delta" in key for key in payload)


def test_jsonl_roundtrip_is_lossless(tmp_path: Path) -> None:
    clock = MutableClock(T0)
    path = tmp_path / "session"
    ledger = CitationLedger(path, clock=clock)
    labels = ["chan-a", "chan-b"]
    first = ledger.record_citation(
        task_id="task-src",
        source_task_id="task-src",
        ref={"type": "fork", "id": "f1", "revision": "r9"},
        role="applied",
        source_channel_labels=labels,
        policy_version="p-v1",
        bridge_id="bridge-9",
    )
    labels.append("mutated")
    assert first.source_channel_labels == ("chan-a", "chan-b")
    with pytest.raises(FrozenInstanceError):
        first.role = "rejected"  # type: ignore[misc]

    first_text = ledger.path.read_text(encoding="utf-8")
    clock.current = T0 + timedelta(days=1)
    ledger.record_hint_exposure(
        task_id="task-other",
        source_task_id="task-src",
        ref=REF,
        source_channel_labels=["chan-a"],
        policy_version="p-v1",
    )
    ledger.record_index_level_exposure(
        task_id="task-other",
        source_task_id="task-src",
        ref=REF,
        source_channel_labels=["chan-b"],
        policy_version="p-v1",
    )
    second_text = ledger.path.read_text(encoding="utf-8")
    assert second_text.startswith(first_text)
    assert first_text in second_text

    reloaded = CitationLedger(path, clock=clock)
    assert reloaded.events(require_admin=True) == ledger.events(require_admin=True)
    for event in reloaded.events(require_admin=True):
        restored = CitationEvent.from_dict(json.loads(json.dumps(event.to_dict())))
        assert restored == event

    policy = PolicyVersion(policy_id="p-custom", in_task_half_life_days=3, cross_task_half_life_days=10)
    assert PolicyVersion.from_dict(json.loads(json.dumps(policy.to_dict()))) == policy
    assert [event.kind for event in reloaded.events(require_admin=True)] == [
        "citation",
        "hint_exposure",
        "index_level_exposure",
    ]
    assert reloaded.events(require_admin=True)[1].occurred_at != reloaded.events(require_admin=True)[0].occurred_at


def test_materialize_strength_rejects_none_and_events_for_is_scoped(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    policy = PolicyVersion(policy_id="p-v1")
    _cite(ledger, task_id="task-src", labels=["alpha"])
    _cite(ledger, task_id="task-other", labels=["beta"])
    _cite(ledger, task_id="task-remote", labels=[])

    with pytest.raises(ValueError, match="visible_strength"):
        ledger.materialize_strength(REF, T0, policy, None)

    alpha = ledger.events_for(["alpha"])
    beta = ledger.events_for(["beta"])
    assert [event.source_channel_labels for event in alpha] == [("alpha",)]
    assert [event.source_channel_labels for event in beta] == [("beta",)]
    assert ledger.events_for(["gamma"]) == []
    assert ledger.events_for([]) == []
    assert ledger.events_for(["alpha", "beta"]) == alpha + beta

    with pytest.raises(PermissionError, match="offline/admin only"):
        ledger.events()
    assert len(ledger.events(require_admin=True)) == 3


def test_offline_counts_warn_once(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    citation_ledger_module._OFFLINE_WARNED.clear()
    ledger = _ledger(tmp_path)
    _cite(ledger, task_id="task-src", bridge_id="bridge-1")
    ledger.record_index_level_exposure(
        task_id="task-src",
        source_task_id="task-src",
        ref=REF,
        source_channel_labels=["chan-a"],
        policy_version="p-v1",
    )
    with caplog.at_level(logging.WARNING):
        assert ledger.count_index_exposures(REF) == 1
        assert ledger.count_index_exposures(REF, offline_only=True) == 1
        assert ledger.bridge_citation_count("bridge-1") == 1
        assert ledger.bridge_citation_count("bridge-1", offline_only=True) == 1
    warnings = [record.message for record in caplog.records if record.levelno == logging.WARNING]
    assert sum("count_index_exposures" in message for message in warnings) == 1
    assert sum("bridge_citation_count" in message for message in warnings) == 1


def test_fsync_on_append_reloads_complete_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fsync_calls: list[int] = []
    monkeypatch.setattr(citation_ledger_module.os, "fsync", lambda fd: fsync_calls.append(fd))
    assert (
        inspect.signature(CitationLedger.__init__).parameters["fsync_on_append"].default is True
    )

    batch = 4
    for enabled in (True, False):
        path = tmp_path / f"fsync-{enabled}.jsonl"
        ledger = CitationLedger(path, clock=MutableClock(T0), fsync_on_append=enabled)
        before = len(fsync_calls)
        for index in range(batch):
            _cite(ledger, task_id=f"task-{index}")
        assert len(fsync_calls) - before == (batch if enabled else 0)
        reloaded = CitationLedger(path, clock=MutableClock(T0))
        assert len(reloaded.events(require_admin=True)) == batch

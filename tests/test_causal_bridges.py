# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Unit tests for HypothesisBridge / EvidenceBridge governance (slice S9)."""

from __future__ import annotations

from pathlib import Path

import pytest

from openviking.session.causal_bridges import (
    PATH_MAX_ENTITIES,
    PATH_MAX_HYPOTHESIS_HOPS,
    PER_NODE_ACTIVE_CAP,
    PER_RUN_WRITE_QUOTA,
    BridgeRecord,
    CausalBridgeStore,
    HypothesisJudgment,
    QuotaExceeded,
    check_path_budget,
)

_CAUSAL_CONFIG = {"skill_trajectory_mode": "causal"}
_SOURCE = {"type": "fork", "id": "fork-src"}


def _target(index: int) -> dict[str, str]:
    return {"type": "memory", "id": f"mem-{index}"}


def _store(tmp_path: Path, **kwargs: object) -> CausalBridgeStore:
    return CausalBridgeStore(tmp_path, _CAUSAL_CONFIG, **kwargs)  # type: ignore[arg-type]


def _judge(
    store: CausalBridgeStore,
    *,
    index: int,
    run_id: str,
    relevance: float,
    relation_type: str = "explores",
    source: dict[str, str] | None = None,
    judgment_id: str | None = None,
    timestamp: str | None = None,
) -> BridgeRecord:
    return store.record_judgment(
        run_id=run_id,
        source_ref=source or _SOURCE,
        target_ref=_target(index),
        relation_type=relation_type,
        relevance=relevance,
        timestamp=timestamp or f"2026-09-24T00:00:{index:02d}.000Z",
        judgment_id=judgment_id or f"j-{index}",
        direct_relevance=relevance,
    )


def test_same_endpoints_collapse_to_one_bridge_and_keep_every_judgment(tmp_path: Path) -> None:
    store = _store(tmp_path)
    first = _judge(store, index=1, run_id="run-a", relevance=0.25, judgment_id="j-1")
    second = store.record_judgment(
        run_id="run-b",
        source_ref=_SOURCE,
        target_ref=_target(1),
        relation_type="explores",
        relevance=0.75,
        timestamp="2026-09-24T00:01:00.000Z",
        judgment_id="j-2",
        direct_relevance=0.75,
    )
    other_relation = _judge(
        store, index=1, run_id="run-c", relevance=0.5, relation_type="cites", judgment_id="j-3"
    )

    assert first.bridge_id == second.bridge_id
    assert first.kind == "hypothesis"
    assert other_relation.bridge_id != first.bridge_id
    judgments = store.judgments_for(first.bridge_id)
    assert [item.judgment_id for item in judgments] == ["j-1", "j-2"]
    assert [item.relevance for item in judgments] == [0.25, 0.75]
    assert judgments[0].canonical_key == first.canonical_key == second.canonical_key
    assert store.find_bridge(_SOURCE, _target(1), "explores") == second


def test_run_quota_rejects_thirteenth_write_and_node_cap_inactivates_lowest(
    tmp_path: Path,
) -> None:
    assert PER_RUN_WRITE_QUOTA == 12
    assert PER_NODE_ACTIVE_CAP == 32
    quota_store = _store(tmp_path / "quota")
    for index in range(PER_RUN_WRITE_QUOTA):
        _judge(quota_store, index=index, run_id="iter-1", relevance=float(index + 1))
    with pytest.raises(QuotaExceeded) as raised:
        _judge(quota_store, index=99, run_id="iter-1", relevance=1.0)
    assert raised.value.quota == PER_RUN_WRITE_QUOTA
    assert quota_store.find_bridge(_SOURCE, _target(99), "explores") is None
    assert len(quota_store.list_bridges()) == PER_RUN_WRITE_QUOTA

    cap_store = _store(tmp_path / "cap")
    created: list[BridgeRecord] = []
    for index in range(PER_NODE_ACTIVE_CAP + 1):
        created.append(
            _judge(
                cap_store,
                index=index,
                run_id=f"iter-{index // PER_RUN_WRITE_QUOTA}",
                relevance=1.0 if index == PER_NODE_ACTIVE_CAP else float(index + 10),
            )
        )
    lowest = cap_store.get_bridge(created[-1].bridge_id)
    assert lowest is not None
    assert lowest.status == "inactive"
    assert lowest.governance_score == 1.0
    visible = {bridge.bridge_id for bridge in cap_store.traversable_edges(_SOURCE)}
    assert lowest.bridge_id not in visible
    assert len(visible) == PER_NODE_ACTIVE_CAP
    assert [item.judgment_id for item in cap_store.judgments_for(lowest.bridge_id)] == [
        f"j-{PER_NODE_ACTIVE_CAP}"
    ]
    assert sum(len(cap_store.judgments_for(bridge.bridge_id)) for bridge in created) == 33


def test_exposure_cannot_reactivate_but_new_judgment_can_when_capacity_allows(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    created: list[BridgeRecord] = []
    for index in range(PER_NODE_ACTIVE_CAP + 1):
        created.append(
            _judge(
                store,
                index=index,
                run_id=f"iter-{index // PER_RUN_WRITE_QUOTA}",
                relevance=1.0 if index == PER_NODE_ACTIVE_CAP else float(index + 10),
            )
        )
    dormant = store.get_bridge(created[-1].bridge_id)
    assert dormant is not None and dormant.status == "inactive"

    with pytest.raises(ValueError, match="exposure"):
        store.reactivate(dormant.bridge_id, "exposure")
    with pytest.raises(ValueError, match="traversal"):
        store.reactivate(dormant.bridge_id, "traversal")
    assert store.get_bridge(dormant.bridge_id) == dormant

    restored = store.reactivate(
        dormant.bridge_id,
        "new_judgment",
        direct_relevance=100.0,
        at="2026-09-24T01:00:00.000Z",
    )
    assert restored.status == "reactivated"
    assert restored.status_events[-1].signal == "new_judgment"
    visible = {bridge.bridge_id for bridge in store.traversable_edges(_SOURCE)}
    assert restored.bridge_id in visible
    assert len(visible) == PER_NODE_ACTIVE_CAP
    displaced = [bridge for bridge in store.list_bridges() if bridge.status == "inactive"]
    assert len(displaced) == 1
    assert displaced[0].bridge_id != restored.bridge_id
    assert store.judgments_for(restored.bridge_id)


def test_promotion_requires_direct_evidence_and_switches_kind(tmp_path: Path) -> None:
    store = _store(tmp_path)
    bridge = _judge(store, index=1, run_id="run-1", relevance=2.0)
    evidence = {"type": "evidence", "id": "ev-1"}
    check = {"checked": True, "result": "no counterexample"}

    with pytest.raises(ValueError, match="evidence_refs"):
        store.promote_to_evidence(bridge.bridge_id, [], "supports", check)
    with pytest.raises(ValueError, match="relation_semantics"):
        store.promote_to_evidence(bridge.bridge_id, [evidence], "", check)
    with pytest.raises(ValueError, match="counter_example_check"):
        store.promote_to_evidence(bridge.bridge_id, [evidence], "supports", {})
    assert not hasattr(store, "promote_by_citation_count")

    promoted = store.promote_to_evidence(
        bridge.bridge_id, [evidence], "supports", check
    )
    assert promoted.kind == "evidence"
    assert promoted.relation_semantics == "supports"
    assert promoted.evidence_refs[0].to_dict() == evidence
    assert promoted.counter_example_check == check
    assert store.get_bridge(bridge.bridge_id) == promoted
    assert store.get_bridge(bridge.bridge_id).kind == "evidence"  # type: ignore[union-attr]


def test_path_budget_rejects_three_hops_thirteen_entities_and_duplicate_nodes() -> None:
    assert PATH_MAX_HYPOTHESIS_HOPS == 2
    assert PATH_MAX_ENTITIES == 12
    entities = [f"e{index}" for index in range(4)]
    three_hops = [
        {"kind": "hypothesis", "bridge_id": "h1"},
        {"kind": "hypothesis", "bridge_id": "h2"},
        {"kind": "evidence", "bridge_id": "e1"},
        {"kind": "hypothesis", "bridge_id": "h3"},
    ]
    hop_violations = check_path_budget(entities, three_hops)
    assert [item.code for item in hop_violations] == ["hypothesis_hops"]

    too_many = [f"n{index}" for index in range(13)]
    entity_violations = check_path_budget(too_many, [])
    assert [item.code for item in entity_violations] == ["entity_count"]

    dup_violations = check_path_budget(
        ["a", "b", "a"],
        [{"kind": "hypothesis", "bridge_id": "only"}],
    )
    assert [item.code for item in dup_violations] == ["duplicate_node"]

    edge_violations = check_path_budget(
        ["a", "b", "c"],
        [
            {"kind": "hypothesis", "bridge_id": "same"},
            {"kind": "hypothesis", "bridge_id": "same"},
        ],
    )
    assert any(item.code == "duplicate_edge" for item in edge_violations)

    ok = check_path_budget(
        [f"n{index}" for index in range(PATH_MAX_ENTITIES)],
        [
            {"kind": "hypothesis", "bridge_id": "h1"},
            {"kind": "evidence", "bridge_id": "ev"},
            {"kind": "hypothesis", "bridge_id": "h2"},
        ],
    )
    assert ok == []


def test_persistence_roundtrip_is_lossless(tmp_path: Path) -> None:
    store = _store(tmp_path, per_node_active_cap=1)
    first = _judge(store, index=1, run_id="run-1", relevance=1.0, judgment_id="j-keep")
    second_judgment = store.record_judgment(
        run_id="run-1",
        source_ref=_SOURCE,
        target_ref=_target(1),
        relation_type="explores",
        relevance=4.0,
        timestamp="2026-09-24T00:00:30.000Z",
        judgment_id="j-keep-2",
        direct_relevance=4.0,
    )
    assert second_judgment.bridge_id == first.bridge_id
    rival = _judge(store, index=2, run_id="run-2", relevance=8.0, judgment_id="j-rival")
    assert store.get_bridge(first.bridge_id).status == "inactive"  # type: ignore[union-attr]
    restored = store.reactivate(
        first.bridge_id,
        "citation_with_bridge_id",
        direct_relevance=9.0,
        at="2026-09-24T02:00:00.000Z",
    )
    assert restored.status == "reactivated"
    promoted = store.promote_to_evidence(
        rival.bridge_id,
        [{"type": "evidence", "id": "ev-9"}],
        "explains",
        "counterexample search found none",
    )

    reloaded = _store(tmp_path, per_node_active_cap=1)
    assert reloaded.get_bridge(restored.bridge_id) == restored
    assert reloaded.get_bridge(promoted.bridge_id) == promoted
    assert reloaded.judgments_for(restored.bridge_id) == store.judgments_for(restored.bridge_id)
    assert [item.to_dict() for item in reloaded.list_bridges()] == [
        item.to_dict() for item in store.list_bridges()
    ]
    assert BridgeRecord.from_dict(restored.to_dict()) == restored
    judgment = store.judgments_for(restored.bridge_id)[0]
    assert HypothesisJudgment.from_dict(judgment.to_dict()) == judgment
    assert reloaded.traversable_edges(_SOURCE) == store.traversable_edges(_SOURCE)

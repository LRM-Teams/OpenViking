# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Unit tests for verified similarity edges and causal clusters (slice S6/6)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openviking.session.causal_clusters import (
    CLUSTERS_DIRNAME,
    EDGES_DIRNAME,
    EDGES_INDEX_FILENAME,
    CausalClusterStore,
    SimilarityJudgment,
    VerifiedSimilarityEdge,
    canonical_pair_key,
)
from openviking.session.causal_experiences import NAMESPACE_DIRNAME

_CAUSAL_CONFIG = {"skill_trajectory_mode": "causal"}
_LEGACY_CONFIG = {"skill_trajectory_mode": "legacy"}

_FORK_A = "fork-aaa"
_FORK_B = "fork-bbb"
_FORK_C = "fork-ccc"
_REV_A = "rev-a1"
_REV_B = "rev-b1"
_REV_C = "rev-c1"


def _criteria(**overrides: object) -> dict:
    criteria: dict = {
        "task_decision_type": "same recovery decision",
        "anchor_state": "blocked on dirty worktree",
        "failure_mode": "rebase conflict",
        "action_skill_role": "rescue orchestrator",
    }
    criteria.update(overrides)
    return criteria


def _judgment(**overrides: object) -> dict:
    judgment: dict = {
        "judged_by_model_version": "judge-model-v1",
        "prompt_policy_version": "sim-policy-v1",
        "criteria_per_dimension": _criteria(),
        "verdict": "consistent",
        "confidence": 0.9,
        "rationale": "same decision type and failure mechanism",
        "evidence_revision_hashes": ["hash-a", "hash-b"],
        "judged_at": "2026-09-23T12:00:00.000Z",
        "diagnosis_run_id": "diag-1",
    }
    judgment.update(overrides)
    return judgment


def _endpoint(fork_node_id: str, revision_id: str) -> dict[str, str]:
    return {"fork_node_id": fork_node_id, "revision_id": revision_id}


def _edge_payload(
    *,
    fork_a: str = _FORK_A,
    rev_a: str = _REV_A,
    fork_b: str = _FORK_B,
    rev_b: str = _REV_B,
    **overrides: object,
) -> dict:
    payload: dict = {
        "endpoint_a": _endpoint(fork_a, rev_a),
        "endpoint_b": _endpoint(fork_b, rev_b),
        "judgment": _judgment(),
        "direction": "undirected",
        "created_at": "2026-09-23T12:00:00.000Z",
    }
    payload.update(overrides)
    return payload


def _store(tmp_path: Path, config: dict | None = None) -> CausalClusterStore:
    return CausalClusterStore(tmp_path, config=config if config is not None else _CAUSAL_CONFIG)


def _submit_positive(
    store: CausalClusterStore,
    *,
    fork_a: str,
    fork_b: str,
    key: str,
    verdict: str = "consistent",
    rationale: str = "same decision type and failure mechanism",
    rev_a: str | None = None,
    rev_b: str | None = None,
) -> dict:
    return store.submit_similarity_edge(
        _edge_payload(
            fork_a=fork_a,
            rev_a=rev_a or f"rev-{fork_a}",
            fork_b=fork_b,
            rev_b=rev_b or f"rev-{fork_b}",
            judgment=_judgment(verdict=verdict, rationale=rationale, diagnosis_run_id=key),
        ),
        idempotency_key=key,
    )


def test_judgment_rejects_missing_dimension_illegal_verdict_and_confidence() -> None:
    incomplete = _criteria()
    del incomplete["failure_mode"]
    with pytest.raises(ValueError, match="criteria_per_dimension"):
        SimilarityJudgment.from_dict(_judgment(criteria_per_dimension=incomplete))

    with pytest.raises(ValueError, match="verdict"):
        SimilarityJudgment.from_dict(_judgment(verdict="similar"))

    with pytest.raises(ValueError, match="confidence"):
        SimilarityJudgment.from_dict(_judgment(confidence=1.5))
    with pytest.raises(ValueError, match="confidence"):
        SimilarityJudgment.from_dict(_judgment(confidence=-0.01))


def test_positive_edges_index_and_negative_samples_excluded(tmp_path: Path) -> None:
    store = _store(tmp_path)
    ns = tmp_path / NAMESPACE_DIRNAME

    consistent = store.submit_similarity_edge(
        _edge_payload(judgment=_judgment(verdict="consistent")),
        idempotency_key="edge-consistent",
    )
    analogous = store.submit_similarity_edge(
        _edge_payload(
            fork_a=_FORK_A,
            rev_a=_REV_A,
            fork_b=_FORK_C,
            rev_b=_REV_C,
            judgment=_judgment(verdict="analogous", diagnosis_run_id="diag-2"),
        ),
        idempotency_key="edge-analogous",
    )
    assert consistent["negative"] is False
    assert analogous["negative"] is False
    assert consistent["created"] is True
    assert analogous["created"] is True

    dissimilar = store.submit_similarity_edge(
        _edge_payload(
            fork_a=_FORK_B,
            fork_b=_FORK_C,
            judgment=_judgment(verdict="dissimilar", diagnosis_run_id="diag-neg-1"),
        ),
        idempotency_key="edge-dissimilar",
    )
    insufficient = store.submit_similarity_edge(
        _edge_payload(
            fork_a=_FORK_B,
            rev_a="rev-b2",
            fork_b=_FORK_C,
            rev_b="rev-c2",
            judgment=_judgment(verdict="insufficient", diagnosis_run_id="diag-neg-2"),
        ),
        idempotency_key="edge-insufficient",
    )
    assert dissimilar["negative"] is True
    assert insufficient["negative"] is True

    for result in (consistent, analogous):
        path = ns / EDGES_DIRNAME / f"{result['edge_id']}.json"
        assert path.is_file()
        body = json.loads(path.read_text(encoding="utf-8"))
        assert body["negative"] is False

    for result in (dissimilar, insufficient):
        path = ns / EDGES_DIRNAME / f"{result['edge_id']}.json"
        assert path.is_file()
        body = json.loads(path.read_text(encoding="utf-8"))
        assert body["negative"] is True

    index_text = (ns / EDGES_INDEX_FILENAME).read_text(encoding="utf-8")
    assert consistent["edge_id"] in index_text
    assert analogous["edge_id"] in index_text
    assert dissimilar["edge_id"] not in index_text
    assert insufficient["edge_id"] not in index_text

    assert store.edges_between(_FORK_B, _FORK_C) == []
    negatives = store.edges_between(_FORK_B, _FORK_C, include_negative=True)
    assert {item["edge_id"] for item in negatives} == {
        dissimilar["edge_id"],
        insufficient["edge_id"],
    }
    assert all(item["negative"] is True for item in negatives)

    with pytest.raises(ValueError, match="positive"):
        store.create_cluster(_FORK_B, [_FORK_B, _FORK_C], dissimilar["edge_id"])


def test_endpoint_canonicalization_dedup_and_new_judgment_revision(tmp_path: Path) -> None:
    store = _store(tmp_path)
    forward = store.submit_similarity_edge(_edge_payload(), idempotency_key="pair-ab")
    assert forward["created"] is True

    pair = canonical_pair_key(_endpoint(_FORK_A, _REV_A), _endpoint(_FORK_B, _REV_B))
    reverse_same = store.submit_similarity_edge(
        _edge_payload(
            fork_a=_FORK_B,
            rev_a=_REV_B,
            fork_b=_FORK_A,
            rev_b=_REV_A,
            created_at="2026-09-23T13:00:00.000Z",
        ),
        idempotency_key="pair-ba-same",
    )
    assert reverse_same["deduplicated"] is True
    assert reverse_same["edge_id"] == forward["edge_id"]
    assert canonical_pair_key(_endpoint(_FORK_B, _REV_B), _endpoint(_FORK_A, _REV_A)) == pair

    reverse_new = store.submit_similarity_edge(
        _edge_payload(
            fork_a=_FORK_B,
            rev_a=_REV_B,
            fork_b=_FORK_A,
            rev_b=_REV_A,
            judgment=_judgment(rationale="updated analogous reading", verdict="analogous"),
            created_at="2026-09-23T14:00:00.000Z",
        ),
        idempotency_key="pair-ba-new",
    )
    assert reverse_new["created"] is True
    assert reverse_new["deduplicated"] is False
    assert reverse_new["edge_id"] != forward["edge_id"]

    edges = store.edges_between(_FORK_A, _FORK_B)
    assert {item["edge_id"] for item in edges} == {forward["edge_id"], reverse_new["edge_id"]}
    for item in edges:
        assert item["endpoint_a"]["fork_node_id"] <= item["endpoint_b"]["fork_node_id"] or (
            item["endpoint_a"]["fork_node_id"] == item["endpoint_b"]["fork_node_id"]
            and item["endpoint_a"]["revision_id"] <= item["endpoint_b"]["revision_id"]
        )
        restored = VerifiedSimilarityEdge.from_dict(item)
        assert restored.pair_key == pair


def test_create_cluster_requires_positive_edge_and_indexes_fork(tmp_path: Path) -> None:
    store = _store(tmp_path)
    with pytest.raises(ValueError, match="unknown initiating_edge_id"):
        store.create_cluster(_FORK_A, [_FORK_A, _FORK_B], "missing-edge")

    negative = store.submit_similarity_edge(
        _edge_payload(judgment=_judgment(verdict="dissimilar")),
        idempotency_key="neg-for-cluster",
    )
    with pytest.raises(ValueError, match="positive"):
        store.create_cluster(_FORK_A, [_FORK_A, _FORK_B], negative["edge_id"])

    submitted = _submit_positive(store, fork_a=_FORK_A, fork_b=_FORK_B, key="pos-ab")
    missing_member = store.submit_similarity_edge(
        _edge_payload(
            fork_a=_FORK_A,
            fork_b=_FORK_C,
            judgment=_judgment(verdict="insufficient", diagnosis_run_id="diag-ac-neg"),
        ),
        idempotency_key="neg-ac",
    )
    assert missing_member["negative"] is True
    with pytest.raises(ValueError, match="no positive verified edge"):
        store.create_cluster(_FORK_A, [_FORK_A, _FORK_C], submitted["edge_id"])

    cluster = store.create_cluster(_FORK_A, [_FORK_A, _FORK_B], submitted["edge_id"])
    assert cluster.status == "active"
    assert cluster.center_fork_node_id == _FORK_A
    assert cluster.member_fork_node_ids == [_FORK_A, _FORK_B]
    assert cluster.version == 1
    assert (tmp_path / NAMESPACE_DIRNAME / CLUSTERS_DIRNAME / f"{cluster.cluster_id}.json").is_file()

    hits_a = store.clusters_for_fork(_FORK_A)
    hits_b = store.clusters_for_fork(_FORK_B)
    assert [item["cluster_id"] for item in hits_a] == [cluster.cluster_id]
    assert [item["cluster_id"] for item in hits_b] == [cluster.cluster_id]
    assert store.clusters_for_fork(_FORK_C) == []


def test_add_member_rejects_unlinked_and_retains_history(tmp_path: Path) -> None:
    store = _store(tmp_path)
    edge_ab = _submit_positive(store, fork_a=_FORK_A, fork_b=_FORK_B, key="pos-ab")
    cluster = store.create_cluster(_FORK_A, [_FORK_A, _FORK_B], edge_ab["edge_id"])

    with pytest.raises(ValueError, match="no positive verified edge"):
        store.add_member(cluster.cluster_id, _FORK_C, edge_ab["edge_id"])

    edge_ac = _submit_positive(store, fork_a=_FORK_A, fork_b=_FORK_C, key="pos-ac")
    updated = store.add_member(cluster.cluster_id, _FORK_C, edge_ac["edge_id"])
    assert updated.cluster_id == cluster.cluster_id
    assert updated.version == 2
    assert updated.member_fork_node_ids == [_FORK_A, _FORK_B, _FORK_C]

    loaded = store.get_cluster(cluster.cluster_id)
    assert loaded is not None
    assert loaded["latest"]["member_fork_node_ids"] == [_FORK_A, _FORK_B, _FORK_C]
    assert [item["version"] for item in loaded["versions"]] == [1, 2]
    assert loaded["versions"][0]["member_fork_node_ids"] == [_FORK_A, _FORK_B]
    hist = (
        tmp_path
        / NAMESPACE_DIRNAME
        / CLUSTERS_DIRNAME
        / f"{cluster.cluster_id}.v1.json"
    )
    assert hist.is_file()
    historical = json.loads(hist.read_text(encoding="utf-8"))
    assert historical["version"] == 1
    assert historical["member_fork_node_ids"] == [_FORK_A, _FORK_B]


def test_edge_submit_idempotency(tmp_path: Path) -> None:
    store = _store(tmp_path)
    payload = _edge_payload()
    first = store.submit_similarity_edge(payload, idempotency_key="diag-1:pair-ab")
    ns = tmp_path / NAMESPACE_DIRNAME
    files_after_create = {path.relative_to(ns) for path in ns.rglob("*") if path.is_file()}

    replay = store.submit_similarity_edge(payload, idempotency_key="diag-1:pair-ab")
    assert replay["deduplicated"] is True
    assert replay["created"] is False
    assert replay["edge_id"] == first["edge_id"]
    files_after_replay = {path.relative_to(ns) for path in ns.rglob("*") if path.is_file()}
    assert files_after_replay == files_after_create

    conflict = _edge_payload(judgment=_judgment(rationale="different rationale"))
    with pytest.raises(ValueError, match="idempotency key conflict"):
        store.submit_similarity_edge(conflict, idempotency_key="diag-1:pair-ab")


def test_legacy_mode_submit_denied(tmp_path: Path) -> None:
    store = CausalClusterStore(tmp_path, config=_LEGACY_CONFIG)
    with pytest.raises(PermissionError, match="evaluator/orchestration"):
        store.submit_similarity_edge(_edge_payload(), idempotency_key="legacy-edge")
    ns = tmp_path / NAMESPACE_DIRNAME
    assert not ns.exists() or not any(ns.rglob("*.json"))


def test_merge_clusters_supersedes_predecessors(tmp_path: Path) -> None:
    store = _store(tmp_path)
    edge_ab = _submit_positive(store, fork_a=_FORK_A, fork_b=_FORK_B, key="pos-ab")
    edge_ac = _submit_positive(store, fork_a=_FORK_C, fork_b=_FORK_A, key="pos-ca", verdict="analogous")
    left = store.create_cluster(_FORK_A, [_FORK_A, _FORK_B], edge_ab["edge_id"])
    right = store.create_cluster(_FORK_C, [_FORK_C, _FORK_A], edge_ac["edge_id"])

    merged = store.merge_clusters([left.cluster_id, right.cluster_id], center_fork_node_id=_FORK_A)
    assert merged.status == "active"
    assert merged.cluster_id not in {left.cluster_id, right.cluster_id}
    assert merged.supersedes_cluster_ids == [left.cluster_id, right.cluster_id]
    assert set(merged.member_fork_node_ids) == {_FORK_A, _FORK_B, _FORK_C}
    assert merged.center_fork_node_id == _FORK_A

    old_left = store.get_cluster(left.cluster_id)
    old_right = store.get_cluster(right.cluster_id)
    assert old_left is not None and old_right is not None
    assert old_left["latest"]["status"] == "superseded"
    assert old_right["latest"]["status"] == "superseded"
    assert old_left["latest"]["closed_at"]
    assert [item["version"] for item in old_left["versions"]] == [1, 2]
    assert (tmp_path / NAMESPACE_DIRNAME / CLUSTERS_DIRNAME / f"{left.cluster_id}.v1.json").is_file()

    active_ids = {item["cluster_id"] for item in store.clusters_for_fork(_FORK_A)}
    assert active_ids == {merged.cluster_id}
    assert left.cluster_id not in active_ids

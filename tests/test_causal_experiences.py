# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Unit tests for causal-experiences fork-node store (slice S5/6)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openviking.session.causal_experiences import (
    NAMESPACE_DIRNAME,
    SEVEN_SECTION_KEYS,
    CausalExperiencesStore,
    EvidenceRef,
    ForkNodeRevision,
    canonical_anchor_key,
    validate_anchor,
    validate_evidence_ref,
)

_CAUSAL_CONFIG = {"skill_trajectory_mode": "causal"}
_LEGACY_CONFIG = {"skill_trajectory_mode": "legacy"}


def _seven_sections(**overrides: str) -> dict[str, str]:
    sections = {key: f"{key} body" for key in SEVEN_SECTION_KEYS}
    sections.update(overrides)
    return sections


def _ao_anchor(*, sequence: int = 10) -> dict:
    return {
        "anchor_kind": "ao",
        "ao_id": "ao-anchor-1",
        "source_session_id": "sess-1",
        "anchor_sequence": sequence,
    }


def _session_state_anchor(**overrides: object) -> dict:
    anchor: dict = {
        "anchor_kind": "session_state",
        "session_id": "sess-live",
        "ledger_snapshot_watermark": "wm-42",
        "live_ao_upper_bound": 7,
        "anchored_at": "2026-09-23T00:00:00.000Z",
    }
    anchor.update(overrides)
    return anchor


def _interaction_edge_anchor() -> dict:
    return {
        "anchor_kind": "interaction_edge",
        "channel_id": "chan-1",
        "interaction_event_id": "evt-9",
        "from_segment_id": "seg-a",
        "to_segment_id": "seg-b",
        "dag_snapshot_watermark": "dag-wm-1",
    }


def _evidence_ref(
    *,
    ao_id: str = "ao-e1",
    sequence: int = 3,
    role: str = "contemporaneous_basis",
) -> dict:
    return {
        "ao_id": ao_id,
        "source_session_id": "sess-1",
        "source_archive_id": "arch-1",
        "archive_commit_watermark": "commit-1",
        "source_sequence": sequence,
        "source_read_snapshot_watermark": "read-wm-1",
        "evidence_role": role,
        "evidence_content_hash": "hash-" + ao_id,
        "captured_state": "committed",
    }


def _payload(**overrides: object) -> dict:
    payload: dict = {
        "fork_node_id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
        "revision_id": "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
        "workspace_id": "ws-1",
        "anchor": _ao_anchor(),
        "seven_sections": _seven_sections(),
        "contemporaneous_basis": [_evidence_ref(sequence=4, role="contemporaneous_basis")],
        "hindsight_attribution": [
            _evidence_ref(ao_id="ao-h1", sequence=12, role="hindsight_attribution")
        ],
        "used_skills": [
            {
                "skill_uri": "viking://~/skills/git-rescue",
                "revision_hash": "sha256:abc",
                "invocation_id": "inv-1",
            }
        ],
        "branches": [],
        "diagnosis_run_id": "diag-1",
        "created_at": "2026-09-23T12:00:00.000Z",
        "model_version": "diag-model-v1",
    }
    payload.update(overrides)
    return payload


def _store(tmp_path: Path, config: dict | None = None) -> CausalExperiencesStore:
    return CausalExperiencesStore(tmp_path, config=config if config is not None else _CAUSAL_CONFIG)


def test_validate_anchor_accepts_three_kinds_and_rejects_invalid() -> None:
    ao = validate_anchor(_ao_anchor())
    assert ao["anchor_kind"] == "ao"
    assert ao["ao_id"] == "ao-anchor-1"
    assert ao["anchor_sequence"] == 10

    state = validate_anchor(_session_state_anchor())
    assert state["live_ao_upper_bound"] == 7
    assert state["ledger_snapshot_watermark"] == "wm-42"

    edge = validate_anchor(_interaction_edge_anchor())
    assert edge["from_segment_id"] == "seg-a"
    assert edge["interaction_event_id"] == "evt-9"
    assert canonical_anchor_key(_ao_anchor(sequence=1)) == canonical_anchor_key(
        _ao_anchor(sequence=99)
    )
    assert canonical_anchor_key(_interaction_edge_anchor()) == (
        '{"anchor_kind":"interaction_edge","channel_id":"chan-1",'
        '"dag_snapshot_watermark":"dag-wm-1","from_segment_id":"seg-a",'
        '"interaction_event_id":"evt-9","to_segment_id":"seg-b"}'
    )

    with pytest.raises(ValueError, match="anchor_kind"):
        validate_anchor({"anchor_kind": "memory", "ao_id": "x", "source_session_id": "s"})
    with pytest.raises(ValueError, match="ao_id"):
        validate_anchor(
            {"anchor_kind": "ao", "source_session_id": "s", "anchor_sequence": 1}
        )
    with pytest.raises(ValueError, match="session_id"):
        validate_anchor(
            {
                "anchor_kind": "session_state",
                "ledger_snapshot_watermark": "wm",
                "live_ao_upper_bound": 1,
                "anchored_at": "t",
            }
        )
    with pytest.raises(ValueError, match="channel_id"):
        validate_anchor(
            {
                "anchor_kind": "interaction_edge",
                "interaction_event_id": "e",
                "from_segment_id": "a",
                "to_segment_id": "b",
                "dag_snapshot_watermark": "w",
            }
        )


def test_seven_sections_and_causal_context_isolation() -> None:
    incomplete = _seven_sections()
    del incomplete["open_issues"]
    with pytest.raises(ValueError, match="seven_sections"):
        ForkNodeRevision.from_dict(_payload(seven_sections=incomplete, content_hash=""))

    with pytest.raises(ValueError, match="contemporaneous_basis source_sequence"):
        ForkNodeRevision.from_dict(
            _payload(
                contemporaneous_basis=[
                    _evidence_ref(sequence=10, role="contemporaneous_basis")
                ],
                content_hash="",
            )
        )
    with pytest.raises(ValueError, match="contemporaneous_basis source_sequence"):
        ForkNodeRevision.from_dict(
            _payload(
                contemporaneous_basis=[
                    _evidence_ref(sequence=11, role="contemporaneous_basis")
                ],
                content_hash="",
            )
        )

    late_hindsight = _payload(
        hindsight_attribution=[
            _evidence_ref(ao_id="ao-late", sequence=99, role="hindsight_attribution")
        ],
        content_hash="",
    )
    revision = ForkNodeRevision.from_dict(late_hindsight)
    assert revision.hindsight_attribution[0].source_sequence == 99
    assert revision.anchor["anchor_sequence"] == 10

    session_state_payload = _payload(
        anchor=_session_state_anchor(),
        contemporaneous_basis=[
            _evidence_ref(sequence=99, role="contemporaneous_basis")
        ],
        content_hash="",
    )
    skipped = ForkNodeRevision.from_dict(session_state_payload)
    assert skipped.anchor["anchor_kind"] == "session_state"
    assert skipped.contemporaneous_basis[0].source_sequence == 99


def test_used_skills_require_revision_hash() -> None:
    with pytest.raises(ValueError, match="revision_hash"):
        ForkNodeRevision.from_dict(
            _payload(
                used_skills=[{"skill_uri": "viking://~/skills/git-rescue"}],
                content_hash="",
            )
        )


def test_upsert_new_anchor_and_idempotency(tmp_path: Path) -> None:
    store = _store(tmp_path)
    payload = _payload()
    first = store.upsert_fork_node(payload, idempotency_key="diag-1:ao-anchor-1")
    assert first["created"] is True
    assert first["deduplicated"] is False
    assert first["fork_node_id"]
    assert first["revision_id"]

    ns = tmp_path / NAMESPACE_DIRNAME
    revision_path = (
        ns / "forks" / first["fork_node_id"] / "revisions" / f"{first['revision_id']}.json"
    )
    assert revision_path.is_file()
    index_path = ns / "index.jsonl"
    index_lines = [line for line in index_path.read_text(encoding="utf-8").splitlines() if line]
    assert len(index_lines) == 1
    index_row = json.loads(index_lines[0])
    assert index_row["fork_node_id"] == first["fork_node_id"]
    assert index_row["revision_id"] == first["revision_id"]
    assert index_row["anchor_key"] == canonical_anchor_key(_ao_anchor())
    files_after_create = {path.relative_to(ns) for path in ns.rglob("*") if path.is_file()}

    replay = store.upsert_fork_node(payload, idempotency_key="diag-1:ao-anchor-1")
    assert replay["deduplicated"] is True
    assert replay["created"] is False
    assert replay["fork_node_id"] == first["fork_node_id"]
    assert replay["revision_id"] == first["revision_id"]
    files_after_replay = {path.relative_to(ns) for path in ns.rglob("*") if path.is_file()}
    assert files_after_replay == files_after_create
    replay_index = [line for line in index_path.read_text(encoding="utf-8").splitlines() if line]
    assert len(replay_index) == 1

    conflict = _payload(diagnosis_run_id="diag-other")
    with pytest.raises(ValueError, match="idempotency key conflict"):
        store.upsert_fork_node(conflict, idempotency_key="diag-1:ao-anchor-1")


def test_same_anchor_reuses_fork_node_and_appends_revision(tmp_path: Path) -> None:
    store = _store(tmp_path)
    first = store.upsert_fork_node(_payload(), idempotency_key="run-a")
    second_payload = _payload(diagnosis_run_id="diag-2")
    second_payload.pop("fork_node_id", None)
    second_payload.pop("revision_id", None)
    second_payload.pop("created_at", None)
    second = store.upsert_fork_node(second_payload, idempotency_key="run-b")
    assert second["created"] is False
    assert second["fork_node_id"] == first["fork_node_id"]
    assert second["revision_id"] != first["revision_id"]

    rev_dir = (
        tmp_path / NAMESPACE_DIRNAME / "forks" / first["fork_node_id"] / "revisions"
    )
    assert (rev_dir / f"{first['revision_id']}.json").is_file()
    assert (rev_dir / f"{second['revision_id']}.json").is_file()
    assert len(list(rev_dir.glob("*.json"))) == 2

    loaded = store.get_fork_node(first["fork_node_id"])
    assert loaded is not None
    assert loaded["fork_node_id"] == first["fork_node_id"]
    assert len(loaded["revisions"]) == 2
    assert loaded["latest_revision"]["diagnosis_run_id"] == "diag-2"
    assert store.get_revision(first["fork_node_id"], first["revision_id"]) is not None


def test_content_hash_roundtrip_and_tamper_detection() -> None:
    revision = ForkNodeRevision.from_dict(_payload(content_hash=""))
    assert revision.content_hash
    assert revision.recompute_content_hash() == revision.content_hash

    restored = ForkNodeRevision.from_dict(revision.to_dict())
    assert restored.content_hash == revision.content_hash
    assert restored.seven_sections["session_title"] == revision.seven_sections["session_title"]

    tampered = revision.to_dict()
    tampered["model_version"] = "tampered-model"
    assert ForkNodeRevision.from_dict({**tampered, "content_hash": ""}).recompute_content_hash() != (
        revision.content_hash
    )
    with pytest.raises(ValueError, match="content_hash mismatch"):
        ForkNodeRevision.from_dict(tampered)


def test_session_state_reuses_fork_node_across_watermarks(tmp_path: Path) -> None:
    first_anchor = _session_state_anchor(
        ledger_snapshot_watermark="wm-1",
        live_ao_upper_bound=3,
        anchored_at="2026-09-23T01:00:00.000Z",
    )
    second_anchor = _session_state_anchor(
        ledger_snapshot_watermark="wm-2",
        live_ao_upper_bound=9,
        anchored_at="2026-09-23T02:00:00.000Z",
    )
    assert canonical_anchor_key(first_anchor) == canonical_anchor_key(second_anchor)
    assert canonical_anchor_key(first_anchor) == (
        '{"anchor_kind":"session_state","session_id":"sess-live"}'
    )

    store = _store(tmp_path)
    first = store.upsert_fork_node(
        _payload(anchor=first_anchor, revision_id="rev-ss-1"),
        idempotency_key="ss-run-1",
    )
    second_payload = _payload(
        revision_id="rev-ss-2",
        diagnosis_run_id="diag-2",
        created_at="2026-09-23T13:00:00.000Z",
        anchor=second_anchor,
        supersedes_revision_id=first["revision_id"],
    )
    second_payload.pop("fork_node_id", None)
    second = store.upsert_fork_node(second_payload, idempotency_key="ss-run-2")
    assert second["created"] is False
    assert second["fork_node_id"] == first["fork_node_id"]
    assert second["revision_id"] != first["revision_id"]

    loaded = store.get_fork_node(first["fork_node_id"])
    assert loaded is not None
    assert len(loaded["revisions"]) == 2
    assert [item["revision_id"] for item in loaded["revisions"]] == [
        first["revision_id"],
        second["revision_id"],
    ]
    assert loaded["revisions"][1]["supersedes_revision_id"] == first["revision_id"]
    assert loaded["latest_revision"]["revision_id"] == second["revision_id"]
    assert loaded["revisions"][0]["anchor"]["ledger_snapshot_watermark"] == "wm-1"
    assert loaded["revisions"][1]["anchor"]["ledger_snapshot_watermark"] == "wm-2"


def test_sibling_revisions_sort_by_created_at_without_inferred_supersede(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    later = store.upsert_fork_node(
        _payload(
            revision_id="rev-later",
            created_at="2026-09-23T14:00:00.000Z",
            diagnosis_run_id="diag-later",
        ),
        idempotency_key="sibling-later",
    )
    earlier_payload = _payload(
        revision_id="rev-earlier",
        created_at="2026-09-23T13:00:00.000Z",
        diagnosis_run_id="diag-earlier",
    )
    earlier_payload.pop("fork_node_id", None)
    earlier = store.upsert_fork_node(earlier_payload, idempotency_key="sibling-earlier")
    assert earlier["fork_node_id"] == later["fork_node_id"]

    loaded = store.get_fork_node(later["fork_node_id"])
    assert loaded is not None
    assert [item["revision_id"] for item in loaded["revisions"]] == [
        "rev-earlier",
        "rev-later",
    ]
    assert loaded["latest_revision"]["revision_id"] == "rev-later"
    assert all(item["supersedes_revision_id"] is None for item in loaded["revisions"])


def test_legacy_mode_upsert_denied(tmp_path: Path) -> None:
    store = CausalExperiencesStore(tmp_path, config=_LEGACY_CONFIG)
    with pytest.raises(PermissionError, match="evaluator/orchestration"):
        store.upsert_fork_node(_payload(), idempotency_key="legacy-run")
    ns = tmp_path / NAMESPACE_DIRNAME
    assert not ns.exists() or not any(ns.rglob("*.json"))


def test_evidence_ref_nine_fields_roundtrip() -> None:
    raw = _evidence_ref()
    ref = validate_evidence_ref(raw)
    assert isinstance(ref, EvidenceRef)
    assert ref.to_dict()["source_archive_id"] == "arch-1"
    assert validate_evidence_ref(ref) is ref
    with pytest.raises(ValueError, match="evidence_role"):
        validate_evidence_ref({**raw, "evidence_role": "unknown"})

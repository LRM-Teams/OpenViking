# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""ForkRevision lifecycle, influence grounding, and intervention history (slice S8)."""

from __future__ import annotations

import inspect
import json

import pytest

from openviking.session import causal_experiences as ce
from openviking.session.causal_experiences import (
    FORK_CANDIDATE_ROLE,
    CausalExperiencesStore,
    ForkStatus,
    InfluenceGrounding,
    InterventionIndexCorruptError,
    ServerChecks,
    influence_position_key,
)
from tests.test_causal_experiences import _ao_anchor, _payload, _seven_sections, _store

_PRINCIPAL = ["ws-1"]

_PASSING_CHECKS = ServerChecks(
    anchor_exists=True,
    refs_exist=True,
    provenance_valid=True,
    acl_allowed=True,
    acl_hook=lambda _draft: True,
)


def _grounding(**overrides: str) -> InfluenceGrounding:
    fields: dict[str, str] = {
        "view_revision_id": "view-rev-1",
        "position_kind": "block",
        "position_ref": "block:42",
        "semantic_summary_snapshot": "frozen summary of the grounded block",
        "content_hash": "sha256:grounding-snapshot",
        "grounded_at": "2026-09-24T00:00:00.000Z",
    }
    fields.update(overrides)
    return InfluenceGrounding(**fields)  # type: ignore[arg-type]


def _observed_branch() -> dict[str, str]:
    return {"branch_id": "br-obs-1", "evidence_status": "real", "ref": "ao:ao-e1"}


def _independent_control() -> dict[str, str]:
    return {"ref_kind": "independent_control", "ref_id": "ctl-1"}


def _validated_evidence() -> dict[str, list[dict[str, str]]]:
    return {
        "observed_branch_refs": [_observed_branch()],
        "independent_control_refs": [_independent_control()],
    }


def test_draft_is_not_authoritative_and_commit_is_idempotent(tmp_path) -> None:
    store = _store(tmp_path)
    draft = store.submit_fork_draft(_payload(), influence_grounding=_grounding())
    assert draft.status == ForkStatus.DRAFT
    assert store.get_fork_node(draft.revision.fork_node_id) is None
    ns = tmp_path / "causal-experiences"
    assert not (ns / "forks").exists()
    assert not (ns / "index.jsonl").exists()

    with pytest.raises(ValueError, match="server check failed"):
        store.commit_provisional(
            draft,
            ServerChecks(anchor_exists=False, refs_exist=True, provenance_valid=True),
        )
    assert store.get_fork_node(draft.revision.fork_node_id) is None
    assert not (ns / "forks").exists()

    first = store.commit_provisional(draft, _PASSING_CHECKS)
    second = store.commit_provisional(draft, _PASSING_CHECKS)
    assert first.status == ForkStatus.PROVISIONAL
    assert first.consumption_role == FORK_CANDIDATE_ROLE
    assert second.revision_id == first.revision_id
    assert second.fork_node_id == first.fork_node_id
    assert second.content_hash == first.content_hash
    rev_dir = ns / "forks" / first.fork_node_id / "revisions"
    assert len(list(rev_dir.glob("*.json"))) == 1


def test_validated_requires_observed_branch_and_control_invalidated_appends(tmp_path) -> None:
    store = _store(tmp_path)
    provisional = store.commit_provisional(
        store.submit_fork_draft(_payload()),
        _PASSING_CHECKS,
    )
    with pytest.raises(ValueError, match="observed branch"):
        store.append_validated(
            provisional.fork_node_id,
            {"independent_control_refs": [_independent_control()]},
        )
    with pytest.raises(ValueError, match="independent_control"):
        store.append_validated(
            provisional.fork_node_id,
            {"observed_branch_refs": [_observed_branch()]},
        )
    with pytest.raises(ValueError, match="observed branch"):
        store.append_validated(
            provisional.fork_node_id,
            {
                "observed_branch_refs": [
                    {"branch_id": "br-imagined", "evidence_status": "imagined_synthetic"}
                ],
                "independent_control_refs": [_independent_control()],
            },
        )
    validated = store.append_validated(provisional.fork_node_id, _validated_evidence())
    assert validated.status == ForkStatus.VALIDATED
    assert validated.supersedes_revision_id == provisional.revision_id
    still_there = store.get_revision(provisional.fork_node_id, provisional.revision_id)
    assert still_there is not None
    assert still_there["status"] == ForkStatus.PROVISIONAL

    other_anchor = _ao_anchor()
    other_anchor["ao_id"] = "ao-invalidate"
    invalidated_source = store.commit_provisional(
        store.submit_fork_draft(
            _payload(
                fork_node_id="cccccccc-cccc-4ccc-8ccc-cccccccccccc",
                revision_id="dddddddd-dddd-4ddd-8ddd-dddddddddddd",
                diagnosis_run_id="diag-invalidate",
                anchor=other_anchor,
            )
        ),
        _PASSING_CHECKS,
    )
    invalidated = store.append_invalidated(
        invalidated_source.fork_node_id,
        {"reason": "mechanism contradicted", "counter_ref": "run-baseline-9"},
    )
    assert invalidated.status == ForkStatus.INVALIDATED
    assert invalidated.supersedes_revision_id == invalidated_source.revision_id
    assert invalidated.revision_id != invalidated_source.revision_id
    previous = store.get_revision(
        invalidated_source.fork_node_id, invalidated_source.revision_id
    )
    assert previous is not None
    assert previous["status"] == ForkStatus.PROVISIONAL
    loaded = store.get_fork_node(invalidated_source.fork_node_id)
    assert loaded is not None
    assert [item["revision_id"] for item in loaded["revisions"]] == [
        invalidated_source.revision_id,
        invalidated.revision_id,
    ]


def test_grounding_snapshot_reverse_index_and_revalidation(tmp_path) -> None:
    store = _store(tmp_path)
    grounding = _grounding()
    other = _grounding(view_revision_id="view-rev-other", position_ref="edge:7", position_kind="edge")
    provisional = store.commit_provisional(
        store.submit_fork_draft(_payload(), influence_grounding=grounding),
        _PASSING_CHECKS,
    )
    snapshot = provisional.influence_grounding
    assert snapshot is not None
    assert snapshot.view_revision_id == grounding.view_revision_id
    assert snapshot.position_kind == "block"
    assert snapshot.position_ref == grounding.position_ref
    assert snapshot.semantic_summary_snapshot == grounding.semantic_summary_snapshot
    assert snapshot.content_hash == grounding.content_hash
    assert snapshot.grounded_at == grounding.grounded_at

    other_anchor = _ao_anchor()
    other_anchor["ao_id"] = "ao-other-view"
    store.commit_provisional(
        store.submit_fork_draft(
            _payload(
                fork_node_id="eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee",
                revision_id="ffffffff-ffff-4fff-8fff-ffffffffffff",
                diagnosis_run_id="diag-other-view",
                anchor=other_anchor,
            ),
            influence_grounding=other,
        ),
        _PASSING_CHECKS,
    )

    position_key = influence_position_key(grounding)
    history = store.query_intervention_history(
        grounding.view_revision_id, position_key, _PRINCIPAL
    )
    assert [row["fork_node_id"] for row in history] == [provisional.fork_node_id]
    assert history[0]["revision_id"] == provisional.revision_id
    assert history[0]["validation_status"] == ForkStatus.PROVISIONAL
    assert history[0]["summary"] == grounding.semantic_summary_snapshot
    assert history[0]["needs_revalidation"] is False

    validated = store.append_validated(provisional.fork_node_id, _validated_evidence())
    history = store.query_intervention_history(
        grounding.view_revision_id, position_key, _PRINCIPAL
    )
    assert [row["revision_id"] for row in history] == [
        provisional.revision_id,
        validated.revision_id,
    ]
    assert history[1]["validation_status"] == ForkStatus.VALIDATED

    revision_path = (
        tmp_path
        / "causal-experiences"
        / "forks"
        / provisional.fork_node_id
        / "revisions"
        / f"{provisional.revision_id}.json"
    )
    before = revision_path.read_bytes()
    flagged = store.mark_needs_revalidation(grounding.view_revision_id)
    assert flagged == 2
    assert revision_path.read_bytes() == before
    history = store.query_intervention_history(
        grounding.view_revision_id, position_key, _PRINCIPAL
    )
    assert len(history) == 2
    assert all(row["needs_revalidation"] is True for row in history)
    other_history = store.query_intervention_history(
        other.view_revision_id, influence_position_key(other), _PRINCIPAL
    )
    assert len(other_history) == 1
    assert other_history[0]["needs_revalidation"] is False


def test_no_public_api_changes_status_from_citation_counts() -> None:
    # Citation counts are not a ForkStatus transition. Introspection guards the surface.
    names: list[str] = []
    for name, obj in vars(ce).items():
        if name.startswith("_"):
            continue
        names.append(name)
        if inspect.isclass(obj):
            for method_name, member in vars(obj).items():
                if method_name.startswith("_"):
                    continue
                names.append(method_name)
                if inspect.isfunction(member) or inspect.ismethod(member):
                    assert "citation" not in (inspect.getdoc(member) or "").lower()
    public = " ".join(names).lower()
    assert "citation" not in public
    store_methods = [
        name
        for name, member in vars(CausalExperiencesStore).items()
        if callable(member) and not name.startswith("_")
    ]
    assert "submit_fork_draft" in store_methods
    assert "commit_provisional" in store_methods
    assert "append_validated" in store_methods
    assert "append_invalidated" in store_methods
    assert not any("citation" in name.lower() for name in store_methods)


def test_retry_recovers_when_revision_exists_without_idempotency_file(tmp_path) -> None:
    store = _store(tmp_path)
    payload = _payload()
    key = "crash-window"
    first = store.upsert_fork_node(payload, idempotency_key=key)
    ns = tmp_path / "causal-experiences"
    idem_files = list((ns / ".idempotency").glob("*.json"))
    assert len(idem_files) == 1
    idem_files[0].unlink()
    index_before = (ns / "index.jsonl").read_text(encoding="utf-8")

    replay = store.upsert_fork_node(payload, idempotency_key=key)
    assert replay["fork_node_id"] == first["fork_node_id"]
    assert replay["revision_id"] == first["revision_id"]
    assert replay["deduplicated"] is True
    assert replay["created"] is False
    assert (ns / "index.jsonl").read_text(encoding="utf-8") == index_before
    assert list((ns / ".idempotency").glob("*.json"))

    for path in (ns / ".idempotency").glob("*.json"):
        path.unlink()
    changed = _payload(seven_sections=_seven_sections(session_title="rewritten session title"))
    with pytest.raises(ValueError, match="idempotency key conflict") as caught:
        store.upsert_fork_node(changed, idempotency_key=key)
    assert "duplicate revision_id" not in str(caught.value)
    stored = store.get_revision(first["fork_node_id"], first["revision_id"])
    assert stored is not None
    assert stored["seven_sections"]["session_title"] == payload["seven_sections"]["session_title"]


def test_corrupt_intervention_index_is_quarantined_and_history_survives_repair(
    tmp_path,
) -> None:
    store = _store(tmp_path)
    grounding = _grounding()
    provisional = store.commit_provisional(
        store.submit_fork_draft(_payload(), influence_grounding=grounding),
        _PASSING_CHECKS,
    )
    index = tmp_path / "causal-experiences" / "intervention-index.json"
    good = index.read_bytes()
    index.write_text("{not-json", encoding="utf-8")

    with pytest.raises(InterventionIndexCorruptError):
        store.mark_needs_revalidation(grounding.view_revision_id)
    assert not index.exists()
    quarantined = list(index.parent.glob("intervention-index.json.corrupt-*"))
    assert len(quarantined) == 1
    assert quarantined[0].read_text(encoding="utf-8") == "{not-json"

    index.write_bytes(good)
    history = store.query_intervention_history(
        grounding.view_revision_id, influence_position_key(grounding), _PRINCIPAL
    )
    assert [row["revision_id"] for row in history] == [provisional.revision_id]
    assert history[0]["needs_revalidation"] is False


def test_intervention_history_hides_rows_the_principal_cannot_see(tmp_path) -> None:
    store = _store(tmp_path)
    grounding = _grounding()
    provisional = store.commit_provisional(
        store.submit_fork_draft(_payload(), influence_grounding=grounding),
        _PASSING_CHECKS,
    )
    position_key = influence_position_key(grounding)
    hidden = store.query_intervention_history(
        grounding.view_revision_id, position_key, ["outsider"]
    )
    assert hidden == []

    visible = store.query_intervention_history(
        grounding.view_revision_id, position_key, _PRINCIPAL
    )
    assert [row["revision_id"] for row in visible] == [provisional.revision_id]
    assert set(visible[0]) == {
        "fork_node_id",
        "revision_id",
        "validation_status",
        "summary",
        "needs_revalidation",
    }

    index = tmp_path / "causal-experiences" / "intervention-index.json"
    payload = json.loads(index.read_text(encoding="utf-8"))
    del payload["entries"][0]["security_labels"]
    index.write_text(json.dumps(payload), encoding="utf-8")
    unlabeled = store.query_intervention_history(
        grounding.view_revision_id, position_key, _PRINCIPAL
    )
    assert unlabeled == []


def test_upsert_rejects_direct_validated_and_invalidated_status(tmp_path) -> None:
    store = _store(tmp_path)
    with pytest.raises(ValueError, match="append_validated/append_invalidated"):
        store.upsert_fork_node(_payload(status=ForkStatus.VALIDATED), idempotency_key="direct-v")
    with pytest.raises(ValueError, match="append_validated/append_invalidated"):
        store.upsert_fork_node(
            _payload(status=ForkStatus.INVALIDATED), idempotency_key="direct-i"
        )
    assert store.get_fork_node(_payload()["fork_node_id"]) is None

    accepted = store.upsert_fork_node(
        _payload(status=ForkStatus.PROVISIONAL), idempotency_key="direct-p"
    )
    stored = store.get_revision(accepted["fork_node_id"], accepted["revision_id"])
    assert stored is not None
    assert stored["status"] == ForkStatus.PROVISIONAL

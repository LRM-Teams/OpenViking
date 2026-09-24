# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Unit tests for Influence projection cards (slice S12)."""

from __future__ import annotations

import json
import logging

import pytest

from openviking.session.influence_projection import (
    CARD_KIND_BLOCK,
    CARD_KIND_CLAIM,
    CARD_STATUS_PROVISIONAL,
    CARD_STATUS_VERIFIED,
    REMOVAL_ACL_CHANGE,
    REMOVAL_SUPERSEDED,
    InfluenceProjection,
    ProjectionRegistry,
    SourcePointer,
)
from openviking.session.influence_view import (
    AORange,
    CoverageManifest,
    InfluenceBlock,
    InfluenceClaim,
    InfluenceViewDraft,
    PURPOSE_FAILURE_DIAGNOSIS,
    PURPOSE_POST_RUN_INDEX,
    SegmentRef,
    SourceRef,
    build_minimal_fallback_view,
)

SESSION = "sess-1"
OTHER = "sess-2"
CREATED = "2026-01-02T00:00:00.000Z"


def _clock():
    from datetime import datetime, timezone

    return datetime(2026, 1, 2, tzinfo=timezone.utc)


def _ref(sequence: int, *, session_id: str = SESSION) -> SourceRef:
    return SourceRef(
        ao_id=f"ao-{sequence}",
        sequence=sequence,
        participant_id="agent-a",
        segment_id="seg-a",
        session_id=session_id,
        content_hash=f"hash-{sequence}",
    )


def _block(block_id: str, sequence: int, *, session_id: str = SESSION) -> InfluenceBlock:
    ref = _ref(sequence, session_id=session_id)
    return InfluenceBlock(
        block_id=block_id,
        role="agent_action",
        participant_id="agent-a",
        segment_ref=SegmentRef(segment_id="seg-a", session_id=session_id),
        ao_start_seq=sequence,
        ao_end_seq=sequence,
        summary=f"summary-{block_id}",
        input="RAW_INPUT_SHOULD_NOT_LEAK",
        output="RAW_OUTPUT_SHOULD_NOT_LEAK",
        authorship="agent-a",
        granularity_reason="one action",
        source_content_hashes=(ref.content_hash,),
        source_refs=(ref,),
    )


def _coverage(blocks: list[InfluenceBlock]) -> CoverageManifest:
    refs = tuple(ref for block in blocks for ref in block.source_refs)
    return CoverageManifest(
        total_ao_range=AORange(start_seq=min(ref.sequence for ref in refs), end_seq=max(ref.sequence for ref in refs)),
        inspected_refs=refs,
        included_ranges=tuple(
            AORange(start_seq=block.ao_start_seq, end_seq=block.ao_end_seq, segment_id="seg-a")
            for block in blocks
        ),
        omitted_ranges=(),
        snapshot_watermark="wm-1",
    )


def _revision_with_claim(session_id: str = SESSION, revision_id: str = "rev-claim"):
    blocks = [_block("block-a", 1, session_id=session_id), _block("block-b", 2, session_id=session_id)]
    claim = InfluenceClaim(
        claim_id="claim-1",
        source_block_id="block-a",
        target_block_id="block-b",
        carried_artifact="patch",
        downstream_effect="tests passed",
        relation_type="influence",
    )
    draft = InfluenceViewDraft(
        view_id=f"view-{revision_id}",
        purpose=PURPOSE_POST_RUN_INDEX,
        session_id=session_id,
        task_run_id=f"task-{revision_id}",
        coverage=_coverage(blocks),
        blocks=blocks,
        claims=(claim,),
    )
    return draft.freeze(revision_id=revision_id, frozen_at=CREATED)


def _fallback(session_id: str, revision_id: str, *, purpose: str = PURPOSE_POST_RUN_INDEX):
    return build_minimal_fallback_view(
        {
            "session_id": session_id,
            "task_run_id": f"task-{revision_id}",
            "purpose": purpose,
            "revision_id": revision_id,
            "view_id": f"view-{revision_id}",
            "frozen_at": CREATED,
            "ao_records": [
                {
                    "ao_id": f"ao-{revision_id}",
                    "sequence": 1,
                    "participant_id": "agent-a",
                    "segment_id": f"seg-{session_id}",
                    "session_id": session_id,
                    "content_hash": f"hash-{revision_id}",
                }
            ],
        }
    )


def _registry(tmp_path, name: str = "projections.json", **kwargs) -> ProjectionRegistry:
    return ProjectionRegistry(tmp_path / name, clock=_clock, **kwargs)


def test_register_projects_blocks_and_claims_without_full_text(tmp_path) -> None:
    registry = _registry(tmp_path)
    revision = _revision_with_claim()
    cards = registry.register(revision, acl_labels=("team-a",))
    assert [card.kind for card in cards] == [CARD_KIND_BLOCK, CARD_KIND_BLOCK, CARD_KIND_CLAIM]
    assert all(card.status == CARD_STATUS_PROVISIONAL for card in cards)
    assert all(card.fork_provenance is None for card in cards)
    claim = registry.search(principal_labels=("team-a",), kind=CARD_KIND_CLAIM)[0]
    assert claim.summary == "patch -> tests passed"
    assert claim.semantic_type == "influence"
    assert claim.source_pointer == SourcePointer("rev-claim", "claim-1")
    assert claim.embedding_stub["dimensions"] == 0
    blob = json.dumps(claim.to_dict())
    assert "RAW_INPUT_SHOULD_NOT_LEAK" not in blob
    assert "RAW_OUTPUT_SHOULD_NOT_LEAK" not in blob
    assert "source_refs" not in claim.to_dict()
    assert InfluenceProjection.from_dict(json.loads(json.dumps(claim.to_dict()))) == claim


def test_search_hides_cards_when_acl_labels_do_not_intersect(tmp_path) -> None:
    registry = _registry(tmp_path, session_acl=lambda session_id: (f"label-{session_id}",))
    inherited = registry.register(_fallback(SESSION, "rev-inherited"))
    assert inherited[0].acl_labels == ("label-sess-1",)
    registry.register(_fallback(SESSION, "rev-secret"), acl_labels=("team-a", "secret"))
    registry.register(_fallback(OTHER, "rev-other"), acl_labels=("team-b",))

    visible = registry.search(principal_labels=("team-a",))
    assert visible
    assert all("team-a" in card.acl_labels for card in visible)
    assert registry.search(principal_labels=("outsider",)) == ()
    assert registry.search(principal_labels=()) == ()
    leaked = registry.search(principal_labels=("secret", "outsider"))
    assert {card.source_pointer.view_revision_id for card in leaked} == {"rev-secret"}
    other = registry.search(principal_labels=("team-b",), kind=CARD_KIND_BLOCK, status=CARD_STATUS_PROVISIONAL)
    assert [card.session_id for card in other] == [OTHER]


def test_acl_change_withdraws_cards_and_keeps_audit(tmp_path) -> None:
    registry = _registry(tmp_path)
    registry.register(_fallback(SESSION, "rev-1"), acl_labels=("team-a",))
    registry.register(_fallback(OTHER, "rev-2"), acl_labels=("team-b",))
    removed = registry.handle_acl_change(SESSION, ("team-b",))
    assert len(removed) == 1
    assert removed[0].reason == REMOVAL_ACL_CHANGE
    assert removed[0].session_id == SESSION
    assert removed[0].new_labels == ("team-b",)
    assert removed[0].card.acl_labels == ("team-a",)
    assert registry.search(principal_labels=("team-a",)) == ()
    assert registry.search(principal_labels=("team-b",))[0].session_id == OTHER
    assert registry.removed_records() == removed

    reloaded = ProjectionRegistry(tmp_path / "projections.json", clock=_clock)
    assert reloaded.search(principal_labels=("team-a",)) == ()
    assert reloaded.removed_records()[0].record_id == removed[0].record_id
    assert reloaded.search(principal_labels=("team-b",))[0].session_id == OTHER


def test_register_supersedes_same_purpose_only(tmp_path) -> None:
    registry = _registry(tmp_path)
    registry.register(_fallback(SESSION, "rev-1"), acl_labels=("team",))
    diagnosis = _fallback(SESSION, "rev-diag", purpose=PURPOSE_FAILURE_DIAGNOSIS)
    registry.register(diagnosis, acl_labels=("team",))
    registry.register(_fallback(SESSION, "rev-2"), acl_labels=("team",))
    found = registry.search(principal_labels=("team",))
    revision_ids = {card.source_pointer.view_revision_id for card in found}
    assert revision_ids == {"rev-2", "rev-diag"}
    superseded = [record for record in registry.removed_records() if record.reason == REMOVAL_SUPERSEDED]
    assert [record.card.source_pointer.view_revision_id for record in superseded] == ["rev-1"]


def test_promote_rejects_without_validated_grounding(tmp_path) -> None:
    registry = _registry(tmp_path, grounding_query=lambda _revision_id, _position_id: False)
    revision = _fallback(SESSION, "rev-1")
    registry.register(revision, acl_labels=("team-a",))
    block_id = revision.blocks[0].block_id
    with pytest.raises(ValueError, match="validated"):
        registry.promote_to_verified(revision.revision_id, block_id, {"fork_revision_id": "fr-1"})
    assert registry.search(principal_labels=("team-a",))[0].status == CARD_STATUS_PROVISIONAL


def test_promote_with_grounding_query_marks_verified(tmp_path) -> None:
    blocks = [_block("block-a", 1)]
    revision = InfluenceViewDraft(
        view_id="view-rev-1",
        purpose=PURPOSE_POST_RUN_INDEX,
        session_id=SESSION,
        task_run_id="task-rev-1",
        coverage=_coverage(blocks),
        blocks=blocks,
    ).freeze(revision_id="rev-1", frozen_at=CREATED)
    block_id = revision.blocks[0].block_id

    def grounding_query(view_revision_id: str, position_id: str):
        if view_revision_id == revision.revision_id and position_id == block_id:
            return {
                "fork_node_id": "fork-1",
                "revision_id": "fr-1",
                "status": "validated",
                "position_match": True,
            }
        return None

    registry = _registry(tmp_path, grounding_query=grounding_query)
    registry.register(revision, acl_labels=("team-a",))
    provenance = {"fork_revision_id": "fr-1", "status": "validated"}
    card = registry.promote_to_verified(revision.revision_id, block_id, provenance)
    provenance["status"] = "tampered"
    assert card.status == CARD_STATUS_VERIFIED
    assert card.fork_provenance == {"fork_revision_id": "fr-1", "status": "validated"}
    assert registry.search(principal_labels=("team-a",), status=CARD_STATUS_PROVISIONAL) == ()
    verified = registry.search(principal_labels=("team-a",), status=CARD_STATUS_VERIFIED)
    assert verified == (card,)


def test_promote_with_registration_validation_marks_verified(tmp_path) -> None:
    revision = _revision_with_claim()
    registry = _registry(tmp_path)
    registry.register(
        revision,
        acl_labels=("team-a",),
        validated_positions=("claim-1",),
    )
    card = registry.promote_to_verified(
        revision.revision_id,
        "claim-1",
        {"fork_node_id": "fork-1", "fork_revision_id": "fr-9"},
    )
    assert card.kind == CARD_KIND_CLAIM
    assert card.status == CARD_STATUS_VERIFIED
    assert card.fork_provenance == {"fork_node_id": "fork-1", "fork_revision_id": "fr-9"}
    with pytest.raises(ValueError, match="validated"):
        registry.promote_to_verified(revision.revision_id, "block-a", {"fork_revision_id": "fr-x"})


def test_registry_round_trip(tmp_path) -> None:
    registry = _registry(tmp_path)
    revision = _revision_with_claim(revision_id="rev-1")
    registry.register(revision, acl_labels=("team-a",), validated_positions=(revision.blocks[0].block_id,))
    registry.promote_to_verified(
        revision.revision_id,
        revision.blocks[0].block_id,
        {"fork_node_id": "fork-1", "fork_revision_id": "fr-1"},
    )
    payload = json.loads(json.dumps(registry.to_dict()))
    restored = ProjectionRegistry.from_dict(payload, path=tmp_path / "copy.json", clock=_clock)
    assert restored.to_dict() == registry.to_dict()
    assert restored.search(principal_labels=("team-a",))[0].status == CARD_STATUS_VERIFIED


def test_withdraw_superseded_keeps_named_revision_and_audits_the_rest(tmp_path) -> None:
    registry = _registry(tmp_path)
    registry.register(_fallback(SESSION, "rev-1"), acl_labels=("team",))
    diagnosis = _fallback(SESSION, "rev-diag", purpose=PURPOSE_FAILURE_DIAGNOSIS)
    registry.register(diagnosis, acl_labels=("team",))
    kept = registry.withdraw_superseded(SESSION, PURPOSE_POST_RUN_INDEX, "rev-1")
    assert kept == ()
    removed = registry.withdraw_superseded(SESSION, PURPOSE_POST_RUN_INDEX, "rev-2")
    assert [record.reason for record in removed] == [REMOVAL_SUPERSEDED]
    assert [record.card.source_pointer.view_revision_id for record in removed] == ["rev-1"]
    assert registry.removed_records() == removed
    found = registry.search(principal_labels=("team",))
    assert {card.source_pointer.view_revision_id for card in found} == {"rev-diag"}
    registry.register(_fallback(SESSION, "rev-2"), acl_labels=("team",))
    found = registry.search(principal_labels=("team",))
    assert {card.source_pointer.view_revision_id for card in found} == {"rev-2", "rev-diag"}


def _authoritative(status: str, position_match: bool) -> dict:
    return {
        "fork_node_id": "fork-1",
        "revision_id": "fr-1",
        "status": status,
        "position_match": position_match,
    }


def test_promote_follows_authoritative_fork_status_and_position(tmp_path) -> None:
    revision = _revision_with_claim()
    block_id = revision.blocks[0].block_id
    state = _authoritative("provisional", True)

    def grounding_query(view_revision_id: str, position_id: str):
        if view_revision_id == revision.revision_id and position_id == block_id:
            return dict(state)
        return None

    registry = _registry(tmp_path, grounding_query=grounding_query)
    registry.register(revision, acl_labels=("team-a",))
    provenance = {"fork_node_id": "fork-1", "revision_id": "fr-1"}
    with pytest.raises(ValueError, match="provisional"):
        registry.promote_to_verified(revision.revision_id, block_id, provenance)
    state["status"] = "validated"
    state["position_match"] = False
    with pytest.raises(ValueError, match="position"):
        registry.promote_to_verified(revision.revision_id, block_id, provenance)
    state["position_match"] = True
    card = registry.promote_to_verified(revision.revision_id, block_id, provenance)
    assert card.status == CARD_STATUS_VERIFIED
    assert card.fork_provenance == provenance


def test_promote_rejects_fallback_view_even_with_validated_grounding(tmp_path) -> None:
    revision = _fallback(SESSION, "rev-fb")
    block_id = revision.blocks[0].block_id

    def grounding_query(view_revision_id: str, position_id: str):
        del view_revision_id, position_id
        return _authoritative("validated", True)

    registry = _registry(tmp_path, grounding_query=grounding_query)
    registry.register(revision, acl_labels=("team-a",))
    with pytest.raises(ValueError, match="ADR #72"):
        registry.promote_to_verified(
            revision.revision_id,
            block_id,
            {"fork_node_id": "fork-1", "revision_id": "fr-1"},
        )
    assert registry.search(principal_labels=("team-a",))[0].status == CARD_STATUS_PROVISIONAL


def test_promote_rejects_static_positions_without_fork_provenance(tmp_path) -> None:
    revision = _revision_with_claim()
    registry = _registry(tmp_path)
    registry.register(revision, acl_labels=("team-a",), validated_positions=("claim-1",))
    with pytest.raises(ValueError, match="fork_provenance"):
        registry.promote_to_verified(revision.revision_id, "claim-1", {"fork_revision_id": "fr-1"})
    assert registry.search(principal_labels=("team-a",), kind=CARD_KIND_CLAIM)[0].status == (
        CARD_STATUS_PROVISIONAL
    )


def test_acl_change_calls_revalidation_callback_with_session(tmp_path) -> None:
    seen: list[str] = []
    registry = _registry(tmp_path, revalidation_callback=seen.append)
    registry.register(_fallback(SESSION, "rev-1"), acl_labels=("team-a",))
    registry.register(_fallback(OTHER, "rev-2"), acl_labels=("team-b",))
    removed = registry.handle_acl_change(SESSION, ("team-b",))
    assert len(removed) == 1
    assert seen == [SESSION]


def test_acl_change_without_revalidation_callback_warns_once(tmp_path, caplog) -> None:
    registry = _registry(tmp_path)
    registry.register(_fallback(SESSION, "rev-1"), acl_labels=("team-a",))
    registry.register(_fallback(OTHER, "rev-2"), acl_labels=("team-b",))
    with caplog.at_level(logging.WARNING):
        registry.handle_acl_change(SESSION, ("team-c",))
        registry.handle_acl_change(OTHER, ("team-d",))
    warnings = [record for record in caplog.records if "revalidation_callback" in record.message]
    assert len(warnings) == 1
    assert warnings[0].levelno == logging.WARNING


def test_withdraw_superseded_calls_revalidation_callback(tmp_path) -> None:
    seen: list[str] = []
    registry = _registry(tmp_path, revalidation_callback=seen.append)
    registry.register(_fallback(SESSION, "rev-1"), acl_labels=("team",))
    registry.register(
        _fallback(SESSION, "rev-diag", purpose=PURPOSE_FAILURE_DIAGNOSIS),
        acl_labels=("team",),
    )
    removed = registry.withdraw_superseded(SESSION, PURPOSE_POST_RUN_INDEX, "rev-2")
    assert [record.card.source_pointer.view_revision_id for record in removed] == ["rev-1"]
    assert seen == ["rev-1"]

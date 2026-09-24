# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Typed hybrid retrieval facade (slice S13)."""

from __future__ import annotations

from pathlib import Path

import pytest

from openviking.session.ao_ledger import AOLedger
from openviking.session.causal_bridges import PATH_MAX_ENTITIES, check_path_budget
from openviking.session.causal_experiences import (
    SEVEN_SECTION_KEYS,
    CausalExperiencesStore,
    ServerChecks,
)
from openviking.session.influence_projection import ProjectionRegistry
from openviking.session.influence_view import (
    AORange,
    CoverageManifest,
    InfluenceBlock,
    InfluenceClaim,
    InfluenceViewDraft,
    PURPOSE_POST_RUN_INDEX,
    SegmentRef,
    SourceRef,
)
from openviking.session.retrieval_facade import (
    FEATURE_SPEC_VERSION,
    ForkBranchAdapter,
    HybridRetrievalFacade,
    InfluenceProjectionAdapter,
    QuotaError,
    RetrievalCard,
    RetrievalProfile,
    SegmentAtomAdapter,
)

_CAUSAL = {"skill_trajectory_mode": "causal"}
_CHECKS = ServerChecks(anchor_exists=True, refs_exist=True, provenance_valid=True)
_EVIDENCE = {
    "observed_branch_refs": [
        {"branch_id": "br-obs", "evidence_status": "real", "ref": "run-1"}
    ],
    "independent_control_refs": [
        {"ref_kind": "independent_control", "ref_id": "ctrl-1"}
    ],
}
_PRINCIPAL = ("team-a",)
_CREATED = "2026-01-02T00:00:00.000Z"


def _sections() -> dict[str, str]:
    return {key: f"{key} body" for key in SEVEN_SECTION_KEYS}


def _evidence_ref() -> dict[str, object]:
    return {
        "ao_id": "ao-e1",
        "source_session_id": "sess-1",
        "source_archive_id": "arch-1",
        "archive_commit_watermark": "commit-1",
        "source_sequence": 4,
        "source_read_snapshot_watermark": "read-wm-1",
        "evidence_role": "contemporaneous_basis",
        "evidence_content_hash": "hash-ao-e1",
        "captured_state": "committed",
    }


def _store(root: Path) -> CausalExperiencesStore:
    return CausalExperiencesStore(root, config=_CAUSAL)


def _commit_fork(
    store: CausalExperiencesStore,
    *,
    ao_id: str,
    workspace: str,
    status: str,
    branches: list[dict[str, object]] | None = None,
) -> str:
    draft = store.submit_fork_draft(
        {
            "workspace_id": workspace,
            "anchor": {
                "anchor_kind": "ao",
                "ao_id": ao_id,
                "source_session_id": "sess-1",
                "anchor_sequence": 10,
            },
            "seven_sections": _sections(),
            "contemporaneous_basis": [_evidence_ref()],
            "hindsight_attribution": [],
            "used_skills": [],
            "branches": branches or [],
            "diagnosis_run_id": f"diag-{ao_id}",
            "model_version": "diag-model-v1",
        }
    )
    provisional = store.commit_provisional(draft, _CHECKS)
    if status == "validated":
        store.append_validated(provisional.fork_node_id, _EVIDENCE)
    elif status != "provisional":
        raise AssertionError(status)
    return provisional.fork_node_id


def _ref(sequence: int, *, session_id: str = "sess-1") -> SourceRef:
    return SourceRef(
        ao_id=f"ao-{session_id}-{sequence}",
        sequence=sequence,
        participant_id="agent-a",
        segment_id="seg-a",
        session_id=session_id,
        content_hash=f"hash-{session_id}-{sequence}",
    )


def _block(block_id: str, sequence: int, *, session_id: str = "sess-1") -> InfluenceBlock:
    ref = _ref(sequence, session_id=session_id)
    return InfluenceBlock(
        block_id=block_id,
        role="agent_action",
        participant_id="agent-a",
        segment_ref=SegmentRef(segment_id="seg-a", session_id=session_id),
        ao_start_seq=sequence,
        ao_end_seq=sequence,
        summary=f"summary-{block_id}",
        input="",
        output="",
        authorship="agent-a",
        granularity_reason="one action",
        source_content_hashes=(ref.content_hash,),
        source_refs=(ref,),
    )


def _coverage(blocks: list[InfluenceBlock]) -> CoverageManifest:
    refs = tuple(ref for block in blocks for ref in block.source_refs)
    return CoverageManifest(
        total_ao_range=AORange(
            start_seq=min(ref.sequence for ref in refs),
            end_seq=max(ref.sequence for ref in refs),
        ),
        inspected_refs=refs,
        included_ranges=tuple(
            AORange(
                start_seq=block.ao_start_seq,
                end_seq=block.ao_end_seq,
                segment_id="seg-a",
            )
            for block in blocks
        ),
        omitted_ranges=(),
        snapshot_watermark="wm-1",
    )


def _freeze(
    blocks: list[InfluenceBlock],
    claims: list[InfluenceClaim],
    *,
    revision_id: str,
    session_id: str = "sess-1",
):
    draft = InfluenceViewDraft(
        view_id=f"view-{revision_id}",
        purpose=PURPOSE_POST_RUN_INDEX,
        session_id=session_id,
        task_run_id=f"task-{revision_id}",
        coverage=_coverage(blocks),
        blocks=blocks,
        claims=claims,
    )
    return draft.freeze(revision_id=revision_id, frozen_at=_CREATED)


def _facade(
    root: Path,
    store: CausalExperiencesStore,
    registry: ProjectionRegistry,
    session_dir: Path,
    session_id: str,
    *,
    views: list | None = None,
) -> HybridRetrievalFacade:
    projection = InfluenceProjectionAdapter(registry, views=views)
    return HybridRetrievalFacade(
        ForkBranchAdapter(root, store),
        projection,
        SegmentAtomAdapter(session_dir, session_id),
    )


def _append_ao(ledger: AOLedger, *, labels: list[str], record_kind: str = "atom") -> str:
    record = ledger.append(
        {"tool": "read_file", "record_kind": record_kind, "acl_labels": labels},
        {"kind": "text", "summary": [record_kind]},
    )
    return record.ao_id


def test_corpus_quotas_and_profile_over_cap(tmp_path: Path) -> None:
    root = tmp_path / "lane"
    store = _store(root)
    fork_ids = [
        _commit_fork(store, ao_id=f"ao-fork-{index}", workspace="team-a", status="validated")
        for index in range(8)
    ]
    blocks = [_block(f"b{index}", index) for index in range(1, 7)]
    revision = _freeze(blocks, [], revision_id="rev-quota")
    registry = ProjectionRegistry(root / "projections.json")
    registry.register(revision, acl_labels=_PRINCIPAL)
    session_dir = root / "sess-facts"
    ledger = AOLedger(session_dir, "sess-facts")
    for _ in range(4):
        _append_ao(ledger, labels=["team-a"])

    facade = _facade(root, store, registry, session_dir, "sess-facts", views=[revision])
    result = facade.retrieve({}, _PRINCIPAL, RetrievalProfile())
    forks = [card for card in result.cards if card.kind == "fork"]
    projections = [card for card in result.cards if card.kind.endswith("_projection")]
    facts = [card for card in result.cards if card.kind in {"segment", "atom"}]
    assert len(forks) == 6
    assert [card.source_pointer["fork_node_id"] for card in forks] == fork_ids[:6]
    assert len(projections) == 4
    assert len(facts) == 2
    assert {card.kind for card in facts} == {"atom"}
    assert result.feature_version == FEATURE_SPEC_VERSION == facade.feature_version
    assert RetrievalCard.from_dict(result.cards[0].to_dict()) == result.cards[0]

    with pytest.raises(QuotaError) as raised:
        facade.retrieve(
            {},
            _PRINCIPAL,
            RetrievalProfile(fork_branch=7, projection=4, segment_atom=2),
        )
    assert raised.value.total == 13
    assert raised.value.cap == 12


def test_branch_cards_follow_parent_channel(tmp_path: Path) -> None:
    root = tmp_path / "branches"
    store = _store(root)
    branch = {
        "branch_id": "br-1",
        "evidence_status": "real",
        "ao_sequence": ["ao-step"],
        "guidance": {"summary": "retry the read"},
    }
    _commit_fork(
        store,
        ao_id="ao-prov",
        workspace="team-a",
        status="provisional",
        branches=[branch],
    )
    _commit_fork(
        store,
        ao_id="ao-val",
        workspace="team-a",
        status="validated",
        branches=[branch],
    )
    registry = ProjectionRegistry(root / "projections.json")
    session_dir = root / "sess"
    AOLedger(session_dir, "sess")
    facade = _facade(root, store, registry, session_dir, "sess")
    result = facade.retrieve({}, _PRINCIPAL, RetrievalProfile(fork_branch=4, projection=0, segment_atom=0))
    assert [(card.kind, card.channel, card.status_label) for card in result.cards] == [
        ("fork", "verified", "validated"),
        ("branch", "verified", "validated"),
        ("fork", "provisional", "ForkCandidate"),
        ("branch", "provisional", "ForkCandidate"),
    ]


def test_channel_backfill_keeps_in_channel_rank(tmp_path: Path) -> None:
    verified_root = tmp_path / "verified-backfill"
    verified_store = _store(verified_root)
    validated_ids = [
        _commit_fork(
            verified_store,
            ao_id=f"ao-v-{index}",
            workspace="team-a",
            status="validated",
        )
        for index in range(9)
    ]
    provisional_ids = [
        _commit_fork(
            verified_store,
            ao_id=f"ao-p-{index}",
            workspace="team-a",
            status="provisional",
        )
        for index in range(3)
    ]
    registry = ProjectionRegistry(verified_root / "projections.json")
    session_dir = verified_root / "sess"
    AOLedger(session_dir, "sess")
    facade = _facade(verified_root, verified_store, registry, session_dir, "sess")
    result = facade.retrieve(
        {},
        _PRINCIPAL,
        RetrievalProfile(fork_branch=12, projection=0, segment_atom=0),
    )
    by_fork = {card.source_pointer["fork_node_id"]: card for card in result.cards}
    assert by_fork[validated_ids[8]].channel == "verified"
    assert by_fork[validated_ids[8]].in_channel_rank == 9
    assert by_fork[validated_ids[8]].status_label == "validated"
    assert [by_fork[fork_id].in_channel_rank for fork_id in provisional_ids] == [1, 2, 3]
    assert {by_fork[fork_id].channel for fork_id in provisional_ids} == {"provisional"}

    short_root = tmp_path / "provisional-backfill"
    short_store = _store(short_root)
    for index in range(2):
        _commit_fork(short_store, ao_id=f"ao-sv-{index}", workspace="team-a", status="validated")
    provisional_ids = [
        _commit_fork(
            short_store,
            ao_id=f"ao-sp-{index}",
            workspace="team-a",
            status="provisional",
        )
        for index in range(6)
    ]
    short_registry = ProjectionRegistry(short_root / "projections.json")
    short_session = short_root / "sess"
    AOLedger(short_session, "sess")
    short_facade = _facade(short_root, short_store, short_registry, short_session, "sess")
    backfilled = short_facade.retrieve(
        {},
        _PRINCIPAL,
        RetrievalProfile(fork_branch=8, projection=0, segment_atom=0),
    )
    provisional_cards = [card for card in backfilled.cards if card.channel == "provisional"]
    assert sorted(card.in_channel_rank for card in provisional_cards) == [1, 2, 3, 4, 5, 6]
    kept = {card.source_pointer["fork_node_id"]: card for card in provisional_cards}
    assert kept[provisional_ids[4]].in_channel_rank == 5
    assert kept[provisional_ids[5]].in_channel_rank == 6
    assert kept[provisional_ids[5]].status_label == "ForkCandidate"
    assert len([card for card in backfilled.cards if card.channel == "verified"]) == 2


def test_acl_filters_non_intersecting_labels(tmp_path: Path) -> None:
    root = tmp_path / "acl"
    store = _store(root)
    visible_fork = _commit_fork(store, ao_id="ao-visible", workspace="team-a", status="validated")
    hidden_fork = _commit_fork(store, ao_id="ao-hidden", workspace="secret", status="validated")
    visible_rev = _freeze([_block("b-vis", 1, session_id="sess-vis")], [], revision_id="rev-vis", session_id="sess-vis")
    hidden_rev = _freeze(
        [_block("b-hid", 1, session_id="sess-hid")],
        [],
        revision_id="rev-hid",
        session_id="sess-hid",
    )
    registry = ProjectionRegistry(root / "projections.json")
    registry.register(visible_rev, acl_labels=("team-a",))
    registry.register(hidden_rev, acl_labels=("secret",))
    session_dir = root / "sess-acl"
    ledger = AOLedger(session_dir, "sess-acl")
    visible_ao = _append_ao(ledger, labels=["team-a"])
    hidden_ao = _append_ao(ledger, labels=["secret"])

    facade = _facade(root, store, registry, session_dir, "sess-acl", views=[visible_rev, hidden_rev])
    result = facade.retrieve({}, _PRINCIPAL)
    pointers = [card.source_pointer for card in result.cards]
    assert any(item.get("fork_node_id") == visible_fork for item in pointers)
    assert all(item.get("fork_node_id") != hidden_fork for item in pointers)
    assert any(item.get("view_revision_id") == "rev-vis" for item in pointers)
    assert all(item.get("view_revision_id") != "rev-hid" for item in pointers)
    assert any(item.get("ao_id") == visible_ao for item in pointers)
    assert all(item.get("ao_id") != hidden_ao for item in pointers)
    assert result.cards
    assert all(set(card.acl_labels) & set(_PRINCIPAL) for card in result.cards)


def test_dependency_pattern_and_bounded_neighborhood(tmp_path: Path) -> None:
    root = tmp_path / "deps"
    store = _store(root)
    blocks = [_block(f"b{index}", index) for index in range(1, 4)]
    match = InfluenceClaim(
        claim_id="c-match",
        source_block_id="b1",
        target_block_id="b2",
        carried_artifact="patch",
        downstream_effect="tests passed",
        relation_type="influence",
    )
    other = InfluenceClaim(
        claim_id="c-other",
        source_block_id="b2",
        target_block_id="b3",
        carried_artifact="notes",
        downstream_effect="no change",
        relation_type="influence",
    )
    revision = _freeze(blocks, [match, other], revision_id="rev-dep")
    registry = ProjectionRegistry(root / "projections.json")
    registry.register(revision, acl_labels=_PRINCIPAL, validated_positions=("c-match",))
    registry.promote_to_verified(
        revision.revision_id,
        "c-match",
        {"fork_revision_id": "fr-1"},
    )
    session_dir = root / "sess"
    AOLedger(session_dir, "sess")
    facade = _facade(root, store, registry, session_dir, "sess", views=[revision])
    result = facade.retrieve(
        {"dependency_pattern": {"wanted_artifact": "patch", "wanted_effect": "tests passed"}},
        _PRINCIPAL,
        RetrievalProfile(fork_branch=0, projection=4, segment_atom=0),
    )
    claims = [card for card in result.cards if card.kind == "claim_projection"]
    assert len(claims) == 1
    assert claims[0].source_pointer["carried_artifact"] == "patch"
    assert claims[0].source_pointer["downstream_effect"] == "tests passed"
    assert claims[0].channel == "verified"
    assert claims[0].status_label == "verified"
    assert all(card.source_pointer.get("block_or_claim_id") != "c-other" for card in result.cards)

    chain_blocks = [_block(f"n{index}", index) for index in range(1, 15)]
    chain_claims = [
        InfluenceClaim(
            claim_id=f"edge-{index}",
            source_block_id=f"n{index}",
            target_block_id=f"n{index + 1}",
            carried_artifact=f"art-{index}",
            downstream_effect=f"eff-{index}",
            relation_type="influence",
        )
        for index in range(1, 14)
    ]
    chain = _freeze(chain_blocks, chain_claims, revision_id="rev-chain")
    cycle_blocks = [_block(f"c{index}", index, session_id="sess-cycle") for index in range(1, 4)]
    cycle_claims = [
        InfluenceClaim(
            claim_id="cy-1",
            source_block_id="c1",
            target_block_id="c2",
            carried_artifact="loop",
            downstream_effect="step",
            relation_type="influence",
        ),
        InfluenceClaim(
            claim_id="cy-2",
            source_block_id="c2",
            target_block_id="c3",
            carried_artifact="loop",
            downstream_effect="step",
            relation_type="influence",
        ),
        InfluenceClaim(
            claim_id="cy-3",
            source_block_id="c3",
            target_block_id="c1",
            carried_artifact="loop",
            downstream_effect="step",
            relation_type="influence",
        ),
    ]
    cycle = _freeze(cycle_blocks, cycle_claims, revision_id="rev-cycle", session_id="sess-cycle")
    facade.projection.bind_view(chain)
    facade.projection.bind_view(cycle)
    registry_bytes = (root / "projections.json").read_bytes()
    causal_before = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in (root / "causal-experiences").rglob("*")
        if path.is_file()
    }

    neighborhood = facade.expand_claim_neighborhood("rev-chain", "edge-1", max_entities=12)
    assert len(neighborhood.entities) == PATH_MAX_ENTITIES == 12
    assert len(neighborhood.entities) == len(set(neighborhood.entities))
    assert neighborhood.truncated is True
    assert neighborhood.entities[0] == "n1"
    assert check_path_budget(
        list(neighborhood.entities),
        [
            {
                "kind": "evidence",
                "bridge_id": claim_id,
                "source_ref": {"type": "block", "id": source},
                "target_ref": {"type": "block", "id": target},
                "relation_type": "claim",
            }
            for claim_id, source, target in neighborhood.edges
        ],
        max_entities=12,
    ) == []

    cycled = facade.expand_claim_neighborhood("rev-cycle", "cy-1")
    assert cycled.truncated is True
    assert cycled.entities == ("c1", "c2", "c3")
    assert len(cycled.entities) == len(set(cycled.entities)) <= 12
    assert (root / "projections.json").read_bytes() == registry_bytes
    causal_after = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in (root / "causal-experiences").rglob("*")
        if path.is_file()
    }
    assert causal_after == causal_before


def test_fact_channel_does_not_consume_state_slots(tmp_path: Path) -> None:
    root = tmp_path / "facts"
    store = _store(root)
    validated_ids = [
        _commit_fork(store, ao_id=f"ao-fact-{index}", workspace="team-a", status="validated")
        for index in range(10)
    ]
    registry = ProjectionRegistry(root / "projections.json")
    session_dir = root / "sess-facts"
    ledger = AOLedger(session_dir, "sess-facts")
    segment_id = _append_ao(ledger, labels=["team-a"], record_kind="segment")
    atom_id = _append_ao(ledger, labels=["team-a"], record_kind="atom")
    facade = _facade(root, store, registry, session_dir, "sess-facts")
    result = facade.retrieve(
        {},
        _PRINCIPAL,
        RetrievalProfile(fork_branch=10, projection=0, segment_atom=2),
    )
    verified = [card for card in result.cards if card.channel == "verified"]
    facts = [card for card in result.cards if card.kind in {"segment", "atom"}]
    assert [card.source_pointer["fork_node_id"] for card in verified] == validated_ids
    assert [card.in_channel_rank for card in verified] == list(range(1, 11))
    assert len(facts) == 2
    assert {card.channel for card in facts} == {"fact"}
    assert {card.status_label for card in facts} == {"fact"}
    assert {card.kind for card in facts} == {"segment", "atom"}
    assert {card.source_pointer["ao_id"] for card in facts} == {segment_id, atom_id}
    assert all(card.in_channel_rank >= 1 for card in facts)
    assert not any(card.channel in {"verified", "provisional"} for card in facts)

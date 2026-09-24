# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Typed hybrid retrieval facade for the causal-memory lane (Q112-A / Q122-A).

Three corpus adapters share one card envelope. Storage and identity stay
separate: fork/branch, block/claim projection, and segment/atom facts are
not collapsed into one corpus. Verified/provisional slotting (8/4, with
backfill) applies only to stateful cards. Segment/atom cards stay on the
fact channel. Dependency-pattern hits expand inside one frozen view at read
time and do not write a stored object.
"""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Protocol

from openviking.session.ao_ledger_reader import list_ao_ledger
from openviking.session.causal_bridges import (
    PATH_MAX_ENTITIES,
    PATH_MAX_HYPOTHESIS_HOPS,
    check_path_budget,
)
from openviking.session.causal_experiences import (
    INDEX_FILENAME,
    NAMESPACE_DIRNAME,
    FORK_CANDIDATE_ROLE,
    CausalExperiencesStore,
    ForkStatus,
)
from openviking.session.influence_projection import InfluenceProjection, ProjectionRegistry
from openviking.session.influence_view import InfluenceClaim, InfluenceViewRevision

FEATURE_SPEC_VERSION = "ranker-feature-spec-v1"

CAUSAL_MEMORY_CAP = 12
DEFAULT_FORK_BRANCH_QUOTA = 6
DEFAULT_PROJECTION_QUOTA = 4
DEFAULT_SEGMENT_ATOM_QUOTA = 2
VERIFIED_SLOTS = 8
PROVISIONAL_SLOTS = 4

CardKind = Literal[
    "fork",
    "branch",
    "block_projection",
    "claim_projection",
    "segment",
    "atom",
]
CardChannel = Literal["verified", "provisional", "fact"]

CARD_KINDS = frozenset(
    ("fork", "branch", "block_projection", "claim_projection", "segment", "atom")
)
CARD_CHANNELS = frozenset(("verified", "provisional", "fact"))
STATEFUL_CHANNELS = frozenset(("verified", "provisional"))
FACT_KINDS = frozenset(("segment", "atom"))

AclFilter = Callable[["RetrievalCard", Sequence[str]], bool]
RerankFn = Callable[[Sequence["RetrievalCard"], Mapping[str, Any]], Sequence["RetrievalCard"]]


class QuotaError(ValueError):
    """Raised when a retrieval profile exceeds the causal-memory lane cap."""

    def __init__(self, message: str, *, total: int, cap: int = CAUSAL_MEMORY_CAP) -> None:
        self.total = total
        self.cap = cap
        super().__init__(message)


def _require_mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return dict(value)


def _json_copy(value: Any, field: str) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be JSON-serializable") from exc


def _require_str(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty str")
    return value


def _require_rank(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("in_channel_rank must be an int >= 0")
    return value


def _normalize_labels(value: Sequence[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("acl_labels must be a sequence of str")
    labels: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = _require_str(item, "acl_labels")
        if text in seen:
            continue
        seen.add(text)
        labels.append(text)
    return tuple(labels)


@dataclass(frozen=True)
class RetrievalCard:
    """Cross-corpus envelope. ``kind`` keeps each corpus identity intact."""

    kind: CardKind
    channel: CardChannel
    in_channel_rank: int
    status_label: str
    source_pointer: dict[str, Any]
    payload_ref: str
    score_breakdown: dict[str, Any]
    acl_labels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in CARD_KINDS:
            raise ValueError(f"kind must be one of {sorted(CARD_KINDS)}, got {self.kind!r}")
        if self.channel not in CARD_CHANNELS:
            raise ValueError(
                f"channel must be one of {sorted(CARD_CHANNELS)}, got {self.channel!r}"
            )
        object.__setattr__(self, "in_channel_rank", _require_rank(self.in_channel_rank))
        object.__setattr__(self, "status_label", _require_str(self.status_label, "status_label"))
        object.__setattr__(self, "payload_ref", _require_str(self.payload_ref, "payload_ref"))
        pointer = _json_copy(_require_mapping(self.source_pointer, "source_pointer"), "source_pointer")
        if not isinstance(pointer, dict):
            raise ValueError("source_pointer must be a dict")
        breakdown = _json_copy(
            _require_mapping(self.score_breakdown, "score_breakdown"), "score_breakdown"
        )
        if not isinstance(breakdown, dict):
            raise ValueError("score_breakdown must be a dict")
        object.__setattr__(self, "source_pointer", pointer)
        object.__setattr__(self, "score_breakdown", breakdown)
        object.__setattr__(self, "acl_labels", _normalize_labels(self.acl_labels))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "channel": self.channel,
            "in_channel_rank": self.in_channel_rank,
            "status_label": self.status_label,
            "source_pointer": _json_copy(self.source_pointer, "source_pointer"),
            "payload_ref": self.payload_ref,
            "score_breakdown": _json_copy(self.score_breakdown, "score_breakdown"),
            "acl_labels": list(self.acl_labels),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RetrievalCard:
        payload = _require_mapping(data, "retrieval_card")
        labels = payload.get("acl_labels", [])
        if not isinstance(labels, list):
            raise ValueError("acl_labels must be a list")
        return cls(
            kind=payload.get("kind"),  # type: ignore[arg-type]
            channel=payload.get("channel"),  # type: ignore[arg-type]
            in_channel_rank=payload.get("in_channel_rank", 0),
            status_label=str(payload.get("status_label") or ""),
            source_pointer=_require_mapping(payload.get("source_pointer"), "source_pointer"),
            payload_ref=str(payload.get("payload_ref") or ""),
            score_breakdown=_require_mapping(payload.get("score_breakdown"), "score_breakdown"),
            acl_labels=tuple(str(item) for item in labels),
        )


@dataclass(frozen=True)
class RetrievalQuery:
    """Query text plus an optional dependency-pattern signature (Q122-A)."""

    text: str = ""
    dependency_pattern: dict[str, str] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise ValueError("text must be a str")
        pattern = self.dependency_pattern
        if pattern is None:
            return
        copied = _require_mapping(pattern, "dependency_pattern")
        normalized: dict[str, str] = {}
        for key in ("wanted_artifact", "wanted_effect"):
            if key not in copied or copied[key] is None:
                continue
            normalized[key] = _require_str(copied[key], f"dependency_pattern.{key}")
        object.__setattr__(self, "dependency_pattern", normalized or None)

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "dependency_pattern": self.dependency_pattern}


def coerce_query(value: RetrievalQuery | Mapping[str, Any] | str | None) -> RetrievalQuery:
    if value is None:
        return RetrievalQuery()
    if isinstance(value, RetrievalQuery):
        return value
    if isinstance(value, str):
        return RetrievalQuery(text=value)
    payload = _require_mapping(value, "query")
    pattern = payload.get("dependency_pattern")
    return RetrievalQuery(
        text=str(payload.get("text") or ""),
        dependency_pattern=None if pattern is None else dict(pattern),
    )


@dataclass(frozen=True)
class RetrievalProfile:
    """Per-corpus budgets inside the causal-memory lane. Their sum cannot exceed 12."""

    fork_branch: int = DEFAULT_FORK_BRANCH_QUOTA
    projection: int = DEFAULT_PROJECTION_QUOTA
    segment_atom: int = DEFAULT_SEGMENT_ATOM_QUOTA

    def to_dict(self) -> dict[str, int]:
        return {
            "fork_branch": self.fork_branch,
            "projection": self.projection,
            "segment_atom": self.segment_atom,
        }


def _quota_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise QuotaError(f"{field} must be an int", total=0)
    if value < 0:
        raise QuotaError(f"{field} must be >= 0", total=value)
    return value


def coerce_profile(value: RetrievalProfile | Mapping[str, Any] | None) -> RetrievalProfile:
    if value is None:
        profile = RetrievalProfile()
    elif isinstance(value, RetrievalProfile):
        profile = value
    else:
        payload = _require_mapping(value, "profile")
        profile = RetrievalProfile(
            fork_branch=_quota_int(payload.get("fork_branch", DEFAULT_FORK_BRANCH_QUOTA), "fork_branch"),
            projection=_quota_int(payload.get("projection", DEFAULT_PROJECTION_QUOTA), "projection"),
            segment_atom=_quota_int(
                payload.get("segment_atom", DEFAULT_SEGMENT_ATOM_QUOTA), "segment_atom"
            ),
        )
    fork_branch = _quota_int(profile.fork_branch, "fork_branch")
    projection = _quota_int(profile.projection, "projection")
    segment_atom = _quota_int(profile.segment_atom, "segment_atom")
    total = fork_branch + projection + segment_atom
    if total > CAUSAL_MEMORY_CAP:
        raise QuotaError(
            f"retrieval profile sums to {total}, above the causal-memory cap {CAUSAL_MEMORY_CAP}",
            total=total,
        )
    return RetrievalProfile(
        fork_branch=fork_branch, projection=projection, segment_atom=segment_atom
    )


@dataclass(frozen=True)
class AssembledPath:
    """Read-time claim neighborhood. Not a stored object."""

    view_revision_id: str
    claim_id: str
    entities: tuple[str, ...]
    edges: tuple[tuple[str, str, str], ...]
    truncated: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "view_revision_id": self.view_revision_id,
            "claim_id": self.claim_id,
            "entities": list(self.entities),
            "edges": [
                {"claim_id": claim_id, "source_block_id": source, "target_block_id": target}
                for claim_id, source, target in self.edges
            ],
            "truncated": self.truncated,
        }


@dataclass(frozen=True)
class RetrievalResult:
    cards: tuple[RetrievalCard, ...]
    feature_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "cards": [card.to_dict() for card in self.cards],
            "feature_version": self.feature_version,
        }


class CorpusAdapter(Protocol):
    """One corpus search. Results keep that corpus's kind."""

    name: str

    def search(self, query: RetrievalQuery, limit: int) -> list[RetrievalCard]:
        """Return at most ``limit`` cards for ``query``."""


class RerankHook(Protocol):
    def rerank(
        self, cards: Sequence[RetrievalCard], context: Mapping[str, Any]
    ) -> Sequence[RetrievalCard]:
        """Return cards in reranked order. The default hook is identity."""


class IdentityRerank:
    """Placeholder ranker. Records no mutation; feature version lives on the facade."""

    def rerank(
        self, cards: Sequence[RetrievalCard], context: Mapping[str, Any]
    ) -> Sequence[RetrievalCard]:
        del context
        return list(cards)


def default_acl_filter(card: RetrievalCard, principal_labels: Sequence[str]) -> bool:
    """Keep a card when its labels intersect the principal's labels."""
    if isinstance(principal_labels, (str, bytes)) or not isinstance(principal_labels, Sequence):
        raise ValueError("principal_labels must be a sequence of str")
    principal = set(principal_labels)
    return bool(principal.intersection(card.acl_labels))


def _text_hits(query: RetrievalQuery, *parts: str) -> bool:
    if not query.text:
        return True
    haystack = "\n".join(parts)
    return query.text in haystack


def _score(query: RetrievalQuery) -> dict[str, Any]:
    return {"lexical": 1 if not query.text else 1}


def _channel_for_fork_status(status: str | None) -> CardChannel | None:
    if status == ForkStatus.VALIDATED:
        return "verified"
    if status == ForkStatus.PROVISIONAL:
        return "provisional"
    return None


def _status_label_for_fork(status: str) -> str:
    if status == ForkStatus.PROVISIONAL:
        return FORK_CANDIDATE_ROLE
    return status


def _limit(limit: int) -> int:
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise ValueError("limit must be an int >= 0")
    return limit


class ForkBranchAdapter:
    """Wraps ``CausalExperiencesStore`` reads. Latest revision only.

    Validated revisions enter the verified channel. Provisional revisions are
    consumed as ForkCandidate and enter the provisional channel. Draft and
    invalidated revisions stay out of retrieval. Branches share the parent
    revision's channel. ACL labels are the fork workspace id: the store has
    no separate label column.
    """

    name = "fork_branch"

    def __init__(self, root: str | Path, store: CausalExperiencesStore | None = None) -> None:
        self._root = Path(root)
        self._store = store if store is not None else CausalExperiencesStore(self._root)

    def search(self, query: RetrievalQuery | Mapping[str, Any] | None, limit: int) -> list[RetrievalCard]:
        parsed = coerce_query(query)
        cap = _limit(limit)
        cards: list[RetrievalCard] = []
        for fork_node_id in self._fork_ids():
            loaded = self._store.get_fork_node(fork_node_id)
            if loaded is None:
                continue
            latest = loaded.get("latest_revision")
            if not isinstance(latest, dict):
                continue
            channel = _channel_for_fork_status(
                latest.get("status") if isinstance(latest.get("status"), str) else None
            )
            if channel is None:
                continue
            status = str(latest["status"])
            blob = json.dumps(latest, ensure_ascii=False, sort_keys=True)
            if not _text_hits(parsed, blob):
                continue
            labels = _normalize_labels((str(latest.get("workspace_id") or ""),))
            revision_id = str(latest.get("revision_id") or "")
            fork_card = RetrievalCard(
                kind="fork",
                channel=channel,
                in_channel_rank=0,
                status_label=_status_label_for_fork(status),
                source_pointer={
                    "corpus": self.name,
                    "fork_node_id": fork_node_id,
                    "revision_id": revision_id,
                },
                payload_ref=f"fork:{fork_node_id}@{revision_id}",
                score_breakdown=_score(parsed),
                acl_labels=labels,
            )
            cards.append(fork_card)
            branches = latest.get("branches") if isinstance(latest.get("branches"), list) else []
            ordered = sorted(
                (item for item in branches if isinstance(item, dict)),
                key=lambda item: str(item.get("branch_id") or ""),
            )
            for branch in ordered:
                branch_id = str(branch.get("branch_id") or "")
                if not branch_id:
                    continue
                cards.append(
                    RetrievalCard(
                        kind="branch",
                        channel=channel,
                        in_channel_rank=0,
                        status_label=_status_label_for_fork(status),
                        source_pointer={
                            "corpus": self.name,
                            "fork_node_id": fork_node_id,
                            "revision_id": revision_id,
                            "branch_id": branch_id,
                            "evidence_status": branch.get("evidence_status"),
                        },
                        payload_ref=f"branch:{fork_node_id}@{revision_id}/{branch_id}",
                        score_breakdown=_score(parsed),
                        acl_labels=labels,
                    )
                )
            if len(cards) >= cap:
                break
        return cards[:cap]

    def _fork_ids(self) -> list[str]:
        index_path = self._root / NAMESPACE_DIRNAME / INDEX_FILENAME
        if not index_path.is_file():
            return []
        ordered: list[str] = []
        seen: set[str] = set()
        for line in index_path.read_text(encoding="utf-8").splitlines():
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            fork_node_id = row.get("fork_node_id")
            if isinstance(fork_node_id, str) and fork_node_id and fork_node_id not in seen:
                seen.add(fork_node_id)
                ordered.append(fork_node_id)
        return ordered


def _claim_fields(
    card: InfluenceProjection, views: Mapping[str, InfluenceViewRevision]
) -> dict[str, str] | None:
    if card.kind != "claim":
        return None
    view = views.get(card.source_pointer.view_revision_id)
    if view is not None:
        for claim in view.claims:
            if claim.claim_id == card.source_pointer.block_or_claim_id:
                return {
                    "carried_artifact": claim.carried_artifact,
                    "downstream_effect": claim.downstream_effect,
                }
    summary = card.summary
    marker = " -> "
    if marker in summary:
        artifact, effect = summary.split(marker, 1)
        if artifact and effect:
            return {"carried_artifact": artifact, "downstream_effect": effect}
    return {"carried_artifact": "", "downstream_effect": ""}


def _pattern_matches(fields: Mapping[str, str], pattern: Mapping[str, str]) -> bool:
    artifact = pattern.get("wanted_artifact")
    effect = pattern.get("wanted_effect")
    if artifact and fields.get("carried_artifact") != artifact:
        return False
    if effect and fields.get("downstream_effect") != effect:
        return False
    return True


class InfluenceProjectionAdapter:
    """Wraps ``ProjectionRegistry``. Card status maps onto the retrieval channel.

    Claim signature fields are read from the bound frozen view (the projection
    card stores them only inside its summary). Neighborhood expansion walks
    claim edges of that same view and never writes the registry.
    """

    name = "projection"

    def __init__(
        self,
        registry: ProjectionRegistry,
        views: Sequence[InfluenceViewRevision] | None = None,
    ) -> None:
        self._registry = registry
        self._views: dict[str, InfluenceViewRevision] = {}
        for revision in views or ():
            self.bind_view(revision)

    def bind_view(self, revision: InfluenceViewRevision) -> None:
        if not isinstance(revision, InfluenceViewRevision):
            raise ValueError("revision must be an InfluenceViewRevision")
        self._views[revision.revision_id] = revision

    def search(self, query: RetrievalQuery | Mapping[str, Any] | None, limit: int) -> list[RetrievalCard]:
        parsed = coerce_query(query)
        cap = _limit(limit)
        pattern = parsed.dependency_pattern
        raw_cards = [
            InfluenceProjection.from_dict(entry["card"])
            for entry in self._registry.to_dict()["entries"]
            if isinstance(entry, dict) and isinstance(entry.get("card"), dict)
        ]
        raw_cards.sort(key=lambda card: (card.created_at, card.card_id))
        cards: list[RetrievalCard] = []
        for card in raw_cards:
            fields = _claim_fields(card, self._views)
            if card.kind == "claim" and pattern is not None:
                if fields is None or not _pattern_matches(fields, pattern):
                    continue
            if not _text_hits(parsed, card.summary, card.card_id, card.semantic_type):
                continue
            kind: CardKind = "claim_projection" if card.kind == "claim" else "block_projection"
            pointer: dict[str, Any] = {
                "corpus": self.name,
                "view_revision_id": card.source_pointer.view_revision_id,
                "block_or_claim_id": card.source_pointer.block_or_claim_id,
                "card_id": card.card_id,
                "session_id": card.session_id,
            }
            if fields is not None:
                pointer["carried_artifact"] = fields["carried_artifact"]
                pointer["downstream_effect"] = fields["downstream_effect"]
            cards.append(
                RetrievalCard(
                    kind=kind,
                    channel=card.status,  # type: ignore[arg-type]
                    in_channel_rank=0,
                    status_label=card.status,
                    source_pointer=pointer,
                    payload_ref=card.card_id,
                    score_breakdown=_score(parsed),
                    acl_labels=card.acl_labels,
                )
            )
            if len(cards) >= cap:
                break
        return cards[:cap]

    def expand_claim_neighborhood(
        self,
        view_revision_id: str,
        claim_id: str,
        max_entities: int = PATH_MAX_ENTITIES,
    ) -> AssembledPath:
        """Assemble a bounded path along claim edges of one frozen view.

        The walk stops at ``max_entities`` (never above ``PATH_MAX_ENTITIES``)
        and when the next block is already on the path. Nothing is persisted.
        """
        if isinstance(max_entities, bool) or not isinstance(max_entities, int):
            raise ValueError("max_entities must be an int")
        if max_entities < 1:
            raise ValueError("max_entities must be >= 1")
        if max_entities > PATH_MAX_ENTITIES:
            raise ValueError(
                f"max_entities cannot exceed the global ceiling {PATH_MAX_ENTITIES}"
            )
        revision_id = _require_str(view_revision_id, "view_revision_id")
        seed_id = _require_str(claim_id, "claim_id")
        view = self._views.get(revision_id)
        if view is None:
            raise ValueError(f"unknown view revision {revision_id}")
        claims = [claim for claim in view.claims if isinstance(claim, InfluenceClaim)]
        seed = next((claim for claim in claims if claim.claim_id == seed_id), None)
        if seed is None:
            raise ValueError(f"unknown claim {seed_id} on view {revision_id}")

        entities: list[str] = []
        seen: set[str] = set()
        edges: list[tuple[str, str, str]] = []
        truncated = False

        def _try_add(block_id: str) -> bool:
            nonlocal truncated
            if block_id in seen:
                truncated = True
                return False
            if len(seen) >= max_entities:
                truncated = True
                return False
            seen.add(block_id)
            entities.append(block_id)
            return True

        _try_add(seed.source_block_id)
        added_target = _try_add(seed.target_block_id)
        if seed.source_block_id in seen and seed.target_block_id in seen:
            edges.append((seed.claim_id, seed.source_block_id, seed.target_block_id))
        elif not added_target and seed.target_block_id not in seen:
            truncated = True

        adjacency: dict[str, list[InfluenceClaim]] = {}
        for claim in claims:
            adjacency.setdefault(claim.source_block_id, []).append(claim)
            adjacency.setdefault(claim.target_block_id, []).append(claim)
        for bucket in adjacency.values():
            bucket.sort(key=lambda claim: claim.claim_id)

        visited_claims = {seed.claim_id}
        queue: deque[str] = deque(block_id for block_id in entities)
        while queue and not (len(seen) >= max_entities and truncated):
            node = queue.popleft()
            for claim in adjacency.get(node, ()):
                if claim.claim_id in visited_claims:
                    continue
                other = (
                    claim.target_block_id if claim.source_block_id == node else claim.source_block_id
                )
                if other in seen:
                    visited_claims.add(claim.claim_id)
                    truncated = True
                    continue
                if len(seen) >= max_entities:
                    truncated = True
                    continue
                visited_claims.add(claim.claim_id)
                seen.add(other)
                entities.append(other)
                edges.append((claim.claim_id, claim.source_block_id, claim.target_block_id))
                queue.append(other)

        usages = [
            {
                "kind": "evidence",
                "bridge_id": edge_claim,
                "source_ref": {"type": "block", "id": source},
                "target_ref": {"type": "block", "id": target},
                "relation_type": "claim",
            }
            for edge_claim, source, target in edges
        ]
        violations = check_path_budget(
            entities,
            usages,
            max_entities=max_entities,
            max_hypothesis_hops=PATH_MAX_HYPOTHESIS_HOPS,
        )
        if violations:
            raise ValueError(
                "assembled claim path exceeds the path budget: "
                + ", ".join(item.code for item in violations)
            )
        return AssembledPath(
            view_revision_id=revision_id,
            claim_id=seed_id,
            entities=tuple(entities),
            edges=tuple(edges),
            truncated=truncated,
        )


class SegmentAtomAdapter:
    """Wraps ``list_ao_ledger``. Facts use channel ``fact`` and skip 8/4 slotting.

    ``action.record_kind == "segment"`` selects a segment card; every other AO
    is an atom. Labels come from ``action.acl_labels`` when present, otherwise
    the session id.
    """

    name = "segment_atom"

    def __init__(self, session_dir: str | Path, session_id: str) -> None:
        self._session_dir = Path(session_dir)
        self._session_id = _require_str(session_id, "session_id")

    def search(self, query: RetrievalQuery | Mapping[str, Any] | None, limit: int) -> list[RetrievalCard]:
        parsed = coerce_query(query)
        cap = _limit(limit)
        cards: list[RetrievalCard] = []
        for item in self._items():
            action = item.get("action") if isinstance(item.get("action"), dict) else {}
            observation = item.get("observation") if isinstance(item.get("observation"), dict) else {}
            blob = json.dumps({"action": action, "observation": observation}, ensure_ascii=False)
            if not _text_hits(parsed, blob, str(item.get("ao_id") or "")):
                continue
            record_kind = action.get("record_kind")
            kind: CardKind = "segment" if record_kind == "segment" else "atom"
            raw_labels = action.get("acl_labels")
            if isinstance(raw_labels, list) and raw_labels:
                labels = _normalize_labels(tuple(str(label) for label in raw_labels))
            else:
                labels = (self._session_id,)
            ao_id = str(item.get("ao_id") or "")
            cards.append(
                RetrievalCard(
                    kind=kind,
                    channel="fact",
                    in_channel_rank=0,
                    status_label="fact",
                    source_pointer={
                        "corpus": self.name,
                        "session_id": self._session_id,
                        "ao_id": ao_id,
                        "sequence": item.get("sequence"),
                        "captured_state": item.get("captured_state"),
                    },
                    payload_ref=f"{kind}:{self._session_id}/{ao_id}",
                    score_breakdown=_score(parsed),
                    acl_labels=labels,
                )
            )
            if len(cards) >= cap:
                break
        return cards[:cap]

    def _items(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        cursor: str | None = None
        watermark: str | None = None
        while True:
            page = list_ao_ledger(
                self._session_dir,
                self._session_id,
                cursor=cursor,
                limit=200,
                snapshot_watermark=watermark,
                include_live=True,
            )
            items.extend(item for item in page.items if isinstance(item, dict))
            if not page.next_cursor:
                return items
            cursor = page.next_cursor
            watermark = page.snapshot_watermark


def _assign_ranks(cards: Sequence[RetrievalCard], channel: str) -> list[RetrievalCard]:
    group = [card for card in cards if card.channel == channel]
    return [replace(card, in_channel_rank=index) for index, card in enumerate(group, start=1)]


def _allocate_stateful(cards: Sequence[RetrievalCard]) -> list[RetrievalCard]:
    """Fill 8 verified and 4 provisional slots. A short channel is backfilled.

    Backfilled cards keep the channel and in-channel rank they had before
    slotting. Fact cards never enter this function.
    """
    verified = _assign_ranks(cards, "verified")
    provisional = _assign_ranks(cards, "provisional")
    primary_verified = verified[:VERIFIED_SLOTS]
    primary_provisional = provisional[:PROVISIONAL_SLOTS]
    verified_deficit = VERIFIED_SLOTS - len(primary_verified)
    provisional_deficit = PROVISIONAL_SLOTS - len(primary_provisional)
    backfill_verified_slots = provisional[PROVISIONAL_SLOTS : PROVISIONAL_SLOTS + verified_deficit]
    backfill_provisional_slots = verified[VERIFIED_SLOTS : VERIFIED_SLOTS + provisional_deficit]
    return [
        *primary_verified,
        *backfill_verified_slots,
        *primary_provisional,
        *backfill_provisional_slots,
    ]


class HybridRetrievalFacade:
    """One retrieve call over fork/branch, projection, and segment/atom."""

    def __init__(
        self,
        fork_branch: ForkBranchAdapter,
        projection: InfluenceProjectionAdapter,
        segment_atom: SegmentAtomAdapter,
        *,
        rerank: RerankHook | RerankFn | None = None,
        acl_filter: AclFilter | None = None,
        feature_version: str = FEATURE_SPEC_VERSION,
    ) -> None:
        self.fork_branch = fork_branch
        self.projection = projection
        self.segment_atom = segment_atom
        self.rerank = rerank if rerank is not None else IdentityRerank()
        self.acl_filter = acl_filter if acl_filter is not None else default_acl_filter
        self.feature_version = _require_str(feature_version, "feature_version")

    def retrieve(
        self,
        query: RetrievalQuery | Mapping[str, Any] | str | None,
        principal_labels: Sequence[str],
        profile: RetrievalProfile | Mapping[str, Any] | None = None,
    ) -> RetrievalResult:
        parsed = coerce_query(query)
        budgets = coerce_profile(profile)
        if isinstance(principal_labels, (str, bytes)) or not isinstance(principal_labels, Sequence):
            raise ValueError("principal_labels must be a sequence of str")
        fork_cards = self._visible(
            self.fork_branch.search(parsed, budgets.fork_branch), principal_labels
        )
        projection_cards = self._visible(
            self.projection.search(parsed, budgets.projection), principal_labels
        )
        fact_cards = self._visible(
            self.segment_atom.search(parsed, budgets.segment_atom), principal_labels
        )
        stateful = _allocate_stateful([*fork_cards, *projection_cards])
        facts = _assign_ranks(fact_cards, "fact")
        selected = [*stateful, *facts]
        context: dict[str, Any] = {
            "feature_version": self.feature_version,
            "query": parsed.to_dict(),
            "profile": budgets.to_dict(),
        }
        reranked = self._call_rerank(selected, context)
        return RetrievalResult(cards=tuple(reranked), feature_version=self.feature_version)

    def expand_claim_neighborhood(
        self,
        view_revision_id: str,
        claim_id: str,
        max_entities: int = PATH_MAX_ENTITIES,
    ) -> AssembledPath:
        """Read-time assembly. Delegates to the projection adapter and writes nothing."""
        return self.projection.expand_claim_neighborhood(
            view_revision_id, claim_id, max_entities=max_entities
        )

    def _visible(
        self, cards: Sequence[RetrievalCard], principal_labels: Sequence[str]
    ) -> list[RetrievalCard]:
        return [card for card in cards if self.acl_filter(card, principal_labels)]

    def _call_rerank(
        self, cards: Sequence[RetrievalCard], context: Mapping[str, Any]
    ) -> list[RetrievalCard]:
        hook = self.rerank
        if isinstance(hook, IdentityRerank) or hasattr(hook, "rerank"):
            reranked = hook.rerank(cards, context)  # type: ignore[union-attr]
        else:
            reranked = hook(cards, context)  # type: ignore[operator]
        return [card if isinstance(card, RetrievalCard) else RetrievalCard.from_dict(card) for card in reranked]

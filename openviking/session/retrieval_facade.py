# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Typed hybrid retrieval facade for the causal-memory lane (Q112-A / Q122-A).

Three corpus adapters share one card envelope. Storage and identity stay
separate: fork/branch, block/claim projection, and segment/atom facts are
not collapsed into one corpus. Verified/provisional slotting (8/4, with
backfill) applies only to stateful cards. When the provisional quota is
``n >= 2``, those seats are ``n - 1`` exploitation cards in in-channel rank
order plus one seeded exploration card drawn from the provisional candidates
exploitation did not take. Segment/atom cards stay on the fact channel.
Dependency-pattern hits expand inside one frozen view at read time and do
not write a stored object.

Online retrieval applies principal ACL inside each adapter before the quota
is spent, then clamps the reranked list back to the causal-memory cap.
Offline management reads are not part of this path and online code must not
call them: ``materialize_strength(None)`` (strength with no principal),
``CitationLedger.events()``, ``CausalBridgeStore.list_bridges``, and
``ProjectionRegistry.to_dict``. Those dumps skip principal filtering.
Online projection lookup goes through ``ProjectionRegistry.search``.
"""

from __future__ import annotations

import hashlib
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
    branch_retrieval_channel,
)
from openviking.session.influence_projection import InfluenceProjection, ProjectionRegistry
from openviking.session.influence_view import InfluenceClaim, InfluenceViewRevision

FEATURE_SPEC_VERSION = "ranker-feature-spec-v1"
SELECTION_POLICY_VERSION = "provisional-selection-v1"
NO_ELIGIBLE_EXPLORATION_REASON = "no_eligible_exploration_candidate"

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
SelectionMode = Literal["exploitation", "exploration"]

CARD_KINDS = frozenset(
    ("fork", "branch", "block_projection", "claim_projection", "segment", "atom")
)
CARD_CHANNELS = frozenset(("verified", "provisional", "fact"))
SELECTION_MODES = frozenset(("exploitation", "exploration"))
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
    selection_mode: SelectionMode = "exploitation"

    def __post_init__(self) -> None:
        if self.kind not in CARD_KINDS:
            raise ValueError(f"kind must be one of {sorted(CARD_KINDS)}, got {self.kind!r}")
        if self.channel not in CARD_CHANNELS:
            raise ValueError(
                f"channel must be one of {sorted(CARD_CHANNELS)}, got {self.channel!r}"
            )
        if self.selection_mode not in SELECTION_MODES:
            raise ValueError(
                "selection_mode must be one of "
                f"{sorted(SELECTION_MODES)}, got {self.selection_mode!r}"
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
            "selection_mode": self.selection_mode,
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
            selection_mode=payload.get("selection_mode", "exploitation"),
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
    """Per-corpus budgets inside the causal-memory lane. Their sum cannot exceed 12.

    ``provisional_slots`` is the provisional channel quota (default 4). It is
    not a fourth corpus budget and is not added to that sum. For ``n >= 2``
    the quota is filled as ``n - 1`` exploitation seats plus one exploration
    seat.
    """

    fork_branch: int = DEFAULT_FORK_BRANCH_QUOTA
    projection: int = DEFAULT_PROJECTION_QUOTA
    segment_atom: int = DEFAULT_SEGMENT_ATOM_QUOTA
    provisional_slots: int = PROVISIONAL_SLOTS

    def to_dict(self) -> dict[str, int]:
        return {
            "fork_branch": self.fork_branch,
            "projection": self.projection,
            "segment_atom": self.segment_atom,
            "provisional_slots": self.provisional_slots,
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
            provisional_slots=_quota_int(
                payload.get("provisional_slots", PROVISIONAL_SLOTS), "provisional_slots"
            ),
        )
    fork_branch = _quota_int(profile.fork_branch, "fork_branch")
    projection = _quota_int(profile.projection, "projection")
    segment_atom = _quota_int(profile.segment_atom, "segment_atom")
    provisional_slots = _quota_int(profile.provisional_slots, "provisional_slots")
    total = fork_branch + projection + segment_atom
    if total > CAUSAL_MEMORY_CAP:
        raise QuotaError(
            f"retrieval profile sums to {total}, above the causal-memory cap {CAUSAL_MEMORY_CAP}",
            total=total,
        )
    return RetrievalProfile(
        fork_branch=fork_branch,
        projection=projection,
        segment_atom=segment_atom,
        provisional_slots=provisional_slots,
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
class SelectionAudit:
    """Logged draw for the provisional exploration seat (Q87-A)."""

    eligible_set: tuple[str, ...]
    selection_propensity: float | None
    seed: str
    selection_policy_version: str
    backfill: bool
    backfill_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "eligible_set": list(self.eligible_set),
            "selection_propensity": self.selection_propensity,
            "seed": self.seed,
            "selection_policy_version": self.selection_policy_version,
            "backfill": self.backfill,
            "backfill_reason": self.backfill_reason,
        }


@dataclass(frozen=True)
class RetrievalResult:
    cards: tuple[RetrievalCard, ...]
    feature_version: str
    truncated_by_rerank_clamp: bool = False
    selection_audit: SelectionAudit | None = None

    def to_dict(self) -> dict[str, Any]:
        audit = None if self.selection_audit is None else self.selection_audit.to_dict()
        return {
            "cards": [card.to_dict() for card in self.cards],
            "feature_version": self.feature_version,
            "truncated_by_rerank_clamp": self.truncated_by_rerank_clamp,
            "selection_audit": audit,
        }


class CorpusAdapter(Protocol):
    """One corpus search. Results keep that corpus's kind."""

    name: str

    def search(
        self,
        query: RetrievalQuery,
        limit: int,
        principal_labels: Sequence[str],
    ) -> list[RetrievalCard]:
        """Return at most ``limit`` cards visible to ``principal_labels``."""


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
    """Case-insensitive substring match (``str.casefold`` on both sides).

    An empty query text matches every haystack.
    """
    if not query.text:
        return True
    haystack = "\n".join(parts).casefold()
    return query.text.casefold() in haystack


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


def _principal_set(principal_labels: Sequence[str]) -> set[str]:
    if isinstance(principal_labels, (str, bytes)) or not isinstance(principal_labels, Sequence):
        raise ValueError("principal_labels must be a sequence of str")
    return set(principal_labels)


def _labels_visible(labels: Sequence[str], principal: set[str]) -> bool:
    return bool(principal.intersection(labels))


class ForkBranchAdapter:
    """Wraps ``CausalExperiencesStore`` reads. Latest revision only.

    Validated revisions enter the verified channel. Provisional revisions are
    consumed as ForkCandidate and enter the provisional channel. Draft and
    invalidated revisions stay out of retrieval. A branch card uses
    ``branch_retrieval_channel`` on its own evidence status; it does not
    inherit the parent revision channel. ACL labels are the fork workspace id:
    the store has no separate label column. Invisible cards are dropped before
    the quota is spent.

    Online code must not call ``materialize_strength(None)``, ``events()``,
    ``list_bridges``, or ``ProjectionRegistry.to_dict``. Those are offline
    management dumps and do not apply principal ACL.
    """

    name = "fork_branch"

    def __init__(self, root: str | Path, store: CausalExperiencesStore | None = None) -> None:
        self._root = Path(root)
        self._store = store if store is not None else CausalExperiencesStore(self._root)

    def search(
        self,
        query: RetrievalQuery | Mapping[str, Any] | None,
        limit: int,
        principal_labels: Sequence[str] = (),
    ) -> list[RetrievalCard]:
        """ACL-filter fork and branch cards, then keep at most ``limit`` visible ones."""
        parsed = coerce_query(query)
        cap = _limit(limit)
        principal = _principal_set(principal_labels)
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
            raw_workspace = latest.get("workspace_id")
            if not isinstance(raw_workspace, str) or not raw_workspace:
                continue
            labels = _normalize_labels((raw_workspace,))
            if not _labels_visible(labels, principal):
                continue
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
                branch_channel = branch_retrieval_channel(branch, status)
                if branch_channel not in CARD_CHANNELS or branch_channel == "fact":
                    continue
                cards.append(
                    RetrievalCard(
                        kind="branch",
                        channel=branch_channel,  # type: ignore[arg-type]
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


def _remember_ao_id(ao_id: str, ordered: list[str], seen: set[str]) -> None:
    if ao_id and ao_id not in seen:
        seen.add(ao_id)
        ordered.append(ao_id)


def _projection_source_ao_ids(
    view: InfluenceViewRevision, kind: str, entity_id: str
) -> list[str]:
    """AO ids behind one block or claim. Used only to resolve match text."""
    if kind == "block":
        for block in view.blocks:
            if block.block_id == entity_id:
                return [ref.ao_id for ref in block.source_refs if ref.ao_id]
        return []
    if kind != "claim":
        return []
    for claim in view.claims:
        if claim.claim_id != entity_id:
            continue
        ordered: list[str] = []
        seen: set[str] = set()
        for ref in (*claim.source_evidence_refs, *claim.target_evidence_refs):
            _remember_ao_id(ref.ao_id, ordered, seen)
        linked = {claim.source_block_id, claim.target_block_id}
        for block in view.blocks:
            if block.block_id in linked:
                for ref in block.source_refs:
                    _remember_ao_id(ref.ao_id, ordered, seen)
        return ordered
    return []


class InfluenceProjectionAdapter:
    """Wraps ``ProjectionRegistry``. Card status maps onto the retrieval channel.

    Claim signature fields are read from the bound frozen view (the projection
    card stores them only inside its summary). Neighborhood expansion walks
    claim edges of that same view and never writes the registry. Online search
    calls ``ProjectionRegistry.search`` with ``principal_labels`` and never
    ``to_dict``. ``materialize_strength(None)``, ``events()``, and
    ``list_bridges`` are offline management paths; online code must not call
    them.

    ``content_resolver`` maps an AO id to that AO's body text. Resolved text
    is concatenated into the hit test only. It is not written onto the
    returned card or its source pointer. A missing resolver, an unbound view,
    or an empty resolution leaves matching on the summary, card id, and
    semantic type.
    """

    name = "projection"

    def __init__(
        self,
        registry: ProjectionRegistry,
        views: Sequence[InfluenceViewRevision] | None = None,
        content_resolver: Callable[[str], str] | None = None,
    ) -> None:
        self._registry = registry
        self._views: dict[str, InfluenceViewRevision] = {}
        self._content_resolver = content_resolver
        for revision in views or ():
            self.bind_view(revision)

    def bind_view(self, revision: InfluenceViewRevision) -> None:
        if not isinstance(revision, InfluenceViewRevision):
            raise ValueError("revision must be an InfluenceViewRevision")
        self._views[revision.revision_id] = revision

    def search(
        self,
        query: RetrievalQuery | Mapping[str, Any] | None,
        limit: int,
        principal_labels: Sequence[str] = (),
    ) -> list[RetrievalCard]:
        """ACL via ``ProjectionRegistry.search``, then keep at most ``limit`` hits.

        Does not call ``ProjectionRegistry.to_dict``. That dump, along with
        ``materialize_strength(None)``, ``events()``, and ``list_bridges``, is
        an offline management path and must not be used online.
        """
        parsed = coerce_query(query)
        cap = _limit(limit)
        pattern = parsed.dependency_pattern
        raw_cards = list(self._registry.search(principal_labels=principal_labels))
        cards: list[RetrievalCard] = []
        for card in raw_cards:
            fields = _claim_fields(card, self._views)
            if card.kind == "claim" and pattern is not None:
                if fields is None or not _pattern_matches(fields, pattern):
                    continue
            if not _text_hits(
                parsed,
                card.summary,
                card.card_id,
                card.semantic_type,
                *self._resolved_match_parts(card),
            ):
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

    def _resolved_match_parts(self, card: InfluenceProjection) -> tuple[str, ...]:
        """Body snippets for the hit test. Never copied onto the card."""
        resolver = self._content_resolver
        if resolver is None:
            return ()
        view = self._views.get(card.source_pointer.view_revision_id)
        if view is None:
            return ()
        parts: list[str] = []
        for ao_id in _projection_source_ao_ids(
            view, card.kind, card.source_pointer.block_or_claim_id
        ):
            text = resolver(ao_id)
            if isinstance(text, str) and text:
                parts.append(text)
        return tuple(parts)

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
    is an atom. Labels come only from ``action.acl_labels``. An action with no
    labels is rejected and stays invisible: the session id is not used as a
    stand-in label, and the row does not consume quota.
    """

    name = "segment_atom"

    def __init__(self, session_dir: str | Path, session_id: str) -> None:
        self._session_dir = Path(session_dir)
        self._session_id = _require_str(session_id, "session_id")

    def search(
        self,
        query: RetrievalQuery | Mapping[str, Any] | None,
        limit: int,
        principal_labels: Sequence[str] = (),
    ) -> list[RetrievalCard]:
        """Drop unlabeled or non-intersecting facts, then keep at most ``limit``."""
        parsed = coerce_query(query)
        cap = _limit(limit)
        principal = _principal_set(principal_labels)
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
            if not isinstance(raw_labels, list) or not raw_labels:
                continue
            labels = _normalize_labels(tuple(str(label) for label in raw_labels))
            if not _labels_visible(labels, principal):
                continue
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


def _candidate_id(card: RetrievalCard) -> str:
    return card.payload_ref


def _derive_query_seed(query: RetrievalQuery) -> str:
    """Seed from the query envelope: sha256 digest, first 8 bytes, hex."""
    payload = json.dumps(
        query.to_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).digest()[:8].hex()


def _normalize_seed(seed: str | int | None) -> str | None:
    if seed is None:
        return None
    if isinstance(seed, bool) or not isinstance(seed, (str, int)):
        raise ValueError("seed must be a str, int, or None")
    text = str(seed)
    if not text:
        raise ValueError("seed must be non-empty when provided")
    return text


def _exploration_index(seed: str, eligible_count: int) -> int:
    """Uniform index in ``[0, eligible_count)`` from a versioned seed digest.

    Replay: ``eligible_set[_exploration_index(seed, len(eligible_set))]``.
    """
    material = f"{SELECTION_POLICY_VERSION}\n{seed}\n{eligible_count}".encode("utf-8")
    draw = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    return draw % eligible_count


def _select_provisional(
    ranked: Sequence[RetrievalCard],
    slots: int,
    seed: str,
) -> tuple[list[RetrievalCard], list[RetrievalCard], SelectionAudit]:
    """Fill provisional seats: ``n - 1`` exploitation plus one seeded exploration.

    ``ranked`` is in-channel rank order. Exploitation keeps that prefix.
    Exploration is drawn uniformly from provisional cards exploitation did
    not take (already ACL- and status-gated by the adapters). When that
    eligible set is empty, the exploration seat stays open for the existing
    exploitation backfill and the audit records ``backfill``. Unchosen cards
    stay in rank order so a verified-slot shortage can still absorb them.
    """
    if slots < 2:
        primary = [replace(card, selection_mode="exploitation") for card in ranked[:slots]]
        overflow = [replace(card, selection_mode="exploitation") for card in ranked[slots:]]
        audit = SelectionAudit(
            eligible_set=(),
            selection_propensity=None,
            seed=seed,
            selection_policy_version=SELECTION_POLICY_VERSION,
            backfill=False,
        )
        return primary, overflow, audit

    exploit_count = slots - 1
    exploitation = list(ranked[:exploit_count])
    eligible = list(ranked[exploit_count:])
    if not eligible:
        primary = [replace(card, selection_mode="exploitation") for card in exploitation]
        audit = SelectionAudit(
            eligible_set=(),
            selection_propensity=None,
            seed=seed,
            selection_policy_version=SELECTION_POLICY_VERSION,
            backfill=True,
            backfill_reason=NO_ELIGIBLE_EXPLORATION_REASON,
        )
        return primary, [], audit

    chosen = eligible[_exploration_index(seed, len(eligible))]
    overflow = [
        replace(card, selection_mode="exploitation") for card in eligible if card is not chosen
    ]
    primary = [
        *[replace(card, selection_mode="exploitation") for card in exploitation],
        replace(chosen, selection_mode="exploration"),
    ]
    audit = SelectionAudit(
        eligible_set=tuple(_candidate_id(card) for card in eligible),
        selection_propensity=1.0 / len(eligible),
        seed=seed,
        selection_policy_version=SELECTION_POLICY_VERSION,
        backfill=False,
    )
    return primary, overflow, audit


def _allocate_stateful(
    cards: Sequence[RetrievalCard],
    *,
    provisional_slots: int,
    seed: str,
) -> tuple[list[RetrievalCard], SelectionAudit]:
    """Fill 8 verified slots and ``provisional_slots`` provisional slots.

    A short channel is backfilled from the other channel's overflow.
    Backfilled cards keep the channel and in-channel rank they had before
    slotting. Fact cards never enter this function. Verified cards stay
    ``selection_mode="exploitation"``.
    """
    verified = [
        replace(card, selection_mode="exploitation") for card in _assign_ranks(cards, "verified")
    ]
    provisional = _assign_ranks(cards, "provisional")
    primary_verified = verified[:VERIFIED_SLOTS]
    primary_provisional, provisional_overflow, audit = _select_provisional(
        provisional, provisional_slots, seed
    )
    verified_deficit = VERIFIED_SLOTS - len(primary_verified)
    provisional_deficit = provisional_slots - len(primary_provisional)
    backfill_verified_slots = provisional_overflow[:verified_deficit]
    backfill_provisional_slots = verified[VERIFIED_SLOTS : VERIFIED_SLOTS + provisional_deficit]
    return [
        *primary_verified,
        *backfill_verified_slots,
        *primary_provisional,
        *backfill_provisional_slots,
    ], audit


class HybridRetrievalFacade:
    """One retrieve call over fork/branch, projection, and segment/atom.

    Online code must not call ``materialize_strength(None)``, ``events()``,
    ``list_bridges``, or ``ProjectionRegistry.to_dict``. Those are offline
    management paths and do not apply principal ACL.
    """

    def __init__(
        self,
        fork_branch: ForkBranchAdapter,
        projection: InfluenceProjectionAdapter,
        segment_atom: SegmentAtomAdapter,
        *,
        rerank: RerankHook | RerankFn | None = None,
        acl_filter: AclFilter | None = None,
        feature_version: str = FEATURE_SPEC_VERSION,
        seed: str | int | None = None,
    ) -> None:
        self.fork_branch = fork_branch
        self.projection = projection
        self.segment_atom = segment_atom
        self.rerank = rerank if rerank is not None else IdentityRerank()
        self.acl_filter = acl_filter if acl_filter is not None else default_acl_filter
        self.feature_version = _require_str(feature_version, "feature_version")
        self.seed = _normalize_seed(seed)

    def retrieve(
        self,
        query: RetrievalQuery | Mapping[str, Any] | str | None,
        principal_labels: Sequence[str],
        profile: RetrievalProfile | Mapping[str, Any] | None = None,
    ) -> RetrievalResult:
        """Filter inside each adapter, then clamp reranked cards back to 12.

        ``materialize_strength(None)``, ``events()``, ``list_bridges``, and
        ``ProjectionRegistry.to_dict`` are offline management paths. This
        method does not call them.
        """
        parsed = coerce_query(query)
        budgets = coerce_profile(profile)
        _principal_set(principal_labels)
        seed = self.seed if self.seed is not None else _derive_query_seed(parsed)
        fork_cards = self._visible(
            self.fork_branch.search(parsed, budgets.fork_branch, principal_labels),
            principal_labels,
        )
        projection_cards = self._visible(
            self.projection.search(parsed, budgets.projection, principal_labels),
            principal_labels,
        )
        fact_cards = self._visible(
            self.segment_atom.search(parsed, budgets.segment_atom, principal_labels),
            principal_labels,
        )
        stateful, audit = _allocate_stateful(
            [*fork_cards, *projection_cards],
            provisional_slots=budgets.provisional_slots,
            seed=seed,
        )
        facts = [
            replace(card, selection_mode="exploitation")
            for card in _assign_ranks(fact_cards, "fact")
        ]
        selected = [*stateful, *facts]
        context: dict[str, Any] = {
            "feature_version": self.feature_version,
            "query": parsed.to_dict(),
            "profile": budgets.to_dict(),
        }
        reranked = self._call_rerank(selected, context)
        truncated = len(reranked) > CAUSAL_MEMORY_CAP
        clamped = reranked[:CAUSAL_MEMORY_CAP]
        return RetrievalResult(
            cards=tuple(clamped),
            feature_version=self.feature_version,
            truncated_by_rerank_clamp=truncated,
            selection_audit=audit,
        )

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

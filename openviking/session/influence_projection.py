# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Lightweight Influence projection cards and their retrieval registry.

Cards carry a semantic summary, an embedding stub, and a source pointer.
They do not store the view body or raw AO text. Verified promotion follows
validated ForkRevision grounding (Q113-A). An ACL change withdraws every
card from that session and keeps an audit record (Q114-A).
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from openviking.session.influence_view import InfluenceViewRevision

REGISTRY_VERSION = 1

CARD_KIND_BLOCK = "block"
CARD_KIND_CLAIM = "claim"
CARD_KINDS = frozenset((CARD_KIND_BLOCK, CARD_KIND_CLAIM))

CARD_STATUS_PROVISIONAL = "provisional"
CARD_STATUS_VERIFIED = "verified"
CARD_STATUSES = frozenset((CARD_STATUS_PROVISIONAL, CARD_STATUS_VERIFIED))

REMOVAL_ACL_CHANGE = "acl_change"
REMOVAL_SUPERSEDED = "superseded"
REMOVAL_REASONS = frozenset((REMOVAL_ACL_CHANGE, REMOVAL_SUPERSEDED))

CardKind = Literal["block", "claim"]
CardStatus = Literal["provisional", "verified"]
RemovalReason = Literal["acl_change", "superseded"]
GroundingQuery = Callable[[str, str], bool]
SessionAcl = Callable[[str], Sequence[str]]
Clock = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _format_dt(value: datetime) -> str:
    return _as_utc(value).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _parse_iso(value: str, field: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{field} must be ISO-8601, got {value!r}") from exc
    return _as_utc(parsed)


def _require_mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a dict")
    return value


def _require_str(value: Any, field: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a str")
    if not allow_empty and not value.strip():
        raise ValueError(f"{field} must be a non-empty str")
    return value


def _require_choice(value: Any, field: str, choices: frozenset[str]) -> str:
    text = _require_str(value, field)
    if text not in choices:
        raise ValueError(f"{field} must be one of {sorted(choices)}, got {text!r}")
    return text


def _canonical_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _require_timestamp(value: Any, field: str) -> str:
    return _format_dt(_parse_iso(_require_str(value, field), field))


def _normalize_labels(value: Sequence[str] | None, field: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{field} must be a sequence of str")
    labels: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = _require_str(item, field)
        if text in seen:
            continue
        seen.add(text)
        labels.append(text)
    return tuple(labels)


def _copy_json_dict(value: Any, field: str, *, allow_empty: bool) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a dict")
    try:
        cloned = json.loads(_canonical_dumps(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be JSON-serializable") from exc
    if not isinstance(cloned, dict):
        raise ValueError(f"{field} must be a dict")
    if not allow_empty and not cloned:
        raise ValueError(f"{field} must be a non-empty object")
    return cloned


def _embedding_stub(content_hash: str) -> dict[str, Any]:
    return {"provider": "stub", "dimensions": 0, "digest": content_hash}


def _card_content_hash(
    *,
    kind: str,
    summary: str,
    semantic_type: str,
    view_revision_id: str,
    block_or_claim_id: str,
    session_id: str,
) -> str:
    payload = {
        "kind": kind,
        "summary": summary,
        "semantic_type": semantic_type,
        "view_revision_id": view_revision_id,
        "block_or_claim_id": block_or_claim_id,
        "session_id": session_id,
    }
    return _sha256_text(_canonical_dumps(payload))


@dataclass(frozen=True)
class SourcePointer:
    """Pointer back to one frozen block or claim. Not a copy of the view."""

    view_revision_id: str
    block_or_claim_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "view_revision_id", _require_str(self.view_revision_id, "view_revision_id")
        )
        object.__setattr__(
            self, "block_or_claim_id", _require_str(self.block_or_claim_id, "block_or_claim_id")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "view_revision_id": self.view_revision_id,
            "block_or_claim_id": self.block_or_claim_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SourcePointer:
        payload = _require_mapping(data, "source_pointer")
        return cls(
            view_revision_id=str(payload.get("view_revision_id") or ""),
            block_or_claim_id=str(payload.get("block_or_claim_id") or ""),
        )


@dataclass(frozen=True)
class InfluenceProjection:
    """Retrieval card for one frozen block or claim."""

    card_id: str
    kind: CardKind
    summary: str
    semantic_type: str
    status: CardStatus
    source_pointer: SourcePointer
    embedding_stub: dict[str, Any]
    acl_labels: tuple[str, ...]
    content_hash: str
    created_at: str
    session_id: str
    fork_provenance: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "card_id", _require_str(self.card_id, "card_id"))
        object.__setattr__(self, "kind", _require_choice(self.kind, "kind", CARD_KINDS))
        object.__setattr__(self, "summary", _require_str(self.summary, "summary", allow_empty=True))
        object.__setattr__(
            self, "semantic_type", _require_str(self.semantic_type, "semantic_type")
        )
        object.__setattr__(self, "status", _require_choice(self.status, "status", CARD_STATUSES))
        if not isinstance(self.source_pointer, SourcePointer):
            raise ValueError("source_pointer must be a SourcePointer")
        object.__setattr__(
            self, "embedding_stub", _copy_json_dict(self.embedding_stub, "embedding_stub", allow_empty=True)
        )
        object.__setattr__(self, "acl_labels", _normalize_labels(self.acl_labels, "acl_labels"))
        object.__setattr__(self, "content_hash", _require_str(self.content_hash, "content_hash"))
        object.__setattr__(self, "created_at", _require_timestamp(self.created_at, "created_at"))
        object.__setattr__(self, "session_id", _require_str(self.session_id, "session_id"))
        provenance = self.fork_provenance
        if provenance is None:
            cloned: dict[str, Any] | None = None
        else:
            cloned = _copy_json_dict(provenance, "fork_provenance", allow_empty=False)
        object.__setattr__(self, "fork_provenance", cloned)
        if self.status == CARD_STATUS_VERIFIED and not self.fork_provenance:
            raise ValueError("verified projection requires fork_provenance")
        if self.status == CARD_STATUS_PROVISIONAL and self.fork_provenance is not None:
            raise ValueError("provisional projection must not carry fork_provenance")

    def to_dict(self) -> dict[str, Any]:
        return {
            "card_id": self.card_id,
            "kind": self.kind,
            "summary": self.summary,
            "semantic_type": self.semantic_type,
            "status": self.status,
            "source_pointer": self.source_pointer.to_dict(),
            "embedding_stub": self.embedding_stub,
            "acl_labels": list(self.acl_labels),
            "content_hash": self.content_hash,
            "created_at": self.created_at,
            "session_id": self.session_id,
            "fork_provenance": self.fork_provenance,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InfluenceProjection:
        payload = _require_mapping(data, "projection")
        provenance = payload.get("fork_provenance")
        labels = payload.get("acl_labels")
        if not isinstance(labels, list):
            raise ValueError("acl_labels must be a list")
        return cls(
            card_id=str(payload.get("card_id") or ""),
            kind=payload.get("kind"),  # type: ignore[arg-type]
            summary=payload.get("summary") if isinstance(payload.get("summary"), str) else "",
            semantic_type=str(payload.get("semantic_type") or ""),
            status=payload.get("status"),  # type: ignore[arg-type]
            source_pointer=SourcePointer.from_dict(
                _require_mapping(payload.get("source_pointer"), "source_pointer")
            ),
            embedding_stub=_require_mapping(payload.get("embedding_stub"), "embedding_stub"),
            acl_labels=tuple(labels),
            content_hash=str(payload.get("content_hash") or ""),
            created_at=str(payload.get("created_at") or ""),
            session_id=str(payload.get("session_id") or ""),
            fork_provenance=None if provenance is None else provenance,
        )


@dataclass(frozen=True)
class RemovalRecord:
    """Audit row for a card taken out of the search index."""

    record_id: str
    card: InfluenceProjection
    reason: RemovalReason
    session_id: str
    new_labels: tuple[str, ...]
    removed_at: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "record_id", _require_str(self.record_id, "record_id"))
        if not isinstance(self.card, InfluenceProjection):
            raise ValueError("card must be an InfluenceProjection")
        object.__setattr__(self, "reason", _require_choice(self.reason, "reason", REMOVAL_REASONS))
        object.__setattr__(self, "session_id", _require_str(self.session_id, "session_id"))
        object.__setattr__(self, "new_labels", _normalize_labels(self.new_labels, "new_labels"))
        object.__setattr__(self, "removed_at", _require_timestamp(self.removed_at, "removed_at"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "card": self.card.to_dict(),
            "reason": self.reason,
            "session_id": self.session_id,
            "new_labels": list(self.new_labels),
            "removed_at": self.removed_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RemovalRecord:
        payload = _require_mapping(data, "removal")
        labels = payload.get("new_labels")
        if not isinstance(labels, list):
            raise ValueError("new_labels must be a list")
        return cls(
            record_id=str(payload.get("record_id") or ""),
            card=InfluenceProjection.from_dict(_require_mapping(payload.get("card"), "projection")),
            reason=payload.get("reason"),  # type: ignore[arg-type]
            session_id=str(payload.get("session_id") or ""),
            new_labels=tuple(labels),
            removed_at=str(payload.get("removed_at") or ""),
        )


@dataclass(frozen=True)
class _ActiveEntry:
    card: InfluenceProjection
    purpose: str

    def __post_init__(self) -> None:
        if not isinstance(self.card, InfluenceProjection):
            raise ValueError("card must be an InfluenceProjection")
        object.__setattr__(self, "purpose", _require_str(self.purpose, "purpose"))

    def to_dict(self) -> dict[str, Any]:
        return {"card": self.card.to_dict(), "purpose": self.purpose}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> _ActiveEntry:
        payload = _require_mapping(data, "entry")
        return cls(
            card=InfluenceProjection.from_dict(_require_mapping(payload.get("card"), "projection")),
            purpose=str(payload.get("purpose") or ""),
        )


class ProjectionRegistry:
    """Search index of projection cards plus an append-only removal audit."""

    def __init__(
        self,
        path: str | Path,
        *,
        grounding_query: GroundingQuery | None = None,
        session_acl: SessionAcl | None = None,
        clock: Clock | None = None,
    ) -> None:
        self.path = Path(path)
        self._grounding_query = grounding_query
        self._session_acl = session_acl
        self._clock = clock or _utc_now
        self._lock = threading.RLock()
        self._entries: dict[str, _ActiveEntry] = {}
        self._removed: list[RemovalRecord] = []
        self._validated: set[tuple[str, str]] = set()
        self._removal_seq = 0
        self._load()

    def register(
        self,
        revision: InfluenceViewRevision,
        *,
        acl_labels: Sequence[str] | None = None,
        validated_positions: Sequence[str] | None = None,
    ) -> tuple[InfluenceProjection, ...]:
        """Project every block and claim on a frozen revision.

        A later revision for the same session and purpose supersedes older
        cards: they leave the search index and stay in the audit log.
        """
        if not isinstance(revision, InfluenceViewRevision):
            raise ValueError("revision must be a frozen InfluenceViewRevision")
        with self._lock:
            self._remember_validated(revision.revision_id, validated_positions)
            existing = self._cards_for_revision_locked(revision.revision_id)
            if existing:
                self._save_locked()
                return existing
            labels = self._labels_for(revision.session_id, acl_labels)
            self._supersede_locked(revision, labels)
            created_at = _format_dt(_as_utc(self._clock()))
            cards = _cards_from_revision(revision, acl_labels=labels, created_at=created_at)
            for card in cards:
                self._entries[card.card_id] = _ActiveEntry(card=card, purpose=revision.purpose)
            self._save_locked()
            return cards

    def search(
        self,
        *,
        principal_labels: Sequence[str],
        kind: CardKind | None = None,
        status: CardStatus | None = None,
    ) -> tuple[InfluenceProjection, ...]:
        """Return active cards whose ACL labels intersect ``principal_labels``."""
        principal = set(_normalize_labels(principal_labels, "principal_labels"))
        if kind is not None:
            _require_choice(kind, "kind", CARD_KINDS)
        if status is not None:
            _require_choice(status, "status", CARD_STATUSES)
        with self._lock:
            matched = [
                entry.card
                for entry in self._entries.values()
                if principal.intersection(entry.card.acl_labels)
                and (kind is None or entry.card.kind == kind)
                and (status is None or entry.card.status == status)
            ]
        matched.sort(key=lambda card: (card.created_at, card.card_id))
        return tuple(matched)

    def promote_to_verified(
        self,
        view_revision_id: str,
        block_or_claim_id: str,
        fork_provenance: Mapping[str, Any],
    ) -> InfluenceProjection:
        """Move one card to the verified channel when grounding is validated."""
        revision_id = _require_str(view_revision_id, "view_revision_id")
        position_id = _require_str(block_or_claim_id, "block_or_claim_id")
        provenance = _copy_json_dict(dict(fork_provenance), "fork_provenance", allow_empty=False)
        with self._lock:
            if not self._is_validated_locked(revision_id, position_id):
                raise ValueError(
                    "cannot promote without validated ForkRevision grounding "
                    f"for {revision_id}:{position_id}"
                )
            entry = self._find_locked(revision_id, position_id)
            if entry is None:
                raise ValueError(
                    f"no active projection for {revision_id}:{position_id}"
                )
            promoted = InfluenceProjection(
                card_id=entry.card.card_id,
                kind=entry.card.kind,
                summary=entry.card.summary,
                semantic_type=entry.card.semantic_type,
                status=CARD_STATUS_VERIFIED,
                source_pointer=entry.card.source_pointer,
                embedding_stub=entry.card.embedding_stub,
                acl_labels=entry.card.acl_labels,
                content_hash=entry.card.content_hash,
                created_at=entry.card.created_at,
                session_id=entry.card.session_id,
                fork_provenance=provenance,
            )
            self._entries[promoted.card_id] = _ActiveEntry(card=promoted, purpose=entry.purpose)
            self._save_locked()
            return promoted

    def handle_acl_change(
        self,
        session_id: str,
        new_labels: Sequence[str],
    ) -> tuple[RemovalRecord, ...]:
        """Withdraw every active card from ``session_id`` and keep an audit row."""
        session = _require_str(session_id, "session_id")
        labels = _normalize_labels(new_labels, "new_labels")
        with self._lock:
            doomed = [
                entry
                for entry in self._entries.values()
                if entry.card.session_id == session
            ]
            doomed.sort(key=lambda entry: entry.card.card_id)
            removed = tuple(
                self._withdraw_locked(entry, reason=REMOVAL_ACL_CHANGE, new_labels=labels)
                for entry in doomed
            )
            if removed:
                self._save_locked()
            return removed

    def removed_records(self) -> tuple[RemovalRecord, ...]:
        with self._lock:
            return tuple(self._removed)

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            entries = [
                self._entries[card_id].to_dict()
                for card_id in sorted(self._entries)
            ]
            positions = [
                {"view_revision_id": revision_id, "block_or_claim_id": position_id}
                for revision_id, position_id in sorted(self._validated)
            ]
            return {
                "version": REGISTRY_VERSION,
                "entries": entries,
                "removed": [record.to_dict() for record in self._removed],
                "validated_positions": positions,
                "removal_seq": self._removal_seq,
            }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        path: str | Path,
        grounding_query: GroundingQuery | None = None,
        session_acl: SessionAcl | None = None,
        clock: Clock | None = None,
    ) -> ProjectionRegistry:
        registry = cls(
            path,
            grounding_query=grounding_query,
            session_acl=session_acl,
            clock=clock,
        )
        registry._apply(data, persist=True)
        return registry

    def _labels_for(self, session_id: str, acl_labels: Sequence[str] | None) -> tuple[str, ...]:
        if acl_labels is not None:
            return _normalize_labels(acl_labels, "acl_labels")
        if self._session_acl is None:
            return ()
        return _normalize_labels(self._session_acl(session_id), "acl_labels")

    def _remember_validated(
        self,
        view_revision_id: str,
        validated_positions: Sequence[str] | None,
    ) -> None:
        if not validated_positions:
            return
        if isinstance(validated_positions, (str, bytes)):
            raise ValueError("validated_positions must be a sequence of str")
        for position_id in validated_positions:
            self._validated.add((view_revision_id, _require_str(position_id, "validated_positions")))

    def _cards_for_revision_locked(self, view_revision_id: str) -> tuple[InfluenceProjection, ...]:
        cards = [
            entry.card
            for entry in self._entries.values()
            if entry.card.source_pointer.view_revision_id == view_revision_id
        ]
        cards.sort(key=lambda card: card.card_id)
        return tuple(cards)

    def _supersede_locked(
        self,
        revision: InfluenceViewRevision,
        new_labels: tuple[str, ...],
    ) -> None:
        doomed = [
            entry
            for entry in self._entries.values()
            if entry.card.session_id == revision.session_id
            and entry.purpose == revision.purpose
            and entry.card.source_pointer.view_revision_id != revision.revision_id
        ]
        doomed.sort(key=lambda entry: entry.card.card_id)
        for entry in doomed:
            self._withdraw_locked(entry, reason=REMOVAL_SUPERSEDED, new_labels=new_labels)

    def _withdraw_locked(
        self,
        entry: _ActiveEntry,
        *,
        reason: RemovalReason,
        new_labels: tuple[str, ...],
    ) -> RemovalRecord:
        self._entries.pop(entry.card.card_id, None)
        self._removal_seq += 1
        record = RemovalRecord(
            record_id=f"removal-{self._removal_seq}",
            card=entry.card,
            reason=reason,
            session_id=entry.card.session_id,
            new_labels=new_labels,
            removed_at=_format_dt(_as_utc(self._clock())),
        )
        self._removed.append(record)
        return record

    def _is_validated_locked(self, view_revision_id: str, block_or_claim_id: str) -> bool:
        if (view_revision_id, block_or_claim_id) in self._validated:
            return True
        if self._grounding_query is None:
            return False
        return bool(self._grounding_query(view_revision_id, block_or_claim_id))

    def _find_locked(self, view_revision_id: str, block_or_claim_id: str) -> _ActiveEntry | None:
        for entry in self._entries.values():
            pointer = entry.card.source_pointer
            if pointer.view_revision_id == view_revision_id and pointer.block_or_claim_id == block_or_claim_id:
                return entry
        return None

    def _apply(self, data: dict[str, Any], *, persist: bool) -> None:
        payload = _require_mapping(data, "registry")
        if payload.get("version") != REGISTRY_VERSION:
            raise ValueError(f"unsupported projection registry version {payload.get('version')!r}")
        entries_raw = payload.get("entries")
        removed_raw = payload.get("removed")
        positions_raw = payload.get("validated_positions")
        if not isinstance(entries_raw, list):
            raise ValueError("entries must be a list")
        if not isinstance(removed_raw, list):
            raise ValueError("removed must be a list")
        if not isinstance(positions_raw, list):
            raise ValueError("validated_positions must be a list")
        seq = payload.get("removal_seq")
        if isinstance(seq, bool) or not isinstance(seq, int) or seq < 0:
            raise ValueError("removal_seq must be an int >= 0")
        entries: dict[str, _ActiveEntry] = {}
        for item in entries_raw:
            entry = _ActiveEntry.from_dict(_require_mapping(item, "entry"))
            entries[entry.card.card_id] = entry
        removed = [
            RemovalRecord.from_dict(_require_mapping(item, "removal")) for item in removed_raw
        ]
        validated: set[tuple[str, str]] = set()
        for item in positions_raw:
            position = _require_mapping(item, "validated_position")
            validated.add(
                (
                    _require_str(position.get("view_revision_id"), "view_revision_id"),
                    _require_str(position.get("block_or_claim_id"), "block_or_claim_id"),
                )
            )
        with self._lock:
            self._entries = entries
            self._removed = removed
            self._validated = validated
            self._removal_seq = seq
            if persist:
                self._save_locked()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        self._apply(payload, persist=False)

    def _save_locked(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(self.path)


def _cards_from_revision(
    revision: InfluenceViewRevision,
    *,
    acl_labels: tuple[str, ...],
    created_at: str,
) -> tuple[InfluenceProjection, ...]:
    cards: list[InfluenceProjection] = []
    for block in revision.blocks:
        cards.append(
            _make_card(
                revision=revision,
                kind=CARD_KIND_BLOCK,
                entity_id=block.block_id,
                summary=block.summary,
                semantic_type=block.role,
                acl_labels=acl_labels,
                created_at=created_at,
            )
        )
    for claim in revision.claims:
        summary = f"{claim.carried_artifact} -> {claim.downstream_effect}"
        cards.append(
            _make_card(
                revision=revision,
                kind=CARD_KIND_CLAIM,
                entity_id=claim.claim_id,
                summary=summary,
                semantic_type=claim.relation_type,
                acl_labels=acl_labels,
                created_at=created_at,
            )
        )
    return tuple(cards)


def _make_card(
    *,
    revision: InfluenceViewRevision,
    kind: CardKind,
    entity_id: str,
    summary: str,
    semantic_type: str,
    acl_labels: tuple[str, ...],
    created_at: str,
) -> InfluenceProjection:
    pointer = SourcePointer(view_revision_id=revision.revision_id, block_or_claim_id=entity_id)
    content_hash = _card_content_hash(
        kind=kind,
        summary=summary,
        semantic_type=semantic_type,
        view_revision_id=revision.revision_id,
        block_or_claim_id=entity_id,
        session_id=revision.session_id,
    )
    return InfluenceProjection(
        card_id=f"{kind}:{revision.revision_id}:{entity_id}",
        kind=kind,
        summary=summary,
        semantic_type=semantic_type,
        status=CARD_STATUS_PROVISIONAL,
        source_pointer=pointer,
        embedding_stub=_embedding_stub(content_hash),
        acl_labels=acl_labels,
        content_hash=content_hash,
        created_at=created_at,
        session_id=revision.session_id,
    )

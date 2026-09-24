# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Append-only citation and exposure ledger.

Citation events are immutable and never rewritten. Reference strength is a
pure projection: in-task and cross-task dimensions stay separate, each
citation is equal-weight inside its dimension, and exponential decay is
applied at read time from a ``PolicyVersion``. Index-level exposure is
stored for offline recall evaluation and never enters strength.
"""

from __future__ import annotations

import json
import threading
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

LEDGER_FILENAME = "citation-ledger.jsonl"

KIND_CITATION = "citation"
KIND_HINT_EXPOSURE = "hint_exposure"
KIND_INDEX_LEVEL_EXPOSURE = "index_level_exposure"
_KINDS = frozenset({KIND_CITATION, KIND_HINT_EXPOSURE, KIND_INDEX_LEVEL_EXPOSURE})

ROLE_CONSIDERED = "considered"
ROLE_APPLIED = "applied"
ROLE_REJECTED = "rejected"
ROLE_COMPARED = "compared"
_ROLES = frozenset({ROLE_CONSIDERED, ROLE_APPLIED, ROLE_REJECTED, ROLE_COMPARED})

_DAY = timedelta(days=1)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _format_dt(value: datetime) -> str:
    return _as_utc(value).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _parse_dt(value: str) -> datetime:
    text = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    return _as_utc(parsed)


def _require_str(value: Any, field: str) -> str:
    if not isinstance(value, str) or value == "":
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _decay_weight(age_days: float, half_life_days: float) -> float:
    """``weight(age) = 2^(-age/half_life)``."""
    return 2.0 ** (-age_days / half_life_days)


@dataclass(frozen=True)
class CitationRef:
    """Structured memory reference: ``{type, id, revision}``."""

    type: str
    id: str
    revision: str

    def __post_init__(self) -> None:
        _require_str(self.type, "ref.type")
        _require_str(self.id, "ref.id")
        if not isinstance(self.revision, str):
            raise ValueError("ref.revision must be a string")

    def to_dict(self) -> dict[str, str]:
        return {"type": self.type, "id": self.id, "revision": self.revision}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CitationRef:
        try:
            ref_type = data["type"]
            ref_id = data["id"]
            revision = data["revision"]
        except KeyError as exc:
            raise ValueError("ref requires type, id, and revision") from exc
        return cls(
            type=_require_str(ref_type, "ref.type"),
            id=_require_str(ref_id, "ref.id"),
            revision=revision if isinstance(revision, str) else _bad_revision(),
        )


def _bad_revision() -> str:
    raise ValueError("ref.revision must be a string")


def _coerce_ref(ref: CitationRef | Mapping[str, Any]) -> CitationRef:
    if isinstance(ref, CitationRef):
        return ref
    if isinstance(ref, Mapping):
        return CitationRef.from_dict(ref)
    raise TypeError("ref must be a CitationRef or a mapping with type, id, and revision")


@dataclass(frozen=True)
class PolicyVersion:
    """Decay policy used to project strength. Changing it never rewrites events."""

    policy_id: str
    in_task_half_life_days: float = 7
    cross_task_half_life_days: float = 90

    def __post_init__(self) -> None:
        _require_str(self.policy_id, "policy_id")
        in_task = float(self.in_task_half_life_days)
        cross_task = float(self.cross_task_half_life_days)
        if in_task <= 0 or cross_task <= 0:
            raise ValueError("half-life days must be positive")
        object.__setattr__(self, "in_task_half_life_days", in_task)
        object.__setattr__(self, "cross_task_half_life_days", cross_task)

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "in_task_half_life_days": self.in_task_half_life_days,
            "cross_task_half_life_days": self.cross_task_half_life_days,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PolicyVersion:
        return cls(
            policy_id=_require_str(data["policy_id"], "policy_id"),
            in_task_half_life_days=float(data.get("in_task_half_life_days", 7)),
            cross_task_half_life_days=float(data.get("cross_task_half_life_days", 90)),
        )


@dataclass(frozen=True)
class ReferenceStrength:
    """Two independent dimensions. No hidden-event count is part of this value."""

    in_task_reference_strength: float
    cross_task_reference_strength: float

    def to_dict(self) -> dict[str, float]:
        return {
            "in_task_reference_strength": self.in_task_reference_strength,
            "cross_task_reference_strength": self.cross_task_reference_strength,
        }


@dataclass(frozen=True)
class CitationEvent:
    """Immutable consumption event. The ledger appends these and never overwrites them."""

    event_id: str
    kind: str
    task_id: str
    source_task_id: str
    ref: CitationRef
    role: str | None
    bridge_id: str | None
    source_channel_labels: tuple[str, ...]
    occurred_at: str
    policy_version: str

    def __post_init__(self) -> None:
        _require_str(self.event_id, "event_id")
        if self.kind not in _KINDS:
            raise ValueError(f"kind must be one of {sorted(_KINDS)}")
        _require_str(self.task_id, "task_id")
        _require_str(self.source_task_id, "source_task_id")
        if not isinstance(self.ref, CitationRef):
            raise TypeError("ref must be a CitationRef")
        if not isinstance(self.source_channel_labels, tuple):
            raise TypeError("source_channel_labels must be a tuple of strings")
        for label in self.source_channel_labels:
            if not isinstance(label, str) or label == "":
                raise ValueError("source_channel_labels entries must be non-empty strings")
        _parse_dt(self.occurred_at)
        _require_str(self.policy_version, "policy_version")
        if self.kind == KIND_CITATION:
            if self.role not in _ROLES:
                raise ValueError(f"citation role must be one of {sorted(_ROLES)}")
            if self.bridge_id is not None and (
                not isinstance(self.bridge_id, str) or self.bridge_id == ""
            ):
                raise ValueError("bridge_id must be a non-empty string when set")
        else:
            if self.role is not None:
                raise ValueError("role is only valid on citation events")
            if self.bridge_id is not None:
                raise ValueError("bridge_id is only valid on citation events")

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "kind": self.kind,
            "task_id": self.task_id,
            "source_task_id": self.source_task_id,
            "ref": self.ref.to_dict(),
            "role": self.role,
            "bridge_id": self.bridge_id,
            "source_channel_labels": list(self.source_channel_labels),
            "occurred_at": self.occurred_at,
            "policy_version": self.policy_version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CitationEvent:
        raw_ref = data["ref"]
        if not isinstance(raw_ref, Mapping):
            raise TypeError("ref must be a mapping")
        labels = data["source_channel_labels"]
        if not isinstance(labels, (list, tuple)):
            raise TypeError("source_channel_labels must be a list of strings")
        role = data["role"]
        bridge_id = data["bridge_id"]
        return cls(
            event_id=_require_str(data["event_id"], "event_id"),
            kind=_require_str(data["kind"], "kind"),
            task_id=_require_str(data["task_id"], "task_id"),
            source_task_id=_require_str(data["source_task_id"], "source_task_id"),
            ref=CitationRef.from_dict(raw_ref),
            role=None if role is None else _require_str(role, "role"),
            bridge_id=None if bridge_id is None else _require_str(bridge_id, "bridge_id"),
            source_channel_labels=tuple(labels),
            occurred_at=_require_str(data["occurred_at"], "occurred_at"),
            policy_version=_require_str(data["policy_version"], "policy_version"),
        )


def _labels_overlap(event_labels: tuple[str, ...], principal_labels: frozenset[str]) -> bool:
    if not event_labels or not principal_labels:
        return False
    return any(label in principal_labels for label in event_labels)


class CitationLedger:
    """Append-only JSONL ledger of citation and exposure events."""

    def __init__(
        self,
        path: str | Path,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._path = _resolve_ledger_path(Path(path))
        self._clock = clock or _utc_now
        self._lock = threading.Lock()
        self._events: list[CitationEvent] = []
        self._by_id: dict[str, CitationEvent] = {}
        self.skipped_line_count = 0
        self._load()

    @property
    def path(self) -> Path:
        return self._path

    def record_citation(
        self,
        *,
        task_id: str,
        source_task_id: str,
        ref: CitationRef | Mapping[str, Any],
        role: str,
        source_channel_labels: Iterable[str],
        policy_version: str,
        bridge_id: str | None = None,
    ) -> CitationEvent:
        """Append one structured citation.

        Contract: only a structured citation is recorded. The caller must
        already hold ``{memory_ref, role}`` where role is considered, applied,
        rejected, or compared. A natural-language mention of an identifier is
        not a citation and must not be passed here. This ledger does not
        inspect prose; the caller guarantees every invocation is structured.

        ``bridge_id`` counts toward a bridge only when the caller explicitly
        passes a non-empty id. Events are append-only and never overwritten.
        """
        stored_bridge = _optional_bridge_id(bridge_id)
        event = self._new_event(
            kind=KIND_CITATION,
            task_id=task_id,
            source_task_id=source_task_id,
            ref=_coerce_ref(ref),
            role=_require_str(role, "role"),
            bridge_id=stored_bridge,
            source_channel_labels=_freeze_labels(source_channel_labels),
            policy_version=policy_version,
        )
        return self._append(event)

    def record_hint_exposure(
        self,
        *,
        task_id: str,
        source_task_id: str,
        ref: CitationRef | Mapping[str, Any],
        source_channel_labels: Iterable[str],
        policy_version: str,
    ) -> CitationEvent:
        """Append a Memory Hint exposure.

        Hint exposure is not a citation and does not contribute to either
        reference-strength dimension.
        """
        event = self._new_event(
            kind=KIND_HINT_EXPOSURE,
            task_id=task_id,
            source_task_id=source_task_id,
            ref=_coerce_ref(ref),
            role=None,
            bridge_id=None,
            source_channel_labels=_freeze_labels(source_channel_labels),
            policy_version=policy_version,
        )
        return self._append(event)

    def record_index_level_exposure(
        self,
        *,
        task_id: str,
        source_task_id: str,
        ref: CitationRef | Mapping[str, Any],
        source_channel_labels: Iterable[str],
        policy_version: str,
    ) -> CitationEvent:
        """Append an index-level exposure for a projection-card hit.

        Projection hits are stored here and never enter strength. They are
        excluded from ``materialize_strength`` and ``visible_strength``.
        """
        event = self._new_event(
            kind=KIND_INDEX_LEVEL_EXPOSURE,
            task_id=task_id,
            source_task_id=source_task_id,
            ref=_coerce_ref(ref),
            role=None,
            bridge_id=None,
            source_channel_labels=_freeze_labels(source_channel_labels),
            policy_version=policy_version,
        )
        return self._append(event)

    def events(self) -> list[CitationEvent]:
        with self._lock:
            return list(self._events)

    def materialize_strength(
        self,
        ref: CitationRef | Mapping[str, Any],
        as_of: datetime,
        policy: PolicyVersion,
    ) -> ReferenceStrength:
        """Project dual reference strength at ``as_of`` under ``policy``.

        Pure read: events are not modified and the JSONL file is not rewritten.
        Only ``citation`` events for ``ref`` at or before ``as_of`` contribute.
        ``task_id == source_task_id`` accumulates in-task strength; any other
        id accumulates cross-task strength. The two sums are returned separately.
        Each citation has base weight 1, decayed as ``2^(-age_days/half_life)``.
        A different ``PolicyVersion`` recomputes the projection only.
        """
        target = _coerce_ref(ref)
        moment = _as_utc(as_of)
        with self._lock:
            snapshot = list(self._events)
        return _project(snapshot, target, moment, policy, principal_labels=None)

    def visible_strength(
        self,
        ref: CitationRef | Mapping[str, Any],
        principal_labels: Iterable[str],
        policy: PolicyVersion,
    ) -> ReferenceStrength:
        """ACL-safe strength for one principal.

        A citation contributes only when ``source_channel_labels`` intersects
        ``principal_labels``. The return value has only the two strength
        fields — no hidden count, no filtered delta, and no task id. The
        projection instant is the injected clock. Hint exposure and
        index-level exposure never contribute.
        """
        target = _coerce_ref(ref)
        allowed = frozenset(_freeze_labels(principal_labels))
        moment = _as_utc(self._clock())
        with self._lock:
            snapshot = list(self._events)
        return _project(snapshot, target, moment, policy, principal_labels=allowed)

    def count_index_exposures(self, ref: CitationRef | Mapping[str, Any]) -> int:
        """Count index-level exposures for ``ref``.

        Offline-only: for recall-quality evaluation. Not an input to online
        ranking, reference strength, or utility.
        """
        target = _coerce_ref(ref)
        with self._lock:
            return sum(
                1
                for event in self._events
                if event.kind == KIND_INDEX_LEVEL_EXPOSURE and event.ref == target
            )

    def bridge_citation_count(self, bridge_id: str) -> int:
        """Count citations that explicitly include ``bridge_id``.

        Citations with no bridge id, and every non-citation event, contribute
        nothing.
        """
        if not isinstance(bridge_id, str) or bridge_id == "":
            return 0
        with self._lock:
            return sum(
                1
                for event in self._events
                if event.kind == KIND_CITATION and event.bridge_id == bridge_id
            )

    def _new_event(
        self,
        *,
        kind: str,
        task_id: str,
        source_task_id: str,
        ref: CitationRef,
        role: str | None,
        bridge_id: str | None,
        source_channel_labels: tuple[str, ...],
        policy_version: str,
    ) -> CitationEvent:
        occurred_at = _format_dt(_as_utc(self._clock()))
        return CitationEvent(
            event_id=uuid.uuid4().hex,
            kind=kind,
            task_id=_require_str(task_id, "task_id"),
            source_task_id=_require_str(source_task_id, "source_task_id"),
            ref=ref,
            role=role,
            bridge_id=bridge_id,
            source_channel_labels=source_channel_labels,
            occurred_at=occurred_at,
            policy_version=_require_str(policy_version, "policy_version"),
        )

    def _append(self, event: CitationEvent) -> CitationEvent:
        line = json.dumps(event.to_dict(), ensure_ascii=False) + "\n"
        with self._lock:
            if event.event_id in self._by_id:
                raise ValueError(f"event already exists: {event.event_id}")
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()
            self._events.append(event)
            self._by_id[event.event_id] = event
            return event

    def _load(self) -> None:
        events: list[CitationEvent] = []
        by_id: dict[str, CitationEvent] = {}
        skipped = 0
        if self._path.is_file():
            with self._path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    raw = line.strip()
                    if not raw:
                        continue
                    try:
                        payload = json.loads(raw)
                        if not isinstance(payload, dict):
                            raise TypeError("ledger line must be a JSON object")
                        event = CitationEvent.from_dict(payload)
                    except (
                        json.JSONDecodeError,
                        TypeError,
                        KeyError,
                        ValueError,
                    ):
                        skipped += 1
                        continue
                    if event.event_id in by_id:
                        skipped += 1
                        continue
                    events.append(event)
                    by_id[event.event_id] = event
        with self._lock:
            self._events = events
            self._by_id = by_id
            self.skipped_line_count = skipped


def _resolve_ledger_path(path: Path) -> Path:
    if path.suffix == ".jsonl":
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    path.mkdir(parents=True, exist_ok=True)
    return path / LEDGER_FILENAME


def _freeze_labels(labels: Iterable[str]) -> tuple[str, ...]:
    frozen: list[str] = []
    for label in labels:
        frozen.append(_require_str(label, "label"))
    return tuple(frozen)


def _optional_bridge_id(bridge_id: str | None) -> str | None:
    if bridge_id is None:
        return None
    if not isinstance(bridge_id, str):
        raise TypeError("bridge_id must be a string or None")
    if bridge_id == "":
        return None
    return bridge_id


def _project(
    events: list[CitationEvent],
    ref: CitationRef,
    as_of: datetime,
    policy: PolicyVersion,
    principal_labels: frozenset[str] | None,
) -> ReferenceStrength:
    in_task = 0.0
    cross_task = 0.0
    for event in events:
        if event.kind != KIND_CITATION or event.ref != ref:
            continue
        occurred_at = _parse_dt(event.occurred_at)
        if occurred_at > as_of:
            continue
        if principal_labels is not None and not _labels_overlap(
            event.source_channel_labels,
            principal_labels,
        ):
            continue
        age_days = (as_of - occurred_at) / _DAY
        if event.task_id == event.source_task_id:
            in_task += _decay_weight(age_days, policy.in_task_half_life_days)
        else:
            cross_task += _decay_weight(age_days, policy.cross_task_half_life_days)
    return ReferenceStrength(
        in_task_reference_strength=in_task,
        cross_task_reference_strength=cross_task,
    )

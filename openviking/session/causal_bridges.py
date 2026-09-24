# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""HypothesisBridge / EvidenceBridge typing and hypothesis-graph governance (ADR-0011).

Bridges aggregate by canonical directed endpoint pair + relation_type. Every
model judgment is an append-only audit event. Active outgoing hypothesis edges
are capped per node; excess edges leave the traversal index without deleting
judgments. Writes are gated by ``is_causal_mode_enabled``.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from openviking.session.causal_experiences import NAMESPACE_DIRNAME
from openviking_cli.utils.config.memory_config import is_causal_mode_enabled

logger = logging.getLogger(__name__)

BRIDGES_DIRNAME = "bridges"
EVENTS_FILENAME = "events.jsonl"

PER_NODE_ACTIVE_CAP = 32
PER_RUN_WRITE_QUOTA = 12
PATH_MAX_HYPOTHESIS_HOPS = 2
PATH_MAX_ENTITIES = 12

BridgeKind = Literal["hypothesis", "evidence"]
BridgeStatus = Literal["active", "inactive", "reactivated"]

BRIDGE_KINDS = frozenset(("hypothesis", "evidence"))
BRIDGE_STATUSES = frozenset(("active", "inactive", "reactivated"))
TRAVERSABLE_STATUSES = frozenset(("active", "reactivated"))
REACTIVATION_SIGNALS = frozenset(
    (
        "new_judgment",
        "citation_with_bridge_id",
        "adoption_or_outcome",
        "direct_relevance_recheck",
    )
)

ScoreFn = Callable[[float, float, float], float]


class QuotaExceeded(Exception):
    """Raised when one evaluation run exceeds the hypothesis judgment write quota."""

    def __init__(self, run_id: str, quota: int) -> None:
        self.run_id = run_id
        self.quota = quota
        super().__init__(
            f"per-run hypothesis judgment quota exceeded for run {run_id!r}: limit {quota}"
        )


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _canonical_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _require_mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a dict")
    return value


def _require_str(value: Any, field: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a str")
    if not allow_empty and not value:
        raise ValueError(f"{field} must be a non-empty str")
    return value


def _require_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number")
    return float(value)


def _require_optional_str(value: Any, field: str) -> str | None:
    if value is None:
        return None
    return _require_str(value, field)


def _tighten_limit(requested: int, ceiling: int, field: str) -> int:
    if isinstance(requested, bool) or not isinstance(requested, int):
        raise ValueError(f"{field} must be an int")
    if requested < 1:
        raise ValueError(f"{field} must be >= 1")
    if requested > ceiling:
        raise ValueError(
            f"{field} cannot exceed the global ceiling {ceiling} (workspace may only tighten)"
        )
    return requested


def default_governance_score(
    direct_relevance: float, time_decay: float, citation_strength: float
) -> float:
    """Direct relevance × time decay × citation strength."""
    return direct_relevance * time_decay * citation_strength


@dataclass(frozen=True)
class NodeRef:
    """Endpoint reference ``{type, id}``."""

    type: str
    id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "type", _require_str(self.type, "type"))
        object.__setattr__(self, "id", _require_str(self.id, "id"))

    def to_dict(self) -> dict[str, str]:
        return {"type": self.type, "id": self.id}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NodeRef:
        payload = _require_mapping(data, "node_ref")
        return cls(
            type=str(payload.get("type") or ""),
            id=str(payload.get("id") or ""),
        )


def validate_node_ref(value: NodeRef | Mapping[str, Any], field: str) -> NodeRef:
    if isinstance(value, NodeRef):
        return value
    payload = _require_mapping(value, field)
    return NodeRef(
        type=_require_str(payload.get("type"), f"{field}.type"),
        id=_require_str(payload.get("id"), f"{field}.id"),
    )


def canonical_bridge_key(source_ref: NodeRef, target_ref: NodeRef, relation_type: str) -> str:
    """Directed canonical key: endpoint pair + relation_type.

    ``(source, target, relation)`` and the swapped pair are different bridges.
    Outgoing caps count edges whose ``source_ref`` is the node.
    """
    return _canonical_dumps(
        {
            "relation_type": relation_type,
            "source": source_ref.to_dict(),
            "target": target_ref.to_dict(),
        }
    )


@dataclass(frozen=True)
class Provenance:
    """Creator run and judgment event that minted the bridge."""

    run_id: str
    judgment_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _require_str(self.run_id, "provenance.run_id"))
        object.__setattr__(
            self, "judgment_id", _require_str(self.judgment_id, "provenance.judgment_id")
        )

    def to_dict(self) -> dict[str, str]:
        return {"run_id": self.run_id, "judgment_id": self.judgment_id}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Provenance:
        payload = _require_mapping(data, "provenance")
        return cls(
            run_id=str(payload.get("run_id") or ""),
            judgment_id=str(payload.get("judgment_id") or ""),
        )


@dataclass(frozen=True)
class StatusEvent:
    """One entry in the bridge status event stream."""

    status: BridgeStatus
    at: str
    signal: str | None = None
    detail: str | None = None

    def __post_init__(self) -> None:
        if self.status not in BRIDGE_STATUSES:
            raise ValueError(
                f"status must be one of {sorted(BRIDGE_STATUSES)}, got {self.status!r}"
            )
        object.__setattr__(self, "at", _require_str(self.at, "status_event.at"))
        object.__setattr__(self, "signal", _require_optional_str(self.signal, "status_event.signal"))
        object.__setattr__(self, "detail", _require_optional_str(self.detail, "status_event.detail"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "at": self.at,
            "signal": self.signal,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StatusEvent:
        payload = _require_mapping(data, "status_event")
        return cls(
            status=payload.get("status"),  # type: ignore[arg-type]
            at=str(payload.get("at") or ""),
            signal=payload.get("signal"),
            detail=payload.get("detail"),
        )


@dataclass(frozen=True)
class HypothesisJudgment:
    """Immutable model judgment. All events are retained for audit."""

    judgment_id: str
    canonical_key: str
    relevance: float
    timestamp: str
    run_id: str
    bridge_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "judgment_id", _require_str(self.judgment_id, "judgment_id"))
        object.__setattr__(self, "canonical_key", _require_str(self.canonical_key, "canonical_key"))
        object.__setattr__(self, "relevance", _require_number(self.relevance, "relevance"))
        object.__setattr__(self, "timestamp", _require_str(self.timestamp, "timestamp"))
        object.__setattr__(self, "run_id", _require_str(self.run_id, "run_id"))
        object.__setattr__(self, "bridge_id", _require_str(self.bridge_id, "bridge_id"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "judgment_id": self.judgment_id,
            "canonical_key": self.canonical_key,
            "relevance": self.relevance,
            "timestamp": self.timestamp,
            "run_id": self.run_id,
            "bridge_id": self.bridge_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HypothesisJudgment:
        payload = _require_mapping(data, "judgment")
        return cls(
            judgment_id=str(payload.get("judgment_id") or ""),
            canonical_key=str(payload.get("canonical_key") or ""),
            relevance=payload.get("relevance"),  # type: ignore[arg-type]
            timestamp=str(payload.get("timestamp") or ""),
            run_id=str(payload.get("run_id") or ""),
            bridge_id=str(payload.get("bridge_id") or ""),
        )


def _status_events_from(value: Any) -> tuple[StatusEvent, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("status_events must be a non-empty list")
    return tuple(
        item if isinstance(item, StatusEvent) else StatusEvent.from_dict(item) for item in value
    )


def _evidence_refs_from(value: Any) -> tuple[NodeRef, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError("evidence_refs must be a list")
    refs: list[NodeRef] = []
    for index, item in enumerate(value):
        refs.append(
            item if isinstance(item, NodeRef) else validate_node_ref(item, f"evidence_refs[{index}]")
        )
    return tuple(refs)


@dataclass(frozen=True)
class BridgeRecord:
    """Typed bridge. ``status`` is the latest event in ``status_events``."""

    bridge_id: str
    kind: BridgeKind
    source_ref: NodeRef
    target_ref: NodeRef
    relation_type: str
    provenance: Provenance
    status_events: tuple[StatusEvent, ...]
    created_at: str
    canonical_key: str
    direct_relevance: float
    time_decay: float
    citation_strength: float
    governance_score: float
    evidence_refs: tuple[NodeRef, ...] = ()
    relation_semantics: str | None = None
    counter_example_check: dict[str, Any] | str | None = None
    security_labels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "bridge_id", _require_str(self.bridge_id, "bridge_id"))
        if self.kind not in BRIDGE_KINDS:
            raise ValueError(f"kind must be one of {sorted(BRIDGE_KINDS)}, got {self.kind!r}")
        object.__setattr__(self, "source_ref", validate_node_ref(self.source_ref, "source_ref"))
        object.__setattr__(self, "target_ref", validate_node_ref(self.target_ref, "target_ref"))
        object.__setattr__(self, "relation_type", _require_str(self.relation_type, "relation_type"))
        provenance = self.provenance
        if not isinstance(provenance, Provenance):
            provenance = Provenance.from_dict(provenance)
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(self, "status_events", _status_events_from(self.status_events))
        object.__setattr__(self, "created_at", _require_str(self.created_at, "created_at"))
        object.__setattr__(self, "canonical_key", _require_str(self.canonical_key, "canonical_key"))
        object.__setattr__(
            self, "direct_relevance", _require_number(self.direct_relevance, "direct_relevance")
        )
        object.__setattr__(self, "time_decay", _require_number(self.time_decay, "time_decay"))
        object.__setattr__(
            self, "citation_strength", _require_number(self.citation_strength, "citation_strength")
        )
        object.__setattr__(
            self, "governance_score", _require_number(self.governance_score, "governance_score")
        )
        object.__setattr__(self, "evidence_refs", _evidence_refs_from(self.evidence_refs))
        object.__setattr__(
            self,
            "relation_semantics",
            _require_optional_str(self.relation_semantics, "relation_semantics"),
        )
        check = self.counter_example_check
        if check is not None and not isinstance(check, (dict, str)):
            raise ValueError("counter_example_check must be a dict, str, or None")
        if isinstance(check, dict):
            object.__setattr__(self, "counter_example_check", dict(check))
        object.__setattr__(
            self, "security_labels", _security_labels_from(self.security_labels)
        )

    @property
    def status(self) -> BridgeStatus:
        return self.status_events[-1].status

    def to_dict(self) -> dict[str, Any]:
        check = self.counter_example_check
        return {
            "bridge_id": self.bridge_id,
            "kind": self.kind,
            "source_ref": self.source_ref.to_dict(),
            "target_ref": self.target_ref.to_dict(),
            "relation_type": self.relation_type,
            "provenance": self.provenance.to_dict(),
            "status": self.status,
            "status_events": [event.to_dict() for event in self.status_events],
            "created_at": self.created_at,
            "canonical_key": self.canonical_key,
            "direct_relevance": self.direct_relevance,
            "time_decay": self.time_decay,
            "citation_strength": self.citation_strength,
            "governance_score": self.governance_score,
            "evidence_refs": [ref.to_dict() for ref in self.evidence_refs],
            "relation_semantics": self.relation_semantics,
            "counter_example_check": dict(check) if isinstance(check, dict) else check,
            "security_labels": list(self.security_labels),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BridgeRecord:
        payload = _require_mapping(data, "bridge")
        events = payload.get("status_events")
        if not events and payload.get("status"):
            events = [
                {
                    "status": payload.get("status"),
                    "at": payload.get("created_at") or "",
                    "signal": None,
                    "detail": None,
                }
            ]
        return cls(
            bridge_id=str(payload.get("bridge_id") or ""),
            kind=payload.get("kind"),  # type: ignore[arg-type]
            source_ref=payload.get("source_ref") or {},
            target_ref=payload.get("target_ref") or {},
            relation_type=str(payload.get("relation_type") or ""),
            provenance=payload.get("provenance") or {},
            status_events=events or [],
            created_at=str(payload.get("created_at") or ""),
            canonical_key=str(payload.get("canonical_key") or ""),
            direct_relevance=payload.get("direct_relevance", 0),
            time_decay=payload.get("time_decay", 1),
            citation_strength=payload.get("citation_strength", 1),
            governance_score=payload.get("governance_score", 0),
            evidence_refs=payload.get("evidence_refs") or [],
            relation_semantics=payload.get("relation_semantics"),
            counter_example_check=payload.get("counter_example_check"),
            security_labels=payload.get("security_labels") or (),
        )


@dataclass(frozen=True)
class PathViolation:
    """One path-budget failure. An empty list means the path is within budget."""

    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message}


def _entity_key(entity: Any) -> str:
    if isinstance(entity, NodeRef):
        return _canonical_dumps(entity.to_dict())
    if isinstance(entity, str):
        if not entity:
            raise ValueError("entity must be a non-empty str")
        return _canonical_dumps({"id": entity})
    if isinstance(entity, Mapping):
        if "type" in entity and "id" in entity:
            return _canonical_dumps(
                {
                    "type": _require_str(entity.get("type"), "entity.type"),
                    "id": _require_str(entity.get("id"), "entity.id"),
                }
            )
        return _canonical_dumps(dict(entity))
    raise ValueError("entity must be a NodeRef, str, or dict")


def _usage_mapping(item: Any, index: int) -> dict[str, Any]:
    if isinstance(item, BridgeRecord):
        return {
            "kind": item.kind,
            "bridge_id": item.bridge_id,
            "source_ref": item.source_ref.to_dict(),
            "target_ref": item.target_ref.to_dict(),
            "relation_type": item.relation_type,
        }
    return _require_mapping(item, f"bridge_usage[{index}]")


def _usage_kind(item: Mapping[str, Any]) -> str:
    kind = item.get("kind", item.get("bridge_kind", "hypothesis"))
    if kind not in BRIDGE_KINDS:
        raise ValueError(f"bridge kind must be one of {sorted(BRIDGE_KINDS)}, got {kind!r}")
    return str(kind)


def _usage_edge_key(item: Mapping[str, Any]) -> str:
    bridge_id = item.get("bridge_id")
    if isinstance(bridge_id, str) and bridge_id:
        return f"id:{bridge_id}"
    return _canonical_dumps(
        {
            "kind": _usage_kind(item),
            "relation_type": item.get("relation_type"),
            "source": item.get("source_ref", item.get("source")),
            "target": item.get("target_ref", item.get("target")),
        }
    )


def check_path_budget(
    entity_sequence: Sequence[Any],
    bridge_usage: Sequence[Any],
    *,
    max_hypothesis_hops: int = PATH_MAX_HYPOTHESIS_HOPS,
    max_entities: int = PATH_MAX_ENTITIES,
) -> list[PathViolation]:
    """Return path-budget violations.

    Limits cannot be loosened past ``PATH_MAX_HYPOTHESIS_HOPS`` and
    ``PATH_MAX_ENTITIES``. Duplicate nodes and edges are always violations.
    """
    hop_limit = _tighten_limit(
        max_hypothesis_hops, PATH_MAX_HYPOTHESIS_HOPS, "max_hypothesis_hops"
    )
    entity_limit = _tighten_limit(max_entities, PATH_MAX_ENTITIES, "max_entities")
    if not isinstance(entity_sequence, Sequence) or isinstance(entity_sequence, (str, bytes)):
        raise ValueError("entity_sequence must be a sequence")
    if not isinstance(bridge_usage, Sequence) or isinstance(bridge_usage, (str, bytes)):
        raise ValueError("bridge_usage must be a sequence")

    violations: list[PathViolation] = []
    keys: list[str] = []
    seen: set[str] = set()
    for entity in entity_sequence:
        key = _entity_key(entity)
        if key in seen:
            violations.append(
                PathViolation(
                    code="duplicate_node",
                    message=f"duplicate entity {key}",
                )
            )
        else:
            seen.add(key)
        keys.append(key)
    if len(keys) > entity_limit:
        violations.append(
            PathViolation(
                code="entity_count",
                message=f"entity count {len(keys)} exceeds {entity_limit}",
            )
        )

    hop_count = 0
    seen_edges: set[str] = set()
    for index, raw in enumerate(bridge_usage):
        usage = _usage_mapping(raw, index)
        if _usage_kind(usage) == "hypothesis":
            hop_count += 1
        edge_key = _usage_edge_key(usage)
        if edge_key in seen_edges:
            violations.append(
                PathViolation(
                    code="duplicate_edge",
                    message=f"duplicate edge {edge_key}",
                )
            )
        else:
            seen_edges.add(edge_key)
    if hop_count > hop_limit:
        violations.append(
            PathViolation(
                code="hypothesis_hops",
                message=f"hypothesis hops {hop_count} exceeds {hop_limit}",
            )
        )
    return violations


def _security_labels_from(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ValueError("security_labels must be a list of strings")
    return tuple(_require_str(item, "security_labels") for item in value)


def _principal_labels(labels: Any) -> frozenset[str]:
    if labels is None:
        raise ValueError("principal_labels is required")
    if isinstance(labels, (str, bytes)) or not isinstance(labels, Sequence):
        raise TypeError("principal_labels must be a sequence of strings")
    return frozenset(_require_str(item, "principal_labels") for item in labels)


def _bridge_visible(bridge: BridgeRecord, principal: frozenset[str]) -> bool:
    if not bridge.security_labels or not principal:
        return False
    return any(label in principal for label in bridge.security_labels)


def _is_traversable_status(status: str) -> bool:
    return status in TRAVERSABLE_STATUSES


def _append_status(
    bridge: BridgeRecord,
    status: BridgeStatus,
    *,
    at: str,
    signal: str | None = None,
    detail: str | None = None,
) -> BridgeRecord:
    event = StatusEvent(status=status, at=at, signal=signal, detail=detail)
    return BridgeRecord.from_dict(
        {
            **bridge.to_dict(),
            "status_events": [item.to_dict() for item in (*bridge.status_events, event)],
        }
    )


def _with_scores(
    bridge: BridgeRecord,
    *,
    direct_relevance: float,
    time_decay: float,
    citation_strength: float,
    governance_score: float,
) -> BridgeRecord:
    payload = bridge.to_dict()
    payload.update(
        {
            "direct_relevance": direct_relevance,
            "time_decay": time_decay,
            "citation_strength": citation_strength,
            "governance_score": governance_score,
        }
    )
    return BridgeRecord.from_dict(payload)


class CausalBridgeStore:
    """Append-only JSONL store for hypothesis/evidence bridges and judgments."""

    def __init__(
        self,
        root: str | Path,
        config: Any = None,
        *,
        per_node_active_cap: int = PER_NODE_ACTIVE_CAP,
        per_run_write_quota: int = PER_RUN_WRITE_QUOTA,
        score_fn: ScoreFn | None = None,
    ) -> None:
        self._root = Path(root)
        self._config = config
        self._dir = self._root / NAMESPACE_DIRNAME / BRIDGES_DIRNAME
        self._events_path = self._dir / EVENTS_FILENAME
        self._per_node_active_cap = _tighten_limit(
            per_node_active_cap, PER_NODE_ACTIVE_CAP, "per_node_active_cap"
        )
        self._per_run_write_quota = _tighten_limit(
            per_run_write_quota, PER_RUN_WRITE_QUOTA, "per_run_write_quota"
        )
        if score_fn is not None and not callable(score_fn):
            raise ValueError("score_fn must be callable")
        self._score_fn = score_fn
        self._lock = threading.Lock()
        self._bridges: dict[str, BridgeRecord] = {}
        self._by_key: dict[str, str] = {}
        self._judgments: dict[str, HypothesisJudgment] = {}
        self._judgments_by_bridge: dict[str, list[str]] = {}
        self._run_counts: dict[str, int] = {}
        self.skipped_line_count = 0
        self._load()

    def _require_causal_write(self) -> None:
        if not is_causal_mode_enabled(self._config):
            raise PermissionError(
                "causal bridge writes require skill_trajectory_mode=causal"
            )

    def _load(self) -> None:
        skipped = 0
        if not self._events_path.is_file():
            self.skipped_line_count = 0
            return
        with self._events_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                raw = line.strip()
                if not raw:
                    continue
                try:
                    event = json.loads(raw)
                    if not isinstance(event, dict):
                        raise TypeError(
                            f"bridge event at line {line_number} must be an object"
                        )
                    self._apply_event(event)
                except (json.JSONDecodeError, TypeError, KeyError, ValueError) as exc:
                    skipped += 1
                    logger.error(
                        "skipping corrupt bridge event at line %s: %s",
                        line_number,
                        exc,
                    )
        self.skipped_line_count = skipped

    def _apply_event(self, event: Mapping[str, Any]) -> None:
        event_type = event.get("event_type")
        if event_type == "judgment_commit":
            self._apply_judgment_commit(event)
            return
        if event_type == "judgment":
            judgment = HypothesisJudgment.from_dict(event.get("judgment") or {})
            self._remember_judgment(judgment)
            return
        if event_type == "bridge":
            bridge = BridgeRecord.from_dict(event.get("bridge") or {})
            self._remember_bridge(bridge)
            return
        raise ValueError(f"unknown bridge event_type {event_type!r}")

    def _remember_bridge(self, bridge: BridgeRecord) -> None:
        self._bridges[bridge.bridge_id] = bridge
        self._by_key[bridge.canonical_key] = bridge.bridge_id

    def _remember_judgment(self, judgment: HypothesisJudgment) -> bool:
        """Record a judgment once. Returns False when ``judgment_id`` was already applied."""
        if judgment.judgment_id in self._judgments:
            return False
        self._judgments[judgment.judgment_id] = judgment
        self._judgments_by_bridge.setdefault(judgment.bridge_id, []).append(
            judgment.judgment_id
        )
        self._run_counts[judgment.run_id] = self._run_counts.get(judgment.run_id, 0) + 1
        return True

    def _apply_judgment_commit(self, event: Mapping[str, Any]) -> None:
        judgment = HypothesisJudgment.from_dict(event.get("judgment") or {})
        raw_bridges = event.get("bridges")
        if not isinstance(raw_bridges, list) or not raw_bridges:
            raise ValueError("judgment_commit requires a non-empty bridges list")
        bridges = [
            BridgeRecord.from_dict(item if isinstance(item, dict) else {})
            for item in raw_bridges
        ]
        if not any(bridge.bridge_id == judgment.bridge_id for bridge in bridges):
            raise ValueError("judgment_commit bridges do not include judgment.bridge_id")
        if not self._remember_judgment(judgment):
            return
        for bridge in bridges:
            self._remember_bridge(bridge)

    def _append_event(self, event: dict[str, Any]) -> None:
        self._events_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n"
        with self._events_path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())

    def _persist_judgment(self, judgment: HypothesisJudgment) -> None:
        self._append_event({"event_type": "judgment", "judgment": judgment.to_dict()})

    def _persist_bridge(self, bridge: BridgeRecord) -> None:
        self._bridges[bridge.bridge_id] = bridge
        self._by_key[bridge.canonical_key] = bridge.bridge_id
        self._append_event({"event_type": "bridge", "bridge": bridge.to_dict()})

    def _score(
        self,
        direct_relevance: float,
        time_decay: float,
        citation_strength: float,
        score_fn: ScoreFn | None,
    ) -> float:
        fn = score_fn or self._score_fn or default_governance_score
        scored = fn(direct_relevance, time_decay, citation_strength)
        return _require_number(scored, "governance_score")

    def _rank_key(self, bridge: BridgeRecord) -> tuple[float, str, str]:
        return (-bridge.governance_score, bridge.created_at, bridge.bridge_id)

    def _apply_cap(
        self,
        source: NodeRef,
        *,
        contender: BridgeRecord | None,
        signal: str | None,
        at: str,
    ) -> tuple[BridgeRecord | None, list[BridgeRecord]]:
        """Plan cap updates against the full bridge set. Does not persist."""
        pool: dict[str, BridgeRecord] = {}
        for bridge in self._bridges.values():
            if bridge.kind != "hypothesis" or bridge.source_ref != source:
                continue
            if _is_traversable_status(bridge.status):
                pool[bridge.bridge_id] = bridge
        if (
            contender is not None
            and contender.kind == "hypothesis"
            and contender.source_ref == source
        ):
            pool[contender.bridge_id] = contender

        ranked = sorted(pool.values(), key=self._rank_key)
        keep_ids = {bridge.bridge_id for bridge in ranked[: self._per_node_active_cap]}
        updated = contender
        pending: list[BridgeRecord] = []
        for bridge in list(pool.values()):
            kept = bridge.bridge_id in keep_ids
            traversable = _is_traversable_status(bridge.status)
            if traversable and not kept:
                revised = _append_status(
                    bridge, "inactive", at=at, detail="per_node_active_cap"
                )
            elif not traversable and kept:
                revised = _append_status(
                    bridge,
                    "reactivated",
                    at=at,
                    signal=signal or "new_judgment",
                    detail="reentered_top_cap",
                )
            else:
                revised = bridge
            if self._bridges.get(revised.bridge_id) != revised:
                pending.append(revised)
            if updated is not None and revised.bridge_id == updated.bridge_id:
                updated = revised
        return updated, pending

    def record_judgment(
        self,
        *,
        run_id: str,
        source_ref: NodeRef | Mapping[str, Any],
        target_ref: NodeRef | Mapping[str, Any],
        relation_type: str,
        relevance: float,
        timestamp: str | None = None,
        judgment_id: str | None = None,
        direct_relevance: float | None = None,
        time_decay: float = 1.0,
        citation_strength: float = 1.0,
        score_fn: ScoreFn | None = None,
        security_labels: Sequence[str] | None = None,
    ) -> BridgeRecord:
        """Record one judgment, aggregating onto the canonical bridge.

        The same endpoint pair and ``relation_type`` reuse one ``bridge_id``.
        Each new judgment counts against the per-run write quota.
        """
        self._require_causal_write()
        source = validate_node_ref(source_ref, "source_ref")
        target = validate_node_ref(target_ref, "target_ref")
        relation = _require_str(relation_type, "relation_type")
        run = _require_str(run_id, "run_id")
        score_relevance = _require_number(relevance, "relevance")
        decay = _require_number(time_decay, "time_decay")
        citation = _require_number(citation_strength, "citation_strength")
        direct = (
            score_relevance
            if direct_relevance is None
            else _require_number(direct_relevance, "direct_relevance")
        )
        when = _require_str(timestamp, "timestamp") if timestamp is not None else _utc_now_iso()
        jid = _require_str(judgment_id, "judgment_id") if judgment_id is not None else str(uuid.uuid4())
        labels = None if security_labels is None else _security_labels_from(security_labels)
        key = canonical_bridge_key(source, target, relation)

        with self._lock:
            existing_judgment = self._judgments.get(jid)
            if existing_judgment is not None:
                if (
                    existing_judgment.canonical_key != key
                    or existing_judgment.relevance != score_relevance
                    or existing_judgment.run_id != run
                    or existing_judgment.timestamp != when
                ):
                    raise ValueError(
                        "judgment_id conflict: same judgment_id with different content"
                    )
                found = self._bridges.get(existing_judgment.bridge_id)
                if found is None:
                    raise ValueError(f"judgment {jid!r} has no bridge")
                return found

            if self._run_counts.get(run, 0) >= self._per_run_write_quota:
                raise QuotaExceeded(run, self._per_run_write_quota)

            bridge_id = self._by_key.get(key)
            if bridge_id is None:
                bridge_id = str(uuid.uuid4())
                bridge = BridgeRecord(
                    bridge_id=bridge_id,
                    kind="hypothesis",
                    source_ref=source,
                    target_ref=target,
                    relation_type=relation,
                    provenance=Provenance(run_id=run, judgment_id=jid),
                    status_events=(StatusEvent(status="active", at=when),),
                    created_at=when,
                    canonical_key=key,
                    direct_relevance=direct,
                    time_decay=decay,
                    citation_strength=citation,
                    governance_score=0.0,
                    security_labels=labels or (),
                )
                was_inactive = False
            else:
                current = self._bridges[bridge_id]
                if current.kind != "hypothesis":
                    raise ValueError(
                        "cannot record a hypothesis judgment on an evidence bridge"
                    )
                bridge = current
                if labels is not None:
                    payload = bridge.to_dict()
                    payload["security_labels"] = list(labels)
                    bridge = BridgeRecord.from_dict(payload)
                was_inactive = not _is_traversable_status(current.status)

            governance = self._score(direct, decay, citation, score_fn)
            bridge = _with_scores(
                bridge,
                direct_relevance=direct,
                time_decay=decay,
                citation_strength=citation,
                governance_score=governance,
            )
            judgment = HypothesisJudgment(
                judgment_id=jid,
                canonical_key=key,
                relevance=score_relevance,
                timestamp=when,
                run_id=run,
                bridge_id=bridge.bridge_id,
            )
            updated, pending = self._apply_cap(
                source,
                contender=bridge,
                signal="new_judgment" if was_inactive else None,
                at=when,
            )
            if updated is None:
                raise ValueError("failed to persist bridge")
            if all(item.bridge_id != updated.bridge_id for item in pending):
                pending.append(updated)
            self._append_event(
                {
                    "event_type": "judgment_commit",
                    "judgment": judgment.to_dict(),
                    "bridges": [item.to_dict() for item in pending],
                }
            )
            self._apply_event(
                {
                    "event_type": "judgment_commit",
                    "judgment": judgment.to_dict(),
                    "bridges": [item.to_dict() for item in pending],
                }
            )
            stored = self._bridges.get(updated.bridge_id)
            if stored is None:
                raise ValueError("failed to persist bridge")
            return stored

    def reactivate(
        self,
        bridge_id: str,
        signal: str,
        *,
        direct_relevance: float | None = None,
        time_decay: float | None = None,
        citation_strength: float | None = None,
        score_fn: ScoreFn | None = None,
        at: str | None = None,
        acl_valid: bool = True,
    ) -> BridgeRecord:
        """Recompute governance and restore active only when the edge re-enters top-N.

        ``exposure`` and ``traversal`` are not legal signals.
        """
        self._require_causal_write()
        if not isinstance(signal, str) or signal not in REACTIVATION_SIGNALS:
            raise ValueError(
                "reactivation signal must be one of "
                f"{sorted(REACTIVATION_SIGNALS)}; exposure/traversal cannot reactivate "
                f"a bridge, got {signal!r}"
            )
        if not acl_valid:
            raise ValueError("reactivation requires a valid ACL")
        bid = _require_str(bridge_id, "bridge_id")
        when = _require_str(at, "at") if at is not None else _utc_now_iso()

        with self._lock:
            bridge = self._bridges.get(bid)
            if bridge is None:
                raise ValueError(f"unknown bridge_id {bid!r}")
            if bridge.kind != "hypothesis":
                raise ValueError("only hypothesis bridges use governance reactivation")
            if _is_traversable_status(bridge.status):
                return bridge
            direct = (
                bridge.direct_relevance
                if direct_relevance is None
                else _require_number(direct_relevance, "direct_relevance")
            )
            decay = (
                bridge.time_decay
                if time_decay is None
                else _require_number(time_decay, "time_decay")
            )
            citation = (
                bridge.citation_strength
                if citation_strength is None
                else _require_number(citation_strength, "citation_strength")
            )
            governance = self._score(direct, decay, citation, score_fn)
            contender = _with_scores(
                bridge,
                direct_relevance=direct,
                time_decay=decay,
                citation_strength=citation,
                governance_score=governance,
            )
            updated, pending = self._apply_cap(
                contender.source_ref, contender=contender, signal=signal, at=when
            )
            for revised in pending:
                self._persist_bridge(revised)
            if updated is None:
                raise ValueError(f"unknown bridge_id {bid!r}")
            stored = self._bridges.get(updated.bridge_id)
            return stored if stored is not None else updated

    def promote_to_evidence(
        self,
        bridge_id: str,
        evidence_refs: Sequence[NodeRef | Mapping[str, Any]],
        relation_semantics: str,
        counter_example_check: dict[str, Any] | str,
    ) -> BridgeRecord:
        """Promote hypothesis → evidence only with explicit diagnosis inputs.

        Citation count is not an input. There is no citation-threshold promotion.
        """
        self._require_causal_write()
        bid = _require_str(bridge_id, "bridge_id")
        if not isinstance(evidence_refs, Sequence) or isinstance(evidence_refs, (str, bytes)):
            raise ValueError("evidence_refs must be a non-empty list of direct evidence refs")
        refs = _evidence_refs_from(list(evidence_refs))
        if not refs:
            raise ValueError("evidence_refs must be a non-empty list of direct evidence refs")
        semantics = _require_str(relation_semantics, "relation_semantics")
        if isinstance(counter_example_check, str):
            check: dict[str, Any] | str = _require_str(
                counter_example_check, "counter_example_check"
            )
        elif isinstance(counter_example_check, dict):
            if not counter_example_check:
                raise ValueError("counter_example_check must be a non-empty dict or str")
            check = dict(counter_example_check)
        else:
            raise ValueError("counter_example_check must be a non-empty dict or str")

        with self._lock:
            bridge = self._bridges.get(bid)
            if bridge is None:
                raise ValueError(f"unknown bridge_id {bid!r}")
            if bridge.kind == "evidence":
                return bridge
            payload = bridge.to_dict()
            payload["kind"] = "evidence"
            payload["evidence_refs"] = [ref.to_dict() for ref in refs]
            payload["relation_semantics"] = semantics
            payload["counter_example_check"] = check
            promoted = BridgeRecord.from_dict(payload)
            self._persist_bridge(promoted)
            return promoted

    def traversable_edges(
        self,
        node_ref: NodeRef | Mapping[str, Any],
        principal_labels: Sequence[str],
    ) -> list[BridgeRecord]:
        """Active and reactivated edges incident to ``node_ref`` and visible to ``principal_labels``.

        Bridges with no security labels are invisible to every principal.
        Inactive edges are omitted. Governance still ranks the full set.
        """
        node = validate_node_ref(node_ref, "node_ref")
        principal = _principal_labels(principal_labels)
        edges = [
            bridge
            for bridge in self._bridges.values()
            if _bridge_visible(bridge, principal)
            and _is_traversable_status(bridge.status)
            and (bridge.source_ref == node or bridge.target_ref == node)
        ]
        edges.sort(key=lambda bridge: (bridge.created_at, bridge.bridge_id))
        return edges

    def get_bridge(self, bridge_id: str) -> BridgeRecord | None:
        return self._bridges.get(bridge_id)

    def find_bridge(
        self,
        source_ref: NodeRef | Mapping[str, Any],
        target_ref: NodeRef | Mapping[str, Any],
        relation_type: str,
    ) -> BridgeRecord | None:
        source = validate_node_ref(source_ref, "source_ref")
        target = validate_node_ref(target_ref, "target_ref")
        relation = _require_str(relation_type, "relation_type")
        bridge_id = self._by_key.get(canonical_bridge_key(source, target, relation))
        if bridge_id is None:
            return None
        return self._bridges.get(bridge_id)

    def judgments_for(self, bridge_id: str) -> list[HypothesisJudgment]:
        ids = self._judgments_by_bridge.get(bridge_id, [])
        return [self._judgments[judgment_id] for judgment_id in ids]

    def list_bridges(self, principal_labels: Sequence[str]) -> list[BridgeRecord]:
        """Return bridges whose security labels intersect ``principal_labels``.

        Bridges with no security labels are invisible to every principal.
        """
        principal = _principal_labels(principal_labels)
        bridges = [
            bridge
            for bridge in self._bridges.values()
            if _bridge_visible(bridge, principal)
        ]
        bridges.sort(key=lambda bridge: (bridge.created_at, bridge.bridge_id))
        return bridges

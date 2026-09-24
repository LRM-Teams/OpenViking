# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Directed Memory Hint delivery, disposition, and throttle (ADR-0013, Q119-A).

A Memory Hint is a task/run-scoped terminal consumption event. It is not
executable, target agent decides. Delivery is ``memory@target_agent`` only:
the hint is not an explore hop and this module exposes no execute API.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openviking.session.citation_ledger import CitationLedger

STORE_FILENAME = "memory-hints.json"

SOURCE_KIND_FORK = "fork"
SOURCE_KIND_BRANCH = "branch"
SOURCE_KIND_GUIDANCE = "guidance"
SOURCE_KIND_SKILL = "skill"
SOURCE_KINDS = frozenset(
    {
        SOURCE_KIND_FORK,
        SOURCE_KIND_BRANCH,
        SOURCE_KIND_GUIDANCE,
        SOURCE_KIND_SKILL,
    }
)

DECISION_ADOPT = "adopt"
DECISION_REJECT = "reject"
DECISION_DEFER = "defer"
DECISIONS = frozenset({DECISION_ADOPT, DECISION_REJECT, DECISION_DEFER})

FOLLOWTHROUGH_OBSERVED = "observed"
FOLLOWTHROUGH_PARTIAL = "partial"
FOLLOWTHROUGH_NOT_OBSERVED = "not_observed"
FOLLOWTHROUGH_CONTRADICTED = "contradicted"
FOLLOWTHROUGHS = frozenset(
    {
        FOLLOWTHROUGH_OBSERVED,
        FOLLOWTHROUGH_PARTIAL,
        FOLLOWTHROUGH_NOT_OBSERVED,
        FOLLOWTHROUGH_CONTRADICTED,
    }
)

OUTCOME_POSITIVE = "positive"
OUTCOME_NEGATIVE = "negative"
OUTCOME_NEUTRAL = "neutral"
OUTCOMES = frozenset({OUTCOME_POSITIVE, OUTCOME_NEGATIVE, OUTCOME_NEUTRAL})

STATE_PENDING = "pending"
STATE_DISPOSED = "disposed"
STATE_EXPIRED = "expired"

DEFAULT_PENDING_CAP = 2
DEFAULT_COOLDOWN_SECONDS = 300.0
PREFERENCE_DEFER_REASON = "preference_defer_all"
DEFAULT_POLICY_VERSION = "memory-hint-v1"

Clock = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _format_dt(value: datetime) -> str:
    text = _as_utc(value).isoformat()
    if text.endswith("+00:00"):
        return text[:-6] + "Z"
    return text


def _parse_dt(value: Any, field: str) -> datetime:
    if isinstance(value, datetime):
        return _as_utc(value)
    if not isinstance(value, str) or value == "":
        raise ValueError(f"{field} must be a datetime or ISO-8601 string")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return _as_utc(parsed)


def _require_str(value: Any, field: str) -> str:
    if not isinstance(value, str) or value == "":
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _freeze_strs(values: Iterable[Any], field: str) -> tuple[str, ...]:
    frozen: list[str] = []
    for item in values:
        frozen.append(_require_str(item, field))
    return tuple(frozen)


def _canonical_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class HintSource:
    """One fork, branch, guidance, or skill cited by a hint."""

    kind: str
    ref: str
    revision: str

    def __post_init__(self) -> None:
        if self.kind not in SOURCE_KINDS:
            raise ValueError(f"kind must be one of {sorted(SOURCE_KINDS)}")
        _require_str(self.ref, "ref")
        if not isinstance(self.revision, str):
            raise ValueError("revision must be a string")

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "ref": self.ref, "revision": self.revision}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> HintSource:
        try:
            kind = data["kind"]
            ref = data["ref"]
            revision = data["revision"]
        except KeyError as exc:
            raise ValueError("source requires kind, ref, and revision") from exc
        return cls(kind=kind, ref=ref, revision=revision if isinstance(revision, str) else _bad_revision())


def _bad_revision() -> str:
    raise ValueError("revision must be a string")


def _coerce_source(value: HintSource | Mapping[str, Any]) -> HintSource:
    if isinstance(value, HintSource):
        return value
    if isinstance(value, Mapping):
        return HintSource.from_dict(value)
    raise TypeError("source must be a HintSource or a mapping")


def provenance_content_hash(sources: Iterable[HintSource]) -> str:
    """Content hash of the source set. Order and duplicates do not change it."""
    unique = {
        (source.kind, source.ref, source.revision): {
            "kind": source.kind,
            "ref": source.ref,
            "revision": source.revision,
        }
        for source in sources
    }
    payload = [unique[key] for key in sorted(unique)]
    return hashlib.sha256(_canonical_dumps(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class MemoryHint:
    """Task/run-scoped terminal consumption event.

    not executable, target agent decides. Sources may synthesize several
    fork, branch, guidance, or skill refs. ``provenance`` is the generating
    run. The hint is delivered only to ``target_agent_id`` and is not an
    explore hop.
    """

    hint_id: str
    task_id: str
    run_id: str
    target_agent_id: str
    sources: tuple[HintSource, ...]
    match_reason: str
    anchor_state_summary: str
    applicability: str
    confidence_status: str
    provenance: str
    ttl_expires_at: datetime
    created_at: datetime

    def __post_init__(self) -> None:
        _require_str(self.hint_id, "hint_id")
        _require_str(self.task_id, "task_id")
        _require_str(self.run_id, "run_id")
        _require_str(self.target_agent_id, "target_agent_id")
        if not isinstance(self.sources, tuple) or not self.sources:
            raise ValueError("sources must be a non-empty tuple of HintSource")
        sources = tuple(_coerce_source(source) for source in self.sources)
        object.__setattr__(self, "sources", sources)
        _require_str(self.match_reason, "match_reason")
        _require_str(self.anchor_state_summary, "anchor_state_summary")
        _require_str(self.applicability, "applicability")
        _require_str(self.confidence_status, "confidence_status")
        _require_str(self.provenance, "provenance")
        object.__setattr__(self, "ttl_expires_at", _parse_dt(self.ttl_expires_at, "ttl_expires_at"))
        object.__setattr__(self, "created_at", _parse_dt(self.created_at, "created_at"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "hint_id": self.hint_id,
            "task_id": self.task_id,
            "run_id": self.run_id,
            "target_agent_id": self.target_agent_id,
            "sources": [source.to_dict() for source in self.sources],
            "match_reason": self.match_reason,
            "anchor_state_summary": self.anchor_state_summary,
            "applicability": self.applicability,
            "confidence_status": self.confidence_status,
            "provenance": self.provenance,
            "ttl_expires_at": _format_dt(self.ttl_expires_at),
            "created_at": _format_dt(self.created_at),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MemoryHint:
        raw_sources = data["sources"]
        if not isinstance(raw_sources, (list, tuple)):
            raise TypeError("sources must be a list")
        return cls(
            hint_id=_require_str(data["hint_id"], "hint_id"),
            task_id=_require_str(data["task_id"], "task_id"),
            run_id=_require_str(data["run_id"], "run_id"),
            target_agent_id=_require_str(data["target_agent_id"], "target_agent_id"),
            sources=tuple(_coerce_source(item) for item in raw_sources),
            match_reason=_require_str(data["match_reason"], "match_reason"),
            anchor_state_summary=_require_str(data["anchor_state_summary"], "anchor_state_summary"),
            applicability=_require_str(data["applicability"], "applicability"),
            confidence_status=_require_str(data["confidence_status"], "confidence_status"),
            provenance=_require_str(data["provenance"], "provenance"),
            ttl_expires_at=_parse_dt(data["ttl_expires_at"], "ttl_expires_at"),
            created_at=_parse_dt(data["created_at"], "created_at"),
        )


@dataclass(frozen=True)
class MemoryDisposition:
    """Target agent's exclusive adopt, reject, or defer decision.

    The delivery side does not accept a proxy judgment. Recording a
    disposition does not rewrite followthrough or outcome events.
    """

    hint_id: str
    target_agent_id: str
    decision: str
    reason: str
    intended_action_refs: tuple[str, ...]
    decided_at: datetime

    def __post_init__(self) -> None:
        _require_str(self.hint_id, "hint_id")
        _require_str(self.target_agent_id, "target_agent_id")
        if self.decision not in DECISIONS:
            raise ValueError(f"decision must be one of {sorted(DECISIONS)}")
        _require_str(self.reason, "reason")
        if not isinstance(self.intended_action_refs, tuple):
            raise TypeError("intended_action_refs must be a tuple of strings")
        object.__setattr__(
            self,
            "intended_action_refs",
            _freeze_strs(self.intended_action_refs, "intended_action_refs"),
        )
        object.__setattr__(self, "decided_at", _parse_dt(self.decided_at, "decided_at"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "hint_id": self.hint_id,
            "target_agent_id": self.target_agent_id,
            "decision": self.decision,
            "reason": self.reason,
            "intended_action_refs": list(self.intended_action_refs),
            "decided_at": _format_dt(self.decided_at),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MemoryDisposition:
        refs = data["intended_action_refs"]
        if not isinstance(refs, (list, tuple)):
            raise TypeError("intended_action_refs must be a list of strings")
        return cls(
            hint_id=_require_str(data["hint_id"], "hint_id"),
            target_agent_id=_require_str(data["target_agent_id"], "target_agent_id"),
            decision=_require_str(data["decision"], "decision"),
            reason=_require_str(data["reason"], "reason"),
            intended_action_refs=tuple(refs),
            decided_at=_parse_dt(data["decided_at"], "decided_at"),
        )


@dataclass(frozen=True)
class FollowthroughRecord:
    """Behavior match recorded independently of disposition. Does not overwrite it."""

    hint_id: str
    followthrough: str
    behavior_evidence_refs: tuple[str, ...]
    recorded_at: datetime

    def __post_init__(self) -> None:
        _require_str(self.hint_id, "hint_id")
        if self.followthrough not in FOLLOWTHROUGHS:
            raise ValueError(f"followthrough must be one of {sorted(FOLLOWTHROUGHS)}")
        if not isinstance(self.behavior_evidence_refs, tuple):
            raise TypeError("behavior_evidence_refs must be a tuple of strings")
        object.__setattr__(
            self,
            "behavior_evidence_refs",
            _freeze_strs(self.behavior_evidence_refs, "behavior_evidence_refs"),
        )
        object.__setattr__(self, "recorded_at", _parse_dt(self.recorded_at, "recorded_at"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "hint_id": self.hint_id,
            "followthrough": self.followthrough,
            "behavior_evidence_refs": list(self.behavior_evidence_refs),
            "recorded_at": _format_dt(self.recorded_at),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FollowthroughRecord:
        refs = data["behavior_evidence_refs"]
        if not isinstance(refs, (list, tuple)):
            raise TypeError("behavior_evidence_refs must be a list of strings")
        return cls(
            hint_id=_require_str(data["hint_id"], "hint_id"),
            followthrough=_require_str(data["followthrough"], "followthrough"),
            behavior_evidence_refs=tuple(refs),
            recorded_at=_parse_dt(data["recorded_at"], "recorded_at"),
        )


@dataclass(frozen=True)
class OutcomeRecord:
    """Task outcome recorded independently of disposition. Does not overwrite it."""

    hint_id: str
    outcome: str
    evidence_refs: tuple[str, ...]
    recorded_at: datetime

    def __post_init__(self) -> None:
        _require_str(self.hint_id, "hint_id")
        if self.outcome not in OUTCOMES:
            raise ValueError(f"outcome must be one of {sorted(OUTCOMES)}")
        if not isinstance(self.evidence_refs, tuple):
            raise TypeError("evidence_refs must be a tuple of strings")
        object.__setattr__(self, "evidence_refs", _freeze_strs(self.evidence_refs, "evidence_refs"))
        object.__setattr__(self, "recorded_at", _parse_dt(self.recorded_at, "recorded_at"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "hint_id": self.hint_id,
            "outcome": self.outcome,
            "evidence_refs": list(self.evidence_refs),
            "recorded_at": _format_dt(self.recorded_at),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> OutcomeRecord:
        refs = data["evidence_refs"]
        if not isinstance(refs, (list, tuple)):
            raise TypeError("evidence_refs must be a list of strings")
        return cls(
            hint_id=_require_str(data["hint_id"], "hint_id"),
            outcome=_require_str(data["outcome"], "outcome"),
            evidence_refs=tuple(refs),
            recorded_at=_parse_dt(data["recorded_at"], "recorded_at"),
        )


@dataclass(frozen=True)
class TargetPreference:
    """Target-agent delivery preference. ``defer_all`` is not a proxy adopt/reject.

    ``silence_all`` is the user-silence preference: deliveries still record
    exposure and do not expect a disposition.
    """

    target_agent_id: str
    defer_all: bool = False
    blocked_kinds: tuple[str, ...] = ()
    silence_all: bool = False

    def __post_init__(self) -> None:
        _require_str(self.target_agent_id, "target_agent_id")
        if not isinstance(self.defer_all, bool):
            raise TypeError("defer_all must be a bool")
        if not isinstance(self.silence_all, bool):
            raise TypeError("silence_all must be a bool")
        if not isinstance(self.blocked_kinds, tuple):
            raise TypeError("blocked_kinds must be a tuple of source kinds")
        kinds = _freeze_strs(self.blocked_kinds, "blocked_kinds")
        unknown = [kind for kind in kinds if kind not in SOURCE_KINDS]
        if unknown:
            raise ValueError(f"blocked_kinds must be subset of {sorted(SOURCE_KINDS)}")
        object.__setattr__(self, "blocked_kinds", kinds)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_agent_id": self.target_agent_id,
            "defer_all": self.defer_all,
            "blocked_kinds": list(self.blocked_kinds),
            "silence_all": self.silence_all,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TargetPreference:
        kinds = data.get("blocked_kinds", [])
        if not isinstance(kinds, (list, tuple)):
            raise TypeError("blocked_kinds must be a list of source kinds")
        return cls(
            target_agent_id=_require_str(data["target_agent_id"], "target_agent_id"),
            defer_all=bool(data.get("defer_all", False)),
            blocked_kinds=tuple(kinds),
            silence_all=bool(data.get("silence_all", False)),
        )


@dataclass(frozen=True)
class DispositionAmendment:
    """Correction appended after the authoritative disposition.

    The first disposition remains the decision. An amendment may replace
    ``reason`` and ``intended_action_refs`` only; it cannot flip ``decision``.
    """

    hint_id: str
    reason: str
    intended_action_refs: tuple[str, ...]
    amended_at: datetime

    def __post_init__(self) -> None:
        _require_str(self.hint_id, "hint_id")
        _require_str(self.reason, "reason")
        if not isinstance(self.intended_action_refs, tuple):
            raise TypeError("intended_action_refs must be a tuple of strings")
        object.__setattr__(
            self,
            "intended_action_refs",
            _freeze_strs(self.intended_action_refs, "intended_action_refs"),
        )
        object.__setattr__(self, "amended_at", _parse_dt(self.amended_at, "amended_at"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "hint_id": self.hint_id,
            "reason": self.reason,
            "intended_action_refs": list(self.intended_action_refs),
            "amended_at": _format_dt(self.amended_at),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DispositionAmendment:
        refs = data["intended_action_refs"]
        if not isinstance(refs, (list, tuple)):
            raise TypeError("intended_action_refs must be a list of strings")
        return cls(
            hint_id=_require_str(data["hint_id"], "hint_id"),
            reason=_require_str(data["reason"], "reason"),
            intended_action_refs=tuple(refs),
            amended_at=_parse_dt(data["amended_at"], "amended_at"),
        )


class DuplicateDisposition(Exception):
    """Raised when a hint already has a disposition."""

    def __init__(self, hint_id: str, existing: MemoryDisposition) -> None:
        self.hint_id = hint_id
        self.existing = existing
        super().__init__(
            f"duplicate disposition for {hint_id}: existing decision {existing.decision}"
        )


class PendingCapExceeded(Exception):
    """Raised when a task/target pair already has ``cap`` undisposed hints."""

    def __init__(self, task_id: str, target_agent_id: str, pending: int, cap: int) -> None:
        self.task_id = task_id
        self.target_agent_id = target_agent_id
        self.pending = pending
        self.cap = cap
        super().__init__(
            f"pending hint cap {cap} exceeded for task {task_id} target {target_agent_id}"
        )


class CooldownActive(Exception):
    """Raised when the target's last delivery is still inside the cooldown window."""

    def __init__(self, target_agent_id: str, remaining_seconds: float) -> None:
        self.target_agent_id = target_agent_id
        self.remaining_seconds = remaining_seconds
        super().__init__(
            f"cooldown active for {target_agent_id}: {remaining_seconds}s remaining"
        )


class BlockedKindRejected(Exception):
    """Raised when a source kind is on the target agent's block list."""

    def __init__(self, target_agent_id: str, kinds: tuple[str, ...]) -> None:
        self.target_agent_id = target_agent_id
        self.kinds = kinds
        super().__init__(f"delivery rejected for {target_agent_id}: blocked kinds {list(kinds)}")


class HintDeliveryService:
    """In-process directed delivery with JSON persistence.

    ``deliver`` addresses ``memory@target_agent`` and records hint exposure on
    the injected citation ledger. It does not execute the hint. Throttle
    defaults follow Q119-A: pending cap 2, cooldown 300 seconds, and same
    source-set provenance refreshes TTL instead of sending again.
    """

    def __init__(
        self,
        path: str | Path,
        ledger: CitationLedger,
        *,
        clock: Clock | None = None,
        pending_cap: int = DEFAULT_PENDING_CAP,
        cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
        policy_version: str = DEFAULT_POLICY_VERSION,
        source_channel_labels: Iterable[str] = ("memory",),
    ) -> None:
        if not isinstance(ledger, CitationLedger):
            raise TypeError("ledger must be a CitationLedger")
        if isinstance(pending_cap, bool) or not isinstance(pending_cap, int) or pending_cap < 1:
            raise ValueError("pending_cap must be a positive int")
        cooldown = float(cooldown_seconds)
        if cooldown < 0:
            raise ValueError("cooldown_seconds must be >= 0")
        self._path = _resolve_store_path(Path(path))
        self._ledger = ledger
        self._clock: Clock = clock or _utc_now
        self._pending_cap = pending_cap
        self._cooldown_seconds = cooldown
        self._policy_version = _require_str(policy_version, "policy_version")
        self._source_channel_labels = _freeze_strs(source_channel_labels, "source_channel_labels")
        self._lock = threading.Lock()
        self._hints: dict[str, MemoryHint] = {}
        self._hint_order: list[str] = []
        self._provenance_index: dict[tuple[str, str, str], str] = {}
        self._dispositions: list[MemoryDisposition] = []
        self._amendments: list[DispositionAmendment] = []
        self._silenced_hint_ids: set[str] = set()
        self._followthroughs: list[FollowthroughRecord] = []
        self._outcomes: list[OutcomeRecord] = []
        self._preferences: dict[str, TargetPreference] = {}
        self._expired: set[str] = set()
        self._last_delivery_at: dict[str, datetime] = {}
        self._load()

    @property
    def path(self) -> Path:
        return self._path

    def register_preference(
        self,
        target_agent_id: str,
        defer_all: bool = False,
        blocked_kinds: Iterable[str] | None = None,
    ) -> TargetPreference:
        """Store the target agent's defer-all and kind filter. Replaces the prior preference."""
        preference = TargetPreference(
            target_agent_id=target_agent_id,
            defer_all=defer_all,
            blocked_kinds=tuple(blocked_kinds or ()),
        )
        with self._lock:
            self._preferences[preference.target_agent_id] = preference
            self._persist()
        return preference

    def silence(self, target_agent_id: str) -> TargetPreference:
        """Record user silence for ``target_agent_id``.

        Deliveries during silence still record exposure and do not expect a
        disposition. This extends the defer-all preference with ``silence_all``
        instead of writing a proxy adopt/reject.
        """
        agent = _require_str(target_agent_id, "target_agent_id")
        with self._lock:
            current = self._preferences.get(agent)
            preference = TargetPreference(
                target_agent_id=agent,
                defer_all=current.defer_all if current is not None else False,
                blocked_kinds=current.blocked_kinds if current is not None else (),
                silence_all=True,
            )
            self._preferences[agent] = preference
            self._persist()
        return preference

    def deliver(
        self,
        hint: MemoryHint,
        clock: Clock | datetime | None = None,
    ) -> MemoryHint:
        """Deliver ``hint`` to its target agent.

        Records one hint exposure when a new hint is accepted. Same source-set
        provenance does not resend: TTL is refreshed and the original hint_id
        is returned. ``defer_all`` still records exposure, then appends a
        system disposition with reason ``preference_defer_all``. ``silence_all``
        still records exposure and does not append a disposition.
        """
        if not isinstance(hint, MemoryHint):
            raise TypeError("hint must be a MemoryHint")
        now = self._resolve_now(clock)
        with self._lock:
            preference = self._preferences.get(hint.target_agent_id)
            if preference is not None:
                blocked = tuple(
                    source.kind for source in hint.sources if source.kind in preference.blocked_kinds
                )
                if blocked:
                    raise BlockedKindRejected(hint.target_agent_id, blocked)
            dedup_key = (
                hint.task_id,
                hint.target_agent_id,
                provenance_content_hash(hint.sources),
            )
            existing_id = self._provenance_index.get(dedup_key)
            if existing_id is not None:
                refreshed = replace(self._hints[existing_id], ttl_expires_at=hint.ttl_expires_at)
                self._hints[existing_id] = refreshed
                self._persist()
                return refreshed
            pending = self._pending_count_locked(hint.task_id, hint.target_agent_id)
            if pending >= self._pending_cap:
                raise PendingCapExceeded(
                    hint.task_id,
                    hint.target_agent_id,
                    pending,
                    self._pending_cap,
                )
            last = self._last_delivery_at.get(hint.target_agent_id)
            if last is not None:
                elapsed = (now - last).total_seconds()
                if elapsed < self._cooldown_seconds:
                    raise CooldownActive(
                        hint.target_agent_id,
                        self._cooldown_seconds - elapsed,
                    )
            if hint.hint_id in self._hints:
                raise ValueError(f"hint already exists: {hint.hint_id}")
            self._hints[hint.hint_id] = hint
            self._hint_order.append(hint.hint_id)
            self._provenance_index[dedup_key] = hint.hint_id
            try:
                self._record_exposure(hint)
            except Exception:
                self._hints.pop(hint.hint_id, None)
                self._hint_order.pop()
                self._provenance_index.pop(dedup_key, None)
                raise
            self._last_delivery_at[hint.target_agent_id] = now
            if preference is not None and preference.defer_all:
                self._dispositions.append(
                    MemoryDisposition(
                        hint_id=hint.hint_id,
                        target_agent_id=hint.target_agent_id,
                        decision=DECISION_DEFER,
                        reason=PREFERENCE_DEFER_REASON,
                        intended_action_refs=(),
                        decided_at=now,
                    )
                )
            elif preference is not None and preference.silence_all:
                self._silenced_hint_ids.add(hint.hint_id)
            self._persist()
            return hint

    def record_disposition(self, disposition: MemoryDisposition) -> MemoryDisposition:
        """Append the first target-agent disposition. A mismatched agent is rejected.

        Each hint accepts one disposition. A later call raises
        ``DuplicateDisposition`` carrying the existing decision. Corrections go
        through ``amend_disposition`` and cannot flip that decision.
        """
        if not isinstance(disposition, MemoryDisposition):
            raise TypeError("disposition must be a MemoryDisposition")
        with self._lock:
            hint = self._require_hint(disposition.hint_id)
            if disposition.target_agent_id != hint.target_agent_id:
                raise ValueError("disposition target_agent_id must match the hint target")
            existing = self._first_disposition_locked(disposition.hint_id)
            if existing is not None:
                raise DuplicateDisposition(disposition.hint_id, existing)
            self._dispositions.append(disposition)
            self._silenced_hint_ids.discard(disposition.hint_id)
            self._persist()
            return disposition

    def amend_disposition(
        self,
        hint_id: str,
        *,
        reason: str = "correction",
        intended_action_refs: Iterable[str] | None = None,
        decision: str | None = None,
        amended_at: datetime | None = None,
    ) -> DispositionAmendment:
        """Append a correction event. The first disposition stays authoritative.

        Amendments may correct ``reason`` and ``intended_action_refs`` only.
        They cannot flip ``decision``: a different ``decision`` raises
        ``ValueError``. The stored first record is not rewritten; readers of
        ``dispositions`` see the latest corrected reason and refs with the
        original decision.
        """
        with self._lock:
            existing = self._first_disposition_locked(hint_id)
            if existing is None:
                self._require_hint(hint_id)
                raise ValueError(f"no disposition to amend for {hint_id}")
            if decision is not None and decision != existing.decision:
                raise ValueError("amend_disposition cannot flip decision")
            current = self._effective_disposition_locked(existing)
            refs = (
                current.intended_action_refs
                if intended_action_refs is None
                else _freeze_strs(intended_action_refs, "intended_action_refs")
            )
            amendment = DispositionAmendment(
                hint_id=existing.hint_id,
                reason=reason,
                intended_action_refs=refs,
                amended_at=amended_at or self._clock(),
            )
            self._amendments.append(amendment)
            self._persist()
            return amendment

    def record_followthrough(self, record: FollowthroughRecord) -> FollowthroughRecord:
        """Append followthrough. Does not overwrite disposition or outcome."""
        if not isinstance(record, FollowthroughRecord):
            raise TypeError("record must be a FollowthroughRecord")
        with self._lock:
            self._require_hint(record.hint_id)
            self._followthroughs.append(record)
            self._persist()
            return record

    def record_outcome(self, record: OutcomeRecord) -> OutcomeRecord:
        """Append outcome. Does not overwrite disposition or followthrough."""
        if not isinstance(record, OutcomeRecord):
            raise TypeError("record must be an OutcomeRecord")
        with self._lock:
            self._require_hint(record.hint_id)
            self._outcomes.append(record)
            self._persist()
            return record

    def expire_stale(self, clock: Clock | datetime | None = None) -> tuple[str, ...]:
        """Mark TTL-expired hints that still have no disposition as expired.

        Expired hints count only as exposure. Hints that already have a
        disposition are left unchanged.
        """
        now = self._resolve_now(clock)
        newly: list[str] = []
        with self._lock:
            for hint_id in self._hint_order:
                if hint_id in self._expired:
                    continue
                if self._has_disposition_locked(hint_id):
                    continue
                hint = self._hints[hint_id]
                if hint.ttl_expires_at <= now:
                    self._expired.add(hint_id)
                    newly.append(hint_id)
            if newly:
                self._persist()
        return tuple(newly)

    def get_hint(self, hint_id: str) -> MemoryHint:
        with self._lock:
            return self._require_hint(hint_id)

    def hints(self) -> tuple[MemoryHint, ...]:
        with self._lock:
            return tuple(self._hints[hint_id] for hint_id in self._hint_order)

    def dispositions(self, hint_id: str | None = None) -> tuple[MemoryDisposition, ...]:
        """Authoritative decisions. Later amendments correct reason and refs only."""
        with self._lock:
            rows = self._dispositions
            if hint_id is not None:
                rows = [row for row in rows if row.hint_id == hint_id]
            return tuple(self._effective_disposition_locked(row) for row in rows)

    def amendments(self, hint_id: str | None = None) -> tuple[DispositionAmendment, ...]:
        with self._lock:
            rows = self._amendments
            if hint_id is not None:
                rows = [row for row in rows if row.hint_id == hint_id]
            return tuple(rows)

    def followthroughs(self, hint_id: str | None = None) -> tuple[FollowthroughRecord, ...]:
        with self._lock:
            rows = self._followthroughs
            if hint_id is not None:
                rows = [row for row in rows if row.hint_id == hint_id]
            return tuple(rows)

    def outcomes(self, hint_id: str | None = None) -> tuple[OutcomeRecord, ...]:
        with self._lock:
            rows = self._outcomes
            if hint_id is not None:
                rows = [row for row in rows if row.hint_id == hint_id]
            return tuple(rows)

    def state(self, hint_id: str) -> str:
        """``disposed`` when any disposition exists, else ``expired`` or ``pending``."""
        with self._lock:
            self._require_hint(hint_id)
            if self._has_disposition_locked(hint_id):
                return STATE_DISPOSED
            if hint_id in self._expired:
                return STATE_EXPIRED
            return STATE_PENDING

    def pending_count(self, task_id: str, target_agent_id: str) -> int:
        with self._lock:
            return self._pending_count_locked(task_id, target_agent_id)

    def _pending_count_locked(self, task_id: str, target_agent_id: str) -> int:
        count = 0
        for hint_id, hint in self._hints.items():
            if hint.task_id != task_id or hint.target_agent_id != target_agent_id:
                continue
            if hint_id in self._expired or hint_id in self._silenced_hint_ids:
                continue
            if self._has_disposition_locked(hint_id):
                continue
            count += 1
        return count

    def _has_disposition_locked(self, hint_id: str) -> bool:
        return any(row.hint_id == hint_id for row in self._dispositions)

    def _first_disposition_locked(self, hint_id: str) -> MemoryDisposition | None:
        for row in self._dispositions:
            if row.hint_id == hint_id:
                return row
        return None

    def _effective_disposition_locked(self, row: MemoryDisposition) -> MemoryDisposition:
        latest = None
        for amendment in self._amendments:
            if amendment.hint_id == row.hint_id:
                latest = amendment
        if latest is None:
            return row
        return replace(
            row,
            reason=latest.reason,
            intended_action_refs=latest.intended_action_refs,
        )

    def _require_hint(self, hint_id: str) -> MemoryHint:
        hint = self._hints.get(hint_id)
        if hint is None:
            raise KeyError(hint_id)
        return hint

    def _record_exposure(self, hint: MemoryHint) -> None:
        self._ledger.record_hint_exposure(
            task_id=hint.task_id,
            source_task_id=hint.task_id,
            ref={
                "type": "memory_hint",
                "id": hint.hint_id,
                "revision": provenance_content_hash(hint.sources),
            },
            source_channel_labels=self._source_channel_labels,
            policy_version=self._policy_version,
        )

    def _resolve_now(self, clock: Clock | datetime | None) -> datetime:
        if clock is None:
            value = self._clock()
        elif isinstance(clock, datetime):
            value = clock
        else:
            value = clock()
        if not isinstance(value, datetime):
            raise TypeError("clock must return a datetime")
        return _as_utc(value)

    def _dump_state(self) -> dict[str, Any]:
        return {
            "hints": [self._hints[hint_id].to_dict() for hint_id in self._hint_order],
            "dispositions": [row.to_dict() for row in self._dispositions],
            "amendments": [row.to_dict() for row in self._amendments],
            "silenced_hint_ids": sorted(self._silenced_hint_ids),
            "followthroughs": [row.to_dict() for row in self._followthroughs],
            "outcomes": [row.to_dict() for row in self._outcomes],
            "preferences": [pref.to_dict() for pref in self._preferences.values()],
            "expired_hint_ids": sorted(self._expired),
            "last_delivery_at": {
                agent: _format_dt(moment) for agent, moment in sorted(self._last_delivery_at.items())
            },
        }

    def _persist(self) -> None:
        text = json.dumps(self._dump_state(), ensure_ascii=False, indent=2) + "\n"
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(self._path)

    def _load(self) -> None:
        if not self._path.is_file():
            return
        payload = json.loads(self._path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("memory hint store must be a JSON object")
        hints: dict[str, MemoryHint] = {}
        order: list[str] = []
        index: dict[tuple[str, str, str], str] = {}
        for raw in payload.get("hints", []):
            hint = MemoryHint.from_dict(raw)
            hints[hint.hint_id] = hint
            order.append(hint.hint_id)
            index[(hint.task_id, hint.target_agent_id, provenance_content_hash(hint.sources))] = (
                hint.hint_id
            )
        dispositions = [MemoryDisposition.from_dict(raw) for raw in payload.get("dispositions", [])]
        amendments = [DispositionAmendment.from_dict(raw) for raw in payload.get("amendments", [])]
        silenced = payload.get("silenced_hint_ids", [])
        followthroughs = [
            FollowthroughRecord.from_dict(raw) for raw in payload.get("followthroughs", [])
        ]
        outcomes = [OutcomeRecord.from_dict(raw) for raw in payload.get("outcomes", [])]
        preferences = {
            pref.target_agent_id: pref
            for pref in (
                TargetPreference.from_dict(raw) for raw in payload.get("preferences", [])
            )
        }
        expired = set(payload.get("expired_hint_ids", []))
        last_raw = payload.get("last_delivery_at", {})
        last_delivery = {
            _require_str(agent, "last_delivery_at agent"): _parse_dt(moment, "last_delivery_at")
            for agent, moment in last_raw.items()
        }
        self._hints = hints
        self._hint_order = order
        self._provenance_index = index
        self._dispositions = dispositions
        self._amendments = amendments
        self._silenced_hint_ids = {hint_id for hint_id in silenced if isinstance(hint_id, str)}
        self._followthroughs = followthroughs
        self._outcomes = outcomes
        self._preferences = preferences
        self._expired = {hint_id for hint_id in expired if isinstance(hint_id, str)}
        self._last_delivery_at = last_delivery


def _resolve_store_path(path: Path) -> Path:
    if path.suffix == ".json":
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    path.mkdir(parents=True, exist_ok=True)
    return path / STORE_FILENAME

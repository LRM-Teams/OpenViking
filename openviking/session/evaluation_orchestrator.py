# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""EvaluationOrchestrator state machine (ADR-0012).

The orchestrator exclusively owns the evaluation-loop transitions. Memory and
diagnosis agents may only submit trigger or schedule proposals; a proposal is
never a legal transition event. The state log is append-only and persisted as
JSON.

One iteration shares five worker seats. Independent diagnosis reservations
stay at or below three. A full pool fails at once with the current holders
instead of waiting.
"""

from __future__ import annotations

import json
import threading
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from openviking.session.handoff_envelope import content_address

STORE_FILENAME = "evaluation-orchestrator.json"

MEMORY_AGENT = "memory_agent"
DIAGNOSIS_AGENT = "diagnosis_agent"
AGENT_ROLES = frozenset({MEMORY_AGENT, DIAGNOSIS_AGENT})

PROPOSAL_TRIGGER = "trigger"
PROPOSAL_SCHEDULE = "schedule"
PROPOSAL_KINDS = frozenset({PROPOSAL_TRIGGER, PROPOSAL_SCHEDULE})

EVENT_MEMORY_RETRIEVE = "memory_retrieve"
EVENT_TASK_RUN = "task_run"
EVENT_SCORE = "score"
EVENT_FREEZE_BRIEF = "freeze_brief"
EVENT_DIAGNOSIS = "diagnosis"
EVENT_COMMIT_REVISIONS = "commit_revisions"
EVENT_NEXT_RUN = "next_run"
EVENT_DONE = "done"
EVENT_STOP = "stop"
EVENT_WAIT_EXTERNAL = "wait_external"
EVENT_RESUME = "resume"

_PROPOSAL_EVENTS = frozenset(
    {
        "proposal",
        "trigger_proposal",
        "schedule_proposal",
        "submit_trigger_proposal",
        "submit_schedule_proposal",
        PROPOSAL_TRIGGER,
        PROPOSAL_SCHEDULE,
    }
)

NO_GAIN_LIMIT = 2
WORKER_SEAT_TOTAL = 5
DIAGNOSIS_SEAT_LIMIT = 3

Clock = Callable[[], datetime]
IdFactory = Callable[[], str]


class OrchestratorState(str, Enum):
    """Evaluation loop states. DONE, STOPPED, and WAITING_EXTERNAL are terminals.

    WAITING_EXTERNAL is terminal until a future evidence event resumes a
    one-shot task run.
    """

    CREATED = "CREATED"
    MEMORY_RETRIEVE = "MEMORY_RETRIEVE"
    TASK_RUN = "TASK_RUN"
    SCORE = "SCORE"
    FREEZE_BRIEF = "FREEZE_BRIEF"
    DIAGNOSIS = "DIAGNOSIS"
    COMMIT_REVISIONS = "COMMIT_REVISIONS"
    NEXT_RUN = "NEXT_RUN"
    DONE = "DONE"
    STOPPED = "STOPPED"
    WAITING_EXTERNAL = "WAITING_EXTERNAL"


TERMINAL_STATES = frozenset({OrchestratorState.DONE, OrchestratorState.STOPPED})

# event -> state it requests, used when the guard rejects the move
_EVENT_TARGET: dict[str, OrchestratorState] = {
    EVENT_MEMORY_RETRIEVE: OrchestratorState.MEMORY_RETRIEVE,
    EVENT_TASK_RUN: OrchestratorState.TASK_RUN,
    EVENT_SCORE: OrchestratorState.SCORE,
    EVENT_FREEZE_BRIEF: OrchestratorState.FREEZE_BRIEF,
    EVENT_DIAGNOSIS: OrchestratorState.DIAGNOSIS,
    EVENT_COMMIT_REVISIONS: OrchestratorState.COMMIT_REVISIONS,
    EVENT_NEXT_RUN: OrchestratorState.NEXT_RUN,
    EVENT_DONE: OrchestratorState.DONE,
    EVENT_STOP: OrchestratorState.STOPPED,
    EVENT_WAIT_EXTERNAL: OrchestratorState.WAITING_EXTERNAL,
    EVENT_RESUME: OrchestratorState.TASK_RUN,
}

TRANSITION_GUARD: dict[tuple[OrchestratorState, str], OrchestratorState] = {
    (OrchestratorState.CREATED, EVENT_MEMORY_RETRIEVE): OrchestratorState.MEMORY_RETRIEVE,
    (OrchestratorState.MEMORY_RETRIEVE, EVENT_TASK_RUN): OrchestratorState.TASK_RUN,
    (OrchestratorState.TASK_RUN, EVENT_SCORE): OrchestratorState.SCORE,
    (OrchestratorState.SCORE, EVENT_FREEZE_BRIEF): OrchestratorState.FREEZE_BRIEF,
    (OrchestratorState.FREEZE_BRIEF, EVENT_DIAGNOSIS): OrchestratorState.DIAGNOSIS,
    (OrchestratorState.DIAGNOSIS, EVENT_COMMIT_REVISIONS): OrchestratorState.COMMIT_REVISIONS,
    (OrchestratorState.COMMIT_REVISIONS, EVENT_NEXT_RUN): OrchestratorState.NEXT_RUN,
    (OrchestratorState.NEXT_RUN, EVENT_MEMORY_RETRIEVE): OrchestratorState.MEMORY_RETRIEVE,
    (OrchestratorState.NEXT_RUN, EVENT_DONE): OrchestratorState.DONE,
    (OrchestratorState.NEXT_RUN, EVENT_STOP): OrchestratorState.STOPPED,
    (OrchestratorState.NEXT_RUN, EVENT_WAIT_EXTERNAL): OrchestratorState.WAITING_EXTERNAL,
    (OrchestratorState.WAITING_EXTERNAL, EVENT_RESUME): OrchestratorState.TASK_RUN,
}

HAPPY_PATH_EVENTS: tuple[str, ...] = (
    EVENT_MEMORY_RETRIEVE,
    EVENT_TASK_RUN,
    EVENT_SCORE,
    EVENT_FREEZE_BRIEF,
    EVENT_DIAGNOSIS,
    EVENT_COMMIT_REVISIONS,
    EVENT_NEXT_RUN,
    EVENT_DONE,
)


class MissingTransitionEvidence(Exception):
    """Raised when a guarded transition has no envelope or revision evidence."""

    def __init__(self, event: str) -> None:
        self.event = event
        super().__init__(f"missing transition evidence for {event}")


class IllegalTransition(Exception):
    """Raised when ``advance`` requests a transition absent from the guard."""

    def __init__(self, current: OrchestratorState, requested: OrchestratorState | str) -> None:
        self.current = current
        self.requested = requested
        requested_text = requested.value if isinstance(requested, OrchestratorState) else str(requested)
        super().__init__(f"illegal transition from {current.value} to {requested_text}")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _format_dt(value: datetime) -> str:
    return _as_utc(value).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _parse_dt(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return _as_utc(parsed)


def _require_str(value: Any, field: str) -> str:
    if not isinstance(value, str) or value == "":
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _copy_evidence(evidence: Mapping[str, Any]) -> dict[str, Any]:
    copied: dict[str, Any] = {}
    for key, value in evidence.items():
        if isinstance(value, (list, tuple)):
            copied[str(key)] = [str(item) for item in value]
        else:
            copied[str(key)] = value
    return copied


def _payload_hash(payload: Mapping[str, Any] | None) -> str:
    """Content address of the event payload.

    ``sha256`` of canonical JSON (sorted keys, compact separators), the same
    encoding as handoff ``content_address``. An empty payload hashes ``{}``,
    not a stand-in digest.
    """
    body = dict(payload) if payload else {}
    return content_address(body)


_JSON_MAP = "map"
_JSON_LIST = "list"


def _freeze_json(value: Any) -> Any:
    """Convert JSON-like values into hashable, order-stable structures."""
    if isinstance(value, Mapping):
        return (_JSON_MAP, tuple((str(key), _freeze_json(value[key])) for key in sorted(value)))
    if isinstance(value, (list, tuple)):
        return (_JSON_LIST, tuple(_freeze_json(item) for item in value))
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"proposal values must be JSON-compatible, got {type(value).__name__}")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, tuple) and len(value) == 2 and value[0] == _JSON_MAP:
        return {key: _thaw_json(item) for key, item in value[1]}
    if isinstance(value, tuple) and len(value) == 2 and value[0] == _JSON_LIST:
        return [_thaw_json(item) for item in value[1]]
    return value


@dataclass
class NoGainTracker:
    """Consecutive iterations with no new evidence refs and a stable judgment.

    The first observation is the baseline. Each later iteration that adds no
    evidence ref and keeps the same key judgment increments ``consecutive``.
    Two such iterations in a row satisfy the no-gain stop condition.
    """

    consecutive: int = 0
    previous_refs: tuple[str, ...] | None = None
    previous_judgment: str | None = None

    def observe(self, evidence_refs: Iterable[str], key_judgment: str) -> int:
        refs = tuple(_require_str(ref, "evidence_refs") for ref in evidence_refs)
        judgment = key_judgment if isinstance(key_judgment, str) else _require_str(key_judgment, "key_judgment")
        if self.previous_refs is None:
            self.previous_refs = refs
            self.previous_judgment = judgment
            self.consecutive = 0
            return self.consecutive
        no_new = frozenset(refs) <= frozenset(self.previous_refs)
        same = judgment == self.previous_judgment
        self.consecutive = self.consecutive + 1 if no_new and same else 0
        self.previous_refs = refs
        self.previous_judgment = judgment
        return self.consecutive

    def to_dict(self) -> dict[str, Any]:
        return {
            "consecutive": self.consecutive,
            "previous_refs": list(self.previous_refs) if self.previous_refs is not None else None,
            "previous_judgment": self.previous_judgment,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> NoGainTracker:
        raw_refs = data.get("previous_refs")
        refs = None if raw_refs is None else tuple(_require_str(ref, "previous_refs") for ref in raw_refs)
        judgment = data.get("previous_judgment")
        if judgment is not None:
            judgment = _require_str(judgment, "previous_judgment")
        consecutive = data.get("consecutive", 0)
        if isinstance(consecutive, bool) or not isinstance(consecutive, int):
            raise ValueError("consecutive must be an int")
        return cls(consecutive=consecutive, previous_refs=refs, previous_judgment=judgment)


@dataclass(frozen=True)
class IterationResult:
    """Caller-supplied facts for the stop hook. Any one stop flag is enough."""

    terminal: bool = False
    ineligible: bool = False
    budget_exhausted: bool = False
    evidence_refs: tuple[str, ...] = ()
    key_judgment: str = ""
    no_gain_streak: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_refs", tuple(_require_str(ref, "evidence_refs") for ref in self.evidence_refs))
        if not isinstance(self.key_judgment, str):
            raise ValueError("key_judgment must be a string")
        if self.no_gain_streak is not None and (
            isinstance(self.no_gain_streak, bool) or not isinstance(self.no_gain_streak, int)
        ):
            raise ValueError("no_gain_streak must be an int")

    def to_dict(self) -> dict[str, Any]:
        return {
            "terminal": self.terminal,
            "ineligible": self.ineligible,
            "budget_exhausted": self.budget_exhausted,
            "evidence_refs": list(self.evidence_refs),
            "key_judgment": self.key_judgment,
            "no_gain_streak": self.no_gain_streak,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> IterationResult:
        streak = data.get("no_gain_streak")
        return cls(
            terminal=bool(data.get("terminal", False)),
            ineligible=bool(data.get("ineligible", False)),
            budget_exhausted=bool(data.get("budget_exhausted", False)),
            evidence_refs=tuple(data.get("evidence_refs", ())),
            key_judgment=data.get("key_judgment", "") or "",
            no_gain_streak=streak,
        )


@dataclass(frozen=True)
class WaitingCondition:
    """One dossier entry that a future evidence event may satisfy."""

    condition_id: str
    description: str
    evidence_match: str

    def __post_init__(self) -> None:
        _require_str(self.condition_id, "condition_id")
        _require_str(self.description, "description")
        _require_str(self.evidence_match, "evidence_match")

    def to_dict(self) -> dict[str, str]:
        return {
            "condition_id": self.condition_id,
            "description": self.description,
            "evidence_match": self.evidence_match,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> WaitingCondition:
        return cls(
            condition_id=_require_str(data["condition_id"], "condition_id"),
            description=_require_str(data["description"], "description"),
            evidence_match=_require_str(data["evidence_match"], "evidence_match"),
        )

    def matches(self, evidence: Mapping[str, Any]) -> bool:
        if evidence.get("condition_id") == self.condition_id:
            return True
        if evidence.get("evidence_match") == self.evidence_match:
            return True
        return self.evidence_match in evidence.values()


@dataclass(frozen=True)
class StateEvent:
    """Immutable transition record. The log appends these and never rewrites them."""

    state_event_id: str
    from_state: OrchestratorState
    to_state: OrchestratorState
    event: str
    payload_hash: str
    occurred_at: str
    evidence: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        _require_str(self.state_event_id, "state_event_id")
        if not isinstance(self.from_state, OrchestratorState):
            raise TypeError("from_state must be an OrchestratorState")
        if not isinstance(self.to_state, OrchestratorState):
            raise TypeError("to_state must be an OrchestratorState")
        _require_str(self.event, "event")
        _require_str(self.payload_hash, "payload_hash")
        _parse_dt(self.occurred_at)
        if self.evidence is not None:
            object.__setattr__(self, "evidence", _copy_evidence(self.evidence))

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "state_event_id": self.state_event_id,
            "from": self.from_state.value,
            "to": self.to_state.value,
            "event": self.event,
            "payload_hash": self.payload_hash,
            "occurred_at": self.occurred_at,
        }
        if self.evidence:
            payload["evidence"] = _copy_evidence(self.evidence)
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> StateEvent:
        return cls(
            state_event_id=_require_str(data["state_event_id"], "state_event_id"),
            from_state=OrchestratorState(_require_str(data["from"], "from")),
            to_state=OrchestratorState(_require_str(data["to"], "to")),
            event=_require_str(data["event"], "event"),
            payload_hash=_require_str(data["payload_hash"], "payload_hash"),
            occurred_at=_require_str(data["occurred_at"], "occurred_at"),
            evidence=_copy_evidence(data["evidence"]) if isinstance(data.get("evidence"), Mapping) else None,
        )


@dataclass(frozen=True)
class AgentProposal:
    """A trigger or schedule proposal. Recording it does not move the state."""

    proposal_id: str
    kind: str
    agent_role: str
    proposal: tuple[tuple[str, Any], ...]
    proposed_at: str

    def __post_init__(self) -> None:
        _require_str(self.proposal_id, "proposal_id")
        if self.kind not in PROPOSAL_KINDS:
            raise ValueError(f"kind must be one of {sorted(PROPOSAL_KINDS)}")
        if self.agent_role not in AGENT_ROLES:
            raise ValueError(f"agent_role must be one of {sorted(AGENT_ROLES)}")
        _parse_dt(self.proposed_at)

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "kind": self.kind,
            "agent_role": self.agent_role,
            "proposal": _thaw_json(self.proposal),
            "proposed_at": self.proposed_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AgentProposal:
        proposal = data.get("proposal", {})
        if not isinstance(proposal, Mapping):
            raise ValueError("proposal must be a mapping")
        return cls(
            proposal_id=_require_str(data["proposal_id"], "proposal_id"),
            kind=_require_str(data["kind"], "kind"),
            agent_role=_require_str(data["agent_role"], "agent_role"),
            proposal=_freeze_json(dict(proposal)),
            proposed_at=_require_str(data["proposed_at"], "proposed_at"),
        )


@dataclass(frozen=True)
class SeatHold:
    """One held seat: holder, acquisition time, and purpose label."""

    seat_id: str
    holder: str
    acquired_at: str
    purpose: str
    diagnosis: bool = False

    def __post_init__(self) -> None:
        _require_str(self.seat_id, "seat_id")
        _require_str(self.holder, "holder")
        _require_str(self.purpose, "purpose")
        _parse_dt(self.acquired_at)


class NoSeatError(Exception):
    """Raised when a seat cannot be granted. Carries the holders at refusal."""

    def __init__(self, holders: Iterable[SeatHold], *, requested: int, reason: str) -> None:
        self.holders = tuple(holders)
        self.requested = requested
        self.reason = reason
        listing = ", ".join(f"{item.holder}:{item.purpose}" for item in self.holders) or "(none)"
        super().__init__(f"{reason}: requested {requested}, holders [{listing}]")


class SeatLease:
    """Pairs one ``acquire`` with ``release``. Also usable as a context manager."""

    def __init__(self, pool: WorkerSeatPool, hold: SeatHold) -> None:
        self._pool = pool
        self._hold = hold
        self._released = False

    @property
    def hold(self) -> SeatHold:
        return self._hold

    @property
    def released(self) -> bool:
        return self._released

    def release(self) -> None:
        self._pool.release(self)

    def __enter__(self) -> SeatLease:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if not self._released:
            self.release()


class WorkerSeatPool:
    """Deterministic seat ledger. No locks and no waiting.

    Interaction, sub-diagnosis, and explorer work share ``total`` seats.
    ``reserve_diagnosis`` grants a batch only when ``n`` is within the
    diagnosis cap and within the seats free at that moment, counting seats
    already held for diagnosis. Timestamps use the injected ``clock``.
    """

    def __init__(
        self,
        *,
        total: int = WORKER_SEAT_TOTAL,
        diagnosis_limit: int = DIAGNOSIS_SEAT_LIMIT,
        clock: Clock | None = None,
    ) -> None:
        if isinstance(total, bool) or not isinstance(total, int) or total < 1:
            raise ValueError("total must be a positive int")
        if (
            isinstance(diagnosis_limit, bool)
            or not isinstance(diagnosis_limit, int)
            or diagnosis_limit < 0
        ):
            raise ValueError("diagnosis_limit must be a non-negative int")
        if diagnosis_limit > total:
            raise ValueError("diagnosis_limit cannot exceed total")
        self._total = total
        self._diagnosis_limit = diagnosis_limit
        self._clock = clock or _utc_now
        self._held: list[SeatHold] = []
        self._seq = 0

    @property
    def total(self) -> int:
        return self._total

    @property
    def diagnosis_limit(self) -> int:
        return self._diagnosis_limit

    @property
    def free(self) -> int:
        return self._total - len(self._held)

    def acquire(self, holder: str, purpose: str) -> SeatLease:
        """Take one seat, or raise ``NoSeatError`` with the current holders."""
        holder_name = _require_str(holder, "holder")
        purpose_label = _require_str(purpose, "purpose")
        if len(self._held) >= self._total:
            raise NoSeatError(self.held_snapshot(), requested=1, reason="pool_full")
        return self._grant(holder_name, purpose_label, diagnosis=False)

    def reserve_diagnosis(self, n: int, holder: str, purpose: str) -> tuple[SeatLease, ...]:
        """Grant ``n`` diagnosis seats, or raise ``NoSeatError`` without granting any."""
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError("n must be a positive int")
        holder_name = _require_str(holder, "holder")
        purpose_label = _require_str(purpose, "purpose")
        diagnosis_held = sum(1 for item in self._held if item.diagnosis)
        if n > self._diagnosis_limit or diagnosis_held + n > self._diagnosis_limit:
            raise NoSeatError(self.held_snapshot(), requested=n, reason="diagnosis_batch")
        if n > self.free:
            raise NoSeatError(self.held_snapshot(), requested=n, reason="insufficient_free")
        return tuple(self._grant(holder_name, purpose_label, diagnosis=True) for _ in range(n))

    def release(self, lease: SeatLease) -> None:
        """Return a seat granted by this pool. A second release is unpaired."""
        if lease._pool is not self:
            raise ValueError("lease does not belong to this pool")
        if lease._released:
            raise ValueError("seat release is not paired with an active acquire")
        index = next(
            (pos for pos, item in enumerate(self._held) if item.seat_id == lease._hold.seat_id),
            None,
        )
        if index is None:
            raise ValueError("seat release is not paired with an active acquire")
        lease._released = True
        del self._held[index]

    def held_snapshot(self) -> tuple[SeatHold, ...]:
        """Seats still held, in acquisition order, for audit and leak checks."""
        return tuple(self._held)

    def _grant(self, holder: str, purpose: str, *, diagnosis: bool) -> SeatLease:
        self._seq += 1
        hold = SeatHold(
            seat_id=f"seat-{self._seq}",
            holder=holder,
            acquired_at=_format_dt(self._clock()),
            purpose=purpose,
            diagnosis=diagnosis,
        )
        self._held.append(hold)
        return SeatLease(self, hold)


def _coerce_iteration(value: IterationResult | Mapping[str, Any]) -> IterationResult:
    if isinstance(value, IterationResult):
        return value
    if isinstance(value, Mapping):
        return IterationResult.from_dict(value)
    raise TypeError("iteration_result must be an IterationResult or a mapping")


def _coerce_conditions(payload: Mapping[str, Any]) -> tuple[WaitingCondition, ...]:
    raw = payload.get("waiting_conditions")
    if not isinstance(raw, list) or not raw:
        raise ValueError("waiting_conditions must be a non-empty list")
    return tuple(WaitingCondition.from_dict(item) for item in raw)


class EvaluationOrchestrator:
    """In-process evaluation loop plus a JSON state log.

    ``advance`` is the only transition API. Agent proposals are stored beside
    the log and are rejected if passed to ``advance``. ``seats`` is the shared
    five-worker pool; when omitted, the orchestrator builds one on its clock.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        evaluation_id: str | None = None,
        clock: Clock | None = None,
        event_id_factory: IdFactory | None = None,
        write_hooks: Mapping[str, Callable[..., Any]] | None = None,
        seats: WorkerSeatPool | None = None,
    ) -> None:
        self._path = _resolve_store_path(path) if path is not None else None
        self._clock = clock or _utc_now
        self._ids = event_id_factory or (lambda: uuid.uuid4().hex)
        self._write_hooks = dict(write_hooks) if write_hooks else {}
        self._seats = (
            seats
            if seats is not None
            else WorkerSeatPool(
                total=WORKER_SEAT_TOTAL,
                diagnosis_limit=DIAGNOSIS_SEAT_LIMIT,
                clock=self._clock,
            )
        )
        self._lock = threading.Lock()
        self._evaluation_id = _require_str(evaluation_id, "evaluation_id") if evaluation_id else uuid.uuid4().hex
        self._state = OrchestratorState.CREATED
        self._events: list[StateEvent] = []
        self._proposals: list[AgentProposal] = []
        self._waiting_conditions: list[WaitingCondition] = []
        self._satisfied_conditions: list[str] = []
        if self._path is not None and self._path.is_file():
            self._load()

    @property
    def evaluation_id(self) -> str:
        return self._evaluation_id

    @property
    def state(self) -> OrchestratorState:
        return self._state

    @property
    def events(self) -> tuple[StateEvent, ...]:
        return tuple(self._events)

    @property
    def proposals(self) -> tuple[AgentProposal, ...]:
        return tuple(self._proposals)

    @property
    def waiting_conditions(self) -> tuple[WaitingCondition, ...]:
        return tuple(self._waiting_conditions)

    @property
    def satisfied_conditions(self) -> tuple[str, ...]:
        return tuple(self._satisfied_conditions)

    @property
    def seats(self) -> WorkerSeatPool:
        return self._seats

    def advance(
        self,
        event: Any,
        payload: Mapping[str, Any] | None = None,
        *,
        brief_envelope_id: str | None = None,
        revision_ids: Iterable[str] | None = None,
        outcome_envelope_id: str | None = None,
        diagnosis_batch: int | None = None,
    ) -> StateEvent:
        """Move one step. Illegal events, including any proposal, raise.

        Entering ``FREEZE_BRIEF`` requires ``brief_envelope_id``. Entering
        ``COMMIT_REVISIONS`` requires ``revision_ids`` or
        ``outcome_envelope_id``. When ``write_hooks`` supplies
        ``on_brief_envelope`` or ``on_revision_commit``, the hook runs before
        the transition and its returned id is stored on the event ``evidence``
        field beside ``payload_hash``. ``payload_hash`` is the content
        address of the payload. Missing evidence raises
        ``MissingTransitionEvidence`` and does not move state. Entering
        ``DIAGNOSIS`` with ``diagnosis_batch`` above the diagnosis seat limit
        raises ``NoSeatError`` and does not move state. That check sits beside
        the transition-evidence guard and does not replace it.
        """
        with self._lock:
            self._reject_if_proposal(event)
            if not isinstance(event, str) or event == "":
                raise IllegalTransition(self._state, str(event))
            requested: OrchestratorState | str = _EVENT_TARGET.get(event, event)
            target = TRANSITION_GUARD.get((self._state, event))
            if target is None:
                raise IllegalTransition(self._state, requested)
            body = dict(payload) if payload else {}
            evidence = self._transition_evidence(
                target,
                body,
                brief_envelope_id=brief_envelope_id,
                revision_ids=revision_ids,
                outcome_envelope_id=outcome_envelope_id,
            )
            self._guard_diagnosis_batch(target, body, diagnosis_batch)
            if target is OrchestratorState.WAITING_EXTERNAL:
                conditions = _coerce_conditions(body)
            else:
                conditions = None
            if self._state is OrchestratorState.WAITING_EXTERNAL:
                self._require_resume(body)
            record = StateEvent(
                state_event_id=self._ids(),
                from_state=self._state,
                to_state=target,
                event=event,
                payload_hash=_payload_hash(body),
                occurred_at=_format_dt(self._clock()),
                evidence=evidence or None,
            )
            self._events.append(record)
            self._state = target
            if conditions is not None:
                self._waiting_conditions = list(conditions)
            self._persist()
            return record

    def submit_trigger_proposal(self, agent_role: str, proposal: Mapping[str, Any]) -> AgentProposal:
        return self._submit(PROPOSAL_TRIGGER, agent_role, proposal)

    def submit_schedule_proposal(self, agent_role: str, proposal: Mapping[str, Any]) -> AgentProposal:
        return self._submit(PROPOSAL_SCHEDULE, agent_role, proposal)

    def should_stop(
        self,
        iteration_result: IterationResult | Mapping[str, Any],
        no_gain: NoGainTracker | int | None = None,
    ) -> bool:
        """Return whether the loop is STOPPED.

        Any one of the four conditions is sufficient: the iteration or the
        orchestrator is already terminal, replay/continuation is ineligible,
        the parent budget/time/branch cap is exhausted, or two consecutive
        iterations added no evidence ref while the key judgment stayed the
        same. When the decision is stop and ``stop`` is a legal move, the
        orchestrator enters STOPPED.
        """
        result = _coerce_iteration(iteration_result)
        streak = self._no_gain_streak(result, no_gain)
        stop = (
            result.terminal
            or self._state in TERMINAL_STATES
            or result.ineligible
            or result.budget_exhausted
            or streak >= NO_GAIN_LIMIT
        )
        if stop and (self._state, EVENT_STOP) in TRANSITION_GUARD:
            reason = _stop_reason(result, streak)
            self.advance(EVENT_STOP, {"reason": reason})
        return stop

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluation_id": self._evaluation_id,
            "state": self._state.value,
            "events": [event.to_dict() for event in self._events],
            "proposals": [proposal.to_dict() for proposal in self._proposals],
            "waiting_conditions": [condition.to_dict() for condition in self._waiting_conditions],
            "satisfied_conditions": list(self._satisfied_conditions),
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        path: str | Path | None = None,
        clock: Clock | None = None,
        event_id_factory: IdFactory | None = None,
        seats: WorkerSeatPool | None = None,
    ) -> EvaluationOrchestrator:
        orchestrator = cls(
            path=None,
            evaluation_id=_require_str(data["evaluation_id"], "evaluation_id"),
            clock=clock,
            event_id_factory=event_id_factory,
            seats=seats,
        )
        orchestrator._state = OrchestratorState(_require_str(data["state"], "state"))
        orchestrator._events = [StateEvent.from_dict(item) for item in data.get("events", [])]
        orchestrator._proposals = [AgentProposal.from_dict(item) for item in data.get("proposals", [])]
        orchestrator._waiting_conditions = [
            WaitingCondition.from_dict(item) for item in data.get("waiting_conditions", [])
        ]
        orchestrator._satisfied_conditions = [
            _require_str(item, "satisfied_conditions") for item in data.get("satisfied_conditions", [])
        ]
        if path is not None:
            orchestrator._path = _resolve_store_path(path)
            orchestrator._persist()
        return orchestrator

    def _submit(self, kind: str, agent_role: str, proposal: Mapping[str, Any]) -> AgentProposal:
        if agent_role not in AGENT_ROLES:
            raise ValueError(f"agent_role must be one of {sorted(AGENT_ROLES)}")
        if not isinstance(proposal, Mapping):
            raise TypeError("proposal must be a mapping")
        with self._lock:
            record = AgentProposal(
                proposal_id=self._ids(),
                kind=kind,
                agent_role=agent_role,
                proposal=_freeze_json(dict(proposal)),
                proposed_at=_format_dt(self._clock()),
            )
            self._proposals.append(record)
            self._persist()
            return record

    def _transition_evidence(
        self,
        target: OrchestratorState,
        body: Mapping[str, Any],
        *,
        brief_envelope_id: str | None,
        revision_ids: Iterable[str] | None,
        outcome_envelope_id: str | None,
    ) -> dict[str, Any]:
        if target is OrchestratorState.FREEZE_BRIEF:
            envelope = brief_envelope_id or body.get("brief_envelope_id")
            hook = self._write_hooks.get("on_brief_envelope")
            if hook is not None:
                returned = hook(body)
                if isinstance(returned, str) and returned:
                    envelope = returned
            if not isinstance(envelope, str) or envelope == "":
                raise MissingTransitionEvidence(EVENT_FREEZE_BRIEF)
            return {"brief_envelope_id": envelope}
        if target is OrchestratorState.COMMIT_REVISIONS:
            revisions = list(revision_ids) if revision_ids is not None else body.get("revision_ids")
            outcome = outcome_envelope_id or body.get("outcome_envelope_id")
            hook = self._write_hooks.get("on_revision_commit")
            if hook is not None:
                returned = hook(body)
                if isinstance(returned, (list, tuple)):
                    revisions = list(returned)
                elif isinstance(returned, str) and returned:
                    outcome = returned
            rev_list: list[str] = []
            if isinstance(revisions, (list, tuple)):
                rev_list = [item for item in revisions if isinstance(item, str) and item]
            outcome_id = outcome if isinstance(outcome, str) and outcome else ""
            if not rev_list and not outcome_id:
                raise MissingTransitionEvidence(EVENT_COMMIT_REVISIONS)
            evidence: dict[str, Any] = {}
            if rev_list:
                evidence["revision_ids"] = rev_list
            if outcome_id:
                evidence["outcome_envelope_id"] = outcome_id
            return evidence
        return {}

    def _guard_diagnosis_batch(
        self,
        target: OrchestratorState,
        body: Mapping[str, Any],
        diagnosis_batch: int | None,
    ) -> None:
        """Refuse a diagnosis entry whose declared batch exceeds the seat cap.

        Absent batch means the caller did not declare one. Evidence requirements
        stay in ``_transition_evidence`` and still run before this check.
        """
        if target is not OrchestratorState.DIAGNOSIS:
            return
        raw = diagnosis_batch if diagnosis_batch is not None else body.get("diagnosis_batch")
        if raw is None:
            return
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise ValueError("diagnosis_batch must be an int")
        if raw < 0:
            raise ValueError("diagnosis_batch must be >= 0")
        if raw > self._seats.diagnosis_limit:
            raise NoSeatError(
                self._seats.held_snapshot(),
                requested=raw,
                reason="diagnosis_batch",
            )

    def _reject_if_proposal(self, event: Any) -> None:
        if isinstance(event, AgentProposal):
            raise IllegalTransition(self._state, event.kind)
        if isinstance(event, Mapping) and (
            "proposal" in event or event.get("kind") in PROPOSAL_KINDS or "agent_role" in event
        ):
            requested = event.get("kind", "proposal")
            raise IllegalTransition(self._state, str(requested))
        if isinstance(event, str) and (
            event in _PROPOSAL_EVENTS
            or event.startswith("proposal:")
            or any(item.proposal_id == event for item in self._proposals)
        ):
            raise IllegalTransition(self._state, event)

    def _require_resume(self, payload: Mapping[str, Any]) -> None:
        evidence = payload.get("evidence_event", payload)
        if not isinstance(evidence, Mapping):
            raise ValueError("evidence_event must be a mapping")
        matched = [item for item in self._waiting_conditions if item.matches(evidence)]
        if not matched:
            raise ValueError("evidence event does not satisfy any waiting condition")
        for item in matched:
            if item.condition_id not in self._satisfied_conditions:
                self._satisfied_conditions.append(item.condition_id)

    def _no_gain_streak(self, result: IterationResult, no_gain: NoGainTracker | int | None) -> int:
        if isinstance(no_gain, NoGainTracker):
            return no_gain.observe(result.evidence_refs, result.key_judgment)
        if isinstance(no_gain, int) and not isinstance(no_gain, bool):
            return no_gain
        if result.no_gain_streak is not None:
            return result.no_gain_streak
        return 0

    def _persist(self) -> None:
        if self._path is None:
            return
        text = json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n"
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(self._path)

    def _load(self) -> None:
        assert self._path is not None
        payload = json.loads(self._path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("evaluation orchestrator store must be a JSON object")
        restored = EvaluationOrchestrator.from_dict(payload, clock=self._clock, event_id_factory=self._ids)
        self._evaluation_id = restored.evaluation_id
        self._state = restored.state
        self._events = list(restored.events)
        self._proposals = list(restored.proposals)
        self._waiting_conditions = list(restored.waiting_conditions)
        self._satisfied_conditions = list(restored.satisfied_conditions)


def _stop_reason(result: IterationResult, streak: int) -> str:
    if result.terminal:
        return "terminal"
    if result.ineligible:
        return "ineligible"
    if result.budget_exhausted:
        return "budget_exhausted"
    if streak >= NO_GAIN_LIMIT:
        return "no_gain"
    return "terminal_state"


def _resolve_store_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.suffix == ".json":
        candidate.parent.mkdir(parents=True, exist_ok=True)
        return candidate
    candidate.mkdir(parents=True, exist_ok=True)
    return candidate / STORE_FILENAME

# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Immutable content-addressed handoff envelopes (ADR-0012, Q93-A).

Idempotency key is ``iteration_id + handoff_kind + input_manifest_hash``.
The same key must yield the same result: a repeat whose ``envelope_id``
matches the stored delivery returns that result and writes nothing new.
The same key with a different body raises ``EnvelopeConflict``; create a
new iteration instead. ``envelope_id`` is ``sha256`` of the canonical
JSON document. ``input_manifest_hash`` is supplied by the caller;
``verify_manifest`` checks it against the canonical JSON hash of the body.

Envelopes carry no credential fields. Names are matched case-insensitively
as substrings of token, secret, key, credential, password, bearer,
authorization, cookie, apikey, and session. Established fields that end
with ``_hash`` are an exact allowlist: ``input_manifest_hash``,
``content_hash``, ``manifest_hash``, ``input_brief_hash``, and
``memory_task_state_hash``. Any other name containing a forbidden
substring is rejected, including names that merely contain ``hash``.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any

STORE_FILENAME = "handoff-envelopes.json"

HANDOFF_KIND_BRIEF = "brief"
HANDOFF_KIND_OUTCOME = "outcome"
HANDOFF_KINDS = frozenset({HANDOFF_KIND_BRIEF, HANDOFF_KIND_OUTCOME})

_FORBIDDEN_NAME_PARTS = (
    "token",
    "secret",
    "key",
    "credential",
    "password",
    "bearer",
    "authorization",
    "cookie",
    "apikey",
    "session",
)

# Exact names, not a "contains hash" exemption. Each entry ends with ``_hash``.
_ALLOWED_HASH_FIELDS = frozenset(
    {
        "content_hash",
        "input_brief_hash",
        "input_manifest_hash",
        "manifest_hash",
        "memory_task_state_hash",
    }
)

Clock = Callable[[], datetime]


class EnvelopeConflict(ValueError):
    """Same idempotency key was reused with a different envelope body."""


class ManifestMismatch(ValueError):
    """``input_manifest_hash`` does not match the canonical hash of the body."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _format_dt(value: datetime) -> str:
    return _as_utc(value).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _parse_dt(value: str) -> datetime:
    return _as_utc(datetime.fromisoformat(value.replace("Z", "+00:00")))


def _require_str(value: Any, field: str) -> str:
    if not isinstance(value, str) or value == "":
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _canonical_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _is_credential_name(name: str) -> bool:
    """Return whether ``name`` looks like a credential field.

    Matching is a case-insensitive substring against
    ``_FORBIDDEN_NAME_PARTS``. ``_ALLOWED_HASH_FIELDS`` is an exact
    allowlist of established ``*_hash`` names. A name that merely
    contains ``hash`` is not exempt.
    """
    lowered = name.lower()
    if lowered in _ALLOWED_HASH_FIELDS:
        return False
    return any(part in lowered for part in _FORBIDDEN_NAME_PARTS)


def _reject_credential_fields(value: Any, field: str = "envelope") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{field} keys must be strings")
            if _is_credential_name(key):
                raise ValueError(f"{field} field {key!r} looks like a credential")
            _reject_credential_fields(item, f"{field}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_credential_fields(item, f"{field}[{index}]")


def _freeze_strs(values: Any, field: str) -> tuple[str, ...]:
    if isinstance(values, str) or not isinstance(values, (list, tuple)):
        raise ValueError(f"{field} must be a list of strings")
    return tuple(_require_str(item, field) for item in values)


def _freeze_map(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    _reject_credential_fields(value, field)
    return MappingProxyType({_require_str(key, field): _freeze_value(item, f"{field}.{key}") for key, item in value.items()})


def _freeze_value(value: Any, field: str) -> Any:
    if isinstance(value, bool) or isinstance(value, str) or value is None:
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, Mapping):
        return _freeze_map(value, field)
    if isinstance(value, (list, tuple)) and not isinstance(value, str):
        return tuple(_freeze_value(item, field) for item in value)
    raise ValueError(f"{field} must be JSON-compatible")


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def content_address(document: Mapping[str, Any]) -> str:
    """``sha256`` of canonical JSON. Key order does not change the id."""
    return hashlib.sha256(_canonical_dumps(dict(document)).encode("utf-8")).hexdigest()


def idempotency_key(iteration_id: str, handoff_kind: str, input_manifest_hash: str) -> str:
    return f"{iteration_id}\x1f{handoff_kind}\x1f{input_manifest_hash}"


def verify_manifest(envelope: HandoffEnvelope, body: Mapping[str, Any]) -> None:
    """Check ``input_manifest_hash`` against the canonical JSON hash of ``body``.

    The digest is ``sha256`` of canonical JSON (sorted keys, compact
    separators), the same encoding as ``content_address``. A match returns.
    A mismatch raises ``ManifestMismatch``. Callers that supply
    ``input_manifest_hash`` can self-check immediately after construction.
    The hash is still caller-supplied; delivery does not recompute it.
    """
    if not isinstance(envelope, HandoffEnvelope):
        raise TypeError("envelope must be a HandoffEnvelope")
    if not isinstance(body, Mapping):
        raise ValueError("body must be a mapping")
    digest = content_address(_thaw(body))
    if digest != envelope.input_manifest_hash:
        raise ManifestMismatch(
            "input_manifest_hash does not match the canonical JSON hash of the body"
        )


@dataclass(frozen=True)
class DiagnosisBrief:
    """Frozen brief handed from scoring to diagnosis. No credential fields."""

    evaluation_id: str
    task_id: str
    source_run_id: str
    iteration_id: str
    memory_task_state_hash: str
    selected_refs: tuple[str, ...]
    snapshot_watermarks: Mapping[str, Any]
    policy_versions: Mapping[str, Any]
    sub_budgets: Mapping[str, Any]
    input_manifest_hash: str

    def __post_init__(self) -> None:
        _require_str(self.evaluation_id, "evaluation_id")
        _require_str(self.task_id, "task_id")
        _require_str(self.source_run_id, "source_run_id")
        _require_str(self.iteration_id, "iteration_id")
        _require_str(self.memory_task_state_hash, "memory_task_state_hash")
        _require_str(self.input_manifest_hash, "input_manifest_hash")
        if not isinstance(self.selected_refs, tuple):
            raise TypeError("selected_refs must be a tuple of strings")
        for ref in self.selected_refs:
            _require_str(ref, "selected_refs")
        _reject_credential_fields(self.to_dict(), "DiagnosisBrief")

    @property
    def handoff_kind(self) -> str:
        return HANDOFF_KIND_BRIEF

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluation_id": self.evaluation_id,
            "task_id": self.task_id,
            "source_run_id": self.source_run_id,
            "iteration_id": self.iteration_id,
            "memory_task_state_hash": self.memory_task_state_hash,
            "selected_refs": list(self.selected_refs),
            "snapshot_watermarks": _thaw(self.snapshot_watermarks),
            "policy_versions": _thaw(self.policy_versions),
            "sub_budgets": _thaw(self.sub_budgets),
            "input_manifest_hash": self.input_manifest_hash,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DiagnosisBrief:
        _reject_credential_fields(data, "DiagnosisBrief")
        return cls(
            evaluation_id=_require_str(data["evaluation_id"], "evaluation_id"),
            task_id=_require_str(data["task_id"], "task_id"),
            source_run_id=_require_str(data["source_run_id"], "source_run_id"),
            iteration_id=_require_str(data["iteration_id"], "iteration_id"),
            memory_task_state_hash=_require_str(data["memory_task_state_hash"], "memory_task_state_hash"),
            selected_refs=_freeze_strs(data.get("selected_refs", []), "selected_refs"),
            snapshot_watermarks=_freeze_map(data.get("snapshot_watermarks", {}), "snapshot_watermarks"),
            policy_versions=_freeze_map(data.get("policy_versions", {}), "policy_versions"),
            sub_budgets=_freeze_map(data.get("sub_budgets", {}), "sub_budgets"),
            input_manifest_hash=_require_str(data["input_manifest_hash"], "input_manifest_hash"),
        )


@dataclass(frozen=True)
class DiagnosisOutcome:
    """Frozen diagnosis result handed back toward the next memory run."""

    diagnosis_run_id: str
    input_brief_hash: str
    fork_revision_refs: tuple[str, ...]
    bridge_judgment_refs: tuple[str, ...]
    open_questions: tuple[str, ...]
    waiting_conditions: tuple[str, ...]
    budget_spent: Mapping[str, Any]
    versions: Mapping[str, Any]

    def __post_init__(self) -> None:
        _require_str(self.diagnosis_run_id, "diagnosis_run_id")
        _require_str(self.input_brief_hash, "input_brief_hash")
        for field_name, values in (
            ("fork_revision_refs", self.fork_revision_refs),
            ("bridge_judgment_refs", self.bridge_judgment_refs),
            ("open_questions", self.open_questions),
            ("waiting_conditions", self.waiting_conditions),
        ):
            if not isinstance(values, tuple):
                raise TypeError(f"{field_name} must be a tuple of strings")
            for item in values:
                _require_str(item, field_name)
        _reject_credential_fields(self.to_dict(), "DiagnosisOutcome")

    @property
    def handoff_kind(self) -> str:
        return HANDOFF_KIND_OUTCOME

    def to_dict(self) -> dict[str, Any]:
        return {
            "diagnosis_run_id": self.diagnosis_run_id,
            "input_brief_hash": self.input_brief_hash,
            "fork_revision_refs": list(self.fork_revision_refs),
            "bridge_judgment_refs": list(self.bridge_judgment_refs),
            "open_questions": list(self.open_questions),
            "waiting_conditions": list(self.waiting_conditions),
            "budget_spent": _thaw(self.budget_spent),
            "versions": _thaw(self.versions),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DiagnosisOutcome:
        _reject_credential_fields(data, "DiagnosisOutcome")
        return cls(
            diagnosis_run_id=_require_str(data["diagnosis_run_id"], "diagnosis_run_id"),
            input_brief_hash=_require_str(data["input_brief_hash"], "input_brief_hash"),
            fork_revision_refs=_freeze_strs(data.get("fork_revision_refs", []), "fork_revision_refs"),
            bridge_judgment_refs=_freeze_strs(data.get("bridge_judgment_refs", []), "bridge_judgment_refs"),
            open_questions=_freeze_strs(data.get("open_questions", []), "open_questions"),
            waiting_conditions=_freeze_strs(data.get("waiting_conditions", []), "waiting_conditions"),
            budget_spent=_freeze_map(data.get("budget_spent", {}), "budget_spent"),
            versions=_freeze_map(data.get("versions", {}), "versions"),
        )


@dataclass(frozen=True)
class HandoffEnvelope:
    """Content-addressed envelope. ``envelope_id`` is derived, not chosen."""

    envelope_id: str
    handoff_kind: str
    iteration_id: str
    input_manifest_hash: str
    body: Mapping[str, Any]
    delivered_at: str | None = None

    def __post_init__(self) -> None:
        if self.handoff_kind not in HANDOFF_KINDS:
            raise ValueError(f"handoff_kind must be one of {sorted(HANDOFF_KINDS)}")
        _require_str(self.iteration_id, "iteration_id")
        _require_str(self.input_manifest_hash, "input_manifest_hash")
        _reject_credential_fields(self.to_dict(), "HandoffEnvelope")
        expected = content_address(self.canonical_body())
        if self.envelope_id != expected:
            raise ValueError("envelope_id does not match the content address")
        if self.delivered_at is not None:
            _parse_dt(self.delivered_at)

    def canonical_body(self) -> dict[str, Any]:
        return {
            "handoff_kind": self.handoff_kind,
            "iteration_id": self.iteration_id,
            "input_manifest_hash": self.input_manifest_hash,
            "body": _thaw(self.body),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self.canonical_body()
        payload["envelope_id"] = self.envelope_id
        if self.delivered_at is not None:
            payload["delivered_at"] = self.delivered_at
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> HandoffEnvelope:
        _reject_credential_fields(data, "HandoffEnvelope")
        kind = _require_str(data["handoff_kind"], "handoff_kind")
        iteration_id = _require_str(data["iteration_id"], "iteration_id")
        manifest = _require_str(data["input_manifest_hash"], "input_manifest_hash")
        body = data.get("body", {})
        if not isinstance(body, Mapping):
            raise ValueError("body must be a mapping")
        document = {
            "handoff_kind": kind,
            "iteration_id": iteration_id,
            "input_manifest_hash": manifest,
            "body": dict(body),
        }
        envelope_id = data.get("envelope_id") or content_address(document)
        delivered_at = data.get("delivered_at")
        return cls(
            envelope_id=_require_str(envelope_id, "envelope_id"),
            handoff_kind=kind,
            iteration_id=iteration_id,
            input_manifest_hash=manifest,
            body=_freeze_map(body, "body"),
            delivered_at=delivered_at,
        )

    @classmethod
    def from_brief(cls, brief: DiagnosisBrief, *, delivered_at: str | None = None) -> HandoffEnvelope:
        return cls.from_dict(
            {
                "handoff_kind": HANDOFF_KIND_BRIEF,
                "iteration_id": brief.iteration_id,
                "input_manifest_hash": brief.input_manifest_hash,
                "body": brief.to_dict(),
                "delivered_at": delivered_at,
            }
        )

    @classmethod
    def from_outcome(
        cls,
        outcome: DiagnosisOutcome,
        *,
        iteration_id: str,
        input_manifest_hash: str,
        delivered_at: str | None = None,
    ) -> HandoffEnvelope:
        return cls.from_dict(
            {
                "handoff_kind": HANDOFF_KIND_OUTCOME,
                "iteration_id": iteration_id,
                "input_manifest_hash": input_manifest_hash,
                "body": outcome.to_dict(),
                "delivered_at": delivered_at,
            }
        )


@dataclass(frozen=True)
class DeliveryResult:
    """First successful delivery. Repeats return this same value."""

    envelope_id: str
    idempotency_key: str
    handoff_kind: str
    iteration_id: str
    input_manifest_hash: str
    delivered_at: str
    envelope: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "envelope_id": self.envelope_id,
            "idempotency_key": self.idempotency_key,
            "handoff_kind": self.handoff_kind,
            "iteration_id": self.iteration_id,
            "input_manifest_hash": self.input_manifest_hash,
            "delivered_at": self.delivered_at,
            "envelope": _thaw(self.envelope),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DeliveryResult:
        envelope = data["envelope"]
        if not isinstance(envelope, Mapping):
            raise ValueError("envelope must be a mapping")
        return cls(
            envelope_id=_require_str(data["envelope_id"], "envelope_id"),
            idempotency_key=_require_str(data["idempotency_key"], "idempotency_key"),
            handoff_kind=_require_str(data["handoff_kind"], "handoff_kind"),
            iteration_id=_require_str(data["iteration_id"], "iteration_id"),
            input_manifest_hash=_require_str(data["input_manifest_hash"], "input_manifest_hash"),
            delivered_at=_require_str(data["delivered_at"], "delivered_at"),
            envelope=_freeze_map(envelope, "envelope"),
        )


@dataclass(frozen=True)
class AckRecord:
    checkpoint_id: str
    acked_at: str
    envelope_id: str | None = None

    def __post_init__(self) -> None:
        _require_str(self.checkpoint_id, "checkpoint_id")
        _parse_dt(self.acked_at)
        if self.envelope_id is not None:
            _require_str(self.envelope_id, "envelope_id")

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "acked_at": self.acked_at,
            "envelope_id": self.envelope_id,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AckRecord:
        return cls(
            checkpoint_id=_require_str(data["checkpoint_id"], "checkpoint_id"),
            acked_at=_require_str(data["acked_at"], "acked_at"),
            envelope_id=data.get("envelope_id"),
        )


class HandoffEnvelopeService:
    """In-process delivery log.

    The same idempotency key returns the original delivery when
    ``envelope_id`` matches. A different body under that key raises
    ``EnvelopeConflict``.
    """

    def __init__(self, path: str | Path | None = None, *, clock: Clock | None = None) -> None:
        self._path = _resolve_store_path(path) if path is not None else None
        self._clock = clock or _utc_now
        self._lock = threading.Lock()
        self._deliveries: list[DeliveryResult] = []
        self._by_key: dict[str, DeliveryResult] = {}
        self._by_id: dict[str, DeliveryResult] = {}
        self._acks: dict[str, AckRecord] = {}
        if self._path is not None and self._path.is_file():
            self._load()

    @property
    def deliveries(self) -> tuple[DeliveryResult, ...]:
        return tuple(self._deliveries)

    @property
    def acks(self) -> tuple[AckRecord, ...]:
        return tuple(self._acks[key] for key in sorted(self._acks))

    def deliver(
        self,
        envelope: HandoffEnvelope | DiagnosisBrief | DiagnosisOutcome | Mapping[str, Any],
        *,
        iteration_id: str | None = None,
        input_manifest_hash: str | None = None,
    ) -> DeliveryResult:
        """Deliver once per idempotency key.

        A repeat with the same ``envelope_id`` returns the first result.
        A repeat with a different body raises ``EnvelopeConflict``.
        """
        with self._lock:
            sealed = self._seal(envelope, iteration_id=iteration_id, input_manifest_hash=input_manifest_hash)
            key = idempotency_key(sealed.iteration_id, sealed.handoff_kind, sealed.input_manifest_hash)
            existing = self._by_key.get(key)
            if existing is not None:
                if existing.envelope_id != sealed.envelope_id:
                    raise EnvelopeConflict(
                        "same idempotency key with a different body "
                        f"(stored envelope_id {existing.envelope_id}, "
                        f"new envelope_id {sealed.envelope_id}); "
                        "create a new iteration"
                    )
                return existing
            delivered_at = sealed.delivered_at or _format_dt(self._clock())
            stored = HandoffEnvelope(
                envelope_id=sealed.envelope_id,
                handoff_kind=sealed.handoff_kind,
                iteration_id=sealed.iteration_id,
                input_manifest_hash=sealed.input_manifest_hash,
                body=sealed.body,
                delivered_at=delivered_at,
            )
            result = DeliveryResult(
                envelope_id=stored.envelope_id,
                idempotency_key=key,
                handoff_kind=stored.handoff_kind,
                iteration_id=stored.iteration_id,
                input_manifest_hash=stored.input_manifest_hash,
                delivered_at=delivered_at,
                envelope=_freeze_map(stored.to_dict(), "envelope"),
            )
            self._deliveries.append(result)
            self._by_key[key] = result
            self._by_id[result.envelope_id] = result
            self._persist()
            return result

    def ack(self, checkpoint_id: str) -> AckRecord:
        """Record a checkpoint. Repeating the same id returns the first ack."""
        with self._lock:
            checkpoint_id = _require_str(checkpoint_id, "checkpoint_id")
            existing = self._acks.get(checkpoint_id)
            if existing is not None:
                return existing
            linked = self._by_id.get(checkpoint_id)
            record = AckRecord(
                checkpoint_id=checkpoint_id,
                acked_at=_format_dt(self._clock()),
                envelope_id=linked.envelope_id if linked is not None else None,
            )
            self._acks[checkpoint_id] = record
            self._persist()
            return record

    def to_dict(self) -> dict[str, Any]:
        return {
            "deliveries": [item.to_dict() for item in self._deliveries],
            "acks": [self._acks[key].to_dict() for key in sorted(self._acks)],
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        path: str | Path | None = None,
        clock: Clock | None = None,
    ) -> HandoffEnvelopeService:
        service = cls(path=None, clock=clock)
        for raw in data.get("deliveries", []):
            result = DeliveryResult.from_dict(raw)
            service._deliveries.append(result)
            service._by_key[result.idempotency_key] = result
            service._by_id[result.envelope_id] = result
        for raw in data.get("acks", []):
            ack = AckRecord.from_dict(raw)
            service._acks[ack.checkpoint_id] = ack
        if path is not None:
            service._path = _resolve_store_path(path)
            service._persist()
        return service

    def _seal(
        self,
        envelope: HandoffEnvelope | DiagnosisBrief | DiagnosisOutcome | Mapping[str, Any],
        *,
        iteration_id: str | None,
        input_manifest_hash: str | None,
    ) -> HandoffEnvelope:
        if isinstance(envelope, HandoffEnvelope):
            return envelope
        if isinstance(envelope, DiagnosisBrief):
            return HandoffEnvelope.from_brief(envelope)
        if isinstance(envelope, DiagnosisOutcome):
            return HandoffEnvelope.from_outcome(
                envelope,
                iteration_id=_require_str(iteration_id, "iteration_id"),
                input_manifest_hash=_require_str(input_manifest_hash, "input_manifest_hash"),
            )
        if isinstance(envelope, Mapping):
            _reject_credential_fields(envelope, "envelope")
            return HandoffEnvelope.from_dict(envelope)
        raise TypeError("envelope must be a HandoffEnvelope, DiagnosisBrief, DiagnosisOutcome, or mapping")

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
            raise ValueError("handoff envelope store must be a JSON object")
        restored = HandoffEnvelopeService.from_dict(payload, clock=self._clock)
        self._deliveries = list(restored.deliveries)
        self._by_key = {item.idempotency_key: item for item in self._deliveries}
        self._by_id = {item.envelope_id: item for item in self._deliveries}
        self._acks = {item.checkpoint_id: item for item in restored.acks}


def _resolve_store_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.suffix == ".json":
        candidate.parent.mkdir(parents=True, exist_ok=True)
        return candidate
    candidate.mkdir(parents=True, exist_ok=True)
    return candidate / STORE_FILENAME

# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""causal-experiences namespace: fork-node model and ForkNodeUpsert validation.

File-level authoritative store for fork nodes (ADR-0001 / ADR-0003, Q25/Q26).
Directory layout is ``<root>/causal-experiences/``; the URI layer is a later
slice. Writes are gated by ``is_causal_mode_enabled``. Evaluator / orchestration
trigger semantics (CONSENSUS #27 / line-28) are deferred to a later slice.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from openviking_cli.utils.config.memory_config import is_causal_mode_enabled

NAMESPACE_DIRNAME = "causal-experiences"
FORKS_DIRNAME = "forks"
REVISIONS_DIRNAME = "revisions"
INDEX_FILENAME = "index.jsonl"
IDEMPOTENCY_DIRNAME = ".idempotency"

ANCHOR_KIND_AO = "ao"
ANCHOR_KIND_SESSION_STATE = "session_state"
ANCHOR_KIND_INTERACTION_EDGE = "interaction_edge"
ANCHOR_KINDS = frozenset(
    (ANCHOR_KIND_AO, ANCHOR_KIND_SESSION_STATE, ANCHOR_KIND_INTERACTION_EDGE)
)

SEVEN_SECTION_KEYS = (
    "session_title",
    "current_state",
    "task_and_goals",
    "key_facts_and_decisions",
    "files_and_context",
    "errors_and_corrections",
    "open_issues",
)

EVIDENCE_ROLES = frozenset(
    (
        "contemporaneous_basis",
        "hindsight_attribution",
        "branch_basis",
        "guidance_basis",
    )
)
CAPTURED_STATES = frozenset(("committed", "live"))
BRANCH_EVIDENCE_STATUSES = frozenset(
    ("real", "imagined_synthetic", "imagined_unverified")
)

EvidenceRole = Literal[
    "contemporaneous_basis",
    "hindsight_attribution",
    "branch_basis",
    "guidance_basis",
]
CapturedState = Literal["committed", "live"]

_AO_IDENTITY = ("anchor_kind", "ao_id", "source_session_id")
_SESSION_STATE_REQUIRED = (
    "anchor_kind",
    "session_id",
    "ledger_snapshot_watermark",
    "live_ao_upper_bound",
    "anchored_at",
)
_INTERACTION_EDGE_REQUIRED = (
    "anchor_kind",
    "channel_id",
    "interaction_event_id",
    "from_segment_id",
    "to_segment_id",
    "dag_snapshot_watermark",
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _canonical_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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


def _require_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an int")
    return value


def _require_optional_str(value: Any, field: str) -> str | None:
    if value is None:
        return None
    return _require_str(value, field)


def canonical_anchor_key(anchor: dict[str, Any]) -> str:
    """Stable serialization of an anchor (sorted keys; ao drops sequence).

    Identity is {workspace_id, canonical_anchor_key}. ``anchor_sequence`` is a
    Q29 isolation bound, not part of the stable ao identity (Q25).
    """
    validated = validate_anchor(anchor)
    kind = validated["anchor_kind"]
    if kind == ANCHOR_KIND_AO:
        identity = {key: validated[key] for key in _AO_IDENTITY}
    elif kind == ANCHOR_KIND_SESSION_STATE:
        identity = {key: validated[key] for key in _SESSION_STATE_REQUIRED}
    else:
        identity = {key: validated[key] for key in _INTERACTION_EDGE_REQUIRED}
    return _canonical_dumps(identity)


def validate_anchor(anchor: dict[str, Any]) -> dict[str, Any]:
    """Validate a discriminated-union anchor. Returns a normalized copy."""
    payload = _require_mapping(anchor, "anchor")
    kind = payload.get("anchor_kind")
    if kind not in ANCHOR_KINDS:
        raise ValueError(
            f"anchor_kind must be one of {sorted(ANCHOR_KINDS)}, got {kind!r}"
        )
    if kind == ANCHOR_KIND_AO:
        return {
            "anchor_kind": ANCHOR_KIND_AO,
            "ao_id": _require_str(payload.get("ao_id"), "anchor.ao_id"),
            "source_session_id": _require_str(
                payload.get("source_session_id"), "anchor.source_session_id"
            ),
            "anchor_sequence": _require_int(
                payload.get("anchor_sequence"), "anchor.anchor_sequence"
            ),
        }
    if kind == ANCHOR_KIND_SESSION_STATE:
        return {
            "anchor_kind": ANCHOR_KIND_SESSION_STATE,
            "session_id": _require_str(payload.get("session_id"), "anchor.session_id"),
            "ledger_snapshot_watermark": _require_str(
                payload.get("ledger_snapshot_watermark"),
                "anchor.ledger_snapshot_watermark",
            ),
            "live_ao_upper_bound": _require_int(
                payload.get("live_ao_upper_bound"), "anchor.live_ao_upper_bound"
            ),
            "anchored_at": _require_str(payload.get("anchored_at"), "anchor.anchored_at"),
        }
    return {
        "anchor_kind": ANCHOR_KIND_INTERACTION_EDGE,
        "channel_id": _require_str(payload.get("channel_id"), "anchor.channel_id"),
        "interaction_event_id": _require_str(
            payload.get("interaction_event_id"), "anchor.interaction_event_id"
        ),
        "from_segment_id": _require_str(
            payload.get("from_segment_id"), "anchor.from_segment_id"
        ),
        "to_segment_id": _require_str(payload.get("to_segment_id"), "anchor.to_segment_id"),
        "dag_snapshot_watermark": _require_str(
            payload.get("dag_snapshot_watermark"), "anchor.dag_snapshot_watermark"
        ),
    }


@dataclass(frozen=True)
class EvidenceRef:
    """Nine-field cross-commit provenance (CONSENSUS #22 / Q25 special case)."""

    ao_id: str
    source_session_id: str
    source_archive_id: str | None
    archive_commit_watermark: str | None
    source_sequence: int
    source_read_snapshot_watermark: str
    evidence_role: EvidenceRole
    evidence_content_hash: str
    captured_state: CapturedState

    def to_dict(self) -> dict[str, Any]:
        return {
            "ao_id": self.ao_id,
            "source_session_id": self.source_session_id,
            "source_archive_id": self.source_archive_id,
            "archive_commit_watermark": self.archive_commit_watermark,
            "source_sequence": self.source_sequence,
            "source_read_snapshot_watermark": self.source_read_snapshot_watermark,
            "evidence_role": self.evidence_role,
            "evidence_content_hash": self.evidence_content_hash,
            "captured_state": self.captured_state,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvidenceRef:
        return validate_evidence_ref(data)


def validate_evidence_ref(ref: dict[str, Any] | EvidenceRef) -> EvidenceRef:
    if isinstance(ref, EvidenceRef):
        return ref
    payload = _require_mapping(ref, "evidence_ref")
    role = payload.get("evidence_role")
    if role not in EVIDENCE_ROLES:
        raise ValueError(
            f"evidence_role must be one of {sorted(EVIDENCE_ROLES)}, got {role!r}"
        )
    captured = payload.get("captured_state")
    if captured not in CAPTURED_STATES:
        raise ValueError(
            f"captured_state must be one of {sorted(CAPTURED_STATES)}, got {captured!r}"
        )
    return EvidenceRef(
        ao_id=_require_str(payload.get("ao_id"), "evidence_ref.ao_id"),
        source_session_id=_require_str(
            payload.get("source_session_id"), "evidence_ref.source_session_id"
        ),
        source_archive_id=_require_optional_str(
            payload.get("source_archive_id"), "evidence_ref.source_archive_id"
        ),
        archive_commit_watermark=_require_optional_str(
            payload.get("archive_commit_watermark"),
            "evidence_ref.archive_commit_watermark",
        ),
        source_sequence=_require_int(
            payload.get("source_sequence"), "evidence_ref.source_sequence"
        ),
        source_read_snapshot_watermark=_require_str(
            payload.get("source_read_snapshot_watermark"),
            "evidence_ref.source_read_snapshot_watermark",
        ),
        evidence_role=role,
        evidence_content_hash=_require_str(
            payload.get("evidence_content_hash"), "evidence_ref.evidence_content_hash"
        ),
        captured_state=captured,
    )


def _validate_seven_sections(sections: Any) -> dict[str, str]:
    payload = _require_mapping(sections, "seven_sections")
    missing = [key for key in SEVEN_SECTION_KEYS if key not in payload]
    if missing:
        raise ValueError(f"seven_sections missing keys: {missing}")
    extra = [key for key in payload if key not in SEVEN_SECTION_KEYS]
    if extra:
        raise ValueError(f"seven_sections unexpected keys: {extra}")
    out: dict[str, str] = {}
    for key in SEVEN_SECTION_KEYS:
        value = payload[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"seven_sections[{key}] must be a non-empty str")
        out[key] = value
    return out


def _validate_used_skills(items: Any) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        raise ValueError("used_skills must be a list")
    out: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        entry = _require_mapping(item, f"used_skills[{index}]")
        skill_uri = entry.get("skill_uri")
        revision_hash = entry.get("revision_hash")
        if not isinstance(skill_uri, str) or not skill_uri:
            raise ValueError(f"used_skills[{index}] must contain skill_uri")
        if not isinstance(revision_hash, str) or not revision_hash:
            raise ValueError(f"used_skills[{index}] must contain revision_hash")
        normalized: dict[str, Any] = {
            "skill_uri": skill_uri,
            "revision_hash": revision_hash,
        }
        if "invocation_id" in entry and entry["invocation_id"] is not None:
            normalized["invocation_id"] = _require_str(
                entry["invocation_id"], f"used_skills[{index}].invocation_id"
            )
        out.append(normalized)
    return out


def _validate_branches(items: Any) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        raise ValueError("branches must be a list")
    out: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        entry = _require_mapping(item, f"branches[{index}]")
        status = entry.get("evidence_status")
        if status not in BRANCH_EVIDENCE_STATUSES:
            raise ValueError(
                f"branches[{index}].evidence_status must be one of "
                f"{sorted(BRANCH_EVIDENCE_STATUSES)}, got {status!r}"
            )
        sequence = entry.get("ao_sequence")
        if not isinstance(sequence, list) or not all(isinstance(x, str) for x in sequence):
            raise ValueError(f"branches[{index}].ao_sequence must be a list of str")
        guidance = entry.get("guidance")
        if guidance is not None and not isinstance(guidance, dict):
            raise ValueError(f"branches[{index}].guidance must be a dict or None")
        out.append(
            {
                "branch_id": _require_str(
                    entry.get("branch_id"), f"branches[{index}].branch_id"
                ),
                "evidence_status": status,
                "ao_sequence": list(sequence),
                "guidance": dict(guidance) if guidance is not None else None,
            }
        )
    return out


def _validate_evidence_list(items: Any, field: str) -> list[EvidenceRef]:
    if not isinstance(items, list):
        raise ValueError(f"{field} must be a list")
    return [validate_evidence_ref(item) for item in items]


def _hash_fields(
    *,
    fork_node_id: str,
    workspace_id: str,
    anchor: dict[str, Any],
    revision_id: str,
    supersedes_revision_id: str | None,
    seven_sections: dict[str, str],
    contemporaneous_basis: list[EvidenceRef],
    hindsight_attribution: list[EvidenceRef],
    used_skills: list[dict[str, Any]],
    branches: list[dict[str, Any]],
    diagnosis_run_id: str,
    created_at: str,
    model_version: str,
) -> str:
    payload = {
        "fork_node_id": fork_node_id,
        "workspace_id": workspace_id,
        "anchor": anchor,
        "revision_id": revision_id,
        "supersedes_revision_id": supersedes_revision_id,
        "seven_sections": seven_sections,
        "contemporaneous_basis": [item.to_dict() for item in contemporaneous_basis],
        "hindsight_attribution": [item.to_dict() for item in hindsight_attribution],
        "used_skills": used_skills,
        "branches": branches,
        "diagnosis_run_id": diagnosis_run_id,
        "created_at": created_at,
        "model_version": model_version,
    }
    return _sha256_text(_canonical_dumps(payload))


def _check_contemporaneous_isolation(
    anchor: dict[str, Any], contemporaneous_basis: list[EvidenceRef]
) -> None:
    if anchor.get("anchor_kind") != ANCHOR_KIND_AO:
        return
    bound = _require_int(anchor.get("anchor_sequence"), "anchor.anchor_sequence")
    for ref in contemporaneous_basis:
        if ref.source_sequence >= bound:
            raise ValueError(
                "contemporaneous_basis source_sequence must be strictly less "
                f"than anchor_sequence {bound}, got {ref.source_sequence}"
            )


@dataclass(frozen=True)
class ForkNodeRevision:
    """Immutable fork-node revision (stable identity + append-only revisions)."""

    fork_node_id: str
    workspace_id: str
    anchor: dict[str, Any]
    revision_id: str
    supersedes_revision_id: str | None
    seven_sections: dict[str, str]
    contemporaneous_basis: list[EvidenceRef]
    hindsight_attribution: list[EvidenceRef]
    used_skills: list[dict[str, Any]]
    branches: list[dict[str, Any]]
    diagnosis_run_id: str
    created_at: str
    content_hash: str
    model_version: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "fork_node_id", _require_str(self.fork_node_id, "fork_node_id"))
        object.__setattr__(self, "workspace_id", _require_str(self.workspace_id, "workspace_id"))
        object.__setattr__(self, "revision_id", _require_str(self.revision_id, "revision_id"))
        object.__setattr__(
            self,
            "supersedes_revision_id",
            _require_optional_str(self.supersedes_revision_id, "supersedes_revision_id"),
        )
        object.__setattr__(
            self, "diagnosis_run_id", _require_str(self.diagnosis_run_id, "diagnosis_run_id")
        )
        object.__setattr__(self, "created_at", _require_str(self.created_at, "created_at"))
        object.__setattr__(
            self, "model_version", _require_str(self.model_version, "model_version")
        )
        object.__setattr__(self, "anchor", validate_anchor(self.anchor))
        object.__setattr__(self, "seven_sections", _validate_seven_sections(self.seven_sections))
        object.__setattr__(
            self,
            "contemporaneous_basis",
            _validate_evidence_list(self.contemporaneous_basis, "contemporaneous_basis"),
        )
        object.__setattr__(
            self,
            "hindsight_attribution",
            _validate_evidence_list(self.hindsight_attribution, "hindsight_attribution"),
        )
        object.__setattr__(self, "used_skills", _validate_used_skills(self.used_skills))
        object.__setattr__(self, "branches", _validate_branches(self.branches))
        _check_contemporaneous_isolation(self.anchor, self.contemporaneous_basis)
        expected = self.recompute_content_hash()
        stored = self.content_hash
        if stored:
            if stored != expected:
                raise ValueError(
                    f"content_hash mismatch: stored {stored!r} != computed {expected!r}"
                )
        else:
            object.__setattr__(self, "content_hash", expected)

    def recompute_content_hash(self) -> str:
        return _hash_fields(
            fork_node_id=self.fork_node_id,
            workspace_id=self.workspace_id,
            anchor=self.anchor,
            revision_id=self.revision_id,
            supersedes_revision_id=self.supersedes_revision_id,
            seven_sections=self.seven_sections,
            contemporaneous_basis=self.contemporaneous_basis,
            hindsight_attribution=self.hindsight_attribution,
            used_skills=self.used_skills,
            branches=self.branches,
            diagnosis_run_id=self.diagnosis_run_id,
            created_at=self.created_at,
            model_version=self.model_version,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "fork_node_id": self.fork_node_id,
            "workspace_id": self.workspace_id,
            "anchor": dict(self.anchor),
            "revision_id": self.revision_id,
            "supersedes_revision_id": self.supersedes_revision_id,
            "seven_sections": dict(self.seven_sections),
            "contemporaneous_basis": [item.to_dict() for item in self.contemporaneous_basis],
            "hindsight_attribution": [item.to_dict() for item in self.hindsight_attribution],
            "used_skills": [dict(item) for item in self.used_skills],
            "branches": [dict(item) for item in self.branches],
            "diagnosis_run_id": self.diagnosis_run_id,
            "created_at": self.created_at,
            "content_hash": self.content_hash,
            "model_version": self.model_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ForkNodeRevision:
        payload = _require_mapping(data, "fork_node_revision")
        return cls(
            fork_node_id=str(payload.get("fork_node_id") or ""),
            workspace_id=str(payload.get("workspace_id") or ""),
            anchor=dict(payload.get("anchor") or {}),
            revision_id=str(payload.get("revision_id") or ""),
            supersedes_revision_id=payload.get("supersedes_revision_id"),
            seven_sections=dict(payload.get("seven_sections") or {}),
            contemporaneous_basis=list(payload.get("contemporaneous_basis") or []),
            hindsight_attribution=list(payload.get("hindsight_attribution") or []),
            used_skills=list(payload.get("used_skills") or []),
            branches=list(payload.get("branches") or []),
            diagnosis_run_id=str(payload.get("diagnosis_run_id") or ""),
            created_at=str(payload.get("created_at") or ""),
            content_hash=str(payload.get("content_hash") or ""),
            model_version=str(payload.get("model_version") or ""),
        )


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


class CausalExperiencesStore:
    """File-level causal-experiences store keyed at ``<root>/causal-experiences/``."""

    def __init__(self, root: str | Path, config: Any = None) -> None:
        self._root = Path(root)
        self._config = config
        self._ns = self._root / NAMESPACE_DIRNAME
        self._index_path = self._ns / INDEX_FILENAME
        self._idempotency_dir = self._ns / IDEMPOTENCY_DIRNAME
        self._lock = threading.Lock()

    def _require_causal_write(self) -> None:
        if not is_causal_mode_enabled(self._config):
            raise PermissionError(
                "causal-experiences writes require skill_trajectory_mode=causal; "
                "evaluator/orchestration trigger semantics are implemented in a later slice"
            )

    def _fork_dir(self, fork_node_id: str) -> Path:
        return self._ns / FORKS_DIRNAME / fork_node_id

    def _revisions_dir(self, fork_node_id: str) -> Path:
        return self._fork_dir(fork_node_id) / REVISIONS_DIRNAME

    def _revision_path(self, fork_node_id: str, revision_id: str) -> Path:
        return self._revisions_dir(fork_node_id) / f"{revision_id}.json"

    def _idempotency_path(self, idempotency_key: str) -> Path:
        digest = _sha256_text(idempotency_key)
        return self._idempotency_dir / f"{digest}.json"

    def _read_index_rows(self) -> list[dict[str, Any]]:
        if not self._index_path.is_file():
            return []
        rows: list[dict[str, Any]] = []
        with self._index_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
        return rows

    def _lookup_fork_node_id(self, workspace_id: str, anchor_key: str) -> str | None:
        for row in self._read_index_rows():
            if row.get("anchor_key") != anchor_key:
                continue
            if row.get("workspace_id") not in (None, workspace_id):
                continue
            fork_node_id = row.get("fork_node_id")
            if isinstance(fork_node_id, str) and fork_node_id:
                return fork_node_id
        return None

    def _append_index(self, row: dict[str, Any]) -> None:
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        with self._index_path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())

    def _revision_ids_on_disk(self, fork_node_id: str) -> set[str]:
        rev_dir = self._revisions_dir(fork_node_id)
        if not rev_dir.is_dir():
            return set()
        return {path.stem for path in rev_dir.glob("*.json") if path.is_file()}

    def upsert_fork_node(self, payload: dict[str, Any], *, idempotency_key: str) -> dict[str, Any]:
        """Validate and persist a fork-node revision (Q26 server-side minimum).

        Evaluator / orchestration is not triggered here; that is a later slice.
        """
        self._require_causal_write()
        if not isinstance(payload, dict):
            raise ValueError("payload must be a dict")
        if not isinstance(idempotency_key, str) or not idempotency_key:
            raise ValueError("idempotency_key must be a non-empty str")

        incoming = dict(payload)
        payload_hash = _sha256_text(_canonical_dumps(incoming))
        idem_path = self._idempotency_path(idempotency_key)

        with self._lock:
            if idem_path.is_file():
                stored = json.loads(idem_path.read_text(encoding="utf-8"))
                if stored.get("payload_hash") != payload_hash:
                    raise ValueError(
                        "idempotency key conflict: same idempotency_key with different content"
                    )
                return {
                    "fork_node_id": stored["fork_node_id"],
                    "revision_id": stored["revision_id"],
                    "deduplicated": True,
                    "created": False,
                }

            workspace_id = _require_str(incoming.get("workspace_id"), "workspace_id")
            anchor = validate_anchor(incoming.get("anchor") or {})
            _validate_seven_sections(incoming.get("seven_sections") or {})
            contemporaneous = _validate_evidence_list(
                incoming.get("contemporaneous_basis") or [], "contemporaneous_basis"
            )
            _validate_evidence_list(
                incoming.get("hindsight_attribution") or [], "hindsight_attribution"
            )
            _validate_used_skills(incoming.get("used_skills") or [])
            _validate_branches(incoming.get("branches") or [])
            _check_contemporaneous_isolation(anchor, contemporaneous)
            _require_str(incoming.get("diagnosis_run_id"), "diagnosis_run_id")
            _require_str(incoming.get("model_version"), "model_version")

            anchor_key = canonical_anchor_key(anchor)
            existing_id = self._lookup_fork_node_id(workspace_id, anchor_key)
            requested_id = incoming.get("fork_node_id")
            if requested_id is not None:
                requested_id = _require_str(requested_id, "fork_node_id")

            if existing_id is not None:
                if requested_id is not None and requested_id != existing_id:
                    raise ValueError(
                        "fork_node_id does not match the stable identity for this "
                        f"workspace_id + canonical_anchor_key (expected {existing_id})"
                    )
                fork_node_id = existing_id
                created = False
            else:
                fork_node_id = requested_id or str(uuid.uuid4())
                created = True

            revision_id = incoming.get("revision_id") or str(uuid.uuid4())
            revision_id = _require_str(revision_id, "revision_id")
            if revision_id in self._revision_ids_on_disk(fork_node_id):
                raise ValueError(
                    f"duplicate revision_id {revision_id!r} under fork_node_id {fork_node_id}"
                )

            created_at = incoming.get("created_at") or _utc_now_iso()
            draft = dict(incoming)
            draft["fork_node_id"] = fork_node_id
            draft["revision_id"] = revision_id
            draft["created_at"] = created_at
            draft["anchor"] = anchor
            draft["content_hash"] = ""
            revision = ForkNodeRevision.from_dict(draft)

            revision_path = self._revision_path(fork_node_id, revision.revision_id)
            _atomic_write_json(revision_path, revision.to_dict())
            self._append_index(
                {
                    "fork_node_id": revision.fork_node_id,
                    "revision_id": revision.revision_id,
                    "workspace_id": revision.workspace_id,
                    "anchor_key": anchor_key,
                    "created_at": revision.created_at,
                }
            )
            result = {
                "fork_node_id": revision.fork_node_id,
                "revision_id": revision.revision_id,
                "deduplicated": False,
                "created": created,
            }
            _atomic_write_json(
                idem_path,
                {
                    "payload_hash": payload_hash,
                    "content_hash": revision.content_hash,
                    "fork_node_id": revision.fork_node_id,
                    "revision_id": revision.revision_id,
                    "created": created,
                    "result": result,
                },
            )
            return result

    def get_fork_node(self, fork_node_id: str) -> dict[str, Any] | None:
        rev_dir = self._revisions_dir(fork_node_id)
        if not rev_dir.is_dir():
            return None
        by_id: dict[str, dict[str, Any]] = {}
        for path in rev_dir.glob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            revision = ForkNodeRevision.from_dict(payload).to_dict()
            by_id[str(revision["revision_id"])] = revision
        if not by_id:
            return None
        ordered_ids = [
            str(row["revision_id"])
            for row in self._read_index_rows()
            if row.get("fork_node_id") == fork_node_id and row.get("revision_id") in by_id
        ]
        leftover = [rev_id for rev_id in by_id if rev_id not in ordered_ids]
        leftover.sort()
        revisions = [by_id[rev_id] for rev_id in ordered_ids + leftover]
        return {
            "fork_node_id": fork_node_id,
            "latest_revision": revisions[-1],
            "revisions": revisions,
        }

    def get_revision(self, fork_node_id: str, revision_id: str) -> dict[str, Any] | None:
        path = self._revision_path(fork_node_id, revision_id)
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("revision file must contain a JSON object")
        return ForkNodeRevision.from_dict(payload).to_dict()

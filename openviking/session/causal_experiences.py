# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""causal-experiences namespace: fork-node model and ForkNodeUpsert validation.

File-level authoritative store for fork nodes (ADR-0001 / ADR-0003, Q25/Q26).
Directory layout is ``<root>/causal-experiences/``; the URI layer is a later
slice. Writes are gated by ``is_causal_mode_enabled``. Evaluator / orchestration
trigger semantics (CONSENSUS #27 / line-28) are deferred to a later slice.

Fork revisions also follow the three-phase lifecycle (ADR-0012): ``draft``
stays in memory, ``commit_provisional`` is the authoritative accept, and
``validated`` / ``invalidated`` are append-only superseding revisions.
``influence_grounding`` snapshots and the intervention-history reverse index
implement ADR-0010 grounding reachability. Citation counts do not change
``ForkStatus``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

from openviking_cli.utils.config.memory_config import is_causal_mode_enabled

NAMESPACE_DIRNAME = "causal-experiences"
FORKS_DIRNAME = "forks"
REVISIONS_DIRNAME = "revisions"
INDEX_FILENAME = "index.jsonl"
IDEMPOTENCY_DIRNAME = ".idempotency"
INTERVENTION_INDEX_FILENAME = "intervention-index.json"

FORK_CANDIDATE_ROLE = "ForkCandidate"

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

POSITION_KIND_BLOCK = "block"
POSITION_KIND_EDGE = "edge"
POSITION_KIND_CUT = "cut"
POSITION_KINDS = frozenset(
    (POSITION_KIND_BLOCK, POSITION_KIND_EDGE, POSITION_KIND_CUT)
)
PositionKind = Literal["block", "edge", "cut"]

INDEPENDENT_CONTROL_REF_KIND = "independent_control"
OBSERVED_BRANCH_EVIDENCE_STATUS = "real"


class InterventionIndexCorruptError(RuntimeError):
    """Intervention-history index could not be read and was not treated as empty."""


class ForkStatus:
    """Fork revision lifecycle. No transition reads a citation count."""

    DRAFT = "draft"
    PROVISIONAL = "provisional"
    VALIDATED = "validated"
    INVALIDATED = "invalidated"
    ALL = frozenset((DRAFT, PROVISIONAL, VALIDATED, INVALIDATED))


EvidenceRole = Literal[
    "contemporaneous_basis",
    "hindsight_attribution",
    "branch_basis",
    "guidance_basis",
]
CapturedState = Literal["committed", "live"]

_AO_IDENTITY = ("anchor_kind", "ao_id", "source_session_id")
_SESSION_STATE_IDENTITY = ("anchor_kind", "session_id")
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
    """Stable serialization of an anchor identity (sorted keys).

    Identity is {workspace_id, canonical_anchor_key}. ``anchor_sequence`` is a
    Q29 isolation bound, not part of the stable ao identity (Q25). session_state
    identity is {anchor_kind, session_id}; snapshot watermarks belong to a
    revision's frozen evidence view, not the subject.
    """
    validated = validate_anchor(anchor)
    kind = validated["anchor_kind"]
    if kind == ANCHOR_KIND_AO:
        identity = {key: validated[key] for key in _AO_IDENTITY}
    elif kind == ANCHOR_KIND_SESSION_STATE:
        identity = {key: validated[key] for key in _SESSION_STATE_IDENTITY}
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


@dataclass(frozen=True)
class InfluenceGrounding:
    """Content-addressed snapshot of a fork's influence-view position (ADR-0010)."""

    view_revision_id: str
    position_kind: PositionKind
    position_ref: str
    semantic_summary_snapshot: str
    content_hash: str
    grounded_at: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "view_revision_id", _require_str(self.view_revision_id, "view_revision_id")
        )
        kind = self.position_kind
        if kind not in POSITION_KINDS:
            raise ValueError(
                f"position_kind must be one of {sorted(POSITION_KINDS)}, got {kind!r}"
            )
        object.__setattr__(self, "position_kind", kind)
        object.__setattr__(self, "position_ref", _require_str(self.position_ref, "position_ref"))
        object.__setattr__(
            self,
            "semantic_summary_snapshot",
            _require_str(self.semantic_summary_snapshot, "semantic_summary_snapshot"),
        )
        object.__setattr__(self, "content_hash", _require_str(self.content_hash, "content_hash"))
        object.__setattr__(self, "grounded_at", _require_str(self.grounded_at, "grounded_at"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "view_revision_id": self.view_revision_id,
            "position_kind": self.position_kind,
            "position_ref": self.position_ref,
            "semantic_summary_snapshot": self.semantic_summary_snapshot,
            "content_hash": self.content_hash,
            "grounded_at": self.grounded_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InfluenceGrounding:
        payload = _require_mapping(data, "influence_grounding")
        return cls(
            view_revision_id=str(payload.get("view_revision_id") or ""),
            position_kind=payload.get("position_kind"),  # type: ignore[arg-type]
            position_ref=str(payload.get("position_ref") or ""),
            semantic_summary_snapshot=str(payload.get("semantic_summary_snapshot") or ""),
            content_hash=str(payload.get("content_hash") or ""),
            grounded_at=str(payload.get("grounded_at") or ""),
        )


def influence_position_key(grounding: InfluenceGrounding | dict[str, Any]) -> str:
    """Stable key for one grounded position inside a view revision."""
    if not isinstance(grounding, InfluenceGrounding):
        grounding = InfluenceGrounding.from_dict(
            _require_mapping(grounding, "influence_grounding")
        )
    return _canonical_dumps(
        {"position_kind": grounding.position_kind, "position_ref": grounding.position_ref}
    )


def _coerce_influence_grounding(value: Any) -> InfluenceGrounding | None:
    if value is None:
        return None
    if isinstance(value, InfluenceGrounding):
        return value
    return InfluenceGrounding.from_dict(_require_mapping(value, "influence_grounding"))


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
    status: str | None = None,
    influence_grounding: dict[str, Any] | None = None,
    validation_evidence: dict[str, Any] | None = None,
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
        "status": status,
        "influence_grounding": influence_grounding,
        "validation_evidence": validation_evidence,
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
    status: str | None = None
    influence_grounding: InfluenceGrounding | None = None
    validation_evidence: dict[str, Any] | None = None

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
        if self.status is None:
            normalized_status: str | None = None
        else:
            normalized_status = _require_str(self.status, "status")
            if normalized_status not in ForkStatus.ALL:
                raise ValueError(
                    f"status must be one of {sorted(ForkStatus.ALL)}, got {normalized_status!r}"
                )
        object.__setattr__(self, "status", normalized_status)
        object.__setattr__(
            self,
            "influence_grounding",
            _coerce_influence_grounding(self.influence_grounding),
        )
        evidence = self.validation_evidence
        if evidence is not None:
            evidence = json.loads(
                _canonical_dumps(_require_mapping(evidence, "validation_evidence"))
            )
        object.__setattr__(self, "validation_evidence", evidence)
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
            status=self.status,
            influence_grounding=(
                None
                if self.influence_grounding is None
                else self.influence_grounding.to_dict()
            ),
            validation_evidence=self.validation_evidence,
        )

    @property
    def consumption_role(self) -> str | None:
        """Provisional revisions are consumed as ForkCandidate."""
        if self.status == ForkStatus.PROVISIONAL:
            return FORK_CANDIDATE_ROLE
        return None

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
            "status": self.status,
            "influence_grounding": (
                None
                if self.influence_grounding is None
                else self.influence_grounding.to_dict()
            ),
            "validation_evidence": self.validation_evidence,
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
            status=payload.get("status"),
            influence_grounding=payload.get("influence_grounding"),
            validation_evidence=payload.get("validation_evidence"),
        )


# Authoritative lifecycle object. A provisional ForkRevision is consumed as ForkCandidate.
ForkRevision = ForkNodeRevision


@dataclass(frozen=True)
class ForkDraft:
    """In-memory draft. Never written to the authoritative fork store."""

    revision: ForkNodeRevision

    def __post_init__(self) -> None:
        if self.revision.status != ForkStatus.DRAFT:
            raise ValueError("ForkDraft requires status=draft")

    @property
    def status(self) -> str:
        return ForkStatus.DRAFT


@dataclass(frozen=True)
class ServerChecks:
    """Attestations required before a draft may enter authoritative storage.

    Schema is enforced by this module. Anchor/ref existence, ACL, and provenance
    depend on ledger services, so the caller supplies those results. ``acl_hook``
    is invoked when present.
    """

    anchor_exists: bool
    refs_exist: bool
    provenance_valid: bool
    acl_allowed: bool = True
    acl_hook: Callable[[ForkDraft], bool] | None = None

    def failure_reason(self, draft: ForkDraft) -> str | None:
        if not self.anchor_exists:
            return "anchor does not exist"
        if not self.refs_exist:
            return "referenced evidence does not exist"
        if self.acl_hook is not None and not self.acl_hook(draft):
            return "ACL hook denied"
        if not self.acl_allowed:
            return "ACL denied"
        if not self.provenance_valid:
            return "provenance check failed"
        return None


def _coerce_server_checks(value: ServerChecks | dict[str, Any]) -> ServerChecks:
    if isinstance(value, ServerChecks):
        return value
    payload = _require_mapping(value, "server_checks")
    hook = payload.get("acl_hook")
    if hook is not None and not callable(hook):
        raise ValueError("server_checks.acl_hook must be callable")
    return ServerChecks(
        anchor_exists=bool(payload.get("anchor_exists")),
        refs_exist=bool(payload.get("refs_exist")),
        provenance_valid=bool(payload.get("provenance_valid")),
        acl_allowed=bool(payload.get("acl_allowed", True)),
        acl_hook=hook,
    )


def _require_ref_list(items: Any, field: str) -> list[Any]:
    if not isinstance(items, list) or not items:
        raise ValueError(f"validated evidence requires at least one {field}")
    return items


def _normalize_validated_evidence(evidence: Any) -> dict[str, Any]:
    payload = _require_mapping(evidence, "evidence")
    observed_in = _require_ref_list(
        payload.get("observed_branch_refs"), "observed branch ref"
    )
    observed: list[dict[str, Any]] = []
    for index, item in enumerate(observed_in):
        entry = _require_mapping(item, f"observed_branch_refs[{index}]")
        if entry.get("evidence_status") != OBSERVED_BRANCH_EVIDENCE_STATUS:
            continue
        observed.append(
            {
                "branch_id": _require_str(
                    entry.get("branch_id"), f"observed_branch_refs[{index}].branch_id"
                ),
                "evidence_status": OBSERVED_BRANCH_EVIDENCE_STATUS,
                "ref": _require_optional_str(entry.get("ref"), f"observed_branch_refs[{index}].ref"),
            }
        )
    if not observed:
        raise ValueError(
            "validated evidence requires at least one observed branch ref "
            f"with evidence_status={OBSERVED_BRANCH_EVIDENCE_STATUS!r}"
        )
    control_in = _require_ref_list(
        payload.get("independent_control_refs"), "independent_control ref"
    )
    controls: list[dict[str, Any]] = []
    for index, item in enumerate(control_in):
        entry = _require_mapping(item, f"independent_control_refs[{index}]")
        if entry.get("ref_kind") != INDEPENDENT_CONTROL_REF_KIND:
            continue
        controls.append(
            {
                "ref_kind": INDEPENDENT_CONTROL_REF_KIND,
                "ref_id": _require_str(
                    entry.get("ref_id"), f"independent_control_refs[{index}].ref_id"
                ),
            }
        )
    if not controls:
        raise ValueError("validated evidence requires at least one independent_control ref")
    return {
        "observed_branch_refs": observed,
        "independent_control_refs": controls,
    }


def _normalize_counter_evidence(counter_evidence: Any) -> dict[str, Any]:
    payload = _require_mapping(counter_evidence, "counter_evidence")
    if not payload:
        raise ValueError("counter_evidence must be a non-empty dict")
    return json.loads(_canonical_dumps(payload))


def _provisional_idempotency_key(revision: ForkNodeRevision) -> str:
    return _canonical_dumps(
        {
            "diagnosis_run_id": revision.diagnosis_run_id,
            "anchor": canonical_anchor_key(revision.anchor),
            "content_hash": revision.content_hash,
        }
    )


def _normalize_principal_labels(labels: Any) -> frozenset[str]:
    if isinstance(labels, (str, bytes)) or not isinstance(labels, Sequence):
        raise ValueError("principal_labels must be a sequence of str")
    return frozenset(_require_str(item, "principal_labels") for item in labels)


def _security_labels_of(row: dict[str, Any]) -> frozenset[str]:
    raw = row.get("security_labels")
    if not isinstance(raw, list) or not raw:
        return frozenset()
    labels: list[str] = []
    for item in raw:
        if not isinstance(item, str) or not item:
            return frozenset()
        labels.append(item)
    return frozenset(labels)


def _quarantine_intervention_index(path: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    dest = path.with_name(f"{path.name}.corrupt-{stamp}")
    os.replace(path, dest)
    return dest


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


def _revision_sort_key(revision: dict[str, Any]) -> tuple[str, str]:
    return (str(revision.get("created_at") or ""), str(revision.get("revision_id") or ""))


def _order_revisions(by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Authoritative revision order: explicit supersedes chain, then siblings.

    Walk each explicit ``supersedes_revision_id`` chain head-to-tail. Concurrent
    sibling roots that do not form a chain are ordered by ``created_at`` then
    ``revision_id``. Mutual supersede is never inferred from timestamps.
    """
    successors: dict[str, list[str]] = {rev_id: [] for rev_id in by_id}
    has_predecessor_in_set: set[str] = set()
    for rev_id, revision in by_id.items():
        pred = revision.get("supersedes_revision_id")
        if isinstance(pred, str) and pred in by_id:
            successors[pred].append(rev_id)
            has_predecessor_in_set.add(rev_id)
    for children in successors.values():
        children.sort(key=lambda child: _revision_sort_key(by_id[child]))

    roots = [rev_id for rev_id in by_id if rev_id not in has_predecessor_in_set]
    roots.sort(key=lambda rev_id: _revision_sort_key(by_id[rev_id]))

    ordered: list[dict[str, Any]] = []
    visited: set[str] = set()

    def walk(rev_id: str) -> None:
        if rev_id in visited:
            return
        visited.add(rev_id)
        ordered.append(by_id[rev_id])
        for child in successors[rev_id]:
            walk(child)

    for root in roots:
        walk(root)

    leftover = [rev_id for rev_id in by_id if rev_id not in visited]
    leftover.sort(key=lambda rev_id: _revision_sort_key(by_id[rev_id]))
    ordered.extend(by_id[rev_id] for rev_id in leftover)
    return ordered


class CausalExperiencesStore:
    """File-level causal-experiences store keyed at ``<root>/causal-experiences/``."""

    def __init__(self, root: str | Path, config: Any = None) -> None:
        self._root = Path(root)
        self._config = config
        self._ns = self._root / NAMESPACE_DIRNAME
        self._index_path = self._ns / INDEX_FILENAME
        self._idempotency_dir = self._ns / IDEMPOTENCY_DIRNAME
        self._intervention_index_path = self._ns / INTERVENTION_INDEX_FILENAME
        self._drafts: dict[str, ForkDraft] = {}
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

    def upsert_fork_node(
        self,
        payload: dict[str, Any],
        *,
        idempotency_key: str,
        _allow_terminal_status: bool = False,
    ) -> dict[str, Any]:
        """Validate and persist a fork-node revision (Q26 server-side minimum).

        Evaluator / orchestration is not triggered here; that is a later slice.
        ``validated`` / ``invalidated`` are accepted only from the append APIs.
        """
        self._require_causal_write()
        if not isinstance(payload, dict):
            raise ValueError("payload must be a dict")
        if not isinstance(idempotency_key, str) or not idempotency_key:
            raise ValueError("idempotency_key must be a non-empty str")
        status = payload.get("status")
        if status == ForkStatus.DRAFT:
            raise ValueError("draft status cannot be written to authoritative storage")
        if status in (ForkStatus.VALIDATED, ForkStatus.INVALIDATED) and not _allow_terminal_status:
            raise ValueError(
                f"status {status!r} cannot be written by upsert_fork_node; "
                "only append_validated/append_invalidated may write validated|invalidated"
            )

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

            created_at = incoming.get("created_at") or _utc_now_iso()
            draft = dict(incoming)
            draft["fork_node_id"] = fork_node_id
            draft["revision_id"] = revision_id
            draft["created_at"] = created_at
            draft["anchor"] = anchor
            draft["content_hash"] = ""
            revision = ForkNodeRevision.from_dict(draft)

            if revision.revision_id in self._revision_ids_on_disk(fork_node_id):
                return self._recover_missing_idempotency(
                    fork_node_id=fork_node_id,
                    revision=revision,
                    idem_path=idem_path,
                    payload_hash=payload_hash,
                )

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

    def _recover_missing_idempotency(
        self,
        *,
        fork_node_id: str,
        revision: ForkNodeRevision,
        idem_path: Path,
        payload_hash: str,
    ) -> dict[str, Any]:
        """Finish a crash window where the revision is durable and idempotency is not.

        Matching content is an idempotent hit: rewrite the idempotency file and
        return the original identity. Differing content is a conflict. This path
        never reports ``duplicate revision_id``.
        """
        path = self._revision_path(fork_node_id, revision.revision_id)
        stored = ForkNodeRevision.from_dict(
            json.loads(path.read_text(encoding="utf-8"))
        )
        if stored.content_hash != revision.content_hash:
            raise ValueError(
                "idempotency key conflict: revision already stored with different content "
                f"for revision_id {revision.revision_id!r}"
            )
        result = {
            "fork_node_id": stored.fork_node_id,
            "revision_id": stored.revision_id,
            "deduplicated": True,
            "created": False,
        }
        _atomic_write_json(
            idem_path,
            {
                "payload_hash": payload_hash,
                "content_hash": stored.content_hash,
                "fork_node_id": stored.fork_node_id,
                "revision_id": stored.revision_id,
                "created": False,
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
        revisions = _order_revisions(by_id)
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

    def submit_fork_draft(
        self,
        payload: dict[str, Any],
        *,
        influence_grounding: InfluenceGrounding | dict[str, Any] | None = None,
    ) -> ForkDraft:
        """Validate a fork draft in memory. Does not touch authoritative storage."""
        self._require_causal_write()
        if not isinstance(payload, dict):
            raise ValueError("payload must be a dict")
        incoming = dict(payload)
        incoming["status"] = ForkStatus.DRAFT
        incoming["content_hash"] = ""
        if influence_grounding is not None:
            incoming["influence_grounding"] = (
                influence_grounding.to_dict()
                if isinstance(influence_grounding, InfluenceGrounding)
                else influence_grounding
            )
        if not incoming.get("fork_node_id"):
            incoming["fork_node_id"] = str(uuid.uuid4())
        if not incoming.get("revision_id"):
            incoming["revision_id"] = str(uuid.uuid4())
        if not incoming.get("created_at"):
            incoming["created_at"] = _utc_now_iso()
        draft = ForkDraft(ForkNodeRevision.from_dict(incoming))
        with self._lock:
            self._drafts[draft.revision.revision_id] = draft
        return draft

    def commit_provisional(
        self,
        draft: ForkDraft,
        server_checks: ServerChecks | dict[str, Any],
    ) -> ForkRevision:
        """Persist ``draft`` as a provisional ForkRevision when server checks pass.

        Idempotency key is ``diagnosis_run_id + anchor + content_hash``. The
        returned object is consumed in the ForkCandidate role.
        """
        self._require_causal_write()
        if not isinstance(draft, ForkDraft):
            raise ValueError("draft must be a ForkDraft")
        checks = _coerce_server_checks(server_checks)
        reason = checks.failure_reason(draft)
        if reason is not None:
            raise ValueError(f"server check failed: {reason}")
        provisional_payload = draft.revision.to_dict()
        provisional_payload["status"] = ForkStatus.PROVISIONAL
        provisional_payload["content_hash"] = ""
        provisional = ForkNodeRevision.from_dict(provisional_payload)
        result = self.upsert_fork_node(
            provisional.to_dict(),
            idempotency_key=_provisional_idempotency_key(provisional),
        )
        stored = self.get_revision(result["fork_node_id"], result["revision_id"])
        if stored is None:
            raise ValueError("provisional revision missing after commit")
        revision = ForkNodeRevision.from_dict(stored)
        self._index_intervention(revision)
        with self._lock:
            self._drafts.pop(draft.revision.revision_id, None)
        return revision

    def append_validated(self, fork_node_id: str, evidence: dict[str, Any]) -> ForkRevision:
        """Append a validated superseding revision. Does not overwrite history."""
        self._require_causal_write()
        normalized = _normalize_validated_evidence(evidence)
        return self._append_status_revision(
            fork_node_id, ForkStatus.VALIDATED, normalized
        )

    def append_invalidated(
        self, fork_node_id: str, counter_evidence: dict[str, Any]
    ) -> ForkRevision:
        """Append an invalidated superseding revision. Does not overwrite history."""
        self._require_causal_write()
        normalized = _normalize_counter_evidence(counter_evidence)
        return self._append_status_revision(
            fork_node_id, ForkStatus.INVALIDATED, normalized
        )

    def query_intervention_history(
        self,
        view_revision_id: str,
        position_key: str,
        principal_labels: Sequence[str],
    ) -> list[dict[str, Any]]:
        """Return intervention-history rows visible to ``principal_labels``.

        Each row is kept only when its security labels intersect the principal.
        Rows with missing labels are omitted. The result has no hidden-count fields.
        """
        view_revision_id = _require_str(view_revision_id, "view_revision_id")
        position_key = _require_str(position_key, "position_key")
        principal = _normalize_principal_labels(principal_labels)
        with self._lock:
            rows = self._read_intervention_entries()
        matched = [
            _public_intervention_row(row)
            for row in rows
            if row.get("view_revision_id") == view_revision_id
            and row.get("position_key") == position_key
            and principal.intersection(_security_labels_of(row))
        ]
        return matched

    def mark_needs_revalidation(self, view_revision_id: str) -> int:
        """Flag groundings of ``view_revision_id`` without deleting or rewriting revisions."""
        self._require_causal_write()
        view_revision_id = _require_str(view_revision_id, "view_revision_id")
        with self._lock:
            rows = self._read_intervention_entries()
            updated = 0
            for row in rows:
                if row.get("view_revision_id") != view_revision_id:
                    continue
                if row.get("needs_revalidation") is True:
                    continue
                row["needs_revalidation"] = True
                updated += 1
            if updated:
                self._write_intervention_entries(rows)
            return updated

    def _append_status_revision(
        self,
        fork_node_id: str,
        status: str,
        validation_evidence: dict[str, Any],
    ) -> ForkRevision:
        fork_node_id = _require_str(fork_node_id, "fork_node_id")
        loaded = self.get_fork_node(fork_node_id)
        if loaded is None:
            raise ValueError(f"unknown fork_node_id {fork_node_id!r}")
        latest = dict(loaded["latest_revision"])
        payload = {
            "fork_node_id": fork_node_id,
            "workspace_id": latest["workspace_id"],
            "anchor": latest["anchor"],
            "revision_id": str(uuid.uuid4()),
            "supersedes_revision_id": latest["revision_id"],
            "seven_sections": latest["seven_sections"],
            "contemporaneous_basis": latest["contemporaneous_basis"],
            "hindsight_attribution": latest["hindsight_attribution"],
            "used_skills": latest["used_skills"],
            "branches": latest["branches"],
            "diagnosis_run_id": latest["diagnosis_run_id"],
            "created_at": _utc_now_iso(),
            "model_version": latest["model_version"],
            "status": status,
            "influence_grounding": latest.get("influence_grounding"),
            "validation_evidence": validation_evidence,
            "content_hash": "",
        }
        revision = ForkNodeRevision.from_dict(payload)
        result = self.upsert_fork_node(
            revision.to_dict(),
            idempotency_key=_canonical_dumps(
                {
                    "op": status,
                    "diagnosis_run_id": revision.diagnosis_run_id,
                    "anchor": canonical_anchor_key(revision.anchor),
                    "content_hash": revision.content_hash,
                }
            ),
            _allow_terminal_status=True,
        )
        stored = self.get_revision(result["fork_node_id"], result["revision_id"])
        if stored is None:
            raise ValueError("appended revision missing after write")
        appended = ForkNodeRevision.from_dict(stored)
        self._index_intervention(appended)
        return appended

    def _index_intervention(self, revision: ForkNodeRevision) -> None:
        grounding = revision.influence_grounding
        if grounding is None:
            return
        entry = {
            "view_revision_id": grounding.view_revision_id,
            "position_key": influence_position_key(grounding),
            "fork_node_id": revision.fork_node_id,
            "revision_id": revision.revision_id,
            "validation_status": revision.status,
            "summary": grounding.semantic_summary_snapshot,
            "needs_revalidation": False,
            "security_labels": [revision.workspace_id],
        }
        with self._lock:
            rows = self._read_intervention_entries()
            if any(row.get("revision_id") == revision.revision_id for row in rows):
                return
            rows.append(entry)
            self._write_intervention_entries(rows)

    def _read_intervention_entries(self) -> list[dict[str, Any]]:
        path = self._intervention_index_path
        if not path.exists():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self._raise_corrupt_intervention_index(path, exc)
        if not isinstance(payload, dict) or not isinstance(payload.get("entries"), list):
            self._raise_corrupt_intervention_index(
                path, ValueError("intervention index entries are missing")
            )
        entries = payload["entries"]
        if any(not isinstance(item, dict) for item in entries):
            self._raise_corrupt_intervention_index(
                path, ValueError("intervention index entry is not an object")
            )
        return [dict(item) for item in entries]

    def _raise_corrupt_intervention_index(self, path: Path, exc: BaseException) -> None:
        try:
            quarantined = _quarantine_intervention_index(path)
        except OSError:
            logger.error("intervention index unreadable and could not be quarantined: %s", path)
            raise InterventionIndexCorruptError(
                f"intervention index corrupt at {path}; quarantine failed"
            ) from exc
        logger.error(
            "intervention index unreadable at %s; quarantined to %s",
            path,
            quarantined,
        )
        raise InterventionIndexCorruptError(
            f"intervention index corrupt; quarantined to {quarantined}"
        ) from exc

    def _write_intervention_entries(self, rows: list[dict[str, Any]]) -> None:
        _atomic_write_json(self._intervention_index_path, {"entries": rows})


def _public_intervention_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "fork_node_id": row.get("fork_node_id"),
        "revision_id": row.get("revision_id"),
        "validation_status": row.get("validation_status"),
        "summary": row.get("summary"),
        "needs_revalidation": bool(row.get("needs_revalidation")),
    }

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
from collections.abc import Callable, Mapping, Sequence
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

APPLICABILITY_DIMENSIONS = (
    "task_decision_type",
    "anchor_state",
    "failure_mode",
    "action_skill_role",
)
APPLICABILITY_CONCLUSION_APPLICABLE = "applicable"
APPLICABILITY_CONCLUSION_PARTIALLY = "partially_applicable"
APPLICABILITY_CONCLUSION_NOT = "not_applicable"
APPLICABILITY_CONCLUSION_INSUFFICIENT = "insufficient"
APPLICABILITY_CONCLUSIONS = frozenset(
    (
        APPLICABILITY_CONCLUSION_APPLICABLE,
        APPLICABILITY_CONCLUSION_PARTIALLY,
        APPLICABILITY_CONCLUSION_NOT,
        APPLICABILITY_CONCLUSION_INSUFFICIENT,
    )
)
APPLICABILITY_PASSING_CONCLUSIONS = frozenset(
    (APPLICABILITY_CONCLUSION_APPLICABLE, APPLICABILITY_CONCLUSION_PARTIALLY)
)
COUNTER_EXAMPLE_CONCLUSION_CLEAR = "no_counter_example"
COUNTER_EXAMPLE_CONCLUSION_FOUND = "counter_example_found"
COUNTER_EXAMPLE_CONCLUSIONS = frozenset(
    (COUNTER_EXAMPLE_CONCLUSION_CLEAR, COUNTER_EXAMPLE_CONCLUSION_FOUND)
)

_CONTROL_CONTRACT_FIELDS = (
    "anchor_state_snapshot",
    "task_input",
    "agent_model_prompt_policy",
    "tool_dependency_versions",
    "permissions",
    "budget",
    "evaluator",
    "frozen_at",
)
_CONTROL_CONTRACT_STRING_FIELDS = (
    "anchor_state_snapshot",
    "task_input",
    "agent_model_prompt_policy",
    "permissions",
    "budget",
    "evaluator",
    "frozen_at",
)


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


@dataclass(frozen=True)
class AnchorValidity:
    """Whether the provisional anchor and its refs were still valid, and when."""

    valid: bool
    checked_at: str

    def __post_init__(self) -> None:
        if not isinstance(self.valid, bool):
            raise ValueError("anchor_valid.valid must be a bool")
        if not isinstance(self.checked_at, str):
            raise ValueError("anchor_valid.checked_at must be a str")

    def to_dict(self) -> dict[str, Any]:
        return {"valid": self.valid, "checked_at": self.checked_at}


@dataclass(frozen=True)
class ApplicabilityCheck:
    """One conclusion tier for each of the four applicability dimensions."""

    task_decision_type: str
    anchor_state: str
    failure_mode: str
    action_skill_role: str

    def __post_init__(self) -> None:
        for dimension in APPLICABILITY_DIMENSIONS:
            value = getattr(self, dimension)
            if not isinstance(value, str):
                raise ValueError(f"applicability_check.{dimension} must be a str")
            if value not in APPLICABILITY_CONCLUSIONS:
                raise ValueError(
                    f"applicability_check.{dimension} must be one of "
                    f"{sorted(APPLICABILITY_CONCLUSIONS)}, got {value!r}"
                )

    def to_dict(self) -> dict[str, str]:
        return {dimension: getattr(self, dimension) for dimension in APPLICABILITY_DIMENSIONS}


@dataclass(frozen=True)
class CounterExampleCheck:
    """Counter-example conclusion plus the refs the check examined."""

    conclusion: str
    refs: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.conclusion not in COUNTER_EXAMPLE_CONCLUSIONS:
            raise ValueError(
                "counter_example_check.conclusion must be one of "
                f"{sorted(COUNTER_EXAMPLE_CONCLUSIONS)}, got {self.conclusion!r}"
            )
        object.__setattr__(self, "refs", tuple(self.refs))
        for index, ref in enumerate(self.refs):
            _require_str(ref, f"counter_example_check.refs[{index}]")

    def to_dict(self) -> dict[str, Any]:
        return {"conclusion": self.conclusion, "refs": list(self.refs)}


@dataclass(frozen=True)
class ControlContract:
    """Frozen paired-control contract (Q85-A). Empty evaluator or frozen_at does not validate."""

    anchor_state_snapshot: str
    task_input: str
    agent_model_prompt_policy: str
    tool_dependency_versions: dict[str, str]
    permissions: str
    budget: str
    evaluator: str
    frozen_at: str

    def __post_init__(self) -> None:
        for field in _CONTROL_CONTRACT_STRING_FIELDS:
            value = getattr(self, field)
            if not isinstance(value, str):
                raise ValueError(f"control_contract.{field} must be a str")
        versions = self.tool_dependency_versions
        if not isinstance(versions, dict):
            raise ValueError("control_contract.tool_dependency_versions must be a dict")
        normalized: dict[str, str] = {}
        for key, value in versions.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError(
                    "control_contract.tool_dependency_versions keys must be non-empty str"
                )
            if not isinstance(value, str):
                raise ValueError(
                    f"control_contract.tool_dependency_versions[{key!r}] must be a str"
                )
            normalized[key] = value
        object.__setattr__(self, "tool_dependency_versions", dict(sorted(normalized.items())))

    def to_dict(self) -> dict[str, Any]:
        return {
            "anchor_state_snapshot": self.anchor_state_snapshot,
            "task_input": self.task_input,
            "agent_model_prompt_policy": self.agent_model_prompt_policy,
            "tool_dependency_versions": dict(self.tool_dependency_versions),
            "permissions": self.permissions,
            "budget": self.budget,
            "evaluator": self.evaluator,
            "frozen_at": self.frozen_at,
        }


@dataclass(frozen=True)
class ValidationEvidence:
    """Q81/Q85 evidence required before a fork revision may be validated."""

    observed_branch_refs: tuple[dict[str, Any], ...]
    independent_control_refs: tuple[dict[str, Any], ...]
    anchor_valid: AnchorValidity
    applicability_check: ApplicabilityCheck
    counter_example_check: CounterExampleCheck
    no_high_severity_contradictions: bool
    control_contract: ControlContract

    def __post_init__(self) -> None:
        object.__setattr__(self, "observed_branch_refs", tuple(self.observed_branch_refs))
        object.__setattr__(
            self, "independent_control_refs", tuple(self.independent_control_refs)
        )
        if not isinstance(self.anchor_valid, AnchorValidity):
            raise ValueError("anchor_valid must be an AnchorValidity")
        if not isinstance(self.applicability_check, ApplicabilityCheck):
            raise ValueError("applicability_check must be an ApplicabilityCheck")
        if not isinstance(self.counter_example_check, CounterExampleCheck):
            raise ValueError("counter_example_check must be a CounterExampleCheck")
        if not isinstance(self.no_high_severity_contradictions, bool):
            raise ValueError("no_high_severity_contradictions must be a bool")
        if not isinstance(self.control_contract, ControlContract):
            raise ValueError("control_contract must be a ControlContract")

    def to_dict(self) -> dict[str, Any]:
        return {
            "observed_branch_refs": [dict(item) for item in self.observed_branch_refs],
            "independent_control_refs": [dict(item) for item in self.independent_control_refs],
            "anchor_valid": self.anchor_valid.to_dict(),
            "applicability_check": self.applicability_check.to_dict(),
            "counter_example_check": self.counter_example_check.to_dict(),
            "no_high_severity_contradictions": self.no_high_severity_contradictions,
            "control_contract": self.control_contract.to_dict(),
        }


@dataclass(frozen=True)
class ValidationOutcome:
    """Returned when validation gates fail. The stored fork stays provisional."""

    status: str
    missing_requirements: tuple[str, ...]
    fork_node_id: str
    revision_id: str

    def __post_init__(self) -> None:
        if self.status != ForkStatus.PROVISIONAL:
            raise ValueError("ValidationOutcome status must be provisional")
        object.__setattr__(self, "missing_requirements", tuple(self.missing_requirements))
        if not self.missing_requirements:
            raise ValueError("ValidationOutcome requires missing_requirements")
        object.__setattr__(self, "fork_node_id", _require_str(self.fork_node_id, "fork_node_id"))
        object.__setattr__(self, "revision_id", _require_str(self.revision_id, "revision_id"))


def _nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _reject_unexpected_keys(payload: dict[str, Any], allowed: Sequence[str], field: str) -> None:
    extra = [key for key in payload if key not in allowed]
    if extra:
        raise ValueError(f"{field} unexpected keys: {sorted(extra)}")


def _mapping_or_dict(value: Any, field: str) -> dict[str, Any]:
    to_dict = getattr(value, "to_dict", None)
    if to_dict is not None and not isinstance(value, dict):
        value = to_dict()
    return _require_mapping(value, field)


def _assess_anchor_valid(payload: dict[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    if "anchor_valid" not in payload or payload["anchor_valid"] is None:
        return None, ["anchor_valid"]
    raw = _mapping_or_dict(payload["anchor_valid"], "anchor_valid")
    _reject_unexpected_keys(raw, ("valid", "checked_at"), "anchor_valid")
    if "valid" not in raw:
        checked = raw.get("checked_at")
        if checked is not None and not isinstance(checked, str):
            raise ValueError("anchor_valid.checked_at must be a str")
        return None, ["anchor_valid"]
    valid = raw["valid"]
    if not isinstance(valid, bool):
        raise ValueError("anchor_valid.valid must be a bool")
    checked = raw.get("checked_at")
    if checked is not None and not isinstance(checked, str):
        raise ValueError("anchor_valid.checked_at must be a str")
    missing: list[str] = []
    if valid is not True:
        missing.append("anchor_valid")
    if not _nonempty_text(checked):
        missing.append("anchor_valid.checked_at")
    if missing:
        return None, missing
    return AnchorValidity(valid=True, checked_at=str(checked).strip()).to_dict(), []


def _assess_applicability(payload: dict[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    if "applicability_check" not in payload or payload["applicability_check"] is None:
        return None, ["applicability_check"]
    raw = _mapping_or_dict(payload["applicability_check"], "applicability_check")
    _reject_unexpected_keys(raw, APPLICABILITY_DIMENSIONS, "applicability_check")
    missing: list[str] = []
    conclusions: dict[str, str] = {}
    for dimension in APPLICABILITY_DIMENSIONS:
        if dimension not in raw or raw[dimension] is None:
            missing.append(f"applicability_check.{dimension}")
            continue
        value = raw[dimension]
        if not isinstance(value, str):
            raise ValueError(f"applicability_check.{dimension} must be a str")
        if value not in APPLICABILITY_CONCLUSIONS:
            raise ValueError(
                f"applicability_check.{dimension} must be one of "
                f"{sorted(APPLICABILITY_CONCLUSIONS)}, got {value!r}"
            )
        if value not in APPLICABILITY_PASSING_CONCLUSIONS:
            missing.append(f"applicability_check.{dimension}")
            continue
        conclusions[dimension] = value
    if missing:
        return None, missing
    return ApplicabilityCheck(**conclusions).to_dict(), []


def _assess_counter_example(payload: dict[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    if "counter_example_check" not in payload or payload["counter_example_check"] is None:
        return None, ["counter_example_check"]
    raw = _mapping_or_dict(payload["counter_example_check"], "counter_example_check")
    _reject_unexpected_keys(raw, ("conclusion", "refs"), "counter_example_check")
    missing: list[str] = []
    conclusion = raw.get("conclusion")
    if conclusion is None:
        missing.append("counter_example_check")
    elif not isinstance(conclusion, str):
        raise ValueError("counter_example_check.conclusion must be a str")
    elif conclusion not in COUNTER_EXAMPLE_CONCLUSIONS:
        raise ValueError(
            "counter_example_check.conclusion must be one of "
            f"{sorted(COUNTER_EXAMPLE_CONCLUSIONS)}, got {conclusion!r}"
        )
    elif conclusion != COUNTER_EXAMPLE_CONCLUSION_CLEAR:
        missing.append("counter_example_check")
    refs = raw.get("refs")
    normalized_refs: list[str] = []
    if refs is None:
        missing.append("counter_example_check.refs")
    elif isinstance(refs, (str, bytes)) or not isinstance(refs, list):
        raise ValueError("counter_example_check.refs must be a list")
    else:
        for index, item in enumerate(refs):
            normalized_refs.append(_require_str(item, f"counter_example_check.refs[{index}]"))
    if missing:
        return None, missing
    return (
        CounterExampleCheck(conclusion=str(conclusion), refs=tuple(normalized_refs)).to_dict(),
        [],
    )


def _assess_no_high_severity(payload: dict[str, Any]) -> tuple[bool | None, list[str]]:
    if "no_high_severity_contradictions" not in payload:
        return None, ["no_high_severity_contradictions"]
    value = payload["no_high_severity_contradictions"]
    if value is None:
        return None, ["no_high_severity_contradictions"]
    if not isinstance(value, bool):
        raise ValueError("no_high_severity_contradictions must be a bool")
    if value is not True:
        return None, ["no_high_severity_contradictions"]
    return True, []


def _assess_control_contract(payload: dict[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    if "control_contract" not in payload or payload["control_contract"] is None:
        return None, ["control_contract"]
    raw = _mapping_or_dict(payload["control_contract"], "control_contract")
    _reject_unexpected_keys(raw, _CONTROL_CONTRACT_FIELDS, "control_contract")
    missing: list[str] = []
    strings: dict[str, str] = {}
    for field in _CONTROL_CONTRACT_STRING_FIELDS:
        if field not in raw or raw[field] is None:
            missing.append(f"control_contract.{field}")
            continue
        value = raw[field]
        if not isinstance(value, str):
            raise ValueError(f"control_contract.{field} must be a str")
        if not value.strip():
            missing.append(f"control_contract.{field}")
            continue
        strings[field] = value.strip()
    versions: dict[str, str] | None = None
    if "tool_dependency_versions" not in raw or raw["tool_dependency_versions"] is None:
        missing.append("control_contract.tool_dependency_versions")
    else:
        raw_versions = raw["tool_dependency_versions"]
        if not isinstance(raw_versions, dict):
            raise ValueError("control_contract.tool_dependency_versions must be a dict")
        versions = {}
        for key, value in raw_versions.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError(
                    "control_contract.tool_dependency_versions keys must be non-empty str"
                )
            if not isinstance(value, str):
                raise ValueError(
                    f"control_contract.tool_dependency_versions[{key!r}] must be a str"
                )
            versions[key.strip()] = value
    if missing or versions is None:
        return None, missing
    contract = ControlContract(
        anchor_state_snapshot=strings["anchor_state_snapshot"],
        task_input=strings["task_input"],
        agent_model_prompt_policy=strings["agent_model_prompt_policy"],
        tool_dependency_versions=versions,
        permissions=strings["permissions"],
        budget=strings["budget"],
        evaluator=strings["evaluator"],
        frozen_at=strings["frozen_at"],
    )
    return contract.to_dict(), []


def _assess_validation_gates(evidence: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Return normalized Q81/Q85 fields and any failed requirement ids.

    Illegal shapes raise ``ValueError``. A missing or failed gate is a requirement
    id, not an exception.
    """
    payload = _require_mapping(evidence, "evidence")
    normalized: dict[str, Any] = {}
    missing: list[str] = []
    anchor, anchor_missing = _assess_anchor_valid(payload)
    missing.extend(anchor_missing)
    if anchor is not None:
        normalized["anchor_valid"] = anchor
    applicability, applicability_missing = _assess_applicability(payload)
    missing.extend(applicability_missing)
    if applicability is not None:
        normalized["applicability_check"] = applicability
    counter, counter_missing = _assess_counter_example(payload)
    missing.extend(counter_missing)
    if counter is not None:
        normalized["counter_example_check"] = counter
    contradictions, contradiction_missing = _assess_no_high_severity(payload)
    missing.extend(contradiction_missing)
    if contradictions is not None:
        normalized["no_high_severity_contradictions"] = contradictions
    contract, contract_missing = _assess_control_contract(payload)
    missing.extend(contract_missing)
    if contract is not None:
        normalized["control_contract"] = contract
    return normalized, missing


def _validation_evidence_dict(
    core: dict[str, Any], gates: dict[str, Any]
) -> dict[str, Any]:
    evidence = ValidationEvidence(
        observed_branch_refs=tuple(core["observed_branch_refs"]),
        independent_control_refs=tuple(core["independent_control_refs"]),
        anchor_valid=AnchorValidity(
            valid=gates["anchor_valid"]["valid"],
            checked_at=gates["anchor_valid"]["checked_at"],
        ),
        applicability_check=ApplicabilityCheck(**gates["applicability_check"]),
        counter_example_check=CounterExampleCheck(
            conclusion=gates["counter_example_check"]["conclusion"],
            refs=tuple(gates["counter_example_check"]["refs"]),
        ),
        no_high_severity_contradictions=gates["no_high_severity_contradictions"],
        control_contract=ControlContract(**gates["control_contract"]),
    )
    return evidence.to_dict()


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

    def append_validated(
        self, fork_node_id: str, evidence: dict[str, Any]
    ) -> ForkRevision | ValidationOutcome:
        """Append a validated revision only when every Q81/Q85 gate passes.

        A real observed branch and an independent-control ref are still required
        (``ValueError`` when they are absent or malformed). Anchor validity,
        four-dimension applicability, the counter-example check, absence of
        high-severity contradictions, and a frozen ``ControlContract`` are the
        remaining gates: any miss leaves the stored fork provisional and returns
        ``ValidationOutcome`` instead of writing a validated revision.
        """
        self._require_causal_write()
        fork_node_id = _require_str(fork_node_id, "fork_node_id")
        core = _normalize_validated_evidence(evidence)
        gates, missing = _assess_validation_gates(evidence)
        if missing:
            loaded = self.get_fork_node(fork_node_id)
            if loaded is None:
                raise ValueError(f"unknown fork_node_id {fork_node_id!r}")
            latest = loaded["latest_revision"]
            return ValidationOutcome(
                status=ForkStatus.PROVISIONAL,
                missing_requirements=tuple(missing),
                fork_node_id=fork_node_id,
                revision_id=str(latest["revision_id"]),
            )
        return self._append_status_revision(
            fork_node_id,
            ForkStatus.VALIDATED,
            _validation_evidence_dict(core, gates),
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


def branch_retrieval_channel(branch: Mapping[str, Any], parent_status: str) -> str:
    """Read-only retrieval channel for one branch. Does not write storage.

    The parent revision channel is not inherited. Verified only when the branch
    itself is real and the parent fork is validated (``evidence_status`` is
    ``real`` and ``parent_status`` is ``validated``), or when an imagined branch
    is itself validated (``imagined_synthetic`` under a validated parent).
    ``imagined_unverified`` stays provisional even on a validated parent.
    """
    evidence = branch.get("evidence_status")
    if evidence == "imagined_unverified":
        return "provisional"
    parent_validated = parent_status == ForkStatus.VALIDATED
    if evidence == "real" and parent_validated:
        return "verified"
    if evidence == "imagined_synthetic" and parent_validated:
        return "verified"
    return "provisional"

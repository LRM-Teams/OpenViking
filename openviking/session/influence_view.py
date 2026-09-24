# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""L2 Influence View: mutable draft, frozen revision, structural checks.

Post-run and failure-diagnosis views are derived trajectory indexes over the
AO ledger (ADR-0010). Drafts stay mutable for at most three critic rounds;
``freeze`` yields an immutable revision. The supersede index keeps older
revisions as history and points ``latest`` at the newest freeze for a
``(session_id, purpose)`` pair.
"""

from __future__ import annotations

import hashlib
import json
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

BLOCK_ROLES = frozenset(("task", "orchestrator", "agent_action", "conclusion"))
PURPOSES = frozenset(("post_run_index", "failure_diagnosis"))
CLAIM_STATUSES = frozenset(("hypothesis", "supported", "contradicted", "insufficient"))
CONSTRUCTION_MODES = frozenset(("adaptive", "fallback"))
SEVERITIES = frozenset(("high", "low"))

CLAIM_STATUS_HYPOTHESIS = "hypothesis"
CONSTRUCTION_MODE_ADAPTIVE = "adaptive"
CONSTRUCTION_MODE_FALLBACK = "fallback"
PURPOSE_POST_RUN_INDEX = "post_run_index"
PURPOSE_FAILURE_DIAGNOSIS = "failure_diagnosis"
ROLE_TASK = "task"
ROLE_CONCLUSION = "conclusion"
RELATION_INFLUENCE = "influence"
RELATION_SKIP = "skip"
DETERMINISTIC_FALLBACK_REASON = "deterministic_fallback"
OMITTED_GAP_REASON = "not_in_selected_ao_range"
MAX_CRITIC_ROUNDS = 3
INDEX_VERSION = 1

BlockRole = Literal["task", "orchestrator", "agent_action", "conclusion"]
Purpose = Literal["post_run_index", "failure_diagnosis"]
ClaimStatus = Literal["hypothesis", "supported", "contradicted", "insufficient"]
ConstructionMode = Literal["adaptive", "fallback"]
Severity = Literal["high", "low"]


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
    if not allow_empty and not value.strip():
        raise ValueError(f"{field} must be a non-empty str")
    return value


def _require_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an int")
    return value


def _require_choice(value: Any, field: str, choices: frozenset[str]) -> str:
    text = _require_str(value, field)
    if text not in choices:
        raise ValueError(f"{field} must be one of {sorted(choices)}, got {text!r}")
    return text


@dataclass(frozen=True)
class SegmentRef:
    """Single segment a block is grounded in."""

    segment_id: str
    session_id: str

    def __post_init__(self) -> None:
        _require_str(self.segment_id, "segment_ref.segment_id")
        _require_str(self.session_id, "segment_ref.session_id")

    def to_dict(self) -> dict[str, Any]:
        return {"segment_id": self.segment_id, "session_id": self.session_id}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SegmentRef:
        payload = _require_mapping(data, "segment_ref")
        return cls(
            segment_id=_require_str(payload.get("segment_id"), "segment_ref.segment_id"),
            session_id=_require_str(payload.get("session_id"), "segment_ref.session_id"),
        )


@dataclass(frozen=True)
class SourceRef:
    """Readable pointer from a block back to one AO (ao_id + sequence)."""

    ao_id: str
    sequence: int
    participant_id: str
    segment_id: str
    session_id: str
    content_hash: str

    def __post_init__(self) -> None:
        _require_str(self.ao_id, "source_ref.ao_id")
        _require_int(self.sequence, "source_ref.sequence")
        _require_str(self.participant_id, "source_ref.participant_id")
        _require_str(self.segment_id, "source_ref.segment_id")
        _require_str(self.session_id, "source_ref.session_id")
        _require_str(self.content_hash, "source_ref.content_hash")

    def to_dict(self) -> dict[str, Any]:
        return {
            "ao_id": self.ao_id,
            "sequence": self.sequence,
            "participant_id": self.participant_id,
            "segment_id": self.segment_id,
            "session_id": self.session_id,
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SourceRef:
        payload = _require_mapping(data, "source_ref")
        return cls(
            ao_id=_require_str(payload.get("ao_id"), "source_ref.ao_id"),
            sequence=_require_int(payload.get("sequence"), "source_ref.sequence"),
            participant_id=_require_str(
                payload.get("participant_id"), "source_ref.participant_id"
            ),
            segment_id=_require_str(payload.get("segment_id"), "source_ref.segment_id"),
            session_id=_require_str(payload.get("session_id"), "source_ref.session_id"),
            content_hash=_require_str(payload.get("content_hash"), "source_ref.content_hash"),
        )


@dataclass(frozen=True)
class InfluenceEvidenceRef:
    """Nine-field provenance, same shape as causal-experiences ``EvidenceRef``."""

    ao_id: str
    source_session_id: str
    source_archive_id: str | None
    archive_commit_watermark: str | None
    source_sequence: int
    source_read_snapshot_watermark: str
    evidence_role: str
    evidence_content_hash: str
    captured_state: str

    def __post_init__(self) -> None:
        _require_str(self.ao_id, "evidence_ref.ao_id")
        _require_str(self.source_session_id, "evidence_ref.source_session_id")
        if self.source_archive_id is not None:
            _require_str(self.source_archive_id, "evidence_ref.source_archive_id")
        if self.archive_commit_watermark is not None:
            _require_str(
                self.archive_commit_watermark, "evidence_ref.archive_commit_watermark"
            )
        _require_int(self.source_sequence, "evidence_ref.source_sequence")
        _require_str(
            self.source_read_snapshot_watermark,
            "evidence_ref.source_read_snapshot_watermark",
        )
        _require_str(self.evidence_role, "evidence_ref.evidence_role")
        _require_str(self.evidence_content_hash, "evidence_ref.evidence_content_hash")
        _require_str(self.captured_state, "evidence_ref.captured_state")

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
    def from_dict(cls, data: dict[str, Any]) -> InfluenceEvidenceRef:
        payload = _require_mapping(data, "evidence_ref")
        archive_id = payload.get("source_archive_id")
        watermark = payload.get("archive_commit_watermark")
        return cls(
            ao_id=_require_str(payload.get("ao_id"), "evidence_ref.ao_id"),
            source_session_id=_require_str(
                payload.get("source_session_id"), "evidence_ref.source_session_id"
            ),
            source_archive_id=(
                None
                if archive_id is None
                else _require_str(archive_id, "evidence_ref.source_archive_id")
            ),
            archive_commit_watermark=(
                None
                if watermark is None
                else _require_str(watermark, "evidence_ref.archive_commit_watermark")
            ),
            source_sequence=_require_int(
                payload.get("source_sequence"), "evidence_ref.source_sequence"
            ),
            source_read_snapshot_watermark=_require_str(
                payload.get("source_read_snapshot_watermark"),
                "evidence_ref.source_read_snapshot_watermark",
            ),
            evidence_role=_require_str(payload.get("evidence_role"), "evidence_ref.evidence_role"),
            evidence_content_hash=_require_str(
                payload.get("evidence_content_hash"), "evidence_ref.evidence_content_hash"
            ),
            captured_state=_require_str(payload.get("captured_state"), "evidence_ref.captured_state"),
        )


@dataclass(frozen=True)
class AORange:
    """Inclusive AO sequence interval."""

    start_seq: int
    end_seq: int
    segment_id: str | None = None

    def __post_init__(self) -> None:
        start = _require_int(self.start_seq, "ao_range.start_seq")
        end = _require_int(self.end_seq, "ao_range.end_seq")
        if start > end:
            raise ValueError("ao range must be non-empty and contiguous")
        if self.segment_id is not None:
            _require_str(self.segment_id, "ao_range.segment_id")

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_seq": self.start_seq,
            "end_seq": self.end_seq,
            "segment_id": self.segment_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AORange:
        payload = _require_mapping(data, "ao_range")
        segment_id = payload.get("segment_id")
        return cls(
            start_seq=_require_int(payload.get("start_seq"), "ao_range.start_seq"),
            end_seq=_require_int(payload.get("end_seq"), "ao_range.end_seq"),
            segment_id=None if segment_id is None else _require_str(segment_id, "ao_range.segment_id"),
        )

    def overlaps(self, start_seq: int, end_seq: int) -> bool:
        return self.start_seq <= end_seq and start_seq <= self.end_seq


@dataclass(frozen=True)
class OmittedRange:
    """AO interval left out of the view. ``reason`` may be empty until validation."""

    start_seq: int
    end_seq: int
    reason: str
    segment_id: str | None = None

    def __post_init__(self) -> None:
        start = _require_int(self.start_seq, "omitted_range.start_seq")
        end = _require_int(self.end_seq, "omitted_range.end_seq")
        if start > end:
            raise ValueError("omitted range must be non-empty")
        if not isinstance(self.reason, str):
            raise ValueError("omitted_range.reason must be a str")
        if self.segment_id is not None:
            _require_str(self.segment_id, "omitted_range.segment_id")

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_seq": self.start_seq,
            "end_seq": self.end_seq,
            "reason": self.reason,
            "segment_id": self.segment_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OmittedRange:
        payload = _require_mapping(data, "omitted_range")
        segment_id = payload.get("segment_id")
        reason = payload.get("reason")
        if not isinstance(reason, str):
            raise ValueError("omitted_range.reason must be a str")
        return cls(
            start_seq=_require_int(payload.get("start_seq"), "omitted_range.start_seq"),
            end_seq=_require_int(payload.get("end_seq"), "omitted_range.end_seq"),
            reason=reason,
            segment_id=(
                None if segment_id is None else _require_str(segment_id, "omitted_range.segment_id")
            ),
        )


@dataclass(frozen=True)
class InfluenceBlock:
    """Action-level unit: one participant, one segment, one contiguous AO span."""

    block_id: str
    role: BlockRole
    participant_id: str
    segment_ref: SegmentRef
    ao_start_seq: int
    ao_end_seq: int
    summary: str
    input: str
    output: str
    authorship: str
    granularity_reason: str
    source_content_hashes: tuple[str, ...]
    source_refs: tuple[SourceRef, ...]

    def __post_init__(self) -> None:
        ordered = tuple(sorted(self.source_refs, key=lambda ref: (ref.sequence, ref.ao_id)))
        object.__setattr__(self, "source_refs", ordered)
        object.__setattr__(self, "source_content_hashes", tuple(self.source_content_hashes))
        self.validate()

    def validate(self) -> None:
        """Reject cross-participant, cross-segment, empty, or gapped spans."""
        _require_str(self.block_id, "block_id")
        _require_choice(self.role, "role", BLOCK_ROLES)
        _require_str(self.participant_id, "participant_id")
        if not isinstance(self.segment_ref, SegmentRef):
            raise ValueError("segment_ref must be a single SegmentRef")
        _require_str(self.summary, "summary", allow_empty=True)
        _require_str(self.input, "input", allow_empty=True)
        _require_str(self.output, "output", allow_empty=True)
        _require_str(self.authorship, "authorship", allow_empty=True)
        if not isinstance(self.granularity_reason, str) or not self.granularity_reason.strip():
            raise ValueError("granularity_reason is required")
        start = _require_int(self.ao_start_seq, "ao_start_seq")
        end = _require_int(self.ao_end_seq, "ao_end_seq")
        if start > end:
            raise ValueError("AO range must be contiguous and non-empty")
        if not self.source_refs:
            raise ValueError("AO range must be contiguous and non-empty")
        participants = {ref.participant_id for ref in self.source_refs}
        if participants != {self.participant_id}:
            raise ValueError("InfluenceBlock must belong to a single participant")
        segments = {ref.segment_id for ref in self.source_refs}
        if segments != {self.segment_ref.segment_id}:
            raise ValueError("InfluenceBlock must reference a single segment")
        sessions = {ref.session_id for ref in self.source_refs}
        if sessions != {self.segment_ref.session_id}:
            raise ValueError("InfluenceBlock must stay within a single session")
        sequences = [ref.sequence for ref in self.source_refs]
        if len(sequences) != len(set(sequences)):
            raise ValueError("AO range must be contiguous and non-empty")
        ordered = sorted(sequences)
        expected = list(range(ordered[0], ordered[-1] + 1))
        if ordered != expected or ordered[0] != start or ordered[-1] != end:
            raise ValueError("AO range must be contiguous and non-empty")
        hashes = tuple(ref.content_hash for ref in self.source_refs)
        if tuple(self.source_content_hashes) != hashes:
            raise ValueError("source_content_hashes must align with source refs in sequence order")

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "role": self.role,
            "participant_id": self.participant_id,
            "segment_ref": self.segment_ref.to_dict(),
            "ao_start_seq": self.ao_start_seq,
            "ao_end_seq": self.ao_end_seq,
            "summary": self.summary,
            "input": self.input,
            "output": self.output,
            "authorship": self.authorship,
            "granularity_reason": self.granularity_reason,
            "source_content_hashes": list(self.source_content_hashes),
            "source_refs": [ref.to_dict() for ref in self.source_refs],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InfluenceBlock:
        payload = _require_mapping(data, "block")
        refs = tuple(
            SourceRef.from_dict(_require_mapping(item, "source_ref"))
            for item in payload.get("source_refs") or []
        )
        hashes = payload.get("source_content_hashes")
        if not isinstance(hashes, list):
            raise ValueError("source_content_hashes must be a list")
        return cls(
            block_id=_require_str(payload.get("block_id"), "block_id"),
            role=_require_choice(payload.get("role"), "role", BLOCK_ROLES),  # type: ignore[arg-type]
            participant_id=_require_str(payload.get("participant_id"), "participant_id"),
            segment_ref=SegmentRef.from_dict(
                _require_mapping(payload.get("segment_ref"), "segment_ref")
            ),
            ao_start_seq=_require_int(payload.get("ao_start_seq"), "ao_start_seq"),
            ao_end_seq=_require_int(payload.get("ao_end_seq"), "ao_end_seq"),
            summary=_require_str(payload.get("summary"), "summary", allow_empty=True),
            input=_require_str(payload.get("input"), "input", allow_empty=True),
            output=_require_str(payload.get("output"), "output", allow_empty=True),
            authorship=_require_str(payload.get("authorship"), "authorship", allow_empty=True),
            granularity_reason=_require_str(
                payload.get("granularity_reason"), "granularity_reason", allow_empty=True
            ),
            source_content_hashes=tuple(str(item) for item in hashes),
            source_refs=refs,
        )


@dataclass(frozen=True)
class InfluenceClaim:
    """Directed hypothesis from an earlier block to a later block.

    ``status`` is the reader evidence verdict. ``critic_verdict`` is a separate
    column and never overwrites ``status``. New claims start as hypothesis.
    """

    claim_id: str
    source_block_id: str
    target_block_id: str
    carried_artifact: str
    downstream_effect: str
    relation_type: str
    source_evidence_refs: tuple[InfluenceEvidenceRef, ...] = ()
    target_evidence_refs: tuple[InfluenceEvidenceRef, ...] = ()
    status: ClaimStatus = CLAIM_STATUS_HYPOTHESIS
    critic_verdict: str | None = None

    def __post_init__(self) -> None:
        _require_str(self.claim_id, "claim_id")
        _require_str(self.source_block_id, "source_block_id")
        _require_str(self.target_block_id, "target_block_id")
        _require_str(self.carried_artifact, "carried_artifact")
        _require_str(self.downstream_effect, "downstream_effect")
        _require_str(self.relation_type, "relation_type")
        _require_choice(self.status, "status", CLAIM_STATUSES)
        if self.critic_verdict is not None:
            _require_str(self.critic_verdict, "critic_verdict", allow_empty=True)
        object.__setattr__(self, "source_evidence_refs", tuple(self.source_evidence_refs))
        object.__setattr__(self, "target_evidence_refs", tuple(self.target_evidence_refs))
        for ref in self.source_evidence_refs:
            if not isinstance(ref, InfluenceEvidenceRef):
                raise ValueError("source_evidence_refs must contain InfluenceEvidenceRef")
        for ref in self.target_evidence_refs:
            if not isinstance(ref, InfluenceEvidenceRef):
                raise ValueError("target_evidence_refs must contain InfluenceEvidenceRef")

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "source_block_id": self.source_block_id,
            "target_block_id": self.target_block_id,
            "carried_artifact": self.carried_artifact,
            "downstream_effect": self.downstream_effect,
            "relation_type": self.relation_type,
            "source_evidence_refs": [ref.to_dict() for ref in self.source_evidence_refs],
            "target_evidence_refs": [ref.to_dict() for ref in self.target_evidence_refs],
            "status": self.status,
            "critic_verdict": self.critic_verdict,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InfluenceClaim:
        payload = _require_mapping(data, "claim")
        source_refs = tuple(
            InfluenceEvidenceRef.from_dict(_require_mapping(item, "evidence_ref"))
            for item in payload.get("source_evidence_refs") or []
        )
        target_refs = tuple(
            InfluenceEvidenceRef.from_dict(_require_mapping(item, "evidence_ref"))
            for item in payload.get("target_evidence_refs") or []
        )
        verdict = payload.get("critic_verdict")
        return cls(
            claim_id=_require_str(payload.get("claim_id"), "claim_id"),
            source_block_id=_require_str(payload.get("source_block_id"), "source_block_id"),
            target_block_id=_require_str(payload.get("target_block_id"), "target_block_id"),
            carried_artifact=_require_str(payload.get("carried_artifact"), "carried_artifact"),
            downstream_effect=_require_str(payload.get("downstream_effect"), "downstream_effect"),
            relation_type=_require_str(payload.get("relation_type"), "relation_type"),
            source_evidence_refs=source_refs,
            target_evidence_refs=target_refs,
            status=_require_choice(  # type: ignore[arg-type]
                payload.get("status", CLAIM_STATUS_HYPOTHESIS), "status", CLAIM_STATUSES
            ),
            critic_verdict=None if verdict is None else _require_str(verdict, "critic_verdict", allow_empty=True),
        )


@dataclass(frozen=True)
class CoverageManifest:
    """Bounded coverage of a snapshot: included spans plus omitted spans with reasons."""

    total_ao_range: AORange
    inspected_refs: tuple[SourceRef, ...]
    included_ranges: tuple[AORange, ...]
    omitted_ranges: tuple[OmittedRange, ...]
    snapshot_watermark: str

    def __post_init__(self) -> None:
        if not isinstance(self.total_ao_range, AORange):
            raise ValueError("total_ao_range must be an AORange")
        _require_str(self.snapshot_watermark, "snapshot_watermark")
        object.__setattr__(self, "inspected_refs", tuple(self.inspected_refs))
        object.__setattr__(self, "included_ranges", tuple(self.included_ranges))
        object.__setattr__(self, "omitted_ranges", tuple(self.omitted_ranges))
        for ref in self.inspected_refs:
            if not isinstance(ref, SourceRef):
                raise ValueError("inspected_refs must contain SourceRef")
        for item in self.included_ranges:
            if not isinstance(item, AORange):
                raise ValueError("included_ranges must contain AORange")
        for item in self.omitted_ranges:
            if not isinstance(item, OmittedRange):
                raise ValueError("omitted_ranges must contain OmittedRange")

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_ao_range": self.total_ao_range.to_dict(),
            "inspected_refs": [ref.to_dict() for ref in self.inspected_refs],
            "included_ranges": [item.to_dict() for item in self.included_ranges],
            "omitted_ranges": [item.to_dict() for item in self.omitted_ranges],
            "snapshot_watermark": self.snapshot_watermark,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CoverageManifest:
        payload = _require_mapping(data, "coverage")
        return cls(
            total_ao_range=AORange.from_dict(
                _require_mapping(payload.get("total_ao_range"), "total_ao_range")
            ),
            inspected_refs=tuple(
                SourceRef.from_dict(_require_mapping(item, "source_ref"))
                for item in payload.get("inspected_refs") or []
            ),
            included_ranges=tuple(
                AORange.from_dict(_require_mapping(item, "ao_range"))
                for item in payload.get("included_ranges") or []
            ),
            omitted_ranges=tuple(
                OmittedRange.from_dict(_require_mapping(item, "omitted_range"))
                for item in payload.get("omitted_ranges") or []
            ),
            snapshot_watermark=_require_str(
                payload.get("snapshot_watermark"), "snapshot_watermark"
            ),
        )


@dataclass(frozen=True)
class Violation:
    """Deterministic structural finding. ``location`` points at the offending node."""

    code: str
    severity: Severity
    message: str
    location: str

    def __post_init__(self) -> None:
        _require_str(self.code, "violation.code")
        _require_choice(self.severity, "violation.severity", SEVERITIES)
        _require_str(self.message, "violation.message")
        _require_str(self.location, "violation.location")

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "location": self.location,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Violation:
        payload = _require_mapping(data, "violation")
        return cls(
            code=_require_str(payload.get("code"), "violation.code"),
            severity=_require_choice(  # type: ignore[arg-type]
                payload.get("severity"), "violation.severity", SEVERITIES
            ),
            message=_require_str(payload.get("message"), "violation.message"),
            location=_require_str(payload.get("location"), "violation.location"),
        )


@dataclass(frozen=True)
class CriticPatchRecord:
    """Provenance for one refiner round that addressed a violation."""

    round_index: int
    violation_code: str
    location: str
    severity: Severity
    note: str

    def __post_init__(self) -> None:
        index = _require_int(self.round_index, "patch.round_index")
        if index < 1 or index > MAX_CRITIC_ROUNDS:
            raise ValueError(f"patch.round_index must be in 1..{MAX_CRITIC_ROUNDS}")
        _require_str(self.violation_code, "patch.violation_code")
        _require_str(self.location, "patch.location")
        _require_choice(self.severity, "patch.severity", SEVERITIES)
        _require_str(self.note, "patch.note", allow_empty=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_index": self.round_index,
            "violation_code": self.violation_code,
            "location": self.location,
            "severity": self.severity,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CriticPatchRecord:
        payload = _require_mapping(data, "patch")
        return cls(
            round_index=_require_int(payload.get("round_index"), "patch.round_index"),
            violation_code=_require_str(payload.get("violation_code"), "patch.violation_code"),
            location=_require_str(payload.get("location"), "patch.location"),
            severity=_require_choice(  # type: ignore[arg-type]
                payload.get("severity"), "patch.severity", SEVERITIES
            ),
            note=_require_str(payload.get("note", ""), "patch.note", allow_empty=True),
        )


def compute_content_hash(
    blocks: Sequence[InfluenceBlock],
    claims: Sequence[InfluenceClaim],
    coverage: CoverageManifest,
) -> str:
    """sha256 of canonical JSON for blocks, claims, and coverage."""
    payload = {
        "blocks": [block.to_dict() for block in blocks],
        "claims": [claim.to_dict() for claim in claims],
        "coverage": coverage.to_dict(),
    }
    return _sha256_text(_canonical_dumps(payload))


class InfluenceViewDraft:
    """Mutable view. Builder and refiner edits stop after three critic rounds."""

    def __init__(
        self,
        *,
        view_id: str,
        purpose: Purpose,
        session_id: str,
        task_run_id: str,
        coverage: CoverageManifest,
        blocks: Sequence[InfluenceBlock] | None = None,
        claims: Sequence[InfluenceClaim] | None = None,
        critic_rounds: int = 0,
        patch_provenance: Sequence[CriticPatchRecord] | None = None,
    ) -> None:
        self.view_id = _require_str(view_id, "view_id")
        self.purpose = _require_choice(purpose, "purpose", PURPOSES)
        self.session_id = _require_str(session_id, "session_id")
        self.task_run_id = _require_str(task_run_id, "task_run_id")
        if not isinstance(coverage, CoverageManifest):
            raise ValueError("coverage must be a CoverageManifest")
        self.coverage = coverage
        rounds = _require_int(critic_rounds, "critic_rounds")
        if rounds < 0 or rounds > MAX_CRITIC_ROUNDS:
            raise ValueError(f"critic_rounds must be in 0..{MAX_CRITIC_ROUNDS}")
        self.critic_rounds = rounds
        self.blocks: list[InfluenceBlock] = []
        self.claims: list[InfluenceClaim] = []
        self.patch_provenance: list[CriticPatchRecord] = []
        for block in blocks or ():
            self._append_block(block)
        for claim in claims or ():
            self._append_claim(claim, enforce_hypothesis=False)
        for record in patch_provenance or ():
            if not isinstance(record, CriticPatchRecord):
                raise ValueError("patch_provenance must contain CriticPatchRecord")
            self.patch_provenance.append(record)

    def _ensure_editable(self) -> None:
        if self.critic_rounds >= MAX_CRITIC_ROUNDS:
            raise ValueError(
                f"critic_rounds limit is {MAX_CRITIC_ROUNDS}; refusing further edits"
            )

    def _append_block(self, block: InfluenceBlock) -> None:
        if not isinstance(block, InfluenceBlock):
            raise ValueError("block must be an InfluenceBlock")
        if any(existing.block_id == block.block_id for existing in self.blocks):
            raise ValueError(f"duplicate block_id {block.block_id}")
        self.blocks.append(block)

    def _append_claim(self, claim: InfluenceClaim, *, enforce_hypothesis: bool) -> None:
        if not isinstance(claim, InfluenceClaim):
            raise ValueError("claim must be an InfluenceClaim")
        if enforce_hypothesis and claim.status != CLAIM_STATUS_HYPOTHESIS:
            raise ValueError("new claims must start as hypothesis")
        if any(existing.claim_id == claim.claim_id for existing in self.claims):
            raise ValueError(f"duplicate claim_id {claim.claim_id}")
        self.claims.append(claim)

    def add_block(self, block: InfluenceBlock) -> None:
        self._ensure_editable()
        self._append_block(block)

    def add_claim(self, claim: InfluenceClaim) -> None:
        self._ensure_editable()
        self._append_claim(claim, enforce_hypothesis=True)

    def apply_critic_patch(
        self,
        violation: Violation,
        *,
        note: str = "",
        blocks: Sequence[InfluenceBlock] | None = None,
        claims: Sequence[InfluenceClaim] | None = None,
        coverage: CoverageManifest | None = None,
    ) -> None:
        """Apply one refiner patch and record which violation it addresses."""
        if not isinstance(violation, Violation):
            raise ValueError("violation must be a Violation")
        self._ensure_editable()
        next_blocks = list(self.blocks)
        next_claims = list(self.claims)
        next_coverage = self.coverage
        if blocks is not None:
            next_blocks = []
            seen: set[str] = set()
            for block in blocks:
                if not isinstance(block, InfluenceBlock):
                    raise ValueError("block must be an InfluenceBlock")
                if block.block_id in seen:
                    raise ValueError(f"duplicate block_id {block.block_id}")
                seen.add(block.block_id)
                next_blocks.append(block)
        if claims is not None:
            next_claims = []
            seen_claims: set[str] = set()
            for claim in claims:
                if not isinstance(claim, InfluenceClaim):
                    raise ValueError("claim must be an InfluenceClaim")
                if claim.status != CLAIM_STATUS_HYPOTHESIS:
                    raise ValueError("new claims must start as hypothesis")
                if claim.claim_id in seen_claims:
                    raise ValueError(f"duplicate claim_id {claim.claim_id}")
                seen_claims.add(claim.claim_id)
                next_claims.append(claim)
        if coverage is not None:
            if not isinstance(coverage, CoverageManifest):
                raise ValueError("coverage must be a CoverageManifest")
            next_coverage = coverage
        self.blocks = next_blocks
        self.claims = next_claims
        self.coverage = next_coverage
        self.critic_rounds += 1
        self.patch_provenance.append(
            CriticPatchRecord(
                round_index=self.critic_rounds,
                violation_code=violation.code,
                location=violation.location,
                severity=violation.severity,
                note=note,
            )
        )

    def freeze(
        self,
        *,
        construction_mode: ConstructionMode = CONSTRUCTION_MODE_ADAPTIVE,
        revision_id: str | None = None,
        frozen_at: str | None = None,
    ) -> InfluenceViewRevision:
        mode = _require_choice(construction_mode, "construction_mode", CONSTRUCTION_MODES)
        stamp = _utc_now_iso() if frozen_at is None else _require_str(frozen_at, "frozen_at")
        identity = revision_id or uuid.uuid4().hex
        blocks = tuple(self.blocks)
        claims = tuple(self.claims)
        return InfluenceViewRevision(
            revision_id=_require_str(identity, "revision_id"),
            view_id=self.view_id,
            purpose=self.purpose,  # type: ignore[arg-type]
            session_id=self.session_id,
            task_run_id=self.task_run_id,
            blocks=blocks,
            claims=claims,
            coverage=self.coverage,
            critic_rounds=self.critic_rounds,
            patch_provenance=tuple(self.patch_provenance),
            construction_mode=mode,  # type: ignore[arg-type]
            content_hash=compute_content_hash(blocks, claims, self.coverage),
            frozen_at=stamp,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "view_id": self.view_id,
            "purpose": self.purpose,
            "session_id": self.session_id,
            "task_run_id": self.task_run_id,
            "blocks": [block.to_dict() for block in self.blocks],
            "claims": [claim.to_dict() for claim in self.claims],
            "coverage": self.coverage.to_dict(),
            "critic_rounds": self.critic_rounds,
            "patch_provenance": [item.to_dict() for item in self.patch_provenance],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InfluenceViewDraft:
        payload = _require_mapping(data, "draft")
        return cls(
            view_id=_require_str(payload.get("view_id"), "view_id"),
            purpose=_require_choice(payload.get("purpose"), "purpose", PURPOSES),  # type: ignore[arg-type]
            session_id=_require_str(payload.get("session_id"), "session_id"),
            task_run_id=_require_str(payload.get("task_run_id"), "task_run_id"),
            coverage=CoverageManifest.from_dict(
                _require_mapping(payload.get("coverage"), "coverage")
            ),
            blocks=tuple(
                InfluenceBlock.from_dict(_require_mapping(item, "block"))
                for item in payload.get("blocks") or []
            ),
            claims=tuple(
                InfluenceClaim.from_dict(_require_mapping(item, "claim"))
                for item in payload.get("claims") or []
            ),
            critic_rounds=_require_int(payload.get("critic_rounds", 0), "critic_rounds"),
            patch_provenance=tuple(
                CriticPatchRecord.from_dict(_require_mapping(item, "patch"))
                for item in payload.get("patch_provenance") or []
            ),
        )


@dataclass(frozen=True)
class InfluenceViewRevision:
    """Immutable freeze of a draft. Content hash covers blocks, claims, and coverage."""

    revision_id: str
    view_id: str
    purpose: Purpose
    session_id: str
    task_run_id: str
    blocks: tuple[InfluenceBlock, ...]
    claims: tuple[InfluenceClaim, ...]
    coverage: CoverageManifest
    critic_rounds: int
    patch_provenance: tuple[CriticPatchRecord, ...]
    construction_mode: ConstructionMode
    content_hash: str
    frozen_at: str

    def __post_init__(self) -> None:
        _require_str(self.revision_id, "revision_id")
        _require_str(self.view_id, "view_id")
        _require_choice(self.purpose, "purpose", PURPOSES)
        _require_str(self.session_id, "session_id")
        _require_str(self.task_run_id, "task_run_id")
        _require_choice(self.construction_mode, "construction_mode", CONSTRUCTION_MODES)
        _require_str(self.content_hash, "content_hash")
        _require_str(self.frozen_at, "frozen_at")
        rounds = _require_int(self.critic_rounds, "critic_rounds")
        if rounds < 0 or rounds > MAX_CRITIC_ROUNDS:
            raise ValueError(f"critic_rounds must be in 0..{MAX_CRITIC_ROUNDS}")
        if not isinstance(self.coverage, CoverageManifest):
            raise ValueError("coverage must be a CoverageManifest")
        object.__setattr__(self, "blocks", tuple(self.blocks))
        object.__setattr__(self, "claims", tuple(self.claims))
        object.__setattr__(self, "patch_provenance", tuple(self.patch_provenance))

    def to_dict(self) -> dict[str, Any]:
        return {
            "revision_id": self.revision_id,
            "view_id": self.view_id,
            "purpose": self.purpose,
            "session_id": self.session_id,
            "task_run_id": self.task_run_id,
            "blocks": [block.to_dict() for block in self.blocks],
            "claims": [claim.to_dict() for claim in self.claims],
            "coverage": self.coverage.to_dict(),
            "critic_rounds": self.critic_rounds,
            "patch_provenance": [item.to_dict() for item in self.patch_provenance],
            "construction_mode": self.construction_mode,
            "content_hash": self.content_hash,
            "frozen_at": self.frozen_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InfluenceViewRevision:
        payload = _require_mapping(data, "revision")
        return cls(
            revision_id=_require_str(payload.get("revision_id"), "revision_id"),
            view_id=_require_str(payload.get("view_id"), "view_id"),
            purpose=_require_choice(payload.get("purpose"), "purpose", PURPOSES),  # type: ignore[arg-type]
            session_id=_require_str(payload.get("session_id"), "session_id"),
            task_run_id=_require_str(payload.get("task_run_id"), "task_run_id"),
            blocks=tuple(
                InfluenceBlock.from_dict(_require_mapping(item, "block"))
                for item in payload.get("blocks") or []
            ),
            claims=tuple(
                InfluenceClaim.from_dict(_require_mapping(item, "claim"))
                for item in payload.get("claims") or []
            ),
            coverage=CoverageManifest.from_dict(
                _require_mapping(payload.get("coverage"), "coverage")
            ),
            critic_rounds=_require_int(payload.get("critic_rounds", 0), "critic_rounds"),
            patch_provenance=tuple(
                CriticPatchRecord.from_dict(_require_mapping(item, "patch"))
                for item in payload.get("patch_provenance") or []
            ),
            construction_mode=_require_choice(  # type: ignore[arg-type]
                payload.get("construction_mode"), "construction_mode", CONSTRUCTION_MODES
            ),
            content_hash=_require_str(payload.get("content_hash"), "content_hash"),
            frozen_at=_require_str(payload.get("frozen_at"), "frozen_at"),
        )


def _view_parts(
    view: InfluenceViewDraft | InfluenceViewRevision,
) -> tuple[str, str, Sequence[InfluenceBlock], Sequence[InfluenceClaim], CoverageManifest]:
    return view.purpose, view.session_id, view.blocks, view.claims, view.coverage


def _readable_source_violations(blocks: Sequence[InfluenceBlock]) -> list[Violation]:
    findings: list[Violation] = []
    for block in blocks:
        if not block.source_refs:
            findings.append(
                Violation(
                    code="missing_source_ref",
                    severity="high",
                    message="block has no readable source ref",
                    location=f"block:{block.block_id}",
                )
            )
            continue
        for index, ref in enumerate(block.source_refs):
            if not ref.ao_id.strip() or not ref.content_hash.strip():
                findings.append(
                    Violation(
                        code="unreadable_source_ref",
                        severity="high",
                        message="source ref must expose ao_id and content hash",
                        location=f"block:{block.block_id}/source_refs[{index}]",
                    )
                )
        if not block.segment_ref.segment_id.strip():
            findings.append(
                Violation(
                    code="unreadable_source_ref",
                    severity="high",
                    message="block segment ref is empty",
                    location=f"block:{block.block_id}",
                )
            )
    return findings


def _temporal_and_cycle_violations(
    blocks: Sequence[InfluenceBlock],
    claims: Sequence[InfluenceClaim],
) -> list[Violation]:
    by_id = {block.block_id: block for block in blocks}
    findings: list[Violation] = []
    graph: dict[str, list[tuple[str, InfluenceClaim]]] = {}
    nodes: set[str] = set()
    for claim in claims:
        source = by_id.get(claim.source_block_id)
        target = by_id.get(claim.target_block_id)
        if source is None or target is None:
            findings.append(
                Violation(
                    code="dangling_claim",
                    severity="high",
                    message="claim endpoints must name blocks in this view",
                    location=f"claim:{claim.claim_id}",
                )
            )
        elif source.ao_end_seq >= target.ao_start_seq:
            findings.append(
                Violation(
                    code="temporal_order",
                    severity="high",
                    message="source AO interval must be strictly before the target interval",
                    location=f"claim:{claim.claim_id}",
                )
            )
        graph.setdefault(claim.source_block_id, []).append((claim.target_block_id, claim))
        nodes.add(claim.source_block_id)
        nodes.add(claim.target_block_id)

    white, gray, black = 0, 1, 2
    color = {node: white for node in nodes}

    def walk(node: str) -> None:
        color[node] = gray
        for nxt, claim in graph.get(node, ()):
            if color.get(nxt, white) == gray:
                findings.append(
                    Violation(
                        code="cycle",
                        severity="high",
                        message="influence view must be acyclic",
                        location=f"claim:{claim.claim_id}",
                    )
                )
            elif color.get(nxt, white) == white:
                walk(nxt)
        color[node] = black

    for node in sorted(nodes):
        if color[node] == white:
            walk(node)
    return findings


def _weak_component(root_id: str, claims: Sequence[InfluenceClaim]) -> set[str]:
    undirected: dict[str, set[str]] = {}
    for claim in claims:
        undirected.setdefault(claim.source_block_id, set()).add(claim.target_block_id)
        undirected.setdefault(claim.target_block_id, set()).add(claim.source_block_id)
    seen: set[str] = set()
    stack = [root_id]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(undirected.get(node, set()) - seen)
    return seen


def _diagnosis_connectivity_violations(
    purpose: str,
    blocks: Sequence[InfluenceBlock],
    claims: Sequence[InfluenceClaim],
) -> list[Violation]:
    if purpose != PURPOSE_FAILURE_DIAGNOSIS:
        return []
    findings: list[Violation] = []
    roots = [block for block in blocks if block.role == ROLE_TASK]
    sinks = [block for block in blocks if block.role == ROLE_CONCLUSION]
    if len(roots) != 1:
        findings.append(
            Violation(
                code="missing_task_root",
                severity="high",
                message="failure_diagnosis view requires exactly one virtual task root",
                location="view",
            )
        )
    if len(sinks) != 1:
        findings.append(
            Violation(
                code="missing_outcome_sink",
                severity="high",
                message="failure_diagnosis view requires exactly one observed-outcome sink",
                location="view",
            )
        )
    if len(roots) != 1 or len(sinks) != 1:
        return findings
    root = roots[0]
    sink = sinks[0]
    component = _weak_component(root.block_id, claims)
    if sink.block_id not in component:
        findings.append(
            Violation(
                code="root_sink_not_weakly_connected",
                severity="high",
                message="virtual task root and observed-outcome sink are not weakly connected",
                location="view",
            )
        )
        return findings
    for block in blocks:
        if block.block_id not in component:
            findings.append(
                Violation(
                    code="block_outside_root_sink_subgraph",
                    severity="high",
                    message="included block is outside the root-to-sink weak subgraph",
                    location=f"block:{block.block_id}",
                )
            )
    return findings


def _ranges_overlap_gap(omitted: OmittedRange, start_seq: int, end_seq: int) -> bool:
    if start_seq > end_seq:
        return False
    return omitted.start_seq <= end_seq and start_seq <= omitted.end_seq


def _coverage_violations(
    blocks: Sequence[InfluenceBlock],
    claims: Sequence[InfluenceClaim],
    coverage: CoverageManifest,
) -> list[Violation]:
    findings: list[Violation] = []
    if not coverage.snapshot_watermark.strip():
        findings.append(
            Violation(
                code="coverage_incomplete",
                severity="high",
                message="coverage manifest requires a snapshot watermark",
                location="coverage.snapshot_watermark",
            )
        )
    for index, omitted in enumerate(coverage.omitted_ranges):
        if not omitted.reason.strip():
            findings.append(
                Violation(
                    code="omitted_range_missing_reason",
                    severity="high",
                    message="every omitted range requires a reason",
                    location=f"coverage.omitted_ranges[{index}]",
                )
            )
    by_id = {block.block_id: block for block in blocks}
    for claim in claims:
        source = by_id.get(claim.source_block_id)
        target = by_id.get(claim.target_block_id)
        if source is None or target is None:
            continue
        gap_start = source.ao_end_seq + 1
        gap_end = target.ao_start_seq - 1
        crosses = any(
            _ranges_overlap_gap(omitted, gap_start, gap_end) for omitted in coverage.omitted_ranges
        )
        if not crosses:
            continue
        if claim.relation_type != RELATION_SKIP:
            findings.append(
                Violation(
                    code="omitted_range_skip_required",
                    severity="high",
                    message="a claim that crosses an omitted range must use relation_type skip",
                    location=f"claim:{claim.claim_id}",
                )
            )
        if not claim.source_evidence_refs or not claim.target_evidence_refs:
            findings.append(
                Violation(
                    code="skip_missing_evidence",
                    severity="high",
                    message="a skip across an omitted range must cite evidence on both ends",
                    location=f"claim:{claim.claim_id}",
                )
            )
    return findings


def validate_view(draft: InfluenceViewDraft | InfluenceViewRevision) -> list[Violation]:
    """Server-side structural checks. Semantic critic verdicts are out of scope."""
    purpose, session_id, blocks, claims, coverage = _view_parts(draft)
    findings: list[Violation] = []
    for block in blocks:
        if block.segment_ref.session_id != session_id:
            findings.append(
                Violation(
                    code="session_mismatch",
                    severity="high",
                    message="block segment session must match the view session",
                    location=f"block:{block.block_id}",
                )
            )
    findings.extend(_readable_source_violations(blocks))
    findings.extend(_temporal_and_cycle_violations(blocks, claims))
    findings.extend(_diagnosis_connectivity_violations(purpose, blocks, claims))
    findings.extend(_coverage_violations(blocks, claims, coverage))
    return findings


def _as_record(item: Mapping[str, Any], default_session: str) -> SourceRef:
    payload = _require_mapping(item, "ao_record")
    session_id = payload.get("session_id", default_session)
    return SourceRef(
        ao_id=_require_str(payload.get("ao_id"), "ao_record.ao_id"),
        sequence=_require_int(payload.get("sequence"), "ao_record.sequence"),
        participant_id=_require_str(payload.get("participant_id"), "ao_record.participant_id"),
        segment_id=_require_str(payload.get("segment_id"), "ao_record.segment_id"),
        session_id=_require_str(session_id, "ao_record.session_id"),
        content_hash=_require_str(payload.get("content_hash"), "ao_record.content_hash"),
    )


def _contiguous_runs(records: Sequence[SourceRef]) -> list[list[SourceRef]]:
    ordered = sorted(records, key=lambda ref: (ref.sequence, ref.ao_id))
    if not ordered:
        return []
    runs: list[list[SourceRef]] = [[ordered[0]]]
    for ref in ordered[1:]:
        previous = runs[-1][-1]
        if ref.sequence == previous.sequence + 1:
            runs[-1].append(ref)
        elif ref.sequence == previous.sequence:
            raise ValueError("fallback AO sequences must be unique within a segment")
        else:
            raise ValueError(
                "fallback AO interval within a segment must be contiguous; "
                f"gap before sequence {ref.sequence}"
            )
    return runs


def build_minimal_fallback_view(session: Mapping[str, Any]) -> InfluenceViewRevision:
    """One block per segment over the given AO interval, with no claims.

    ``granularity_reason`` is always ``deterministic_fallback`` and
    ``construction_mode`` is ``fallback``.
    """
    payload = _require_mapping(dict(session), "session")
    session_id = _require_str(payload.get("session_id"), "session_id")
    task_run_id = _require_str(payload.get("task_run_id"), "task_run_id")
    purpose = _require_choice(
        payload.get("purpose", PURPOSE_POST_RUN_INDEX), "purpose", PURPOSES
    )
    raw_records = payload.get("ao_records")
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError("ao_records must be a non-empty list")
    records = [_as_record(_require_mapping(item, "ao_record"), session_id) for item in raw_records]
    if any(ref.session_id != session_id for ref in records):
        raise ValueError("fallback AO records must belong to the view session")

    grouped: dict[str, list[SourceRef]] = {}
    for ref in records:
        grouped.setdefault(ref.segment_id, []).append(ref)

    segment_ids = sorted(grouped, key=lambda segment_id: (min(ref.sequence for ref in grouped[segment_id]), segment_id))
    blocks: list[InfluenceBlock] = []
    included: list[AORange] = []
    for segment_id in segment_ids:
        segment_records = grouped[segment_id]
        participants = {ref.participant_id for ref in segment_records}
        if len(participants) != 1:
            raise ValueError("fallback segment must belong to a single participant")
        runs = _contiguous_runs(segment_records)
        if len(runs) != 1:
            raise ValueError("fallback segment must be one contiguous AO interval")
        run = runs[0]
        start = run[0].sequence
        end = run[-1].sequence
        participant_id = run[0].participant_id
        summary = str(payload.get("summary") or f"segment {segment_id} sequences {start}-{end}")
        blocks.append(
            InfluenceBlock(
                block_id=f"fallback-{segment_id}-{start}-{end}",
                role="agent_action",
                participant_id=participant_id,
                segment_ref=SegmentRef(segment_id=segment_id, session_id=session_id),
                ao_start_seq=start,
                ao_end_seq=end,
                summary=summary,
                input="",
                output="",
                authorship=participant_id,
                granularity_reason=DETERMINISTIC_FALLBACK_REASON,
                source_content_hashes=tuple(ref.content_hash for ref in run),
                source_refs=tuple(run),
            )
        )
        included.append(AORange(start_seq=start, end_seq=end, segment_id=segment_id))

    included_sorted = tuple(sorted(included, key=lambda item: (item.start_seq, item.segment_id or "")))
    omitted: list[OmittedRange] = []
    cursor = included_sorted[0].start_seq
    for item in included_sorted:
        if item.start_seq > cursor:
            omitted.append(
                OmittedRange(
                    start_seq=cursor,
                    end_seq=item.start_seq - 1,
                    reason=OMITTED_GAP_REASON,
                )
            )
        cursor = max(cursor, item.end_seq + 1)
    all_sequences = [ref.sequence for ref in records]
    total = AORange(start_seq=min(all_sequences), end_seq=max(all_sequences))
    watermark = payload.get("snapshot_watermark")
    if watermark is None:
        watermark = f"fallback:{session_id}:{total.end_seq}"
    coverage = CoverageManifest(
        total_ao_range=total,
        inspected_refs=tuple(sorted(records, key=lambda ref: (ref.sequence, ref.ao_id))),
        included_ranges=included_sorted,
        omitted_ranges=tuple(omitted),
        snapshot_watermark=_require_str(watermark, "snapshot_watermark"),
    )
    view_id = payload.get("view_id") or f"fallback-{session_id}-{purpose}"
    draft = InfluenceViewDraft(
        view_id=_require_str(view_id, "view_id"),
        purpose=purpose,  # type: ignore[arg-type]
        session_id=session_id,
        task_run_id=task_run_id,
        coverage=coverage,
        blocks=tuple(blocks),
    )
    frozen_at = payload.get("frozen_at")
    revision_id = payload.get("revision_id")
    return draft.freeze(
        construction_mode=CONSTRUCTION_MODE_FALLBACK,
        revision_id=None if revision_id is None else _require_str(revision_id, "revision_id"),
        frozen_at=None if frozen_at is None else _require_str(frozen_at, "frozen_at"),
    )


class ViewSupersedeIndex:
    """Latest frozen revision per ``(session_id, purpose)``, with history retained."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._history: dict[tuple[str, str], list[InfluenceViewRevision]] = {}
        self._load()

    def register(self, revision: InfluenceViewRevision) -> None:
        if not isinstance(revision, InfluenceViewRevision):
            raise ValueError("revision must be an InfluenceViewRevision")
        key = (revision.session_id, revision.purpose)
        with self._lock:
            self._history.setdefault(key, []).append(revision)
            self._save_locked()

    def latest(self, session_id: str, purpose: str) -> InfluenceViewRevision | None:
        purpose_value = _require_choice(purpose, "purpose", PURPOSES)
        with self._lock:
            items = self._history.get((session_id, purpose_value), [])
            if not items:
                return None
            return items[-1]

    def history(self, session_id: str, purpose: str) -> list[InfluenceViewRevision]:
        purpose_value = _require_choice(purpose, "purpose", PURPOSES)
        with self._lock:
            return list(self._history.get((session_id, purpose_value), []))

    def to_dict(self) -> dict[str, Any]:
        keys: list[dict[str, Any]] = []
        for (session_id, purpose), revisions in sorted(self._history.items()):
            keys.append(
                {
                    "session_id": session_id,
                    "purpose": purpose,
                    "revisions": [revision.to_dict() for revision in revisions],
                }
            )
        return {"version": INDEX_VERSION, "keys": keys}

    def _load(self) -> None:
        if not self.path.is_file():
            return
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        parsed = _require_mapping(payload, "index")
        history: dict[tuple[str, str], list[InfluenceViewRevision]] = {}
        for entry in parsed.get("keys") or []:
            item = _require_mapping(entry, "index.keys")
            session_id = _require_str(item.get("session_id"), "index.session_id")
            purpose = _require_choice(item.get("purpose"), "index.purpose", PURPOSES)
            revisions = [
                InfluenceViewRevision.from_dict(_require_mapping(raw, "revision"))
                for raw in item.get("revisions") or []
            ]
            history[(session_id, purpose)] = revisions
        self._history = history

    def _save_locked(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(self.path)

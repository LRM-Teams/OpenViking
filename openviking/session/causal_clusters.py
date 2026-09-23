# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""verified_similarity_edge and causal_experience_cluster store (ADR-0007, Q39–Q41).

File-level store beside fork nodes under ``<root>/causal-experiences/``.
Writes are gated by ``is_causal_mode_enabled``. Cluster IDs are minted once
and never recomputed; merge/split create successor clusters (Q40).
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

from openviking.session.causal_experiences import IDEMPOTENCY_DIRNAME, NAMESPACE_DIRNAME
from openviking_cli.utils.config.memory_config import is_causal_mode_enabled

# Helpers below are copied from openviking.session.causal_experiences (S5)
# rather than importing private underscore names or exporting new public aliases.

EDGES_DIRNAME = "edges"
CLUSTERS_DIRNAME = "clusters"
EDGES_INDEX_FILENAME = "edges-index.jsonl"
CLUSTERS_INDEX_FILENAME = "clusters-index.jsonl"

CRITERIA_DIMENSIONS = (
    "task_decision_type",
    "anchor_state",
    "failure_mode",
    "action_skill_role",
)
VERDICTS = frozenset(("consistent", "analogous", "insufficient", "dissimilar"))
POSITIVE_VERDICTS = frozenset(("consistent", "analogous"))
CLUSTER_STATUSES = frozenset(("active", "superseded", "split_closed"))
EDGE_DIRECTION: Literal["undirected"] = "undirected"

Verdict = Literal["consistent", "analogous", "insufficient", "dissimilar"]
ClusterStatus = Literal["active", "superseded", "split_closed"]


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


def _require_optional_str(value: Any, field: str) -> str | None:
    if value is None:
        return None
    return _require_str(value, field)


def _require_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an int")
    return value


def _require_confidence(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number")
    number = float(value)
    if number < 0.0 or number > 1.0:
        raise ValueError(f"{field} must be between 0 and 1 inclusive, got {value!r}")
    return number


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


def _require_str_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    out: list[str] = []
    for index, item in enumerate(value):
        out.append(_require_str(item, f"{field}[{index}]"))
    return out


def validate_endpoint(endpoint: Any, field: str) -> dict[str, str]:
    payload = _require_mapping(endpoint, field)
    return {
        "fork_node_id": _require_str(payload.get("fork_node_id"), f"{field}.fork_node_id"),
        "revision_id": _require_str(payload.get("revision_id"), f"{field}.revision_id"),
    }


def canonicalize_endpoints(
    endpoint_a: dict[str, str], endpoint_b: dict[str, str]
) -> tuple[dict[str, str], dict[str, str]]:
    """Lexicographic pair order so (a,b) and (b,a) share one canonical key."""
    key_a = (endpoint_a["fork_node_id"], endpoint_a["revision_id"])
    key_b = (endpoint_b["fork_node_id"], endpoint_b["revision_id"])
    if key_a <= key_b:
        return dict(endpoint_a), dict(endpoint_b)
    return dict(endpoint_b), dict(endpoint_a)


def canonical_pair_key(endpoint_a: dict[str, str], endpoint_b: dict[str, str]) -> str:
    left, right = canonicalize_endpoints(endpoint_a, endpoint_b)
    return _canonical_dumps({"a": left, "b": right})


def _validate_criteria(criteria: Any) -> dict[str, Any]:
    payload = _require_mapping(criteria, "criteria_per_dimension")
    missing = [key for key in CRITERIA_DIMENSIONS if key not in payload]
    if missing:
        raise ValueError(f"criteria_per_dimension missing keys: {missing}")
    extra = [key for key in payload if key not in CRITERIA_DIMENSIONS]
    if extra:
        raise ValueError(f"criteria_per_dimension unexpected keys: {extra}")
    return {key: payload[key] for key in CRITERIA_DIMENSIONS}


def _judgment_hash(judgment: SimilarityJudgment | dict[str, Any]) -> str:
    payload = judgment.to_dict() if isinstance(judgment, SimilarityJudgment) else dict(judgment)
    return _sha256_text(_canonical_dumps(payload))


def _hash_edge_fields(
    *,
    edge_id: str,
    endpoint_a: dict[str, str],
    endpoint_b: dict[str, str],
    judgment: dict[str, Any],
    direction: str,
    created_at: str,
) -> str:
    payload = {
        "created_at": created_at,
        "direction": direction,
        "edge_id": edge_id,
        "endpoint_a": endpoint_a,
        "endpoint_b": endpoint_b,
        "judgment": judgment,
    }
    return _sha256_text(_canonical_dumps(payload))


@dataclass(frozen=True)
class SimilarityJudgment:
    """Q39 four-dimension semantic judgment with four-valued verdict."""

    judged_by_model_version: str
    prompt_policy_version: str
    criteria_per_dimension: dict[str, Any]
    verdict: Verdict
    confidence: float
    rationale: str
    evidence_revision_hashes: list[str]
    judged_at: str
    diagnosis_run_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "judged_by_model_version",
            _require_str(self.judged_by_model_version, "judged_by_model_version"),
        )
        object.__setattr__(
            self,
            "prompt_policy_version",
            _require_str(self.prompt_policy_version, "prompt_policy_version"),
        )
        object.__setattr__(
            self, "criteria_per_dimension", _validate_criteria(self.criteria_per_dimension)
        )
        if self.verdict not in VERDICTS:
            raise ValueError(
                f"verdict must be one of {sorted(VERDICTS)}, got {self.verdict!r}"
            )
        object.__setattr__(self, "confidence", _require_confidence(self.confidence, "confidence"))
        object.__setattr__(self, "rationale", _require_str(self.rationale, "rationale"))
        object.__setattr__(
            self,
            "evidence_revision_hashes",
            _require_str_list(self.evidence_revision_hashes, "evidence_revision_hashes"),
        )
        object.__setattr__(self, "judged_at", _require_str(self.judged_at, "judged_at"))
        object.__setattr__(
            self, "diagnosis_run_id", _require_str(self.diagnosis_run_id, "diagnosis_run_id")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "judged_by_model_version": self.judged_by_model_version,
            "prompt_policy_version": self.prompt_policy_version,
            "criteria_per_dimension": dict(self.criteria_per_dimension),
            "verdict": self.verdict,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "evidence_revision_hashes": list(self.evidence_revision_hashes),
            "judged_at": self.judged_at,
            "diagnosis_run_id": self.diagnosis_run_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SimilarityJudgment:
        payload = _require_mapping(data, "judgment")
        return cls(
            judged_by_model_version=str(payload.get("judged_by_model_version") or ""),
            prompt_policy_version=str(payload.get("prompt_policy_version") or ""),
            criteria_per_dimension=dict(payload.get("criteria_per_dimension") or {}),
            verdict=payload.get("verdict"),  # type: ignore[arg-type]
            confidence=payload.get("confidence"),  # type: ignore[arg-type]
            rationale=str(payload.get("rationale") or ""),
            evidence_revision_hashes=list(payload.get("evidence_revision_hashes") or []),
            judged_at=str(payload.get("judged_at") or ""),
            diagnosis_run_id=str(payload.get("diagnosis_run_id") or ""),
        )


@dataclass(frozen=True)
class VerifiedSimilarityEdge:
    """Immutable undirected edge between two fork-node revisions (Q40)."""

    edge_id: str
    endpoint_a: dict[str, str]
    endpoint_b: dict[str, str]
    judgment: SimilarityJudgment
    direction: Literal["undirected"]
    created_at: str
    content_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "edge_id", _require_str(self.edge_id, "edge_id"))
        left = validate_endpoint(self.endpoint_a, "endpoint_a")
        right = validate_endpoint(self.endpoint_b, "endpoint_b")
        if (left["fork_node_id"], left["revision_id"]) == (
            right["fork_node_id"],
            right["revision_id"],
        ):
            raise ValueError("endpoint_a and endpoint_b must be distinct")
        left, right = canonicalize_endpoints(left, right)
        object.__setattr__(self, "endpoint_a", left)
        object.__setattr__(self, "endpoint_b", right)
        judgment = (
            self.judgment
            if isinstance(self.judgment, SimilarityJudgment)
            else SimilarityJudgment.from_dict(self.judgment)
        )
        object.__setattr__(self, "judgment", judgment)
        if self.direction != EDGE_DIRECTION:
            raise ValueError(f"direction must be {EDGE_DIRECTION!r}, got {self.direction!r}")
        object.__setattr__(self, "created_at", _require_str(self.created_at, "created_at"))
        expected = self.recompute_content_hash()
        stored = self.content_hash
        if stored:
            if stored != expected:
                raise ValueError(
                    f"content_hash mismatch: stored {stored!r} != computed {expected!r}"
                )
        else:
            object.__setattr__(self, "content_hash", expected)

    @property
    def negative(self) -> bool:
        return self.judgment.verdict not in POSITIVE_VERDICTS

    @property
    def judgment_hash(self) -> str:
        return _judgment_hash(self.judgment)

    @property
    def pair_key(self) -> str:
        return canonical_pair_key(self.endpoint_a, self.endpoint_b)

    def recompute_content_hash(self) -> str:
        return _hash_edge_fields(
            edge_id=self.edge_id,
            endpoint_a=self.endpoint_a,
            endpoint_b=self.endpoint_b,
            judgment=self.judgment.to_dict(),
            direction=self.direction,
            created_at=self.created_at,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "endpoint_a": dict(self.endpoint_a),
            "endpoint_b": dict(self.endpoint_b),
            "judgment": self.judgment.to_dict(),
            "direction": self.direction,
            "created_at": self.created_at,
            "content_hash": self.content_hash,
            "negative": self.negative,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VerifiedSimilarityEdge:
        payload = _require_mapping(data, "verified_similarity_edge")
        return cls(
            edge_id=str(payload.get("edge_id") or ""),
            endpoint_a=dict(payload.get("endpoint_a") or {}),
            endpoint_b=dict(payload.get("endpoint_b") or {}),
            judgment=SimilarityJudgment.from_dict(dict(payload.get("judgment") or {})),
            direction=payload.get("direction") or EDGE_DIRECTION,
            created_at=str(payload.get("created_at") or ""),
            content_hash=str(payload.get("content_hash") or ""),
        )


@dataclass(frozen=True)
class CausalCluster:
    """Center-based overlapping cluster over verified positive edges (Q40)."""

    cluster_id: str
    center_fork_node_id: str
    member_fork_node_ids: list[str]
    status: ClusterStatus
    created_at: str
    supersedes_cluster_ids: list[str]
    split_from_cluster_id: str | None
    closed_at: str | None
    version: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "cluster_id", _require_str(self.cluster_id, "cluster_id"))
        object.__setattr__(
            self,
            "center_fork_node_id",
            _require_str(self.center_fork_node_id, "center_fork_node_id"),
        )
        members = _require_str_list(self.member_fork_node_ids, "member_fork_node_ids")
        object.__setattr__(
            self, "member_fork_node_ids", _normalize_members(self.center_fork_node_id, members)
        )
        if self.status not in CLUSTER_STATUSES:
            raise ValueError(
                f"status must be one of {sorted(CLUSTER_STATUSES)}, got {self.status!r}"
            )
        object.__setattr__(self, "created_at", _require_str(self.created_at, "created_at"))
        object.__setattr__(
            self,
            "supersedes_cluster_ids",
            _require_str_list(self.supersedes_cluster_ids, "supersedes_cluster_ids"),
        )
        object.__setattr__(
            self,
            "split_from_cluster_id",
            _require_optional_str(self.split_from_cluster_id, "split_from_cluster_id"),
        )
        object.__setattr__(
            self, "closed_at", _require_optional_str(self.closed_at, "closed_at")
        )
        object.__setattr__(self, "version", _require_int(self.version, "version"))
        if self.version < 1:
            raise ValueError("version must be >= 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "center_fork_node_id": self.center_fork_node_id,
            "member_fork_node_ids": list(self.member_fork_node_ids),
            "status": self.status,
            "created_at": self.created_at,
            "supersedes_cluster_ids": list(self.supersedes_cluster_ids),
            "split_from_cluster_id": self.split_from_cluster_id,
            "closed_at": self.closed_at,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CausalCluster:
        payload = _require_mapping(data, "causal_cluster")
        return cls(
            cluster_id=str(payload.get("cluster_id") or ""),
            center_fork_node_id=str(payload.get("center_fork_node_id") or ""),
            member_fork_node_ids=list(payload.get("member_fork_node_ids") or []),
            status=payload.get("status") or "active",  # type: ignore[arg-type]
            created_at=str(payload.get("created_at") or ""),
            supersedes_cluster_ids=list(payload.get("supersedes_cluster_ids") or []),
            split_from_cluster_id=payload.get("split_from_cluster_id"),
            closed_at=payload.get("closed_at"),
            version=payload.get("version") if payload.get("version") is not None else 1,
        )


def _normalize_members(center: str, members: list[str]) -> list[str]:
    out = [center]
    for member in members:
        if member not in out:
            out.append(member)
    return out


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
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


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def _load_json_object(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


class CausalClusterStore:
    """File-level edge/cluster store under ``<root>/causal-experiences/``."""

    def __init__(self, root: str | Path, config: Any = None) -> None:
        self._root = Path(root)
        self._config = config
        self._ns = self._root / NAMESPACE_DIRNAME
        self._edges_dir = self._ns / EDGES_DIRNAME
        self._clusters_dir = self._ns / CLUSTERS_DIRNAME
        self._edges_index_path = self._ns / EDGES_INDEX_FILENAME
        self._clusters_index_path = self._ns / CLUSTERS_INDEX_FILENAME
        self._idempotency_dir = self._ns / IDEMPOTENCY_DIRNAME
        self._lock = threading.Lock()

    def _require_causal_write(self) -> None:
        if not is_causal_mode_enabled(self._config):
            raise PermissionError(
                "causal-experiences writes require skill_trajectory_mode=causal; "
                "evaluator/orchestration trigger semantics are implemented in a later slice"
            )

    def _idempotency_path(self, idempotency_key: str) -> Path:
        digest = _sha256_text(idempotency_key)
        return self._idempotency_dir / f"{digest}.json"

    def _edge_path(self, edge_id: str) -> Path:
        return self._edges_dir / f"{edge_id}.json"

    def _cluster_latest_path(self, cluster_id: str) -> Path:
        return self._clusters_dir / f"{cluster_id}.json"

    def _cluster_revision_path(self, cluster_id: str, version: int) -> Path:
        return self._clusters_dir / f"{cluster_id}.v{version}.json"

    def _read_edges_index(self) -> list[dict[str, Any]]:
        return _read_jsonl(self._edges_index_path)

    def _read_clusters_index(self) -> list[dict[str, Any]]:
        return _read_jsonl(self._clusters_index_path)

    def _find_positive_pair(
        self, pair_key: str, judgment_hash: str
    ) -> dict[str, Any] | None:
        for row in self._read_edges_index():
            if row.get("canonical_pair") != pair_key:
                continue
            if row.get("judgment_hash") == judgment_hash:
                return row
        return None

    def _has_positive_edge(self, fork_a: str, fork_b: str) -> bool:
        if fork_a == fork_b:
            return True
        pair = frozenset((fork_a, fork_b))
        for row in self._read_edges_index():
            ends = frozenset(
                (str(row.get("fork_a") or ""), str(row.get("fork_b") or ""))
            )
            if ends == pair:
                return True
        return False

    def _load_edge(self, edge_id: str) -> VerifiedSimilarityEdge | None:
        payload = _load_json_object(self._edge_path(edge_id))
        if payload is None:
            return None
        return VerifiedSimilarityEdge.from_dict(payload)

    def _load_latest_cluster(self, cluster_id: str) -> CausalCluster | None:
        payload = _load_json_object(self._cluster_latest_path(cluster_id))
        if payload is None:
            return None
        return CausalCluster.from_dict(payload)

    def _require_initiating_edge(
        self, edge_id: str, center_fork_node_id: str, member_fork_node_id: str | None = None
    ) -> VerifiedSimilarityEdge:
        edge = self._load_edge(edge_id)
        if edge is None:
            raise ValueError(f"unknown initiating_edge_id {edge_id!r}")
        if edge.negative:
            raise ValueError("initiating_edge_id must refer to a positive edge")
        ends = {edge.endpoint_a["fork_node_id"], edge.endpoint_b["fork_node_id"]}
        if center_fork_node_id not in ends:
            raise ValueError("initiating_edge_id must connect the cluster center")
        if member_fork_node_id is not None and member_fork_node_id not in ends:
            raise ValueError("initiating_edge_id must connect the new member")
        return edge

    def _require_members_linked_to_center(
        self, center_fork_node_id: str, member_fork_node_ids: list[str]
    ) -> None:
        for member in member_fork_node_ids:
            if member == center_fork_node_id:
                continue
            if not self._has_positive_edge(center_fork_node_id, member):
                raise ValueError(
                    f"member {member!r} has no positive verified edge to center "
                    f"{center_fork_node_id!r}"
                )

    def _persist_cluster(self, cluster: CausalCluster, *, previous: CausalCluster | None) -> None:
        if previous is not None:
            hist_path = self._cluster_revision_path(previous.cluster_id, previous.version)
            if not hist_path.is_file():
                _atomic_write_json(hist_path, previous.to_dict())
        _atomic_write_json(self._cluster_latest_path(cluster.cluster_id), cluster.to_dict())
        _append_jsonl(
            self._clusters_index_path,
            {
                "cluster_id": cluster.cluster_id,
                "version": cluster.version,
                "center_fork_node_id": cluster.center_fork_node_id,
                "member_fork_node_ids": list(cluster.member_fork_node_ids),
                "status": cluster.status,
                "created_at": cluster.created_at,
                "closed_at": cluster.closed_at,
            },
        )

    def _close_cluster(self, cluster: CausalCluster, *, status: ClusterStatus) -> CausalCluster:
        closed = CausalCluster(
            cluster_id=cluster.cluster_id,
            center_fork_node_id=cluster.center_fork_node_id,
            member_fork_node_ids=list(cluster.member_fork_node_ids),
            status=status,
            created_at=cluster.created_at,
            supersedes_cluster_ids=list(cluster.supersedes_cluster_ids),
            split_from_cluster_id=cluster.split_from_cluster_id,
            closed_at=_utc_now_iso(),
            version=cluster.version + 1,
        )
        self._persist_cluster(closed, previous=cluster)
        return closed

    def _latest_index_row_by_cluster(self) -> dict[str, dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}
        for row in self._read_clusters_index():
            cluster_id = row.get("cluster_id")
            if not isinstance(cluster_id, str) or not cluster_id:
                continue
            previous = latest.get(cluster_id)
            if previous is None or int(row.get("version") or 0) >= int(
                previous.get("version") or 0
            ):
                latest[cluster_id] = row
        return latest

    def submit_similarity_edge(
        self, payload: dict[str, Any], *, idempotency_key: str
    ) -> dict[str, Any]:
        """Validate and persist a similarity edge (Q41 server-side minimum)."""
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
                result = dict(stored.get("result") or {})
                result["deduplicated"] = True
                result["created"] = False
                return result

            edge_id = incoming.get("edge_id") or str(uuid.uuid4())
            edge_id = _require_str(edge_id, "edge_id")
            created_at = incoming.get("created_at") or _utc_now_iso()
            draft = dict(incoming)
            draft["edge_id"] = edge_id
            draft["created_at"] = created_at
            draft["direction"] = incoming.get("direction") or EDGE_DIRECTION
            draft["content_hash"] = incoming.get("content_hash") or ""
            edge = VerifiedSimilarityEdge.from_dict(draft)

            if not edge.negative:
                existing = self._find_positive_pair(edge.pair_key, edge.judgment_hash)
                if existing is not None:
                    result = {
                        "edge_id": existing["edge_id"],
                        "deduplicated": True,
                        "created": False,
                        "negative": False,
                    }
                    _atomic_write_json(
                        idem_path,
                        {
                            "payload_hash": payload_hash,
                            "content_hash": existing.get("content_hash"),
                            "edge_id": existing["edge_id"],
                            "created": False,
                            "negative": False,
                            "result": result,
                        },
                    )
                    return result

            _atomic_write_json(self._edge_path(edge.edge_id), edge.to_dict())
            if not edge.negative:
                _append_jsonl(
                    self._edges_index_path,
                    {
                        "edge_id": edge.edge_id,
                        "canonical_pair": edge.pair_key,
                        "judgment_hash": edge.judgment_hash,
                        "content_hash": edge.content_hash,
                        "fork_a": edge.endpoint_a["fork_node_id"],
                        "fork_b": edge.endpoint_b["fork_node_id"],
                        "revision_a": edge.endpoint_a["revision_id"],
                        "revision_b": edge.endpoint_b["revision_id"],
                        "verdict": edge.judgment.verdict,
                        "created_at": edge.created_at,
                    },
                )
            result = {
                "edge_id": edge.edge_id,
                "deduplicated": False,
                "created": True,
                "negative": edge.negative,
            }
            _atomic_write_json(
                idem_path,
                {
                    "payload_hash": payload_hash,
                    "content_hash": edge.content_hash,
                    "edge_id": edge.edge_id,
                    "created": True,
                    "negative": edge.negative,
                    "result": result,
                },
            )
            return result

    def create_cluster(
        self,
        center_fork_node_id: str,
        member_fork_node_ids: list[str],
        initiating_edge_id: str,
    ) -> CausalCluster:
        """Mint a stable cluster_id; members must each have a positive edge to center."""
        self._require_causal_write()
        center = _require_str(center_fork_node_id, "center_fork_node_id")
        members = _require_str_list(member_fork_node_ids, "member_fork_node_ids")
        initiating = _require_str(initiating_edge_id, "initiating_edge_id")
        members = _normalize_members(center, members)
        if len(members) < 2:
            raise ValueError("create_cluster requires at least one member besides the center")

        with self._lock:
            self._require_initiating_edge(initiating, center)
            self._require_members_linked_to_center(center, members)
            cluster = CausalCluster(
                cluster_id=str(uuid.uuid4()),
                center_fork_node_id=center,
                member_fork_node_ids=members,
                status="active",
                created_at=_utc_now_iso(),
                supersedes_cluster_ids=[],
                split_from_cluster_id=None,
                closed_at=None,
                version=1,
            )
            self._persist_cluster(cluster, previous=None)
            return cluster

    def add_member(
        self,
        cluster_id: str,
        member_fork_node_id: str,
        initiating_edge_id: str,
    ) -> CausalCluster:
        """Append a member on a new cluster version; history is retained."""
        self._require_causal_write()
        cid = _require_str(cluster_id, "cluster_id")
        member = _require_str(member_fork_node_id, "member_fork_node_id")
        initiating = _require_str(initiating_edge_id, "initiating_edge_id")

        with self._lock:
            current = self._load_latest_cluster(cid)
            if current is None:
                raise ValueError(f"unknown cluster_id {cid!r}")
            if current.status != "active":
                raise ValueError(f"cluster {cid!r} is not active")
            if member in current.member_fork_node_ids:
                return current
            self._require_members_linked_to_center(current.center_fork_node_id, [member])
            self._require_initiating_edge(initiating, current.center_fork_node_id, member)
            updated = CausalCluster(
                cluster_id=current.cluster_id,
                center_fork_node_id=current.center_fork_node_id,
                member_fork_node_ids=[*current.member_fork_node_ids, member],
                status="active",
                created_at=current.created_at,
                supersedes_cluster_ids=list(current.supersedes_cluster_ids),
                split_from_cluster_id=current.split_from_cluster_id,
                closed_at=None,
                version=current.version + 1,
            )
            self._persist_cluster(updated, previous=current)
            return updated

    def merge_clusters(
        self,
        cluster_ids: list[str],
        *,
        center_fork_node_id: str | None = None,
        member_fork_node_ids: list[str] | None = None,
    ) -> CausalCluster:
        """Create an active successor; close predecessors as superseded (Q40)."""
        self._require_causal_write()
        ids = _require_str_list(cluster_ids, "cluster_ids")
        if len(ids) < 2:
            raise ValueError("merge_clusters requires at least two cluster_ids")

        with self._lock:
            predecessors: list[CausalCluster] = []
            union: list[str] = []
            seen: set[str] = set()
            for cid in ids:
                cluster = self._load_latest_cluster(cid)
                if cluster is None:
                    raise ValueError(f"unknown cluster_id {cid!r}")
                if cluster.status != "active":
                    raise ValueError(f"cluster {cid!r} is not active")
                predecessors.append(cluster)
                for member in cluster.member_fork_node_ids:
                    if member not in seen:
                        seen.add(member)
                        union.append(member)

            if member_fork_node_ids is not None:
                requested = _require_str_list(member_fork_node_ids, "member_fork_node_ids")
                for member in requested:
                    if member not in seen:
                        raise ValueError(
                            f"member {member!r} is not in any predecessor cluster"
                        )
                members = requested
            else:
                members = union

            center = (
                _require_str(center_fork_node_id, "center_fork_node_id")
                if center_fork_node_id is not None
                else predecessors[0].center_fork_node_id
            )
            if center not in seen and center not in members:
                raise ValueError("merge center must belong to a predecessor cluster")
            members = _normalize_members(center, members)

            for pred in predecessors:
                self._close_cluster(pred, status="superseded")

            successor = CausalCluster(
                cluster_id=str(uuid.uuid4()),
                center_fork_node_id=center,
                member_fork_node_ids=members,
                status="active",
                created_at=_utc_now_iso(),
                supersedes_cluster_ids=list(ids),
                split_from_cluster_id=None,
                closed_at=None,
                version=1,
            )
            self._persist_cluster(successor, previous=None)
            return successor

    def split_cluster(
        self,
        cluster_id: str,
        *,
        successors: list[dict[str, Any]],
    ) -> list[CausalCluster]:
        """Create successor clusters; close the source as split_closed (Q40)."""
        self._require_causal_write()
        cid = _require_str(cluster_id, "cluster_id")
        if not isinstance(successors, list) or not successors:
            raise ValueError("successors must be a non-empty list")

        with self._lock:
            source = self._load_latest_cluster(cid)
            if source is None:
                raise ValueError(f"unknown cluster_id {cid!r}")
            if source.status != "active":
                raise ValueError(f"cluster {cid!r} is not active")
            allowed = set(source.member_fork_node_ids)
            created: list[CausalCluster] = []
            for index, spec in enumerate(successors):
                entry = _require_mapping(spec, f"successors[{index}]")
                center = _require_str(
                    entry.get("center_fork_node_id"),
                    f"successors[{index}].center_fork_node_id",
                )
                members = _require_str_list(
                    entry.get("member_fork_node_ids") or [],
                    f"successors[{index}].member_fork_node_ids",
                )
                members = _normalize_members(center, members)
                for member in members:
                    if member not in allowed:
                        raise ValueError(
                            f"split member {member!r} is not in predecessor cluster"
                        )
                created.append(
                    CausalCluster(
                        cluster_id=str(uuid.uuid4()),
                        center_fork_node_id=center,
                        member_fork_node_ids=members,
                        status="active",
                        created_at=_utc_now_iso(),
                        supersedes_cluster_ids=[],
                        split_from_cluster_id=source.cluster_id,
                        closed_at=None,
                        version=1,
                    )
                )
            self._close_cluster(source, status="split_closed")
            for cluster in created:
                self._persist_cluster(cluster, previous=None)
            return created

    def propose_cluster_change(
        self, payload: dict[str, Any]
    ) -> CausalCluster | list[CausalCluster]:
        """Q41 ClusterChangePropose minimum: create / add_member / merge / split."""
        self._require_causal_write()
        data = _require_mapping(payload, "payload")
        action = _require_str(data.get("action"), "action")
        if action == "create_cluster":
            return self.create_cluster(
                center_fork_node_id=_require_str(
                    data.get("center_fork_node_id"), "center_fork_node_id"
                ),
                member_fork_node_ids=list(data.get("member_fork_node_ids") or []),
                initiating_edge_id=_require_str(
                    data.get("initiating_edge_id"), "initiating_edge_id"
                ),
            )
        if action == "add_member":
            return self.add_member(
                cluster_id=_require_str(data.get("cluster_id"), "cluster_id"),
                member_fork_node_id=_require_str(
                    data.get("member_fork_node_id"), "member_fork_node_id"
                ),
                initiating_edge_id=_require_str(
                    data.get("initiating_edge_id"), "initiating_edge_id"
                ),
            )
        if action == "merge_clusters":
            return self.merge_clusters(
                list(data.get("cluster_ids") or []),
                center_fork_node_id=data.get("center_fork_node_id"),
                member_fork_node_ids=data.get("member_fork_node_ids"),
            )
        if action == "split_cluster":
            return self.split_cluster(
                _require_str(data.get("cluster_id"), "cluster_id"),
                successors=list(data.get("successors") or []),
            )
        raise ValueError(f"unknown cluster change action: {action!r}")

    def get_cluster(self, cluster_id: str) -> dict[str, Any] | None:
        cid = _require_str(cluster_id, "cluster_id")
        latest = self._load_latest_cluster(cid)
        if latest is None:
            return None
        versions: list[dict[str, Any]] = []
        for path in self._clusters_dir.glob(f"{cid}.v*.json"):
            payload = _load_json_object(path)
            if payload is None:
                continue
            versions.append(CausalCluster.from_dict(payload).to_dict())
        versions.sort(key=lambda item: int(item.get("version") or 0))
        versions.append(latest.to_dict())
        return {
            "cluster_id": cid,
            "latest": latest.to_dict(),
            "versions": versions,
        }

    def clusters_for_fork(self, fork_node_id: str) -> list[dict[str, Any]]:
        fork = _require_str(fork_node_id, "fork_node_id")
        hits: list[dict[str, Any]] = []
        for row in self._latest_index_row_by_cluster().values():
            if row.get("status") != "active":
                continue
            members = row.get("member_fork_node_ids") or []
            if fork not in members:
                continue
            loaded = self.get_cluster(str(row["cluster_id"]))
            if loaded is not None:
                hits.append(loaded["latest"])
        return hits

    def edges_between(
        self, fork_a: str, fork_b: str, *, include_negative: bool = False
    ) -> list[dict[str, Any]]:
        left = _require_str(fork_a, "fork_a")
        right = _require_str(fork_b, "fork_b")
        wanted = frozenset((left, right))
        found: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in self._read_edges_index():
            ends = frozenset(
                (str(row.get("fork_a") or ""), str(row.get("fork_b") or ""))
            )
            if ends != wanted:
                continue
            edge = self._load_edge(str(row.get("edge_id") or ""))
            if edge is None:
                continue
            found.append(edge.to_dict())
            seen.add(edge.edge_id)
        if include_negative and self._edges_dir.is_dir():
            for path in self._edges_dir.glob("*.json"):
                payload = _load_json_object(path)
                if payload is None:
                    continue
                edge = VerifiedSimilarityEdge.from_dict(payload)
                if edge.edge_id in seen:
                    continue
                ends = frozenset(
                    (edge.endpoint_a["fork_node_id"], edge.endpoint_b["fork_node_id"])
                )
                if ends != wanted:
                    continue
                found.append(edge.to_dict())
        return found

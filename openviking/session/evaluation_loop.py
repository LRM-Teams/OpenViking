# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Minimal in-process Memory → Diagnosis → Memory evaluation loop (ADR-0012).

``MinimalEvaluationLoop`` drives one evaluation iteration through
``EvaluationOrchestrator.advance``. It does not invent transitions or stop
conditions. Freeze and commit carry the F2e envelope and revision evidence.
Entering diagnosis passes ``diagnosis_batch`` through the seat guard.

Collaborators are injected. A stub may stand in for diagnosis and for hint
composition. Fork proposals are accepted only as drafts and are stored with
``commit_provisional``. Validated and invalidated revisions stay on the
append-only APIs this loop does not call. Replaying the same iteration
re-enters each store; idempotency keys suppress a second brief, provisional
fork, hint, and citation.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from openviking.session.causal_experiences import (
    CausalExperiencesStore,
    ForkDraft,
    ForkNodeRevision,
    ForkStatus,
    ServerChecks,
)
from openviking.session.citation_ledger import (
    KIND_CITATION,
    ROLE_CONSIDERED,
    CitationLedger,
)
from openviking.session.evaluation_orchestrator import (
    EVENT_COMMIT_REVISIONS,
    EVENT_DIAGNOSIS,
    EVENT_FREEZE_BRIEF,
    EVENT_MEMORY_RETRIEVE,
    EVENT_NEXT_RUN,
    EVENT_SCORE,
    EVENT_TASK_RUN,
    TERMINAL_STATES,
    EvaluationOrchestrator,
    IllegalTransition,
    IterationResult,
    OrchestratorState,
)
from openviking.session.handoff_envelope import (
    DeliveryResult,
    DiagnosisBrief,
    HandoffEnvelope,
    HandoffEnvelopeService,
    content_address,
    verify_manifest,
)
from openviking.session.memory_hints import HintDeliveryService, MemoryHint

DiagnosisFn = Callable[[Mapping[str, Any]], Sequence[Any]]
HintComposer = Callable[[Sequence[ForkNodeRevision]], MemoryHint | Mapping[str, Any]]

_STATE_RANK: dict[OrchestratorState, int] = {
    OrchestratorState.CREATED: 0,
    OrchestratorState.MEMORY_RETRIEVE: 1,
    OrchestratorState.TASK_RUN: 2,
    OrchestratorState.SCORE: 3,
    OrchestratorState.FREEZE_BRIEF: 4,
    OrchestratorState.DIAGNOSIS: 5,
    OrchestratorState.COMMIT_REVISIONS: 6,
    OrchestratorState.NEXT_RUN: 7,
    OrchestratorState.DONE: 8,
    OrchestratorState.STOPPED: 8,
}

_TERMINAL_WRITES = frozenset({ForkStatus.VALIDATED, ForkStatus.INVALIDATED})


def _require_str(value: Any, field: str) -> str:
    if not isinstance(value, str) or value == "":
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _json_mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return {str(key): item for key, item in value.items()}


@dataclass(frozen=True)
class EvaluationIteration:
    """Facts for one evaluation iteration. Stop flags are an ``IterationResult``."""

    iteration_id: str
    task_id: str
    source_run_id: str
    memory_task_state_hash: str
    selected_refs: tuple[str, ...]
    snapshot_watermarks: Mapping[str, Any]
    policy_versions: Mapping[str, Any]
    sub_budgets: Mapping[str, Any]
    target_agent_id: str
    task_lineage_id: str
    iteration_result: IterationResult
    diagnosis_batch: int = 1

    def __post_init__(self) -> None:
        _require_str(self.iteration_id, "iteration_id")
        _require_str(self.task_id, "task_id")
        _require_str(self.source_run_id, "source_run_id")
        _require_str(self.memory_task_state_hash, "memory_task_state_hash")
        _require_str(self.target_agent_id, "target_agent_id")
        _require_str(self.task_lineage_id, "task_lineage_id")
        if isinstance(self.selected_refs, str) or not isinstance(self.selected_refs, tuple):
            raise TypeError("selected_refs must be a tuple of strings")
        object.__setattr__(
            self,
            "selected_refs",
            tuple(_require_str(ref, "selected_refs") for ref in self.selected_refs),
        )
        object.__setattr__(self, "snapshot_watermarks", _json_mapping(self.snapshot_watermarks, "snapshot_watermarks"))
        object.__setattr__(self, "policy_versions", _json_mapping(self.policy_versions, "policy_versions"))
        object.__setattr__(self, "sub_budgets", _json_mapping(self.sub_budgets, "sub_budgets"))
        result = self.iteration_result
        if isinstance(result, Mapping):
            result = IterationResult.from_dict(result)
        elif not isinstance(result, IterationResult):
            raise TypeError("iteration_result must be an IterationResult or a mapping")
        object.__setattr__(self, "iteration_result", result)
        batch = self.diagnosis_batch
        if isinstance(batch, bool) or not isinstance(batch, int):
            raise ValueError("diagnosis_batch must be an int")
        if batch < 0:
            raise ValueError("diagnosis_batch must be >= 0")


@dataclass(frozen=True)
class EvaluationLoopResult:
    """Identities produced by one pass. A replay returns the original ids."""

    iteration_id: str
    state: OrchestratorState
    stopped: bool
    brief_envelope_id: str
    input_manifest_hash: str
    revision_ids: tuple[str, ...]
    fork_node_ids: tuple[str, ...]
    hint_id: str
    task_lineage_id: str


class MinimalEvaluationLoop:
    """One iteration: retrieve, score, hand off a brief, commit provisional forks, deliver a hint.

    Every move goes through ``EvaluationOrchestrator.advance``. When the
    orchestrator is already past a step, that advance is not repeated, because
    the guard would reject it. Store writes still run and rely on each store's
    idempotency key.
    """

    def __init__(
        self,
        orchestrator: EvaluationOrchestrator,
        handoff: HandoffEnvelopeService,
        experiences: CausalExperiencesStore,
        hints: HintDeliveryService,
        ledger: CitationLedger,
        diagnosis_fn: DiagnosisFn,
        hint_composer: HintComposer,
        server_checks: ServerChecks,
        *,
        policy_version: str = "memory-hint-v1",
        source_channel_labels: Iterable[str] = ("memory",),
    ) -> None:
        if not isinstance(orchestrator, EvaluationOrchestrator):
            raise TypeError("orchestrator must be an EvaluationOrchestrator")
        if not isinstance(handoff, HandoffEnvelopeService):
            raise TypeError("handoff must be a HandoffEnvelopeService")
        if not isinstance(experiences, CausalExperiencesStore):
            raise TypeError("experiences must be a CausalExperiencesStore")
        if not isinstance(hints, HintDeliveryService):
            raise TypeError("hints must be a HintDeliveryService")
        if not isinstance(ledger, CitationLedger):
            raise TypeError("ledger must be a CitationLedger")
        if not callable(diagnosis_fn) or not callable(hint_composer):
            raise TypeError("diagnosis_fn and hint_composer must be callable")
        if not isinstance(server_checks, ServerChecks):
            raise TypeError("server_checks must be a ServerChecks")
        labels = tuple(source_channel_labels)
        if not labels or any(not isinstance(label, str) or label == "" for label in labels):
            raise ValueError("source_channel_labels must be a non-empty sequence of strings")
        self._orchestrator = orchestrator
        self._handoff = handoff
        self._experiences = experiences
        self._hints = hints
        self._ledger = ledger
        self._diagnosis_fn = diagnosis_fn
        self._hint_composer = hint_composer
        self._server_checks = server_checks
        self._policy_version = _require_str(policy_version, "policy_version")
        self._source_channel_labels = labels
        self._iteration_id: str | None = None

    def run(self, iteration: EvaluationIteration | Mapping[str, Any]) -> EvaluationLoopResult:
        """Run or replay one iteration. A second iteration id is rejected."""
        spec = iteration if isinstance(iteration, EvaluationIteration) else _iteration_from_mapping(iteration)
        self._bind(spec)
        self._step(
            OrchestratorState.CREATED,
            EVENT_MEMORY_RETRIEVE,
            {
                "iteration_id": spec.iteration_id,
                "selected_refs": list(spec.selected_refs),
            },
        )
        self._step(
            OrchestratorState.MEMORY_RETRIEVE,
            EVENT_TASK_RUN,
            {
                "iteration_id": spec.iteration_id,
                "task_id": spec.task_id,
                "source_run_id": spec.source_run_id,
            },
        )
        self._step(
            OrchestratorState.TASK_RUN,
            EVENT_SCORE,
            {
                "iteration_id": spec.iteration_id,
                "key_judgment": spec.iteration_result.key_judgment,
                "evidence_refs": list(spec.iteration_result.evidence_refs),
            },
        )
        brief, manifest_hash, delivery = self._deliver_brief(spec)
        self._step(
            OrchestratorState.SCORE,
            EVENT_FREEZE_BRIEF,
            {
                "iteration_id": spec.iteration_id,
                "brief_envelope_id": delivery.envelope_id,
                "input_manifest_hash": manifest_hash,
            },
            brief_envelope_id=delivery.envelope_id,
        )
        self._step(
            OrchestratorState.FREEZE_BRIEF,
            EVENT_DIAGNOSIS,
            {
                "iteration_id": spec.iteration_id,
                "diagnosis_batch": spec.diagnosis_batch,
                "brief_envelope_id": delivery.envelope_id,
            },
            diagnosis_batch=spec.diagnosis_batch,
        )
        revisions = self._commit_proposals(self._diagnosis_fn(self._brief_view(brief, delivery)))
        revision_ids = [revision.revision_id for revision in revisions]
        self._step(
            OrchestratorState.DIAGNOSIS,
            EVENT_COMMIT_REVISIONS,
            {
                "iteration_id": spec.iteration_id,
                "revision_ids": revision_ids,
            },
            revision_ids=revision_ids,
        )
        hint = self._deliver_hint(spec, revisions)
        self._step(
            OrchestratorState.COMMIT_REVISIONS,
            EVENT_NEXT_RUN,
            {
                "iteration_id": spec.iteration_id,
                "hint_id": hint.hint_id,
                "task_lineage_id": spec.task_lineage_id,
            },
        )
        stopped = self._finish(spec)
        return EvaluationLoopResult(
            iteration_id=spec.iteration_id,
            state=self._orchestrator.state,
            stopped=stopped,
            brief_envelope_id=delivery.envelope_id,
            input_manifest_hash=manifest_hash,
            revision_ids=tuple(revision.revision_id for revision in revisions),
            fork_node_ids=tuple(revision.fork_node_id for revision in revisions),
            hint_id=hint.hint_id,
            task_lineage_id=spec.task_lineage_id,
        )

    def _bind(self, spec: EvaluationIteration) -> None:
        if self._iteration_id is None:
            self._iteration_id = spec.iteration_id
            return
        if self._iteration_id != spec.iteration_id:
            raise ValueError(
                f"iteration_id {spec.iteration_id!r} does not match "
                f"bound iteration {self._iteration_id!r}"
            )

    def _step(self, expected: OrchestratorState, event: str, payload: Mapping[str, Any], **kwargs: Any) -> None:
        """Advance only from ``expected``. A later state has already passed this guard."""
        current = self._orchestrator.state
        rank = _STATE_RANK.get(current)
        if rank is None:
            raise IllegalTransition(current, event)
        if current is expected:
            self._orchestrator.advance(event, payload, **kwargs)
            return
        if rank > _STATE_RANK[expected]:
            return
        raise IllegalTransition(current, event)

    def _finish(self, spec: EvaluationIteration) -> bool:
        state = self._orchestrator.state
        if state is OrchestratorState.NEXT_RUN or state in TERMINAL_STATES:
            return self._orchestrator.should_stop(spec.iteration_result)
        raise IllegalTransition(state, OrchestratorState.STOPPED)

    def _deliver_brief(self, spec: EvaluationIteration) -> tuple[DiagnosisBrief, str, DeliveryResult]:
        manifest = {
            "evaluation_id": self._orchestrator.evaluation_id,
            "task_id": spec.task_id,
            "source_run_id": spec.source_run_id,
            "iteration_id": spec.iteration_id,
            "memory_task_state_hash": spec.memory_task_state_hash,
            "selected_refs": list(spec.selected_refs),
            "snapshot_watermarks": dict(spec.snapshot_watermarks),
            "policy_versions": dict(spec.policy_versions),
            "sub_budgets": dict(spec.sub_budgets),
        }
        manifest_hash = content_address(manifest)
        brief = DiagnosisBrief.from_dict({**manifest, "input_manifest_hash": manifest_hash})
        sealed = HandoffEnvelope.from_brief(brief)
        verify_manifest(sealed, manifest)
        delivery = self._handoff.deliver(sealed)
        return brief, manifest_hash, delivery

    def _brief_view(self, brief: DiagnosisBrief, delivery: DeliveryResult) -> dict[str, Any]:
        return {
            "envelope_id": delivery.envelope_id,
            "handoff_kind": delivery.handoff_kind,
            "iteration_id": delivery.iteration_id,
            "input_manifest_hash": delivery.input_manifest_hash,
            "brief": brief.to_dict(),
        }

    def _commit_proposals(self, proposals: Any) -> tuple[ForkNodeRevision, ...]:
        if isinstance(proposals, (str, bytes, Mapping)) or not isinstance(proposals, Sequence):
            raise TypeError("diagnosis_fn must return a sequence of fork proposals")
        revisions: list[ForkNodeRevision] = []
        for proposal in proposals:
            revisions.append(self._experiences.commit_provisional(self._as_draft(proposal), self._server_checks))
        return tuple(revisions)

    def _as_draft(self, proposal: Any) -> ForkDraft:
        if isinstance(proposal, ForkDraft):
            return proposal
        if isinstance(proposal, Mapping):
            status = proposal.get("status")
            if status in _TERMINAL_WRITES:
                raise ValueError(
                    "fork proposals cannot write validated or invalidated; "
                    "commit_provisional accepts drafts only"
                )
            return self._experiences.submit_fork_draft(dict(proposal))
        raise TypeError("fork proposal must be a ForkDraft or a mapping")

    def _deliver_hint(self, spec: EvaluationIteration, revisions: Sequence[ForkNodeRevision]) -> MemoryHint:
        composed = self._hint_composer(revisions)
        hint = composed if isinstance(composed, MemoryHint) else MemoryHint.from_dict(composed)
        delivered = self._hints.deliver(hint, task_lineage_id=spec.task_lineage_id)
        self._record_citations(spec, revisions)
        return delivered

    def _record_citations(self, spec: EvaluationIteration, revisions: Sequence[ForkNodeRevision]) -> None:
        """Record a structured citation once per provisional fork.

        Hint exposure is written by ``HintDeliveryService.deliver``. This pass
        adds the fork citation. A replay sees the ledger event and does not
        append another one.
        """
        existing = {
            (event.ref.type, event.ref.id, event.ref.revision)
            for event in self._ledger.events_for(self._source_channel_labels)
            if event.kind == KIND_CITATION
        }
        for revision in revisions:
            identity = ("fork", revision.fork_node_id, revision.revision_id)
            if identity in existing:
                continue
            self._ledger.record_citation(
                task_id=spec.task_id,
                source_task_id=spec.task_id,
                ref={
                    "type": "fork",
                    "id": revision.fork_node_id,
                    "revision": revision.revision_id,
                },
                role=ROLE_CONSIDERED,
                source_channel_labels=self._source_channel_labels,
                policy_version=self._policy_version,
            )
            existing.add(identity)


def _iteration_from_mapping(data: Mapping[str, Any]) -> EvaluationIteration:
    if not isinstance(data, Mapping):
        raise TypeError("iteration must be an EvaluationIteration or a mapping")
    refs = data.get("selected_refs", ())
    if isinstance(refs, str) or not isinstance(refs, Sequence):
        raise TypeError("selected_refs must be a sequence of strings")
    result = data.get("iteration_result", {})
    return EvaluationIteration(
        iteration_id=_require_str(data.get("iteration_id"), "iteration_id"),
        task_id=_require_str(data.get("task_id"), "task_id"),
        source_run_id=_require_str(data.get("source_run_id"), "source_run_id"),
        memory_task_state_hash=_require_str(data.get("memory_task_state_hash"), "memory_task_state_hash"),
        selected_refs=tuple(refs),
        snapshot_watermarks=_json_mapping(data.get("snapshot_watermarks", {}), "snapshot_watermarks"),
        policy_versions=_json_mapping(data.get("policy_versions", {}), "policy_versions"),
        sub_budgets=_json_mapping(data.get("sub_budgets", {}), "sub_budgets"),
        target_agent_id=_require_str(data.get("target_agent_id"), "target_agent_id"),
        task_lineage_id=_require_str(data.get("task_lineage_id"), "task_lineage_id"),
        iteration_result=result if isinstance(result, IterationResult) else IterationResult.from_dict(result),
        diagnosis_batch=data.get("diagnosis_batch", 1),
    )

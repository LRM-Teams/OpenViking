# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Positive outcome utility (Q89-A).

Adoption, followthrough, and outcome stay separate events. This module
aggregates them without writing state or reading the clock. A positive
utility exists only when all three are present for the same object:

* adoption: a disposition event with ``decision`` ``adopt``
* completion: a followthrough event with ``followthrough`` ``observed``
* positive outcome: an outcome event with ``outcome`` ``positive``

Hint exposure and citation events never count as adoption, followthrough,
or outcome. ``policy_version`` selects one column: events tagged with a
different ``policy_version`` are left out. Events with no ``policy_version``
belong to the requested column, matching hint records that do not store one.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from openviking.session.citation_ledger import (
    KIND_CITATION,
    KIND_HINT_EXPOSURE,
    KIND_INDEX_LEVEL_EXPOSURE,
)
from openviking.session.memory_hints import (
    DECISION_ADOPT,
    FOLLOWTHROUGH_OBSERVED,
    OUTCOME_POSITIVE,
)

DEFAULT_OBJECT_KEY = "hint_id"

_LEDGER_KINDS = frozenset({KIND_CITATION, KIND_HINT_EXPOSURE, KIND_INDEX_LEVEL_EXPOSURE})


class _Bucket:
    __slots__ = ("adopted", "completed", "positive_count", "evidence_refs")

    def __init__(self) -> None:
        self.adopted = False
        self.completed = False
        self.positive_count = 0
        self.evidence_refs: list[str] = []


def positive_outcome_utility(
    events: Sequence[Mapping[str, Any]],
    policy_version: str,
    *,
    object_key: str = DEFAULT_OBJECT_KEY,
) -> dict[str, dict[str, Any]]:
    """Aggregate positive outcome utility for one policy column.

    Returns ``{object_key: {positive_count, evidence_refs, policy_version}}``.
    ``positive_count`` is the number of positive outcome events once adoption
    and a completed followthrough are both present. Missing any of the three
    omits the object. The same inputs always return the same mapping.
    """
    if not isinstance(policy_version, str) or policy_version == "":
        raise ValueError("policy_version must be a non-empty string")
    if not isinstance(object_key, str) or object_key == "":
        raise ValueError("object_key must be a non-empty string")
    if isinstance(events, (str, bytes, Mapping)) or not isinstance(events, Sequence):
        raise TypeError("events must be a sequence of event mappings")

    order: list[str] = []
    buckets: dict[str, _Bucket] = {}
    for event in events:
        if not isinstance(event, Mapping):
            raise TypeError("events must be a sequence of event mappings")
        if event.get("kind") in _LEDGER_KINDS:
            continue
        if not _in_policy(event, policy_version):
            continue
        key = event.get(object_key)
        if not isinstance(key, str) or key == "":
            continue
        bucket = buckets.get(key)
        if bucket is None:
            bucket = _Bucket()
            buckets[key] = bucket
            order.append(key)
        if event.get("decision") == DECISION_ADOPT:
            bucket.adopted = True
        if event.get("followthrough") == FOLLOWTHROUGH_OBSERVED:
            bucket.completed = True
        if event.get("outcome") == OUTCOME_POSITIVE:
            bucket.positive_count += 1
            _extend_evidence(bucket.evidence_refs, event.get("evidence_refs"))

    result: dict[str, dict[str, Any]] = {}
    for key in order:
        bucket = buckets[key]
        if not (bucket.adopted and bucket.completed and bucket.positive_count):
            continue
        result[key] = {
            "positive_count": bucket.positive_count,
            "evidence_refs": list(bucket.evidence_refs),
            "policy_version": policy_version,
        }
    return result


def _in_policy(event: Mapping[str, Any], policy_version: str) -> bool:
    tagged = event.get("policy_version")
    if tagged is None:
        return True
    return tagged == policy_version


def _extend_evidence(sink: list[str], raw: Any) -> None:
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        return
    seen = set(sink)
    for item in raw:
        if isinstance(item, str) and item != "" and item not in seen:
            sink.append(item)
            seen.add(item)

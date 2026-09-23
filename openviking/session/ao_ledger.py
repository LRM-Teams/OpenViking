# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Append-only action-observation ledger for a live session.

AO identities are issued here (globally unique ``ao_id``, per-session
``sequence``). Archive attribution is a separate append-only record and
never mutates ``AORecord`` (ADR-0002).
"""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openviking.session.tool_result_synopsis import (
    ToolResultSynopsis,
    generate_tool_result_synopsis,
)

LEDGER_FILENAME = "ao-ledger.jsonl"
CAPTURED_STATE_LIVE = "live"
_ACTION_PARAM_LIMIT = 2048
_SYNOPSIS_PREVIEW_CHARS = 500
_PARAM_KEYS = ("arguments", "params", "parameters", "args")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _clip_text(value: str, limit: int) -> str:
    if limit <= 0:
        return ""
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)] + "..."


def _truncate_tree(value: Any, limit: int) -> tuple[Any, bool]:
    if isinstance(value, str):
        if len(value) > limit:
            return _clip_text(value, limit), True
        return value, False
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        truncated = False
        for key, child in value.items():
            new_child, was_truncated = _truncate_tree(child, limit)
            out[str(key)] = new_child
            truncated = truncated or was_truncated
        return out, truncated
    if isinstance(value, list):
        out_list: list[Any] = []
        truncated = False
        for child in value:
            new_child, was_truncated = _truncate_tree(child, limit)
            out_list.append(new_child)
            truncated = truncated or was_truncated
        return out_list, truncated
    return value, False


def _sanitize_action(action: dict[str, Any]) -> dict[str, Any]:
    """Keep tool name and a compact param summary; drop large originals."""
    sanitized, truncated = _truncate_tree(dict(action), _ACTION_PARAM_LIMIT)
    if not isinstance(sanitized, dict):
        sanitized = {"value": sanitized}
        truncated = True
    sanitized.pop("truncated", None)
    for key in _PARAM_KEYS:
        if key not in sanitized:
            continue
        encoded = json.dumps(sanitized[key], ensure_ascii=False, default=str)
        if len(encoded) > _ACTION_PARAM_LIMIT:
            sanitized[key] = _clip_text(encoded, _ACTION_PARAM_LIMIT)
            truncated = True
    sanitized["truncated"] = bool(truncated)
    return sanitized


def extract_skill_invocations(tool_call: dict[str, Any]) -> list[dict[str, Any]]:
    """Runtime hook that collects skill invocations for an AO record.

    v1 always returns ``[]``. Later slices will attach real skill_uri /
    revision_hash / invocation_id from the call path. Callers must still
    persist the field (empty list, never omitted).
    """
    del tool_call
    return []


def build_observation(tool_result_content: str | dict[str, Any]) -> dict[str, Any]:
    """Build a synopsis observation dict from tool result content.

    Strings are handed to ``generate_tool_result_synopsis`` as text content
    (the generator still classifies JSON/YAML/code when the payload looks
    like those kinds). Dicts are JSON-serialized so the same public entry
    can classify structured json/yaml results. The returned dict is
    ``ToolResultSynopsis.to_dict()`` (kind/title/summary/structure/
    notable_items/sample). Callers may add ``artifact_ref`` when a large
    result is stored out-of-line.
    """
    if isinstance(tool_result_content, dict):
        content = json.dumps(tool_result_content, ensure_ascii=False, default=str)
        mime_type = "application/json"
    else:
        content = tool_result_content
        mime_type = "text/plain"
    synopsis: ToolResultSynopsis = generate_tool_result_synopsis(
        content,
        preview_chars=_SYNOPSIS_PREVIEW_CHARS,
        mime_type=mime_type,
    )
    return synopsis.to_dict()


@dataclass(frozen=True)
class AORecord:
    """Immutable live action-observation pair. Archive ownership is not here."""

    ao_id: str
    session_id: str
    sequence: int
    action: dict[str, Any]
    observation: dict[str, Any]
    skill_invocations: list[dict[str, Any]]
    message_ref: dict[str, Any] | None
    created_at: str
    captured_state: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AORecord:
        raw_ref = data.get("message_ref")
        return cls(
            ao_id=str(data["ao_id"]),
            session_id=str(data["session_id"]),
            sequence=int(data["sequence"]),
            action=dict(data.get("action") or {}),
            observation=dict(data.get("observation") or {}),
            skill_invocations=[dict(item) for item in (data.get("skill_invocations") or [])],
            message_ref=dict(raw_ref) if raw_ref is not None else None,
            created_at=str(data["created_at"]),
            captured_state=str(data.get("captured_state") or CAPTURED_STATE_LIVE),
        )


@dataclass(frozen=True)
class AOAttribution:
    """Independent append-only archive attribution. Never rewrites AORecord."""

    ao_id: str
    archive_id: str
    commit_watermark: str
    attributed_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AOAttribution:
        return cls(
            ao_id=str(data["ao_id"]),
            archive_id=str(data["archive_id"]),
            commit_watermark=str(data["commit_watermark"]),
            attributed_at=str(data["attributed_at"]),
        )


class AOLedger:
    """Per-session append-only AO ledger persisted as ``ao-ledger.jsonl``."""

    def __init__(self, session_dir: Path, session_id: str) -> None:
        self._session_dir = Path(session_dir)
        self._session_id = session_id
        self._path = self._session_dir / LEDGER_FILENAME
        self._lock = threading.Lock()
        self._records: list[AORecord] = []
        self._by_id: dict[str, AORecord] = {}
        self._next_sequence = 1
        self.skipped_line_count = 0
        self._session_dir.mkdir(parents=True, exist_ok=True)
        self.load_existing()

    def load_existing(self) -> None:
        """Load ``ao-ledger.jsonl``. Corrupt lines are skipped and counted."""
        records: list[AORecord] = []
        by_id: dict[str, AORecord] = {}
        skipped = 0
        if self._path.is_file():
            with self._path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    raw = line.strip()
                    if not raw:
                        continue
                    try:
                        payload = json.loads(raw)
                        if not isinstance(payload, dict):
                            raise TypeError("ledger line must be a JSON object")
                        record = AORecord.from_dict(payload)
                    except (json.JSONDecodeError, TypeError, KeyError, ValueError):
                        skipped += 1
                        continue
                    records.append(record)
                    by_id[record.ao_id] = record
        records.sort(key=lambda item: item.sequence)
        with self._lock:
            self._records = records
            self._by_id = by_id
            self.skipped_line_count = skipped
            self._next_sequence = records[-1].sequence + 1 if records else 1

    def append(
        self,
        action: dict[str, Any],
        observation: dict[str, Any],
        skill_invocations: list[dict[str, Any]] | None = None,
        message_ref: dict[str, Any] | None = None,
    ) -> AORecord:
        if skill_invocations is None:
            skill_invocations = extract_skill_invocations(action)
        sanitized_action = _sanitize_action(action)
        stored_observation = dict(observation)
        stored_skills = [dict(item) for item in skill_invocations]
        stored_ref = dict(message_ref) if message_ref is not None else None
        with self._lock:
            record = AORecord(
                ao_id=uuid.uuid4().hex,
                session_id=self._session_id,
                sequence=self._next_sequence,
                action=sanitized_action,
                observation=stored_observation,
                skill_invocations=stored_skills,
                message_ref=stored_ref,
                created_at=_utc_now_iso(),
                captured_state=CAPTURED_STATE_LIVE,
            )
            line = json.dumps(record.to_dict(), ensure_ascii=False) + "\n"
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()
            self._records.append(record)
            self._by_id[record.ao_id] = record
            self._next_sequence += 1
            return record

    def records(
        self,
        start_sequence: int | None = None,
        end_sequence: int | None = None,
    ) -> list[AORecord]:
        with self._lock:
            items = list(self._records)
        items.sort(key=lambda item: item.sequence)
        if start_sequence is not None:
            items = [item for item in items if item.sequence >= start_sequence]
        if end_sequence is not None:
            items = [item for item in items if item.sequence <= end_sequence]
        return items

    def get(self, ao_id: str) -> AORecord | None:
        with self._lock:
            return self._by_id.get(ao_id)

    def count(self) -> int:
        with self._lock:
            return len(self._records)

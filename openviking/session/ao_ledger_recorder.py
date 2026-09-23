# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Runtime AO ledger hooks: live tool-exchange capture and archive attribution.

Session wiring stays outside this module. ``session.py`` only calls the two
``maybe_*`` helpers; failures here are swallowed so the live session path
cannot break (ADR-0002: attribution is a separate append-only file).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openviking.session.ao_ledger import (
    AOAttribution,
    AOLedger,
    AORecord,
    LEDGER_FILENAME,
    build_observation,
    extract_skill_invocations,
)
from openviking_cli.utils import get_logger
from openviking_cli.utils.config.memory_config import is_causal_mode_enabled

ATTRIBUTION_FILENAME = "ao-attribution.jsonl"
HISTORY_DIRNAME = "history"
_REF_KEYS = (
    "artifact_ref",
    "tool_output_ref",
    "tool_output_storage_uri",
    "storage_uri",
)
_PARAM_KEYS = ("arguments", "params", "parameters", "args", "tool_input")
_CONTENT_KEYS = ("content", "output", "tool_output", "text")

logger = get_logger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _read_attribution_ao_ids(path: Path) -> set[str]:
    """Return ao_ids from one attribution file. Missing/unreadable → empty."""
    ids: set[str] = set()
    if not path.is_file():
        return ids
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict) and payload.get("ao_id"):
                    ids.add(str(payload["ao_id"]))
    except OSError:
        return set()
    return ids


def collect_attributed_ao_ids(session_dir: Path | str) -> set[str]:
    """Union ao_ids from every ``history/archive_*/ao-attribution.jsonl``.

    Missing history or files are treated as empty (no attribution yet).
    """
    attributed: set[str] = set()
    history = Path(session_dir) / HISTORY_DIRNAME
    if not history.is_dir():
        return attributed
    for child in history.iterdir():
        if child.is_dir():
            attributed.update(_read_attribution_ao_ids(child / ATTRIBUTION_FILENAME))
    return attributed


def write_archive_attribution(session_dir: Path | str, archive_dir: Path | str) -> int:
    """Attribute every still-unattributed AO record to ``archive_dir``.

    Never rewrites ``ao-ledger.jsonl``. Returns the number of new attribution
    lines written. Re-running against an already-attributed set is a no-op.
    """
    session_path = Path(session_dir)
    archive_path = Path(archive_dir)
    ledger_file = session_path / LEDGER_FILENAME
    if not ledger_file.is_file():
        return 0

    attributed = collect_attributed_ao_ids(session_path)
    attributed.update(_read_attribution_ao_ids(archive_path / ATTRIBUTION_FILENAME))

    ledger = AOLedger(session_path, session_path.name)
    pending = [record for record in ledger.records() if record.ao_id not in attributed]
    if not pending:
        return 0

    archive_id = archive_path.name
    attributed_at = _utc_now_iso()
    commit_watermark = f"{archive_id}@{attributed_at}"
    archive_path.mkdir(parents=True, exist_ok=True)
    dest = archive_path / ATTRIBUTION_FILENAME
    written = 0
    with dest.open("a", encoding="utf-8") as handle:
        for record in pending:
            attr = AOAttribution(
                ao_id=record.ao_id,
                archive_id=archive_id,
                commit_watermark=commit_watermark,
                attributed_at=attributed_at,
            )
            handle.write(json.dumps(attr.to_dict(), ensure_ascii=False) + "\n")
            written += 1
        handle.flush()
    return written


def _action_from_tool_call(tool_call: dict[str, Any]) -> dict[str, Any]:
    tool_name = (
        tool_call.get("tool")
        or tool_call.get("tool_name")
        or tool_call.get("name")
        or ""
    )
    action: dict[str, Any] = {"tool": str(tool_name)}
    tool_id = tool_call.get("tool_id")
    if tool_id:
        action["tool_id"] = tool_id
    for key in _PARAM_KEYS:
        if key not in tool_call or tool_call[key] is None:
            continue
        action["arguments"] = tool_call[key]
        break
    return action


def _extract_artifact_ref(
    tool_call: dict[str, Any],
    tool_result: Any,
    message_ref: dict[str, Any] | None,
) -> Any:
    if isinstance(tool_result, dict):
        for key in _REF_KEYS:
            value = tool_result.get(key)
            if value:
                return value
    if message_ref:
        value = message_ref.get("artifact_ref")
        if value:
            return value
    return tool_call.get("artifact_ref")


def _observation_content(tool_result: Any) -> str | dict[str, Any]:
    if isinstance(tool_result, dict):
        for key in _CONTENT_KEYS:
            if key not in tool_result:
                continue
            value = tool_result[key]
            if isinstance(value, (str, dict)):
                return value
            return str(value)
        return {key: value for key, value in tool_result.items() if key not in _REF_KEYS}
    if isinstance(tool_result, str):
        return tool_result
    return str(tool_result)


def _is_tool_part(part: Any) -> bool:
    return hasattr(part, "tool_name") and (
        hasattr(part, "tool_output") or hasattr(part, "tool_input")
    )


def _has_tool_result(part: Any) -> bool:
    if getattr(part, "tool_output", None):
        return True
    if getattr(part, "tool_output_ref", None):
        return True
    if getattr(part, "tool_output_storage_uri", None):
        return True
    return str(getattr(part, "tool_status", "") or "") in {"completed", "error"}


def _tool_call_dict(part: Any) -> dict[str, Any]:
    return {
        "tool": getattr(part, "tool_name", "") or "",
        "tool_name": getattr(part, "tool_name", "") or "",
        "tool_id": getattr(part, "tool_id", "") or "",
        "arguments": getattr(part, "tool_input", None) or {},
    }


def _tool_result_payload(part: Any) -> Any:
    artifact_ref = getattr(part, "tool_output_ref", None) or getattr(
        part, "tool_output_storage_uri", None
    )
    output = getattr(part, "tool_output", None) or ""
    if artifact_ref:
        return {"content": output, "artifact_ref": artifact_ref}
    return output


def _session_uri_path(session: Any, uri: str | None) -> Path | None:
    viking_fs = getattr(session, "_viking_fs", None)
    if viking_fs is None or not uri:
        return None
    try:
        raw = viking_fs._uri_to_path(uri, ctx=getattr(session, "ctx", None))
    except Exception:
        return None
    if not raw:
        return None
    return Path(raw)


def _get_or_create_recorder(session: Any) -> AOLedgerRecorder | None:
    recorder = getattr(session, "_ao_ledger_recorder", None)
    if recorder is not None:
        return recorder
    session_dir = _session_uri_path(session, getattr(session, "_session_uri", None))
    session_id = getattr(session, "session_id", None)
    if session_dir is None or not session_id:
        return None
    recorder = AOLedgerRecorder(session_dir, str(session_id))
    session._ao_ledger_recorder = recorder
    return recorder


class AOLedgerRecorder:
    """Per-session live capture of tool call / result pairs into ``AOLedger``."""

    def __init__(self, session_dir: Path | str, session_id: str) -> None:
        self.session_id = session_id
        self.session_dir = Path(session_dir)
        self.ledger = AOLedger(self.session_dir, session_id)

    def record_tool_exchange(
        self,
        tool_call: dict[str, Any],
        tool_result: Any,
        message_ref: dict[str, Any] | None = None,
    ) -> AORecord | None:
        """Append one AO pair. Failures are logged and do not raise."""
        try:
            action = _action_from_tool_call(dict(tool_call))
            observation = build_observation(_observation_content(tool_result))
            artifact_ref = _extract_artifact_ref(tool_call, tool_result, message_ref)
            if artifact_ref:
                observation["artifact_ref"] = artifact_ref
            return self.ledger.append(
                action=action,
                observation=observation,
                skill_invocations=extract_skill_invocations(tool_call),
                message_ref=message_ref,
            )
        except Exception:
            logger.warning("AO ledger record_tool_exchange failed", exc_info=True)
            return None

    def record_messages(
        self,
        messages: list[Any],
        lookup_messages: list[Any] | None = None,
    ) -> list[AORecord]:
        """Record completed tool exchanges found in ``messages``.

        ``lookup_messages`` (typically the live session list) supplies earlier
        tool calls so a later result can recover name/params by ``tool_id``.
        """
        call_parts: dict[str, Any] = {}
        for message in list(lookup_messages or []) + list(messages):
            for part in getattr(message, "parts", None) or []:
                if not _is_tool_part(part):
                    continue
                tool_id = str(getattr(part, "tool_id", "") or "")
                if tool_id and getattr(part, "tool_input", None):
                    call_parts[tool_id] = part

        recorded: list[AORecord] = []
        for message in messages:
            for part in getattr(message, "parts", None) or []:
                if not _is_tool_part(part) or not _has_tool_result(part):
                    continue
                tool_id = str(getattr(part, "tool_id", "") or "")
                call_part = (
                    part
                    if getattr(part, "tool_input", None)
                    else call_parts.get(tool_id, part)
                )
                message_ref: dict[str, Any] = {}
                message_id = getattr(message, "id", None)
                if message_id:
                    message_ref["message_id"] = message_id
                if tool_id:
                    message_ref["tool_id"] = tool_id
                record = self.record_tool_exchange(
                    _tool_call_dict(call_part),
                    _tool_result_payload(part),
                    message_ref or None,
                )
                if record is not None:
                    recorded.append(record)
        return recorded


def maybe_record_session_tool_exchanges(session: Any, messages: list[Any]) -> None:
    """Causal-only live hook. Legacy is a single boolean check and returns."""
    try:
        if not is_causal_mode_enabled(session_id=getattr(session, "session_id", None)):
            return
        recorder = _get_or_create_recorder(session)
        if recorder is None:
            return
        recorder.record_messages(
            messages,
            lookup_messages=getattr(session, "_messages", None),
        )
    except Exception:
        logger.warning("AO ledger recording failed", exc_info=True)


def maybe_write_session_archive_attribution(session: Any, archive_uri: str) -> None:
    """Causal-only Phase 1 hook. Writes ``ao-attribution.jsonl`` next to archive."""
    try:
        if not is_causal_mode_enabled(session_id=getattr(session, "session_id", None)):
            return
        session_dir = _session_uri_path(session, getattr(session, "_session_uri", None))
        archive_dir = _session_uri_path(session, archive_uri)
        if session_dir is None or archive_dir is None:
            return
        write_archive_attribution(session_dir, archive_dir)
    except Exception:
        logger.warning("AO archive attribution failed", exc_info=True)

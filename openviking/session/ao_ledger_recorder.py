# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Runtime AO ledger hooks: live tool-exchange capture and archive attribution.

Session wiring stays outside this module. ``session.py`` only calls the two
``maybe_*`` helpers; failures here are swallowed so the live session path
cannot break (ADR-0002: attribution is a separate append-only file).
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Callable
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


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _format_dt(value: datetime) -> str:
    return _as_utc(value).isoformat(timespec="milliseconds").replace("+00:00", "Z")


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


def write_archive_attribution(
    session_dir: Path | str,
    archive_dir: Path | str,
    *,
    clock: Callable[[], datetime] | None = None,
) -> int:
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

    resolved_clock = clock or _utc_now
    ledger = AOLedger(session_path, session_path.name, clock=resolved_clock)
    pending = [record for record in ledger.records() if record.ao_id not in attributed]
    if not pending:
        return 0

    archive_id = archive_path.name
    attributed_at = _format_dt(resolved_clock())
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
    payload: dict[str, Any] = {
        "tool": getattr(part, "tool_name", "") or "",
        "tool_name": getattr(part, "tool_name", "") or "",
        "tool_id": getattr(part, "tool_id", "") or "",
        "arguments": getattr(part, "tool_input", None) or {},
    }
    skill_uri = getattr(part, "skill_uri", "") or ""
    if skill_uri:
        payload["skill_uri"] = skill_uri
    return payload


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _exchange_dedup_key(message_ref: dict[str, Any] | None) -> tuple[str, str] | None:
    """Primary dedup key ``(message_id, tool_call_id)``.

    Both ids are required. A missing id is not a primary key; callers build a
    ``key_kind=fallback`` key instead of skipping dedup.
    """
    if not message_ref:
        return None
    message_id = str(message_ref.get("message_id") or "")
    tool_call_id = str(message_ref.get("tool_call_id") or message_ref.get("tool_id") or "")
    if not message_id or not tool_call_id:
        return None
    return message_id, tool_call_id


def _fallback_dedup_digest(
    session_id: str,
    tool_name: str,
    params: Any,
    observation_summary: str,
) -> str:
    """Stable fallback identity: session, tool, canonical params, observation summary."""
    body = {
        "session_id": session_id,
        "tool_name": tool_name,
        "params": params,
        "observation_summary": observation_summary,
    }
    return hashlib.sha256(_canonical_json(body).encode("utf-8")).hexdigest()


def _fallback_digest_from_parts(session_id: str, action: dict[str, Any], observation: dict[str, Any]) -> str | None:
    tool_name = str(action.get("tool") or "")
    if not session_id or not tool_name or "arguments" not in action:
        return None
    summary = observation.get("summary")
    if isinstance(summary, str):
        summary_text = summary
    elif isinstance(summary, list) and summary:
        summary_text = "\n".join(str(item) for item in summary)
    else:
        return None
    return _fallback_dedup_digest(session_id, tool_name, action.get("arguments"), summary_text)


def _stored_dedup_identity(record: AORecord) -> tuple[str, ...]:
    """Identity already written on a ledger row, else a recomputed fallback."""
    message_ref = record.message_ref or {}
    if message_ref.get("key_kind") == "fallback" and message_ref.get("dedup_key"):
        return ("fallback", str(message_ref["dedup_key"]))
    primary = _exchange_dedup_key(record.message_ref)
    if primary is not None:
        return ("primary", primary[0], primary[1])
    digest = _fallback_digest_from_parts(record.session_id, record.action, record.observation)
    if digest is None:
        return ()
    return ("fallback", digest)


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

    def __init__(
        self,
        session_dir: Path | str,
        session_id: str,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.session_id = session_id
        self.session_dir = Path(session_dir)
        self._clock = clock or _utc_now
        self.ledger = AOLedger(self.session_dir, session_id, clock=self._clock)
        self.duplicate_skip_count = 0
        self._seen_exchange_keys: set[tuple[str, ...]] = set()
        self._dedup_lock = threading.Lock()
        for record in self.ledger.records():
            key = _stored_dedup_identity(record)
            if key:
                self._seen_exchange_keys.add(key)

    def record_tool_exchange(
        self,
        tool_call: dict[str, Any],
        tool_result: Any,
        message_ref: dict[str, Any] | None = None,
    ) -> AORecord | None:
        """Append one AO pair.

        Dedup prefers ``(message_id, tool_call_id)`` (``tool_call_id`` or the
        session ``tool_id`` alias). When either id is missing, a fallback key
        is recorded with ``key_kind=fallback``: ``session_id``, tool name,
        canonical parameter hash, and the paired observation-summary hash.
        A repeat of either key is skipped, counted on ``duplicate_skip_count``,
        and logged at debug. If neither key can be built, the AO is not written
        and ``ValueError`` is raised. Other failures are logged and do not raise.
        """
        claimed = False
        dedup_key: tuple[str, ...] | None = None
        try:
            primary = _exchange_dedup_key(message_ref)
            action = _action_from_tool_call(dict(tool_call))
            observation: dict[str, Any] | None = None
            if primary is not None:
                dedup_key = ("primary", primary[0], primary[1])
            else:
                if tool_result is not None and action.get("tool") and "arguments" in action:
                    observation = build_observation(_observation_content(tool_result))
                    digest = _fallback_digest_from_parts(self.session_id, action, observation)
                else:
                    digest = None
                if digest is None:
                    raise ValueError(
                        "cannot construct AO dedup key: message_id/tool_call_id missing "
                        "and fallback key incomplete"
                    )
                dedup_key = ("fallback", digest)
            with self._dedup_lock:
                if dedup_key in self._seen_exchange_keys:
                    self.duplicate_skip_count += 1
                    logger.debug(
                        "skip duplicate AO record key_kind=%s dedup_key=%s",
                        dedup_key[0],
                        dedup_key[-1],
                    )
                    return None
                self._seen_exchange_keys.add(dedup_key)
                claimed = True
            if observation is None:
                observation = build_observation(_observation_content(tool_result))
            artifact_ref = _extract_artifact_ref(tool_call, tool_result, message_ref)
            if artifact_ref:
                observation["artifact_ref"] = artifact_ref
            stored_ref = dict(message_ref) if message_ref else None
            if dedup_key[0] == "fallback":
                stored_ref = dict(stored_ref or {})
                stored_ref["key_kind"] = "fallback"
                stored_ref["dedup_key"] = dedup_key[1]
            return self.ledger.append(
                action=action,
                observation=observation,
                skill_invocations=extract_skill_invocations(
                    dict(tool_call), session_id=self.session_id
                ),
                message_ref=stored_ref,
            )
        except ValueError:
            if claimed and dedup_key is not None:
                with self._dedup_lock:
                    self._seen_exchange_keys.discard(dedup_key)
            if claimed:
                logger.warning("AO ledger record_tool_exchange failed", exc_info=True)
                return None
            raise
        except Exception:
            if claimed and dedup_key is not None:
                with self._dedup_lock:
                    self._seen_exchange_keys.discard(dedup_key)
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

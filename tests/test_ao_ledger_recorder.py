# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

"""Unit tests for AO ledger session hooks (slice S2/6)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from openviking.message import Message, TextPart, ToolPart
from openviking.session.ao_ledger import AOLedger, AORecord, LEDGER_FILENAME
from openviking.session.ao_ledger_recorder import (
    AOLedgerRecorder,
    ATTRIBUTION_FILENAME,
    collect_attributed_ao_ids,
    maybe_record_session_tool_exchanges,
    maybe_write_session_archive_attribution,
    write_archive_attribution,
)
from openviking.session.session import Session


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tool_call(name: str = "read_file", **arguments: object) -> dict:
    return {"tool": name, "arguments": dict(arguments) or {"path": "notes.md"}}


class _DummyFS:
    def __init__(self, root: Path, session_uri: str) -> None:
        self.root = Path(root)
        self.session_uri = session_uri
        self.files: dict[str, str] = {}
        self._async_agfs = AsyncMock()

    def _uri_to_path(self, uri: str, ctx=None) -> str:
        if uri == self.session_uri:
            return str(self.root)
        prefix = self.session_uri + "/"
        if uri.startswith(prefix):
            return str(self.root / uri[len(prefix) :])
        return str(self.root)

    async def read_file(self, uri, ctx=None):
        if uri not in self.files:
            raise FileNotFoundError(uri)
        return self.files[uri]

    async def write_file(self, uri, content, ctx=None, lease_ref=None):
        self.files[uri] = content

    async def append_file(self, uri, content, ctx=None):
        self.files[uri] = self.files.get(uri, "") + content

    async def exists(self, uri, ctx=None):
        return uri in self.files or any(path.startswith(f"{uri}/") for path in self.files)

    async def ls(self, uri, ctx=None):
        prefix = f"{uri}/"
        names = {
            path[len(prefix) :].split("/")[0] for path in self.files if path.startswith(prefix)
        }
        return [{"name": name} for name in sorted(names)]


def test_record_tool_exchange_dict_and_str(tmp_path: Path) -> None:
    recorder = AOLedgerRecorder(tmp_path, "sess-rec")

    dict_record = recorder.record_tool_exchange(
        _tool_call(path="a.md"),
        {"ok": True, "count": 2},
        {"message_id": "m-dict"},
    )
    str_record = recorder.record_tool_exchange(
        _tool_call(path="b.md"),
        "plain tool output without structure",
        {"message_id": "m-str"},
    )

    assert isinstance(dict_record, AORecord)
    assert isinstance(str_record, AORecord)
    assert dict_record is not None and str_record is not None
    assert dict_record.action["tool"] == "read_file"
    assert str_record.action["tool"] == "read_file"
    assert "path" in json.dumps(dict_record.action["arguments"])
    for record in (dict_record, str_record):
        assert record.skill_invocations == []
        assert {"kind", "title", "summary", "structure", "notable_items", "sample"} <= set(
            record.observation
        )
        assert "artifact_ref" not in record.observation
    assert recorder.ledger.count() == 2


def test_fallback_dedup_key_skips_repeat_without_message_id(tmp_path: Path) -> None:
    recorder = AOLedgerRecorder(tmp_path / "sess-fallback", "sess-fallback")
    call = {"tool": "read_file", "arguments": {"path": "a.md", "n": 1}}
    first = recorder.record_tool_exchange(call, "observed text", None)
    swapped = {"tool": "read_file", "arguments": {"n": 1, "path": "a.md"}}
    second = recorder.record_tool_exchange(swapped, "observed text", None)
    assert first is not None
    assert second is None
    assert recorder.ledger.count() == 1
    assert recorder.duplicate_skip_count == 1
    assert first.message_ref is not None
    assert first.message_ref["key_kind"] == "fallback"
    assert first.message_ref["dedup_key"]

    reloaded = AOLedgerRecorder(tmp_path / "sess-fallback", "sess-fallback")
    assert reloaded.record_tool_exchange(call, "observed text", None) is None
    assert reloaded.ledger.count() == 1
    assert reloaded.duplicate_skip_count == 1


def test_unkeyed_exchange_is_rejected(tmp_path: Path) -> None:
    recorder = AOLedgerRecorder(tmp_path / "sess-nokey", "sess-nokey")
    with pytest.raises(ValueError, match="dedup key"):
        recorder.record_tool_exchange({}, None, None)
    assert recorder.ledger.count() == 0
    assert not (tmp_path / "sess-nokey" / LEDGER_FILENAME).exists()


def test_record_messages_skips_duplicate_message_and_tool_call(tmp_path: Path) -> None:
    recorder = AOLedgerRecorder(tmp_path / "sess-dedup", "sess-dedup")
    tool_msg = Message(
        id="m-dup",
        role="assistant",
        parts=[
            ToolPart(
                tool_id="call-1",
                tool_name="read_file",
                tool_input={"path": "a.md"},
                tool_output="hello",
                tool_status="completed",
            )
        ],
    )
    first = recorder.record_messages([tool_msg])
    second = recorder.record_messages([tool_msg])
    assert len(first) == 1
    assert second == []
    assert recorder.ledger.count() == 1
    assert recorder.duplicate_skip_count == 1

    reloaded = AOLedgerRecorder(tmp_path / "sess-dedup", "sess-dedup")
    assert reloaded.record_messages([tool_msg]) == []
    assert reloaded.ledger.count() == 1
    assert reloaded.duplicate_skip_count == 1


def test_record_tool_exchange_adds_artifact_ref(tmp_path: Path) -> None:
    recorder = AOLedgerRecorder(tmp_path, "sess-ref")
    artifact = "viking://user/default/sessions/s/tool-results/tr_abc"
    record = recorder.record_tool_exchange(
        _tool_call(path="big.md"),
        {
            "content": "x" * 80,
            "artifact_ref": artifact,
        },
    )
    assert record is not None
    assert record.observation["artifact_ref"] == artifact
    assert record.observation["kind"]
    assert record.skill_invocations == []


def test_write_archive_attribution_is_incremental_and_idempotent(tmp_path: Path) -> None:
    session_dir = tmp_path / "sess-attr"
    ledger = AOLedger(session_dir, "sess-attr")
    first_ids = [
        ledger.append(
            {"tool": "t", "arguments": {"n": idx}},
            {"kind": "text", "title": str(idx)},
        ).ao_id
        for idx in range(3)
    ]
    ledger_path = session_dir / LEDGER_FILENAME
    before = ledger_path.read_bytes()
    before_hash = _sha256(ledger_path)

    archive_1 = session_dir / "history" / "archive_001"
    written = write_archive_attribution(session_dir, archive_1)
    assert written == 3
    attr_1 = archive_1 / ATTRIBUTION_FILENAME
    lines_1 = [json.loads(line) for line in attr_1.read_text(encoding="utf-8").splitlines() if line]
    assert [row["ao_id"] for row in lines_1] == first_ids
    assert all(row["archive_id"] == "archive_001" for row in lines_1)
    assert all(row["commit_watermark"].startswith("archive_001@") for row in lines_1)

    assert write_archive_attribution(session_dir, archive_1) == 0
    assert len(attr_1.read_text(encoding="utf-8").splitlines()) == 3

    after_first = ledger_path.read_bytes()
    assert after_first == before
    assert _sha256(ledger_path) == before_hash

    extra_ids = [
        ledger.append(
            {"tool": "t", "arguments": {"n": idx}},
            {"kind": "text", "title": str(idx)},
        ).ao_id
        for idx in (3, 4)
    ]
    after_append_hash = _sha256(ledger_path)

    archive_2 = session_dir / "history" / "archive_002"
    written_2 = write_archive_attribution(session_dir, archive_2)
    assert written_2 == 2
    lines_2 = [
        json.loads(line)
        for line in (archive_2 / ATTRIBUTION_FILENAME).read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert [row["ao_id"] for row in lines_2] == extra_ids
    assert collect_attributed_ao_ids(session_dir) == set(first_ids + extra_ids)
    assert _sha256(ledger_path) == after_append_hash


def test_attribution_leaves_ledger_bytes_unchanged(tmp_path: Path) -> None:
    session_dir = tmp_path / "sess-hash"
    ledger = AOLedger(session_dir, "sess-hash")
    for idx in range(3):
        ledger.append({"tool": "hash", "arguments": {"n": idx}}, {"kind": "text", "title": "t"})
    ledger_path = session_dir / LEDGER_FILENAME
    before = _sha256(ledger_path)
    write_archive_attribution(session_dir, session_dir / "history" / "archive_001")
    assert _sha256(ledger_path) == before
    assert ledger_path.read_bytes() == (session_dir / LEDGER_FILENAME).read_bytes()


def test_collect_attributed_ao_ids_missing_files_are_empty(tmp_path: Path) -> None:
    assert collect_attributed_ao_ids(tmp_path / "missing") == set()
    session_dir = tmp_path / "empty-hist"
    (session_dir / "history" / "archive_001").mkdir(parents=True)
    assert collect_attributed_ao_ids(session_dir) == set()


def test_legacy_hooks_are_noop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "openviking.session.ao_ledger_recorder.is_causal_mode_enabled",
        lambda **_kwargs: False,
    )
    session_uri = "viking://user/default/sessions/session-1"
    storage = _DummyFS(tmp_path, session_uri)
    session = SimpleNamespace(
        session_id="session-1",
        _session_uri=session_uri,
        ctx=None,
        _viking_fs=storage,
        _messages=[],
    )
    tool_msg = Message(
        id="m1",
        role="user",
        parts=[
            ToolPart(
                tool_id="t1",
                tool_name="read_file",
                tool_input={"path": "a.md"},
                tool_output="hello",
                tool_status="completed",
            )
        ],
    )
    maybe_record_session_tool_exchanges(session, [tool_msg])
    maybe_write_session_archive_attribution(
        session, f"{session_uri}/history/archive_001"
    )
    assert not (tmp_path / LEDGER_FILENAME).exists()
    assert not (tmp_path / "history" / "archive_001" / ATTRIBUTION_FILENAME).exists()
    assert getattr(session, "_ao_ledger_recorder", None) is None


def test_causal_hooks_record_and_attribute(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "openviking.session.ao_ledger_recorder.is_causal_mode_enabled",
        lambda **_kwargs: True,
    )
    session_uri = "viking://user/default/sessions/session-1"
    storage = _DummyFS(tmp_path, session_uri)
    session = SimpleNamespace(
        session_id="session-1",
        _session_uri=session_uri,
        ctx=None,
        _viking_fs=storage,
        _messages=[],
    )
    tool_msg = Message(
        id="m1",
        role="user",
        parts=[
            ToolPart(
                tool_id="t1",
                tool_name="read_file",
                tool_input={"path": "a.md"},
                tool_output="hello from tool",
                tool_status="completed",
            )
        ],
    )
    maybe_record_session_tool_exchanges(session, [tool_msg])
    ledger_path = tmp_path / LEDGER_FILENAME
    assert ledger_path.is_file()
    records = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    assert records[0]["action"]["tool"] == "read_file"
    assert records[0]["skill_invocations"] == []

    archive_uri = f"{session_uri}/history/archive_001"
    maybe_write_session_archive_attribution(session, archive_uri)
    attr_path = tmp_path / "history" / "archive_001" / ATTRIBUTION_FILENAME
    assert attr_path.is_file()
    rows = [json.loads(line) for line in attr_path.read_text(encoding="utf-8").splitlines()]
    assert [row["ao_id"] for row in rows] == [records[0]["ao_id"]]
    assert rows[0]["archive_id"] == "archive_001"


@pytest.mark.asyncio
async def test_session_commit_seam_causal_vs_legacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Seam-level Session commit: causal writes attribution; legacy writes neither file.

    Uses the in-memory VikingFS pattern from tests/unit/session/test_session_commit_resume.py
    so this does not need cluster.json / a live service.
    """
    from openviking.service.task_tracker import TaskTracker, set_task_tracker

    class _TaskStore:
        def __init__(self):
            self.tasks = {}

        async def create(self, task):
            self.tasks[task.task_id] = task

        async def update(self, task):
            self.tasks[task.task_id] = task

        async def get(self, task_id, *, account_id=None, user_id=None):
            return None

        async def list(self, account_id, *, user_id=None):
            return []

        async def delete(self, task_id, *, account_id, user_id=None):
            self.tasks.pop(task_id, None)

    session_uri = "viking://user/default/sessions/session-1"

    async def _commit_once(*, causal: bool, root: Path) -> None:
        monkeypatch.setattr(
            "openviking.session.ao_ledger_recorder.is_causal_mode_enabled",
            lambda **_kwargs: causal,
        )
        storage = _DummyFS(root, session_uri)
        tracker = TaskTracker(_TaskStore())
        monkeypatch.setattr("openviking.session.session._enabled_memory_types", lambda: set())
        monkeypatch.setattr("openviking.service.task_tracker.get_task_tracker", lambda: tracker)
        monkeypatch.setattr(
            "openviking.storage.queuefs.get_queue_manager",
            lambda: SimpleNamespace(enqueue=AsyncMock()),
        )
        set_task_tracker(tracker)
        session = Session(viking_fs=storage, session_id="session-1", session_uri=session_uri)
        try:
            await session.add_messages_async(
                [
                    {
                        "role": "assistant",
                        "parts": [
                            ToolPart(
                                tool_id="t1",
                                tool_name="read_file",
                                tool_input={"path": "a.md"},
                                tool_output="tool body",
                                tool_status="completed",
                            )
                        ],
                    }
                ]
            )
            result = await session.commit_async(keep_recent_count=0)
            assert result.get("archived") is True
        finally:
            set_task_tracker(None)

    causal_root = tmp_path / "causal"
    causal_root.mkdir()
    await _commit_once(causal=True, root=causal_root)
    assert (causal_root / LEDGER_FILENAME).is_file()
    attr = causal_root / "history" / "archive_001" / ATTRIBUTION_FILENAME
    assert attr.is_file()
    assert len(attr.read_text(encoding="utf-8").splitlines()) == 1

    legacy_root = tmp_path / "legacy"
    legacy_root.mkdir()
    await _commit_once(causal=False, root=legacy_root)
    assert not (legacy_root / LEDGER_FILENAME).exists()
    assert not (legacy_root / "history" / "archive_001" / ATTRIBUTION_FILENAME).exists()

# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

"""Skill invocation capture into the AO ledger (slice S11)."""

from __future__ import annotations

from pathlib import Path

from openviking.session.ao_ledger import UNKNOWN_SKILL_FIELD, note_skill_invocation
from openviking.session.ao_ledger_recorder import AOLedgerRecorder


def test_registered_skill_execution_is_captured_on_matching_tool_call(tmp_path: Path) -> None:
    session_id = "sess-skill"
    note_skill_invocation(
        session_id,
        skill_uri="viking://user/default/skills/git-rescue",
        revision_hash="sha256:abc",
        invocation_id="inv-1",
        tool_call_id="call-9",
    )
    recorder = AOLedgerRecorder(tmp_path / session_id, session_id)
    record = recorder.record_tool_exchange(
        {"tool": "read_file", "tool_id": "call-9", "arguments": {"path": "a.md"}},
        "rescued",
        {"message_id": "m1", "tool_id": "call-9"},
    )
    assert record is not None
    assert record.skill_invocations == [
        {
            "skill_uri": "viking://user/default/skills/git-rescue",
            "revision_hash": "sha256:abc",
            "invocation_id": "inv-1",
        }
    ]

    other = recorder.record_tool_exchange(
        {"tool": "read_file", "tool_id": "call-9", "arguments": {"path": "b.md"}},
        "again",
        {"message_id": "m2", "tool_id": "call-9"},
    )
    assert other is not None
    assert other.skill_invocations == []


def test_missing_revision_hash_is_unknown_not_invented(tmp_path: Path) -> None:
    session_id = "sess-skill-unknown"
    note_skill_invocation(
        session_id,
        skill_uri="viking://user/default/skills/git-rescue",
        tool_call_id="call-1",
    )
    recorder = AOLedgerRecorder(tmp_path / session_id, session_id)
    record = recorder.record_tool_exchange(
        {"tool": "read_file", "tool_id": "call-1", "arguments": {}},
        "ok",
        {"message_id": "m1", "tool_id": "call-1"},
    )
    assert record is not None
    assert record.skill_invocations == [
        {
            "skill_uri": "viking://user/default/skills/git-rescue",
            "revision_hash": UNKNOWN_SKILL_FIELD,
        }
    ]

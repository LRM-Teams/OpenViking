# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

"""Unit tests for the session AO ledger (slice S1/6)."""

from __future__ import annotations

import json
import threading
from dataclasses import FrozenInstanceError, fields
from datetime import datetime, timezone
from pathlib import Path

from openviking.session.ao_ledger import (
    AOAttribution,
    AOLedger,
    AORecord,
    LEDGER_FILENAME,
    build_observation,
    extract_skill_invocations,
)


def _action(tool: str = "read_file", **params: object) -> dict:
    return {"tool": tool, "arguments": dict(params) or {"path": "notes.md"}}


def _observation(title: str = "ok") -> dict:
    return {
        "kind": "text",
        "title": title,
        "summary": ["ok"],
        "structure": [],
        "notable_items": [],
        "sample": "",
    }


def test_ao_id_is_globally_unique_across_sessions(tmp_path: Path) -> None:
    ledger_a = AOLedger(tmp_path / "sess-a", "sess-a")
    ledger_b = AOLedger(tmp_path / "sess-b", "sess-b")
    for idx in range(4):
        ledger_a.append(_action(idx=idx), _observation(f"a-{idx}"))
        ledger_b.append(_action(idx=idx), _observation(f"b-{idx}"))

    all_ids = [record.ao_id for record in ledger_a.records()] + [
        record.ao_id for record in ledger_b.records()
    ]
    assert len(all_ids) == 8
    assert len(set(all_ids)) == 8
    assert all(len(ao_id) == 32 for ao_id in all_ids)


def test_sequence_is_strictly_increasing_and_gapless_under_concurrency(tmp_path: Path) -> None:
    ledger = AOLedger(tmp_path / "sess-c", "sess-c")
    errors: list[BaseException] = []

    def _worker(label: str) -> None:
        try:
            for idx in range(50):
                ledger.append(_action(worker=label, idx=idx), _observation(f"{label}-{idx}"))
        except BaseException as exc:  # noqa: BLE001 — surface in parent thread
            errors.append(exc)

    threads = [
        threading.Thread(target=_worker, args=("t1",)),
        threading.Thread(target=_worker, args=("t2",)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    sequences = [record.sequence for record in ledger.records()]
    assert sequences == list(range(1, 101))
    assert ledger.count() == 100
    assert len({record.ao_id for record in ledger.records()}) == 100


def test_aorecord_is_immutable_without_archive_fields_and_roundtrips(tmp_path: Path) -> None:
    ledger = AOLedger(tmp_path / "sess-d", "sess-d")
    record = ledger.append(
        _action(path="demo.md"),
        _observation("demo"),
        message_ref={"message_id": "m1", "span": [0, 12]},
    )

    names = {item.name for item in fields(AORecord)}
    assert "archive_id" not in names
    assert "commit_watermark" not in names
    assert "attributed_at" not in names
    assert not any("archive" in name for name in names)
    assert record.captured_state == "live"

    attr_names = {item.name for item in fields(AOAttribution)}
    assert attr_names == {"ao_id", "archive_id", "commit_watermark", "attributed_at"}

    try:
        record.sequence = 99  # type: ignore[misc]
        raised = False
    except FrozenInstanceError:
        raised = True
    assert raised

    restored = AORecord.from_dict(record.to_dict())
    assert restored == record
    json_restored = AORecord.from_dict(json.loads(json.dumps(record.to_dict())))
    assert json_restored == record


def test_jsonl_persists_and_skips_corrupt_lines(tmp_path: Path) -> None:
    session_dir = tmp_path / "sess-e"
    ledger = AOLedger(session_dir, "sess-e")
    first = ledger.append(_action(path="one.md"), _observation("one"))
    second = ledger.append(_action(path="two.md"), _observation("two"))

    reloaded = AOLedger(session_dir, "sess-e")
    assert reloaded.count() == 2
    assert reloaded.get(first.ao_id) == first
    assert reloaded.get(second.ao_id) == second
    assert [item.sequence for item in reloaded.records()] == [1, 2]
    assert reloaded.records(start_sequence=2, end_sequence=2) == [second]

    ledger_path = session_dir / "ao-ledger.jsonl"
    with ledger_path.open("a", encoding="utf-8") as handle:
        handle.write("{not-valid-json\n")
        handle.write("null\n")

    tolerant = AOLedger(session_dir, "sess-e")
    assert tolerant.count() == 2
    assert tolerant.skipped_line_count == 2
    assert tolerant.get(first.ao_id) is not None
    assert tolerant.get("missing") is None


def test_build_observation_covers_json_string_and_plain_text() -> None:
    json_obs = build_observation('{"ok": true, "count": 3}')
    assert json_obs["kind"]
    assert json_obs["title"]
    assert {"kind", "title", "summary", "structure", "notable_items", "sample"} <= set(json_obs)

    text_obs = build_observation("plain tool output without structure")
    assert text_obs["kind"]
    assert text_obs["title"]
    assert {"kind", "title", "summary", "structure", "notable_items", "sample"} <= set(text_obs)


def test_skill_invocations_default_to_empty_list_and_are_serialized(tmp_path: Path) -> None:
    assert extract_skill_invocations({"tool": "read_file"}) == []

    ledger = AOLedger(tmp_path / "sess-f", "sess-f")
    record = ledger.append(_action(), _observation())
    assert record.skill_invocations == []
    payload = record.to_dict()
    assert "skill_invocations" in payload
    assert payload["skill_invocations"] == []
    assert json.loads(json.dumps(payload))["skill_invocations"] == []


def test_injected_clock_stamps_persisted_created_at(tmp_path: Path) -> None:
    fixed = datetime(2026, 4, 5, 6, 7, 8, 901000, tzinfo=timezone.utc)
    expected = "2026-04-05T06:07:08.901Z"
    session_dir = tmp_path / "sess-clock"
    ledger = AOLedger(session_dir, "sess-clock", clock=lambda: fixed)
    record = ledger.append(_action(), _observation())
    assert record.created_at == expected
    line = json.loads((session_dir / LEDGER_FILENAME).read_text(encoding="utf-8").strip())
    assert line["created_at"] == expected
    reloaded = AOLedger(session_dir, "sess-clock")
    assert reloaded.records()[0].created_at == expected

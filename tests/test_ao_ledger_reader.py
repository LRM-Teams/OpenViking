# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

"""Unit tests for AO ledger read APIs (slice S4/6)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openviking.session.ao_ledger import AOAttribution, AOLedger
from openviking.session.ao_ledger_recorder import ATTRIBUTION_FILENAME
from openviking.session.ao_ledger_reader import (
    MAX_PAGE_LIMIT,
    LedgerSnapshot,
    get_ao_evidence,
    issue_snapshot,
    list_ao_ledger,
)


def _action(n: int) -> dict:
    return {"tool": "read_file", "arguments": {"n": n}}


def _observation(n: int) -> dict:
    return {
        "kind": "text",
        "title": str(n),
        "summary": [str(n)],
        "structure": [],
        "notable_items": [],
        "sample": "",
    }


def _append_n(ledger: AOLedger, count: int) -> list:
    return [ledger.append(_action(idx), _observation(idx)) for idx in range(count)]


def _write_attribution(session_dir: Path, archive_id: str, records: list) -> None:
    archive = session_dir / "history" / archive_id
    archive.mkdir(parents=True, exist_ok=True)
    dest = archive / ATTRIBUTION_FILENAME
    attributed_at = "2026-09-23T00:00:00.000Z"
    with dest.open("a", encoding="utf-8") as handle:
        handle.write("{not-valid-json\n")
        handle.write("null\n")
        for record in records:
            attr = AOAttribution(
                ao_id=record.ao_id,
                archive_id=archive_id,
                commit_watermark=f"{archive_id}@{attributed_at}",
                attributed_at=attributed_at,
            )
            handle.write(json.dumps(attr.to_dict(), ensure_ascii=False) + "\n")


def test_snapshot_freezes_ledger_upper_bound(tmp_path: Path) -> None:
    session_dir = tmp_path / "sess-freeze"
    session_id = "sess-freeze"
    ledger = AOLedger(session_dir, session_id)
    _append_n(ledger, 3)

    frozen = issue_snapshot(session_dir, session_id)
    assert frozen.ledger_line_count == 3
    _append_n(ledger, 2)

    page = list_ao_ledger(
        session_dir,
        session_id,
        snapshot_watermark=frozen.watermark,
        include_live=True,
    )
    assert len(page.items) == 3
    assert page.total_in_snapshot == 3
    assert page.snapshot_watermark == frozen.watermark
    assert [item["sequence"] for item in page.items] == [1, 2, 3]

    reissued = list_ao_ledger(session_dir, session_id, include_live=True)
    assert len(reissued.items) == 5
    assert reissued.total_in_snapshot == 5
    assert reissued.snapshot_watermark != frozen.watermark
    assert [item["sequence"] for item in reissued.items] == [1, 2, 3, 4, 5]


def test_cursor_pagination_and_watermark_mismatch(tmp_path: Path) -> None:
    session_dir = tmp_path / "sess-page"
    session_id = "sess-page"
    ledger = AOLedger(session_dir, session_id)
    records = _append_n(ledger, 5)
    expected_ids = [record.ao_id for record in records]

    first = list_ao_ledger(session_dir, session_id, limit=2, include_live=True)
    watermark = first.snapshot_watermark
    assert [item["ao_id"] for item in first.items] == expected_ids[:2]
    assert first.next_cursor is not None
    assert first.total_in_snapshot == 5

    second = list_ao_ledger(
        session_dir,
        session_id,
        cursor=first.next_cursor,
        limit=2,
        snapshot_watermark=watermark,
        include_live=True,
    )
    assert [item["ao_id"] for item in second.items] == expected_ids[2:4]
    assert second.next_cursor is not None
    assert second.snapshot_watermark == watermark

    third = list_ao_ledger(
        session_dir,
        session_id,
        cursor=second.next_cursor,
        limit=2,
        snapshot_watermark=watermark,
        include_live=True,
    )
    assert [item["ao_id"] for item in third.items] == expected_ids[4:]
    assert third.next_cursor is None

    seen = [item["ao_id"] for item in first.items + second.items + third.items]
    assert seen == expected_ids
    assert len(set(seen)) == 5

    other = issue_snapshot(session_dir, session_id)
    assert other.watermark != watermark
    with pytest.raises(ValueError):
        list_ao_ledger(
            session_dir,
            session_id,
            cursor=first.next_cursor,
            snapshot_watermark=other.watermark,
            include_live=True,
        )
    with pytest.raises(ValueError):
        list_ao_ledger(
            session_dir,
            session_id,
            cursor=first.next_cursor,
            include_live=True,
        )


def test_committed_semantics_and_include_live(tmp_path: Path) -> None:
    session_dir = tmp_path / "sess-commit"
    session_id = "sess-commit"
    ledger = AOLedger(session_dir, session_id)
    records = _append_n(ledger, 3)
    _write_attribution(session_dir, "archive_001", records[:2])

    committed = list_ao_ledger(session_dir, session_id, include_live=False)
    assert committed.live_included is False
    assert len(committed.items) == 2
    assert committed.total_in_snapshot == 2
    assert {item["ao_id"] for item in committed.items} == {records[0].ao_id, records[1].ao_id}
    assert all(item["committed"] is True for item in committed.items)
    assert all(item["archive_id"] == "archive_001" for item in committed.items)

    live = list_ao_ledger(
        session_dir,
        session_id,
        snapshot_watermark=committed.snapshot_watermark,
        include_live=True,
    )
    assert live.live_included is True
    assert len(live.items) == 3
    assert live.total_in_snapshot == 3
    by_id = {item["ao_id"]: item for item in live.items}
    assert by_id[records[2].ao_id]["archive_id"] is None
    assert by_id[records[2].ao_id]["committed"] is False
    assert by_id[records[0].ao_id]["committed"] is True
    assert by_id[records[0].ao_id]["archive_id"] == "archive_001"


def test_get_ao_evidence_attribution_and_acl(tmp_path: Path) -> None:
    session_dir = tmp_path / "sess-ev"
    session_id = "sess-ev"
    ledger = AOLedger(session_dir, session_id)
    attributed, live = _append_n(ledger, 2)
    _write_attribution(session_dir, "archive_001", [attributed])

    committed = get_ao_evidence(session_dir, attributed.ao_id)
    assert committed is not None
    assert committed["record"]["ao_id"] == attributed.ao_id
    assert committed["attribution"] is not None
    assert committed["attribution"]["archive_id"] == "archive_001"
    assert committed["attribution"]["ao_id"] == attributed.ao_id

    unattributed = get_ao_evidence(session_dir, live.ao_id)
    assert unattributed is not None
    assert unattributed["record"]["ao_id"] == live.ao_id
    assert unattributed["attribution"] is None

    assert get_ao_evidence(session_dir, "missing-ao") is None

    with pytest.raises(PermissionError):
        get_ao_evidence(
            session_dir,
            attributed.ao_id,
            acl_check=lambda _session, _record: False,
        )

    allowed = get_ao_evidence(
        session_dir,
        attributed.ao_id,
        acl_check=lambda source_session_id, record: (
            source_session_id == session_id and record["ao_id"] == attributed.ao_id
        ),
    )
    assert allowed is not None
    assert allowed["record"]["ao_id"] == attributed.ao_id


def test_watermark_encode_decode_roundtrip(tmp_path: Path) -> None:
    session_dir = tmp_path / "sess-wm"
    session_id = "sess-wm"
    ledger = AOLedger(session_dir, session_id)
    _append_n(ledger, 2)
    (session_dir / "history" / "archive_001").mkdir(parents=True)

    snapshot = issue_snapshot(session_dir, session_id)
    restored = LedgerSnapshot.decode(snapshot.encode())
    assert restored.session_id == snapshot.session_id
    assert restored.ledger_line_count == snapshot.ledger_line_count
    assert restored.archive_ids == snapshot.archive_ids
    assert restored.issued_at == snapshot.issued_at
    assert restored.watermark == snapshot.watermark
    assert restored.encode() == snapshot.encode()
    assert LedgerSnapshot.decode(snapshot.watermark) == restored

    over_limit = list_ao_ledger(
        session_dir,
        session_id,
        limit=MAX_PAGE_LIMIT + 50,
        include_live=True,
    )
    assert len(over_limit.items) == 2
    assert over_limit.total_in_snapshot == 2

# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Read-only AO ledger APIs: snapshot pagination and evidence drill-down.

``list_ao_ledger`` pages a watermark-frozen view (archive set + live upper
bound). ``get_ao_evidence`` returns one record plus its attribution; an
optional ``acl_check`` hook is the source-session ACL authority (Q28).
When the hook is omitted, a registered ``AclPolicy`` is consulted.
Unregistered sessions are default-deny; nothing silently allows every read.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from openviking.session.ao_ledger import AOAttribution, AOLedger, AORecord
from openviking.session.ao_ledger_recorder import (
    ATTRIBUTION_FILENAME,
    HISTORY_DIRNAME,
    collect_attributed_ao_ids,
)

MAX_PAGE_LIMIT = 200
_CURSOR_PREFIX = "seq:"
_SNAPSHOT_VERSION = 1

logger = logging.getLogger(__name__)


class AclPolicy(Protocol):
    """Source-session visibility policy. Implementations must fail closed."""

    def allows(self, session_id: str, record: dict[str, Any]) -> bool:
        """Return whether ``record`` from ``session_id`` may be read as evidence."""


class DefaultDenyAcl:
    """Explicit deny. Used when no session policy has been registered."""

    def allows(self, session_id: str, record: dict[str, Any]) -> bool:
        del session_id, record
        return False


_acl_lock = threading.Lock()
_session_acls: dict[str, AclPolicy] = {}
_DEFAULT_ACL = DefaultDenyAcl()


def register_session_acl(session_id: str, policy: AclPolicy) -> None:
    """Install the visibility policy for one source session.

    Later ``get_ao_evidence`` calls that omit ``acl_check`` use this policy.
    Registering is the only way to permit a read; the default is deny.
    """
    if not session_id:
        raise ValueError("session_id is required")
    with _acl_lock:
        _session_acls[session_id] = policy


def _session_acl(session_id: str) -> AclPolicy:
    with _acl_lock:
        return _session_acls.get(session_id, _DEFAULT_ACL)


def _registered_acl_check(session_id: str, record: dict[str, Any]) -> bool:
    policy = _session_acl(session_id)
    try:
        return bool(policy.allows(session_id, record))
    except Exception:
        logger.warning("session ACL check failed; denying", exc_info=True)
        return False


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _encode_payload(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return base64.urlsafe_b64encode(raw.encode("utf-8")).decode("ascii")


def _decode_payload(token: str) -> dict[str, Any]:
    padded = token + "=" * (-len(token) % 4)
    try:
        raw = base64.urlsafe_b64decode(padded.encode("ascii"))
        payload = json.loads(raw.decode("utf-8"))
    except (ValueError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("invalid snapshot watermark") from exc
    if not isinstance(payload, dict):
        raise ValueError("invalid snapshot watermark")
    return payload


def _watermark_fingerprint(watermark: str) -> str:
    return hashlib.sha256(watermark.encode("utf-8")).hexdigest()[:16]


def _encode_cursor(sequence: int, watermark: str) -> str:
    return f"{_CURSOR_PREFIX}{sequence}:{_watermark_fingerprint(watermark)}"


def _parse_cursor(cursor: str) -> tuple[int, str | None]:
    if not cursor.startswith(_CURSOR_PREFIX):
        raise ValueError(f"invalid cursor: {cursor!r}")
    rest = cursor[len(_CURSOR_PREFIX) :]
    if not rest:
        raise ValueError(f"invalid cursor: {cursor!r}")
    sequence_text, _sep, fingerprint = rest.partition(":")
    try:
        sequence = int(sequence_text)
    except ValueError as exc:
        raise ValueError(f"invalid cursor: {cursor!r}") from exc
    if sequence < 0:
        raise ValueError(f"invalid cursor: {cursor!r}")
    return sequence, fingerprint or None


def _existing_archive_ids(session_dir: Path) -> tuple[str, ...]:
    history = session_dir / HISTORY_DIRNAME
    if not history.is_dir():
        return ()
    try:
        names = [child.name for child in history.iterdir() if child.is_dir()]
    except OSError:
        return ()
    return tuple(sorted(names))


def _read_attributions(path: Path) -> dict[str, AOAttribution]:
    """Parse one ``ao-attribution.jsonl``. Missing/corrupt lines are skipped."""
    found: dict[str, AOAttribution] = {}
    if not path.is_file():
        return found
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    payload = json.loads(raw)
                    if not isinstance(payload, dict):
                        continue
                    attr = AOAttribution.from_dict(payload)
                except (json.JSONDecodeError, TypeError, KeyError, ValueError):
                    continue
                found.setdefault(attr.ao_id, attr)
    except OSError:
        return {}
    return found


def _attributions_for_archives(
    session_dir: Path, archive_ids: tuple[str, ...]
) -> dict[str, AOAttribution]:
    found: dict[str, AOAttribution] = {}
    history = session_dir / HISTORY_DIRNAME
    for archive_id in archive_ids:
        found.update(_read_attributions(history / archive_id / ATTRIBUTION_FILENAME))
    return found


def _find_attribution(session_dir: Path, ao_id: str) -> AOAttribution | None:
    if ao_id not in collect_attributed_ao_ids(session_dir):
        return None
    history = session_dir / HISTORY_DIRNAME
    if not history.is_dir():
        return None
    try:
        children = list(history.iterdir())
    except OSError:
        return None
    for child in children:
        if not child.is_dir():
            continue
        attr = _read_attributions(child / ATTRIBUTION_FILENAME).get(ao_id)
        if attr is not None:
            return attr
    return None


def _frozen_records(session_dir: Path, session_id: str, line_count: int) -> list[AORecord]:
    ledger = AOLedger(session_dir, session_id)
    return ledger.records()[: max(0, line_count)]


def _item_dict(record: AORecord, attr: AOAttribution | None) -> dict[str, Any]:
    item = record.to_dict()
    item["archive_id"] = attr.archive_id if attr is not None else None
    item["committed"] = attr is not None
    return item


@dataclass(frozen=True)
class LedgerSnapshot:
    """Opaque read-view freeze: archive set + live ledger upper bound."""

    watermark: str
    ledger_line_count: int
    archive_ids: tuple[str, ...]
    issued_at: str
    session_id: str

    def encode(self) -> str:
        """Serialize the frozen state into an opaque watermark token."""
        return _encode_payload(
            {
                "v": _SNAPSHOT_VERSION,
                "session_id": self.session_id,
                "ledger_line_count": self.ledger_line_count,
                "archive_ids": list(self.archive_ids),
                "issued_at": self.issued_at,
            }
        )

    @classmethod
    def decode(cls, watermark: str) -> LedgerSnapshot:
        """Restore a snapshot from a watermark issued by ``encode`` / ``issue_snapshot``."""
        payload = _decode_payload(watermark)
        try:
            archive_ids = tuple(str(item) for item in payload["archive_ids"])
            snapshot = cls(
                watermark="",
                ledger_line_count=int(payload["ledger_line_count"]),
                archive_ids=archive_ids,
                issued_at=str(payload["issued_at"]),
                session_id=str(payload["session_id"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid snapshot watermark") from exc
        token = snapshot.encode()
        return cls(
            watermark=token,
            ledger_line_count=snapshot.ledger_line_count,
            archive_ids=snapshot.archive_ids,
            issued_at=snapshot.issued_at,
            session_id=snapshot.session_id,
        )


def issue_snapshot(session_dir: Path | str, session_id: str) -> LedgerSnapshot:
    """Sign a read view of the current ledger line count and archive set."""
    session_path = Path(session_dir)
    ledger = AOLedger(session_path, session_id)
    issued_at = _utc_now_iso()
    archive_ids = _existing_archive_ids(session_path)
    draft = LedgerSnapshot(
        watermark="",
        ledger_line_count=ledger.count(),
        archive_ids=archive_ids,
        issued_at=issued_at,
        session_id=session_id,
    )
    token = draft.encode()
    return LedgerSnapshot(
        watermark=token,
        ledger_line_count=draft.ledger_line_count,
        archive_ids=draft.archive_ids,
        issued_at=draft.issued_at,
        session_id=draft.session_id,
    )


@dataclass
class LedgerPage:
    """One page of ``list_ao_ledger`` under a frozen snapshot."""

    items: list[dict[str, Any]]
    next_cursor: str | None
    snapshot_watermark: str
    total_in_snapshot: int
    live_included: bool


def list_ao_ledger(
    session_dir: Path | str,
    session_id: str,
    *,
    cursor: str | None = None,
    limit: int = 50,
    snapshot_watermark: str | None = None,
    include_live: bool = False,
) -> LedgerPage:
    """Page AO records under a frozen snapshot watermark.

    First call (no watermark) issues a snapshot. A cursor must be paired
    with the same watermark that produced it; a mismatch raises
    ``ValueError``. ``include_live=False`` returns only attributed
    (committed) AOs. ``include_live=True`` also returns live rows still
    inside the frozen snapshot. This list does not apply source-session
    ACL: ``include_live`` is not an evidence bypass. Drill-down goes
    through ``get_ao_evidence``, which is default-deny unless
    ``register_session_acl`` installed a policy or the caller passes
    ``acl_check``. ``limit`` above ``MAX_PAGE_LIMIT`` is truncated.
    """
    if cursor and not snapshot_watermark:
        raise ValueError("cursor requires snapshot_watermark")

    if snapshot_watermark is None:
        snapshot = issue_snapshot(session_dir, session_id)
    else:
        snapshot = LedgerSnapshot.decode(snapshot_watermark)
        if snapshot.session_id != session_id:
            raise ValueError(
                f"snapshot watermark session_id {snapshot.session_id!r} "
                f"does not match {session_id!r}"
            )

    after_sequence = 0
    if cursor:
        after_sequence, fingerprint = _parse_cursor(cursor)
        expected = _watermark_fingerprint(snapshot.watermark)
        if fingerprint is not None and fingerprint != expected:
            raise ValueError("cursor does not match snapshot_watermark")

    effective_limit = limit
    if effective_limit > MAX_PAGE_LIMIT:
        logger.warning(
            "list_ao_ledger limit %s exceeds server max %s; truncating",
            effective_limit,
            MAX_PAGE_LIMIT,
        )
        effective_limit = MAX_PAGE_LIMIT
    if effective_limit < 1:
        effective_limit = 0

    session_path = Path(session_dir)
    records = _frozen_records(session_path, session_id, snapshot.ledger_line_count)
    attributions = _attributions_for_archives(session_path, snapshot.archive_ids)

    visible: list[tuple[AORecord, AOAttribution | None]] = []
    for record in records:
        attr = attributions.get(record.ao_id)
        # include_live adds unattributed rows from the frozen snapshot only.
        # Source-session ACL is enforced on get_ao_evidence (default-deny
        # unless register_session_acl installed a policy). Listing live rows
        # does not grant an evidence read.
        if attr is None and not include_live:
            continue
        visible.append((record, attr))

    remaining = [pair for pair in visible if pair[0].sequence > after_sequence]
    page_pairs = remaining[:effective_limit]
    items = [_item_dict(record, attr) for record, attr in page_pairs]
    next_cursor = None
    if len(remaining) > len(page_pairs) and page_pairs:
        next_cursor = _encode_cursor(page_pairs[-1][0].sequence, snapshot.watermark)

    return LedgerPage(
        items=items,
        next_cursor=next_cursor,
        snapshot_watermark=snapshot.watermark,
        total_in_snapshot=len(visible),
        live_included=include_live,
    )


def get_ao_evidence(
    session_dir: Path | str,
    ao_id: str,
    *,
    acl_check: Callable[[str, dict], bool] | None = None,
) -> dict[str, Any] | None:
    """Return one AO record and its archive attribution, if any.

    ``acl_check(source_session_id, record_dict)`` overrides the registered
    source-session policy (Q28). When it is omitted, ``register_session_acl``
    supplies the policy. No registration means ``DefaultDenyAcl`` — the read
    is refused. A check that returns ``False`` raises ``PermissionError``.
    Missing ``ao_id`` returns ``None`` before the ACL check.
    """
    session_path = Path(session_dir)
    ledger = AOLedger(session_path, session_path.name)
    record = ledger.get(ao_id)
    if record is None:
        return None

    record_dict = record.to_dict()
    checker = acl_check if acl_check is not None else _registered_acl_check
    if not checker(record.session_id, record_dict):
        raise PermissionError(f"ACL denied for ao_id {ao_id}")

    attr = _find_attribution(session_path, ao_id)
    return {
        "record": record_dict,
        "attribution": attr.to_dict() if attr is not None else None,
    }

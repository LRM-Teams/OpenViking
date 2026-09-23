# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Unit tests for memory.skill_trajectory_mode (slice S3/6)."""

from __future__ import annotations

import pytest

from openviking_cli.utils.config import OPENVIKING_CONFIG_ENV
from openviking_cli.utils.config.memory_config import (
    SKILL_TRAJECTORY_MODE_CAUSAL,
    SKILL_TRAJECTORY_MODE_LEGACY,
    MemoryConfig,
    is_causal_mode_enabled,
    resolve_skill_trajectory_mode,
)
from openviking_cli.utils.config.open_viking_config import (
    OpenVikingConfig,
    OpenVikingConfigSingleton,
    set_openviking_config,
)


def test_skill_trajectory_mode_defaults_to_legacy() -> None:
    memory = MemoryConfig()
    assert memory.skill_trajectory_mode == SKILL_TRAJECTORY_MODE_LEGACY
    assert resolve_skill_trajectory_mode(memory) == SKILL_TRAJECTORY_MODE_LEGACY
    assert is_causal_mode_enabled(memory) is False

    from_empty = MemoryConfig.from_dict({})
    assert from_empty.skill_trajectory_mode == SKILL_TRAJECTORY_MODE_LEGACY
    assert resolve_skill_trajectory_mode(from_empty) == SKILL_TRAJECTORY_MODE_LEGACY


def test_explicit_causal_mode_is_resolved(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(OPENVIKING_CONFIG_ENV, "/tmp/codex-no-config.json")
    memory = MemoryConfig.from_dict({"skill_trajectory_mode": SKILL_TRAJECTORY_MODE_CAUSAL})
    assert resolve_skill_trajectory_mode(memory) == SKILL_TRAJECTORY_MODE_CAUSAL
    assert is_causal_mode_enabled(memory) is True

    full = OpenVikingConfig.from_dict({"memory": {"skill_trajectory_mode": "causal"}})
    assert resolve_skill_trajectory_mode(full) == SKILL_TRAJECTORY_MODE_CAUSAL
    assert is_causal_mode_enabled(full) is True
    assert resolve_skill_trajectory_mode({"skill_trajectory_mode": "causal"}) == (
        SKILL_TRAJECTORY_MODE_CAUSAL
    )
    assert is_causal_mode_enabled({"memory": {"skill_trajectory_mode": "causal"}}) is True


def test_invalid_skill_trajectory_mode_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(OPENVIKING_CONFIG_ENV, "/tmp/codex-no-config.json")
    with pytest.raises(ValueError, match="skill_trajectory_mode"):
        MemoryConfig.from_dict({"skill_trajectory_mode": "fast"})
    with pytest.raises(ValueError, match="skill_trajectory_mode"):
        OpenVikingConfig.from_dict({"memory": {"skill_trajectory_mode": "fast"}})
    with pytest.raises(ValueError, match="skill_trajectory_mode"):
        resolve_skill_trajectory_mode({"skill_trajectory_mode": "fast"})


def test_omitted_skill_trajectory_mode_parses_as_legacy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(OPENVIKING_CONFIG_ENV, "/tmp/codex-no-config.json")
    legacy_payload = {"extraction_enabled": True}
    assert "skill_trajectory_mode" not in legacy_payload
    memory = MemoryConfig.from_dict(legacy_payload)
    assert memory.skill_trajectory_mode == SKILL_TRAJECTORY_MODE_LEGACY
    assert resolve_skill_trajectory_mode(memory) == SKILL_TRAJECTORY_MODE_LEGACY
    assert is_causal_mode_enabled(memory) is False

    config = OpenVikingConfig.from_dict({"memory": {"session_skill_extraction_enabled": True}})
    assert config.memory.skill_trajectory_mode == SKILL_TRAJECTORY_MODE_LEGACY
    dumped = config.memory.to_dict()
    assert dumped["skill_trajectory_mode"] == SKILL_TRAJECTORY_MODE_LEGACY
    assert dumped["session_skill_extraction_enabled"] is True


def test_ao_ledger_imports_under_legacy_without_circular_dependency() -> None:
    import openviking.session.ao_ledger as ao_ledger

    assert resolve_skill_trajectory_mode(MemoryConfig()) == SKILL_TRAJECTORY_MODE_LEGACY
    assert is_causal_mode_enabled(MemoryConfig()) is False
    assert ao_ledger.LEDGER_FILENAME == "ao-ledger.jsonl"
    assert ao_ledger.AOLedger is not None
    assert ao_ledger.extract_skill_invocations({"tool": "read_file"}) == []


def test_resolve_reads_process_singleton_when_config_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(OPENVIKING_CONFIG_ENV, "/tmp/codex-no-config.json")
    OpenVikingConfigSingleton.reset_instance()
    set_openviking_config(
        OpenVikingConfig.from_dict({"memory": {"skill_trajectory_mode": "causal"}})
    )
    try:
        assert resolve_skill_trajectory_mode() == SKILL_TRAJECTORY_MODE_CAUSAL
        assert is_causal_mode_enabled() is True
    finally:
        OpenVikingConfigSingleton.reset_instance()

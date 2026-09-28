# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from openviking_cli.utils.logger import get_logger

logger = get_logger(__name__)

SKILL_TRAJECTORY_MODE_LEGACY = "legacy"
SKILL_TRAJECTORY_MODE_CAUSAL = "causal"
SKILL_TRAJECTORY_MODES = frozenset(
    (SKILL_TRAJECTORY_MODE_LEGACY, SKILL_TRAJECTORY_MODE_CAUSAL)
)


class SessionAutoCommitConfig(BaseModel):
    """Server-wide controls for automatic session commits."""

    enabled: bool = Field(
        default=False,
        description=(
            "Master switch for automatic session commits. When enabled, newly "
            "created sessions without an explicit auto_commit_policy get a "
            "default policy, and the idle-timeout background scheduler is "
            "started. When disabled, neither happens."
        ),
    )
    check_interval_seconds: float = Field(default=600.0, gt=0)
    scan_rate_limit_files_per_second: float = Field(
        default=2.0,
        gt=0,
        description=(
            "Maximum number of session .meta.json files read per second during the "
            "idle auto-commit scan. Used to bound background IO pressure when the "
            "sessions directory is very large."
        ),
    )


class MemoryConfig(BaseModel):
    """Memory configuration for OpenViking."""

    version: str = Field(
        default="v3",
        description="Deprecated and ignored. Memory extraction always uses v3.",
    )
    custom_templates_dir: str = Field(
        default="",
        description="Custom memory templates directory. If set, templates from this directory will be loaded in addition to built-in templates",
    )
    experimental_memory_switch: bool = Field(
        default=False,
        description=(
            "Experimental memory switch for experimental testing. When enabled, "
            "experimental memory templates are loaded."
        ),
    )
    eager_prefetch: bool = Field(
        default=True,
        description=(
            "When enabled, prefetch will execute search + read to preload all memory file contents "
            "into the context, and no read/search tools will be provided to the LLM. "
            "When disabled (default), LLM has read tool and reads files on-demand."
        ),
    )
    prefetch_search_topn: int = Field(
        default=5,
        ge=1,
        description=(
            "Number of top search results to read during prefetch. "
            "Only applies when eager_prefetch is enabled. "
            "When multiple directories are searched, results are merged and top-N are read."
        ),
    )
    maintenance_review_tokens: int = Field(
        default=1000,
        gt=0,
        description=(
            "Estimated token count above which a full memory read includes an LLM maintenance "
            "notice asking it to choose between coherent splitting and single-file compaction."
        ),
    )
    extraction_enabled: bool = Field(
        default=True,
        description=(
            "When enabled (default), memory extraction runs on session commit "
            "to produce long-term memories. When disabled, sessions are archived "
            "but no memory extraction is performed. Useful for read-only or "
            "stateless deployments."
        ),
    )
    extraction_output_format: Literal["json", "python"] = Field(
        default="python",
        description=(
            "Final model-output protocol used by every memory extraction loop. "
            "'python' uses the restricted internal memory SDK DSL (default); "
            "'json' preserves the legacy structured JSON protocol."
        ),
    )
    session_skill_extraction_enabled: bool = Field(
        default=False,
        description=(
            "When enabled, session commit also extracts reusable skills from the archived "
            "conversation and writes them into the current user's skill directory. Disabled by "
            "default."
        ),
    )
    skill_trajectory_mode: Literal["legacy", "causal"] = Field(
        default=SKILL_TRAJECTORY_MODE_LEGACY,
        description=(
            "Versioned switch for skill-trajectory behavior (ADR-0004). "
            "'legacy' keeps original OpenViking behavior. "
            "'causal' gates the AO ledger, fork construction, proposal, replay, "
            "guidance, and aggregator pipeline. This field is the global default; "
            "workspace override and session-creation freeze are later slices."
        ),
    )
    link_enabled: bool = Field(
        default=False,
        description=(
            "When enabled, memory extraction supports link extraction between "
            "memory items (page_id, links field, and link resolution). When disabled (default), "
            "no page_id or link fields are generated, and link resolution is skipped."
        ),
    )
    session_auto_commit: SessionAutoCommitConfig = Field(
        default_factory=SessionAutoCommitConfig,
        description="Server-wide controls for automatic session commits.",
    )

    @model_validator(mode="before")
    @classmethod
    def drop_deprecated_memory_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
            if "agent_memory_enabled" in data:
                data.pop("agent_memory_enabled", None)
                logger.debug(
                    "memory.agent_memory_enabled is deprecated and ignored; "
                    "use session memory_policy.memory_types to control trajectory/experience extraction"
                )
            if "working_memory_enabled" in data:
                data.pop("working_memory_enabled", None)
                logger.debug(
                    "memory.working_memory_enabled is deprecated and ignored; "
                    "use session memory_policy.working_memory.enabled to control archive summaries"
                )
        return data

    @field_validator("version", mode="before")
    @classmethod
    def accept_deprecated_version(cls, value: Any) -> str:
        if value not in (None, ""):
            logger.debug(
                "memory.version is deprecated and ignored; memory extraction always uses v3"
            )
        return "v3"

    @field_validator("skill_trajectory_mode", mode="before")
    @classmethod
    def validate_skill_trajectory_mode(cls, value: Any) -> str:
        if value is None:
            return SKILL_TRAJECTORY_MODE_LEGACY
        if not isinstance(value, str):
            raise ValueError(
                "memory.skill_trajectory_mode must be "
                f"{SKILL_TRAJECTORY_MODE_LEGACY!r} or {SKILL_TRAJECTORY_MODE_CAUSAL!r}"
            )
        normalized = value.strip()
        if normalized not in SKILL_TRAJECTORY_MODES:
            raise ValueError(
                "memory.skill_trajectory_mode must be "
                f"{SKILL_TRAJECTORY_MODE_LEGACY!r} or {SKILL_TRAJECTORY_MODE_CAUSAL!r}"
            )
        return normalized

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "MemoryConfig":
        """Create configuration from dictionary."""
        return cls(**config)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.model_dump()


def _memory_config_from_source(config: Any) -> MemoryConfig:
    """Normalize a config source to ``MemoryConfig``.

    Accepts the shapes callers already pass around: ``None`` (process singleton),
    ``MemoryConfig``, an object with a ``memory`` section (``OpenVikingConfig``),
    or a mapping (memory section or full config dict).
    """
    if config is None:
        from openviking_cli.utils.config.open_viking_config import get_openviking_config

        return get_openviking_config().memory
    if isinstance(config, MemoryConfig):
        return config
    memory = getattr(config, "memory", None)
    if isinstance(memory, MemoryConfig):
        return memory
    if isinstance(config, dict):
        payload = config
        nested = config.get("memory")
        if isinstance(nested, dict):
            payload = nested
        return MemoryConfig.from_dict(payload)
    mode = getattr(config, "skill_trajectory_mode", None)
    if mode is not None:
        return MemoryConfig(skill_trajectory_mode=mode)
    raise TypeError(
        "config must be MemoryConfig, OpenVikingConfig, a mapping, or None"
    )


def resolve_skill_trajectory_mode(
    config: Any = None,
    *,
    workspace_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> str:
    """Return the effective skill trajectory mode (``legacy`` or ``causal``).

    This slice resolves only the **global default** from
    ``memory.skill_trajectory_mode``. ``workspace_id`` and ``session_id`` are
    reserved for later slices (workspace override, then session-creation freeze)
    and are ignored here.

    ``config`` follows existing config-read habits: omit it to use
    ``get_openviking_config()``, or pass ``MemoryConfig``, ``OpenVikingConfig``,
    or a mapping (memory section or full config dict).
    """
    del workspace_id, session_id
    return _memory_config_from_source(config).skill_trajectory_mode


def is_causal_mode_enabled(
    config: Any = None,
    *,
    workspace_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> bool:
    """Return whether the resolved skill trajectory mode is ``causal``.

    Scope and ``config`` sources match :func:`resolve_skill_trajectory_mode`.
    Workspace override and session freeze are later slices; ``workspace_id``
    and ``session_id`` are reserved and ignored here.
    """
    return (
        resolve_skill_trajectory_mode(
            config,
            workspace_id=workspace_id,
            session_id=session_id,
        )
        == SKILL_TRAJECTORY_MODE_CAUSAL
    )

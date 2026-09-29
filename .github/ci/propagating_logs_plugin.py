"""Make OpenViking log records visible to pytest's ``caplog``.

Why this exists
---------------
``openviking_cli/utils/logger.py::reconfigure_logging`` configures the
``openviking`` / ``openviking_cli`` loggers with the shared handler attached to
the *managed root* logger and ``propagate = root_name != logger_name`` — so the
``openviking`` logger keeps the handler and stops propagating, and records never
reach the stdlib root logger. pytest's ``caplog`` handler lives on the stdlib
root logger, so from the moment configuration is loaded in a process,
``caplog.records`` stays empty for every ``openviking.*`` record.

Five fork tests assert on ``caplog`` records
(``test_causal_bridges.py::test_corrupt_jsonl_line_is_skipped_and_store_stays_usable``,
``test_citation_ledger.py::test_offline_counts_warn_once``,
``test_influence_projection.py::test_acl_change_without_revalidation_callback_warns_once``,
``test_trajectory_index.py::test_missing_projection_registry_warns_once_and_still_indexes``,
``test_trajectory_index.py::test_supersede_without_revalidation_callback_warns_once``).
They fail in a clean checkout for that reason alone: their state assertions pass
and the expected messages are emitted, but the records are not observable.

This plugin forces ``propagate=True`` whenever the fork configures a logger, so
the records reach both the shared handler and ``caplog``. It is CI-only glue and
changes no repository code: remove it once the fork's logging configuration and
its ``caplog``-based tests agree (fix the tests to attach their own handler, or
keep propagation on in ``reconfigure_logging``).
"""

from __future__ import annotations

import openviking_cli.utils.logger as ov_logger

_original_configure_logger_instance = ov_logger._configure_logger_instance


def _configure_logger_instance(logger, level, handler, propagate):  # noqa: ANN001
    return _original_configure_logger_instance(
        logger, level=level, handler=handler, propagate=True
    )


# reconfigure_logging() calls the module-global name, so patching the module
# attribute covers both the call already made and every later one.
ov_logger._configure_logger_instance = _configure_logger_instance

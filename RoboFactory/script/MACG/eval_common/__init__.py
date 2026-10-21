"""Shared evaluation helpers for MACG rule-based rollouts."""

from .progress_tracker import (
    ProgressSpecError,
    ProgressTracker,
    SignalSample,
    load_progress_spec,
    validate_progress_spec,
)

__all__ = [
    "ProgressSpecError",
    "ProgressTracker",
    "SignalSample",
    "load_progress_spec",
    "validate_progress_spec",
]


def __getattr__(name: str):
    if name in {
        "build_progress_tracker",
        "emit_metrics_line",
        "extract_progress_signals",
        "finalize_episode_metrics",
        "sample_progress",
    }:
        from . import eval_hooks

        return getattr(eval_hooks, name)
    raise AttributeError(name)

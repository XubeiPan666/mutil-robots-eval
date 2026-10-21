"""Helpers for wiring ProgressTracker into per-task eval rollouts."""

from __future__ import annotations

import json
from typing import Any, Mapping, Optional

from .progress_tracker import ProgressTracker, SignalSample


def _to_bool(value: Any) -> bool:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return False
            return bool(value.reshape(-1)[0].item())
    except Exception:
        pass
    try:
        import numpy as np

        arr = np.asarray(value).reshape(-1)
        if arr.size == 0:
            return False
        return bool(arr[0])
    except Exception:
        if isinstance(value, (list, tuple)):
            return bool(value[0]) if value else False
        return bool(value)


def _info_bool(info: Mapping[str, Any], *keys: str, default: bool = False) -> bool:
    for key in keys:
        if key in info:
            return _to_bool(info[key])
    return default


def extract_progress_signals(
    task: str,
    info: Mapping[str, Any],
    observation: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Map task-specific env info into ProgressTracker signal names."""

    del observation  # reserved for future TCP-based attribution
    task = str(task or "")

    if task == "stack_cube":
        return {
            "agent0_holds_cubeA": _info_bool(info, "is_cubeA_grasped_left", "is_cubeA_grasped"),
            "agent1_holds_cubeB": _info_bool(info, "is_cubeB_grasped_right", "is_cubeB_grasped"),
            "cubeA_placed_on_target": (
                _info_bool(info, "cubeA_placed")
                and not _info_bool(info, "is_cubeA_grasped")
            ),
            "cubeB_placed_on_target": (
                _info_bool(info, "cubeB_placed")
                and not _info_bool(info, "is_cubeB_grasped")
            ),
            "cubeA_on_cubeB": (
                _info_bool(info, "is_cubeA_on_cubeB")
                and not _info_bool(info, "is_cubeA_grasped")
                and not _info_bool(info, "is_cubeB_grasped")
            ),
            "cubeB_on_cubeA": (
                _info_bool(info, "is_cubeB_on_cubeA")
                and not _info_bool(info, "is_cubeA_grasped")
                and not _info_bool(info, "is_cubeB_grasped")
            ),
        }
    if task == "stack_bowl":
        return {
            "agent0_holds_bowlA": _info_bool(info, "is_bowlA_grasped_left", "is_bowlA_grasped"),
            "agent1_holds_bowlB": _info_bool(info, "is_bowlB_grasped_right", "is_bowlB_grasped"),
            "bowlA_placed_on_target": (
                _info_bool(info, "bowlA_placed")
                and not _info_bool(info, "is_bowlA_grasped")
            ),
            "bowlB_placed_on_target": (
                _info_bool(info, "bowlB_placed")
                and not _info_bool(info, "is_bowlB_grasped")
            ),
            "bowlA_on_bowlB": (
                _info_bool(info, "is_bowlA_on_bowlB")
                and not _info_bool(info, "is_bowlA_grasped")
                and not _info_bool(info, "is_bowlB_grasped")
            ),
            "bowlB_on_bowlA": (
                _info_bool(info, "is_bowlB_on_bowlA")
                and not _info_bool(info, "is_bowlA_grasped")
                and not _info_bool(info, "is_bowlB_grasped")
            ),
        }
    if task == "push_cube":
        cube_a = _info_bool(info, "cubeA_pushed_by_agent0", "is_cubeA_placed", "cubeA_placed")
        cube_b = _info_bool(info, "cubeB_pushed_by_agent1", "is_cubeB_placed", "cubeB_placed")
        return {
            "cubeA_pushed_to_goal_by_agent0": SignalSample(cube_a, "agent0"),
            "cubeB_pushed_to_goal_by_agent1": SignalSample(cube_b, "agent1"),
        }
    if task == "place_can_basket":
        return {
            "agent0_holds_can": _info_bool(info, "is_can_grasped_left", "is_can_grasped"),
            "agent1_holds_basket": _info_bool(
                info, "is_basket_grasped_right", "is_basket_grasped"
            ),
            "can_in_basket": (
                _info_bool(info, "can_in_basket")
                and not _info_bool(info, "is_can_grasped")
            ),
        }
    if task == "place_burger_fries":
        return {
            "agent0_holds_burger": _info_bool(
                info, "is_burger_grasped_left", "is_burger_grasped"
            ),
            "agent1_holds_fries": _info_bool(
                info, "is_fries_grasped_right", "is_fries_grasped"
            ),
            "burger_placed_on_tray": (
                _info_bool(info, "burger_placed")
                and not _info_bool(info, "is_burger_grasped")
            ),
            "fries_placed_on_tray": (
                _info_bool(info, "fries_placed")
                and not _info_bool(info, "is_fries_grasped")
            ),
        }
    if task == "place_object_cabinet":
        return {
            "agent0_holds_object": _info_bool(
                info, "is_object_grasped_left", "is_object_grasped"
            ),
            "agent1_holds_drawer_handle": _info_bool(info, "is_handle_grasped"),
            "drawer_opened_by_agent1": _info_bool(info, "drawer_open"),
            "object_in_drawer": (
                _info_bool(info, "tag") and not _info_bool(info, "is_object_grasped")
            ),
        }
    if task == "place_object_bowl_cabinet":
        return {
            "agent0_holds_bowl": _info_bool(
                info, "is_bowl_grasped_left", "is_bowl_grasped"
            ),
            "agent1_holds_drawer_handle": _info_bool(info, "is_handle_grasped"),
            "drawer_opened_by_agent1": _info_bool(info, "drawer_open"),
            "bowl_in_drawer": (
                _info_bool(info, "bowl_in_cabinet")
                and not _info_bool(info, "is_bowl_grasped")
            ),
        }
    if task == "place_cube_in_bowl":
        return {
            "agent0_holds_cube": _info_bool(
                info, "is_cube_grasped_left", "is_cube_grasped"
            ),
            "agent1_holds_bowl": _info_bool(
                info, "is_bowl_grasped_right", "is_bowl_grasped"
            ),
            "bowl_placed_on_tray": (
                _info_bool(info, "bowl_placed")
                and not _info_bool(info, "is_bowl_grasped")
            ),
            "cube_in_bowl": (
                _info_bool(info, "cube_in_bowl")
                and not _info_bool(info, "is_cube_grasped")
            ),
        }
    if task == "handover_bottles":
        left1 = _info_bool(info, "is_bottle1_grasped_left")
        right1 = _info_bool(info, "is_bottle1_grasped_right")
        left2 = _info_bool(info, "is_bottle2_grasped_left")
        right2 = _info_bool(info, "is_bottle2_grasped_right")
        return {
            "agent0_holds_bottle1": left1,
            "agent1_holds_bottle1": right1,
            "agent0_holds_bottle2": left2,
            "agent1_holds_bottle2": right2,
            "bottle1_handed_to_agent1": right1 and not left1,
            "bottle2_handed_to_agent1": right2 and not left2,
            "bottle1_in_basket": (
                _info_bool(info, "bottle1_in_basket") and not left1 and not right1
            ),
            "bottle2_in_basket": (
                _info_bool(info, "bottle2_in_basket") and not left2 and not right2
            ),
        }
    raise ValueError(f"Unsupported progress task: {task!r}")


def build_progress_tracker(
    *,
    env_id: Any = None,
    status: Any = None,
    phase_config: Any = None,
    control_freq: Any = None,
    control_dt: Optional[float] = None,
) -> Optional[ProgressTracker]:
    """Create a tracker from eval context. Returns None if no matching spec."""

    if control_dt is None and control_freq:
        try:
            control_dt = 1.0 / float(control_freq)
        except Exception:
            control_dt = None
    try:
        return ProgressTracker.from_context(
            env_id=env_id,
            status=status,
            phase_config=phase_config,
            control_dt=control_dt,
        )
    except Exception as exc:
        print(f"[progress] disabled: {exc}")
        return None


def sample_progress(
    tracker: Optional[ProgressTracker],
    info: Mapping[str, Any],
    observation: Optional[Mapping[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    if tracker is None:
        return None
    signals = extract_progress_signals(tracker.spec.get("task"), info, observation)
    timestamp = None
    if "elapsed_steps" in info and tracker.control_dt is not None:
        try:
            raw = info["elapsed_steps"]
            if hasattr(raw, "reshape"):
                steps = float(raw.reshape(-1)[0])
            elif isinstance(raw, (list, tuple)):
                steps = float(raw[0])
            else:
                steps = float(raw)
            timestamp = steps * float(tracker.control_dt)
        except Exception:
            timestamp = None
    return tracker.sample(signals, timestamp=timestamp)


def finalize_episode_metrics(
    tracker: Optional[ProgressTracker],
    *,
    task_success: bool,
    seed: Any = None,
    env_id: Any = None,
    status: Any = None,
    phase_config: Any = None,
    policy_steps: Any = None,
) -> dict[str, Any]:
    """Convert tracker snapshot into aggregate_progress-compatible episode JSON."""

    if tracker is None:
        return {
            "task_id": str(env_id or "unknown"),
            "mode": "unknown",
            "task_success": bool(task_success),
            "normalized_progress": 1.0 if task_success else 0.0,
            "conditional_completion": {},
            "unconditional_completion": {},
            "seed": seed,
            "status": status,
            "phase_config": phase_config,
            "policy_steps": policy_steps,
        }

    snap = tracker.snapshot()
    order = [item["id"] for item in tracker.spec["milestones"]]
    compliant = snap.get("compliant", {})
    unconditional = snap.get("diagnostics", {}).get("unconditional", {})

    predecessors = {
        item["id"]: set(item.get("depends_on", []))
        for item in tracker.spec["milestones"]
    }
    for constraint in tracker.spec.get("order_constraints", []):
        predecessors[constraint["after"]].add(constraint["before"])
    eligible = {
        name: all(compliant.get(dep, False) for dep in predecessors[name])
        for name in order
    }

    return {
        "task_id": snap.get("task") or tracker.spec.get("task"),
        "env_id": env_id,
        "mode": snap.get("mode") or tracker.spec.get("mode"),
        "status": status,
        "phase_config": phase_config,
        "seed": seed,
        "policy_steps": policy_steps,
        "task_success": bool(task_success),
        "normalized_progress": float(snap.get("normalized_progress", 0.0)),
        "completed_count": int(snap.get("completed_count", 0)),
        "milestone_count": int(snap.get("milestone_count", 0)),
        "conditional_completion": {
            name: bool(compliant.get(name, False)) for name in order
        },
        "unconditional_completion": {
            name: bool(unconditional.get(name, False)) for name in order
        },
        "eligible": eligible,
        "continuous_prefix": float(
            snap.get("diagnostics", {}).get("continuous_prefix_progress", 0.0)
        ),
        "violations": snap.get("diagnostics", {}).get("violations", {}),
        "timeline": snap.get("timeline", []),
    }


def emit_metrics_line(metrics: Mapping[str, Any]) -> None:
    """Print a shell-parseable METRICS_JSON line (before success/failed)."""

    print(
        "METRICS_JSON="
        + json.dumps(dict(metrics), ensure_ascii=False, separators=(",", ":"))
    )


__all__ = [
    "build_progress_tracker",
    "emit_metrics_line",
    "extract_progress_signals",
    "finalize_episode_metrics",
    "sample_progress",
]

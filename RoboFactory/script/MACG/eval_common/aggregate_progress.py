"""Aggregate per-episode progress JSONL files.

The evaluator writes one self-contained JSON object per rollout.  This module
keeps aggregation independent from the rollout implementation so old binary
success logs remain unchanged.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def _first(mapping: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default


def _completion_map(episode: dict[str, Any], key: str) -> dict[str, bool]:
    value = episode.get(key, {})
    if isinstance(value, dict):
        return {str(name): bool(done) for name, done in value.items()}
    return {}


def _progress(episode: dict[str, Any]) -> float:
    value = _first(
        episode,
        "normalized_progress",
        "conditional_progress",
        "continuous_prefix",
        default=0.0,
    )
    return float(value)


def _task_key(episode: dict[str, Any]) -> str:
    task = str(_first(episode, "task_id", "env_id", "task", default="unknown"))
    mode = str(_first(episode, "mode", "instruction_mode", default="unknown"))
    return f"{task}:{mode}"


def load_episodes(paths: Iterable[Path]) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, raw_line in enumerate(stream, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
                if not isinstance(item, dict):
                    raise ValueError(f"{path}:{line_number}: episode must be a JSON object")
                episodes.append(item)
    return episodes


def _rate_map(
    episodes: list[dict[str, Any]], key: str, denominator: str = "all"
) -> dict[str, dict[str, float | int]]:
    names = sorted(
        {
            name
            for episode in episodes
            for name in _completion_map(episode, key)
        }
    )
    result: dict[str, dict[str, float | int]] = {}
    for name in names:
        if denominator == "eligible":
            selected = [
                episode
                for episode in episodes
                if bool(episode.get("eligible", {}).get(name, False))
            ]
        else:
            selected = episodes
        completed = sum(_completion_map(item, key).get(name, False) for item in selected)
        total = len(selected)
        result[name] = {
            "completed": completed,
            "total": total,
            "rate": completed / total if total else 0.0,
        }
    return result


def aggregate(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        grouped[_task_key(episode)].append(episode)

    tasks: dict[str, Any] = {}
    for key, items in sorted(grouped.items()):
        count = len(items)
        successes = sum(
            bool(_first(item, "task_success", "final_success", "success", default=False))
            for item in items
        )
        tasks[key] = {
            "rollouts": count,
            "task_successes": successes,
            "task_success_rate": successes / count if count else 0.0,
            "normalized_progress": (
                sum(_progress(item) for item in items) / count if count else 0.0
            ),
            # Paper-facing per-step rates always use all rollouts as denominator.
            "subtask_completion": _rate_map(items, "conditional_completion"),
            "unconditional_completion": _rate_map(items, "unconditional_completion"),
            # Optional diagnostic; emitted only when the tracker provides eligibility.
            "conditional_on_reached": _rate_map(
                items, "conditional_completion", denominator="eligible"
            ),
        }

    task_values = list(tasks.values())
    return {
        "rollouts": len(episodes),
        "tasks": tasks,
        "macro_task_success": (
            sum(item["task_success_rate"] for item in task_values) / len(task_values)
            if task_values
            else 0.0
        ),
        "macro_normalized_progress": (
            sum(item["normalized_progress"] for item in task_values) / len(task_values)
            if task_values
            else 0.0
        ),
    }


def _resolve_inputs(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            paths.extend(sorted(path.rglob("*_episodes.jsonl")))
        else:
            paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate strict subtask completion metrics."
    )
    parser.add_argument("inputs", nargs="+", help="Episode JSONL files or directories")
    parser.add_argument("-o", "--output", help="Summary JSON path; defaults to stdout")
    args = parser.parse_args()

    episodes = load_episodes(_resolve_inputs(args.inputs))
    summary = aggregate(episodes)
    rendered = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()

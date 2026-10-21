"""Strict low-level subtask progress tracking for MACG evaluations.

The tracker is deliberately independent from prompt phases.  Call ``sample``
after every low-level ``env.step`` with physical boolean signals and simulation
time.  Only milestones satisfying agent, dependency, ordering, debounce and
synchronization constraints contribute to ``normalized_progress``.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional


SPEC_DIR = Path(__file__).with_name("progress_specs")


class ProgressSpecError(ValueError):
    """Raised when a progress specification is invalid or ambiguous."""


@dataclass(frozen=True)
class SignalSample:
    """One physical signal value, optionally attributed to an agent."""

    value: bool
    agent: Optional[str] = None


@dataclass
class _DebouncedSignal:
    stable: bool = False
    stable_agent: Optional[str] = None
    candidate: bool = False
    candidate_agent: Optional[str] = None
    count: int = 0


def _key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _phase_name(path: Any) -> str:
    return Path(str(path or "")).stem.lower()


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _matches(value: Any, aliases: Iterable[str]) -> bool:
    candidate = _key(value)
    if not candidate:
        return False
    return any(_key(alias) == candidate or _key(alias) in candidate for alias in aliases)


def load_progress_spec(
    env_id: Any = None,
    status: Any = None,
    phase_config: Any = None,
    *,
    spec_dir: Path | str = SPEC_DIR,
) -> Dict[str, Any]:
    """Select and validate a task/mode spec from runtime context.

    Task selection uses ``env_id`` and ``status`` aliases.  Mode selection uses
    the phase-config basename (for example ``stack_cubes_sync.yaml``).
    """

    phase = _phase_name(phase_config)
    candidates = []
    for path in sorted(Path(spec_dir).glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        selectors = raw.get("selectors", {})
        env_match = _matches(env_id, selectors.get("env_id", []))
        status_match = _matches(status, selectors.get("status", []))
        if env_match or status_match:
            candidates.append((int(env_match) + int(status_match), path, raw))

    if not candidates:
        raise ProgressSpecError(
            f"No progress spec matches env_id={env_id!r}, status={status!r}"
        )
    best_score = max(item[0] for item in candidates)
    best = [item for item in candidates if item[0] == best_score]
    if len(best) != 1:
        names = ", ".join(item[1].name for item in best)
        raise ProgressSpecError(f"Ambiguous progress specs: {names}")

    _, path, raw = best[0]
    mode_name = None
    for name, mode in raw.get("modes", {}).items():
        basenames = mode.get("phase_basenames", [name])
        if phase and phase in {_phase_name(item) for item in basenames}:
            mode_name = name
            break
    if mode_name is None:
        available = sorted(raw.get("modes", {}))
        raise ProgressSpecError(
            f"{path.name} has no mode for phase_config={phase_config!r}; "
            f"available modes: {available}"
        )

    common = {key: value for key, value in raw.items() if key != "modes"}
    spec = _deep_merge(common, raw["modes"][mode_name])
    spec["mode"] = mode_name
    spec["_source"] = str(path)
    validate_progress_spec(spec)
    return spec


def _predicate_signals(predicate: Any) -> set[str]:
    if isinstance(predicate, str):
        return {predicate}
    if not isinstance(predicate, Mapping):
        raise ProgressSpecError(f"Invalid predicate: {predicate!r}")
    if "signal" in predicate:
        return {str(predicate["signal"])}
    operator = "all" if "all" in predicate else "any" if "any" in predicate else None
    if operator:
        result: set[str] = set()
        for child in predicate[operator]:
            result.update(_predicate_signals(child))
        return result
    if "not" in predicate:
        return _predicate_signals(predicate["not"])
    raise ProgressSpecError(f"Predicate has no supported operator: {predicate!r}")


def validate_progress_spec(spec: Mapping[str, Any]) -> None:
    """Validate identifiers, physical signals, DAG and sync groups."""

    milestones = spec.get("milestones")
    if not isinstance(milestones, list) or not milestones:
        raise ProgressSpecError("Spec must contain a non-empty milestones list")
    by_id = {item.get("id"): item for item in milestones}
    if None in by_id or len(by_id) != len(milestones):
        raise ProgressSpecError("Milestone ids must be present and unique")

    declared_signals = set(spec.get("signals", {}))
    predecessors: Dict[str, set[str]] = {
        item_id: set(item.get("depends_on", [])) for item_id, item in by_id.items()
    }
    for constraint in spec.get("order_constraints", []):
        before, after = constraint.get("before"), constraint.get("after")
        if before not in by_id or after not in by_id:
            raise ProgressSpecError(f"Unknown order constraint endpoint: {constraint}")
        predecessors[after].add(before)

    for item_id, item in by_id.items():
        unknown = predecessors[item_id] - set(by_id)
        if unknown:
            raise ProgressSpecError(f"{item_id} has unknown dependencies: {sorted(unknown)}")
        used = _predicate_signals(item.get("predicate"))
        if item.get("hold_signal"):
            used.add(item["hold_signal"])
        unknown_signals = used - declared_signals
        if unknown_signals:
            raise ProgressSpecError(
                f"{item_id} uses undeclared signals: {sorted(unknown_signals)}"
            )
        agent = item.get("agent")
        if not agent:
            raise ProgressSpecError(f"{item_id} must specify an agent")
        for signal in _predicate_signals(item["predicate"]):
            signal_agent = spec["signals"].get(signal, {}).get("agent")
            if signal_agent and signal_agent != agent:
                raise ProgressSpecError(
                    f"{item_id} agent {agent} conflicts with {signal} agent {signal_agent}"
                )

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            raise ProgressSpecError(f"Dependency/order graph contains a cycle at {node}")
        if node in visited:
            return
        visiting.add(node)
        for dependency in predecessors[node]:
            visit(dependency)
        visiting.remove(node)
        visited.add(node)

    for node in by_id:
        visit(node)

    grouped: set[str] = set()
    for group in spec.get("sync_groups", []):
        members = group.get("members", [])
        if len(members) < 2 or any(member not in by_id for member in members):
            raise ProgressSpecError(f"Invalid sync group: {group}")
        overlap = grouped.intersection(members)
        if overlap:
            raise ProgressSpecError(f"Milestones occur in multiple sync groups: {overlap}")
        grouped.update(members)
        for member in members:
            if by_id[member].get("sync_group") != group.get("id"):
                raise ProgressSpecError(f"{member} does not reference sync group {group.get('id')}")


class ProgressTracker:
    """Stateful strict milestone tracker. One instance represents one rollout."""

    def __init__(self, spec: Mapping[str, Any], *, control_dt: Optional[float] = None):
        validate_progress_spec(spec)
        self.spec = copy.deepcopy(dict(spec))
        self.control_dt = control_dt
        self._milestones = {item["id"]: item for item in self.spec["milestones"]}
        self._order = [item["id"] for item in self.spec["milestones"]]
        self._predecessors = {
            item_id: set(item.get("depends_on", []))
            for item_id, item in self._milestones.items()
        }
        for constraint in self.spec.get("order_constraints", []):
            self._predecessors[constraint["after"]].add(constraint["before"])
        self._groups = {
            group["id"]: group for group in self.spec.get("sync_groups", [])
        }
        self.reset()

    @classmethod
    def from_context(
        cls,
        env_id: Any = None,
        status: Any = None,
        phase_config: Any = None,
        *,
        control_dt: Optional[float] = None,
        spec_dir: Path | str = SPEC_DIR,
    ) -> "ProgressTracker":
        return cls(
            load_progress_spec(env_id, status, phase_config, spec_dir=spec_dir),
            control_dt=control_dt,
        )

    def reset(self) -> None:
        self._sample_index = 0
        self._time: Optional[float] = None
        self._signals: Dict[str, _DebouncedSignal] = {}
        self._expr_previous = {item_id: False for item_id in self._milestones}
        self._unconditional: Dict[str, float] = {}
        self._conditional: Dict[str, float] = {}
        self._compliant: Dict[str, float] = {}
        self._violations: Dict[str, list[Dict[str, Any]]] = {
            item_id: [] for item_id in self._milestones
        }
        self._disqualified: set[str] = set()
        self._sync_candidates: Dict[str, Dict[str, float]] = {
            group_id: {} for group_id in self._groups
        }
        self._failed_groups: set[str] = set()
        self._timeline: list[Dict[str, Any]] = []

    def _timestamp(self, timestamp: Optional[float], dt: Optional[float]) -> float:
        if timestamp is not None:
            value = float(timestamp)
            if self._time is not None and value < self._time:
                raise ValueError("timestamp must be monotonic")
            return value
        increment = dt if dt is not None else self.control_dt
        if increment is None:
            raise ValueError("Pass timestamp, dt, or construct with control_dt")
        return (self._time or 0.0) + float(increment)

    @staticmethod
    def _coerce_sample(value: Any) -> SignalSample:
        if isinstance(value, SignalSample):
            return value
        if isinstance(value, Mapping):
            return SignalSample(bool(value.get("value")), value.get("agent"))
        return SignalSample(bool(value))

    def _debounce(self, raw: Mapping[str, Any]) -> None:
        defaults = self.spec.get("defaults", {})
        default_samples = int(defaults.get("debounce_samples", 2))
        for name in self.spec.get("signals", {}):
            sample = self._coerce_sample(raw.get(name, False))
            state = self._signals.setdefault(name, _DebouncedSignal())
            if sample.value == state.candidate and sample.agent == state.candidate_agent:
                state.count += 1
            else:
                state.candidate = sample.value
                state.candidate_agent = sample.agent
                state.count = 1
            needed = int(
                self.spec["signals"].get(name, {}).get("debounce_samples", default_samples)
            )
            if state.count >= needed:
                state.stable = state.candidate
                state.stable_agent = state.candidate_agent

    def _eval_predicate(self, predicate: Any, *, enforce_agent: bool = False) -> bool:
        if isinstance(predicate, str):
            predicate = {"signal": predicate}
        if "signal" in predicate:
            name = predicate["signal"]
            state = self._signals.get(name, _DebouncedSignal())
            expected = predicate.get("value", True)
            if state.stable is not bool(expected):
                return False
            if enforce_agent:
                expected_agent = predicate.get("agent")
                configured_agent = self.spec["signals"].get(name, {}).get("agent")
                expected_agent = expected_agent or configured_agent
                if state.stable_agent and expected_agent and state.stable_agent != expected_agent:
                    return False
            return True
        if "all" in predicate:
            return all(
                self._eval_predicate(child, enforce_agent=enforce_agent)
                for child in predicate["all"]
            )
        if "any" in predicate:
            return any(
                self._eval_predicate(child, enforce_agent=enforce_agent)
                for child in predicate["any"]
            )
        if "not" in predicate:
            return not self._eval_predicate(
                predicate["not"], enforce_agent=enforce_agent
            )
        return False

    def _record(self, kind: str, milestone: str, **details: Any) -> None:
        event = {
            "time": self._time,
            "sample": self._sample_index,
            "kind": kind,
            "milestone": milestone,
        }
        event.update(details)
        self._timeline.append(event)

    def _violate(self, milestone: str, reason: str, **details: Any) -> None:
        violation = {"time": self._time, "reason": reason, **details}
        self._violations[milestone].append(violation)
        self._disqualified.add(milestone)
        self._record("violation", milestone, reason=reason, **details)

    def _commit(self, milestone: str, timestamp: Optional[float] = None) -> None:
        if milestone in self._compliant or milestone in self._disqualified:
            return
        completed_at = self._time if timestamp is None else timestamp
        self._compliant[milestone] = float(completed_at)
        self._record("compliant", milestone)

    def _check_pending_sync(self) -> None:
        default_window = float(self.spec.get("defaults", {}).get("sync_window_seconds", 45.0))
        for group_id, candidates in self._sync_candidates.items():
            if not candidates or group_id in self._failed_groups:
                continue
            group = self._groups[group_id]
            members = group["members"]
            if group.get("reject_early_release", True):
                for member in tuple(candidates):
                    hold_signal = self._milestones[member].get("hold_signal")
                    if hold_signal and not self._signals.get(
                        hold_signal, _DebouncedSignal()
                    ).stable:
                        self._fail_group(group_id, "early_release", released=member)
                        break
            if group_id in self._failed_groups:
                continue
            window = float(group.get("window_seconds", default_window))
            first = min(candidates.values())
            if len(candidates) < len(members) and self._time - first > window:
                self._fail_group(group_id, "sync_timeout", window_seconds=window)

    def _fail_group(self, group_id: str, reason: str, **details: Any) -> None:
        self._failed_groups.add(group_id)
        group = self._groups[group_id]
        for member in group["members"]:
            self._violate(member, reason, sync_group=group_id, **details)

    def _candidate(self, milestone: str) -> None:
        item = self._milestones[milestone]
        physical = self._eval_predicate(item["predicate"], enforce_agent=False)
        strict = self._eval_predicate(item["predicate"], enforce_agent=True)
        if not physical:
            return
        self._unconditional.setdefault(milestone, float(self._time))
        self._record("unconditional", milestone)

        deps_met = self._predecessors[milestone].issubset(self._compliant)
        if deps_met:
            self._conditional.setdefault(milestone, float(self._time))
        if not strict:
            self._violate(milestone, "wrong_agent", expected=item["agent"])
            return
        if not deps_met:
            missing = sorted(self._predecessors[milestone] - set(self._compliant))
            self._violate(milestone, "order_violation", missing=missing)
            return
        group_id = item.get("sync_group")
        if not group_id:
            self._commit(milestone)
            return
        if group_id in self._failed_groups:
            self._violate(milestone, "sync_group_failed", sync_group=group_id)
            return
        candidates = self._sync_candidates[group_id]
        candidates.setdefault(milestone, float(self._time))
        group = self._groups[group_id]
        if len(candidates) == len(group["members"]):
            default_window = float(
                self.spec.get("defaults", {}).get("sync_window_seconds", 45.0)
            )
            window = float(group.get("window_seconds", default_window))
            spread = max(candidates.values()) - min(candidates.values())
            if spread <= window:
                for member in group["members"]:
                    self._commit(member, candidates[member])
            else:
                self._fail_group(
                    group_id, "sync_window_exceeded", spread=spread, window_seconds=window
                )

    def sample(
        self,
        signals: Mapping[str, Any],
        *,
        timestamp: Optional[float] = None,
        dt: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Consume one L0 sample and return a serializable snapshot."""

        self._time = self._timestamp(timestamp, dt)
        self._sample_index += 1
        self._debounce(signals)
        self._check_pending_sync()
        for milestone, item in self._milestones.items():
            current = self._eval_predicate(item["predicate"], enforce_agent=False)
            if current and not self._expr_previous[milestone]:
                self._candidate(milestone)
            self._expr_previous[milestone] = current
        self._check_pending_sync()
        return self.snapshot()

    update = sample

    def snapshot(self) -> Dict[str, Any]:
        """Return strict metric plus all diagnostic views."""

        prefix = 0
        for milestone in self._order:
            if milestone not in self._compliant:
                break
            prefix += 1
        total = len(self._order)
        return {
            "task": self.spec.get("task"),
            "mode": self.spec.get("mode"),
            "sample_level": "L0",
            "samples": self._sample_index,
            "time": self._time,
            "normalized_progress": len(self._compliant) / total,
            "completed_count": len(self._compliant),
            "milestone_count": total,
            "compliant": {item: item in self._compliant for item in self._order},
            "diagnostics": {
                "unconditional": {
                    item: item in self._unconditional for item in self._order
                },
                "conditional": {
                    item: item in self._conditional for item in self._order
                },
                "continuous_prefix_count": prefix,
                "continuous_prefix_progress": prefix / total,
                "violations": copy.deepcopy(self._violations),
            },
            "timeline": copy.deepcopy(self._timeline),
        }


__all__ = [
    "ProgressSpecError",
    "ProgressTracker",
    "SignalSample",
    "load_progress_spec",
    "validate_progress_spec",
]

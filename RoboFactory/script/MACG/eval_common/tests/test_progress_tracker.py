"""Unit tests for strict ProgressTracker semantics and progress specs."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from script.MACG.eval_common.aggregate_progress import aggregate
from script.MACG.eval_common.eval_hooks import (
    extract_progress_signals,
    finalize_episode_metrics,
)
from script.MACG.eval_common.progress_tracker import (
    ProgressSpecError,
    ProgressTracker,
    SignalSample,
    load_progress_spec,
    validate_progress_spec,
)


SPEC_DIR = Path(__file__).resolve().parents[1] / "progress_specs"


def _serial_spec():
    return {
        "task": "toy",
        "mode": "indomain",
        "defaults": {"debounce_samples": 1, "sync_window_seconds": 1.0},
        "signals": {
            "a0_hold": {"agent": "agent0"},
            "a0_place": {"agent": "agent0"},
            "a1_hold": {"agent": "agent1"},
            "a1_place": {"agent": "agent1"},
        },
        "milestones": [
            {"id": "pick0", "agent": "agent0", "predicate": "a0_hold"},
            {"id": "place0", "agent": "agent0", "predicate": "a0_place", "depends_on": ["pick0"]},
            {"id": "pick1", "agent": "agent1", "predicate": "a1_hold", "depends_on": ["place0"]},
            {"id": "place1", "agent": "agent1", "predicate": "a1_place", "depends_on": ["pick1"]},
        ],
        "order_constraints": [
            {"before": "pick0", "after": "place0"},
            {"before": "place0", "after": "pick1"},
            {"before": "pick1", "after": "place1"},
        ],
    }


def _sync_spec():
    return {
        "task": "toy_sync",
        "mode": "sync",
        "defaults": {"debounce_samples": 1, "sync_window_seconds": 1.0},
        "signals": {
            "a0_hold": {"agent": "agent0"},
            "a1_hold": {"agent": "agent1"},
            "a0_place": {"agent": "agent0"},
        },
        "milestones": [
            {
                "id": "pick0",
                "agent": "agent0",
                "predicate": "a0_hold",
                "hold_signal": "a0_hold",
                "sync_group": "sync_pick",
            },
            {
                "id": "pick1",
                "agent": "agent1",
                "predicate": "a1_hold",
                "hold_signal": "a1_hold",
                "sync_group": "sync_pick",
            },
            {
                "id": "place0",
                "agent": "agent0",
                "predicate": "a0_place",
                "depends_on": ["pick0", "pick1"],
            },
        ],
        "sync_groups": [
            {
                "id": "sync_pick",
                "members": ["pick0", "pick1"],
                "reject_early_release": True,
            }
        ],
        "order_constraints": [
            {"before": "pick0", "after": "place0"},
            {"before": "pick1", "after": "place0"},
        ],
    }


class ProgressTrackerTests(unittest.TestCase):
    def test_serial_progress_and_latch(self):
        tracker = ProgressTracker(_serial_spec(), control_dt=0.05)
        tracker.sample({"a0_hold": True}, timestamp=0.0)
        self.assertAlmostEqual(tracker.snapshot()["normalized_progress"], 0.25)
        # Flicker off should not revoke progress.
        tracker.sample({"a0_hold": False}, timestamp=0.05)
        self.assertAlmostEqual(tracker.snapshot()["normalized_progress"], 0.25)
        tracker.sample({"a0_place": True}, timestamp=0.1)
        tracker.sample({"a1_hold": True}, timestamp=0.15)
        tracker.sample({"a1_place": True}, timestamp=0.2)
        snap = tracker.snapshot()
        self.assertEqual(snap["normalized_progress"], 1.0)
        self.assertTrue(all(snap["compliant"].values()))

    def test_wrong_agent_is_unconditional_only(self):
        tracker = ProgressTracker(_serial_spec(), control_dt=0.05)
        tracker.sample(
            {"a0_hold": SignalSample(True, agent="agent1")},
            timestamp=0.0,
        )
        snap = tracker.snapshot()
        self.assertTrue(snap["diagnostics"]["unconditional"]["pick0"])
        self.assertFalse(snap["compliant"]["pick0"])
        self.assertEqual(snap["normalized_progress"], 0.0)
        self.assertTrue(snap["diagnostics"]["violations"]["pick0"])

    def test_order_violation_does_not_count_for_progress(self):
        tracker = ProgressTracker(_serial_spec(), control_dt=0.05)
        tracker.sample({"a0_place": True}, timestamp=0.0)  # place before pick
        snap = tracker.snapshot()
        self.assertTrue(snap["diagnostics"]["unconditional"]["place0"])
        self.assertFalse(snap["compliant"]["place0"])
        self.assertEqual(snap["normalized_progress"], 0.0)

    def test_sync_window_boundary_and_timeout(self):
        ok = ProgressTracker(_sync_spec(), control_dt=0.05)
        ok.sample({"a0_hold": True}, timestamp=0.0)
        ok.sample({"a0_hold": True, "a1_hold": True}, timestamp=1.0)
        self.assertEqual(ok.snapshot()["normalized_progress"], 2 / 3)

        bad = ProgressTracker(_sync_spec(), control_dt=0.05)
        bad.sample({"a0_hold": True}, timestamp=0.0)
        bad.sample({"a0_hold": True, "a1_hold": True}, timestamp=1.01)
        snap = bad.snapshot()
        self.assertEqual(snap["normalized_progress"], 0.0)
        self.assertTrue(snap["diagnostics"]["violations"]["pick0"])

    def test_early_release_fails_sync_group(self):
        tracker = ProgressTracker(_sync_spec(), control_dt=0.05)
        tracker.sample({"a0_hold": True}, timestamp=0.0)
        tracker.sample({"a0_hold": False}, timestamp=0.1)
        snap = tracker.snapshot()
        self.assertEqual(snap["normalized_progress"], 0.0)
        self.assertTrue(
            any(
                item["reason"] == "early_release"
                for item in snap["diagnostics"]["violations"]["pick0"]
            )
        )

    def test_continuous_prefix_stops_on_gap(self):
        tracker = ProgressTracker(_serial_spec(), control_dt=0.05)
        tracker.sample({"a0_hold": True}, timestamp=0.0)
        tracker.sample({"a1_hold": True}, timestamp=0.1)  # skip place0
        snap = tracker.snapshot()
        self.assertEqual(snap["diagnostics"]["continuous_prefix_count"], 1)
        self.assertAlmostEqual(snap["diagnostics"]["continuous_prefix_progress"], 0.25)

    def test_debounce_requires_stable_samples(self):
        spec = _serial_spec()
        spec["defaults"]["debounce_samples"] = 2
        tracker = ProgressTracker(spec, control_dt=0.05)
        tracker.sample({"a0_hold": True}, timestamp=0.0)
        self.assertEqual(tracker.snapshot()["normalized_progress"], 0.0)
        tracker.sample({"a0_hold": True}, timestamp=0.05)
        self.assertAlmostEqual(tracker.snapshot()["normalized_progress"], 0.25)


class ProgressSpecTests(unittest.TestCase):
    def test_all_specs_validate_and_load_by_phase(self):
        cases = [
            ("stack_cube.json", "TwoArmsStackCube-rf", "two_arms_stack_cube_12", "stack_cubes_sync.yaml", "sync"),
            ("stack_bowl.json", "TwoArmsStackBowl-rf", "two_arms_stack_bowl_21", "stack_bowls_syncreorder.yaml", "syncreorder"),
            ("push_cube.json", "TwoArmsPushCube-rf", "two_arms_push_cube_12", "push_cubes_indomain.yaml", "indomain"),
            ("place_can_basket.json", "TwoArmsPlaceCanBasket-rf", "two_arms_place_can_basket", "place_can_basket_sync.yaml", "sync"),
            ("place_burger_fries.json", "TwoArmsPlaceBurgerFries-rf", "two_arms_place_burger_fries_12", "place_burger_fries_reorder.yaml", "reorder"),
            ("place_object_cabinet.json", "TwoArmsPlaceObjectCabinet-rf", "two_arms_place_object_cabinet", "place_object_cabinet_indomain.yaml", "indomain"),
            ("place_object_bowl_cabinet.json", "TwoArmsPlaceObjectBowlCabinet-rf", "two_arms_place_object_bowl_cabinet", "place_object_bowl_cabinet_sync.yaml", "sync"),
            ("place_cube_in_bowl.json", "TwoArmsCubeInBowl-rf", "two_arms_place_cube_in_bowl", "place_cube_in_bowl_indomain.yaml", "indomain"),
            ("handover_bottles.json", "TwoArmsHandoverBottles-rf", "two_arms_handover_bottles_12", "handover_bottles_reorder.yaml", "reorder"),
        ]
        for filename, env_id, status, phase, mode in cases:
            path = SPEC_DIR / filename
            raw = json.loads(path.read_text(encoding="utf-8"))
            for mode_name, mode_body in raw["modes"].items():
                merged = {k: v for k, v in raw.items() if k != "modes"}
                merged.update(mode_body)
                merged["mode"] = mode_name
                validate_progress_spec(merged)
            loaded = load_progress_spec(env_id, status, phase, spec_dir=SPEC_DIR)
            self.assertEqual(loaded["mode"], mode)
            ProgressTracker(loaded, control_dt=0.05)

    def test_missing_phase_raises(self):
        with self.assertRaises(ProgressSpecError):
            load_progress_spec(
                "TwoArmsStackCube-rf",
                "two_arms_stack_cube_12",
                None,
                spec_dir=SPEC_DIR,
            )


class EvalHooksTests(unittest.TestCase):
    def test_extract_stack_cube_signals(self):
        signals = extract_progress_signals(
            "stack_cube",
            {
                "is_cubeA_grasped_left": True,
                "is_cubeB_grasped_right": False,
                "cubeA_placed": True,
                "is_cubeA_grasped": False,
                "is_cubeB_on_cubeA": True,
                "is_cubeB_grasped": False,
            },
        )
        self.assertTrue(signals["agent0_holds_cubeA"])
        self.assertTrue(signals["cubeA_placed_on_target"])
        self.assertTrue(signals["cubeB_on_cubeA"])

    def test_finalize_maps_to_aggregate_schema(self):
        tracker = ProgressTracker(_serial_spec(), control_dt=0.05)
        tracker.sample({"a0_hold": True}, timestamp=0.0)
        metrics = finalize_episode_metrics(
            tracker,
            task_success=False,
            seed=1000,
            env_id="Toy",
            status="toy",
            phase_config="toy_indomain.yaml",
            policy_steps=3,
        )
        summary = aggregate([metrics])
        key = f"{metrics['task_id']}:{metrics['mode']}"
        self.assertEqual(key, "toy:indomain")
        self.assertAlmostEqual(summary["tasks"][key]["normalized_progress"], 0.25)
        self.assertEqual(
            summary["tasks"][key]["subtask_completion"]["pick0"]["rate"],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()

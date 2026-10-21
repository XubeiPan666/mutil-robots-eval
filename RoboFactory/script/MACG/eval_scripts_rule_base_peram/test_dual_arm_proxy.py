#!/usr/bin/env python3

from __future__ import annotations

import unittest

import numpy as np

from dual_arm_proxy import DualArmProxy, ProxyError


def make_example() -> dict:
    return {
        "global_image": np.full((4, 4, 3), 1, dtype=np.uint8),
        "image_agent0": np.full((4, 4, 3), 2, dtype=np.uint8),
        "image_agent1": np.full((4, 4, 3), 3, dtype=np.uint8),
        "joint_position_agent0": np.arange(7, dtype=np.float32),
        "gripper_position_agent0": 0.01,
        "joint_position_agent1": np.arange(7, dtype=np.float32) + 10,
        "gripper_position_agent1": 0.02,
        "task": "Pick the can | Hold the basket",
        "unused": "must not be forwarded",
    }


class FakeClient:
    def __init__(self, arm_name, server_url, timeout_s, results, calls):
        self.arm_name = arm_name
        self.server_url = server_url
        self.results = results
        self.calls = calls

    def health(self):
        return {"ok": True, "url": self.server_url}

    def infer(self, example):
        self.calls[self.arm_name] = example
        result = self.results[self.arm_name]
        if isinstance(result, Exception):
            raise result
        return result


def make_proxy(results):
    calls = {}

    def factory(arm_name, server_url, timeout_s):
        return FakeClient(arm_name, server_url, timeout_s, results, calls)

    return DualArmProxy("http://arm0", "http://arm1", client_factory=factory), calls


class DualArmProxyTest(unittest.TestCase):
    def test_splits_inputs_and_concatenates_actions(self):
        arm0_actions = np.zeros((50, 8), dtype=np.float32)
        arm1_actions = np.ones((50, 8), dtype=np.float32)
        proxy, calls = make_proxy(
            {
                "arm0": {"actions": arm0_actions},
                "arm1": {"actions": arm1_actions},
            }
        )

        result = proxy.infer_combined(make_example())

        self.assertEqual(result["actions"].shape, (50, 16))
        np.testing.assert_array_equal(result["actions"][:, :8], arm0_actions)
        np.testing.assert_array_equal(result["actions"][:, 8:], arm1_actions)
        self.assertEqual(
            set(calls["arm0"]),
            {
                "global_image",
                "image_agent0",
                "joint_position_agent0",
                "gripper_position_agent0",
                "task",
            },
        )
        self.assertEqual(
            set(calls["arm1"]),
            {
                "global_image",
                "image_agent1",
                "joint_position_agent1",
                "gripper_position_agent1",
                "task",
            },
        )
        self.assertEqual(calls["arm0"]["task"], calls["arm1"]["task"])

    def test_rejects_wrong_action_width(self):
        proxy, _ = make_proxy(
            {
                "arm0": {"actions": np.zeros((50, 7))},
                "arm1": {"actions": np.zeros((50, 8))},
            }
        )
        with self.assertRaisesRegex(ProxyError, r"arm0.*\[T, 8\]"):
            proxy.infer_combined(make_example())

    def test_rejects_short_horizon(self):
        proxy, _ = make_proxy(
            {
                "arm0": {"actions": np.zeros((49, 8))},
                "arm1": {"actions": np.zeros((49, 8))},
            }
        )
        with self.assertRaisesRegex(ProxyError, "at least 50"):
            proxy.infer_combined(make_example())

    def test_rejects_mismatched_horizons(self):
        proxy, _ = make_proxy(
            {
                "arm0": {"actions": np.zeros((50, 8))},
                "arm1": {"actions": np.zeros((51, 8))},
            }
        )
        with self.assertRaisesRegex(ProxyError, "horizons differ"):
            proxy.infer_combined(make_example())

    def test_propagates_upstream_failure(self):
        proxy, _ = make_proxy(
            {
                "arm0": {"actions": np.zeros((50, 8))},
                "arm1": ProxyError("arm1 unavailable"),
            }
        )
        with self.assertRaisesRegex(ProxyError, "arm1 unavailable"):
            proxy.infer_combined(make_example())


if __name__ == "__main__":
    unittest.main()

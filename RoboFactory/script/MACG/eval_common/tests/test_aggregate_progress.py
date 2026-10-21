import unittest

from script.MACG.eval_common.aggregate_progress import aggregate


class AggregateProgressTest(unittest.TestCase):
    def test_uses_all_rollouts_for_subtask_rates_and_macro_averages_tasks(self):
        episodes = [
            {
                "task_id": "TaskA",
                "mode": "sync",
                "task_success": True,
                "normalized_progress": 1.0,
                "conditional_completion": {"pick": True, "place": True},
                "unconditional_completion": {"pick": True, "place": True},
                "eligible": {"pick": True, "place": True},
            },
            {
                "task_id": "TaskA",
                "mode": "sync",
                "task_success": False,
                "normalized_progress": 0.5,
                "conditional_completion": {"pick": True, "place": False},
                "unconditional_completion": {"pick": True, "place": False},
                "eligible": {"pick": True, "place": True},
            },
            {
                "task_id": "TaskB",
                "mode": "indomain",
                "task_success": False,
                "normalized_progress": 0.25,
                "conditional_completion": {
                    "open": True,
                    "pick": False,
                    "place": False,
                },
                "unconditional_completion": {
                    "open": True,
                    "pick": False,
                    "place": False,
                },
                "eligible": {"open": True, "pick": True, "place": False},
            },
        ]

        summary = aggregate(episodes)

        self.assertAlmostEqual(summary["tasks"]["TaskA:sync"]["normalized_progress"], 0.75)
        self.assertAlmostEqual(
            summary["tasks"]["TaskA:sync"]["subtask_completion"]["place"]["rate"],
            0.5,
        )
        self.assertAlmostEqual(summary["macro_normalized_progress"], 0.5)
        self.assertAlmostEqual(summary["macro_task_success"], 0.25)

    def test_final_success_does_not_override_strict_progress(self):
        summary = aggregate(
            [
                {
                    "task_id": "TaskA",
                    "mode": "reorder",
                    "task_success": True,
                    "normalized_progress": 0.5,
                    "conditional_completion": {"pick": True, "place": False},
                }
            ]
        )

        task = summary["tasks"]["TaskA:reorder"]
        self.assertEqual(task["task_success_rate"], 1.0)
        self.assertEqual(task["normalized_progress"], 0.5)


if __name__ == "__main__":
    unittest.main()

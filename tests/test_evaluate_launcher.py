"""Exercise public argument handling and process wiring without a simulator."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "evaluate.sh"
TASKS = [
    "stack_bowls", "stack_cubes", "push_cubes", "place_can_basket",
    "place_burger_fries", "place_object_cabinet", "handover_bottles",
    "place_cube_in_bowl", "place_object_bowl_cabinet",
]


class EvaluateLauncherTests(unittest.TestCase):
    def run_launcher(self, *args, launcher=LAUNCHER, env=None, cwd="/tmp"):
        clean_env = dict(os.environ)
        for key in (
            "VLA_SERVER_URL", "ARM0_VLA_SERVER_URL", "ARM1_VLA_SERVER_URL",
            "PERARM_PROXY_PORT", "PYTHON_BIN",
        ):
            clean_env.pop(key, None)
        clean_env.update(env or {})
        return subprocess.run(
            ["bash", str(launcher), *args], cwd=cwd, env=clean_env,
            capture_output=True, text=True, check=False,
        )

    def test_help_and_task_list_need_no_endpoint(self):
        help_result = self.run_launcher("--help")
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("1000–1099", help_result.stdout)
        listed = self.run_launcher("--list-tasks")
        self.assertEqual(listed.returncode, 0, listed.stderr)
        self.assertEqual(listed.stdout.splitlines(), TASKS)

    def test_dry_run_resolves_paths_from_another_directory(self):
        result = self.run_launcher(
            "--task", "stack_cubes", "--task", "push_cubes", "--dry-run",
            "--server-url", "http://127.0.0.1:8000",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(str(ROOT / "RoboFactory"), result.stdout)
        commands = [line for line in result.stdout.splitlines() if "command:" in line]
        self.assertEqual(len(commands), 2)
        self.assertIn("stack_cubes.sh", commands[0])
        self.assertIn("push_cubes.sh", commands[1])
        self.assertNotIn("run_with_proxy.sh", result.stdout)

    def test_default_suite_and_all_alias_keep_original_order(self):
        for selection in [[], ["--task", "all"]]:
            with self.subTest(selection=selection):
                result = self.run_launcher(
                    *selection, "--dry-run", env={"VLA_SERVER_URL": "http://model:8000"},
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                commands = [line for line in result.stdout.splitlines() if "command:" in line]
                self.assertEqual(len(commands), len(TASKS))
                for task, command in zip(TASKS, commands):
                    self.assertIn(f"/{task}.sh", command)

    def test_per_arm_mode_uses_proxy_and_environment(self):
        result = self.run_launcher(
            "--mode", "per-arm", "--task", "stack_bowls", "--dry-run",
            env={"ARM0_VLA_SERVER_URL": "http://arm0:8000", "ARM1_VLA_SERVER_URL": "http://arm1:8001"},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("run_with_proxy.sh", result.stdout)
        self.assertIn("ARM0_VLA_SERVER_URL=http://arm0:8000", result.stdout)
        self.assertIn("ARM1_VLA_SERVER_URL=http://arm1:8001", result.stdout)

    def test_invalid_arguments_fail_before_running(self):
        cases = [
            [], ["--task"], ["--server-url"], ["--unknown"],
            ["--mode", "invalid"], ["--task", "unknown"],
            ["--task", "stack_cubes", "--task", "stack_cubes"],
            ["--task", "all", "--task", "stack_cubes"],
            ["--server-url", "file:///tmp/model"],
            ["--server-url", "http://"],
            ["--server-url", "http://model with spaces"],
            ["--arm0-url", "http://arm0:8000"],
            ["--mode", "per-arm", "--arm0-url", "http://arm0:8000"],
            ["--mode", "per-arm", "--server-url", "http://model:8000"],
        ]
        for args in cases:
            with self.subTest(args=args):
                result = self.run_launcher(*args, "--dry-run")
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertIn("Error:", result.stderr)
                self.assertNotIn("[evaluate] command:", result.stdout)

    def test_invalid_proxy_port_fails(self):
        for port in ["0", "65536", "not-a-port", "000000"]:
            with self.subTest(port=port):
                result = self.run_launcher(
                    "--mode", "per-arm", "--dry-run",
                    env={"ARM0_VLA_SERVER_URL": "http://arm0:8000", "ARM1_VLA_SERVER_URL": "http://arm1:8001", "PERARM_PROXY_PORT": port},
                )
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("PERARM_PROXY_PORT", result.stderr)

    def test_execution_forwards_environment_and_stops_on_nested_failure(self):
        with tempfile.TemporaryDirectory(prefix="evaluation launcher ") as directory:
            root = Path(directory)
            (root / "scripts").mkdir()
            launcher = root / "scripts" / "evaluate.sh"
            shutil.copy2(LAUNCHER, launcher)
            task_dir = root / "RoboFactory/script/MACG/eval_scripts_rule_base"
            task_dir.mkdir(parents=True)
            capture = root / "capture.txt"
            (task_dir / "stack_cubes.sh").write_text(
                'printf "%s\\n" "$PWD" "$VLA_SERVER_URL" "$CUSTOM_EVAL_SETTING" > "$CAPTURE"\n'
                'bash -c "exit 7"\n'
                'printf "unexpected continuation\\n" >> "$CAPTURE"\n',
            )
            (task_dir / "push_cubes.sh").write_text('printf "unexpected task\\n" >> "$CAPTURE"\n')
            bin_dir = root / "bin"
            bin_dir.mkdir()
            for name in ["python", "bc"]:
                executable = bin_dir / name
                executable.write_text("#!/bin/sh\nexit 0\n")
                executable.chmod(0o755)
            result = self.run_launcher(
                "--task", "stack_cubes", "--task", "push_cubes",
                "--server-url", "http://127.0.0.1:8000", launcher=launcher,
                env={"CAPTURE": str(capture), "CUSTOM_EVAL_SETTING": "preserved", "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"]},
            )
            self.assertEqual(result.returncode, 7, result.stderr)
            self.assertEqual(capture.read_text().splitlines(), [
                str(root / "RoboFactory"), "http://127.0.0.1:8000", "preserved",
            ])


class LegacyTaskSafetyTests(unittest.TestCase):
    def make_stack_cube_fixture(self, directory, fail=False):
        root = Path(directory) / "RoboFactory"
        task_dir = root / "script/MACG/eval_scripts_rule_base"
        task_dir.mkdir(parents=True)
        source_dir = ROOT / "RoboFactory/script/MACG/eval_scripts_rule_base"
        for name in ["stack_cubes.sh", "_common.sh"]:
            shutil.copy2(source_dir / name, task_dir / name)
        evaluator = root / "script/MACG/two_arms_stack_cube/eval_two_arms_stackcube.sh"
        evaluator.parent.mkdir()
        evaluator.write_text(
            '#!/usr/bin/env bash\n'
            'printf "called\\n" >> calls.txt\n'
            + ('exit 7\n' if fail else 'mkdir -p eval_video/TwoArmsStackCube-rf\n')
        )
        return root, task_dir / "stack_cubes.sh"

    def test_existing_outputs_are_refused_before_the_next_evaluation(self):
        with tempfile.TemporaryDirectory(prefix="task safety ") as directory:
            root, wrapper = self.make_stack_cube_fixture(directory)
            result = subprocess.run(["bash", str(wrapper)], cwd="/tmp", capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((root / "calls.txt").read_text().splitlines(), ["called"] * 4)
            for mode in ["in-domain", "reorder", "sync", "syncreorder"]:
                output = root / f"eval_video/TwoArmsStackCube-rf-{mode}-pi0"
                self.assertTrue(output.is_dir())
                (output / "keep.txt").write_text(mode)
            rerun = subprocess.run(["bash", str(wrapper)], cwd="/tmp", capture_output=True, text=True)
            self.assertNotEqual(rerun.returncode, 0)
            self.assertIn("output already exists", rerun.stderr)
            self.assertEqual((root / "calls.txt").read_text().splitlines(), ["called"] * 4)
            self.assertEqual(len(list((root / "eval_video").glob("*/keep.txt"))), 4)

    def test_child_process_failure_stops_legacy_wrapper(self):
        with tempfile.TemporaryDirectory(prefix="task failure ") as directory:
            root, wrapper = self.make_stack_cube_fixture(directory, fail=True)
            result = subprocess.run(["bash", str(wrapper)], cwd="/tmp", capture_output=True, text=True)
            self.assertEqual(result.returncode, 7, result.stderr)
            self.assertEqual((root / "calls.txt").read_text().splitlines(), ["called"])


if __name__ == "__main__":
    unittest.main()

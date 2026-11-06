"""Startup regressions without training packages, weights, GPU or Slurm."""

import importlib.util
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("wait_for_policy", ROOT / "scripts/slurm/wait_for_policy.py")
WAIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WAIT)


class StartupTests(unittest.TestCase):
    def test_launcher_resolves_default_and_explicit_model_configs(self):
        bash = os.environ.get("TEST_BASH") or shutil.which("bash")
        if not bash:
            self.skipTest("Bash is required to check launcher configuration expansion")
        script = (ROOT / "scripts/slurm/eval_official.sh").read_text()
        prefix = script.split('test -d "$CHECKPOINT_DIR/params"', 1)[0]
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ, FAKE_WS=Path(tmp).as_posix(), CHECKPOINT_DIR=tmp)
            env.pop("POLICY_CONFIG", None)
            for selected in (None, "pi0_two_arms_multi_task", "pi05_two_arms_multi_task"):
                candidate = dict(env)
                if selected:
                    candidate["POLICY_CONFIG"] = selected
                command = ('ws_find() { printf "%s" "$FAKE_WS"; }\n' + prefix
                           + '\nprintf "%s" "$POLICY_CONFIG"')
                result = subprocess.run([bash, "-c", command], env=candidate,
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout, selected or "aevla_pi05_skill_lora_two_arms_multitask")

    def test_dead_launcher_reports_original_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "server.log").write_text("ModuleNotFoundError: No module named 'openpi'")
            with patch.object(WAIT.os, "kill", side_effect=ProcessLookupError):
                with self.assertRaisesRegex(RuntimeError, "ModuleNotFoundError"):
                    WAIT.wait_for_policy(12345, root)

    def test_timeout_reports_startup_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "server.log").write_text("Still compiling")
            # os.kill(pid, 0) is a Linux existence probe, not a portable signal.
            with patch.object(WAIT.os, "kill", return_value=None):
                with self.assertRaisesRegex(RuntimeError, "Still compiling"):
                    WAIT.wait_for_policy(os.getpid(), tmp, timeout=0)

    def test_two_live_endpoints_on_same_host(self):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"model_loaded":true,"status":"healthy"}')

            def log_message(self, *args):
                pass

        with tempfile.TemporaryDirectory() as tmp:
            servers = [ThreadingHTTPServer(("127.0.0.1", 0), Handler) for _ in range(2)]
            for server in servers:
                threading.Thread(target=server.serve_forever, daemon=True).start()
            try:
                urls = []
                for i, server in enumerate(servers):
                    folder = Path(tmp) / str(i)
                    folder.mkdir()
                    url = f"http://127.0.0.1:{server.server_port}"
                    (folder / "server_ready.json").write_text(json.dumps({"pid": os.getpid(), "url": url}))
                    with patch.object(WAIT.os, "kill", return_value=None):
                        urls.append(WAIT.wait_for_policy(os.getpid(), folder, timeout=3))
                self.assertNotEqual(urls[0], urls[1])
            finally:
                for server in servers:
                    server.shutdown()
                    server.server_close()

    def test_external_wrapper_imports_src_and_restores_warmup_rng(self):
        # A src-layout checkout deliberately not installed into this interpreter.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train, run = root / "training", root / "run"
            run.mkdir()
            files = {
                "src/openpi/__init__.py": "MARKER = 'training-src'\n",
                "packages/openpi-client/src/openpi_client/__init__.py": "MARKER = 'client-src'\n",
                "numpy.py": """
uint8 = float64 = None
def zeros(*args, **kwargs): return None
class Actions:
    ndim = 2
    shape = (50, 16)
    def all(self): return True
def asarray(value): return Actions()
def isfinite(value): return Actions()
""",
                "yaml.py": "def safe_load(text): return {'phase_instructions': {1: {'agent0':'Pick', 'agent1':'Wait'}}}\n",
                "server_agent0.py": """
import openpi
import openpi_client
assert openpi.MARKER == 'training-src'
assert openpi_client.MARKER == 'client-src'
class App:
    def run(self, **kwargs): raise AssertionError('warmup wrapper not installed')
app = App()
class Policy:
    _rng = 0
    _is_pytorch_model = False
    def infer(self, obs):
        self._rng += 1
        return {'actions': None}
policy = Policy()
""",
                "deploy_minimal.py": """
import argparse
import server_agent0
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='aevla_pi05_skill_lora_two_arms_multitask')
    parser.add_argument('--checkpoint-dir')
    args = parser.parse_args()
    print('Selected config:', args.config)
    server_agent0.app.run(host='127.0.0.1', port=0, debug=False)
""",
                "werkzeug/__init__.py": "",
                "werkzeug/serving.py": """
import server_agent0
class Server:
    server_port = 54321
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def serve_forever(self):
        assert server_agent0.policy._rng == 0
        print('RNG restored')
def make_server(host, port, app, threaded):
    assert host == '127.0.0.1' and port == 0 and threaded
    return Server()
""",
            }
            for name, content in files.items():
                path = train / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
            phase = root / "phase.yaml"
            phase.write_text("fixture")
            env = dict(os.environ, TRAIN_ROOT=str(train), RUN_DIR=str(run),
                       WARMUP_PHASE_CONFIG=str(phase), PYTHONIOENCODING="utf-8")
            wrapper = str(ROOT / "scripts/slurm/serve_warmed.py")
            for args in (["--check-imports"], [],
                         ["--config", "pi0_two_arms_multi_task", "--checkpoint-dir", "pi0/29999"],
                         ["--config", "pi05_two_arms_multi_task", "--checkpoint-dir", "pi05/29999"]):
                result = subprocess.run([sys.executable, "-I", wrapper, *args], env=env,
                                        cwd=run, capture_output=True, encoding="utf-8")
                self.assertEqual(result.returncode, 0, result.stderr)
                if args != ["--check-imports"]:
                    self.assertIn("RNG restored", result.stdout)
                    expected = args[1] if args else "aevla_pi05_skill_lora_two_arms_multitask"
                    self.assertIn("Selected config: " + expected, result.stdout)
            ready = json.loads((run / "server_ready.json").read_text())
            self.assertEqual(ready["url"], "http://127.0.0.1:54321")


if __name__ == "__main__":
    unittest.main()

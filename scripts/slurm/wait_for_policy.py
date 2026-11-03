"""Wait for this job's bound policy endpoint, reporting the actual startup log."""

import argparse
import json
import os
from pathlib import Path
import time
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, build_opener


def startup_error(run_dir, reason):
    log = run_dir / "server.log"
    tail = "\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-80:]) if log.exists() else "(not created)"
    return RuntimeError(f"{reason}\nModel startup log: {log}\n{tail}")


def wait_for_policy(pid, run_dir, timeout=1800, interval=3):
    run_dir = Path(run_dir)
    opener = build_opener(ProxyHandler({}))
    deadline = time.monotonic() + timeout
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            raise startup_error(run_dir, "Model launcher exited before readiness") from None
        try:
            ready = json.loads((run_dir / "server_ready.json").read_text(encoding="utf-8"))
            url = ready["url"]
            parsed = urlsplit(url)
            if parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or not parsed.port:
                raise ValueError(f"Invalid local policy endpoint: {url}")
            os.kill(int(ready["pid"]), 0)
            with opener.open(url + "/health", timeout=5) as reply:
                health = json.load(reply)
            if health.get("model_loaded") is True:
                return url
        except (FileNotFoundError, URLError, TimeoutError, json.JSONDecodeError):
            pass
        except ProcessLookupError:
            raise startup_error(run_dir, "Model process exited after binding its port") from None
        if time.monotonic() >= deadline:
            raise startup_error(run_dir, f"Model startup/warmup exceeded {timeout} seconds")
        time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    print(wait_for_policy(args.pid, args.run_dir), flush=True)

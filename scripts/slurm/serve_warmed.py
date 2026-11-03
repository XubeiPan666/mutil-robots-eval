"""Start the repository's HTTP server after non-scoring JAX warmup.

Run with the training project's uv environment. The official deployment main,
policy configuration, transforms and HTTP handlers are reused without edits.
"""

import os
import sys
import json
from importlib import metadata
from pathlib import Path

# The wrapper lives outside the training repository. Its src-layout packages
# must be resolved from that checkout, even when it is not installed editable.
train_root = Path(os.environ["TRAIN_ROOT"]).resolve()
sys.path[:0] = [str(train_root), str(train_root / "src"),
                str(train_root / "packages/openpi-client/src")]
import numpy as np
import yaml
import openpi
import openpi_client
import deploy_minimal
import server_agent0


def main():
    print("Training Python:", sys.executable, flush=True)
    print("OpenPI source:", openpi.__file__, flush=True)
    print("OpenPI client source:", openpi_client.__file__, flush=True)
    if sys.argv[1:] == ["--check-imports"]:
        print("Training server imports OK; no model loaded.", flush=True)
        return

    def warmed_run(*args, **kwargs):
        from werkzeug.serving import make_server

        policy = server_agent0.policy
        if not hasattr(policy, "_rng") or policy._is_pytorch_model:
            raise RuntimeError("This warmup wrapper is for the verified JAX policy only")
        saved_rng = policy._rng
        phase_config = yaml.safe_load(Path(os.environ["WARMUP_PHASE_CONFIG"]).read_text())
        print("Training Python:", sys.executable, flush=True)
        versions = sorted(f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions())
        (Path(os.environ["RUN_DIR"]) / "training-packages.txt").write_text("\n".join(versions) + "\n")
        try:
            for phase, instructions in sorted(phase_config["phase_instructions"].items()):
                prompt = " | ".join(instructions[key] for key in sorted(instructions))
                example = {
                    "image_agent0": np.zeros((224, 224, 3), dtype=np.uint8),
                    "image_agent1": np.zeros((224, 224, 3), dtype=np.uint8),
                    "global_image": np.zeros((224, 224, 3), dtype=np.uint8),
                    "joint_position_agent0": np.zeros(7, dtype=np.float64),
                    "joint_position_agent1": np.zeros(7, dtype=np.float64),
                    "gripper_position_agent0": 1.0,
                    "gripper_position_agent1": 1.0,
                    "task": prompt, "prompt": prompt,
                }
                print(f"Non-scoring warmup phase {phase}: {prompt}", flush=True)
                actions = np.asarray(policy.infer(example)["actions"])
                if actions.ndim != 2 or actions.shape[0] < 50 or actions.shape[1] < 16:
                    raise ValueError(f"Unexpected actions: {actions.shape}")
                if not np.isfinite(actions).all():
                    raise ValueError("Non-finite warmup actions")
        finally:
            policy._rng = saved_rng
        print("Warmup complete; original policy RNG restored; starting HTTP server.", flush=True)
        host = kwargs.get("host", "127.0.0.1")
        port = kwargs.get("port", 0)
        # Bind port 0 and let the OS allocate a free port atomically. Different
        # Slurm jobs can share a node's network even with separate GPUs.
        with make_server(host, port, server_agent0.app, threaded=True) as server:
            root = Path(os.environ["RUN_DIR"])
            ready = {"pid": os.getpid(), "url": f"http://{host}:{server.server_port}"}
            temporary = root / "server_ready.json.tmp"
            temporary.write_text(json.dumps(ready), encoding="utf-8")
            temporary.replace(root / "server_ready.json")
            print(f"Serving policy at {ready['url']}", flush=True)
            server.serve_forever()

    server_agent0.app.run = warmed_run
    deploy_minimal.main()


if __name__ == "__main__":
    main()

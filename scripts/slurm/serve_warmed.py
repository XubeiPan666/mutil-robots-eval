"""Start the repository's HTTP server after non-scoring JAX warmup.

Run with the training project's uv environment. The official deployment main,
policy configuration, transforms and HTTP handlers are reused without edits.
"""

import os
import sys
from importlib import metadata
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, os.environ["TRAIN_ROOT"])
import deploy_minimal
import server_agent0


def main():
    original_run = server_agent0.app.run

    def warmed_run(*args, **kwargs):
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
        print("Warmup complete; original policy RNG restored; starting official HTTP server.", flush=True)
        return original_run(*args, **kwargs)

    server_agent0.app.run = warmed_run
    deploy_minimal.main()


if __name__ == "__main__":
    main()

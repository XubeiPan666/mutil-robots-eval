"""VLM-driven entry point for two-arm stack-bowl evaluation.

The control loop is shared with the rule-based evaluator to prevent policy,
planner, success-check, or recording behavior from drifting between baselines.
"""

import os
import sys

sys.path.append("./")
sys.path.insert(0, "./policy/Diffusion-Policy")
sys.path.insert(0, "./script/MACG/two_arms_stack_bowl")

# This must be set before importing the shared evaluator, which creates its
# OpenPI client at module import time.
os.environ.setdefault("VLA_SERVER_URL", "http://localhost:20020")

import tyro

from script.MACG.two_arms_stack_bowl.eval_stackbowl import Args, main


if __name__ == "__main__":
    main(tyro.cli(Args))

"""
Record perturbation manifests for all tasks.

Iterates over each (task, perturbation_yaml) pair, resets the environment
once per seed (no policy server needed), and writes a per-task manifest JSON.

Usage (run from the RoboFactory root directory):
    python scripts/perturbation/record_manifests.py \
        --seeds 1000-1099 \
        --out-dir release/manifests \
        [--asset-bundle-manifest release/perturbation_assets_v1/perturbation_assets_v1.manifest.json] \
        [--task two_arms_stack_cube]   # omit to record all tasks

The resulting manifests are used by eval_*.py via:
    --replay_manifest release/manifests/<task>_perturbed.json
"""

from __future__ import annotations

import argparse
import os
import sys

# Ensure RoboFactory root is on the path when run from the RoboFactory dir.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import gymnasium as gym
import numpy as np
import yaml

from tasks import *  # noqa: F401,F403 – registers all gym envs

from utils.perturbation.recorder import PerturbRecorder
from utils.perturbation.manifest import ManifestWriter


# ============================================================
# Task registry: (task_key, scene_config, perturbation_config, task_env_id, status)
# ============================================================
TASKS = [
    {
        "key": "two_arms_stack_cube",
        "scene_config": "configs/table/two_arms_stack_cube.yaml",
        "perturbation_config": "configs/perturbation/visual_perturbation_stackcube.yaml",
        "status": "two_arms_stack_cube_12",
    },
    {
        "key": "two_arms_stack_bowl",
        "scene_config": "configs/table/two_arms_stack_bowl.yaml",
        "perturbation_config": "configs/perturbation/visual_perturbation.yaml",
        "status": "two_arms_stack_bowl_12",
    },
    {
        "key": "two_arms_push_cube",
        "scene_config": "configs/table/two_arms_push_cube.yaml",
        "perturbation_config": "configs/perturbation/visual_perturbation_pushcube.yaml",
        "status": "two_arms_push_cube_12",
    },
    {
        "key": "two_arms_place_cube_in_bowl",
        "scene_config": "configs/table/two_arms_place_cube_in_bowl.yaml",
        "perturbation_config": "configs/perturbation/visual_perturbation.yaml",
        "status": "two_arms_place_cube_in_bowl",
    },
    {
        "key": "two_arms_handover_bottles",
        "scene_config": "configs/table/two_arms_handover_bottles.yaml",
        "perturbation_config": "configs/perturbation/visual_perturbation_handover_bottles.yaml",
        "status": "two_arms_handover_bottles_12",
    },
    {
        "key": "two_arms_place_burger_fries",
        "scene_config": "configs/table/two_arms_place_burger_fries.yaml",
        "perturbation_config": "configs/perturbation/visual_perturbation.yaml",
        "status": "two_arms_place_burger_fries_12",
    },
    {
        "key": "two_arms_place_can_basket",
        "scene_config": "configs/table/two_arms_place_can_basket.yaml",
        "perturbation_config": "configs/perturbation/visual_perturbation.yaml",
        "status": "two_arms_place_can_basket",
    },
    {
        "key": "two_arms_place_object_cabinet",
        "scene_config": "configs/table/two_arms_place_object_cabinet.yaml",
        "perturbation_config": "configs/perturbation/visual_perturbation_place_object_cabinet.yaml",
        "status": "two_arms_place_object_cabinet",
    },
    {
        "key": "two_arms_place_object_bowl_cabinet",
        "scene_config": "configs/table/two_arms_place_object_bowl_cabinet.yaml",
        "perturbation_config": "configs/perturbation/visual_perturbation_place_object_bowl_cabinet.yaml",
        "status": "two_arms_place_object_bowl_cabinet",
    },
    {
        "key": "three_arms_place_objects",
        "scene_config": "configs/table/three_arms_place_objects.yaml",
        "perturbation_config": "configs/perturbation/visual_perturbation_three_arms_place_objects.yaml",
        "status": "three_arms_place_objects_123_123",
    },
]


def parse_seed_range(seed_arg: str) -> list[int]:
    """Parse '1000-1099' or '1000,1001,1002' or '1000' into a list of ints."""
    seeds = []
    for part in seed_arg.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(part))
    return seeds


def _expand_yaml_envvars(text: str) -> str:
    """Expand ${VAR} or $VAR patterns in a YAML string using os.environ."""
    import re
    def replacer(m):
        return os.environ.get(m.group(1) or m.group(2), m.group(0))
    return re.sub(r'\$\{(\w+)\}|\$(\w+)', replacer, text)


def record_task(task_cfg: dict, seeds: list[int], out_dir: str, asset_bundle_manifest: str):
    key = task_cfg["key"]
    scene_config = task_cfg["scene_config"]
    perturbation_config = task_cfg["perturbation_config"]
    status = task_cfg.get("status")

    print(f"\n{'='*60}")
    print(f"Recording task: {key}  ({len(seeds)} seeds)")
    print(f"  scene:       {scene_config}")
    print(f"  perturbation: {perturbation_config}")
    print(f"{'='*60}")

    # Read the perturbation YAML (expand env vars for asset paths)
    with open(perturbation_config, "r") as f:
        raw = f.read()
    raw = _expand_yaml_envvars(raw)
    pcfg_data = yaml.safe_load(raw)
    domain_rand = pcfg_data.get("domain_randomization", {}) if pcfg_data else {}

    # Resolve the task env_id from the scene config
    with open(scene_config, "r") as f:
        scene_data = yaml.safe_load(f)
    env_id = scene_data["task_name"] + "-rf"

    writer = ManifestWriter(
        task_name=key,
        perturbation_config=perturbation_config,
        scene_config=scene_config,
        asset_bundle_manifest_path=asset_bundle_manifest,
    )

    env_kwargs = dict(
        config=scene_config,
        obs_mode="rgb",
        reward_mode=None,
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        sensor_configs=dict(shader_pack="default"),
        human_render_camera_configs=dict(shader_pack="default"),
        viewer_camera_configs=dict(shader_pack="default"),
        num_envs=1,
        sim_backend="cpu",
        enable_shadow=True,
        parallel_in_single_scene=False,
    )
    if domain_rand:
        env_kwargs["domain_randomization"] = domain_rand
    if status:
        env_kwargs["status"] = status

    # Create the env ONCE and reuse across all seeds to avoid GPU/rendering
    # context exhaustion that causes segfaults when creating/destroying per seed.
    try:
        env = gym.make(env_id, **env_kwargs)
    except Exception as exc:
        print(f"ERROR: failed to create env: {exc}")
        import traceback
        traceback.print_exc()
        return None

    try:
        for seed in seeds:
            print(f"  seed {seed} ... ", end="", flush=True)
            np.random.seed(seed)

            recorder = PerturbRecorder()
            try:
                # Attach recorder before the seeded reset so _build_background,
                # _build_distractors, _load_lighting and initialize() all run
                # with the recorder active. reconfigure=True forces the scene to
                # be fully rebuilt (not just _initialize_episode).
                env._perturb_recorder = recorder
                env.reset(seed=[seed], options={"reconfigure": True})
            except Exception as exc:
                print(f"ERROR: {exc}")
                import traceback
                traceback.print_exc()
                continue
            finally:
                env._perturb_recorder = None  # detach so next reset is clean

            writer.add_seed(seed, recorder.to_dict())
            print("ok")
    finally:
        env.close()

    out_path = os.path.join(out_dir, f"{key}_perturbed.json")
    writer.write(out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Record perturbation manifests for all (or selected) tasks."
    )
    parser.add_argument(
        "--seeds",
        default="1000-1099",
        help="Seed range, e.g. '1000-1099' or '1000,1002,1005' (default: 1000-1099)",
    )
    parser.add_argument(
        "--out-dir",
        default="release/manifests",
        help="Output directory for manifest JSON files (default: release/manifests)",
    )
    parser.add_argument(
        "--asset-bundle-manifest",
        default=None,
        help="Path to perturbation_assets_v*.manifest.json (from build_asset_bundle.py). "
             "If provided, its sha256 is embedded in each task manifest for integrity checking.",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="Record only this task key (e.g. 'two_arms_stack_cube'). "
             "Omit to record all tasks.",
    )
    args = parser.parse_args()

    seeds = parse_seed_range(args.seeds)
    os.makedirs(args.out_dir, exist_ok=True)

    task_list = TASKS
    if args.task:
        task_list = [t for t in TASKS if t["key"] == args.task]
        if not task_list:
            print(f"ERROR: unknown task key '{args.task}'. Known keys:")
            for t in TASKS:
                print(f"  {t['key']}")
            sys.exit(1)

    for task_cfg in task_list:
        if not os.path.exists(task_cfg["scene_config"]):
            print(f"SKIP {task_cfg['key']}: scene config not found: {task_cfg['scene_config']}")
            continue
        if not os.path.exists(task_cfg["perturbation_config"]):
            print(f"SKIP {task_cfg['key']}: perturbation config not found: {task_cfg['perturbation_config']}")
            continue
        record_task(task_cfg, seeds, args.out_dir, args.asset_bundle_manifest)

    print(f"\nDone. Manifests written to: {args.out_dir}/")


if __name__ == "__main__":
    main()

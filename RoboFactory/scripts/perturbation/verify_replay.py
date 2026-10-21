"""
Verify that replay mode on this machine reproduces the recorded perturbation.

For each seed in the manifest:
1. Loads the env in RECORD mode -> captures fresh perturbation state
2. Loads the env in REPLAY mode -> replays from manifest
3. Compares object poses, agent poses, distractor placement, lighting

A non-zero exit code means at least one seed failed verification.

Usage (run from the RoboFactory root):
    python scripts/perturbation/verify_replay.py \
        --task two_arms_stack_cube \
        --manifest release/manifests/two_arms_stack_cube_perturbed.json \
        --seeds 1000-1005 \
        [--asset-bundle-root /path/to/perturbation_assets_v1]

    # Or verify all tasks at once (requires manifests for each):
    python scripts/perturbation/verify_replay.py \
        --manifest-dir release/manifests \
        --seeds 1000-1005
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import gymnasium as gym
import numpy as np
import yaml

from tasks import *  # noqa: F401,F403

from utils.perturbation.recorder import PerturbRecorder
from utils.perturbation.manifest import ManifestReader, ManifestWriter
from scripts.perturbation.record_manifests import TASKS, parse_seed_range

POSE_TOL = 1e-4   # tolerance for position/quat comparison (metres / unit)
COLOR_TOL = 1e-5  # tolerance for lighting colour comparison


def _arr_close(a, b, tol=POSE_TOL) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    try:
        return float(np.max(np.abs(np.array(a, dtype=float) - np.array(b, dtype=float)))) <= tol
    except Exception:
        return False


def _task_cfg_for_key(key: str):
    for t in TASKS:
        if t["key"] == key:
            return t
    return None


def record_one_seed(task_cfg: dict, seed: int) -> dict:
    """Run one env.reset() in record mode and return the recorder data."""
    scene_config = task_cfg["scene_config"]
    perturbation_config = task_cfg["perturbation_config"]
    status = task_cfg.get("status")

    with open(perturbation_config, "r") as f:
        import re
        raw = f.read()
        raw = re.sub(r'\$\{(\w+)\}|\$(\w+)',
                     lambda m: os.environ.get(m.group(1) or m.group(2), m.group(0)), raw)
    pcfg_data = yaml.safe_load(raw)
    domain_rand = pcfg_data.get("domain_randomization", {}) if pcfg_data else {}

    with open(scene_config, "r") as f:
        scene_data = yaml.safe_load(f)
    env_id = scene_data["task_name"] + "-rf"

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

    np.random.seed(seed)
    recorder = PerturbRecorder()
    env = gym.make(env_id, **env_kwargs)
    env._perturb_recorder = recorder
    env.reset(seed=[seed])
    env.close()
    del env
    return recorder.to_dict()


def replay_one_seed(task_cfg: dict, seed: int, reader: ManifestReader) -> dict:
    """Run one env.reset() in replay mode and return the recorder data (for comparison)."""
    scene_config = task_cfg["scene_config"]
    perturbation_config = task_cfg["perturbation_config"]
    status = task_cfg.get("status")

    with open(perturbation_config, "r") as f:
        import re
        raw = f.read()
        raw = re.sub(r'\$\{(\w+)\}|\$(\w+)',
                     lambda m: os.environ.get(m.group(1) or m.group(2), m.group(0)), raw)
    pcfg_data = yaml.safe_load(raw)
    domain_rand = pcfg_data.get("domain_randomization", {}) if pcfg_data else {}

    with open(scene_config, "r") as f:
        scene_data = yaml.safe_load(f)
    env_id = scene_data["task_name"] + "-rf"

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

    replayer = reader.get_replayer(seed)
    # Also attach a recorder to capture what was actually applied
    recorder = PerturbRecorder()

    env = gym.make(env_id, **env_kwargs)
    env._perturb_replayer = replayer
    env._perturb_recorder = recorder
    env.reset(seed=[seed])
    env.close()
    del env
    return recorder.to_dict()


def compare(recorded: dict, replayed: dict, seed: int) -> list[str]:
    """Return a list of failure messages (empty = pass)."""
    failures = []

    # Background
    bg_rec = recorded.get("background", {})
    bg_rep = replayed.get("background", {})
    if bg_rec.get("use_clean") != bg_rep.get("use_clean"):
        failures.append(
            f"seed {seed}: background use_clean mismatch: "
            f"recorded={bg_rec.get('use_clean')} replay={bg_rep.get('use_clean')}"
        )
    if not bg_rec.get("use_clean"):
        if bg_rec.get("texture_relpath") != bg_rep.get("texture_relpath"):
            failures.append(
                f"seed {seed}: background texture mismatch: "
                f"recorded={bg_rec.get('texture_relpath')} replay={bg_rep.get('texture_relpath')}"
            )

    # Distractor placement
    dp_rec = recorded.get("distractor_placement", [])
    dp_rep = replayed.get("distractor_placement", [])
    if len(dp_rec) != len(dp_rep):
        failures.append(
            f"seed {seed}: distractor count mismatch: recorded={len(dp_rec)} replay={len(dp_rep)}"
        )
    else:
        for i, (r, p) in enumerate(zip(dp_rec, dp_rep)):
            for field in ("x", "y", "z", "yaw"):
                if not _arr_close([r.get(field, 0)], [p.get(field, 0)]):
                    failures.append(
                        f"seed {seed}: distractor[{i}].{field} mismatch: "
                        f"recorded={r.get(field)} replay={p.get(field)}"
                    )
            if r.get("fallback") != p.get("fallback"):
                failures.append(
                    f"seed {seed}: distractor[{i}].fallback mismatch: "
                    f"recorded={r.get('fallback')} replay={p.get('fallback')}"
                )

    # Movable object poses
    poses_rec = recorded.get("movable_object_poses", {})
    poses_rep = replayed.get("movable_object_poses", {})
    for name in set(list(poses_rec.keys()) + list(poses_rep.keys())):
        pr = poses_rec.get(name, {})
        pp = poses_rep.get(name, {})
        for field in ("p", "q"):
            if not _arr_close(pr.get(field), pp.get(field)):
                failures.append(
                    f"seed {seed}: {name}.{field} pose mismatch: "
                    f"recorded={pr.get(field)} replay={pp.get(field)}"
                )

    # Agent poses
    ap_rec = recorded.get("agent_base_poses", {})
    ap_rep = replayed.get("agent_base_poses", {})
    for key in set(list(ap_rec.keys()) + list(ap_rep.keys())):
        pr = ap_rec.get(key, {})
        pp = ap_rep.get(key, {})
        for field in ("p", "q"):
            if not _arr_close(pr.get(field), pp.get(field)):
                failures.append(
                    f"seed {seed}: agent {key}.{field} pose mismatch: "
                    f"recorded={pr.get(field)} replay={pp.get(field)}"
                )

    # Lighting
    lt_rec = recorded.get("lighting", {})
    lt_rep = replayed.get("lighting", {})
    if lt_rec.get("enabled") != lt_rep.get("enabled"):
        failures.append(
            f"seed {seed}: lighting.enabled mismatch: "
            f"recorded={lt_rec.get('enabled')} replay={lt_rep.get('enabled')}"
        )
    if lt_rec.get("enabled"):
        for field in ("ambient", "dir1_color", "dir2_color"):
            if not _arr_close(lt_rec.get(field), lt_rep.get(field), tol=COLOR_TOL):
                failures.append(
                    f"seed {seed}: lighting.{field} mismatch: "
                    f"recorded={lt_rec.get(field)} replay={lt_rep.get(field)}"
                )

    return failures


def verify_task(task_key: str, manifest_path: str, seeds: list[int], asset_bundle_root: str | None) -> bool:
    task_cfg = _task_cfg_for_key(task_key)
    if task_cfg is None:
        print(f"ERROR: unknown task key '{task_key}'")
        return False
    if not os.path.exists(manifest_path):
        print(f"ERROR: manifest not found: {manifest_path}")
        return False

    reader = ManifestReader(manifest_path, asset_bundle_root=asset_bundle_root)
    print(f"\nVerifying task '{task_key}' with {len(seeds)} seeds ...")

    all_pass = True
    for seed in seeds:
        if not reader.has_seed(seed):
            print(f"  seed {seed}: SKIP (not in manifest)")
            continue
        try:
            rec_data = record_one_seed(task_cfg, seed)
            rep_data = replay_one_seed(task_cfg, seed, reader)
            failures = compare(rec_data, rep_data, seed)
            if failures:
                all_pass = False
                for msg in failures:
                    print(f"  FAIL: {msg}")
            else:
                print(f"  seed {seed}: PASS")
        except Exception as exc:
            print(f"  seed {seed}: ERROR: {exc}")
            import traceback
            traceback.print_exc()
            all_pass = False

    return all_pass


def main():
    parser = argparse.ArgumentParser(description="Verify perturbation replay against recorded manifests.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--manifest", help="Path to a single task manifest JSON.")
    group.add_argument("--manifest-dir", help="Directory containing all task manifest JSONs.")
    parser.add_argument("--task", help="Task key (required when --manifest is used).")
    parser.add_argument("--seeds", default="1000-1005",
                        help="Seeds to verify (default: 1000-1005).")
    parser.add_argument("--asset-bundle-root", default=None,
                        help="Root of extracted asset bundle. Overrides ROBOFACTORY_PERTURB_ASSETS.")
    args = parser.parse_args()

    seeds = parse_seed_range(args.seeds)
    asset_root = args.asset_bundle_root or os.environ.get("ROBOFACTORY_PERTURB_ASSETS")

    all_pass = True

    if args.manifest:
        if not args.task:
            parser.error("--task is required when --manifest is used")
        ok = verify_task(args.task, args.manifest, seeds, asset_root)
        all_pass = all_pass and ok
    else:
        mdir = args.manifest_dir
        for t in TASKS:
            manifest_path = os.path.join(mdir, f"{t['key']}_perturbed.json")
            if not os.path.exists(manifest_path):
                print(f"SKIP {t['key']}: no manifest at {manifest_path}")
                continue
            ok = verify_task(t["key"], manifest_path, seeds, asset_root)
            all_pass = all_pass and ok

    if all_pass:
        print("\nAll seeds passed verification.")
        sys.exit(0)
    else:
        print("\nSome seeds FAILED verification. See messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

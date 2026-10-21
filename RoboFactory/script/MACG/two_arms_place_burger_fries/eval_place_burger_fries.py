import sys
sys.path.append('./')
sys.path.insert(0, './policy/Diffusion-Policy')

import os
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional, Annotated, Union

import numpy as np
import yaml

import gymnasium as gym
import sapien

from mani_skill.envs.sapien_env import BaseEnv
from utils.wrappers.record import RecordEpisodeMA

import tyro

from planner.motionplanner import PandaArmMotionPlanningSolver
from client import *
from tasks import *

from script.MACG.eval_common.eval_hooks import (
    build_progress_tracker,
    emit_metrics_line,
    finalize_episode_metrics,
    sample_progress,
)

# ==========================
# Server configuration
# ==========================
SERVER_URL = os.getenv("VLA_SERVER_URL", "http://localhost:20020")
client = OpenPIClient(SERVER_URL)
print("=== Checking unified model server status ===")
try:
    health = client.health_check()
    print(f"Server health: {health}")
except Exception as e:
    print(f"Failed to connect to server at {SERVER_URL}: {e}")


def _to_bool_flag(value) -> bool:
    """Best-effort convert tensor/ndarray/scalar into Python bool (first element if array)."""
    try:
        import torch
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return False
            return bool(value.reshape(-1)[0].item())
    except Exception:
        pass
    try:
        arr = np.array(value).reshape(-1)
        if arr.size == 0:
            return False
        return bool(arr[0])
    except Exception:
        return bool(value)


class RuleBasedPhaseDetector:
    """Runtime phase detector driven by task contact-based grasp states."""

    CONDITION_HANDLERS = {
        "agent0_grip",
        "agent0_release",
        "agent1_grip",
        "agent1_release",
        "both_grip",
        "any_grip",
    }

    def __init__(self, transitions, initial_phase, phase_instructions_dict,
                 gripper_threshold=0.035, min_steps_per_phase=2, verbose=False):
        self.transitions = transitions
        self.current_phase = int(initial_phase)
        self.phase_instructions_dict = phase_instructions_dict
        self.gripper_threshold = gripper_threshold
        self.min_steps = min_steps_per_phase
        self.verbose = verbose

        self.steps_in_phase = 0
        self.prev_agent0_grip = False
        self.prev_agent1_grip = False

        # Pending delayed transition: (target_phase, remaining_delay)
        self._pending_transition = None
        self._latched_conditions = set()

    def _gripper_closed(self, qpos) -> bool:
        return float(qpos[-1]) < self.gripper_threshold

    def _check_condition(self, condition: str,
                         a0_grip: bool, a1_grip: bool) -> bool:
        if condition == "agent0_grip":
            return (not self.prev_agent0_grip) and a0_grip
        if condition == "agent0_release":
            return self.prev_agent0_grip and (not a0_grip)
        if condition == "agent1_grip":
            return (not self.prev_agent1_grip) and a1_grip
        if condition == "agent1_release":
            return self.prev_agent1_grip and (not a1_grip)
        if condition == "both_grip":
            return a0_grip and a1_grip
        if condition == "any_grip":
            return a0_grip or a1_grip
        raise ValueError(f"Unknown transition condition: {condition}")

    def update(self, grasp_info) -> int:
        """Observe new contact grasp states and return the current phase.

        Supports per-transition ``delay``: when a condition fires and
        ``delay > 0``, the detector keeps returning the *current* phase for
        ``delay`` more steps before actually switching.
        """
        a0_grip = _to_bool_flag(grasp_info["is_burger_grasped"])
        a1_grip = _to_bool_flag(grasp_info["is_fries_grasped"])
        self.steps_in_phase += 1

        # If a delayed transition is pending, count down
        if self._pending_transition is not None:
            target_phase, remaining = self._pending_transition
            remaining -= 1
            if remaining <= 0:
                old_phase = self.current_phase
                self.current_phase = target_phase
                self.steps_in_phase = 0
                self._pending_transition = None
                self._latched_conditions.clear()
                if self.verbose:
                    print(f"[RuleDetector] Phase {old_phase} -> {self.current_phase} "
                          f"(delayed transition complete)")
            else:
                self._pending_transition = (target_phase, remaining)
        else:
            for t in self.transitions:
                if int(t["from"]) != self.current_phase:
                    continue
                condition = t["condition"]
                if self._check_condition(condition, a0_grip, a1_grip):
                    self._latched_conditions.add(condition)
                min_req = int(t.get("min_steps", self.min_steps))
                if self.steps_in_phase < min_req:
                    continue
                if condition in self._latched_conditions:
                    delay = int(t.get("delay", 0))
                    if delay > 0:
                        self._pending_transition = (int(t["to"]), delay)
                        if self.verbose:
                            print(f"[RuleDetector] Phase {self.current_phase}: condition "
                                  f"{t['condition']} fired, delaying {delay} step(s)")
                    else:
                        old_phase = self.current_phase
                        self.current_phase = int(t["to"])
                        self.steps_in_phase = 0
                        self._latched_conditions.clear()
                        if self.verbose:
                            print(f"[RuleDetector] Phase {old_phase} -> {self.current_phase} "
                                  f"(condition={condition}, a0_grip={a0_grip}, a1_grip={a1_grip})")
                    break

        self.prev_agent0_grip = a0_grip
        self.prev_agent1_grip = a1_grip
        return self.current_phase


@dataclass
class Args:
    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = ""
    """The environment ID of the task you want to simulate"""

    config: str = "configs/table/two_arms_place_burger_fries.yaml"
    """Configuration to build scenes, assets and agents."""

    obs_mode: Annotated[str, tyro.conf.arg(aliases=["-o"])] = "rgb"
    """Observation mode"""

    robot_uids: Annotated[Optional[str], tyro.conf.arg(aliases=["-r"])] = None
    """Robot UID(s) to use. Can be a comma separated list of UIDs or empty string to have no agents. If not given then defaults to the environments default robot"""

    sim_backend: Annotated[str, tyro.conf.arg(aliases=["-b"])] = "auto"
    """Which simulation backend to use. Can be 'auto', 'cpu', 'gpu'"""

    reward_mode: Optional[str] = None
    """Reward mode"""

    num_envs: Annotated[int, tyro.conf.arg(aliases=["-n"])] = 1
    """Number of environments to run."""

    control_mode: Annotated[Optional[str], tyro.conf.arg(aliases=["-c"])] = "pd_joint_pos"
    """Control mode"""

    render_mode: str = "rgb_array"
    """Render mode"""

    shader: str = "default"
    """Change shader used for all cameras in the environment for rendering."""

    record_dir: Optional[str] = './eval_video/{env_id}'
    """Directory to save recordings"""

    pause: Annotated[bool, tyro.conf.arg(aliases=["-p"])] = False
    """If using human render mode, auto pauses the simulation upon loading"""

    quiet: bool = False
    """Disable verbose output."""

    seed: Annotated[Optional[Union[int, List[int]]], tyro.conf.arg(aliases=["-s"])] = 10000
    """Seed(s) for random actions and simulator. Can be a single integer or a list of integers."""

    data_num: int = 100
    """The number of episode data used for training the policy (metadata only, not used here)"""

    checkpoint_num: int = 300
    """The number of training epoch of the checkpoint (metadata only, not used here)"""

    max_steps: int = 40
    """Maximum number of high-level steps (each containing up to 50 substeps)"""

    timeout_s: float = 120.0
    """Wall-clock timeout in seconds; declare task failed when exceeded."""

    prompt: str = 'place the burger and fries on the tray'
    """Text prompt sent to the unified model"""

    status: Optional[str] = "two_arms_place_burger_fries_12"
    """Status/task type for the environment (e.g., 'two_arms_place_burger_fries_12')"""

    phase_config: Optional[str] = None
    """Path to phase configuration file (YAML/JSON). If provided, overrides phase_schedule and uses custom phase_instructions."""

    phase_schedule: str = "1-2:1,3-6:2,7-9:3,10-13:4,14+:5"
    """Phase schedule in format 'step:phase,step:phase' or 'step-step:phase'. Use '+' for all remaining steps. Ignored if phase_config is provided."""

    perturbation_config: Optional[str] = None
    """Path to a YAML file with domain_randomization settings for visual perturbation during evaluation."""

    replay_manifest: Optional[str] = None
    """Path to a per-task manifest JSON for bit-exact perturbation replay on any machine.
    Generate once with scripts/perturbation/record_manifests.py. Set ROBOFACTORY_PERTURB_ASSETS
    env var to the extracted asset bundle root on the target machine."""

    record_camera: Optional[str] = "head_camera_global,wrist_camera_agent0,wrist_camera_agent1"
    """Comma-separated camera names to record in video (concatenated horizontally)."""


# Default phase instructions for two arms place burger fries task (in-domain ordering)
phase_instructions = {
    1: {
        "agent0": "Pick up the burger.",
        "agent1": "Wait for the other robot."
    },
    2: {
        "agent0": "Place the burger on the tray.",
        "agent1": "Wait for the other robot."
    },
    3: {
        "agent0": "Recover.",
        "agent1": "Pick up the fries."
    },
    4: {
        "agent0": "Wait for the other robot.",
        "agent1": "Place the fries on the tray."
    },
    5: {
        "agent0": "Wait for the other robot.",
        "agent1": "Recover."
    }
}


def load_phase_config(config_path: str) -> dict:
    """Load phase configuration from YAML or JSON file.

    Returns a dict with keys:
        phase_instructions  – dict[int, dict[str, str]]
        phase_schedule      – str  (fallback timestep schedule)
        mode                – "rule_based" | "timestep" (default)
        transitions         – list[dict] | None
        initial_phase       – int
        gripper_threshold   – float
        min_steps_per_phase – int
    """
    import json

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Phase config file not found: {config_path}")

    if config_path.endswith('.json'):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

    phase_instr_raw = config.get('phase_instructions', {})
    phase_instructions_int = {int(k): v for k, v in phase_instr_raw.items()}

    phase_schedule = config.get('phase_schedule', "1:1,2+:1")
    if isinstance(phase_schedule, dict):
        schedule_parts = [f"{sr}:{ph}" for sr, ph in phase_schedule.items()]
        phase_schedule = ",".join(schedule_parts)

    return {
        "phase_instructions": phase_instructions_int,
        "phase_schedule": phase_schedule,
        "mode": config.get("mode", "timestep"),
        "transitions": config.get("transitions", None),
        "initial_phase": int(config.get("initial_phase", 1)),
        "gripper_threshold": float(config.get("gripper_threshold", 0.035)),
        "min_steps_per_phase": int(config.get("min_steps_per_phase", 2)),
    }


def parse_phase_schedule(schedule_str: str) -> dict:
    """
    Parse phase schedule string into a mapping function.
    Format: 'step:phase,step:phase' or 'step-step:phase'
    Example: '1-2:1,3-6:2,7-9:3,10-13:4,14+:5'
    Returns a dict with step -> phase mapping and a default phase for remaining steps.
    """
    schedule = {}
    default_phase = 1

    parts = schedule_str.split(',')
    for part in parts:
        part = part.strip()
        if ':' not in part:
            continue

        step_part, phase_str = part.split(':')
        phase = int(phase_str)

        if '+' in step_part:
            step_num = int(step_part.replace('+', ''))
            default_phase = phase
            schedule['default'] = (step_num, phase)
        elif '-' in step_part:
            start, end = map(int, step_part.split('-'))
            for step in range(start, end + 1):
                schedule[step] = phase
        else:
            step = int(step_part)
            schedule[step] = phase

    return schedule


def get_phase_for_step(step: int, schedule: dict) -> int:
    """Get phase for a given step based on schedule."""
    if step in schedule:
        return schedule[step]
    elif 'default' in schedule:
        threshold, phase = schedule['default']
        if step >= threshold:
            return phase
    return 1  # Fallback to phase 1


def build_prompt_for_phase(phase: int, phase_instructions_dict: dict) -> str:
    """Build prompt string from phase instructions dictionary.

    Output format: "instruction1. | instruction2."
    Example: "Pick up the burger. | Wait for the other robot."
    """
    phase = int(phase)
    if phase not in phase_instructions_dict:
        phase = 1  # Default to phase 1
    ins = phase_instructions_dict[phase]

    prompt_parts = []
    for agent_key in sorted(ins.keys()):
        prompt_parts.append(ins[agent_key])

    return " | ".join(prompt_parts)


def get_unified_model_input(observation, agent0_pos, agent1_pos, prompt: str):
    """Get unified model input for two agents."""
    import cv2

    cam0 = 'wrist_camera_agent0'
    cam1 = 'wrist_camera_agent1'

    img0 = observation['sensor_data'][cam0]['rgb'].squeeze(0).numpy()
    img1 = observation['sensor_data'][cam1]['rgb'].squeeze(0).numpy()
    global_img = observation['sensor_data']['head_camera_global']['rgb'].squeeze(0).numpy()

    timestamp = int(time.time())
    save_dir = "./tmp/eval_two_arms_place_burger_fries"
    try:
        os.makedirs(save_dir, exist_ok=True)
        img0_bgr = cv2.cvtColor(img0, cv2.COLOR_RGB2BGR)
        img1_bgr = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
        global_bgr = cv2.cvtColor(global_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(save_dir, f"unified_agent0_{timestamp}.jpg"), img0_bgr)
        cv2.imwrite(os.path.join(save_dir, f"unified_agent1_{timestamp}.jpg"), img1_bgr)
        cv2.imwrite(os.path.join(save_dir, f"unified_global_{timestamp}.jpg"), global_bgr)
    except Exception:
        pass

    # 7-dim joints + scalar gripper per agent (2 agents total)
    joint_position_0 = agent0_pos[:7]
    gripper_position_0 = float(agent0_pos[7])

    joint_position_1 = agent1_pos[:7]
    gripper_position_1 = float(agent1_pos[7])

    return {
        'image_agent0': img0,
        'image_agent1': img1,
        'global_image': global_img,
        'joint_position_agent0': joint_position_0,
        'gripper_position_agent0': gripper_position_0,
        'joint_position_agent1': joint_position_1,
        'gripper_position_agent1': gripper_position_1,
        'task': prompt,
    }


def get_unified_model_answer(example):
    result = client.infer(example)
    return result


def main(args: Args):
    np.set_printoptions(suppress=True, precision=5)
    verbose = not args.quiet

    if isinstance(args.seed, int):
        args.seed = [args.seed]
    if args.seed is not None:
        np.random.seed(args.seed[0])

    parallel_in_single_scene = args.render_mode == "human"
    if args.render_mode == "human" and args.obs_mode in ["sensor_data", "rgb", "rgbd", "depth", "point_cloud"]:
        print("Disabling parallel single scene/GUI render as observation mode is a visual one. Change observation mode to state or state_dict to see a parallel env render")
        parallel_in_single_scene = False
    if args.render_mode == "human" and args.num_envs == 1:
        parallel_in_single_scene = False

    env_id = args.env_id
    if env_id == "":
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
            env_id = config['task_name'] + '-rf'

    env_kwargs = dict(
        config=args.config,
        obs_mode=args.obs_mode,
        reward_mode=args.reward_mode,
        control_mode=args.control_mode,
        render_mode=args.render_mode,
        sensor_configs=dict(shader_pack=args.shader),
        human_render_camera_configs=dict(shader_pack=args.shader),
        viewer_camera_configs=dict(shader_pack=args.shader),
        num_envs=args.num_envs,
        sim_backend=args.sim_backend,
        enable_shadow=True,
        parallel_in_single_scene=parallel_in_single_scene,
    )
    if args.status is not None:
        env_kwargs["status"] = args.status
    if args.perturbation_config is not None:
        with open(args.perturbation_config, "r") as _pf:
            _pcfg = yaml.safe_load(_pf)
        if _pcfg and "domain_randomization" in _pcfg:
            env_kwargs["domain_randomization"] = _pcfg["domain_randomization"]
    if args.robot_uids is not None:
        env_kwargs["robot_uids"] = tuple(args.robot_uids.split(","))

    env: BaseEnv = gym.make(env_id, **env_kwargs)

    if args.replay_manifest is not None:
        import os as _os
        from utils.perturbation import ManifestReader as _ManifestReader
        _seed_val = args.seed[0] if isinstance(args.seed, list) else (args.seed or 1000)
        _asset_root = _os.environ.get('ROBOFACTORY_PERTURB_ASSETS')
        _reader = _ManifestReader(args.replay_manifest, asset_bundle_root=_asset_root)
        _replayer = _reader.get_replayer(_seed_val)
        if _replayer is not None:
            env._perturb_replayer = _replayer
        else:
            print(f"[replay] WARNING: seed {_seed_val} not in {args.replay_manifest}, running without replay.")

    record_dir = args.record_dir + '/' + str(args.seed) + '_' + str(args.data_num) + '_' + str(args.checkpoint_num)
    if record_dir:
        record_dir = record_dir.format(env_id=env_id)
        env = RecordEpisodeMA(env, record_dir, info_on_video=False, save_trajectory=False, max_steps_per_video=30000, record_camera=args.record_camera)

    raw_obs, _ = env.reset(seed=args.seed)

    progress_tracker = build_progress_tracker(
        env_id=env_id,
        status=args.status,
        phase_config=args.phase_config,
        control_freq=getattr(env.unwrapped, "control_freq", None),
    )

    planner = PandaArmMotionPlanningSolver(
        env,
        debug=False,
        vis=verbose,
        base_pose=[agent.robot.pose for agent in env.agent.agents],
        visualize_target_grasp_pose=verbose,
        print_env_info=False,
        is_multi_agent=True
    )

    agent_num = planner.agent_num
    if agent_num < 2:
        raise RuntimeError(f"Two arms evaluation expects at least 2 agents, found {agent_num}")
    if agent_num > 2 and verbose:
        print(f"Warning: found {agent_num} agents. Only the first two agents will be controlled by the 16-DoF policy.")

    if args.seed is not None and env.action_space is not None:
        env.action_space.seed(args.seed[0])

    if args.render_mode is not None:
        viewer = env.render()
        if isinstance(viewer, sapien.utils.Viewer):
            viewer.paused = args.pause
        env.render()

    # Build initial unified input for two agents
    init_qpos_0 = raw_obs['agent']['panda-0']['qpos'].squeeze(0)[:-2].numpy()
    init_qpos_0 = np.append(init_qpos_0, planner.gripper_state[0])

    init_qpos_1 = raw_obs['agent']['panda-1']['qpos'].squeeze(0)[:-2].numpy()
    init_qpos_1 = np.append(init_qpos_1, planner.gripper_state[1])

    # Load phase configuration
    phase_detector = None  # will be set if mode == "rule_based"
    if args.phase_config is not None:
        if verbose:
            print(f"Loading phase config from: {args.phase_config}")
        pcfg = load_phase_config(args.phase_config)
        phase_instructions_dict = pcfg["phase_instructions"]
        phase_schedule_str = pcfg["phase_schedule"]
        if verbose:
            print(f"Loaded {len(phase_instructions_dict)} phase instructions")
            print(f"Mode: {pcfg['mode']}")

        if pcfg["mode"] == "rule_based" and pcfg["transitions"]:
            phase_detector = RuleBasedPhaseDetector(
                transitions=pcfg["transitions"],
                initial_phase=pcfg["initial_phase"],
                phase_instructions_dict=phase_instructions_dict,
                gripper_threshold=pcfg["gripper_threshold"],
                min_steps_per_phase=pcfg["min_steps_per_phase"],
                verbose=verbose,
            )
            if verbose:
                print(f"Rule-based detector active: initial_phase={pcfg['initial_phase']}, "
                      f"threshold={pcfg['gripper_threshold']}, "
                      f"min_steps={pcfg['min_steps_per_phase']}, "
                      f"transitions={len(pcfg['transitions'])}")
        else:
            if verbose:
                print(f"Phase schedule (timestep): {phase_schedule_str}")
    else:
        phase_instructions_dict = phase_instructions
        phase_schedule_str = args.phase_schedule
        if verbose:
            print(f"Using default phase configuration")
            print(f"Phase schedule: {phase_schedule_str}")

    # Timestep-based schedule (used as fallback when detector is None)
    phase_schedule = parse_phase_schedule(phase_schedule_str)
    if verbose and phase_detector is None:
        print(f"Parsed schedule: {phase_schedule}")

    # Initial prompt
    if phase_detector is not None:
        initial_phase = phase_detector.current_phase
    else:
        initial_phase = get_phase_for_step(1, phase_schedule)
    initial_prompt = build_prompt_for_phase(initial_phase, phase_instructions_dict)
    if verbose:
        print(f"Initial phase: {initial_phase}, prompt: {initial_prompt}")

    unified_example = get_unified_model_input(raw_obs, init_qpos_0, init_qpos_1, initial_prompt)

    start_time = time.time()

    cnt = 0
    while True:
        if verbose:
            print("Iteration:", cnt)
        cnt += 1
        if cnt > args.max_steps:
            break
        # Timeout failure check
        if (time.time() - start_time) > args.timeout_s:
            if verbose:
                print(f"Timeout reached: {args.timeout_s}s. Failing.")
            break

        action_dict = defaultdict(list)
        action_step_dict = defaultdict(list)

        # Query unified model once for both agents
        infer_result = get_unified_model_answer(unified_example)
        action_array = np.asarray(infer_result['actions'])
        if action_array.ndim == 1:
            action_array = action_array.reshape(1, -1)
        if action_array.shape[1] >= 16:  # 2 agents × 8 DoF = 16
            action_array = action_array[:, :16]
        else:
            raise AssertionError(f"Expected at least 16 dims, got {action_array.shape}")

        # For each high-level step, compute time-parametrized paths for both agents
        for i in range(50):
            now_action16 = action_array[i]

            # Split for each agent: 8 DoF per agent (7 joints + 1 gripper)
            now0 = now_action16[:8]
            now1 = now_action16[8:16]

            raw_obs = env.get_obs()

            # Agent 0 current qpos
            if i == 0:
                current_qpos_0 = raw_obs['agent']['panda-0']['qpos'].squeeze(0)[:-2].numpy()
            else:
                current_qpos_0 = action_array[i - 1][:7]

            # Agent 1 current qpos
            if i == 0:
                current_qpos_1 = raw_obs['agent']['panda-1']['qpos'].squeeze(0)[:-2].numpy()
            else:
                current_qpos_1 = action_array[i - 1][8:15]

            # Plan for agent 0
            path0 = np.vstack((current_qpos_0, now0[:-1]))
            try:
                times0, pos0, v0, a0, dur0 = planner.planner[0].TOPP(path0, 0.05, verbose=True)
            except Exception as e:
                print(f"Planner error (agent 0): {e}")
                action_now0 = np.hstack([now0[:-1], now0[-1]])
                action_dict['panda-0'].append(action_now0)
                action_step_dict['panda-0'].append(1)
                pos0 = np.empty((0, 7))

            # Plan for agent 1
            path1 = np.vstack((current_qpos_1, now1[:-1]))
            try:
                times1, pos1, v1, a1, dur1 = planner.planner[1].TOPP(path1, 0.05, verbose=True)
            except Exception as e:
                print(f"Planner error (agent 1): {e}")
                action_now1 = np.hstack([now1[:-1], now1[-1]])
                action_dict['panda-1'].append(action_now1)
                action_step_dict['panda-1'].append(1)
                pos1 = np.empty((0, 7))

            n_step0 = pos0.shape[0] if isinstance(pos0, np.ndarray) else 0
            n_step1 = pos1.shape[0] if isinstance(pos1, np.ndarray) else 0

            if n_step0 == 0:
                action_now0 = np.hstack([now0[:-1], now0[-1]])
                action_dict['panda-0'].append(action_now0)
                action_step_dict['panda-0'].append(1)
            else:
                action_step_dict['panda-0'].append(n_step0)
                for j in range(n_step0):
                    true_action0 = np.hstack([pos0[j], now0[-1]])
                    action_dict['panda-0'].append(true_action0)

            if n_step1 == 0:
                action_now1 = np.hstack([now1[:-1], now1[-1]])
                action_dict['panda-1'].append(action_now1)
                action_step_dict['panda-1'].append(1)
            else:
                action_step_dict['panda-1'].append(n_step1)
                for j in range(n_step1):
                    true_action1 = np.hstack([pos1[j], now1[-1]])
                    action_dict['panda-1'].append(true_action1)

        # Execute synchronized steps for two agents
        start_idx = [0, 0]
        for i in range(50):
            max_step = max(action_step_dict['panda-0'][i],
                          action_step_dict['panda-1'][i])
            for j in range(max_step):
                true_action = dict()
                now_step0 = min(j, action_step_dict['panda-0'][i] - 1)
                now_step1 = min(j, action_step_dict['panda-1'][i] - 1)
                true_action['panda-0'] = action_dict['panda-0'][start_idx[0] + now_step0]
                true_action['panda-1'] = action_dict['panda-1'][start_idx[1] + now_step1]
                observation, reward, terminated, truncated, info = env.step(true_action)
                sample_progress(progress_tracker, info, observation)
                if verbose and args.render_mode == "human":
                    env.render_human()
            if verbose:
                print(true_action)
                print("max_step", max_step)

            start_idx[0] += action_step_dict['panda-0'][i]
            start_idx[1] += action_step_dict['panda-1'][i]

        # Build next model input from latest observation and last commanded targets
        last0 = action_dict['panda-0'][start_idx[0] - 1]
        last1 = action_dict['panda-1'][start_idx[1] - 1]

        # Determine phase: rule-based (gripper state) or timestep-based
        if phase_detector is not None:
            prompt_phase = phase_detector.update(info)
        else:
            prompt_phase = get_phase_for_step(cnt, phase_schedule)
        next_prompt = build_prompt_for_phase(prompt_phase, phase_instructions_dict)
        if verbose:
            print(f"Step {cnt}, Phase {prompt_phase}: {next_prompt}")

        unified_example = get_unified_model_input(
            observation,
            last0,  # 8 dims (7 joints + 1 gripper)
            last1,
            next_prompt,
        )

        if verbose:
            print("info", info)
        if args.render_mode is not None:
            env.render()
        # Robust success detection: handle torch tensors / arrays
        success_flag = _to_bool_flag(info.get('success', False))
        # Fallback on terminated if success not provided
        try:
            success_flag = success_flag or _to_bool_flag(terminated)
        except Exception:
            pass
        if success_flag:
            emit_metrics_line(
                finalize_episode_metrics(
                    progress_tracker,
                    task_success=True,
                    seed=args.seed[0] if isinstance(args.seed, list) else args.seed,
                    env_id=env_id,
                    status=args.status,
                    phase_config=args.phase_config,
                    policy_steps=cnt,
                )
            )
            env.close()
            if record_dir:
                print(f"Saving video to {record_dir}")
            print("success")
            return

    emit_metrics_line(
        finalize_episode_metrics(
            progress_tracker,
            task_success=False,
            seed=args.seed[0] if isinstance(args.seed, list) else args.seed,
            env_id=env_id,
            status=args.status,
            phase_config=args.phase_config,
            policy_steps=cnt if "cnt" in locals() else None,
        )
    )
    env.close()
    if record_dir:
        print(f"Saving video to {record_dir}")
    print("failed")


if __name__ == "__main__":
    parsed_args = tyro.cli(Args)
    main(parsed_args)

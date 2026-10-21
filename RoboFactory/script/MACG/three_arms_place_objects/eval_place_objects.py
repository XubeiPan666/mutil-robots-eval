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
    """Runtime phase detector for 3-agent tasks based on contact grasp events.

    Transitions are defined in a YAML config (edge-triggered for per-agent
    grip/release, level-triggered for all_grip / all_release / any_grip /
    a01_grip / a01_release).
    """

    CONDITION_HANDLERS = {
        "agent0_grip",
        "agent0_release",
        "agent1_grip",
        "agent1_release",
        "agent2_grip",
        "agent2_release",
        "all_grip",
        "all_release",
        "any_grip",
        "a01_grip",
        "a01_release",
    }

    def __init__(self, transitions, initial_phase, phase_instructions_dict,
                 gripper_threshold=0.032, min_steps_per_phase=1, verbose=False):
        self.transitions = transitions
        self.current_phase = int(initial_phase)
        self.phase_instructions_dict = phase_instructions_dict
        self.gripper_threshold = gripper_threshold
        self.min_steps = min_steps_per_phase
        self.verbose = verbose

        self.steps_in_phase = 0
        self.prev_agent0_grip = False
        self.prev_agent1_grip = False
        self.prev_agent2_grip = False

        self._pending_transition = None
        self._latched_conditions = set()

    def _gripper_closed(self, qpos) -> bool:
        return float(qpos[-1]) < self.gripper_threshold

    def _check_condition(self, condition: str,
                         a0_grip: bool, a1_grip: bool, a2_grip: bool) -> bool:
        if condition == "agent0_grip":
            return (not self.prev_agent0_grip) and a0_grip
        if condition == "agent0_release":
            return self.prev_agent0_grip and (not a0_grip)
        if condition == "agent1_grip":
            return (not self.prev_agent1_grip) and a1_grip
        if condition == "agent1_release":
            return self.prev_agent1_grip and (not a1_grip)
        if condition == "agent2_grip":
            return (not self.prev_agent2_grip) and a2_grip
        if condition == "agent2_release":
            return self.prev_agent2_grip and (not a2_grip)
        if condition == "all_grip":
            return a0_grip and a1_grip and a2_grip
        if condition == "all_release":
            return (not a0_grip) and (not a1_grip) and (not a2_grip)
        if condition == "any_grip":
            return a0_grip or a1_grip or a2_grip
        if condition == "a01_grip":
            # Level-triggered: agent0 AND agent1 are both gripping (agent2 ignored)
            return a0_grip and a1_grip
        if condition == "a01_release":
            # Level-triggered: agent0 AND agent1 are both released (agent2 ignored)
            return (not a0_grip) and (not a1_grip)
        raise ValueError(f"Unknown transition condition: {condition}")

    def update(self, grasp_info) -> int:
        """Observe new contact grasp states and return the current phase.

        Supports per-transition ``delay``: when a condition fires and
        ``delay > 0``, the detector keeps returning the *current* phase for
        ``delay`` more steps before actually switching.
        """
        a0_grip = _to_bool_flag(grasp_info["is_can_grasped"])
        a1_grip = _to_bool_flag(grasp_info["is_teabox_grasped"])
        a2_grip = _to_bool_flag(grasp_info["is_hamburg_grasped"])
        self.steps_in_phase += 1

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
                if self._check_condition(condition, a0_grip, a1_grip, a2_grip):
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
                                  f"(condition={condition}, "
                                  f"a0={a0_grip}, a1={a1_grip}, a2={a2_grip})")
                    break

        self.prev_agent0_grip = a0_grip
        self.prev_agent1_grip = a1_grip
        self.prev_agent2_grip = a2_grip
        return self.current_phase


@dataclass
class Args:
    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = ""
    """The environment ID of the task you want to simulate"""

    config: str = "configs/table/three_arms_place_objects.yaml"
    """Configuration to build scenes, assets and agents."""

    obs_mode: Annotated[str, tyro.conf.arg(aliases=["-o"])] = "rgb"
    """Observation mode"""

    robot_uids: Annotated[Optional[str], tyro.conf.arg(aliases=["-r"])] = None
    """Robot UID(s) to use. Comma-separated list or empty string for no agents."""

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
    """Seed(s) for random actions and simulator."""

    data_num: int = 100
    """The number of episode data used for training the policy (metadata only)"""

    checkpoint_num: int = 300
    """The number of training epoch of the checkpoint (metadata only)"""

    max_steps: int = 50
    """Maximum number of high-level steps (each containing up to 50 substeps)"""

    timeout_s: float = 60.0
    """Wall-clock timeout in seconds; declare task failed when exceeded."""

    prompt: str = 'place objects in the tray'
    """Text prompt sent to the unified model (fallback when no phase_config)"""

    status: Optional[str] = "three_arms_place_objects_123_123"
    """Status/task type for the environment"""

    phase_config: Optional[str] = None
    """Path to phase configuration YAML. Enables rule-based phase detection."""

    phase_schedule: str = "1:1,2+:1"
    """Timestep-based fallback schedule. Ignored when phase_config is provided."""

    perturbation_config: Optional[str] = None
    """Path to a YAML file with domain_randomization settings for visual perturbation."""

    replay_manifest: Optional[str] = None
    """Path to a per-task manifest JSON for bit-exact perturbation replay on any machine.
    Generate once with scripts/perturbation/record_manifests.py. Set ROBOFACTORY_PERTURB_ASSETS
    env var to the extracted asset bundle root on the target machine."""

    record_camera: Optional[str] = (
        "head_camera_global,wrist_camera_agent0,wrist_camera_agent1,wrist_camera_agent2"
    )
    """Comma-separated camera names to record in video (concatenated horizontally)."""


def load_phase_config(config_path: str) -> dict:
    """Load phase configuration from a YAML file.

    Returns a dict with keys:
        phase_instructions  – dict[int, dict[str, str]]
        phase_schedule      – str  (fallback timestep schedule)
        mode                – "rule_based" | "timestep"
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
        "gripper_threshold": float(config.get("gripper_threshold", 0.032)),
        "min_steps_per_phase": int(config.get("min_steps_per_phase", 1)),
    }


def parse_phase_schedule(schedule_str: str) -> dict:
    """Parse phase schedule string into a step->phase mapping.

    Format: 'step:phase,step:phase' or 'step-step:phase'
    Example: '1-5:1,5-12:2,12-18:3,18-25:4,25-32:5,32-40:6,40+:7'
    """
    schedule = {}

    parts = schedule_str.split(',')
    for part in parts:
        part = part.strip()
        if ':' not in part:
            continue
        step_part, phase_str = part.split(':')
        phase = int(phase_str)

        if '+' in step_part:
            step_num = int(step_part.replace('+', ''))
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
    return 1


def build_prompt_for_phase(phase: int, phase_instructions_dict: dict) -> str:
    """Build the ' | '-separated prompt string for all agents in the given phase."""
    phase = int(phase)
    if phase not in phase_instructions_dict:
        phase = 1
    ins = phase_instructions_dict[phase]
    prompt_parts = [ins[k] for k in sorted(ins.keys())]
    return " | ".join(prompt_parts)


def get_unified_model_input(observation, agent0_pos, agent1_pos, agent2_pos, prompt: str):
    """Build the inference dict for the unified 3-agent model."""
    import cv2

    img0 = observation['sensor_data']['wrist_camera_agent0']['rgb'].squeeze(0).numpy()
    img1 = observation['sensor_data']['wrist_camera_agent1']['rgb'].squeeze(0).numpy()
    img2 = observation['sensor_data']['wrist_camera_agent2']['rgb'].squeeze(0).numpy()
    global_img = observation['sensor_data']['head_camera_global']['rgb'].squeeze(0).numpy()

    timestamp = int(time.time())
    save_dir = "/tmp/three_arms_place_objects"
    try:
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(os.path.join(save_dir, f"agent0_{timestamp}.jpg"),
                    cv2.cvtColor(img0, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(save_dir, f"agent1_{timestamp}.jpg"),
                    cv2.cvtColor(img1, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(save_dir, f"agent2_{timestamp}.jpg"),
                    cv2.cvtColor(img2, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(save_dir, f"global_{timestamp}.jpg"),
                    cv2.cvtColor(global_img, cv2.COLOR_RGB2BGR))
    except Exception:
        pass

    return {
        'image_agent0': img0,
        'image_agent1': img1,
        'image_agent2': img2,
        'global_image': global_img,
        'joint_position_agent0': agent0_pos[:7],
        'gripper_position_agent0': float(agent0_pos[7]),
        'joint_position_agent1': agent1_pos[:7],
        'gripper_position_agent1': float(agent1_pos[7]),
        'joint_position_agent2': agent2_pos[:7],
        'gripper_position_agent2': float(agent2_pos[7]),
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
        print("Disabling parallel single scene/GUI render for visual obs mode.")
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
        env = RecordEpisodeMA(
            env, record_dir,
            info_on_video=False, save_trajectory=False,
            max_steps_per_video=30000, record_camera=args.record_camera,
        )

    raw_obs, _ = env.reset(seed=args.seed)

    planner = PandaArmMotionPlanningSolver(
        env,
        debug=False,
        vis=verbose,
        base_pose=[agent.robot.pose for agent in env.agent.agents],
        visualize_target_grasp_pose=verbose,
        print_env_info=False,
        is_multi_agent=True,
    )

    agent_num = planner.agent_num
    if agent_num < 3:
        raise RuntimeError(f"Three arms evaluation expects at least 3 agents, found {agent_num}")

    if args.seed is not None and env.action_space is not None:
        env.action_space.seed(args.seed[0])

    if args.render_mode is not None:
        viewer = env.render()
        if isinstance(viewer, sapien.utils.Viewer):
            viewer.paused = args.pause
        env.render()

    # Build initial 8-DoF qpos for each agent (7 joints + 1 gripper)
    def _init_qpos(obs, panda_key, gripper_idx):
        q = obs['agent'][panda_key]['qpos'].squeeze(0)[:-2].numpy()
        return np.append(q, planner.gripper_state[gripper_idx])

    init_qpos_0 = _init_qpos(raw_obs, 'panda-0', 0)
    init_qpos_1 = _init_qpos(raw_obs, 'panda-1', 1)
    init_qpos_2 = _init_qpos(raw_obs, 'panda-2', 2)

    # Load phase configuration
    phase_detector = None
    if args.phase_config is not None:
        if verbose:
            print(f"Loading phase config from: {args.phase_config}")
        pcfg = load_phase_config(args.phase_config)
        phase_instructions_dict = pcfg["phase_instructions"]
        phase_schedule_str = pcfg["phase_schedule"]
        if verbose:
            print(f"Mode: {pcfg['mode']}, phases: {len(phase_instructions_dict)}")

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
        phase_instructions_dict = {
            1: {"agent0": args.prompt, "agent1": args.prompt, "agent2": args.prompt}
        }
        phase_schedule_str = args.phase_schedule
        if verbose:
            print(f"No phase config; using fixed prompt: '{args.prompt}'")

    phase_schedule = parse_phase_schedule(phase_schedule_str)

    if phase_detector is not None:
        initial_phase = phase_detector.current_phase
    else:
        initial_phase = get_phase_for_step(1, phase_schedule)
    initial_prompt = build_prompt_for_phase(initial_phase, phase_instructions_dict)
    if verbose:
        print(f"Initial phase: {initial_phase}, prompt: {initial_prompt}")

    unified_example = get_unified_model_input(
        raw_obs, init_qpos_0, init_qpos_1, init_qpos_2, initial_prompt
    )

    start_time = time.time()
    cnt = 0

    while True:
        if verbose:
            print("Iteration:", cnt)
        cnt += 1
        if cnt > args.max_steps:
            break
        if (time.time() - start_time) > args.timeout_s:
            if verbose:
                print(f"Timeout reached: {args.timeout_s}s. Failing.")
            break

        action_dict = defaultdict(list)
        action_step_dict = defaultdict(list)

        infer_result = get_unified_model_answer(unified_example)
        action_array = np.asarray(infer_result['actions'])
        if action_array.ndim == 1:
            action_array = action_array.reshape(1, -1)
        if action_array.shape[1] >= 24:
            action_array = action_array[:, :24]
        else:
            raise AssertionError(f"Expected at least 24 dims, got {action_array.shape}")

        for i in range(50):
            now_action24 = action_array[i]
            now0 = now_action24[:8]
            now1 = now_action24[8:16]
            now2 = now_action24[16:24]

            raw_obs = env.get_obs()

            if i == 0:
                current_qpos_0 = raw_obs['agent']['panda-0']['qpos'].squeeze(0)[:-2].numpy()
                current_qpos_1 = raw_obs['agent']['panda-1']['qpos'].squeeze(0)[:-2].numpy()
                current_qpos_2 = raw_obs['agent']['panda-2']['qpos'].squeeze(0)[:-2].numpy()
            else:
                current_qpos_0 = action_array[i - 1][:7]
                current_qpos_1 = action_array[i - 1][8:15]
                current_qpos_2 = action_array[i - 1][16:23]

            path0 = np.vstack((current_qpos_0, now0[:-1]))
            try:
                _, pos0, _, _, _ = planner.planner[0].TOPP(path0, 0.05, verbose=True)
            except Exception as e:
                print(f"Planner error (agent 0): {e}")
                action_dict['panda-0'].append(np.hstack([now0[:-1], now0[-1]]))
                action_step_dict['panda-0'].append(1)
                pos0 = np.empty((0, 7))

            path1 = np.vstack((current_qpos_1, now1[:-1]))
            try:
                _, pos1, _, _, _ = planner.planner[1].TOPP(path1, 0.05, verbose=True)
            except Exception as e:
                print(f"Planner error (agent 1): {e}")
                action_dict['panda-1'].append(np.hstack([now1[:-1], now1[-1]]))
                action_step_dict['panda-1'].append(1)
                pos1 = np.empty((0, 7))

            path2 = np.vstack((current_qpos_2, now2[:-1]))
            try:
                _, pos2, _, _, _ = planner.planner[2].TOPP(path2, 0.05, verbose=True)
            except Exception as e:
                print(f"Planner error (agent 2): {e}")
                action_dict['panda-2'].append(np.hstack([now2[:-1], now2[-1]]))
                action_step_dict['panda-2'].append(1)
                pos2 = np.empty((0, 7))

            n_step0 = pos0.shape[0] if isinstance(pos0, np.ndarray) else 0
            n_step1 = pos1.shape[0] if isinstance(pos1, np.ndarray) else 0
            n_step2 = pos2.shape[0] if isinstance(pos2, np.ndarray) else 0

            if n_step0 == 0:
                action_dict['panda-0'].append(np.hstack([now0[:-1], now0[-1]]))
                action_step_dict['panda-0'].append(1)
            else:
                action_step_dict['panda-0'].append(n_step0)
                for j in range(n_step0):
                    action_dict['panda-0'].append(np.hstack([pos0[j], now0[-1]]))

            if n_step1 == 0:
                action_dict['panda-1'].append(np.hstack([now1[:-1], now1[-1]]))
                action_step_dict['panda-1'].append(1)
            else:
                action_step_dict['panda-1'].append(n_step1)
                for j in range(n_step1):
                    action_dict['panda-1'].append(np.hstack([pos1[j], now1[-1]]))

            if n_step2 == 0:
                action_dict['panda-2'].append(np.hstack([now2[:-1], now2[-1]]))
                action_step_dict['panda-2'].append(1)
            else:
                action_step_dict['panda-2'].append(n_step2)
                for j in range(n_step2):
                    action_dict['panda-2'].append(np.hstack([pos2[j], now2[-1]]))

        # Execute synchronized substeps for all three agents
        start_idx = [0, 0, 0]
        for i in range(50):
            max_step = max(
                action_step_dict['panda-0'][i],
                action_step_dict['panda-1'][i],
                action_step_dict['panda-2'][i],
            )
            for j in range(max_step):
                now_step0 = min(j, action_step_dict['panda-0'][i] - 1)
                now_step1 = min(j, action_step_dict['panda-1'][i] - 1)
                now_step2 = min(j, action_step_dict['panda-2'][i] - 1)
                true_action = {
                    'panda-0': action_dict['panda-0'][start_idx[0] + now_step0],
                    'panda-1': action_dict['panda-1'][start_idx[1] + now_step1],
                    'panda-2': action_dict['panda-2'][start_idx[2] + now_step2],
                }
                observation, reward, terminated, truncated, info = env.step(true_action)
                if verbose and args.render_mode == "human":
                    env.render_human()
            if verbose:
                print("max_step", max_step)

            start_idx[0] += action_step_dict['panda-0'][i]
            start_idx[1] += action_step_dict['panda-1'][i]
            start_idx[2] += action_step_dict['panda-2'][i]

        last0 = action_dict['panda-0'][start_idx[0] - 1]
        last1 = action_dict['panda-1'][start_idx[1] - 1]
        last2 = action_dict['panda-2'][start_idx[2] - 1]

        # Update phase: rule-based (gripper state) or timestep-based fallback
        if phase_detector is not None:
            prompt_phase = phase_detector.update(info)
        else:
            prompt_phase = get_phase_for_step(cnt, phase_schedule)
        next_prompt = build_prompt_for_phase(prompt_phase, phase_instructions_dict)
        if verbose:
            print(f"Step {cnt}, Phase {prompt_phase}: {next_prompt}")

        unified_example = get_unified_model_input(
            observation, last0, last1, last2, next_prompt,
        )

        if verbose:
            print("info", info)
        if args.render_mode is not None:
            env.render()

        success_flag = _to_bool_flag(info.get('success', False))
        try:
            success_flag = success_flag or _to_bool_flag(terminated)
        except Exception:
            pass
        if success_flag:
            env.close()
            if record_dir:
                print(f"Saving video to {record_dir}")
            print("success")
            return

    env.close()
    if record_dir:
        print(f"Saving video to {record_dir}")
    print("failed")


if __name__ == "__main__":
    parsed_args = tyro.cli(Args)
    main(parsed_args)

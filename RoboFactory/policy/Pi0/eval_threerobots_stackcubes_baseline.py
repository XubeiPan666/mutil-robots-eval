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
SERVER_URL = os.getenv("OPENPI_SERVER_URL", "http://localhost:8000")
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
        import torch  # local import in case torch is not used elsewhere
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


@dataclass
class Args:
    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = ""
    """The environment ID of the task you want to simulate"""

    config: str = "configs/robocasa/take_photo.yaml"
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

    max_steps: int = 250
    """Maximum number of high-level steps (each containing up to 50 substeps)"""

    timeout_s: float = 70.0
    """Wall-clock timeout in seconds; declare task failed when exceeded."""

    prompt: str = 'Stack the cubes.'
    """Text prompt sent to the unified model"""

    use_rule_prompt: bool = False
    """Enable rule-based prompt switching based on gripper state phases"""

    close_threshold: float = 0.015
    """Gripper summed finger-joint threshold (<=) considered closed"""

    open_threshold: float = 0.04
    """Gripper summed finger-joint threshold (>=) considered open (hysteresis)"""


# Phase instructions (rule-based prompts) for three agents
phase_instructions = {
    0: {
        "agent0": "Grasp the object",
        "agent1": "Wait for the other agents to finish the job",
        "agent2": "Wait for the other agents to finish the job",
    },
    1: {
        "agent0": "Grasp the object",
        "agent1": "Wait for the other agents to finish the job", 
        "agent2": "Wait for the other agents to finish the job",
    },
    2: {
        "agent0": "Place the object",
        "agent1": "Wait for the other agents to finish the job",
        "agent2": "Wait for the other agents to finish the job",
    },
    3: {
        "agent0": "Wait for the other agents to finish the job",
        "agent1": "Grasp the object",
        "agent2": "Wait for the other agents to finish the job",
    },
    4: {
        "agent0": "Wait for the other agents to finish the job",
        "agent1": "Place the object on the target",
        "agent2": "Wait for the other agents to finish the job",
    },
    5: {
        "agent0": "Wait for the other agents to finish the job",
        "agent1": "Wait for the other agents to finish the job",
        "agent2": "Grasp the object",
    },
    6: {
        "agent0": "Wait for the other agents to finish the job",
        "agent1": "Wait for the other agents to finish the job",
        "agent2": "Place the object on the final target",
    },
}


def build_prompt_for_phase(phase: int) -> str:
    phase = int(phase)
    if phase not in phase_instructions:
        phase = 0
    ins = phase_instructions[phase]
    return f"agent0: {ins['agent0']} agent1: {ins['agent1']} agent2: {ins['agent2']}"


def get_gripper_opening_from_obs(observation, agent_key: str) -> float:
    """Return an approximate gripper opening from observation qpos (sum of the two finger joints)."""
    try:
        qpos = observation['agent'][agent_key]['qpos'].squeeze(0).numpy()
        if qpos.shape[0] >= 2:
            # Panda gripper is usually last two joints
            finger_joints = qpos[-2:]
            return float(np.sum(finger_joints))
    except Exception:
        pass
    return float('inf')


def get_unified_model_input(observation, agent0_pos, agent1_pos, agent2_pos, prompt: str):
    import cv2

    cam0 = 'head_camera_agent0'
    cam1 = 'head_camera_agent1'
    cam2 = 'head_camera_agent2'

    img0 = observation['sensor_data'][cam0]['rgb'].squeeze(0).numpy()
    img1 = observation['sensor_data'][cam1]['rgb'].squeeze(0).numpy()
    img2 = observation['sensor_data'][cam2]['rgb'].squeeze(0).numpy()
    global_img = observation['sensor_data']['head_camera_global']['rgb'].squeeze(0).numpy()

    timestamp = int(time.time())
    save_dir = os.getenv("MAS_VLA_IMAGE_DIR", "tmp")
    try:
        os.makedirs(save_dir, exist_ok=True)
        img0_bgr = cv2.cvtColor(img0, cv2.COLOR_RGB2BGR)
        img1_bgr = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
        img2_bgr = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
        global_bgr = cv2.cvtColor(global_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(save_dir, f"unified_agent0_{timestamp}.jpg"), img0_bgr)
        cv2.imwrite(os.path.join(save_dir, f"unified_agent1_{timestamp}.jpg"), img1_bgr)
        cv2.imwrite(os.path.join(save_dir, f"unified_agent2_{timestamp}.jpg"), img2_bgr)
        cv2.imwrite(os.path.join(save_dir, f"unified_global_{timestamp}.jpg"), global_bgr)
    except Exception:
        pass

    # 7-dim joints + scalar gripper per agent (3 agents total)
    joint_position_0 = agent0_pos[:7]
    gripper_position_0 = float(agent0_pos[7])

    joint_position_1 = agent1_pos[:7]
    gripper_position_1 = float(agent1_pos[7])

    joint_position_2 = agent2_pos[:7]
    gripper_position_2 = float(agent2_pos[7])

    return {
        'image_agent0': img0,
        'image_agent1': img1,
        'image_agent2': img2,
        'global_image': global_img,
        'joint_position_agent0': joint_position_0,
        'gripper_position_agent0': gripper_position_0,
        'joint_position_agent1': joint_position_1,
        'gripper_position_agent1': gripper_position_1,
        'joint_position_agent2': joint_position_2,
        'gripper_position_agent2': gripper_position_2,
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
    if args.robot_uids is not None:
        env_kwargs["robot_uids"] = tuple(args.robot_uids.split(","))

    env: BaseEnv = gym.make(env_id, **env_kwargs)

    record_dir = args.record_dir + '/' + str(args.seed) + '_' + str(args.data_num) + '_' + str(args.checkpoint_num)
    if record_dir:
        record_dir = record_dir.format(env_id=env_id)
        env = RecordEpisodeMA(env, record_dir, info_on_video=False, save_trajectory=False, max_steps_per_video=30000)

    raw_obs, _ = env.reset(seed=args.seed)

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
    if agent_num < 3:
        raise RuntimeError(f"Unified 24-DoF evaluation expects at least 3 agents, found {agent_num}")
    if agent_num > 3 and verbose:
        print(f"Warning: found {agent_num} agents. Only the first three agents will be controlled by the 24-DoF policy.")

    if args.seed is not None and env.action_space is not None:
        env.action_space.seed(args.seed[0])

    if args.render_mode is not None:
        viewer = env.render()
        if isinstance(viewer, sapien.utils.Viewer):
            viewer.paused = args.pause
        env.render()

    # Build initial unified input for three agents
    init_qpos_0 = raw_obs['agent']['panda-0']['qpos'].squeeze(0)[:-2].numpy()
    init_qpos_0 = np.append(init_qpos_0, planner.gripper_state[0])

    init_qpos_1 = raw_obs['agent']['panda-1']['qpos'].squeeze(0)[:-2].numpy()
    init_qpos_1 = np.append(init_qpos_1, planner.gripper_state[1])

    init_qpos_2 = raw_obs['agent']['panda-2']['qpos'].squeeze(0)[:-2].numpy()
    init_qpos_2 = np.append(init_qpos_2, planner.gripper_state[2])

    # Rule-based phase initialization
    phase = 0
    dynamic_prompt = build_prompt_for_phase(phase) if args.use_rule_prompt else args.prompt
    if verbose and args.use_rule_prompt:
        print(f"Initial phase: {phase}, prompt: {dynamic_prompt}")

    unified_example = get_unified_model_input(raw_obs, init_qpos_0, init_qpos_1, init_qpos_2, dynamic_prompt)

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

        # Query unified model once for all three agents
        infer_result = get_unified_model_answer(unified_example)
        action_array = np.asarray(infer_result['actions'])
        if action_array.ndim == 1:
            action_array = action_array.reshape(1, -1)
        if action_array.shape[1] >= 24:  # 3 agents × 8 DoF = 24
            action_array = action_array[:, :24]
        else:
            raise AssertionError(f"Expected at least 24 dims, got {action_array.shape}")

        # For each high-level step, compute time-parametrized paths for all three agents
        horizon = min(50, action_array.shape[0])
        for i in range(50):
            now_action24 = action_array[i]

            # Split for each agent: 8 DoF per agent (7 joints + 1 gripper)
            now0 = now_action24[:8]
            now1 = now_action24[8:16]
            now2 = now_action24[16:24]

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

            # Agent 2 current qpos
            if i == 0:
                current_qpos_2 = raw_obs['agent']['panda-2']['qpos'].squeeze(0)[:-2].numpy()
            else:
                current_qpos_2 = action_array[i - 1][16:23]

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

            # Plan for agent 2
            path2 = np.vstack((current_qpos_2, now2[:-1]))
            try:
                times2, pos2, v2, a2, dur2 = planner.planner[2].TOPP(path2, 0.05, verbose=True)
            except Exception as e:
                print(f"Planner error (agent 2): {e}")
                action_now2 = np.hstack([now2[:-1], now2[-1]])
                action_dict['panda-2'].append(action_now2)
                action_step_dict['panda-2'].append(1)
                pos2 = np.empty((0, 7))

            n_step0 = pos0.shape[0] if isinstance(pos0, np.ndarray) else 0
            n_step1 = pos1.shape[0] if isinstance(pos1, np.ndarray) else 0
            n_step2 = pos2.shape[0] if isinstance(pos2, np.ndarray) else 0

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

            if n_step2 == 0:
                action_now2 = np.hstack([now2[:-1], now2[-1]])
                action_dict['panda-2'].append(action_now2)
                action_step_dict['panda-2'].append(1)
            else:
                action_step_dict['panda-2'].append(n_step2)
                for j in range(n_step2):
                    true_action2 = np.hstack([pos2[j], now2[-1]])
                    action_dict['panda-2'].append(true_action2)

        # Execute synchronized steps for three agents
        start_idx = [0, 0, 0]
        for i in range(50):
            max_step = max(action_step_dict['panda-0'][i], 
                          action_step_dict['panda-1'][i],
                          action_step_dict['panda-2'][i])
            for j in range(max_step):
                true_action = dict()
                now_step0 = min(j, action_step_dict['panda-0'][i] - 1)
                now_step1 = min(j, action_step_dict['panda-1'][i] - 1)
                now_step2 = min(j, action_step_dict['panda-2'][i] - 1)
                true_action['panda-0'] = action_dict['panda-0'][start_idx[0] + now_step0]
                true_action['panda-1'] = action_dict['panda-1'][start_idx[1] + now_step1]
                true_action['panda-2'] = action_dict['panda-2'][start_idx[2] + now_step2]
                observation, reward, terminated, truncated, info = env.step(true_action)
                if verbose:
                    env.render_human()
            if verbose:
                print(true_action)
                print("max_step", max_step)

            start_idx[0] += action_step_dict['panda-0'][i]
            start_idx[1] += action_step_dict['panda-1'][i]
            start_idx[2] += action_step_dict['panda-2'][i]

            # Phase transition logic based on gripper openings (rule-based) for three agents
            if args.use_rule_prompt:
                g0 = get_gripper_opening_from_obs(observation, 'panda-0')
                g1 = get_gripper_opening_from_obs(observation, 'panda-1')
                g2 = get_gripper_opening_from_obs(observation, 'panda-2')
                agent0_closed = (g0 <= args.close_threshold)
                agent0_open = (g0 >= args.open_threshold)
                agent1_closed = (g1 <= args.close_threshold)
                agent1_open = (g1 >= args.open_threshold)
                agent2_closed = (g2 <= args.close_threshold)

                prev_phase = phase
                if phase < 2 and agent0_closed:
                    phase = 2
                elif phase < 3 and phase >= 2 and agent0_open:
                    phase = 3
                elif phase < 4 and phase >= 3 and agent1_closed:
                    phase = 4
                elif phase < 5 and phase >= 4 and agent1_open:
                    phase = 5
                elif phase < 6 and phase >= 5 and agent2_closed:
                    phase = 6
                if verbose and prev_phase != phase:
                    print(f"Phase transition: {prev_phase} -> {phase}")

            # Build next model input from latest observation and last commanded targets
            last0 = action_dict['panda-0'][start_idx[0] - 1]
            last1 = action_dict['panda-1'][start_idx[1] - 1]
            last2 = action_dict['panda-2'][start_idx[2] - 1]
            next_prompt = build_prompt_for_phase(phase) if args.use_rule_prompt else args.prompt
            if verbose and args.use_rule_prompt:
                print(f"Current prompt: {next_prompt}")
            unified_example = get_unified_model_input(
                observation,
                last0,  # 8 dims (7 joints + 1 gripper)
                last1,
                last2,
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

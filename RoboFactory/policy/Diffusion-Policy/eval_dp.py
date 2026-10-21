# import sys
# sys.path.append('./') 
# sys.path.insert(0, './policy/Diffusion-Policy') 

# import torch  
# import os

# import hydra
# from pathlib import Path
# from collections import deque
# import traceback

# import yaml
# from datetime import datetime
# import importlib
# import dill
# from argparse import ArgumentParser
# from diffusion_policy.workspace.robotworkspace import RobotWorkspace
# from diffusion_policy.common.pytorch_util import dict_apply
# from diffusion_policy.policy.base_image_policy import BaseImagePolicy
# from diffusion_policy.env_runner.dp_runner import DPRunner
# from planner.motionplanner import PandaArmMotionPlanningSolver


# import gymnasium as gym
# import numpy as np
# import sapien

# from mani_skill.envs.sapien_env import BaseEnv
# from mani_skill.utils import gym_utils
# from utils.wrappers.record import RecordEpisodeMA

# import tyro
# from dataclasses import dataclass
# from typing import List, Optional, Annotated, Union
# import pdb

# @dataclass
# class Args:
#     env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "StackCube"
#     """The environment ID of the task you want to simulate"""

#     obs_mode: Annotated[str, tyro.conf.arg(aliases=["-o"])] = "rgb"
#     """Observation mode"""

#     robot_uids: Annotated[Optional[str], tyro.conf.arg(aliases=["-r"])] = None
#     """Robot UID(s) to use. Can be a comma separated list of UIDs or empty string to have no agents. If not given then defaults to the environments default robot"""

#     sim_backend: Annotated[str, tyro.conf.arg(aliases=["-b"])] = "auto"
#     """Which simulation backend to use. Can be 'auto', 'cpu', 'gpu'"""

#     reward_mode: Optional[str] = None
#     """Reward mode"""

#     num_envs: Annotated[int, tyro.conf.arg(aliases=["-n"])] = 1
#     """Number of environments to run."""

#     control_mode: Annotated[Optional[str], tyro.conf.arg(aliases=["-c"])] = "pd_joint_pos"
#     """Control mode"""

#     render_mode: str = "rgb_array"
#     """Render mode"""

#     shader: str = "default"
#     """Change shader used for all cameras in the environment for rendering. Default is 'minimal' which is very fast. Can also be 'rt' for ray tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer"""

#     record_dir: Optional[str] = './testvideo/{env_id}'
#     """Directory to save recordings"""

#     pause: Annotated[bool, tyro.conf.arg(aliases=["-p"])] = False
#     """If using human render mode, auto pauses the simulation upon loading"""

#     quiet: bool = False
#     """Disable verbose output."""

#     seed: Annotated[Optional[Union[int, List[int]]], tyro.conf.arg(aliases=["-s"])] = 10000
#     """Seed(s) for random actions and simulator. Can be a single integer or a list of integers. Default is None (no seeds)"""

#     data_num: int = 100
#     """The number of episode data used for training the policy"""

#     checkpoint_num: int = 300
#     """The number of training epoch of the checkpoint"""

#     record_dir: Optional[str] = './eval_video/{env_id}'
#     """Directory to save recordings"""

#     max_steps: int = 250
#     """Maximum number of steps to run the simulation"""

# def get_policy(checkpoint, output_dir, device):
#     # load checkpoint
#     payload = torch.load(open('./policy/Diffusion-Policy/'+checkpoint, 'rb'), pickle_module=dill)
#     cfg = payload['cfg']
#     cls = hydra.utils.get_class(cfg._target_)
#     workspace = cls(cfg, output_dir=output_dir)
#     workspace: RobotWorkspace
#     workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
#     # get policy from workspace
#     policy = workspace.model
#     if cfg.training.use_ema:
#         policy = workspace.ema_model
    
#     device = torch.device(device)
#     policy.to(device)
#     policy.eval()

#     return policy


# class DP:
#     def __init__(self, task_name, checkpoint_num: int, data_num: int):
#         # self.policy = get_policy(f'checkpoints/{task_name}_{data_num}/{checkpoint_num}.ckpt', None, 'cuda:0')
#         self.policy = get_policy(f'checkpoints/pick_car_clean20_dual_cam_Agent0_20_pre/{checkpoint_num}.ckpt', None, 'cuda:0')
#         self.runner = DPRunner(output_dir=None)

#     def update_obs(self, observation):
#         self.runner.update_obs(observation)
    
#     def get_action(self, observation=None):
#         action = self.runner.get_action(self.policy, observation)
#         return action

#     def get_last_obs(self):
#         return self.runner.obs[-1]

# def get_model_input(observation, agent_pos):
#     head_cam = np.moveaxis(observation['sensor_data']['head_camera']['rgb'].squeeze(0).numpy(), -1, 0) / 255
#     return dict(
#         head_cam = head_cam,
#         agent_pos = agent_pos,
#     )


# def main(args: Args):
#     np.set_printoptions(suppress=True, precision=3)
#     verbose = not args.quiet
#     if isinstance(args.seed, int):
#         args.seed = [args.seed]
#     if args.seed is not None:
#         np.random.seed(args.seed[0])
#     parallel_in_single_scene = args.render_mode == "human"
#     if args.render_mode == "human" and args.obs_mode in ["sensor_data", "rgb", "rgbd", "depth", "point_cloud"]:
#         print("Disabling parallel single scene/GUI render as observation mode is a visual one. Change observation mode to state or state_dict to see a parallel env render")
#         parallel_in_single_scene = False
#     if args.render_mode == "human" and args.num_envs == 1:
#         parallel_in_single_scene = False
#     env_kwargs = dict(
#         obs_mode=args.obs_mode,
#         reward_mode=args.reward_mode,
#         control_mode=args.control_mode,
#         render_mode=args.render_mode,
#         sensor_configs=dict(shader_pack=args.shader),
#         human_render_camera_configs=dict(shader_pack=args.shader),
#         viewer_camera_configs=dict(shader_pack=args.shader),
#         num_envs=args.num_envs,
#         sim_backend=args.sim_backend,
#         enable_shadow=True,
#         parallel_in_single_scene=parallel_in_single_scene,
#     )
#     if args.robot_uids is not None:
#         env_kwargs["robot_uids"] = tuple(args.robot_uids.split(","))
#     env: BaseEnv = gym.make(args.env_id, **env_kwargs)

#     record_dir = args.record_dir + '/' + str(args.seed) + '_' + str(args.data_num) + '_' + str(args.checkpoint_num)
#     if record_dir:
#         record_dir = record_dir.format(env_id=args.env_id)
#         env = RecordEpisodeMA(env, record_dir, info_on_video=False, save_trajectory=False, max_steps_per_video=30000)

#     planner = PandaArmMotionPlanningSolver(
#         env,
#         debug=False,
#         vis=verbose,
#         base_pose=env.unwrapped.agent.robot.pose,
#         visualize_target_grasp_pose=verbose,
#         print_env_info=False,
#     )
#     dp = DP(args.env_id, args.checkpoint_num, args.data_num)
#     raw_obs, _ = env.reset(seed=args.seed, options=dict(reconfigure=True))
#     if args.seed is not None and env.action_space is not None:
#         env.action_space.seed(args.seed[0])
#     if args.render_mode is not None:
#         viewer = env.render()
#         if isinstance(viewer, sapien.utils.Viewer):
#             viewer.paused = args.pause
#         env.render()
#     initial_qpos = raw_obs['agent']['qpos'].squeeze(0)[:-2].numpy()
#     initial_qpos = np.append(initial_qpos, planner.gripper_state)
#     obs = get_model_input(raw_obs, initial_qpos)
#     dp.update_obs(obs)
#     cnt = 0
#     while True:
#         if verbose:
#             print("Iteration:", cnt)
#         cnt = cnt + 1
#         if cnt > args.max_steps:
#             break
#         action = env.action_space.sample() if env.action_space is not None else None
#         action = dp.get_action()
#         for i in range(6):
#             raw_obs = env.get_obs()
#             current_qpos = raw_obs['agent']['qpos'].squeeze(0)[:-2].numpy()
#             path = np.vstack((current_qpos, action[i][:-1]))
#             error_flag = False
#             try:
#                 times, right_pos, right_vel, acc, duration = planner.planner[0].TOPP(path, 0.05, verbose=True)
#             except Exception as e:
#                 error_flag = True
#                 print(f"Error occurred: {e}")
#             if error_flag:
#                 continue
#             result = dict()
#             result['position'] = right_pos
#             n_step = result["position"].shape[0]
#             gripper_state = action[i][-1]
#             if verbose:
#                 print(gripper_state)
#                 print(path, n_step)
#             for j in range(n_step):
#                 true_action = np.hstack([result['position'][j], gripper_state])
#                 if j != n_step - 1:
#                     observation, reward, terminated, truncated, info = env.step(true_action)
#                 else:
#                     observation, reward, terminated, truncated, info = env.step(true_action)
#                 if verbose:
#                     env.render_human()
#             obs = get_model_input(observation, true_action)
#             dp.update_obs(obs)
#         if verbose:
#             print("info", info)
#         if args.render_mode is not None:
#             env.render()
#         if info['success'] == True:
#             env.close()
#             if record_dir:
#                 print(f"Saving video to {record_dir}")
#             print("success")
#             return
#     env.close() 
#     if record_dir:
#         print(f"Saving video to {record_dir}")
#     print("failed")

# if __name__ == "__main__":
#     parsed_args = tyro.cli(Args)
#     main(parsed_args)

import sys
sys.path.append('./') 
sys.path.insert(0, './policy/Diffusion-Policy') 

import torch  
import os

import hydra
from pathlib import Path
from collections import deque
from tasks import *
import traceback

import yaml
from datetime import datetime
import importlib
import dill
from argparse import ArgumentParser
from diffusion_policy.workspace.robotworkspace import RobotWorkspace
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.env_runner.dp_runner import DPRunner
from planner.motionplanner import PandaArmMotionPlanningSolver


import gymnasium as gym
import numpy as np
import sapien

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import gym_utils
from utils.wrappers.record import RecordEpisodeMA

import tyro
from dataclasses import dataclass
from typing import List, Optional, Annotated, Union

@dataclass
class Args:
    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = ""
    """The environment ID of the task you want to simulate"""

    config: str = "${CONFIG_DIR}/robocasa/take_photo.yaml"
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
    """Change shader used for all cameras in the environment for rendering. Default is 'minimal' which is very fast. Can also be 'rt' for ray tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer"""

    record_dir: Optional[str] = './testvideo/{env_id}'
    """Directory to save recordings"""

    pause: Annotated[bool, tyro.conf.arg(aliases=["-p"])] = False
    """If using human render mode, auto pauses the simulation upon loading"""

    quiet: bool = False
    """Disable verbose output."""

    seed: Annotated[Optional[Union[int, List[int]]], tyro.conf.arg(aliases=["-s"])] = 10000
    """Seed(s) for random actions and simulator. Can be a single integer or a list of integers. Default is None (no seeds)"""

    data_num: int = 100
    """The number of episode data used for training the policy"""

    checkpoint_num: int = 300
    """The number of training epoch of the checkpoint"""

    # record_dir: Optional[str] = './eval_video/{env_id}'
    # """Directory to save recordings"""
    record_dir: Optional[str] = './eval_video/{env_id}_{test_id}'
    """Directory to save recordings"""

    max_steps: int = 50
    """Maximum number of steps to run the simulation"""
     
    status: str = "success"
    """Status of the evaluation"""

    test_id: int = 0
    """Test ID for different policies"""

def get_policy(checkpoint, output_dir, device):
    # load checkpoint
    payload = torch.load(open('./'+checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: RobotWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    device = torch.device(device)
    policy.to(device)
    policy.eval()

    return policy


class DP:
    def __init__(self, task_name, checkpoint_num: int, data_num: int, id: int = 0, test_id: int = 0):
        # self.policy = get_policy(f'checkpoints/{task_name}_{data_num}/{checkpoint_num}.ckpt', None, 'cuda:0')
        if test_id == 0:
            self.policy = get_policy(f'checkpoints/pick_car_clean1_dual_cam_Agent0_20/{checkpoint_num}.ckpt', None, 'cuda:0')
        elif test_id == 1:
            self.policy = get_policy(f'checkpoints/pick_car_clean1_dual_cam_Agent0_20_origin0204/{checkpoint_num}.ckpt', None, 'cuda:0')
        # Detect what observation keys the policy expects from normalizer
        normalizer_keys = set(self.policy.normalizer.params_dict.keys())
        self.expects_dual_cam = 'img1' in normalizer_keys and 'img2' in normalizer_keys
        self.expects_head_cam = 'head_cam' in normalizer_keys
        print(f"Policy normalizer keys: {normalizer_keys}")
        print(f"Expects dual_cam (img1/img2): {self.expects_dual_cam}")
        print(f"Expects head_cam: {self.expects_head_cam}")
        self.runner = DPRunner(output_dir=None, policy=self.policy)

    def update_obs(self, observation):
        self.runner.update_obs(observation)
    
    def get_action(self, observation=None):
        action = self.runner.get_action(self.policy, observation)
        return action

    def get_last_obs(self):
        return self.runner.obs[-1]

def get_model_input(observation, agent_pos, expects_dual_cam=False):
    wrist_cam = np.moveaxis(observation['sensor_data']['wrist_camera_agent0']['rgb'].squeeze(0).numpy(), -1, 0) / 255
    
    if expects_dual_cam:
        # For dual camera mode, need both wrist and head camera
        if 'head_camera_global' in observation['sensor_data']:
            head_cam = np.moveaxis(observation['sensor_data']['head_camera_global']['rgb'].squeeze(0).numpy(), -1, 0) / 255
        else:
            # Fallback: use wrist camera for both if head camera not available
            print("Warning: head_camera_global not found, using wrist camera for img2")
            head_cam = wrist_cam.copy()
        return dict(
            img1 = wrist_cam,  # wrist camera
            img2 = head_cam,    # head camera
            agent_pos = agent_pos,
        )
    else:
        # Single camera mode
        return dict(
            head_cam = wrist_cam,
            agent_pos = agent_pos,
        )

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
        status=args.status
    )
    if args.robot_uids is not None:
        env_kwargs["robot_uids"] = tuple(args.robot_uids.split(","))
    env: BaseEnv = gym.make(env_id, **env_kwargs)

    record_dir = args.record_dir + '/' + str(args.seed) + '_' + str(args.data_num) + '_' + str(args.checkpoint_num)
    if record_dir:
        # record_dir = record_dir.format(env_id=env_id)
        record_dir = record_dir.format(env_id=env_id, test_id=args.test_id)
        env = RecordEpisodeMA(env, record_dir, info_on_video=False, save_trajectory=False, max_steps_per_video=30000)
    raw_obs, _ = env.reset(seed=args.seed)
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=False,
        vis=verbose,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=verbose,
        print_env_info=False,
    )
    
    
    dp = DP(env_id, args.checkpoint_num, args.data_num, id=0, test_id=args.test_id)
    if args.seed is not None and env.action_space is not None:
        env.action_space.seed(args.seed[0])
    if args.render_mode is not None:
        viewer = env.render()
        if isinstance(viewer, sapien.utils.Viewer):
            viewer.paused = args.pause
        env.render()
    # Use full qpos (9 dims: 7 joints + 2 grippers) to match training data format
    initial_qpos = raw_obs['agent']['qpos'].squeeze(0).numpy()
    obs = get_model_input(raw_obs, initial_qpos, expects_dual_cam=dp.expects_dual_cam)
    dp.update_obs(obs)
    cnt = 0
    while True:
        if verbose:
            print("Iteration:", cnt)
        cnt = cnt + 1
        if cnt > args.max_steps:
            break
        action = env.action_space.sample() if env.action_space is not None else None
        action = dp.get_action()
        # Use horizon from policy instead of n_action_steps
        # action_pred has shape (horizon, action_dim) matching training data format
        horizon = 16 #dp.policy.horizon
        if verbose:
            print(f"Action shape: {action.shape}, horizon: {horizon}, n_action_steps: {dp.policy.n_action_steps}")
        # Validate action shape matches expected horizon
        if action.shape[0] < horizon:
            print(f"Warning: Action shape {action.shape[0]} is less than horizon {horizon}")
            horizon = action.shape[0]
        for i in range(horizon):
            raw_obs = env.get_obs()
            current_qpos = raw_obs['agent']['qpos'].squeeze(0)[:-2].numpy()
            path = np.vstack((current_qpos, action[i][:-1]))
            error_flag = False
            try:
                times, right_pos, right_vel, acc, duration = planner.planner[0].TOPP(path, 0.05, verbose=True)
            except Exception as e:
                error_flag = True
                print(f"Error occurred: {e}")
            if error_flag:
                continue
            result = dict()
            result['position'] = right_pos
            n_step = result["position"].shape[0]
            gripper_state = action[i][-1]
            if verbose:
                print(gripper_state)
                print(path, n_step)
            for j in range(n_step):
                true_action = np.hstack([result['position'][j], gripper_state])
                if j != n_step - 1:
                    observation, reward, terminated, truncated, info = env.step(true_action)
                else:
                    observation, reward, terminated, truncated, info = env.step(true_action)
                if verbose:
                    env.render_human()
            # Use full qpos (9 dims) from observation to match training data format
            current_qpos_full = observation['agent']['qpos'].squeeze(0).numpy()
            obs = get_model_input(observation, current_qpos_full, expects_dual_cam=dp.expects_dual_cam)
            dp.update_obs(obs)
        if verbose:
            print("info", info)
        if args.render_mode is not None:
            env.render()
        if info['success'] == True:
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
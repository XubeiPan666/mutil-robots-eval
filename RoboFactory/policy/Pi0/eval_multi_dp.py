import sys
sys.path.append('./') 
sys.path.insert(0, './policy/Diffusion-Policy') 

import torch  
import os
import time

import hydra
from pathlib import Path
from collections import deque, defaultdict
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
from client import *


SERVER_URL1 = os.getenv("OPENPI_SERVER_URL_ARM0", "http://localhost:8000")
SERVER_URL2 = os.getenv("OPENPI_SERVER_URL_ARM1", "http://localhost:8001")
client1 = OpenPIClient(SERVER_URL1)
client2 = OpenPIClient(SERVER_URL2)
print("=== 检查服务器状态 ===")
health = client1.health_check()
if health:
    print(f"服务器状态: {health}")
else:
    print("无法连接到服务器")

def get_model_answer(example, agent_id):
    if agent_id == 0:
        result = client1.infer(example)
    elif agent_id==1:
        result = client2.infer(example)
    else:
        raise ValueError
    return result



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

    record_dir: Optional[str] = './eval_video/{env_id}'
    """Directory to save recordings"""

    max_steps: int = 250
    """Maximum number of steps to run the simulation"""

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
    def __init__(self, task_name, checkpoint_num: int, data_num: int, id: int = 0):
        self.policy = get_policy(f'checkpoints/{task_name}_Agent{id}_{data_num}/{checkpoint_num}.ckpt', None, 'cuda:0')
        self.runner = DPRunner(output_dir=None)

    def update_obs(self, observation):
        self.runner.update_obs(observation)
    
    def get_action(self, observation=None):
        action = self.runner.get_action(self.policy, observation)
        return action

    def get_last_obs(self):
        return self.runner.obs[-1]

def get_model_input(observation, agent_pos, agent_id):
    camera_name = 'head_camera' + '_agent' + str(agent_id)
    head_cam = np.moveaxis(observation['sensor_data'][camera_name]['rgb'].squeeze(0).numpy(), -1, 0) / 255
    return dict(
        head_cam = head_cam,
        agent_pos = agent_pos,
    )
    
    
    
def get_model_input_1(observation, agent_pos, agent_id):
    import cv2

    camera_name = 'head_camera' + '_agent' + str(agent_id)

    # 提取图像并去掉 batch 维度，保持 (240, 320, 3)
    img = observation['sensor_data'][camera_name]['rgb'].squeeze(0).numpy()
    global_img=observation['sensor_data']['head_camera_global']['rgb'].squeeze(0).numpy()
    # Resize 到 (224, 224, 3)
    # resized_img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
    # global_resized_img = cv2.resize(global_img, (256, 256), interpolation=cv2.INTER_LINEAR)
    # # 获取当前时间戳
    timestamp = int(time.time())

    # # 构造文件名
    filename1 = f"tmp/head_{agent_id}_{timestamp}.jpg"
    filename2 = f"tmp/global_{agent_id}_image_{timestamp}.jpg"
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    global_img = cv2.cvtColor(global_img, cv2.COLOR_RGB2BGR)
    # 保存图片
    cv2.imwrite(filename1, img)
    cv2.imwrite(filename2, global_img)
    # joint_position 取前7维
    joint_position = agent_pos[:7]

    # gripper_position 取第8维
    gripper_position = agent_pos[7:8]
    print(gripper_position)
    

    # 返回格式与示例完全一致
    return {
        'image': img,
        'wrist_image': global_img,#传入不对
        'joint_position': joint_position,
        'gripper_position': gripper_position,
        'prompt': 'pass the shoe with two arms'
    }
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

    # Load multi dp policy
    agent_num = planner.agent_num
    # dp_models = []
    # for i in range(agent_num):
    #     dp_models.append(DP(env_id, args.checkpoint_num, args.data_num, id=i))

    if args.seed is not None and env.action_space is not None:
        env.action_space.seed(args.seed[0])
    if args.render_mode is not None:
        viewer = env.render()
        if isinstance(viewer, sapien.utils.Viewer):
            viewer.paused = args.pause
        env.render()
    vla_models = [None for _ in range(agent_num)]
    for id in range(agent_num):
        initial_qpos = raw_obs['agent'][f'panda-{id}']['qpos'].squeeze(0)[:-2].numpy()
        initial_qpos = np.append(initial_qpos, planner.gripper_state[id])
        # obs = get_model_input(raw_obs, initial_qpos, id)
        obs1=get_model_input_1(raw_obs,initial_qpos, id)
        # dp_models[id].update_obs(obs)
        vla_models[id]=obs1
    cnt = 0
    
    #### test test
    #test=get_model_answer(vla_models[0])
    
    
    while True:
        if verbose:
            print("Iteration:", cnt)
        cnt = cnt + 1
        if cnt > args.max_steps:
            break
        action_dict = defaultdict(list)
        action_step_dict = defaultdict(list)
        for id in range(agent_num):
            #action_list = dp_models[id].get_action()
            action_list = get_model_answer(vla_models[id], id)['actions']
            # action_list[:, -1] = action_list[:, -1]-0.1
            # action_list[:, -1][action_list[:, -1] < 0.5] -= 1
            # if action_list[-1]>0:
            #     action_list[-1] = 1.0
            # else:
            #     action_list[-1] = -1.0
                
            print(f'{id}:{action_list}')
            for i in range(50):
                now_action = action_list[i]
                raw_obs = env.get_obs()
                if i == 0:
                    current_qpos = raw_obs['agent'][f'panda-{id}']['qpos'].squeeze(0)[:-2].numpy()
                else:
                    current_qpos = action_list[i - 1][:-1]
                path = np.vstack((current_qpos, now_action[:-1]))
                try:
                    times, position, right_vel, acc, duration = planner.planner[id].TOPP(path, 0.06, verbose=True)
                except Exception as e:
                    print(f"Error occurred: {e}")
                    action_now = np.hstack([now_action[:-1], now_action[-1]])
                    action_dict[f'panda-{id}'].append(action_now)
                    action_step_dict[f'panda-{id}'].append(1)
                    continue
                n_step = position.shape[0]
                gripper_state = now_action[-1]
                if n_step == 0:
                    action_now = np.hstack([now_action[:-1], gripper_state])
                    action_dict[f'panda-{id}'].append(action_now)
                    action_step_dict[f'panda-{id}'].append(1)
                else:
                    action_step_dict[f'panda-{id}'].append(n_step)
                    for j in range(n_step):
                        true_action = np.hstack([position[j], gripper_state])
                        action_dict[f'panda-{id}'].append(true_action)
        
        start_idx = []
        for id in range(agent_num):
            start_idx.append(0)
        for i in range(50):
            max_step = 0
            for id in range(agent_num):
                max_step = max(max_step, action_step_dict[f'panda-{id}'][i])
            for j in range(max_step):
                true_action = dict()
                for id in range(agent_num):
                    now_step = min(j, action_step_dict[f'panda-{id}'][i] - 1)
                    true_action[f'panda-{id}'] = action_dict[f'panda-{id}'][start_idx[id] + now_step]
                observation, reward, terminated, truncated, info = env.step(true_action)
                if verbose:
                    env.render_human()
            if verbose:
                print(true_action)
                print("max_step", max_step)
            for id in range(agent_num):
                start_idx[id] += action_step_dict[f'panda-{id}'][i]
                # if action_step_dict[f'panda-{id}'][i] == 0:
                #     continue
                #obs = get_model_input(observation, true_action[f'panda-{id}'], id)
                obs1 = get_model_input_1(observation, true_action[f'panda-{id}'], id)
                print("-----obs------",i," no. ",id,obs1['gripper_position'])
                #dp_models[id].update_obs(obs)
                vla_models[id]=obs1
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

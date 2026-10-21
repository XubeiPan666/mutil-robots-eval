from typing import Any, Dict, Tuple

import os.path as osp
import numpy as np
import sapien
import torch
import json
import yaml
import copy
from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
import utils.scenes
from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SimConfig

@register_env("ThreeArmsStackCubes-rf", max_episode_steps=1500)
class ThreeArmsStackCubesEnv(BaseEnv):
    SUPPORTED_ROBOTS = [("panda", "panda", "panda")]
    agent: MultiAgent[Tuple[Panda, Panda, Panda]]
    
    goal_thresh = 0.025
    cube_color = np.concatenate((np.array([187, 116, 175]) / 255, [1]))
    cube_half_size = 0.02

    def __init__(
        self, *args, robot_uids=("panda", "panda", "panda"), robot_init_qpos_noise=0.02, **kwargs
    ):
        self.status = kwargs.pop('status', None)
        self.robot_init_qpos_noise = robot_init_qpos_noise
        assert 'config' in kwargs
        with open(kwargs['config'], 'r', encoding='utf-8') as f:
            self.cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
        del kwargs['config']
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        cfg = copy.deepcopy(self.cfg)
        camera_cfg = cfg.get('cameras', {})
        sensor_cfg = camera_cfg.get('sensor', [])
        all_camera_configs =[]
        for sensor in sensor_cfg:
            pose = sensor['pose']
            if pose['type'] == 'pose':
                sensor['pose'] = sapien.Pose(*pose['params'])
            elif pose['type'] == 'look_at':
                sensor['pose'] = sapien_utils.look_at(*pose['params'])
            
            # 如果是手腕相机，挂载到对应 agent 的 TCP link
            if 'wrist_camera_agent' in sensor['uid']:
                agent_id = int(sensor['uid'].replace('wrist_camera_agent', ''))
                if hasattr(self, 'agent') and self.agent is not None and agent_id < len(self.agent.agents):
                    sensor['mount'] = self.agent.agents[agent_id].tcp
                    
            all_camera_configs.append(CameraConfig(**sensor))
        return all_camera_configs

    @property
    def _default_human_render_camera_configs(self):
        cfg = copy.deepcopy(self.cfg)
        camera_cfg = cfg.get('cameras', {})
        render_cfg = camera_cfg.get('human_render', [])
        all_camera_configs =[]
        for render in render_cfg:
            pose = render['pose']
            if pose['type'] == 'pose':
                render['pose'] = sapien.Pose(*pose['params'])
            elif pose['type'] == 'look_at':
                render['pose'] = sapien_utils.look_at(*pose['params'])
            all_camera_configs.append(CameraConfig(**render))
        return all_camera_configs

    # @property
    # def _default_sim_config(self):
    #     return SimConfig(
    #         sim_freq=1250,  # 提高模拟频率以改善夹取稳定性
    #         control_freq=250,
    #         gpu_memory_config=GPUMemoryConfig(
    #             found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18
    #         )
    #     )
    def _load_agent(self, options: dict):
        cfg = copy.deepcopy(self.cfg)
        init_poses = []
        for agent_cfg in cfg['agents']:
            init_poses.append(sapien.Pose(p=agent_cfg['pos']['ppos']['p']))
        super()._load_agent(options, init_poses)

    def _load_scene(self, options: dict):
        cfg = copy.deepcopy(self.cfg)
        scene_name = cfg['scene']['name']
        self.cube_half_size = common.to_tensor([0.02] * 3, device=self.device)
        self.bowl_half_size = common.to_tensor([0.02, 0.02, 0.0125], device=self.device)
        scene_builder = getattr(utils.scenes, f'{scene_name}SceneBuilder')
        self.scene_builder = scene_builder(env=self, cfg=cfg)
        self.scene_builder.build()

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            self.scene_builder.initialize(env_idx)
        # alignment between cube and camera
        # cube_ppose = self.cube.pose.p
        # temp_pose = self.camera.pose
        # temp_pose.p[:, 0] = cube_ppose[:, 0]
        # temp_pose.p[:, 1] = cube_ppose[:, 1]
        # self.camera.set_pose(temp_pose)

    @property
    def left_agent(self) -> Panda:
        return self.agent.agents[0]

    @property
    def right_agent(self) -> Panda:
        return self.agent.agents[2]

    @property
    def middle_agent(self) -> Panda:
        return self.agent.agents[1]
        
    def evaluate(self):
        if self.status == "three_arms_stack_cubes":
            pos_A = self.cube1.pose.p
            pos_B = self.cube2.pose.p
            pos_C = self.cube3.pose.p
            offset =  pos_B - pos_A
            xy_flag = (
                torch.linalg.norm(offset[..., :2], axis=1)
                <= torch.linalg.norm(self.cube_half_size[:2]) + 0.005
            )
            z_flag = torch.abs(offset[..., 2] - self.cube_half_size[..., 2] * 2) <= 0.03
            is_cubeB_on_cubeA = torch.logical_and(xy_flag, z_flag)
            offset = pos_C - pos_B
            xy_flag = (
                torch.linalg.norm(offset[..., :2], axis=1)
                <= torch.linalg.norm(self.cube_half_size[:2]) + 0.005
            )
            z_flag = torch.abs(offset[..., 2] - self.cube_half_size[..., 2] * 2) <= 0.03
            is_cubeC_on_cubeB = torch.logical_and(xy_flag, z_flag)
            
            # Check if cubeB and cubeC are within rectangle centered at goal_region
            # Rectangle size: x = 0.1, y = 0.18
            goal_x = self.goal_region.pose.p[..., 0]
            goal_y = self.goal_region.pose.p[..., 1]
            
            # Rectangle bounds: half_size in each direction
            x_half = 0.1 / 2  # 0.05
            y_half = 0.18 / 2  # 0.09
            
            # Check bowlB
            cubeB_x = self.cube2.pose.p[..., 0]
            cubeB_y = self.cube2.pose.p[..., 1]
            cubeB_x_in_range = torch.abs(cubeB_x - goal_x) <= x_half
            cubeB_y_in_range = torch.abs(cubeB_y - goal_y) <= y_half
            cubeB_placed = cubeB_x_in_range & cubeB_y_in_range
            
            # Check bowlC
            cubeC_x = self.cube3.pose.p[..., 0]
            cubeC_y = self.cube3.pose.p[..., 1]
            cubeC_x_in_range = torch.abs(cubeC_x - goal_x) <= x_half
            cubeC_y_in_range = torch.abs(cubeC_y - goal_y) <= y_half
            cubeC_placed = cubeC_x_in_range & cubeC_y_in_range
            
            # Check if bowls are grasped, with error handling to avoid memory allocation errors
            try:
                is_cubeA_grasped = self.left_agent.is_grasping(self.cube1)
            except (RuntimeError, Exception) as e:
                # If is_grasping fails (e.g., memory allocation error), assume not grasped
                is_cubeA_grasped = torch.zeros(len(cubeC_placed), dtype=torch.bool, device=self.device)
                if hasattr(self, 'debug') and self.debug:
                    print(f"Warning: is_grasping failed for cube1: {e}")
            
            try:
                is_cubeB_grasped = self.middle_agent.is_grasping(self.cube2)
            except (RuntimeError, Exception) as e:
                is_cubeB_grasped = torch.zeros(len(cubeC_placed), dtype=torch.bool, device=self.device)
                if hasattr(self, 'debug') and self.debug:
                    print(f"Warning: is_grasping failed for cube2: {e}")
            
            try:
                is_cubeC_grasped = self.right_agent.is_grasping(self.cube3)
            except (RuntimeError, Exception) as e:
                is_cubeC_grasped = torch.zeros(len(cubeC_placed), dtype=torch.bool, device=self.device)
                if hasattr(self, 'debug') and self.debug:
                    print(f"Warning: is_grasping failed for cube3: {e}")
            success = (
                is_cubeC_on_cubeB * is_cubeB_on_cubeA * cubeC_placed * cubeB_placed * (~is_cubeA_grasped) * (~is_cubeB_grasped) * (~is_cubeC_grasped)
            )
            return {
                "is_cubeA_grasped": is_cubeA_grasped,
                "is_cubeB_grasped": is_cubeB_grasped,
                "is_cubeC_grasped": is_cubeC_grasped,
                "is_cubeC_on_cubeB": is_cubeC_on_cubeB,
                "is_cubeB_on_cubeA": is_cubeB_on_cubeA,
                "cubeB_placed": cubeB_placed,
                "success": success.bool(),
            }
    
    def _get_obs_extra(self, info: Dict):
        return {}

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return 0

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return 0
    

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

@register_env("PlaceFactory-rf", max_episode_steps=1500)
class PlaceFactoryEnv(BaseEnv):
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
        if self.status == "place_factory_123" or self.status == "place_factory_132" or self.status == "place_factory_231" or self.status == "place_factory_213" or self.status == "place_factory_123_2bd":
            # Check if meat1, meat2, meat3 are all within a rectangle centered at tray
            # Rectangle size: x = 0.1, y = 0.18
            tray_x = self.tray.pose.p[..., 0]
            tray_y = self.tray.pose.p[..., 1]
            
            # Rectangle bounds: half_size in each direction
            x_half = 0.1 / 2  # 0.05
            y_half = 0.18 / 2  # 0.09
            
            # Helper function to check if a meat is within rectangle
            def check_meat_in_range(meat):
                meat_x = meat.pose.p[..., 0]
                meat_y = meat.pose.p[..., 1]
                x_in_range = torch.abs(meat_x - tray_x) <= x_half
                y_in_range = torch.abs(meat_y - tray_y) <= y_half
                return x_in_range & y_in_range
            
            # Check all three meats
            meat1_success = check_meat_in_range(self.meat1)
            meat2_success = check_meat_in_range(self.meat2)
            meat3_success = check_meat_in_range(self.meat3)
            
            # All meats must be within the rectangle
            success = meat1_success & meat2_success & meat3_success
            return {
            "success": success,
        }
        
        elif self.status == "place_factory_bowl_123" or self.status == "place_factory_bowl_123_2bd" or self.status == "place_factory_bowl_123_1bd" or self.status == "place_factory_bowl_123_3bd" or self.status == "place_factory_bowl_132_123":
            
            pos_A = self.bowl1.pose.p
            pos_B = self.bowl2.pose.p
            pos_C = self.bowl3.pose.p
            offset =  pos_B - pos_A
            xy_flag = (
                torch.linalg.norm(offset[..., :2], axis=1)
                <= torch.linalg.norm(self.bowl_half_size[:2]) + 0.005
            )
            z_flag = torch.abs(offset[..., 2] - self.bowl_half_size[..., 2] * 2) <= 0.01
            is_bowlB_on_bowlA = torch.logical_and(xy_flag, z_flag)
            offset = pos_C - pos_B
            xy_flag = (
                torch.linalg.norm(offset[..., :2], axis=1)
                <= torch.linalg.norm(self.bowl_half_size[:2]) + 0.005
            )
            z_flag = torch.abs(offset[..., 2] - self.bowl_half_size[..., 2] * 2) <= 0.01
            is_bowlC_on_bowlB = torch.logical_and(xy_flag, z_flag)
            
            # Check if bowlB and bowlC are within rectangle centered at goal_region
            # Rectangle size: x = 0.1, y = 0.18
            goal_x = self.tray.pose.p[..., 0]
            goal_y = self.tray.pose.p[..., 1]
            
            # Rectangle bounds: half_size in each direction
            x_half = 0.1 / 2  # 0.05
            y_half = 0.18 / 2  # 0.09
            
            # Check bowlB
            bowlB_x = self.bowl2.pose.p[..., 0]
            bowlB_y = self.bowl2.pose.p[..., 1]
            bowlB_x_in_range = torch.abs(bowlB_x - goal_x) <= x_half
            bowlB_y_in_range = torch.abs(bowlB_y - goal_y) <= y_half
            bowlB_placed = bowlB_x_in_range & bowlB_y_in_range
            
            # Check bowlC
            bowlC_x = self.bowl3.pose.p[..., 0]
            bowlC_y = self.bowl3.pose.p[..., 1]
            bowlC_x_in_range = torch.abs(bowlC_x - goal_x) <= x_half
            bowlC_y_in_range = torch.abs(bowlC_y - goal_y) <= y_half
            bowlC_placed = bowlC_x_in_range & bowlC_y_in_range
            
            # Check if bowls are grasped, with error handling to avoid memory allocation errors
            try:
                is_bowlA_grasped = self.left_agent.is_grasping(self.bowl1)
            except (RuntimeError, Exception) as e:
                # If is_grasping fails (e.g., memory allocation error), assume not grasped
                is_bowlA_grasped = torch.zeros(len(bowlC_placed), dtype=torch.bool, device=self.device)
                if hasattr(self, 'debug') and self.debug:
                    print(f"Warning: is_grasping failed for bowl1: {e}")
            
            try:
                is_bowlB_grasped = self.middle_agent.is_grasping(self.bowl2)
            except (RuntimeError, Exception) as e:
                is_bowlB_grasped = torch.zeros(len(bowlC_placed), dtype=torch.bool, device=self.device)
                if hasattr(self, 'debug') and self.debug:
                    print(f"Warning: is_grasping failed for bowl2: {e}")
            
            try:
                is_bowlC_grasped = self.right_agent.is_grasping(self.bowl3)
            except (RuntimeError, Exception) as e:
                is_bowlC_grasped = torch.zeros(len(bowlC_placed), dtype=torch.bool, device=self.device)
                if hasattr(self, 'debug') and self.debug:
                    print(f"Warning: is_grasping failed for bowl3: {e}")
            success = (
                is_bowlC_on_bowlB * is_bowlB_on_bowlA * bowlC_placed * bowlB_placed * (~is_bowlA_grasped) * (~is_bowlB_grasped) * (~is_bowlC_grasped)
            )
            return {
                "is_bowlA_grasped": is_bowlA_grasped,
                "is_bowlB_grasped": is_bowlB_grasped,
                "is_bowlC_grasped": is_bowlC_grasped,
                "is_bowlC_on_bowlB": is_bowlC_on_bowlB,
                "is_bowlB_on_bowlA": is_bowlB_on_bowlA,
                "bowlB_placed": bowlB_placed,
                "success": success.bool(),
            }
        
        elif self.status == "place_factory_cube_123_123" or self.status == "place_factory_cube_123_132" or self.status == "place_factory_cube_123_213":
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
        
        elif self.status in ("place_factory_object_123", "place_factory_object_132", "place_factory_object_213"):
            # Check if can1, teabox2, hamburg3 are all within a rectangle centered at tray
            tray_x = self.tray.pose.p[..., 0]
            tray_y = self.tray.pose.p[..., 1]
            
            x_half = 0.1 / 2  # 0.05
            y_half = 0.18 / 2  # 0.09
            
            def check_obj_in_range(obj):
                obj_x = obj.pose.p[..., 0]
                obj_y = obj.pose.p[..., 1]
                x_in_range = torch.abs(obj_x - tray_x) <= x_half
                y_in_range = torch.abs(obj_y - tray_y) <= y_half
                return x_in_range & y_in_range
            
            obj1_success = check_obj_in_range(self.can1)
            obj2_success = check_obj_in_range(self.teabox2)
            obj3_success = check_obj_in_range(self.hamburg3)
            
            success = obj1_success & obj2_success & obj3_success
            return {
                "success": success,
            }
        
        elif self.status == "place_factory_cup_123":
            success = self.cup1.pose.p[..., 2] > self.agent.agents[0].robot.pose.p[0, 2] 
            return {
            "success": success,
        }
        else:
            # Default fallback for unknown status
            return {
                "success": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            }
    def _get_obs_extra(self, info: Dict):
        return {}

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return 0

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return 0
    

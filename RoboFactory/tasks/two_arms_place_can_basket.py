from typing import Any, Dict, Tuple

import numpy as np
import sapien
import torch
import json
import yaml

from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SimConfig
from transforms3d.euler import euler2quat
from mani_skill.utils.structs.pose import Pose
import utils.scenes

@register_env("TwoArmsPlaceCanBasket-rf", max_episode_steps=800)
class TwoArmsPlaceCanBasketEnv(BaseEnv):
    SUPPORTED_ROBOTS = [("panda", "panda")]
    agent: MultiAgent[Tuple[Panda, Panda]]
    
    goal_thresh = 0.025
    cube_color = np.concatenate((np.array([187, 116, 175]) / 255, [1]))
    light_cube_color = np.concatenate((np.array([187, 116, 175]) / 255, [0.5]))
    cube_half_size = 0.02

    def __init__(
        self, *args, robot_uids=("panda", "panda"), robot_init_qpos_noise=0.02, **kwargs
    ):
        self.status = kwargs.pop('status', None)
        self.domain_randomization = kwargs.pop('domain_randomization', {})
        assert 'config' in kwargs
        with open(kwargs['config'], 'r', encoding='utf-8') as f:
            self.cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
        del kwargs['config']
        if self.domain_randomization:
            self.cfg['domain_randomization'] = self.domain_randomization
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        camera_cfg = self.cfg.get('cameras', {})
        sensor_cfg = camera_cfg.get('sensor', [])
        all_camera_configs =[]
        for sensor in sensor_cfg:
            pose = sensor['pose']
            if not isinstance(pose, sapien.Pose):
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
        camera_cfg = self.cfg.get('cameras', {})
        render_cfg = camera_cfg.get('human_render', [])
        all_camera_configs =[]
        for render in render_cfg:
            pose = render['pose']
            if not isinstance(pose, sapien.Pose):
                if pose['type'] == 'pose':
                    render['pose'] = sapien.Pose(*pose['params'])
                elif pose['type'] == 'look_at':
                    render['pose'] = sapien_utils.look_at(*pose['params'])
            all_camera_configs.append(CameraConfig(**render))
        return all_camera_configs

    # @property
    # def _default_sim_config(self):
    #     return SimConfig(
    #         sim_freq=20,
    #         gpu_memory_config=GPUMemoryConfig(
    #             found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18
    #         )
    #     )
    
    def _load_agent(self, options: dict):
        init_poses = []
        for agent_cfg in self.cfg['agents']:
            init_poses.append(sapien.Pose(p=agent_cfg['pos']['ppos']['p']))
        super()._load_agent(options, init_poses)

    def _load_scene(self, options: dict):
        scene_name = self.cfg['scene']['name']
        scene_builder = getattr(utils.scenes, f'{scene_name}SceneBuilder')
        self.scene_builder = scene_builder(env=self, cfg=self.cfg)
        self.scene_builder.build()

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            self.scene_builder.initialize(env_idx)
        # alignment between pot and meat
        # pot_ppose = self.pot.pose.p
        # self.meat.pose.p[:, 0] = pot_ppose[:, 0]
        # self.meat.pose.p[:, 1] = pot_ppose[:, 1] - 0.25
        # self.meat.set_pose(self.meat.pose)

    def evaluate(self):
        if self.status == "two_arms_place_can_basket":
            can_pose = self.can.pose.p
            basket_pose = self.basket.pose.p
            distance = torch.linalg.norm(can_pose[..., :2] - basket_pose[..., :2], axis=1)
            can_in_basket = distance < 0.1
            basket_lifted = (
                self.basket.pose.p[..., 2]
                > 0.1 + self.agent.agents[1].robot.pose.p[0, 2]
            )
            success = can_in_basket & basket_lifted
            is_can_grasped_left = self.agent.agents[0].is_grasping(self.can)
            is_can_grasped_right = self.agent.agents[1].is_grasping(self.can)
            is_basket_grasped_left = self.agent.agents[0].is_grasping(self.basket)
            is_basket_grasped_right = self.agent.agents[1].is_grasping(self.basket)
            is_can_grasped = is_can_grasped_left
            is_basket_grasped = is_basket_grasped_right
            return {
                "is_can_grasped": is_can_grasped,
                "is_basket_grasped": is_basket_grasped,
                "is_can_grasped_left": is_can_grasped_left,
                "is_can_grasped_right": is_can_grasped_right,
                "is_basket_grasped_left": is_basket_grasped_left,
                "is_basket_grasped_right": is_basket_grasped_right,
                "can_in_basket": can_in_basket,
                "can_to_basket_distance": distance,
                "basket_lifted": basket_lifted,
                "success": success,
            }
        return {
            "success": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        }

    def _get_obs_extra(self, info: Dict):
        return {}

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return {}

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return {}


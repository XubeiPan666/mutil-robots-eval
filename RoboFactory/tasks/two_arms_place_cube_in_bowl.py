from typing import Any, Dict, Tuple

import numpy as np
import sapien
import torch
import yaml
from transforms3d.euler import euler2quat

from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.agents.robots.panda import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils.randomization.pose import random_quaternions
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
import utils.scenes


@register_env("TwoArmsCubeInBowl-rf", max_episode_steps=800)
class TwoArmsCubeInBowlEnv(BaseEnv):
    """Two-arm task: agent1 (right) places bowl at center, then agent0 (left) places cube inside bowl."""

    SUPPORTED_ROBOTS = [("panda", "panda")]
    agent: MultiAgent[Tuple[Panda, Panda]]

    goal_radius = 0.11

    def __init__(
        self,
        *args,
        robot_uids=("panda", "panda"),
        robot_init_qpos_noise=0.02,
        **kwargs
    ):
        self.status = kwargs.pop('status', None)
        self.domain_randomization = kwargs.pop('domain_randomization', {})
        self.robot_init_qpos_noise = robot_init_qpos_noise
        assert "config" in kwargs
        with open(kwargs["config"], "r", encoding="utf-8") as f:
            self.cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
        del kwargs["config"]
        if self.domain_randomization:
            self.cfg['domain_randomization'] = self.domain_randomization
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        camera_cfg = self.cfg.get('cameras', {})
        sensor_cfg = camera_cfg.get('sensor', [])
        all_camera_configs = []
        for sensor in sensor_cfg:
            pose = sensor['pose']
            if not isinstance(pose, sapien.Pose):
                if pose['type'] == 'pose':
                    sensor['pose'] = sapien.Pose(*pose['params'])
                elif pose['type'] == 'look_at':
                    sensor['pose'] = sapien_utils.look_at(*pose['params'])
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
        all_camera_configs = []
        for render in render_cfg:
            pose = render['pose']
            if not isinstance(pose, sapien.Pose):
                if pose['type'] == 'pose':
                    render['pose'] = sapien.Pose(*pose['params'])
                elif pose['type'] == 'look_at':
                    render['pose'] = sapien_utils.look_at(*pose['params'])
            all_camera_configs.append(CameraConfig(**render))
        return all_camera_configs

    def _load_agent(self, options: dict):
        init_poses = []
        for agent_cfg in self.cfg["agents"]:
            init_poses.append(sapien.Pose(p=agent_cfg["pos"]["ppos"]["p"]))
        super()._load_agent(options, init_poses)

    def _load_scene(self, options: dict):
        self.cube_half_size = common.to_tensor([0.02] * 3, device=self.device)
        self.bowl_half_size = common.to_tensor([0.08, 0.08, 0.016], device=self.device)
        scene_name = self.cfg["scene"]["name"]
        scene_builder = getattr(utils.scenes, f"{scene_name}SceneBuilder")
        self.scene_builder = scene_builder(env=self, cfg=self.cfg)
        self.scene_builder.build()

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            self.scene_builder.initialize(env_idx)

    @property
    def left_agent(self) -> Panda:
        return self.agent.agents[0]

    @property
    def right_agent(self) -> Panda:
        return self.agent.agents[1]

    def evaluate(self):
        if self.status and "two_arms_place_cube_in_bowl" in self.status:
            bowl_pos = self.bowl.pose.p
            cube_pos = self.cubeA.pose.p

            # Check bowl is placed on the tray (center of table)
            goal_x = self.tray.pose.p[..., 0]
            goal_y = self.tray.pose.p[..., 1]
            x_half = 0.1 / 2
            y_half = 0.18 / 2
            bowl_placed = (
                (torch.abs(bowl_pos[..., 0] - goal_x) <= x_half) &
                (torch.abs(bowl_pos[..., 1] - goal_y) <= y_half)
            )

            # Check cube is inside the bowl: xy close to bowl center, z within bowl cavity
            xy_dist = torch.linalg.norm(cube_pos[..., :2] - bowl_pos[..., :2], dim=-1)
            cube_in_bowl_xy = xy_dist < 0.06
            # Cube should be at approximately bowl surface height or slightly above
            z_diff = cube_pos[..., 2] - bowl_pos[..., 2]
            cube_in_bowl_z = (z_diff >= -0.01) & (z_diff <= 0.08)
            cube_in_bowl = cube_in_bowl_xy & cube_in_bowl_z

            is_bowl_grasped_left = self.left_agent.is_grasping(self.bowl)
            is_bowl_grasped_right = self.right_agent.is_grasping(self.bowl)
            is_cube_grasped_left = self.left_agent.is_grasping(self.cubeA)
            is_cube_grasped_right = self.right_agent.is_grasping(self.cubeA)
            is_bowl_grasped = is_bowl_grasped_right
            is_cube_grasped = is_cube_grasped_left

            success = bowl_placed & cube_in_bowl & (~is_bowl_grasped) & (~is_cube_grasped)

            return {
                "is_bowl_grasped": is_bowl_grasped,
                "is_cube_grasped": is_cube_grasped,
                "is_bowl_grasped_left": is_bowl_grasped_left,
                "is_bowl_grasped_right": is_bowl_grasped_right,
                "is_cube_grasped_left": is_cube_grasped_left,
                "is_cube_grasped_right": is_cube_grasped_right,
                "bowl_placed": bowl_placed,
                "cube_in_bowl": cube_in_bowl,
                "cube_in_bowl_xy": cube_in_bowl_xy,
                "cube_in_bowl_z": cube_in_bowl_z,
                "success": success.bool(),
            }
        return {"success": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)}

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            left_arm_tcp=self.left_agent.tcp.pose.raw_pose,
            right_arm_tcp=self.right_agent.tcp.pose.raw_pose,
        )
        if "state" in self.obs_mode:
            obs.update(
                bowl_pose=self.bowl.pose.raw_pose,
                cubeA_pose=self.cubeA.pose.raw_pose,
                left_arm_tcp_to_cubeA_pos=self.cubeA.pose.p - self.left_agent.tcp.pose.p,
                right_arm_tcp_to_bowl_pos=self.bowl.pose.p - self.right_agent.tcp.pose.p,
                cubeA_to_bowl_pos=self.bowl.pose.p - self.cubeA.pose.p,
            )
        return obs

    def _load_lighting(self, options: dict):
        dr = self.cfg.get('domain_randomization', {})
        shadow = self.enable_shadow

        if dr.get('random_light', False):
            ambient = np.clip(np.random.rand(3) * 0.4 + 0.2, 0, 1).tolist()
            self.scene.set_ambient_light(ambient)
            dir1_color = np.random.rand(3).tolist()
            dir2_color = np.random.rand(3).tolist()
            self.scene.add_directional_light(
                [1, 1, -1], dir1_color, shadow=shadow, shadow_scale=5, shadow_map_size=2048
            )
            self.scene.add_directional_light([0, 0, -1], dir2_color)
        else:
            super()._load_lighting(options)

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return {}

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return {}

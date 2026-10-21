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
# from mani_skill.utils.scene_builder.table import TableSceneBuilder
# from utils.scenes import TableSceneBuilder, RobocasaSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
import utils.scenes

@register_env("TwoArmsPushCube-rf", max_episode_steps=500)
class TwoArmsPushCubeEnv(BaseEnv):
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
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25,
                max_rigid_patch_count=2**19,
                max_rigid_contact_count=2**21,
            )
        )
    
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

    def _load_agent(self, options: dict):
        init_poses = []
        for agent_cfg in self.cfg["agents"]:
            init_poses.append(sapien.Pose(p=agent_cfg["pos"]["ppos"]["p"]))
        super()._load_agent(options, init_poses)


    def _load_scene(self, options: dict):
        self.cube_half_size = common.to_tensor([0.02] * 3, device=self.device)
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
        if self.status and "two_arms_push_cube" in self.status:
            is_cubeA_placed = (
                torch.linalg.norm(
                    self.cubeA.pose.p[..., :2] - self.goal_region.pose.p[..., :2], axis=1
                )
                < self.goal_radius
            ) & (self.cubeA.pose.p[..., 2] < self.cube_half_size[2] + 5e-3)

            is_cubeB_placed = (
                torch.linalg.norm(
                    self.cubeB.pose.p[..., :2] - self.goal_region.pose.p[..., :2], axis=1
                )
                < self.goal_radius
            ) & (self.cubeB.pose.p[..., 2] < self.cube_half_size[2] + 5e-3)

            success = is_cubeA_placed & is_cubeB_placed

            left_tcp = self.left_agent.tcp.pose.p
            right_tcp = self.right_agent.tcp.pose.p
            cubeA_xy = self.cubeA.pose.p[..., :2]
            cubeB_xy = self.cubeB.pose.p[..., :2]
            near_radius = 0.12
            agent0_near_cubeA = (
                torch.linalg.norm(left_tcp[..., :2] - cubeA_xy, axis=1) < near_radius
            )
            agent1_near_cubeA = (
                torch.linalg.norm(right_tcp[..., :2] - cubeA_xy, axis=1) < near_radius
            )
            agent0_near_cubeB = (
                torch.linalg.norm(left_tcp[..., :2] - cubeB_xy, axis=1) < near_radius
            )
            agent1_near_cubeB = (
                torch.linalg.norm(right_tcp[..., :2] - cubeB_xy, axis=1) < near_radius
            )

            return {
                "success": success,
                "is_cubeA_placed": is_cubeA_placed,
                "is_cubeB_placed": is_cubeB_placed,
                "cubeA_placed": is_cubeA_placed,
                "cubeB_placed": is_cubeB_placed,
                "cubeA_xy": cubeA_xy,
                "cubeB_xy": cubeB_xy,
                "agent0_near_cubeA": agent0_near_cubeA,
                "agent1_near_cubeA": agent1_near_cubeA,
                "agent0_near_cubeB": agent0_near_cubeB,
                "agent1_near_cubeB": agent1_near_cubeB,
                # Default agent mapping: cubeA->agent0, cubeB->agent1.
                "cubeA_pushed_by_agent0": is_cubeA_placed
                & (agent0_near_cubeA | (~agent1_near_cubeA)),
                "cubeB_pushed_by_agent1": is_cubeB_placed
                & (agent1_near_cubeB | (~agent0_near_cubeB)),
            }

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            left_arm_tcp=self.left_agent.tcp.pose.raw_pose,
            right_arm_tcp=self.right_agent.tcp.pose.raw_pose,
        )
        if "state" in self.obs_mode:
            obs.update(
                goal_region_pos=self.goal_region.pose.p,
                cubeA_pose=self.cubeA.pose.raw_pose,
                cubeB_pose=self.cubeB.pose.raw_pose,
                left_arm_tcp_to_cubeA_pos=self.cubeA.pose.p
                - self.left_agent.tcp.pose.p,
                right_arm_tcp_to_cubeB_pos=self.cubeB.pose.p
                - self.right_agent.tcp.pose.p,
                cubeA_to_cubeB_pos=self.cubeB.pose.p - self.cubeA.pose.p,
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

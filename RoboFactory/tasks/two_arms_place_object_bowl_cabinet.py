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


@register_env("TwoArmsPlaceObjectBowlCabinet-rf", max_episode_steps=800)
class TwoArmsPlaceObjectBowlCabinetEnv(BaseEnv):
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

    def _get_cabinet_handle_link(self, functional_point_id: int = 0):
        """Return the cabinet link containing the annotated drawer handle."""
        functional_points = self.annotation_data["cabinet"].get("functional_points", [])
        if functional_point_id >= len(functional_points):
            return None
        base_link_name = functional_points[functional_point_id].get("base")
        if base_link_name and hasattr(self.cabinet, "get_links"):
            for link in self.cabinet.get_links():
                if link.get_name() == base_link_name:
                    return link
        return None

    def _get_drawer_state(self, functional_point_id: int = 0):
        """Read the annotated drawer's qpos and normalized opening fraction."""
        handle_link = self._get_cabinet_handle_link(functional_point_id)
        if handle_link is not None:
            for joint in self.cabinet.get_active_joints():
                if joint.get_child_link().get_name() == handle_link.get_name():
                    joint_index = int(joint.active_index[0].item())
                    drawer_qpos = self.cabinet.get_qpos()[..., joint_index]
                    limits = self.cabinet.get_qlimits()[..., joint_index, :]
                    joint_range = (limits[..., 1] - limits[..., 0]).clamp_min(1e-6)
                    open_fraction = (
                        (drawer_qpos - limits[..., 0]) / joint_range
                    ).clamp(0.0, 1.0)
                    return drawer_qpos, open_fraction

        zeros = torch.zeros(self.num_envs, device=self.device)
        return zeros, zeros

    def _get_cabinet_target_pos(self, functional_point_id: int = 0):
        """Compute world-space target position from cabinet functional_points annotation."""
        cabinet_annotation = self.annotation_data['cabinet']
        if 'functional_points' not in cabinet_annotation:
            cabinet_pose = self.cabinet.pose.p
            target = cabinet_pose.clone()
            target[..., 0] += 0.1
            target[..., 2] += 0.1
            return target

        fp_data = cabinet_annotation['functional_points'][functional_point_id]
        local_matrix = np.asarray(fp_data['matrix'])

        scale = cabinet_annotation.get('scale', 1.0)
        if isinstance(scale, (list, np.ndarray)):
            scale = scale[0] if len(scale) > 0 else 1.0
        local_matrix[:3, 3] *= scale

        base_link_name = fp_data.get('base', None)
        if base_link_name and hasattr(self.cabinet, 'get_links'):
            link = None
            for l in self.cabinet.get_links():
                if l.get_name() == base_link_name:
                    link = l
                    break
            if link is not None:
                link_matrix = link.pose.to_transformation_matrix()
                if len(link_matrix.shape) == 3:
                    link_matrix = link_matrix[0]
                world_matrix = link_matrix.cpu().numpy() @ local_matrix
            else:
                cabinet_matrix = self.cabinet.pose.to_transformation_matrix()
                if len(cabinet_matrix.shape) == 3:
                    cabinet_matrix = cabinet_matrix[0]
                world_matrix = cabinet_matrix.cpu().numpy() @ local_matrix
        else:
            cabinet_matrix = self.cabinet.pose.to_transformation_matrix()
            if len(cabinet_matrix.shape) == 3:
                cabinet_matrix = cabinet_matrix[0]
            world_matrix = cabinet_matrix.cpu().numpy() @ local_matrix

        return torch.from_numpy(world_matrix[:3, 3]).to(device=self.device, dtype=torch.float32)

    def evaluate(self):
        if self.status is not None and "two_arms_place_object_bowl_cabinet" in self.status:
            bowl_pose = self.bowl.pose.p

            target_pos = self._get_cabinet_target_pos(functional_point_id=0)

            # Expand target to match batch dim if needed
            if len(bowl_pose.shape) > 1 and bowl_pose.shape[0] > 1:
                target_pos = target_pos.unsqueeze(0).expand(bowl_pose.shape[0], -1)

            tolerance = torch.tensor([0.06, 0.06], device=self.device, dtype=bowl_pose.dtype)

            bowl_xy_diff = torch.abs(bowl_pose[..., :2] - target_pos[..., :2])
            bowl_in_cabinet = torch.all(bowl_xy_diff < tolerance, dim=-1)

            is_bowl_grasped_left = self.left_agent.is_grasping(self.bowl)
            is_bowl_grasped_right = self.right_agent.is_grasping(self.bowl)
            is_bowl_grasped = is_bowl_grasped_left | is_bowl_grasped_right
            handle_link = self._get_cabinet_handle_link()
            is_handle_grasped = (
                self.right_agent.is_grasping(handle_link)
                if handle_link is not None
                else torch.zeros_like(is_bowl_grasped_left)
            )
            drawer_qpos, drawer_open_fraction = self._get_drawer_state()
            drawer_open = drawer_open_fraction >= 0.2

            success = bowl_in_cabinet & (~is_bowl_grasped)

            return {
                "bowl_in_cabinet": bowl_in_cabinet,
                "is_bowl_grasped": is_bowl_grasped,
                "is_bowl_grasped_left": is_bowl_grasped_left,
                "is_bowl_grasped_right": is_bowl_grasped_right,
                "is_handle_grasped": is_handle_grasped,
                "drawer_qpos": drawer_qpos,
                "drawer_open_fraction": drawer_open_fraction,
                "drawer_open": drawer_open,
                "bowl_xy_diff": bowl_xy_diff,
                "success": success.bool(),
            }

        return {"success": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)}

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            left_arm_tcp=self.left_agent.tcp.pose.raw_pose,
            right_arm_tcp=self.right_agent.tcp.pose.raw_pose,
        )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return {}

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return {}

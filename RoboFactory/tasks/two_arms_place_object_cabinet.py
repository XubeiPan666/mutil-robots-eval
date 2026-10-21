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


@register_env("TwoArmsPlaceObjectCabinet-rf", max_episode_steps=800)
class TwoArmsPlaceObjectCabinetEnv(BaseEnv):
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

    # @property
    # def _default_sim_config(self):
    #     return SimConfig(
    #         gpu_memory_config=GPUMemoryConfig(
    #             found_lost_pairs_capacity=2**25,
    #             max_rigid_patch_count=2**19,
    #             max_rigid_contact_count=2**21,
    #         )
    #     )
    
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
        fp_data = self.annotation_data["cabinet"].get("functional_points", [])[functional_point_id]
        base_link_name = fp_data.get("base")
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

    def evaluate(self):
        if self.status == "two_arms_place_object_cabinet":
            # 先获取物体位置，用于后续的 batch 维度处理
            object_pose = self.object.pose.p  # (torch tensor)
            
            # 手动获取 functional point（参考 ArticulationActor.get_point 实现）
            cabinet_annotation = self.annotation_data['cabinet']
            functional_point_id = 0
            
            # 检查是否有 functional_points
            if 'functional_points' in cabinet_annotation:
                # URDF 格式：从 functional_points 中获取
                fp_data = cabinet_annotation['functional_points'][functional_point_id]
                local_matrix = np.asarray(fp_data['matrix'])
                
                # 应用 scale
                scale = cabinet_annotation.get('scale', 1.0)
                if isinstance(scale, (list, np.ndarray)):
                    scale = scale[0] if len(scale) > 0 else 1.0
                local_matrix[:3, 3] *= scale
                
                # 获取对应的 link
                base_link_name = fp_data.get('base', None)
                if base_link_name and hasattr(self.cabinet, 'get_links'):
                    # 找到对应的 link
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
                        # 如果找不到 link，使用 cabinet 的 pose
                        cabinet_matrix = self.cabinet.pose.to_transformation_matrix()
                        if len(cabinet_matrix.shape) == 3:
                            cabinet_matrix = cabinet_matrix[0]
                        world_matrix = cabinet_matrix.cpu().numpy() @ local_matrix
                else:
                    # 没有 base 字段，使用 cabinet 的 pose
                    cabinet_matrix = self.cabinet.pose.to_transformation_matrix()
                    if len(cabinet_matrix.shape) == 3:
                        cabinet_matrix = cabinet_matrix[0]
                    world_matrix = cabinet_matrix.cpu().numpy() @ local_matrix
                
                # 提取位置并转换为 torch tensor
                target_pos = torch.from_numpy(world_matrix[:3, 3]).to(
                    device=self.device, dtype=object_pose.dtype
                )
                # 处理 batch 维度
                if len(object_pose.shape) > 1 and object_pose.shape[0] > 1:
                    target_pos = target_pos.unsqueeze(0).expand(object_pose.shape[0], -1)
                target_pose = target_pos
            else:
                # Fallback: 使用 cabinet 位置加上偏移
                cabinet_pose = self.cabinet.pose.p
                target_pose = cabinet_pose.clone()
                target_pose[..., 0] += 0.1  # Forward into cabinet
                target_pose[..., 2] += 0.1  # At appropriate height
            
            # 判断物体是否在目标位置的 XY 平面容差范围内
            # 转换为 torch 操作，容差为 [0.05, 0.05]
            xy_diff = torch.abs(object_pose[..., :2] - target_pose[..., :2])
            tolerance = torch.tensor([0.05, 0.05], device=self.device, dtype=object_pose.dtype)
            tag = torch.all(xy_diff < tolerance, dim=-1)
            
            # 检查夹爪是否打开（物体未被抓取）
            is_object_grasped_left = self.left_agent.is_grasping(self.object)
            is_object_grasped_right = self.right_agent.is_grasping(self.object)
            is_object_grasped = is_object_grasped_left | is_object_grasped_right
            handle_link = self._get_cabinet_handle_link()
            is_handle_grasped = (
                self.right_agent.is_grasping(handle_link)
                if handle_link is not None
                else torch.zeros_like(is_object_grasped_left)
            )
            drawer_qpos, drawer_open_fraction = self._get_drawer_state()
            drawer_open = drawer_open_fraction >= 0.2
            
            # 成功条件：物体在目标位置 XY 平面容差内，且未被抓取
            success = tag & (~is_object_grasped)
            
            return {
                "tag": tag,
                "is_object_grasped": is_object_grasped,
                "is_object_grasped_left": is_object_grasped_left,
                "is_object_grasped_right": is_object_grasped_right,
                "is_handle_grasped": is_handle_grasped,
                "drawer_qpos": drawer_qpos,
                "drawer_open_fraction": drawer_open_fraction,
                "drawer_open": drawer_open,
                "xy_diff": xy_diff,
                "success": success.bool(),
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

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return {}

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return {}

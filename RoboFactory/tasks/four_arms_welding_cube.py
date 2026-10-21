from typing import Any, Dict, Tuple

import numpy as np
import sapien
import torch
import yaml

from mani_skill.agents.robots import Fetch, Panda
from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from transforms3d.euler import euler2quat
from transforms3d.quaternions import quat2mat
import utils.scenes

@register_env("FourArmsWeldingCube-rf", max_episode_steps=1500)
class FourArmsWeldingCubeEnv(BaseEnv):

    SUPPORTED_ROBOTS = [("panda", "panda", "panda", "panda")]
    agent: MultiAgent[Tuple[Panda, Panda, Panda, Panda]]
    goal_thresh = 0.025
    cube_color = np.concatenate((np.array([187, 116, 175]) / 255, [1]))
    cube_half_size = 0.02

    def __init__(
        self, *args, robot_uids=("panda", "panda", "panda", "panda"), **kwargs
    ):
        self.status = kwargs.pop('status', None)
        assert 'config' in kwargs
        with open(kwargs['config'], 'r', encoding='utf-8') as f:
            self.cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
        del kwargs['config']
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        camera_cfg = self.cfg.get('cameras', {})
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
        camera_cfg = self.cfg.get('cameras', {})
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
            
            # Create welding spheres at the four corners of hollow_cube after it's initialized
            if hasattr(self, 'hollow_cube') and self.hollow_cube is not None:
                # Get hollow_cube parameters from config
                hollow_cube_cfg = None
                for primitive_cfg in self.cfg['scene']['primitives']:
                    if primitive_cfg['name'] == 'hollow_cube':
                        hollow_cube_cfg = primitive_cfg
                        break
                
                if hollow_cube_cfg:
                    half_size = hollow_cube_cfg['params']['half_size']
                    edge_thickness = hollow_cube_cfg['params'].get('edge_thickness', 0.01)
                    sphere_radius = edge_thickness / 2.0  # Sphere radius matches corner gap size
                    
                    # Get hollow_cube's world pose
                    cube_pose = self.hollow_cube.pose
                    # Handle both torch tensor and numpy array
                    if hasattr(cube_pose.p, 'cpu'):
                        cube_pos = cube_pose.p[0].cpu().numpy()
                    elif hasattr(cube_pose.p, '__getitem__'):
                        cube_pos = np.array(cube_pose.p[0]) if isinstance(cube_pose.p, (list, tuple)) else cube_pose.p[0]
                    else:
                        cube_pos = np.array([cube_pose.p.x, cube_pose.p.y, cube_pose.p.z])
                    
                    # Get cube rotation (quaternion) to transform local corners to world coordinates
                    if hasattr(cube_pose, 'q'):
                        if hasattr(cube_pose.q, 'cpu'):
                            cube_quat = cube_pose.q[0].cpu().numpy()
                        else:
                            cube_quat = np.array(cube_pose.q[0]) if hasattr(cube_pose.q, '__getitem__') else np.array([cube_pose.q.w, cube_pose.q.x, cube_pose.q.y, cube_pose.q.z])
                    else:
                        cube_quat = np.array([1.0, 0.0, 0.0, 0.0])  # No rotation
                    
                    # Calculate four corner positions in local coordinates (relative to cube center)
                    local_corners = [
                        [-half_size, -half_size, half_size],  # top-back-left
                        [ half_size, -half_size, half_size],  # top-back-right
                        [ half_size,  half_size, half_size],  # top-front-right
                        [-half_size,  half_size, half_size],  # top-front-left
                    ]
                    
                    # Initialize annotation_data if not exists
                    if not hasattr(self, 'annotation_data'):
                        self.annotation_data = {}
                    
                    # Create 4 welding spheres
                    sphere_names = ['welding_sphere_1', 'welding_sphere_2', 'welding_sphere_3', 'welding_sphere_4']
                    for i, (corner_name, local_corner) in enumerate(zip(sphere_names, local_corners)):
                        # Transform local corner position to world coordinates
                        # Apply rotation if cube is rotated
                        if len(cube_quat) == 4:
                            # quat2mat expects [w, x, y, z] format
                            rot_matrix = quat2mat(cube_quat)
                            rotated_corner = rot_matrix[:3, :3] @ np.array(local_corner)
                            world_corner = cube_pos + rotated_corner
                        else:
                            # No rotation, just translate
                            world_corner = cube_pos + np.array(local_corner)
                        
                        # Check if sphere already exists (from previous episode reset)
                        if hasattr(self, corner_name) and getattr(self, corner_name) is not None:
                            # Sphere already exists, just update its pose
                            sphere_actor = getattr(self, corner_name)
                            sphere_actor.set_pose(sapien.Pose(p=world_corner))
                        else:
                            # Create new sphere actor using ManiSkill's build_sphere
                            sphere_actor = actors.build_sphere(
                                scene=self.scene,
                                radius=sphere_radius,
                                name=corner_name,
                                color=[1.0, 0.0, 0.0, 0.8],  # Red color
                                body_type="dynamic",
                                add_collision=True,
                                initial_pose=sapien.Pose(p=world_corner),
                            )
                            
                            # Set sphere pose (in case initial_pose doesn't work)
                            sphere_actor.set_pose(sapien.Pose(p=world_corner))
                            
                            # Store sphere actor
                            setattr(self, corner_name, sphere_actor)
                        
                        # Create/update actor_data with contact points for grasping
                        # Contact point 0: top of sphere (Z+ direction)
                        contact_0_matrix = np.eye(4)
                        contact_0_matrix[:3, 3] = [0, 0, sphere_radius]  # Position at top
                        contact_0_matrix[:3, :3] = np.array([
                            [1, 0, 0],  # X axis
                            [0, 1, 0],  # Y axis
                            [0, 0, 1],  # Z axis (pointing outward, up)
                        ])
                        
                        # Contact point 1: bottom of sphere (Z- direction)
                        contact_1_matrix = np.eye(4)
                        contact_1_matrix[:3, 3] = [0, 0, -sphere_radius]  # Position at bottom
                        contact_1_matrix[:3, :3] = np.array([
                            [1, 0,  0],  # X axis
                            [0, 1,  0],  # Y axis
                            [0, 0, -1],  # Z axis (pointing outward, down)
                        ])
                        
                        sphere_actor_data = {
                            "contact_points_pose": [
                                contact_0_matrix.tolist(),
                                contact_1_matrix.tolist(),
                            ],
                            "scale": 1.0,
                        }
                        
                        # Store annotation data
                        self.annotation_data[corner_name] = sphere_actor_data
                        
                        # Add to movable_objects if exists
                        if hasattr(self.scene_builder, 'movable_objects'):
                            if self.scene_builder.movable_objects is None:
                                self.scene_builder.movable_objects = {}
                            self.scene_builder.movable_objects[corner_name] = sphere_actor


    def evaluate(self):
        if self.status == "four_arms_welding_cube_1234_1234" or self.status == "four_arms_welding_cube_1234_2341" or self.status == "four_arms_welding_cube_1234_4123":
            # success = abs(self.stick_1.pose.p[..., 0] - self.goal_region.pose.p[..., 0]) ** 2 + abs(self.stick_1.pose.p[..., 1] - self.goal_region.pose.p[..., 1]) ** 2 < 0.01
            half_cube = self.hollow_cube.pose.p[..., 2] / 2
            # success = self.welding_sphere_1.pose.p[..., 2] < half_cube and self.welding_sphere_2.pose.p[..., 2] < half_cube and self.welding_sphere_3.pose.p[..., 2] < half_cube and self.welding_sphere_4.pose.p[..., 2] < half_cube
            success = self.welding_sphere_2.pose.p[..., 2] < 10
        return {
            "success": success,
        }

    def _get_obs_extra(self, info: Dict):
        return {}

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return {}

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return {}
    

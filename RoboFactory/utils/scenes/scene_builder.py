# Local RoboFactory/MAS-VLA adaptation of ManiSkill utilities.
# Modified for multi-agent tasks and local scene/recording integration.
# See THIRD_PARTY_NOTICES.md and licenses/ for upstream licenses.

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Union, Optional
from functools import cached_property

import torch
import numpy as np
from gymnasium import spaces
import sapien
from transforms3d.euler import euler2quat

if TYPE_CHECKING:
    from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.structs import Actor, Articulation
from mani_skill.utils.structs.types import Array
from mani_skill.utils.structs.pose import Pose
from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.envs.utils.randomization.pose import random_quaternions



class SceneBuilder:
    """Base class for defining scene builders that can be reused across tasks"""

    env: BaseEnv
    """Env which scenebuilder will build in."""

    robot_init_qpos_noise: float = 0.02
    """Robot init qpos noise"""

    builds_lighting: bool = False
    """Whether this scene builder will add its own lighting when build is called. If False, ManiSkill will add some default lighting"""

    build_configs: Optional[List[Any]] = None
    """List of scene configuration information that can be used to **build** scenes during reconfiguration (i.e. `env.reset(seed=seed, options=dict(reconfigure=True))`). Can be a dictionary, a path to a json file, or some other data. If a scene needs to load build config data, it will index/sample such build configs from this list."""
    init_configs: Optional[List[Any]] = None
    """List of scene configuration information that can be used to **init** scenes during reconfiguration (i.e. `env.reset()`). Can be a dictionary, a path to a json file, or some other data. If a scene needs to load init config data, it will index/sample such init configs from this list."""

    scene_objects: Optional[Dict[str, Actor]] = None
    """Scene objects are any dynamic, kinematic, or static Actor built by the scene builder. Useful for accessing objects in the scene directly."""
    movable_objects: Optional[Dict[str, Actor]] = None
    """Movable objects are any **dynamic** Actor built by the scene builder. movable_objects is a subset of scene_objects. Can be used to query dynamic objects for e.g. task initialization."""
    articulations: Optional[Dict[str, Articulation]] = None
    """Articulations are any articulation loaded in by the scene builder."""

    navigable_positions: Optional[List[Union[Array, spaces.Box]]] = None
    """Some scenes allow for mobile robots to move through these scene. In this case, a list of navigable positions per env_idx (e.g. loaded from a navmesh) should be provided for easy initialization. Can be a discretized list, range, spaces.Box, etc."""

    def __init__(self, env, robot_init_qpos_noise=0.02):
        self.env = env
        self.robot_init_qpos_noise = robot_init_qpos_noise

    def build(self, build_config_idxs: List[int] = None):
        """
        Should create actor/articulation builders and only build objects into the scene without initializing pose, qpos, velocities etc.
        """
        raise NotImplementedError()

    def initialize(self, env_idx: torch.Tensor, init_config_idxs: List[int] = None):
        """
        Should initialize the scene, which can include e.g. setting the pose of all objects, changing the qpos/pose of articulations/robots etc.
        """
        raise NotImplementedError()

    def sample_build_config_idxs(self) -> List[int]:
        """
        Sample idxs of build configs for easy scene randomization. Should be changed to fit shape of self.build_configs.
        """
        return torch.randint(
            low=0, high=len(self.build_configs), size=(self.env.num_envs,)
        ).tolist()

    def sample_init_config_idxs(self) -> List[int]:
        """
        Sample idxs of init configs for easy scene randomization. Should be changed to fit shape of self.init_configs.
        """
        return torch.randint(
            low=0, high=len(self.init_configs), size=(self.env.num_envs,)
        ).tolist()

    @cached_property
    def build_config_names_to_idxs(self) -> Dict[str, int]:
        return dict((v, i) for i, v in enumerate(self.build_configs))

    @cached_property
    def init_config_names_to_idxs(self) -> Dict[str, int]:
        return dict((v, i) for i, v in enumerate(self.init_configs))

    @property
    def scene(self):
        return self.env.scene


class RFSceneBuilder(SceneBuilder):
    def __init__(self, env, cfg, **kwargs):
        super().__init__(env, **kwargs)
        self.cfg = cfg
        self.env.annotation_data = {}

    @staticmethod
    def _pose_to_lists(pose_obj):
        """Convert a sapien Pose (or similar) to plain Python float lists."""
        def _to_floats(x):
            try:
                import torch
                if isinstance(x, torch.Tensor):
                    return x.detach().cpu().numpy().flatten().tolist()
            except ImportError:
                pass
            try:
                return np.array(x).flatten().tolist()
            except Exception:
                return [float(v) for v in x]
        return {"p": _to_floats(pose_obj.p), "q": _to_floats(pose_obj.q)}

    def initialize(self, env_idx: torch.Tensor):
        replayer = getattr(self.env, '_perturb_replayer', None)
        recorder = getattr(self.env, '_perturb_recorder', None)
        b = len(env_idx)
        scene_cfg = self.cfg['scene']

        # --- REPLAY: apply recorded poses and return early -------------------
        if replayer is not None:
            self._initialize_from_replay(replayer, b)
            return

        # --- ORIGINAL randomisation ------------------------------------------

        # primitive
        if 'primitives' in scene_cfg:
            for primitive_cfg in scene_cfg['primitives']:
                asset = getattr(self.env, primitive_cfg['name'], None)
                if not asset:
                    raise AttributeError(f'Attribute "{primitive_cfg["name"]}" not found in SceneBuilder.')
                ppos = primitive_cfg['pos']['ppos']['p']
                if 'randp_scale' in primitive_cfg['pos']:
                    if isinstance(primitive_cfg['pos']['randp_scale'], list):
                        selected_idx = np.random.randint(0, len(primitive_cfg['pos']['randp_scale']))
                        ppos = np.array(ppos) + np.array(primitive_cfg['pos']['randp_scale'][selected_idx]) * np.random.rand((len(ppos)))
                        ppos = ppos.tolist()
                    else:
                        ppos = np.array(ppos) + np.array(primitive_cfg['pos']['randp_scale']) * np.random.rand((len(ppos)))
                        ppos = ppos.tolist()
                qpos = np.array(primitive_cfg['pos']['qpos'])
                if 'randq_scale' in primitive_cfg:
                    qpos = np.array(qpos) + np.array(primitive_cfg['pos']['randq_scale'])* np.random.rand((len(qpos)))
                if 'random_quaternions' in primitive_cfg['pos']:
                    qpos = random_quaternions(
                        b,
                        lock_x=primitive_cfg['pos']['random_quaternions'][0],
                        lock_y=primitive_cfg['pos']['random_quaternions'][1],
                        lock_z=primitive_cfg['pos']['random_quaternions'][2]
                    )
                asset.set_pose(Pose.create_from_pq(ppos, qpos))

        # objects
        if 'objects' in self.cfg:
            objects_cfg = self.cfg['objects']
            self.movable_objects = {}
            for asset_cfg in objects_cfg:
                asset = getattr(self.env, asset_cfg['name'], None)
                if not asset:
                    raise AttributeError(f'Attribute "{asset_cfg["name"]}" not found in SceneBuilder.')
                base_ppos = np.array(asset_cfg['pos']['ppos']['p'])
                randp_scale_raw = asset_cfg['pos'].get('randp_scale', [0., 0., 0.])
                    
                # Support selecting from a list of randp_scale options
                # If randp_scale is a list of lists, randomly select one
                if isinstance(randp_scale_raw, list) and len(randp_scale_raw) > 0:
                    if isinstance(randp_scale_raw[0], (list, np.ndarray, tuple)):
                        # It's a list of options, randomly select one
                        selected_scale = randp_scale_raw[np.random.randint(len(randp_scale_raw))]
                        randp_scale = np.array(selected_scale)
                    else:
                        # It's a single vector, use it directly
                        randp_scale = np.array(randp_scale_raw)
                else:
                    randp_scale = np.array(randp_scale_raw)
                
                # Calculate initial position with random offset
                ppos = base_ppos.copy()
                if 'randp_scale' in asset_cfg['pos']:
                    ppos = base_ppos + randp_scale * np.random.rand((len(base_ppos)))
                    ppos = ppos.tolist()
                qpos = asset_cfg['pos']['qpos']
                if 'randq_scale' in asset_cfg['pos']:
                    qpos = np.array(qpos) + np.array(asset_cfg['pos']['randq_scale'])* np.random.rand((len(qpos)))
                if 'random_quaternions' in asset_cfg['pos']:
                    qpos = random_quaternions(
                        b,
                        lock_x=asset_cfg['pos']['random_quaternions'][0],
                        lock_y=asset_cfg['pos']['random_quaternions'][1],
                        lock_z=asset_cfg['pos']['random_quaternions'][2]
                    )
                asset.set_pose(Pose.create_from_pq(ppos, qpos))
                # Optional articulation joint initialization for URDF-like objects.
                # Keep backward compatibility: only apply when config provides joint_qpos.
                if 'joint_qpos' in asset_cfg and hasattr(asset, "set_qpos"):
                    joint_qpos = np.array(asset_cfg['joint_qpos'])
                    if joint_qpos.size > 0:
                        asset.set_qpos(joint_qpos)
                self.movable_objects[asset_cfg['name']] = asset
        # agents
        if 'agents' in self.cfg:
            agents_cfg = self.cfg['agents']
            is_multi_agent = (len(agents_cfg) > 1)
            agent = self.env.agent
            self.articulations = {}
            for idx, agent_cfg in enumerate(agents_cfg):
                pos_cfg = agent_cfg['pos']
                ppos = pos_cfg['ppos']['p']
                if 'randp_scale' in pos_cfg:
                    ppos = np.array(ppos) + np.array(agent_cfg['pos']['randp_scale']) * np.random.rand((len(ppos)))
                    ppos = ppos.tolist()
                ppos = sapien.Pose(ppos, q=euler2quat(*pos_cfg['ppos']['q']))
                qpos = np.array((pos_cfg['qpos']))
                if 'randq_scale' in pos_cfg:
                    qpos = np.tile(qpos, (b, 1)) + np.tile(np.array(agent_cfg['pos']['randq_scale']), (b, 1)) * np.random.rand(b, (len(qpos)))
                if 'random_quaternions' in agent_cfg['pos']:
                    qpos = random_quaternions(
                        b,
                        lock_x=agent_cfg['pos']['random_quaternions'][0],
                        lock_y=agent_cfg['pos']['random_quaternions'][1],
                        lock_z=agent_cfg['pos']['random_quaternions'][2]
                    )
                if is_multi_agent:
                    agent.agents[idx].reset(qpos)
                    agent.agents[idx].robot.set_pose(ppos)
                    self.articulations[agent_cfg['robot_uid']] = agent.agents[idx]
                else:
                    agent.reset(qpos)
                    agent.robot.set_pose(ppos)
                    self.articulations[agent_cfg['robot_uid']] = agent

        # --- RECORD: snapshot final poses ------------------------------------
        if recorder is not None:
            self._record_final_poses(recorder)

    def _initialize_from_replay(self, replayer, b: int):
        """Apply recorded scene state without any np.random.* calls."""
        scene_cfg = self.cfg['scene']

        # Primitives: apply base poses from config (no randomness replayed for primitives)
        if 'primitives' in scene_cfg:
            for primitive_cfg in scene_cfg['primitives']:
                asset = getattr(self.env, primitive_cfg['name'], None)
                if not asset:
                    continue
                ppos = primitive_cfg['pos']['ppos']['p']
                qpos = np.array(primitive_cfg['pos']['qpos'])
                asset.set_pose(Pose.create_from_pq(ppos, qpos))

        # Movable objects: apply recorded poses
        movable_poses = replayer.get_movable_poses()
        if 'objects' in self.cfg:
            self.movable_objects = {}
            for asset_cfg in self.cfg['objects']:
                name = asset_cfg['name']
                asset = getattr(self.env, name, None)
                if asset is None:
                    raise AttributeError(f'Attribute "{name}" not found in env.')
                if name in movable_poses:
                    rp = movable_poses[name]
                    asset.set_pose(Pose.create_from_pq(rp['p'], rp['q']))
                else:
                    ppos = asset_cfg['pos']['ppos']['p']
                    qpos = asset_cfg['pos']['qpos']
                    asset.set_pose(Pose.create_from_pq(ppos, qpos))
                if 'joint_qpos' in asset_cfg and hasattr(asset, "set_qpos"):
                    joint_qpos = np.array(asset_cfg['joint_qpos'])
                    if joint_qpos.size > 0:
                        asset.set_qpos(joint_qpos)
                self.movable_objects[name] = asset

        # Agents: apply recorded qpos and base poses
        agent_poses = replayer.get_agent_poses()
        agent_qpos_dict = replayer.get_agent_qpos()
        if 'agents' in self.cfg:
            agents_cfg = self.cfg['agents']
            is_multi_agent = len(agents_cfg) > 1
            agent = self.env.agent
            self.articulations = {}
            for idx, agent_cfg in enumerate(agents_cfg):
                key = agent_cfg['robot_uid']
                pos_cfg = agent_cfg['pos']
                if key in agent_poses:
                    ap = agent_poses[key]
                    ppos = sapien.Pose(ap['p'], ap['q'])
                else:
                    ppos = sapien.Pose(pos_cfg['ppos']['p'], q=euler2quat(*pos_cfg['ppos']['q']))
                if key in agent_qpos_dict:
                    qpos = np.array(agent_qpos_dict[key])
                else:
                    qpos = np.array(pos_cfg['qpos'])
                if is_multi_agent:
                    agent.agents[idx].reset(qpos)
                    agent.agents[idx].robot.set_pose(ppos)
                    self.articulations[key] = agent.agents[idx]
                else:
                    agent.reset(qpos)
                    agent.robot.set_pose(ppos)
                    self.articulations[key] = agent

    def _record_final_poses(self, recorder):
        """Snapshot final poses of all movable objects and agents."""
        movable_poses = {}
        if hasattr(self, 'movable_objects') and self.movable_objects:
            for name, actor in self.movable_objects.items():
                try:
                    movable_poses[name] = self._pose_to_lists(actor.pose)
                except Exception:
                    pass
        recorder.record_movable_poses(movable_poses)

        agent_poses = {}
        agent_qpos_dict = {}
        if hasattr(self, 'articulations') and self.articulations:
            for key, art in self.articulations.items():
                robot = getattr(art, 'robot', art)
                try:
                    agent_poses[key] = self._pose_to_lists(robot.pose)
                except Exception:
                    pass
                try:
                    qpos_val = robot.qpos
                    try:
                        import torch
                        if isinstance(qpos_val, torch.Tensor):
                            qpos_val = qpos_val.detach().cpu().numpy()
                    except ImportError:
                        pass
                    agent_qpos_dict[key] = np.array(qpos_val).flatten().tolist()
                except Exception:
                    pass
        recorder.record_agent_poses(agent_poses)
        recorder.record_agent_qpos(agent_qpos_dict)


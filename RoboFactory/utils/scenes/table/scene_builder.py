import os
import os.path as osp
import importlib
import json
import re
import copy
from pathlib import Path
from typing import List
import numpy as np
import sapien
import sapien.render
import sapien.pysapien.physx
import torch
from transforms3d.euler import euler2quat

from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.agents.robots.fetch import FETCH_WHEELS_COLLISION_BIT
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.scene_builder.scene_builder import SceneBuilder
from mani_skill.utils.structs.pose import Pose

from ..scene_builder import SceneBuilder, RFSceneBuilder

class TableSceneBuilder(RFSceneBuilder):
    def build(self):
        # scene
        scene_cfg = self.cfg['scene']
        altitude = 0
        self.scene_objects = {}
        if 'primitives' in scene_cfg:
            for primitive_cfg in scene_cfg['primitives']:
                primitive_name = primitive_cfg['name']
                builder_module_name, builder_class_name = primitive_cfg['builder'].rsplit('.', maxsplit=1)
                builder_module = importlib.import_module(builder_module_name)
                builder = getattr(builder_module, builder_class_name)
                params = copy.deepcopy(primitive_cfg['params'])
                if 'initial_pose' in params:
                    ip = params['initial_pose']
                    if not isinstance(ip, sapien.Pose):
                        params['initial_pose'] = sapien.Pose(p=ip['p'])
                primitive = builder(self.env.scene, **params)
                setattr(self.env, primitive_name, primitive)
                self.scene_objects[primitive_name] = getattr(self.env, primitive_name)
        if 'assets' in scene_cfg:        
            for asset_cfg in scene_cfg['assets']:
                asset_file = asset_cfg['file_path']
                asset_type = osp.splitext(asset_file)[-1]
                if asset_type in ['.obj', '.glb']:
                    builder = self.scene.create_actor_builder()
                    asset_name = asset_cfg['name']
                    # TODO: update different collision types
                    if asset_cfg['collision']['type'] == 'box':
                        builder.add_box_collision(
                            pose=sapien.Pose(p=asset_cfg['collision']['pos']['p']),
                            half_size=asset_cfg['collision']['pos']['half_size'],
                    )
                    temp_pose = sapien.Pose(q=euler2quat(*asset_cfg['pos']['ppos']['q']))
                    builder.add_visual_from_file(
                        filename=asset_file, scale=asset_cfg['scale'], pose=temp_pose
                    )
                    initial_ppos = asset_cfg['pos']['ppos']['p']
                    if 'randp_scale' in asset_cfg['pos']:
                        initial_ppos = np.array(initial_ppos) + np.array(asset_cfg['pos']['randp_scale']) * np.random.rand((len(initial_ppos)))
                        initial_ppos = initial_ppos.tolist()
                    builder.initial_pose = sapien.Pose(
                        p=asset_cfg['pos']['ppos']['p'], q=euler2quat(*asset_cfg['pos']['ppos']['q'])
                    )
                    setattr(self.env, asset_name, builder.build_kinematic(name=f"{scene_cfg['name']}-Workspace"))
                    aabb = (
                        getattr(self.env, asset_name)._objs[0]
                        .find_component_by_type(sapien.render.RenderBodyComponent)
                        .compute_global_aabb_tight()
                    )
                    height = aabb[1, 2] - aabb[0, 2]
                    altitude = min(altitude, -height)    # make the plane at 0 in z
                    self.scene_objects[asset_name] = getattr(self.env, asset_name)
                elif asset_type in ['.urdf']:
                    pass
        self.ground = build_ground(
            self.scene, floor_width=scene_cfg['env']['floor_width'], altitude=altitude
        )
        self.scene_objects['ground'] = self.ground

        # objects
        if 'objects' in self.cfg:
            objects_cfg = self.cfg['objects']
            self.movable_objects = {}
            for object_cfg in objects_cfg:
                object_file_path = object_cfg['file_path']
                object_type = osp.splitext(object_file_path)[-1]
                object_name = object_cfg['name']
                object_annotation_path = object_cfg['annotation_path']
                with open(object_annotation_path, 'r') as f:
                    object_annotation_data = json.load(f)
                self.env.annotation_data[object_name] = object_annotation_data
                if object_type in ['.obj', '.glb']:
                    builder = self.scene.create_actor_builder()
                    builder.set_physx_body_type("dynamic")
                    visual_params = {}
                    collision_params = {}
                    if 'material' in object_cfg:
                        physx, material = object_cfg['material']['type'].rsplit('.', maxsplit=1)
                        physx_module = importlib.import_module(physx)
                        material_builder = getattr(physx_module, material)
                        object_material = material_builder(**object_cfg['material']['params'])
                        collision_params['material'] = object_material
                    visual_cfg = object_cfg['visual']
                    visual_params.update(visual_cfg)
                    if object_cfg.get('collision', None):
                        collision_params.update(object_cfg['collision'])
                    else:
                        collision_params.update(visual_params)    # use visual cfg as default
                    if collision_params['type'] == 'nonconvex':
                        del(collision_params['type'])
                        builder.add_nonconvex_collision_from_file(**collision_params)
                    elif collision_params['type'] == 'multiple_convex':
                        del(collision_params['type'])
                        # 支持多凸分解，参数可选
                        # decomposition = collision_params.pop('decomposition', 'coacd')
                        # decomposition_params = collision_params.pop('decomposition_params', {})
                        builder.add_multiple_convex_collisions_from_file(
                            **collision_params,
                            # decomposition=decomposition,
                            # decomposition_params=decomposition_params
                        )
                    else:
                        del(collision_params['type'])
                        builder.add_convex_collision_from_file(**collision_params)
                    builder.add_visual_from_file(**visual_params)
                    if object_cfg.get('mass_params', None):
                        mass_params = copy.deepcopy(object_cfg['mass_params'])
                        if 'cmass_local_pose' in mass_params:
                            mass_params['cmass_local_pose'] = sapien.Pose(mass_params['cmass_local_pose'])
                        builder.set_mass_and_inertia(**mass_params)
                    setattr(self.env, object_name, builder.build(name=object_name))
                elif object_type in ['.urdf']:
                    # use nonconvex collision
                    def create_nonconvex_urdf_loader(scene):
                        from utils.building.nonconvex_urdf_loader import NonconvexURDFLoader
                        loader = NonconvexURDFLoader()
                        loader.set_scene(scene)
                        return loader
                    urdf_builder = create_nonconvex_urdf_loader(self.scene)
                    urdf_builder.fix_root_link = True
                    # urdf_builder.load_multiple_collisions_from_file = False
                    urdf_builder.load_multiple_collisions_from_file = object_cfg.get('load_multiple_collisions', False)
                    if 'scale' in object_cfg:
                        urdf_builder.scale = object_cfg['scale']
                    if 'density' in object_cfg:
                        urdf_builder._density = object_cfg['density']
                    setattr(self.env, object_name, urdf_builder.load(object_file_path))
                else:
                    raise ValueError
                self.movable_objects[object_name] = getattr(self.env, object_name)

        # domain randomization (visual perturbation for evaluation)
        self.domain_rand = self.cfg.get('domain_randomization', {})
        self.distractor_objects = {}
        if self.domain_rand.get('random_background', False):
            self._build_background()
        if self.domain_rand.get('cluttered_table', False):
            self._build_distractors()

    def _build_background(self):
        """Build tablecloth overlay with a random texture on the table surface."""
        replayer = getattr(self.env, '_perturb_replayer', None)
        recorder = getattr(self.env, '_perturb_recorder', None)

        # --- REPLAY: use recorded outcome, skip np.random.* entirely ----------
        if replayer is not None:
            use_clean, tex_abs_path = replayer.get_background()
            if not use_clean and tex_abs_path and os.path.exists(tex_abs_path):
                self._create_textured_plane(
                    name="tablecloth",
                    tex_path=tex_abs_path,
                    half_size=[0.65, 0.55, 0.0005],
                    position=[0.25, 0., 0.001],
                )
            return

        # --- ORIGINAL randomisation ------------------------------------------
        texture_root = self.domain_rand.get('background_texture_root', '')
        if not texture_root or not os.path.exists(texture_root):
            return

        texture_type = self.domain_rand.get('texture_type', 'seen')
        texture_dir = os.path.join(texture_root, texture_type)
        if not os.path.exists(texture_dir):
            texture_dir = texture_root
        texture_files = [f for f in os.listdir(texture_dir) if f.endswith('.png')]
        if not texture_files:
            return

        clean_rate = self.domain_rand.get('clean_background_rate', 0.02)

        if np.random.rand() > clean_rate:
            tex_path = os.path.join(texture_dir, np.random.choice(texture_files))
            self._create_textured_plane(
                name="tablecloth",
                tex_path=tex_path,
                half_size=[0.65, 0.55, 0.0005],
                position=[0.25, 0., 0.001],
            )
            if recorder is not None:
                tex_rel = "background_texture/" + os.path.relpath(
                    tex_path, texture_root
                ).replace(os.sep, "/")
                recorder.record_background(use_clean=False, texture_relpath=tex_rel)
        else:
            if recorder is not None:
                recorder.record_background(use_clean=True, texture_relpath=None)

    def _create_textured_plane(self, name, tex_path, half_size, position):
        """Create a kinematic plane actor with a texture applied."""
        builder = self.scene.create_actor_builder()
        render_mat = sapien.render.RenderMaterial()
        texture2d = sapien.render.RenderTexture2D(tex_path)
        render_mat.set_base_color_texture(texture2d)
        render_mat.base_color = [1, 1, 1, 1]
        render_mat.metallic = 0.0
        render_mat.roughness = 0.85
        builder.add_box_visual(half_size=half_size, material=render_mat)
        actor = builder.build_kinematic(name=name)
        actor.set_pose(sapien.Pose(p=position))
        setattr(self.env, name, actor)
        self.scene_objects[name] = actor

    def _build_distractors(self):
        """Build distractor objects from the OOD objects directory."""
        replayer = getattr(self.env, '_perturb_replayer', None)
        recorder = getattr(self.env, '_perturb_recorder', None)

        # --- REPLAY: rebuild from recorded selection, skip np.random.* -------
        if replayer is not None:
            selection = replayer.get_distractor_selection()
            for entry in selection:
                vis_path = replayer.get_distractor_visual_path(entry)
                if not vis_path or not os.path.exists(vis_path):
                    continue
                scale = entry.get('scale', [0.06, 0.06, 0.06])
                dist_name = entry['name']
                builder = self.scene.create_actor_builder()
                builder.add_visual_from_file(filename=vis_path, scale=scale)
                actor = builder.build_kinematic(name=dist_name)
                setattr(self.env, dist_name, actor)
                self.distractor_objects[dist_name] = actor
            return

        # --- ORIGINAL randomisation ------------------------------------------
        objects_root = self.domain_rand.get('distractor_objects_root', '')
        if not objects_root or not os.path.exists(objects_root):
            return
        num_distractors = self.domain_rand.get('num_distractors', 5)

        task_categories = set()
        if 'objects' in self.cfg:
            for obj in self.cfg['objects']:
                for part in obj.get('file_path', '').split('/'):
                    if re.match(r'\d{3}_', part):
                        task_categories.add(part)
                        break

        available = []
        for d in sorted(os.listdir(objects_root)):
            full = os.path.join(objects_root, d)
            if not os.path.isdir(full) or not re.match(r'\d{3}_', d):
                continue
            if d in task_categories:
                continue
            vis_dir = os.path.join(full, 'visual')
            col_dir = os.path.join(full, 'collision')
            if not os.path.isdir(vis_dir) or not os.path.isdir(col_dir):
                continue
            vis_files = [f for f in os.listdir(vis_dir) if f.endswith('.glb')]
            if vis_files:
                available.append((d, full, vis_files))

        if not available:
            return

        n_select = min(num_distractors, len(available))
        selected_indices = np.random.choice(len(available), size=n_select, replace=False)

        selection_entries = []
        for i, idx in enumerate(selected_indices):
            cat_name, cat_path, vis_files = available[idx]
            chosen_vis = np.random.choice(vis_files)
            variant = chosen_vis.replace('base', '').replace('.glb', '')
            col_file = f"base{variant}.glb"
            model_data_file = f"model_data{variant}.json"

            vis_path = os.path.join(cat_path, 'visual', chosen_vis)
            col_path = os.path.join(cat_path, 'collision', col_file)
            md_path = os.path.join(cat_path, model_data_file)

            if not os.path.exists(col_path):
                continue

            scale = [0.06, 0.06, 0.06]
            if os.path.exists(md_path):
                try:
                    with open(md_path, 'r') as f:
                        md = json.load(f)
                    if 'scale' in md:
                        s = md['scale']
                        if isinstance(s, list) and len(s) == 3:
                            scale = [float(v) for v in s]
                except Exception:
                    pass

            dist_name = f"distractor_{i}_{cat_name}"
            builder = self.scene.create_actor_builder()
            builder.add_visual_from_file(filename=vis_path, scale=scale)

            actor = builder.build_kinematic(name=dist_name)
            setattr(self.env, dist_name, actor)
            self.distractor_objects[dist_name] = actor

            if recorder is not None:
                vis_rel = "distractors/" + os.path.relpath(
                    vis_path, objects_root
                ).replace(os.sep, "/")
                col_rel = "distractors/" + os.path.relpath(
                    col_path, objects_root
                ).replace(os.sep, "/")
                selection_entries.append({
                    "name": dist_name,
                    "category": cat_name,
                    "visual_relpath": vis_rel,
                    "collision_relpath": col_rel,
                    "scale": scale,
                })

        if recorder is not None:
            recorder.record_distractor_selection(selection_entries)

    def initialize(self, env_idx: torch.Tensor):
        super().initialize(env_idx)
        if self.distractor_objects:
            self._initialize_distractors()

    def _initialize_distractors(self):
        """Randomly place distractor objects on the table, far from task objects and robots."""
        replayer = getattr(self.env, '_perturb_replayer', None)
        recorder = getattr(self.env, '_perturb_recorder', None)

        # --- REPLAY: apply recorded placements, skip np.random.* entirely ----
        if replayer is not None:
            placements = replayer.get_distractor_placement()
            for i, (name, actor) in enumerate(self.distractor_objects.items()):
                if i >= len(placements):
                    actor.set_pose(Pose.create_from_pq([10, 10, -10], [1, 0, 0, 0]))
                    continue
                p = placements[i]
                if p.get('fallback', False):
                    actor.set_pose(Pose.create_from_pq([10, 10, -10], [1, 0, 0, 0]))
                else:
                    q = euler2quat(0, 0, p['yaw'])
                    actor.set_pose(Pose.create_from_pq([p['x'], p['y'], p['z']], q))
            return

        # --- ORIGINAL randomisation ------------------------------------------
        exclusion_zones = []

        # Use actual current positions of task objects (after randp_scale has been applied)
        if hasattr(self, 'movable_objects') and self.movable_objects:
            for obj_name, obj_actor in self.movable_objects.items():
                try:
                    p = obj_actor.pose.p
                    pos = np.array([float(p[0][0]), float(p[0][1])])
                except Exception:
                    p = obj_actor.pose.p
                    pos = np.array([float(p[0]), float(p[1])])
                exclusion_zones.append(pos)

        # Also add robot base positions as exclusion zones
        if 'agents' in self.cfg:
            for agent_cfg in self.cfg['agents']:
                rp = np.array(agent_cfg['pos']['ppos']['p'])
                exclusion_zones.append(rp[:2])

        # Support multiple placement zones; fall back to single xlim/ylim for compatibility
        zones = self.domain_rand.get('distractor_zones', None)
        if zones:
            zone_list = [(z['xlim'], z['ylim']) for z in zones]
        else:
            xlim = self.domain_rand.get('distractor_xlim', [-0.05, 0.65])
            ylim = self.domain_rand.get('distractor_ylim', [-0.7, 0.7])
            zone_list = [(xlim, ylim)]
        min_dist = self.domain_rand.get('distractor_min_dist', 0.25)

        placement_records = []
        placed = list(exclusion_zones)
        for i, (name, actor) in enumerate(self.distractor_objects.items()):
            # Cycle through zones so distractors spread across all defined areas
            xlim, ylim = zone_list[i % len(zone_list)]
            success = False
            for _ in range(100):
                x = np.random.uniform(xlim[0], xlim[1])
                y = np.random.uniform(ylim[0], ylim[1])
                too_close = any(
                    np.linalg.norm(np.array([x, y]) - np.array(pp)) < min_dist
                    for pp in placed
                )
                if not too_close:
                    yaw = np.random.uniform(0, 2 * np.pi)
                    q = euler2quat(0, 0, yaw)
                    actor.set_pose(Pose.create_from_pq([x, y, 0.02], q))
                    placed.append(np.array([x, y]))
                    success = True
                    if recorder is not None:
                        placement_records.append({
                            "name": name, "x": float(x), "y": float(y),
                            "z": 0.02, "yaw": float(yaw), "fallback": False,
                        })
                    break
            if not success:
                actor.set_pose(Pose.create_from_pq([10, 10, -10], [1, 0, 0, 0]))
                if recorder is not None:
                    placement_records.append({
                        "name": name, "x": 10.0, "y": 10.0,
                        "z": -10.0, "yaw": 0.0, "fallback": True,
                    })

        if recorder is not None:
            recorder.record_distractor_placement(placement_records)


def get_scene_builder():
    return TableSceneBuilder
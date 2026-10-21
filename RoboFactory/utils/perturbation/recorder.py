"""
Perturbation recorder and replayer.

PerturbRecorder:  attached to an env during a recording run. Scene-builder hooks
                  push every random outcome into this object, which can then be
                  serialised to a seed-entry dict.

PerturbReplayer:  attached to an env during a replay run. Scene-builder hooks
                  read from this object instead of calling np.random.*, guaranteeing
                  bit-exact reproduction of the recorded perturbation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class PerturbRecorder:
    """Collects all perturbation outcomes produced during a single env.reset()."""

    def __init__(self):
        self._data: Dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    # Background texture
    # ------------------------------------------------------------------ #

    def record_background(self, use_clean: bool, texture_relpath: Optional[str]):
        """
        Args:
            use_clean:       True when the original table texture is kept (no overlay).
            texture_relpath: Path relative to the asset-bundle root, or None when
                             use_clean is True.
        """
        self._data["background"] = {
            "use_clean": use_clean,
            "texture_relpath": texture_relpath,
        }

    # ------------------------------------------------------------------ #
    # Distractors built during _build_distractors
    # ------------------------------------------------------------------ #

    def record_distractor_selection(
        self,
        entries: List[Dict[str, Any]],
    ):
        """
        Args:
            entries: list of dicts, one per distractor, with keys:
                name, category, visual_relpath, collision_relpath, scale
                (positions/yaws added later by record_distractor_placement)
        """
        self._data["distractor_selection"] = entries

    # ------------------------------------------------------------------ #
    # Distractor placement during _initialize_distractors
    # ------------------------------------------------------------------ #

    def record_distractor_placement(self, placements: List[Dict[str, Any]]):
        """
        Args:
            placements: list of dicts in the same order as distractor_selection:
                name, x, y, z, yaw, fallback (bool – True when placed off-table)
        """
        self._data["distractor_placement"] = placements

    # ------------------------------------------------------------------ #
    # Lighting
    # ------------------------------------------------------------------ #

    def record_lighting(
        self,
        enabled: bool,
        ambient: Optional[List[float]] = None,
        dir1_color: Optional[List[float]] = None,
        dir2_color: Optional[List[float]] = None,
    ):
        self._data["lighting"] = {
            "enabled": enabled,
            "ambient": ambient,
            "dir1_color": dir1_color,
            "dir2_color": dir2_color,
        }

    # ------------------------------------------------------------------ #
    # Movable object poses (after RFSceneBuilder.initialize)
    # ------------------------------------------------------------------ #

    def record_movable_poses(self, poses: Dict[str, Dict[str, List[float]]]):
        """
        Args:
            poses: {object_name: {"p": [x,y,z], "q": [w,x,y,z]}}
        """
        self._data["movable_object_poses"] = poses

    # ------------------------------------------------------------------ #
    # Agent base poses (after RFSceneBuilder.initialize)
    # ------------------------------------------------------------------ #

    def record_agent_poses(self, poses: Dict[str, Dict[str, List[float]]]):
        """
        Args:
            poses: {agent_key: {"p": [x,y,z], "q": [w,x,y,z]}}
        """
        self._data["agent_base_poses"] = poses

    # ------------------------------------------------------------------ #
    # Agent joint qpos (after RFSceneBuilder.initialize)
    # ------------------------------------------------------------------ #

    def record_agent_qpos(self, qpos_dict: Dict[str, List[float]]):
        """
        Args:
            qpos_dict: {agent_key: [q0, q1, ..., qN]}
        """
        self._data["agent_qpos"] = qpos_dict

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._data)


# ====================================================================== #

class PerturbReplayer:
    """Provides recorded perturbation outcomes so that np.random.* is not called."""

    def __init__(self, seed_entry: Dict[str, Any], asset_bundle_root: Optional[str] = None):
        """
        Args:
            seed_entry:        One entry from the manifest (the dict for this seed).
            asset_bundle_root: Root directory of the extracted asset bundle.
                               Relative paths in the manifest are resolved against this.
        """
        self._d = seed_entry
        self._asset_root = asset_bundle_root

    def _resolve(self, relpath: Optional[str]) -> Optional[str]:
        if relpath is None:
            return None
        if self._asset_root is None:
            return relpath
        import os
        return os.path.join(self._asset_root, relpath)

    # ------------------------------------------------------------------ #

    def get_background(self):
        """Returns (use_clean: bool, texture_abs_path: Optional[str])."""
        bg = self._d.get("background", {})
        return bg.get("use_clean", True), self._resolve(bg.get("texture_relpath"))

    def get_distractor_selection(self) -> List[Dict[str, Any]]:
        """Returns the recorded list from record_distractor_selection."""
        return self._d.get("distractor_selection", [])

    def get_distractor_placement(self) -> List[Dict[str, Any]]:
        """Returns the recorded placements from record_distractor_placement."""
        return self._d.get("distractor_placement", [])

    def get_distractor_visual_path(self, entry: Dict[str, Any]) -> str:
        return self._resolve(entry["visual_relpath"])

    def get_distractor_collision_path(self, entry: Dict[str, Any]) -> str:
        return self._resolve(entry["collision_relpath"])

    def get_lighting(self):
        """Returns dict with keys: enabled, ambient, dir1_color, dir2_color."""
        return self._d.get("lighting", {"enabled": False})

    def get_movable_poses(self) -> Dict[str, Dict[str, List[float]]]:
        return self._d.get("movable_object_poses", {})

    def get_agent_poses(self) -> Dict[str, Dict[str, List[float]]]:
        return self._d.get("agent_base_poses", {})

    def get_agent_qpos(self) -> Dict[str, List[float]]:
        return self._d.get("agent_qpos", {})

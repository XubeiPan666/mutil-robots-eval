"""
Manifest writer and reader for perturbation reproducibility.

Schema (per-task manifest JSON):
{
  "schema_version": 2,
  "task_name": "...",
  "perturbation_config": "configs/perturbation/visual_perturbation_*.yaml",
  "perturbation_config_sha256": "...",
  "scene_config": "configs/table/*.yaml",
  "scene_config_sha256": "...",
  "asset_bundle_version": "v1",
  "asset_bundle_sha256": "...",          # sha256 of perturbation_assets_v1.tar.gz
  "seeds": {
    "1000": { <PerturbRecorder.to_dict()> },
    "1001": { ... },
    ...
  }
}
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

SCHEMA_VERSION = 2


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class ManifestWriter:
    """Accumulates per-seed recorder data and writes a manifest JSON."""

    def __init__(
        self,
        task_name: str,
        perturbation_config: str,
        scene_config: str,
        asset_bundle_manifest_path: Optional[str] = None,
    ):
        self.task_name = task_name
        self.perturbation_config = perturbation_config
        self.scene_config = scene_config

        self._perturb_sha256 = (
            sha256_file(perturbation_config) if perturbation_config and os.path.exists(perturbation_config) else None
        )
        self._scene_sha256 = (
            sha256_file(scene_config) if scene_config and os.path.exists(scene_config) else None
        )

        # Extract asset bundle version + sha256 from its manifest
        self._bundle_version = None
        self._bundle_tarball_sha256 = None
        if asset_bundle_manifest_path and os.path.exists(asset_bundle_manifest_path):
            with open(asset_bundle_manifest_path) as f:
                bm = json.load(f)
            self._bundle_version = bm.get("bundle_version")
            self._bundle_tarball_sha256 = bm.get("tarball_sha256")

        self._seeds: Dict[str, Any] = {}

    def add_seed(self, seed: int, recorder_data: Dict[str, Any]):
        self._seeds[str(seed)] = recorder_data

    def write(self, out_path: str):
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        data = {
            "schema_version": SCHEMA_VERSION,
            "task_name": self.task_name,
            "perturbation_config": self.perturbation_config,
            "perturbation_config_sha256": self._perturb_sha256,
            "scene_config": self.scene_config,
            "scene_config_sha256": self._scene_sha256,
            "asset_bundle_version": self._bundle_version,
            "asset_bundle_sha256": self._bundle_tarball_sha256,
            "seeds": self._seeds,
        }
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[ManifestWriter] Wrote {len(self._seeds)} seeds -> {out_path}")


class ManifestReader:
    """Loads a manifest JSON and provides per-seed replayer objects."""

    def __init__(
        self,
        manifest_path: str,
        asset_bundle_root: Optional[str] = None,
        verify_bundle_sha256: bool = False,
        asset_bundle_tarball_path: Optional[str] = None,
    ):
        """
        Args:
            manifest_path:            Path to the task manifest JSON.
            asset_bundle_root:        Root of the extracted asset bundle.
            verify_bundle_sha256:     If True, re-compute sha256 of the tarball and
                                      compare against the manifest.
            asset_bundle_tarball_path: Path to the .tar.gz to verify (only used when
                                       verify_bundle_sha256 is True).
        """
        with open(manifest_path) as f:
            self._data = json.load(f)

        self._asset_root = asset_bundle_root

        if verify_bundle_sha256 and asset_bundle_tarball_path:
            expected = self._data.get("asset_bundle_sha256")
            if expected:
                actual = sha256_file(asset_bundle_tarball_path)
                if actual != expected:
                    raise ValueError(
                        f"Asset bundle sha256 mismatch!\n"
                        f"  expected: {expected}\n"
                        f"  actual:   {actual}\n"
                        f"  tarball:  {asset_bundle_tarball_path}"
                    )
                print(f"[ManifestReader] Asset bundle sha256 OK: {actual[:16]}...")

    @property
    def task_name(self) -> str:
        return self._data.get("task_name", "")

    @property
    def available_seeds(self):
        return [int(k) for k in self._data.get("seeds", {}).keys()]

    def has_seed(self, seed: int) -> bool:
        return str(seed) in self._data.get("seeds", {})

    def get_replayer(self, seed: int):
        """Return a PerturbReplayer for the given seed, or None if not recorded."""
        from .recorder import PerturbReplayer
        seeds = self._data.get("seeds", {})
        entry = seeds.get(str(seed))
        if entry is None:
            return None
        return PerturbReplayer(entry, asset_bundle_root=self._asset_root)

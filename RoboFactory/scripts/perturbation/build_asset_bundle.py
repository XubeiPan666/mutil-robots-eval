"""
Build a versioned asset bundle (tar.gz + sha256 manifest) from the background texture
and distractor objects directories.

Usage (run from RoboFactory root):
    python scripts/perturbation/build_asset_bundle.py \
        --texture-root /path/to/background_texture \
        --distractor-root /path/to/objects \
        --out-dir release/perturbation_assets_v1

Outputs:
    <out-dir>/perturbation_assets_v1/   -- extracted tree
    <out-dir>/perturbation_assets_v1.manifest.json  -- sha256 per file + version
    <out-dir>/perturbation_assets_v1.tar.gz  -- compressed bundle
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import tarfile
from pathlib import Path

BUNDLE_VERSION = "v1"
SCHEMA_VERSION = 1


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_with_manifest(src: Path, dst: Path, entries: list, rel_root: Path):
    """Recursively copy src -> dst and append sha256 entries."""
    if not src.exists():
        print(f"  WARNING: source path does not exist, skipping: {src}")
        return
    dst.mkdir(parents=True, exist_ok=True)
    for item in sorted(src.iterdir()):
        target = dst / item.name
        if item.is_dir():
            copy_with_manifest(item, target, entries, rel_root)
        elif item.is_file():
            shutil.copy2(item, target)
            rel = target.relative_to(rel_root).as_posix()
            entries.append({"path": rel, "sha256": sha256_file(target)})


def build_bundle(texture_root: str, distractor_root: str, out_dir: str):
    out_dir = Path(out_dir)
    bundle_name = f"perturbation_assets_{BUNDLE_VERSION}"
    bundle_dir = out_dir / bundle_name
    bundle_dir.mkdir(parents=True, exist_ok=True)

    entries = []

    # --- Background textures ------------------------------------------------
    tex_root = Path(texture_root)
    tex_dst = bundle_dir / "background_texture"
    print(f"Copying background textures: {tex_root} -> {tex_dst}")
    copy_with_manifest(tex_root, tex_dst, entries, bundle_dir)

    # --- Distractor objects -------------------------------------------------
    dist_root = Path(distractor_root)
    dist_dst = bundle_dir / "distractors"
    dist_dst.mkdir(parents=True, exist_ok=True)
    print(f"Copying distractor objects: {dist_root} -> {dist_dst}")

    if dist_root.exists():
        for category in sorted(dist_root.iterdir()):
            if not category.is_dir() or not re.match(r"\d{3}_", category.name):
                continue
            cat_dst = dist_dst / category.name
            cat_dst.mkdir(parents=True, exist_ok=True)
            for sub in ["visual", "collision"]:
                sub_src = category / sub
                if sub_src.is_dir():
                    copy_with_manifest(sub_src, cat_dst / sub, entries, bundle_dir)
            for f in sorted(category.iterdir()):
                if f.is_file() and f.name.startswith("model_data"):
                    target = cat_dst / f.name
                    shutil.copy2(f, target)
                    rel = target.relative_to(bundle_dir).as_posix()
                    entries.append({"path": rel, "sha256": sha256_file(target)})
    else:
        print(f"  WARNING: distractor root does not exist: {dist_root}")

    # --- Write sha256 manifest ----------------------------------------------
    manifest_data = {
        "bundle_name": bundle_name,
        "bundle_version": BUNDLE_VERSION,
        "schema_version": SCHEMA_VERSION,
        "files": entries,
        # convenience lookup: path -> sha256
        "sha256_by_path": {e["path"]: e["sha256"] for e in entries},
    }
    manifest_path = out_dir / f"{bundle_name}.manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f, indent=2)
    print(f"Manifest written: {manifest_path}  ({len(entries)} files)")

    # --- Build tarball -------------------------------------------------------
    tarball_path = out_dir / f"{bundle_name}.tar.gz"
    print(f"Building tarball: {tarball_path}")
    with tarfile.open(tarball_path, "w:gz") as tar:
        tar.add(bundle_dir, arcname=bundle_name)
        tar.add(manifest_path, arcname=manifest_path.name)

    tarball_sha256 = sha256_file(tarball_path)
    print(f"Done. Tarball sha256: {tarball_sha256}")
    print(f"  tarball: {tarball_path}")
    print(f"  manifest: {manifest_path}")

    # Append tarball sha256 to manifest for convenience
    manifest_data["tarball_sha256"] = tarball_sha256
    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Build perturbation asset bundle")
    parser.add_argument(
        "--texture-root",
        required=True,
        help="Path to background_texture root (contains seen/, unseen/ subdirs)",
    )
    parser.add_argument(
        "--distractor-root",
        required=True,
        help="Path to distractor objects root (contains NNN_* category dirs)",
    )
    parser.add_argument(
        "--out-dir",
        default="release/perturbation_assets_v1",
        help="Output directory (created if absent)",
    )
    args = parser.parse_args()
    build_bundle(args.texture_root, args.distractor_root, args.out_dir)


if __name__ == "__main__":
    main()

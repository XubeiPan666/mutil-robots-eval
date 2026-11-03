"""Extract the downloaded RoboTwin objects and map the existing scene paths.

No scene configuration or object data is modified. Existing unrelated outputs
are never overwritten. Failed staging directories are retained for inspection.
"""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
from zipfile import ZipFile


REQUIRED = (
    "objects/002_bowl/model_data3.json", "objects/002_bowl/visual/base3.glb",
    "objects/002_bowl/collision/base3.glb", "objects/008_tray/model_data0.json",
    "objects/008_tray/visual/base0.glb", "objects/008_tray/collision/base0.glb",
    "objects/036_cabinet/46653/mobility.urdf",
)


def prepare(archive, destination, robo_root, revision):
    archive, destination, robo_root = map(lambda p: Path(p).resolve(), (archive, destination, robo_root))
    digest = hashlib.sha256()
    with archive.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    signature = {"revision": revision, "archive_sha256": digest.hexdigest()}
    if destination.exists():
        manifest = destination / "objects_archive.json"
        if not manifest.is_file() or json.loads(manifest.read_text()) != signature:
            raise FileExistsError(f"Destination exists with different/unknown provenance: {destination}")
        if any(not (destination / name).is_file() for name in REQUIRED):
            raise FileNotFoundError("Existing asset installation is incomplete")
    else:
        with ZipFile(archive) as zipped:
            names = set(zipped.namelist())
            missing = set(REQUIRED) - names
            if missing:
                raise FileNotFoundError(f"Archive missing required scene files: {sorted(missing)}")
            entries = []
            for entry in zipped.infolist():
                name = PurePosixPath(entry.filename)
                if name.is_absolute() or ".." in name.parts or "\\" in entry.filename:
                    raise ValueError(f"Unsafe ZIP path: {entry.filename}")
                if stat.S_ISLNK(entry.external_attr >> 16):
                    raise ValueError(f"ZIP symlink is not supported: {entry.filename}")
                if name.parts and name.parts[0] == "objects":
                    entries.append(entry)
            destination.parent.mkdir(parents=True, exist_ok=True)
            stage = Path(tempfile.mkdtemp(prefix=".robotwin-objects-", dir=destination.parent))
            print(f"Extracting {len(entries)} entries to {stage}", flush=True)
            for number, entry in enumerate(entries, 1):
                target = stage.joinpath(*PurePosixPath(entry.filename).parts)
                if entry.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zipped.open(entry) as src, target.open("xb") as dst:
                        shutil.copyfileobj(src, dst, length=1024 * 1024)
                if number % 500 == 0:
                    print(f"Extracted {number}/{len(entries)}", flush=True)
        (stage / "objects_archive.json").write_text(json.dumps(signature, indent=2) + "\n")
        if destination.exists():
            raise FileExistsError(f"Destination appeared during extraction: {destination}; stage retained: {stage}")
        stage.rename(destination)

    objects = destination / "objects"
    alias = robo_root / "release/perturbation_assets_v1/perturbation_assets_v1/distractors"
    if alias.is_symlink():
        if alias.resolve() != objects:
            raise FileExistsError(f"Existing asset symlink points elsewhere: {alias}")
    elif alias.exists():
        raise FileExistsError(f"Refusing to replace existing asset directory: {alias}")
    else:
        if not alias.parent.resolve().is_relative_to(robo_root):
            raise ValueError("Asset alias parent resolves outside this checkout")
        alias.parent.mkdir(parents=True, exist_ok=True)
        alias.symlink_to(objects, target_is_directory=True)
    print(f"Scene asset mapping ready: {alias} -> {objects}", flush=True)
    print(f"Archive SHA256: {signature['archive_sha256']}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--robo-root", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    prepare(args.archive, args.destination, args.robo_root, args.revision)

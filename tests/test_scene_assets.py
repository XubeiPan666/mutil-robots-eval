"""Asset extraction/path checks with tiny fixtures; no simulation or downloads."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts/slurm" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PREPARE = load("prepare_scene_assets")
CHECK = load("check_scene_assets")


class SceneAssetsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.archive = self.root / "objects.zip"
        self.destination = self.root / "installed"
        self.robo = self.root / "repo/RoboFactory"
        self.robo.mkdir(parents=True)
        with ZipFile(self.archive, "w") as zipped:
            for name in PREPARE.REQUIRED:
                zipped.writestr(name, b"original asset bytes")
            zipped.writestr("objects/036_cabinet/46653/textures/texture.png", b"texture")

    def test_extract_and_map_preserve_contents_and_are_repeatable(self):
        try:
            PREPARE.prepare(self.archive, self.destination, self.robo, "fixture")
        except OSError as error:
            if getattr(error, "winerror", None) == 1314:
                self.skipTest("Windows user cannot create symlinks")
            raise
        alias = self.robo / "release/perturbation_assets_v1/perturbation_assets_v1/distractors"
        self.assertEqual((alias / "002_bowl/model_data3.json").read_bytes(), b"original asset bytes")
        self.assertEqual((alias / "036_cabinet/46653/textures/texture.png").read_bytes(), b"texture")
        PREPARE.prepare(self.archive, self.destination, self.robo, "fixture")
        with self.assertRaises(FileExistsError):
            PREPARE.prepare(self.archive, self.destination, self.robo, "different revision")

    def test_unsafe_archive_rejected_before_extraction(self):
        with ZipFile(self.archive, "a") as zipped:
            zipped.writestr("objects/../../escape.txt", b"unsafe")
        with self.assertRaises(ValueError):
            PREPARE.prepare(self.archive, self.destination, self.robo, "fixture")
        self.assertFalse(self.destination.exists())
        self.assertFalse((self.root / "escape.txt").exists())

    def test_extraction_preserves_bytes_and_does_not_replace_alias(self):
        alias = self.robo / "release/perturbation_assets_v1/perturbation_assets_v1/distractors"
        alias.mkdir(parents=True)
        marker = alias / "keep.txt"
        marker.write_text("do not replace")
        with self.assertRaises(FileExistsError):
            PREPARE.prepare(self.archive, self.destination, self.robo, "fixture")
        for name in PREPARE.REQUIRED:
            self.assertEqual((self.destination / name).read_bytes(), b"original asset bytes")
        self.assertEqual((self.destination / "objects/036_cabinet/46653/textures/texture.png").read_bytes(), b"texture")
        self.assertEqual(marker.read_text(), "do not replace")
        self.assertEqual(json.loads((self.destination / "objects_archive.json").read_text())["revision"], "fixture")

    def test_existing_directory_not_overwritten(self):
        self.destination.mkdir()
        marker = self.destination / "important.txt"
        marker.write_text("keep")
        with self.assertRaises(FileExistsError):
            PREPARE.prepare(self.archive, self.destination, self.robo, "fixture")
        self.assertEqual(marker.read_text(), "keep")

    def test_selected_tasks_only_and_nested_file_references(self):
        repo = self.robo.parent
        (repo / "scripts").mkdir()
        (repo / "scripts/evaluate.sh").write_text("ALL_TASKS=(stack_cubes stack_bowls)\n")
        wrappers = self.robo / "script/MACG/eval_scripts_rule_base"
        wrappers.mkdir(parents=True)
        configs = self.robo / "configs/table"
        configs.mkdir(parents=True)
        for task in ("stack_cubes", "stack_bowls"):
            (wrappers / f"{task}.sh").write_text(f"# configs/table/ignored.yaml\nSCENE=configs/table/{task}.yaml\n")
            (configs / f"{task}.yaml").write_text(json.dumps({"objects": [
                {"annotation_path": f"assets/{task}.json", "visual": {"filename": f"assets/{task}.glb"}}
            ]}))
        (self.robo / "assets").mkdir()
        for extension in ("json", "glb"):
            (self.robo / f"assets/stack_cubes.{extension}").write_text("fixture")
        self.assertEqual(CHECK.check(repo, ["stack_cubes"])["missing"], [])
        report = CHECK.check(repo, ["all"])
        self.assertEqual(len(report["missing"]), 2)
        self.assertEqual(report["unique_files"], 4)
        self.assertTrue(all("stack_bowls" in entry["path"] for entry in report["missing"]))


if __name__ == "__main__":
    unittest.main()

"""Check declared scene files for selected official task wrappers, without GPU."""

import argparse
import json
import os
from pathlib import Path
import re

import yaml


def file_references(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"file_path", "annotation_path", "filename"} and isinstance(child, str):
                yield child
            else:
                yield from file_references(child)
    elif isinstance(value, list):
        for child in value:
            yield from file_references(child)


def check(repo, tasks=None):
    repo = Path(repo).resolve()
    text = (repo / "scripts/evaluate.sh").read_text(encoding="utf-8")
    match = re.search(r"ALL_TASKS=\((.*?)\)", text, re.S)
    if not match:
        raise ValueError("Cannot read official task inventory")
    available = match[1].split()
    tasks = tasks or ["stack_cubes"]
    if tasks == ["all"]:
        tasks = available
    if any(task not in available for task in tasks):
        raise ValueError(f"Unsupported tasks: {tasks}")
    robo = repo / "RoboFactory"
    configs = set()
    for task in tasks:
        wrapper = robo / "script/MACG/eval_scripts_rule_base" / f"{task}.sh"
        active = "\n".join(line.split("#", 1)[0] for line in wrapper.read_text(encoding="utf-8").splitlines())
        found = re.findall(r"configs/table/[A-Za-z0-9_/-]+\.yaml", active)
        if not found:
            raise ValueError(f"Cannot identify scene config for {task}")
        configs.update(found)
    missing, checked = [], set()
    for name in sorted(configs):
        data = yaml.safe_load((robo / name).read_text(encoding="utf-8"))
        for reference in sorted(set(file_references(data))):
            path = Path(os.path.expandvars(reference)).expanduser()
            if not path.is_absolute():
                path = robo / path
            checked.add(str(path))
            if not path.is_file():
                missing.append({"config": name, "path": str(path)})
    return {"tasks": tasks, "configs": sorted(configs), "unique_files": len(checked), "missing": missing}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--task", action="append")
    args, _ = parser.parse_known_args()
    report = check(args.repo, args.task)
    print(json.dumps(report, indent=2))
    if report["missing"]:
        raise SystemExit("Scene assets are incomplete; model startup was not attempted")

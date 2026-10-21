#!/usr/bin/env python3
"""Export checked working-tree sources without Git history or ignored files."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from check_release import ROOT, check, source_files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path, help='New, empty directory outside the checkout')
    args = parser.parse_args()
    destination = args.destination.resolve()
    if destination == ROOT or ROOT in destination.parents:
        parser.error('destination must be outside this checkout')
    if destination.exists() and any(destination.iterdir()):
        parser.error('destination must be absent or empty')
    errors = check()
    if errors:
        parser.exit(1, '\n'.join(errors) + '\n')
    paths = source_files()
    destination.mkdir(parents=True, exist_ok=True)
    manifest = []
    for source in paths:
        relative = source.relative_to(ROOT)
        # A fresh export owns its own manifest; never nest a prior snapshot.
        if relative.as_posix() == 'PUBLIC_RELEASE_MANIFEST.json':
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        manifest.append({'path': relative.as_posix(), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()})
    (destination / 'PUBLIC_RELEASE_MANIFEST.json').write_text(
        json.dumps({'schema_version': 1, 'repository': 'https://github.com/zhangzaibin/future-robots-eval', 'files': manifest}, indent=2) + '\n', encoding='utf-8',
    )
    print(f'Exported {len(manifest)} checked files to {destination}; no Git history or ignored files copied.')


if __name__ == '__main__':
    main()

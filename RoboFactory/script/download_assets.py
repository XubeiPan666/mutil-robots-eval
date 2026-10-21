"""Download RoboFactory scene assets into a predictable repository location."""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_REPO_ID = "sparklexfantasy/RoboFactory_asset"
DEFAULT_ASSET_DIR = Path(__file__).resolve().parents[1] / "assets"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id", default=DEFAULT_REPO_ID, help="Hugging Face dataset repository."
    )
    parser.add_argument(
        "--revision", default=None, help="Optional dataset commit hash, tag, or branch."
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=DEFAULT_ASSET_DIR,
        help="Download directory (default: RoboFactory/assets, independent of cwd).",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use files already cached locally; do not contact Hugging Face.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface-hub is missing. Activate the simulator environment and run "
            "python -m pip install -r RoboFactory/requirements.txt from the repository root."
        ) from exc

    destination = args.local_dir.expanduser().resolve()
    print(f"Downloading {args.repo_id} into {destination}", flush=True)
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=str(destination),
        local_files_only=args.local_files_only,
    )
    print(f"Assets ready: {destination}")


if __name__ == "__main__":
    main()

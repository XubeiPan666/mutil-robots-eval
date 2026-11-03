#!/bin/bash -l
#SBATCH --job-name=aevla-scene-assets
#SBATCH --partition=a100mig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00

set -eo pipefail
source "$(ws_find mutil-robots)/load_eval_env.sh"
cd "$PROJECT_WS/code/mutil-robots-eval"
REVISION="$(cat "$PROJECT_WS/robotwin_objects_revision.txt")"
test "$REVISION" = ba1ebdd89aac671899a768e62dfe4dc6b29956c8
python scripts/slurm/prepare_scene_assets.py \
    --archive "$PROJECT_WS/downloads/robotwin2/objects.zip" \
    --destination "$PROJECT_WS/assets/robotwin2" \
    --robo-root "$PWD/RoboFactory" --revision "$REVISION"
python scripts/slurm/check_scene_assets.py --repo "$PWD" --task all
echo "Asset extraction, mapping and declared scene-file checks completed: $(date -Is)"

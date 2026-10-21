# Reproducible visual perturbation experiments

The current convenience suite runs clean conditions. Its visual-perturbation commands are commented out. A message saying that a replay manifest was found does **not** enable perturbation evaluation.

Use the maintained [visual perturbation workflow](../../../../docs/evaluation.md#visual-perturbations-and-reproducibility) for verified commands, asset bundle layout, record/replay options, and experiment reporting. Commands below run from `RoboFactory/`.

## Build and record your assets

```bash
python scripts/perturbation/build_asset_bundle.py \
  --texture-root /path/to/background_texture \
  --distractor-root /path/to/objects \
  --out-dir release/perturbation_assets_v1

python scripts/perturbation/record_manifests.py \
  --task two_arms_stack_cube --seeds 1000-1099 \
  --out-dir release/manifests \
  --asset-bundle-manifest release/perturbation_assets_v1/perturbation_assets_v1.manifest.json
```

The builder writes an outer output directory containing a manifest, a tarball, and the nested tree `perturbation_assets_v1/`. Inspect warnings for missing source folders. Only redistribute assets whose terms permit it.

## Verify replay before evaluation

```bash
export ROBOFACTORY_PERTURB_ASSETS="$PWD/release/perturbation_assets_v1/perturbation_assets_v1"
python scripts/perturbation/verify_replay.py \
  --task two_arms_stack_cube \
  --manifest release/manifests/two_arms_stack_cube_perturbed.json \
  --asset-bundle-root "$ROBOFACTORY_PERTURB_ASSETS" \
  --seeds 1000-1005
```

Use the actual recorded manifests and matching asset bundle on every machine. The environment remaps recorded asset paths using `ROBOFACTORY_PERTURB_ASSETS`; fresh randomization still reads source paths in the perturbation YAML. Keep both configurations consistent.

To enable perturbations for a direct Python rollout, supply `--perturbation-config` and `--replay-manifest`. For the task-specific batch shell script, pass the perturbation YAML as its seventh positional argument and set `REPLAY_MANIFEST_FILE`. See the linked evaluation guide for complete commands.

Record source / environment versions, scene and phase YAMLs, assets and their revisions, bundle hashes, manifests, model checkpoint, seed list, and result files. A seed alone does not guarantee cross-machine identical visual input. The verification tool checks recorded state within numerical tolerances; it does not promise identical pixels or policy outputs across hardware. The released code does not bundle the original paper-run perturbation assets or manifests.

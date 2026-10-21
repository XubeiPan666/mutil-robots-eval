# Installation

## Two environments

The **simulator environment** runs this repository: ManiSkill/SAPIEN, task scenes, phase detectors, HTTP clients, and metrics. The **model-server environment** runs your trained VLA model. Follow [Future Robots](https://github.com/zhangzaibin/future-robots) for model installation, checkpoints, and serving. Keep these environments separate so their PyTorch, CUDA, and model dependencies can be managed independently. They may run on the same machine or on separate machines connected over HTTP.

Use Linux with an NVIDIA GPU and working Vulkan rendering for the simulator. The shipped evaluators use CPU physics (`--sim-backend cpu`) and RGB observations; CPU physics still needs GPU rendering. This release does not specify a measured minimum VRAM requirement. Model GPU requirements depend on the checkpoint and server configuration. See the [official ManiSkill installation guide](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html#vulkan) for driver and Vulkan troubleshooting.

## 1. Create the simulator environment

From a terminal:

```bash
git clone https://github.com/zhangzaibin/future-robots-eval.git
cd future-robots-eval
conda create -n future-robots-eval python=3.9 -y
conda activate future-robots-eval
python -m pip install --upgrade pip
python -m pip install -r RoboFactory/requirements.txt
python -m pip check
```

The requirements retain the existing `mani_skill==3.0.0b12`, `torch==2.6.0`, and `torchvision==0.21.0` versions. NumPy remains on the 1.x ABI used by the legacy motion-planning extensions; see the [ManiSkill maintainer's compatibility guidance](https://github.com/mani-skill/ManiSkill/issues/1100). Install the matching PyTorch build for your CUDA driver if your machine needs a specific build, while keeping these version numbers aligned. Avoid upgrading ManiSkill or MPlib independently: this code uses their older interfaces.

These requirements are a dependency baseline, not a fully locked and benchmark-validated environment export. Once your Linux setup passes the smoke test, save its exact package versions with the experiment:

```bash
python -m pip freeze > simulator-environment.txt
```

On Ubuntu/Debian, the shell entry points also require `bash`, `curl`, and `bc`. Install rendering libraries if they are absent:

```bash
sudo apt update
sudo apt install -y curl bc libvulkan1 vulkan-tools libgl1 libglvnd0 libegl1 libgles2 libopengl0
nvidia-smi
vulkaninfo --summary
```

A headless server can render with `rgb_array` without opening a desktop window. The NVIDIA driver and Vulkan device must still be available inside the process/container. Missing Vulkan devices, `libGLX_nvidia.so`, or EGL vendor configuration should be fixed before diagnosing task code. Do not switch to `human` rendering on a headless machine.

## 2. Download scene assets

From the repository root:

```bash
python RoboFactory/script/download_assets.py
```

The downloader fetches the [upstream RoboFactory asset dataset](https://huggingface.co/datasets/sparklexfantasy/RoboFactory_asset) into `RoboFactory/assets/`, regardless of the current working directory. Downloading may take time and requires network access and free disk space. Hugging Face reuses already downloaded files on subsequent runs.

Check a file referenced by the table scenes:

```bash
test -f RoboFactory/assets/scenes/table/table.glb
```

For a controlled experiment, pin the asset dataset revision:

```bash
python RoboFactory/script/download_assets.py --revision DATASET_COMMIT_HASH
```

Replace `DATASET_COMMIT_HASH` with the actual revision you selected; no paper-specific asset revision is bundled here. `--local-dir PATH` changes the download destination, but existing scene YAML files resolve `assets/...` relative to `RoboFactory/`. Keep that path available or update the relevant YAML when using a custom destination. `--local-files-only` requests cached files without network access.

The optional RoboCasa scenes need an additional dataset. They are not required by the default two-arm table evaluation suite:

```bash
python -m mani_skill.utils.download_asset RoboCasa
```

Downloaded scene assets and optional background/distractor bundles retain their original terms; see [third-party notices](../THIRD_PARTY_NOTICES.md).

## 3. Test the simulator without a model

The motion-planning runner can collect one trajectory without a VLA server:

```bash
cd RoboFactory
python -m planner.run \
  --config configs/table/two_arms_stack_cube.yaml \
  --status two_arms_stack_cube_12 \
  --num-traj 1 \
  --seed 1000 \
  --obs-mode rgb \
  --sim-backend cpu \
  --render-mode rgb_array \
  --record-dir demos/smoke \
  --save-video
```

The runner writes a trajectory and video under `demos/smoke/two_arms_stack_cube_12/motionplanning/`. A motion planner can fail a rollout, so inspect its success output as well as the generated files. This checks task import, assets, rendering, and recording; it does not validate a VLA checkpoint.

For a desktop GUI, the convenience runner requires **both** a configuration and a status:

```bash
python script/run_task.py configs/table/two_arms_stack_cube.yaml two_arms_stack_cube_12
```

## 4. Connect the model

Start a compatible model server in its own environment, then confirm connectivity from the simulator machine:

```bash
export VLA_SERVER_URL=http://127.0.0.1:20011
curl --fail "$VLA_SERVER_URL/health"
```

Replace the URL with your actual server address. A generic OpenAI-compatible chat API or an unadapted WebSocket policy server does not implement this evaluator's `/infer` protocol. See the exact [model-server contract](architecture.md#model-server-contract), then proceed to [evaluation](evaluation.md).

## Troubleshooting

| Symptom | Check |
| --- | --- |
| `No module named mani_skill`, `requests`, or `cv2` | Activate `future-robots-eval`; use `python -m pip install -r RoboFactory/requirements.txt` from the repository root |
| Scene mesh/file not found | Confirm `RoboFactory/assets/` exists and run low-level scripts from `RoboFactory/` |
| Renderer fails or no Vulkan device is found | Confirm `nvidia-smi`, `vulkaninfo`, driver libraries, and container GPU visibility |
| MPlib import or NumPy ABI error | Check that legacy dependencies were retained and `numpy==1.26.4` is installed |
| `bc: command not found` | Install `bc`; legacy task wrappers use it for success-rate arithmetic |
| `/health` is unreachable | Check server address, port, network routing, and whether the checkpoint finished loading |
| Per-arm proxy fails to become healthy | Both upstream `/health` responses must report `status: healthy`; inspect the proxy log |
| Action horizon or dimension error | Return at least 50 actions with 16 dual-arm coordinates, or 8 coordinates per upstream arm |

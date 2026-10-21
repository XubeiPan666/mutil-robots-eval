# Architecture and policy integration

## Repository map

```text
future-robots-eval/
├── scripts/evaluate.sh                 # Explicit, validated public evaluation entry point
├── docs/                               # Installation, evaluation, and integration guides
├── RoboFactory/
│   ├── configs/table/                   # Scene, robots, objects, and camera definitions
│   ├── configs/perturbation/            # Background, distractor, and lighting settings
│   ├── tasks/                          # Registered ManiSkill task environments
│   ├── planner/                        # Motion planning and demonstration collection
│   ├── script/MACG/
│   │   ├── eval_scripts_rule_base/      # Task suites and instruction-condition selections
│   │   ├── eval_scripts_rule_base_peram/ # Per-arm proxy and compatible task entry points
│   │   ├── two_arms_*/                  # Rollout evaluators, phase YAML, and HTTP clients
│   │   ├── three_arms_place_objects/    # Separate three-arm task integration
│   │   └── eval_common/                # Subtask progress tracking and aggregation
│   ├── scripts/perturbation/           # Asset bundles and perturbation record/replay tools
│   ├── utils/                          # Scene builders, recording, and planner helpers
│   └── policy/                         # Inherited research baselines and training utilities
└── DataAugmentation/                   # Experimental view/trajectory augmentation tools
```

The public launcher selects existing task scripts. A task script chooses a scene, environment status, and phase configuration, then rolls out seeds. At each model query, the evaluator sends RGB views, robot state, and the current instruction to the model server. The returned joint/gripper targets are interpolated by the motion planner and executed in the simulator. Environment information drives the rule-based phase detector and the separate progress tracker.

The default documented path is the two-arm rule-based suite. The inherited Diffusion Policy/Pi0 utilities, VLM phase-selection experiments, three-/four-arm tasks, and augmentation scripts are research extensions with their own configurations and dependencies; they are not all exercised by the public launcher.

## Model-server contract

### Transport

Both single-server and per-arm modes use the same HTTP envelope:

| Request | Expected response |
| --- | --- |
| `GET /health` | JSON health object; per-arm mode requires `{"status": "healthy"}` and rejects `model_loaded: false` |
| `POST /infer` with `{"example": "BASE64_PICKLED_DICT"}` | `{"success": true, "result": "BASE64_PICKLED_RESULT"}` |
| Failed inference | `{"success": false, "error": "description"}` |

The task clients serialize Python objects using `pickle`, then encode the bytes with base64. The decoded inference result is a dictionary containing `actions`, a numeric array. The single-server clients use a 10-second health-check timeout and a 60-second inference timeout; the proxy has a separately configurable upstream timeout.

**Use trusted model servers on a private network.** Pickle decoding can execute code, and this research protocol provides no authentication or encrypted transport by itself. Do not expose the proxy to untrusted clients. Its default bind address is `127.0.0.1`.

### Two-arm observation

The decoded `example` dictionary contains:

| Key | Value |
| --- | --- |
| `global_image` | Global RGB image, typically a `uint8` array of shape `(224, 224, 3)` in the supplied table scenes |
| `image_agent0`, `image_agent1` | Arm-specific wrist RGB images with the same layout |
| `joint_position_agent0`, `joint_position_agent1` | Seven joint positions for each Panda arm |
| `gripper_position_agent0`, `gripper_position_agent1` | Scalar gripper positions |
| `task` | Current compound instruction string assembled from the phase configuration |

Image dimensions come from the selected scene configuration. Preserve RGB channel order; the diagnostic JPEG conversion in the evaluator is only for files written to disk.

### Actions

Single-server dual-arm models return `actions` with shape `(T, 16)`, where **`T >= 50`**. Each row is ordered as:

```text
[arm0_joint_0 ... arm0_joint_6, arm0_gripper,
 arm1_joint_0 ... arm1_joint_6, arm1_gripper]
```

The provided two-arm evaluators consume 50 targets per query. Joint values are absolute position targets used by the existing `pd_joint_pos` control and trajectory interpolation. Your serving adapter must produce the evaluator's expected joint/gripper scale and ordering after model-specific denormalization; an end-effector delta action head cannot be connected directly.

For **per-arm mode**, the local proxy forwards `global_image`, `task`, and the chosen arm's image/state keys to each upstream server in parallel. It does not rename agent keys: the arm1 server receives `image_agent1`, `joint_position_agent1`, and `gripper_position_agent1`. Each server returns shape `(T, 8)`. Both horizons must be equal and at least 50. The proxy concatenates arm0 then arm1 along the last dimension, producing `(T, 16)` for the unchanged evaluator. Model-specific image padding, prompt splitting, and single-arm preprocessing belong in the upstream serving adapter.

## Extending evaluation

To connect a new policy, implement the contract above, run one seed as described in [evaluation](evaluation.md#one-episode-smoke-test), and inspect the videos before launching the suite. Record the checkpoint, preprocessing, action representation, and server revision with your results.

To add a task, register its environment under `RoboFactory/tasks/`, define its scene and phase configurations, add an evaluator and task wrapper, and define progress milestones under `script/MACG/eval_common/progress_specs/`. A milestone's physical signal mapping belongs in `eval_common/eval_hooks.py`. Keep environment success and constraint-compliant progress as distinct reported metrics. Add the task to the public launcher's list only after its rollout, output, and server interfaces are verified.

Existing three-arm integration is separate and requires a matching multi-arm server. The two-upstream per-arm proxy supports exactly two arms.

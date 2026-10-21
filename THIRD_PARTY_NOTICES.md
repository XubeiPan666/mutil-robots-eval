# Third-party notices

MAS-VLA / Future Robots Eval builds on RoboFactory and includes adaptations of
other research software. The repository [MIT license](LICENSE) retains the
RoboFactory / MARS-EAI attribution. It does not replace third-party licenses.
Existing copyright and license notices in source files are retained.

## Included and adapted code

| Project | Relevant local paths | Upstream license and source |
| --- | --- | --- |
| RoboFactory | `RoboFactory/`; original project documentation in `README-RF.md` | MIT; [MARS-EAI/RoboFactory](https://github.com/MARS-EAI/RoboFactory); attribution retained in `LICENSE` |
| Diffusion Policy | `RoboFactory/policy/Diffusion-Policy/diffusion_policy/` and the legacy copy at `RoboFactory/policy/Pi0/diffusion_policy/` | [MIT text](licenses/diffusion_policy.txt); [upstream](https://github.com/real-stanford/diffusion_policy) |
| ManiSkill | Local simulator, scene and recording adaptations under `RoboFactory/utils/envs/`, `RoboFactory/utils/scenes/`, `RoboFactory/utils/building/nonconvex_urdf_loader.py`, and `RoboFactory/utils/wrappers/record.py`; also installed as a dependency | [Apache-2.0 text](licenses/maniskill.txt); [upstream](https://github.com/mani-skill/ManiSkill) |
| RoboCasa | Kitchen scene, fixture and object utilities under `RoboFactory/utils/scenes/robocasa/`, including code ported through ManiSkill | [MIT text and MuJoCo notice](licenses/robocasa.txt); [upstream](https://github.com/robocasa/robocasa) |
| robosuite | Object/MJCF utilities referenced by the RoboCasa port, including `fixtures/mujoco_object.py` | [MIT text and MuJoCo notice](licenses/robosuite.txt); [upstream](https://github.com/ARISE-Initiative/robosuite) |
| minGPT | `model/bet/libraries/mingpt/` in both included `diffusion_policy` trees | Existing [MIT text](RoboFactory/policy/Diffusion-Policy/diffusion_policy/model/bet/libraries/mingpt/LICENSE); [upstream](https://github.com/karpathy/minGPT) |
| Multiclass focal loss | `model/bet/libraries/loss_fn.py` in both included `diffusion_policy` trees | MIT notice retained in each file; [upstream](https://github.com/AdeelH/pytorch-multi-class-focal-loss) |
| Pymunk adaptation | `common/pymunk_override.py` in both included `diffusion_policy` trees | MIT notice retained in each file; [upstream](https://github.com/viblo/pymunk) |

The simulator and scene files are local adaptations for RoboFactory/MAS-VLA:
they support multi-agent tasks, custom scene and asset handling, nonconvex URDF
loading, and local recording behavior. They are not unmodified upstream releases.
This release adds public configuration defaults and documentation; upstream
algorithm and task ownership remain with their respective authors. The repository
does not record exact original upstream revisions for all inherited files.

The four license texts in `licenses/` were retrieved from the official upstream
repositories on 2026-10-21. They supplement the notices already included in the
source tree; their retrieval date does not identify the revision of adapted code.

## Assets, datasets and model weights

Runtime asset downloads are obtained from their original RoboFactory, ManiSkill,
RoboCasa or Hugging Face sources. Those assets, datasets, model checkpoints and
remote policy servers have their own terms. A source-code license does not grant
additional rights to them. Retain source attribution and follow the original
asset or dataset license when redistributing downloaded files.

`RoboFactory/utils/scenes/table/assets/table.glb` is retained from the inherited
RoboFactory source tree. It is not a new asset authored by Evolving Machines Lab.
The small `example.pkl` file is an inherited protocol example; only deserialize
pickle data from a source you trust.

When publishing results, cite Future Robots as shown in the root README and also
cite the simulator, benchmark, dataset and policy projects used in the experiment.

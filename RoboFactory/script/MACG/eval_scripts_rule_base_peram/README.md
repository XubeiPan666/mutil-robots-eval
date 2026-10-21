# Per-arm 双 VLA 规则评测

这个目录提供隔离的双 server 入口，不修改
`script/MACG/eval_scripts_rule_base` 或各任务原有的 Python、shell、client 文件。

## 数据流

本地代理接收原评测的统一观测，然后并行请求：

- arm0：`http://localhost:20011`，输入 global image、arm0 wrist/state 和复合指令。
- arm1：`http://localhost:20012`，输入 global image、arm1 wrist/state 和复合指令。

每个上游必须返回 `actions.shape == (T, 8)`，两者 T 相同且 T >= 50。代理按
`[arm0, arm1]` 拼成 `(T, 16)`，原评测继续按前 8 维控制
`panda-0`、后 8 维控制 `panda-1`。

黑色 padding 图像和单臂指令拆分由 `perarmpi05` server 的
`single_arm_id` 预处理完成，代理不会重复处理。

## 运行

所有命令从 `RoboFactory` 根目录执行。

单个任务（原脚本会评测其配置的 100 个 seeds）：

```bash
bash script/MACG/eval_scripts_rule_base_peram/place_can_basket.sh
```

批量运行原 `run.sh` 当前启用的双臂任务：

```bash
bash script/MACG/eval_scripts_rule_base_peram/run.sh
```

其他可用入口：

```text
stack_cubes.sh
push_cubes.sh
stack_bowls.sh
place_cube_in_bowl.sh
handover_bottles.sh
place_burger_fries.sh
place_can_basket.sh
place_object_cabinet.sh
place_object_bowl_cabinet.sh
```

三臂任务不适用于两个单臂 server，因此本目录没有三臂入口。

## 单 seed smoke test

先用 seed 1000 验证两个 server、动作拼接和环境执行：

```bash
bash script/MACG/eval_scripts_rule_base_peram/run_with_proxy.sh \
  python script/MACG/two_arms_place_can_basket/eval_place_can_basket.py \
  --config configs/table/two_arms_place_can_basket.yaml \
  --data-num 1 \
  --checkpoint-num 300 \
  --render-mode rgb_array \
  --obs-mode rgb \
  --sim-backend cpu \
  --num-envs 1 \
  --seed 1000 \
  --status two_arms_place_can_basket \
  --phase-config script/MACG/two_arms_place_can_basket/place_can_basket_indomain.yaml \
  --quiet
```

## 地址与超时覆盖

```bash
ARM0_VLA_SERVER_URL=http://localhost:20011 \
ARM1_VLA_SERVER_URL=http://localhost:20012 \
PERARM_PROXY_PORT=31888 \
PERARM_INFER_TIMEOUT_S=60 \
bash script/MACG/eval_scripts_rule_base_peram/place_can_basket.sh
```

可用环境变量：

- `ARM0_VLA_SERVER_URL`、`ARM1_VLA_SERVER_URL`：上游 server。
- `PERARM_PROXY_HOST`、`PERARM_PROXY_PORT`：本地兼容代理监听地址。
- `PERARM_INFER_TIMEOUT_S`：单次上游推理超时。
- `PERARM_PROXY_LOG`：代理日志路径；默认写入 `/tmp`。
- `PYTHON_BIN`：启动代理所用的 Python 命令。

`run_with_proxy.sh` 只对子评测进程设置 `VLA_SERVER_URL`，退出时会自动关闭
本地代理，不会改变当前 shell 的环境变量。

# 三机器人堆叠方块 GBR 环境说明

## 概述

本项目现在支持**两种**堆叠判定标准：

### 1. 原始环境 (`ThreeRobotsStackCube-rf`)
- **文件**: `three_robots_stack_cube.py`
- **堆叠顺序**: 
  - 蓝色 (cubeA) 在最下面
  - 绿色 (cubeB) 在第二层
  - 红色 (cubeC) 在最上面
- **使用脚本**: `eval_threerobots_stackcubes_atom.py`

### 2. GBR 环境 (`ThreeRobotsStackCubeGBR-rf`)
- **文件**: `three_robots_stack_cube_gbr.py`
- **堆叠顺序**: 
  - 绿色 (cubeB) 在最下面 ✅
  - 蓝色 (cubeA) 在第二层 ✅
  - 红色 (cubeC) 在最上面 ✅
- **使用脚本**: `eval_threerobots_stackcubes_atom_gbr.py`

## 方块颜色定义

根据配置文件 `configs/*/three_robots_stack_cube.yaml`:

- **cubeA**: 蓝色 `[0.047, 0.165, 0.627, 1]`
- **cubeB**: 绿色 `[0, 1, 0, 1]`
- **cubeC**: 红色 `[1, 0, 0, 1]`

## 成功判定条件

### GBR 环境的成功条件：

```python
success = (
    is_cubeC_on_cubeA *        # 红色在蓝色上面
    is_cubeA_on_cubeB *        # 蓝色在绿色上面
    cubeB_placed *             # 绿色在目标区域
    cubeA_placed *             # 蓝色在目标区域
    cubeC_placed *             # 红色在目标区域
    (~is_cubeA_grasped) *      # 蓝色未被抓取
    (~is_cubeB_grasped) *      # 绿色未被抓取
    (~is_cubeC_grasped)        # 红色未被抓取
)
```

## 使用方法

### 使用原始环境（保留原判定标准）：

```bash
python policy/Pi0/eval_threerobots_stackcubes_atom.py \
    --config configs/robocasa/three_robots_stack_cube.yaml
```

### 使用 GBR 环境（绿-蓝-红堆叠顺序）：

```bash
python policy/Pi0/eval_threerobots_stackcubes_atom_gbr.py \
    --config configs/robocasa/three_robots_stack_cube.yaml
```

**注意**: `eval_threerobots_stackcubes_atom_gbr.py` 会自动将环境 ID 从 `ThreeRobotsStackCube-rf` 转换为 `ThreeRobotsStackCubeGBR-rf`。

## 修改的文件

1. **新增**: `RoboFactory/tasks/three_robots_stack_cube_gbr.py`
   - 新的环境类 `ThreeRobotsStackCubeGBREnv`
   - 注册为 `ThreeRobotsStackCubeGBR-rf`

2. **修改**: `RoboFactory/tasks/__init__.py`
   - 添加了 `ThreeRobotsStackCubeGBREnv` 的导入

3. **修改**: `RoboFactory/policy/Pi0/eval_threerobots_stackcubes_atom_gbr.py`
   - 自动使用 GBR 环境进行评估

4. **保留**: `RoboFactory/tasks/three_robots_stack_cube.py`
   - 保持原始判定标准不变

## 技术细节

### 堆叠检测逻辑

判断方块 X 是否在方块 Y 上面：

```python
offset = pos_X - pos_Y
xy_flag = (
    torch.linalg.norm(offset[..., :2], axis=1)
    <= torch.linalg.norm(cube_half_size[:2]) + 0.005
)
z_flag = torch.abs(offset[..., 2] - cube_half_size[..., 2] * 2) <= 0.005
is_X_on_Y = torch.logical_and(xy_flag, z_flag)
```

### 目标区域检测

```python
cube_to_goal_dist = torch.linalg.norm(
    cube.pose.p[:, :2] - goal_region.pose.p[..., :2], axis=1
)
cube_placed = cube_to_goal_dist < goal_radius  # goal_radius = 0.12
```

## 兼容性

- 两个环境使用相同的配置文件
- 两个环境使用相同的场景和机器人设置
- 唯一的区别是成功判定的堆叠顺序


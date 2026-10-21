# Data Augmentation - New View Image Generation

从 h5 文件中提取数据，进行分割、修复和视角变换，生成新视角图像的工具。

## 功能概述

该工具实现了从机器人轨迹数据（h5 格式）中生成新视角图像的功能，主要应用于数据增强场景。通过以下步骤生成新视角图像：

1. **数据提取**：从 h5 文件中提取多个时间步的相机图像、内参和外参
2. **目标外参计算**：基于机器人关节角度（qpos）计算目标相机外参
3. **图像分割与修复**：对图像进行分割，并使用 inpainting 技术修复分割区域
4. **视角变换**：将修复后的图像变换到目标视角
5. **图像合成**：将原始分割 mask 应用到变换后的图像上

## 主要功能

### 核心功能

- **H5 数据读取**：支持从 h5 文件中读取相机数据（图像、内参、外参）和机器人关节角度（qpos）
- **相机外参计算**：基于 URDF 模型和关节角度计算目标相机外参（使用 `qpos2camext` 模块）
- **图像分割**：通过分割服务器对图像进行分割，获取物体 mask
- **图像修复**：使用 OpenCV 的 inpainting 算法修复分割区域（支持 Telea 和 Navier-Stokes 方法）
- **视角变换**：调用变换服务器将图像从源视角变换到目标视角
- **姿态扰动**：支持对机器人关节角度进行扰动，生成不同姿态下的新视角图像
- **对比图生成**：可选生成原始图像与生成图像的对比图

### 辅助功能

- **多时间步提取**：支持提取连续或间隔的多个时间步数据
- **灵活的参数配置**：支持多种参数组合方式
- **错误处理**：完善的错误处理和降级策略

## 工作流程

```
┌─────────────────┐
│  读取 H5 数据    │ → 提取 n 个时间步的图像、内参、外参
└─────────────────┘
         ↓
┌─────────────────┐
│  计算目标外参    │ → 使用 qpos2camext 或使用最后一步外参
└─────────────────┘
         ↓
┌─────────────────┐
│  图像分割与修复  │ → 对每张图像进行分割，并用 inpainting 修复
└─────────────────┘
         ↓
┌─────────────────┐
│   视角变换       │ → 调用变换服务器进行视角变换
└─────────────────┘
         ↓
┌─────────────────┐
│   图像合成       │ → 将原始 mask 应用到变换后的图像
└─────────────────┘
         ↓
┌─────────────────┐
│   输出结果       │ → 生成最终图像和可选对比图
└─────────────────┘
```

## 依赖项

### Python 包

- `h5py`：H5 文件读取
- `numpy`：数值计算
- `opencv-python` (cv2)：图像处理
- `Pillow` (PIL)：图像处理
- `requests`：HTTP 请求（用于调用分割和变换服务器）

### 外部服务

- **分割服务器**：提供图像分割服务（默认：环境变量 `TRANSFORM_SERVER_URL`，未设置时为 `http://localhost:8001`）
  - 端点：`/predict_base64`
  - 输入：base64 编码的图像
  - 输出：base64 编码的 mask

- **变换服务器**：提供视角变换服务（默认：环境变量 `TRANSFORM_SERVER_URL`，未设置时为 `http://localhost:8001`）
  - 端点：`/inference`
  - 输入：图像列表、内参、外参、目标外参
  - 输出：变换后的图像

### 本地模块

- `qpos2camext.py`：从关节角度计算相机外参
- `plan_new_pose.py`：机器人姿态规划和验证

## 使用方法

### 基本用法

```bash
python get_new_view_image.py \
    --h5_path /path/to/data.h5 \
    --traj_idx 0 \
    --timestep 10 \
    --agent_idx 0 \
    --output outputs/new_view_image.png
```

### 使用 qpos2camext 计算目标外参

```bash
python get_new_view_image.py \
    --h5_path /path/to/data.h5 \
    --traj_idx 0 \
    --timestep 10 \
    --agent_idx 0 \
    --urdf_filepath panda/panda_v2.urdf \
    --robot_base_position -0.2 -0.4 0.0 \
    --camera_translation_relative_to_tcp 0.2 0 -0.18 \
    --camera_rotation_relative_to_tcp 0.0 0.707107 0.0 0.707107 \
    --read_qpos_from_h5 \
    --qpos_timestep_offset 5 \
    --output outputs/new_view_image.png
```

### 使用姿态扰动

```bash
python get_new_view_image.py \
    --h5_path /path/to/data.h5 \
    --traj_idx 0 \
    --timestep 10 \
    --agent_idx 0 \
    --urdf_filepath panda/panda_v2.urdf \
    --robot_base_position -0.2 -0.4 0.0 \
    --camera_translation_relative_to_tcp 0.2 0 -0.18 \
    --camera_rotation_relative_to_tcp 0.0 0.707107 0.0 0.707107 \
    --joint_indices 0 2 4 \
    --perturbations 0.1 -0.1 0.05 \
    --output outputs/new_view_image.png
```

## 参数说明

### 必需参数

- `--h5_path`：H5 文件路径
- `--traj_idx`：轨迹序号（从 0 开始）
- `--timestep`：起始时间步 t

### 可选参数

#### 基本参数

- `--agent_idx`：Agent 序号（默认：0）
- `--n`：提取的步数，包含起始时间步（默认：3）
- `--step_size`：时间步长，提取的时间步为 t, t+a, t+2a, ...（默认：10）
- `--segment_server`：分割服务器 URL（默认：环境变量 `SEGMENT_SERVER_URL`，未设置时为 `http://localhost:8000`）
- `--transform_server`：变换服务器 URL（默认：环境变量 `TRANSFORM_SERVER_URL`，未设置时为 `http://localhost:8001`）
- `--output`：输出图像路径（默认：`outputs/new_view_image.png`）

#### qpos2camext 相关参数

- `--urdf_filepath`：URDF 文件路径（用于计算目标外参）
- `--robot_base_position`：机器人基座位置 [x, y, z]
- `--camera_translation_relative_to_tcp`：相机相对于 TCP 的平移 [x, y, z]
- `--camera_rotation_relative_to_tcp`：相机相对于 TCP 的旋转 [w, x, y, z]（四元数，wxyz 格式）
- `--read_qpos_from_h5`：从 h5 文件中读取 qpos（标志位）
- `--qpos_timestep_offset`：读取 qpos 时的时间步偏移量（相对于 timestep，默认：5）

#### 姿态扰动相关参数

- `--joint_indices`：需要扰动的关节索引列表（例如：`0 2 4`）
- `--perturbations`：对应的扰动值列表（例如：`0.1 -0.1 0.05`）

## 函数说明

### 主要函数

#### `get_new_view_image()`

主函数，执行完整的新视角图像生成流程。

**参数**：
- `h5_path`：H5 文件路径
- `traj_idx`：轨迹序号
- `timestep`：起始时间步
- `agent_idx`：Agent 序号
- `n`：提取的步数（默认：3）
- `step_size`：时间步长（默认：1）
- `segment_server_url`：分割服务器 URL
- `transform_server_url`：变换服务器 URL
- `tgt_extrinsic`：目标外参（可选，3x4 矩阵）
- `urdf_filepath`：URDF 文件路径（可选）
- `qpos`：关节角度（可选）
- `robot_base_position`：机器人基座位置（可选）
- `camera_translation_relative_to_tcp`：相机相对于 TCP 的平移（可选）
- `camera_rotation_relative_to_tcp`：相机相对于 TCP 的旋转（可选）
- `read_qpos_from_h5`：是否从 h5 读取 qpos（默认：False）
- `qpos_timestep_offset`：qpos 时间步偏移（默认：30）
- `joint_indices`：需要扰动的关节索引（可选）
- `perturbations`：扰动值列表（可选）

**返回**：
- `final_image`：最终生成的图像（numpy 数组，[H, W, 3]，RGB 格式）
- `comparison_image`：对比图（如果 `read_qpos_from_h5=True`，否则为 None）

### 辅助函数

- `read_h5_camera_data()`：从 h5 文件读取相机数据
- `read_h5_qpos()`：从 h5 文件读取关节角度
- `compute_tgt_extrinsic_from_qpos()`：基于 qpos 计算目标外参
- `segment_image()`：对图像进行分割
- `inpaint_image()`：使用 inpainting 修复图像
- `transform_viewpoint()`：进行视角变换
- `apply_mask_to_image()`：将 mask 应用到图像
- `create_comparison_image()`：创建对比图

## 注意事项

### 重要提示

1. **相机位置要求**：为了让 DA3（视角变换）尺度对齐准确，传入的图片差异要大，即相机的位置尽量不要在一条基线上。

2. **时间步选择**：
   - `step_size` 建议设置为较大的值（如 10），以确保提取的图像有足够的视角差异
   - 确保 `timestep + (n-1) * step_size` 不超过轨迹的最大时间步

3. **服务器连接**：
   - 确保分割服务器和变换服务器正在运行
   - 检查服务器 URL 是否正确
   - 网络连接正常

4. **参数组合**：
   - 如果提供了 `urdf_filepath` 和相关相机参数，将使用 `qpos2camext` 计算目标外参
   - 如果未提供或计算失败，将使用最后一步的外参作为目标外参
   - 如果同时提供了 `joint_indices` 和 `perturbations`，将对 qpos 进行扰动后再计算外参

5. **输出文件**：
   - 如果使用 `--read_qpos_from_h5`，会额外生成对比图（文件名添加 `_comparison` 后缀）
   - 输出目录会自动创建

## 示例输出

程序执行时会输出详细的进度信息：

```
[1/5] Extracting camera data from h5 file...
  Extracting timesteps: [10, 20, 30]
  ✓ Extracted 3 images
[2/5] Computing target extrinsic...
  ✓ Read qpos from h5 file at timestep 15: shape (9,)
  ✓ Computed tgt_extrinsic from qpos2camext
[3/5] Segmenting and inpainting images at timestep 10...
  ✓ All 3 images segmented and inpainted
[4/5] Transforming viewpoint...
  ✓ Viewpoint transformation completed
[5/5] Applying mask to transformed image...
  ✓ Final image generated

[Extra] Creating comparison image...
  ✓ Read original image at timestep 15
  ✓ Comparison image created

✓ Final image saved to: outputs/new_view_image.png
✓ Comparison image saved to: outputs/new_view_image_comparison.png
```

## 相关文件

- `get_new_view_image.py`：主程序文件
- `qpos2camext.py`：相机外参计算模块
- `plan_new_pose.py`：姿态规划和验证模块
- `segment_client.py`：分割服务客户端（参考）
- `transform_client.py`：变换服务客户端（参考）

## 许可证
@zzb


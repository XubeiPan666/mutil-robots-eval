"""
将包含多个 h5 文件的目录直接转换为单个 zarr 文件。

用途：当你有多个扰动的 h5 文件（每个包含1条轨迹）时，
      可以用此脚本将它们合并成一个 zarr 文件，用于训练。

输入目录结构：
    h5_dir/
    ├── perturbed_0.h5
    ├── perturbed_1.h5
    ├── ...
    └── shared.json  (可选，所有h5共用)

输出 zarr 结构：
    output.zarr/
    ├── data/
    │   ├── img1      (N, 3, H, W) - 手腕相机
    │   ├── img2      (N, 3, H, W) - 全局相机
    │   ├── state     (N, 9) - 关节状态
    │   └── action    (N, 8) - 动作
    └── meta/
        └── episode_ends  (num_episodes,)

使用示例：
    python parse_h5_dir_to_zarr.py \\
        --h5_dir /path/to/h5_files \\
        --dst /path/to/output.zarr \\
        --agent_id 0
"""

import os
import re
import shutil
import argparse
import numpy as np
import h5py
import zarr
from tqdm import tqdm


def natural_sort_key(filename):
    """
    自然排序的 key 函数，使 perturbed_2.h5 排在 perturbed_10.h5 前面
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', filename)]


def parse_h5_dir_to_zarr(h5_dir: str, dst_zarr_path: str, agent_id: int = 0):
    """
    将目录中的多个 h5 文件合并转换为单个 zarr 文件
    
    Args:
        h5_dir: 包含多个 h5 文件的目录路径
        dst_zarr_path: 输出 zarr 文件路径
        agent_id: 要处理的 agent ID (默认 0)
    """
    # 检查源目录是否存在
    if not os.path.exists(h5_dir):
        raise FileNotFoundError(f"H5 directory not found: {h5_dir}")
    
    # 扫描目录，找到所有 h5 文件
    h5_files = [f for f in os.listdir(h5_dir) if f.endswith('.h5')]
    if not h5_files:
        raise ValueError(f"No .h5 files found in {h5_dir}")
    
    # 自然排序（使 perturbed_2 在 perturbed_10 前面）
    h5_files = sorted(h5_files, key=natural_sort_key)
    
    print(f"\n{'='*60}")
    print(f"Converting H5 Directory to Zarr")
    print(f"{'='*60}")
    print(f"  Source directory: {h5_dir}")
    print(f"  Destination: {dst_zarr_path}")
    print(f"  Agent ID: {agent_id}")
    print(f"  Found {len(h5_files)} h5 files:")
    for i, f in enumerate(h5_files[:5]):
        print(f"    [{i}] {f}")
    if len(h5_files) > 5:
        print(f"    ... and {len(h5_files) - 5} more")
    print(f"{'='*60}\n")
    
    # 如果目标已存在，先删除
    if os.path.exists(dst_zarr_path):
        print(f"Removing existing zarr: {dst_zarr_path}")
        shutil.rmtree(dst_zarr_path)
    
    # 定义相机名称
    wrist_camera = f"wrist_camera_agent{agent_id}"
    global_camera = "head_camera_global"
    
    # 收集所有数据
    all_img1 = []
    all_img2 = []
    all_state = []
    all_action = []
    episode_ends = []
    total_steps = 0
    
    print("Processing h5 files...")
    for h5_file in tqdm(h5_files, desc="Converting"):
        h5_path = os.path.join(h5_dir, h5_file)
        
        with h5py.File(h5_path, 'r') as f:
            # 找到轨迹 key（通常是 traj_0）
            traj_keys = [k for k in f.keys() if k.startswith('traj_')]
            if not traj_keys:
                print(f"Warning: No trajectory found in {h5_file}, skipping...")
                continue
            
            for traj_key in traj_keys:
                traj = f[traj_key]
                
                # 读取 action（这决定了有效帧数）
                actions = traj['actions'][:]  # (T, 8)
                n_steps = len(actions)
                
                # 读取 state (qpos)，取前 n_steps 帧与 action 对齐
                state = traj['obs']['agent']['qpos'][:n_steps]  # (T, 9)
                
                # 读取图像，取前 n_steps 帧
                img1 = traj['obs']['sensor_data'][wrist_camera]['rgb'][:n_steps]  # (T, H, W, 3)
                img2 = traj['obs']['sensor_data'][global_camera]['rgb'][:n_steps]  # (T, H, W, 3)
                
                # 追加到列表
                all_img1.append(img1)
                all_img2.append(img2)
                all_state.append(state)
                all_action.append(actions)
                
                # 更新 episode_ends
                total_steps += n_steps
                episode_ends.append(total_steps)
    
    if not all_action:
        raise ValueError("No valid trajectories found in any h5 file")
    
    # 拼接所有数据
    print("\nConcatenating data...")
    all_img1 = np.concatenate(all_img1, axis=0)      # (N, H, W, 3)
    all_img2 = np.concatenate(all_img2, axis=0)      # (N, H, W, 3)
    all_state = np.concatenate(all_state, axis=0)    # (N, 9)
    all_action = np.concatenate(all_action, axis=0)  # (N, 8)
    episode_ends = np.array(episode_ends, dtype='int64')
    
    print(f"  img1 shape: {all_img1.shape}")
    print(f"  img2 shape: {all_img2.shape}")
    print(f"  state shape: {all_state.shape}")
    print(f"  action shape: {all_action.shape}")
    print(f"  episode_ends: {episode_ends}")
    
    # 转换图像格式：NHWC -> NCHW
    print("\nConverting images to NCHW format...")
    all_img1 = np.moveaxis(all_img1, -1, 1)  # (N, 3, H, W)
    all_img2 = np.moveaxis(all_img2, -1, 1)  # (N, 3, H, W)
    print(f"  img1 shape (NCHW): {all_img1.shape}")
    print(f"  img2 shape (NCHW): {all_img2.shape}")
    
    # 创建 zarr
    print("\nSaving to zarr...")
    zarr_root = zarr.group(dst_zarr_path)
    zarr_data = zarr_root.create_group('data')
    zarr_meta = zarr_root.create_group('meta')
    
    # 压缩器
    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
    
    # 保存数据
    img_chunk_size = (min(100, all_img1.shape[0]),) + all_img1.shape[1:]
    state_chunk_size = (min(100, all_state.shape[0]), all_state.shape[1])
    action_chunk_size = (min(100, all_action.shape[0]), all_action.shape[1])
    
    zarr_data.create_dataset('img1', data=all_img1, chunks=img_chunk_size,
                             compressor=compressor)
    zarr_data.create_dataset('img2', data=all_img2, chunks=img_chunk_size,
                             compressor=compressor)
    zarr_data.create_dataset('state', data=all_state.astype('float32'),
                             chunks=state_chunk_size, compressor=compressor)
    zarr_data.create_dataset('action', data=all_action.astype('float32'),
                             chunks=action_chunk_size, compressor=compressor)
    zarr_meta.create_dataset('episode_ends', data=episode_ends, dtype='int64',
                             compressor=compressor)
    
    print(f"\n{'='*60}")
    print(f"✓ Successfully created {dst_zarr_path}")
    print(f"  Episodes: {len(episode_ends)}")
    print(f"  Total steps: {total_steps}")
    print(f"  Keys: img1 (wrist), img2 (global head), state, action")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Convert multiple H5 files to a single Zarr file.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python parse_h5_dir_to_zarr.py \\
        --h5_dir /path/to/h5_files \\
        --dst /path/to/output.zarr \\
        --agent_id 0

The script will:
1. Find all .h5 files in the directory
2. Sort them naturally (perturbed_2.h5 before perturbed_10.h5)
3. Extract img1 (wrist), img2 (global), state, action from each
4. Concatenate and save as a single zarr file
        """
    )
    parser.add_argument('--h5_dir', type=str, required=True,
                        help='Directory containing h5 files')
    parser.add_argument('--dst', type=str, required=True,
                        help='Destination zarr path')
    parser.add_argument('--agent_id', type=int, default=0,
                        help='Agent ID for wrist camera selection (default: 0)')
    
    args = parser.parse_args()
    
    parse_h5_dir_to_zarr(args.h5_dir, args.dst, args.agent_id)


if __name__ == '__main__':
    main()

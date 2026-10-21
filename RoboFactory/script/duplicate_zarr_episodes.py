"""
将单条 episode 的 zarr 复制 n 次，生成包含 n 条相同 episode 的新 zarr。

用途：当你有 1 条干净的 episode 和 n 条扰动的 episode 时，
      可以用此脚本把干净数据复制 n 次，使其与扰动数据的 episode 数量一致，
      从而可以直接用于 robot_image_dataset.py 的数据增强功能。

使用示例：
    python duplicate_zarr_episodes.py \
        --src data/zarr_data/clean_1.zarr \
        --dst data/zarr_data/clean_100.zarr \
        --n 100
"""

import os
import shutil
import argparse
import numpy as np
import zarr


def duplicate_zarr_episodes(src_zarr_path: str, dst_zarr_path: str, n_copies: int):
    """
    将单条 episode 的 zarr 复制 n 次，生成包含 n 条相同 episode 的新 zarr
    
    Args:
        src_zarr_path: 原始 zarr 路径（1条 episode）
        dst_zarr_path: 输出 zarr 路径（n条 episode）
        n_copies: 要复制的次数（应该和扰动 zarr 的 episode 数量一致）
    """
    # 检查源文件是否存在
    if not os.path.exists(src_zarr_path):
        raise FileNotFoundError(f"Source zarr not found: {src_zarr_path}")
    
    # 如果目标已存在，先删除
    if os.path.exists(dst_zarr_path):
        print(f"Removing existing zarr directory: {dst_zarr_path}")
        shutil.rmtree(dst_zarr_path)
    
    # 打开原始 zarr
    src_root = zarr.open(src_zarr_path, 'r')
    
    print(f"\n{'='*60}")
    print(f"Duplicating Zarr Episodes")
    print(f"{'='*60}")
    print(f"  Source: {src_zarr_path}")
    print(f"  Destination: {dst_zarr_path}")
    print(f"  Copies: {n_copies}")
    print(f"{'='*60}\n")
    
    # 验证源 zarr 只有 1 条 episode
    if 'meta' not in src_root or 'episode_ends' not in src_root['meta']:
        raise ValueError("Source zarr does not contain meta/episode_ends")
    
    episode_ends_original = src_root['meta']['episode_ends'][:]
    n_episodes_original = len(episode_ends_original)
    
    if n_episodes_original != 1:
        print(f"⚠ Warning: Source zarr has {n_episodes_original} episodes, not 1.")
        print(f"  Will duplicate all {n_episodes_original} episodes {n_copies} times.")
        print(f"  Result will have {n_episodes_original * n_copies} episodes.")
        response = input("  Continue? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    episode_length = episode_ends_original[-1]  # 总步数
    print(f"Original episode_ends: {episode_ends_original}")
    print(f"Total steps per copy: {episode_length}")
    
    # 创建目标 zarr
    dst_root = zarr.group(dst_zarr_path)
    
    # 复制 meta 信息
    dst_meta = dst_root.create_group('meta')
    
    # 生成新的 episode_ends
    # 如果原始有多条 episode，需要正确处理
    if n_episodes_original == 1:
        # 简单情况：[L, 2L, 3L, ..., nL]
        new_episode_ends = np.array([episode_length * (i + 1) for i in range(n_copies)], dtype='int64')
    else:
        # 复杂情况：需要复制原始的 episode 结构
        new_episode_ends_list = []
        for copy_idx in range(n_copies):
            offset = copy_idx * episode_length
            for orig_end in episode_ends_original:
                new_episode_ends_list.append(orig_end + offset)
        new_episode_ends = np.array(new_episode_ends_list, dtype='int64')
    
    dst_meta.create_dataset('episode_ends', data=new_episode_ends, dtype='int64')
    print(f"New episode_ends: {new_episode_ends}")
    print(f"Total episodes: {len(new_episode_ends)}")
    
    # 复制 data 信息
    if 'data' not in src_root:
        raise ValueError("Source zarr does not contain data group")
    
    dst_data = dst_root.create_group('data')
    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
    
    for key in src_root['data'].keys():
        original_data = src_root['data'][key][:]
        print(f"\nProcessing key: {key}")
        print(f"  Original shape: {original_data.shape}")
        
        # 将数据沿第 0 维复制 n 次
        # np.tile 需要正确的重复维度
        tile_reps = (n_copies,) + (1,) * (original_data.ndim - 1)
        repeated_data = np.tile(original_data, tile_reps)
        print(f"  Repeated shape: {repeated_data.shape}")
        
        # 确定 chunk 大小
        chunk_size = (min(100, repeated_data.shape[0]),) + repeated_data.shape[1:]
        
        dst_data.create_dataset(
            key, 
            data=repeated_data, 
            chunks=chunk_size,
            dtype=repeated_data.dtype,
            compressor=compressor
        )
    
    print(f"\n{'='*60}")
    print(f"✓ Successfully created {dst_zarr_path}")
    print(f"  Episodes: {len(new_episode_ends)}")
    print(f"  Total steps: {new_episode_ends[-1]}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Duplicate zarr episodes for data augmentation.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    # 将 1 条 episode 复制 100 次
    python duplicate_zarr_episodes.py \\
        --src data/zarr_data/clean_1episode.zarr \\
        --dst data/zarr_data/clean_100episodes.zarr \\
        --n 100
    
    # 然后在训练配置中使用：
    # zarr_path: data/zarr_data/clean_100episodes.zarr
    # augmented_zarr_path: data/zarr_data/perturbed_100episodes.zarr
        """
    )
    parser.add_argument('--src', type=str, required=True,
                        help='Source zarr path (with 1 episode)')
    parser.add_argument('--dst', type=str, required=True,
                        help='Destination zarr path')
    parser.add_argument('--n', type=int, required=True,
                        help='Number of copies (should match perturbed zarr episode count)')
    
    args = parser.parse_args()
    
    duplicate_zarr_episodes(args.src, args.dst, args.n)


if __name__ == '__main__':
    main()

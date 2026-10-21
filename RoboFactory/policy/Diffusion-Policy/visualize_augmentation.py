"""
可视化数据增强效果的验证脚本

用法:
    python visualize_augmentation.py \
        --zarr_path data/zarr_data/xxx.zarr \
        --augmented_zarr_path data/zarr_data/xxx_augmented.zarr \
        --sample_idx 0 \
        --output_dir ./augmentation_vis
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
import torch

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.dataset.robot_image_dataset import RobotImageDataset


def visualize_single_sample(dataset, sample_idx, output_dir, prefix=""):
    """
    可视化单个样本的序列数据
    直接调用 __getitem__ 方法，完全模拟训练时的数据加载行为
    """
    # 保存原始状态
    old_prob = dataset.augment_prob
    old_enabled = dataset.augment_enabled
    old_rng_state = dataset.augment_rng.get_state()
    
    # 获取原始采样（不增强）- 通过 __getitem__
    dataset.augment_enabled = False
    original_sample_torch = dataset[sample_idx]  # 调用 __getitem__
    # 转换回 numpy 用于可视化
    original_sample = {k: v.numpy() for k, v in original_sample_torch.items()}
    
    # 获取增强采样 - 通过 __getitem__
    dataset.augment_enabled = True
    dataset.augment_prob = 1.0  # 强制增强
    # 重置随机数生成器以确保一致性
    dataset.augment_rng.seed(42)
    augmented_sample_torch = dataset[sample_idx]  # 调用 __getitem__
    # 转换回 numpy 用于可视化
    augmented_sample = {k: v.numpy() for k, v in augmented_sample_torch.items()}
    
    # 恢复原始状态
    dataset.augment_prob = old_prob
    dataset.augment_enabled = old_enabled
    dataset.augment_rng.set_state(old_rng_state)
    
    # 获取增强帧的信息
    buffer_start_idx = dataset.sampler.indices[sample_idx][0]
    episode_id, frame_in_episode = dataset._get_episode_info(buffer_start_idx)
    aug_frame = dataset._get_augmented_frame(episode_id, frame_in_episode)
    
    # 创建可视化
    horizon = original_sample['head_camera'].shape[0]
    
    # 图1：对比原始序列和增强序列的图像
    fig1, axes = plt.subplots(3, horizon, figsize=(horizon * 2, 6))
    fig1.suptitle(f'Sample {sample_idx} - Episode {episode_id}, Frame {frame_in_episode}\n'
                  f'Prefix: {prefix}', fontsize=12)
    
    # 第一行：原始序列
    for t in range(horizon):
        img = original_sample['head_camera'][t]
        if img.shape[0] == 3:  # CHW -> HWC
            img = np.transpose(img, (1, 2, 0))
        axes[0, t].imshow(img.astype(np.uint8))
        axes[0, t].set_title(f'Orig t={t}', fontsize=8)
        axes[0, t].axis('off')
    axes[0, 0].set_ylabel('Original', fontsize=10)
    
    # 第二行：增强序列
    for t in range(horizon):
        img = augmented_sample['head_camera'][t]
        if img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        axes[1, t].imshow(img.astype(np.uint8))
        if t == 0:
            axes[1, t].set_title(f'Aug t={t}\n(REPLACED)', fontsize=8, color='red')
        else:
            axes[1, t].set_title(f'Aug t={t}\n(unchanged)', fontsize=8)
        axes[1, t].axis('off')
    axes[1, 0].set_ylabel('Augmented', fontsize=10)
    
    # 第三行：差异可视化
    for t in range(horizon):
        orig_img = original_sample['head_camera'][t]
        aug_img = augmented_sample['head_camera'][t]
        if orig_img.shape[0] == 3:
            orig_img = np.transpose(orig_img, (1, 2, 0))
            aug_img = np.transpose(aug_img, (1, 2, 0))
        diff = np.abs(orig_img.astype(float) - aug_img.astype(float))
        diff = (diff / diff.max() * 255).astype(np.uint8) if diff.max() > 0 else diff.astype(np.uint8)
        axes[2, t].imshow(diff)
        axes[2, t].set_title(f'Diff t={t}', fontsize=8)
        axes[2, t].axis('off')
    axes[2, 0].set_ylabel('Difference', fontsize=10)
    
    plt.tight_layout()
    fig1.savefig(os.path.join(output_dir, f'{prefix}sample_{sample_idx}_images.png'), dpi=150)
    plt.close(fig1)
    
    # 图2：对比状态和动作
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8))
    fig2.suptitle(f'Sample {sample_idx} - State and Action Comparison', fontsize=12)
    
    # 状态对比
    orig_state = original_sample['state']
    aug_state = augmented_sample['state']
    
    axes2[0, 0].plot(orig_state, 'o-', alpha=0.7, label=[f'dim{i}' for i in range(orig_state.shape[1])])
    axes2[0, 0].set_title('Original State')
    axes2[0, 0].set_xlabel('Time step')
    axes2[0, 0].set_ylabel('Value')
    axes2[0, 0].legend(fontsize=6, ncol=2)
    axes2[0, 0].grid(True, alpha=0.3)
    
    axes2[0, 1].plot(aug_state, 'o-', alpha=0.7)
    axes2[0, 1].set_title('Augmented State')
    axes2[0, 1].set_xlabel('Time step')
    axes2[0, 1].set_ylabel('Value')
    axes2[0, 1].grid(True, alpha=0.3)
    
    # 动作对比
    orig_action = original_sample['action']
    aug_action = augmented_sample['action']
    
    axes2[1, 0].plot(orig_action, 'o-', alpha=0.7)
    axes2[1, 0].set_title('Original Action')
    axes2[1, 0].set_xlabel('Time step')
    axes2[1, 0].set_ylabel('Value')
    axes2[1, 0].grid(True, alpha=0.3)
    
    axes2[1, 1].plot(aug_action, 'o-', alpha=0.7)
    axes2[1, 1].set_title('Augmented Action')
    axes2[1, 1].set_xlabel('Time step')
    axes2[1, 1].set_ylabel('Value')
    axes2[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig2.savefig(os.path.join(output_dir, f'{prefix}sample_{sample_idx}_state_action.png'), dpi=150)
    plt.close(fig2)
    
    # 打印详细信息
    print(f"\n{'='*60}")
    print(f"Sample {sample_idx} Info:")
    print(f"  Episode ID: {episode_id}")
    print(f"  Frame in Episode: {frame_in_episode}")
    print(f"  Buffer Start Index: {buffer_start_idx}")
    print(f"  Data loaded via: __getitem__ (same as training)")
    print(f"{'='*60}")
    
    print(f"\nOriginal Sample Shapes (from __getitem__, no augmentation):")
    for k, v in original_sample.items():
        print(f"  {k}: {v.shape}")
    
    print(f"\nAugmented Sample Shapes (from __getitem__, with augmentation):")
    for k, v in augmented_sample.items():
        print(f"  {k}: {v.shape}")
    
    print(f"\nAugmented Frame (inserted at t=0):")
    for k, v in aug_frame.items():
        print(f"  {k}: {v.shape}")
    
    # 验证增强逻辑（新逻辑：只替换第 0 帧）
    print(f"\n{'='*60}")
    print("Verification (using __getitem__ output):")
    print("New augmentation logic: only replace frame 0")
    print(f"{'='*60}")
    
    # 检查 head_camera[0] 是否被替换
    aug_first_frame = augmented_sample['head_camera'][0]
    expected_first_frame = aug_frame['head_camera'][0]
    img_first_match = np.allclose(aug_first_frame, expected_first_frame)
    print(f"\n[1] head_camera[0] replaced: {'✓ PASS' if img_first_match else '✗ FAIL'}")
    
    # 检查 state[0] 是否被替换
    state_first_match = np.allclose(augmented_sample['state'][0], aug_frame['state'][0])
    print(f"\n[2] state[0] replaced: {'✓ PASS' if state_first_match else '✗ FAIL'}")
    print(f"    原始 state[0]:     {original_sample['state'][0]}")
    print(f"    增强后 state[0]:   {augmented_sample['state'][0]}")
    print(f"    期望值(aug_frame): {aug_frame['state'][0].flatten()}")
    state_changed = not np.allclose(original_sample['state'][0], augmented_sample['state'][0])
    print(f"    state[0] 确实发生变化: {'✓ YES' if state_changed else '✗ NO (可能原始和增强数据相同)'}")
    
    # 检查 action[0] = 原始 state[1]
    action_first_match = np.allclose(augmented_sample['action'][0], original_sample['state'][1])
    print(f"\n[3] action[0] = original state[1]: {'✓ PASS' if action_first_match else '✗ FAIL'}")
    print(f"    原始 action[0]:    {original_sample['action'][0]}")
    print(f"    增强后 action[0]:  {augmented_sample['action'][0]}")
    print(f"    原始 state[1]:     {original_sample['state'][1]}")
    action_changed = not np.allclose(original_sample['action'][0], augmented_sample['action'][0])
    print(f"    action[0] 确实发生变化: {'✓ YES' if action_changed else '✗ NO (可能 action[0] 本来就等于 state[1])'}")
    
    # 检查其他帧是否保持不变（frame 1 到 horizon-1）
    print(f"\n[4] 检查其他帧 (t=1~{horizon-1}) 是否保持不变:")
    all_other_frames_unchanged = True
    for t in range(1, horizon):
        img_match = np.allclose(original_sample['head_camera'][t], augmented_sample['head_camera'][t])
        state_match = np.allclose(original_sample['state'][t], augmented_sample['state'][t])
        action_match = np.allclose(original_sample['action'][t], augmented_sample['action'][t])
        if not (img_match and state_match and action_match):
            all_other_frames_unchanged = False
            print(f"    Frame t={t}: ✗ FAIL (img={img_match}, state={state_match}, action={action_match})")
    if all_other_frames_unchanged:
        print(f"    所有帧 t=1~{horizon-1}: ✓ PASS (全部保持不变)")
    
    print(f"\n{'='*60}")
    print("总结:")
    all_pass = img_first_match and state_first_match and action_first_match and all_other_frames_unchanged
    print(f"  整体验证: {'✓ ALL PASS' if all_pass else '✗ SOME FAILED'}")
    print(f"{'='*60}\n")


def visualize_augmentation_comparison(zarr_path, augmented_zarr_path, output_dir, 
                                       sample_indices=None, horizon=32, seed=42):
    """
    可视化对比原始数据和增强数据
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading original data from: {zarr_path}")
    print(f"Loading augmented data from: {augmented_zarr_path}")
    
    # 创建数据集
    dataset = RobotImageDataset(
        zarr_path=zarr_path,
        augmented_zarr_path=augmented_zarr_path,
        horizon=horizon,
        pad_before=2,  # n_obs_steps - 1
        pad_after=7,   # n_action_steps - 1
        seed=seed,
        val_ratio=0.0,
        augment_prob=1.0,  # 强制增强用于可视化
        batch_size=64,
        max_train_episodes=None
    )
    
    print(f"\nDataset Info:")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Horizon: {horizon}")
    print(f"  N Episodes (original): {dataset.replay_buffer.n_episodes}")
    print(f"  N Episodes (augmented): {dataset.augmented_buffer.n_episodes}")
    print(f"  Test method: __getitem__ (identical to training data loading)")
    
    # 默认选择几个样本进行可视化
    if sample_indices is None:
        n_samples = min(5, len(dataset))
        sample_indices = np.linspace(0, len(dataset) - 1, n_samples, dtype=int)
    
    print(f"\nVisualizing samples: {sample_indices}")
    
    for idx in sample_indices:
        visualize_single_sample(dataset, idx, output_dir)
    
    print(f"\nVisualization saved to: {output_dir}")


def verify_consecutive_augmentation(dataset, episode_id, output_dir, start_frame=0, n=1):
    """
    验证连续 n x horizon 个帧的增强是否正确。
    检查从 start_frame 开始的 n 个 horizon 内，每个 horizon 的第 0 帧是否被正确替换。
    
    :param dataset: 数据集对象
    :param episode_id: episode 索引
    :param output_dir: 输出目录
    :param start_frame: 起始帧位置（在 episode 内）
    :param n: 要检查的 horizon 数量，默认 1
    """
    os.makedirs(output_dir, exist_ok=True)
    
    horizon = dataset.horizon
    total_frames_to_check = n * horizon
    
    # 找到属于该 episode 的所有采样索引
    episode_sample_indices = []
    for sample_idx in range(len(dataset)):
        buffer_start_idx = dataset.sampler.indices[sample_idx][0]
        ep_id, frame_in_ep = dataset._get_episode_info(buffer_start_idx)
        if ep_id == episode_id:
            episode_sample_indices.append((sample_idx, frame_in_ep))
    
    if len(episode_sample_indices) == 0:
        print(f"No samples found for episode {episode_id}")
        return False
    
    # 按帧位置排序
    episode_sample_indices.sort(key=lambda x: x[1])
    
    # 找到从 start_frame 开始的连续 horizon 样本
    selected_samples = []
    for sample_idx, frame_in_ep in episode_sample_indices:
        if frame_in_ep >= start_frame and len(selected_samples) < n:
            # 检查这个样本是否正好是 horizon 的起始位置
            if (frame_in_ep - start_frame) % horizon == 0:
                selected_samples.append((sample_idx, frame_in_ep))
    
    if len(selected_samples) < n:
        print(f"Warning: Only found {len(selected_samples)} samples starting from frame {start_frame}, need {n}")
        if len(selected_samples) == 0:
            return False
    
    print(f"\n{'='*80}")
    print(f"Verifying Consecutive Augmentation: {n} x {horizon} = {total_frames_to_check} frames")
    print(f"Episode {episode_id}, starting from frame {start_frame}")
    print(f"{'='*80}")
    
    # 保存原始状态
    old_prob = dataset.augment_prob
    old_enabled = dataset.augment_enabled
    
    all_pass = True
    verification_results = []
    
    for horizon_idx, (sample_idx, frame_in_ep) in enumerate(selected_samples):
        print(f"\n--- Horizon {horizon_idx + 1}/{n} (starting at frame {frame_in_ep}) ---")
        
        # 获取增强帧信息
        buffer_start_idx = dataset.sampler.indices[sample_idx][0]
        aug_frame = dataset._get_augmented_frame(episode_id, frame_in_ep)
        
        # 获取增强后的样本（通过 __getitem__）
        dataset.augment_enabled = True
        dataset.augment_prob = 1.0
        dataset.augment_rng.seed(42)
        augmented_sample_torch = dataset[sample_idx]
        augmented_sample = {k: v.numpy() for k, v in augmented_sample_torch.items()}
        
        # 获取原始样本（不增强）
        dataset.augment_enabled = False
        original_sample_torch = dataset[sample_idx]
        original_sample = {k: v.numpy() for k, v in original_sample_torch.items()}
        
        # 验证 head_camera[0]
        img_match = np.allclose(augmented_sample['head_camera'][0], aug_frame['head_camera'][0])
        
        # 验证 state[0]
        state_match = np.allclose(augmented_sample['state'][0], aug_frame['state'][0])
        
        # 验证 action[0][:7] = 原始 state[1][:7]（只检查前7维关节角）
        action_match = np.allclose(augmented_sample['action'][0][:7], original_sample['state'][1][:7])
        
        # 验证其他帧保持不变
        other_frames_ok = True
        for t in range(1, horizon):
            img_ok = np.allclose(original_sample['head_camera'][t], augmented_sample['head_camera'][t])
            state_ok = np.allclose(original_sample['state'][t], augmented_sample['state'][t])
            action_ok = np.allclose(original_sample['action'][t], augmented_sample['action'][t])
            if not (img_ok and state_ok and action_ok):
                other_frames_ok = False
                break
        
        horizon_pass = img_match and state_match and action_match and other_frames_ok
        if not horizon_pass:
            all_pass = False
        
        verification_results.append({
            'horizon_idx': horizon_idx,
            'frame_in_ep': frame_in_ep,
            'img_match': img_match,
            'state_match': state_match,
            'action_match': action_match,
            'other_frames_ok': other_frames_ok,
            'pass': horizon_pass
        })
        
        print(f"  head_camera[0] replaced: {'✓' if img_match else '✗'}")
        
        print(f"\n  state[0] replaced: {'✓' if state_match else '✗'}")
        print(f"    原始 state[0]:     {original_sample['state'][0]}")
        print(f"    增强后 state[0]:   {augmented_sample['state'][0]}")
        print(f"    期望值(aug_frame): {aug_frame['state'][0].flatten()}")
        state_changed = not np.allclose(original_sample['state'][0], augmented_sample['state'][0])
        print(f"    state[0] 确实发生变化: {'✓ YES' if state_changed else '✗ NO (可能原始和增强数据相同)'}")
        
        print(f"\n  action[0][:7] = original state[1][:7]: {'✓' if action_match else '✗'}")
        print(f"    原始 action[0]:    {original_sample['action'][0]}")
        print(f"    增强后 action[0]:  {augmented_sample['action'][0]}")
        print(f"    原始 state[1]:     {original_sample['state'][1]}")
        print(f"    检查前7维匹配:     action[0][:7] = {augmented_sample['action'][0][:7]}, state[1][:7] = {original_sample['state'][1][:7]}")
        action_changed = not np.allclose(original_sample['action'][0], augmented_sample['action'][0])
        print(f"    action[0] 确实发生变化: {'✓ YES' if action_changed else '✗ NO (可能 action[0] 本来就等于 state[1])'}")
        
        print(f"\n  Frames t=1~{horizon-1} unchanged: {'✓' if other_frames_ok else '✗'}")
        print(f"  Overall: {'✓ PASS' if horizon_pass else '✗ FAIL'}")
    
    # 为每个 horizon 生成类似 sample_0_images.png 格式的图片
    for horizon_idx, (result, (sample_idx, frame_in_ep)) in enumerate(zip(verification_results, selected_samples)):
        # 获取数据用于可视化
        dataset.augment_enabled = False
        original_sample_torch = dataset[sample_idx]
        original_sample = {k: v.numpy() for k, v in original_sample_torch.items()}
        
        dataset.augment_enabled = True
        dataset.augment_prob = 1.0
        dataset.augment_rng.seed(42)
        augmented_sample_torch = dataset[sample_idx]
        augmented_sample = {k: v.numpy() for k, v in augmented_sample_torch.items()}
        
        # 创建可视化：3 行 x horizon 列（和 sample_0_images.png 格式一样）
        fig, axes = plt.subplots(3, horizon, figsize=(horizon * 2, 6))
        fig.suptitle(f'Episode {episode_id}, Horizon {horizon_idx + 1}/{n} - Starting Frame {frame_in_ep}', fontsize=12)
        
        # 第一行：原始序列
        for t in range(horizon):
            img = original_sample['head_camera'][t]
            if img.shape[0] == 3:  # CHW -> HWC
                img = np.transpose(img, (1, 2, 0))
            axes[0, t].imshow(img.astype(np.uint8))
            axes[0, t].set_title(f'Orig t={t}', fontsize=8)
            axes[0, t].axis('off')
        axes[0, 0].set_ylabel('Original', fontsize=10)
        
        # 第二行：增强序列
        for t in range(horizon):
            img = augmented_sample['head_camera'][t]
            if img.shape[0] == 3:
                img = np.transpose(img, (1, 2, 0))
            axes[1, t].imshow(img.astype(np.uint8))
            if t == 0:
                axes[1, t].set_title(f'Aug t={t}\n(REPLACED)', fontsize=8, color='red')
            else:
                axes[1, t].set_title(f'Aug t={t}\n(unchanged)', fontsize=8)
            axes[1, t].axis('off')
        axes[1, 0].set_ylabel('Augmented', fontsize=10)
        
        # 第三行：差异可视化
        for t in range(horizon):
            orig_img = original_sample['head_camera'][t]
            aug_img = augmented_sample['head_camera'][t]
            if orig_img.shape[0] == 3:
                orig_img = np.transpose(orig_img, (1, 2, 0))
                aug_img = np.transpose(aug_img, (1, 2, 0))
            diff = np.abs(orig_img.astype(float) - aug_img.astype(float))
            diff = (diff / diff.max() * 255).astype(np.uint8) if diff.max() > 0 else diff.astype(np.uint8)
            axes[2, t].imshow(diff)
            axes[2, t].set_title(f'Diff t={t}', fontsize=8)
            axes[2, t].axis('off')
        axes[2, 0].set_ylabel('Difference', fontsize=10)
        
        plt.tight_layout()
        output_filename = f'episode_{episode_id}_horizon_{horizon_idx + 1}_frame_{frame_in_ep}_images.png'
        fig.savefig(os.path.join(output_dir, output_filename), dpi=150)
        plt.close(fig)
        print(f"  Saved visualization: {output_filename}")
    
    # 恢复原始状态
    dataset.augment_prob = old_prob
    dataset.augment_enabled = old_enabled
    
    print(f"\n{'='*80}")
    print(f"Overall Result: {'✓ ALL PASS' if all_pass else '✗ SOME FAILED'}")
    print(f"Visualizations saved to: {output_dir}/episode_{episode_id}_horizon_*_images.png")
    print(f"{'='*80}\n")
    
    return all_pass


def visualize_episode_comparison(zarr_path, augmented_zarr_path, output_dir, episode_id=0):
    """
    可视化对比同一个 episode 在原始数据和增强数据中的差异
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    original_buffer = ReplayBuffer.copy_from_path(
        zarr_path, keys=['head_camera', 'state', 'action']
    )
    augmented_buffer = ReplayBuffer.copy_from_path(
        augmented_zarr_path, keys=['head_camera', 'state', 'action']
    )
    
    # 获取 episode 数据
    orig_episode = original_buffer.get_episode(episode_id)
    aug_episode = augmented_buffer.get_episode(episode_id)
    
    print(f"\nEpisode {episode_id} Info:")
    print(f"  Original length: {len(orig_episode['head_camera'])}")
    print(f"  Augmented length: {len(aug_episode['head_camera'])}")
    
    # 可视化前几帧的对比
    n_frames = min(10, len(orig_episode['head_camera']), len(aug_episode['head_camera']))
    
    fig, axes = plt.subplots(2, n_frames, figsize=(n_frames * 2, 4))
    fig.suptitle(f'Episode {episode_id} - Original vs Augmented', fontsize=12)
    
    for t in range(n_frames):
        # 原始
        img = orig_episode['head_camera'][t]
        if img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        axes[0, t].imshow(img.astype(np.uint8))
        axes[0, t].set_title(f't={t}', fontsize=8)
        axes[0, t].axis('off')
        
        # 增强
        img = aug_episode['head_camera'][t]
        if img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        axes[1, t].imshow(img.astype(np.uint8))
        axes[1, t].axis('off')
    
    axes[0, 0].set_ylabel('Original', fontsize=10)
    axes[1, 0].set_ylabel('Augmented', fontsize=10)
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f'episode_{episode_id}_comparison.png'), dpi=150)
    plt.close(fig)
    
    print(f"Episode comparison saved to: {output_dir}/episode_{episode_id}_comparison.png")


def main():
    parser = argparse.ArgumentParser(description='Visualize data augmentation')
    parser.add_argument('--zarr_path', type=str, required=True,
                        help='Path to original zarr data')
    parser.add_argument('--augmented_zarr_path', type=str, required=True,
                        help='Path to augmented zarr data')
    parser.add_argument('--output_dir', type=str, default='./augmentation_vis',
                        help='Output directory for visualizations')
    parser.add_argument('--sample_idx', type=int, nargs='+', default=None,
                        help='Sample indices to visualize (default: auto select)')
    parser.add_argument('--episode_id', type=int, default=0,
                        help='Episode ID for episode comparison')
    parser.add_argument('--horizon', type=int, default=32,
                        help='Sequence horizon')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--mode', type=str, choices=['sample', 'episode', 'consecutive', 'all'], 
                        default='all', help='Visualization mode: sample, episode, consecutive, or all')
    parser.add_argument('--n', type=int, default=1,
                        help='Number of consecutive horizons to verify (for consecutive mode), default=1')
    parser.add_argument('--start_frame', type=int, default=0,
                        help='Starting frame position in episode (for consecutive mode), default=0')
    
    args = parser.parse_args()
    
    # 创建数据集用于 consecutive 模式
    dataset = None
    if args.mode in ['consecutive', 'all']:
        dataset = RobotImageDataset(
            zarr_path=args.zarr_path,
            augmented_zarr_path=args.augmented_zarr_path,
            horizon=args.horizon,
            pad_before=2,
            pad_after=7,
            seed=args.seed,
            val_ratio=0.0,
            augment_prob=1.0,
            batch_size=64,
            max_train_episodes=None
        )
    
    if args.mode in ['sample', 'all']:
        print("\n" + "="*60)
        print("Visualizing Sample Augmentation")
        print("="*60)
        visualize_augmentation_comparison(
            zarr_path=args.zarr_path,
            augmented_zarr_path=args.augmented_zarr_path,
            output_dir=args.output_dir,
            sample_indices=args.sample_idx,
            horizon=args.horizon,
            seed=args.seed
        )
    
    if args.mode in ['episode', 'all']:
        print("\n" + "="*60)
        print("Visualizing Episode Comparison")
        print("="*60)
        visualize_episode_comparison(
            zarr_path=args.zarr_path,
            augmented_zarr_path=args.augmented_zarr_path,
            output_dir=args.output_dir,
            episode_id=args.episode_id
        )
    
    if args.mode in ['consecutive', 'all']:
        print("\n" + "="*60)
        print(f"Verifying Consecutive Augmentation: {args.n} x {args.horizon} = {args.n * args.horizon} frames")
        print("="*60)
        verify_consecutive_augmentation(
            dataset=dataset,
            episode_id=args.episode_id,
            output_dir=args.output_dir,
            start_frame=args.start_frame,
            n=args.n
        )


if __name__ == '__main__':
    main()

from typing import Dict, Optional
import numbers
import numba
import torch
import numpy as np
import copy
import zarr
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer
import pdb

class RobotImageDataset(BaseImageDataset):
    def __init__(self,
                 zarr_path, 
                 horizon=1,
                 pad_before=0,
                 pad_after=0,
                 seed=42,
                 val_ratio=0.0,
                 batch_size=64,
                 max_train_episodes=None,
                 augmented_zarr_path: Optional[str]=None,
                 augment_prob: float=0.5):
        """
        初始化数据集对象，配置采样器，数据缓冲区等。

        :param zarr_path: 数据集的路径，包含回放缓存（replay buffer）的文件。
        :param horizon: 采样序列的长度。
        :param pad_before: 序列前面需要填充的空白。
        :param pad_after: 序列后面需要填充的空白。
        :param seed: 随机种子，用于生成验证集掩码。
        :param val_ratio: 验证集所占比例。
        :param batch_size: 批次大小。
        :param max_train_episodes: 最大训练集的集数，如果为 `None` 则没有限制。
        :param augmented_zarr_path: 增强数据的 zarr 路径，为 None 时不加载增强数据。
        :param augment_prob: 增强概率，范围 [0, 1]，默认 0.5。
        """
        
        # 自动检测zarr文件中的camera keys
        # 支持单相机(head_camera)和双相机(img1, img2)格式
        super().__init__()
        
        # 检测zarr中的keys
        zarr_root = zarr.open(zarr_path, 'r')
        available_keys = list(zarr_root['data'].keys())
        
        # 确定使用哪些image keys
        image_keys = []
        if 'img1' in available_keys and 'img2' in available_keys:
            # 双相机模式 (用于pairwise预训练权重)
            image_keys = ['img1', 'img2']
            print(f"\n{'='*60}")
            print(f"Detected DUAL CAMERA mode:")
            print(f"  img1 = wrist camera")
            print(f"  img2 = global head camera")
            print(f"{'='*60}\n")
        elif 'head_camera' in available_keys:
            # 单相机模式 (传统方式)
            image_keys = ['head_camera']
            print(f"\n{'='*60}")
            print(f"Detected SINGLE CAMERA mode:")
            print(f"  head_camera")
            print(f"{'='*60}\n")
        else:
            raise ValueError(
                f"Could not find valid camera keys in zarr data.\n"
                f"Expected either ['head_camera'] or ['img1', 'img2'].\n"
                f"Available keys: {available_keys}"
            )
        
        # 加载回放缓存，指定数据键（包括摄像头图像、状态和动作）
        load_keys = image_keys + ['state', 'action']
        print(f"Loading zarr data with keys: {load_keys}")
        
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path,
            keys=load_keys
        )
        self.image_keys = image_keys  # 保存image keys供后续使用
        
        # 加载增强数据的回放缓存（如果提供了路径）
        self.augmented_buffer = None
        self.augment_prob = augment_prob
        self.augment_enabled = True  # 训练时启用，验证时禁用
        self.augment_count = 0  # 统计增强被应用的次数
        self.total_samples = 0  # 统计总样本数
        
        print(f"\n{'='*60}")
        print(f"Data Augmentation Configuration:")
        print(f"  augmented_zarr_path: {augmented_zarr_path}")
        print(f"  augment_prob: {augment_prob}")
        print(f"  augment_enabled: {self.augment_enabled}")
        
        if augmented_zarr_path is not None:
            print(f"  Loading augmented data from: {augmented_zarr_path}")
            self.augmented_buffer = ReplayBuffer.copy_from_path(
                augmented_zarr_path,
                keys=load_keys  # 使用与原始数据相同的keys
            )
            # 验证增强数据与原始数据的 episode 数量一致
            assert self.augmented_buffer.n_episodes == self.replay_buffer.n_episodes, \
                f"增强数据 episode 数量 ({self.augmented_buffer.n_episodes}) 与原始数据 ({self.replay_buffer.n_episodes}) 不一致"
            print(f"  ✓ Augmented data loaded successfully!")
            print(f"    - Original episodes: {self.replay_buffer.n_episodes}")
            print(f"    - Augmented episodes: {self.augmented_buffer.n_episodes}")
            print(f"    - Expected augmentation rate: {augment_prob * 100:.1f}%")
        else:
            print(f"  ✗ No augmented data (augmented_zarr_path is None)")
        print(f"{'='*60}\n")
        
        # 创建用于增强的随机数生成器（可复现）
        self.augment_rng = np.random.RandomState(seed)
            
        # 生成验证集掩码
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask  # 训练集掩码（对验证集掩码取反）
        
        # 根据最大训练集数量对训练集掩码进行下采样
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        # 创建 SequenceSampler，控制采样序列的长度和填充方式
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        
        # 保存训练集掩码、序列长度和填充参数
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.batch_size = batch_size
        self.seed = seed
        
        # 预计算 episode 起始索引，用于快速查找
        episode_ends = self.replay_buffer.episode_ends[:]
        self.episode_ends = episode_ends
        self.episode_starts = np.concatenate([[0], episode_ends[:-1]])

        # 初始化缓冲区，存储每个样本的序列
        sequence_length = self.sampler.sequence_length
        self.buffers = {
            k: np.zeros((batch_size, sequence_length, *v.shape[1:]), dtype=v.dtype)
            for k, v in self.sampler.replay_buffer.items()
        }
        # 将缓冲区从 NumPy 转换为 PyTorch 张量，并固定内存
        # 注意：pin_memory() 返回新张量，需要保存返回值
        self.buffers_torch = {k: torch.from_numpy(v).pin_memory() for k, v in self.buffers.items()}

    def get_validation_dataset(self):
        """
        获取验证数据集对象，通过复制当前数据集并使用验证集掩码创建新的采样器。
        验证集禁用数据增强。

        :return: 验证数据集对象
        """
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask  # 使用反转的训练集掩码
        )
        val_set.train_mask = ~self.train_mask  # 验证集掩码
        val_set.augment_enabled = False  # 验证集禁用增强
        # 为验证集创建独立的随机数生成器
        val_set.augment_rng = np.random.RandomState(self.seed + 1000)
        return val_set
    
    def _get_episode_info(self, buffer_idx: int):
        """
        根据全局 buffer 索引，反推出 episode_id 和该索引在 episode 内的帧号。

        :param buffer_idx: 全局 buffer 索引
        :return: (episode_id, frame_in_episode) 元组
        """
        # 使用二分查找找到 episode_id
        episode_id = np.searchsorted(self.episode_ends, buffer_idx, side='right')
        # 计算在 episode 内的帧号
        episode_start = self.episode_starts[episode_id]
        frame_in_episode = buffer_idx - episode_start
        return episode_id, frame_in_episode
    
    def _get_augmented_frame(self, episode_id: int, frame_in_episode: int):
        """
        从增强 zarr 中获取指定 episode 和帧号的数据。
        获取图像(单相机或双相机)和 state，action 由原始数据的 state[1] 提供。

        :param episode_id: episode 索引
        :param frame_in_episode: 在 episode 内的帧号
        :return: 包含单帧数据的字典（图像 + state）
        """
        # 计算增强数据中的全局索引
        aug_episode_ends = self.augmented_buffer.episode_ends[:]
        aug_episode_start = 0 if episode_id == 0 else aug_episode_ends[episode_id - 1]
        aug_global_idx = aug_episode_start + frame_in_episode
        
        # 确保索引在有效范围内
        aug_episode_end = aug_episode_ends[episode_id]
        aug_global_idx = min(aug_global_idx, aug_episode_end - 1)
        
        # 获取单帧数据（图像keys + state，action 由原始数据提供）
        result = {}
        for key in self.image_keys + ['state']:
            result[key] = self.augmented_buffer[key][aug_global_idx:aug_global_idx+1]
        return result
    
    def _apply_augmentation(self, sample: dict, idx: int):
        """
        对采样的序列应用增强：直接替换第 0 帧。
        - 图像[0] = 增强 zarr 的图像 (支持单/双相机)
        - state[0] = 增强 zarr 的 state
        - action[0] = 原始数据的 state[1]
        - 其他帧保持不变

        :param sample: 原始采样的序列数据
        :param idx: 采样索引
        :return: 增强后的序列数据
        """
        # 检查 horizon 是否大于 1（需要 state[1] 来设置 action[0]）
        if sample['state'].shape[0] < 2:
            # horizon=1 时无法获取 state[1]，跳过增强
            return sample
        
        # 获取采样的起始全局索引
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = self.sampler.indices[idx]
        
        # 反推 episode_id 和帧号
        episode_id, frame_in_episode = self._get_episode_info(buffer_start_idx)
        
        # 从增强 zarr 获取对应帧（图像 + state）
        aug_frame = self._get_augmented_frame(episode_id, frame_in_episode)
        
        # 创建增强后的样本（复制原始数据）
        augmented_sample = {key: sample[key].copy() for key in sample.keys()}
        
        # 替换第 0 帧的图像和 state
        # 双相机模式下只替换 img1 (wrist camera)，img2 (head camera) 保持不变
        if 'img1' in self.image_keys:
            augmented_sample['img1'][0] = aug_frame['img1'][0]
        else:
            # 单相机模式，替换 head_camera
            for img_key in self.image_keys:
                augmented_sample[img_key][0] = aug_frame[img_key][0]
        augmented_sample['state'][0] = aug_frame['state'][0]
        
        # action[0] = 原始数据的 action[0]（只赋值关节角，夹爪保持不变）
        augmented_sample['action'][0] = sample['action'][0]
        
        return augmented_sample
    
    def _apply_batch_augmentation(self, idx: np.ndarray):
        """
        对批量采样的数据应用增强：直接替换第 0 帧。
        - 图像[0] = 增强 zarr 的图像 (支持单/双相机)
        - state[0] = 增强 zarr 的 state
        - action[0] = 原始数据的 action[0]
        直接修改 self.buffers 和 self.buffers_torch。

        :param idx: 批量索引数组
        """
        # 检查 horizon 是否大于 1
        if self.sampler.sequence_length < 2:
            return
        
        # 预计算增强数据的 episode 信息
        aug_episode_ends = self.augmented_buffer.episode_ends[:]
        aug_episode_starts = np.concatenate([[0], aug_episode_ends[:-1]])
        
        for i, sample_idx in enumerate(idx):
            # 判断是否对该样本应用增强
            self.total_samples += 1
            if self.augment_rng.random() >= self.augment_prob:
                continue
            self.augment_count += 1
            
            # 获取采样的起始全局索引
            buffer_start_idx = self.sampler.indices[sample_idx][0]
            
            # 反推 episode_id 和帧号
            episode_id, frame_in_episode = self._get_episode_info(buffer_start_idx)
            
            # 计算增强数据中的全局索引
            aug_global_idx = aug_episode_starts[episode_id] + frame_in_episode
            aug_global_idx = min(aug_global_idx, aug_episode_ends[episode_id] - 1)
            
            # 保存原始 action[0]
            original_action_0 = self.buffers['action'][i, 0].copy()
            
            # 替换第 0 帧的图像和 state
            # 双相机模式下只替换 img1 (wrist camera)，img2 (head camera) 保持不变
            if 'img1' in self.image_keys:
                self.buffers['img1'][i, 0] = self.augmented_buffer['img1'][aug_global_idx]
            else:
                # 单相机模式，替换 head_camera
                for img_key in self.image_keys:
                    self.buffers[img_key][i, 0] = self.augmented_buffer[img_key][aug_global_idx]
            self.buffers['state'][i, 0] = self.augmented_buffer['state'][aug_global_idx]

            # action[0] 保持不变（根据任务需求可调整）
            # self.buffers['action'][i, 0] = original_action_0
        
        # 打印增强统计信息
        # if self.total_samples > 0 and self.total_samples % (200 * self.batch_size) == 0:
        #     rate = self.augment_count / self.total_samples * 100
        #     print(f"[Augmentation Stats] Applied: {self.augment_count}/{self.total_samples} ({rate:.1f}%), Expected: {self.augment_prob*100:.1f}%")

        # 更新 torch 缓冲区
        for k in self.buffers.keys():
            self.buffers_torch[k].copy_(torch.from_numpy(self.buffers[k]))

    @staticmethod
    def qpos_to_action(qpos: np.ndarray) -> np.ndarray:
        """
        Convert 9-dim qpos to 8-dim action (numpy version for normalizer computation).
        
        qpos: [..., 9] - [joint_0, ..., joint_6, gripper_left, gripper_right]
        action: [..., 8] - [joint_0, ..., joint_6, gripper_normalized]
        
        Gripper mapping:
        - qpos gripper range: [0.032, 0.04] (physical width in meters)
        - action gripper range: [-1, 1] (normalized command)
        """
        joints = qpos[..., :7]  # [..., 7]
        
        # Average of two gripper fingers
        gripper_phys = (qpos[..., 7] + qpos[..., 8]) / 2  # [...]
        
        # Map from [0.032, 0.04] to [-1, 1]
        gripper_min, gripper_max = 0.032, 0.04
        gripper_norm = 2 * (gripper_phys - gripper_min) / (gripper_max - gripper_min) - 1
        gripper_norm = np.clip(gripper_norm, -1, 1)
        
        # Concatenate: [..., 7] + [..., 1] -> [..., 8]
        return np.concatenate([joints, gripper_norm[..., np.newaxis]], axis=-1)

    def get_normalizer(self, mode='limits', **kwargs):
        """
        获取数据归一化器，返回用于归一化动作和状态的归一化器。

        :param mode: 归一化模式（如 'limits'）。
        :param kwargs: 其他归一化器的参数。
        :return: 正常化器对象
        """
        # 从回放缓存中提取数据
        action = self.replay_buffer['action']  # (N, 8)
        state = self.replay_buffer['state']    # (N, 9)
        
        data = {
            'action': action,
            'agent_pos': state
        }
        
        # 计算 delta_action normalizer
        # NOTE: Only joints (dim 0-6) use relative position (delta), gripper (dim 7) uses absolute position
        # delta_action = [action[:7] - state[:7], action[7]] (gripper is absolute, not delta)
        state_as_action = self.qpos_to_action(state)  # (N, 8)
        delta_joints = action[..., :7] - state_as_action[..., :7]  # (N, 7) - relative position
        gripper_abs = action[..., 7:8]  # (N, 1) - absolute position (not delta)
        delta_action = np.concatenate([delta_joints, gripper_abs], axis=-1)  # (N, 8)
        data['delta_action'] = delta_action
        
        # 创建并拟合归一化器
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        
        # 打印 delta_action 统计信息
        print(f"\n{'='*60}")
        print(f"Delta Action Statistics (for relative action mode):")
        print(f"  NOTE: Joints (dim 0-6) are delta, Gripper (dim 7) is absolute")
        print(f"  delta_action shape: {delta_action.shape}")
        print(f"  delta_action mean:  {np.mean(delta_action, axis=0)}")
        print(f"  delta_action std:   {np.std(delta_action, axis=0)}")
        print(f"  delta_action min:   {np.min(delta_action, axis=0)}")
        print(f"  delta_action max:   {np.max(delta_action, axis=0)}")
        print(f"{'='*60}\n")
        
        # 为相机图像获取归一化器 (支持单/双相机)
        # 单相机: head_cam
        # 双相机: img1, img2
        if 'img1' in self.image_keys and 'img2' in self.image_keys:
            normalizer['img1'] = get_image_range_normalizer()
            normalizer['img2'] = get_image_range_normalizer()
        else:
            normalizer['head_cam'] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        """
        返回数据集的大小（即采样器的大小）。

        :return: 数据集大小
        """
        return len(self.sampler)

    def _sample_to_data(self, sample):
        """
        将样本从原始数据格式转换为标准化的字典格式。
        注意：此函数当前未被使用，保留用于未来可能的需求。

        :param sample: 一个样本，包含摄像头图像、状态和动作。
        :return: 标准化后的数据字典
        """
        agent_pos = sample['state'].astype(np.float32)  # 处理状态信息
        head_cam = np.moveaxis(sample['head_camera'], -1, 1) / 255  # 归一化头部摄像头图像

        # 将处理后的数据存储在字典中
        data = {
            'obs': {
                'head_cam': head_cam,    # T, 3, H, W
                'agent_pos': agent_pos,  # T, D
            },
            'action': sample['action'].astype(np.float32)  # T, D
        }
        return data

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """
        通过索引获取数据集中的一个样本，并转换为 PyTorch 张量格式。
        如果启用了增强且随机数小于 augment_prob，则应用增强。

        :param idx: 索引，支持整数、切片和批量索引。
        :return: 转换后的数据字典
        """
        # import pdb; pdb.set_trace()
        if isinstance(idx, slice):
            raise NotImplementedError  # 尚未实现切片索引
        elif isinstance(idx, numbers.Integral):
            # 获取单个样本（支持 int 和 np.int64 等整数类型）
            idx = int(idx)  # 转换为 Python int
            sample = self.sampler.sample_sequence(idx)
            
            # 判断是否应用增强
            self.total_samples += 1
            if (self.augmented_buffer is not None and 
                self.augment_enabled and 
                self.augment_rng.random() < self.augment_prob):
                sample = self._apply_augmentation(sample, idx)
                self.augment_count += 1
                # 每20个样本打印一次统计信息（适合小数据集，20条episode）
                if self.total_samples % 20 == 0:
                    rate = self.augment_count / self.total_samples * 100
                    print(f"[Augmentation Stats] Applied: {self.augment_count}/{self.total_samples} ({rate:.1f}%), Expected: {self.augment_prob*100:.1f}%")
            
            # 转换为 PyTorch 张量
            sample = dict_apply(sample, torch.from_numpy)
            return sample
        elif isinstance(idx, np.ndarray):
            # 批量索引
            assert len(idx) == self.batch_size
            # 从回放缓存中批量采样数据
            for k, v in self.sampler.replay_buffer.items():
                batch_sample_sequence(self.buffers[k], v, self.sampler.indices, idx, self.sampler.sequence_length)
            
            # 对批量数据应用增强
            if (self.augmented_buffer is not None and 
                self.augment_enabled):
                self._apply_batch_augmentation(idx)
            else:
                # 即使不增强，也需要将 numpy 数据同步到 torch 缓冲区
                # 因为 pin_memory() 后两者不再共享内存
                for k in self.buffers.keys():
                    self.buffers_torch[k].copy_(torch.from_numpy(self.buffers[k]))
            
            return self.buffers_torch
        else:
            raise ValueError(idx)

    def get_augmentation_stats(self):
        """
        获取数据增强的统计信息
        
        Returns:
            dict: 包含增强统计信息的字典
        """
        if self.total_samples == 0:
            return {
                'total_samples': 0,
                'augmented_samples': 0,
                'augmentation_rate': 0.0,
                'expected_rate': self.augment_prob * 100,
                'is_enabled': self.augment_enabled and self.augmented_buffer is not None
            }
        
        actual_rate = self.augment_count / self.total_samples * 100
        return {
            'total_samples': self.total_samples,
            'augmented_samples': self.augment_count,
            'augmentation_rate': actual_rate,
            'expected_rate': self.augment_prob * 100,
            'is_enabled': self.augment_enabled and self.augmented_buffer is not None
        }
    
    def postprocess(self, samples, device):
        """
        将采样的数据转换到设备并标准化。

        :param samples: 采样的数据。
        :param device: PyTorch 设备。
        :return: 经过标准化后的数据字典
        """
        agent_pos = samples['state'].to(device, non_blocking=True)
        action = samples['action'].to(device, non_blocking=True)
        
        # 构建obs字典，支持单/双相机
        obs = {'agent_pos': agent_pos}  # B, T, D
        
        if 'img1' in self.image_keys and 'img2' in self.image_keys:
            # 双相机模式
            obs['img1'] = samples['img1'].to(device, non_blocking=True) / 255.0  # B, T, 3, H, W
            obs['img2'] = samples['img2'].to(device, non_blocking=True) / 255.0  # B, T, 3, H, W
        else:
            # 单相机模式
            obs['head_cam'] = samples['head_camera'].to(device, non_blocking=True) / 255.0  # B, T, 3, H, W
        
        return {
            'obs': obs,
            'action': action  # B, T, D
        }

def _batch_sample_sequence(data: np.ndarray, input_arr: np.ndarray, indices: np.ndarray, idx: np.ndarray, sequence_length: int):
    """
    批量采样序列，将数据从回放缓存复制到目标数据缓冲区。

    :param data: 目标数据数组。
    :param input_arr: 输入数组（来自回放缓存）。
    :param indices: 索引数组，用于确定每个序列的起始和结束位置。
    :param idx: 需要采样的索引。
    :param sequence_length: 序列长度。
    """
    for i in numba.prange(len(idx)):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = indices[idx[i]]
        # 将数据从缓存复制到目标数据数组
        data[i, sample_start_idx:sample_end_idx] = input_arr[buffer_start_idx:buffer_end_idx]
        if sample_start_idx > 0:
            data[i, :sample_start_idx] = data[i, sample_start_idx]  # 前部分填充
        if sample_end_idx < sequence_length:
            data[i, sample_end_idx:] = data[i, sample_end_idx - 1]  # 后部分填充

# 并行和顺序版本的采样函数
_batch_sample_sequence_sequential = numba.jit(_batch_sample_sequence, nopython=True, parallel=False)
_batch_sample_sequence_parallel = numba.jit(_batch_sample_sequence, nopython=True, parallel=True)

def batch_sample_sequence(data: np.ndarray, input_arr: np.ndarray, indices: np.ndarray, idx: np.ndarray, sequence_length: int):
    """
    根据批次大小选择适当的并行或顺序版本的批量采样。

    :param data: 目标数据数组。
    :param input_arr: 输入数据数组。
    :param indices: 索引数组。
    :param idx: 当前批次的索引。
    :param sequence_length: 序列长度。
    """
    batch_size = len(idx)
    assert data.shape == (batch_size, sequence_length, *input_arr.shape[1:])
    if batch_size >= 16 and data.nbytes // batch_size >= 2 ** 16:
        # 如果数据较大，使用并行版本
        _batch_sample_sequence_parallel(data, input_arr, indices, idx, sequence_length)
    else:
        # 否则使用顺序版本
        _batch_sample_sequence_sequential(data, input_arr, indices, idx, sequence_length)

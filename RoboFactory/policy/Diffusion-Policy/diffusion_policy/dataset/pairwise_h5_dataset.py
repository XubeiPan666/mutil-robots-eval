"""
Pairwise H5 Dataset for learning visual-state correspondence.

Given 21 h5 files (1 clean + 20 perturbed), for each frame index,
we create pairwise combinations (img1, state1, img2, state2) where:
- Input: state1, img1, img2
- Target: action2 (converted from state2)

The model learns to predict what action to take given the current state
and visual differences between two perturbation conditions.

This version is adapted to work with the existing Diffusion Policy architecture.
"""

from typing import Dict, List, Optional, Tuple
import copy
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from itertools import combinations_with_replacement

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset


class PairwiseH5Dataset(BaseImageDataset):
    """
    Dataset for pairwise combination of multiple H5 files.
    
    For each frame index across all trajectories, creates pairs from 
    different H5 files (different perturbation conditions).
    
    Adapted to work with existing Diffusion Policy architecture:
    - Output format matches DP expectations (with time dimension)
    - Provides get_normalizer() and postprocess() methods
    """
    
    def __init__(
        self,
        h5_files: Optional[List[str]] = None,
        h5_dir: Optional[str] = None,
        h5_clean: Optional[str] = None,
        h5_pattern: str = "*.h5",
        agent_id: int = 0,
        camera_name: str = 'head_camera_global',
        pairs_per_index: int = 50,
        seed: int = 42,
        val_ratio: float = 0.02,
        preload_to_memory: bool = True,
        # DP compatibility parameters
        horizon: int = 8,  # UNet needs horizon >= 8 for convolutions
        pad_before: int = 0,
        pad_after: int = 0,
        max_train_episodes: Optional[int] = None,
        n_obs_steps: int = 1,  # Number of observation frames
    ):
        """
        Initialize the pairwise H5 dataset.
        
        Args:
            h5_files: List of paths to H5 files (e.g., 1 clean + 20 perturbed)
                      If None, h5_dir or h5_clean+h5_dir must be provided.
            h5_dir: Directory containing H5 files. All .h5 files will be auto-discovered.
                    If h5_clean is also provided, clean file will be prepended to the list.
            h5_clean: Path to clean H5 file. If provided with h5_dir, clean file will be
                     added first, followed by all files from h5_dir.
            h5_pattern: Glob pattern for H5 files (default: "*.h5")
            agent_id: Which agent to use (0 or 1 for panda-0 or panda-1)
            camera_name: Camera to use for images
            pairs_per_index: Number of random pairs to sample per frame index
            seed: Random seed for reproducibility
            val_ratio: Ratio of trajectories to use for validation
            preload_to_memory: Whether to preload all data into memory
            horizon: DP compatibility (UNet needs >= 8)
            pad_before: DP compatibility (always 0)
            pad_after: DP compatibility (always 0)
            max_train_episodes: Maximum number of training episodes
        """
        super().__init__()
        
        # Store DP compatibility params
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.max_train_episodes = max_train_episodes
        
        # Discover H5 files
        import glob
        import os
        
        if h5_files is None:
            h5_files = []
        
        # If h5_dir is provided, discover files from directory
        if h5_dir is not None:
            discovered_files = sorted(glob.glob(os.path.join(h5_dir, h5_pattern)))
            print(f"\n{'='*60}")
            print(f"Auto-discovered {len(discovered_files)} H5 files from: {h5_dir}")
            for f in discovered_files:
                print(f"  - {os.path.basename(f)}")
            print(f"{'='*60}\n")
            h5_files.extend(discovered_files)
        
        # If h5_clean is provided, prepend it to the list
        if h5_clean is not None:
            if not os.path.exists(h5_clean):
                raise FileNotFoundError(f"Clean H5 file not found: {h5_clean}")
            h5_files.insert(0, h5_clean)
            print(f"Added clean H5 file: {os.path.basename(h5_clean)}")
        
        if len(h5_files) == 0:
            raise ValueError("No H5 files found. Provide h5_files, h5_dir, or h5_clean+h5_dir")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_files = []
        for f in h5_files:
            if f not in seen:
                seen.add(f)
                unique_files.append(f)
        h5_files = unique_files
        
        self.h5_files = h5_files
        self.n_files = len(h5_files)
        self.agent_id = agent_id
        self.camera_name = camera_name
        
        # Auto-detect single-arm vs dual-arm format
        self._detect_data_format()
        self.pairs_per_index = pairs_per_index
        self.seed = seed
        self.val_ratio = val_ratio
        self.preload_to_memory = preload_to_memory
        
        # Random number generator for pair sampling
        self.rng = np.random.RandomState(seed)
        
        # Generate all possible pairs (including self-pairs)
        # combinations_with_replacement gives (0,0), (0,1), (0,2), ..., (1,1), (1,2), ...
        self.all_pairs = list(combinations_with_replacement(range(self.n_files), 2))
        self.n_pairs = len(self.all_pairs)  # C(n,2) + n = n*(n+1)/2
        print(f"Total unique pairs (including self): {self.n_pairs}")
        
        # Open H5 files and get structure info
        self._open_h5_files()
        self._build_index()
        
        # Preload data if requested
        if preload_to_memory:
            self._preload_data()
        
        # Split train/val
        self._create_train_val_split()
        
        print(f"\n{'='*60}")
        print(f"PairwiseH5Dataset initialized:")
        print(f"  H5 files: {self.n_files}")
        print(f"  Agent: {self.agent_name if not self.is_single_arm else 'single-arm'}")
        print(f"  Camera: {self.camera_name}")
        print(f"  Total trajectories (per file): {self.n_trajs}")
        print(f"  Total frames: {self.total_frames}")
        print(f"  Pairs per index: {self.pairs_per_index}")
        print(f"  Total samples: {len(self)}")
        print(f"{'='*60}\n")
    
    def _detect_data_format(self):
        """
        Auto-detect whether the H5 files are single-arm or dual-arm format.
        
        Single-arm format: traj_X/obs/agent/qpos
        Dual-arm format: traj_X/obs/agent/panda-0/qpos
        """
        # Temporarily open first file to detect format
        with h5py.File(self.h5_files[0], 'r') as h:
            traj_keys = [k for k in h.keys() if k.startswith('traj_')]
            first_traj = traj_keys[0]
            
            # Check if it's dual-arm format
            if f'{first_traj}/obs/agent/panda-0' in h:
                self.is_single_arm = False
                self.agent_name = f'panda-{self.agent_id}'
                print(f"Detected: Dual-arm format (using {self.agent_name})")
            else:
                self.is_single_arm = True
                self.agent_name = None  # Not used for single-arm
                print(f"Detected: Single-arm format")
    
    def _get_qpos_path(self, traj_key: str) -> str:
        """Get the path to qpos data based on data format."""
        if self.is_single_arm:
            return f'{traj_key}/obs/agent/qpos'
        else:
            return f'{traj_key}/obs/agent/{self.agent_name}/qpos'
    
    def _open_h5_files(self):
        """Open all H5 files and store handles."""
        self.h5_handles = []
        for path in self.h5_files:
            h = h5py.File(path, 'r')
            self.h5_handles.append(h)
    
    def _build_index(self):
        """Build index mapping from flat index to (traj_id, frame_id)."""
        # Get trajectory keys from first file (assume all files have same structure)
        h = self.h5_handles[0]
        self.traj_keys = sorted([k for k in h.keys() if k.startswith('traj_')],
                                 key=lambda x: int(x.split('_')[1]))
        self.n_trajs = len(self.traj_keys)
        
        # Build frame index: list of (traj_key, frame_idx, traj_length)
        self.frame_index = []
        self.traj_lengths = []
        
        for traj_key in self.traj_keys:
            # Get qpos shape to determine trajectory length
            qpos = h[self._get_qpos_path(traj_key)]
            traj_len = qpos.shape[0]  # T+1 frames (including initial state)
            self.traj_lengths.append(traj_len)
            
            # Add all frames except the last one (which has no corresponding action)
            for frame_idx in range(traj_len - 1):
                self.frame_index.append((traj_key, frame_idx))
        
        self.total_frames = len(self.frame_index)
    
    def _preload_data(self):
        """Preload all data into memory for faster access."""
        print("Preloading data into memory...")
        self.data_cache = []
        
        for file_idx, h in enumerate(self.h5_handles):
            file_data = {}
            for traj_key in self.traj_keys:
                traj_data = {
                    'qpos': h[self._get_qpos_path(traj_key)][:],
                    'image': h[f'{traj_key}/obs/sensor_data/{self.camera_name}/rgb'][:],
                }
                file_data[traj_key] = traj_data
            self.data_cache.append(file_data)
            print(f"  Loaded file {file_idx + 1}/{self.n_files}")
        
        print("Preloading complete.")
    
    def _create_train_val_split(self):
        """Create train/validation split based on trajectories."""
        n_val = int(self.n_trajs * self.val_ratio)
        n_train = self.n_trajs - n_val
        
        # Shuffle trajectory indices
        traj_indices = np.arange(self.n_trajs)
        self.rng.shuffle(traj_indices)
        
        self.train_traj_indices = set(traj_indices[:n_train])
        self.val_traj_indices = set(traj_indices[n_train:])
        
        # Build train/val frame indices
        self.train_frame_indices = []
        self.val_frame_indices = []
        
        for idx, (traj_key, frame_idx) in enumerate(self.frame_index):
            traj_id = int(traj_key.split('_')[1])
            if traj_id in self.train_traj_indices:
                self.train_frame_indices.append(idx)
            else:
                self.val_frame_indices.append(idx)
        
        # Default to training mode
        self.is_training = True
        self._active_indices = self.train_frame_indices
    
    def set_training(self, is_training: bool):
        """Switch between training and validation mode."""
        self.is_training = is_training
        if is_training:
            self._active_indices = self.train_frame_indices
        else:
            self._active_indices = self.val_frame_indices
    
    def get_validation_dataset(self) -> 'PairwiseH5Dataset':
        """Return a validation version of this dataset."""
        val_set = copy.copy(self)
        val_set.is_training = False
        val_set._active_indices = self.val_frame_indices
        # Use different random seed for validation
        val_set.rng = np.random.RandomState(self.seed + 1000)
        return val_set
    
    @staticmethod
    def qpos_to_action(qpos: np.ndarray) -> np.ndarray:
        """
        Convert 9-dim qpos to 8-dim action.
        
        qpos: [joint_0, ..., joint_6, gripper_left, gripper_right] (9 dim)
        action: [joint_0, ..., joint_6, gripper_normalized] (8 dim)
        
        Gripper mapping:
        - qpos gripper range: [0.032, 0.04] (physical width in meters)
        - action gripper range: [-1, 1] (normalized command)
        """
        joints = qpos[:7]
        
        # Average of two gripper fingers
        gripper_phys = (qpos[7] + qpos[8]) / 2
        
        # Map from [0.032, 0.04] to [-1, 1]
        gripper_min, gripper_max = 0.032, 0.04
        gripper_norm = 2 * (gripper_phys - gripper_min) / (gripper_max - gripper_min) - 1
        gripper_norm = np.clip(gripper_norm, -1, 1)
        
        return np.concatenate([joints, [gripper_norm]]).astype(np.float32)
    
    def _get_frame_data(self, file_idx: int, traj_key: str, frame_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get image and qpos for a specific frame.
        
        Returns:
            image: (H, W, 3) uint8
            qpos: (9,) float32
        """
        if self.preload_to_memory:
            data = self.data_cache[file_idx][traj_key]
            image = data['image'][frame_idx]
            qpos = data['qpos'][frame_idx]
        else:
            h = self.h5_handles[file_idx]
            image = h[f'{traj_key}/obs/sensor_data/{self.camera_name}/rgb'][frame_idx]
            qpos = h[self._get_qpos_path(traj_key)][frame_idx]
        
        return image, qpos.astype(np.float32)
    
    def _sample_pair_index(self, local_idx: int) -> int:
        """
        Sample a pair index for a given local index within pairs_per_index.
        
        Uses deterministic sampling based on local_idx to ensure reproducibility.
        """
        # Use the local index to deterministically select from all pairs
        # This ensures the same pair is selected for the same idx
        pair_idx = (local_idx * 7919) % self.n_pairs  # 7919 is a prime for better distribution
        return pair_idx
    
    def __len__(self) -> int:
        """Total number of samples = active frames × pairs per index."""
        return len(self._active_indices) * self.pairs_per_index
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """
        Get a pairwise sample, formatted for Diffusion Policy compatibility.
        
        Args:
            idx: Sample index (int or np.ndarray for batch indexing)
            
        Returns:
            Dictionary containing (with time dimension T=1 for DP compatibility):
            - obs:
                - agent_pos: (T, 9) float32 - qpos from file i (state1)
                - img1: (T, 3, H, W) float32 - image from file i
                - img2: (T, 3, H, W) float32 - image from file j
            - action: (T, 8) float32 - action converted from state2 (file j)
        """
        if isinstance(idx, np.ndarray):
            # Batch indexing
            return self._get_batch(idx)
        
        # Single sample indexing
        # Decompose idx into frame index and pair index
        frame_list_idx = idx // self.pairs_per_index
        local_pair_idx = idx % self.pairs_per_index
        
        # Get actual frame index from active indices
        frame_idx = self._active_indices[frame_list_idx]
        traj_key, frame_in_traj = self.frame_index[frame_idx]
        
        # Sample a pair of files
        pair_idx = self._sample_pair_index(local_pair_idx)
        file_i, file_j = self.all_pairs[pair_idx]
        
        # Get data from both files
        img1, state1 = self._get_frame_data(file_i, traj_key, frame_in_traj)
        img2, state2 = self._get_frame_data(file_j, traj_key, frame_in_traj)
        
        # Convert state2 to action format
        action2 = self.qpos_to_action(state2)
        
        # Normalize images: (H, W, 3) uint8 -> (3, H, W) float32 [0, 1]
        img1 = np.moveaxis(img1, -1, 0).astype(np.float32) / 255.0
        img2 = np.moveaxis(img2, -1, 0).astype(np.float32) / 255.0
        
        # Add time dimension for DP compatibility
        # obs has n_obs_steps=1, action has horizon steps (repeated)
        # Convert to tensors
        action_seq = np.tile(action2, (self.horizon, 1))  # Repeat action for horizon steps
        
        return {
            'obs': {
                'agent_pos': torch.from_numpy(state1).unsqueeze(0),   # (1, 9)
                'img1': torch.from_numpy(img1).unsqueeze(0),          # (1, 3, H, W)
                'img2': torch.from_numpy(img2).unsqueeze(0),          # (1, 3, H, W)
            },
            'action': torch.from_numpy(action_seq.astype(np.float32)),  # (horizon, 8)
        }
    
    def _get_batch(self, indices: np.ndarray) -> Dict[str, torch.Tensor]:
        """
        Get a batch of samples for efficient batch loading.
        
        Args:
            indices: Array of sample indices
            
        Returns:
            Batched dictionary with shape (B, T, ...)
        """
        batch_size = len(indices)
        
        # Pre-allocate arrays
        # obs has n_obs_steps=1, action has horizon steps
        img1_batch = np.zeros((batch_size, 1, 3, 224, 224), dtype=np.float32)
        img2_batch = np.zeros((batch_size, 1, 3, 224, 224), dtype=np.float32)
        state_batch = np.zeros((batch_size, 1, 9), dtype=np.float32)
        action_batch = np.zeros((batch_size, self.horizon, 8), dtype=np.float32)
        
        for i, idx in enumerate(indices):
            # Decompose idx
            frame_list_idx = idx // self.pairs_per_index
            local_pair_idx = idx % self.pairs_per_index
            
            frame_idx = self._active_indices[frame_list_idx]
            traj_key, frame_in_traj = self.frame_index[frame_idx]
            
            pair_idx = self._sample_pair_index(local_pair_idx)
            file_i, file_j = self.all_pairs[pair_idx]
            
            # Get data
            img1, state1 = self._get_frame_data(file_i, traj_key, frame_in_traj)
            img2, state2 = self._get_frame_data(file_j, traj_key, frame_in_traj)
            
            # Process
            img1_batch[i, 0] = np.moveaxis(img1, -1, 0).astype(np.float32) / 255.0
            img2_batch[i, 0] = np.moveaxis(img2, -1, 0).astype(np.float32) / 255.0
            state_batch[i, 0] = state1
            
            # Repeat action for horizon steps
            action = self.qpos_to_action(state2)
            action_batch[i] = np.tile(action, (self.horizon, 1))
        
        return {
            'obs': {
                'agent_pos': torch.from_numpy(state_batch),   # (B, 1, 9)
                'img1': torch.from_numpy(img1_batch),         # (B, 1, 3, H, W)
                'img2': torch.from_numpy(img2_batch),         # (B, 1, 3, H, W)
            },
            'action': torch.from_numpy(action_batch),         # (B, horizon, 8)
        }
    
    def get_normalizer(self, mode='limits', **kwargs):
        """
        Get data normalizer for Diffusion Policy.
        
        Args:
            mode: Normalization mode ('limits', 'gaussian', etc.)
            **kwargs: Additional arguments for LinearNormalizer
            
        Returns:
            LinearNormalizer instance
        """
        # Collect sample data for fitting normalizer
        # Sample from first H5 file
        h = self.h5_handles[0]
        
        all_actions = []
        all_states = []
        
        for traj_key in self.traj_keys[:min(10, len(self.traj_keys))]:  # Use first 10 trajs
            qpos = h[self._get_qpos_path(traj_key)][:]
            
            # Convert qpos to action format
            for t in range(len(qpos) - 1):
                action = self.qpos_to_action(qpos[t])
                all_actions.append(action)
                all_states.append(qpos[t])
        
        all_actions = np.array(all_actions)
        all_states = np.array(all_states)
        
        # Create normalizer
        data = {
            'action': all_actions,
            'agent_pos': all_states,
        }
        
        # Compute delta_action for relative action mode
        # NOTE: Only joints (dim 0-6) use relative position (delta), gripper (dim 7) uses absolute position
        state_as_action = np.array([self.qpos_to_action(s) for s in all_states])  # (N, 8)
        delta_joints = all_actions[..., :7] - state_as_action[..., :7]  # (N, 7) - relative position
        gripper_abs = all_actions[..., 7:8]  # (N, 1) - absolute position (not delta)
        delta_action = np.concatenate([delta_joints, gripper_abs], axis=-1)  # (N, 8)
        data['delta_action'] = delta_action
        
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        
        # Print delta_action statistics
        print(f"\n{'='*60}")
        print(f"Delta Action Statistics (for relative action mode):")
        print(f"  NOTE: Joints (dim 0-6) are delta, Gripper (dim 7) is absolute")
        print(f"  delta_action shape: {delta_action.shape}")
        print(f"  delta_action mean:  {np.mean(delta_action, axis=0)}")
        print(f"  delta_action std:   {np.std(delta_action, axis=0)}")
        print(f"  delta_action min:   {np.min(delta_action, axis=0)}")
        print(f"  delta_action max:   {np.max(delta_action, axis=0)}")
        print(f"{'='*60}\n")
        
        # Add image normalizers
        normalizer['img1'] = get_image_range_normalizer()
        normalizer['img2'] = get_image_range_normalizer()
        
        return normalizer
    
    def postprocess(self, samples: Dict, device: torch.device) -> Dict:
        """
        Postprocess batch samples for training.
        
        Moves data to device and formats for DP model.
        
        Args:
            samples: Batch dictionary from dataloader
            device: Target device
            
        Returns:
            Processed batch dictionary
        """
        agent_pos = samples['obs']['agent_pos'].to(device, non_blocking=True)
        img1 = samples['obs']['img1'].to(device, non_blocking=True)
        img2 = samples['obs']['img2'].to(device, non_blocking=True)
        action = samples['action'].to(device, non_blocking=True)
        
        return {
            'obs': {
                'agent_pos': agent_pos,    # (B, T, 9)
                'img1': img1,              # (B, T, 3, H, W)
                'img2': img2,              # (B, T, 3, H, W)
            },
            'action': action,              # (B, T, 8)
        }
    
    def close(self):
        """Close all H5 file handles."""
        for h in self.h5_handles:
            h.close()
        self.h5_handles = []
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.close()
        except:
            pass


def create_pairwise_dataset(
    clean_h5: str,
    perturbed_h5_dir: str,
    perturbed_pattern: str = 'perturbed_*.h5',
    **kwargs
) -> PairwiseH5Dataset:
    """
    Convenience function to create a PairwiseH5Dataset.
    
    Args:
        clean_h5: Path to the clean H5 file
        perturbed_h5_dir: Directory containing perturbed H5 files
        perturbed_pattern: Glob pattern for perturbed files
        **kwargs: Additional arguments for PairwiseH5Dataset
        
    Returns:
        PairwiseH5Dataset instance
    """
    import glob
    import os
    
    # Collect all H5 files
    h5_files = [clean_h5]
    perturbed_files = sorted(glob.glob(os.path.join(perturbed_h5_dir, perturbed_pattern)))
    h5_files.extend(perturbed_files)
    
    print(f"Found {len(h5_files)} H5 files:")
    for f in h5_files:
        print(f"  - {f}")
    
    return PairwiseH5Dataset(h5_files=h5_files, **kwargs)


if __name__ == '__main__':
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5_files', nargs='+', required=True, help='List of H5 files')
    parser.add_argument('--agent_id', type=int, default=0)
    parser.add_argument('--pairs_per_index', type=int, default=50)
    args = parser.parse_args()
    
    dataset = PairwiseH5Dataset(
        h5_files=args.h5_files,
        agent_id=args.agent_id,
        pairs_per_index=args.pairs_per_index,
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Test getting a sample
    sample = dataset[0]
    print(f"\nSample structure:")
    print(f"  state1: {sample['obs']['state1'].shape}")
    print(f"  img1: {sample['obs']['img1'].shape}")
    print(f"  img2: {sample['obs']['img2'].shape}")
    print(f"  action: {sample['action'].shape}")
    
    dataset.close()

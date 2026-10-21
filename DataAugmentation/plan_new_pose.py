"""
Motion planning validation tool for robot joint perturbations.
Reads qpos from H5 trajectory files and validates perturbed configurations.
"""

import sys
import os
from typing import List, Dict, Optional, Tuple
import numpy as np
import h5py

# Add RoboFactory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from RoboFactory.utils.mplib_utils import FlexiblePlanner

# Fixed paths for URDF and SRDF
URDF_PATH = os.path.join(os.path.dirname(__file__), 'panda', 'panda_v2.urdf')
SRDF_PATH = os.path.join(os.path.dirname(__file__), 'panda', 'panda_v2.srdf')

# Panda robot configuration
LINK_NAMES = [
    "panda_link0", "panda_link1", "panda_link2", "panda_link3",
    "panda_link4", "panda_link5", "panda_link6", "panda_link7",
    "panda_link8", "panda_hand", "panda_hand_tcp",
    "panda_leftfinger", "panda_rightfinger"
]

JOINT_NAMES = [
    "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
    "panda_joint5", "panda_joint6", "panda_joint7",
    "panda_finger_joint1", "panda_finger_joint2"
]

# Global planner instance (lazy initialization)
_planner: Optional[FlexiblePlanner] = None


def load_h5_data(h5_group) -> dict:
    """
    Recursively load data from h5 group into a dictionary.
    
    Args:
        h5_group: h5py Group or File object
    
    Returns:
        Dictionary containing the data
    """
    result = {}
    for key in h5_group.keys():
        item = h5_group[key]
        if isinstance(item, h5py.Group):
            result[key] = load_h5_data(item)
        elif isinstance(item, h5py.Dataset):
            result[key] = item[()]
    return result


def read_qpos_from_h5(h5_path: str, traj_idx: int, timestep: int, agent_idx: int) -> np.ndarray:
    """
    Read joint positions (qpos) from H5 trajectory file.
    
    Args:
        h5_path: Path to H5 file
        traj_idx: Trajectory index
        timestep: Timestep within trajectory
        agent_idx: Agent/robot index
    
    Returns:
        qpos: Joint positions as numpy array (9 DOF: 7 arm + 2 gripper)
    
    Raises:
        ValueError: If trajectory/timestep/agent not found
    """
    with h5py.File(h5_path, 'r') as f:
        traj_key = f"traj_{traj_idx}"
        if traj_key not in f:
            raise ValueError(f"Trajectory {traj_idx} not found in h5 file")
        
        trajectory = f[traj_key]
        trajectory_data = load_h5_data(trajectory)
        
        # Try to find qpos in the data structure
        agent_key = f"panda-{agent_idx}"
        qpos = None
        
        # Path 1: obs/agent/panda-X/qpos
        try:
            if "obs" in trajectory_data and "agent" in trajectory_data["obs"]:
                if agent_key in trajectory_data["obs"]["agent"]:
                    if "qpos" in trajectory_data["obs"]["agent"][agent_key]:
                        qpos_data = trajectory_data["obs"]["agent"][agent_key]["qpos"]
                        if len(qpos_data) > timestep:
                            qpos = qpos_data[timestep]
        except (KeyError, IndexError, TypeError):
            pass
        
        if qpos is None:
            raise ValueError(
                f"Could not find qpos for trajectory {traj_idx}, "
                f"timestep {timestep}, agent {agent_idx}"
            )
        
        # Ensure qpos is a numpy array and flatten if needed
        qpos = np.array(qpos).flatten()
        return qpos


def apply_perturbations(
    source_qpos: np.ndarray,
    joint_indices: List[int],
    perturbations: List[float]
) -> np.ndarray:
    """
    Apply perturbations to specified joints.
    
    Args:
        source_qpos: Original joint positions
        joint_indices: List of joint indices to perturb (e.g., [2, 4, 5])
        perturbations: List of perturbation values (e.g., [0.1, 0.2, 0.1])
    
    Returns:
        goal_qpos: Perturbed joint positions
    
    Raises:
        ValueError: If inputs are invalid
    """
    if len(joint_indices) != len(perturbations):
        raise ValueError(
            f"Length mismatch: joint_indices has {len(joint_indices)} elements, "
            f"perturbations has {len(perturbations)} elements"
        )
    
    # Validate joint indices
    max_joint_idx = len(source_qpos) - 1
    for idx in joint_indices:
        if idx < 0 or idx > max_joint_idx:
            raise ValueError(
                f"Invalid joint index {idx}. Must be in range [0, {max_joint_idx}]"
            )
    
    # Create goal_qpos by copying source and applying perturbations
    goal_qpos = source_qpos.copy()
    for joint_idx, delta in zip(joint_indices, perturbations):
        goal_qpos[joint_idx] += delta
    
    return goal_qpos


def get_planner() -> FlexiblePlanner:
    """
    Get or create the FlexiblePlanner instance (singleton pattern).
    
    Returns:
        FlexiblePlanner instance
    """
    global _planner
    
    if _planner is None:
        print("Initializing FlexiblePlanner...")
        
        if not os.path.exists(URDF_PATH):
            raise FileNotFoundError(f"URDF file not found: {URDF_PATH}")
        if not os.path.exists(SRDF_PATH):
            raise FileNotFoundError(f"SRDF file not found: {SRDF_PATH}")
        
        _planner = FlexiblePlanner(
            urdf=URDF_PATH,
            srdf=SRDF_PATH,
            user_link_names=LINK_NAMES,
            user_joint_names=JOINT_NAMES,
            move_group="panda_hand_tcp",
            joint_vel_limits=np.ones(7) * 1.0,
            joint_acc_limits=np.ones(7) * 1.0,
        )
        print("FlexiblePlanner initialized successfully!")
    
    return _planner


def validate_qpos_perturbation(
    h5_path: str,
    traj_idx: int,
    timestep: int,
    agent_idx: int,
    joint_indices: List[int],
    perturbations: List[float],
    planning_time: float = 5.0,
    verbose: bool = True
) -> Dict:
    """
    Validate if a perturbed joint configuration is reachable from the original.
    
    Args:
        h5_path: Path to H5 trajectory file
        traj_idx: Trajectory index
        timestep: Timestep within trajectory
        agent_idx: Agent/robot index
        joint_indices: List of joint indices to perturb
        perturbations: List of perturbation values
        planning_time: Maximum planning time in seconds
        verbose: Print detailed information
    
    Returns:
        Dictionary containing:
            - source_qpos: Original joint positions
            - goal_qpos: Perturbed joint positions
            - status: Planning status ('Success' or error message)
            - path: Planned path (if successful)
            - duration: Path duration (if successful)
    """
    result = {
        'source_qpos': None,
        'goal_qpos': None,
        'status': None,
        'path': None,
        'duration': None
    }
    
    # Step 1: Read source qpos from H5 file
    if verbose:
        print(f"\n{'='*60}")
        print(f"Reading qpos from H5 file...")
        print(f"  h5_path: {h5_path}")
        print(f"  traj_idx: {traj_idx}, timestep: {timestep}, agent_idx: {agent_idx}")
    
    try:
        source_qpos = read_qpos_from_h5(h5_path, traj_idx, timestep, agent_idx)
        result['source_qpos'] = source_qpos
        if verbose:
            print(f"  source_qpos: {source_qpos}")
    except Exception as e:
        result['status'] = f"Failed to read qpos: {e}"
        print(f"Error: {result['status']}")
        return result
    
    # Step 2: Apply perturbations
    if verbose:
        print(f"\nApplying perturbations...")
        print(f"  joint_indices: {joint_indices}")
        print(f"  perturbations: {perturbations}")
    
    try:
        goal_qpos = apply_perturbations(source_qpos, joint_indices, perturbations)
        result['goal_qpos'] = goal_qpos
        if verbose:
            print(f"  goal_qpos: {goal_qpos}")
            print(f"\n  Changes:")
            for idx, delta in zip(joint_indices, perturbations):
                print(f"    Joint {idx}: {source_qpos[idx]:.4f} -> {goal_qpos[idx]:.4f} (delta: {delta:+.4f})")
    except Exception as e:
        result['status'] = f"Failed to apply perturbations: {e}"
        print(f"Error: {result['status']}")
        return result
    
    # Step 3: Get planner
    try:
        planner = get_planner()
    except Exception as e:
        result['status'] = f"Failed to initialize planner: {e}"
        print(f"Error: {result['status']}")
        return result
    
    # Step 4: Plan from source to goal
    if verbose:
        print(f"\n{'='*60}")
        print(f"Planning path from source to goal...")
        print(f"  planning_time: {planning_time}s")
    
    try:
        plan_result = planner.plan_qpos_to_qpos(
            goal_qposes=[goal_qpos],
            current_qpos=source_qpos,
            time_step=0.1,
            rrt_range=0.1,
            planning_time=planning_time,
            verbose=verbose
        )
        
        result['status'] = plan_result['status']
        
        if plan_result['status'] == 'Success':
            result['path'] = plan_result.get('position')
            result['duration'] = plan_result.get('duration')
            if verbose:
                print(f"\n{'='*60}")
                print(f"Planning Result: SUCCESS")
                print(f"  Path shape: {result['path'].shape}")
                print(f"  Duration: {result['duration']}")
        else:
            if verbose:
                print(f"\n{'='*60}")
                print(f"Planning Result: FAILED")
                print(f"  Status: {result['status']}")
                
    except Exception as e:
        result['status'] = f"Planning error: {e}"
        if verbose:
            print(f"Error: {result['status']}")
            import traceback
            traceback.print_exc()
    
    return result


def main():
    """Example usage: read from H5 file, apply perturbations, and validate planning."""
    
    # ============ Configuration ============
    # H5 file path (modify this to your actual file)
    h5_path = "data/trajectory.h5"  # TODO: Replace with actual path
    
    # Trajectory parameters
    traj_idx = 1      # Trajectory index
    timestep = 10     # Timestep to read
    agent_idx = 0     # Agent/robot index
    
    # Perturbation parameters
    joint_indices = [0, 2, 4]           # Joints to perturb (0-indexed)
    perturbations = [0.1, -0.1, 0.05]   # Perturbation values (radians)
    
    # Planning parameters
    planning_time = 5.0  # Maximum planning time in seconds
    
    # ============ Run Validation ============
    print("=" * 60)
    print("Validate Qpos Perturbation")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  H5 file: {h5_path}")
    print(f"  Trajectory: {traj_idx}, Timestep: {timestep}, Agent: {agent_idx}")
    print(f"  Joint indices: {joint_indices}")
    print(f"  Perturbations: {perturbations}")
    print(f"  Planning time: {planning_time}s")
    
    # Call the main validation function
    result = validate_qpos_perturbation(
        h5_path=h5_path,
        traj_idx=traj_idx,
        timestep=timestep,
        agent_idx=agent_idx,
        joint_indices=joint_indices,
        perturbations=perturbations,
        planning_time=planning_time,
        verbose=True
    )
    
    # ============ Summary ============
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Source qpos: {result['source_qpos']}")
    print(f"Goal qpos:   {result['goal_qpos']}")
    print(f"Status:      {result['status']}")
    
    if result['status'] == 'Success':
        print(f"Path shape:  {result['path'].shape}")
        print(f"Duration:    {result['duration']}")
        print("\n✓ Perturbation is valid - path found!")
    else:
        print("\n✗ Perturbation is invalid - no path found!")


if __name__ == "__main__":
    main()

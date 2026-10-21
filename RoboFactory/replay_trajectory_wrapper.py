#!/usr/bin/env python
"""
Wrapper script for replaying ManiSkill trajectories using qpos instead of actions.
This script reads qpos from trajectory observations and converts them to actions
for replay, ensuring that tasks module is imported before use.
"""
import copy
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass
from typing import Annotated, Optional, Dict, List

import gymnasium as gym
import h5py
import numpy as np
import torch
import tyro
from tqdm import tqdm

# Add the RoboFactory directory to the path if needed
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import tasks module first to register all custom environments
try:
    import tasks
    print("✓ Successfully imported tasks module - custom environments registered")
except ImportError as e:
    print(f"Warning: Could not import tasks module: {e}")
    print("Custom environments may not be available")

import mani_skill.envs
# from mani_skill.envs.utils.system.backend import CPU_SIM_BACKENDS
from mani_skill.trajectory import utils as trajectory_utils
from mani_skill.trajectory.merge_trajectory import merge_trajectories
from mani_skill.utils import common, io_utils, wrappers
from mani_skill.utils.logging_utils import logger
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode


@dataclass
class Args:
    traj_path: str
    """Path to the trajectory .h5 file to replay"""
    sim_backend: Annotated[Optional[str], tyro.conf.arg(aliases=["-b"])] = None
    """Which simulation backend to use. Can be 'physx_cpu', 'physx_gpu'. If not specified the backend used is the same as the one used to collect the trajectory data."""
    obs_mode: Annotated[Optional[str], tyro.conf.arg(aliases=["-o"])] = None
    """Target observation mode to record in the trajectory."""
    target_control_mode: Annotated[Optional[str], tyro.conf.arg(aliases=["-c"])] = None
    """Target control mode to convert the demonstration actions to."""
    verbose: bool = False
    """Whether to print verbose information during trajectory replays"""
    save_traj: bool = False
    """Whether to save trajectories to disk. This will not override the original trajectory file."""
    save_video: bool = False
    """Whether to save videos"""
    max_retry: int = 0
    """Maximum number of times to try and replay a trajectory until the task reaches a success state at the end."""
    discard_timeout: bool = False
    """Whether to discard episodes that timeout and are truncated"""
    allow_failure: bool = False
    """Whether to include episodes that fail in saved videos and trajectory data"""
    vis: bool = False
    """Whether to visualize the trajectory replay via the GUI."""
    use_env_states: bool = False
    """Whether to replay by environment states instead of actions."""
    use_first_env_state: bool = False
    """Use the first env state in the trajectory to set initial state."""
    count: Optional[int] = None
    """Number of demonstrations to replay before exiting."""
    reward_mode: Optional[str] = None
    """Specifies the reward type that the env should use."""
    record_rewards: bool = False
    """Whether the replayed trajectory should include rewards"""
    shader: Optional[str] = None
    """Change shader used for rendering for all cameras."""
    video_fps: Optional[int] = None
    """The FPS of saved videos. Defaults to the control frequency"""
    render_mode: str = "rgb_array"
    """The render mode used for saving videos."""
    num_envs: Annotated[int, tyro.conf.arg(aliases=["-n"])] = 1
    """Number of environments to run to replay trajectories."""
    action_delay: float = 0.0
    """Delay time (in seconds) between each action execution. Set to a positive value to slow down the replay."""
    perturbed_traj_path: Annotated[Optional[str], tyro.conf.arg(aliases=["-p"])] = None
    """Path to the perturbed trajectory .h5 file. If provided, will replace qpos from this file into the normal trajectory."""
    perturbed_timestep: Optional[int] = None
    """Timestep in the perturbed trajectory to extract qpos from."""
    normal_timestep: Optional[int] = None
    """Timestep in the normal trajectory to replace qpos at."""
    episode_id: Optional[int] = None
    """Episode ID to process. If None, uses the first episode (episode_id=0)."""


@dataclass
class ReplayResult:
    num_replays: int
    successful_replays: int


def qpos_to_action(qpos: np.ndarray) -> np.ndarray:
    """
    Convert 9-dim qpos to 8-dim action.
    
    Args:
        qpos: [joint_0, ..., joint_6, gripper_left, gripper_right] (9 dim)
    
    Returns:
        action: [joint_0, ..., joint_6, gripper] (8 dim)
        - Joints are used directly
        - Gripper is averaged from left and right gripper positions
    """
    joints = qpos[:7]  # 7 joint positions
    gripper_phys = (qpos[7] + qpos[8]) / 2  # Average of two gripper positions
    return np.concatenate([joints, [gripper_phys]])


def get_qpos_from_trajectory(traj, agent_name: Optional[str] = None) -> np.ndarray:
    """
    Extract qpos from trajectory observations.
    
    Supports both single-arm format (obs/agent/qpos) and multi-arm format (obs/agent/panda-X/qpos).
    
    Args:
        traj: HDF5 trajectory group
        agent_name: Optional agent name for multi-arm (e.g., 'panda-0', 'panda-1')
    
    Returns:
        qpos array of shape (T+1, 9) where T is trajectory length
    """
    obs = traj["obs"]
    agent_obs = obs["agent"]
    
    # Try multi-arm format first if agent_name is provided
    if agent_name is not None and agent_name in agent_obs:
        if "qpos" in agent_obs[agent_name]:
            return np.array(agent_obs[agent_name]["qpos"][:])
    
    # Try single-arm format
    if "qpos" in agent_obs:
        return np.array(agent_obs["qpos"][:])
    
    # Try to detect multi-arm format automatically
    for key in agent_obs.keys():
        if isinstance(agent_obs[key], h5py.Group) and "qpos" in agent_obs[key]:
            return np.array(agent_obs[key]["qpos"][:])
    
    raise ValueError("Could not find qpos in trajectory. Available keys: " + str(list(agent_obs.keys())))


def get_qpos_at_timestep(traj, timestep: int, agent_name: Optional[str] = None) -> np.ndarray:
    """
    Extract qpos at a specific timestep from trajectory observations.
    
    Args:
        traj: HDF5 trajectory group
        timestep: Timestep index (0-indexed)
        agent_name: Optional agent name for multi-arm (e.g., 'panda-0', 'panda-1')
    
    Returns:
        qpos array of shape (9,)
    """
    obs = traj["obs"]
    agent_obs = obs["agent"]
    
    # Try multi-arm format first if agent_name is provided
    if agent_name is not None and agent_name in agent_obs:
        if "qpos" in agent_obs[agent_name]:
            qpos_data = agent_obs[agent_name]["qpos"]
            if timestep < len(qpos_data):
                return np.array(qpos_data[timestep])
            else:
                raise ValueError(f"Timestep {timestep} out of range (trajectory length: {len(qpos_data)})")
    
    # Try single-arm format
    if "qpos" in agent_obs:
        qpos_data = agent_obs["qpos"]
        if timestep < len(qpos_data):
            return np.array(qpos_data[timestep])
        else:
            raise ValueError(f"Timestep {timestep} out of range (trajectory length: {len(qpos_data)})")
    
    # Try to detect multi-arm format automatically
    for key in agent_obs.keys():
        if isinstance(agent_obs[key], h5py.Group) and "qpos" in agent_obs[key]:
            qpos_data = agent_obs[key]["qpos"]
            if timestep < len(qpos_data):
                return np.array(qpos_data[timestep])
            else:
                raise ValueError(f"Timestep {timestep} out of range (trajectory length: {len(qpos_data)})")
    
    raise ValueError("Could not find qpos in trajectory. Available keys: " + str(list(agent_obs.keys())))


def replace_qpos_in_trajectory(
    normal_traj_path: str,
    perturbed_traj_path: str,
    perturbed_timestep: int,
    normal_timestep: int,
    episode_id: int = 0,
    output_path: Optional[str] = None
) -> str:
    """
    Replace qpos at a specific timestep in normal trajectory with qpos from perturbed trajectory.
    
    Args:
        normal_traj_path: Path to normal trajectory .h5 file
        perturbed_traj_path: Path to perturbed trajectory .h5 file
        perturbed_timestep: Timestep in perturbed trajectory to extract qpos from
        normal_timestep: Timestep in normal trajectory to replace qpos at
        episode_id: Episode ID to process (default: 0)
        output_path: Optional output path. If None, creates a temporary file.
    
    Returns:
        Path to the modified trajectory file
    """
    import tempfile
    import shutil
    
    # Create output path if not provided
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(normal_traj_path))[0]
        output_dir = os.path.dirname(normal_traj_path)
        output_path = os.path.join(output_dir, f"{base_name}_replaced_t{normal_timestep}.h5")
    
    # Copy normal trajectory to output
    shutil.copy2(normal_traj_path, output_path)
    
    # Open both files
    with h5py.File(output_path, "r+") as normal_file, h5py.File(perturbed_traj_path, "r") as perturbed_file:
        traj_key = f"traj_{episode_id}"
        
        if traj_key not in normal_file:
            raise ValueError(f"Episode {episode_id} not found in normal trajectory file")
        if traj_key not in perturbed_file:
            raise ValueError(f"Episode {episode_id} not found in perturbed trajectory file")
        
        normal_traj = normal_file[traj_key]
        perturbed_traj = perturbed_file[traj_key]
        
        # Get qpos from perturbed trajectory
        perturbed_qpos = get_qpos_at_timestep(perturbed_traj, perturbed_timestep)
        print(f"\n{'='*60}")
        print(f"Extracting qpos from perturbed trajectory:")
        print(f"  Timestep: {perturbed_timestep}")
        print(f"  Qpos: {perturbed_qpos}")
        
        # Get qpos sequence from normal trajectory
        normal_qpos_sequence = get_qpos_from_trajectory(normal_traj)
        print(f"\nNormal trajectory qpos sequence length: {len(normal_qpos_sequence)}")
        if normal_timestep >= len(normal_qpos_sequence):
            raise ValueError(
                f"Normal timestep {normal_timestep} out of range "
                f"(trajectory length: {len(normal_qpos_sequence)})"
            )
        
        original_qpos = normal_qpos_sequence[normal_timestep].copy()
        print(f"\nReplacing qpos in normal trajectory:")
        print(f"  Timestep: {normal_timestep}")
        print(f"  Original qpos: {original_qpos}")
        print(f"  New qpos:      {perturbed_qpos}")
        print(f"  Note: This will affect action[{normal_timestep}] = qpos_to_action(qpos[{normal_timestep}])")
        print(f"{'='*60}\n")
        
        # Replace qpos in normal trajectory
        obs = normal_traj["obs"]
        agent_obs = obs["agent"]
        
        # Find and replace qpos
        replaced = False
        qpos_dataset = None
        replaced_key = None
        
        # Try multi-arm format
        for key in agent_obs.keys():
            if isinstance(agent_obs[key], h5py.Group) and "qpos" in agent_obs[key]:
                qpos_dataset = agent_obs[key]["qpos"]
                if normal_timestep < len(qpos_dataset):
                    qpos_dataset[normal_timestep] = perturbed_qpos
                    replaced = True
                    replaced_key = key
                    print(f"Replaced qpos in {key} at timestep {normal_timestep}")
                    break
        
        # Try single-arm format
        if not replaced and "qpos" in agent_obs:
            qpos_dataset = agent_obs["qpos"]
            if normal_timestep < len(qpos_dataset):
                qpos_dataset[normal_timestep] = perturbed_qpos
                replaced = True
                replaced_key = "agent"
                print(f"Replaced qpos at timestep {normal_timestep}")
        
        if not replaced:
            raise ValueError("Could not find qpos dataset to replace in normal trajectory")
        
        # Force flush to ensure changes are written to disk
        normal_file.flush()
        
        # Verify the replacement was successful by reading back
        if replaced_key and replaced_key != "agent":
            # Multi-arm format
            verify_qpos = np.array(qpos_dataset[normal_timestep])
        else:
            # Single-arm format
            verify_qpos = np.array(agent_obs["qpos"][normal_timestep])
        
        if not np.allclose(verify_qpos, perturbed_qpos, atol=1e-6):
            raise ValueError(
                f"Verification failed: Replaced qpos does not match! "
                f"Expected: {perturbed_qpos}, Got: {verify_qpos}"
            )
        
        print(f"✓ Successfully replaced qpos at timestep {normal_timestep}")
        print(f"  Original: {original_qpos}")
        print(f"  New:      {perturbed_qpos}")
        print(f"  Verified: {verify_qpos}")
        print(f"  Output saved to: {output_path}")
    
    # Verify again after file is closed by reopening
    with h5py.File(output_path, "r") as verify_file:
        verify_traj = verify_file[traj_key]
        verify_qpos_after_close = get_qpos_at_timestep(verify_traj, normal_timestep)
        if not np.allclose(verify_qpos_after_close, perturbed_qpos, atol=1e-6):
            raise ValueError(
                f"Verification after file close failed: Replaced qpos does not match! "
                f"Expected: {perturbed_qpos}, Got: {verify_qpos_after_close}"
            )
        print(f"✓ Verified qpos replacement persists after file close")
        print(f"  Final qpos at timestep {normal_timestep}: {verify_qpos_after_close}")
    
    return output_path


def sanity_check_and_format_seed(episode):
    """sanity checks the trajectory seed aligns with the episode seed. reformats the reset kwargs seed if missing or formatted wrong"""
    if "seed" in episode["reset_kwargs"]:
        if isinstance(episode["reset_kwargs"]["seed"], list):
            assert (
                len(episode["reset_kwargs"]["seed"]) == 1
            ), f"found multiple seeds for one trajectory (id={episode['episode_id']}) in the reset kwargs which means it is ambiguous which seed to use"
            episode["reset_kwargs"]["seed"] = episode["reset_kwargs"]["seed"][0]
        assert (
            episode["reset_kwargs"]["seed"] == episode["episode_seed"]
        ), f"found mismatch between trajectory seed and episode seed (id={episode['episode_id']})"
    else:
        episode["reset_kwargs"]["seed"] = episode["episode_seed"]


def replay_parallelized_sim(
    args: Args, env: RecordEpisode, pbar, episodes, trajectories
):
    """Replay trajectories using qpos in parallelized simulation (GPU)."""
    pbar.reset(total=len(episodes))
    warned_reset_kwargs_options = False
    
    # Split episodes into batches
    episode_pad = (args.num_envs - len(episodes) % args.num_envs) % args.num_envs
    batches = np.pad(
        np.array(episodes),
        (0, episode_pad),
        mode="constant",
        constant_values=episodes[-1],
    ).reshape(-1, args.num_envs)

    successful_replays = 0
    if pbar is not None:
        pbar.reset(total=len(episodes))
    
    for episode_batch_index, episode_batch in enumerate(batches):
        trajectory_ids = [episode["episode_id"] for episode in episode_batch]
        episode_lens = np.array([episode["elapsed_steps"] for episode in episode_batch])
        ori_control_mode = episode_batch[0]["control_mode"]
        assert all(
            [episode["control_mode"] == ori_control_mode for episode in episode_batch]
        ), "Replay trajectory with parallelized environments is only supported for trajectories with the same control mode"
        
        episode_batch_max_len = max(episode_lens)
        seeds = torch.tensor(
            [episode["episode_seed"] for episode in episode_batch],
            device=env.base_env.device,
        )
        env.reset(seed=seeds)

        # Generate batched env states and actions from qpos
        env_states_list = []
        actions_batch = []
        env_states_batch = []
        
        for i, trajectory_id in enumerate(trajectory_ids):
            traj = trajectories[f"traj_{trajectory_id}"]
            episode = episode_batch[i]
            sanity_check_and_format_seed(episode)
            if not warned_reset_kwargs_options and "options" in episode["reset_kwargs"]:
                logger.warning(
                    f"Reset kwargs includes options, which are not supported in GPU sim replay and will be ignored."
                )
                warned_reset_kwargs_options = True

            # Get qpos from trajectory
            qpos_sequence = get_qpos_from_trajectory(traj)
            
            # Verify qpos replacement if using perturbed trajectory
            if args.perturbed_traj_path is not None and i == 0 and episode_batch_index == 0:
                normal_timestep = args.normal_timestep
                if normal_timestep < len(qpos_sequence):
                    print(f"\n{'='*60}")
                    print(f"Verifying qpos replacement during replay:")
                    print(f"  Timestep: {normal_timestep}")
                    print(f"  Qpos at timestep {normal_timestep}: {qpos_sequence[normal_timestep]}")
                    print(f"  This will be used to generate action[{normal_timestep}]")
                    print(f"{'='*60}\n")
            
            # Convert qpos to actions (skip the last qpos as it's the final state)
            # qpos_sequence has shape (T+1, 9), we need T actions
            actions = []
            for t in range(len(qpos_sequence) - 1):
                action = qpos_to_action(qpos_sequence[t])
                actions.append(action)
            actions = np.array(actions)

            # Get env_states
            env_states = trajectory_utils.dict_to_list_of_dicts(traj["env_states"])

            # Padding
            for _ in range(episode_batch_max_len + 1 - len(env_states)):
                env_states.append(env_states[-1])
            if len(actions) < episode_batch_max_len:
                actions = np.concatenate(
                    [
                        actions,
                        np.zeros(
                            (episode_batch_max_len - len(actions), actions.shape[1])
                        ),
                    ],
                    axis=0,
                )
            env_states_list.append(env_states)
            actions_batch.append(actions)
        
        for t in range(episode_batch_max_len + 1):
            env_states_batch.append(
                trajectory_utils.list_of_dicts_to_dict(
                    [env_states_list[i][t] for i in range(len(env_states_list))]
                )
            )

        actions_batch = np.stack(actions_batch, axis=1)
        
        # When using qpos-generated actions, set initial state from qpos[0] if needed
        # Only use env_states for initial state if use_first_env_state is True
        # Don't use env_states during replay as they would override action effects
        if args.use_first_env_state:
            env.base_env.set_state_dict(env_states_batch[0])
            if args.save_traj:
                def recursive_replace(x, y):
                    if isinstance(x, np.ndarray):
                        x[-1, :] = y[-1, :]
                    else:
                        for k in x.keys():
                            recursive_replace(x[k], y[k])

                recursive_replace(
                    env._trajectory_buffer.state, common.batch(env_states_batch[0])
                )
                recursive_replace(
                    env._trajectory_buffer.observation,
                    common.to_numpy(common.batch(env.base_env.get_obs())),
                )

        # Replay with actions converted from qpos
        if (
            args.target_control_mode is None
            or ori_control_mode == args.target_control_mode
        ):
            flushed_trajectories = np.zeros(len(episode_batch), dtype=bool)
            if episode_batch_index == len(batches) - 1 and episode_pad > 0:
                flushed_trajectories[-episode_pad:] = True
            
            for t, a in enumerate(actions_batch):
                _, _, _, truncated, info = env.step(a)
                if args.action_delay > 0:
                    time.sleep(args.action_delay)
                # Don't use env_states when replaying with qpos-generated actions,
                # as it would override the action effects
                # This script always uses qpos to generate actions, so env_states should be disabled
                # if args.use_env_states:
                #     env.base_env.set_state_dict(env_states_batch[t])
                if args.vis:
                    env.base_env.render_human()

                if args.save_traj:
                    envs_to_flush = (t >= episode_lens - 1) & (~flushed_trajectories)
                    flushed_trajectories |= envs_to_flush
                    if envs_to_flush.sum() > 0:
                        pbar.update(n=envs_to_flush.sum())
                        if not args.allow_failure:
                            if "success" in info:
                                envs_to_flush &= (info["success"] == True).cpu().numpy()
                        if args.discard_timeout:
                            envs_to_flush &= (truncated == False).cpu().numpy()
                        successful_replays += envs_to_flush.sum()
                        env.flush_trajectory(
                            env_idxs_to_flush=np.where(envs_to_flush)[0]
                        )
        else:
            raise NotImplementedError(
                "Replay with different control modes are not supported when replaying on GPU parallelized environments"
            )
    
    return ReplayResult(
        num_replays=len(episodes), successful_replays=successful_replays
    )


def replay_cpu_sim(
    args: Args, env: RecordEpisode, ori_env, pbar, episodes, trajectories
):
    """Replay trajectories using qpos in CPU simulation."""
    successful_replays = 0
    
    for episode in episodes:
        sanity_check_and_format_seed(episode)
        episode_id = episode["episode_id"]
        traj_id = f"traj_{episode_id}"
        reset_kwargs = episode["reset_kwargs"]
        ori_control_mode = episode["control_mode"]
        
        if pbar is not None:
            pbar.set_description(f"Replaying {traj_id}")
        if traj_id not in trajectories:
            tqdm.write(f"{traj_id} does not exist in {args.traj_path}")
            continue

        for _ in range(args.max_retry + 1):
            env.reset(**reset_kwargs)
            if ori_env is not None:
                ori_env.reset(**reset_kwargs)

            # Get qpos from trajectory and convert to actions
            # When using qpos to generate actions, we should not use env_states
            # as they would override the action effects
            qpos_sequence = get_qpos_from_trajectory(trajectories[traj_id])
            
            # Verify qpos replacement if using perturbed trajectory
            if args.perturbed_traj_path is not None:
                normal_timestep = args.normal_timestep
                if normal_timestep < len(qpos_sequence):
                    print(f"\n{'='*60}")
                    print(f"Verifying qpos replacement during replay (CPU):")
                    print(f"  Trajectory ID: {traj_id}")
                    print(f"  Timestep: {normal_timestep}")
                    print(f"  Qpos at timestep {normal_timestep}: {qpos_sequence[normal_timestep]}")
                    print(f"  This will be used to generate action[{normal_timestep}]")
                    print(f"{'='*60}\n")
            
            # Convert qpos sequence to actions
            # qpos_sequence has shape (T+1, 9), we need T actions
            # action[t] is generated from qpos[t]
            actions = []
            for t in range(len(qpos_sequence) - 1):
                action = qpos_to_action(qpos_sequence[t])
                actions.append(action)
            actions = np.array(actions)
            
            info = {}

            # Replay with actions converted from qpos
            assert (
                args.target_control_mode is None
                or ori_control_mode == args.target_control_mode
                or not args.use_env_states
            ), "Cannot use env states when trying to convert from one control mode to another."
            
            if (
                args.target_control_mode is None
                or ori_control_mode == args.target_control_mode
            ):
                n = len(actions)
                if pbar is not None:
                    pbar.reset(total=n)
                
                # When using qpos-generated actions, don't use env_states as they override action effects
                # Only set initial state if use_first_env_state is True
                if args.use_first_env_state:
                    ori_env_states = trajectory_utils.dict_to_list_of_dicts(
                        trajectories[traj_id]["env_states"]
                    )
                    if ori_env is not None:
                        ori_env.set_state_dict(ori_env_states[0])
                    env.base_env.set_state_dict(ori_env_states[0])
                    if args.save_traj:
                        def recursive_replace(x, y):
                            if isinstance(x, np.ndarray):
                                x[-1, :] = y[-1, :]
                            else:
                                for k in x.keys():
                                    recursive_replace(x[k], y[k])

                        recursive_replace(
                            env._trajectory_buffer.state, common.batch(ori_env_states[0])
                        )
                        fixed_obs = env.base_env.get_obs()
                        recursive_replace(
                            env._trajectory_buffer.observation,
                            common.to_numpy(common.batch(fixed_obs)),
                        )
                
                # Don't use env_states during replay when using qpos-generated actions
                # as they would override the action effects
                # This script always uses qpos to generate actions, so env_states should be disabled
                ori_env_states = None
                
                for t, a in enumerate(actions):
                    if pbar is not None:
                        pbar.update()
                    _, _, _, truncated, info = env.step(a)
                    if args.action_delay > 0:
                        time.sleep(args.action_delay)
                    # Don't use env_states when replaying with qpos-generated actions
                    # as it would override the action effects
                    # This script always uses qpos to generate actions, so env_states should be disabled
                    # if args.use_env_states and ori_env_states is not None:
                    #     if t < len(ori_env_states):
                    #         env.base_env.set_state_dict(ori_env_states[t])
                    if args.vis:
                        env.base_env.render_human()
            else:
                raise NotImplementedError(
                    f"Script currently does not support converting {ori_control_mode} to {args.target_control_mode} when using qpos"
                )

            success = info.get("success", False)
            if args.discard_timeout:
                success = success and (not truncated)

            if success or args.allow_failure:
                successful_replays += 1
                if args.save_traj:
                    env.flush_trajectory()
                if args.save_video:
                    env.flush_video(ignore_empty_transition=False)
                break
            else:
                if args.verbose:
                    print("info", info)
        else:
            env.flush_video(save=False)
            tqdm.write(f"Episode {episode_id} is not replayed successfully. Skipping")

    return ReplayResult(
        num_replays=len(episodes), successful_replays=successful_replays
    )


def _main_helper(x):
    return _main(*x)


def _main(
    args: Args,
    use_cpu_backend,
    env_id,
    env_kwargs,
    ori_env_kwargs,
    record_episode_kwargs,
    proc_id: int = 0,
    num_procs=1,
):
    pbar = tqdm(position=proc_id, leave=None, unit="step", dynamic_ncols=True)

    # Load HDF5 containing trajectories
    traj_path = args.traj_path
    ori_h5_file = h5py.File(traj_path, "r")

    # Load associated json
    json_path = traj_path.replace(".h5", ".json")
    json_data = io_utils.load_json(json_path)
    env = gym.make(env_id, **env_kwargs)
    if isinstance(env.action_space, gym.spaces.Dict):
        logger.warning(
            "We currently do not track which wrappers are used when recording trajectories but majority of the time in multi-agent envs with dictionary action spaces the actions are stored as flat vectors. We will flatten the action space with the ManiSkill provided FlattenActionSpaceWrapper. If you do not want this behavior you can copy the replay trajectory code yourself and modify it as needed."
        )
        env = FlattenActionSpaceWrapper(env)

    # Prepare output directory and trajectory name
    output_dir = os.path.dirname(traj_path)
    ori_traj_name = os.path.splitext(os.path.basename(traj_path))[0]
    parts = ori_traj_name.split(".")
    if len(parts) > 1:
        ori_traj_name = parts[0]
    suffix = "{}.{}.{}".format(
        env.unwrapped.obs_mode,
        env.unwrapped.control_mode,
        # env.unwrapped.backend.sim_backend,
        'cpu'
    )
    new_traj_name = ori_traj_name + "." + suffix
    if use_cpu_backend:
        if num_procs > 1:
            new_traj_name = new_traj_name + "." + str(proc_id)
        if args.target_control_mode is not None:
            ori_env = gym.make(env_id, **ori_env_kwargs)
        else:
            ori_env = None
    else:
        pass

    env = wrappers.RecordEpisode(
        env,
        output_dir,
        trajectory_name=new_traj_name,
        video_fps=(
            args.video_fps if args.video_fps is not None else env.unwrapped.control_freq
        ),
        **record_episode_kwargs,
    )

    if env.save_trajectory:
        output_h5_path = env._h5_file.filename
        assert not os.path.samefile(output_h5_path, traj_path)
    else:
        output_h5_path = None

    episodes = json_data["episodes"][: args.count]
    if use_cpu_backend:
        inds = np.arange(len(episodes))
        inds = np.array_split(inds, num_procs)[proc_id]
        replay_result = replay_cpu_sim(
            args, env, ori_env, pbar, [episodes[index] for index in inds], ori_h5_file
        )
    else:
        replay_result = replay_parallelized_sim(args, env, pbar, episodes, ori_h5_file)

    env.close()
    ori_h5_file.close()
    return output_h5_path, replay_result


def parse_args(args=None):
    return tyro.cli(Args, args=args)


def main(args: Args):
    traj_path = args.traj_path
    temp_traj_path = None  # Track temporary trajectory file for cleanup
    
    # If perturbed trajectory is provided, replace qpos first
    if args.perturbed_traj_path is not None:
        if args.perturbed_timestep is None or args.normal_timestep is None:
            raise ValueError(
                "When --perturbed-traj-path is provided, both --perturbed-timestep "
                "and --normal-timestep must be specified"
            )
        
        episode_id = args.episode_id if args.episode_id is not None else 0
        print(f"\n{'='*60}")
        print(f"Replacing qpos from perturbed trajectory")
        print(f"  Perturbed file: {args.perturbed_traj_path}")
        print(f"  Normal file:    {traj_path}")
        print(f"  Episode ID:     {episode_id}")
        print(f"  Perturbed timestep: {args.perturbed_timestep}")
        print(f"  Normal timestep:    {args.normal_timestep}")
        print(f"{'='*60}\n")
        
        # Create modified trajectory file
        modified_traj_path = replace_qpos_in_trajectory(
            normal_traj_path=traj_path,
            perturbed_traj_path=args.perturbed_traj_path,
            perturbed_timestep=args.perturbed_timestep,
            normal_timestep=args.normal_timestep,
            episode_id=episode_id
        )
        
        # Track temporary file for cleanup
        temp_traj_path = modified_traj_path
        
        # Use modified trajectory for replay
        traj_path = modified_traj_path
        # Update args.traj_path so _main function uses the modified trajectory
        args.traj_path = modified_traj_path
        print(f"\nUsing modified trajectory: {traj_path}\n")
        print(f"Updated args.traj_path to: {args.traj_path}\n")
    
    # Load trajectory metadata json
    json_path = traj_path.replace(".h5", ".json")
    if not os.path.exists(json_path):
        # If using modified trajectory, try to use original json
        original_json_path = args.traj_path.replace(".h5", ".json")
        if os.path.exists(original_json_path):
            json_path = original_json_path
            print(f"Using original JSON file: {json_path}")
        else:
            raise FileNotFoundError(f"JSON file not found: {json_path}")
    json_data = io_utils.load_json(json_path)

    env_info = json_data["env_info"]
    env_id = env_info["env_id"]
    ori_env_kwargs = env_info["env_kwargs"]
    env_kwargs = ori_env_kwargs.copy()

    # Warn if use_env_states is enabled - this script always uses qpos to generate actions
    # so env_states would override the action effects and should be disabled
    if args.use_env_states:
        logger.warning(
            "Warning: --use-env-states is enabled, but this script always uses qpos to generate actions. "
            "env_states will be disabled during replay to avoid overriding action effects. "
            "If you need to use env_states, please use the original replay_trajectory script instead."
        )

    # Checks and setting up env kwargs
    # Note: use_env_states is disabled in this script as it always uses qpos to generate actions
    if (
        "sim_backend" in ori_env_kwargs
        and ori_env_kwargs["sim_backend"] != args.sim_backend
    ):
        logger.warning(
            f"Warning: Using different backend ({args.sim_backend}) than the original used to collect the trajectory data "
            f"({ori_env_kwargs['sim_backend']}). This may cause replay failures due to "
            f"differences in simulation/physics engine backend. Use the same backend by passing -b {ori_env_kwargs['sim_backend']}."
        )
    if args.sim_backend is None:
        if "sim_backend" not in ori_env_kwargs:
            args.sim_backend = "physx_cpu"
        else:
            args.sim_backend = ori_env_kwargs["sim_backend"]

    ori_env_kwargs["sim_backend"] = args.sim_backend
    env_kwargs["sim_backend"] = args.sim_backend

    # Modify the env kwargs according to the users inputs
    target_obs_mode = args.obs_mode
    target_control_mode = args.target_control_mode
    if target_obs_mode is not None:
        env_kwargs["obs_mode"] = target_obs_mode
    if target_control_mode is not None:
        env_kwargs["control_mode"] = target_control_mode
    if args.shader is not None:
        env_kwargs["shader_dir"] = args.shader
    env_kwargs["reward_mode"] = args.reward_mode
    env_kwargs["render_mode"] = args.render_mode

    record_episode_kwargs = dict(
        save_on_reset=False,
        save_trajectory=args.save_traj,
        save_video=args.save_video,
        record_reward=args.record_rewards,
    )

    if args.count is not None and args.count > len(json_data["episodes"]):
        logger.warning(
            f"Warning: Requested to replay {args.count} demos but there are only {len(json_data['episodes'])} demos collected, replaying all demos now"
        )
        args.count = len(json_data["episodes"])
    elif args.count is None:
        args.count = len(json_data["episodes"])

    pbar = tqdm(total=args.count, unit="step", dynamic_ncols=True)

    # Determine backend
    if "sim_backend" not in env_kwargs or (
        env_kwargs["sim_backend"] == "auto"
        and ("num_envs" not in env_kwargs or env_kwargs["num_envs"] == 1)
    ):
        env_kwargs["sim_backend"] = "physx_cpu"
    env_kwargs["num_envs"] = args.num_envs
    
    if env_kwargs["sim_backend"] not in ["cpu"]:
        record_episode_kwargs["max_steps_per_video"] = env_info["max_episode_steps"]
        _, replay_result = _main(
            args,
            use_cpu_backend=False,
            env_id=env_id,
            env_kwargs=env_kwargs,
            ori_env_kwargs=ori_env_kwargs,
            record_episode_kwargs=record_episode_kwargs,
            proc_id=0,
            num_procs=1,
        )
    else:
        env_kwargs["num_envs"] = 1
        ori_env_kwargs["num_envs"] = 1
        if args.num_envs > 1:
            pool = mp.Pool(args.num_envs)
            proc_args = [
                (
                    copy.deepcopy(args),
                    True,
                    env_id,
                    env_kwargs,
                    ori_env_kwargs,
                    record_episode_kwargs,
                    i,
                    args.num_envs,
                )
                for i in range(args.num_envs)
            ]
            res = list(tqdm(pool.imap(_main_helper, proc_args), total=args.count))
            replay_results_list = [x[1] for x in res]
            trajectory_paths = [x[0] for x in res]
            pool.close()
            if args.save_traj:
                output_path = trajectory_paths[0][: -len("0.h5")] + "h5"
                merge_trajectories(output_path, trajectory_paths)
                for h5_path in trajectory_paths:
                    tqdm.write(f"Remove {h5_path}")
                    os.remove(h5_path)
                    json_path = h5_path.replace(".h5", ".json")
                    tqdm.write(f"Remove {json_path}")
                    os.remove(json_path)
            replay_result = ReplayResult(
                num_replays=sum([x.num_replays for x in replay_results_list]),
                successful_replays=sum(
                    [x.successful_replays for x in replay_results_list]
                ),
            )
        else:
            _, replay_result = _main(
                args,
                use_cpu_backend=True,
                env_id=env_id,
                env_kwargs=env_kwargs,
                ori_env_kwargs=ori_env_kwargs,
                record_episode_kwargs=record_episode_kwargs,
                proc_id=0,
                num_procs=1,
            )

    pbar.close()
    print(
        f"Replayed {replay_result.num_replays} episodes, "
        f"{replay_result.successful_replays}/{replay_result.num_replays}={replay_result.successful_replays/replay_result.num_replays*100:.2f}% demos saved"
    )
    
    # Clean up temporary trajectory files if they were created
    if temp_traj_path is not None and os.path.exists(temp_traj_path):
        try:
            os.remove(temp_traj_path)
            print(f"✓ Removed temporary trajectory file: {temp_traj_path}")
        except Exception as e:
            print(f"Warning: Failed to remove temporary trajectory file {temp_traj_path}: {e}")
        
        # Also remove corresponding JSON file if it exists (only if it's not the original JSON)
        temp_json_path = temp_traj_path.replace(".h5", ".json")
        original_json_path = args.traj_path.replace(".h5", ".json")
        # Only delete if the temp JSON exists and is different from the original JSON
        if os.path.exists(temp_json_path):
            try:
                # Check if it's the same file as the original JSON
                if os.path.exists(original_json_path) and os.path.samefile(temp_json_path, original_json_path):
                    # Don't delete if it's the original JSON file
                    pass
                else:
                    # Delete the temporary JSON file
                    os.remove(temp_json_path)
                    print(f"✓ Removed temporary JSON file: {temp_json_path}")
            except (OSError, ValueError) as e:
                # OSError for samefile, ValueError for other issues
                # If samefile fails, we'll be conservative and not delete
                print(f"Warning: Could not verify if JSON is original file, skipping deletion: {e}")
            except Exception as e:
                print(f"Warning: Failed to remove temporary JSON file {temp_json_path}: {e}")


if __name__ == "__main__":
    # spawn is needed due to warp init issue
    mp.set_start_method("spawn")
    main(parse_args())

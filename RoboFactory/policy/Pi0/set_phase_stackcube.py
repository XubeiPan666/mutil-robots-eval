import h5py
import numpy as np
import matplotlib.pyplot as plt

def analyze_gripper_state_adaptive(qpos, close_ratio=0.3):
    """
    Adaptive gripper state analysis based on the actual data range.
    Uses relative thresholds based on min/max values in the trajectory.
    
    Args:
        qpos: Array of joint positions
        close_ratio: Ratio from min value to consider as closed (0-1)
    """
    # Typically gripper position is in the last column of qpos
    gripper_pos = qpos[:, -1]  # Last joint is usually gripper
    
    # Calculate adaptive thresholds based on actual data range
    min_pos = gripper_pos.min()
    max_pos = gripper_pos.max()
    pos_range = max_pos - min_pos
    
    # Close threshold: values closer to minimum are considered closed
    close_threshold = min_pos + pos_range * close_ratio
    
    # Open threshold: values closer to maximum are considered open (with hysteresis)
    open_threshold = min_pos + pos_range * (close_ratio + 0.2)
    
    print(f"    Gripper range: [{min_pos:.4f}, {max_pos:.4f}]")
    print(f"    Close threshold: {close_threshold:.4f}, Open threshold: {open_threshold:.4f}")
    
    # Initialize state array
    n_steps = len(gripper_pos)
    gripper_state = np.zeros(n_steps, dtype=int)
    
    # First state based on close threshold
    gripper_state[0] = 1 if gripper_pos[0] <= close_threshold else 0
    
    # Apply hysteresis for remaining states
    for i in range(1, n_steps):
        prev_state = gripper_state[i-1]
        curr_pos = gripper_pos[i]
        
        if prev_state == 0:  # Was open
            gripper_state[i] = 1 if curr_pos <= close_threshold else 0
        else:  # Was closed
            gripper_state[i] = 0 if curr_pos >= open_threshold else 1
            
    return gripper_state, close_threshold, open_threshold

def apply_unified_gripper_analysis(gripper_pos, close_threshold, open_threshold):
    """
    Apply unified gripper thresholds with hysteresis to determine gripper state.
    
    Args:
        gripper_pos: Array of gripper positions for one agent
        close_threshold: Unified close threshold
        open_threshold: Unified open threshold
    
    Returns:
        Binary array: 1 for closed, 0 for open
    """
    n_steps = len(gripper_pos)
    gripper_state = np.zeros(n_steps, dtype=int)
    
    # First state based on close threshold
    gripper_state[0] = 1 if gripper_pos[0] <= close_threshold else 0
    
    # Apply hysteresis for remaining states
    for i in range(1, n_steps):
        prev_state = gripper_state[i-1]
        curr_pos = gripper_pos[i]
        
        if prev_state == 0:  # Was open
            gripper_state[i] = 1 if curr_pos <= close_threshold else 0
        else:  # Was closed
            gripper_state[i] = 0 if curr_pos >= open_threshold else 1
            
    return gripper_state

def analyze_height_position(qpos, high_threshold_percentile=70):
    """
    Analyze if robot is at high position based on z-coordinate (assuming joint 2 controls height).
    Returns binary array: 1 for high position, 0 for low position
    
    Args:
        qpos: Array of joint positions
        high_threshold_percentile: Percentile to determine high position threshold
    """
    # Assuming joint 2 (index 2) controls the height/vertical movement
    height_pos = qpos[:, 2]
    
    # Calculate threshold based on percentile of height positions
    high_threshold = np.percentile(height_pos, high_threshold_percentile)
    
    print(f"    Height range: [{height_pos.min():.4f}, {height_pos.max():.4f}]")
    print(f"    High threshold ({high_threshold_percentile}%): {high_threshold:.4f}")
    
    # Binary array: 1 for high, 0 for low
    is_high = (height_pos >= high_threshold).astype(int)
    
    return is_high, high_threshold

def detect_grip_to_release_cycles(gripper_state, min_grip_duration=10):
    """
    Detect complete grip-to-release cycles (grip -> hold -> release).
    Returns list of (grip_start, release_time) tuples.
    """
    cycles = []
    grip_start = None
    
    for i in range(1, len(gripper_state)):
        if gripper_state[i-1] == 0 and gripper_state[i] == 1:
            # Start of grip
            grip_start = i
        elif gripper_state[i-1] == 1 and gripper_state[i] == 0 and grip_start is not None:
            # End of grip (release)
            grip_duration = i - grip_start
            if grip_duration >= min_grip_duration:  # Filter out short grips
                cycles.append((grip_start, i))
            grip_start = None
    
    # Handle case where gripper is still closed at the end
    if grip_start is not None and len(gripper_state) - grip_start >= min_grip_duration:
        cycles.append((grip_start, len(gripper_state)-1))
    
    return cycles

def detect_phases_simple(agent0_gripper, agent1_gripper, agent2_gripper):
    """
    基于gripper状态的三阶段检测：
    Phase 1: 所有agent下去抓cube，抬起来，直到agent0将cube放到target上（agent0从grip到release）
    Phase 2: agent0开启后到agent1开启（agent0 release到agent1 release）
    Phase 3: agent1开启后到agent2开启（agent1 release到agent2 release）
    """
    n_steps = len(agent0_gripper)
    phases = np.ones(n_steps, dtype=int)  # Initialize all steps as phase 1
    
    print(f"\n  Three-phase detection for {n_steps} steps...")
    
    # 找到每个agent的gripper状态变化点
    # 寻找从closed(1)到open(0)的转换点（release动作）
    agent0_releases = []
    agent1_releases = []
    agent2_releases = []
    
    # 检测agent0的release时间点
    for i in range(1, n_steps):
        if agent0_gripper[i-1] == 1 and agent0_gripper[i] == 0:
            agent0_releases.append(i)
    
    # 检测agent1的release时间点
    for i in range(1, n_steps):
        if agent1_gripper[i-1] == 1 and agent1_gripper[i] == 0:
            agent1_releases.append(i)
    
    # 检测agent2的release时间点  
    for i in range(1, n_steps):
        if agent2_gripper[i-1] == 1 and agent2_gripper[i] == 0:
            agent2_releases.append(i)
    
    print(f"  Agent0 releases at: {agent0_releases}")
    print(f"  Agent1 releases at: {agent1_releases}")
    print(f"  Agent2 releases at: {agent2_releases}")
    
    # Phase 1: 从开始到agent0第一次release（放下cube）
    if agent0_releases:
        phase1_end = agent0_releases[0]
        phases[:phase1_end] = 1
        print(f"  Phase 1: 0-{phase1_end} (所有agent抓取cube，agent0放置到target)")
        
        # Phase 2: 从agent0 release到agent1 release
        valid_agent1_releases = [t for t in agent1_releases if t >= phase1_end]
        if valid_agent1_releases:
            phase2_end = valid_agent1_releases[0]
            phases[phase1_end:phase2_end] = 2
            print(f"  Phase 2: {phase1_end}-{phase2_end} (agent0开启后到agent1开启)")
            
            # Phase 3: 从agent1 release到agent2 release
            valid_agent2_releases = [t for t in agent2_releases if t >= phase2_end]
            if valid_agent2_releases:
                phase3_end = valid_agent2_releases[0]
                phases[phase2_end:phase3_end] = 3
                print(f"  Phase 3: {phase2_end}-{phase3_end} (agent1开启后到agent2开启)")
                
                # agent2 release之后的步骤仍保持在phase 3
                if phase3_end < n_steps:
                    phases[phase3_end:] = 3
            else:
                # 如果agent2没有release，剩余步骤保持在phase 2
                phases[phase2_end:] = 2
                print(f"  Phase 2 continues: {phase2_end}-{n_steps} (agent2未开启)")
        else:
            # 如果agent1没有release，剩余步骤保持在phase 1
            phases[phase1_end:] = 1
            print(f"  Phase 1 continues: {phase1_end}-{n_steps} (agent1未开启)")
    else:
        # 如果agent0没有release，所有步骤都是phase 1
        print(f"  Warning: Agent0没有release动作，所有步骤为Phase 1")
    
    # 打印phase统计信息
    phase_counts = np.bincount(phases)
    print(f"  Phase distribution: {dict(enumerate(phase_counts))}")
    
    return phases

def create_detailed_visualization(file_path, traj_key, close_ratio=0.3):
    """
    Create detailed visualization for a single trajectory focusing on gripper states
    """
    with h5py.File(file_path, 'r') as f:
        # Load data
        agent0_qpos = f[f"{traj_key}/obs/agent/panda-0/qpos"][:]
        agent1_qpos = f[f"{traj_key}/obs/agent/panda-1/qpos"][:]
        agent2_qpos = f[f"{traj_key}/obs/agent/panda-2/qpos"][:]
        
        print(f"\n=== Detailed Analysis of {traj_key} ===")
        
        # Use unified thresholds for all agents
        all_gripper_pos = np.concatenate([
            agent0_qpos[:, -1], agent1_qpos[:, -1], agent2_qpos[:, -1]
        ])
        gripper_min = all_gripper_pos.min()
        gripper_max = all_gripper_pos.max()
        gripper_range = gripper_max - gripper_min
        unified_close_threshold = gripper_min + gripper_range * close_ratio
        unified_open_threshold = gripper_min + gripper_range * (close_ratio + 0.2)
        
        # Apply unified thresholds to all agents
        agent0_gripper = apply_unified_gripper_analysis(agent0_qpos[:, -1], unified_close_threshold, unified_open_threshold)
        agent1_gripper = apply_unified_gripper_analysis(agent1_qpos[:, -1], unified_close_threshold, unified_open_threshold)
        agent2_gripper = apply_unified_gripper_analysis(agent2_qpos[:, -1], unified_close_threshold, unified_open_threshold)
        
        # Detect phases
        phases = detect_phases_simple(
            agent0_gripper, agent1_gripper, agent2_gripper
        )
        
        # Create visualization
        fig, axes = plt.subplots(3, 1, figsize=(16, 12))
        time_steps = np.arange(len(phases))
        
        # 1. Raw gripper positions
        ax = axes[0]
        ax.plot(time_steps, agent0_qpos[:, -1], 'b-', linewidth=1, label='Agent0', alpha=0.7)
        ax.plot(time_steps, agent1_qpos[:, -1], 'r-', linewidth=1, label='Agent1', alpha=0.7)
        ax.plot(time_steps, agent2_qpos[:, -1], 'g-', linewidth=1, label='Agent2', alpha=0.7)
        ax.axhline(y=unified_close_threshold, color='k', linestyle='--', alpha=0.5, label=f'Close Th: {unified_close_threshold:.3f}')
        ax.axhline(y=unified_open_threshold, color='k', linestyle=':', alpha=0.5, label=f'Open Th: {unified_open_threshold:.3f}')
        ax.set_ylabel('Raw Gripper Position')
        ax.set_title(f'{traj_key} - Detailed Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Binary gripper states
        ax = axes[1]
        ax.plot(time_steps, agent0_gripper, 'b-', linewidth=2, label='Agent0 Gripper')
        ax.plot(time_steps, agent1_gripper + 0.1, 'r-', linewidth=2, label='Agent1 Gripper')
        ax.plot(time_steps, agent2_gripper + 0.2, 'g-', linewidth=2, label='Agent2 Gripper')
        ax.set_ylabel('Gripper State (Binary)')
        ax.set_ylim(-0.1, 1.4)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Final phases
        ax = axes[2]
        phase_colors = ['lightblue', 'lightgreen', 'lightcoral']
        phase_labels = ['Phase 1 (抓取+Agent0放置)', 'Phase 2 (Agent0→Agent1)', 'Phase 3 (Agent1→Agent2)']
        for phase in range(1, 4):
            phase_mask = phases == phase
            if np.any(phase_mask):
                ax.fill_between(time_steps, 0, phase_mask * phase, alpha=0.6, 
                             color=phase_colors[phase-1], label=phase_labels[phase-1])
        
        ax.plot(time_steps, phases, 'k-', linewidth=2, marker='o', markersize=2)
        ax.set_ylabel('Phase')
        ax.set_xlabel('Time Step')
        ax.set_ylim(0.5, 3.5)
        ax.set_yticks([1, 2, 3])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'detailed_analysis_{traj_key}.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Print summary statistics
        phase_counts = np.bincount(phases)
        print(f"\nPhase distribution:")
        for i, count in enumerate(phase_counts):
            if count > 0:
                print(f"Phase {i}: {count} steps ({count/len(phases)*100:.1f}%)")
        
        print(f"\nGripper state statistics:")
        print(f"Agent0 closed ratio: {agent0_gripper.mean():.3f}")
        print(f"Agent1 closed ratio: {agent1_gripper.mean():.3f}")
        print(f"Agent2 closed ratio: {agent2_gripper.mean():.3f}")

def add_phases_to_h5_v2(file_path, output_path=None, close_ratio=0.3):
    """
    基于gripper开关状态的三阶段检测：
    Phase 1: 所有agent抓取cube并抬起，直到agent0将cube放到target上
    Phase 2: agent0开启后到agent1开启  
    Phase 3: agent1开启后到agent2开启
    """
    if output_path is None:
        output_path = file_path.replace('.h5', '_with_phases_v2.h5')
    
    # Read the original file
    with h5py.File(file_path, 'r') as src_file:
        # Create new file
        with h5py.File(output_path, 'w') as dst_file:
            print(f"Processing {file_path} -> {output_path}")
            
            # Copy all data and add phases
            traj_keys = [key for key in src_file.keys() if key.startswith('traj_')]
            print(f"Found {len(traj_keys)} trajectories")
            
            for traj_key in traj_keys:
                print(f"Processing {traj_key}...")
                
                # Copy the entire trajectory group
                src_file.copy(traj_key, dst_file)
                
                # Get gripper data for all three agents
                try:
                    agent0_qpos = src_file[f"{traj_key}/obs/agent/panda-0/qpos"][:]
                    agent1_qpos = src_file[f"{traj_key}/obs/agent/panda-1/qpos"][:]
                    agent2_qpos = src_file[f"{traj_key}/obs/agent/panda-2/qpos"][:]
                    
                    # Use unified thresholds for all agents
                    # Calculate unified gripper threshold from all agents
                    all_gripper_pos = np.concatenate([
                        agent0_qpos[:, -1], agent1_qpos[:, -1], agent2_qpos[:, -1]
                    ])
                    gripper_min = all_gripper_pos.min()
                    gripper_max = all_gripper_pos.max()
                    gripper_range = gripper_max - gripper_min
                    unified_close_threshold = gripper_min + gripper_range * close_ratio
                    unified_open_threshold = gripper_min + gripper_range * (close_ratio + 0.2)
                    
                    print(f"  Unified gripper thresholds: close={unified_close_threshold:.4f}, open={unified_open_threshold:.4f}")
                    
                    # Apply unified thresholds to all agents
                    agent0_gripper = apply_unified_gripper_analysis(agent0_qpos[:, -1], unified_close_threshold, unified_open_threshold)
                    agent1_gripper = apply_unified_gripper_analysis(agent1_qpos[:, -1], unified_close_threshold, unified_open_threshold)
                    agent2_gripper = apply_unified_gripper_analysis(agent2_qpos[:, -1], unified_close_threshold, unified_open_threshold)
                    
                    print(f"  Gripper state ratios: A0={agent0_gripper.mean():.3f}, A1={agent1_gripper.mean():.3f}, A2={agent2_gripper.mean():.3f}")
                    
                    # Detect phases using only gripper states
                    phases = detect_phases_simple(agent0_gripper, agent1_gripper, agent2_gripper)
                    
                    # Add phases to the destination file
                    if f"{traj_key}/phase" in dst_file:
                        del dst_file[f"{traj_key}/phase"]  # Delete existing dataset if it exists
                    dst_file.create_dataset(f"{traj_key}/phase", data=phases)
                    
                    phase_counts = np.bincount(phases)
                    print(f"  Added phases: {phase_counts}")
                    
                except Exception as e:
                    print(f"  Error processing {traj_key}: {e}")
    
    print(f"Completed! Output saved to: {output_path}")
    return output_path

if __name__ == "__main__":
    # File path
    file_path = "/18141169908/zhangzaibin/dataset/original/three_robots_stack_cube--only-count-success.h5"
    
    # Create detailed visualization for sample trajectories
    print("=== Creating detailed visualizations ===")
    for traj_id in ["traj_0", "traj_1", "traj_5"]:
        try:
            create_detailed_visualization(
                file_path, 
                traj_id, 
                close_ratio=0.3      # 30% from min value is considered closed
            )
        except Exception as e:
            print(f"Error visualizing {traj_id}: {e}")
    
    # Process all trajectories with improved detection
    print("\n=== 基于gripper开关状态的三阶段检测 ===")
    try:
        output_path = add_phases_to_h5_v2(
            file_path, 
            close_ratio=0.3          # Adaptive gripper threshold
        )
        
        print(f"\nDone! Check the output file: {output_path}")
    except Exception as e:
        print(f"Error processing file: {e}")

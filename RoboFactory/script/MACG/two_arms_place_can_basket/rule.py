import h5py
import matplotlib.pyplot as plt
import numpy as np


def analyze_gripper_state(qpos, gripper_threshold=0.03):
    """
    Returns binary array: 1 for closed/gripping, 0 for open.
    Panda gripper: 0.04 = fully open, ~0.02/0.007 = closed on object.
    """
    gripper_pos = qpos[:, -1]
    gripper_closed = gripper_pos < gripper_threshold
    return gripper_closed.astype(int)


def detect_phases(agent0_gripper, agent1_gripper, agent0_joint_state, agent1_joint_state):
    """
    Detect phases for two_arms_place_can_basket task.

    Task flow: Agent1 grabs basket first, then Agent0 grabs can and places it.

    Phase 1: Agent1 approaches, grasps the basket and lifts → first stop
    Phase 2: Agent0 approaches and grasps the can → first stop (Agent1 holds)
    Phase 3: Agent0 carries and places the can into the basket → second stop (Agent1 holds)
    Phase 4: Agent0 releases and returns to rest (Agent1 holds)
    """
    n_steps = len(agent0_gripper)
    phases = np.zeros(n_steps, dtype=int)

    agent1_grip_start = None
    agent1_first_stop = None
    agent0_grip_start = None
    agent0_first_stop = None
    agent0_grip_end = None
    agent0_second_stop = None

    window_size = 3
    threshold = 0.0005

    # --- Agent1 events (acts first) ---

    # Detect agent1 grip start (transition from 0 to 1)
    for i in range(1, n_steps):
        if agent1_gripper[i - 1] == 0 and agent1_gripper[i] == 1:
            agent1_grip_start = i
            break

    # Detect agent1 first stop (stable state 20+ frames after gripping)
    if agent1_grip_start is not None:
        search_start = min(agent1_grip_start + 20, len(agent1_joint_state) - window_size)
        for i in range(search_start, len(agent1_joint_state) - window_size + 1):
            is_stable = True
            for j in range(i, i + window_size - 1):
                if abs(agent1_joint_state[j + 1] - agent1_joint_state[j]) > threshold:
                    is_stable = False
                    break
            if is_stable:
                agent1_first_stop = i
                break

    # --- Agent0 events (acts second) ---

    # Detect agent0 grip start (transition from 0 to 1, after agent1 first stop)
    search_start = agent1_first_stop if agent1_first_stop is not None else 0
    for i in range(search_start + 1, n_steps):
        if agent0_gripper[i - 1] == 0 and agent0_gripper[i] == 1:
            agent0_grip_start = i
            break

    # Detect agent0 first stop (stable state 30+ frames after gripping)
    if agent0_grip_start is not None:
        search_start = min(agent0_grip_start + 30, len(agent0_joint_state) - window_size)
        for i in range(search_start, len(agent0_joint_state) - window_size + 1):
            is_stable = True
            for j in range(i, i + window_size - 1):
                if abs(agent0_joint_state[j + 1] - agent0_joint_state[j]) > threshold:
                    is_stable = False
                    break
            if is_stable:
                agent0_first_stop = i
                break

    # Detect agent0 grip end (transition from 1 to 0)
    if agent0_first_stop is not None:
        for i in range(agent0_first_stop + 1, n_steps):
            if agent0_gripper[i - 1] == 1 and agent0_gripper[i] == 0:
                agent0_grip_end = i
                break

    # Detect agent0 second stop (stable state 30+ frames after releasing)
    if agent0_grip_end is not None:
        search_start = min(agent0_grip_end + 30, len(agent0_joint_state) - window_size)
        for i in range(search_start, len(agent0_joint_state) - window_size + 1):
            is_stable = True
            for j in range(i, i + window_size - 1):
                if abs(agent0_joint_state[j + 1] - agent0_joint_state[j]) > threshold:
                    is_stable = False
                    break
            if is_stable:
                agent0_second_stop = i
                break

    # --- Assign phases ---
    if agent1_first_stop is not None:
        phases[:agent1_first_stop] = 1

        if agent0_first_stop is not None:
            phases[agent1_first_stop:agent0_first_stop] = 2

            if agent0_second_stop is not None:
                phases[agent0_first_stop:agent0_second_stop] = 3
                phases[agent0_second_stop:] = 4
            else:
                phases[agent0_first_stop:] = 3
                print("Warning: agent0_second_stop not detected")
        else:
            phases[agent1_first_stop:] = 2
            print("Warning: agent0_first_stop not detected")
    else:
        phases[:] = 1
        print("Warning: agent1_first_stop not detected")

    return phases


def add_phases_to_h5(file_path, output_path=None, gripper_threshold=0.03, visualize=False):
    if output_path is None:
        output_path = file_path.replace(".h5", "_with_phases.h5")

    with h5py.File(file_path, "r") as src_file:
        with h5py.File(output_path, "w") as dst_file:
            print(f"Processing {file_path} -> {output_path}")

            traj_keys = [key for key in src_file.keys() if key.startswith("traj_")]
            print(f"Found {len(traj_keys)} trajectories")

            for traj_key in traj_keys:
                print(f"Processing {traj_key}...")

                src_file.copy(traj_key, dst_file)

                try:
                    agent0_qpos = src_file[f"{traj_key}/obs/agent/panda-0/qpos"][:]
                    agent1_qpos = src_file[f"{traj_key}/obs/agent/panda-1/qpos"][:]

                    agent0_gripper = analyze_gripper_state(agent0_qpos, gripper_threshold)
                    agent1_gripper = analyze_gripper_state(agent1_qpos, gripper_threshold)

                    agent0_joint_state = agent0_qpos[:, 1]
                    agent1_joint_state = agent1_qpos[:, 1]

                    phases = detect_phases(agent0_gripper, agent1_gripper, agent0_joint_state, agent1_joint_state)

                    dst_file.create_dataset(f"{traj_key}/phase", data=phases)

                    print(f"  Added phases: {np.bincount(phases)}")

                    if visualize and int(traj_key.split("_")[1]) < 5:
                        visualize_phases(agent0_gripper, agent1_gripper, phases, traj_key)

                except Exception as e:
                    print(f"  Error processing {traj_key}: {e}")

    print(f"Completed! Output saved to: {output_path}")
    return output_path


def visualize_phases(agent0_gripper, agent1_gripper, phases, traj_key):
    plt.figure(figsize=(12, 6))

    time_steps = np.arange(len(phases))

    plt.subplot(3, 1, 1)
    plt.plot(time_steps, agent0_gripper, "b-", linewidth=2, label="Agent0 Gripper")
    plt.ylabel("Agent0 Gripper")
    plt.title(f"{traj_key} - Gripper States and Phases")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(time_steps, agent1_gripper, "r-", linewidth=2, label="Agent1 Gripper")
    plt.ylabel("Agent1 Gripper")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(time_steps, phases, "g-", linewidth=2, marker="o", markersize=3)
    plt.ylabel("Phase")
    plt.xlabel("Time Step")
    plt.ylim(0.5, 4.5)
    plt.yticks([1, 2, 3, 4])
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"phase_analysis_{traj_key}.png", dpi=150, bbox_inches="tight")
    plt.close()


def analyze_single_trajectory(file_path, traj_key, gripper_threshold=0.03):
    with h5py.File(file_path, "r") as f:
        agent0_qpos = f[f"{traj_key}/obs/agent/panda-0/qpos"][:]
        agent1_qpos = f[f"{traj_key}/obs/agent/panda-1/qpos"][:]

        print(f"\n=== Analysis of {traj_key} ===")
        print(f"Trajectory length: {len(agent0_qpos)} steps")
        print(f"Agent0 qpos shape: {agent0_qpos.shape}")
        print(f"Agent1 qpos shape: {agent1_qpos.shape}")

        agent0_gripper_pos = agent0_qpos[:, -1]
        agent1_gripper_pos = agent1_qpos[:, -1]

        print("\nGripper position ranges:")
        print(f"Agent0 gripper: [{agent0_gripper_pos.min():.4f}, {agent0_gripper_pos.max():.4f}]")
        print(f"Agent1 gripper: [{agent1_gripper_pos.min():.4f}, {agent1_gripper_pos.max():.4f}]")

        agent0_gripper = analyze_gripper_state(agent0_qpos, gripper_threshold)
        agent1_gripper = analyze_gripper_state(agent1_qpos, gripper_threshold)

        print("\nGripper states (closed ratio):")
        print(f"Agent0: {agent0_gripper.mean():.3f}")
        print(f"Agent1: {agent1_gripper.mean():.3f}")

        agent0_joint_state = agent0_qpos[:, 1]
        agent1_joint_state = agent1_qpos[:, 1]

        phases = detect_phases(agent0_gripper, agent1_gripper, agent0_joint_state, agent1_joint_state)
        phase_counts = np.bincount(phases)

        print("\nPhase distribution:")
        for i, count in enumerate(phase_counts):
            if count > 0:
                print(f"Phase {i}: {count} steps ({count / len(phases) * 100:.1f}%)")

        return agent0_gripper, agent1_gripper, phases


if __name__ == "__main__":
    file_path = "/18141169908/zhangzaibin/multivla-pi0/dataset/macg_twoarms/two_arms_place_can_basket/two_arms_place_can_basket.h5"

    print("=== Analyzing sample trajectories ===")
    analyze_single_trajectory(file_path, "traj_0")
    analyze_single_trajectory(file_path, "traj_1")

    print("\n=== Adding phases to all trajectories ===")
    output_path = add_phases_to_h5(
        file_path,
        gripper_threshold=0.03,
        visualize=True,
    )

    print(f"\nDone! Check the output file: {output_path}")

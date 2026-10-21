import numpy as np
import sapien

from tasks import ThreeArmsPlaceObjectsEnv
from planner.motionplanner import PandaArmMotionPlanningSolver

def solve(env: ThreeArmsPlaceObjectsEnv, seed=None, debug=False, vis=False, status=None):
    env.reset(seed=seed)
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=False,
        vis=vis,
        base_pose=[agent.robot.pose for agent in env.agent.agents],
        visualize_target_grasp_pose=vis,
        print_env_info=False,
        is_multi_agent=True
    )
    env=env.unwrapped
    

    # Sequential pick-and-place: each agent picks its object and places it into the tray
    # Object mapping: agent0 -> can1, agent1 -> teabox2, agent2 -> oilbottle3
    init_pose0 = env.agent.agents[0].tcp.pose
    init_pose1 = env.agent.agents[1].tcp.pose
    init_pose2 = env.agent.agents[2].tcp.pose

    # Define the order based on status
    # Digits encode which object each agent picks: "132" = agent0→obj1, agent1→obj3, agent2→obj2
    # Execution order is always agent0 → agent1 → agent2
    # Tray placement positions are FIXED per object (same across all statuses)
    obj_tray_pos = {
        'can1':     (0.025, -0.085),   # can1 always goes to left edge
        'teabox2':  (-0.025, 0.0),     # teabox2 always goes to center
        'hamburg3': (0.025, 0.085),    # hamburg3 always goes to right edge
    }
    if status == "three_arms_place_objects_123_123":
        # agent0→can1(obj1), agent1→teabox2(obj2), agent2→hamburg3(obj3)
        order = [
            (0, env.can1, init_pose0, 'can1'),
            (1, env.teabox2, init_pose1, 'teabox2'),
            (2, env.hamburg3, init_pose2, 'hamburg3'),
        ]
    elif status == "three_arms_place_objects_123_132":
        # agent0→can1(obj1), agent1→hamburg3(obj3), agent2→teabox2(obj2)
        order = [
            (0, env.can1, init_pose0, 'can1'),
            (1, env.hamburg3, init_pose1, 'hamburg3'),
            (2, env.teabox2, init_pose2, 'teabox2'),
        ]
    elif status == "three_arms_place_objects_123_213":
        # agent0→teabox2(obj2), agent1→can1(obj1), agent2→hamburg3(obj3)
        order = [
            (0, env.teabox2, init_pose0, 'teabox2'),
            (1, env.can1, init_pose1, 'can1'),
            (2, env.hamburg3, init_pose2, 'hamburg3'),
        ]

    # Execute sequentially: each agent picks up its object and places it in the tray
    for agent_id, obj_actor, init_pose, obj_name in order:
        tray_x_offset, tray_y_offset = obj_tray_pos[obj_name]
        # Get grasp pose using OBB (auto-searches for reachable approach direction)
        grasp_pose = planner.get_grasp_pose_from_obb(obj_actor, agent_id)
        # Step 1: Move above the object
        grasp_pose[2] += 0.12
        planner.move_to_pose_with_screw([grasp_pose], move_id=[agent_id])
        # Step 2: Move down to grasp
        grasp_pose[2] -= 0.14
        planner.move_to_pose_with_screw([grasp_pose], move_id=[agent_id])
        # Step 3: Close gripper
        planner.close_gripper(close_id=[agent_id])
        # Step 4: Lift up
        grasp_pose[2] += 0.2
        planner.move_to_pose_with_screw([grasp_pose], move_id=[agent_id])
        # Step 5: Move to above tray (with x/y offset, zigzag to avoid collision)
        grasp_pose[:2] = env.tray.pose.p[0, :2]
        grasp_pose[0] += tray_x_offset
        grasp_pose[1] += tray_y_offset
        planner.move_to_pose_with_screw([grasp_pose], move_id=[agent_id])
        # Step 6: Lower into tray
        grasp_pose[2] -= 0.18
        planner.move_to_pose_with_screw([grasp_pose], move_id=[agent_id])
        # Step 7: Release object
        planner.open_gripper(open_id=[agent_id])
        # Step 8: Retract upward
        grasp_pose[2] += 0.1
        planner.move_to_pose_with_screw([grasp_pose], move_id=[agent_id])
        # Step 9: Return to initial pose
        res = planner.move_to_pose_with_screw([init_pose], move_id=[agent_id])


    planner.close()
    return res


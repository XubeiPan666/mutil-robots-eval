import numpy as np
import sapien

from tasks import ThreeArmsStackCubesEnv
from planner.motionplanner import PandaArmMotionPlanningSolver

def solve(env: ThreeArmsStackCubesEnv, seed=None, debug=False, vis=False, status=None):
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
    

    if status == "three_arms_stack_cubes":
        init_pose1 = env.agent.agents[0].tcp.pose
        init_pose2 = env.agent.agents[1].tcp.pose
        init_pose3 = env.agent.agents[2].tcp.pose
        # Step 1: 获取抓取姿态
        pose1 = planner.get_grasp_pose_from_obb(env.cube1, 0)
        pose2 = planner.get_grasp_pose_from_obb(env.cube2, 1)
        pose3 = planner.get_grasp_pose_from_obb(env.cube3, 2)


        pose1[2] += 0.12
        # pose2[2] += 0.12
        # pose3[2] += 0.12
        planner.move_to_pose_with_screw([pose1], move_id=[0])
        pose1[2] -= 0.14
        # pose2[2] -= 0.14
        # pose3[2] -= 0.14
        planner.move_to_pose_with_screw([pose1], move_id=[0])
        planner.close_gripper(close_id=[0])
        pose1[2] += 0.2
        # pose2[2] += 0.2
        # pose2[0] += 0.2
        # pose3[2] += 0.2
        planner.move_to_pose_with_screw([pose1], move_id=[0])
        

        pose1[:2] = env.goal_region.pose.p[0, :2]
        # pose1[0] -= 0.05
        planner.move_to_pose_with_screw([pose1], move_id=[0])
        pose1[2] -= 0.18
        planner.move_to_pose_with_screw([pose1], move_id=[0]) #time_step越小越慢
        planner.open_gripper(open_id=[0])
        pose1[2] += 0.1
        planner.move_to_pose_with_screw([pose1], move_id=[0])
        planner.move_to_pose_with_screw([init_pose1], move_id=[0])

        pose2[2] += 0.12
        planner.move_to_pose_with_screw([pose2], move_id=[1])
        pose2[2] -= 0.14
        planner.move_to_pose_with_screw([pose2], move_id=[1])
        planner.close_gripper(close_id=[1])
        pose2[2] += 0.2
        pose2[0] += 0.2
        planner.move_to_pose_with_screw([pose2], move_id=[1])
        pose2[:2] = env.cube1.pose.p[0, :2]
        # pose2[0] += 0.04
        # pose2[0] += 0.08
        planner.move_to_pose_with_screw([pose2], move_id=[1])
        pose2[2] -= 0.14
        planner.move_to_pose_with_screw([pose2], move_id=[1])
        planner.open_gripper(open_id=[1])
        pose2[2] += 0.1
        planner.move_to_pose_with_screw([pose2], move_id=[1])
        planner.move_to_pose_with_screw([init_pose2], move_id=[1])

        pose3[2] += 0.12
        planner.move_to_pose_with_screw([pose3], move_id=[2])
        pose3[2] -= 0.14
        planner.move_to_pose_with_screw([pose3], move_id=[2])
        planner.close_gripper(close_id=[2])
        pose3[2] += 0.2
        planner.move_to_pose_with_screw([pose3], move_id=[2])
        pose3[:2] = env.cube2.pose.p[0, :2]
        # pose3[0] -= 0.04
        # pose3[0] -= 0.08
        planner.move_to_pose_with_screw([pose3], move_id=[2])
        pose3[2] -= 0.1
        planner.move_to_pose_with_screw([pose3], move_id=[2])
        planner.open_gripper(open_id=[2])
        pose3[2] += 0.1
        planner.move_to_pose_with_screw([pose3], move_id=[2])
        res = planner.move_to_pose_with_screw([init_pose3], move_id=[2])


    planner.close()
    return res


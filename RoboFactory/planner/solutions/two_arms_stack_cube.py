import numpy as np
import sapien
import torch

from tasks import TwoArmsStackCubeEnv
from planner.motionplanner import PandaArmMotionPlanningSolver

def solve(env: TwoArmsStackCubeEnv, seed=101, debug=False, vis=False, status=None):
    env.reset(seed=seed)
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=[agent.robot.pose for agent in env.agent.agents],
        visualize_target_grasp_pose=vis,
        print_env_info=False,
        is_multi_agent=True
    )
    env = env.unwrapped
    # Task decomposition
    # Step 1: Two robots pick up the cubes
    if status == "two_arms_stack_cube":
        init_poseA = env.agent.agents[0].tcp.pose
        init_poseB = env.agent.agents[1].tcp.pose
        grasp_poseA = planner.get_grasp_pose_from_obb(env.cubeA, 0)
        grasp_poseB = planner.get_grasp_pose_from_obb(env.cubeB, 1)
        
        grasp_poseA[2] += 0.12
        # grasp_poseB[2] += 0.12
        planner.move_to_pose_with_screw([grasp_poseA], move_id=[0])
        grasp_poseA[2] -= 0.14
        # grasp_poseB[2] -= 0.14
        planner.move_to_pose_with_screw([grasp_poseA], move_id=[0])
        planner.close_gripper([0])
        grasp_poseA[2] += 0.2
        # grasp_poseB[2] += 0.2
        planner.move_to_pose_with_screw([grasp_poseA], move_id=[0])

        # Step 2: Stack the cubes
        # target_poseA = planner.get_grasp_pose_for_stack(grasp_poseA, env.goal_region)
        grasp_poseA[:2] = env.goal_region.pose.p[0, :2]
        planner.move_to_pose_with_screw(grasp_poseA, move_id=0)
        grasp_poseA[2] -= 0.15
        planner.move_to_pose_with_screw(grasp_poseA, move_id=0)
        planner.open_gripper([0])
        grasp_poseA[2] += 0.2
        planner.move_to_pose_with_screw(grasp_poseA, move_id=0)
        planner.move_to_pose_with_screw(init_poseA, move_id=0)

        if torch.linalg.norm(
                env.cubeA.pose.p[:, :2] - env.goal_region.pose.p[..., :2], axis=1
            )  < env.goal_radius:
            grasp_poseB[2] += 0.12
            planner.move_to_pose_with_screw([grasp_poseB], move_id=[1])
            grasp_poseB[2] -= 0.14
            planner.move_to_pose_with_screw([grasp_poseB], move_id=[1])
            planner.close_gripper([1])
            grasp_poseB[2] += 0.2
            planner.move_to_pose_with_screw([grasp_poseB], move_id=[1])

            grasp_poseB[:2] = env.cubeA.pose.p[0, :2]
            planner.move_to_pose_with_screw(grasp_poseB, move_id=1)
            grasp_poseB[2] -= 0.15
            planner.move_to_pose_with_screw(grasp_poseB, move_id=1)
            planner.open_gripper([1])
            grasp_poseB[2] += 0.2
            planner.move_to_pose_with_screw(grasp_poseB, move_id=1)
            res = planner.move_to_pose_with_screw(init_poseB, move_id=1)
        else:
            res = -1
        
        planner.close()
    return res

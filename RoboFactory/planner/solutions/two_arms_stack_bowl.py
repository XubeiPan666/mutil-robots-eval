import numpy as np
import sapien

from tasks import TwoArmsStackBowlEnv
from planner.motionplanner import PandaArmMotionPlanningSolver

def solve(env: TwoArmsStackBowlEnv, seed=None, debug=False, vis=False, status=None):
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
    if status == "two_arms_stack_bowl":
        init_poseA = env.agent.agents[0].tcp.pose
        init_poseB = env.agent.agents[1].tcp.pose
        grasp_poseA = planner.get_grasp_pose_w_labeled_direction(actor=env.bowl1, actor_data=env.annotation_data['bowl1'], pre_dis=0, id=3)
        # grasp_poseB = planner.get_grasp_pose_w_labeled_direction(actor=env.bowl2, actor_data=env.annotation_data['bowl2'], pre_dis=0, id=1)
        grasp_poseA[2] += 0.12
        # grasp_poseB[2] += 0.12
        planner.move_to_pose_with_screw([grasp_poseA], move_id=[0])
        grasp_poseA[2] -= 0.14
        # grasp_poseB[2] -= 0.12
        planner.move_to_pose_with_screw([grasp_poseA], move_id=[0])
        planner.close_gripper([0])
        grasp_poseA[2] += 0.3
        # grasp_poseB[2] += 0.3
        planner.move_to_pose_with_screw([grasp_poseA], move_id=[0])

        # Step 2: Stack the bowls
        target_poseA = planner.get_grasp_pose_for_stack(grasp_poseA, env.tray)
        target_poseA[0] -= 0.07   
        target_poseA[2] += 0.2
        planner.move_to_pose_with_screw(target_poseA, move_id=0)
        target_poseA[2] -= 0.1
        planner.move_to_pose_with_screw(target_poseA, move_id=0)
        planner.open_gripper([0])
        target_poseA[2] += 0.2
        planner.move_to_pose_with_screw(target_poseA, move_id=0)
        planner.move_to_pose_with_screw(init_poseA, move_id=0)

        grasp_poseB = planner.get_grasp_pose_w_labeled_direction(actor=env.bowl2, actor_data=env.annotation_data['bowl2'], pre_dis=0, id=3)
        grasp_poseB[2] += 0.12
        # grasp_poseB[2] += 0.12
        planner.move_to_pose_with_screw([grasp_poseB], move_id=[1])
        grasp_poseB[2] -= 0.14
        # grasp_poseB[2] -= 0.12
        planner.move_to_pose_with_screw([grasp_poseB], move_id=[1])
        planner.close_gripper([1])
        grasp_poseB[2] += 0.3
        # grasp_poseB[2] += 0.3
        planner.move_to_pose_with_screw([grasp_poseB], move_id=[1])

        target_poseB = planner.get_grasp_pose_for_stack(grasp_poseB, env.bowl1)
        target_poseB[0] -= 0.08 
        target_poseB[2] += 0.2
        planner.move_to_pose_with_screw(target_poseB, move_id=1)
        target_poseB[2] -= 0.15
        planner.move_to_pose_with_screw(target_poseB, move_id=1)
        planner.open_gripper([1])
        target_poseB[2] += 0.2
        planner.move_to_pose_with_screw(target_poseB, move_id=1)
        res = planner.move_to_pose_with_screw(init_poseB, move_id=1)
        planner.close()
    return res

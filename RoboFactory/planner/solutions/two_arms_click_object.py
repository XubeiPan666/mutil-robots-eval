import numpy as np
import sapien

from tasks import TwoArmsClickObjectEnv
from planner.motionplanner import PandaArmMotionPlanningSolver

def solve(env: TwoArmsClickObjectEnv, seed=None, debug=False, vis=False, status=None):
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
    if status == "two_arms_click_object":
        init_poseA = env.agent.agents[0].tcp.pose
        init_poseB = env.agent.agents[1].tcp.pose
        grasp_poseA = planner.get_grasp_pose_w_labeled_direction(actor=env.alarm_clock, actor_data=env.annotation_data['alarm_clock'], pre_dis=0, id=0)
        planner.close_gripper([0])
        # grasp_poseB = planner.get_grasp_pose_w_labeled_direction(actor=env.bowl2, actor_data=env.annotation_data['bowl2'], pre_dis=0, id=1)
        grasp_poseA[2] += 0.12
        # grasp_poseB[2] += 0.12
        planner.move_to_pose_with_screw([grasp_poseA], move_id=[0])
        grasp_poseA[2] -= 0.12
        # grasp_poseB[2] -= 0.12
        planner.move_to_pose_with_screw([grasp_poseA], move_id=[0], time_step=0.05)
        grasp_poseA[2] += 0.05
        # grasp_poseB[2] += 0.3
        planner.move_to_pose_with_screw([grasp_poseA], move_id=[0])
        # planner.move_to_pose_with_screw(init_poseA, move_id=0)
        # planner.open_gripper([0])


        grasp_poseB = planner.get_grasp_pose_w_labeled_direction(actor=env.bell, actor_data=env.annotation_data['bell'], pre_dis=0, id=0)
        planner.close_gripper([1])
        grasp_poseB[2] += 0.12
        # grasp_poseB[2] += 0.12
        planner.move_to_pose_with_screw([grasp_poseB], move_id=[1])
        grasp_poseB[2] -= 0.12
        # grasp_poseB[2] -= 0.12
        planner.move_to_pose_with_screw([grasp_poseB], move_id=[1], time_step=0.05)
        grasp_poseB[2] += 0.05
        # grasp_poseB[2] += 0.3
        res = planner.move_to_pose_with_screw([grasp_poseB], move_id=[1])
        # planner.move_to_pose_with_screw(init_poseB, move_id=1)
        # res = planner.open_gripper([1])
        planner.close()
    return res

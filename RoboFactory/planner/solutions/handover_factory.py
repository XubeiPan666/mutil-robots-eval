import numpy as np
import sapien
import torch

from tasks import LongPipelineDeliveryEnv
from planner.motionplanner import PandaArmMotionPlanningSolver
from transforms3d.euler import euler2quat
from transforms3d.quaternions import qmult

def solve(env: LongPipelineDeliveryEnv, seed=None, debug=False, vis=False, status=None):
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

    env = env.unwrapped
    if status == "four_arms_welding_cube":
        init_pose1 = env.agent.agents[0].tcp.pose
        init_pose2 = env.agent.agents[1].tcp.pose
        init_pose3 = env.agent.agents[2].tcp.pose
        init_pose4 = env.agent.agents[3].tcp.pose

        # while 1:
        planner.close_gripper(close_id=[0, 1, 2, 3])
        
        pose1 = planner.get_grasp_pose_from_obb(actor=env.welding_sphere_1, agent_id=0)
        pose1[2] += 0.15
        planner.move_to_pose_with_screw(pose1, move_id=[0])
        pose1[2] -= 0.16
        planner.move_to_pose_with_screw(pose1, move_id=[0], time_step=0.05)
        pose1[2] += 0.15
        planner.move_to_pose_with_screw(pose1, move_id=[0])
        planner.move_to_pose_with_screw(init_pose1, move_id=[0])

        pose2 = planner.get_grasp_pose_from_obb(actor=env.welding_sphere_2, agent_id=2)
        pose2[2] += 0.15
        planner.move_to_pose_with_screw(pose2, move_id=[2])
        pose2[2] -= 0.16
        planner.move_to_pose_with_screw(pose2, move_id=[2], time_step=0.05)
        pose2[2] += 0.15
        planner.move_to_pose_with_screw(pose2, move_id=[2])
        planner.move_to_pose_with_screw(init_pose3, move_id=[2])


        pose3 = planner.get_grasp_pose_from_obb(actor=env.welding_sphere_3, agent_id=1)
        pose3[2] += 0.15
        planner.move_to_pose_with_screw(pose3, move_id=[1])
        pose3[2] -= 0.16
        planner.move_to_pose_with_screw(pose3, move_id=[1], time_step=0.05)
        pose3[2] += 0.15
        planner.move_to_pose_with_screw(pose3, move_id=[1])
        planner.move_to_pose_with_screw(init_pose2, move_id=[1])

        pose4 = planner.get_grasp_pose_from_obb(actor=env.welding_sphere_4, agent_id=3)
        pose4[2] += 0.15
        planner.move_to_pose_with_screw(pose4, move_id=[3])
        pose4[2] -= 0.16
        planner.move_to_pose_with_screw(pose4, move_id=[3])
        pose4[2] += 0.15
        planner.move_to_pose_with_screw(pose4, move_id=[3])
        res = planner.move_to_pose_with_screw(init_pose4, move_id=[3])
    
    planner.close()
    return res

import numpy as np
import sapien
import torch

from tasks import FourArmsWeldingCubeEnv
from planner.motionplanner import PandaArmMotionPlanningSolver
from transforms3d.euler import euler2quat
from transforms3d.quaternions import qmult

def solve(env: FourArmsWeldingCubeEnv, seed=None, debug=False, vis=False, status=None):
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
    # if status == "four_arms_welding_cube":
    init_pose1 = env.agent.agents[0].tcp.pose
    init_pose2 = env.agent.agents[1].tcp.pose
    init_pose3 = env.agent.agents[2].tcp.pose
    init_pose4 = env.agent.agents[3].tcp.pose

    # prepare_work_pose_0 = np.array([0, -0.156, 0.419, 0, 0.707, 0.707, 0])
    # prepare_work_pose_1 = np.array([0.156, 0, 0.419, 0., 0., 1, 0.])
    # prepare_work_pose_2 = np.array([0, 0.156, 0.419, 0, 0.707, -0.707, 0])
    # prepare_work_pose_3 = np.array([-0.156, 0, 0.419, 0., 1., 0., 0.])

        # while 1:
        #     planner.close_gripper(close_id=[0, 1, 2, 3])
    if status == "four_arms_welding_cube_1234_1234":
        
        planner.close_gripper(close_id=[0])
        pose1 = planner.get_grasp_pose_from_obb(actor=env.welding_sphere_1, agent_id=0)
        pose1[2] += 0.15
        planner.move_to_pose_with_screw(pose1, move_id=[0])
        pose1[2] -= 0.16
        planner.move_to_pose_with_screw(pose1, move_id=[0])
        pose1[2] += 0.15
        planner.move_to_pose_with_screw(pose1, move_id=[0])
        planner.move_to_pose_with_screw(init_pose1, move_id=[0])
        planner.open_gripper(open_id=[0])

        planner.close_gripper(close_id=[1])
        pose2 = planner.get_grasp_pose_from_obb(actor=env.welding_sphere_2, agent_id=1)
        pose2[2] += 0.15
        planner.move_to_pose_with_screw(pose2, move_id=[1])
        pose2[2] -= 0.16
        planner.move_to_pose_with_screw(pose2, move_id=[1])
        pose2[2] += 0.15
        planner.move_to_pose_with_screw(pose2, move_id=[1])
        planner.move_to_pose_with_screw(init_pose2, move_id=[1])
        planner.open_gripper(open_id=[1])

        planner.close_gripper(close_id=[2])
        pose3 = planner.get_grasp_pose_from_obb(actor=env.welding_sphere_3, agent_id=2)
        pose3[2] += 0.15
        planner.move_to_pose_with_screw(pose3, move_id=[2])
        pose3[2] -= 0.16
        planner.move_to_pose_with_screw(pose3, move_id=[2])
        pose3[2] += 0.15
        planner.move_to_pose_with_screw(pose3, move_id=[2])
        planner.move_to_pose_with_screw(init_pose3, move_id=[2])
        planner.open_gripper(open_id=[2])

        planner.close_gripper(close_id=[3])
        pose4 = planner.get_grasp_pose_from_obb(actor=env.welding_sphere_4, agent_id=3)
        pose4[2] += 0.15
        planner.move_to_pose_with_screw(pose4, move_id=[3])
        pose4[2] -= 0.16
        planner.move_to_pose_with_screw(pose4, move_id=[3])
        pose4[2] += 0.15
        planner.move_to_pose_with_screw(pose4, move_id=[3])
        planner.move_to_pose_with_screw(init_pose4, move_id=[3])
        res = planner.open_gripper(open_id=[3])
    
    elif status == "four_arms_welding_cube_1234_2341":
        
        planner.close_gripper(close_id=[0])
        pose1 = planner.get_grasp_pose_from_obb(actor=env.welding_sphere_2, agent_id=0)
        current_quat_1 = pose1[3:7].copy()  # 保存当前旋转四元数
        # 创建绕z轴旋转90度的四元数
        rotation_quat_1 = euler2quat(0, np.pi/2, 0, 'sxyz')  # 绕z轴旋转90度
        # 组合旋转：先应用当前旋转，再应用新旋转
        new_quat_1 = qmult(current_quat_1, rotation_quat_1)
        # 更新pose1，保持位置不变，只改变旋转
        pose1[1] -= 0.05
        pose1[3:7] = new_quat_1
        planner.move_to_pose_with_screw(pose1, move_id=[0])
        pose1[1] += 0.05
        planner.move_to_pose_with_screw(pose1, move_id=[0])
        planner.move_to_pose_with_screw(init_pose1, move_id=[0])
        planner.open_gripper(open_id=[0])

        planner.close_gripper(close_id=[1])
        # pose2 = planner.get_grasp_pose_from_obb(actor=env.welding_sphere_3, agent_id=1)
        # pose2[2] += 0.12
        # planner.move_to_pose_with_screw(pose2, move_id=[1])
        # pose2[2] -= 0.12
        # planner.move_to_pose_with_screw(pose2, move_id=[1])
        # planner.move_to_pose_with_screw(init_pose2, move_id=[1])
        # planner.open_gripper(open_id=[1])
        pose2 = planner.get_grasp_pose_from_obb(actor=env.welding_sphere_3, agent_id=1)
        current_quat_2 = pose2[3:7].copy()  # 保存当前旋转四元数
        # 创建绕z轴旋转90度的四元数
        rotation_quat_2 = euler2quat(0, np.pi/2, 0, 'sxyz')  # 绕z轴旋转90度
        # 组合旋转：先应用当前旋转，再应用新旋转
        new_quat_2 = qmult(current_quat_2, rotation_quat_2)
        # 更新pose1，保持位置不变，只改变旋转
        pose2[1] += 0.05
        pose2[0] += 0.05
        pose2[3:7] = new_quat_2
        planner.move_to_pose_with_screw(pose2, move_id=[1])
        pose2[1] -= 0.05
        planner.move_to_pose_with_screw(pose2, move_id=[1])
        planner.move_to_pose_with_screw(init_pose2, move_id=[1])
        planner.open_gripper(open_id=[1])

        planner.close_gripper(close_id=[2])
        pose3 = planner.get_grasp_pose_from_obb(actor=env.welding_sphere_4, agent_id=2)
        current_quat_2 = pose3[3:7].copy()  # 保存当前旋转四元数
        rotation_quat_2 = euler2quat(0, np.pi/2, 0, 'sxyz')  # 绕z轴旋转90度
        new_quat_2 = qmult(current_quat_2, rotation_quat_2)
        pose3[1] += 0.05
        pose3[3:7] = new_quat_2
        planner.move_to_pose_with_screw(pose3, move_id=[2])
        pose3[1] -= 0.05
        planner.move_to_pose_with_screw(pose3, move_id=[2])
        planner.move_to_pose_with_screw(init_pose3, move_id=[2])
        planner.open_gripper(open_id=[2])

        planner.close_gripper(close_id=[3])
        pose4 = planner.get_grasp_pose_from_obb(actor=env.welding_sphere_1, agent_id=3)
        pose4[2] += 0.12
        planner.move_to_pose_with_screw(pose4, move_id=[3])
        pose4[2] -= 0.12
        planner.move_to_pose_with_screw(pose4, move_id=[3])
        planner.move_to_pose_with_screw(init_pose4, move_id=[3])
        res = planner.open_gripper(open_id=[3])
    


        # while 1:
        #     planner.close_gripper(close_id=[0])
        # pose1[2] -= 0.16
        # planner.move_to_pose_with_screw(pose1, move_id=[0])
        # pose1[2] += 0.15
        # planner.move_to_pose_with_screw(pose1, move_id=[0])
        # planner.move_to_pose_with_screw(init_pose1, move_id=[0])
        # planner.open_gripper(open_id=[0])
    

    planner.close()
    return res

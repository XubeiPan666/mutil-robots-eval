import numpy as np
import sapien
import torch
import time

from tasks import PickMeatInBasketEnv
from planner.motionplanner import PandaArmMotionPlanningSolver
from transforms3d.euler import euler2quat
from transforms3d.quaternions import qmult

def solve(env: PickMeatInBasketEnv, seed=None, debug=False, vis=False, status=None):
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
    
    if status == "pick_meat_in_basket":
        pose_rob1 = env.agent.agents[0].tcp.pose
        # while 1:
        #     planner.open_gripper(open_id=[0])
        
        pose1 = planner.get_grasp_pose_w_labeled_direction(actor=env.meat, actor_data=env.annotation_data['meat'], pre_dis=0, id=0)
        pose2 = planner.get_grasp_pose_w_labeled_direction(actor=env.basket, actor_data=env.annotation_data['basket'], pre_dis=0, id=1)
        pose1[2] += 0.12
        pose2[2] += 0.12
        planner.move_to_pose_with_screw([pose1, pose2], move_id=[0,1])
        pose1[2] -= 0.12
        pose2[2] -= 0.12
        planner.move_to_pose_with_screw([pose1, pose2], move_id=[0,1])
        planner.close_gripper(close_id=[0,1])
        pose1[:2] = (env.agent.agents[0].robot.pose.p[0, :2] + env.agent.agents[1].robot.pose.p[0, :2]) / 2
        pose2[:2] = (env.agent.agents[0].robot.pose.p[0, :2] + env.agent.agents[1].robot.pose.p[0, :2]) / 2
        # pose1[0] += 0.4
        # pose2[0] += 0.1
        pose1[0] += 0.6
        pose2[0] += 0.4
        pose1[2] += 0.4
        pose2[2] += 0.15
        # pose2[1] += 0.2
        planner.move_to_pose_with_screw([pose1, pose2], move_id=[0,1])
        grasppose_basket = pose2
        grasppose_basket2 = pose2
        
        # Adjust grasp pose: translation and rotation for tilted end-effector
        # Extract position and quaternion
        pos = grasppose_basket[:3]  # [x, y, z]
        quat = grasppose_basket[3:]  # [qx, qy, qz, qw]

        pos2 = grasppose_basket2[:3]  # [x, y, z]
        quat2 = grasppose_basket2[3:]  # [qx, qy, qz, qw]
        # Apply translation offset (adjust these values as needed)
        translation_offset = np.array([0.05, 0.0, 0.0])  # [x_offset, y_offset, z_offset]
        pos_adjusted = pos + translation_offset
        
        # Apply rotation for tilted angle
        # Create rotation quaternion for tilt (e.g., rotate around y-axis by tilt_angle degrees)
        
        tilt_angle_x1 = np.radians(0)  # Tilt angle around x-axis in degrees (adjust as needed)
        tilt_angle_y1 = np.radians(-75)   # Tilt angle around y-axis in degrees (adjust as needed)
        tilt_angle_z1 = np.radians(90)   # Tilt angle around z-axis in degrees (adjust as needed)
        tilt_quat1 = euler2quat(tilt_angle_z1, tilt_angle_y1, tilt_angle_x1, 'rzyx')

        tilt_angle_x2 = np.radians(0)  # Tilt angle around x-axis in degrees (adjust as needed)
        tilt_angle_y2 = np.radians(0)   # Tilt angle around y-axis in degrees (adjust as needed)
        tilt_angle_z2 = np.radians(0)   # Tilt angle around z-axis in degrees (adjust as needed)
        tilt_quat2 = euler2quat(tilt_angle_z2, tilt_angle_y2, tilt_angle_x2, 'rzyx')
        
        # Combine original quaternion with tilt rotation
        
        quat_adjusted = qmult(quat, tilt_quat1)
        quat_adjusted2 = qmult(quat2, tilt_quat2)
        
        # Reconstruct adjusted pose
        grasppose_basket = np.concatenate([pos_adjusted, quat_adjusted])
        grasppose_basket2 = np.concatenate([pos2, quat_adjusted2])
        
        # Make first action faster by using smaller time_step
        # Smaller time_step means faster execution (more aggressive time parameterization)
        # Use list of time_steps to make first action faster while both execute simultaneously
        # faster_time_step = env.control_timestep * 0.7  # 30% faster for first action (adjust as needed)
        # normal_time_step = env.control_timestep        # Normal speed for second action
        
        # Execute both actions simultaneously, but first one with faster time_step
        planner.move_to_pose_with_screw([grasppose_basket, grasppose_basket2], move_id=[0, 1])
        # while 1:
        #     planner.close_gripper()
        
        # planner.move_to_pose_with_screw(pose1, move_id=[0])
        res = planner.open_gripper(open_id=[0])
        # pose1[2] += 0.2
        # planner.move_to_pose_with_screw(pose1, move_id=[0])
        # pose1 = planner.get_grasp_pose_w_labeled_direction(actor=env.basket, actor_data=env.annotation_data['basket'], pre_dis=0, id=1)
        # pose1[1] += 0.05
        # pose1[2] += 0.05
        # planner.move_to_pose_with_screw(pose1, move_id=[0])
        # planner.open_gripper(open_id=[0])
        # pose1[2] += 0.1
        # planner.move_to_pose_with_screw(pose1, move_id=[0])
        # planner.move_to_pose_with_screw(pose_rob1, move_id=[0])
        
        # pose2 = planner.get_grasp_pose_w_labeled_direction(actor=env.basket, actor_data=env.annotation_data['basket'], pre_dis=0, id=1)
        # pose2[2] += 0.12
        # planner.move_to_pose_with_screw(pose2, move_id=[1])
        # pose2[2] -= 0.12
        # planner.move_to_pose_with_screw(pose2, move_id=[1])
        # planner.close_gripper(close_id=[1])
        # pose2[2] += 0.2
        # res = planner.move_to_pose_with_screw(pose2, move_id=[1])
    
    elif status == "pick_meat_in_basket_order":
        # while 1:
        #     planner.open_gripper(open_id=[0])
        init_pose1 = env.agent.agents[0].tcp.pose
        pose1 = planner.get_grasp_pose_w_labeled_direction(actor=env.meat, actor_data=env.annotation_data['meat'], pre_dis=0, id=0)
        pose2 = planner.get_grasp_pose_w_labeled_direction(actor=env.basket, actor_data=env.annotation_data['basket'], pre_dis=0, id=1)
        pose1[2] += 0.12
        # pose2[2] += 0.12
        planner.move_to_pose_with_screw([pose1], move_id=[0])
        pose1[2] -= 0.12
        # pose2[2] -= 0.12
        planner.move_to_pose_with_screw([pose1], move_id=[0])
        planner.close_gripper(close_id=[0])
        pose1[2] += 0.3
        planner.move_to_pose_with_screw([pose1], move_id=[0])
        pose1 = pose2
        pose1[0] -= 0.05
        pose1[2] += 0.05
        # planner.move_to_pose_with_screw([pose1], move_id=[0])

        current_pos = pose1[:3].copy()  # 保存当前位置
        current_quat = pose1[3:7].copy()  # 保存当前旋转四元数
        # 创建绕z轴旋转90度的四元数
        rotation_quat = euler2quat(0, 0, np.pi/2, 'sxyz')  # 绕z轴旋转90度
        # 组合旋转：先应用当前旋转，再应用新旋转
        new_quat = qmult(current_quat, rotation_quat)
        # 更新pose1，保持位置不变，只改变旋转
        pose1[:3] = current_pos
        pose1[3:7] = new_quat
        planner.move_to_pose_with_screw([pose1], move_id=[0])

        planner.open_gripper(open_id=[0])
        planner.move_to_pose_with_screw([init_pose1], move_id=[0])
        pose2 = planner.get_grasp_pose_w_labeled_direction(actor=env.basket, actor_data=env.annotation_data['basket'], pre_dis=0, id=1)
        pose2[2] += 0.12
        planner.move_to_pose_with_screw([pose2], move_id=[1])
        pose2[2] -= 0.12
        planner.move_to_pose_with_screw([pose2], move_id=[1])
        planner.close_gripper(close_id=[1])
        pose2[2] += 0.2
        res = planner.move_to_pose_with_screw([pose2], move_id=[1])

        

    planner.close()
    return res

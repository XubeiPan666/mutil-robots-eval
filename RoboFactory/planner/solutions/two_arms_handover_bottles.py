import numpy as np
import sapien

from tasks import TwoArmsHandoverBottlesEnv
from planner.motionplanner import PandaArmMotionPlanningSolver
from transforms3d.euler import euler2quat
from transforms3d.quaternions import qmult

def solve(env: TwoArmsHandoverBottlesEnv, seed=None, debug=False, vis=False, status=None):
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
    if status == "two_arms_handover_bottles":
        # while 1:
        #     planner.open_gripper([0, 1])

        init_poseA = env.agent.agents[0].tcp.pose
        init_poseB = env.agent.agents[1].tcp.pose

        #handover bottle1 to basket
        grasp_poseA = planner.get_grasp_pose_w_labeled_direction(actor=env.bottle1, actor_data=env.annotation_data['bottle1'], pre_dis=0, id=1)
        grasp_poseA[2] += 0.12
        planner.move_to_pose_with_screw([grasp_poseA], move_id=[0])
        grasp_poseA[2] -= 0.14
        planner.move_to_pose_with_screw([grasp_poseA], move_id=[0])
        

        # grasp_poseA = planner.get_grasp_pose_w_labeled_direction(actor=env.bottle1, actor_data=env.annotation_data['bottle1'], pre_dis=0, id=1)
        # # grasp_poseA[2] += 0.12
        # # planner.move_to_pose_with_screw([grasp_poseA], move_id=[0])
        # # grasp_poseA[2] -= 0.14
        # planner.move_to_pose_with_screw([grasp_poseA], move_id=[0])
        planner.close_gripper([0])
        grasp_poseA[2] += 0.2
        planner.move_to_pose_with_screw([grasp_poseA], move_id=[0])


        grasp_poseA[:2] = (env.agent.agents[0].robot.pose.p[0, :2] + env.agent.agents[1].robot.pose.p[0, :2]) / 2
        grasp_poseA[0] += 0.6
        grasp_poseA[2] += 0.3

        current_quat = grasp_poseA[3:7].copy()  # 保存当前旋转四元数
        # 创建绕z轴旋转90度的四元数
        rotation_quat = euler2quat(0, np.pi/2, -np.pi/2, 'sxyz')  # 绕z轴旋转90度
        # 组合旋转：先应用当前旋转，再应用新旋转
        new_quat_1 = qmult(current_quat, rotation_quat)
        # 更新pose1，保持位置不变，只改变旋转
        grasp_poseA[3:7] = new_quat_1

        planner.move_to_pose_with_screw(grasp_poseA, move_id=0)
        
        
        grasp_poseB = planner.get_grasp_pose_w_labeled_direction(actor=env.bottle1, actor_data=env.annotation_data['bottle1'], pre_dis=-0.05, id=3)
        planner.move_to_pose_with_screw(grasp_poseB, move_id=1)
        grasp_poseB = planner.get_grasp_pose_w_labeled_direction(actor=env.bottle1, actor_data=env.annotation_data['bottle1'], pre_dis=0, id=3)
        grasp_poseB[2] -= 0.05 
        planner.move_to_pose_with_screw(grasp_poseB, move_id=1)
        planner.close_gripper([1])
        planner.open_gripper([0])
        grasp_poseA[1] -= 0.1
        planner.move_to_pose_with_screw(grasp_poseA, move_id=0)
        planner.move_to_pose_with_screw(init_poseA, move_id=0)
        grasp_poseB[:2] = env.basket.pose.p[0, :2]
        grasp_poseB[1] -= 0.05
        grasp_poseB[2] -= 0.15
        

        # current_quat = grasp_poseB[3:7].copy()  # 保存当前旋转四元数
        # # 创建绕z轴旋转90度的四元数
        # rotation_quat = euler2quat(np.pi/2, 0, -np.pi/2, 'sxyz')  # 绕z轴旋转90度
        # # 组合旋转：先应用当前旋转，再应用新旋转
        # new_quat_1 = qmult(current_quat, rotation_quat)
        # # 更新pose1，保持位置不变，只改变旋转
        # grasp_poseB[3:7] = new_quat_1
        # planner.move_to_pose_with_screw(grasp_poseB, move_id=1)
        # rotation_quat = euler2quat(0, 0, np.pi/2, 'sxyz')  # 绕z轴旋转90度
        # new_quat_1 = qmult(current_quat, rotation_quat)
        # grasp_poseB[3:7] = new_quat_1
        planner.move_to_pose_with_screw(grasp_poseB, move_id=1)
        planner.open_gripper([1])
        grasp_poseB[1] += 0.2
        grasp_poseB[0] -= 0.2
        grasp_poseB[2] += 0.1
        planner.move_to_pose_with_screw(grasp_poseB, move_id=1)
        planner.move_to_pose_with_screw(init_poseB, move_id=1)

        #handover bottle2 to basket
        # grasp_poseA = planner.get_grasp_pose_w_labeled_direction(actor=env.bottle2, actor_data=env.annotation_data['bottle2'], pre_dis=0, id=1)
        # grasp_poseA[2] += 0.12
        # planner.move_to_pose_with_screw([grasp_poseA], move_id=[0])
        # grasp_poseA[2] -= 0.14
        # planner.move_to_pose_with_screw([grasp_poseA], move_id=[0])
        # planner.close_gripper([0])
        # grasp_poseA[2] += 0.3
        # planner.move_to_pose_with_screw([grasp_poseA], move_id=[0])


        # grasp_poseA[:2] = (env.agent.agents[0].robot.pose.p[0, :2] + env.agent.agents[1].robot.pose.p[0, :2]) / 2
        # grasp_poseA[0] += 0.6
        # planner.move_to_pose_with_screw(grasp_poseA, move_id=0)
        

        # grasp_poseB = planner.get_grasp_pose_w_labeled_direction(actor=env.bottle2, actor_data=env.annotation_data['bottle2'], pre_dis=-0.05, id=2)
        # planner.move_to_pose_with_screw(grasp_poseB, move_id=1)
        # grasp_poseB = planner.get_grasp_pose_w_labeled_direction(actor=env.bottle2, actor_data=env.annotation_data['bottle2'], pre_dis=0, id=2)
        # planner.move_to_pose_with_screw(grasp_poseB, move_id=1)
        # planner.close_gripper([1])
        # planner.open_gripper([0])
        # grasp_poseA[1] -= 0.1
        # planner.move_to_pose_with_screw(grasp_poseA, move_id=0)
        # planner.move_to_pose_with_screw(init_poseA, move_id=0)
        # grasp_poseB[:2] = env.basket.pose.p[0, :2]
        # grasp_poseB[1] -= 0.05
        # planner.move_to_pose_with_screw(grasp_poseB, move_id=1)
        # planner.open_gripper([1])
        # grasp_poseB[1] -= 0.2
        # grasp_poseB[0] -= 0.2
        # planner.move_to_pose_with_screw(grasp_poseB, move_id=1)
        # res = planner.move_to_pose_with_screw(init_poseB, move_id=1)

        grasp_poseA = planner.get_grasp_pose_w_labeled_direction(actor=env.bottle2, actor_data=env.annotation_data['bottle2'], pre_dis=0, id=5)
        grasp_poseA[2] += 0.12
        planner.move_to_pose_with_screw([grasp_poseA], move_id=[0])
        grasp_poseA[2] -= 0.14
        planner.move_to_pose_with_screw([grasp_poseA], move_id=[0])
        

        # grasp_poseA = planner.get_grasp_pose_w_labeled_direction(actor=env.bottle1, actor_data=env.annotation_data['bottle1'], pre_dis=0, id=1)
        # # grasp_poseA[2] += 0.12
        # # planner.move_to_pose_with_screw([grasp_poseA], move_id=[0])
        # # grasp_poseA[2] -= 0.14
        # planner.move_to_pose_with_screw([grasp_poseA], move_id=[0])
        planner.close_gripper([0])
        grasp_poseA[2] += 0.2
        planner.move_to_pose_with_screw([grasp_poseA], move_id=[0])


        grasp_poseA[:2] = (env.agent.agents[0].robot.pose.p[0, :2] + env.agent.agents[1].robot.pose.p[0, :2]) / 2
        grasp_poseA[0] += 0.6
        grasp_poseA[2] += 0.3

        current_quat = grasp_poseA[3:7].copy()  # 保存当前旋转四元数
        # 创建绕z轴旋转90度的四元数
        rotation_quat = euler2quat(0, np.pi/2, -np.pi/2, 'sxyz')  # 绕z轴旋转90度
        # 组合旋转：先应用当前旋转，再应用新旋转
        new_quat_1 = qmult(current_quat, rotation_quat)
        # 更新pose1，保持位置不变，只改变旋转
        grasp_poseA[3:7] = new_quat_1

        planner.move_to_pose_with_screw(grasp_poseA, move_id=0)
        

        grasp_poseB = planner.get_grasp_pose_w_labeled_direction(actor=env.bottle2, actor_data=env.annotation_data['bottle2'], pre_dis=-0.05, id=1)
        planner.move_to_pose_with_screw(grasp_poseB, move_id=1)
        grasp_poseB = planner.get_grasp_pose_w_labeled_direction(actor=env.bottle2, actor_data=env.annotation_data['bottle2'], pre_dis=0, id=1)
        grasp_poseB[2] -= 0.05 
        planner.move_to_pose_with_screw(grasp_poseB, move_id=1)
        planner.close_gripper([1])
        planner.open_gripper([0])
        grasp_poseA[1] -= 0.1
        planner.move_to_pose_with_screw(grasp_poseA, move_id=0)
        planner.move_to_pose_with_screw(init_poseA, move_id=0)
        grasp_poseB[:2] = env.basket.pose.p[0, :2]
        grasp_poseB[1] += 0.05
        grasp_poseB[2] -= 0.15

        # current_quat = grasp_poseB[3:7].copy()  # 保存当前旋转四元数
        # # 创建绕z轴旋转90度的四元数
        # rotation_quat = euler2quat(np.pi/2, 0, -np.pi/2, 'sxyz')  # 绕z轴旋转90度
        # # 组合旋转：先应用当前旋转，再应用新旋转
        # new_quat_1 = qmult(current_quat, rotation_quat)
        # # 更新pose1，保持位置不变，只改变旋转
        # grasp_poseB[3:7] = new_quat_1
        # planner.move_to_pose_with_screw(grasp_poseB, move_id=1)
        # rotation_quat = euler2quat(0, 0, np.pi/2, 'sxyz')  # 绕z轴旋转90度
        # new_quat_1 = qmult(current_quat, rotation_quat)
        # grasp_poseB[3:7] = new_quat_1
        planner.move_to_pose_with_screw(grasp_poseB, move_id=1)
        planner.open_gripper([1])
        grasp_poseB[1] += 0.2
        grasp_poseB[0] -= 0.2
        grasp_poseB[2] += 0.1
        planner.move_to_pose_with_screw(grasp_poseB, move_id=1)
        res = planner.move_to_pose_with_screw(init_poseB, move_id=1)
        
        planner.close()
    return res

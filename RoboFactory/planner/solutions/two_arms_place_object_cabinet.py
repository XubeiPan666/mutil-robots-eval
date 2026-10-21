import numpy as np
import sapien

from tasks import TwoArmsPlaceObjectCabinetEnv
from planner.motionplanner import PandaArmMotionPlanningSolver
from transforms3d.euler import euler2quat
from transforms3d.quaternions import qmult

def solve(env: TwoArmsPlaceObjectCabinetEnv, seed=None, debug=False, vis=False, status=None):
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
    if status == "two_arms_place_object_cabinet":
        init_poseA = env.agent.agents[0].tcp.pose
        init_poseB = env.agent.agents[1].tcp.pose
        # while 1:
        #     res = planner.close_gripper()
        # Step 1: Pull the cabinet 
        pose1 = planner.get_grasp_pose_w_labeled_direction(actor=env.cabinet, actor_data=env.annotation_data['cabinet'], pre_dis=0, id=1)
        pose1[0] -= 0.12
        planner.move_to_pose_with_screw(pose1, move_id=[1])
        pose1[0] += 0.13
        planner.move_to_pose_with_screw(pose1, move_id=[1])
        planner.close_gripper(close_id=[1])
        pose1[0] -= 0.15
        planner.move_to_pose_with_screw(pose1, move_id=[1])

        # Step 2: Place the object in the cabinet
        pose1 = planner.get_grasp_pose_w_labeled_direction(actor=env.object, actor_data=env.annotation_data['object'], pre_dis=0, id=2)
        pose1[2] += 0.12
        planner.move_to_pose_with_screw(pose1, move_id=[0])
        pose1[2] -= 0.14
        planner.move_to_pose_with_screw(pose1, move_id=[0])
        planner.close_gripper(close_id=[0])
        # pose1[0] -= 0.2
        planner.move_to_pose_with_screw(init_poseA, move_id=[0])
        target_pose = planner.get_grasp_pose_w_labeled_direction(actor=env.cabinet, actor_data=env.annotation_data['cabinet'], pre_dis=0, id=0)
        current_pos = target_pose[:3].copy()  # 保存当前位置
        current_quat = target_pose[3:7].copy()  # 保存当前旋转四元数
        # 创建绕z轴旋转90度的四元数
        rotation_quat = euler2quat(np.pi/2, -np.pi/2, 0, 'sxyz')  # 绕z轴旋转90度
        # 组合旋转：先应用当前旋转，再应用新旋转
        new_quat = qmult(current_quat, rotation_quat)
        current_pos[0] += 0.1
        current_pos[2] += 0.1
        # 更新pose1，改变位置和旋转
        target_pose[:3] = current_pos
        target_pose[3:7] = new_quat
        planner.move_to_pose_with_screw(target_pose, move_id=[0])
        target_pose[2] -= 0.15
        planner.move_to_pose_with_screw(target_pose, move_id=[0])
        res = planner.open_gripper(open_id=[0])

        
        # while 1:
        #     res = planner.close_gripper()
        # grasp_poseA = planner.get_grasp_pose_w_labeled_direction(actor=env.object, actor_data=env.annotation_data['shoe1'], pre_dis=0, id=0)
        # # grasp_poseB = planner.get_grasp_pose_w_labeled_direction(actor=env.bowl2, actor_data=env.annotation_data['bowl2'], pre_dis=0, id=1)
        # grasp_poseA[2] += 0.12
        # # grasp_poseB[2] += 0.12
        # planner.move_to_pose_with_screw([grasp_poseA], move_id=[0])
        # grasp_poseA[2] -= 0.14
        # # grasp_poseB[2] -= 0.12
        # planner.move_to_pose_with_screw([grasp_poseA], move_id=[0])
        # planner.close_gripper([0])
        # grasp_poseA[2] += 0.3
        # # grasp_poseB[2] += 0.3
        # planner.move_to_pose_with_screw([grasp_poseA], move_id=[0])

        # # Step 2: Stack the bowls
        # target_poseA = planner.get_grasp_pose_for_stack(grasp_poseA, env.shoebox)
        # target_poseA[0] -= 0.05
        # target_poseA[1] -= 0.05   
        # target_poseA[2] += 0.2
        # planner.move_to_pose_with_screw(target_poseA, move_id=0)
        # target_poseA[2] -= 0.15
        # planner.move_to_pose_with_screw(target_poseA, move_id=0)
        # planner.open_gripper([0])
        # target_poseA[2] += 0.2
        # planner.move_to_pose_with_screw(target_poseA, move_id=0)
        # planner.move_to_pose_with_screw(init_poseA, move_id=0)

        # grasp_poseB = planner.get_grasp_pose_w_labeled_direction(actor=env.shoe2, actor_data=env.annotation_data['shoe2'], pre_dis=0, id=0)
        # grasp_poseB[2] += 0.12
        # # grasp_poseB[2] += 0.12
        # planner.move_to_pose_with_screw([grasp_poseB], move_id=[1])
        # grasp_poseB[2] -= 0.14
        # # grasp_poseB[2] -= 0.12
        # planner.move_to_pose_with_screw([grasp_poseB], move_id=[1])
        # planner.close_gripper([1])
        # grasp_poseB[2] += 0.3
        # # grasp_poseB[2] += 0.3
        # planner.move_to_pose_with_screw([grasp_poseB], move_id=[1])

        # target_poseB = planner.get_grasp_pose_for_stack(grasp_poseB, env.shoebox)
        # target_poseB[0] -= 0.05
        # target_poseB[1] += 0.05 
        # target_poseB[2] += 0.2
        # planner.move_to_pose_with_screw(target_poseB, move_id=1)
        # target_poseB[2] -= 0.15
        # planner.move_to_pose_with_screw(target_poseB, move_id=1)
        # planner.open_gripper([1])
        # target_poseB[2] += 0.2
        # planner.move_to_pose_with_screw(target_poseB, move_id=1)
        # res = planner.move_to_pose_with_screw(init_poseB, move_id=1)
        # planner.close()
    return res

import numpy as np
import sapien

from tasks import TwoArmsPlaceObjectEnv
from planner.motionplanner import PandaArmMotionPlanningSolver
from transforms3d.euler import euler2quat
from transforms3d.quaternions import qmult

def solve(env: TwoArmsPlaceObjectEnv, seed=None, debug=False, vis=False, status=None):
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
    if status == "two_arms_place_object_12":
        # while 1:
        #     planner.open_gripper([0, 1])
        # (0.5, -0.6), (0.1, -0.6), (0.3, -0.8), (0.3, -0.4), (0.4, -0.7), (0.4, -0.5), (0.2, -0.7), (0.2, -0.5)
        # init_poseA = env.agent.agents[0].tcp.pose
        # init_poseB = env.agent.agents[1].tcp.pose
        # pose_A = init_poseA.p[0].cpu().numpy()
        # quat_A = init_poseA.q[0].cpu().numpy()
        # pose_A[0], pose_A[1] = 0.5, -0.6
        # pose_A = np.concatenate([pose_A, quat_A])
        # planner.move_to_pose_with_screw(init_poseA, move_id=0)
        while 1:
            planner.open_gripper()
        grasp_poseA = planner.get_grasp_pose_w_labeled_direction(actor=env.toycar, actor_data=env.annotation_data['toycar'], pre_dis=0, id=1)
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
        grasp_poseA[:2] = env.woodenbox1.pose.p[0, :2]
        grasp_poseA[0] += 0.02
        grasp_poseA[2] -= 0.2
        current_pos = grasp_poseA[:3].copy()  # 保存当前位置
        current_quat = grasp_poseA[3:7].copy()  # 保存当前旋转四元数
        # 创建绕z轴旋转90度的四元数
        rotation_quat = euler2quat(0, 0, np.pi/2, 'sxyz')  # 绕z轴旋转90度
        # 组合旋转：先应用当前旋转，再应用新旋转
        new_quat = qmult(current_quat, rotation_quat)
        # 更新pose1，保持位置不变，只改变旋转
        grasp_poseA[:3] = current_pos
        grasp_poseA[3:7] = new_quat
        planner.move_to_pose_with_screw(grasp_poseA, move_id=0)
        # grasp_poseA[2] -= 0.1
        # planner.move_to_pose_with_screw(grasp_poseA, move_id=0)
        planner.open_gripper([0])
        grasp_poseA[2] += 0.2
        planner.move_to_pose_with_screw(grasp_poseA, move_id=0)
        planner.move_to_pose_with_screw(init_poseA, move_id=0)

        grasp_poseB = planner.get_grasp_pose_w_labeled_direction(actor=env.hammer, actor_data=env.annotation_data['hammer'], pre_dis=0, id=0)
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

        grasp_poseB[:2] = env.woodenbox2.pose.p[0, :2]
        grasp_poseB[0] += 0.02
        grasp_poseB[2] -= 0.2
        current_pos = grasp_poseB[:3].copy()  # 保存当前位置
        current_quat = grasp_poseB[3:7].copy()  # 保存当前旋转四元数
        # 创建绕z轴旋转90度的四元数
        rotation_quat = euler2quat(0, 0, np.pi/2, 'sxyz')  # 绕z轴旋转90度
        # 组合旋转：先应用当前旋转，再应用新旋转
        new_quat = qmult(current_quat, rotation_quat)
        # 更新pose1，保持位置不变，只改变旋转
        grasp_poseB[:3] = current_pos
        grasp_poseB[3:7] = new_quat
        planner.move_to_pose_with_screw(grasp_poseB, move_id=1)
        # grasp_poseA[2] -= 0.1
        # planner.move_to_pose_with_screw(grasp_poseB, move_id=0)
        planner.open_gripper([1])
        grasp_poseB[2] += 0.2
        planner.move_to_pose_with_screw(grasp_poseB, move_id=1)
        res = planner.move_to_pose_with_screw(init_poseB, move_id=1)
        planner.close()
    return res

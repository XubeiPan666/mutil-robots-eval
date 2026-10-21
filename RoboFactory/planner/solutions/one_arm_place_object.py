import numpy as np
import sapien

from tasks import OneArmPlaceObjectEnv
from planner.motionplanner import PandaArmMotionPlanningSolver
from transforms3d.euler import euler2quat
from transforms3d.quaternions import qmult

def solve(env: OneArmPlaceObjectEnv, seed=None, debug=False, vis=False, status=None):
    env.reset(seed=seed)
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=False,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )
    env = env.unwrapped
    # pose = planner.get_grasp_pose_w_labeled_direction(actor=env.object, actor_data=env.annotation_data['object'], pre_dis=0, id=1)
    # planner.move_to_pose_with_screw(pose)
    # planner.close_gripper()
    # pose[2] += 0.2
    # res = planner.move_to_pose_with_screw(pose)
    # planner.close()
    # return res
    while 1:
        planner.open_gripper()
    init_poseA = env.agent.tcp.pose
    # init_poseB = env.agent.agents[1].tcp.pose
    grasp_poseA = planner.get_grasp_pose_w_labeled_direction(actor=env.object, actor_data=env.annotation_data['object'], pre_dis=0, id=1)
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
    grasp_poseA[:2] = env.woodenbox.pose.p[0, :2]
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
    res = planner.move_to_pose_with_screw(init_poseA, move_id=0)
    planner.close()
    return res


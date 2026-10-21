import numpy as np
import sapien


from tasks import TwoArmsPlaceCanBasketEnv
from planner.motionplanner import PandaArmMotionPlanningSolver
from transforms3d.euler import euler2quat
from transforms3d.quaternions import qmult

def solve(env: TwoArmsPlaceCanBasketEnv, seed=None, debug=False, vis=False, status=None):
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

    if status == "two_arms_place_can_basket":
        init_pose1 = env.agent.agents[0].tcp.pose
        pose1 = planner.get_grasp_pose_w_labeled_direction(actor=env.can, actor_data=env.annotation_data['can'], pre_dis=0, id=12)  #id=12
        pose1_copy = pose1.copy()
        pose2 = planner.get_grasp_pose_w_labeled_direction(actor=env.basket, actor_data=env.annotation_data['basket'], pre_dis=0, id=1)
        pose2[2] += 0.12
        # pose2[2] += 0.15
        planner.move_to_pose_with_screw([pose2], move_id=[1])
        pose2[2] -= 0.13
        # pose2[2] -= 0.15
        planner.move_to_pose_with_screw([pose2], move_id=[1])
        planner.close_gripper(close_id=[1])
        pose2[:2] = (env.agent.agents[0].robot.pose.p[0, :2] + env.agent.agents[1].robot.pose.p[0, :2]) / 2
        pose2[2] += 0.2
        pose2[0] += 0.5
        # pose1[1] += 0.2
        # pose2[2] += 0.15

        planner.move_to_pose_with_screw([pose2], move_id=[1])



        pose1[2] += 0.12
        planner.move_to_pose_with_screw([pose1], move_id=[0])
        pose1[2] -= 0.13
        planner.move_to_pose_with_screw([pose1], move_id=[0])
        planner.close_gripper(close_id=[0])
        pose1 = planner.get_grasp_pose_w_labeled_direction(actor=env.basket, actor_data=env.annotation_data['basket'], pre_dis=0, id=1)
        pose1[1] -= 0.3
        # pose1[0] += 0.1
        pose1[2] -= 0.03
        # pose1[2] += 0.01
        current_quat = pose1[3:7].copy()  # 保存当前旋转四元数
        # 创建绕z轴旋转90度的四元数
        rotation_quat = euler2quat(0, -np.pi/2, np.pi/2, 'sxyz')  # 绕z轴旋转90度
        # 组合旋转：先应用当前旋转，再应用新旋转
        new_quat = qmult(current_quat, rotation_quat)
        pose1[3:7] = new_quat

        planner.move_to_pose_with_screw([pose1], move_id=[0])

        # while 1:
        #     planner.close_gripper(close_id=[0, 1])
        pose1[1] += 0.2
        planner.move_to_pose_with_screw([pose1], move_id=[0])
        current_quat = pose1[3:7].copy()  # 保存当前旋转四元数
        # 创建绕z轴旋转90度的四元数
        rotation_quat = euler2quat(0, np.pi/8, 0, 'sxyz')  # 绕z轴旋转90度
        # 组合旋转：先应用当前旋转，再应用新旋转
        new_quat = qmult(current_quat, rotation_quat)
        pose1[3:7] = new_quat
        planner.move_to_pose_with_screw([pose1], move_id=[0])
        planner.open_gripper(open_id=[0])
        pose1[1] -= 0.3
        planner.move_to_pose_with_screw([pose1], move_id=[0])
        res = planner.move_to_pose_with_screw([init_pose1], move_id=[0])
        planner.close()

    return res

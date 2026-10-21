import numpy as np
import sapien


from tasks import TwoArmsPlaceFoodEnv
from planner.motionplanner import PandaArmMotionPlanningSolver
from transforms3d.euler import euler2quat
from transforms3d.quaternions import qmult

def solve(env: TwoArmsPlaceFoodEnv, seed=None, debug=False, vis=False, status=None):
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

    if status == "two_arms_place_food":
        pose1 = planner.get_grasp_pose_w_labeled_direction(actor=env.pot, actor_data=env.annotation_data['pot'], pre_dis=0, id=3)
        pose1_copy = pose1.copy()
        pose2 = planner.get_grasp_pose_w_labeled_direction(actor=env.meat, actor_data=env.annotation_data['meat'], pre_dis=0, id=0)
        pose1[2] += 0.2
        # pose2[2] += 0.15
        planner.move_to_pose_with_screw([pose1], move_id=[1])
        pose1[2] -= 0.22
        # pose2[2] -= 0.15
        planner.move_to_pose_with_screw([pose1], move_id=[1])
        planner.close_gripper(close_id=[1])
        pose1[2] += 0.2
        # pose1[1] += 0.2
        # pose2[2] += 0.15
        planner.move_to_pose_with_screw([pose1], move_id=[1])

        pose2[2] += 0.15
        planner.move_to_pose_with_screw([pose2], move_id=[0])
        pose2[2] -= 0.15
        planner.move_to_pose_with_screw([pose2], move_id=[0])
        planner.close_gripper(close_id=[0])
        pose2[2] += 0.15
        planner.move_to_pose_with_screw([pose2], move_id=[0])

        # pose3 = planner.get_target_pose_w_labeled_direction(actor=env.pot, actor_data=env.annotation_data['pot'], pre_dis=0)
        pose3 = pose1_copy
        # pose3[1] -= 0.2
        current_quat = pose3[3:7].copy()  # 保存当前旋转四元数
        # 创建绕z轴旋转90度的四元数
        rotation_quat = euler2quat(0, -np.pi/2, 0, 'sxyz')  # 绕z轴旋转90度
        # 组合旋转：先应用当前旋转，再应用新旋转
        new_quat = qmult(current_quat, rotation_quat)
        pose3[3:7] = new_quat
        
        planner.move_to_pose_with_screw(pose3, move_id=[0])
        # planner.open_gripper(open_id=[0])
        # planner.open_gripper(open_id=[0])
        res = planner.open_gripper(open_id=[0])
        planner.close()

    return res

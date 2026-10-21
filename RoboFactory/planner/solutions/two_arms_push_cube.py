import numpy as np
import sapien

from tasks import TwoArmsPushCubeEnv
from planner.motionplanner import PandaArmMotionPlanningSolver
from transforms3d.euler import euler2quat
from transforms3d.quaternions import qmult

def solve(env: TwoArmsPushCubeEnv, seed=101, debug=False, vis=False, status=None):
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
    
    if status == "two_arms_push_cube":
        # 关闭两个机械臂的夹爪
        init_poseA = env.agent.agents[0].tcp.pose
        init_poseB = env.agent.agents[1].tcp.pose

        planner.close_gripper([0, 1])
        
        
        push_poseA = planner.get_grasp_pose_from_obb(env.cubeA, 0)
        push_poseB = planner.get_grasp_pose_from_obb(env.cubeB, 1)
        push_poseA[1] -= 0.05
        push_poseA[2] += 0.12
        push_poseB[1] += 0.05
        push_poseB[2] += 0.12

        
        current_quat_A = push_poseA[3:7].copy()  # 保存当前旋转四元数
        current_quat_B = push_poseB[3:7].copy()  # 保存当前旋转四元数
        # 创建绕z轴旋转90度的四元数
        rotation_quat_A = euler2quat(0, 0, -np.pi/2, 'sxyz')  # 绕z轴旋转90度
        rotation_quat_B = euler2quat(0, 0, np.pi/2, 'sxyz')  # 绕z轴旋转90度
        # 组合旋转：先应用当前旋转，再应用新旋转
        new_quat_A = qmult(current_quat_A, rotation_quat_A)
        new_quat_B = qmult(current_quat_B, rotation_quat_B)
        push_poseA[3:7] = new_quat_A
        push_poseB[3:7] = new_quat_B

        planner.move_to_pose_with_screw([push_poseA], move_id=[0])
        push_poseA[2] -= 0.12
        planner.move_to_pose_with_screw([push_poseA], move_id=[0])

        # planner.move_to_pose_with_screw([push_poseA, push_poseB], move_id=[0, 1])
        # push_poseA[2] -= 0.12
        # push_poseB[2] -= 0.12
        # planner.move_to_pose_with_screw([push_poseA, push_poseB], move_id=[0, 1])

        

        # while 1:
        #     planner.close_gripper([0, 1])
        
        # Step 2: 两个机械臂推方块到目标位置
        target_poseA = push_poseA.copy()
        target_poseA[:2] = env.goal_region.pose.p[0, :2]

        planner.move_to_pose_with_screw([target_poseA], move_id=[0])
        planner.move_to_pose_with_screw([init_poseA], move_id=[0])

        planner.move_to_pose_with_screw([push_poseB], move_id=[1])
        push_poseB[2] -= 0.12
        planner.move_to_pose_with_screw([push_poseB], move_id=[1])
        target_poseB = push_poseB.copy()
        target_poseB[:2] = env.goal_region.pose.p[0, :2]
        planner.move_to_pose_with_screw([target_poseB], move_id=[1])
        res = planner.move_to_pose_with_screw([init_poseB], move_id=[1])

        # res = planner.move_to_pose_with_screw([target_poseA, target_poseB], move_id=[0, 1])
        
        planner.close()
    return res

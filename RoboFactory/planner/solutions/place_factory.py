import numpy as np
import sapien

from tasks import PlaceFactoryEnv
from planner.motionplanner import PandaArmMotionPlanningSolver

def solve(env: PlaceFactoryEnv, seed=None, debug=False, vis=False, status=None):
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
    env=env.unwrapped
    

    if status == "place_factory_bowl_123":
        # while 1:
        #     planner.open_gripper()
        init_pose1 = env.agent.agents[0].tcp.pose
        init_pose2 = env.agent.agents[1].tcp.pose
        init_pose3 = env.agent.agents[2].tcp.pose
        # Step 1: 获取抓取姿态
        pose1 = planner.get_grasp_pose_w_labeled_direction(actor=env.bowl1, actor_data=env.annotation_data['bowl1'], pre_dis=0, id=2)
        pose2 = planner.get_grasp_pose_w_labeled_direction(actor=env.bowl2, actor_data=env.annotation_data['bowl2'], pre_dis=0, id=1)
        pose3 = planner.get_grasp_pose_w_labeled_direction(actor=env.bowl3, actor_data=env.annotation_data['bowl3'], pre_dis=0, id=2)


        pose1[2] += 0.12
        pose2[2] += 0.12
        pose3[2] += 0.12
        planner.move_to_pose_with_screw([pose1, pose2, pose3], move_id=[0, 1, 2])
        pose1[2] -= 0.13
        pose2[2] -= 0.13
        pose3[2] -= 0.13
        planner.move_to_pose_with_screw([pose1, pose2, pose3], move_id=[0, 1, 2])
        planner.close_gripper(close_id=[0, 1, 2])
        pose1[2] += 0.2
        pose2[2] += 0.2
        pose3[2] += 0.2
        planner.move_to_pose_with_screw([pose1, pose2, pose3], move_id=[0, 1, 2])
        

        pose1[:2] = env.tray.pose.p[0, :2]
        pose1[0] -= 0.05
        planner.move_to_pose_with_screw([pose1], move_id=[0])
        pose1[2] -= 0.19
        planner.move_to_pose_with_screw([pose1], move_id=[0]) #time_step越小越慢
        planner.open_gripper(open_id=[0])
        pose1[2] += 0.1
        planner.move_to_pose_with_screw([pose1], move_id=[0])
        planner.move_to_pose_with_screw([init_pose1], move_id=[0])

        # pose2[2] += 0.12
        # planner.move_to_pose_with_screw([pose2], move_id=[1])
        # pose2[2] -= 0.13
        # planner.move_to_pose_with_screw([pose2], move_id=[1])
        # planner.close_gripper(close_id=[1])
        # pose2[2] += 0.2
        # planner.move_to_pose_with_screw([pose2], move_id=[1])
        pose2[:2] = env.bowl1.pose.p[0, :2]
        pose2[0] += 0.04
        # pose2[0] += 0.08
        planner.move_to_pose_with_screw([pose2], move_id=[1])
        pose2[2] -= 0.16
        planner.move_to_pose_with_screw([pose2], move_id=[1])
        planner.open_gripper(open_id=[1])
        pose2[2] += 0.1
        planner.move_to_pose_with_screw([pose2], move_id=[1])
        planner.move_to_pose_with_screw([init_pose2], move_id=[1])

        # pose3[2] += 0.12
        # planner.move_to_pose_with_screw([pose3], move_id=[2])
        # pose3[2] -= 0.13
        # planner.move_to_pose_with_screw([pose3], move_id=[2])
        # planner.close_gripper(close_id=[2])
        # pose3[2] += 0.2
        # planner.move_to_pose_with_screw([pose3], move_id=[2])
        pose3[:2] = env.bowl2.pose.p[0, :2]
        pose3[0] -= 0.04
        # pose3[0] -= 0.08
        planner.move_to_pose_with_screw([pose3], move_id=[2])
        pose3[2] -= 0.13
        planner.move_to_pose_with_screw([pose3], move_id=[2])
        planner.open_gripper(open_id=[2])
        pose3[2] += 0.1
        planner.move_to_pose_with_screw([pose3], move_id=[2])
        res = planner.move_to_pose_with_screw([init_pose3], move_id=[2])
    
    elif status == "place_factory_bowl_132_123":
        # while 1:
        #     planner.open_gripper()
        init_pose1 = env.agent.agents[0].tcp.pose
        init_pose2 = env.agent.agents[1].tcp.pose
        init_pose3 = env.agent.agents[2].tcp.pose
        # Step 1: 获取抓取姿态
        pose1 = planner.get_grasp_pose_w_labeled_direction(actor=env.bowl1, actor_data=env.annotation_data['bowl1'], pre_dis=0, id=2)
        pose2 = planner.get_grasp_pose_w_labeled_direction(actor=env.bowl2, actor_data=env.annotation_data['bowl2'], pre_dis=0, id=2)
        pose3 = planner.get_grasp_pose_w_labeled_direction(actor=env.bowl3, actor_data=env.annotation_data['bowl3'], pre_dis=0, id=1)


        pose1[2] += 0.12
        pose2[2] += 0.12
        pose3[2] += 0.12
        planner.move_to_pose_with_screw([pose1, pose2, pose3], move_id=[0, 2, 1])
        pose1[2] -= 0.13
        pose2[2] -= 0.13
        pose3[2] -= 0.13
        planner.move_to_pose_with_screw([pose1, pose2, pose3], move_id=[0, 2, 1])
        planner.close_gripper(close_id=[0, 1, 2])
        pose1[2] += 0.2
        pose2[2] += 0.2
        pose3[2] += 0.2
        planner.move_to_pose_with_screw([pose1, pose2, pose3], move_id=[0, 2, 1])
        planner.move_to_pose_with_screw([init_pose1, init_pose2, init_pose3], move_id=[0, 1, 2])
        

        pose1[:2] = env.tray.pose.p[0, :2]
        pose1[0] -= 0.05
        planner.move_to_pose_with_screw([pose1], move_id=[0])
        pose1[2] -= 0.19
        planner.move_to_pose_with_screw([pose1], move_id=[0]) #time_step越小越慢
        planner.open_gripper(open_id=[0])
        pose1[2] += 0.1
        planner.move_to_pose_with_screw([pose1], move_id=[0])
        planner.move_to_pose_with_screw([init_pose1], move_id=[0])

        # pose2[2] += 0.12
        # planner.move_to_pose_with_screw([pose2], move_id=[1])
        # pose2[2] -= 0.13
        # planner.move_to_pose_with_screw([pose2], move_id=[1])
        # planner.close_gripper(close_id=[1])
        # pose2[2] += 0.2
        # planner.move_to_pose_with_screw([pose2], move_id=[1])
        # pose2[:2] = env.bowl1.pose.p[0, :2]
        # pose2[0] += 0.04
        # # pose2[0] += 0.08
        # planner.move_to_pose_with_screw([pose2], move_id=[1])
        # pose2[2] -= 0.16
        # planner.move_to_pose_with_screw([pose2], move_id=[1])
        # planner.open_gripper(open_id=[1])
        # pose2[2] += 0.1
        # planner.move_to_pose_with_screw([pose2], move_id=[1])
        # planner.move_to_pose_with_screw([init_pose2], move_id=[1])

        # pose3[2] += 0.12
        # planner.move_to_pose_with_screw([pose3], move_id=[2])
        # pose3[2] -= 0.13
        # planner.move_to_pose_with_screw([pose3], move_id=[2])
        # planner.close_gripper(close_id=[2])
        # pose3[2] += 0.2
        # planner.move_to_pose_with_screw([pose3], move_id=[2])
        pose3[:2] = env.bowl1.pose.p[0, :2]
        pose3[0] += 0.04
        # pose3[0] -= 0.08
        planner.move_to_pose_with_screw([pose3], move_id=[1])
        pose3[2] -= 0.13
        planner.move_to_pose_with_screw([pose3], move_id=[1])
        planner.open_gripper(open_id=[1])
        pose3[2] += 0.1
        planner.move_to_pose_with_screw([pose3], move_id=[1])
        planner.move_to_pose_with_screw([init_pose2], move_id=[1])

        pose2[:2] = env.bowl3.pose.p[0, :2]
        pose2[0] -= 0.04
        # pose2[0] += 0.08
        planner.move_to_pose_with_screw([pose2], move_id=[2])
        pose2[2] -= 0.16
        planner.move_to_pose_with_screw([pose2], move_id=[2])
        planner.open_gripper(open_id=[2])
        pose2[2] += 0.1
        planner.move_to_pose_with_screw([pose2], move_id=[2])
        res = planner.move_to_pose_with_screw([init_pose3], move_id=[2])
    
    elif status == "place_factory_bowl_123_2bd":
        
        init_pose1 = env.agent.agents[0].tcp.pose
        init_pose2 = env.agent.agents[1].tcp.pose
        init_pose3 = env.agent.agents[2].tcp.pose
        # Step 1: 获取抓取姿态
        pose1 = planner.get_grasp_pose_w_labeled_direction(actor=env.bowl1, actor_data=env.annotation_data['bowl1'], pre_dis=0, id=2)
        pose2 = planner.get_grasp_pose_w_labeled_direction(actor=env.bowl2, actor_data=env.annotation_data['bowl2'], pre_dis=0, id=2)
        pose3 = planner.get_grasp_pose_w_labeled_direction(actor=env.bowl3, actor_data=env.annotation_data['bowl3'], pre_dis=0, id=2)


        pose1[2] += 0.12
        pose2[2] += 0.12
        # pose3[2] += 0.12
        planner.move_to_pose_with_screw([pose1, pose2], move_id=[0, 2])
        pose1[2] -= 0.13
        pose2[2] -= 0.13
        # pose3[2] -= 0.13
        planner.move_to_pose_with_screw([pose1, pose2], move_id=[0, 2])
        planner.close_gripper(close_id=[0, 2])
        pose1[2] += 0.2
        pose2[2] += 0.2
        # pose3[2] += 0.2
        planner.move_to_pose_with_screw([pose1, pose2], move_id=[0, 2])
        

        pose1[:2] = env.tray.pose.p[0, :2]
        pose1[0] -= 0.05
        planner.move_to_pose_with_screw([pose1], move_id=[0])
        pose1[2] -= 0.19
        planner.move_to_pose_with_screw([pose1], move_id=[0]) #time_step越小越慢
        planner.open_gripper(open_id=[0])
        pose1[2] += 0.1
        planner.move_to_pose_with_screw([pose1], move_id=[0])
        planner.move_to_pose_with_screw([init_pose1], move_id=[0])

        # pose2[2] += 0.12
        # planner.move_to_pose_with_screw([pose2], move_id=[1])
        # pose2[2] -= 0.13
        # planner.move_to_pose_with_screw([pose2], move_id=[1])
        # planner.close_gripper(close_id=[1])
        # pose2[2] += 0.2
        # planner.move_to_pose_with_screw([pose2], move_id=[1])
        pose2[:2] = env.bowl1.pose.p[0, :2]
        pose2[0] -= 0.04
        # pose2[0] += 0.08
        planner.move_to_pose_with_screw([pose2], move_id=[2])
        pose2[2] -= 0.16
        planner.move_to_pose_with_screw([pose2], move_id=[2])
        planner.open_gripper(open_id=[2])
        pose2[2] += 0.1
        planner.move_to_pose_with_screw([pose2], move_id=[2])
        planner.move_to_pose_with_screw([init_pose3], move_id=[2])

        pose3[2] += 0.12
        planner.move_to_pose_with_screw([pose3], move_id=[2])
        pose3[2] -= 0.13
        planner.move_to_pose_with_screw([pose3], move_id=[2])
        planner.close_gripper(close_id=[2])
        pose3[2] += 0.2
        planner.move_to_pose_with_screw([pose3], move_id=[2])

        pose3[:2] = env.bowl2.pose.p[0, :2]
        pose3[0] -= 0.04
        # pose3[0] -= 0.08
        planner.move_to_pose_with_screw([pose3], move_id=[2])
        pose3[2] -= 0.13
        planner.move_to_pose_with_screw([pose3], move_id=[2])
        planner.open_gripper(open_id=[2])
        pose3[2] += 0.1
        planner.move_to_pose_with_screw([pose3], move_id=[2])
        res = planner.move_to_pose_with_screw([init_pose3], move_id=[2])
    
    elif status == "place_factory_cube_123_123":
        # while 1:
        #     planner.open_gripper()
        init_pose1 = env.agent.agents[0].tcp.pose
        init_pose2 = env.agent.agents[1].tcp.pose
        init_pose3 = env.agent.agents[2].tcp.pose
        # Step 1: 获取抓取姿态
        pose1 = planner.get_grasp_pose_from_obb(env.cube1, 0)
        pose2 = planner.get_grasp_pose_from_obb(env.cube2, 1)
        pose3 = planner.get_grasp_pose_from_obb(env.cube3, 2)


        pose1[2] += 0.12
        pose2[2] += 0.12
        pose3[2] += 0.12
        planner.move_to_pose_with_screw([pose1, pose2, pose3], move_id=[0, 1, 2])
        pose1[2] -= 0.14
        pose2[2] -= 0.14
        pose3[2] -= 0.14
        planner.move_to_pose_with_screw([pose1, pose2, pose3], move_id=[0, 1, 2])
        planner.close_gripper(close_id=[0, 1, 2])
        pose1[2] += 0.2
        pose2[2] += 0.2
        pose2[0] += 0.2
        pose3[2] += 0.2
        planner.move_to_pose_with_screw([pose1, pose2, pose3], move_id=[0, 1, 2])
        

        pose1[:2] = env.goal_region.pose.p[0, :2]
        # pose1[0] -= 0.05
        planner.move_to_pose_with_screw([pose1], move_id=[0])
        pose1[2] -= 0.18
        planner.move_to_pose_with_screw([pose1], move_id=[0]) #time_step越小越慢
        planner.open_gripper(open_id=[0])
        pose1[2] += 0.1
        planner.move_to_pose_with_screw([pose1], move_id=[0])
        planner.move_to_pose_with_screw([init_pose1], move_id=[0])

        # pose2[2] += 0.12
        # planner.move_to_pose_with_screw([pose2], move_id=[1])
        # pose2[2] -= 0.13
        # planner.move_to_pose_with_screw([pose2], move_id=[1])
        # planner.close_gripper(close_id=[1])
        # pose2[2] += 0.2
        # planner.move_to_pose_with_screw([pose2], move_id=[1])
        pose2[:2] = env.cube1.pose.p[0, :2]
        # pose2[0] += 0.04
        # pose2[0] += 0.08
        planner.move_to_pose_with_screw([pose2], move_id=[1])
        pose2[2] -= 0.14
        planner.move_to_pose_with_screw([pose2], move_id=[1])
        planner.open_gripper(open_id=[1])
        pose2[2] += 0.1
        planner.move_to_pose_with_screw([pose2], move_id=[1])
        planner.move_to_pose_with_screw([init_pose2], move_id=[1])

        # pose3[2] += 0.12
        # planner.move_to_pose_with_screw([pose3], move_id=[2])
        # pose3[2] -= 0.13
        # planner.move_to_pose_with_screw([pose3], move_id=[2])
        # planner.close_gripper(close_id=[2])
        # pose3[2] += 0.2
        # planner.move_to_pose_with_screw([pose3], move_id=[2])
        pose3[:2] = env.cube2.pose.p[0, :2]
        # pose3[0] -= 0.04
        # pose3[0] -= 0.08
        planner.move_to_pose_with_screw([pose3], move_id=[2])
        pose3[2] -= 0.1
        planner.move_to_pose_with_screw([pose3], move_id=[2])
        planner.open_gripper(open_id=[2])
        pose3[2] += 0.1
        planner.move_to_pose_with_screw([pose3], move_id=[2])
        res = planner.move_to_pose_with_screw([init_pose3], move_id=[2])
    
    elif status == "place_factory_cube_123_132":
        # while 1:
        #     planner.open_gripper()
        init_pose1 = env.agent.agents[0].tcp.pose
        init_pose2 = env.agent.agents[1].tcp.pose
        init_pose3 = env.agent.agents[2].tcp.pose
        # Step 1: 获取抓取姿态
        pose1 = planner.get_grasp_pose_from_obb(env.cube1, 0)
        pose2 = planner.get_grasp_pose_from_obb(env.cube2, 1)
        pose3 = planner.get_grasp_pose_from_obb(env.cube3, 2)


        pose1[2] += 0.12
        pose3[2] += 0.12
        # pose3[2] += 0.12
        planner.move_to_pose_with_screw([pose1, pose3], move_id=[0, 1])
        pose1[2] -= 0.14
        pose3[2] -= 0.14
        # pose3[2] -= 0.14
        planner.move_to_pose_with_screw([pose1, pose3], move_id=[0, 1])
        planner.close_gripper(close_id=[0, 1])
        pose1[2] += 0.2
        pose3[2] += 0.2
        pose3[0] += 0.2
        pose3[1] -= 0.2
        # pose3[2] += 0.2
        planner.move_to_pose_with_screw([pose1, pose3], move_id=[0, 1])

        pose1[:2] = env.goal_region.pose.p[0, :2]
        # pose1[0] -= 0.05
        planner.move_to_pose_with_screw([pose1], move_id=[0])
        pose1[2] -= 0.18
        planner.move_to_pose_with_screw([pose1], move_id=[0]) #time_step越小越慢
        planner.open_gripper(open_id=[0])
        pose1[2] += 0.1
        planner.move_to_pose_with_screw([pose1], move_id=[0])
        planner.move_to_pose_with_screw([init_pose1], move_id=[0])

        pose2[2] += 0.12
        planner.move_to_pose_with_screw([pose2], move_id=[2])
        pose2[2] -= 0.14
        planner.move_to_pose_with_screw([pose2], move_id=[2])
        planner.close_gripper(close_id=[2])
        pose2[2] += 0.2
        planner.move_to_pose_with_screw([pose2], move_id=[2])
        pose2[:2] = env.cube1.pose.p[0, :2]
        # pose2[0] += 0.04
        # pose2[0] += 0.08
        planner.move_to_pose_with_screw([pose2], move_id=[2])
        pose2[2] -= 0.14
        planner.move_to_pose_with_screw([pose2], move_id=[2])
        planner.open_gripper(open_id=[2])
        pose2[2] += 0.1
        planner.move_to_pose_with_screw([pose2], move_id=[2])
        planner.move_to_pose_with_screw([init_pose3], move_id=[2])

        # pose3[2] += 0.12
        # planner.move_to_pose_with_screw([pose3], move_id=[1])
        # pose3[2] -= 0.14
        # planner.move_to_pose_with_screw([pose3], move_id=[1])
        # planner.close_gripper(close_id=[1])
        # pose3[2] += 0.2
        # pose3[0] += 0.1
        # pose3[1] -= 0.1
        # planner.move_to_pose_with_screw([pose3], move_id=[1])
        pose3[:2] = env.cube2.pose.p[0, :2]
        # pose3[0] -= 0.04
        # pose3[0] -= 0.08
        planner.move_to_pose_with_screw([pose3], move_id=[1])
        pose3[2] -= 0.1
        planner.move_to_pose_with_screw([pose3], move_id=[1])
        planner.open_gripper(open_id=[1])
        pose3[2] += 0.1
        planner.move_to_pose_with_screw([pose3], move_id=[1])
        res = planner.move_to_pose_with_screw([init_pose2], move_id=[1])
    
    elif status == "place_factory_cube_123_213":

        # while 1:
        #     planner.open_gripper()
        init_pose1 = env.agent.agents[0].tcp.pose
        init_pose2 = env.agent.agents[1].tcp.pose
        init_pose3 = env.agent.agents[2].tcp.pose
        # Step 1: 获取抓取姿态
        pose1 = planner.get_grasp_pose_from_obb(env.cube1, 0)
        pose2 = planner.get_grasp_pose_from_obb(env.cube2, 1)
        pose3 = planner.get_grasp_pose_from_obb(env.cube3, 2)


        pose1[2] += 0.12
        pose3[2] += 0.12
        # pose3[2] += 0.12
        planner.move_to_pose_with_screw([pose1, pose3], move_id=[1, 2])
        pose1[2] -= 0.14
        pose3[2] -= 0.14
        # pose3[2] -= 0.14
        planner.move_to_pose_with_screw([pose1, pose3], move_id=[1, 2])
        planner.close_gripper(close_id=[1, 2])
        pose1[2] += 0.2
        pose1[0] += 0.2
        pose1[1] += 0.2
        pose3[2] += 0.2 
        # pose3[2] += 0.2
        planner.move_to_pose_with_screw([pose1, pose3], move_id=[1, 2])

        pose1[:2] = env.goal_region.pose.p[0, :2]
        # pose1[0] -= 0.05
        planner.move_to_pose_with_screw([pose1], move_id=[1])
        pose1[2] -= 0.18
        planner.move_to_pose_with_screw([pose1], move_id=[1]) #time_step越小越慢
        planner.open_gripper(open_id=[1])
        pose1[2] += 0.1
        planner.move_to_pose_with_screw([pose1], move_id=[1])
        planner.move_to_pose_with_screw([init_pose2], move_id=[1])

        pose2[2] += 0.12
        planner.move_to_pose_with_screw([pose2], move_id=[0])
        pose2[2] -= 0.14
        planner.move_to_pose_with_screw([pose2], move_id=[0])
        planner.close_gripper(close_id=[0])
        pose2[2] += 0.2
        # pose2[0] -= 0.1 
        planner.move_to_pose_with_screw([pose2], move_id=[0])
        pose2[:2] = env.cube1.pose.p[0, :2]
        # pose2[0] += 0.04
        # pose2[0] += 0.08
        planner.move_to_pose_with_screw([pose2], move_id=[0])
        pose2[2] -= 0.14
        planner.move_to_pose_with_screw([pose2], move_id=[0])
        planner.open_gripper(open_id=[0])
        pose2[2] += 0.1
        planner.move_to_pose_with_screw([pose2], move_id=[0])
        planner.move_to_pose_with_screw([init_pose1], move_id=[0])

        # pose3[2] += 0.12
        # planner.move_to_pose_with_screw([pose3], move_id=[2])
        # pose3[2] -= 0.14
        # planner.move_to_pose_with_screw([pose3], move_id=[2])
        # planner.close_gripper(close_id=[2])
        # pose3[2] += 0.2
        # planner.move_to_pose_with_screw([pose3], move_id=[2])
        pose3[:2] = env.cube2.pose.p[0, :2]
        # pose3[0] -= 0.04
        # pose3[0] -= 0.08
        planner.move_to_pose_with_screw([pose3], move_id=[2])
        pose3[2] -= 0.1
        planner.move_to_pose_with_screw([pose3], move_id=[2])
        planner.open_gripper(open_id=[2])
        pose3[2] += 0.1
        planner.move_to_pose_with_screw([pose3], move_id=[2])
        res = planner.move_to_pose_with_screw([init_pose3], move_id=[2])



    
    elif status == "place_factory_cup_123":
        # while 1:
        #     planner.open_gripper()
        init_pose1 = env.agent.agents[0].tcp.pose
        init_pose2 = env.agent.agents[1].tcp.pose
        init_pose3 = env.agent.agents[2].tcp.pose
        # Step 1: 获取抓取姿态
        pose1 = planner.get_grasp_pose_w_labeled_direction(actor=env.cup1, actor_data=env.annotation_data['cup1'], pre_dis=0, id=2)
        pose2 = planner.get_grasp_pose_w_labeled_direction(actor=env.cup2, actor_data=env.annotation_data['cup2'], pre_dis=0, id=1)
        pose3 = planner.get_grasp_pose_w_labeled_direction(actor=env.cup3, actor_data=env.annotation_data['cup3'], pre_dis=0, id=2)


        pose1[2] += 0.12
        pose2[2] += 0.12
        pose3[2] += 0.12
        planner.move_to_pose_with_screw([pose1, pose2, pose3], move_id=[0, 1, 2])
        pose1[2] -= 0.13
        pose2[2] -= 0.13
        pose3[2] -= 0.13
        planner.move_to_pose_with_screw([pose1, pose2, pose3], move_id=[0, 1, 2])
        planner.close_gripper(close_id=[0, 1, 2])
        pose1[2] += 0.2
        pose2[2] += 0.2
        pose3[2] += 0.2
        planner.move_to_pose_with_screw([pose1, pose2, pose3], move_id=[0, 1, 2])
        

        pose1[:2] = env.tray.pose.p[0, :2]
        pose1[0] -= 0.05
        planner.move_to_pose_with_screw([pose1], move_id=[0])
        pose1[2] -= 0.19
        planner.move_to_pose_with_screw([pose1], move_id=[0]) #time_step越小越慢
        planner.open_gripper(open_id=[0])
        pose1[2] += 0.1
        planner.move_to_pose_with_screw([pose1], move_id=[0])
        planner.move_to_pose_with_screw([init_pose1], move_id=[0])

        # pose2[2] += 0.12
        # planner.move_to_pose_with_screw([pose2], move_id=[1])
        # pose2[2] -= 0.13
        # planner.move_to_pose_with_screw([pose2], move_id=[1])
        # planner.close_gripper(close_id=[1])
        # pose2[2] += 0.2
        # planner.move_to_pose_with_screw([pose2], move_id=[1])
        pose2[:2] = env.cup1.pose.p[0, :2]
        pose2[0] += 0.04
        # pose2[0] += 0.08
        planner.move_to_pose_with_screw([pose2], move_id=[1])
        pose2[2] -= 0.16
        planner.move_to_pose_with_screw([pose2], move_id=[1])
        planner.open_gripper(open_id=[1])
        pose2[2] += 0.1
        planner.move_to_pose_with_screw([pose2], move_id=[1])
        planner.move_to_pose_with_screw([init_pose2], move_id=[1])

        # pose3[2] += 0.12
        # planner.move_to_pose_with_screw([pose3], move_id=[2])
        # pose3[2] -= 0.13
        # planner.move_to_pose_with_screw([pose3], move_id=[2])
        # planner.close_gripper(close_id=[2])
        # pose3[2] += 0.2
        # planner.move_to_pose_with_screw([pose3], move_id=[2])
        pose3[:2] = env.cup2.pose.p[0, :2]
        pose3[0] -= 0.04
        # pose3[0] -= 0.08
        planner.move_to_pose_with_screw([pose3], move_id=[2])
        pose3[2] -= 0.13
        planner.move_to_pose_with_screw([pose3], move_id=[2])
        planner.open_gripper(open_id=[2])
        pose3[2] += 0.1
        planner.move_to_pose_with_screw([pose3], move_id=[2])
        res = planner.move_to_pose_with_screw([init_pose3], move_id=[2])

    elif status in ("place_factory_object_123", "place_factory_object_132", "place_factory_object_213"):
        # Sequential pick-and-place: each agent picks its object and places it into the tray
        # Object mapping: agent0 -> can1, agent1 -> teabox2, agent2 -> oilbottle3
        init_pose0 = env.agent.agents[0].tcp.pose
        init_pose1 = env.agent.agents[1].tcp.pose
        init_pose2 = env.agent.agents[2].tcp.pose

        # Define the order based on status
        # Digits encode which object each agent picks: "132" = agent0→obj1, agent1→obj3, agent2→obj2
        # Execution order is always agent0 → agent1 → agent2
        # Tray placement positions are FIXED per object (same across all statuses)
        obj_tray_pos = {
            'can1':     (0.025, -0.085),   # can1 always goes to left edge
            'teabox2':  (-0.025, 0.0),     # teabox2 always goes to center
            'hamburg3': (0.025, 0.085),    # hamburg3 always goes to right edge
        }
        if status == "place_factory_object_123":
            # agent0→can1(obj1), agent1→teabox2(obj2), agent2→hamburg3(obj3)
            order = [
                (0, env.can1, init_pose0, 'can1'),
                (1, env.teabox2, init_pose1, 'teabox2'),
                (2, env.hamburg3, init_pose2, 'hamburg3'),
            ]
        elif status == "place_factory_object_132":
            # agent0→can1(obj1), agent1→hamburg3(obj3), agent2→teabox2(obj2)
            order = [
                (0, env.can1, init_pose0, 'can1'),
                (1, env.hamburg3, init_pose1, 'hamburg3'),
                (2, env.teabox2, init_pose2, 'teabox2'),
            ]
        elif status == "place_factory_object_213":
            # agent0→teabox2(obj2), agent1→can1(obj1), agent2→hamburg3(obj3)
            order = [
                (0, env.teabox2, init_pose0, 'teabox2'),
                (1, env.can1, init_pose1, 'can1'),
                (2, env.hamburg3, init_pose2, 'hamburg3'),
            ]

        # Execute sequentially: each agent picks up its object and places it in the tray
        for agent_id, obj_actor, init_pose, obj_name in order:
            tray_x_offset, tray_y_offset = obj_tray_pos[obj_name]
            # Get grasp pose using OBB (auto-searches for reachable approach direction)
            grasp_pose = planner.get_grasp_pose_from_obb(obj_actor, agent_id)
            # Step 1: Move above the object
            grasp_pose[2] += 0.12
            planner.move_to_pose_with_screw([grasp_pose], move_id=[agent_id])
            # Step 2: Move down to grasp
            grasp_pose[2] -= 0.14
            planner.move_to_pose_with_screw([grasp_pose], move_id=[agent_id])
            # Step 3: Close gripper
            planner.close_gripper(close_id=[agent_id])
            # Step 4: Lift up
            grasp_pose[2] += 0.2
            planner.move_to_pose_with_screw([grasp_pose], move_id=[agent_id])
            # Step 5: Move to above tray (with x/y offset, zigzag to avoid collision)
            grasp_pose[:2] = env.tray.pose.p[0, :2]
            grasp_pose[0] += tray_x_offset
            grasp_pose[1] += tray_y_offset
            planner.move_to_pose_with_screw([grasp_pose], move_id=[agent_id])
            # Step 6: Lower into tray
            grasp_pose[2] -= 0.18
            planner.move_to_pose_with_screw([grasp_pose], move_id=[agent_id])
            # Step 7: Release object
            planner.open_gripper(open_id=[agent_id])
            # Step 8: Retract upward
            grasp_pose[2] += 0.1
            planner.move_to_pose_with_screw([grasp_pose], move_id=[agent_id])
            # Step 9: Return to initial pose
            res = planner.move_to_pose_with_screw([init_pose], move_id=[agent_id])


    planner.close()
    return res


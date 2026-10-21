import pinocchio  #pip install pin
import numpy as np
from os.path import join, dirname, abspath
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
import h5py
import os
import sys

def get_tcp_pose(model, data, tcp_frame_name='panda_hand_tcp'):
    """
    获取TCP（末端执行器）在世界坐标系中的位姿
    
    Args:
        model: pinocchio模型
        data: pinocchio数据
        tcp_frame_name: TCP帧的名称，默认为'panda_hand_tcp'
    
    Returns:
        T_world_tcp: 4x4齐次变换矩阵，从世界坐标系到TCP坐标系
    """
    # 查找TCP帧的索引
    try:
        tcp_frame_id = model.getFrameId(tcp_frame_name)
    except:
        # 如果找不到指定的帧，尝试使用最后一个关节
        print(f"Warning: Frame '{tcp_frame_name}' not found, using last frame")
        tcp_frame_id = len(model.frames) - 1
    
    # 获取TCP在世界坐标系中的位姿
    oMf = pinocchio.updateFramePlacement(model, data, tcp_frame_id)
    
    # 转换为4x4齐次变换矩阵
    T_world_tcp = oMf.homogeneous
    
    return T_world_tcp, tcp_frame_id

def create_transformation_matrix(translation, rotation=None, quat_format='wxyz'):
    """
    创建齐次变换矩阵
    
    Args:
        translation: 3x1平移向量 [x, y, z]
        rotation: 旋转，可以是以下格式之一：
            - 3x3旋转矩阵
            - 四元数（格式由 quat_format 指定）
            - 欧拉角 [roll, pitch, yaw] (弧度)
            - None: 使用单位旋转矩阵
        quat_format: 四元数格式，'wxyz' (SAPIEN/MuJoCo格式) 或 'xyzw' (scipy/ROS格式)
                     默认 'wxyz' 与 SAPIEN 仿真环境一致
    
    Returns:
        T: 4x4齐次变换矩阵
    """
    T = np.eye(4)
    T[:3, 3] = np.array(translation).flatten()
    
    if rotation is not None:
        rotation = np.array(rotation)
        if rotation.shape == (3, 3):
            # 3x3旋转矩阵
            T[:3, :3] = rotation
        elif rotation.shape == (4,):
            # 四元数处理
            # scipy.spatial.transform.Rotation.from_quat 需要 [x, y, z, w] 格式
            if quat_format == 'wxyz':
                # 输入是 SAPIEN 格式 [w, x, y, z]，转换为 scipy 格式 [x, y, z, w]
                quat_scipy = rotation[[1, 2, 3, 0]]
            else:  # 'xyzw'
                # 输入已经是 scipy 格式 [x, y, z, w]
                quat_scipy = rotation
            r = R.from_quat(quat_scipy)
            T[:3, :3] = r.as_matrix()
        elif rotation.shape == (3,):
            # 欧拉角 [roll, pitch, yaw]
            r = R.from_euler('xyz', rotation)
            T[:3, :3] = r.as_matrix()
    else:
        T[:3, :3] = np.eye(3)
    
    return T

def compute_camera_extrinsics(T_world_tcp, T_tcp_camera):
    """
    计算相机的外参矩阵（相机坐标系到世界坐标系的变换）
    
    Args:
        T_world_tcp: 4x4矩阵，世界坐标系到TCP坐标系的变换
        T_tcp_camera: 4x4矩阵，TCP坐标系到相机坐标系的变换
    
    Returns:
        T_world_camera: 4x4矩阵，世界坐标系到相机坐标系的变换（外参）
        T_camera_world: 4x4矩阵，相机坐标系到世界坐标系的变换（通常用于外参表示）
    """
    # 世界坐标系到相机坐标系
    T_world_camera = T_world_tcp @ T_tcp_camera
    
    # 相机坐标系到世界坐标系（这是通常说的外参）
    T_camera_world = np.linalg.inv(T_world_camera)
    
    return T_world_camera, T_camera_world

def sapien_pose_to_cam2world_gl(T_world_camera_sapien):
    """
    将 SAPIEN 相机位姿转换为 OpenGL 格式的 cam2world 矩阵
    
    SAPIEN 内部的 camera.get_model_matrix() 会应用一个坐标系变换：
    - SAPIEN link frame 和 OpenGL camera frame 有不同的轴定义
    
    从分析得到的变换矩阵:
    R_link_to_cam = R_pose^T @ R_gl
                 = [[0, 0, -1],
                    [-1, 0, 0],
                    [0, 1, 0]]
    
    Args:
        T_world_camera_sapien: SAPIEN 格式的相机位姿 (4x4)
        
    Returns:
        cam2world_gl: OpenGL 格式的 cam2world 矩阵 (4x4)
    """
    # SAPIEN link frame 到 OpenGL camera frame 的变换
    R_link_to_cam = np.array([
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0]
    ])
    
    T_link_to_cam = np.eye(4)
    T_link_to_cam[:3, :3] = R_link_to_cam
    
    # cam2world_gl = T_world_camera_sapien @ T_link_to_cam
    cam2world_gl = T_world_camera_sapien @ T_link_to_cam
    
    return cam2world_gl

def cam2world_gl_to_extrinsic_cv(cam2world_gl):
    """
    将 OpenGL 格式的 cam2world 矩阵转换为 OpenCV 格式的外参矩阵
    
    OpenGL 相机坐标系: X-右, Y-上, Z-后 (光轴 -Z)
    OpenCV 相机坐标系: X-右, Y-下, Z-前 (光轴 +Z)
    
    转换矩阵: R_gl_to_cv = [[1,0,0], [0,-1,0], [0,0,-1]]
    
    Args:
        cam2world_gl: OpenGL 格式的 cam2world 矩阵 (4x4)
        
    Returns:
        extrinsic_cv: OpenCV 格式的外参矩阵 (3x4 或 4x4)
    """
    # OpenGL 到 OpenCV 的坐标系转换（绕X轴旋转180度）
    R_gl_to_cv = np.eye(4)
    R_gl_to_cv[1, 1] = -1
    R_gl_to_cv[2, 2] = -1
    
    # world2cam_gl = inv(cam2world_gl)
    world2cam_gl = np.linalg.inv(cam2world_gl)
    
    # extrinsic_cv = R_gl_to_cv @ world2cam_gl
    extrinsic_cv = R_gl_to_cv @ world2cam_gl
    
    return extrinsic_cv

def compute_full_camera_params(T_world_tcp, T_tcp_camera):
    """
    计算完整的相机参数，与 SAPIEN camera.get_params() 输出格式一致
    
    Args:
        T_world_tcp: 世界到TCP的变换矩阵 (4x4)
        T_tcp_camera: TCP到相机的变换矩阵 (4x4)
        
    Returns:
        dict: 包含 cam2world_gl 和 extrinsic_cv
    """
    # 1. 计算 SAPIEN 格式的相机位姿
    T_world_camera_sapien = T_world_tcp @ T_tcp_camera
    
    # 2. 转换为 OpenGL 格式的 cam2world
    cam2world_gl = sapien_pose_to_cam2world_gl(T_world_camera_sapien)
    
    # 3. 计算 OpenCV 格式的外参
    extrinsic_cv = cam2world_gl_to_extrinsic_cv(cam2world_gl)
    
    return {
        'cam2world_gl': cam2world_gl,
        'extrinsic_cv': extrinsic_cv[:3, :],  # 返回 3x4 格式
        'T_world_camera_sapien': T_world_camera_sapien  # SAPIEN pose 格式
    }

def extrinsic_cv_to_cam2world_gl(extrinsic_cv):
    """
    将 OpenCV 格式的外参矩阵转换为 OpenGL 格式的 cam2world 矩阵
    
    Args:
        extrinsic_cv: OpenCV 格式的外参矩阵 (3x4 或 4x4)
        
    Returns:
        cam2world_gl: OpenGL 格式的 cam2world 矩阵 (4x4)
    """
    # 如果是 3x4，补全为 4x4
    if extrinsic_cv.shape == (3, 4):
        extrinsic_cv_full = np.eye(4)
        extrinsic_cv_full[:3, :] = extrinsic_cv
        extrinsic_cv = extrinsic_cv_full
    
    # OpenCV 到 OpenGL 的坐标系转换（绕X轴旋转180度）
    R_cv_to_gl = np.eye(4)
    R_cv_to_gl[1, 1] = -1
    R_cv_to_gl[2, 2] = -1
    
    # world2cam_cv -> world2cam_gl
    world2cam_gl = R_cv_to_gl @ extrinsic_cv
    
    # cam2world_gl = inv(world2cam_gl)
    cam2world_gl = np.linalg.inv(world2cam_gl)
    
    return cam2world_gl

def cam2world_gl_to_sapien_pose(cam2world_gl):
    """
    将 OpenGL 格式的 cam2world 矩阵转换为 SAPIEN 格式的相机位姿
    
    Args:
        cam2world_gl: OpenGL 格式的 cam2world 矩阵 (4x4)
        
    Returns:
        T_world_camera_sapien: SAPIEN 格式的相机位姿 (4x4)
    """
    # SAPIEN link frame 到 OpenGL camera frame 的逆变换
    R_link_to_cam = np.array([
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0]
    ])
    
    T_link_to_cam = np.eye(4)
    T_link_to_cam[:3, :3] = R_link_to_cam
    
    # T_world_camera_sapien = cam2world_gl @ inv(T_link_to_cam)
    T_world_camera_sapien = cam2world_gl @ np.linalg.inv(T_link_to_cam)
    
    return T_world_camera_sapien

def camera_extrinsics_to_tcp_pose(camera_extrinsics, T_tcp_camera, robot_base_position, debug=False):
    """
    从相机外参反推TCP位姿
    
    Args:
        camera_extrinsics: 相机外参，可以是以下格式之一：
            - dict: 包含 'cam2world_gl' 或 'extrinsic_cv' 或 'T_world_camera_sapien'
            - np.ndarray: cam2world_gl (4x4) 或 extrinsic_cv (3x4 或 4x4)
        T_tcp_camera: TCP到相机的变换矩阵 (4x4)
        robot_base_position: 机器人基座位置，3个维度
        debug: 是否输出调试信息
        
    Returns:
        T_world_tcp: 世界到TCP的变换矩阵 (4x4)，相对于基座坐标系
    """
    # 确定输入格式并转换为 SAPIEN 格式的相机位姿
    if isinstance(camera_extrinsics, dict):
        if 'T_world_camera_sapien' in camera_extrinsics:
            T_world_camera_sapien = camera_extrinsics['T_world_camera_sapien']
        elif 'cam2world_gl' in camera_extrinsics:
            cam2world_gl = camera_extrinsics['cam2world_gl']
            T_world_camera_sapien = cam2world_gl_to_sapien_pose(cam2world_gl)
        elif 'extrinsic_cv' in camera_extrinsics:
            extrinsic_cv = camera_extrinsics['extrinsic_cv']
            cam2world_gl = extrinsic_cv_to_cam2world_gl(extrinsic_cv)
            T_world_camera_sapien = cam2world_gl_to_sapien_pose(cam2world_gl)
        else:
            raise ValueError("camera_extrinsics dict must contain one of: 'T_world_camera_sapien', 'cam2world_gl', 'extrinsic_cv'")
    elif isinstance(camera_extrinsics, np.ndarray):
        if camera_extrinsics.shape == (3, 4):
            # 假设是 extrinsic_cv (3x4)
            cam2world_gl = extrinsic_cv_to_cam2world_gl(camera_extrinsics)
            T_world_camera_sapien = cam2world_gl_to_sapien_pose(cam2world_gl)
        elif camera_extrinsics.shape == (4, 4):
            # 可能是 cam2world_gl 或 extrinsic_cv (4x4)，默认假设是 cam2world_gl
            # 如果用户传入的是 extrinsic_cv (4x4)，需要明确指定
            T_world_camera_sapien = cam2world_gl_to_sapien_pose(camera_extrinsics)
        else:
            raise ValueError(f"Unsupported camera_extrinsics shape: {camera_extrinsics.shape}")
    else:
        raise ValueError("camera_extrinsics must be dict or np.ndarray")
    
    if debug:
        print(f"\n[调试] camera_extrinsics_to_tcp_pose:")
        print(f"  T_world_camera_sapien (世界坐标系，包含基座位置):")
        print(f"    translation: {T_world_camera_sapien[:3, 3]}")
    
    # 从相机位姿反推TCP位姿: T_world_tcp = T_world_camera @ inv(T_tcp_camera)
    # 此时 T_world_tcp 是世界坐标系下的TCP位姿（包含基座位置）
    T_world_tcp = T_world_camera_sapien @ np.linalg.inv(T_tcp_camera)
    
    if debug:
        print(f"  T_world_tcp (世界坐标系，包含基座位置):")
        print(f"    translation: {T_world_tcp[:3, 3]}")
        print(f"  robot_base_position: {robot_base_position}")
    
    # 减去基座偏移，得到相对于基座坐标系的TCP位姿
    # 这是逆运动学需要的目标位姿（相对于机器人基座）
    T_world_tcp[:3, 3] -= robot_base_position
    
    if debug:
        print(f"  T_world_tcp (基座坐标系，已减去基座偏移):")
        print(f"    translation: {T_world_tcp[:3, 3]}")
    
    return T_world_tcp

def get_random_configuration(model):
    """
    生成随机关节配置，在关节限位范围内
    
    Args:
        model: pinocchio模型
        
    Returns:
        qpos: 随机关节角度
    """
    qpos = np.zeros(model.nq)
    for i in range(model.nq):
        q_min = model.lowerPositionLimit[i]
        q_max = model.upperPositionLimit[i]
        qpos[i] = np.random.uniform(q_min, q_max)
    return qpos

def estimate_initial_qpos_from_target_pose(model, target_pose):
    """
    根据目标TCP位姿，使用几何启发式方法估算一个粗略的初始关节配置
    
    注意：这不是真正的逆运动学求解，而是一个启发式估计，用于为优化算法提供更好的初始值
    
    对于Panda机器人（7自由度），启发式策略：
    1. 根据TCP位置，估算前3个关节的角度（控制位置）
    2. 根据TCP方向，估算后4个关节的角度（控制姿态）
    3. 对于无法估算的关节，使用中性位置或随机值
    
    Args:
        model: pinocchio模型
        target_pose: 目标TCP位姿，pinocchio.SE3对象或4x4矩阵
        
    Returns:
        qpos_estimate: 估算的关节角度
    """
    # 转换目标位姿
    if isinstance(target_pose, np.ndarray):
        target_se3 = pinocchio.SE3(target_pose)
    else:
        target_se3 = target_pose
    
    target_pos = target_se3.translation
    target_rot = target_se3.rotation
    
    # 初始化估算的关节角度（使用中性位置作为基础）
    qpos_estimate = pinocchio.neutral(model).copy()
    
    # 对于Panda机器人，尝试使用几何方法估算前几个关节
    # 注意：这是一个简化的启发式，实际效果取决于机器人结构
    
    # 策略1: 根据TCP的x, y位置，估算第一个关节（绕z轴旋转）
    if model.nq >= 1:
        # 第一个关节通常控制机器人的朝向
        # 计算TCP在xy平面上的角度
        angle_xy = np.arctan2(target_pos[1], target_pos[0])
        # 将角度限制在关节限位内
        q_min = model.lowerPositionLimit[0]
        q_max = model.upperPositionLimit[0]
        qpos_estimate[0] = np.clip(angle_xy, q_min, q_max)
    
    # 策略2: 根据TCP的径向距离，估算第二个关节
    if model.nq >= 2:
        # 计算TCP到基座的距离（在xy平面上的投影）
        radial_distance = np.sqrt(target_pos[0]**2 + target_pos[1]**2)
        # 根据距离估算第二个关节角度（这是一个粗略估计）
        # 假设第二个关节主要影响径向距离
        q_min = model.lowerPositionLimit[1]
        q_max = model.upperPositionLimit[1]
        q_mid = (q_min + q_max) / 2
        # 根据距离调整（距离越大，角度越大，但这是简化假设）
        qpos_estimate[1] = q_mid + (radial_distance - 0.5) * 0.5  # 粗略调整
        qpos_estimate[1] = np.clip(qpos_estimate[1], q_min, q_max)
    
    # 策略3: 根据TCP的z高度，估算第三个关节
    if model.nq >= 3:
        # 第三个关节通常影响高度
        q_min = model.lowerPositionLimit[2]
        q_max = model.upperPositionLimit[2]
        q_mid = (q_min + q_max) / 2
        # 根据z高度调整
        qpos_estimate[2] = q_mid + (target_pos[2] - 0.5) * 0.3  # 粗略调整
        qpos_estimate[2] = np.clip(qpos_estimate[2], q_min, q_max)
    
    # 对于剩余的关节（控制姿态），使用中性位置或小幅随机扰动
    # 因为这些关节主要影响姿态，难以用简单的几何方法估算
    for i in range(3, model.nq):
        q_min = model.lowerPositionLimit[i]
        q_max = model.upperPositionLimit[i]
        q_mid = (q_min + q_max) / 2
        # 使用中性位置附近的小幅随机扰动
        qpos_estimate[i] = q_mid + np.random.uniform(-0.2, 0.2)
        qpos_estimate[i] = np.clip(qpos_estimate[i], q_min, q_max)
    
    return qpos_estimate

def wrap_joint_limit(model, q):
    """
    检查并调整关节角度到限位范围内
    对于旋转关节，将角度包装到 [q_min, q_min+2*pi) 范围内
    
    Args:
        model: pinocchio模型
        q: 关节配置，可能会被修改
        
    Returns:
        flag: True 如果 q 可以调整到关节限位内
    """
    flag = True
    for i in range(model.nq):
        # 检查是否是旋转关节：检查关节限位范围是否大于 2*pi
        q_range = model.upperPositionLimit[i] - model.lowerPositionLimit[i]
        if q_range > 5.0:  # 假设旋转关节的范围大于 5 弧度（约 286 度）
            # 旋转关节：包装到 [q_min, q_min+2*pi)
            q_min = model.lowerPositionLimit[i]
            if -1e-3 <= q[i] - q_min < 0:
                continue
            q[i] -= 2 * np.pi * np.floor((q[i] - q_min) / (2 * np.pi))
            if q[i] > model.upperPositionLimit[i] + 1e-3:
                flag = False
        else:
            # 非旋转关节：直接检查限位
            if q[i] < model.lowerPositionLimit[i] - 1e-3 or q[i] > model.upperPositionLimit[i] + 1e-3:
                flag = False
    return flag

def distance_6D(p1, q1, p2, q2):
    """
    计算两个位姿之间的6D距离
    
    Args:
        p1: 位置1 [x, y, z]
        q1: 四元数1 [w, x, y, z] 或 [x, y, z, w]
        p2: 位置2 [x, y, z]
        q2: 四元数2 [w, x, y, z] 或 [x, y, z, w]
        
    Returns:
        distance: 6D距离
    """
    pos_dist = np.linalg.norm(p1 - p2)
    # 四元数距离（考虑 q 和 -q 表示同一旋转）
    quat_dist = min(np.linalg.norm(q1 - q2), np.linalg.norm(q1 + q2))
    return pos_dist + quat_dist

def solve_ik(model, data, tcp_frame_id, target_pose, initial_qpos=None, n_init_qpos=20, max_iterations=100, tolerance=1e-3):
    """
    使用优化方法求解逆运动学，参考 mplib_utils.py 的实现策略
    多次尝试不同的初始配置以提高成功率
    
    Args:
        model: pinocchio模型
        data: pinocchio数据
        tcp_frame_id: TCP帧的索引
        target_pose: 目标位姿，pinocchio.SE3 对象或 4x4 矩阵
        initial_qpos: 初始关节角度，如果为None则使用模型的中立位置
        n_init_qpos: 尝试的初始配置数量（类似 mplib_utils.py 的 n_init_qpos）
        max_iterations: 每次优化的最大迭代次数
        tolerance: 收敛容差
        
    Returns:
        qpos: 求解得到的关节角度
        success: 是否成功求解
    """
    # 转换目标位姿为 pinocchio.SE3
    if isinstance(target_pose, np.ndarray):
        target_se3 = pinocchio.SE3(target_pose)
    else:
        target_se3 = target_pose
    
    # 将目标位姿转换为 [x, y, z, qw, qx, qy, qz] 格式用于距离计算
    target_pos = target_se3.translation
    target_quat = pinocchio.Quaternion(target_se3.rotation).coeffs()  # [x, y, z, w]
    target_quat_wxyz = np.array([target_quat[3], target_quat[0], target_quat[1], target_quat[2]])  # [w, x, y, z]
    
    # 定义优化目标函数
    def objective(q):
        pinocchio.forwardKinematics(model, data, q)
        pinocchio.updateFramePlacement(model, data, tcp_frame_id)
        current_pose = data.oMf[tcp_frame_id]
        
        # 计算位置误差
        pos_error = current_pose.translation - target_se3.translation
        
        # 计算旋转误差（使用对数映射）
        rot_error = pinocchio.log3(current_pose.rotation.T @ target_se3.rotation)
        
        # 总误差
        error = np.concatenate([pos_error, rot_error])
        return 0.5 * np.dot(error, error)
    
    # 定义约束（关节限位）
    bounds = [(model.lowerPositionLimit[i], model.upperPositionLimit[i]) 
              for i in range(model.nq)]
    
    # 初始化
    if initial_qpos is None:
        # 如果没有提供初始值，尝试使用启发式方法估算
        start_qpos = estimate_initial_qpos_from_target_pose(model, target_se3)
    else:
        start_qpos = initial_qpos.copy()
    
    qpos0 = start_qpos.copy()
    results = []
    min_dis = 1e9
    best_qpos = None
    
    # 多次尝试不同的初始配置
    for attempt in range(n_init_qpos):
        # 使用当前初始配置或随机配置
        if attempt == 0:
            # 第一次尝试：使用提供的初始值或启发式估算值
            qpos_init = start_qpos.copy()
        elif attempt == 1:
            # 第二次尝试：使用启发式估算值（如果第一次不是启发式估算）
            if initial_qpos is not None:
                qpos_init = estimate_initial_qpos_from_target_pose(model, target_se3)
            else:
                qpos_init = get_random_configuration(model)
        else:
            # 后续尝试：使用随机配置
            qpos_init = get_random_configuration(model)
        
        # 优化求解
        try:
            result = minimize(
                objective,
                qpos_init,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iterations, 'ftol': tolerance}
            )
            
            # 检查关节限位
            qpos_candidate = result.x.copy()
            flag = wrap_joint_limit(model, qpos_candidate)
            
            if flag:
                # 验证结果：计算实际位姿
                pinocchio.forwardKinematics(model, data, qpos_candidate)
                pinocchio.updateFramePlacement(model, data, tcp_frame_id)
                final_pose = data.oMf[tcp_frame_id]
                
                # 计算6D距离
                final_pos = final_pose.translation
                final_quat = pinocchio.Quaternion(final_pose.rotation).coeffs()  # [x, y, z, w]
                final_quat_wxyz = np.array([final_quat[3], final_quat[0], final_quat[1], final_quat[2]])  # [w, x, y, z]
                
                tmp_dis = distance_6D(target_pos, target_quat_wxyz, final_pos, final_quat_wxyz)
                
                if tmp_dis < min_dis:
                    min_dis = tmp_dis
                    best_qpos = qpos_candidate.copy()
                
                if tmp_dis < tolerance:
                    # 检查是否与已有解不同（避免重复解）
                    unique = True
                    for existing_result in results:
                        if np.linalg.norm(existing_result - qpos_candidate) < 0.1:
                            unique = False
                            break
                    if unique:
                        results.append(qpos_candidate.copy())
        except:
            # 优化失败，继续尝试下一个初始配置
            pass
    
    # 返回结果
    if len(results) > 0:
        # 返回第一个满足容差的解
        success = True
        print(f"逆运动学求解成功，找到 {len(results)} 个满足容差的解")
        return results[0], success
    elif best_qpos is not None:
        # 返回最佳解（即使未完全满足容差）
        # 如果最小距离在容差的合理范围内（比如 2 倍容差），仍然认为成功
        # 这样可以处理数值误差和优化算法的精度限制
        success = min_dis < tolerance * 2
        if not success:
            print(f"警告: 逆运动学未完全收敛，最小6D距离: {min_dis:.6f}, 容差: {tolerance:.6f}")
            # 计算位置和旋转误差
            pinocchio.forwardKinematics(model, data, best_qpos)
            pinocchio.updateFramePlacement(model, data, tcp_frame_id)
            final_pose = data.oMf[tcp_frame_id]
            pos_error = np.linalg.norm(final_pose.translation - target_se3.translation)
            rot_error = np.linalg.norm(pinocchio.log3(final_pose.rotation.T @ target_se3.rotation))
            print(f"  位置误差: {pos_error:.6f} m, 旋转误差: {rot_error:.6f} rad")
            print(f"  建议: 如果误差在可接受范围内，可以继续使用该解")
        else:
            print(f"逆运动学求解成功（接近解），最小距离: {min_dis:.6f}, 容差: {tolerance:.6f}")
        return best_qpos, success
    else:
        # 所有尝试都失败
        success = False
        print(f"错误: 逆运动学求解完全失败，所有 {n_init_qpos} 次尝试都未找到有效解")
        return start_qpos, success

def get_qpos_from_camera_extrinsics(
    urdf_filepath: str,
    camera_extrinsics,
    robot_base_position: np.ndarray,
    camera_translation_relative_to_tcp: np.ndarray,
    camera_rotation_relative_to_tcp: np.ndarray,
    initial_qpos: np.ndarray = None,
    max_iterations: int = 200,  # 增加迭代次数以提高精度
    tolerance: float = 1e-4,  # 降低容差以提高精度
    n_init_qpos: int = 50,  # 增加随机尝试次数
    ground_truth_qpos: np.ndarray = None,  # 真实qpos，用于验证误差
    verbose: bool = True  # 是否输出详细信息
):
    """
    从相机外参通过逆运动学求解出qpos
    
    Args:
        urdf_filepath: URDF文件路径
        camera_extrinsics: 相机外参，可以是以下格式之一：
            - dict: 包含 'cam2world_gl' 或 'extrinsic_cv' 或 'T_world_camera_sapien'
            - np.ndarray: cam2world_gl (4x4) 或 extrinsic_cv (3x4 或 4x4)
        robot_base_position: 机器人基座位置，3个维度
        camera_translation_relative_to_tcp: 相机相对于TCP的平移，3个维度
        camera_rotation_relative_to_tcp: 相机相对于TCP的旋转，4个维度 (SAPIEN格式 [w, x, y, z])
        initial_qpos: 初始关节角度，用于逆运动学求解的初始值（可选）
        max_iterations: 逆运动学最大迭代次数
        tolerance: 逆运动学收敛容差
        ground_truth_qpos: 真实qpos，用于验证误差（可选）
        
    Returns:
        qpos: 求解得到的关节角度，9个维度（前7个是关节角度，后2个是夹爪状态，夹爪状态保持为0）
        success: 是否成功求解
        error_info: 如果提供了ground_truth_qpos，返回误差信息dict，否则返回None
    """
    # 构建模型
    model = pinocchio.buildModelFromUrdf(urdf_filepath)
    data = model.createData()
    
    # 获取TCP帧ID
    try:
        tcp_frame_id = model.getFrameId('panda_hand_tcp')
    except:
        print(f"Warning: Frame 'panda_hand_tcp' not found, using last frame")
        tcp_frame_id = len(model.frames) - 1
    
    # 创建TCP到相机的变换矩阵
    T_tcp_camera = create_transformation_matrix(
        camera_translation_relative_to_tcp,
        camera_rotation_relative_to_tcp,
        quat_format='wxyz'  # SAPIEN 格式
    )
    
    # 从相机外参反推TCP位姿
    T_world_tcp = camera_extrinsics_to_tcp_pose(
        camera_extrinsics,
        T_tcp_camera,
        robot_base_position,
        debug=verbose  # 根据verbose参数决定是否输出调试信息
    )
    
    # 转换为 pinocchio.SE3
    target_pose = pinocchio.SE3(T_world_tcp)
    
    # 准备初始关节角度（只使用前7个关节，忽略夹爪）
    if initial_qpos is not None:
        qpos_init = initial_qpos[:model.nq].copy()
    else:
        # 使用模型的中性位置作为逆运动学优化的初始值
        # 中性位置通常是URDF中定义的默认配置，或全零配置
        qpos_init = pinocchio.neutral(model)
    
    # 求解逆运动学
    if verbose:
        print(f"正在从相机外参求解关节角度，目标TCP位姿: translation={target_pose.translation}, rotation=\n{target_pose.rotation}")
        print(f"  参数: n_init_qpos={n_init_qpos}, max_iterations={max_iterations}, tolerance={tolerance}")
    qpos_joints, success = solve_ik(
        model, data, tcp_frame_id, target_pose,
        initial_qpos=qpos_init,
        n_init_qpos=n_init_qpos,  # 使用参数控制尝试次数
        max_iterations=max_iterations,
        tolerance=tolerance
    )
    
    # 构建完整的qpos（9维：7个关节 + 2个夹爪状态）
    qpos = np.zeros(9)
    qpos[:model.nq] = qpos_joints  #这里的model.nq是9
    # 夹爪状态保持为0（可以根据需要修改）
    
    if verbose:
        if success:
            print(f"逆运动学求解成功: qpos = {qpos}")
        else:
            print(f"警告: 逆运动学求解可能未完全收敛")
    
    # 如果提供了真实qpos，计算误差
    # error_info = None
    # if ground_truth_qpos is not None:
    #     error_info = compare_qpos_with_ground_truth(
    #         model, data, tcp_frame_id, qpos, ground_truth_qpos, 
    #         robot_base_position, target_pose
    #     )
    
    return qpos, success

def compare_qpos_with_ground_truth(
    model, data, tcp_frame_id, computed_qpos, ground_truth_qpos,
    robot_base_position, target_pose
):
    """
    比较计算得到的qpos和真实qpos，计算TCP位姿误差
    
    Args:
        model: pinocchio模型
        data: pinocchio数据
        tcp_frame_id: TCP帧的索引
        computed_qpos: 通过逆运动学计算得到的qpos
        ground_truth_qpos: 真实的qpos
        robot_base_position: 机器人基座位置
        target_pose: 目标TCP位姿（从相机外参反推得到的）
        
    Returns:
        error_info: 包含各种误差信息的字典
    """
    print("\n" + "="*60)
    print("开始验证逆运动学求解结果与真实qpos的误差")
    print("="*60)
    
    # 1. 计算真实qpos对应的TCP位姿
    pinocchio.forwardKinematics(model, data, ground_truth_qpos[:model.nq])
    pinocchio.updateFramePlacement(model, data, tcp_frame_id)
    true_tcp_pose = data.oMf[tcp_frame_id]
    true_tcp_matrix = true_tcp_pose.homogeneous
    
    # 加上基座偏移（与正向计算保持一致）
    true_tcp_matrix[:3, 3] += robot_base_position
    true_tcp_se3 = pinocchio.SE3(true_tcp_matrix)
    
    # 2. 计算逆运动学得到的qpos对应的TCP位姿
    pinocchio.forwardKinematics(model, data, computed_qpos[:model.nq])
    pinocchio.updateFramePlacement(model, data, tcp_frame_id)
    computed_tcp_pose = data.oMf[tcp_frame_id]
    computed_tcp_matrix = computed_tcp_pose.homogeneous
    
    # 加上基座偏移
    computed_tcp_matrix[:3, 3] += robot_base_position
    computed_tcp_se3 = pinocchio.SE3(computed_tcp_matrix)
    
    # 3. 计算TCP位姿误差
    # 位置误差
    pos_error = np.linalg.norm(computed_tcp_se3.translation - true_tcp_se3.translation)
    
    # 旋转误差（使用对数映射）
    rot_error_vec = pinocchio.log3(computed_tcp_se3.rotation.T @ true_tcp_se3.rotation)
    rot_error = np.linalg.norm(rot_error_vec)
    rot_error_deg = np.degrees(rot_error)
    
    # 与目标位姿的误差（验证反推是否正确）
    # 注意：target_pose是基座坐标系下的，需要加上基座偏移才能与世界坐标系下的位姿比较
    target_pose_world = pinocchio.SE3(target_pose.homogeneous.copy())
    target_pose_world.translation += robot_base_position
    
    target_pos_error = np.linalg.norm(computed_tcp_se3.translation - target_pose_world.translation)
    target_rot_error_vec = pinocchio.log3(computed_tcp_se3.rotation.T @ target_pose_world.rotation)
    target_rot_error = np.linalg.norm(target_rot_error_vec)
    target_rot_error_deg = np.degrees(target_rot_error)
    
    # 4. 计算关节角度误差
    joint_error = np.abs(computed_qpos[:model.nq] - ground_truth_qpos[:model.nq])
    max_joint_error = np.max(joint_error)
    mean_joint_error = np.mean(joint_error)
    joint_error_deg = np.degrees(joint_error)
    max_joint_error_deg = np.max(joint_error_deg)
    mean_joint_error_deg = np.mean(joint_error_deg)
    
    # 5. 打印结果
    print(f"\n【TCP位姿误差分析】")
    print(f"  真实TCP位姿: translation={true_tcp_se3.translation}")
    print(f"  计算TCP位姿: translation={computed_tcp_se3.translation}")
    print(f"  位置误差: {pos_error*1000:.4f} mm")
    print(f"  旋转误差: {rot_error:.6f} rad ({rot_error_deg:.4f} 度)")
    
    print(f"\n【与目标位姿的误差（验证反推正确性）】")
    print(f"  目标TCP位姿（基座坐标系）: translation={target_pose.translation}")
    print(f"  目标TCP位姿（世界坐标系，用于比较）: translation={target_pose_world.translation}")
    print(f"  计算TCP位姿（世界坐标系）: translation={computed_tcp_se3.translation}")
    print(f"  位置误差: {target_pos_error*1000:.4f} mm")
    print(f"  旋转误差: {target_rot_error:.6f} rad ({target_rot_error_deg:.4f} 度)")
    
    print(f"\n【关节角度误差分析】")
    print(f"  真实qpos: {ground_truth_qpos[:model.nq]}")
    print(f"  计算qpos: {computed_qpos[:model.nq]}")
    print(f"  最大关节误差: {max_joint_error:.6f} rad ({max_joint_error_deg:.4f} 度)")
    print(f"  平均关节误差: {mean_joint_error:.6f} rad ({mean_joint_error_deg:.4f} 度)")
    print(f"  各关节误差 (rad): {joint_error}")
    print(f"  各关节误差 (度): {joint_error_deg}")
    
    # 6. 判断是否在可接受范围内
    pos_acceptable = pos_error < 0.01  # 1cm
    rot_acceptable = rot_error_deg < 5.0  # 5度
    joint_acceptable = max_joint_error_deg < 10.0  # 10度
    
    print(f"\n【误差评估】")
    print(f"  位置误差是否可接受 (<1cm): {pos_acceptable} ({'✓' if pos_acceptable else '✗'})")
    print(f"  旋转误差是否可接受 (<5度): {rot_acceptable} ({'✓' if rot_acceptable else '✗'})")
    print(f"  关节误差是否可接受 (<10度): {joint_acceptable} ({'✓' if joint_acceptable else '✗'})")
    
    overall_acceptable = pos_acceptable and rot_acceptable
    print(f"  整体评估: {'✓ 误差在可接受范围内' if overall_acceptable else '✗ 误差较大，可能需要检查'}")
    
    print("="*60 + "\n")
    
    # 返回误差信息
    return {
        'tcp_position_error_mm': pos_error * 1000,
        'tcp_rotation_error_rad': rot_error,
        'tcp_rotation_error_deg': rot_error_deg,
        'target_position_error_mm': target_pos_error * 1000,
        'target_rotation_error_rad': target_rot_error,
        'target_rotation_error_deg': target_rot_error_deg,
        'max_joint_error_rad': max_joint_error,
        'mean_joint_error_rad': mean_joint_error,
        'max_joint_error_deg': max_joint_error_deg,
        'mean_joint_error_deg': mean_joint_error_deg,
        'joint_errors_rad': joint_error,
        'joint_errors_deg': joint_error_deg,
        'pos_acceptable': pos_acceptable,
        'rot_acceptable': rot_acceptable,
        'joint_acceptable': joint_acceptable,
        'overall_acceptable': overall_acceptable,
        'true_tcp_pose': true_tcp_se3,
        'computed_tcp_pose': computed_tcp_se3,
        'target_tcp_pose': target_pose
    }

def get_camera_extrinsics(urdf_filepath: str, qpos: np.ndarray, robot_base_position: np.ndarray, camera_translation_relative_to_tcp: np.ndarray, camera_rotation_relative_to_tcp: np.ndarray):
    """
    urdf_filename: URDF文件路径
    qpos: 当前关节角度，9个维度，前7个维度是关节角度，后2个是夹爪状态
    robot_base_position: 机器人基座位置，3个维度,  FIXME 这里有个问题，为什么只考虑了基座的偏移，没有考虑基座的旋转？
    camera_translation_relative_to_tcp: 相机相对于TCP的平移，3个维度
    camera_rotation_relative_to_tcp: 相机相对于TCP的旋转，4个维度
    """
    model = pinocchio.buildModelFromUrdf(urdf_filepath)
    data = model.createData()
    print("正在对该关节角度进行相应的外参计算: %s" % qpos.T)
    pinocchio.forwardKinematics(model, data, qpos)
    T_world_tcp, tcp_frame_id = get_tcp_pose(model, data)
    
    # 将基座偏移加到TCP位置上
    T_world_tcp[:3, 3] += robot_base_position

    # 创建TCP到相机的变换矩阵
    # quat_format='wxyz' 表示输入四元数是 SAPIEN 格式 [w, x, y, z]
    T_tcp_camera = create_transformation_matrix(
    camera_translation_relative_to_tcp,
    camera_rotation_relative_to_tcp,
    quat_format='wxyz'  # SAPIEN 格式
    )

    # 计算相机外参
    # T_world_camera, T_camera_world = compute_camera_extrinsics(T_world_tcp, T_tcp_camera)

    #计算与sapien仿真环境获取一致的外参
    camera_params = compute_full_camera_params(T_world_tcp, T_tcp_camera)

    return camera_params

def verify_conversion_chain_symmetry(urdf_filepath, qpos, robot_base_position,
                                    camera_translation_relative_to_tcp,
                                    camera_rotation_relative_to_tcp):
    """
    验证正向和反向转换链的对称性
    
    Args:
        urdf_filepath: URDF文件路径
        qpos: 测试用的qpos
        robot_base_position: 基座位置
        camera_translation_relative_to_tcp: 相机相对于TCP的平移
        camera_rotation_relative_to_tcp: 相机相对于TCP的旋转
        
    Returns:
        result: 验证结果
    """
    print("\n" + "="*80)
    print("验证正向和反向转换链的对称性")
    print("="*80)
    
    # 正向：qpos → extrinsic_cv
    camera_params = get_camera_extrinsics(
        urdf_filepath,
        qpos,
        robot_base_position,
        camera_translation_relative_to_tcp,
        camera_rotation_relative_to_tcp
    )
    extrinsic_cv_forward = camera_params['extrinsic_cv']
    T_world_camera_sapien_forward = camera_params['T_world_camera_sapien']
    
    # 计算正向的TCP位姿（用于对比）
    model = pinocchio.buildModelFromUrdf(urdf_filepath)
    data = model.createData()
    pinocchio.forwardKinematics(model, data, qpos)
    T_world_tcp_base_frame, _ = get_tcp_pose(model, data)  # 基座坐标系
    T_world_tcp_world_frame = T_world_tcp_base_frame.copy()
    T_world_tcp_world_frame[:3, 3] += robot_base_position  # 世界坐标系
    
    print(f"\n【正向转换】")
    print(f"  TCP位姿（基座坐标系）: {T_world_tcp_base_frame[:3, 3]}")
    print(f"  TCP位姿（世界坐标系）: {T_world_tcp_world_frame[:3, 3]}")
    print(f"  相机位姿（世界坐标系）: {T_world_camera_sapien_forward[:3, 3]}")
    
    # 反向：extrinsic_cv → TCP位姿
    T_tcp_camera = create_transformation_matrix(
        camera_translation_relative_to_tcp,
        camera_rotation_relative_to_tcp,
        quat_format='wxyz'
    )
    
    camera_extrinsics = {'extrinsic_cv': extrinsic_cv_forward}
    T_world_tcp_reverse = camera_extrinsics_to_tcp_pose(
        camera_extrinsics,
        T_tcp_camera,
        robot_base_position,
        debug=True
    )
    
    print(f"\n【反向转换】")
    print(f"  TCP位姿（从extrinsic_cv反推，基座坐标系）: {T_world_tcp_reverse[:3, 3]}")
    
    # 比较
    pos_error = np.linalg.norm(T_world_tcp_base_frame[:3, 3] - T_world_tcp_reverse[:3, 3])
    rot_error = np.linalg.norm(pinocchio.log3(
        T_world_tcp_base_frame[:3, :3].T @ T_world_tcp_reverse[:3, :3]
    ))
    
    print(f"\n【对称性验证】")
    print(f"  位置误差: {pos_error*1000:.4f} mm")
    print(f"  旋转误差: {rot_error:.6f} rad ({np.degrees(rot_error):.4f} 度)")
    
    if pos_error < 0.01 and rot_error < 0.01:  # 1cm和约0.57度
        print(f"  ✓ 转换链对称性验证通过！")
        return True
    else:
        print(f"  ✗ 转换链对称性验证失败！")
        print(f"  可能的问题：")
        print(f"    - 基座位置处理不正确")
        print(f"    - 坐标系转换有误")
        print(f"    - 数值误差累积")
        return False

def check_extrinsic_contains_base_position(extrinsic_cv, robot_base_position, 
                                          urdf_filepath, ground_truth_qpos,
                                          camera_translation_relative_to_tcp,
                                          camera_rotation_relative_to_tcp):
    """
    检查extrinsic_cv中是否已经包含了基座位置
    
    Args:
        extrinsic_cv: 输入的相机外参 (3x4)
        robot_base_position: 基座位置
        urdf_filepath: URDF文件路径
        ground_truth_qpos: 真实qpos
        camera_translation_relative_to_tcp: 相机相对于TCP的平移
        camera_rotation_relative_to_tcp: 相机相对于TCP的旋转
        
    Returns:
        analysis: 分析结果字典
    """
    print("\n" + "="*80)
    print("检查extrinsic_cv中是否包含基座位置")
    print("="*80)
    
    # 1. 从真实qpos计算extrinsic_cv（包含基座位置）
    camera_params = get_camera_extrinsics(
        urdf_filepath,
        ground_truth_qpos,
        robot_base_position,
        camera_translation_relative_to_tcp,
        camera_rotation_relative_to_tcp
    )
    extrinsic_cv_with_base = camera_params['extrinsic_cv']
    T_world_camera_sapien_with_base = camera_params['T_world_camera_sapien']
    
    # 2. 从extrinsic_cv反推相机在世界坐标系中的位置
    # extrinsic_cv是world2cam，格式为 [R^T | -R^T @ t]
    # 其中t是相机在世界坐标系中的位置
    extrinsic_cv_4x4 = np.eye(4)
    extrinsic_cv_4x4[:3, :] = extrinsic_cv
    
    # 提取旋转和平移
    R_world2cam = extrinsic_cv_4x4[:3, :3]
    t_world2cam = extrinsic_cv_4x4[:3, 3]
    
    # 相机在世界坐标系中的位置: t = -R @ t_world2cam
    camera_position_from_extrinsic = -R_world2cam.T @ t_world2cam
    
    # 从包含基座位置的T_world_camera_sapien提取相机位置
    camera_position_with_base = T_world_camera_sapien_with_base[:3, 3]
    
    # 从真实qpos计算TCP位置（不包含基座）
    model = pinocchio.buildModelFromUrdf(urdf_filepath)
    data = model.createData()
    pinocchio.forwardKinematics(model, data, ground_truth_qpos)
    T_world_tcp_no_base, _ = get_tcp_pose(model, data)
    tcp_position_no_base = T_world_tcp_no_base[:3, 3]
    
    # 从真实qpos计算TCP位置（包含基座）
    tcp_position_with_base = tcp_position_no_base + robot_base_position
    
    # 计算相机位置（从TCP位置 + TCP到相机的变换）
    T_tcp_camera = create_transformation_matrix(
        camera_translation_relative_to_tcp,
        camera_rotation_relative_to_tcp,
        quat_format='wxyz'
    )
    camera_position_from_tcp = tcp_position_with_base + T_tcp_camera[:3, 3]
    
    print(f"\n【分析结果】")
    print(f"1. 基座位置: {robot_base_position}")
    print(f"2. TCP位置（不包含基座）: {tcp_position_no_base}")
    print(f"3. TCP位置（包含基座）: {tcp_position_with_base}")
    print(f"4. 相机位置（从包含基座的T_world_camera_sapien）: {camera_position_with_base}")
    print(f"5. 相机位置（从extrinsic_cv反推）: {camera_position_from_extrinsic}")
    print(f"6. 相机位置（从TCP位置计算）: {camera_position_from_tcp}")
    
    # 比较
    diff1 = np.linalg.norm(camera_position_from_extrinsic - camera_position_with_base)
    diff2 = np.linalg.norm(camera_position_from_extrinsic - camera_position_from_tcp)
    
    print(f"\n【差异分析】")
    print(f"  相机位置差异（extrinsic vs T_world_camera_sapien）: {diff1*1000:.4f} mm")
    print(f"  相机位置差异（extrinsic vs 从TCP计算）: {diff2*1000:.4f} mm")
    
    # 检查extrinsic_cv的平移部分是否包含基座位置的影响
    contains_base = diff1 < 0.01  # 1cm容差
    
    print(f"\n【结论】")
    if contains_base:
        print(f"  ✓ extrinsic_cv中已经包含了基座位置的影响")
        print(f"  相机位置差异: {diff1*1000:.4f} mm (在容差范围内)")
    else:
        print(f"  ✗ extrinsic_cv中可能不包含基座位置，或者有其他问题")
        print(f"  相机位置差异: {diff1*1000:.4f} mm (超出容差)")
    
    # 比较输入的extrinsic_cv和从真实qpos计算的extrinsic_cv
    extrinsic_diff = np.abs(extrinsic_cv - extrinsic_cv_with_base)
    max_diff = np.max(extrinsic_diff)
    print(f"\n  输入的extrinsic_cv与从真实qpos计算的extrinsic_cv的最大差异: {max_diff:.10f}")
    
    if max_diff < 1e-6:
        print(f"  ✓ 输入的extrinsic_cv与从真实qpos计算的extrinsic_cv几乎相同（数值误差范围内）")
    else:
        print(f"  ✗ 输入的extrinsic_cv与从真实qpos计算的extrinsic_cv有显著差异")
    
    print("="*80 + "\n")
    
    return {
        'contains_base': contains_base,
        'camera_position_from_extrinsic': camera_position_from_extrinsic,
        'camera_position_with_base': camera_position_with_base,
        'camera_position_from_tcp': camera_position_from_tcp,
        'position_diff_mm': diff1 * 1000,
        'extrinsic_diff': max_diff
    }


def load_h5_data(h5_group):
    """
    Recursively load data from h5 group into a dictionary.
    
    Args:
        h5_group: h5py Group or File object
    
    Returns:
        Dictionary containing the data
    """
    result = {}
    for key in h5_group.keys():
        item = h5_group[key]
        if isinstance(item, h5py.Group):
            result[key] = load_h5_data(item)
        elif isinstance(item, h5py.Dataset):
            result[key] = item[()]
    return result


def read_h5_extrinsic(h5_path: str, traj_idx: int, timestep: int, agent_idx: int) -> np.ndarray:
    """
    从h5文件中读取指定轨迹、时间步和agent的相机外参
    
    Args:
        h5_path: h5文件路径
        traj_idx: 轨迹序号
        timestep: 时间步t
        agent_idx: agent序号
    
    Returns:
        extrinsic_cv: 相机外参，3x4矩阵
    """
    camera_name = f"wrist_camera_agent{agent_idx}"
    
    with h5py.File(h5_path, 'r') as f:
        traj_key = f"traj_{traj_idx}"
        if traj_key not in f:
            raise ValueError(f"Trajectory {traj_idx} not found in h5 file")
        
        trajectory = f[traj_key]
        trajectory_data = load_h5_data(trajectory)
        
        # 检查时间步范围
        max_timestep = len(trajectory_data["obs"]["sensor_data"][camera_name]["rgb"]) - 1
        if timestep > max_timestep:
            raise ValueError(f"Timestep {timestep} exceeds max timestep {max_timestep}")
        
        # 读取外参 [3, 4]
        extrinsic = trajectory_data["obs"]["sensor_param"][camera_name]["extrinsic_cv"][timestep]
        
    return np.array(extrinsic)


def read_h5_qpos(h5_path: str, traj_idx: int, timestep: int, agent_idx: int) -> np.ndarray:
    """
    从h5文件中读取指定轨迹、时间步和agent的qpos
    
    Args:
        h5_path: h5文件路径
        traj_idx: 轨迹序号
        timestep: 时间步t
        agent_idx: agent序号
    
    Returns:
        qpos: 关节角度，numpy数组 (9 DOF: 7 arm + 2 gripper)
    """
    with h5py.File(h5_path, 'r') as f:
        traj_key = f"traj_{traj_idx}"
        if traj_key not in f:
            raise ValueError(f"Trajectory {traj_idx} not found in h5 file")
        
        trajectory = f[traj_key]
        trajectory_data = load_h5_data(trajectory)
        
        # 尝试不同的路径来读取qpos
        agent_key = f"panda-{agent_idx}"
        qpos = None
        
        # Path 1: obs/agent/panda-X/qpos
        try:
            if "obs" in trajectory_data and "agent" in trajectory_data["obs"]:
                if agent_key in trajectory_data["obs"]["agent"]:
                    if "qpos" in trajectory_data["obs"]["agent"][agent_key]:
                        qpos_data = trajectory_data["obs"]["agent"][agent_key]["qpos"]
                        if len(qpos_data) > timestep:
                            qpos = qpos_data[timestep]
        except (KeyError, IndexError, TypeError):
            pass
        
        if qpos is None:
            raise ValueError(
                f"Could not find qpos for trajectory {traj_idx}, "
                f"timestep {timestep}, agent {agent_idx}"
            )
        
        # Ensure qpos is a numpy array and flatten if needed
        qpos = np.array(qpos).flatten()
        return qpos


def validate_ik_with_planning(
    h5_path: str,
    traj_idx: int,
    initial_timestep: int,
    target_timestep: int,
    agent_idx: int,
    urdf_filepath: str,
    robot_base_position: np.ndarray,
    camera_translation_relative_to_tcp: np.ndarray,
    camera_rotation_relative_to_tcp: np.ndarray,
    planning_time: float = 5.0,
    n_init_qpos: int = 50,  # 逆运动学随机尝试次数
    max_iterations: int = 200,  # 每次优化的最大迭代次数
    tolerance: float = 1e-4,  # 逆运动学收敛容差
    verbose: bool = True
) -> dict:
    """
    验证从扰动的关节角度能否回到正常的角度
    
    流程：
    1. 从H5文件读取第initial_timestep个时间步的qpos作为初始值
    2. 从H5文件读取第target_timestep个时间步的相机外参
    3. 使用第initial_timestep个时间步的qpos作为初始值，通过逆运动学求解第target_timestep个时间步外参对应的qpos
    4. 使用规划方法，验证能否从求解得到的qpos规划到第initial_timestep个时间步的qpos
    
    Args:
        h5_path: H5文件路径
        traj_idx: 轨迹序号
        initial_timestep: 初始时间步（第0个时间步）
        target_timestep: 目标时间步（第10个时间步）
        agent_idx: agent序号
        urdf_filepath: URDF文件路径
        robot_base_position: 机器人基座位置
        camera_translation_relative_to_tcp: 相机相对于TCP的平移
        camera_rotation_relative_to_tcp: 相机相对于TCP的旋转
        planning_time: 规划时间限制
        n_init_qpos: 逆运动学随机尝试次数（默认50，越大越精确但越慢）
        max_iterations: 每次优化的最大迭代次数（默认200）
        tolerance: 逆运动学收敛容差（默认1e-4，越小越精确）
        verbose: 是否输出详细信息
    
    Returns:
        result: 包含验证结果的字典
    """
    result = {
        'initial_qpos': None,
        'target_extrinsic': None,
        'computed_qpos': None,
        'ik_success': False,
        'planning_success': False,
        'planning_status': None,
        'path': None,
        'duration': None,
        'error_info': None,
        'extrinsic_verification': None
    }
    
    # Step 1: 读取初始qpos（第0个时间步）
    if verbose:
        print(f"\n{'='*80}")
        print(f"步骤1: 读取初始qpos（时间步 {initial_timestep}）")
        print(f"{'='*80}")
    
    try:
        initial_qpos = read_h5_qpos(h5_path, traj_idx, initial_timestep, agent_idx)
        result['initial_qpos'] = initial_qpos
        if verbose:
            print(f"  初始qpos: {initial_qpos}")
    except Exception as e:
        print(f"错误: 无法读取初始qpos: {e}")
        return result
    
    # Step 2: 读取目标相机外参（第10个时间步）
    if verbose:
        print(f"\n{'='*80}")
        print(f"步骤2: 读取目标相机外参（时间步 {target_timestep}）")
        print(f"{'='*80}")
    
    try:
        target_extrinsic = read_h5_extrinsic(h5_path, traj_idx, target_timestep, agent_idx)
        result['target_extrinsic'] = target_extrinsic
        if verbose:
            print(f"  目标extrinsic_cv (3x4):")
            print(f"  {target_extrinsic}")
    except Exception as e:
        print(f"错误: 无法读取目标相机外参: {e}")
        return result
    
    # Step 3: 使用初始qpos作为初始值，通过逆运动学求解目标外参对应的qpos
    if verbose:
        print(f"\n{'='*80}")
        print(f"步骤3: 使用逆运动学求解目标外参对应的qpos")
        print(f"  使用初始qpos（时间步 {initial_timestep}）作为初始值")
        print(f"  目标外参来自时间步 {target_timestep}")
        print(f"{'='*80}")
    
    try:
        camera_extrinsics = {'extrinsic_cv': target_extrinsic}
        computed_qpos, ik_success, error_info = get_qpos_from_camera_extrinsics(
            urdf_filepath,
            camera_extrinsics,
            robot_base_position,
            camera_translation_relative_to_tcp,
            camera_rotation_relative_to_tcp,
            initial_qpos=initial_qpos,  # 使用初始qpos作为初始值
            max_iterations=max_iterations,
            tolerance=tolerance,
            n_init_qpos=n_init_qpos,
            ground_truth_qpos=None,  # 不提供真实qpos，因为我们不知道目标时间步的真实qpos
            verbose=verbose
        )
        
        result['computed_qpos'] = computed_qpos
        result['ik_success'] = ik_success
        result['error_info'] = error_info
        
        if verbose:
            if ik_success:
                print(f"  ✓ 逆运动学求解成功")
                print(f"  计算得到的qpos: {computed_qpos[:7]}")
            else:
                print(f"  ✗ 逆运动学求解失败或未完全收敛")
                if error_info:
                    print(f"  TCP位置误差: {error_info.get('tcp_position_error_mm', 'N/A')} mm")
                    print(f"  TCP旋转误差: {error_info.get('tcp_rotation_error_deg', 'N/A')} 度")
    except Exception as e:
        print(f"错误: 逆运动学求解失败: {e}")
        import traceback
        traceback.print_exc()
        return result
    
    # Step 3.5: 验证：从计算得到的qpos通过正运动学计算相机外参，与输入的外参比较
    if verbose:
        print(f"\n{'='*80}")
        print(f"步骤3.5: 验证逆运动学求解的正确性")
        print(f"  使用计算得到的qpos，通过正运动学计算相机外参")
        print(f"  与输入的外参（时间步 {target_timestep}）进行比较")
        print(f"{'='*80}")
    
    extrinsic_verification = {
        'success': False,
        'extrinsic_from_computed_qpos': None,
        'max_diff': None,
        'mean_diff': None,
        'position_error_mm': None,
        'rotation_error_deg': None
    }
    
    try:
        # 从计算得到的qpos计算相机外参
        camera_params_from_computed_qpos = get_camera_extrinsics(
            urdf_filepath,
            computed_qpos,
            robot_base_position,
            camera_translation_relative_to_tcp,
            camera_rotation_relative_to_tcp
        )
        computed_extrinsic = camera_params_from_computed_qpos['extrinsic_cv']
        extrinsic_verification['extrinsic_from_computed_qpos'] = computed_extrinsic
        
        # 比较两个外参矩阵
        extrinsic_diff = np.abs(target_extrinsic - computed_extrinsic)
        max_diff = np.max(extrinsic_diff)
        mean_diff = np.mean(extrinsic_diff)
        extrinsic_verification['max_diff'] = max_diff
        extrinsic_verification['mean_diff'] = mean_diff
        
        # 计算相机位置的误差
        # 从extrinsic_cv反推相机位置
        # extrinsic_cv是world2cam，格式为 [R^T | -R^T @ t]
        # 其中t是相机在世界坐标系中的位置
        R_world2cam_target = target_extrinsic[:3, :3]
        t_world2cam_target = target_extrinsic[:3, 3]
        camera_pos_target = -R_world2cam_target.T @ t_world2cam_target
        
        R_world2cam_computed = computed_extrinsic[:3, :3]
        t_world2cam_computed = computed_extrinsic[:3, 3]
        camera_pos_computed = -R_world2cam_computed.T @ t_world2cam_computed
        
        position_error = np.linalg.norm(camera_pos_target - camera_pos_computed)
        extrinsic_verification['position_error_mm'] = position_error * 1000
        
        # 计算旋转误差
        R_target = R_world2cam_target.T  # 相机到世界的旋转
        R_computed = R_world2cam_computed.T
        rot_error_vec = pinocchio.log3(R_target.T @ R_computed)
        rot_error = np.linalg.norm(rot_error_vec)
        extrinsic_verification['rotation_error_deg'] = np.degrees(rot_error)
        
        # 判断是否在可接受范围内
        # 外参矩阵差异 < 0.01，位置误差 < 10mm，旋转误差 < 5度
        acceptable = (max_diff < 0.01 and 
                     position_error < 0.01 and 
                     rot_error < np.radians(5))
        extrinsic_verification['success'] = acceptable
        
        if verbose:
            print(f"  输入的外参（时间步 {target_timestep}）:")
            print(f"  {target_extrinsic}")
            print(f"  从计算得到的qpos计算的外参:")
            print(f"  {computed_extrinsic}")
            print(f"\n  外参矩阵差异:")
            print(f"    最大差异: {max_diff:.10f}")
            print(f"    平均差异: {mean_diff:.10f}")
            print(f"  相机位置误差: {position_error*1000:.4f} mm")
            print(f"  相机旋转误差: {np.degrees(rot_error):.4f} 度")
            
            if acceptable:
                print(f"\n  ✓ 验证通过：从计算得到的qpos计算的外参与输入的外参一致（误差在可接受范围内）")
            else:
                print(f"\n  ✗ 验证失败：从计算得到的qpos计算的外参与输入的外参有较大差异")
                if max_diff >= 0.01:
                    print(f"    外参矩阵差异过大 (>= 0.01)")
                if position_error >= 0.01:
                    print(f"    相机位置误差过大 (>= 10mm)")
                if rot_error >= np.radians(5):
                    print(f"    相机旋转误差过大 (>= 5度)")
    
    except Exception as e:
        print(f"错误: 外参验证失败: {e}")
        import traceback
        traceback.print_exc()
    
    result['extrinsic_verification'] = extrinsic_verification
    
    # Step 4: 使用规划方法，验证能否从求解得到的qpos规划到初始qpos
    if verbose:
        print(f"\n{'='*80}")
        print(f"步骤4: 使用规划方法验证能否从求解得到的qpos回到初始qpos")
        print(f"  起点: 计算得到的qpos（来自时间步 {target_timestep} 的外参）")
        print(f"  终点: 初始qpos（时间步 {initial_timestep}）")
        print(f"{'='*80}")
    
    try:
        # 导入规划器
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from RoboFactory.utils.mplib_utils import FlexiblePlanner
        
        # 获取规划器（使用plan_new_pose.py中的配置）
        SRDF_PATH = os.path.join(os.path.dirname(urdf_filepath), 'panda_v2.srdf')
        
        LINK_NAMES = [
            "panda_link0", "panda_link1", "panda_link2", "panda_link3",
            "panda_link4", "panda_link5", "panda_link6", "panda_link7",
            "panda_link8", "panda_hand", "panda_hand_tcp",
            "panda_leftfinger", "panda_rightfinger"
        ]
        
        JOINT_NAMES = [
            "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
            "panda_joint5", "panda_joint6", "panda_joint7",
            "panda_finger_joint1", "panda_finger_joint2"
        ]
        
        planner = FlexiblePlanner(
            urdf=urdf_filepath,
            srdf=SRDF_PATH,
            user_link_names=LINK_NAMES,
            user_joint_names=JOINT_NAMES,
            move_group="panda_hand_tcp",
            joint_vel_limits=np.ones(7) * 1.0,
            joint_acc_limits=np.ones(7) * 1.0,
        )
        
        # 规划从计算得到的qpos到初始qpos
        plan_result = planner.plan_qpos_to_qpos(
            goal_qposes=[initial_qpos],
            current_qpos=computed_qpos,
            time_step=0.1,
            rrt_range=0.1,
            planning_time=planning_time,
            verbose=verbose
        )
        
        result['planning_status'] = plan_result['status']
        result['planning_success'] = (plan_result['status'] == 'Success')
        
        if result['planning_success']:
            result['path'] = plan_result.get('position')
            result['duration'] = plan_result.get('duration')
            if verbose:
                print(f"  ✓ 规划成功！")
                print(f"  路径形状: {result['path'].shape}")
                print(f"  持续时间: {result['duration']}")
        else:
            if verbose:
                print(f"  ✗ 规划失败")
                print(f"  状态: {result['planning_status']}")
                
    except Exception as e:
        print(f"错误: 规划失败: {e}")
        import traceback
        traceback.print_exc()
        result['planning_status'] = f"Planning error: {e}"
    
    # Step 5: 总结
    if verbose:
        print(f"\n{'='*80}")
        print(f"验证总结")
        print(f"{'='*80}")
        print(f"  初始qpos（时间步 {initial_timestep}）: {initial_qpos[:7]}")
        print(f"  计算得到的qpos（来自时间步 {target_timestep} 的外参）: {computed_qpos[:7] if computed_qpos is not None else 'N/A'}")
        print(f"  逆运动学求解: {'✓ 成功' if result['ik_success'] else '✗ 失败'}")
        
        # 外参验证结果
        if result.get('extrinsic_verification') is not None:
            ext_ver = result['extrinsic_verification']
            print(f"  外参验证: {'✓ 通过' if ext_ver.get('success', False) else '✗ 失败'}")
            if ext_ver.get('max_diff') is not None:
                print(f"    外参矩阵最大差异: {ext_ver['max_diff']:.10f}")
                print(f"    相机位置误差: {ext_ver.get('position_error_mm', 'N/A'):.4f} mm")
                print(f"    相机旋转误差: {ext_ver.get('rotation_error_deg', 'N/A'):.4f} 度")
        
        print(f"  规划验证: {'✓ 成功' if result['planning_success'] else '✗ 失败'}")
        
        # 综合评估
        all_passed = (result['ik_success'] and 
                     result.get('extrinsic_verification', {}).get('success', False) and
                     result['planning_success'])
        
        if all_passed:
            print(f"\n  ✓ 所有验证通过：")
            print(f"    - 逆运动学求解成功")
            print(f"    - 从计算得到的qpos计算的外参与输入外参一致")
            print(f"    - 可以从扰动的关节角度规划回到正常的角度")
        else:
            print(f"\n  ✗ 部分验证失败：")
            if not result['ik_success']:
                print(f"    - 逆运动学求解失败或未完全收敛")
            if result.get('extrinsic_verification', {}).get('success', True) == False:
                print(f"    - 从计算得到的qpos计算的外参与输入外参不一致")
            if not result['planning_success']:
                print(f"    - 无法从扰动的关节角度规划回到正常的角度")
                if result['planning_status']:
                    print(f"      原因: {result['planning_status']}")
        print(f"{'='*80}\n")
    
    return result


if __name__ == "__main__":
    urdf_filepath = "panda/panda_v2.urdf"
    robot_base_position = np.array([-0.2, -0.4, 0.0])
    camera_translation_relative_to_tcp = np.array([0.2, 0, -0.18])
    camera_rotation_relative_to_tcp = np.array([0.0, 0.707107, 0.0, 0.707107])
    
    # 使用H5文件验证从扰动的关节角度能否回到正常的角度
    h5_path = "data/trajectory.h5"
    traj_idx = 0
    initial_timestep = 0  # 第0个时间步
    target_timestep = 10  # 第10个时间步
    agent_idx = 0
    
    # 逆运动学参数（可调整以平衡精度和速度）
    # n_init_qpos: 随机尝试次数，越大越精确但越慢（推荐：20-100）
    # max_iterations: 每次优化的最大迭代次数（推荐：100-500）
    # tolerance: 收敛容差，越小越精确（推荐：1e-3 到 1e-5）
    
    result = validate_ik_with_planning(
        h5_path=h5_path,
        traj_idx=traj_idx,
        initial_timestep=initial_timestep,
        target_timestep=target_timestep,
        agent_idx=agent_idx,
        urdf_filepath=urdf_filepath,
        robot_base_position=robot_base_position,
        camera_translation_relative_to_tcp=camera_translation_relative_to_tcp,
        camera_rotation_relative_to_tcp=camera_rotation_relative_to_tcp,
        planning_time=5.0,
        n_init_qpos=100,  # 增加随机尝试次数以提高精度
        max_iterations=300,  # 增加迭代次数
        tolerance=1e-5,  # 降低容差以提高精度
        verbose=True
    )

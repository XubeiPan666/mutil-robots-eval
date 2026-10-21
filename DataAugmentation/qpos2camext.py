import pinocchio  #pip install pin
import numpy as np
from os.path import join, dirname, abspath
from scipy.spatial.transform import Rotation as R

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
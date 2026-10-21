#!/usr/bin/env python3
"""
从h5文件中提取数据，进行分割、修复和视角变换，生成新视角图像
"""

import h5py
import numpy as np
import cv2
import base64
import requests
import json
import os
import tempfile
import random
from pathlib import Path
from PIL import Image
import io
from typing import List, Tuple, Optional

# 导入qpos2camext模块
import sys
from pathlib import Path
# 添加当前目录到路径，以便导入同目录下的模块
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))
from qpos2camext import get_camera_extrinsics
from plan_new_pose import validate_qpos_perturbation
from camext2qpos import (
    get_qpos_from_camera_extrinsics,
    get_camera_extrinsics as get_camera_extrinsics_from_qpos_ik,
    read_h5_extrinsic as read_h5_extrinsic_camext
)
import pinocchio
from scipy.spatial.transform import Rotation as R


def load_h5_data(data):
    """递归加载h5数据到内存"""
    out = dict()
    for k in data.keys():
        if isinstance(data[k], h5py.Dataset):
            out[k] = data[k][:]
        else:
            out[k] = load_h5_data(data[k])
    return out


def read_h5_camera_data(h5_path: str, traj_idx: int, timestep: int, agent_idx: int, n: int = 3, step_size: int = 1) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    从h5文件中读取指定轨迹、时间步和agent的相机数据
    
    Args:
        h5_path: h5文件路径
        traj_idx: 轨迹序号
        timestep: 起始时间步t
        agent_idx: agent序号
        n: 需要提取的步数（包含t本身），默认3
        step_size: 时间步长a，提取的时间步为 t, t+a, t+2a, ...，默认1
    
    Returns:
        images: 图像列表，每个元素是numpy数组 [H, W, 3]
        intrinsics: 内参列表，每个元素是3x3矩阵
        extrinsics: 外参列表，每个元素是3x4矩阵
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
        max_needed_timestep = timestep + (n - 1) * step_size
        if max_needed_timestep > max_timestep:
            raise ValueError(f"Timestep {max_needed_timestep} (t={timestep} + {n-1}*{step_size}) exceeds max timestep {max_timestep}")
        
        images = []
        intrinsics = []
        extrinsics = []
        
        for i in range(n):
            if timestep >= 290: #FIXME 有点太规则了，需要修改
                t = timestep - i * step_size  
            else:
                t = timestep + i * step_size  # 提取 t, t+a, t+2a, ...
            # 读取图像 [H, W, 3]
            img = trajectory_data["obs"]["sensor_data"][camera_name]["rgb"][t]
            images.append(img)
            
            # 读取内参 [3, 3]
            intrinsic = trajectory_data["obs"]["sensor_param"][camera_name]["intrinsic_cv"][t]
            intrinsics.append(intrinsic)
            
            # 读取外参 [3, 4]
            extrinsic = trajectory_data["obs"]["sensor_param"][camera_name]["extrinsic_cv"][t]
            extrinsics.append(extrinsic)
    
    return images, intrinsics, extrinsics


def read_h5_qpos(h5_path: str, traj_idx: int, timestep: int, agent_idx: int) -> np.ndarray:
    """
    从h5文件中读取指定轨迹、时间步和agent的qpos
    
    Args:
        h5_path: h5文件路径
        traj_idx: 轨迹序号
        timestep: 时间步t
        agent_idx: agent序号
    
    Returns:
        qpos: 关节角度，numpy数组
    """
    with h5py.File(h5_path, 'r') as f:
        traj_key = f"traj_{traj_idx}"
        if traj_key not in f:
            raise ValueError(f"Trajectory {traj_idx} not found in h5 file")
        
        trajectory = f[traj_key]
        trajectory_data = load_h5_data(trajectory)
        
        # 尝试不同的路径来读取qpos
        qpos_paths = [
            f"obs/agent/panda-{agent_idx}/qpos",
            f"obs/agent/panda-{agent_idx}/qpos",
            f"actions/panda-{agent_idx}",  # 有时qpos在actions中
            f"obs/agent/qpos",
        ]
        
        qpos = None
        for path in qpos_paths:
            try:
                keys = path.split('/')
                data = trajectory_data
                for key in keys:
                    if key in data:
                        data = data[key]
                    else:
                        break
                else:
                    if isinstance(data, np.ndarray) and len(data) > timestep:
                        qpos = data[timestep]
                        break
            except (KeyError, IndexError, TypeError):
                continue
        
        if qpos is None:
            # 如果找不到，尝试直接从obs中读取
            try:
                if "obs" in trajectory_data and "agent" in trajectory_data["obs"]:
                    agent_key = f"panda-{agent_idx}"
                    if agent_key in trajectory_data["obs"]["agent"]:
                        if "qpos" in trajectory_data["obs"]["agent"][agent_key]:
                            qpos_data = trajectory_data["obs"]["agent"][agent_key]["qpos"]
                            if len(qpos_data) > timestep:
                                qpos = qpos_data[timestep]
            except (KeyError, IndexError, TypeError):
                pass
        
        if qpos is None:
            raise ValueError(f"Could not find qpos for trajectory {traj_idx}, timestep {timestep}, agent {agent_idx}")
        
        return np.array(qpos)


def compute_tgt_extrinsic_from_qpos(
    urdf_filepath: str,
    qpos: np.ndarray,
    robot_base_position: np.ndarray,
    camera_translation_relative_to_tcp: np.ndarray,
    camera_rotation_relative_to_tcp: np.ndarray
) -> np.ndarray:
    """
    使用qpos2camext计算目标外参
    
    Args:
        urdf_filepath: URDF文件路径
        qpos: 关节角度，9个维度（前7个是关节角度，后2个是夹爪状态）
        robot_base_position: 机器人基座位置，3个维度
        camera_translation_relative_to_tcp: 相机相对于TCP的平移，3个维度
        camera_rotation_relative_to_tcp: 相机相对于TCP的旋转，4个维度（wxyz格式）
    
    Returns:
        tgt_extrinsic: 目标外参，3x4矩阵
    """
    camera_params = get_camera_extrinsics(
        urdf_filepath,
        qpos,
        robot_base_position,
        camera_translation_relative_to_tcp,
        camera_rotation_relative_to_tcp
    )
    
    # 返回extrinsic_cv (3x4格式)
    return camera_params['extrinsic_cv']


def apply_gaussian_perturbation_to_extrinsic(
    extrinsic: np.ndarray,
    translation_std: float = 0.01,
    rotation_std: float = 0.01
) -> np.ndarray:
    """
    对相机外参施加高斯分布的平移和旋转扰动
    
    Args:
        extrinsic: 输入外参，3x4矩阵，格式为 [R^T | -R^T @ t]
            - R: 旋转矩阵（相机到世界）
            - t: 相机在世界坐标系中的位置
        translation_std: 平移扰动的标准差（米），默认0.01m (1cm)
        rotation_std: 旋转扰动的标准差（弧度），默认0.01rad (约0.57度)
    
    Returns:
        perturbed_extrinsic: 扰动后的外参，3x4矩阵
    """
    # 从外参矩阵提取旋转和平移
    # extrinsic = [R^T | -R^T @ t]
    R_world2cam = extrinsic[:3, :3]  # R^T (世界到相机)
    t_world2cam = extrinsic[:3, 3]    # -R^T @ t
    
    # 反推相机位置和旋转
    R_cam2world = R_world2cam.T  # 相机到世界的旋转
    camera_position = -R_cam2world @ t_world2cam  # 相机在世界坐标系中的位置
    
    # 1. 施加平移扰动（高斯分布）
    translation_noise = np.random.normal(0, translation_std, size=3)
    perturbed_camera_position = camera_position + translation_noise
    
    # 2. 施加旋转扰动（高斯分布）
    # 使用旋转向量（轴角表示）来施加扰动
    rotation_noise = np.random.normal(0, rotation_std, size=3)
    rotation_perturbation = R.from_rotvec(rotation_noise)
    
    # 将旋转矩阵转换为Rotation对象
    R_cam2world_original = R.from_matrix(R_cam2world)
    
    # 应用旋转扰动
    R_cam2world_perturbed = rotation_perturbation * R_cam2world_original
    R_cam2world_perturbed_matrix = R_cam2world_perturbed.as_matrix()
    
    # 3. 重新构建外参矩阵
    # extrinsic = [R^T | -R^T @ t]
    R_world2cam_perturbed = R_cam2world_perturbed_matrix.T
    t_world2cam_perturbed = -R_world2cam_perturbed @ perturbed_camera_position
    
    perturbed_extrinsic = np.zeros((3, 4))
    perturbed_extrinsic[:3, :3] = R_world2cam_perturbed
    perturbed_extrinsic[:3, 3] = t_world2cam_perturbed
    
    return perturbed_extrinsic


def segment_image(image: np.ndarray, segment_server_url: str = "http://localhost:8000") -> Tuple[np.ndarray, np.ndarray]:
    """
    对图像进行分割，获取mask
    
    Args:
        image: 输入图像，numpy数组 [H, W, 3]，uint8
        segment_server_url: 分割服务器URL
    
    Returns:
        mask: 分割mask，numpy数组 [H, W]，uint8，0或255
        mask_binary: 二值mask，numpy数组 [H, W]，bool
    """
    # 将图像编码为base64
    _, buffer = cv2.imencode('.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    
    # 发送请求
    url = f"{segment_server_url}/predict_base64"
    payload = {"image_base64": image_base64}
    
    response = requests.post(url, json=payload, timeout=60)
    
    if response.status_code != 200:
        raise RuntimeError(f"Segmentation failed: {response.status_code}, {response.text}")
    
    result = response.json()
    mask_base64 = result['mask_base64']
    
    # 解码mask
    mask_bytes = base64.b64decode(mask_base64)
    mask_image = Image.open(io.BytesIO(mask_bytes))
    mask = np.array(mask_image)
    
    # 如果mask是RGB，转换为灰度
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    
    # 确保mask是二值的（0或255）
    mask = (mask > 127).astype(np.uint8) * 255
    mask_binary = mask > 0
    
    return mask, mask_binary


def inpaint_image(image: np.ndarray, mask: np.ndarray, method: str = 'ns') -> np.ndarray:
    """
    使用inpaint修复图像中的mask区域
    
    Args:
        image: 输入图像，numpy数组 [H, W, 3]，uint8，RGB格式
        mask: 需要修复的区域mask，numpy数组 [H, W]，uint8，0或255
        method: inpaint方法，'telea' 或 'ns' (Navier-Stokes)
    
    Returns:
        inpainted_image: 修复后的图像，numpy数组 [H, W, 3]，uint8，RGB格式
    """
    # cv2.inpaint需要BGR格式
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # 选择inpaint方法
    if method == 'telea':
        inpaint_method = cv2.INPAINT_TELEA
    elif method == 'ns':
        inpaint_method = cv2.INPAINT_NS
    else:
        raise ValueError(f"Unknown inpaint method: {method}")
    
    # 执行inpaint
    inpainted_bgr = cv2.inpaint(image_bgr, mask, 3, inpaint_method)
    
    # 转换回RGB
    inpainted_image = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)
    
    return inpainted_image


def transform_viewpoint(images: List[np.ndarray], intrinsics: List[np.ndarray], 
                       extrinsics: List[np.ndarray], tgt_extrinsic: np.ndarray,
                       transform_server_url: str = "http://localhost:8000",
                       src_idx: int = 0) -> np.ndarray:
    """
    调用transform_client进行视角变换
    
    Args:
        images: 图像列表，每个元素是numpy数组 [H, W, 3]
        intrinsics: 内参列表，每个元素是3x3矩阵
        extrinsics: 外参列表，每个元素是3x4矩阵
        tgt_extrinsic: 目标外参，3x4矩阵
        transform_server_url: 变换服务器URL
        src_idx: 源图像索引，默认0（使用第一张图像）
    
    Returns:
        warped_image: 变换后的图像，numpy数组 [H, W, 3]
    """
    # 将图像编码为base64
    images_base64 = []
    for img in images:
        _, buffer = cv2.imencode('.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        images_base64.append(img_base64)
    
    # 准备内参（需要转换为列表格式，每个是3x3）
    align_cam_intrinsics = [intrinsic.tolist() for intrinsic in intrinsics]
    
    # 准备外参（需要转换为列表格式，每个是3x4）
    align_cam_extrinsics = [extrinsic.tolist() for extrinsic in extrinsics]
    
    # 准备目标外参
    tgt_cam_extrinsics = tgt_extrinsic.tolist()
    
    # 发送请求
    payload = {
        "images": images_base64,
        "src_idx": src_idx,
        "align_cam_intrinsics": align_cam_intrinsics,
        "align_cam_extrinsics": align_cam_extrinsics,
        "tgt_cam_extrinsics": tgt_cam_extrinsics
    }
    
    response = requests.post(
        f"{transform_server_url}/inference",
        json=payload,
        timeout=300  # 5分钟超时
    )
    
    if response.status_code != 200:
        raise RuntimeError(f"Transform failed: {response.status_code}, {response.text}")
    
    result = response.json()
    
    if not result.get('success'):
        raise RuntimeError(f"Transform failed: {result.get('message')}")
    
    # 解码变换后的图像
    warped_image_base64 = result['warped_image']
    warped_image_bytes = base64.b64decode(warped_image_base64)
    warped_image = Image.open(io.BytesIO(warped_image_bytes))
    warped_image = np.array(warped_image)
    
    # 如果图像是RGB，确保格式正确
    if len(warped_image.shape) == 3 and warped_image.shape[2] == 3:
        # 检查是否是BGR格式（cv2默认），如果是则转换为RGB
        # 这里假设服务器返回的是RGB格式
        pass
    
    return warped_image


def read_h5_image_at_timestep(h5_path: str, traj_idx: int, timestep: int, agent_idx: int) -> np.ndarray:
    """
    从h5文件中读取指定轨迹、时间步和agent的图像
    
    Args:
        h5_path: h5文件路径
        traj_idx: 轨迹序号
        timestep: 时间步
        agent_idx: agent序号
    
    Returns:
        image: 图像，numpy数组 [H, W, 3]，RGB格式
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
        
        # 读取图像 [H, W, 3]
        image = trajectory_data["obs"]["sensor_data"][camera_name]["rgb"][timestep]
    
    return image


def create_comparison_image(image1: np.ndarray, image2: np.ndarray, 
                            label1: str = "Original", label2: str = "New View",
                            padding: int = 10, font_scale: float = 1.0) -> np.ndarray:
    """
    创建两张图片的对比图（并排显示）
    
    Args:
        image1: 第一张图片，numpy数组 [H, W, 3]，RGB格式
        image2: 第二张图片，numpy数组 [H, W, 3]，RGB格式
        label1: 第一张图片的标签
        label2: 第二张图片的标签
        padding: 图片之间的间距和边距
        font_scale: 字体大小
    
    Returns:
        comparison_image: 对比图，numpy数组，RGB格式
    """
    # 确保两张图片尺寸相同（调整第二张图片）
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    
    # 统一高度（使用较大的高度）
    target_h = max(h1, h2)
    
    # 调整图片尺寸
    if h1 != target_h:
        image1 = cv2.resize(image1, (int(w1 * target_h / h1), target_h), interpolation=cv2.INTER_LINEAR)
    if h2 != target_h:
        image2 = cv2.resize(image2, (int(w2 * target_h / h2), target_h), interpolation=cv2.INTER_LINEAR)
    
    h, w = image1.shape[:2]
    w2 = image2.shape[1]
    
    # 创建对比图（并排）
    comparison_h = h + padding * 3 + 40  # 高度：图片 + 上下padding + 标签高度
    comparison_w = w + w2 + padding * 3  # 宽度：两张图片 + 中间padding + 左右padding
    
    comparison_image = np.ones((comparison_h, comparison_w, 3), dtype=np.uint8) * 255
    
    # 放置第一张图片
    y_start = padding + 40
    comparison_image[y_start:y_start+h, padding:padding+w] = image1
    
    # 放置第二张图片
    x2_start = padding + w + padding
    comparison_image[y_start:y_start+h, x2_start:x2_start+w2] = image2
    
    # 添加标签
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_thickness = 2
    text_color = (0, 0, 0)
    
    # 第一张图片的标签
    (text_width1, text_height1), _ = cv2.getTextSize(label1, font, font_scale, font_thickness)
    text_x1 = padding + (w - text_width1) // 2
    text_y1 = padding + 30
    cv2.putText(comparison_image, label1, (text_x1, text_y1), font, font_scale, text_color, font_thickness)
    
    # 第二张图片的标签
    (text_width2, text_height2), _ = cv2.getTextSize(label2, font, font_scale, font_thickness)
    text_x2 = x2_start + (w2 - text_width2) // 2
    text_y2 = padding + 30
    cv2.putText(comparison_image, label2, (text_x2, text_y2), font, font_scale, text_color, font_thickness)
    
    return comparison_image


def apply_mask_to_image(transformed_image: np.ndarray, original_image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    将原始图像中mask区域的像素直接添加到变换后的图像上
    
    Args:
        transformed_image: 变换后的图像，numpy数组 [H, W, 3]，uint8，RGB格式
        original_image: 原始图像（时间步t的图像），numpy数组 [H, W, 3]，uint8，RGB格式
        mask: 分割mask，numpy数组 [H, W]，uint8，0或255
    
    Returns:
        result_image: 合成后的图像，numpy数组 [H, W, 3]，uint8，RGB格式
    """
    result_image = transformed_image.copy()
    
    # 确保mask和图像尺寸匹配
    if mask.shape[:2] != original_image.shape[:2]:
        # 调整mask尺寸以匹配原始图像
        mask = cv2.resize(mask, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.uint8) * 255
    
    # 如果变换后的图像尺寸与原始图像不同，需要调整mask
    if mask.shape[:2] != transformed_image.shape[:2]:
        # 调整mask尺寸以匹配变换后的图像
        mask_resized = cv2.resize(mask, (transformed_image.shape[1], transformed_image.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask_resized = (mask_resized > 127).astype(np.uint8) * 255
        
        # 同时需要调整原始图像尺寸
        original_resized = cv2.resize(original_image, (transformed_image.shape[1], transformed_image.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        # 在mask区域使用原始图像的像素
        mask_binary = mask_resized > 0
        result_image[mask_binary] = original_resized[mask_binary]
    else:
        # 尺寸匹配，直接使用
        mask_binary = mask > 0
        result_image[mask_binary] = original_image[mask_binary]
    
    return result_image


def get_new_view_image(h5_path: str, traj_idx: int, timestep: int, agent_idx: int, 
                       n: int = 3,
                       step_size: int = 1,
                       segment_server_url: str = "http://localhost:8000",
                       transform_server_url: str = "http://localhost:8000",
                       tgt_extrinsic: Optional[np.ndarray] = None,
                       urdf_filepath: Optional[str] = None,
                       robot_base_position: Optional[np.ndarray] = None,
                       camera_translation_relative_to_tcp: Optional[np.ndarray] = None,
                       camera_rotation_relative_to_tcp: Optional[np.ndarray] = None,
                       read_extrinsic_from_h5: bool = False,
                       extrinsic_timestep_offset: int = 30,
                       translation_std: float = 0.01,
                       rotation_std: float = 0.01,
                       ik_n_init_qpos: int = 50,
                       ik_max_iterations: int = 200,
                       ik_tolerance: float = 1e-4,
                       max_perturbation_attempts: int = 100,
                       planning_time: float = 5.0,
                       return_perturbed_data: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[dict]]:
    """
    主函数：从h5文件生成新视角图像
    
    功能流程：
    1. 提取时间步t, t+a, t+2a, ...（共n步）的wrist_camera内参、外参和图像
    2. 计算目标外参（使用qpos2camext，如果提供了相关参数）
    3. 将时间步t的图片进行分割，用inpaint修复分割区域
    4. 将修复后的图和内参外参送入transform_client，得到变换后的视角图
    5. 将分割mask添加到变换后的视角图片上
    
    Args:
        h5_path: h5文件路径
        traj_idx: 轨迹序号
        timestep: 时间步t
        agent_idx: agent序号
        n: 提取的步数（包含t），默认3
        step_size: 时间步长a，提取的时间步为 t, t+a, t+2a, ...，默认1
        segment_server_url: 分割服务器URL
        transform_server_url: 变换服务器URL
        urdf_filepath: URDF文件路径（用于计算tgt_qpos）
        robot_base_position: 机器人基座位置，3个维度（用于计算tgt_qpos）
        camera_translation_relative_to_tcp: 相机相对于TCP的平移，3个维度（用于计算tgt_qpos）
        camera_rotation_relative_to_tcp: 相机相对于TCP的旋转，4个维度wxyz格式（用于计算tgt_qpos）
        read_extrinsic_from_h5: 是否从h5文件中读取外参，默认False
            - 如果True: 从h5文件读取 timestep + extrinsic_timestep_offset 时间步的外参
            - 如果False: 对当前timestep的外参加高斯扰动，并通过逆运动学求解qpos
        extrinsic_timestep_offset: 读取外参时的时间步偏移量（相对于timestep），默认30
        translation_std: 平移扰动的标准差（米），默认0.01m (1cm)，仅当read_extrinsic_from_h5=False时使用
        rotation_std: 旋转扰动的标准差（弧度），默认0.01rad (约0.57度)，仅当read_extrinsic_from_h5=False时使用
        ik_n_init_qpos: 逆运动学随机尝试次数，默认50
        ik_max_iterations: 逆运动学每次优化的最大迭代次数，默认200
        ik_tolerance: 逆运动学收敛容差，默认1e-4
        max_perturbation_attempts: 最大扰动尝试次数（如果验证失败会重试），默认100
        planning_time: 规划时间限制（秒），默认5.0
    
    Returns:
        final_image: 最终输出的图像，numpy数组 [H, W, 3]，uint8，RGB格式
        comparison_image: 对比图（如果read_extrinsic_from_h5为True），否则为None
        perturbed_data: 扰动数据字典（如果return_perturbed_data为True），包含:
            - 'extrinsic': 扰动后的相机外参 (3x4)
            - 'qpos': 扰动后的qpos (9维，如果read_extrinsic_from_h5=False)
            - 'translation_std': 平移扰动标准差（如果read_extrinsic_from_h5=False）
            - 'rotation_std': 旋转扰动标准差（如果read_extrinsic_from_h5=False）
            否则为None
    """
    # 步骤1: 提取相机数据
    print(f"[1/5] Extracting camera data from h5 file...")
    print(f"  Extracting timesteps: {[timestep + i * step_size for i in range(n)]}")
    images, intrinsics, extrinsics = read_h5_camera_data(h5_path, traj_idx, timestep, agent_idx, n, step_size)
    print(f"  ✓ Extracted {len(images)} images")

    
    # 步骤2: 计算目标外参
    print(f"[2/5] Computing target extrinsic...")
    
    qpos = None
    tgt_extrinsic = None
    
    if tgt_extrinsic is None:
        if read_extrinsic_from_h5:
            # 从h5文件读取指定时间步的外参
            extrinsic_timestep = timestep + extrinsic_timestep_offset
            try:
                tgt_extrinsic = read_h5_extrinsic_camext(h5_path, traj_idx, extrinsic_timestep, agent_idx)
                print(f"  ✓ Read extrinsic from h5 file at timestep {extrinsic_timestep}")
            except Exception as e:
                print(f"  ✗ Error: Could not read extrinsic from h5: {e}")
                tgt_extrinsic = extrinsics[-1]  # 使用最后一步的外参作为备选
                print(f"  → Using last step extrinsic instead")
        else:
            # 对当前timestep的外参加扰动，并通过逆运动学求解qpos
            if urdf_filepath is None or robot_base_position is None or \
               camera_translation_relative_to_tcp is None or camera_rotation_relative_to_tcp is None:
                raise ValueError(
                    "When read_extrinsic_from_h5=False, the following parameters are required: "
                    "urdf_filepath, robot_base_position, camera_translation_relative_to_tcp, "
                    "camera_rotation_relative_to_tcp"
                )
            
            base_extrinsic = extrinsics[0]  # 使用第一张图像的外参（时间步t）
            print(f"  → Applying Gaussian perturbation to extrinsic and solving IK...")
            print(f"     Base extrinsic from timestep {timestep}")
            print(f"     Translation std: {translation_std*1000:.2f} mm")
            print(f"     Rotation std: {np.degrees(rotation_std):.2f} degrees")
            
            # 读取当前时间步的qpos作为逆运动学的初始值和规划的目标
            initial_qpos = None
            target_qpos_for_planning = None
            try:
                initial_qpos = read_h5_qpos(h5_path, traj_idx, timestep, agent_idx)
                target_qpos_for_planning = initial_qpos.copy()  # 用于规划验证的目标qpos
                print(f"  ✓ Read initial qpos from timestep {timestep} for IK initialization and planning target")
            except Exception as e:
                print(f"  ⚠ Warning: Could not read initial qpos: {e}")
                initial_qpos = None
                target_qpos_for_planning = None
            
            # 循环尝试扰动，直到找到合理的qpos
            attempt = 0
            valid_qpos_found = False
            best_qpos = None
            best_extrinsic = None
            best_error = float('inf')
            best_planning_success = False
            
            while attempt < max_perturbation_attempts and not valid_qpos_found:
                attempt += 1
                if attempt > 1:
                    print(f"  → Retry attempt {attempt}/{max_perturbation_attempts}: Generating new perturbation...")
                
                # 1. 对基础外参加扰动
                perturbed_extrinsic = apply_gaussian_perturbation_to_extrinsic(
                    base_extrinsic,
                    translation_std=translation_std,
                    rotation_std=rotation_std
                )
                
                # 2. 通过逆运动学求解qpos
                try:
                    camera_extrinsics = {'extrinsic_cv': perturbed_extrinsic}
                    computed_qpos, ik_success = get_qpos_from_camera_extrinsics(
                        urdf_filepath,
                        camera_extrinsics,
                        robot_base_position,
                        camera_translation_relative_to_tcp,
                        camera_rotation_relative_to_tcp,
                        initial_qpos=initial_qpos,
                        max_iterations=ik_max_iterations,
                        tolerance=ik_tolerance,
                        n_init_qpos=ik_n_init_qpos,
                        ground_truth_qpos=None,
                        verbose=False  # 不输出详细信息，避免过多输出
                    )
                    
                    
                    
                    # 3. 验证qpos的合理性：从qpos计算外参，与扰动后的外参比较
                    camera_params_from_qpos = get_camera_extrinsics_from_qpos_ik(
                        urdf_filepath,
                        computed_qpos,
                        robot_base_position,
                        camera_translation_relative_to_tcp,
                        camera_rotation_relative_to_tcp
                    )
                    computed_extrinsic = camera_params_from_qpos['extrinsic_cv']
                    
                    # 计算误差
                    extrinsic_diff = np.abs(perturbed_extrinsic - computed_extrinsic)
                    max_diff = np.max(extrinsic_diff)
                    mean_diff = np.mean(extrinsic_diff)
                    
                    # 计算相机位置误差
                    R_world2cam_perturbed = perturbed_extrinsic[:3, :3]
                    t_world2cam_perturbed = perturbed_extrinsic[:3, 3]
                    camera_pos_perturbed = -R_world2cam_perturbed.T @ t_world2cam_perturbed
                    
                    R_world2cam_computed = computed_extrinsic[:3, :3]
                    t_world2cam_computed = computed_extrinsic[:3, 3]
                    camera_pos_computed = -R_world2cam_computed.T @ t_world2cam_computed
                    
                    position_error = np.linalg.norm(camera_pos_perturbed - camera_pos_computed)
                    
                    # 计算旋转误差
                    R_perturbed = R_world2cam_perturbed.T
                    R_computed = R_world2cam_computed.T
                    rot_error_vec = pinocchio.log3(R_perturbed.T @ R_computed)
                    rot_error = np.linalg.norm(rot_error_vec)
                    
                    # 判断外参验证是否通过：外参矩阵差异 < 0.01，位置误差 < 10mm，旋转误差 < 5度
                    extrinsic_valid = (max_diff < 0.01 and 
                                     position_error < 0.01 and 
                                     rot_error < np.radians(5))
                    
                    # 4. 如果外参验证通过，进行规划验证
                    planning_success = False
                    if extrinsic_valid and target_qpos_for_planning is not None:
                        try:
                            # 导入规划器
                            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
                            from RoboFactory.utils.mplib_utils import FlexiblePlanner
                            
                            # 获取规划器配置
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
                            
                            # 规划从扰动后的qpos到当前timestep的qpos
                            # 优先使用 RRTConnect，失败后再使用 RRTstar
                            plan_result = planner.plan_qpos_to_qpos(
                                goal_qposes=[target_qpos_for_planning],
                                current_qpos=computed_qpos,
                                time_step=0.1,
                                rrt_range=0.1,
                                planning_time=planning_time,
                                planner_name="RRTConnect",
                                verbose=False  # 不输出详细信息
                            )
                            
                            planning_success = (plan_result['status'] == 'Success')
                            
                            # 如果 RRTConnect 失败，尝试使用 RRTstar
                            if not planning_success:
                                print(f"    ⚠ RRTConnect failed (status: {plan_result.get('status', 'Unknown')}), trying RRTstar...")
                                plan_result = planner.plan_qpos_to_qpos(
                                    goal_qposes=[target_qpos_for_planning],
                                    current_qpos=computed_qpos,
                                    time_step=0.1,
                                    rrt_range=0.1,
                                    planning_time=planning_time,
                                    planner_name="RRTstar",
                                    verbose=False  # 不输出详细信息
                                )
                                planning_success = (plan_result['status'] == 'Success')
                                
                                if not planning_success:
                                    print(f"    ✗ RRTstar also failed (status: {plan_result.get('status', 'Unknown')}), retrying...")
                        except Exception as e:
                            print(f"    ✗ Planning error: {e}, retrying...")
                            planning_success = False
                    
                    # 综合判断：外参验证和规划验证都通过
                    is_valid = extrinsic_valid and planning_success
                    
                    # 记录最佳尝试（即使验证失败）
                    total_error = max_diff + position_error + rot_error
                    if total_error < best_error:
                        best_error = total_error
                        best_qpos = computed_qpos.copy()
                        best_extrinsic = perturbed_extrinsic.copy()
                        best_planning_success = planning_success
                    
                    if is_valid:
                        tgt_extrinsic = perturbed_extrinsic
                        qpos = computed_qpos
                        valid_qpos_found = True
                        print(f"  ✓ Valid perturbation found after {attempt} attempt(s)")
                        print(f"     Max extrinsic diff: {max_diff:.6f}")
                        print(f"     Position error: {position_error*1000:.2f} mm")
                        print(f"     Rotation error: {np.degrees(rot_error):.2f} degrees")
                        print(f"     Planning: ✓ Success")
                    else:
                        if attempt < max_perturbation_attempts:
                            if not extrinsic_valid:
                                print(f"    ✗ Extrinsic validation failed (max_diff={max_diff:.6f}, pos_err={position_error*1000:.2f}mm, rot_err={np.degrees(rot_error):.2f}deg), retrying...")
                            elif not planning_success:
                                print(f"    ✗ Planning validation failed, retrying...")
                        else:
                            print(f"    ✗ Validation failed after {max_perturbation_attempts} attempts")
                            if best_qpos is not None:
                                print(f"     Using best attempt (total error: {best_error:.6f}, planning: {'✓' if best_planning_success else '✗'})")
                                # 使用最佳尝试的结果
                                tgt_extrinsic = best_extrinsic
                                qpos = best_qpos
                                valid_qpos_found = True
                            else:
                                print(f"     No valid attempt found, using base extrinsic")
                                tgt_extrinsic = base_extrinsic
                                valid_qpos_found = True  # 标记为完成，避免继续循环
                
                except Exception as e:
                    print(f"    ✗ Error during IK or validation: {e}")
                    if attempt >= max_perturbation_attempts:
                        # 最后一次尝试失败
                        if best_qpos is not None:
                            print(f"  ⚠ Warning: Failed after {max_perturbation_attempts} attempts, using best attempt")
                            tgt_extrinsic = best_extrinsic
                            qpos = best_qpos
                            valid_qpos_found = True
                        else:
                            print(f"  ⚠ Warning: Failed after {max_perturbation_attempts} attempts, using base extrinsic")
                            tgt_extrinsic = base_extrinsic
                            valid_qpos_found = True  # 标记为完成，避免继续循环
                        break
                    continue
            
            if not valid_qpos_found:
                print(f"  ⚠ Warning: Could not find valid qpos after {max_perturbation_attempts} attempts")
                if best_qpos is not None:
                    print(f"     Using best attempt found")
                    tgt_extrinsic = best_extrinsic
                    qpos = best_qpos
                else:
                    tgt_extrinsic = base_extrinsic
    else:
        print(f"  ✓ Using provided tgt_extrinsic")
    
    # 收集扰动数据（如果需要返回）
    perturbed_data = None
    if return_perturbed_data and tgt_extrinsic is not None:
        perturbed_data = {
            'extrinsic': tgt_extrinsic.copy() if isinstance(tgt_extrinsic, np.ndarray) else np.array(tgt_extrinsic)
        }
        if qpos is not None:
            perturbed_data['qpos'] = qpos.copy() if isinstance(qpos, np.ndarray) else np.array(qpos)
        print(f"  ✓ Collected perturbed data: extrinsic shape {perturbed_data['extrinsic'].shape}")
    
    # 步骤3: 分割和修复
    print(f"[3/5] Segmenting and inpainting images at timestep {timestep}...")
    image_t = images[0]  # 时间步t的图像
    
    # 分割和修复：对每张图片分别进行分割，然后用各自的mask进行inpaint
    inpainted_images = []
    masks = []  # 保存所有mask，用于后续步骤
    
    for i, img in enumerate(images):
        # print(f"  Processing image {i+1}/{len(images)}...")
        
        # 对每张图片进行分割，获取各自的mask
        img_mask, img_mask_binary = segment_image(img, segment_server_url)
        masks.append(img_mask)
        # print(f"    ✓ Segmentation completed for image {i+1}, mask shape: {img_mask.shape}")
        
        # 使用各自的mask对各自的图片进行inpaint
        inpainted_img = inpaint_image(img, img_mask)
        inpainted_images.append(inpainted_img)
        # print(f"    ✓ Inpainting completed for image {i+1}")
    
    # 保存第一张图片的mask，用于后续步骤5（添加mask到最终图像）
    mask = masks[0]
    print(f"  ✓ All {len(inpainted_images)} images segmented and inpainted")
    
    # 步骤4: 视角变换
    print(f"[4/5] Transforming viewpoint...")
    
    # INSERT_YOUR_CODE
    # 临时代码：将inpainted_images换成@DataAugmentation/test_inpainted文件夹中的三张图片
    # import os
    # from PIL import Image

    # test_inpainted_dir = os.path.join(os.path.dirname(__file__), "test_inpainted")
    # test_img_names = sorted([f for f in os.listdir(test_inpainted_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])[:3]
    # images_for_transform = []
    # for fname in test_img_names:
    #     img_path = os.path.join(test_inpainted_dir, fname)
    #     img = np.array(Image.open(img_path).convert("RGB"))
    #     images_for_transform.append(img)
    # print(f"  [Test] Loaded {len(images_for_transform)} test inpainted images from {test_inpainted_dir}.")
    # 使用修复后的图像
    images_for_transform = inpainted_images
    
    
    # 调用视角变换
    warped_image = transform_viewpoint(
        images_for_transform, 
        intrinsics, 
        extrinsics, 
        tgt_extrinsic,
        transform_server_url,
        src_idx=0
    )
    print(f"  ✓ Viewpoint transformation completed")
    
    # 步骤5: 添加分割mask
    print(f"[5/5] Applying mask to transformed image...")
    final_image = apply_mask_to_image(warped_image, image_t, mask)
    print(f"  ✓ Final image generated")
    
    # 如果read_extrinsic_from_h5为True，提取对应时间步的图片并创建对比图
    comparison_image = None
    if read_extrinsic_from_h5:
        extrinsic_timestep = timestep + extrinsic_timestep_offset
        try:
            print(f"\n[Extra] Creating comparison image...")
            original_image_at_extrinsic = read_h5_image_at_timestep(h5_path, traj_idx, extrinsic_timestep, agent_idx)
            print(f"  ✓ Read original image at timestep {extrinsic_timestep}")
            
            comparison_image = create_comparison_image(
                original_image_at_extrinsic,
                final_image,
                label1=f"Original (t={extrinsic_timestep})",
                label2="New View (Generated)"
            )
            print(f"  ✓ Comparison image created")
        except Exception as e:
            print(f"  ⚠ Warning: Could not create comparison image: {e}")
            comparison_image = None
    
    return final_image, comparison_image, perturbed_data


if __name__ == "__main__":
    
    #要想让da3尺度对齐准确，必须要要传入的图片的差异大，即相机的位置尽量不要在一条基线上
    #小扰动时间步验证，步长差10步内吧，验证到delta=10步时的视角生成差
    #注意处理双臂或多臂数据时，有些臂在几百步后就会停止运动，而有些臂在几百步后才刚刚开始运动，尺度对齐需要运动的相机
    
    # 示例用法
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate new view image from h5 file')
    parser.add_argument('--h5_path', type=str, required=True, help='h5文件路径')
    parser.add_argument('--traj_idx', type=int, required=True, help='轨迹序号')
    parser.add_argument('--timestep', type=int, required=True, help='时间步t')
    parser.add_argument('--agent_idx', type=int, default=0, help='agent序号')
    parser.add_argument('--n', type=int, default=3, help='提取的步数（包含t）')
    parser.add_argument('--step_size', type=int, default=10, help='时间步长a，提取的时间步为 t, t+a, t+2a, ...，默认10')
    parser.add_argument('--segment_server', type=str, default=os.getenv('SEGMENT_SERVER_URL', 'http://localhost:8000'), help='分割服务器URL')
    parser.add_argument('--transform_server', type=str, default=os.getenv('TRANSFORM_SERVER_URL', 'http://localhost:8001'), help='变换服务器URL')
    parser.add_argument('--output', type=str, default='outputs/new_view_image.png', help='输出图像路径')
    
    # qpos2camext相关参数
    parser.add_argument('--urdf_filepath', type=str, default=None, help='URDF文件路径（用于计算tgt_extrinsic）')
    parser.add_argument('--robot_base_position', type=float, nargs=3, default=None, help='机器人基座位置 [x, y, z]')
    parser.add_argument('--camera_translation_relative_to_tcp', type=float, nargs=3, default=None, help='相机相对于TCP的平移 [x, y, z]')
    parser.add_argument('--camera_rotation_relative_to_tcp', type=float, nargs=4, default=None, help='相机相对于TCP的旋转 [w, x, y, z] (wxyz格式)')
    # 外参相关参数
    parser.add_argument('--read_extrinsic_from_h5', action='store_true', help='从h5文件中读取外参（如果False，则对当前timestep的外参加扰动）')
    parser.add_argument('--extrinsic_timestep_offset', type=int, default=1, help='读取外参时的时间步偏移量（相对于timestep），默认30')
    parser.add_argument('--translation_std', type=float, default=0.01, help='平移扰动的标准差（米），默认0.01m (1cm)')
    parser.add_argument('--rotation_std', type=float, default=0.01, help='旋转扰动的标准差（弧度），默认0.01rad (约0.57度)')
    
    # 逆运动学参数（当read_extrinsic_from_h5=False时使用）
    parser.add_argument('--ik_n_init_qpos', type=int, default=100, help='逆运动学随机尝试次数，默认50')
    parser.add_argument('--ik_max_iterations', type=int, default=300, help='逆运动学每次优化的最大迭代次数，默认200')
    parser.add_argument('--ik_tolerance', type=float, default=1e-5, help='逆运动学收敛容差，默认1e-4')
    parser.add_argument('--max_perturbation_attempts', type=int, default=100, help='最大扰动尝试次数，默认100')
    parser.add_argument('--planning_time', type=float, default=5.0, help='规划时间限制（秒），默认5.0')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # 转换参数格式
    robot_base_position = np.array(args.robot_base_position) if args.robot_base_position is not None else None
    camera_translation = np.array(args.camera_translation_relative_to_tcp) if args.camera_translation_relative_to_tcp is not None else None
    camera_rotation = np.array(args.camera_rotation_relative_to_tcp) if args.camera_rotation_relative_to_tcp is not None else None
    
    # 生成新视角图像
    final_image, comparison_image, perturbed_data = get_new_view_image(
        args.h5_path,
        args.traj_idx,
        args.timestep,
        args.agent_idx,
        args.n,
        args.step_size,
        args.segment_server,
        args.transform_server,
        tgt_extrinsic=None,  # 将根据参数计算
        urdf_filepath=args.urdf_filepath,
        robot_base_position=robot_base_position,
        camera_translation_relative_to_tcp=camera_translation,
        camera_rotation_relative_to_tcp=camera_rotation,
        read_extrinsic_from_h5=args.read_extrinsic_from_h5,
        extrinsic_timestep_offset=args.extrinsic_timestep_offset,
        translation_std=args.translation_std,
        rotation_std=args.rotation_std,
        ik_n_init_qpos=args.ik_n_init_qpos,
        ik_max_iterations=args.ik_max_iterations,
        ik_tolerance=args.ik_tolerance,
        max_perturbation_attempts=args.max_perturbation_attempts,
        planning_time=args.planning_time,
        return_perturbed_data=True
    )
    
    # 保存结果
    Image.fromarray(final_image).save(args.output)
    print(f"\n✓ Final image saved to: {args.output}")
    
    # 如果生成了对比图，保存对比图
    if comparison_image is not None:
        comparison_output = args.output.replace('.png', '_comparison.png').replace('.jpg', '_comparison.jpg')
        Image.fromarray(comparison_image).save(comparison_output)
        print(f"✓ Comparison image saved to: {comparison_output}")



# python get_new_view_image.py \
#     --h5_path data/trajectory.h5 \
#     --traj_idx 0 \
#     --timestep 10 \
#     --agent_idx 0 \
#     --urdf_filepath panda/panda_v2.urdf \
#     --robot_base_position -0.2 -0.4 0.0 \
#     --camera_translation_relative_to_tcp 0.2 0 -0.18 \
#     --camera_rotation_relative_to_tcp 0.0 0.707107 0.0 0.707107 --joint_indices 0 2 4 --perturbations 0.1 -0.1 0.05
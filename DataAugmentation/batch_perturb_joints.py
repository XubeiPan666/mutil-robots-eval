#!/usr/bin/env python3
"""
批量处理脚本：对h5文件中的多个轨迹的多个时间步应用joint扰动，生成新视角图像
并将结果保存到新的h5文件中，同时生成可视化视频
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import numpy as np
import h5py
import cv2
from tqdm import tqdm

# 导入get_new_view_image
import sys
from pathlib import Path
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))
from get_new_view_image import get_new_view_image, load_h5_data


def copy_h5_and_replace_data(
    src_h5_path: str,
    dst_h5_path: str,
    traj_idx: int,
    agent_idx: int,
    timestep_data: Dict[int, Dict[str, Any]],
    verbose: bool = False
) -> None:
    """
    复制h5文件并替换指定时间步的图像、qpos和相机外参
    
    Args:
        src_h5_path: 源h5文件路径
        dst_h5_path: 目标h5文件路径
        traj_idx: 轨迹序号
        agent_idx: agent序号
        timestep_data: 字典，key为timestep，value为包含以下字段的字典:
            - 'image': 新生成的图像 (H, W, 3)
            - 'qpos': 扰动后的qpos (9维)
            - 'extrinsic': 扰动后的相机外参 (3x4)
        verbose: 是否输出详细信息
    """
    if verbose:
        tqdm.write(f"\nCopying h5 file and replacing data...")
        tqdm.write(f"  Source: {src_h5_path}")
        tqdm.write(f"  Target: {dst_h5_path}")
        tqdm.write(f"  Trajectory {traj_idx}, agent {agent_idx}")
        tqdm.write(f"  Replacing {len(timestep_data)} timesteps (image + qpos + extrinsic)")
    
    camera_name = f"wrist_camera_agent{agent_idx}"
    agent_name = f"panda-{agent_idx}"
    
    # 复制h5文件
    if src_h5_path != dst_h5_path:
        shutil.copy2(src_h5_path, dst_h5_path)
    
    # 打开目标h5文件进行修改
    with h5py.File(dst_h5_path, 'r+') as f:
        traj_key = f"traj_{traj_idx}"
        if traj_key not in f:
            raise ValueError(f"Trajectory {traj_idx} not found in h5 file")
        
        trajectory = f[traj_key]
        
        # 获取各个数据集的引用
        try:
            rgb_dataset = trajectory["obs"]["sensor_data"][camera_name]["rgb"]
        except KeyError as e:
            raise ValueError(f"Camera rgb path not found in trajectory {traj_idx}: {e}")
        
        try:
            # qpos_dataset = trajectory["obs"]["agent"][agent_name]["qpos"]
            qpos_dataset = trajectory["obs"]["agent"]["qpos"]
        except KeyError as e:
            raise ValueError(f"Agent qpos path not found in trajectory {traj_idx}: {e}")
        
        try:
            extrinsic_dataset = trajectory["obs"]["sensor_param"][camera_name]["extrinsic_cv"]
        except KeyError as e:
            raise ValueError(f"Camera extrinsic path not found in trajectory {traj_idx}: {e}")
        
        # 替换数据
        for timestep, data in timestep_data.items():
            # 检查时间步范围
            if timestep >= len(rgb_dataset):
                if verbose:
                    tqdm.write(f"  ⚠ Warning: Timestep {timestep} exceeds dataset length {len(rgb_dataset)}, skipping")
                continue
            
            # 1. 替换图像
            if 'image' in data and data['image'] is not None:
                new_image = data['image']
                if new_image.dtype != np.uint8:
                    new_image = (np.clip(new_image, 0, 255)).astype(np.uint8)
                rgb_dataset[timestep] = new_image
            
            # 2. 替换qpos
            if 'qpos' in data and data['qpos'] is not None:
                new_qpos = data['qpos']
                if isinstance(new_qpos, np.ndarray):
                    new_qpos = new_qpos.astype(np.float32)
                qpos_dataset[timestep] = new_qpos
            
            # 3. 替换相机外参
            if 'extrinsic' in data and data['extrinsic'] is not None:
                new_extrinsic = data['extrinsic']
                if isinstance(new_extrinsic, np.ndarray):
                    new_extrinsic = new_extrinsic.astype(np.float32)
                extrinsic_dataset[timestep] = new_extrinsic
    
    if verbose:
        tqdm.write(f"  ✓ H5 file saved with updated image, qpos, and extrinsic: {dst_h5_path}")


def create_visualization_video(
    h5_path: str,
    traj_idx: int,
    output_video_path: str,
    fps: int = 10,
    verbose: bool = False
) -> None:
    """
    从h5文件创建可视化视频，每个时间步显示3张图片（wrist_cam0, wrist_cam1, global_cam）
    
    Args:
        h5_path: h5文件路径
        traj_idx: 轨迹序号
        output_video_path: 输出视频路径
        fps: 视频帧率
        verbose: 是否输出详细信息
    """
    if verbose:
        tqdm.write(f"\nCreating visualization video...")
        tqdm.write(f"  H5 file: {h5_path}")
        tqdm.write(f"  Trajectory: {traj_idx}")
        tqdm.write(f"  Output: {output_video_path}")
    
    # 检查h5文件是否存在
    if not os.path.exists(h5_path):
        raise FileNotFoundError(
            f"H5 file not found: {h5_path}\n"
            f"Please ensure the file exists before creating visualization video."
        )
    
    camera_names = ["wrist_camera_agent0", "wrist_camera_agent1", "global_camera"]
    
    with h5py.File(h5_path, 'r') as f:
        traj_key = f"traj_{traj_idx}"
        if traj_key not in f:
            raise ValueError(f"Trajectory {traj_idx} not found in h5 file")
        
        trajectory = f[traj_key]
        trajectory_data = load_h5_data(trajectory)
        
        # 查找所有可用的相机
        available_cameras = []
        for cam_name in camera_names:
            try:
                sensor_data = trajectory_data.get("obs", {}).get("sensor_data", {})
                if cam_name in sensor_data and "rgb" in sensor_data[cam_name]:
                    available_cameras.append(cam_name)
            except (KeyError, TypeError):
                continue
        
        if not available_cameras:
            raise ValueError(f"No cameras found in trajectory {traj_idx}")
        
        # 获取时间步数量（使用第一个可用相机的长度）
        first_cam = available_cameras[0]
        num_timesteps = len(trajectory_data["obs"]["sensor_data"][first_cam]["rgb"])
        
        # 读取所有图像（使用进度条）
        frames = []
        frame_pbar = tqdm(range(num_timesteps), desc=f"Reading frames (Traj {traj_idx})", 
                         leave=False, unit="frame", ncols=100)
        
        for t in frame_pbar:
            # 读取每个相机的图像
            images = []
            for cam_name in camera_names:
                if cam_name in available_cameras:
                    img = trajectory_data["obs"]["sensor_data"][cam_name]["rgb"][t]
                    # 确保图像是RGB格式，uint8
                    if img.dtype != np.uint8:
                        img = (np.clip(img, 0, 255)).astype(np.uint8)
                    images.append(img)
            
            if not images:
                continue
            
            # 将图像水平拼接成一排
            # 统一高度
            heights = [img.shape[0] for img in images]
            target_h = max(heights)
            
            # 调整所有图像到相同高度
            resized_images = []
            for img in images:
                if img.shape[0] != target_h:
                    scale = target_h / img.shape[0]
                    new_w = int(img.shape[1] * scale)
                    img = cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_LINEAR)
                resized_images.append(img)
            
            # 水平拼接
            combined_image = np.hstack(resized_images)
            frames.append(combined_image)
        
        frame_pbar.close()
        
        if not frames:
            raise ValueError(f"No frames to write to video")
        
        # 获取视频尺寸
        h, w = frames[0].shape[:2]
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))
        
        # 写入所有帧（注意OpenCV使用BGR格式）
        for frame in frames:
            # 确保是RGB格式，转换为BGR
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame
            out.write(frame_bgr)
        
        out.release()
    
    if verbose:
        tqdm.write(f"  ✓ Video saved: {output_video_path}")


def process_single_trajectory(
    h5_path: str,
    traj_idx: int,
    agent_idx: int,
    start_timestep: int,
    end_timestep: int,
    urdf_filepath: Optional[str] = None,
    robot_base_position: Optional[np.ndarray] = None,
    camera_translation_relative_to_tcp: Optional[np.ndarray] = None,
    camera_rotation_relative_to_tcp: Optional[np.ndarray] = None,
    n: int = 3,
    step_size: int = 10,
    segment_server_url: str = "http://localhost:8000",
    transform_server_url: str = "http://localhost:8000",
    read_extrinsic_from_h5: bool = False,
    extrinsic_timestep_offset: int = 30,
    translation_std: float = 0.01,
    rotation_std: float = 0.01,
    ik_n_init_qpos: int = 50,
    ik_max_iterations: int = 200,
    ik_tolerance: float = 1e-4,
    max_perturbation_attempts: int = 100,
    planning_time: float = 5.0
) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Any]]:
    """
    处理单个轨迹的所有时间步，生成新视角图像和扰动数据
    
    Args:
        h5_path: h5文件路径
        traj_idx: 轨迹序号
        agent_idx: agent序号
        start_timestep: 起始时间步
        end_timestep: 结束时间步（包含）
        urdf_filepath: URDF文件路径
        robot_base_position: 机器人基座位置
        camera_translation_relative_to_tcp: 相机相对于TCP的平移
        camera_rotation_relative_to_tcp: 相机相对于TCP的旋转
        n: 提取的步数
        step_size: 时间步长
        segment_server_url: 分割服务器URL
        transform_server_url: 变换服务器URL
        read_extrinsic_from_h5: 是否从h5文件中读取外参，默认False
        extrinsic_timestep_offset: 读取外参时的时间步偏移量（相对于timestep），默认30
        translation_std: 平移扰动的标准差（米），默认0.01m (1cm)
        rotation_std: 旋转扰动的标准差（弧度），默认0.01rad (约0.57度)
        ik_n_init_qpos: 逆运动学随机尝试次数，默认50
        ik_max_iterations: 逆运动学每次优化的最大迭代次数，默认200
        ik_tolerance: 逆运动学收敛容差，默认1e-4
        max_perturbation_attempts: 最大扰动尝试次数，默认100
        planning_time: 规划时间限制（秒），默认5.0
    
    Returns:
        generated_data: 字典，key为timestep，value为包含以下字段的字典:
            - 'image': 生成的图像
            - 'qpos': 扰动后的qpos
            - 'extrinsic': 扰动后的相机外参
        results: 处理结果字典，包含成功和失败的时间步
    """
    generated_data = {}
    results = {
        'traj_idx': traj_idx,
        'agent_idx': agent_idx,
        'success': [],
        'failed': []
    }
    
    print(f"\n{'='*60}")
    print(f"Processing trajectory {traj_idx}, agent {agent_idx}")
    print(f"  Time range: {start_timestep} to {end_timestep}")
    print(f"  Read extrinsic from h5: {read_extrinsic_from_h5}")
    if not read_extrinsic_from_h5:
        print(f"  Translation std: {translation_std*1000:.2f} mm")
        print(f"  Rotation std: {np.degrees(rotation_std):.2f} degrees")
    else:
        print(f"  Extrinsic timestep offset: {extrinsic_timestep_offset}")
    print(f"{'='*60}\n")
    
    # 遍历每个时间步（使用进度条）
    timesteps = list(range(start_timestep, end_timestep + 1))
    pbar = tqdm(timesteps, desc=f"Traj {traj_idx}, Agent {agent_idx}", unit="timestep", ncols=100)
    
    for timestep in pbar:
        pbar.set_postfix({"Current": timestep, "Success": len(results['success']), "Failed": len(results['failed'])})
        
        try:
            # 生成新视角图像和扰动数据
            final_image, _, perturbed_data = get_new_view_image(
                h5_path=h5_path,
                traj_idx=traj_idx,
                timestep=timestep,
                agent_idx=agent_idx,
                n=n,
                step_size=step_size,
                segment_server_url=segment_server_url,
                transform_server_url=transform_server_url,
                tgt_extrinsic=None,
                urdf_filepath=urdf_filepath,
                robot_base_position=robot_base_position,
                camera_translation_relative_to_tcp=camera_translation_relative_to_tcp,
                camera_rotation_relative_to_tcp=camera_rotation_relative_to_tcp,
                read_extrinsic_from_h5=read_extrinsic_from_h5,
                extrinsic_timestep_offset=extrinsic_timestep_offset,
                translation_std=translation_std,
                rotation_std=rotation_std,
                ik_n_init_qpos=ik_n_init_qpos,
                ik_max_iterations=ik_max_iterations,
                ik_tolerance=ik_tolerance,
                max_perturbation_attempts=max_perturbation_attempts,
                planning_time=planning_time,
                return_perturbed_data=True  # 请求返回扰动数据
            )
            
            # 收集完整的扰动数据
            generated_data[timestep] = {
                'image': final_image,
                'qpos': perturbed_data['qpos'] if perturbed_data else None,
                'extrinsic': perturbed_data['extrinsic'] if perturbed_data else None
            }
            results['success'].append(timestep)
            
        except Exception as e:
            error_msg = f"Failed at timestep {timestep}: {str(e)}"
            results['failed'].append({
                'timestep': timestep,
                'error': str(e)
            })
    
    pbar.close()
    
    # 打印汇总
    print(f"\n{'='*60}")
    print(f"Trajectory {traj_idx}, agent {agent_idx} completed:")
    print(f"  Success: {len(results['success'])}/{end_timestep - start_timestep + 1}")
    print(f"  Failed: {len(results['failed'])}/{end_timestep - start_timestep + 1}")
    print(f"{'='*60}\n")
    
    return generated_data, results


def process_multiple_trajectories(
    h5_path: str,
    configs: List[Dict[str, Any]],
    output_dir: str,
    urdf_filepath: Optional[str] = None,
    robot_base_position: Optional[np.ndarray] = None,
    camera_translation_relative_to_tcp: Optional[np.ndarray] = None,
    camera_rotation_relative_to_tcp: Optional[np.ndarray] = None,
    n: int = 3,
    step_size: int = 10,
    segment_server_url: str = "http://localhost:8000",
    transform_server_url: str = "http://localhost:8000",
    read_extrinsic_from_h5: bool = False,
    extrinsic_timestep_offset: int = 30,
    translation_std: float = 0.01,
    rotation_std: float = 0.01,
    ik_n_init_qpos: int = 50,
    ik_max_iterations: int = 200,
    ik_tolerance: float = 1e-4,
    max_perturbation_attempts: int = 100,
    planning_time: float = 5.0
) -> List[Dict[str, Any]]:
    """
    批量处理多个轨迹，生成新h5文件和可视化视频
    每处理完一段轨迹就立即保存该轨迹的可视化视频，h5文件在最后统一保存
    
    Args:
        h5_path: h5文件路径
        configs: 配置列表，每个配置包含 traj_idx, agent_idx, start_timestep, end_timestep
        output_dir: 输出目录
        urdf_filepath: URDF文件路径
        robot_base_position: 机器人基座位置
        camera_translation_relative_to_tcp: 相机相对于TCP的平移
        camera_rotation_relative_to_tcp: 相机相对于TCP的旋转
        n: 提取的步数
        step_size: 时间步长
        segment_server_url: 分割服务器URL
        transform_server_url: 变换服务器URL
        read_extrinsic_from_h5: 是否从h5文件中读取外参，默认False
        extrinsic_timestep_offset: 读取外参时的时间步偏移量（相对于timestep），默认30
        translation_std: 平移扰动的标准差（米），默认0.01m (1cm)
        rotation_std: 旋转扰动的标准差（弧度），默认0.01rad (约0.57度)
        ik_n_init_qpos: 逆运动学随机尝试次数，默认50
        ik_max_iterations: 逆运动学每次优化的最大迭代次数，默认200
        ik_tolerance: 逆运动学收敛容差，默认1e-4
        max_perturbation_attempts: 最大扰动尝试次数，默认100
        planning_time: 规划时间限制（秒），默认5.0
    
    Returns:
        all_results: 所有轨迹的处理结果列表
    """
    all_results = []
    
    # 用于存储每个轨迹和agent的生成的图像 {(traj_idx, agent_idx): {timestep: image}}
    traj_images = {}
    
    # 用于跟踪已完成的轨迹（已创建视频的轨迹）
    completed_trajectories = set()
    
    # # 创建统一的h5文件路径
    # output_h5_path = os.path.join(output_dir, "perturbed.h5")
    
    # # 先复制原始文件（只复制一次）
    # if not os.path.exists(output_h5_path):
    #     try:
    #         shutil.copy2(h5_path, output_h5_path)
    #         print(f"\n  ✓ Copied original h5 file to: {output_h5_path}")
    #     except Exception as e:
    #         raise FileNotFoundError(
    #             f"Failed to copy h5 file from {h5_path} to {output_h5_path}: {e}\n"
    #             f"Please check that the source file exists and you have write permissions to the output directory."
    #         )
    
    # # 验证输出h5文件确实存在
    # if not os.path.exists(output_h5_path):
    #     raise FileNotFoundError(
    #         f"Output h5 file does not exist after copy attempt: {output_h5_path}\n"
    #         f"Please check file permissions and disk space."
    #     )
    
    print(f"\n{'#'*60}")
    print(f"Starting batch processing of {len(configs)} trajectory configurations")
    print(f"Output directory: {output_dir}")
    print(f"{'#'*60}\n")
    
    # 处理所有配置，生成图像，并在每个轨迹完成后立即创建视频
    # 计算总时间步数（用于整体进度显示）
    total_timesteps = sum(config['end_timestep'] - config['start_timestep'] + 1 for config in configs)
    processed_timesteps = 0
    
    config_pbar = tqdm(configs, desc="Processing configurations", unit="config", ncols=100, position=0)
    
    for config_idx, config in enumerate(config_pbar):
        # 提取配置参数
        traj_idx = config['traj_idx']
        agent_idx = config['agent_idx']
        start_timestep = config['start_timestep']
        end_timestep = config['end_timestep']
        # 注意：joint_indices 和 perturbations 已不再使用，保留以支持向后兼容
        # 如果配置文件中存在这些字段，它们将被忽略

        # 创建统一的h5文件路径
        output_h5_path = os.path.join(output_dir, f"perturbed_{config_idx}.h5")
        
        # 先复制原始文件（只复制一次）
        if not os.path.exists(output_h5_path):
            try:
                shutil.copy2(h5_path, output_h5_path)
                print(f"\n  ✓ Copied original h5 file to: {output_h5_path}")
            except Exception as e:
                raise FileNotFoundError(
                    f"Failed to copy h5 file from {h5_path} to {output_h5_path}: {e}\n"
                    f"Please check that the source file exists and you have write permissions to the output directory."
                )
        
        # 验证输出h5文件确实存在
        if not os.path.exists(output_h5_path):
            raise FileNotFoundError(
                f"Output h5 file does not exist after copy attempt: {output_h5_path}\n"
                f"Please check file permissions and disk space."
            )
        
        num_timesteps = end_timestep - start_timestep + 1
        config_pbar.set_postfix({
            "Traj": traj_idx,
            "Agent": agent_idx,
            "Timesteps": f"{start_timestep}-{end_timestep}",
            "Total": f"{processed_timesteps}/{total_timesteps}"
        })
        
        # 处理单个轨迹
        generated_data, result = process_single_trajectory(
            h5_path=h5_path,
            traj_idx=traj_idx,
            agent_idx=agent_idx,
            start_timestep=start_timestep,
            end_timestep=end_timestep,
            urdf_filepath=urdf_filepath,
            robot_base_position=robot_base_position,
            camera_translation_relative_to_tcp=camera_translation_relative_to_tcp,
            camera_rotation_relative_to_tcp=camera_rotation_relative_to_tcp,
            n=n,
            step_size=step_size,
            segment_server_url=segment_server_url,
            transform_server_url=transform_server_url,
            read_extrinsic_from_h5=read_extrinsic_from_h5,
            extrinsic_timestep_offset=extrinsic_timestep_offset,
            translation_std=translation_std,
            rotation_std=rotation_std,
            ik_n_init_qpos=ik_n_init_qpos,
            ik_max_iterations=ik_max_iterations,
            ik_tolerance=ik_tolerance,
            max_perturbation_attempts=max_perturbation_attempts,
            planning_time=planning_time
        )
        
        # 存储生成的数据（包含image, qpos, extrinsic）
        key = (traj_idx, agent_idx)
        if key not in traj_images:
            traj_images[key] = {}
        traj_images[key].update(generated_data)
        
        all_results.append(result)
        processed_timesteps += len(generated_data)
        
        # 立即更新h5文件中该agent的图像、qpos和extrinsic
        try:
            copy_h5_and_replace_data(
                src_h5_path=output_h5_path,
                dst_h5_path=output_h5_path,
                traj_idx=traj_idx,
                agent_idx=agent_idx,
                timestep_data=generated_data,
                verbose=False
            )
        except Exception as e:
            tqdm.write(f"    ✗ Failed to replace data for trajectory {traj_idx}, agent {agent_idx}: {e}")
        
        # 检查该轨迹的所有配置是否都已处理完成
        # 获取该轨迹的所有配置索引
        traj_all_config_indices = set(i for i, c in enumerate(configs) if c['traj_idx'] == traj_idx)
        # 已处理的配置索引（由于是按顺序处理的，所以是 0 到 len(all_results)-1）
        processed_config_indices = set(range(len(all_results)))
        
        # 如果该轨迹的所有配置都已处理完成，且尚未创建视频，则立即创建视频
        # if traj_idx not in completed_trajectories and traj_all_config_indices.issubset(processed_config_indices):
        #     # 该轨迹的所有配置都已处理完成，创建视频
        #     try:
        #         output_video_path = os.path.join(output_dir, f"traj_{traj_idx}_visualization.mp4")
        #         tqdm.write(f"\n  Creating video for trajectory {traj_idx}...")
        #         create_visualization_video(
        #             h5_path=output_h5_path,
        #             traj_idx=traj_idx,
        #             output_video_path=output_video_path,
        #             fps=10,
        #             verbose=False
        #         )
        #         tqdm.write(f"  ✓ Trajectory {traj_idx} video saved: {output_video_path}")
        #         completed_trajectories.add(traj_idx)
        #     except Exception as e:
        #         tqdm.write(f"  ✗ Failed to create video for trajectory {traj_idx}: {e}")
        # if config_idx not in completed_trajectories and traj_all_config_indices.issubset(processed_config_indices):
            # 该轨迹的所有配置都已处理完成，创建视频
        try:
            output_video_path = os.path.join(output_dir, f"traj_{config_idx}_visualization.mp4")
            tqdm.write(f"\n  Creating video for trajectory {config_idx}...")
            create_visualization_video(
                h5_path=output_h5_path,
                traj_idx=traj_idx,
                output_video_path=output_video_path,
                fps=10,
                verbose=False
            )
            tqdm.write(f"  ✓ Trajectory {config_idx} video saved: {output_video_path}")
            completed_trajectories.add(config_idx)
        except Exception as e:
            tqdm.write(f"  ✗ Failed to create video for trajectory {traj_idx}: {e}")
    config_pbar.close()
    
    # 确保所有轨迹的视频都已创建（处理边界情况）
    unique_traj_indices = set(config['traj_idx'] for config in configs)
    for traj_idx in unique_traj_indices:
        if traj_idx not in completed_trajectories:
            try:
                output_video_path = os.path.join(output_dir, f"traj_{traj_idx}_visualization.mp4")
                tqdm.write(f"\n  Creating video for trajectory {traj_idx}...")
                create_visualization_video(
                    h5_path=output_h5_path,
                    traj_idx=traj_idx,
                    output_video_path=output_video_path,
                    fps=10,
                    verbose=False
                )
                tqdm.write(f"  ✓ Trajectory {traj_idx} video saved: {output_video_path}")
                completed_trajectories.add(traj_idx)
            except Exception as e:
                tqdm.write(f"  ✗ Failed to create video for trajectory {traj_idx}: {e}")
    
    # 打印总汇总
    print(f"\n{'#'*60}")
    print(f"Batch processing completed!")
    total_success = sum(len(r['success']) for r in all_results)
    total_failed = sum(len(r['failed']) for r in all_results)
    print(f"  Unified h5 file: {output_h5_path}")
    print(f"  Total trajectories processed: {len(unique_traj_indices)}")
    print(f"  Total successful timesteps: {total_success}")
    print(f"  Total failed timesteps: {total_failed}")
    print(f"{'#'*60}\n")
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Batch process multiple trajectories with extrinsic perturbations'
    )
    
    # 基本参数
    parser.add_argument('--h5_path', type=str, required=True, help='h5文件路径')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    
    # 配置文件或单个配置
    parser.add_argument('--config_file', type=str, default=None, 
                       help='JSON配置文件路径（包含多个轨迹配置）')
    
    # 单个轨迹配置（如果不使用config_file）
    parser.add_argument('--traj_idx', type=int, default=None, help='轨迹序号')
    parser.add_argument('--agent_idx', type=int, default=None, help='agent序号')
    parser.add_argument('--start_timestep', type=int, default=None, help='起始时间步')
    parser.add_argument('--end_timestep', type=int, default=None, help='结束时间步（包含）')
    
    # qpos2camext相关参数
    parser.add_argument('--urdf_filepath', type=str, default=None, 
                       help='URDF文件路径（用于计算tgt_extrinsic）')
    parser.add_argument('--robot_base_position', type=float, nargs=3, default=None, 
                       help='机器人基座位置 [x, y, z]')
    parser.add_argument('--camera_translation_relative_to_tcp', type=float, nargs=3, default=None, 
                       help='相机相对于TCP的平移 [x, y, z]')
    parser.add_argument('--camera_rotation_relative_to_tcp', type=float, nargs=4, default=None, 
                       help='相机相对于TCP的旋转 [w, x, y, z] (wxyz格式)')
    
    # 外参相关参数
    parser.add_argument('--read_extrinsic_from_h5', action='store_true', 
                       help='从h5文件中读取外参（如果False，则对当前timestep的外参加扰动）')
    parser.add_argument('--extrinsic_timestep_offset', type=int, default=30, 
                       help='读取外参时的时间步偏移量（相对于timestep），默认30')
    parser.add_argument('--translation_std', type=float, default=0.02, 
                       help='平移扰动的标准差（米），默认0.01m (1cm)')
    parser.add_argument('--rotation_std', type=float, default=0.02, 
                       help='旋转扰动的标准差（弧度），默认0.01rad (约0.57度)')
    
    # 逆运动学参数（当read_extrinsic_from_h5=False时使用）
    parser.add_argument('--ik_n_init_qpos', type=int, default=100, 
                       help='逆运动学随机尝试次数，默认50')
    parser.add_argument('--ik_max_iterations', type=int, default=300, 
                       help='逆运动学每次优化的最大迭代次数，默认200')
    parser.add_argument('--ik_tolerance', type=float, default=1e-5, 
                       help='逆运动学收敛容差，默认1e-5')
    parser.add_argument('--max_perturbation_attempts', type=int, default=100, 
                       help='最大扰动尝试次数，默认100')
    parser.add_argument('--planning_time', type=float, default=5.0, 
                       help='规划时间限制（秒），默认5.0')
    
    # 处理参数
    parser.add_argument('--n', type=int, default=3, help='提取的步数（包含t）')
    parser.add_argument('--step_size', type=int, default=10, 
                       help='时间步长a，提取的时间步为 t, t+a, t+2a, ...，默认10')
    parser.add_argument('--segment_server', type=str, default=os.getenv('SEGMENT_SERVER_URL', 'http://localhost:8000'),
                       help='分割服务器URL')
    parser.add_argument('--transform_server', type=str, default=os.getenv('TRANSFORM_SERVER_URL', 'http://localhost:8001'),
                       help='变换服务器URL')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 读取配置
    configs = []
    
    if args.config_file:
        # 从JSON文件读取配置
        with open(args.config_file, 'r') as f:
            config_data = json.load(f)
            if isinstance(config_data, list):
                configs = config_data
            else:
                configs = [config_data]
    else:
        # 从命令行参数构建单个配置
        if (args.traj_idx is None or args.agent_idx is None or 
            args.start_timestep is None or args.end_timestep is None):
            parser.error("Either --config_file must be provided, or all of --traj_idx, --agent_idx, "
                        "--start_timestep, --end_timestep must be provided")
        
        configs = [{
            'traj_idx': args.traj_idx,
            'agent_idx': args.agent_idx,
            'start_timestep': args.start_timestep,
            'end_timestep': args.end_timestep
        }]
    
    # 转换参数格式
    robot_base_position = np.array(args.robot_base_position) if args.robot_base_position is not None else None
    camera_translation = np.array(args.camera_translation_relative_to_tcp) if args.camera_translation_relative_to_tcp is not None else None
    camera_rotation = np.array(args.camera_rotation_relative_to_tcp) if args.camera_rotation_relative_to_tcp is not None else None
    
    # 批量处理
    all_results = process_multiple_trajectories(
        h5_path=args.h5_path,
        configs=configs,
        output_dir=args.output_dir,
        urdf_filepath=args.urdf_filepath,
        robot_base_position=robot_base_position,
        camera_translation_relative_to_tcp=camera_translation,
        camera_rotation_relative_to_tcp=camera_rotation,
        n=args.n,
        step_size=args.step_size,
        segment_server_url=args.segment_server,
        transform_server_url=args.transform_server,
        read_extrinsic_from_h5=args.read_extrinsic_from_h5,
        extrinsic_timestep_offset=args.extrinsic_timestep_offset,
        translation_std=args.translation_std,
        rotation_std=args.rotation_std,
        ik_n_init_qpos=args.ik_n_init_qpos,
        ik_max_iterations=args.ik_max_iterations,
        ik_tolerance=args.ik_tolerance,
        max_perturbation_attempts=args.max_perturbation_attempts,
        planning_time=args.planning_time
    )
    
    # 保存结果摘要
    summary_path = os.path.join(args.output_dir, 'batch_processing_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Processing summary saved to: {summary_path}")


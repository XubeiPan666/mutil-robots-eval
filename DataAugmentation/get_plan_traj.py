import h5py
import numpy as np
import cv2
import os
from typing import List, Tuple, Optional
from scipy.spatial.transform import Rotation as R

def extrinsic_cv_to_pose(extrinsic_cv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract camera pose (R, t) from OpenCV extrinsic matrix.
    extrinsic_cv = [R_w2c | t_w2c]
    Returns:
        R_cam2world: 3x3
        t_cam_in_world: 3x1
    """
    R_world2cam = extrinsic_cv[:3, :3]
    t_world2cam = extrinsic_cv[:3, 3]
    
    R_cam2world = R_world2cam.T
    t_cam_in_world = -R_cam2world @ t_world2cam
    
    return R_cam2world, t_cam_in_world

def project_points_to_image(
    points_world: np.ndarray,
    K: np.ndarray,
    extrinsic: np.ndarray
) -> np.ndarray:
    """
    Project 3D world points to 2D image coordinates.
    points_world: (N, 3)
    K: 3x3 intrinsic
    extrinsic: 3x4 [R|t] (world to camera)
    """
    R_w2c = extrinsic[:3, :3]
    t_w2c = extrinsic[:3, 3]
    
    points_cam = (R_w2c @ points_world.T).T + t_w2c
    
    # Project to image plane
    points_img_homo = (K @ points_cam.T).T
    points_img = points_img_homo[:, :2] / points_img_homo[:, 2:3]
    
    return points_img

def quaternion_to_rotation_matrix(quat: np.ndarray, quat_format: str = 'wxyz') -> np.ndarray:
    """
    Convert quaternion to rotation matrix.
    
    Args:
        quat: Quaternion [w, x, y, z] (wxyz) or [x, y, z, w] (xyzw)
        quat_format: 'wxyz' (SAPIEN format) or 'xyzw' (scipy format)
    
    Returns:
        Rotation matrix (3x3)
    """
    quat = np.array(quat)
    if quat_format == 'wxyz':
        # Convert from [w, x, y, z] to [x, y, z, w] for scipy
        quat_scipy = quat[[1, 2, 3, 0]]
    else:
        quat_scipy = quat
    r = R.from_quat(quat_scipy)
    return r.as_matrix()

def get_tcp_3d_in_world(
    extrinsic: np.ndarray,
    camera_to_tcp_offset: np.ndarray = np.array([0.0, 0.0, 0.0]),
    robot_base_position: Optional[np.ndarray] = None,
    camera_to_tcp_rotation: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Get TCP 3D position in world coordinates using camera-to-TCP offset and rotation.
    
    Args:
        extrinsic: Camera extrinsic matrix (World -> Camera) [R|t]
        camera_to_tcp_offset: Offset from camera to TCP [x, y, z] in TCP frame
        robot_base_position: Robot base position relative to world frame [x, y, z]
        camera_to_tcp_rotation: Rotation from TCP to camera (quaternion [w, x, y, z] or rotation matrix 3x3)
    
    Returns:
        TCP position in world coordinates [x, y, z]
    """
    # Get camera pose in world frame
    R_cam2world, t_cam_in_world = extrinsic_cv_to_pose(extrinsic)
    
    # Handle rotation offset
    # camera_to_tcp_offset is "TCP to camera" translation in TCP frame
    # camera_to_tcp_rotation is "TCP to camera" rotation
    if camera_to_tcp_rotation is not None:
        # Convert quaternion to rotation matrix
        if camera_to_tcp_rotation.shape == (4,):
            # Quaternion format [w, x, y, z]
            R_tcp2cam = quaternion_to_rotation_matrix(camera_to_tcp_rotation, 'wxyz')
        elif camera_to_tcp_rotation.shape == (3, 3):
            # Rotation matrix
            R_tcp2cam = camera_to_tcp_rotation
        else:
            raise ValueError(f"Invalid rotation format: shape {camera_to_tcp_rotation.shape}")
        
        # camera_to_tcp_offset is "TCP to camera" in TCP frame: t_tcp2cam
        # To get "camera to TCP" in camera frame: t_cam2tcp = -R_tcp2cam^T @ t_tcp2cam
        # TCP position in camera frame = t_cam2tcp
        offset_cam = -R_tcp2cam.T @ camera_to_tcp_offset
    else:
        # No rotation, offset is "TCP to camera" in TCP frame
        # TCP position in camera frame = -offset
        offset_cam = -camera_to_tcp_offset
    
    # Transform TCP from camera frame to world frame
    # P_world = R_cam2world @ P_cam + t_cam_in_world
    p_world = R_cam2world @ offset_cam + t_cam_in_world
    
    # If extrinsic is relative to robot base (not true world frame),
    # we need to add robot base position to get true world coordinates
    if robot_base_position is not None:
        p_world = p_world + robot_base_position
    
    return p_world

def visualize_trajectory_on_image(
    image: np.ndarray,
    trajectory_2d: np.ndarray,
    gripper_states: Optional[np.ndarray] = None,
    output_path: str = "trajectory_viz.png"
):
    viz_img = image.copy()
    
    # Draw lines
    for i in range(len(trajectory_2d) - 1):
        p1 = tuple(trajectory_2d[i].astype(int))
        p2 = tuple(trajectory_2d[i+1].astype(int))
        cv2.line(viz_img, p1, p2, (0, 255, 0), 2)
        
    # Find key points based on gripper states
    start_idx = 0
    first_close_idx = None
    first_open_after_close_idx = None
    
    if gripper_states is not None:
        # Flatten and check values
        gripper_states = gripper_states.flatten()
        
        # Try to detect if 1 is closed or 0 is closed
        # Usually, if it's binary, 1 is closed. 
        # If it's continuous (width), we already handled it in main.
        closed_indices = np.where(gripper_states > 0.5)[0]
        
        if len(closed_indices) > 0:
            first_close_idx = closed_indices[0]
            # Find first open AFTER the first close
            # We look for a transition from closed to open
            open_after_indices = np.where(gripper_states[first_close_idx:] < 0.5)[0]
            if len(open_after_indices) > 0:
                first_open_after_close_idx = first_close_idx + open_after_indices[0]

    print(f"Key Indices -> Start: {start_idx}, Close: {first_close_idx}, Open: {first_open_after_close_idx}")

    # Draw points
    for i, p in enumerate(trajectory_2d):
        center = tuple(p.astype(int))
        
        # Default point color (small blue dot)
        cv2.circle(viz_img, center, 2, (255, 0, 0), -1)
        
        # Highlight key points in RED with larger circles
        # We use a list to handle multiple labels at the same point
        labels = []
        if i == start_idx:
            labels.append("Start")
        if i == first_close_idx:
            labels.append("First Close")
        if i == first_open_after_close_idx:
            labels.append("First Open")
            
        if labels:
            label_text = " + ".join(labels)
            cv2.circle(viz_img, center, 6, (0, 0, 255), -1) # Solid Red
            cv2.circle(viz_img, center, 8, (255, 255, 255), 1) # White border
            # Add text label
            cv2.putText(viz_img, label_text, (center[0] + 10, center[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imwrite(output_path, cv2.cvtColor(viz_img, cv2.COLOR_RGB2BGR))
    print(f"Saved visualization to {output_path}")

def main(h5_path: str, traj_idx: int, agent_idx: int = 0, tcp_offset: Optional[List[float]] = None, robot_base_position: Optional[List[float]] = None, tcp_rotation: Optional[List[float]] = None, use_gt_tcp: bool = False, camera_type: str = "wrist"):
    with h5py.File(h5_path, 'r') as f:
        traj_key = f"traj_{traj_idx}"
        if traj_key not in f:
            print(f"Error: {traj_key} not found")
            return
            
        group = f[traj_key]
        
        # Select camera based on camera_type
        if camera_type == "global" or camera_type == "head":
            camera_name = "head_camera_global"
        else:
            camera_name = f"wrist_camera_agent{agent_idx}"
        
        # Check if camera exists
        if camera_name not in group["obs"]["sensor_data"]:
            available_cameras = list(group["obs"]["sensor_data"].keys())
            print(f"Error: Camera '{camera_name}' not found!")
            print(f"Available cameras: {available_cameras}")
            if camera_type == "global" or camera_type == "head":
                # Try alternative names
                for alt_name in ["head_camera_global", "global_camera", "head_camera"]:
                    if alt_name in group["obs"]["sensor_data"]:
                        camera_name = alt_name
                        print(f"Using alternative camera name: {camera_name}")
                        break
            if camera_name not in group["obs"]["sensor_data"]:
                return
        
        obs_sensor = group["obs"]["sensor_data"][camera_name]
        param_sensor = group["obs"]["sensor_param"][camera_name]
        
        print(f"Using camera: {camera_name} for visualization")
        
        # 1. First frame as background
        first_frame_rgb = np.array(obs_sensor["rgb"][0])
        K_first = np.array(param_sensor["intrinsic_cv"][0])
        extrinsic_first = np.array(param_sensor["extrinsic_cv"][0])
        
        # 2. Camera to TCP offset and rotation
        # From yaml: params: [[0.2, 0, -0.18], [0.0, 0.707107, 0.0, 0.707107]]
        # This represents TCP->Camera transform: translation and rotation (quaternion)
        if tcp_offset is not None:
            camera_to_tcp_offset = np.array(tcp_offset)
        else:
            # Default offset (you can adjust this based on your calibration)
            camera_to_tcp_offset = np.array([0.2, 0.0, -0.18])
        
        print(f"Using camera-to-TCP offset: {camera_to_tcp_offset} (meters)")
        
        # Debug: Analyze offset in first frame
        if len(obs_sensor["rgb"]) > 0:
            ext_first = np.array(param_sensor["extrinsic_cv"][0])
            R_cam2world, t_cam_in_world = extrinsic_cv_to_pose(ext_first)
            
            # Offset in camera frame
            offset_cam = camera_to_tcp_offset
            # Offset transformed to world frame
            offset_world = R_cam2world @ offset_cam
            
            print(f"\n=== Offset Analysis (First Frame) ===")
            print(f"Offset in camera frame: {offset_cam}")
            print(f"Offset in world frame: {R_cam2world @ offset_cam}")
            print(f"Camera position in world: {t_cam_in_world}")
            print(f"Expected TCP position: {t_cam_in_world + offset_world}")
        
        # Camera to TCP rotation (quaternion [w, x, y, z] format)
        # This is the rotation from TCP to camera
        if tcp_rotation is not None:
            camera_to_tcp_rotation = np.array(tcp_rotation)
            print(f"Using camera-to-TCP rotation (quaternion): {camera_to_tcp_rotation}")
        else:
            # Default rotation from yaml: [0.0, 0.707107, 0.0, 0.707107]
            camera_to_tcp_rotation = np.array([0.0, 0.707107, 0.0, 0.707107])
            print(f"Using default camera-to-TCP rotation (quaternion): {camera_to_tcp_rotation}")
        
        # Robot base position (mechanical arm base offset relative to world)
        if robot_base_position is not None:
            robot_base_position = np.array(robot_base_position)
            print(f"Using robot_base_position: {robot_base_position} (meters)")
        else:
            robot_base_position = None
            print("No robot_base_position provided. Assuming extrinsic is already in world frame.")
        
        # Load ground truth TCP poses
        gt_tcp_poses = None
        if "obs" in group:
            # Try obs/extra/tcp_pose first (most common path)
            if "extra" in group["obs"] and "tcp_pose" in group["obs"]["extra"]:
                gt_tcp_poses = np.array(group["obs"]["extra"]["tcp_pose"])
                print(f"Loaded ground truth TCP poses from obs/extra/tcp_pose: shape {gt_tcp_poses.shape}")
            # Fallback to proprioception paths
            elif "proprioception" in group["obs"]:
                if "agent_tcp_pose" in group["obs"]["proprioception"]:
                    gt_tcp_poses = np.array(group["obs"]["proprioception"]["agent_tcp_pose"])
                    print(f"Loaded ground truth TCP poses from obs/proprioception/agent_tcp_pose: shape {gt_tcp_poses.shape}")
                elif f"agent{agent_idx}_tcp_pose" in group["obs"]["proprioception"]:
                    gt_tcp_poses = np.array(group["obs"]["proprioception"][f"agent{agent_idx}_tcp_pose"])
                    print(f"Loaded ground truth TCP poses from obs/proprioception/agent{agent_idx}_tcp_pose: shape {gt_tcp_poses.shape}")
        
        tcp_positions_world = []
        errors = []  # Store errors for statistics
        num_steps = len(obs_sensor["rgb"])
        
        # If use_gt_tcp is True, directly use ground truth TCP poses
        if use_gt_tcp:
            if gt_tcp_poses is None:
                print("Error: --use_gt_tcp requires ground truth TCP poses, but none found!")
                return
            
            print("Using ground truth TCP poses for visualization")
            for t in range(num_steps):
                if t < len(gt_tcp_poses):
                    gt_pose = gt_tcp_poses[t]
                    # Extract position from pose
                    if gt_pose.shape == (4, 4):
                        gt_pos = gt_pose[:3, 3]
                    elif len(gt_pose.shape) == 1 and len(gt_pose) >= 3:
                        gt_pos = gt_pose[:3]
                    else:
                        gt_pos = None
                    
                    if gt_pos is not None:
                        tcp_positions_world.append(gt_pos)
                    else:
                        # Fallback: use previous position
                        if len(tcp_positions_world) > 0:
                            tcp_positions_world.append(tcp_positions_world[-1])
                        else:
                            print(f"Warning: Could not extract TCP position at step {t}")
                            return
                else:
                    # Use last available position
                    if len(tcp_positions_world) > 0:
                        tcp_positions_world.append(tcp_positions_world[-1])
        else:
            # Calculate TCP positions from camera extrinsics
            for t in range(num_steps):
                ext_t = np.array(param_sensor["extrinsic_cv"][t])
                
                # Directly use camera-to-TCP offset and rotation to get TCP position
                p_world = get_tcp_3d_in_world(ext_t, camera_to_tcp_offset, robot_base_position, camera_to_tcp_rotation)
                tcp_positions_world.append(p_world)
                
                # Compare with ground truth if available
                if gt_tcp_poses is not None and t < len(gt_tcp_poses):
                    gt_pose = gt_tcp_poses[t]
                    # gt_pose might be 4x4 matrix or 7D (position + quaternion)
                    if gt_pose.shape == (4, 4):
                        gt_pos = gt_pose[:3, 3]  # Extract translation
                    elif len(gt_pose.shape) == 1 and len(gt_pose) >= 3:
                        gt_pos = gt_pose[:3]  # First 3 elements are position
                    else:
                        gt_pos = None
                    
                    if gt_pos is not None:
                        error = np.linalg.norm(p_world - gt_pos)
                        errors.append(error)
                        error_vec = p_world - gt_pos  # Error vector (not absolute)
                        error_x = abs(error_vec[0])
                        error_y = abs(error_vec[1])
                        error_z = abs(error_vec[2])
                        
                        # Detect sudden error increase
                        if t > 0 and len(errors) > 0:
                            prev_error = errors[-1]
                            error_jump = error - prev_error
                            if error_jump > 0.01:  # Error increased by more than 1cm
                                ext_t = np.array(param_sensor["extrinsic_cv"][t])
                                R_cam2world, t_cam_in_world = extrinsic_cv_to_pose(ext_t)
                                print(f"\n⚠️  Step {t}: Error jump detected! {prev_error*1000:.2f}mm -> {error*1000:.2f}mm (Δ={error_jump*1000:.2f}mm)")
                                print(f"   Camera position: {t_cam_in_world}")
                                print(f"   Estimated TCP: {p_world}")
                                print(f"   Ground truth TCP: {gt_pos}")
                                print(f"   Error vector: {error_vec*1000} mm")
                        
                        if t < 5 or t % 10 == 0:  # Print first 5 and every 10th step
                            print(f"Step {t:3d}: Estimated={p_world}, GT={gt_pos}, "
                                  f"Error={error*1000:.2f}mm (X:{error_x*1000:.2f}, Y:{error_y*1000:.2f}, Z:{error_z*1000:.2f}mm)")
                        
                        # Detailed analysis for first frame
                        if t == 0:
                            ext_t = np.array(param_sensor["extrinsic_cv"][t])
                            R_cam2world, t_cam_in_world = extrinsic_cv_to_pose(ext_t)
                            offset_world = R_cam2world @ camera_to_tcp_offset
                            print(f"\n=== First Frame Error Analysis ===")
                            print(f"Error vector: {error_vec*1000} mm")
                            print(f"Offset in camera frame: {camera_to_tcp_offset*1000} mm")
                            print(f"Offset in world frame: {offset_world*1000} mm")
                            print(f"Error vs Offset (world): X ratio={error_vec[0]/offset_world[0]:.2f}, "
                                  f"Y ratio={error_vec[1]/offset_world[1] if abs(offset_world[1])>1e-6 else 'N/A'}, "
                                  f"Z ratio={error_vec[2]/offset_world[2]:.2f}")
                            print(f"Note: If error ≈ -offset, the offset direction might be reversed")
        
        # Print error statistics
        if errors:
            errors = np.array(errors)
            print(f"\n=== Error Statistics ===")
            print(f"Mean error: {np.mean(errors)*1000:.2f} mm")
            print(f"Median error: {np.median(errors)*1000:.2f} mm")
            print(f"Std error: {np.std(errors)*1000:.2f} mm")
            print(f"Max error: {np.max(errors)*1000:.2f} mm")
            print(f"Min error: {np.min(errors)*1000:.2f} mm")
            print(f"Valid comparisons: {len(errors)}/{num_steps}")
            
            # Find steps with large errors
            large_error_threshold = np.percentile(errors, 90)  # Top 10% errors
            large_error_indices = np.where(errors > large_error_threshold)[0]
            if len(large_error_indices) > 0:
                print(f"\n=== Large Error Analysis (>{large_error_threshold*1000:.2f}mm) ===")
                print(f"Found {len(large_error_indices)} steps with large errors")
                print(f"Steps: {large_error_indices[:10].tolist()}..." if len(large_error_indices) > 10 else f"Steps: {large_error_indices.tolist()}")
                
                # Analyze camera positions at these steps
                if len(large_error_indices) > 0:
                    sample_idx = large_error_indices[0]
                    ext_sample = np.array(param_sensor["extrinsic_cv"][sample_idx])
                    R_cam2world_sample, t_cam_in_world_sample = extrinsic_cv_to_pose(ext_sample)
                    print(f"\nSample large error step {sample_idx}:")
                    print(f"  Camera position: {t_cam_in_world_sample}")
                    print(f"  Error: {errors[sample_idx]*1000:.2f}mm")
                    
                    # Compare with a good step
                    good_error_indices = np.where(errors < np.percentile(errors, 10))[0]
                    if len(good_error_indices) > 0:
                        good_idx = good_error_indices[0]
                        ext_good = np.array(param_sensor["extrinsic_cv"][good_idx])
                        R_cam2world_good, t_cam_in_world_good = extrinsic_cv_to_pose(ext_good)
                        print(f"\nSample good error step {good_idx}:")
                        print(f"  Camera position: {t_cam_in_world_good}")
                        print(f"  Error: {errors[good_idx]*1000:.2f}mm")
                        print(f"  Camera position difference: {(t_cam_in_world_sample - t_cam_in_world_good)*1000} mm")
        else:
            print("\nWarning: Could not find ground truth TCP poses for comparison")
                
        tcp_positions_world = np.array(tcp_positions_world)
        
        # 3. Project back to first frame
        trajectory_2d = project_points_to_image(tcp_positions_world, K_first, extrinsic_first)
        
        # 4. Gripper states
        gripper_states = None
        if "action" in group and "gripper" in group["action"]:
            gripper_states = np.array(group["action"]["gripper"]).flatten()
        elif "obs" in group and "proprioception" in group["obs"] and "gripper_width" in group["obs"]["proprioception"]:
            # If using width, we need to invert the logic: small width means closed
            widths = np.array(group["obs"]["proprioception"]["gripper_width"]).flatten()
            # Heuristic: if width is less than 50% of max width, consider it "closing/closed"
            if len(widths) > 0:
                max_w = np.max(widths)
                min_w = np.min(widths)
                if max_w > min_w:
                    gripper_states = (widths < (min_w + (max_w - min_w) * 0.3)).astype(float)
        
        if gripper_states is not None:
            print(f"Gripper states: min={gripper_states.min():.3f}, max={gripper_states.max():.3f}")
            
        # 5. Save
        camera_suffix = "wrist" if camera_type == "wrist" else "global"
        output_name = f"trajectory_viz_traj{traj_idx}_{camera_suffix}.png"
        visualize_trajectory_on_image(first_frame_rgb, trajectory_2d, gripper_states, output_name)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=str, required=True, help="Path to a trusted HDF5 trajectory file")
    parser.add_argument("--traj", type=int, default=0)
    parser.add_argument("--agent", type=int, default=0)
    parser.add_argument("--tcp_offset", type=float, nargs=3, default=[-0.2, 0.0, 0.18],
                        help="Camera to TCP offset in camera frame [x, y, z] in meters. "
                             "Positive x: right, Positive y: down, Positive z: forward. "
                             "Default: [-0.2, 0.0, 0.18]")
    parser.add_argument("--robot_base_position", type=float, nargs=3, default=[-0.2, -0.2, 0.],
                        help="Robot base position relative to world frame [x, y, z] in meters. "
                             "If provided, this offset will be added to TCP position. "
                             "Default: [-0.2, -0.2, 0.]")
    parser.add_argument("--tcp_rotation", type=float, nargs=4, default=[0.707107, 0.0, 0.707107, 0.0],
                        help="Camera to TCP rotation as quaternion [w, x, y, z] (SAPIEN format). "
                             "This is the rotation from TCP to camera. "
                             "Default: [0.0, 0.707107, 0.0, 0.707107]")
    parser.add_argument("--use_gt_tcp", action="store_true",
                        help="Use ground truth TCP poses directly for visualization (skip calculation)")
    parser.add_argument("--camera", type=str, default="wrist", choices=["wrist", "global", "head"],
                        help="Camera to use for visualization background. "
                             "Options: 'wrist' (wrist camera), 'global' or 'head' (global/head camera). "
                             "Default: 'wrist'")
    args = parser.parse_args()
    main(args.h5, args.traj, args.agent, args.tcp_offset, args.robot_base_position, args.tcp_rotation, args.use_gt_tcp, args.camera)

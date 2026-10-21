import h5py
import numpy as np
from pathlib import Path
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import shutil
import cv2
import json
import random

def load_instruction_texts(json_path):
    with open(json_path, "r") as f:
        instructions = json.load(f)
    return instructions  # List[str]

def convert_robofactory_to_lerobot(
    h5_path: str,
    output_dir: str,
    instruction_json: str,
    push_to_hub: bool = False
):
    task_desc = "lift the barrier"  # 固定的任务指令

    with h5py.File(h5_path, 'r') as f:
        traj_keys = [k for k in f.keys() if k.startswith("traj_")]
        
        repo_id = "liftbarrier_dual_agents"
        output_path = Path(output_dir) / repo_id
        if output_path.exists():
            shutil.rmtree(output_path)

        print("\n创建双臂机器人数据集...")
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            robot_type="panda",
            fps=30,
            features={
                "image": {
                    "dtype": "image",
                    "shape": (240, 320, 3),
                    "names": ["height", "width", "channel"],
                },
                "wrist_image": {
                    "dtype": "image",
                    "shape": (240, 320, 3),
                    "names": ["height", "width", "channel"],
                },
                "joint_position": {
                    "dtype": "float32",
                    "shape": (7,),  
                    "names": ["joint_position"],
                },
                "gripper_position": {
                    "dtype": "float32",
                    "shape": (1,),  
                    "names": ["gripper_position"],
                },
                "actions": {
                    "dtype": "float32",
                    "shape": (8,),  
                    "names": ["actions"],
                },
            },
            image_writer_threads=4,
        )

        for agent_id in [0, 1]:
            agent_name = f"panda-{agent_id}"
            print(f"\n处理 Agent {agent_name} 的数据...")

            for traj_key in traj_keys:
                try:
                    actions = f[f"{traj_key}/actions/{agent_name}"][:]  # (T, 8)
                    self_image = f[f"{traj_key}/obs/sensor_data/head_camera_agent{agent_id}/rgb"][:]  # (T, H, W, 3)
                    global_image = f[f"{traj_key}/obs/sensor_data/head_camera_global/rgb"][:]  # (T, H, W, 3)
                    T = min(len(actions), len(self_image))     
                    #import pdb;pdb.set_trace()              
                    for t in range(T):
                        full_state = actions[t].astype(np.float32)
                        action_7d = full_state[:7]
                        action_1d = full_state[7]
                        img = self_image[t]
                        img_global = global_image[t]
                        
                        dataset.add_frame({
                            "image": img,
                            "wrist_image": img_global,#直接使用global视角图像
                            "joint_position": action_7d,
                            "gripper_position": np.array([action_1d], dtype=np.float32),
                            "actions": full_state,
                            "task": task_desc 
                        })
                    
                    dataset.save_episode()

                except Exception as e:
                    print(f"[警告] 处理 {traj_key} 失败: {e}")

        if push_to_hub:
            dataset.push_to_hub(
                tags=["robofactory", "panda", "dual_arm"],
                private=True,
                push_videos=True,
                license="apache-2.0",
            )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert LiftBarrier HDF5 trajectories to a local LeRobot dataset.")
    parser.add_argument("--h5-path", required=True, help="Path to a trusted LiftBarrier trajectory file")
    parser.add_argument("--output-dir", default="./datasets", help="Dataset output directory")
    parser.add_argument("--instruction-json", default="data/instructions/lift_barrier.json", help="Legacy argument; currently unused")
    parser.add_argument("--push-to-hub", action="store_true", help="Upload to a private Hugging Face dataset")
    args = parser.parse_args()
    convert_robofactory_to_lerobot(
        h5_path=args.h5_path,
        output_dir=args.output_dir,
        instruction_json=args.instruction_json,
        push_to_hub=args.push_to_hub,
    )

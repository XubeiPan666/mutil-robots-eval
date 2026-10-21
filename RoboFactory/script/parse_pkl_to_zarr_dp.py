import pickle, os
import numpy as np
import pdb
import zarr
import shutil
import argparse


def main():
    parser = argparse.ArgumentParser(description='Process some episodes.')
    parser.add_argument('--task_name', type=str, default='LiftBarrier',
                        help='The name of the task (e.g., LiftBarrier)')
    parser.add_argument('--agent_id', type=int, default='LiftBarrier',
                        help='The id of the agent (e.g., LiftBarrier)')
    parser.add_argument('--load_num', type=int, default=50,
                        help='Number of episodes to process (e.g., 50)')
    args = parser.parse_args()

    task_name = args.task_name
    agent_id = args.agent_id
    num = args.load_num
    load_dir = f'data/pkl_data/{task_name}_Agent{agent_id}'
    
    total_count = 0
    save_dir = f'data/zarr_data/{task_name}_Agent{agent_id}_{num}.zarr'

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    current_ep = 0

    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group('data')
    zarr_meta = zarr_root.create_group('meta')

    head_camera_arrays, front_camera_arrays, left_camera_arrays, right_camera_arrays = [], [], [], []
    episode_ends_arrays, action_arrays, state_arrays = [], [], []
    
    while os.path.isdir(load_dir+f'/episode{current_ep}') and current_ep < num:
        print(f'processing episode: {current_ep + 1} / {num}', end='\r')
        file_num = 0
        
        while os.path.exists(load_dir+f'/episode{current_ep}'+f'/{file_num}.pkl'):
            with open(load_dir+f'/episode{current_ep}'+f'/{file_num}.pkl', 'rb') as file:
                data = pickle.load(file)
            
            head_img = data['observation']['head_camera']['rgb']
            # front_img = data['observation']['front_camera']['rgb']
            # left_img = data['observation']['left_camera']['rgb']
            # right_img = data['observation']['right_camera']['rgb']
            action = data['endpose']
            joint_action = data['joint_action']
            
            # 验证维度
            if action is not None:
                action = np.array(action)
                if action.ndim == 0:
                    action = action.reshape(1)
                assert action.shape[-1] == 8, f"endpose 应该是 8 维，但得到 {action.shape[-1]} 维"
            if joint_action is not None:
                joint_action = np.array(joint_action)
                if joint_action.ndim == 0:
                    joint_action = joint_action.reshape(1)
                assert joint_action.shape[-1] == 9, f"joint_action (qpos) 应该是 9 维，但得到 {joint_action.shape[-1]} 维"

            head_camera_arrays.append(head_img)
            # front_camera_arrays.append(front_img)
            # left_camera_arrays.append(left_img)
            # right_camera_arrays.append(right_img)
            action_arrays.append(action)
            state_arrays.append(joint_action)

            file_num += 1
            total_count += 1
            
        current_ep += 1

        episode_ends_arrays.append(total_count)

    print()
    episode_ends_arrays = np.array(episode_ends_arrays)
    action_arrays = np.array(action_arrays)
    state_arrays = np.array(state_arrays)
    head_camera_arrays = np.array(head_camera_arrays)
    # front_camera_arrays = np.array(front_camera_arrays)
    # left_camera_arrays = np.array(left_camera_arrays)
    # right_camera_arrays = np.array(right_camera_arrays)
    
    # 最终维度验证
    print(f"action_arrays shape: {action_arrays.shape}, 期望最后一维为 8")
    print(f"state_arrays shape: {state_arrays.shape}, 期望最后一维为 9")
    assert action_arrays.shape[-1] == 8, f"action 应该是 8 维，但得到 {action_arrays.shape[-1]} 维"
    assert state_arrays.shape[-1] == 9, f"state 应该是 9 维，但得到 {state_arrays.shape[-1]} 维"

    # pdb.set_trace()
    head_camera_arrays = np.moveaxis(head_camera_arrays, -1, 1)  # NHWC -> NCHW
    # front_camera_arrays = np.moveaxis(front_camera_arrays, -1, 1)
    # left_camera_arrays = np.moveaxis(left_camera_arrays, -1, 1)
    # right_camera_arrays = np.moveaxis(right_camera_arrays, -1, 1)

    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
    action_chunk_size = (100, action_arrays.shape[1])
    state_chunk_size = (100, state_arrays.shape[1])
    head_camera_chunk_size = (100, *head_camera_arrays.shape[1:])
    zarr_data.create_dataset('head_camera', data=head_camera_arrays, chunks=head_camera_chunk_size, overwrite=True, compressor=compressor)
    # zarr_data.create_dataset('front_camera', data=front_camera_arrays, chunks=head_camera_chunk_size, overwrite=True, compressor=compressor)
    # zarr_data.create_dataset('left_camera', data=left_camera_arrays, chunks=head_camera_chunk_size, overwrite=True, compressor=compressor)
    # zarr_data.create_dataset('right_camera', data=right_camera_arrays, chunks=head_camera_chunk_size, overwrite=True, compressor=compressor)
    zarr_data.create_dataset('tcp_action', data=action_arrays, chunks=action_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('state', data=state_arrays, chunks=state_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('action', data=action_arrays, chunks=action_chunk_size, dtype='float32', overwrite=True, compressor=compressor)  # action 应该是 8 维的 endpose，不是 9 维的 joint_action
    zarr_meta.create_dataset('episode_ends', data=episode_ends_arrays, dtype='int64', overwrite=True, compressor=compressor)

if __name__ == '__main__':
    main()
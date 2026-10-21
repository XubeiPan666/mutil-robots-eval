# Local RoboFactory/MAS-VLA adaptation of ManiSkill utilities.
# Modified for multi-agent tasks and local scene/recording integration.
# See THIRD_PARTY_NOTICES.md and licenses/ for upstream licenses.

import copy
import ctypes
import gc
import time
from pathlib import Path
from typing import Callable, Optional, Union, List

import h5py
import numpy as np
import torch

from mani_skill import get_commit_info
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common, gym_utils
from mani_skill.utils.logging_utils import logger
from mani_skill.utils.io_utils import dump_json
from mani_skill.utils.structs.types import Array
from mani_skill.utils.visualization.misc import (
    images_to_video,
    put_info_on_image,
)
from mani_skill.utils.wrappers import CPUGymWrapper
from mani_skill.utils.wrappers.record import RecordEpisode, Step, parse_env_info

from .. import sapien_utils


def _trim_process_memory():
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


class RecordEpisodeMA(RecordEpisode):
    """Record trajectories or videos for episodes. Support multi-agent environments.
    
    """
    def __init__(
        self,
        env: BaseEnv,
        output_dir: str,
        save_trajectory: bool = True,
        trajectory_name: Optional[str] = None,
        save_video: bool = True,
        info_on_video: bool = False,
        save_on_reset: bool = True,
        save_video_trigger: Optional[Callable[[int], bool]] = None,
        max_steps_per_video: Optional[int] = None,
        clean_on_close: bool = True,
        record_reward: bool = True,
        record_env_state: bool = True,
        record_observation: bool = True,
        record_simple_observation: bool = False,
        video_fps: int = 30,
        avoid_overwriting_video: bool = False,
        source_type: Optional[str] = None,
        source_desc: Optional[str] = None,
        record_camera: Optional[str] = None,
    ) -> None:
        super().__init__(
            env=env,
            output_dir=output_dir,
            save_trajectory=save_trajectory,
            trajectory_name=trajectory_name,
            save_video=save_video,
            info_on_video=info_on_video,
            save_on_reset=save_on_reset,
            save_video_trigger=save_video_trigger,
            max_steps_per_video=max_steps_per_video,
            clean_on_close=clean_on_close,
            record_reward=record_reward,
            record_env_state=record_env_state,
            video_fps=video_fps,
            avoid_overwriting_video=avoid_overwriting_video,
            source_type=source_type,
            source_desc=source_desc
        )

        self.record_observation = record_observation
        self.record_simple_observation = record_simple_observation
        self.record_camera = record_camera
        
        if info_on_video and self.num_envs > 1:
            raise ValueError(
                "Cannot turn info_on_video=True when the number of environments parallelized is > 1"
            )

    def _init_observation_buffer(self, obs):
        if not self.record_observation:
            return []
        return [common.to_numpy(common.batch(obs))]

    def _append_observation_buffer(self, obs):
        if not self.record_observation:
            return
        self._trajectory_buffer.observation.append(common.to_numpy(common.batch(obs)))

    def _stack_observation_buffer(self, observation):
        if isinstance(observation, list):
            if len(observation) == 0:
                return observation
            first = observation[0]
            if isinstance(first, dict):
                return {
                    key: self._stack_observation_buffer([step[key] for step in observation])
                    for key in first.keys()
                }
            return np.concatenate(observation, axis=0)
        return observation

    def _slice_observation_buffer(self, observation, idx):
        if isinstance(observation, list):
            return observation[idx]
        return common.index_dict_array(observation, idx)

    def _write_observation_buffer_to_h5(
        self,
        group: h5py.Group,
        observation,
        key,
        start_ptr: int,
        end_ptr: int,
        env_idx: int,
    ):
        write_chunk_size = 64
        if not isinstance(observation, list):
            raise TypeError("streaming observation writer expects list-buffered data")
        if len(observation) == 0:
            return

        first = observation[0]
        if isinstance(first, dict):
            subgrp = group.create_group(key, track_order=True)
            for subkey in first.keys():
                self._write_observation_buffer_to_h5(
                    subgrp,
                    [step[subkey] for step in observation],
                    subkey,
                    start_ptr,
                    end_ptr,
                    env_idx,
                )
            return

        frames = observation[start_ptr:end_ptr]
        if len(frames) == 0:
            return

        if self.cpu_wrapped_env:
            sample = frames[0]
        elif key.startswith("panda-") and key[6:].isdigit():
            sample = frames[0]
        else:
            sample = frames[0][:, env_idx]

        dataset_kwargs = {}
        if key in {"rgb", "depth", "seg"}:
            dataset_kwargs.update(compression="gzip", compression_opts=5)
        dataset = group.create_dataset(
            key,
            shape=(len(frames),) + sample.shape[1:],
            dtype=sample.dtype,
            chunks=(min(write_chunk_size, len(frames)),) + sample.shape[1:],
            **dataset_kwargs,
        )
        for i in range(0, len(frames), write_chunk_size):
            batch = frames[i : i + write_chunk_size]
            if self.cpu_wrapped_env:
                data = np.concatenate(batch, axis=0)
            elif key.startswith("panda-") and key[6:].isdigit():
                data = np.concatenate(batch, axis=0)
            else:
                data = np.concatenate([frame[:, env_idx] for frame in batch], axis=0)
            dataset[i : i + len(batch)] = data
            del data, batch

    def capture_image(self):
        if self.record_camera is not None:
            self.base_env.scene.update_render()
            sensor_images = self.base_env.get_sensor_images()
            camera_patterns = [c.strip() for c in self.record_camera.split(',')]
            if len(camera_patterns) > 1:
                images = []
                for pattern in camera_patterns:
                    for cam_name, cam_data in sensor_images.items():
                        if pattern in cam_name:
                            img = common.to_numpy(cam_data.get("rgb", cam_data))
                            if len(img.shape) > 3 and img.shape[0] == 1:
                                img = img[0]
                            images.append(img)
                            break
                if images:
                    target_h = max(img.shape[0] for img in images)
                    resized = []
                    for img in images:
                        if img.shape[0] != target_h:
                            import cv2
                            scale = target_h / img.shape[0]
                            new_w = int(img.shape[1] * scale)
                            img = cv2.resize(img, (new_w, target_h))
                        resized.append(img)
                    return np.concatenate(resized, axis=1)
            else:
                for cam_name, cam_data in sensor_images.items():
                    if self.record_camera in cam_name:
                        img = common.to_numpy(cam_data.get("rgb", cam_data))
                        if len(img.shape) > 3 and img.shape[0] == 1:
                            img = img[0]
                        return img
        return super().capture_image()

    def reset(
        self,
        *args,
        seed: Optional[Union[int, List[int]]] = None,
        options: Optional[dict] = dict(),
        **kwargs,
    ):

        if self.save_on_reset:
            if self.save_video and self.num_envs == 1:
                self.flush_video()
            # if doing a full reset then we flush all trajectories including incompleted ones
            if self._trajectory_buffer is not None:
                if "env_idx" not in options:
                    self.flush_trajectory(env_idxs_to_flush=np.arange(self.num_envs))
                else:
                    self.flush_trajectory(
                        env_idxs_to_flush=common.to_numpy(options["env_idx"])
                    )

        obs, info = super(RecordEpisode, self).reset(*args, seed=seed, options=options, **kwargs)
        if info["reconfigure"]:
            # if we reconfigure, there is the possibility that state dictionary looks different now
            # so trajectory buffer must be wiped
            self._trajectory_buffer = None
        if self.save_trajectory:
            state_dict = self.base_env.get_state_dict()
            action = common.batch(
                self.env.get_wrapper_attr("single_action_space").sample()
            )
            if isinstance(action, dict):
                action = {
                    agent_id: common.to_numpy(common.batch(act.repeat(self.num_envs, 0)))
                    for agent_id, act in action.items()
                }
            else:
                action = common.to_numpy(common.batch(action.repeat(self.num_envs, 0)))
            # check if state_dict is consistent
            if not sapien_utils.is_state_dict_consistent(state_dict):
                self.record_env_state = False
                if not self._already_warned_about_state_dict_inconsistency:
                    logger.warn(
                        f"State dictionary is not consistent, disabling recording of environment states for {self.env}"
                    )
                    self._already_warned_about_state_dict_inconsistency = True
            first_step = Step(
                state=None,
                observation=self._init_observation_buffer(obs),
                # note first reward/action etc. are ignored when saving trajectories to disk
                # action=common.to_numpy(common.batch(action.repeat(self.num_envs, 0))),
                action=action,
                reward=np.zeros(
                    (
                        1,
                        self.num_envs,
                    ),
                    dtype=float,
                ),
                # terminated and truncated are fixed to be True at the start to indicate the start of an episode.
                # an episode is done when one of these is True otherwise the trajectory is incomplete / a partial episode
                terminated=np.ones((1, self.num_envs), dtype=bool),
                truncated=np.ones((1, self.num_envs), dtype=bool),
                done=np.ones((1, self.num_envs), dtype=bool),
                success=np.zeros((1, self.num_envs), dtype=bool),
                fail=np.zeros((1, self.num_envs), dtype=bool),
                env_episode_ptr=np.zeros((self.num_envs,), dtype=int),
            )
            if self.record_env_state:
                first_step.state = common.to_numpy(common.batch(state_dict))
            env_idx = np.arange(self.num_envs)
            if "env_idx" in options:
                env_idx = common.to_numpy(options["env_idx"])
            # Initialize trajectory buffer on the first episode based on given observation (which should be generated after all wrappers)
            self._trajectory_buffer = first_step
        if options is not None and "env_idx" in options:
            options["env_idx"] = common.to_numpy(options["env_idx"])
        self.last_reset_kwargs = copy.deepcopy(dict(options=options, **kwargs))
        if seed is not None:
            self.last_reset_kwargs.update(seed=seed)
        return obs, info

    def step(self, action):
        if self.save_video and self._video_steps == 0:
            self.render_images.append(self.capture_image())
        obs, rew, terminated, truncated, info = super(RecordEpisode, self).step(action)

        if self.save_trajectory:
            if self.record_env_state:
                state_dict = self.base_env.get_state_dict()
                self._trajectory_buffer.state = common.append_dict_array(
                    self._trajectory_buffer.state,
                    common.to_numpy(common.batch(state_dict)),
                )
            self._append_observation_buffer(obs)
            self._trajectory_buffer.action = common.append_dict_array(
                self._trajectory_buffer.action,
                common.to_numpy(common.batch(action)),
            )
            if self.record_reward:
                self._trajectory_buffer.reward = common.append_dict_array(
                    self._trajectory_buffer.reward,
                    common.to_numpy(common.batch(rew)),
                )
            self._trajectory_buffer.terminated = common.append_dict_array(
                self._trajectory_buffer.terminated,
                common.to_numpy(common.batch(terminated)),
            )
            self._trajectory_buffer.truncated = common.append_dict_array(
                self._trajectory_buffer.truncated,
                common.to_numpy(common.batch(truncated)),
            )
            done = terminated | truncated
            self._trajectory_buffer.done = common.append_dict_array(
                self._trajectory_buffer.done,
                common.to_numpy(common.batch(done)),
            )
            if "success" in info:
                self._trajectory_buffer.success = common.append_dict_array(
                    self._trajectory_buffer.success,
                    common.to_numpy(common.batch(info["success"])),
                )
            else:
                self._trajectory_buffer.success = None
            if "fail" in info:
                self._trajectory_buffer.fail = common.append_dict_array(
                    self._trajectory_buffer.fail,
                    common.to_numpy(common.batch(info["fail"])),
                )
            else:
                self._trajectory_buffer.fail = None
            self._last_info = common.to_numpy(info)

        if self.save_video:
            self._video_steps += 1
            image = self.capture_image()

            if self.info_on_video:
                scalar_info = gym_utils.extract_scalars_from_info(common.to_numpy(info))
                if isinstance(rew, torch.Tensor) and len(rew.shape) > 1:
                    rew = rew[0]
                rew = float(common.to_numpy(rew))
                extra_texts = [
                    f"reward: {rew:.3f}",
                ]
                image = put_info_on_image(image, scalar_info, extras=extra_texts)

            self.render_images.append(image)
            if (
                self.max_steps_per_video is not None
                and self._video_steps >= self.max_steps_per_video
            ):
                self.flush_video()
        self._elapsed_record_steps += 1
        return obs, rew, terminated, truncated, info
    
    def flush_trajectory(
        self,
        verbose=False,
        ignore_empty_transition=True,
        env_idxs_to_flush=None,
        save: bool = True,
    ):
        """
        Flushes a trajectory and by default saves it to disk

        Arguments:
            verbose (bool): whether to print out information about the flushed trajectory
            ignore_empty_transition (bool): whether to ignore trajectories that did not have any actions
            env_idxs_to_flush: which environments by id to flush. If None, all environments are flushed.
            save (bool): whether to save the trajectory to disk
        """
        flush_count = 0
        if env_idxs_to_flush is None:
            env_idxs_to_flush = np.arange(0, self.num_envs)
        for env_idx in env_idxs_to_flush:
            start_ptr = self._trajectory_buffer.env_episode_ptr[env_idx]
            end_ptr = len(self._trajectory_buffer.done)
            if ignore_empty_transition and end_ptr - start_ptr <= 1:
                continue
            flush_count += 1
            if save:
                self._episode_id += 1
                traj_id = "traj_{}".format(self._episode_id)
                group = self._h5_file.create_group(traj_id, track_order=True)

                def recursive_add_to_h5py(
                    group: h5py.Group, data: Union[dict, Array], key
                ):
                    """simple recursive data insertion for nested data structures into h5py, optimizing for visual data as well"""
                    if isinstance(data, dict):
                        subgrp = group.create_group(key, track_order=True)
                        for k in data.keys():
                            recursive_add_to_h5py(subgrp, data[k], k)
                    else:
                        if key == "rgb":
                            # NOTE(jigu): It is more efficient to use gzip than png for a sequence of images.
                            group.create_dataset(
                                "rgb",
                                data=data[start_ptr:end_ptr, env_idx],
                                dtype=data.dtype,
                                compression="gzip",
                                compression_opts=5,
                            )
                        elif key == "depth":
                            # NOTE (stao): By default now cameras in ManiSkill return depth values of type uint16 for numpy
                            group.create_dataset(
                                key,
                                data=data[start_ptr:end_ptr, env_idx],
                                dtype=data.dtype,
                                compression="gzip",
                                compression_opts=5,
                            )
                        elif key == "seg":
                            group.create_dataset(
                                key,
                                data=data[start_ptr:end_ptr, env_idx],
                                dtype=data.dtype,
                                compression="gzip",
                                compression_opts=5,
                            )
                        elif key.startswith("panda-") and key[6:].isdigit():
                            group.create_dataset(
                                key,
                                data=data[start_ptr:end_ptr],
                                dtype=data.dtype,
                            )
                        else:
                            group.create_dataset(
                                key,
                                data=data[start_ptr:end_ptr, env_idx],
                                dtype=data.dtype,
                            )

                # Observations need special processing
                if self.record_observation:
                    if isinstance(self._trajectory_buffer.observation, list):
                        self._write_observation_buffer_to_h5(
                            group,
                            self._trajectory_buffer.observation,
                            "obs",
                            start_ptr,
                            end_ptr,
                            env_idx,
                        )
                    elif isinstance(self._trajectory_buffer.observation, dict):
                        recursive_add_to_h5py(
                            group, self._trajectory_buffer.observation, "obs"
                        )
                    elif isinstance(self._trajectory_buffer.observation, np.ndarray):
                        if self.cpu_wrapped_env:
                            group.create_dataset(
                                "obs",
                                data=self._trajectory_buffer.observation[start_ptr:end_ptr],
                                dtype=self._trajectory_buffer.observation.dtype,
                            )
                        else:
                            group.create_dataset(
                                "obs",
                                data=self._trajectory_buffer.observation[
                                    start_ptr:end_ptr, env_idx
                                ],
                                dtype=self._trajectory_buffer.observation.dtype,
                            )
                    else:
                        raise NotImplementedError(
                            f"RecordEpisode wrapper does not know how to handle observation data of type {type(self._trajectory_buffer.observation)}"
                        )
                episode_info = dict(
                    episode_id=self._episode_id,
                    episode_seed=self.base_env._episode_seed[env_idx],
                    control_mode=self.base_env.control_mode,
                    elapsed_steps=end_ptr - start_ptr - 1,
                )
                if self.num_envs == 1:
                    episode_info.update(reset_kwargs=self.last_reset_kwargs)
                else:
                    # NOTE (stao): With multiple envs in GPU simulation, reset_kwargs do not make much sense
                    episode_info.update(reset_kwargs=dict())

                # slice some data to remove the first dummy frame.
                actions = common.index_dict_array(
                    self._trajectory_buffer.action,
                    (slice(start_ptr + 1, end_ptr), env_idx),
                )
                terminated = self._trajectory_buffer.terminated[
                    start_ptr + 1 : end_ptr, env_idx
                ]
                truncated = self._trajectory_buffer.truncated[
                    start_ptr + 1 : end_ptr, env_idx
                ]
                if isinstance(self._trajectory_buffer.action, dict):
                    recursive_add_to_h5py(group, actions, "actions")
                else:
                    group.create_dataset("actions", data=actions, dtype=np.float32)
                group.create_dataset("terminated", data=terminated, dtype=bool)
                group.create_dataset("truncated", data=truncated, dtype=bool)

                if self._trajectory_buffer.success is not None:
                    end_ptr = len(self._trajectory_buffer.success)
                    group.create_dataset(
                        "success",
                        data=self._trajectory_buffer.success[
                            start_ptr + 1 : end_ptr, env_idx
                        ],
                        dtype=bool,
                    )
                    episode_info.update(
                        success=self._trajectory_buffer.success[end_ptr - 1, env_idx]
                    )
                if self._trajectory_buffer.fail is not None:
                    group.create_dataset(
                        "fail",
                        data=self._trajectory_buffer.fail[
                            start_ptr + 1 : end_ptr, env_idx
                        ],
                        dtype=bool,
                    )
                    episode_info.update(
                        fail=self._trajectory_buffer.fail[end_ptr - 1, env_idx]
                    )
                if self.record_env_state:
                    recursive_add_to_h5py(
                        group, self._trajectory_buffer.state, "env_states"
                    )
                if self.record_reward:
                    group.create_dataset(
                        "rewards",
                        data=self._trajectory_buffer.reward[
                            start_ptr + 1 : end_ptr, env_idx
                        ],
                        dtype=np.float32,
                    )

                self._json_data["episodes"].append(episode_info)
                dump_json(self._json_path, self._json_data, indent=2)
                if verbose:
                    if flush_count == 1:
                        print(f"Recorded episode {self._episode_id}")
                    else:
                        print(
                            f"Recorded episodes {self._episode_id - flush_count} to {self._episode_id}"
                        )

        # truncate self._trajectory_buffer down to save memory
        if flush_count > 0:
            self._trajectory_buffer.env_episode_ptr[env_idxs_to_flush] = (
                len(self._trajectory_buffer.done) - 1
            )
            min_env_ptr = self._trajectory_buffer.env_episode_ptr.min()
            N = len(self._trajectory_buffer.done)

            if self.record_env_state:
                self._trajectory_buffer.state = common.index_dict_array(
                    self._trajectory_buffer.state, slice(min_env_ptr, N)
                )
            self._trajectory_buffer.observation = self._slice_observation_buffer(
                self._trajectory_buffer.observation, slice(min_env_ptr, N)
            )
            self._trajectory_buffer.action = common.index_dict_array(
                self._trajectory_buffer.action, slice(min_env_ptr, N)
            )
            if self.record_reward:
                self._trajectory_buffer.reward = common.index_dict_array(
                    self._trajectory_buffer.reward, slice(min_env_ptr, N)
                )
            self._trajectory_buffer.terminated = common.index_dict_array(
                self._trajectory_buffer.terminated, slice(min_env_ptr, N)
            )
            self._trajectory_buffer.truncated = common.index_dict_array(
                self._trajectory_buffer.truncated, slice(min_env_ptr, N)
            )
            self._trajectory_buffer.done = common.index_dict_array(
                self._trajectory_buffer.done, slice(min_env_ptr, N)
            )
            if self._trajectory_buffer.success is not None:
                self._trajectory_buffer.success = common.index_dict_array(
                    self._trajectory_buffer.success, slice(min_env_ptr, N)
                )
            if self._trajectory_buffer.fail is not None:
                self._trajectory_buffer.fail = common.index_dict_array(
                    self._trajectory_buffer.fail, slice(min_env_ptr, N)
                )
            self._trajectory_buffer.env_episode_ptr -= min_env_ptr
            self._h5_file.flush()
            _trim_process_memory()

    def flush_video(
        self,
        name=None,
        suffix="",
        verbose=False,
        ignore_empty_transition=True,
        save: bool = True,
        split: bool = False,
        split_width: int = 256,
    ):
        """
        Flush a video of the recorded episode(s) anb by default saves it to disk

        Arguments:
            name (str): name of the video file. If None, it will be named with the episode id.
            suffix (str): suffix to add to the video file name
            verbose (bool): whether to print out information about the flushed video
            ignore_empty_transition (bool): whether to ignore trajectories that did not have any actions
            save (bool): whether to save the video to disk
        """
        if len(self.render_images) == 0:
            return
        if ignore_empty_transition and len(self.render_images) == 1:
            return
        if save:
            self._video_id += 1
            if name is None:
                video_name = "{}".format(self._video_id)
                if suffix:
                    video_name += "_" + suffix
                if self._avoid_overwriting_video:
                    while (
                        Path(self.output_dir)
                        / (video_name.replace(" ", "_").replace("\n", "_") + ".mp4")
                    ).exists():
                        self._video_id += 1
                        video_name = "{}".format(self._video_id)
                        if suffix:
                            video_name += "_" + suffix
            else:
                video_name = name
            if split:
                height, width, _ = self.render_images[0].shape
                for i in range(0, width, split_width):
                    split_images = []
                    for j in range(0, len(self.render_images)):
                        split_image = self.render_images[j][:, i : i + split_width, :]
                        split_images.append(split_image)
                    import os
                    output_dir = os.path.join(self.output_dir, video_name)
                    images_to_video(
                        split_images,
                        str(output_dir),
                        video_name=video_name + f"_{(int)(i / height)}",
                        fps=self.video_fps,
                        verbose=verbose,
                    )
            else:
                images_to_video(
                    self.render_images,
                    str(self.output_dir),
                    video_name=video_name,
                    fps=self.video_fps,
                    verbose=verbose,
                )
        self._video_steps = 0
        self.render_images = []

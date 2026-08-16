from typing import Optional, List
import pathlib
import numpy as np
import time
import shutil
import math

from multiprocessing.managers import SharedMemoryManager
from configs.camera_config import CameraConfig, DEFAULT_CAMERA_CONFIG
# from real_world.rokae.rokae_interpolation_controller import RokaeInterpolationController
# from real_world.pgi.pgi_controller import PGIController
from real_world.robot_api.arm.Controller import Controller
from real_world.multi_uvc_camera import MultiUvcCamera, VideoRecorder

from utils.interpolation_util import get_interp1d, PoseInterpolator
from utils.pose_util import pose_to_mat, mat_to_pose, pose_to_pos_quat, pos_quat_to_pose
from utils.cv_util import draw_fisheye_mask

import cv2
import time

from real_world.robot_api.arm.RobotControl_pykin import RobotControl


def _resize_panel_for_model(
    panel: np.ndarray,
    output_resolution: tuple[int, int],
    obs_float32: bool,
) -> np.ndarray:
    """Match collection-time panel resizing and convert decoded BGR to RGB."""
    resized = cv2.resize(
        panel,
        output_resolution,
        interpolation=cv2.INTER_LINEAR,
    )
    resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    if obs_float32:
        resized = resized.astype(np.float32) / 255
    return resized


def _select_alignment_camera_idx(camera_data: dict, num_obs_cameras: int) -> int:
    """Select a camera timestamp that has a non-future frame in every camera."""
    camera_timestamps = {}
    for camera_idx in range(num_obs_cameras):
        timestamps = np.asarray(camera_data[camera_idx]["timestamp"])
        if timestamps.size == 0:
            raise RuntimeError(
                f"camera {camera_idx} returned an empty timestamp buffer"
            )
        camera_timestamps[camera_idx] = timestamps

    align_camera_idx = None
    running_best_error = np.inf

    for camera_idx in range(num_obs_cameras):
        this_timestamp = camera_timestamps[camera_idx][-1]
        this_error = 0.0
        for other_camera_idx in range(num_obs_cameras):
            if other_camera_idx == camera_idx:
                continue

            other_timestamps = camera_timestamps[other_camera_idx]
            other_timestep_idx = (
                np.searchsorted(other_timestamps, this_timestamp, side="right") - 1
            )
            if other_timestep_idx < 0:
                this_error = np.inf
                break

            this_error += this_timestamp - other_timestamps[other_timestep_idx]

        if np.isfinite(this_error) and (
            align_camera_idx is None or this_error < running_best_error
        ):
            running_best_error = this_error
            align_camera_idx = camera_idx

    if align_camera_idx is None:
        timestamp_ranges = ", ".join(
            f"camera {camera_idx}=[{timestamps[0]}, {timestamps[-1]}]"
            for camera_idx, timestamps in camera_timestamps.items()
        )
        raise RuntimeError(
            "camera timestamp buffers have no valid alignment point: "
            f"{timestamp_ranges}"
        )

    return align_camera_idx


class BimanualUmiEnv:
    STARTUP_FAILURE_CLEANUP_TIMEOUT = 1.0

    def __init__(self,
            # required params
            cam_path=None,
            data_type='vision',
            fps_num_points=256,
            control_frequency=10,
            controller_frequency=100,
            obs_image_resolution=(224,224),

            max_obs_buffer_size=60,
            obs_float32=False,
            camera_obs_latency=0.125,

            camera_down_sample_steps=1,
            robot_down_sample_steps=1,
            gripper_down_sample_steps=1,
            camera_obs_horizon=2,
            robot_obs_horizon=2,
            gripper_obs_horizon=2,

            use_fisheye_mask=False,
            fisheye_mask_radius=400,
            fisheye_mask_center=None,
            fisheye_mask_fill_color=(0, 0, 0),

            shm_manager=None,
            quest_2_ee_left:np.ndarray = None,
            quest_2_ee_right:np.ndarray = None,
            width_slope:float = None,
            width_offset:float = None,
            max_gripper_speed:float = 0.01,
            max_pos_speed:float = 0.25,
            max_rot_speed:float = 0.16,
            max_action_pos_delta:float = 0.03,
            max_action_rot_delta:float = 0.35,
            single_arm_mode:bool = False,
            controller_launch_timeout_s:float = 20.0,
            controller_shutdown_timeout_s:float = 3.0,
            camera_config: CameraConfig = DEFAULT_CAMERA_CONFIG,
            ):

        res = camera_config.resolution
        total_width, total_height = res
        if total_width % 3 != 0:
            raise ValueError(
                "camera resolution must contain three equal-width panels; "
                f"width {total_width} is not divisible by 3"
            )
        panel_width = total_width // 3

        if shm_manager is None:
            shm_manager = SharedMemoryManager()
            shm_manager.start()

        resolution = list()
        capture_fps = list()
        cap_buffer_size = list()
        video_recorder = list()
        transform = list()

        for idx in range(len(cam_path)):
            fps = camera_config.capture_fps
            buf = camera_config.buffer_size
            bit_rate = 6000*1000

            # 创建transform函数，处理三联图像的裁剪和resize
            def create_transform_func(idx, res, data_type, obs_image_resolution, obs_float32):
                is_right = (idx == 1)  # idx=0是left hand

                def tf4k(data, input_res=res,
                        use_mask=use_fisheye_mask,
                        mask_radius=fisheye_mask_radius,
                        mask_center=fisheye_mask_center,
                        mask_fill_color=fisheye_mask_fill_color):
                    img = data['color']

                    # === 坏帧防御：解码失败或半截帧 ===
                    if img is None:
                        print(f"[tf4k cam{idx}] dropping bad frame: img is None")
                        return None
                    if not hasattr(img, 'shape') or len(img.shape) < 2:
                        print(f"[tf4k cam{idx}] dropping bad frame: invalid shape {getattr(img, 'shape', None)}")
                        return None
                    # === end 坏帧防御 ===

                    # Apply mask to all visual cameras (non-tactile mode has only visual cameras)
                    if use_mask:
                        # Apply mask before resize, consistent with training data processing order
                        img = draw_fisheye_mask(
                            img,
                            radius=mask_radius,
                            center=mask_center,
                            fill_color=mask_fill_color
                        )

                    # 验证图像尺寸 - 坏帧丢弃，不再 raise 把相机进程崩掉
                    h, w = img.shape[:2]
                    if w != total_width or h != total_height:
                        print(
                            f"[tf4k cam{idx}] dropping bad frame: expected "
                            f"{total_width}x{total_height}, got {w}x{h}"
                        )
                        return None

                    # 裁剪成三部分
                    left_tactile = img[:, 0:panel_width]
                    visual = img[:, panel_width:2*panel_width]
                    right_tactile = img[:, 2*panel_width:3*panel_width]

                    # Process
                    left_tactile = cv2.rotate(left_tactile, cv2.ROTATE_180)
                    visual = visual
                    right_tactile = right_tactile

                    # # left hand的visual旋转180度
                    # if is_right:
                    #     visual = cv2.rotate(visual, cv2.ROTATE_180)

                    # 处理visual（总是需要）
                    visual_resized = _resize_panel_for_model(
                        visual,
                        obs_image_resolution,
                        obs_float32,
                    )
                    data['color'] = visual_resized  # 统一存为color

                    # 根据data_type决定是否处理tactile
                    if data_type == 'vitac':
                        left_tactile_resized = _resize_panel_for_model(
                            left_tactile,
                            obs_image_resolution,
                            obs_float32,
                        )
                        right_tactile_resized = _resize_panel_for_model(
                            right_tactile,
                            obs_image_resolution,
                            obs_float32,
                        )
                        data['left_tactile'] = left_tactile_resized
                        data['right_tactile'] = right_tactile_resized

                    return data

                return tf4k

            transform.append(create_transform_func(idx, res, data_type, obs_image_resolution, obs_float32))

            resolution.append(res)
            capture_fps.append(fps)
            cap_buffer_size.append(buf)
            video_recorder.append(VideoRecorder.create_hevc_nvenc(  # TODO: why use hevc
                fps=fps,
                input_pix_fmt='bgr24',
                bit_rate=bit_rate
            ))

        camera = MultiUvcCamera(
            dev_video_paths=cam_path,
            shm_manager=shm_manager,
            resolution=resolution,
            capture_fps=capture_fps,

            put_fps=camera_config.capture_fps,
            put_downsample=True,

            get_max_k=max_obs_buffer_size,
            receive_latency=camera_obs_latency,
            cap_buffer_size=cap_buffer_size,
            transform=transform,
            video_recorder=video_recorder,
            camera_format=camera_config.pixel_format,
            auto_exposure=camera_config.auto_exposure,
            exposure=camera_config.exposure,
            auto_white_balance=camera_config.auto_white_balance,
            wb_temperature=camera_config.white_balance_temperature,
            brightness=camera_config.brightness,
            gain=camera_config.gain,
            gamma=camera_config.gamma,
            capture_timestamp_delay=camera_config.capture_timestamp_delay,
            verbose=False
        )

        self.camera = camera
        self.single_arm_mode = single_arm_mode
        self.controller = Controller(shm_manager=shm_manager,
            launch_timeout=controller_launch_timeout_s,
            frequency=controller_frequency,
            max_pos_speed=max_pos_speed,
            max_rot_speed=max_rot_speed,
            max_gripper_speed=max_gripper_speed,
            receive_latency=camera_obs_latency,
            single_arm_mode=self.single_arm_mode,
            shutdown_timeout_s=controller_shutdown_timeout_s,
        )
        self._controller_stop_failed = False
        self._controller_stop_error = None
        self.quest_2_ee_left = quest_2_ee_left
        self.quest_2_ee_right = quest_2_ee_right
        self.width_slope = width_slope
        self.width_offset = width_offset
        self.max_action_pos_delta = max_action_pos_delta
        self.max_action_rot_delta = max_action_rot_delta
        self.data_type = data_type
        self.cam_path = cam_path
        self.control_frequency = control_frequency
        self.last_camera_data = None

        self.camera_down_sample_steps = camera_down_sample_steps
        self.robot_down_sample_steps = robot_down_sample_steps
        self.gripper_down_sample_steps = gripper_down_sample_steps
        self.camera_obs_horizon = camera_obs_horizon
        self.robot_obs_horizon = robot_obs_horizon
        self.gripper_obs_horizon = gripper_obs_horizon
        self._last_log_time = {}

    def _rate_limited_log(self, key: str, message: str, interval_sec: float = 2.0) -> None:
        now = time.monotonic()
        last_log_time = self._last_log_time.get(key, 0.0)
        if now - last_log_time >= interval_sec:
            print(message)
            self._last_log_time[key] = now

    # ======== start-stop API =============
    #### 待修改
    @property
    def is_ready(self):
        ready_flag_camera = self.camera.is_ready
        ready_flag = ready_flag_camera and self.controller.is_ready
        return ready_flag

    def start(self, wait=True):
        self.camera.start(wait=False)
        self.controller.start(wait=False)
        if wait:
            try:
                self.start_wait()
            except Exception:
                children = (self.camera, self.controller)
                for child in children:
                    try:
                        child.stop(wait=False)
                    except Exception:
                        pass
                try:
                    self.stop_wait(timeout=self.STARTUP_FAILURE_CLEANUP_TIMEOUT)
                except Exception:
                    pass
                raise

    def stop(self, wait=True):
        first_error = None
        controller_stop_already_failed = getattr(
            self, "_controller_stop_failed", False
        )
        try:
            self.camera.stop(wait=False)
        except Exception as exc:
            first_error = exc

        if not controller_stop_already_failed:
            try:
                self.controller.stop(wait=False)
            except Exception as exc:
                self._controller_stop_failed = True
                self._controller_stop_error = exc
                if first_error is None:
                    first_error = exc
        if wait:
            try:
                self.stop_wait()
            except Exception as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

    def stop_controller(self, wait=True, timeout: float | None = None):
        """Stop and optionally join robot motion without waiting on cameras."""
        if getattr(self, "_controller_stop_failed", False):
            raise self._controller_stop_error
        try:
            if timeout is None:
                self.controller.stop(wait=wait)
            else:
                self.controller.stop(wait=wait, timeout=timeout)
        except Exception as exc:
            self._controller_stop_failed = True
            self._controller_stop_error = exc
            raise

    def check_controller_health(self):
        """Raise the controller's shared fatal error in the parent process."""
        self.controller.check_health()

    def start_wait(self):
        self.camera.start_wait()
        self.controller.start_wait()

    def stop_wait(self, timeout: float | None = None) -> None:
        deadline = None if timeout is None else time.monotonic() + max(timeout, 0.0)
        error = None
        children = [self.camera]
        if not getattr(self, "_controller_stop_failed", False):
            children.append(self.controller)
        for child in children:
            try:
                is_alive = getattr(child, "is_alive", None)
                if callable(is_alive) and not is_alive():
                    continue
                child_timeout = None if deadline is None else max(0.0, deadline - time.monotonic())
                child.stop_wait(timeout=child_timeout)
            except Exception as exc:
                if error is None:
                    error = exc
        if error is not None:
            raise error

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= async env API ==========
    def get_obs(self) -> dict:
        """
        Timestamp alignment policy
        We assume the cameras used for obs are always [0, k - 1], where k is the number of robots
        All other cameras, find corresponding frame with the nearest timestamp
        All low-dim observations, interpolate with respect to 'current' time
        """

        "observation dict"
        self.check_controller_health()
        if not self.camera.is_ready:
            raise RuntimeError("camera group is not ready")

        # get data
        # 60 Hz, camera_calibrated_timestamp (note: cameras capture at 30Hz, but using 60Hz for interpolation)
        k = math.ceil(
            self.camera_obs_horizon * self.camera_down_sample_steps \
            * (30 / self.control_frequency)) + 2 # here 2 is adjustable, typically 1 should be enough
        print(f"[get_obs] k={k}, obs_horizon={self.camera_obs_horizon}, "
      f"down_sample={self.camera_down_sample_steps}, "
      f"ctrl_freq={self.control_frequency}")
        # print('==>k  ', k, self.camera_obs_horizon, self.camera_down_sample_steps, self.control_frequency)

        'camera obs'
        self.last_camera_data = self.camera.get(
            k=k,
            out=self.last_camera_data)
        # print("camera get time:", time.time() - start_time)

        # select align_camera_idx based on calibrated timestamps
        # The timestamps are already calibrated in UvcCamera, so we use them directly
        num_obs_cameras = len(self.cam_path)


        align_camera_idx = _select_alignment_camera_idx(
            self.last_camera_data,
            num_obs_cameras,
        )

        last_timestamp = self.last_camera_data[align_camera_idx]['timestamp'][-1]

        dt = 1 / self.control_frequency

        # align camera obs timestamps
        # Since timestamps are already calibrated in UvcCamera, we can use them directly
        camera_obs_timestamps = last_timestamp - (
            np.arange(self.camera_obs_horizon)[::-1] * self.camera_down_sample_steps * dt)

        camera_obs = dict()
        for camera_idx, value in self.last_camera_data.items():
            this_timestamps = value['timestamp']
            this_idxs = list()
            for t in camera_obs_timestamps:
                nn_idx = np.argmin(np.abs(this_timestamps - t))
                # Optional: Add warning for large timestamp mismatches
                # if np.abs(this_timestamps - t)[nn_idx] > 1.0 / 60:
                #     print(f'WARNING: Large timestamp mismatch for camera {camera_idx}: {np.abs(this_timestamps - t)[nn_idx]:.4f}s')
                this_idxs.append(nn_idx)
            # remap key - 简化逻辑
            # camera_idx=0是left hand, camera_idx=1是right hand
            hand_idx = camera_idx  # 0=left hand (camera0), 1=right hand (camera1)

            # 提取visual (总是存在，存为color)
            camera_obs[f'camera{hand_idx}_rgb'] = value['color'][...,:3][this_idxs]

            # 如果是vitac模式，还需要提取tactile
            if self.data_type == 'vitac':
                if 'left_tactile' in value:
                    camera_obs[f'camera{hand_idx}_left_tactile'] = value['left_tactile'][...,:3][this_idxs]
                if 'right_tactile' in value:
                    camera_obs[f'camera{hand_idx}_right_tactile'] = value['right_tactile'][...,:3][this_idxs]

        '''robot obs'''
        last_robot_data = self.controller.get_all_state()

        # obs_data to return (it only includes camera data at this stage)
        obs_data = dict(camera_obs)
        # include camera timesteps
        obs_data['timestamp'] = camera_obs_timestamps

        # align robot obs
        robot_obs_timestamps = last_timestamp - (
            np.arange(self.robot_obs_horizon)[::-1] * self.robot_down_sample_steps * dt)

        # convert ee pose to quest pose
        quest_pose_left = mat_to_pose(pose_to_mat(last_robot_data['ee_pose_left']) @ self.quest_2_ee_left)
        robot_pose_left_interpolator = PoseInterpolator(
            t = last_robot_data['robot_timestamp'],
            x = quest_pose_left
        )
        robot_pose_left = robot_pose_left_interpolator(robot_obs_timestamps)
        if not self.single_arm_mode:
            quest_pose_right = mat_to_pose(pose_to_mat(last_robot_data['ee_pose_right']) @ self.quest_2_ee_right)
            robot_pose_right_interpolator = PoseInterpolator(
                t = last_robot_data['robot_timestamp'],
                x = quest_pose_right
            )
            robot_pose_right = robot_pose_right_interpolator(robot_obs_timestamps)
        else:
            robot_pose_right = None

        robot_obs = {
            'robot0_eef_pos': robot_pose_left[...,:3],
            'robot0_eef_rot_axis_angle': robot_pose_left[...,3:],
            'robot1_eef_pos': robot_pose_right[...,:3] if robot_pose_right is not None else None,
            'robot1_eef_rot_axis_angle': robot_pose_right[...,3:] if robot_pose_right is not None else None
        }
        obs_data.update(robot_obs)

        '''gripper obs'''
        # align gripper obs
        gripper_obs_timestamps = last_timestamp - (
            np.arange(self.gripper_obs_horizon)[::-1] * self.gripper_down_sample_steps * dt)

        # convert commanded gripper width to actual gripper width
        commanded_gripper_width_left = last_robot_data['gripper_pose_left']
        actual_gripper_width_left = self.width_slope * commanded_gripper_width_left + self.width_offset
        gripper_left_interpolator = get_interp1d(
            t= last_robot_data['robot_timestamp'],
            x= actual_gripper_width_left
        )
        gripper_left = gripper_left_interpolator(gripper_obs_timestamps)
        if not self.single_arm_mode:
            commanded_gripper_width_right = last_robot_data['gripper_pose_right']
            actual_gripper_width_right = self.width_slope * commanded_gripper_width_right + self.width_offset
            gripper_right_interpolator = get_interp1d(
                t= last_robot_data['robot_timestamp'],
                x= actual_gripper_width_right
            )
            gripper_right = gripper_right_interpolator(gripper_obs_timestamps)
        else:
            gripper_right = None

        gripper_obs = {
            'robot0_gripper_width': gripper_left,
            'robot1_gripper_width': gripper_right if gripper_right is not None else None
        }
        obs_data.update(gripper_obs)

        return obs_data

    def exec_actions(self,
            actions: np.ndarray,
            timestamps: np.ndarray):

        self.check_controller_health()
        if not isinstance(actions, np.ndarray):
            actions = np.array(actions)
        if not isinstance(timestamps, np.ndarray):
            timestamps = np.array(timestamps)

        # # 更新动作序列，确保全都是新动作
        # receive_time = time.time()
        # is_new = timestamps > receive_time
        # new_actions = actions[is_new]
        # new_timestamps = timestamps[is_new]

        # print(f"[env] exec {len(new_actions)}/{len(actions)} actions")
        # print("[env] receive_time:", int(receive_time * 1000) % 1000)
        # print("[env] new_timestamps:")
        # print([int(new_timestamps[i] * 1000) % 1000 for i in range(len(new_timestamps))])

        action_debug_records = []

        if len(actions) != 0:
            curr_robot_state = self.controller.get_state()
            prev_ee_pose_left = np.asarray(curr_robot_state['ee_pose_left'], dtype=np.float64)
            prev_ee_pose_right = (
                np.asarray(curr_robot_state['ee_pose_right'], dtype=np.float64)
                if not self.single_arm_mode else None
            )

            for i in range(len(actions)):
                index_left = 0
                index_right = 1

                quest_left_action = actions[i, 7 * index_left + 0: 7 * index_left + 6]
                gripper_left_action = actions[i, 7 * index_left + 6]
                # convert quest pose to ee pose
                target_ee_pose_left = mat_to_pose(pose_to_mat(quest_left_action) @ (np.linalg.inv(self.quest_2_ee_left)))
                # convert actual gripper width to commanded gripper width

                safe_range_max = 0.04
                safe_range_min = 0.01
                commanded_gripper_width_left = np.clip(
                    np.array([(gripper_left_action - self.width_offset) / self.width_slope]),
                    safe_range_min, safe_range_max
                    )

                if not self.single_arm_mode:
                    quest_right_action = actions[i, 7 * index_right + 0: 7 * index_right + 6]
                    gripper_right_action = actions[i, 7 * index_right + 6]
                    target_ee_pose_right = mat_to_pose(pose_to_mat(quest_right_action) @ (np.linalg.inv(self.quest_2_ee_right)))
                    # commanded_gripper_width_right = [(gripper_right_action - self.width_offset) / self.width_slope]
                    commanded_gripper_width_right = np.clip(
                    np.array([(gripper_right_action - self.width_offset) / self.width_slope]),
                        safe_range_min, safe_range_max
                        )
                else:
                    target_ee_pose_right = None
                    commanded_gripper_width_right = None

                delta_pos_left = float(np.linalg.norm(target_ee_pose_left[:3] - prev_ee_pose_left[:3]))
                delta_rot_left = float(np.linalg.norm(target_ee_pose_left[3:] - prev_ee_pose_left[3:]))

                if not self.single_arm_mode:
                    delta_pos_right = float(np.linalg.norm(target_ee_pose_right[:3] - prev_ee_pose_right[:3]))
                    delta_rot_right = float(np.linalg.norm(target_ee_pose_right[3:] - prev_ee_pose_right[3:]))
                else:
                    delta_pos_right = 0.0
                    delta_rot_right = 0.0

                action_record = {
                    "action_index": int(i),
                    "target_time": float(timestamps[i]),
                    "scheduled": False,
                    "skip_reason": None,
                    "left_delta_pos": delta_pos_left,
                    "left_delta_rot": delta_rot_left,
                    "right_delta_pos": delta_pos_right,
                    "right_delta_rot": delta_rot_right,
                    "left_target_pose": target_ee_pose_left.tolist(),
                    "right_target_pose": target_ee_pose_right.tolist() if target_ee_pose_right is not None else None,
                    "left_gripper": commanded_gripper_width_left.tolist(),
                    "right_gripper": commanded_gripper_width_right.tolist() if commanded_gripper_width_right is not None else None,
                }

                # print("[env]gripper:",commanded_gripper_width_left)
                # print(commanded_gripper_width_right)

                # DEBUG: 在这里检查 quest 相对 当前动作 的 dist
                # 使用 get_obs 获取机械臂当前位姿
                # curr_obs = self.get_obs()
                # curr_pos_left = curr_obs['robot0_eef_pos'][-1]
                # curr_rot_left = curr_obs['robot0_eef_rot_axis_angle'][-1]
                # curr_pos_right = curr_obs['robot1_eef_pos'][-1]
                # curr_rot_right = curr_obs['robot1_eef_rot_axis_angle'][-1]
                # curr_pose_left_mat = pose_to_mat(np.concatenate([curr_pos_left, curr_rot_left], axis=-1))
                # curr_pose_right_mat = pose_to_mat(np.concatenate([curr_pos_right, curr_rot_right], axis=-1))
                # 获取相对位姿
                # quest_left_action_mat = pose_to_mat(quest_left_action)
                # quest_right_action_mat = pose_to_mat(quest_right_action)
                # quest_left_action_rel_mat = np.linalg.inv(curr_pose_left_mat) @ quest_left_action_mat
                # quest_right_action_rel_mat = np.linalg.inv(curr_pose_right_mat) @ quest_right_action_mat
                # # 计算并输出Quest坐标系下的相对位姿
                # dist_left = np.linalg.norm(quest_left_action_rel_mat[:3, 3])
                # dist_right = np.linalg.norm(quest_right_action_rel_mat[:3, 3])
                # print(f"[env] #{i} action dist_left: {dist_left}, dist_right: {dist_right}")

                # HACK: 将 target_ee_pose_left 改为左手在curr_pose的基础上增加x，右手不动
                # curr_ee_pose_left_mat = curr_pose_left_mat @ (np.linalg.inv(self.quest_2_ee_left))
                # curr_ee_pose_right_mat = curr_pose_right_mat @ (np.linalg.inv(self.quest_2_ee_right))
                # target_ee_pose_left = mat_to_pose(curr_ee_pose_left_mat)
                # target_ee_pose_left[0] += 0.005
                # target_ee_pose_right = mat_to_pose(curr_ee_pose_right_mat)

                # 把目标动作序列 依次 发送给控制器
                if timestamps[i] < time.time():
                    self._rate_limited_log("late_action", "[env] action is too late")
                    action_record["skip_reason"] = "late_action"
                else:
                    self.controller.schedule_waypoint(
                        pose_left=target_ee_pose_left,
                        gripper_left=commanded_gripper_width_left,
                        pose_right=target_ee_pose_right,
                        gripper_right=commanded_gripper_width_right,
                            target_time=timestamps[i]
                    )
                    action_record["scheduled"] = True
                    prev_ee_pose_left = target_ee_pose_left
                    if not self.single_arm_mode:
                        prev_ee_pose_right = target_ee_pose_right
                action_debug_records.append(action_record)
        else:
            self._rate_limited_log("no_action", "[env] no action received")

        self.check_controller_health()
        return action_debug_records

    def get_debug_info(self):
        return self.controller.get_debug_info()

from typing import Any, Dict, Tuple, List
import numpy as np
from utils.pose_util import (
    pose_to_mat, mat_to_pose, 
    pose10d_to_pose_col)

def get_real_obs_resolution(
        shape_meta: dict
        ) -> Tuple[int, int]:
    out_res = None
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        type = attr.get('type', 'low_dim')
        shape = attr.get('shape')
        if type == 'rgb':
            co,ho,wo = shape
            if out_res is None:
                out_res = (wo, ho)
            assert out_res == (wo, ho)
    return out_res


def get_real_umi_obs_dict(
        env_obs: Dict[str, np.ndarray], 
        shape_meta: dict,
        obs_pose_repr: str='abs',
        episode_start_pose: List[np.ndarray]=None,
        data_type: str=None,
        cam_path: List[str]=None,
        task: str=None,
        no_state_obs_mode:bool=False
        ) -> Dict[str, np.ndarray]:
    # Keep these args in signature for compatibility with existing callers.
    del shape_meta, obs_pose_repr

    # 所有的观测量都只取最近的那一帧（[-1]）
    def _latest_hwc_uint8(key: str) -> np.ndarray:
        if key not in env_obs:
            raise KeyError(f'Missing required observation key: {key}')
        img = env_obs[key][-1]
        if img.dtype == np.uint8:
            return img
        # Convert float images (typically [0,1]) to uint8 for ViTaC policy input.
        if np.issubdtype(img.dtype, np.floating):
            max_v = float(np.max(img)) if img.size > 0 else 1.0
            if max_v <= 1.0:
                img = img * 255.0
            img = np.clip(img, 0.0, 255.0)
            return img.astype(np.uint8)
        return np.clip(img, 0, 255).astype(np.uint8)

    def _latest_feature(key: str) -> np.ndarray:
        if key not in env_obs:
            raise KeyError(f'Missing required observation key: {key}')
        return np.asarray(env_obs[key][-1], dtype=np.float32).reshape(-1)

    def _latest_pose6d(robot_idx: int) -> np.ndarray:
        pose6d = np.concatenate([
            _latest_feature(f'robot{robot_idx}_eef_pos'),
            _latest_feature(f'robot{robot_idx}_eef_rot_axis_angle')
        ], axis=-1)
        if pose6d.shape[0] != 6:
            raise ValueError(f'robot{robot_idx} pose must be 6D, got {pose6d.shape[0]}')
        return pose6d

    def _start_pose6d(robot_idx: int) -> np.ndarray:
        start_pose = np.asarray(episode_start_pose[robot_idx], dtype=np.float32).reshape(-1)
        if start_pose.shape[0] != 6:
            raise ValueError(
                f'episode_start_pose[{robot_idx}] must be 6D (xyz+axis-angle), '
                f'got {start_pose.shape[0]}'
            )
        return start_pose

    # Get state array
    state = []
    if not no_state_obs_mode:
        for idx in range(len(cam_path)):
            pose_mat = pose_to_mat(_latest_pose6d(idx))
            start_pose_mat = pose_to_mat(_start_pose6d(idx))
            rel_start6d = mat_to_pose(np.linalg.inv(start_pose_mat) @ pose_mat).reshape(-1)
            gripper_width = _latest_feature(f'robot{idx}_gripper_width')
            state.extend(rel_start6d)
            state.extend(gripper_width)
        
        if len(cam_path) >= 2:
            left_pose_mat = pose_to_mat(_latest_pose6d(0))
            right_pose_mat = pose_to_mat(_latest_pose6d(1))
            left_rel_right6d = mat_to_pose(np.linalg.inv(right_pose_mat) @ left_pose_mat).reshape(-1)
            state.extend(left_rel_right6d)

            if len(state) != len(cam_path) * 7 + 6:
                raise ValueError(f'Expected 20D state, got {len(state)}D')
        else:
            if len(state) != len(cam_path) * 7:
                raise ValueError(f'Expected 7D state, got {len(state)}D')
    else:
        for idx in range(len(cam_path)):
            # only gripper width in state
            gripper_width = _latest_feature(f'robot{idx}_gripper_width')
            state.extend(gripper_width)

    state = np.array(state, dtype=np.float32)
    # print("[CHECK] state dim:", state.shape)

    # Get cam obs
    obs_dict_np = dict()
    # print("data_type:", data_type)
    if data_type == 'vision':
        for idx in range(len(cam_path)):
            obs_dict_np[f'observation.images.camera{idx}'] = _latest_hwc_uint8(f'camera{idx}_rgb')
    elif data_type == 'vitac':
        for idx in range(len(cam_path)):
            obs_dict_np[f'observation.images.camera{idx}'] = _latest_hwc_uint8(f'camera{idx}_rgb')
            obs_dict_np[f'observation.images.tactile_left_{idx}'] = _latest_hwc_uint8(f'camera{idx}_left_tactile')
            obs_dict_np[f'observation.images.tactile_right_{idx}'] = _latest_hwc_uint8(f'camera{idx}_right_tactile')
    
    obs_dict_np['observation.state'] = state
    obs_dict_np['task'] = task
    
    return obs_dict_np

def get_real_umi_action(
        action: np.ndarray,
        env_obs: Dict[str, np.ndarray], 
        action_pose_repr: str='abs'
    ):

    n_robots = int(action.shape[-1] // 10)
    env_action = list[Any]()
    for robot_idx in range(n_robots):
        # convert pose to mat
        curr2base_mat = pose_to_mat(np.concatenate([
            env_obs[f'robot{robot_idx}_eef_pos'][-1], # is Quest pos
            env_obs[f'robot{robot_idx}_eef_rot_axis_angle'][-1]
        ], axis=-1))

        start = robot_idx * 10
        action_pose10d = action[..., start:start+9]
        action_grip = action[..., start+9:start+10]
        # 注意！！！UMI之前用的一直是旋转矩阵前两*行*，但我们现在改用了前两列。所以这里要改成我们自己的
        action_pose_mat = pose10d_to_pose_col(action_pose10d)

        # # HACK: 验证obs中的action是否有误，先去掉传入的action_pose_mat，用eye(4)代替；grip同理
        # action_pose_mat = np.array([np.eye(4) for _ in range(action_pose_mat.shape[0])])
        # action_grip = np.array([[0.0] for _ in range(action_grip.shape[0])])

        # # HACK: 单独让左手的x增大、右手不动，从而确定输入量与输出量之间是否存在偏差。其他坐标不变
        # if robot_idx == 0:
        #     for i in range(len(action_pose_mat)):
        #         action_pose_mat[i, 0,3] = 0.005 *i
        # elif robot_idx == 1:
        #     # for i in range(len(action_pose_mat)):
        #     #     action_pose_mat[i, 0,3] = -0.001 *i
        #     pass
        
        # DEBUG: 检查每个action的平移距离，保留四位小数
        # dist_list = list[float]()
        # for i in range(len(action_pose_mat)):
        #     dist = np.linalg.norm(action_pose_mat[i, :3, 3])
        #     dist_list.append(round(dist, 4))
        # print("dist_list generated by policy:", dist_list)

        # solve absolute action
        # action_mat = convert_pose_mat_rep(
        #     pose_mat=action_pose_mat, 
        #     base_pose_mat=curr_pose_mat,
        #     pose_rep=action_pose_repr,
        #     backward=True)

        action_mat = np.empty_like(action_pose_mat)
        for i, next2curr_mat in enumerate(action_pose_mat):
            curr2base_mat = curr2base_mat @ next2curr_mat
            action_mat[i] = curr2base_mat

        # convert action to pose
        action_pose = mat_to_pose(action_mat)
        env_action.append(action_pose)
        env_action.append(action_grip)

    env_action = np.concatenate(env_action, axis=-1)
    # print("target_pose (from action):", env_action)
    return env_action

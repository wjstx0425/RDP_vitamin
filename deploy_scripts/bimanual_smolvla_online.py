import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)

from configs.server_config import SERVER_CONFIG

import time
from multiprocessing.managers import SharedMemoryManager
from pathlib import Path
from datetime import datetime
import threading
from queue import Empty, Queue
import json
from typing import NamedTuple

import click
import cv2
import numpy as np

# Plot generation is disabled for deployment; JSONL action logs remain enabled.
# import plotly.graph_objects as go

from client.robot_client import RobotClient
from utils.precise_sleep import precise_wait
from utils.pose_util import mat_to_pose, pose10d_to_pose_col
from real_world.real_inference_util import get_real_umi_obs_dict, get_real_umi_action
from deploy_scripts.vbvla_dry_run import run_dry_run
from deploy_scripts.vbvla_safety import (
    ActionTimeout,
    ClientDisconnected,
    ClientStopRequested,
    FreshActions,
    UnsafeActionError,
    validate_action_chunk,
    validate_safety_limits,
    wait_for_action_or_stop,
    wait_for_start,
)

DEFAULT_TOKEN_LIST_PATH = Path(ROOT_DIR) / "token_list.txt"
REPOSITORY_ROOT = Path(ROOT_DIR).resolve()
DEFAULT_LEFT_CALIBRATION_PATH = REPOSITORY_ROOT / "quest_2_ee_left_hand_fix_quest.npy"
DEFAULT_RIGHT_CALIBRATION_PATH = REPOSITORY_ROOT / "quest_2_ee_right_hand_fix_quest.npy"
SMOLVLA_ACTION_HORIZON = SERVER_CONFIG.action_horizon
SMOLVLA_N_ROBOTS = SERVER_CONFIG.n_robots
SMOLVLA_ACTION_DIM = SERVER_CONFIG.action_dim
ROTATION_6D_EPS = SERVER_CONFIG.rotation_6d_eps
SMOLVLA_MIN_GRIPPER = SERVER_CONFIG.min_gripper
SMOLVLA_MAX_GRIPPER = SERVER_CONFIG.max_gripper
SMOLVLA_OBSERVATION_RESOLUTION = SERVER_CONFIG.observation_resolution
RDP_OBSERVATION_RESOLUTION = SERVER_CONFIG.rdp_observation_resolution


class ActionChunkResult(NamedTuple):
    validated: np.ndarray
    action_timestamps: np.ndarray
    fresh: FreshActions
    conversion_obs: dict
    latency: float
    controller_records: list[dict]


def _validate_max_executed_actions(max_executed_actions: int) -> int:
    expected_range = f"[1, {SMOLVLA_ACTION_HORIZON}]"
    if isinstance(max_executed_actions, (bool, np.bool_)) or not isinstance(
        max_executed_actions, (int, np.integer)
    ):
        raise ValueError(
            f"max_executed_actions must be an integer in {expected_range}"
        )
    value = int(max_executed_actions)
    if not 1 <= value <= SMOLVLA_ACTION_HORIZON:
        raise ValueError(
            f"max_executed_actions must be an integer in {expected_range}"
        )
    return value


def limit_fresh_actions(
    fresh: FreshActions,
    max_executed_actions: int,
) -> FreshActions:
    """Retain at most the first N fresh actions while preserving horizon indices."""
    limit = _validate_max_executed_actions(max_executed_actions)
    selected_indices = np.flatnonzero(fresh.mask)[:limit]
    selected_mask = np.zeros_like(fresh.mask, dtype=bool)
    selected_mask[selected_indices] = True
    selected_count = len(selected_indices)
    return FreshActions(
        mask=selected_mask,
        raw=fresh.raw[:selected_count],
        absolute=fresh.absolute[:selected_count],
        timestamps=fresh.timestamps[:selected_count],
    )


def execute_action_chunk(
    raw_action,
    *,
    action_horizon,
    n_robots,
    max_pos_delta,
    max_rot_delta,
    min_gripper,
    max_gripper,
    obs_timestamp,
    now,
    dt,
    exec_mode,
    env,
    converter,
    action_pose_repr,
    before_execute=None,
    max_executed_actions: int = SMOLVLA_ACTION_HORIZON,
    schedule_from_receive: bool = False,
    command_lead_s: float = SERVER_CONFIG.rdp_command_lead_s,
) -> ActionChunkResult:
    """Validate the full chunk, then convert only capped fresh actions."""
    try:
        observation_time = float(obs_timestamp)
        action_dt = float(dt)
    except (TypeError, ValueError, OverflowError) as exc:
        raise UnsafeActionError("observation timestamp and dt must be finite scalars") from exc
    if not np.isfinite(observation_time):
        raise UnsafeActionError("observation timestamp must be finite")
    if not np.isfinite(action_dt) or action_dt <= 0:
        raise UnsafeActionError("dt must be finite and positive")

    if isinstance(action_horizon, (bool, np.bool_)) or not isinstance(
        action_horizon, (int, np.integer)
    ):
        raise UnsafeActionError(
            f"SmolVLA action_horizon must be an integer in "
            f"[1, {SMOLVLA_ACTION_HORIZON}]"
        )
    action_horizon = int(action_horizon)
    expected_action_shape = (action_horizon, SMOLVLA_ACTION_DIM)
    action_shape = np.asarray(raw_action).shape
    if (
        not 1 <= action_horizon <= SMOLVLA_ACTION_HORIZON
        or n_robots != SMOLVLA_N_ROBOTS
        or action_shape != expected_action_shape
    ):
        raise UnsafeActionError(
            f"SmolVLA action chunk shape {action_shape} does not match configured "
            f"shape {expected_action_shape} for {SMOLVLA_N_ROBOTS} robots; "
            f"action_horizon must be in [1, {SMOLVLA_ACTION_HORIZON}]"
        )
    validated = validate_action_chunk(
        raw_action,
        action_horizon=action_horizon,
        n_robots=SMOLVLA_N_ROBOTS,
        max_pos_delta=max_pos_delta,
        max_rot_delta=max_rot_delta,
        min_gripper=min_gripper,
        max_gripper=max_gripper,
    )

    schedule_now = float(now())
    if not np.isfinite(schedule_now):
        raise UnsafeActionError("schedule time must be finite")
    try:
        command_lead_s = float(command_lead_s)
    except (TypeError, ValueError, OverflowError) as exc:
        raise UnsafeActionError("command lead time must be a finite scalar") from exc
    if not np.isfinite(command_lead_s) or command_lead_s < 0:
        raise UnsafeActionError("command lead time must be finite and non-negative")
    latency = schedule_now - observation_time + 0.01
    if not np.isfinite(latency):
        raise UnsafeActionError("action latency must be finite")
    if schedule_from_receive:
        first_timestamp = schedule_now + command_lead_s
    elif exec_mode == "block":
        first_timestamp = observation_time + latency
    elif exec_mode == "rtc":
        first_timestamp = observation_time
    else:
        raise ValueError(f"unsupported exec_mode: {exec_mode}")
    action_timestamps = (
        np.arange(action_horizon, dtype=np.float64) * action_dt + first_timestamp
    )
    if not np.isfinite(action_timestamps).all():
        raise UnsafeActionError("action timestamps must be finite")
    if not (np.diff(action_timestamps) > 0).all():
        raise UnsafeActionError("action timestamps must be strictly increasing")

    conversion_obs = env.get_obs()
    stale_cutoff = float(now())
    if not np.isfinite(stale_cutoff):
        raise UnsafeActionError("stale cutoff time must be finite")

    execution_limit = _validate_max_executed_actions(max_executed_actions)
    fresh_indices = np.flatnonzero(action_timestamps > stale_cutoff)
    selected_indices = fresh_indices[:execution_limit]
    selected_mask = np.zeros_like(action_timestamps, dtype=bool)
    selected_mask[selected_indices] = True
    selected_raw = validated[selected_indices]

    converted = np.asarray(
        converter(selected_raw, conversion_obs, action_pose_repr)
    )
    expected_converted_shape = (len(selected_raw), SMOLVLA_N_ROBOTS * 7)
    if converted.shape != expected_converted_shape:
        raise UnsafeActionError(
            f"converted action shape {converted.shape} does not match {expected_converted_shape}"
        )
    if not np.isfinite(converted).all():
        raise UnsafeActionError("converted action contains non-finite values")

    fresh = FreshActions(
        mask=selected_mask,
        raw=selected_raw,
        absolute=converted,
        timestamps=action_timestamps[selected_indices],
    )
    if before_execute is not None:
        before_execute()
    if len(fresh.absolute) > 0:
        controller_records = env.exec_actions(
            actions=fresh.absolute,
            timestamps=fresh.timestamps,
        )
    else:
        controller_records = []

    return ActionChunkResult(
        validated=validated,
        action_timestamps=action_timestamps,
        fresh=fresh,
        conversion_obs=conversion_obs,
        latency=latency,
        controller_records=controller_records,
    )


def execute_action_chunk_and_publish_ack(client, obs_seq, *args, **kwargs):
    """Acknowledge only after a valid chunk has been consumed or scheduled."""
    def ensure_action_is_current() -> None:
        stop_check = getattr(client, "is_stop_requested", None)
        if stop_check is not None and stop_check():
            raise ClientStopRequested("client requested stop before action execution")
        if not client.action_origin_is_connected(obs_seq):
            raise ClientDisconnected("action origin disconnected before execution")

    result = execute_action_chunk(
        *args,
        before_execute=ensure_action_is_current,
        **kwargs,
    )
    env = kwargs.get("env")
    health_check = getattr(env, "check_controller_health", None)
    if health_check is not None:
        health_check()
    ensure_action_is_current()
    client.publish_action_ack(obs_seq)
    return result


def _coerce_smolvla_integer(config: dict, key: str) -> int:
    value = config[key]
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"SmolVLA {key} must be a finite integer")
    numeric = float(value)
    if not np.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"SmolVLA {key} must be a finite integer")
    return int(numeric)


def validate_smolvla_config(config_dict: dict) -> dict:
    required = {
        "data_type", "language_prompt", "control_frequency", "controller_frequency",
        "single_arm_mode", "no_state_obs_mode", "steps_per_inference", "action_horizon",
    }
    missing = sorted(required - config_dict.keys())
    if missing:
        raise ValueError(f"Missing SmolVLA config keys: {missing}")
    config = dict(config_dict)
    config["control_frequency"] = float(config["control_frequency"])
    config["policy_type"] = str(config.get("policy_type", "smolvla")).lower()
    if config["policy_type"] not in {"smolvla", "rdp"}:
        raise ValueError("policy_type must be 'smolvla' or 'rdp'")
    config["controller_frequency"] = float(config["controller_frequency"])
    config["steps_per_inference"] = _coerce_smolvla_integer(config, "steps_per_inference")
    config["action_horizon"] = _coerce_smolvla_integer(config, "action_horizon")
    if config["data_type"] not in {"vision", "vitac"}:
        raise ValueError("SmolVLA data_type must be 'vision' or 'vitac'")
    if config["single_arm_mode"] is not False:
        raise ValueError("SmolVLA single_arm_mode must be false")
    if config["no_state_obs_mode"] is not False:
        raise ValueError("SmolVLA no_state_obs_mode must be false")
    if not np.isfinite(config["control_frequency"]) or config["control_frequency"] <= 0:
        raise ValueError("SmolVLA control_frequency must be finite and positive")
    if not np.isfinite(config["controller_frequency"]) or config["controller_frequency"] <= 0:
        raise ValueError("SmolVLA controller_frequency must be finite and positive")
    if not 1 <= config["action_horizon"] <= SMOLVLA_ACTION_HORIZON:
        raise ValueError(
            f"SmolVLA action_horizon must be in [1, {SMOLVLA_ACTION_HORIZON}]"
        )
    if not 1 <= config["steps_per_inference"] <= config["action_horizon"]:
        raise ValueError("SmolVLA steps_per_inference must be in [1, action_horizon]")
    if config["policy_type"] == "rdp" and (
        config["data_type"] != "vitac"
        or config["action_horizon"] != 1
        or config["steps_per_inference"] != 1
    ):
        raise ValueError("RDP requires vitac observations and one-step actions")
    return config


def validate_smolvla_action_chunk(raw_action) -> np.ndarray:
    action = np.asarray(raw_action, dtype=np.float32)
    expected_shape = (SMOLVLA_ACTION_HORIZON, SMOLVLA_ACTION_DIM)
    if action.shape != expected_shape:
        raise ValueError(f"SmolVLA action shape must be {expected_shape}, got {action.shape}")
    if not np.isfinite(action).all():
        raise ValueError("SmolVLA action must contain only finite values")
    for robot_idx in range(SMOLVLA_N_ROBOTS):
        start = robot_idx * 10 + 3
        first_col = action[:, start : start + 3]
        second_col = action[:, start + 3 : start + 6]
        if (np.linalg.norm(first_col, axis=1) <= ROTATION_6D_EPS).any() or (
            np.linalg.norm(second_col, axis=1) <= ROTATION_6D_EPS
        ).any() or (
            np.linalg.norm(np.cross(first_col, second_col), axis=1) <= ROTATION_6D_EPS
        ).any():
            raise ValueError("SmolVLA rotation-6D columns must be non-zero and non-collinear")
    return action


def prepare_smolvla_actions(
        raw_action,
        obs,
        action_timestamps,
        curr_time,
        action_pose_repr,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    action = validate_smolvla_action_chunk(raw_action)
    timestamps = np.asarray(action_timestamps, dtype=np.float64)
    if timestamps.shape != (SMOLVLA_ACTION_HORIZON,):
        raise ValueError(
            "SmolVLA action timestamps must have shape "
            f"({SMOLVLA_ACTION_HORIZON},)"
        )
    if not np.isfinite(timestamps).all():
        raise ValueError("SmolVLA action timestamps must contain only finite values")
    if not (np.diff(timestamps) > 0).all():
        raise ValueError("SmolVLA action timestamps must be strictly increasing")
    current_time = float(curr_time)
    if not np.isfinite(current_time):
        raise ValueError("SmolVLA curr_time must be finite")
    is_new = timestamps > current_time
    new_raw = action[is_new]
    absolute = get_real_umi_action(new_raw, obs, action_pose_repr)
    expected_absolute_shape = (len(new_raw), SMOLVLA_N_ROBOTS * 7)
    if absolute.shape != expected_absolute_shape:
        raise ValueError(f"Decoded SmolVLA action has unexpected shape {absolute.shape}")
    if not np.isfinite(absolute).all():
        raise ValueError("Decoded SmolVLA action must contain only finite values")
    return action, new_raw, absolute, timestamps[is_new], is_new


def _stop_robot_client(client) -> None:
    client.stop()
    client.join(timeout=1.0)


def wait_for_smolvla_config(client, deadline, monotonic=time.monotonic):
    remaining = deadline - monotonic()
    if remaining <= 0 or not client.wait_for_connection(timeout=remaining):
        _stop_robot_client(client)
        return None

    print("Waiting for config", flush=True)
    remaining = deadline - monotonic()
    if remaining <= 0:
        _stop_robot_client(client)
        return None

    config = client.wait_for_config(timeout=remaining)
    if config is None:
        _stop_robot_client(client)
        return None
    return config


def load_token_list(token_file: str) -> list[str]:
    token_path = Path(token_file)
    if not token_path.is_absolute():
        token_path = (Path(ROOT_DIR) / token_path).resolve()

    if not token_path.exists():
        raise click.ClickException(f"Token list file not found: {token_path}")

    token_list = []
    with token_path.open("r", encoding="utf-8") as f:
        for line in f:
            token = line.strip()
            if not token or token.startswith("#"):
                continue
            token_list.append(token)

    if not token_list:
        raise click.ClickException(f"No valid tokens found in {token_path}")

    return token_list


class ObsSaver:
    """异步保存observation数据，不影响eval过程"""

    def __init__(self, save_dir: str, data_type: str):
        """
        Args:
            save_dir: 保存目录
            data_type: 数据类型 ('vision' 或 'vitac')
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = Path(save_dir) / f"eval_obs_{timestamp}"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.data_type = data_type

        # 使用队列进行异步保存
        self.save_queue = Queue(maxsize=100)  # 限制队列大小，避免内存溢出
        self.save_thread = None
        self.running = False
        self.step_count = 0

        print(f"[ObsSaver] Initialized. Save directory: {self.save_dir}")

    def start(self):
        """启动保存线程"""
        self.running = True
        self.save_thread = threading.Thread(target=self._save_worker, daemon=True)
        self.save_thread.start()
        print(f"[ObsSaver] Started saving thread")

    def stop(self):
        """停止保存线程"""
        self.running = False
        if self.save_thread:
            self.save_thread.join(timeout=5.0)
        print(f"[ObsSaver] Stopped. Total steps saved: {self.step_count}")

    def save_obs(self, obs: dict, step_idx: int = None):
        """
        将obs添加到保存队列（非阻塞）

        Args:
            obs: observation字典
            step_idx: 步骤索引（如果为None，使用内部计数器）
        """
        if not self.running:
            return

        if step_idx is None:
            step_idx = self.step_count
            self.step_count += 1

        try:
            # 非阻塞添加，如果队列满了就跳过
            self.save_queue.put_nowait((step_idx, obs))
        except:
            # 队列满了，跳过这次保存
            pass

    def _save_worker(self):
        """后台保存线程"""
        while self.running:
            try:
                # 从队列获取数据，超时1秒
                step_idx, obs = self.save_queue.get(timeout=1.0)
                self._save_single_obs(step_idx, obs)
                self.save_queue.task_done()
            except:
                continue

    def _numpy_to_json_serializable(self, obj):
        """将numpy数组转换为JSON可序列化的格式"""
        if isinstance(obj, np.ndarray):
            # 转换为列表
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            # numpy标量转换为Python原生类型
            return obj.item()
        elif isinstance(obj, dict):
            return {k: self._numpy_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._numpy_to_json_serializable(item) for item in obj]
        else:
            return obj

    def _save_single_obs(self, step_idx: int, obs: dict):
        """保存单个observation - 保存所有obs数据"""
        step_dir = self.save_dir / f"step_{step_idx:06d}"
        step_dir.mkdir(exist_ok=True)

        # 保存时间戳为JSON
        if 'timestamp' in obs:
            timestamp_data = self._numpy_to_json_serializable(obs['timestamp'])
            with open(step_dir / "timestamp.json", 'w') as f:
                json.dump(timestamp_data, f, indent=2)

        # 遍历所有obs数据并保存
        for key, value in obs.items():
            if key == 'timestamp':
                continue

            if isinstance(value, np.ndarray) and len(value.shape) >= 3:
                # 检查是否是图像数据（camera, rgb, tactile相关）
                if 'camera' in key or 'rgb' in key or 'tactile' in key:
                    # 保存为图像文件（取最后一帧）
                    if len(value.shape) == 4:  # (T, H, W, C)
                        img = value[-1]  # 取最后一帧
                    elif len(value.shape) == 3:  # (H, W, C)
                        img = value
                    else:
                        # 不是标准图像格式，保存为JSON
                        json_data = self._numpy_to_json_serializable(value)
                        with open(step_dir / f"{key}.json", 'w') as f:
                            json.dump(json_data, f, indent=2)
                        continue

                    # 转换数据类型和格式
                    if img.dtype == np.float32:
                        img = (img * 255).astype(np.uint8)
                    elif img.max() <= 1.0 and img.dtype in [np.float32, np.float64]:
                        img = (img * 255).astype(np.uint8)

                    # RGB转BGR用于cv2保存
                    # if len(img.shape) == 3 and img.shape[-1] == 3:
                    #     img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    # else:
                    img_path = step_dir / f"{key}.jpg"
                    cv2.imwrite(str(img_path), img)
                else:
                    # 非图像数据，保存为JSON（包括robot pose, gripper width等）
                    json_data = self._numpy_to_json_serializable(value)
                    with open(step_dir / f"{key}.json", 'w') as f:
                        json.dump(json_data, f, indent=2)
            elif isinstance(value, np.ndarray):
                # 低维数据（robot pose, gripper width等），保存为JSON
                json_data = self._numpy_to_json_serializable(value)
                with open(step_dir / f"{key}.json", 'w') as f:
                    json.dump(json_data, f, indent=2)
            else:
                # 其他类型数据，保存为JSON
                json_data = self._numpy_to_json_serializable(value)
                with open(step_dir / f"{key}.json", 'w') as f:
                    json.dump(json_data, f, indent=2)


class ActionDebugLogger:
    """Append action diagnostics to jsonl and plot them at shutdown."""

    def __init__(self, save_dir: str, sides: list[str]):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = Path(save_dir) / "action_debug_logs" / timestamp
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.save_dir / "action_debug.jsonl"
        self.sides = sides
        self._file = self.jsonl_path.open("a", encoding="utf-8", buffering=1)
        print(f"[ActionDebugLogger] Writing action debug logs to {self.jsonl_path}")

    @staticmethod
    def _jsonable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, dict):
            return {k: ActionDebugLogger._jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [ActionDebugLogger._jsonable(v) for v in obj]
        return obj

    def _raw_delta_metrics(self, actions: np.ndarray, n_robots: int) -> dict:
        actions = np.asarray(actions)
        metrics = {}
        if actions.size == 0 or len(actions) == 0:
            return {side: {"pos": [], "rot": [], "grip": []} for side in self.sides[:n_robots]}

        for robot_idx, side in enumerate(self.sides[:n_robots]):
            start = robot_idx * 10
            pose10d = actions[:, start:start + 9]
            grip = actions[:, start + 9:start + 10]
            rel_pose = mat_to_pose(pose10d_to_pose_col(pose10d))
            metrics[side] = {
                "pos": np.linalg.norm(rel_pose[:, :3], axis=-1).tolist(),
                "rot": np.linalg.norm(rel_pose[:, 3:], axis=-1).tolist(),
                "grip": grip.reshape(-1).tolist(),
            }
        return metrics

    def _new_action_jump_metrics(self, actions: np.ndarray, env_obs: dict, n_robots: int) -> dict:
        actions = np.asarray(actions)
        metrics = {}
        for robot_idx, side in enumerate(self.sides[:n_robots]):
            if actions.size == 0 or len(actions) == 0:
                metrics[side] = {"pos": [], "rot": [], "grip": []}
                continue

            start = robot_idx * 7
            prev_pose = np.concatenate([
                env_obs[f'robot{robot_idx}_eef_pos'][-1],
                env_obs[f'robot{robot_idx}_eef_rot_axis_angle'][-1],
            ], axis=-1)
            pos_delta = []
            rot_delta = []
            grip = []
            for target in actions[:, start:start + 7]:
                target_pose = target[:6]
                pos_delta.append(float(np.linalg.norm(target_pose[:3] - prev_pose[:3])))
                rot_delta.append(float(np.linalg.norm(target_pose[3:] - prev_pose[3:])))
                grip.append(float(target[6]))
                prev_pose = target_pose
            metrics[side] = {"pos": pos_delta, "rot": rot_delta, "grip": grip}
        return metrics

    def log_iteration(
            self,
            iter_idx: int,
            obs_seq: int,
            raw_action: np.ndarray,
            new_raw_actions: np.ndarray,
            new_action: np.ndarray,
            new_obs: dict,
            controller_records: list[dict],
            action_timestamps: np.ndarray,
            new_timestamps: np.ndarray,
            is_new: np.ndarray,
            n_robots: int):
        record = {
            "time": time.time(),
            "iter_idx": int(iter_idx),
            "obs_seq": int(obs_seq),
            "raw_action_len": int(len(raw_action)),
            "new_action_len": int(len(new_action)),
            "is_new": np.asarray(is_new, dtype=bool).tolist(),
            "action_timestamps": np.asarray(action_timestamps, dtype=np.float64).tolist(),
            "new_timestamps": np.asarray(new_timestamps, dtype=np.float64).tolist(),
            "raw_delta": self._raw_delta_metrics(raw_action, n_robots),
            "new_raw_delta": self._raw_delta_metrics(new_raw_actions, n_robots),
            "new_action_jump": self._new_action_jump_metrics(new_action, new_obs, n_robots),
            "controller_records": controller_records,
        }
        self._file.write(json.dumps(self._jsonable(record), separators=(",", ":")) + "\n")

    @staticmethod
    def _load_jsonl(path: Path) -> list[dict]:
        records = []
        if not path.exists():
            return records
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    @staticmethod
    def _series_from_action_metrics(records: list[dict], section: str, side: str, metric: str):
        x, y = [], []
        sample_idx = 0
        for record in records:
            values = record.get(section, {}).get(side, {}).get(metric, [])
            for value in values:
                x.append(sample_idx)
                y.append(value)
                sample_idx += 1
        return x, y

    @staticmethod
    def _series_from_controller(records: list[dict], side: str, metric: str):
        x, y = [], []
        key = f"{side}_delta_{metric}"
        sample_idx = 0
        for record in records:
            for item in record.get("controller_records", []):
                if key in item:
                    x.append(sample_idx)
                    y.append(item[key])
                    sample_idx += 1
        return x, y

    @staticmethod
    def _series_from_controller_accept(records: list[dict]):
        x, y = [], []
        sample_idx = 0
        for record in records:
            for item in record.get("controller_records", []):
                x.append(sample_idx)
                y.append(1 if item.get("scheduled") else 0)
                sample_idx += 1
        return x, y

    # Plot generation is disabled for deployment.
    # def plot(self):
    #     if not self._file.closed:
    #         self._file.flush()
    #     records = self._load_jsonl(self.jsonl_path)
    #     if not records:
    #         print("[ActionDebugLogger] No action debug records to plot")
    #         return
    #
    #     plots_dir = self.save_dir / "plots"
    #     plots_dir.mkdir(parents=True, exist_ok=True)
    #
    #     plot_specs = [
    #         ("pos", "Action Position Delta", "delta position (m)", "action_delta_pos.png"),
    #         ("rot", "Action Rotation Delta", "delta rotation (rad)", "action_delta_rot.png"),
    #     ]
    #     for metric, title, y_title, filename in plot_specs:
    #         fig = go.Figure()
    #         for side in self.sides:
    #             for section, label_prefix, dash in [
    #                 ("raw_delta", "raw", "dot"),
    #                 ("new_raw_delta", "raw filtered", "dash"),
    #                 ("new_action_jump", "new_action", "solid"),
    #             ]:
    #                 x, y = self._series_from_action_metrics(records, section, side, metric)
    #                 if y:
    #                     fig.add_trace(go.Scatter(
    #                         x=x, y=y, mode="lines+markers",
    #                         name=f"{label_prefix} {side}",
    #                         line=dict(dash=dash),
    #                     ))
    #             x, y = self._series_from_controller(records, side, metric)
    #             if y:
    #                 fig.add_trace(go.Scatter(
    #                     x=x, y=y, mode="lines+markers",
    #                     name=f"controller {side}",
    #                     line=dict(width=3),
    #                 ))
    #         fig.update_layout(title=title, xaxis_title="sample", yaxis_title=y_title)
    #         fig.write_image(str(plots_dir / filename))
    #
    #     fig = go.Figure()
    #     x, y = self._series_from_controller_accept(records)
    #     if y:
    #         fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name="scheduled"))
    #     fig.update_layout(title="Controller Accepted Waypoints", xaxis_title="sample", yaxis_title="1=scheduled, 0=skipped")
    #     fig.write_image(str(plots_dir / "controller_acceptance.png"))
    #     print(f"[ActionDebugLogger] Saved action debug plots to {plots_dir}")

    def close(self):
        if not self._file.closed:
            self._file.flush()
            self._file.close()


def append_debug_info(env, debug_info):
    """Drain all currently queued controller debug samples into debug_info."""
    try:
        debug_info_new = env.get_debug_info()
    except Empty:
        return 0
    except Exception as e:
        print(f"[debug] Failed to collect debug info: {e}")
        return 0

    n_samples = 0
    for key, values in debug_info_new.items():
        if key not in debug_info:
            debug_info[key] = []
        if isinstance(values, np.ndarray):
            values = values.tolist()
        else:
            values = list(values)
        debug_info[key].extend(values)
        n_samples = max(n_samples, len(values))
    return n_samples


def validate_finite_timeout(ctx, param, value):
    del ctx
    if not np.isfinite(value):
        raise click.BadParameter("must be finite", param=param)
    return value


@click.command()
@click.option('--save_obs', '-so', default=False, help='Save observation data for verification (saves every step)')
@click.option('--cam_path', default=list(SERVER_CONFIG.camera.device_paths), type=list, help="-") #the former one is left hand, the later one is right hand

@click.option('--quest_2_ee_left', default=None, help="-") # eye-hand transform matrix
@click.option('--quest_2_ee_right', default=None, help="-") # eye-hand transform matrix
@click.option('--width_slope', default=SERVER_CONFIG.width_slope, type=float, help="-") # transform between gripper width and commanded width
@click.option('--width_offset', default=SERVER_CONFIG.width_offset, type=float, help="-") # transform between gripper width and commanded width
@click.option('--max_gripper_speed', default=SERVER_CONFIG.max_gripper_speed, type=float, help='max gripper command speed in m/s')
@click.option(
    '--max-pos-speed', '--max_pos_speed', 'max_pos_speed', default=SERVER_CONFIG.max_pos_speed,
    type=click.FloatRange(min=0.0, min_open=True), show_default=True,
    callback=validate_finite_timeout,
    help='maximum end-effector linear speed in m/s')
@click.option(
    '--max-rot-speed', '--max_rot_speed', 'max_rot_speed', default=SERVER_CONFIG.max_rot_speed,
    type=click.FloatRange(min=0.0, min_open=True), show_default=True,
    callback=validate_finite_timeout,
    help='maximum end-effector angular speed in rad/s')
@click.option(
    '--max-executed-actions', default=SERVER_CONFIG.max_executed_actions,
    type=click.IntRange(min=1, max=SMOLVLA_ACTION_HORIZON), show_default=True,
    help='maximum fresh bimanual action timesteps scheduled per chunk')
@click.option(
    '--max_action_pos_delta', default=SERVER_CONFIG.max_action_pos_delta,
    type=click.FloatRange(min=0.0, min_open=True), callback=validate_finite_timeout,
    help='max accepted raw model-step position delta in m')
@click.option(
    '--max_action_rot_delta', default=SERVER_CONFIG.max_action_rot_delta,
    type=click.FloatRange(min=0.0, min_open=True), callback=validate_finite_timeout,
    help='max accepted raw model-step rotation delta in rad')

@click.option('--action_pose_repr', default=SERVER_CONFIG.action_pose_repr, help='action pose representation')
@click.option('--exec_mode', default=SERVER_CONFIG.exec_mode, help='action pose representation') # rtc/block

@click.option('--ip', default=SERVER_CONFIG.host, show_default=True, help='robot bridge listen address')
@click.option('--port', default=SERVER_CONFIG.port, help='port')
@click.option('--token-file', default=str(DEFAULT_TOKEN_LIST_PATH), help='path to the allowed token list file')
@click.option('--cycle_timeout_warn_ms', default=SERVER_CONFIG.cycle_timeout_warn_ms)
@click.option(
    '--dry-run', is_flag=True,
    help='exercise a bounded SmolVLA protocol exchange without robot hardware')
@click.option(
    '--dry-run-iterations', default=SERVER_CONFIG.dry_run_iterations,
    type=click.IntRange(min=1), show_default=True)
@click.option(
    '--controller-start-timeout-s', default=SERVER_CONFIG.controller_start_timeout_s,
    type=click.FloatRange(min=0.0, min_open=True), show_default=True,
    callback=validate_finite_timeout,
    help='maximum seconds to wait for the robot Controller to become ready')
@click.option(
    '--start-timeout-s', default=SERVER_CONFIG.start_timeout_s,
    type=click.FloatRange(min=0.0, min_open=True), show_default=True,
    callback=validate_finite_timeout,
    help='maximum seconds to wait for the start signal')
@click.option(
    '--action-timeout-s', default=SERVER_CONFIG.action_timeout_s,
    type=click.FloatRange(min=0.0, min_open=True), show_default=True,
    callback=validate_finite_timeout,
    help='maximum seconds to wait for each action chunk')

def main(
    save_obs,
    cam_path,
    quest_2_ee_left,
    quest_2_ee_right,
    width_slope,
    width_offset,
    max_gripper_speed,
    max_pos_speed,
    max_rot_speed,
    max_action_pos_delta,
    max_action_rot_delta,
    action_pose_repr,
    exec_mode,
    ip,
    port,
    token_file,
    cycle_timeout_warn_ms,
    dry_run,
    dry_run_iterations,
    start_timeout_s,
    action_timeout_s,
    controller_start_timeout_s=SERVER_CONFIG.controller_start_timeout_s,
    max_executed_actions=SERVER_CONFIG.max_executed_actions,
    ):
    max_executed_actions = _validate_max_executed_actions(max_executed_actions)
    (
        max_action_pos_delta,
        max_action_rot_delta,
        _min_gripper,
        _max_gripper,
    ) = validate_safety_limits(
        max_action_pos_delta,
        max_action_rot_delta,
        SMOLVLA_MIN_GRIPPER,
        SMOLVLA_MAX_GRIPPER,
    )
    token_list = load_token_list(token_file)

    client = RobotClient(host=ip, port=port, allowed_tokens=token_list)
    client.start_background()
    start_deadline = time.monotonic() + start_timeout_s

    print("Waiting for policy client connection")
    raw_config = wait_for_smolvla_config(client, start_deadline)
    if raw_config is None:
        return
    try:
        config_dict = validate_smolvla_config(raw_config)
    except Exception:
        _stop_robot_client(client)
        raise
    client.enable_action_ack()

    if dry_run:
        try:
            remaining_start_s = start_deadline - time.monotonic()
            if remaining_start_s <= 0:
                raise ActionTimeout("timed out waiting for dry-run start")
            return run_dry_run(
                client,
                config_dict,
                iterations=dry_run_iterations,
                start_timeout_s=remaining_start_s,
                action_timeout_s=action_timeout_s,
                limits={
                    "max_pos_delta": max_action_pos_delta,
                    "max_rot_delta": max_action_rot_delta,
                    "min_gripper": SMOLVLA_MIN_GRIPPER,
                    "max_gripper": SMOLVLA_MAX_GRIPPER,
                },
            )
        finally:
            _stop_robot_client(client)

    # Calibration and robot imports are intentionally below negotiation and
    # dry-run so protocol checks never initialize hardware.
    if quest_2_ee_left is None:
        quest_2_ee_left = np.load(SERVER_CONFIG.quest_2_ee_left or DEFAULT_LEFT_CALIBRATION_PATH)
    if quest_2_ee_right is None:
        quest_2_ee_right = np.load(SERVER_CONFIG.quest_2_ee_right or DEFAULT_RIGHT_CALIBRATION_PATH)

    policy_type = config_dict["policy_type"]
    data_type = config_dict["data_type"]
    language_prompt = config_dict["language_prompt"]
    control_frequency = SERVER_CONFIG.control_frequency
    controller_frequency = SERVER_CONFIG.controller_frequency
    single_arm_mode = config_dict["single_arm_mode"]
    no_state_obs_mode = config_dict["no_state_obs_mode"]
    steps_per_inference = min(SERVER_CONFIG.steps_per_inference, config_dict["steps_per_inference"])
    action_horizon = config_dict["action_horizon"]
    effective_max_executed_actions = min(
        steps_per_inference,
        max_executed_actions,
    )

    dt = 1/control_frequency
    cycle_timeout_warn_sec = cycle_timeout_warn_ms / 1000.0
    obs_res = SMOLVLA_OBSERVATION_RESOLUTION
    if policy_type == "rdp":
        obs_res = RDP_OBSERVATION_RESOLUTION
    if single_arm_mode:
        cam_path = [cam_path[0]]

    # DEBUG INFO
    if not single_arm_mode:
        sides = ["left", "right"]
    else:
        sides = ["left"]
    paras = ["x", "y", "z", "rx", "ry", "rz", "g"]
    debug_info = dict()
    for side in sides:
        for para in paras:
            debug_info[f"ee_pose_{side}_{para}"] = []
            debug_info[f"target_pose_{side}_{para}"] = []
    debug_info["time"] = []

    print("steps_per_inference:", steps_per_inference)
    print("cycle_timeout_warn_ms:", cycle_timeout_warn_ms)

    from real_world.bimanual_umi_env import BimanualUmiEnv

    # The environment builds transformed shared-memory examples before forking
    # camera workers. Keep that parent-side OpenCV work single-threaded so the
    # children do not inherit an unusable native worker pool.
    cv2.setNumThreads(1)

    with SharedMemoryManager() as shm_manager:
        with BimanualUmiEnv(
                data_type=data_type,
                cam_path=cam_path,
                control_frequency=control_frequency,
                controller_frequency=controller_frequency,
                obs_image_resolution=obs_res,
                obs_float32=SERVER_CONFIG.obs_float32,
                camera_obs_latency=SERVER_CONFIG.effective_camera_obs_latency,
                camera_obs_horizon=SERVER_CONFIG.camera_obs_horizon,
                robot_obs_horizon=SERVER_CONFIG.robot_obs_horizon,
                gripper_obs_horizon=SERVER_CONFIG.gripper_obs_horizon,
                shm_manager=shm_manager,
                quest_2_ee_left=quest_2_ee_left,
                quest_2_ee_right=quest_2_ee_right,
                width_slope=width_slope,
                width_offset=width_offset,
                max_gripper_speed=max_gripper_speed,
                max_pos_speed=max_pos_speed,
                max_rot_speed=max_rot_speed,
                max_action_pos_delta=max_action_pos_delta,
                max_action_rot_delta=max_action_rot_delta,
                camera_config=SERVER_CONFIG.camera,
                controller_launch_timeout_s=controller_start_timeout_s,
                single_arm_mode=single_arm_mode,
                ) as env:
            cv2.setNumThreads(2)

            print("Waiting for camera")
            time.sleep(SERVER_CONFIG.camera_warmup_s)

            print("Warming up policy inference")
            obs = env.get_obs()
            episode_start_pose = list()

            # record initial robot poses
            for robot_id in range(len(cam_path)):
                pose = np.concatenate([
                    obs[f'robot{robot_id}_eef_pos'],
                    obs[f'robot{robot_id}_eef_rot_axis_angle']
                ], axis=-1)[-1]
                episode_start_pose.append(pose)

            # 在开始前只推送一次 warmup obs，避免 client 等待人工输入时
            # 持续堆积大图像帧把 websocket 写阻塞。
            obs = env.get_obs()
            obs_dict = get_real_umi_obs_dict(
                env_obs=obs, shape_meta=None,
                episode_start_pose=episode_start_pose,
                data_type=data_type,
                cam_path=cam_path,
                task=language_prompt,
                no_state_obs_mode=no_state_obs_mode
            )
            client.publish_obs(obs_dict)
            try:
                remaining_start_s = start_deadline - time.monotonic()
                if remaining_start_s <= 0:
                    raise ActionTimeout("timed out waiting for start")
                wait_for_start(client, timeout_s=remaining_start_s)
            except (ActionTimeout, ClientDisconnected, ClientStopRequested) as exc:
                print(f"[main] {exc}")
                _stop_robot_client(client)
                return

            print('################################## Start! ##################################')

            obs_saver = None
            if save_obs:
                obs_save_dir = os.path.join(ROOT_DIR, "eval_obs_data")
                obs_saver = ObsSaver(obs_save_dir, data_type)
                obs_saver.start()
                print(f"[ObsSaver] Observation saving enabled. Directory: {obs_saver.save_dir}")

            action_debug_logger = ActionDebugLogger(save_dir=ROOT_DIR, sides=sides)

            try:
                start_delay = 1.0
                t_start = time.monotonic() + start_delay
                iter_idx = 0
                last_status_log_time = time.monotonic()

                while True:
                    t_cycle_actual_start = time.monotonic()
                    state = client.get_state_update()
                    if state == "stop":
                        break

                    # 预先计算循环结束的时间点，用于后续的精确等待

                    loop_length_set = steps_per_inference * dt

                    # 获取obs
                    obs = env.get_obs()
                    obs_timestamps = obs['timestamp']

                    # !!!!!!!!!!!!!!!!!!!!!
                    # 保存obs

                    time1 = round(time.time(), 2)
                    if obs_saver is not None:
                        obs_saver.save_obs(obs, step_idx=iter_idx)

                    time2 = round(time.time(), 2)
                    obs_dict = get_real_umi_obs_dict(
                        env_obs=obs, shape_meta=None,
                        episode_start_pose=episode_start_pose,
                        data_type=data_type,
                        cam_path=cam_path,
                        task=language_prompt,
                        no_state_obs_mode=no_state_obs_mode
                    )
                    obs_seq = client.publish_obs(obs_dict)

                    time3 = round(time.time(), 2)
                    try:
                        raw_action = wait_for_action_or_stop(
                            client,
                            obs_seq=obs_seq,
                            timeout_s=action_timeout_s,
                        )
                    except (ActionTimeout, ClientDisconnected, ClientStopRequested) as exc:
                        print(f"[main] {exc}")
                        break

                    time4 = round(time.time(), 2)
                    obs_time_last = round(obs_timestamps[-1],2)
                    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                    # 计算动作执行时间戳
                    # 指定推理出来的每个动作该在什么时间点执行

                    result = execute_action_chunk_and_publish_ack(
                        client,
                        obs_seq,
                        raw_action,
                        action_horizon=action_horizon,
                        n_robots=len(cam_path),
                        max_pos_delta=max_action_pos_delta,
                        max_rot_delta=max_action_rot_delta,
                        min_gripper=SMOLVLA_MIN_GRIPPER,
                        max_gripper=SMOLVLA_MAX_GRIPPER,
                        obs_timestamp=obs_timestamps[-1],
                        now=time.time,
                        dt=dt,
                        exec_mode=exec_mode,
                        env=env,
                        converter=get_real_umi_action,
                        action_pose_repr=action_pose_repr,
                        max_executed_actions=effective_max_executed_actions,
                        schedule_from_receive=policy_type == "rdp",
                        command_lead_s=SERVER_CONFIG.rdp_command_lead_s,
                    )
                    raw_action = result.validated
                    action_timestamps = result.action_timestamps
                    new_raw_actions = result.fresh.raw
                    new_action = result.fresh.absolute
                    new_timestamps = result.fresh.timestamps
                    is_new = result.fresh.mask
                    latency = result.latency
                    controller_action_records = result.controller_records

                    new_obs = env.get_obs()
                    action_debug_logger.log_iteration(
                        iter_idx=iter_idx,
                        obs_seq=obs_seq,
                        raw_action=raw_action,
                        new_raw_actions=new_raw_actions,
                        new_action=new_action,
                        new_obs=new_obs,
                        controller_records=controller_action_records,
                        action_timestamps=action_timestamps,
                        new_timestamps=new_timestamps,
                        is_new=is_new,
                        n_robots=len(cam_path),
                    )

                    now = time.monotonic()
                    if now - last_status_log_time >= 2.0:
                        print(
                            f"[main] iter={iter_idx} obs_seq={obs_seq} "
                            f"infer_latency_ms={latency * 1000.0:.1f} "
                            f"accepted_actions={len(new_raw_actions)}/{len(raw_action)}"
                        )
                        last_status_log_time = now

                    # renew debug info without downsampling or truncation
                    append_debug_info(env, debug_info)

                    loop_length_curr = time.monotonic() - t_cycle_actual_start

                    if exec_mode == "block":
                        loop_length_set += latency + 0.01
                        # condifer latency and HACK in loop

                    if loop_length_curr > loop_length_set:
                        print("[main] loop out of time")
                    while loop_length_curr <= loop_length_set:
                        loop_length_curr = time.monotonic() - t_cycle_actual_start
                        time.sleep(0.01)

                    print("[main] actual loop time:", time.monotonic() - t_cycle_actual_start)
                    print()

                    iter_idx += steps_per_inference

            except KeyboardInterrupt:
                print("Interrupted!")

            finally:
                # Robot motion must be stopped and the Controller child joined
                # before client teardown, debug draining, Plotly, or PNG export.
                controller_stop_error = None
                try:
                    env.stop_controller(wait=True)
                except Exception as exc:
                    controller_stop_error = exc

                if controller_stop_error is not None:
                    # The robot child may still own the dispatch gate. Avoid
                    # slow/debug operations that could hide the shutdown
                    # failure, but still release independent resources.
                    try:
                        client.stop()
                    except Exception as exc:
                        print(f"[cleanup] Failed to stop policy client: {exc}")
                    try:
                        client.join(timeout=1.0)
                    except Exception as exc:
                        print(f"[cleanup] Failed to join policy client: {exc}")
                    if obs_saver is not None:
                        try:
                            obs_saver.stop()
                        except Exception as exc:
                            print(f"[cleanup] Failed to stop observation saver: {exc}")
                    try:
                        action_debug_logger.close()
                    except Exception as exc:
                        print(f"[cleanup] Failed to close action debug logger: {exc}")
                    raise RuntimeError(
                        "failed to stop Controller; skipped debug plots and PNG export"
                    ) from controller_stop_error

                client.stop()
                client.join(timeout=1.0)
                # stop obs saver
                if obs_saver is not None:
                    obs_saver.stop()

                final_debug_samples = append_debug_info(env, debug_info)
                if final_debug_samples > 0:
                    print(f"[debug] Collected {final_debug_samples} final debug samples")

                # Deployment plotting is intentionally disabled because Plotly /
                # Kaleido image export can delay shutdown for a long time.
                # action_debug_logger.plot()
                action_debug_logger.close()

                # ee-vs-target PNG export is intentionally disabled.
                # t = debug_info['time']
                # if len(t) > 0:
                #     t_offset = t[0]
                #     t = [ti - t_offset for ti in t]
                # print("Plotting ee vs target")
                # logs_dir = "./ee_action_logs"
                # if not os.path.exists(logs_dir):
                #     os.makedirs(logs_dir)
                # image_export_failed = False
                #
                # for side in sides:
                #     for para in paras:
                #         fig = go.Figure()
                #         key_ee = f'ee_pose_{side}_{para}'
                #         key_target = f'target_pose_{side}_{para}'
                #         fig.add_trace(go.Scatter(x=t, y=debug_info[key_ee], mode='lines', name=key_ee, line=dict(color='blue', width=2)))
                #         fig.add_trace(go.Scatter(x=t, y=debug_info[key_target], mode='lines', name=key_target, line=dict(color='red', width=2)))
                #         fig.update_layout(title='ee vs target', xaxis_title='t', yaxis_title=para)
                #         png_path = os.path.join(logs_dir, f"{side+' '+para}.png")
                #         try:
                #             fig.write_image(png_path)
                #         except Exception as e:
                #             if not image_export_failed:
                #                 image_export_failed = True
                #                 print(
                #                     "[Warning] Failed to export debug plots as PNG. "
                #                     "This usually means Chrome is not available for kaleido. "
                #                     "Install Chrome via `plotly_get_chrome` or system package manager. "
                #                     f"Original error: {e}"
                #                 )

if __name__ == '__main__':
    main()

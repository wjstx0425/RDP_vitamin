from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

from configs.camera_config import CameraConfig
from configs.camera_config import CameraDeviceConfig


def _require_bool(name: str, value: object) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool")


def _require_int(name: str, value: object, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")


def _require_float(name: str, value: object, *, minimum: float | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a number")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and float(value) < minimum:
        raise ValueError(f"{name} must be at least {minimum}")


def _require_resolution(name: str, value: object) -> None:
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError(f"{name} must be a (width, height) tuple")
    for index, item in enumerate(value):
        _require_int(f"{name}[{index}]", item, minimum=1)


@dataclass(frozen=True)
class SmolVLAServerConfig:
    # Maximum SmolVLA action horizon accepted from a deployment client.
    action_horizon: int = 20
    n_robots: int = 2
    action_dim: int = 20
    rotation_6d_eps: float = 1e-6

    # Camera and observation preprocessing.
    camera: CameraConfig = CameraConfig(
        devices=(
            CameraDeviceConfig(name="left_hand", path="/dev/video0"),
            CameraDeviceConfig(name="right_hand", path="/dev/video2"),
        ),
        pixel_format="MJPG",
        width=3840,
        height=800,
        capture_fps=30,
        buffer_size=3,
        auto_exposure=3,
        exposure=170,
        auto_white_balance=0,
        white_balance_temperature=4600,
        brightness=0,
        gain=40,
        gamma=50,
        capture_timestamp_delay=0.101,
    )
    observation_resolution: tuple[int, int] = (256, 256)
    rdp_observation_resolution: tuple[int, int] = (224, 224)
    obs_float32: bool = False
    camera_obs_horizon: int = 1
    robot_obs_horizon: int = 1
    gripper_obs_horizon: int = 1
    camera_obs_latency: float | None = None
    camera_warmup_s: float = 3.0

    # Control loop and controller limits.
    control_frequency: float = 30.0
    controller_frequency: float = 80.0
    max_gripper_speed: float = 0.3
    max_pos_speed: float = 0.7
    max_rot_speed: float = 0.7
    max_action_pos_delta: float = 0.03
    max_action_rot_delta: float = 0.5
    min_gripper: float = -0.05
    max_gripper: float = 1.05
    max_executed_actions: int = 5
    steps_per_inference: int = 5
    action_pose_repr: str = "relative"
    rdp_command_lead_s: float = 0.05
    exec_mode: str = "rtc"

    # Bridge/runtime defaults.
    host: str = "127.0.0.1"
    port: int = 26421
    cycle_timeout_warn_ms: float = 2.0
    controller_start_timeout_s: float = 20.0
    start_timeout_s: float = 300.0
    action_timeout_s: float = 30.0
    dry_run_iterations: int = 2

    # Calibration and gripper conversion.
    quest_2_ee_left: Path | None = None
    quest_2_ee_right: Path | None = None
    width_slope: float = 1.77
    width_offset: float = 0.050

    def __post_init__(self) -> None:
        _require_int("action_horizon", self.action_horizon, minimum=1)
        _require_int("n_robots", self.n_robots, minimum=1)
        _require_int("action_dim", self.action_dim, minimum=1)
        _require_float("rotation_6d_eps", self.rotation_6d_eps, minimum=0.0)
        if not isinstance(self.camera, CameraConfig):
            raise TypeError("camera must be a CameraConfig")
        _require_resolution("observation_resolution", self.observation_resolution)
        _require_resolution("rdp_observation_resolution", self.rdp_observation_resolution)
        _require_bool("obs_float32", self.obs_float32)
        _require_int("camera_obs_horizon", self.camera_obs_horizon, minimum=1)
        _require_int("robot_obs_horizon", self.robot_obs_horizon, minimum=1)
        _require_int("gripper_obs_horizon", self.gripper_obs_horizon, minimum=1)
        if self.camera_obs_latency is not None:
            _require_float("camera_obs_latency", self.camera_obs_latency, minimum=0.0)
        _require_float("camera_warmup_s", self.camera_warmup_s, minimum=0.0)
        _require_float("control_frequency", self.control_frequency, minimum=0.0)
        _require_float("controller_frequency", self.controller_frequency, minimum=0.0)
        _require_float("max_gripper_speed", self.max_gripper_speed, minimum=0.0)
        _require_float("max_pos_speed", self.max_pos_speed, minimum=0.0)
        _require_float("max_rot_speed", self.max_rot_speed, minimum=0.0)
        _require_float("max_action_pos_delta", self.max_action_pos_delta, minimum=0.0)
        _require_float("max_action_rot_delta", self.max_action_rot_delta, minimum=0.0)
        _require_float("min_gripper", self.min_gripper)
        _require_float("max_gripper", self.max_gripper)
        if self.min_gripper > self.max_gripper:
            raise ValueError("min_gripper must be less than or equal to max_gripper")
        _require_int("max_executed_actions", self.max_executed_actions, minimum=1)
        if self.max_executed_actions > self.action_horizon:
            raise ValueError("max_executed_actions must not exceed action_horizon")
        _require_int("steps_per_inference", self.steps_per_inference, minimum=1)
        if self.steps_per_inference > self.action_horizon:
            raise ValueError("steps_per_inference must not exceed action_horizon")
        _require_float("rdp_command_lead_s", self.rdp_command_lead_s, minimum=0.0)
        if self.exec_mode not in {"rtc", "block"}:
            raise ValueError("exec_mode must be 'rtc' or 'block'")
        if not isinstance(self.action_pose_repr, str) or not self.action_pose_repr.strip():
            raise ValueError("action_pose_repr must be a non-empty string")
        if not isinstance(self.host, str) or not self.host.strip():
            raise ValueError("host must be a non-empty string")
        _require_int("port", self.port, minimum=1)
        if self.port > 65535:
            raise ValueError("port must be at most 65535")
        _require_float("cycle_timeout_warn_ms", self.cycle_timeout_warn_ms, minimum=0.0)
        _require_float("controller_start_timeout_s", self.controller_start_timeout_s, minimum=0.0)
        _require_float("start_timeout_s", self.start_timeout_s, minimum=0.0)
        _require_float("action_timeout_s", self.action_timeout_s, minimum=0.0)
        _require_int("dry_run_iterations", self.dry_run_iterations, minimum=1)
        _require_float("width_slope", self.width_slope)
        _require_float("width_offset", self.width_offset)

    @property
    def effective_camera_obs_latency(self) -> float:
        if self.camera_obs_latency is not None:
            return self.camera_obs_latency
        return self.camera.capture_timestamp_delay


SERVER_CONFIG = SmolVLAServerConfig()

"""NumPy-only safety checks for action chunks sent to the robot server."""

import time
from typing import NamedTuple

import numpy as np

from client.robot_client import RobotClientStopRequested


class UnsafeActionError(ValueError):
    """Raised when an action cannot be sent safely."""


class ClientDisconnected(RuntimeError):  # noqa: N818
    """Raised when the policy client disconnects during a wait."""


class ClientStopRequested(RuntimeError):  # noqa: N818
    """Raised when the policy client requests a stop."""


class ActionTimeout(TimeoutError):  # noqa: N818
    """Raised when a client wait reaches its deadline."""


class FreshActions(NamedTuple):
    """Converted actions retained after timestamp filtering."""

    mask: np.ndarray
    raw: np.ndarray
    absolute: np.ndarray
    timestamps: np.ndarray


def _drain_state_updates(client) -> list[object]:
    states = []
    while True:
        state = client.get_state_update()
        if state is None:
            return states
        states.append(state)


def _stop_is_requested(client, states: list[object]) -> bool:
    sticky_check = getattr(client, "is_stop_requested", None)
    return "stop" in states or (sticky_check is not None and sticky_check())


def wait_for_start(client, timeout_s) -> None:
    """Wait for a start state while enforcing stop, connection, and timeout checks."""
    deadline = time.monotonic() + timeout_s

    while True:
        states = _drain_state_updates(client)
        if _stop_is_requested(client, states):
            raise ClientStopRequested("client requested stop while waiting to start")
        if not client.is_connected():
            raise ClientDisconnected("client disconnected while waiting to start")
        if "start" in states:
            return

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ActionTimeout("timed out waiting for start")
        time.sleep(min(0.1, remaining))


def wait_for_stop(client, timeout_s) -> None:
    """Wait for a stop state while enforcing connection and timeout checks."""
    deadline = time.monotonic() + timeout_s

    while True:
        states = _drain_state_updates(client)
        if _stop_is_requested(client, states):
            return
        if not client.is_connected():
            raise ClientDisconnected("client disconnected while waiting to stop")

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ActionTimeout("timed out waiting for stop")
        time.sleep(min(0.1, remaining))


def wait_for_action_or_stop(client, obs_seq, timeout_s) -> np.ndarray:
    """Wait for an observation's action while enforcing watchdog checks."""
    deadline = time.monotonic() + timeout_s

    while True:
        states = _drain_state_updates(client)
        if _stop_is_requested(client, states):
            raise ClientStopRequested("client requested stop while waiting for action")
        if not client.is_connected():
            raise ClientDisconnected("client disconnected while waiting for action")

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ActionTimeout("timed out waiting for action")
        try:
            action = client.wait_for_action(obs_seq=obs_seq, timeout=min(0.1, remaining))
        except RobotClientStopRequested as exc:
            raise ClientStopRequested("client requested stop while waiting for action") from exc
        if action is not None:
            states = _drain_state_updates(client)
            if _stop_is_requested(client, states):
                raise ClientStopRequested("client requested stop while waiting for action")
            origin_check = getattr(client, "action_origin_is_connected", None)
            if origin_check is not None and not origin_check(obs_seq):
                raise ClientDisconnected("action origin disconnected while waiting for action")
            if not client.is_connected():
                raise ClientDisconnected("client disconnected while waiting for action")
            if time.monotonic() >= deadline:
                raise ActionTimeout("timed out waiting for action")
            return np.asarray(action)


def validate_safety_limits(
    max_pos_delta,
    max_rot_delta,
    min_gripper,
    max_gripper,
) -> tuple[float, float, float, float]:
    """Normalize finite, ordered action safety limits."""
    values = {
        "max_pos_delta": float(max_pos_delta),
        "max_rot_delta": float(max_rot_delta),
        "min_gripper": float(min_gripper),
        "max_gripper": float(max_gripper),
    }
    for name, value in values.items():
        if not np.isfinite(value):
            raise UnsafeActionError(f"{name} must be finite")
    if values["max_pos_delta"] <= 0:
        raise UnsafeActionError("max_pos_delta must be positive")
    if values["max_rot_delta"] <= 0:
        raise UnsafeActionError("max_rot_delta must be positive")
    if values["min_gripper"] > values["max_gripper"]:
        raise UnsafeActionError("gripper bounds must be ordered")
    return (
        values["max_pos_delta"],
        values["max_rot_delta"],
        values["min_gripper"],
        values["max_gripper"],
    )


def validate_action_chunk(
    action,
    action_horizon,
    n_robots,
    max_pos_delta,
    max_rot_delta,
    min_gripper,
    max_gripper,
) -> np.ndarray:
    """Validate a delta-action chunk and return it as float32."""
    max_pos_delta, max_rot_delta, min_gripper, max_gripper = validate_safety_limits(
        max_pos_delta,
        max_rot_delta,
        min_gripper,
        max_gripper,
    )
    action_array = np.asarray(action)
    expected_shape = (action_horizon, n_robots * 10)

    if action_array.shape != expected_shape:
        raise UnsafeActionError(
            f"action shape {action_array.shape} does not match expected {expected_shape}"
        )
    if not np.issubdtype(action_array.dtype, np.floating):
        raise UnsafeActionError(f"action dtype {action_array.dtype} is not floating")
    if not np.all(np.isfinite(action_array)):
        raise UnsafeActionError("action contains non-finite values")
    if np.any(np.abs(action_array) > np.finfo(np.float32).max):
        raise UnsafeActionError("action contains values outside the finite float32 range")

    per_robot = action_array.astype(np.float64, copy=False).reshape(action_horizon, n_robots, 10)
    translation_norms = np.linalg.norm(per_robot[..., :3], axis=-1)
    if np.any(translation_norms > max_pos_delta):
        raise UnsafeActionError("action translation delta exceeds the configured limit")

    rotation_6d = per_robot[..., 3:9]
    first = rotation_6d[..., :3]
    second = rotation_6d[..., 3:]
    epsilon = 1e-6

    first_norm = np.linalg.norm(first, axis=-1, keepdims=True)
    if np.any(first_norm <= epsilon):
        raise UnsafeActionError("action contains a near-zero 6D rotation vector")
    first_unit = first / first_norm

    second_norm = np.linalg.norm(second, axis=-1, keepdims=True)
    if np.any(second_norm <= epsilon):
        raise UnsafeActionError("action contains a near-zero 6D rotation vector")
    second_unit = second / second_norm
    second_orthogonal = (
        second_unit - np.sum(first_unit * second_unit, axis=-1, keepdims=True) * first_unit
    )
    orthogonal_norm = np.linalg.norm(second_orthogonal, axis=-1, keepdims=True)
    if np.any(orthogonal_norm <= epsilon):
        raise UnsafeActionError("action contains collinear 6D rotation vectors")
    second_unit = second_orthogonal / orthogonal_norm
    third_unit = np.cross(first_unit, second_unit)

    rotation_matrices = np.stack((first_unit, second_unit, third_unit), axis=-1)
    traces = np.trace(rotation_matrices, axis1=-2, axis2=-1)
    rotation_angles = np.arccos(np.clip((traces - 1.0) / 2.0, -1.0, 1.0))
    if np.any(rotation_angles > max_rot_delta):
        raise UnsafeActionError("action rotation delta exceeds the configured limit")

    grippers = per_robot[..., 9]
    if np.any((grippers < min_gripper) | (grippers > max_gripper)):
        raise UnsafeActionError("action gripper value is outside the configured bounds")

    return action_array.astype(np.float32, copy=False)


def convert_then_filter_fresh(raw_action, action_timestamps, now, convert) -> FreshActions:
    """Convert the complete chunk, then retain only actions newer than ``now``."""
    raw_array = np.asarray(raw_action)
    absolute_array = np.asarray(convert(raw_action))
    timestamp_array = np.asarray(action_timestamps)

    if raw_array.ndim < 1 or absolute_array.ndim < 1 or timestamp_array.ndim != 1:
        raise UnsafeActionError("actions and timestamps must have leading dimensions")
    leading_dimension = timestamp_array.shape[0]
    if raw_array.shape[0] != leading_dimension or absolute_array.shape[0] != leading_dimension:
        raise UnsafeActionError("raw, absolute, and timestamp leading dimensions must match")

    fresh_mask = timestamp_array > now
    return FreshActions(
        mask=fresh_mask,
        raw=raw_array[fresh_mask],
        absolute=absolute_array[fresh_mask],
        timestamps=timestamp_array[fresh_mask],
    )

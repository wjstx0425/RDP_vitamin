"""Bounded, hardware-free protocol exercise for the robot server."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from deploy_scripts.vbvla_safety import ClientDisconnected
from deploy_scripts.vbvla_safety import ClientStopRequested
from deploy_scripts.vbvla_safety import validate_action_chunk
from deploy_scripts.vbvla_safety import wait_for_action_or_stop
from deploy_scripts.vbvla_safety import wait_for_start
from deploy_scripts.vbvla_safety import wait_for_stop


def make_synthetic_observation(
    task: str,
    image_size: int = 256,
    data_type: str = "vision",
) -> dict[str, np.ndarray | str]:
    """Create a deterministic policy observation without reading hardware."""
    if data_type not in {"vision", "vitac"}:
        raise ValueError("data_type must be 'vision' or 'vitac'")
    image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    observation: dict[str, np.ndarray | str] = {
        "observation.images.camera0": image.copy(),
        "observation.images.camera1": image.copy(),
        "observation.state": np.zeros(20, dtype=np.float32),
        "task": task,
    }
    if data_type == "vitac":
        for hand_index in range(2):
            observation[f"observation.images.tactile_left_{hand_index}"] = image.copy()
            observation[f"observation.images.tactile_right_{hand_index}"] = image.copy()
    return observation


def run_dry_run(
    client: Any,
    config: Mapping[str, Any],
    iterations: int,
    action_timeout_s: float,
    limits: Mapping[str, float],
    start_timeout_s: float | None = None,
) -> int:
    """Exercise a bounded observation/action exchange without executing actions."""
    task = str(config["language_prompt"])
    configured_action_horizon = config["action_horizon"]
    if (
        isinstance(configured_action_horizon, bool)
        or not isinstance(configured_action_horizon, int | np.integer)
        or configured_action_horizon <= 0
    ):
        raise ValueError("action_horizon must be a positive integer")
    action_horizon = int(configured_action_horizon)
    n_robots = 1 if bool(config["single_arm_mode"]) else 2
    data_type = str(config.get("data_type", "vision"))

    client.publish_obs(make_synthetic_observation(task, data_type=data_type))
    wait_for_start(
        client,
        timeout_s=action_timeout_s if start_timeout_s is None else start_timeout_s,
    )

    for _ in range(iterations):
        obs_seq = client.publish_obs(make_synthetic_observation(task, data_type=data_type))
        raw_action = wait_for_action_or_stop(client, obs_seq, timeout_s=action_timeout_s)
        validate_action_chunk(
            raw_action,
            action_horizon=action_horizon,
            n_robots=n_robots,
            max_pos_delta=limits["max_pos_delta"],
            max_rot_delta=limits["max_rot_delta"],
            min_gripper=limits["min_gripper"],
            max_gripper=limits["max_gripper"],
        )
        stop_check = getattr(client, "is_stop_requested", None)
        if stop_check is not None and stop_check():
            raise ClientStopRequested("client requested stop before dry-run acknowledgement")
        origin_check = getattr(client, "action_origin_is_connected", None)
        if origin_check is not None and not origin_check(obs_seq):
            raise ClientDisconnected("action origin disconnected before dry-run acknowledgement")
        client.publish_action_ack(obs_seq)

    wait_for_stop(client, timeout_s=action_timeout_s)
    print(
        f"[dry-run] completed {iterations}/{iterations} dry-run exchanges; "
        "no hardware actions executed"
    )
    return 0

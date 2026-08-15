import time

import numpy as np
import pytest

from real_world.bimanual_umi_env import BimanualUmiEnv


class RecordingController:
    def __init__(self) -> None:
        self.waypoints = []

    def check_health(self) -> None:
        return None

    def get_state(self) -> dict:
        return {
            "ee_pose_left": np.zeros(6, dtype=np.float64),
            "ee_pose_right": np.zeros(6, dtype=np.float64),
        }

    def schedule_waypoint(self, **waypoint) -> None:
        self.waypoints.append(waypoint)


def make_runtime_env() -> tuple[BimanualUmiEnv, RecordingController]:
    env = BimanualUmiEnv.__new__(BimanualUmiEnv)
    controller = RecordingController()
    env.controller = controller
    env.single_arm_mode = False
    env.quest_2_ee_left = np.eye(4)
    env.quest_2_ee_right = np.eye(4)
    env.width_offset = 0.05
    env.width_slope = 1.77
    env.max_action_pos_delta = 0.03
    env.max_action_rot_delta = 0.35
    env._last_log_time = {}
    return env, controller


@pytest.mark.parametrize(
    ("left_pose", "right_pose"),
    [
        (
            np.array([0.10, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.zeros(6),
        ),
        (
            np.zeros(6),
            np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        ),
    ],
    ids=("position-jump", "rotation-jump"),
)
def test_decoded_target_jump_is_scheduled_for_speed_limited_controller(
    left_pose: np.ndarray,
    right_pose: np.ndarray,
) -> None:
    env, controller = make_runtime_env()
    actions = np.concatenate(
        [left_pose, np.array([0.03]), right_pose, np.array([0.03])]
    )[None]

    records = env.exec_actions(
        actions=actions,
        timestamps=np.array([time.time() + 10.0]),
    )

    assert len(controller.waypoints) == 1
    assert records[0]["scheduled"] is True
    assert records[0]["skip_reason"] is None

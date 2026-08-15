import numpy as np

from configs.server_config import SERVER_CONFIG
import deploy_scripts.bimanual_smolvla_online as online
from real_world.real_inference_util import get_real_umi_action


class FakeEnv:
    def __init__(self) -> None:
        self.observation = {
            "robot0_eef_pos": np.zeros((1, 3), dtype=np.float64),
            "robot0_eef_rot_axis_angle": np.zeros((1, 3), dtype=np.float64),
            "robot1_eef_pos": np.zeros((1, 3), dtype=np.float64),
            "robot1_eef_rot_axis_angle": np.zeros((1, 3), dtype=np.float64),
        }
        self.exec_calls = []

    def get_obs(self):
        return self.observation

    def exec_actions(self, *, actions, timestamps):
        self.exec_calls.append((actions.copy(), timestamps.copy()))
        return [{"scheduled": True} for _ in actions]


def make_relative_x_chunk(step_x: float = 0.001) -> np.ndarray:
    action = np.zeros(
        (online.SMOLVLA_ACTION_HORIZON, online.SMOLVLA_ACTION_DIM),
        dtype=np.float32,
    )
    for robot_idx in range(online.SMOLVLA_N_ROBOTS):
        start = robot_idx * 10
        action[:, start + 3 : start + 6] = [1.0, 0.0, 0.0]
        action[:, start + 6 : start + 9] = [0.0, 1.0, 0.0]
    action[:, 0] = step_x
    return action


def execute_chunk(
    action: np.ndarray,
    *,
    now: float,
    exec_mode: str = "rtc",
    max_executed_actions: int | None = None,
):
    env = FakeEnv()
    result = online.execute_action_chunk(
        action,
        action_horizon=online.SMOLVLA_ACTION_HORIZON,
        n_robots=online.SMOLVLA_N_ROBOTS,
        max_pos_delta=SERVER_CONFIG.max_action_pos_delta,
        max_rot_delta=SERVER_CONFIG.max_action_rot_delta,
        min_gripper=SERVER_CONFIG.min_gripper,
        max_gripper=SERVER_CONFIG.max_gripper,
        obs_timestamp=100.0,
        now=lambda: now,
        dt=1.0 / SERVER_CONFIG.control_frequency,
        exec_mode=exec_mode,
        env=env,
        converter=get_real_umi_action,
        action_pose_repr="relative",
        max_executed_actions=(
            online.SMOLVLA_ACTION_HORIZON
            if max_executed_actions is None
            else max_executed_actions
        ),
    )
    return result, env


def test_rtc_rebases_fresh_relative_actions_after_stale_prefix() -> None:
    now = 100.39
    result, env = execute_chunk(make_relative_x_chunk(), now=now)

    action_timestamps = (
        100.0
        + np.arange(online.SMOLVLA_ACTION_HORIZON)
        / SERVER_CONFIG.control_frequency
    )
    expected_indices = np.flatnonzero(action_timestamps > now)
    np.testing.assert_array_equal(np.flatnonzero(result.fresh.mask), expected_indices)
    expected_x = np.arange(1, len(expected_indices) + 1) * 0.001
    np.testing.assert_allclose(result.fresh.absolute[:, 0], expected_x)
    assert len(env.exec_calls) == 1
    np.testing.assert_array_equal(env.exec_calls[0][0], result.fresh.absolute)
    np.testing.assert_array_equal(env.exec_calls[0][1], result.fresh.timestamps)


def test_block_mode_preserves_normal_relative_accumulation() -> None:
    result, env = execute_chunk(
        make_relative_x_chunk(),
        now=100.39,
        exec_mode="block",
    )

    np.testing.assert_array_equal(
        result.fresh.mask,
        np.ones(online.SMOLVLA_ACTION_HORIZON, dtype=bool),
    )
    expected_x = np.arange(1, online.SMOLVLA_ACTION_HORIZON + 1) * 0.001
    np.testing.assert_allclose(result.fresh.absolute[:, 0], expected_x)
    assert len(env.exec_calls) == 1


def test_rtc_does_not_submit_when_every_action_is_stale() -> None:
    after_horizon = (
        100.0
        + online.SMOLVLA_ACTION_HORIZON / SERVER_CONFIG.control_frequency
    )
    result, env = execute_chunk(make_relative_x_chunk(), now=after_horizon)

    np.testing.assert_array_equal(
        result.fresh.mask,
        np.zeros(online.SMOLVLA_ACTION_HORIZON, dtype=bool),
    )
    assert result.fresh.raw.shape == (0, online.SMOLVLA_ACTION_DIM)
    assert result.fresh.absolute.shape == (0, online.SMOLVLA_N_ROBOTS * 7)
    assert result.fresh.timestamps.shape == (0,)
    assert env.exec_calls == []


def test_rtc_execution_cap_preserves_original_horizon_indices() -> None:
    now = 100.39
    result, env = execute_chunk(
        make_relative_x_chunk(),
        now=now,
        max_executed_actions=3,
    )

    action_timestamps = (
        100.0
        + np.arange(online.SMOLVLA_ACTION_HORIZON)
        / SERVER_CONFIG.control_frequency
    )
    expected_indices = np.flatnonzero(action_timestamps > now)[:3]
    np.testing.assert_array_equal(
        np.flatnonzero(result.fresh.mask), expected_indices
    )
    np.testing.assert_allclose(result.fresh.absolute[:, 0], [0.001, 0.002, 0.003])
    np.testing.assert_allclose(
        result.fresh.timestamps,
        action_timestamps[expected_indices],
    )
    assert len(env.exec_calls) == 1

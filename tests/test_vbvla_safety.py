from collections import deque
from pathlib import Path
import threading
import time

from click.testing import CliRunner
import numpy as np
import pytest

from client.robot_client import RobotClient
import deploy_scripts.bimanual_smolvla_online as online
import deploy_scripts.vbvla_safety as safety

UnsafeActionError = safety.UnsafeActionError
convert_then_filter_fresh = safety.convert_then_filter_fresh
validate_action_chunk = safety.validate_action_chunk


ACTION_HORIZON = online.SMOLVLA_ACTION_HORIZON
N_ROBOTS = online.SMOLVLA_N_ROBOTS
MAX_POS_DELTA = 0.03
MAX_ROT_DELTA = 0.35
MIN_GRIPPER = -0.05
MAX_GRIPPER = 1.05


def test_default_calibration_paths_are_owned_by_current_repository() -> None:
    repository_root = Path(online.ROOT_DIR).resolve()

    assert online.DEFAULT_LEFT_CALIBRATION_PATH == (
        repository_root / "quest_2_ee_left_hand_fix_quest.npy"
    )
    assert online.DEFAULT_RIGHT_CALIBRATION_PATH == (
        repository_root / "quest_2_ee_right_hand_fix_quest.npy"
    )
    assert online.DEFAULT_LEFT_CALIBRATION_PATH.is_file()
    assert online.DEFAULT_RIGHT_CALIBRATION_PATH.is_file()


def test_smolvla_entry_has_no_old_server_absolute_path() -> None:
    entry = Path(online.__file__).read_text(encoding="utf-8")

    assert "/home/typhon/vb_robot_server" not in entry


class FakeClient:
    def __init__(self, *, connected=True, states=(), action=None, on_action_wait=None) -> None:
        self.connected = connected
        self.states = deque(states)
        self.action = action
        self.on_action_wait = on_action_wait
        self.action_wait_timeouts = []

    def is_connected(self) -> bool:
        return self.connected

    def get_state_update(self):
        if self.states:
            return self.states.popleft()
        return None

    def wait_for_action(self, obs_seq, timeout=None):
        del obs_seq
        self.action_wait_timeouts.append(timeout)
        if self.on_action_wait is not None:
            self.on_action_wait(self, timeout)
        action, self.action = self.action, None
        return action


class TrackingCondition:
    def __init__(self) -> None:
        self.entered = False
        self.was_entered = False

    def __enter__(self):
        self.entered = True
        self.was_entered = True
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.entered = False


def test_robot_client_reports_connection_state_under_condition() -> None:
    condition = TrackingCondition()
    client = object.__new__(RobotClient)
    client._condition = condition  # noqa: SLF001
    client._connected = True  # noqa: SLF001

    assert client.is_connected() is True
    assert condition.was_entered
    assert condition.entered is False


def test_wait_for_start_returns_immediately_for_start_state() -> None:
    client = FakeClient(states=["start"])

    started_at = time.monotonic()
    result = safety.wait_for_start(client, timeout_s=1.0)

    assert result is None
    assert time.monotonic() - started_at < 0.1


def test_wait_for_stop_returns_for_stop_even_if_transport_disconnected() -> None:
    client = FakeClient(connected=False, states=["stop"])

    result = safety.wait_for_stop(client, timeout_s=1.0)

    assert result is None


def test_wait_for_stop_polls_no_longer_than_point_one_seconds(monkeypatch) -> None:
    client = FakeClient(states=[None, "stop"])
    sleeps = []
    monkeypatch.setattr(safety.time, "sleep", sleeps.append)

    safety.wait_for_stop(client, timeout_s=1.0)

    assert sleeps
    assert all(duration <= 0.1 for duration in sleeps)


def test_wait_for_stop_raises_exact_disconnected_exception() -> None:
    client = FakeClient(connected=False)

    with pytest.raises(safety.ClientDisconnected) as exc_info:
        safety.wait_for_stop(client, timeout_s=1.0)

    assert type(exc_info.value) is safety.ClientDisconnected


def test_wait_for_stop_raises_exact_timeout_after_elapsed_deadline() -> None:
    client = FakeClient()

    started_at = time.monotonic()
    with pytest.raises(safety.ActionTimeout) as exc_info:
        safety.wait_for_stop(client, timeout_s=0.02)
    elapsed = time.monotonic() - started_at

    assert type(exc_info.value) is safety.ActionTimeout
    assert elapsed >= 0.015
    assert elapsed < 0.5


def test_wait_for_action_or_stop_returns_immediate_action() -> None:
    action = np.array([[1.0, 2.0]], dtype=np.float32)
    client = FakeClient(action=action)

    result = safety.wait_for_action_or_stop(client, obs_seq=7, timeout_s=1.0)

    np.testing.assert_array_equal(result, action)
    assert all(timeout <= 0.1 for timeout in client.action_wait_timeouts)


def test_wait_for_action_or_stop_rechecks_stop_before_accepting_action() -> None:
    action = np.array([[1.0, 2.0]], dtype=np.float32)
    client = FakeClient(
        action=action,
        on_action_wait=lambda fake, _timeout: fake.states.append("stop"),
    )

    with pytest.raises(safety.ClientStopRequested):
        safety.wait_for_action_or_stop(client, obs_seq=7, timeout_s=1.0)


def test_wait_for_action_drains_queued_states_and_prioritizes_stop() -> None:
    action = np.array([[1.0, 2.0]], dtype=np.float32)
    client = FakeClient(states=["start", "start", "stop"], action=action)

    with pytest.raises(safety.ClientStopRequested):
        safety.wait_for_action_or_stop(client, obs_seq=7, timeout_s=1.0)

    np.testing.assert_array_equal(client.action, action)


def test_wait_for_action_or_stop_rechecks_disconnect_before_accepting_action() -> None:
    action = np.array([[1.0, 2.0]], dtype=np.float32)
    client = FakeClient(
        action=action,
        on_action_wait=lambda fake, _timeout: setattr(fake, "connected", False),
    )

    with pytest.raises(safety.ClientDisconnected):
        safety.wait_for_action_or_stop(client, obs_seq=7, timeout_s=1.0)


def test_wait_for_action_rejects_disconnected_action_origin() -> None:
    class OriginAwareClient(FakeClient):
        def action_origin_is_connected(self, obs_seq):
            assert obs_seq == 7
            return False

    client = OriginAwareClient(
        connected=True,
        action=np.array([[1.0, 2.0]], dtype=np.float32),
    )

    with pytest.raises(safety.ClientDisconnected, match="origin"):
        safety.wait_for_action_or_stop(client, obs_seq=7, timeout_s=1.0)


def test_async_robot_client_stop_is_translated_to_stop_requested() -> None:
    wait_entered = threading.Event()

    class ObservableRobotClient(RobotClient):
        def wait_for_action(self, obs_seq, timeout=None):
            wait_entered.set()
            return super().wait_for_action(obs_seq=obs_seq, timeout=timeout)

    client = ObservableRobotClient()
    client.enable_action_ack()
    with client._condition:  # noqa: SLF001
        client._active_connection_generations.add(0)  # noqa: SLF001
        client._connected = True  # noqa: SLF001

    def request_stop():
        assert wait_entered.wait(timeout=1.0)
        client._handle_message(  # noqa: SLF001
            {"type": "state", "state": "stop"},
            connection_generation=0,
        )

    stop_thread = threading.Thread(target=request_stop, daemon=True)
    stop_thread.start()

    with pytest.raises(safety.ClientStopRequested, match="stop"):
        safety.wait_for_action_or_stop(client, obs_seq=7, timeout_s=1.0)

    stop_thread.join(timeout=1.0)
    assert not stop_thread.is_alive()


def test_wait_for_action_does_not_translate_protocol_runtime_errors() -> None:
    class ProtocolErrorClient(FakeClient):
        def wait_for_action(self, obs_seq, timeout=None):
            del obs_seq, timeout
            raise RuntimeError("action sequence protocol error")

    with pytest.raises(RuntimeError, match="sequence protocol") as exc_info:
        safety.wait_for_action_or_stop(
            ProtocolErrorClient(),
            obs_seq=7,
            timeout_s=1.0,
        )

    assert type(exc_info.value) is RuntimeError


def test_wait_for_action_or_stop_rechecks_deadline_before_accepting_action() -> None:
    action = np.array([[1.0, 2.0]], dtype=np.float32)
    client = FakeClient(
        action=action,
        on_action_wait=lambda _fake, timeout: time.sleep(timeout + 0.01),
    )

    with pytest.raises(safety.ActionTimeout):
        safety.wait_for_action_or_stop(client, obs_seq=7, timeout_s=0.01)


@pytest.mark.parametrize(
    "wait",
    [
        pytest.param(lambda client: safety.wait_for_start(client, 1.0), id="start"),
        pytest.param(
            lambda client: safety.wait_for_action_or_stop(client, 7, 1.0),
            id="action",
        ),
    ],
)
def test_watchdog_wait_raises_exact_stop_exception(wait) -> None:
    client = FakeClient(states=["stop"])

    with pytest.raises(safety.ClientStopRequested) as exc_info:
        wait(client)

    assert type(exc_info.value) is safety.ClientStopRequested


@pytest.mark.parametrize(
    "wait",
    [
        pytest.param(lambda client: safety.wait_for_start(client, 1.0), id="start"),
        pytest.param(
            lambda client: safety.wait_for_action_or_stop(client, 7, 1.0),
            id="action",
        ),
    ],
)
def test_watchdog_wait_raises_exact_disconnected_exception(wait) -> None:
    client = FakeClient(connected=False)

    with pytest.raises(safety.ClientDisconnected) as exc_info:
        wait(client)

    assert type(exc_info.value) is safety.ClientDisconnected


@pytest.mark.parametrize(
    "wait",
    [
        pytest.param(lambda client: safety.wait_for_start(client, 0.02), id="start"),
        pytest.param(
            lambda client: safety.wait_for_action_or_stop(client, 7, 0.02),
            id="action",
        ),
    ],
)
def test_watchdog_wait_raises_exact_timeout_after_elapsed_deadline(wait) -> None:
    client = FakeClient()

    started_at = time.monotonic()
    with pytest.raises(safety.ActionTimeout) as exc_info:
        wait(client)
    elapsed = time.monotonic() - started_at

    assert type(exc_info.value) is safety.ActionTimeout
    assert elapsed >= 0.015
    assert elapsed < 0.5


def make_valid_chunk(action_horizon: int = ACTION_HORIZON) -> np.ndarray:
    action = np.zeros((action_horizon, N_ROBOTS * 10), dtype=np.float32)
    for robot_index in range(N_ROBOTS):
        offset = robot_index * 10
        action[:, offset + 3] = 1.0
        action[:, offset + 7] = 1.0
    return action


def validate(action: np.ndarray) -> np.ndarray:
    return validate_action_chunk(
        action,
        action_horizon=ACTION_HORIZON,
        n_robots=N_ROBOTS,
        max_pos_delta=MAX_POS_DELTA,
        max_rot_delta=MAX_ROT_DELTA,
        min_gripper=MIN_GRIPPER,
        max_gripper=MAX_GRIPPER,
    )


def test_validate_action_chunk_accepts_valid_float32_identity_delta() -> None:
    action = make_valid_chunk()

    validated = validate(action)

    assert validated.dtype == np.float32
    assert np.all(np.isfinite(validated))
    np.testing.assert_array_equal(validated, action)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        pytest.param({"max_pos_delta": np.nan}, "max_pos_delta", id="nan-position"),
        pytest.param({"max_rot_delta": np.inf}, "max_rot_delta", id="inf-rotation"),
        pytest.param({"max_pos_delta": 0.0}, "max_pos_delta", id="zero-position"),
        pytest.param({"min_gripper": 2.0, "max_gripper": 1.0}, "gripper", id="unordered-gripper"),
    ],
)
def test_validate_action_chunk_rejects_invalid_safety_limits(override, message) -> None:
    limits = {
        "max_pos_delta": MAX_POS_DELTA,
        "max_rot_delta": MAX_ROT_DELTA,
        "min_gripper": MIN_GRIPPER,
        "max_gripper": MAX_GRIPPER,
    }

    with pytest.raises(UnsafeActionError, match=message):
        validate_action_chunk(
            make_valid_chunk(),
            action_horizon=ACTION_HORIZON,
            n_robots=N_ROBOTS,
            **(limits | override),
        )


@pytest.mark.parametrize(
    "action",
    [
        pytest.param(np.zeros((1,), dtype=np.float32), id="one-dimensional"),
        pytest.param(
            np.zeros((ACTION_HORIZON - 1, N_ROBOTS * 10), dtype=np.float32),
            id="wrong-horizon",
        ),
        pytest.param(
            np.zeros((ACTION_HORIZON, N_ROBOTS * 10), dtype=np.int64),
            id="integer",
        ),
    ],
)
def test_validate_action_chunk_rejects_invalid_shape_or_dtype(action: np.ndarray) -> None:
    with pytest.raises(UnsafeActionError):
        validate(action)


@pytest.mark.parametrize("nonfinite", [np.nan, np.inf], ids=["nan", "infinity"])
def test_validate_action_chunk_rejects_nonfinite_values(nonfinite: float) -> None:
    action = make_valid_chunk()
    action[0, 0] = nonfinite

    with pytest.raises(UnsafeActionError):
        validate(action)


def test_validate_action_chunk_rejects_values_that_overflow_float32() -> None:
    action = make_valid_chunk().astype(np.float64)
    for robot_index in range(N_ROBOTS):
        offset = robot_index * 10
        action[:, offset + 3 : offset + 9] *= 1e40

    with pytest.raises(UnsafeActionError):
        validate(action)


@pytest.mark.parametrize(
    "first, second",
    [
        pytest.param([1e-8, 0.0, 0.0], [0.0, 1.0, 0.0], id="near-zero"),
        pytest.param([1.0, 0.0, 0.0], [2.0, 0.0, 0.0], id="collinear"),
    ],
)
def test_validate_action_chunk_rejects_degenerate_6d_rotations(first, second) -> None:
    action = make_valid_chunk()
    action[0, 3:6] = first
    action[0, 6:9] = second

    with pytest.raises(UnsafeActionError):
        validate(action)


def test_validate_action_chunk_rejects_scale_imbalanced_collinear_rotation() -> None:
    action = make_valid_chunk()
    action[0, 3:6] = [1.0, 0.0, 0.0]
    action[0, 6:9] = [1e12, 1e-5, 0.0]

    with pytest.raises(UnsafeActionError):
        validate(action)


def test_validate_action_chunk_rejects_translation_norm_over_limit() -> None:
    action = make_valid_chunk()
    action[0, :3] = [MAX_POS_DELTA + 0.001, 0.0, 0.0]

    with pytest.raises(UnsafeActionError):
        validate(action)


def test_validate_action_chunk_rejects_rotation_angle_over_limit() -> None:
    action = make_valid_chunk()
    angle = MAX_ROT_DELTA + 0.01
    action[0, 3:6] = [np.cos(angle), np.sin(angle), 0.0]
    action[0, 6:9] = [-np.sin(angle), np.cos(angle), 0.0]

    with pytest.raises(UnsafeActionError):
        validate(action)


@pytest.mark.parametrize(
    "gripper",
    [
        pytest.param(MIN_GRIPPER - 0.001, id="below-minimum"),
        pytest.param(MAX_GRIPPER + 0.001, id="above-maximum"),
    ],
)
def test_validate_action_chunk_rejects_gripper_outside_bounds(gripper: float) -> None:
    action = make_valid_chunk()
    action[0, 9] = gripper

    with pytest.raises(UnsafeActionError):
        validate(action)


def test_convert_then_filter_fresh_converts_full_chunk_before_filtering() -> None:
    raw = np.array([[0.01], [0.01]], dtype=np.float32)
    timestamps = np.array([1.0, 2.0])

    fresh = convert_then_filter_fresh(
        raw,
        timestamps,
        now=1.5,
        convert=lambda chunk: np.cumsum(chunk, axis=0),
    )

    np.testing.assert_array_equal(fresh.mask, np.array([False, True]))
    np.testing.assert_array_equal(fresh.raw, np.array([[0.01]], dtype=np.float32))
    np.testing.assert_allclose(fresh.absolute, np.array([[0.02]], dtype=np.float32))
    np.testing.assert_array_equal(fresh.timestamps, np.array([2.0]))


def test_convert_then_filter_fresh_returns_shaped_empty_arrays() -> None:
    raw = np.array([[0.01], [0.01]], dtype=np.float32)
    timestamps = np.array([1.0, 2.0])

    fresh = convert_then_filter_fresh(
        raw,
        timestamps,
        now=2.0,
        convert=lambda chunk: np.cumsum(chunk, axis=0),
    )

    np.testing.assert_array_equal(fresh.mask, np.array([False, False]))
    assert fresh.raw.shape == (0, 1)
    assert fresh.absolute.shape == (0, 1)
    assert fresh.timestamps.shape == (0,)


def make_fresh_actions(mask: np.ndarray) -> safety.FreshActions:
    selected_indices = np.flatnonzero(mask)
    raw = np.repeat(selected_indices[:, None], 20, axis=1).astype(np.float32)
    absolute = np.repeat(selected_indices[:, None], 14, axis=1).astype(np.float32)
    timestamps = selected_indices.astype(np.float64) + 100.0
    return safety.FreshActions(
        mask=mask,
        raw=raw,
        absolute=absolute,
        timestamps=timestamps,
    )


def test_limit_fresh_actions_selects_first_chronologically_fresh_action() -> None:
    mask = np.zeros(ACTION_HORIZON, dtype=bool)
    mask[[2, 7, 11]] = True

    limited = online.limit_fresh_actions(make_fresh_actions(mask), 1)

    expected_mask = np.zeros(ACTION_HORIZON, dtype=bool)
    expected_mask[2] = True
    np.testing.assert_array_equal(limited.mask, expected_mask)
    np.testing.assert_array_equal(limited.raw[:, 0], [2.0])
    np.testing.assert_array_equal(limited.absolute[:, 0], [2.0])
    np.testing.assert_array_equal(limited.timestamps, [102.0])


def test_limit_fresh_actions_selects_first_n_without_reordering() -> None:
    mask = np.zeros(ACTION_HORIZON, dtype=bool)
    mask[[1, 4, 9, ACTION_HORIZON - 1]] = True

    limited = online.limit_fresh_actions(make_fresh_actions(mask), 3)

    expected_mask = np.zeros(ACTION_HORIZON, dtype=bool)
    expected_mask[[1, 4, 9]] = True
    np.testing.assert_array_equal(limited.mask, expected_mask)
    np.testing.assert_array_equal(limited.raw[:, 0], [1.0, 4.0, 9.0])
    np.testing.assert_array_equal(limited.absolute[:, 0], [1.0, 4.0, 9.0])
    np.testing.assert_array_equal(limited.timestamps, [101.0, 104.0, 109.0])


def test_limit_fresh_actions_does_not_pad_when_fewer_actions_are_fresh() -> None:
    mask = np.zeros(ACTION_HORIZON, dtype=bool)
    mask[[3, 8]] = True
    fresh = make_fresh_actions(mask)

    limited = online.limit_fresh_actions(fresh, 10)

    np.testing.assert_array_equal(limited.mask, mask)
    np.testing.assert_array_equal(limited.raw, fresh.raw)
    np.testing.assert_array_equal(limited.absolute, fresh.absolute)
    np.testing.assert_array_equal(limited.timestamps, fresh.timestamps)


def test_limit_fresh_actions_preserves_shaped_empty_arrays() -> None:
    mask = np.zeros(ACTION_HORIZON, dtype=bool)
    fresh = make_fresh_actions(mask)

    limited = online.limit_fresh_actions(fresh, 1)

    np.testing.assert_array_equal(limited.mask, mask)
    assert limited.raw.shape == (0, online.SMOLVLA_ACTION_DIM)
    assert limited.absolute.shape == (0, N_ROBOTS * 7)
    assert limited.timestamps.shape == (0,)


@pytest.mark.parametrize(
    "limit", [0, -1, ACTION_HORIZON + 1, 1.5, "1", True]
)
def test_limit_fresh_actions_rejects_invalid_integer_cap(limit) -> None:
    with pytest.raises(ValueError, match="max_executed_actions"):
        online.limit_fresh_actions(
            make_fresh_actions(np.ones(ACTION_HORIZON, dtype=bool)),
            limit,
        )


@pytest.mark.parametrize(
    "timestamps, convert",
    [
        pytest.param(
            np.array([1.0, 2.0]),
            lambda chunk: chunk[:1],
            id="absolute-action",
        ),
        pytest.param(
            np.array([1.0]),
            lambda chunk: chunk,
            id="timestamps",
        ),
    ],
)
def test_convert_then_filter_fresh_rejects_mismatched_leading_dimensions(
    timestamps, convert
) -> None:
    raw = np.array([[0.01], [0.01]], dtype=np.float32)

    with pytest.raises(UnsafeActionError):
        convert_then_filter_fresh(raw, timestamps, now=0.0, convert=convert)


class ExecutionSpy:
    def __init__(self) -> None:
        self.conversion_obs = {"observation": object()}
        self.get_obs_calls = 0
        self.exec_calls = []

    def get_obs(self):
        self.get_obs_calls += 1
        return self.conversion_obs

    def exec_actions(self, *, actions, timestamps):
        self.exec_calls.append((actions.copy(), timestamps.copy()))
        return [{"scheduled": True} for _ in actions]


class ConverterSpy:
    def __init__(self) -> None:
        self.calls = []

    def __call__(self, raw_action, obs, action_pose_repr):
        self.calls.append((raw_action.copy(), obs, action_pose_repr))
        row = np.arange(len(raw_action), dtype=np.float32)[:, None]
        return np.repeat(row, N_ROBOTS * 7, axis=1)


class FixedClock(float):
    def __call__(self):
        return float(self)


def execute_real_loop_chunk(
    raw_action,
    *,
    now,
    max_executed_actions=ACTION_HORIZON,
    exec_mode="rtc",
):
    env = ExecutionSpy()
    converter = ConverterSpy()
    clock = now if callable(now) else FixedClock(now)
    result = online.execute_action_chunk(
        raw_action,
        action_horizon=ACTION_HORIZON,
        n_robots=N_ROBOTS,
        max_pos_delta=MAX_POS_DELTA,
        max_rot_delta=MAX_ROT_DELTA,
        min_gripper=MIN_GRIPPER,
        max_gripper=MAX_GRIPPER,
        obs_timestamp=100.0,
        now=clock,
        dt=0.1,
        exec_mode=exec_mode,
        env=env,
        converter=converter,
        action_pose_repr="relative",
        max_executed_actions=max_executed_actions,
    )
    return result, env, converter


def test_real_loop_executes_only_first_selected_bimanual_timestep() -> None:
    result, env, _converter = execute_real_loop_chunk(
        make_valid_chunk(),
        now=100.0,
        max_executed_actions=1,
        exec_mode="block",
    )

    assert len(env.exec_calls) == 1
    actions, timestamps = env.exec_calls[0]
    assert result.fresh.raw.shape == (1, N_ROBOTS * 10)
    assert actions.shape == (1, N_ROBOTS * 7)
    np.testing.assert_array_equal(actions, result.fresh.absolute)
    np.testing.assert_allclose(timestamps, [100.01])
    expected_mask = np.zeros(ACTION_HORIZON, dtype=bool)
    expected_mask[0] = True
    np.testing.assert_array_equal(result.fresh.mask, expected_mask)


def test_real_loop_default_limit_preserves_all_fresh_actions() -> None:
    env = ExecutionSpy()

    result = online.execute_action_chunk(
        make_valid_chunk(),
        action_horizon=ACTION_HORIZON,
        n_robots=N_ROBOTS,
        max_pos_delta=MAX_POS_DELTA,
        max_rot_delta=MAX_ROT_DELTA,
        min_gripper=MIN_GRIPPER,
        max_gripper=MAX_GRIPPER,
        obs_timestamp=100.0,
        now=FixedClock(100.0),
        dt=0.1,
        exec_mode="block",
        env=env,
        converter=ConverterSpy(),
        action_pose_repr="relative",
    )

    assert result.fresh.raw.shape == (ACTION_HORIZON, N_ROBOTS * 10)
    assert result.fresh.absolute.shape == (ACTION_HORIZON, N_ROBOTS * 7)
    assert result.fresh.timestamps.shape == (ACTION_HORIZON,)
    np.testing.assert_array_equal(
        result.fresh.mask,
        np.ones(ACTION_HORIZON, dtype=bool),
    )
    assert len(env.exec_calls) == 1
    controller_actions, controller_timestamps = env.exec_calls[0]
    assert controller_actions.shape == (ACTION_HORIZON, N_ROBOTS * 7)
    assert controller_timestamps.shape == (ACTION_HORIZON,)
    np.testing.assert_array_equal(controller_actions, result.fresh.absolute)
    np.testing.assert_array_equal(controller_timestamps, result.fresh.timestamps)


def test_real_loop_filters_stale_actions_before_conversion() -> None:
    raw_action = make_valid_chunk()
    cutoff = 100.0 + (ACTION_HORIZON - 2.5) * 0.1

    result, env, converter = execute_real_loop_chunk(raw_action, now=cutoff)

    assert len(converter.calls) == 1
    converted_raw, converted_obs, converted_pose_repr = converter.calls[0]
    np.testing.assert_array_equal(converted_raw, raw_action[-2:])
    assert converted_raw.shape == (2, N_ROBOTS * 10)
    assert converted_obs is env.conversion_obs
    assert converted_pose_repr == "relative"
    np.testing.assert_allclose(result.fresh.absolute[:, 0], [0.0, 1.0])
    expected_timestamps = 100.0 + np.arange(ACTION_HORIZON - 2, ACTION_HORIZON) * 0.1
    np.testing.assert_allclose(result.fresh.timestamps, expected_timestamps)
    assert len(env.exec_calls) == 1
    np.testing.assert_array_equal(env.exec_calls[0][0], result.fresh.absolute)
    np.testing.assert_array_equal(env.exec_calls[0][1], result.fresh.timestamps)


def test_real_loop_does_not_execute_when_every_action_is_stale() -> None:
    cutoff = 100.0 + (ACTION_HORIZON - 1) * 0.1
    result, env, converter = execute_real_loop_chunk(make_valid_chunk(), now=cutoff)

    assert len(converter.calls) == 1
    assert converter.calls[0][0].shape == (0, N_ROBOTS * 10)
    assert result.fresh.absolute.shape == (0, 14)
    assert result.fresh.timestamps.shape == (0,)
    assert env.exec_calls == []


def test_real_loop_rejects_invalid_chunk_before_conversion_or_execution() -> None:
    raw_action = make_valid_chunk()
    raw_action[0, 0] = np.nan
    env = ExecutionSpy()
    converter = ConverterSpy()

    with pytest.raises(UnsafeActionError):
        online.execute_action_chunk(
            raw_action,
            action_horizon=ACTION_HORIZON,
            n_robots=N_ROBOTS,
            max_pos_delta=MAX_POS_DELTA,
            max_rot_delta=MAX_ROT_DELTA,
            min_gripper=MIN_GRIPPER,
            max_gripper=MAX_GRIPPER,
            obs_timestamp=100.0,
            now=100.0,
            dt=0.1,
            exec_mode="rtc",
            env=env,
            converter=converter,
            action_pose_repr="relative",
        )

    assert converter.calls == []
    assert env.get_obs_calls == 0
    assert env.exec_calls == []


@pytest.mark.parametrize("action_horizon", [1, 7, 19, 20])
def test_real_loop_accepts_configured_action_horizon(action_horizon) -> None:
    raw_action = make_valid_chunk(action_horizon)
    env = ExecutionSpy()
    converter = ConverterSpy()

    result = online.execute_action_chunk(
        raw_action,
        action_horizon=action_horizon,
        n_robots=N_ROBOTS,
        max_pos_delta=MAX_POS_DELTA,
        max_rot_delta=MAX_ROT_DELTA,
        min_gripper=MIN_GRIPPER,
        max_gripper=MAX_GRIPPER,
        obs_timestamp=100.0,
        now=FixedClock(100.0),
        dt=0.1,
        exec_mode="block",
        env=env,
        converter=converter,
        action_pose_repr="relative",
        max_executed_actions=1,
    )

    assert result.validated.shape == (action_horizon, N_ROBOTS * 10)
    assert result.action_timestamps.shape == (action_horizon,)


@pytest.mark.parametrize("actual_horizon", [6, 20])
def test_real_loop_rejects_action_shape_that_disagrees_with_session_horizon(
    actual_horizon,
) -> None:
    env = ExecutionSpy()
    converter = ConverterSpy()

    with pytest.raises(UnsafeActionError, match="does not match configured shape"):
        online.execute_action_chunk(
            make_valid_chunk(actual_horizon),
            action_horizon=7,
            n_robots=N_ROBOTS,
            max_pos_delta=MAX_POS_DELTA,
            max_rot_delta=MAX_ROT_DELTA,
            min_gripper=MIN_GRIPPER,
            max_gripper=MAX_GRIPPER,
            obs_timestamp=100.0,
            now=FixedClock(100.0),
            dt=0.1,
            exec_mode="block",
            env=env,
            converter=converter,
            action_pose_repr="relative",
            max_executed_actions=1,
        )

    assert converter.calls == []
    assert env.get_obs_calls == 0
    assert env.exec_calls == []


def test_real_loop_rejects_single_arm_width_before_conversion() -> None:
    action_horizon = ACTION_HORIZON
    n_robots = 1
    shape = (ACTION_HORIZON, 10)
    raw_action = np.zeros((action_horizon, n_robots, 10), dtype=np.float32)
    raw_action[..., 3] = 1.0
    raw_action[..., 7] = 1.0
    raw_action = raw_action.reshape(shape)
    env = ExecutionSpy()
    converter = ConverterSpy()

    with pytest.raises(UnsafeActionError):
        online.execute_action_chunk(
            raw_action,
            action_horizon=action_horizon,
            n_robots=n_robots,
            max_pos_delta=MAX_POS_DELTA,
            max_rot_delta=MAX_ROT_DELTA,
            min_gripper=MIN_GRIPPER,
            max_gripper=MAX_GRIPPER,
            obs_timestamp=100.0,
            now=FixedClock(100.0),
            dt=0.1,
            exec_mode="rtc",
            env=env,
            converter=converter,
            action_pose_repr="relative",
        )

    assert converter.calls == []
    assert env.get_obs_calls == 0
    assert env.exec_calls == []


class SequencedClock(float):
    def __new__(cls, values, events):
        instance = super().__new__(cls, values[0])
        instance.values = deque(values)
        instance.events = events
        return instance

    def __call__(self):
        self.events.append("clock")
        return self.values.popleft()


def test_real_loop_samples_stale_cutoff_before_conversion() -> None:
    events = []
    stale_cutoff = 100.0 + (ACTION_HORIZON - 1.5) * 0.1
    clock = SequencedClock([100.0, stale_cutoff], events)
    env = ExecutionSpy()

    def converter(raw_action, obs, action_pose_repr):
        del obs, action_pose_repr
        events.append("convert")
        row = np.arange(len(raw_action), dtype=np.float32)[:, None]
        return np.repeat(row, N_ROBOTS * 7, axis=1)

    result = online.execute_action_chunk(
        make_valid_chunk(),
        action_horizon=ACTION_HORIZON,
        n_robots=N_ROBOTS,
        max_pos_delta=MAX_POS_DELTA,
        max_rot_delta=MAX_ROT_DELTA,
        min_gripper=MIN_GRIPPER,
        max_gripper=MAX_GRIPPER,
        obs_timestamp=100.0,
        now=clock,
        dt=0.1,
        exec_mode="rtc",
        env=env,
        converter=converter,
        action_pose_repr="relative",
    )

    assert events == ["clock", "clock", "convert"]
    np.testing.assert_allclose(result.fresh.absolute[:, 0], [0.0])
    np.testing.assert_allclose(
        result.fresh.timestamps,
        [100.0 + (ACTION_HORIZON - 1) * 0.1],
    )


@pytest.mark.parametrize(
    ("obs_timestamp", "clock", "dt"),
    [
        pytest.param(np.inf, FixedClock(100.0), 0.1, id="infinite-observation-time"),
        pytest.param(100.0, FixedClock(np.inf), 0.1, id="infinite-clock"),
        pytest.param(100.0, FixedClock(100.0), 0.0, id="zero-dt"),
    ],
)
def test_active_path_rejects_invalid_times_without_execution_or_ack(
    obs_timestamp, clock, dt
) -> None:
    env = ExecutionSpy()
    converter = ConverterSpy()
    acks = []

    class AckClient:
        def action_origin_is_connected(self, obs_seq):
            return obs_seq == 11

        def publish_action_ack(self, obs_seq):
            acks.append(obs_seq)

    with pytest.raises(UnsafeActionError, match="time|timestamp|dt"):
        online.execute_action_chunk_and_publish_ack(
            AckClient(),
            11,
            make_valid_chunk(),
            action_horizon=ACTION_HORIZON,
            n_robots=N_ROBOTS,
            max_pos_delta=MAX_POS_DELTA,
            max_rot_delta=MAX_ROT_DELTA,
            min_gripper=MIN_GRIPPER,
            max_gripper=MAX_GRIPPER,
            obs_timestamp=obs_timestamp,
            now=clock,
            dt=dt,
            exec_mode="rtc",
            env=env,
            converter=converter,
            action_pose_repr="relative",
        )

    assert env.exec_calls == []
    assert acks == []


def test_active_path_rechecks_action_origin_before_execution() -> None:
    env = ExecutionSpy()
    converter = ConverterSpy()
    acks = []

    class DisconnectedOriginClient:
        def action_origin_is_connected(self, obs_seq):
            assert obs_seq == 11
            return False

        def publish_action_ack(self, obs_seq):
            acks.append(obs_seq)

    with pytest.raises(safety.ClientDisconnected, match="origin"):
        online.execute_action_chunk_and_publish_ack(
            DisconnectedOriginClient(),
            11,
            make_valid_chunk(),
            action_horizon=ACTION_HORIZON,
            n_robots=N_ROBOTS,
            max_pos_delta=MAX_POS_DELTA,
            max_rot_delta=MAX_ROT_DELTA,
            min_gripper=MIN_GRIPPER,
            max_gripper=MAX_GRIPPER,
            obs_timestamp=100.0,
            now=FixedClock(100.0),
            dt=0.1,
            exec_mode="rtc",
            env=env,
            converter=converter,
            action_pose_repr="relative",
        )

    assert env.exec_calls == []
    assert acks == []


def test_active_path_ack_follows_limited_controller_call_return() -> None:
    events = []

    class OrderedEnv(ExecutionSpy):
        def exec_actions(self, *, actions, timestamps):
            assert actions.shape == (1, N_ROBOTS * 7)
            assert timestamps.shape == (1,)
            records = super().exec_actions(actions=actions, timestamps=timestamps)
            events.append("controller-return")
            return records

    class OrderedClient:
        def is_stop_requested(self):
            return False

        def action_origin_is_connected(self, obs_seq):
            assert obs_seq == 11
            return True

        def publish_action_ack(self, obs_seq):
            assert obs_seq == 11
            events.append("ack")

    result = online.execute_action_chunk_and_publish_ack(
        OrderedClient(),
        11,
        make_valid_chunk(),
        action_horizon=ACTION_HORIZON,
        n_robots=N_ROBOTS,
        max_pos_delta=MAX_POS_DELTA,
        max_rot_delta=MAX_ROT_DELTA,
        min_gripper=MIN_GRIPPER,
        max_gripper=MAX_GRIPPER,
        obs_timestamp=100.0,
        now=FixedClock(100.0),
        dt=0.1,
        exec_mode="rtc",
        env=OrderedEnv(),
        converter=ConverterSpy(),
        action_pose_repr="relative",
        max_executed_actions=1,
    )

    assert len(result.fresh.absolute) == 1
    assert events == ["controller-return", "ack"]

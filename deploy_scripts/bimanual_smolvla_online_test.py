import sys
import os
import subprocess
import types
from pathlib import Path

import click
import numpy as np
import pytest
from click.testing import CliRunner

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from deploy_scripts import bimanual_smolvla_online as smolvla


def valid_config(**updates):
    config = {
        "data_type": "vision",
        "language_prompt": "fold the towel",
        "control_frequency": 10,
        "controller_frequency": 100,
        "single_arm_mode": False,
        "no_state_obs_mode": False,
        "steps_per_inference": smolvla.SERVER_CONFIG.steps_per_inference,
        "action_horizon": smolvla.SMOLVLA_ACTION_HORIZON,
    }
    return config | updates


def valid_action_chunk(delta_x=0.0):
    action = np.zeros(
        (smolvla.SMOLVLA_ACTION_HORIZON, smolvla.SMOLVLA_ACTION_DIM),
        dtype=np.float32,
    )
    rotation_6d = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
    for robot_idx in range(2):
        start = robot_idx * 10
        action[:, start] = delta_x
        action[:, start + 3 : start + 9] = rotation_6d
    return action


def zero_robot_obs():
    return {
        "robot0_eef_pos": np.zeros((1, 3), dtype=np.float32),
        "robot0_eef_rot_axis_angle": np.zeros((1, 3), dtype=np.float32),
        "robot1_eef_pos": np.zeros((1, 3), dtype=np.float32),
        "robot1_eef_rot_axis_angle": np.zeros((1, 3), dtype=np.float32),
    }


class FakeStartupClient:
    def __init__(self, connection_result=True, config=None):
        self.connection_result = connection_result
        self.config = valid_config() if config is None else config
        self.connection_timeouts = []
        self.config_timeouts = []
        self.started = False
        self.ack_enabled = False
        self.stopped = False
        self.join_timeouts = []

    def start_background(self):
        self.started = True

    def enable_action_ack(self):
        self.ack_enabled = True

    def wait_for_connection(self, timeout):
        self.connection_timeouts.append(timeout)
        return self.connection_result

    def wait_for_config(self, timeout):
        self.config_timeouts.append(timeout)
        return self.config

    def stop(self):
        self.stopped = True

    def join(self, timeout):
        self.join_timeouts.append(timeout)


def call_main(**updates):
    kwargs = {
        "save_obs": False,
        "cam_path": ["/dev/video0", "/dev/video2"],
        "quest_2_ee_left": np.eye(4),
        "quest_2_ee_right": np.eye(4),
        "width_slope": 1.77,
        "width_offset": 0.05,
        "max_gripper_speed": 0.05,
        "max_pos_speed": smolvla.SERVER_CONFIG.max_pos_speed,
        "max_rot_speed": smolvla.SERVER_CONFIG.max_rot_speed,
        "max_action_pos_delta": 0.03,
        "max_action_rot_delta": 0.35,
        "action_pose_repr": "relative",
        "exec_mode": "rtc",
        "ip": "127.0.0.1",
        "port": 26421,
        "token_file": "unused",
        "cycle_timeout_warn_ms": 2,
        "dry_run": False,
        "dry_run_iterations": 2,
        "start_timeout_s": 30.0,
        "action_timeout_s": 30.0,
    }
    return smolvla.main.callback(**(kwargs | updates))


def test_cli_does_not_expose_removed_control_options():
    result = CliRunner().invoke(smolvla.main, ["--help"])

    assert result.exit_code == 0
    assert "--vel_max" not in result.output
    assert "--trajectory-smoothing" not in result.output
    assert "--smooth-" not in result.output


def test_clean_subprocess_import_and_invalid_cli_keep_hardware_lazy():
    code = """
import sys
from click.testing import CliRunner
import deploy_scripts.bimanual_smolvla_online as server

hardware_modules = (
    'jax',
    'real_world.bimanual_umi_env',
    'real_world.robot_api.arm.Controller',
)
for module_name in hardware_modules:
    assert module_name not in sys.modules

for value in ('0', '-1', str(server.SMOLVLA_ACTION_HORIZON + 1), '1.5', 'one'):
    result = CliRunner().invoke(
        server.main,
        ['--max-executed-actions', value],
    )
    assert result.exit_code == 2, result.output
    for module_name in hardware_modules:
        assert module_name not in sys.modules
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT_DIR)

    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("update", "match"),
    [
        ({"data_type": "depth"}, "data_type"),
        ({"single_arm_mode": True}, "single_arm_mode"),
        ({"no_state_obs_mode": True}, "no_state_obs_mode"),
        ({"action_horizon": 0}, "action_horizon"),
        ({"action_horizon": smolvla.SMOLVLA_ACTION_HORIZON + 1}, "action_horizon"),
        ({"control_frequency": 0}, "control_frequency"),
        ({"steps_per_inference": smolvla.SMOLVLA_ACTION_HORIZON + 1}, "steps_per_inference"),
    ],
)
def test_rejects_non_smolvla_config(update, match):
    config = valid_config() | update
    with pytest.raises(ValueError, match=match):
        smolvla.validate_smolvla_config(config)


@pytest.mark.parametrize("action_horizon", [1, 7, 19, 20])
def test_accepts_action_horizon_up_to_checkpoint_chunk_size(action_horizon):
    config = valid_config(action_horizon=action_horizon, steps_per_inference=1)

    validated = smolvla.validate_smolvla_config(config)

    assert validated["action_horizon"] == action_horizon


@pytest.mark.parametrize("data_type", ["vision", "vitac"])
def test_accepts_supported_observation_modes(data_type):
    validated = smolvla.validate_smolvla_config(valid_config(data_type=data_type))

    assert validated["data_type"] == data_type


@pytest.mark.parametrize("field", ["control_frequency", "controller_frequency"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, 0])
def test_rejects_nonfinite_or_nonpositive_frequency(field, value):
    with pytest.raises(ValueError, match=field):
        smolvla.validate_smolvla_config(valid_config(**{field: value}))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("action_horizon", smolvla.SMOLVLA_ACTION_HORIZON + 0.9),
        ("action_horizon", True),
        ("steps_per_inference", 1.9),
        ("steps_per_inference", True),
    ],
)
def test_rejects_noninteger_or_boolean_count(field, value):
    with pytest.raises(ValueError, match=rf"{field}.*integer"):
        smolvla.validate_smolvla_config(valid_config(**{field: value}))


def test_rejects_invalid_action_chunks():
    with pytest.raises(ValueError, match="shape"):
        smolvla.validate_smolvla_action_chunk(
            np.zeros(
                (smolvla.SMOLVLA_ACTION_HORIZON - 1, smolvla.SMOLVLA_ACTION_DIM),
                dtype=np.float32,
            )
        )
    invalid = valid_action_chunk()
    invalid[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        smolvla.validate_smolvla_action_chunk(invalid)
    invalid = valid_action_chunk()
    invalid[0, 3:9] = 0
    with pytest.raises(ValueError, match="rotation-6D"):
        smolvla.validate_smolvla_action_chunk(invalid)


def test_filters_stale_prefix_before_decoding_relative_actions():
    action = valid_action_chunk(delta_x=0.001)
    timestamps = np.arange(smolvla.SMOLVLA_ACTION_HORIZON, dtype=np.float64)
    _, new_raw, new_absolute, new_timestamps, is_new = smolvla.prepare_smolvla_actions(
        action,
        zero_robot_obs(),
        timestamps,
        curr_time=1.5,
        action_pose_repr="relative",
    )
    np.testing.assert_allclose(new_absolute[0, 0], 0.001, atol=1e-6)
    assert new_raw.shape == (
        smolvla.SMOLVLA_ACTION_HORIZON - 2,
        smolvla.SMOLVLA_ACTION_DIM,
    )
    assert new_timestamps[0] == 2.0
    assert is_new.sum() == smolvla.SMOLVLA_ACTION_HORIZON - 2


def test_returns_empty_batch_when_every_action_is_stale():
    result = smolvla.prepare_smolvla_actions(
        valid_action_chunk(),
        zero_robot_obs(),
        np.arange(smolvla.SMOLVLA_ACTION_HORIZON),
        curr_time=float(smolvla.SMOLVLA_ACTION_HORIZON),
        action_pose_repr="relative",
    )
    assert result[1].shape == (0, smolvla.SMOLVLA_ACTION_DIM)
    assert result[2].shape == (0, smolvla.SMOLVLA_N_ROBOTS * 7)
    assert result[3].shape == (0,)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_rejects_nonfinite_action_timestamps(value):
    timestamps = np.arange(smolvla.SMOLVLA_ACTION_HORIZON, dtype=np.float64)
    timestamps[10] = value

    with pytest.raises(ValueError, match="timestamps.*finite"):
        smolvla.prepare_smolvla_actions(
            valid_action_chunk(),
            zero_robot_obs(),
            timestamps,
            curr_time=1.0,
            action_pose_repr="relative",
        )


@pytest.mark.parametrize("curr_time", [np.nan, np.inf, -np.inf])
def test_rejects_nonfinite_current_time(curr_time):
    with pytest.raises(ValueError, match="curr_time.*finite"):
        smolvla.prepare_smolvla_actions(
            valid_action_chunk(),
            zero_robot_obs(),
            np.arange(smolvla.SMOLVLA_ACTION_HORIZON, dtype=np.float64),
            curr_time=curr_time,
            action_pose_repr="relative",
        )


@pytest.mark.parametrize("replacement", [9.0, 8.0])
def test_rejects_nonincreasing_action_timestamps(replacement):
    timestamps = np.arange(smolvla.SMOLVLA_ACTION_HORIZON, dtype=np.float64)
    timestamps[10] = replacement

    with pytest.raises(ValueError, match="strictly increasing"):
        smolvla.prepare_smolvla_actions(
            valid_action_chunk(),
            zero_robot_obs(),
            timestamps,
            curr_time=1.0,
            action_pose_repr="relative",
        )


@pytest.mark.parametrize("option", ["--start-timeout-s", "--action-timeout-s"])
@pytest.mark.parametrize("value", ["nan", "inf"])
def test_rejects_nonfinite_timeouts(option, value, tmp_path):
    result = CliRunner().invoke(
        smolvla.main,
        [option, value, "--token-file", str(tmp_path / "missing-token-list")],
    )
    assert result.exit_code == 2
    assert "finite" in result.output


@pytest.mark.parametrize("option", ["--max_action_pos_delta", "--max_action_rot_delta"])
@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "0"])
def test_rejects_invalid_cli_safety_limits(option, value, tmp_path):
    result = CliRunner().invoke(
        smolvla.main,
        [option, value, "--token-file", str(tmp_path / "missing-token-list")],
    )
    assert result.exit_code == 2
    assert "finite" in result.output or "range x>0.0" in result.output


def test_max_executed_actions_cli_option_has_fixed_horizon_range_and_default():
    option = next(
        parameter
        for parameter in smolvla.main.params
        if parameter.name == "max_executed_actions"
    )

    assert option.default == smolvla.SERVER_CONFIG.max_executed_actions
    assert isinstance(option.type, click.IntRange)
    assert option.type.min == 1
    assert option.type.max == smolvla.SMOLVLA_ACTION_HORIZON


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("max_pos_speed", smolvla.SERVER_CONFIG.max_pos_speed),
        ("max_rot_speed", smolvla.SERVER_CONFIG.max_rot_speed),
    ],
)
def test_cartesian_speed_cli_options_are_positive_with_current_defaults(
    name, expected
):
    option = next(parameter for parameter in smolvla.main.params if parameter.name == name)

    assert option.default == expected
    assert isinstance(option.type, click.FloatRange)
    assert option.type.min == 0.0
    assert option.type.min_open


@pytest.mark.parametrize("option", ["--max-pos-speed", "--max-rot-speed"])
@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_cli_rejects_invalid_cartesian_speed_before_token_or_client(
    monkeypatch, option, value
):
    calls = []

    def forbidden(label):
        def fail(*_args, **_kwargs):
            calls.append(label)
            raise AssertionError(f"invalid Cartesian speed reached {label}")

        return fail

    monkeypatch.setattr(smolvla, "load_token_list", forbidden("token loading"))
    monkeypatch.setattr(smolvla, "RobotClient", forbidden("RobotClient"))

    result = CliRunner().invoke(smolvla.main, [option, value])

    assert result.exit_code == 2
    assert "Invalid value" in result.output
    assert calls == []


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_cli_rejects_invalid_controller_start_timeout_before_token_or_client(
    monkeypatch, value
):
    calls = []

    def forbidden(label):
        def fail(*_args, **_kwargs):
            calls.append(label)
            raise AssertionError(f"invalid startup timeout reached {label}")

        return fail

    monkeypatch.setattr(smolvla, "load_token_list", forbidden("token loading"))
    monkeypatch.setattr(smolvla, "RobotClient", forbidden("RobotClient"))

    result = CliRunner().invoke(
        smolvla.main,
        ["--controller-start-timeout-s", value],
    )

    assert result.exit_code == 2
    assert "Invalid value" in result.output
    assert "No such option" not in result.output
    assert calls == []


@pytest.mark.parametrize(
    "value",
    ["0", "-1", str(smolvla.SMOLVLA_ACTION_HORIZON + 1), "1.5", "one"],
)
def test_cli_rejects_invalid_max_executed_actions_before_client_construction(
    monkeypatch, value
):
    client_constructions = []

    def forbidden_client(**_kwargs):
        client_constructions.append(True)
        raise AssertionError("invalid action limit constructed RobotClient")

    monkeypatch.setattr(smolvla, "RobotClient", forbidden_client)

    result = CliRunner().invoke(
        smolvla.main,
        ["--max-executed-actions", value],
    )

    assert result.exit_code == 2
    assert "Invalid value for '--max-executed-actions'" in result.output
    assert client_constructions == []


def test_startup_wait_uses_one_deadline_for_connection_and_config():
    client = FakeStartupClient()
    times = iter([10.0, 12.5])

    config = smolvla.wait_for_smolvla_config(
        client,
        deadline=20.0,
        monotonic=lambda: next(times),
    )

    assert config == valid_config()
    assert client.connection_timeouts == [10.0]
    assert client.config_timeouts == [7.5]
    assert not client.stopped


def test_startup_wait_cleans_up_after_connection_timeout():
    client = FakeStartupClient(connection_result=False)

    result = smolvla.wait_for_smolvla_config(
        client,
        deadline=20.0,
        monotonic=lambda: 10.0,
    )

    assert result is None
    assert client.connection_timeouts == [10.0]
    assert client.config_timeouts == []
    assert client.stopped
    assert client.join_timeouts == [1.0]


def test_startup_wait_cleans_up_after_config_timeout():
    client = FakeStartupClient(config=None)
    client.config = None
    times = iter([10.0, 12.0])

    result = smolvla.wait_for_smolvla_config(
        client,
        deadline=20.0,
        monotonic=lambda: next(times),
    )

    assert result is None
    assert client.connection_timeouts == [10.0]
    assert client.config_timeouts == [8.0]
    assert client.stopped
    assert client.join_timeouts == [1.0]


def test_main_starts_one_deadline_and_passes_it_to_startup_helper(monkeypatch):
    client = FakeStartupClient()
    captured = {}

    monkeypatch.setattr(smolvla, "load_token_list", lambda _: ["token"])
    monkeypatch.setattr(smolvla, "RobotClient", lambda **_: client)

    def monotonic():
        assert client.started
        return 100.0

    monkeypatch.setattr(smolvla.time, "monotonic", monotonic)

    def wait_for_config(received_client, deadline):
        captured["client"] = received_client
        captured["deadline"] = deadline
        return None

    monkeypatch.setattr(
        smolvla,
        "wait_for_smolvla_config",
        wait_for_config,
        raising=False,
    )

    assert call_main(start_timeout_s=25.0) is None
    assert captured == {"client": client, "deadline": 125.0}


def test_main_cleans_up_and_reraises_invalid_config(monkeypatch):
    client = FakeStartupClient()
    invalid_config = valid_config(data_type="depth")

    monkeypatch.setattr(smolvla, "load_token_list", lambda _: ["token"])
    monkeypatch.setattr(smolvla, "RobotClient", lambda **_: client)
    monkeypatch.setattr(smolvla.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(
        smolvla,
        "wait_for_smolvla_config",
        lambda _client, _deadline: invalid_config,
        raising=False,
    )

    with pytest.raises(ValueError, match="data_type"):
        call_main()

    assert client.stopped
    assert client.join_timeouts == [1.0]


def test_runtime_safety_limits_are_rejected_before_client_start(monkeypatch):
    client_started = []

    monkeypatch.setattr(smolvla, "load_token_list", lambda _: ["token"])

    def forbidden_client(**_kwargs):
        client_started.append(True)
        raise AssertionError("invalid safety limits started the client")

    monkeypatch.setattr(smolvla, "RobotClient", forbidden_client)

    with pytest.raises(ValueError, match="max_pos_delta"):
        call_main(max_action_pos_delta=np.nan)

    assert client_started == []


@pytest.mark.parametrize("controller_start_timeout_s", [None, 27.5])
def test_main_forwards_options_to_lazy_environment(
    monkeypatch, controller_start_timeout_s
):
    client = FakeStartupClient()
    captured = {}

    class StopBeforeHardware(RuntimeError):
        pass

    class FakeSharedMemoryManager:
        def __enter__(self):
            return object()

        def __exit__(self, *_args):
            return False

    class CaptureBimanualUmiEnv:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            raise StopBeforeHardware

    fake_env_module = types.ModuleType("real_world.bimanual_umi_env")
    fake_env_module.BimanualUmiEnv = CaptureBimanualUmiEnv

    monkeypatch.setattr(smolvla, "load_token_list", lambda _: ["token"])
    monkeypatch.setattr(smolvla, "RobotClient", lambda **_: client)
    monkeypatch.setattr(smolvla.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(
        smolvla,
        "wait_for_smolvla_config",
        lambda _client, _deadline: client.config,
    )
    monkeypatch.setattr(smolvla, "SharedMemoryManager", FakeSharedMemoryManager)
    monkeypatch.setitem(sys.modules, "real_world.bimanual_umi_env", fake_env_module)

    updates = {
        "max_pos_speed": 0.31,
        "max_rot_speed": 0.22,
    }
    if controller_start_timeout_s is not None:
        updates["controller_start_timeout_s"] = controller_start_timeout_s

    with pytest.raises(StopBeforeHardware):
        call_main(**updates)

    assert "trajectory_smoothing" not in captured
    assert not any(name.startswith("smooth_") for name in captured)
    expected_timeout = (
        20.0
        if controller_start_timeout_s is None
        else controller_start_timeout_s
    )
    assert captured["controller_launch_timeout_s"] == pytest.approx(
        expected_timeout
    )
    assert captured["max_pos_speed"] == pytest.approx(0.31)
    assert captured["max_rot_speed"] == pytest.approx(0.22)


@pytest.mark.parametrize(
    "termination",
    [
        "stop",
        "action_timeout",
        "disconnect",
        "keyboard_interrupt",
        "controller_stop_failure",
    ],
)
def test_runtime_termination_joins_controller_before_slow_cleanup(
    monkeypatch, termination
):
    events = []

    class RuntimeClient(FakeStartupClient):
        def publish_obs(self, _obs):
            events.append("publish_obs")
            return 17

        def get_state_update(self):
            events.append("state_update")
            if termination == "keyboard_interrupt":
                raise KeyboardInterrupt
            return (
                "stop"
                if termination in ("stop", "controller_stop_failure")
                else None
            )

        def stop(self):
            super().stop()
            events.append("client_stop")

        def join(self, timeout):
            super().join(timeout)
            events.append("client_join")

    class FakeSharedMemoryManager:
        def __enter__(self):
            return object()

        def __exit__(self, *_args):
            return False

    class FakeRuntimeEnv:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            events.append("env_enter")
            return self

        def __exit__(self, *_args):
            events.append("env_exit")
            return False

        def stop_controller(self, *, wait):
            events.append(("controller_stop_join", wait))
            if termination == "controller_stop_failure":
                raise TimeoutError("controller dispatch gate did not drain")

        def get_obs(self):
            return {
                "timestamp": np.array([100.0]),
                "robot0_eef_pos": np.zeros((1, 3)),
                "robot0_eef_rot_axis_angle": np.zeros((1, 3)),
                "robot1_eef_pos": np.zeros((1, 3)),
                "robot1_eef_rot_axis_angle": np.zeros((1, 3)),
            }

        def get_debug_info(self):
            return {}

    class FakeActionDebugLogger:
        def __init__(self, **_kwargs):
            events.append("logger_init")

        def plot(self):
            events.append("slow_plot")

        def close(self):
            events.append("logger_close")

    class FakeFigure:
        def add_trace(self, _trace):
            pass

        def update_layout(self, **_kwargs):
            pass

        def write_image(self, _path):
            events.append("slow_png")

    client = RuntimeClient()
    fake_env_module = types.ModuleType("real_world.bimanual_umi_env")
    fake_env_module.BimanualUmiEnv = FakeRuntimeEnv

    def wait_for_action(*_args, **_kwargs):
        if termination == "action_timeout":
            raise smolvla.ActionTimeout("timed out")
        if termination == "disconnect":
            raise smolvla.ClientDisconnected("disconnected")
        raise AssertionError("action wait must not run")

    monkeypatch.setattr(smolvla, "load_token_list", lambda _: ["token"])
    monkeypatch.setattr(smolvla, "RobotClient", lambda **_: client)
    monkeypatch.setattr(smolvla, "SharedMemoryManager", FakeSharedMemoryManager)
    monkeypatch.setattr(smolvla, "ActionDebugLogger", FakeActionDebugLogger)
    monkeypatch.setattr(
        smolvla,
        "wait_for_smolvla_config",
        lambda _client, _deadline: client.config,
    )
    monkeypatch.setattr(smolvla, "wait_for_start", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(smolvla, "wait_for_action_or_stop", wait_for_action)
    monkeypatch.setattr(smolvla, "get_real_umi_obs_dict", lambda **_kwargs: {})
    monkeypatch.setattr(
        smolvla,
        "go",
        types.SimpleNamespace(Figure=FakeFigure, Scatter=lambda **kwargs: kwargs),
        raising=False,
    )
    monkeypatch.setattr(smolvla.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(smolvla.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(smolvla.cv2, "setNumThreads", lambda _count: None)
    monkeypatch.setitem(sys.modules, "real_world.bimanual_umi_env", fake_env_module)

    if termination == "controller_stop_failure":
        with pytest.raises(RuntimeError, match="failed to stop Controller"):
            call_main()
    else:
        call_main()

    stop_index = events.index(("controller_stop_join", True))
    assert stop_index < events.index("client_stop")
    assert stop_index < events.index("env_exit")
    assert events.count(("controller_stop_join", True)) == 1
    assert "logger_close" in events
    assert "slow_plot" not in events
    assert "slow_png" not in events


@pytest.mark.parametrize(
    ("runtime_updates", "expected_scheduled"),
    [
        pytest.param({}, 5, id="default-cli-limit"),
        pytest.param({"max_executed_actions": 1}, 1, id="tighter-cli-limit"),
        pytest.param(
            {"max_executed_actions": smolvla.SMOLVLA_ACTION_HORIZON},
            5,
            id="explicit-cli-cannot-loosen",
        ),
    ],
)
def test_main_limits_scheduled_actions_to_negotiated_steps_per_inference(
    monkeypatch, runtime_updates, expected_scheduled
):
    scheduled_counts = []

    class StopAfterSchedule(RuntimeError):
        pass

    class RuntimeClient(FakeStartupClient):
        def publish_obs(self, _obs):
            return 17

        def get_state_update(self):
            return None

        def action_origin_is_connected(self, obs_seq):
            return obs_seq == 17

    class FakeSharedMemoryManager:
        def __enter__(self):
            return object()

        def __exit__(self, *_args):
            return False

    class SchedulingEnv:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def get_obs(self):
            return {
                "timestamp": np.array([100.0]),
                **zero_robot_obs(),
            }

        def exec_actions(self, *, actions, timestamps):
            assert len(actions) == len(timestamps)
            scheduled_counts.append(len(actions))
            raise StopAfterSchedule

        def stop_controller(self, *, wait):
            assert wait is True

        def get_debug_info(self):
            return {}

    class FakeActionDebugLogger:
        def __init__(self, **_kwargs):
            pass

        def plot(self):
            pass

        def close(self):
            pass

    class FakeFigure:
        def add_trace(self, _trace):
            pass

        def update_layout(self, **_kwargs):
            pass

        def write_image(self, _path):
            pass

    client = RuntimeClient(config=valid_config(steps_per_inference=5))
    fake_env_module = types.ModuleType("real_world.bimanual_umi_env")
    fake_env_module.BimanualUmiEnv = SchedulingEnv

    monkeypatch.setattr(smolvla, "load_token_list", lambda _: ["token"])
    monkeypatch.setattr(smolvla, "RobotClient", lambda **_: client)
    monkeypatch.setattr(smolvla, "SharedMemoryManager", FakeSharedMemoryManager)
    monkeypatch.setattr(smolvla, "ActionDebugLogger", FakeActionDebugLogger)
    monkeypatch.setattr(
        smolvla,
        "wait_for_smolvla_config",
        lambda _client, _deadline: client.config,
    )
    monkeypatch.setattr(smolvla, "wait_for_start", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        smolvla,
        "wait_for_action_or_stop",
        lambda *_args, **_kwargs: valid_action_chunk(),
    )
    monkeypatch.setattr(smolvla, "get_real_umi_obs_dict", lambda **_kwargs: {})
    monkeypatch.setattr(
        smolvla,
        "get_real_umi_action",
        lambda action, *_args: np.zeros((len(action), 14), dtype=np.float32),
    )
    monkeypatch.setattr(
        smolvla,
        "go",
        types.SimpleNamespace(Figure=FakeFigure, Scatter=lambda **kwargs: kwargs),
        raising=False,
    )
    monkeypatch.setattr(smolvla.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(smolvla.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(smolvla.time, "time", lambda: 100.0)
    monkeypatch.setattr(smolvla.cv2, "setNumThreads", lambda _count: None)
    monkeypatch.setattr(smolvla.os.path, "exists", lambda _path: True)
    monkeypatch.setitem(sys.modules, "real_world.bimanual_umi_env", fake_env_module)

    with pytest.raises(StopAfterSchedule):
        call_main(exec_mode="block", **runtime_updates)

    assert scheduled_counts == [expected_scheduled]


def test_dry_run_receives_remaining_start_deadline(monkeypatch):
    client = FakeStartupClient()
    captured = {}
    times = iter([100.0, 104.0])

    monkeypatch.setattr(smolvla, "load_token_list", lambda _: ["token"])
    monkeypatch.setattr(smolvla, "RobotClient", lambda **_: client)
    monkeypatch.setattr(smolvla.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(
        smolvla,
        "wait_for_smolvla_config",
        lambda _client, _deadline: valid_config(),
    )

    def run_dry_run(*_args, **kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(smolvla, "run_dry_run", run_dry_run)

    assert call_main(dry_run=True, start_timeout_s=10.0, action_timeout_s=30.0) == 0
    assert captured["start_timeout_s"] == pytest.approx(6.0)
    assert captured["action_timeout_s"] == 30.0

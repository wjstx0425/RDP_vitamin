from __future__ import annotations

from collections import deque
import threading

from click.testing import CliRunner
import numpy as np
import pytest
from websockets.exceptions import ConnectionClosedOK

from client import msgpack_numpy
from client.robot_client import RobotClient

TASK = "fold the towel"
CONFIG = {
    "language_prompt": TASK,
    "action_horizon": 2,
    "steps_per_inference": 2,
    "single_arm_mode": False,
}
LIMITS = {
    "max_pos_delta": 0.03,
    "max_rot_delta": 0.35,
    "min_gripper": -0.05,
    "max_gripper": 1.05,
}


def make_valid_action(action_horizon: int = 2, n_robots: int = 2) -> np.ndarray:
    action = np.zeros((action_horizon, n_robots, 10), dtype=np.float32)
    action[..., 3] = 1.0
    action[..., 7] = 1.0
    action[..., 9] = 0.5
    return action.reshape(action_horizon, n_robots * 10)


class FakeProtocolClient:
    def __init__(self, *, config=CONFIG, actions=(), states=("start",)) -> None:
        self.config = config
        self.actions = deque(actions)
        self.states = deque(states)
        self.published = []
        self.action_waits = []
        self.action_acks = []
        self.events = []
        self.started = False
        self.stopped = False
        self.joined = False

    def start_background(self) -> None:
        self.started = True

    def enable_action_ack(self) -> None:
        self.events.append(("enable_ack", None))

    def wait_for_connection(self, timeout=None) -> bool:
        del timeout
        return True

    def wait_for_config(self, timeout=None):
        del timeout
        return self.config

    def is_connected(self) -> bool:
        return True

    def get_state_update(self):
        state = self.states.popleft() if self.states else None
        if self.action_acks and state == "stop":
            self.events.append(("wait_stop", state))
        return state

    def publish_obs(self, obs) -> int:
        self.published.append(obs)
        return len(self.published) - 1

    def wait_for_action(self, obs_seq, timeout=None):
        self.action_waits.append((obs_seq, timeout))
        self.events.append(("action", obs_seq))
        return self.actions.popleft() if self.actions else None

    def publish_action_ack(self, obs_seq) -> None:
        self.action_acks.append(obs_seq)
        self.events.append(("ack", obs_seq))
        if not self.actions:
            self.states.append("stop")

    def stop(self) -> None:
        self.stopped = True

    def join(self, timeout=None) -> None:
        self.joined = True


def test_make_synthetic_observation_matches_policy_schema() -> None:
    from deploy_scripts.vbvla_dry_run import make_synthetic_observation

    observation = make_synthetic_observation(TASK)

    for key in ("observation.images.camera0", "observation.images.camera1"):
        assert observation[key].shape == (256, 256, 3)
        assert observation[key].dtype == np.uint8
    assert observation["observation.state"].shape == (20,)
    assert observation["observation.state"].dtype == np.float32
    assert np.all(np.isfinite(observation["observation.state"]))
    assert observation["task"] == TASK


def test_run_dry_run_has_one_warmup_and_bounded_validated_exchanges(capsys) -> None:
    from deploy_scripts.vbvla_dry_run import run_dry_run

    client = FakeProtocolClient(actions=[make_valid_action(), make_valid_action()])

    result = run_dry_run(
        client,
        CONFIG,
        iterations=2,
        action_timeout_s=1.0,
        limits=LIMITS,
    )

    assert result == 0
    assert len(client.published) == 3
    assert [obs["task"] for obs in client.published] == [TASK, TASK, TASK]
    assert [obs_seq for obs_seq, _ in client.action_waits] == [1, 2]
    assert client.action_acks == [1, 2]
    assert all(timeout <= 0.1 for _, timeout in client.action_waits)
    assert "completed 2/2 dry-run exchanges" in capsys.readouterr().out


def test_run_dry_run_waits_for_stop_after_validating_and_acknowledging(monkeypatch) -> None:
    import deploy_scripts.vbvla_dry_run as dry_run

    client = FakeProtocolClient(actions=[make_valid_action()])
    validate_action_chunk = dry_run.validate_action_chunk

    def track_validation(*args, **kwargs):
        client.events.append(("validate", None))
        return validate_action_chunk(*args, **kwargs)

    monkeypatch.setattr(dry_run, "validate_action_chunk", track_validation)

    result = dry_run.run_dry_run(client, CONFIG, 1, 1.0, LIMITS)
    client.events.append(("return", result))

    assert client.events[-4:] == [
        ("validate", None),
        ("ack", 1),
        ("wait_stop", "stop"),
        ("return", 0),
    ]


def test_run_dry_run_uses_action_horizon_instead_of_steps_per_inference() -> None:
    from deploy_scripts.vbvla_dry_run import run_dry_run

    config = {
        **CONFIG,
        "action_horizon": 50,
        "steps_per_inference": 5,
    }
    client = FakeProtocolClient(config=config, actions=[make_valid_action(action_horizon=50)])

    result = run_dry_run(client, config, 1, 1.0, LIMITS)

    assert result == 0
    assert client.action_acks == [1]


def test_run_dry_run_fails_closed_when_action_horizon_is_missing() -> None:
    from deploy_scripts.vbvla_dry_run import run_dry_run

    config = {key: value for key, value in CONFIG.items() if key != "action_horizon"}
    client = FakeProtocolClient(config=config, actions=[make_valid_action()])

    with pytest.raises(KeyError, match="action_horizon"):
        run_dry_run(client, config, 1, 1.0, LIMITS)

    assert client.action_acks == []


@pytest.mark.parametrize("invalid_horizon", [None, 0, -1, True, 50.5, "50"])
def test_run_dry_run_fails_closed_when_action_horizon_is_invalid(invalid_horizon) -> None:
    from deploy_scripts.vbvla_dry_run import run_dry_run

    config = {**CONFIG, "action_horizon": invalid_horizon}
    client = FakeProtocolClient(config=config, actions=[make_valid_action()])

    with pytest.raises(ValueError, match="action_horizon must be a positive integer"):
        run_dry_run(client, config, 1, 1.0, LIMITS)

    assert client.action_acks == []


def test_run_dry_run_rejects_an_unsafe_action() -> None:
    from deploy_scripts.vbvla_dry_run import run_dry_run
    from deploy_scripts.vbvla_safety import UnsafeActionError

    unsafe = make_valid_action()
    unsafe[0, 0] = 0.031
    client = FakeProtocolClient(actions=[unsafe])

    with pytest.raises(UnsafeActionError):
        run_dry_run(client, CONFIG, 1, 1.0, LIMITS)

    assert client.action_acks == []


def test_run_dry_run_rechecks_action_origin_after_validation(monkeypatch) -> None:
    import deploy_scripts.vbvla_dry_run as dry_run
    from deploy_scripts.vbvla_safety import ClientDisconnected

    class OriginClient(FakeProtocolClient):
        origin_connected = True

        def action_origin_is_connected(self, obs_seq):
            assert obs_seq == 1
            return self.origin_connected

    client = OriginClient(actions=[make_valid_action()])
    validate = dry_run.validate_action_chunk

    def disconnect_after_validation(*args, **kwargs):
        result = validate(*args, **kwargs)
        client.origin_connected = False
        return result

    monkeypatch.setattr(dry_run, "validate_action_chunk", disconnect_after_validation)

    with pytest.raises(ClientDisconnected, match="origin"):
        dry_run.run_dry_run(client, CONFIG, 1, 1.0, LIMITS)

    assert client.action_acks == []


class FakeWebSocket:
    def __init__(self, client, *, stop_after_publish=False) -> None:
        self.client = client
        self.stop_after_publish = stop_after_publish
        self.events = []
        self.action_returned = False
        self.published = False
        self.close_requested = threading.Event()

    def send(self, payload) -> None:
        message = msgpack_numpy.unpackb(payload)
        self.events.append(("sent", message))
        if message.get("type") == "action_ack":
            self.client.stop()

    def recv(self, timeout=None):
        assert timeout is None
        if not self.action_returned:
            self.action_returned = True
            return msgpack_numpy.packb(
                {
                    "type": "action",
                    "obs_seq": 7,
                    "action": np.array([[7]], dtype=np.float32),
                }
            )
        if not self.published:
            self.published = True
            np.testing.assert_array_equal(
                self.client.wait_for_action(obs_seq=7, timeout=0.0),
                np.array([[7]], dtype=np.float32),
            )
            self.events.append(("publish", 7))
            self.client.publish_action_ack(7)
            if self.stop_after_publish:
                self.client.stop()
        self.close_requested.wait()
        raise ConnectionClosedOK(None, None)

    def close(self) -> None:
        self.close_requested.set()


class BlockingReceiveWebSocket:
    def __init__(self, action_obs_seq=None) -> None:
        self.action_obs_seq = action_obs_seq
        self.action_returned = False
        self.hello_sent = threading.Event()
        self.receive_blocked = threading.Event()
        self.observation_sent = threading.Event()
        self.ack_sent = threading.Event()
        self.close_requested = threading.Event()
        self.sent = []

    def send(self, payload) -> None:
        message = msgpack_numpy.unpackb(payload)
        self.sent.append(message)
        if message["type"] == "hello":
            self.hello_sent.set()
        elif message["type"] == "obs":
            self.observation_sent.set()
        elif message["type"] == "action_ack":
            self.ack_sent.set()

    def recv(self, timeout=None):
        del timeout
        if self.action_obs_seq is not None and not self.action_returned:
            self.action_returned = True
            return msgpack_numpy.packb(
                {
                    "type": "action",
                    "obs_seq": self.action_obs_seq,
                    "action": np.array([[self.action_obs_seq]], dtype=np.float32),
                }
            )
        self.receive_blocked.set()
        self.close_requested.wait()
        raise ConnectionClosedOK(None, None)

    def close(self) -> None:
        self.close_requested.set()


def test_robot_client_sends_observation_while_receive_is_blocked() -> None:
    client = RobotClient()
    websocket = BlockingReceiveWebSocket()
    handler = threading.Thread(
        target=client._handle_connection, args=(websocket,), daemon=True
    )
    handler.start()
    try:
        assert websocket.hello_sent.wait(timeout=1.0)
        assert websocket.receive_blocked.wait(timeout=1.0)
        client.publish_obs({"value": 7})
        assert websocket.observation_sent.wait(timeout=0.2)
    finally:
        websocket.close()
        handler.join(timeout=1.0)
    assert not handler.is_alive()


def test_robot_client_sends_ack_while_receive_is_blocked() -> None:
    client = RobotClient()
    client.enable_action_ack()
    websocket = BlockingReceiveWebSocket(action_obs_seq=7)
    handler = threading.Thread(
        target=client._handle_connection, args=(websocket,), daemon=True
    )
    handler.start()
    try:
        np.testing.assert_array_equal(
            client.wait_for_action(obs_seq=7, timeout=1.0),
            np.array([[7]], dtype=np.float32),
        )
        assert websocket.receive_blocked.wait(timeout=1.0)
        client.publish_action_ack(7)
        assert websocket.ack_sent.wait(timeout=0.2)
    finally:
        websocket.close()
        handler.join(timeout=1.0)
    assert not handler.is_alive()


def test_robot_client_emits_ack_only_after_publish_with_exact_sequence() -> None:
    client = RobotClient()
    client.enable_action_ack()
    websocket = FakeWebSocket(client)

    client._handle_connection(websocket)

    assert websocket.events == [
        ("sent", {"type": "hello", "protocol": "robot-bridge-v1"}),
        ("publish", 7),
        ("sent", {"type": "action_ack", "obs_seq": 7}),
    ]


def test_robot_client_stop_remains_higher_priority_than_pending_ack() -> None:
    client = RobotClient()
    client.enable_action_ack()
    websocket = FakeWebSocket(client, stop_after_publish=True)

    client._handle_connection(websocket)

    assert [message["type"] for event, message in websocket.events if event == "sent"] == ["hello"]


@pytest.mark.parametrize("received_obs_seq", [6, 8], ids=["stale", "future"])
def test_robot_client_rejects_action_sequence_mismatch(received_obs_seq) -> None:
    client = RobotClient()
    client._handle_message(  # noqa: SLF001
        {
            "type": "action",
            "obs_seq": received_obs_seq,
            "action": np.array([[received_obs_seq]], dtype=np.float32),
        },
        connection_generation=0,
    )

    with pytest.raises(RuntimeError, match="sequence"):
        client.wait_for_action(obs_seq=7, timeout=0.0)


def test_legacy_action_consumption_does_not_accumulate_ack_bookkeeping() -> None:
    client = RobotClient()

    for obs_seq in range(1000):
        client._handle_message(  # noqa: SLF001
            {
                "type": "action",
                "obs_seq": obs_seq,
                "action": np.array([[obs_seq]], dtype=np.float32),
            },
            connection_generation=0,
        )
        client.wait_for_action(obs_seq=obs_seq, timeout=0.0)

    assert client._consumed_action_generations == {}  # noqa: SLF001


def test_consumed_action_tracks_its_origin_generation_liveness() -> None:
    client = RobotClient()
    client.enable_action_ack()
    with client._condition:  # noqa: SLF001
        client._active_connection_generations.update({0, 1})  # noqa: SLF001
        client._connected = True  # noqa: SLF001
    client._handle_message(  # noqa: SLF001
        {
            "type": "action",
            "obs_seq": 7,
            "action": np.array([[7]], dtype=np.float32),
        },
        connection_generation=0,
    )
    client.wait_for_action(obs_seq=7, timeout=0.0)
    with client._condition:  # noqa: SLF001
        client._active_connection_generations.remove(0)  # noqa: SLF001

    origin_check = getattr(client, "action_origin_is_connected", None)
    assert origin_check is not None
    assert client.is_connected() is True
    assert origin_check(7) is False


def test_robot_client_stop_is_sticky_and_blocks_action_consumption() -> None:
    client = RobotClient()
    client.enable_action_ack()
    client._handle_message(  # noqa: SLF001
        {"type": "state", "state": "stop"},
        connection_generation=0,
    )
    client._handle_message(  # noqa: SLF001
        {
            "type": "action",
            "obs_seq": 7,
            "action": np.array([[7]], dtype=np.float32),
        },
        connection_generation=0,
    )

    with pytest.raises(RuntimeError, match="stop"):
        client.wait_for_action(obs_seq=7, timeout=0.0)

    assert client.is_stop_requested() is True


class GenerationWebSocket:
    def __init__(self, action_obs_seq=None) -> None:
        self.action_obs_seq = action_obs_seq
        self.action_enabled = threading.Event()
        self.action_returned = threading.Event()
        self.ack_published = threading.Event()
        self.polled_after_ack = threading.Event()
        self.hello_sent = threading.Event()
        self.close_requested = threading.Event()
        self.sent = []

    def send(self, payload) -> None:
        message = msgpack_numpy.unpackb(payload)
        self.sent.append(message)
        if message.get("type") == "hello":
            self.hello_sent.set()

    def recv(self, timeout=None):
        assert timeout is None
        if (
            self.action_obs_seq is not None
            and self.action_enabled.is_set()
            and not self.action_returned.is_set()
        ):
            self.action_returned.set()
            return msgpack_numpy.packb(
                {
                    "type": "action",
                    "obs_seq": self.action_obs_seq,
                    "action": np.array([[self.action_obs_seq]], dtype=np.float32),
                }
            )
        self.close_requested.wait()
        raise ConnectionClosedOK(None, None)

    def close(self) -> None:
        self.close_requested.set()


def test_late_ack_from_disconnected_connection_is_not_sent_to_replacement() -> None:
    client = RobotClient()
    client.enable_action_ack()
    with client._condition:  # noqa: SLF001
        client._active_connection_generations.update({0, 1})  # noqa: SLF001
        client._connected = True  # noqa: SLF001
    client._handle_message(  # noqa: SLF001
        {
            "type": "action",
            "obs_seq": 7,
            "action": np.array([[7]], dtype=np.float32),
        },
        connection_generation=0,
    )
    client.wait_for_action(obs_seq=7, timeout=0.0)
    with client._condition:  # noqa: SLF001
        client._active_connection_generations.remove(0)  # noqa: SLF001

    with pytest.raises(RuntimeError, match="disconnected"):
        client.publish_action_ack(7)
    assert client.is_connected() is True
    assert client._latest_action_ack is None  # noqa: SLF001


def test_old_connection_shutdown_does_not_mark_replacement_disconnected() -> None:
    client = RobotClient()
    connection_a = GenerationWebSocket()
    connection_b = GenerationWebSocket()
    thread_a = threading.Thread(
        target=client._handle_connection, args=(connection_a,), daemon=True
    )
    thread_b = threading.Thread(
        target=client._handle_connection, args=(connection_b,), daemon=True
    )
    thread_a.start()
    assert connection_a.hello_sent.wait(timeout=1.0)
    thread_b.start()
    assert connection_b.hello_sent.wait(timeout=1.0)

    connection_a.close_requested.set()
    thread_a.join(timeout=1.0)

    assert not thread_a.is_alive()
    connected_while_replacement_active = client.is_connected()

    connection_b.close_requested.set()
    thread_b.join(timeout=1.0)
    assert not thread_b.is_alive()
    assert connected_while_replacement_active is True
    assert client.is_connected() is False


def test_real_action_ack_is_published_only_after_successful_execution(monkeypatch) -> None:
    import deploy_scripts.bimanual_smolvla_online as online

    events = []

    def execute(*_args, **_kwargs):
        events.append("execute")
        return "result"

    class AckClient:
        def action_origin_is_connected(self, obs_seq):
            return obs_seq == 11

        def publish_action_ack(self, obs_seq):
            events.append(("ack", obs_seq))

    monkeypatch.setattr(online, "execute_action_chunk", execute)

    result = online.execute_action_chunk_and_publish_ack(AckClient(), 11)

    assert result == "result"
    assert events == ["execute", ("ack", 11)]


def test_real_action_ack_is_not_published_when_execution_fails(monkeypatch) -> None:
    import deploy_scripts.bimanual_smolvla_online as online
    from deploy_scripts.vbvla_safety import UnsafeActionError

    acks = []

    def reject(*_args, **_kwargs):
        raise UnsafeActionError("unsafe")

    class AckClient:
        def publish_action_ack(self, obs_seq):
            acks.append(obs_seq)

    monkeypatch.setattr(online, "execute_action_chunk", reject)

    with pytest.raises(UnsafeActionError):
        online.execute_action_chunk_and_publish_ack(AckClient(), 11)

    assert acks == []


def test_real_action_ack_is_not_published_when_controller_crashes_after_schedule(
    monkeypatch,
) -> None:
    import deploy_scripts.bimanual_smolvla_online as online

    acks = []

    class CrashedEnv:
        def check_controller_health(self):
            raise RuntimeError("controller child failed")

    class AckClient:
        def publish_action_ack(self, obs_seq):
            acks.append(obs_seq)

    monkeypatch.setattr(
        online,
        "execute_action_chunk",
        lambda *_args, **_kwargs: "scheduled",
    )

    with pytest.raises(RuntimeError, match="controller child failed"):
        online.execute_action_chunk_and_publish_ack(
            AckClient(), 11, env=CrashedEnv()
        )

    assert acks == []


def test_click_dry_run_negotiates_then_avoids_all_hardware(monkeypatch, tmp_path) -> None:
    import deploy_scripts.bimanual_smolvla_online as server

    client = FakeProtocolClient(
        config={
            **CONFIG,
            "data_type": "vision",
            "control_frequency": 10,
            "controller_frequency": 100,
            "no_state_obs_mode": False,
            "action_horizon": server.SMOLVLA_ACTION_HORIZON,
            "steps_per_inference": 5,
        },
        actions=[make_valid_action(action_horizon=server.SMOLVLA_ACTION_HORIZON)],
    )
    token_file = tmp_path / "tokens.txt"
    token_file.write_text("test-token\n", encoding="utf-8")

    monkeypatch.setattr(server, "RobotClient", lambda **_kwargs: client)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("dry-run crossed into hardware initialization")

    monkeypatch.setattr(server.np, "load", forbidden)
    monkeypatch.setattr(server, "SharedMemoryManager", forbidden)
    monkeypatch.setattr(server.cv2, "setNumThreads", forbidden)

    result = CliRunner().invoke(
        server.main,
        [
            "--dry-run",
            "--dry-run-iterations",
            "1",
            "--action-timeout-s",
            "0.5",
            "--token-file",
            str(token_file),
        ],
    )

    assert result.exit_code == 0, result.output
    assert client.started
    assert len(client.published) == 2
    assert client.stopped
    assert client.joined
    assert "completed 1/1 dry-run exchanges" in result.output


def test_vitac_dry_run_observation_contains_four_tactile_images() -> None:
    from deploy_scripts.vbvla_dry_run import make_synthetic_observation

    observation = make_synthetic_observation("pick", data_type="vitac")

    assert {
        "observation.images.tactile_left_0",
        "observation.images.tactile_right_0",
        "observation.images.tactile_left_1",
        "observation.images.tactile_right_1",
    }.issubset(observation)
    assert observation["observation.images.tactile_left_0"].dtype == np.uint8

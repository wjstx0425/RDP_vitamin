from __future__ import annotations

from collections import deque
from contextlib import suppress
import threading
import time
from typing import Any

from client import msgpack_numpy
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Request
from websockets.http11 import Response
from websockets.sync.server import Server
from websockets.sync.server import ServerConnection
from websockets.sync.server import serve


class RobotClientStopRequested(RuntimeError):
    """Signal that the remote client requested stop during a blocking operation."""


class RobotClient:
    """Persistent websocket bridge between robot-side control and remote policy."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        allowed_tokens: list[str] | tuple[str, ...] | set[str] | None = None,
    ):
        self.host = host
        self.port = port
        self._allowed_tokens = None if allowed_tokens is None else {str(token) for token in allowed_tokens}

        self._condition = threading.Condition()
        self._packer = msgpack_numpy.Packer()
        self._server: Server | None = None
        self._thread: threading.Thread | None = None
        self._stopped = False
        self._stop_requested = False
        self._connected = False

        self._config: Any = None
        self._state_updates: deque[Any] = deque()

        self._latest_obs: dict[str, Any] | None = None
        self._latest_obs_seq = -1

        self._latest_action: Any = None
        self._latest_action_obs_seq = -1
        self._latest_action_generation: int | None = None
        self._consumed_action_generations: dict[int, int] = {}
        self._action_ack_enabled = False

        self._latest_action_ack: dict[str, int | str] | None = None
        self._latest_action_ack_generation: int | None = None
        self._next_connection_generation = 0
        self._active_connection_generations: set[int] = set()

    def _send_outbound(
        self,
        websocket: ServerConnection,
        connection_generation: int,
        connection_done: threading.Event,
        sender_errors: list[BaseException],
    ) -> None:
        last_sent_obs_seq = -1
        last_sent_action_ack_obs_seq = -1
        initiated_shutdown = False

        try:
            while True:
                with self._condition:
                    self._condition.wait_for(
                        lambda: self._stopped
                        or connection_done.is_set()
                        or (
                            self._latest_action_ack is not None
                            and self._latest_action_ack_generation
                            == connection_generation
                            and int(self._latest_action_ack["obs_seq"])
                            > last_sent_action_ack_obs_seq
                        )
                        or (
                            self._latest_obs is not None
                            and self._latest_obs_seq > last_sent_obs_seq
                        )
                    )
                    if self._stopped or connection_done.is_set():
                        initiated_shutdown = self._stopped
                        break
                    if (
                        self._latest_action_ack is not None
                        and self._latest_action_ack_generation
                        == connection_generation
                        and int(self._latest_action_ack["obs_seq"])
                        > last_sent_action_ack_obs_seq
                    ):
                        outbound = self._latest_action_ack
                    else:
                        outbound = self._latest_obs

                websocket.send(self._packer.pack(outbound))
                if outbound["type"] == "action_ack":
                    last_sent_action_ack_obs_seq = int(outbound["obs_seq"])
                else:
                    last_sent_obs_seq = int(outbound["obs_seq"])
        except ConnectionClosed:
            pass
        except BaseException as exc:
            sender_errors.append(exc)
            initiated_shutdown = True
        finally:
            connection_done.set()
            with self._condition:
                self._condition.notify_all()
            if initiated_shutdown:
                with suppress(ConnectionClosed):
                    websocket.close()

    def _process_request(self, connection: ServerConnection, request: Request) -> Response | None:
        del connection

        if self._allowed_tokens is None:
            return None

        auth_header = request.headers.get("Authorization")
        if auth_header is None:
            return Response(
                401,
                "Unauthorized",
                Headers({"WWW-Authenticate": 'Bearer realm="robot-bridge"'}),
                b"Missing Authorization header.\n",
            )

        scheme, _, token = auth_header.partition(" ")
        if scheme.lower() != "bearer" or not token or token not in self._allowed_tokens:
            return Response(
                401,
                "Unauthorized",
                Headers({"WWW-Authenticate": 'Bearer realm="robot-bridge"'}),
                b"Invalid bearer token.\n",
            )

        return None

    def _handle_connection(self, websocket: ServerConnection) -> None:
        with self._condition:
            connection_generation = self._next_connection_generation
            self._next_connection_generation += 1
            self._active_connection_generations.add(connection_generation)
            self._connected = True
            self._condition.notify_all()

        connection_done = threading.Event()
        sender_errors: list[BaseException] = []
        sender_thread: threading.Thread | None = None
        receive_error: BaseException | None = None

        try:
            websocket.send(
                self._packer.pack(
                    {
                        "type": "hello",
                        "protocol": "robot-bridge-v1",
                    }
                )
            )
            sender_thread = threading.Thread(
                target=self._send_outbound,
                args=(
                    websocket,
                    connection_generation,
                    connection_done,
                    sender_errors,
                ),
                name=f"RobotClientSender-{self.port}-{connection_generation}",
                daemon=True,
            )
            sender_thread.start()

            while True:
                raw_message = websocket.recv()
                if isinstance(raw_message, str):
                    raise RuntimeError("Robot bridge expects binary websocket frames.")

                message = msgpack_numpy.unpackb(raw_message)
                self._handle_message(message, connection_generation)
        except ConnectionClosed:
            pass
        except BaseException as exc:
            receive_error = exc
        finally:
            connection_done.set()
            with self._condition:
                self._condition.notify_all()

            sender_stuck = False
            if sender_thread is not None:
                sender_thread.join(timeout=1.0)
                sender_stuck = sender_thread.is_alive()

            with self._condition:
                self._active_connection_generations.discard(connection_generation)
                self._connected = bool(self._active_connection_generations)
                if self._latest_action_generation == connection_generation:
                    self._latest_action = None
                    self._latest_action_obs_seq = -1
                    self._latest_action_generation = None
                self._condition.notify_all()

        if receive_error is not None:
            raise receive_error
        if sender_stuck:
            raise RuntimeError("Robot bridge sender thread did not stop")
        if sender_errors:
            raise sender_errors[0]

    def _handle_message(self, message: dict[str, Any], connection_generation: int) -> None:
        message_type = message.get("type")

        with self._condition:
            if message_type == "config":
                self._config = message["config"]
            elif message_type == "state":
                state = message["state"]
                self._state_updates.append(state)
                if state == "stop":
                    self._stop_requested = True
            elif message_type == "action":
                self._latest_action = message["action"]
                self._latest_action_obs_seq = int(message["obs_seq"])
                self._latest_action_generation = connection_generation
            else:
                raise ValueError(f"Unsupported websocket message type: {message_type}")

            self._condition.notify_all()

    def run(self) -> None:
        with serve(
            self._handle_connection,
            host=self.host,
            port=self.port,
            process_request=self._process_request,
            compression=None,
            max_size=None,
            # This bridge can legitimately spend long periods inside blocking
            # policy inference or user input before the next recv() call.
            # Disable websocket keepalive to avoid false timeouts on the sync API.
            ping_interval=None,
        ) as server:
            with self._condition:
                self._server = server
                self._condition.notify_all()
            server.serve_forever()

    def start_background(self, daemon: bool = True) -> threading.Thread:
        if self._thread is not None and self._thread.is_alive():
            return self._thread

        with self._condition:
            self._stopped = False
            self._stop_requested = False

        self._thread = threading.Thread(
            target=self.run,
            name=f"RobotClient-{self.port}",
            daemon=daemon,
        )
        self._thread.start()
        return self._thread

    def stop(self) -> None:
        with self._condition:
            self._stopped = True
            server = self._server
            self._condition.notify_all()

        if server is not None:
            server.shutdown()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def is_connected(self) -> bool:
        with self._condition:
            return self._connected

    def is_stop_requested(self) -> bool:
        with self._condition:
            return self._stop_requested

    def enable_action_ack(self) -> None:
        """Enable one-in-flight generation tracking for ACK-aware clients."""
        with self._condition:
            self._action_ack_enabled = True
            self._consumed_action_generations.clear()

    def action_origin_is_connected(self, obs_seq: int) -> bool:
        """Return whether the consumed action still belongs to a live connection."""
        with self._condition:
            generation = self._consumed_action_generations.get(int(obs_seq))
            return generation is not None and generation in self._active_connection_generations

    def wait_for_connection(self, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout

        with self._condition:
            while not self._connected and not self._stopped:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return self._connected

    def wait_for_config(self, timeout: float | None = None) -> Any:
        deadline = None if timeout is None else time.monotonic() + timeout

        with self._condition:
            while self._config is None and not self._stopped:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return None
                self._condition.wait(remaining)
            return self._config

    def get_state_update(self) -> Any:
        with self._condition:
            if self._state_updates:
                return self._state_updates.popleft()
            return None

    def publish_obs(self, obs: Any) -> int:
        with self._condition:
            self._latest_obs_seq += 1
            self._latest_obs = {
                "type": "obs",
                "obs_seq": self._latest_obs_seq,
                "obs": obs,
            }
            self._condition.notify_all()
            return self._latest_obs_seq

    def publish_action_ack(self, obs_seq: int) -> None:
        with self._condition:
            if self._stop_requested:
                raise RuntimeError("Cannot acknowledge an action after a client stop request")
            normalized_obs_seq = int(obs_seq)
            connection_generation = self._consumed_action_generations.get(
                normalized_obs_seq,
                None,
            )
            if connection_generation is None:
                raise RuntimeError(
                    f"Cannot acknowledge unconsumed action for observation {normalized_obs_seq}"
                )
            if connection_generation not in self._active_connection_generations:
                self._consumed_action_generations.pop(normalized_obs_seq, None)
                raise RuntimeError(
                    f"Cannot acknowledge action {normalized_obs_seq}: origin disconnected"
                )
            self._consumed_action_generations.pop(normalized_obs_seq, None)
            self._latest_action_ack = {
                "type": "action_ack",
                "obs_seq": normalized_obs_seq,
            }
            self._latest_action_ack_generation = connection_generation
            self._condition.notify_all()

    def wait_for_action(self, obs_seq: int, timeout: float | None = None) -> Any:
        deadline = None if timeout is None else time.monotonic() + timeout

        with self._condition:
            while not self._stopped:
                if self._stop_requested:
                    raise RobotClientStopRequested(
                        "Client stop requested before action consumption"
                    )
                if self._latest_action is not None and self._latest_action_obs_seq != obs_seq:
                    received_obs_seq = self._latest_action_obs_seq
                    self._latest_action = None
                    self._latest_action_generation = None
                    raise RuntimeError(
                        f"Action sequence {received_obs_seq} does not match requested {obs_seq}"
                    )
                if self._latest_action is not None:
                    action = self._latest_action
                    if self._latest_action_generation is None:
                        raise RuntimeError("Cannot consume an action without a connection generation")
                    if self._action_ack_enabled:
                        self._consumed_action_generations.clear()
                        self._consumed_action_generations[int(obs_seq)] = (
                            self._latest_action_generation
                        )
                    self._latest_action = None
                    self._latest_action_generation = None
                    return action

                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return None
                self._condition.wait(remaining)

        return None

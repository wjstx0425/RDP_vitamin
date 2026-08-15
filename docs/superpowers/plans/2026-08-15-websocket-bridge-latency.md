# WebSocket Bridge Latency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the two 50 ms polling stalls from each robot bridge action cycle while preserving the existing synchronous protocol and public APIs.

**Architecture:** Keep each WebSocket server handler as the connection's only receiver and add one condition-driven sender thread per connection. Publishing an ACK or observation wakes that sender immediately; connection-local shutdown state coordinates clean close and preserves generation isolation.

**Tech Stack:** Python 3.11, `threading`, `websockets>=15.0,<17.0`, msgpack, pytest, Ruff.

## Global Constraints

- Preserve the `robot-bridge-v1` wire format and all public client APIs.
- Preserve outbound priority: stop, action ACK, then newest observation.
- Preserve connection-generation isolation across overlapping connections.
- Do not modify RDP inference, cameras, action scheduling, or Typhon control.
- Do not start cameras, policies, controllers, or physical robot processes.
- A real localhost 50-cycle ACK/observation round-trip must have p95 below 15 ms.
- Do not stage or commit the existing `configs/server_config.py` user change.

---

### Task 1: Deterministic blocked-receive regression

**Files:**
- Modify: `tests/test_vbvla_dry_run.py`

**Interfaces:**
- Consumes: `RobotClient._handle_connection()`, `publish_obs()`, `wait_for_action()`, and `publish_action_ack()`.
- Produces: deterministic regression tests that require outbound sends while `recv()` remains blocked.

- [ ] **Step 1: Add a blocking fake WebSocket**

Add a thread-safe fake whose `recv()` returns an optional first action and then
blocks on an event until `close()` is called. Its `send()` decodes messages and
sets `hello_sent`, `observation_sent`, and `ack_sent` events. `recv()` accepts
the production signature but intentionally doesn't implement timeout polling:

```python
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
```

- [ ] **Step 2: Add observation and ACK wake-up tests**

```python
def test_robot_client_sends_observation_while_receive_is_blocked() -> None:
    client = RobotClient()
    websocket = BlockingReceiveWebSocket()
    handler = threading.Thread(target=client._handle_connection, args=(websocket,), daemon=True)
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
    handler = threading.Thread(target=client._handle_connection, args=(websocket,), daemon=True)
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
```

- [ ] **Step 3: Run the new tests and verify RED**

Run:

```bash
.venv/bin/pytest \
  tests/test_vbvla_dry_run.py::test_robot_client_sends_observation_while_receive_is_blocked \
  tests/test_vbvla_dry_run.py::test_robot_client_sends_ack_while_receive_is_blocked -vv
```

Expected: both tests fail at their outbound event assertions because the
single handler thread is blocked in `recv()`.

### Task 2: Condition-driven sender lifecycle

**Files:**
- Modify: `client/robot_client.py:86-151`
- Modify: `tests/test_vbvla_dry_run.py:235-390`

**Interfaces:**
- Consumes: existing `_condition`, `_stopped`, generation-tagged ACK state, and latest-observation state.
- Produces: private `_send_outbound()` and event-driven `_handle_connection()` behavior; public methods remain unchanged.

- [ ] **Step 1: Add a private sender loop**

Add a method with the exact interface below. It tracks sent sequences locally,
waits on `_condition`, selects ACK before observation, releases the lock before
packing/sending, and closes the connection only when server stop or a sender
failure must unblock `recv()`.

```python
def _send_outbound(
    self,
    websocket: ServerConnection,
    connection_generation: int,
    connection_done: threading.Event,
    sender_errors: list[BaseException],
) -> None:
    last_sent_obs_seq = -1
    last_sent_action_ack_obs_seq = -1
    try:
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._stopped
                    or connection_done.is_set()
                    or (
                        self._latest_action_ack is not None
                        and self._latest_action_ack_generation == connection_generation
                        and int(self._latest_action_ack["obs_seq"])
                        > last_sent_action_ack_obs_seq
                    )
                    or (
                        self._latest_obs is not None
                        and self._latest_obs_seq > last_sent_obs_seq
                    )
                )
                if self._stopped or connection_done.is_set():
                    break
                if (
                    self._latest_action_ack is not None
                    and self._latest_action_ack_generation == connection_generation
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
    finally:
        initiated_shutdown = self._stopped or bool(sender_errors)
        connection_done.set()
        with self._condition:
            self._condition.notify_all()
        if initiated_shutdown:
            with suppress(ConnectionClosed):
                websocket.close()
```

Use a local non-optional `outbound` after the predicate guarantees one exists;
if static analysis requires it, assign it in both ACK and observation branches
rather than introducing a nullable send.

- [ ] **Step 2: Make the handler receive-only after hello**

Import `suppress` from `contextlib`. In `_handle_connection()`, send hello,
create a `threading.Event` and sender-error list, start a daemon sender thread,
then replace the timed receive loop with:

```python
while True:
    raw_message = websocket.recv()
    if isinstance(raw_message, str):
        raise RuntimeError("Robot bridge expects binary websocket frames.")
    message = msgpack_numpy.unpackb(raw_message)
    self._handle_message(message, connection_generation)
```

In `finally`, set the connection event, notify the condition, close the socket
idempotently to unblock either side, join the sender with a bounded timeout,
perform the existing generation cleanup, and raise a clear runtime error if the
sender cannot stop. After cleanup, re-raise the first unexpected sender error
when the receive side ended normally from the sender-initiated close.

- [ ] **Step 3: Adapt legacy fake WebSockets to blocking receive semantics**

Remove assertions that `timeout == 0.05`. Fakes used by ordering and generation
tests must provide `close()` and terminate by raising `ConnectionClosedOK`
rather than an endless stream of immediate `TimeoutError`. Keep all original
ordering and generation assertions unchanged.

- [ ] **Step 4: Run focused bridge tests and verify GREEN**

Run:

```bash
.venv/bin/pytest tests/test_vbvla_dry_run.py -k 'robot_client or connection or ack' -vv
```

Expected: all selected tests pass with no hanging handler or sender thread.

- [ ] **Step 5: Commit the event-driven bridge change**

```bash
git add client/robot_client.py tests/test_vbvla_dry_run.py
git commit -m "fix: wake websocket bridge sends without polling"
```

### Task 3: Real localhost latency acceptance test

**Files:**
- Modify: `tests/test_vbvla_dry_run.py`

**Interfaces:**
- Consumes: production `RobotClient`, `InterfaceClient`, and ACK-aware bridge protocol.
- Produces: 50-cycle p95 regression coverage for the original 100 ms symptom.

- [ ] **Step 1: Add localhost helpers and imports**

Import `socket`, `time`, and `InterfaceClient`. Add a helper that binds a
temporary loopback TCP socket to port zero, reads the selected port, and closes
the socket before returning it:

```python
def get_unused_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
```

- [ ] **Step 2: Add the 50-cycle production protocol test**

```python
def test_robot_bridge_localhost_cycle_p95_is_below_15_ms() -> None:
    port = get_unused_loopback_port()
    server = RobotClient(host="127.0.0.1", port=port)
    server.enable_action_ack()
    server.start_background()
    remote = None
    cycle_ms = []
    try:
        remote = InterfaceClient(ip="127.0.0.1", port=port)
        assert server.wait_for_connection(timeout=1.0)
        for value in range(50):
            started = time.perf_counter()
            obs_seq = server.publish_obs({"value": value})
            received_seq, received_obs = remote.recv_obs(timeout=1.0)
            assert received_seq == obs_seq
            assert received_obs == {"value": value}
            action = np.array([[value]], dtype=np.float32)
            remote.send_action(action, obs_seq)
            np.testing.assert_array_equal(
                server.wait_for_action(obs_seq, timeout=1.0), action
            )
            server.publish_action_ack(obs_seq)
            ack = remote._recv_message(timeout=1.0)  # noqa: SLF001
            assert ack == {"type": "action_ack", "obs_seq": obs_seq}
            cycle_ms.append((time.perf_counter() - started) * 1000.0)
    finally:
        if remote is not None:
            remote.close()
        server.stop()
        server.join(timeout=2.0)

    p95_ms = float(np.percentile(cycle_ms, 95))
    assert p95_ms < 15.0, f"localhost bridge p95 was {p95_ms:.3f} ms: {cycle_ms}"
```

- [ ] **Step 3: Run the localhost acceptance test**

Run:

```bash
.venv/bin/pytest tests/test_vbvla_dry_run.py::test_robot_bridge_localhost_cycle_p95_is_below_15_ms -vv
```

Expected: PASS, reporting no assertion and completing substantially faster
than the approximately five seconds required by the old two-timeout loop.

- [ ] **Step 4: Run full verification**

Run:

```bash
.venv/bin/pytest -q
.venv/bin/ruff check client/robot_client.py tests/test_vbvla_dry_run.py
git diff --check
```

Expected: pytest reports zero failures, Ruff reports no errors, and
`git diff --check` prints nothing.

- [ ] **Step 5: Commit the latency acceptance coverage**

```bash
git add tests/test_vbvla_dry_run.py
git commit -m "test: enforce websocket bridge latency budget"
```


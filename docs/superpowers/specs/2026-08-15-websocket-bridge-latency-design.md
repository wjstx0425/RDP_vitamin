# WebSocket Bridge Latency Design

## Problem

`RobotClient._handle_connection()` currently checks for one pending outbound
message and then calls `websocket.recv(timeout=0.05)`. Publishing an action ACK
or the next observation only notifies `RobotClient._condition`; it cannot wake
the blocking WebSocket receive. The synchronous request/ACK protocol therefore
pays approximately one 50 ms timeout before each ACK and another before each
observation. A localhost reproduction measured about 100 ms per cycle even
with tiny messages and no model or hardware.

## Goals

- Remove timeout-driven outbound latency from the robot bridge.
- Preserve the `robot-bridge-v1` wire format and all public client APIs.
- Preserve outbound priority: stop, action ACK, then newest observation.
- Preserve connection-generation isolation so an ACK is never delivered to a
  replacement connection.
- Support the declared `websockets>=15.0,<17.0` dependency range.
- Achieve a 50-cycle localhost ACK/observation round-trip p95 below 15 ms.
- Keep all verification independent of cameras, policies, controllers, and
  physical robots.

## Non-goals

- Changing RDP inference, observation contents, action scheduling, or the
  Typhon controller.
- Changing the bridge message schema or introducing client-visible methods.
- Replacing the synchronous WebSocket implementation with asyncio.
- Addressing the separate Typhon error-state or gripper oscillation issues.

## Considered Approaches

### Dedicated sender thread per connection

Keep the server handler as the sole receiver and add one sender thread for the
connection. The sender waits on the existing condition variable and can be
woken immediately by observation or ACK publication. `websockets` permits one
thread to receive while another sends, while still rejecting concurrent calls
to the same receive operation. This is the selected approach because it fixes
the root cause without changing the protocol or surrounding synchronous API.

### Shorter receive timeout

Reducing 50 ms to 1 ms would reduce latency but retain polling, CPU overhead,
and scheduler-dependent delay. It would treat the symptom rather than make
outbound delivery event-driven, so it is rejected.

### Asyncio migration

An asynchronous queue and event loop would also solve the wake-up problem, but
would require rewriting server lifecycle and synchronous callers. That scope
and compatibility risk are disproportionate to this isolated latency defect,
so it is rejected.

## Architecture

After registering a connection generation, the handler sends the existing
hello message and starts one daemon sender thread. The handler then performs a
blocking `websocket.recv()` with no polling timeout and remains the only caller
of `recv()`.

The sender thread waits on `RobotClient._condition` until one of these predicates
becomes true:

1. the server is stopped;
2. the connection is closing;
3. a generation-matched action ACK has a newer observation sequence;
4. a newer observation is available.

It selects at most one pending message per wake-up, releases the condition lock,
and sends the packed message. ACK remains higher priority than observation.
Because `publish_obs()` and `publish_action_ack()` already notify the condition,
no polling interval remains on the outbound path.

Connection-local shutdown state must not be stored in shared global booleans,
because old and replacement connections may overlap. A connection-local event
coordinates the handler and sender. A sender failure closes the WebSocket to
unblock the handler's receive; an inbound disconnect signals the sender to exit.
Server-initiated stop wakes the sender, which closes the WebSocket and thereby
unblocks the handler. The handler joins the sender before removing its active
connection generation.

Normal `ConnectionClosed` exceptions remain clean disconnects. Unexpected
sender exceptions are retained and surfaced by the handler after both sides of
the connection have shut down, rather than being silently lost in a background
thread.

## Testing

Testing follows a red-green sequence.

First, deterministic fake-WebSocket tests hold `recv()` blocked indefinitely.
One test publishes a new observation and requires it to be sent while receive
is still blocked. A second test receives and consumes an action, publishes its
ACK, and requires that ACK to be sent while the handler has returned to a
blocked receive. These tests fail under the current single-thread loop without
depending on a guessed 50 ms delay.

Existing connection-generation, stop-priority, stale-action, and disconnect
tests remain in force and are adapted only where their fake WebSocket contract
assumes `timeout=0.05`.

A real localhost integration test runs 50 complete cycles using the production
`RobotClient` and WebSocket client. Each cycle publishes an observation,
receives it, sends an action, consumes and acknowledges it on the server, and
receives the ACK. Its measured p95 must be below 15 ms and must show no 50 or
100 ms timing staircase.

Verification consists of the focused bridge tests, the full pytest suite, Ruff,
and a fresh standalone localhost latency result. No hardware-related process is
started.

## Files

- Modify `client/robot_client.py` for the per-connection sender lifecycle.
- Modify `tests/test_vbvla_dry_run.py` for deterministic concurrency regression
  tests and the real localhost latency test.


import importlib
import multiprocessing as mp
import os
import threading
import time

import numpy as np
import pytest

from utils.shared_memory.shared_memory_queue import Empty


controller_module = importlib.import_module(
    "real_world.robot_api.arm.Controller"
)


class _OwnerDeathController(controller_module.Controller):
    """Minimal no-hardware Controller whose child dies while owning the gate."""

    def __init__(self):
        mp.Process.__init__(self, name="owner_death_controller")
        self.verbose = False
        self.dispatch_lock = mp.Lock()
        self._stop_lock = mp.Lock()
        self.stop_event = mp.Event()
        self._stop_requested = False
        self.input_queue = mp.Queue()
        self.shutdown_timeout_s = 0.3
        self._dispatch_poll_interval_s = 0.02
        self.owner_ready = mp.Event()

    def run(self):
        self.dispatch_lock.acquire()
        self.owner_ready.set()
        os._exit(23)


def _run_owner_death_stop_scenario(result_queue):
    controller = _OwnerDeathController()
    controller.start(wait=False)
    if not controller.owner_ready.wait(timeout=1.0):
        result_queue.put(("error", "child never acquired dispatch lock"))
        return
    deadline = time.monotonic() + 1.0
    while controller.is_alive() and time.monotonic() < deadline:
        time.sleep(0.005)
    started = time.monotonic()
    try:
        controller.stop(wait=True)
    except BaseException as exc:
        result_queue.put(("error", f"{type(exc).__name__}: {exc}"))
        return
    result_queue.put(
        ("ok", time.monotonic() - started, controller.exitcode)
    )


class _FakeQueue:
    def __init__(self):
        self.puts = []

    def put(self, message):
        self.puts.append(message)

    def get_all(self):
        raise Empty


class _FakeRingBuffer:
    def __init__(self):
        self.puts = []
        self.reads = 0

    def put(self, message):
        self.puts.append(message)

    def get(self, **_kwargs):
        self.reads += 1
        return {"state": 1}

    def get_last_k(self, **_kwargs):
        self.reads += 1
        return {"state": 1}

    def get_all(self):
        self.reads += 1
        return {"state": 1}


def _construct_controller(monkeypatch, **controller_kwargs):
    input_queue = _FakeQueue()
    debug_queue = _FakeQueue()
    ring_buffer = _FakeRingBuffer()
    queue_results = iter((input_queue, debug_queue))

    monkeypatch.setattr(
        controller_module.SharedMemoryQueue,
        "create_from_examples",
        lambda **_kwargs: next(queue_results),
    )
    monkeypatch.setattr(
        controller_module.SharedMemoryRingBuffer,
        "create_from_examples",
        lambda **_kwargs: ring_buffer,
    )
    controller = controller_module.Controller(
        shm_manager=object(),
        frequency=80,
        **controller_kwargs,
    )
    return controller, input_queue, debug_queue, ring_buffer


@pytest.fixture
def constructed_controller(monkeypatch):
    return _construct_controller(monkeypatch)


def test_fatal_summary_is_shared_across_process_boundary(constructed_controller):
    controller, *_ = constructed_controller
    process = mp.get_context("fork").Process(
        target=controller._record_fatal_error,
        args=(RuntimeError("synthetic child failure"),),
    )

    process.start()
    process.join(timeout=2.0)

    assert process.exitcode == 0
    assert "RuntimeError: synthetic child failure" in controller.fatal_error_summary


def test_controller_shutdown_timeout_is_configurable_and_validated_before_queues(
    monkeypatch,
):
    allocations = []
    monkeypatch.setattr(
        controller_module.SharedMemoryQueue,
        "create_from_examples",
        lambda **kwargs: allocations.append(kwargs),
    )
    monkeypatch.setattr(
        controller_module.SharedMemoryRingBuffer,
        "create_from_examples",
        lambda **kwargs: allocations.append(kwargs),
    )

    with pytest.raises(ValueError, match="shutdown_timeout_s"):
        controller_module.Controller(
            shm_manager=object(), shutdown_timeout_s=0.0
        )

    assert allocations == []


@pytest.mark.parametrize("launch_timeout", [0, -1, np.nan, np.inf])
def test_controller_launch_timeout_is_finite_and_positive_before_queues(
    monkeypatch, launch_timeout
):
    allocations = []
    monkeypatch.setattr(
        controller_module.SharedMemoryQueue,
        "create_from_examples",
        lambda **kwargs: allocations.append(kwargs),
    )
    monkeypatch.setattr(
        controller_module.SharedMemoryRingBuffer,
        "create_from_examples",
        lambda **kwargs: allocations.append(kwargs),
    )

    with pytest.raises(
        ValueError,
        match="launch_timeout must be finite and positive",
    ):
        controller_module.Controller(
            shm_manager=object(), launch_timeout=launch_timeout
        )

    assert allocations == []


def test_ready_requires_a_live_healthy_child_and_health_checks_guard_reads(
    constructed_controller, monkeypatch
):
    controller, _input, _debug, ring_buffer = constructed_controller
    controller.ready_event.set()
    monkeypatch.setattr(controller, "is_alive", lambda: True)
    assert controller.is_ready

    controller._record_fatal_error(RuntimeError("controller exploded"))

    assert not controller.is_ready
    with pytest.raises(RuntimeError, match="controller exploded"):
        controller.get_state()
    with pytest.raises(RuntimeError, match="controller exploded"):
        controller.get_all_state()
    assert ring_buffer.reads == 0


def test_start_wait_surfaces_child_startup_failure_without_faking_ready(
    monkeypatch,
):
    controller, *_ = _construct_controller(monkeypatch, launch_timeout=0.2)
    controller._record_fatal_error(RuntimeError("startup failed"))
    controller.startup_event.set()
    monkeypatch.setattr(controller, "is_alive", lambda: False)

    started = time.monotonic()
    with pytest.raises(RuntimeError, match="startup failed"):
        controller.start_wait()

    assert time.monotonic() - started < controller.launch_timeout / 2
    assert not controller.ready_event.is_set()
    assert not controller.is_ready


def test_start_wait_accepts_a_healthy_child_ready_within_launch_timeout(
    monkeypatch,
):
    controller, *_ = _construct_controller(monkeypatch, launch_timeout=0.2)
    monkeypatch.setattr(controller, "is_alive", lambda: True)

    def signal_ready():
        time.sleep(0.02)
        controller.ready_event.set()
        controller.startup_event.set()

    signal_thread = threading.Thread(target=signal_ready)
    signal_thread.start()
    try:
        controller.start_wait()
    finally:
        signal_thread.join(timeout=1.0)

    assert not signal_thread.is_alive()
    assert controller.is_ready


def test_schedule_waypoint_uses_explicit_validation_and_health_check(
    constructed_controller, monkeypatch
):
    controller, input_queue, *_ = constructed_controller
    controller.ready_event.set()
    monkeypatch.setattr(controller, "is_alive", lambda: True)

    with pytest.raises(ValueError, match="pose_left"):
        controller.schedule_waypoint(
            pose_left=np.zeros(5),
            pose_right=np.zeros(6),
            gripper_left=np.array([0.02]),
            gripper_right=np.array([0.02]),
            target_time=1.0,
        )
    assert input_queue.puts == []

    controller._record_fatal_error(RuntimeError("child exited"))
    with pytest.raises(RuntimeError, match="child exited"):
        controller.schedule_waypoint(
            pose_left=np.zeros(6),
            pose_right=np.zeros(6),
            gripper_left=np.array([0.02]),
            gripper_right=np.array([0.02]),
            target_time=1.0,
        )
    assert input_queue.puts == []


def test_stop_is_idempotent_and_joins_before_returning(
    constructed_controller, monkeypatch
):
    controller, input_queue, *_ = constructed_controller
    calls = []
    monkeypatch.setattr(controller, "is_alive", lambda: True)
    monkeypatch.setattr(controller, "stop_wait", lambda timeout=None: calls.append(timeout))

    controller.stop(wait=False)
    controller.stop(wait=True)

    assert [message["cmd"] for message in input_queue.puts] == [
        controller_module.Command.STOP.value
    ]
    assert len(calls) == 1
    assert calls[0] is not None
    assert 0.0 <= calls[0] <= controller.shutdown_timeout_s


def test_preexisting_stop_request_prevents_any_fake_robot_target_dispatch(
    constructed_controller, monkeypatch
):
    controller, *_ = constructed_controller
    robot_instances = []

    class FakeRobotControl:
        def __init__(self, **_kwargs):
            self.set_target_calls = 0
            self.execute_calls = 0
            self.stop_calls = 0
            robot_instances.append(self)

        def get_ee_pose(self):
            identity_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
            return {
                "left_arm_ee2rb": identity_pose.copy(),
                "right_arm_ee2rb": identity_pose.copy(),
                "left_gripper": np.array([0.02]),
                "right_gripper": np.array([0.02]),
            }

        def set_target_CP(self, *_args, **_kwargs):
            self.set_target_calls += 1

        def execute(self):
            self.execute_calls += 1

        def stop(self):
            self.stop_calls += 1

    monkeypatch.setattr(controller_module, "RobotControl", FakeRobotControl)
    controller.stop_event.set()

    controller.run()

    assert len(robot_instances) == 1
    assert robot_instances[0].set_target_calls == 0
    assert robot_instances[0].execute_calls == 0
    assert robot_instances[0].stop_calls == 1


def test_stop_linearizes_with_final_dispatch_gate_without_a_toctou_dispatch(
    constructed_controller, monkeypatch
):
    controller, *_ = constructed_controller
    final_check_entered = threading.Event()
    release_final_check = threading.Event()
    stop_returned = threading.Event()
    dispatch_started_after_stop = []
    robot_instances = []

    class PausingStopEvent:
        def __init__(self):
            self._set = False
            self._calls = 0
            self._lock = threading.Lock()

        def clear(self):
            with self._lock:
                self._set = False

        def set(self):
            with self._lock:
                self._set = True

        def is_set(self):
            with self._lock:
                self._calls += 1
                observed = self._set
                call = self._calls
            if call == 2:
                final_check_entered.set()
                assert release_final_check.wait(timeout=2.0)
            return observed

    class FakeRobotControl:
        def __init__(self, **_kwargs):
            self.dispatches = 0
            self.stop_calls = 0
            robot_instances.append(self)

        def get_ee_pose(self):
            identity_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
            return {
                "left_arm_ee2rb": identity_pose.copy(),
                "right_arm_ee2rb": identity_pose.copy(),
                "left_gripper": np.array([0.02]),
                "right_gripper": np.array([0.02]),
            }

        def set_target_CP(self, *_args, **_kwargs):
            dispatch_started_after_stop.append(stop_returned.is_set())
            self.dispatches += 1

        def execute(self):
            pass

        def stop(self):
            self.stop_calls += 1

    controller.stop_event = PausingStopEvent()
    monkeypatch.setattr(controller, "is_alive", lambda: True)
    monkeypatch.setattr(controller_module, "RobotControl", FakeRobotControl)
    runner_errors = []
    runner = threading.Thread(
        target=lambda: _capture_error(controller.run, runner_errors)
    )
    runner.start()
    assert final_check_entered.wait(timeout=2.0)

    stopper = threading.Thread(
        target=lambda: (controller.stop(wait=False), stop_returned.set())
    )
    stopper.start()
    # Old code returns from stop while the runner is paused after its final
    # check. The dispatch lock makes this an in-flight dispatch instead: stop
    # cannot linearize until that one dispatch leaves the critical section.
    stop_returned.wait(timeout=0.1)
    release_final_check.set()

    stopper.join(timeout=2.0)
    runner.join(timeout=2.0)
    assert not stopper.is_alive()
    assert not runner.is_alive()
    assert runner_errors == []
    assert stop_returned.is_set()
    assert dispatch_started_after_stop == [False]
    assert robot_instances[0].dispatches == 1
    assert robot_instances[0].stop_calls == 1


def _capture_error(callback, errors):
    try:
        callback()
    except BaseException as exc:
        errors.append(exc)


def test_dispatch_exception_releases_gate_for_concurrent_stop(
    constructed_controller, monkeypatch
):
    controller, *_ = constructed_controller
    dispatch_entered = threading.Event()
    release_dispatch = threading.Event()
    stop_returned = threading.Event()
    robot_instances = []

    class FailingRobotControl:
        def __init__(self, **_kwargs):
            self.stop_calls = 0
            robot_instances.append(self)

        def get_ee_pose(self):
            identity_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
            return {
                "left_arm_ee2rb": identity_pose.copy(),
                "right_arm_ee2rb": identity_pose.copy(),
                "left_gripper": np.array([0.02]),
                "right_gripper": np.array([0.02]),
            }

        def set_target_CP(self, *_args, **_kwargs):
            dispatch_entered.set()
            assert release_dispatch.wait(timeout=2.0)
            raise RuntimeError("dispatch failed")

        def execute(self):
            raise AssertionError("execute must not follow failed set_target_CP")

        def stop(self):
            self.stop_calls += 1

    monkeypatch.setattr(controller, "is_alive", lambda: True)
    monkeypatch.setattr(controller_module, "RobotControl", FailingRobotControl)
    runner_errors = []
    runner = threading.Thread(
        target=lambda: _capture_error(controller.run, runner_errors)
    )
    runner.start()
    assert dispatch_entered.wait(timeout=2.0)
    stopper = threading.Thread(
        target=lambda: (controller.stop(wait=False), stop_returned.set())
    )
    stopper.start()

    time.sleep(0.02)
    release_dispatch.set()
    stopper.join(timeout=2.0)
    runner.join(timeout=2.0)

    assert not stopper.is_alive()
    assert not runner.is_alive()
    assert stop_returned.is_set()
    assert len(runner_errors) == 1
    assert "dispatch failed" in str(runner_errors[0])
    assert "dispatch failed" in controller.fatal_error_summary
    assert robot_instances[0].stop_calls == 1


def test_concurrent_repeated_stop_enqueues_once_and_never_joins_under_gate(
    constructed_controller, monkeypatch
):
    controller, input_queue, *_ = constructed_controller
    monkeypatch.setattr(controller, "is_alive", lambda: True)
    join_observations = []

    def stop_wait(timeout=None):
        acquired = controller.dispatch_lock.acquire(timeout=0.2)
        join_observations.append((timeout, acquired))
        if acquired:
            controller.dispatch_lock.release()

    monkeypatch.setattr(controller, "stop_wait", stop_wait)
    barrier = threading.Barrier(8)
    threads = [
        threading.Thread(
            target=lambda: (barrier.wait(), controller.stop(wait=True))
        )
        for _ in range(8)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2.0)

    assert all(not thread.is_alive() for thread in threads)
    assert len(input_queue.puts) == 1
    assert len(join_observations) == 8
    assert all(acquired for _timeout, acquired in join_observations)


def test_owner_death_while_holding_dispatch_gate_is_reaped_in_finite_time():
    context = mp.get_context("fork")
    result_queue = context.Queue()
    scenario = context.Process(
        target=_run_owner_death_stop_scenario,
        args=(result_queue,),
    )

    scenario.start()
    scenario.join(timeout=2.0)
    if scenario.is_alive():
        scenario.terminate()
        scenario.join(timeout=1.0)

    assert not scenario.is_alive(), "parent stop hung on an owner-dead dispatch lock"
    result = result_queue.get(timeout=1.0)
    assert result[0] == "ok", result
    assert result[1] < 0.8
    assert result[2] == 23


def test_alive_child_holding_dispatch_gate_has_finite_total_stop_timeout(
    constructed_controller, monkeypatch
):
    controller, *_ = constructed_controller
    monkeypatch.setattr(controller, "is_alive", lambda: True)
    controller.dispatch_lock.acquire()
    started = time.monotonic()
    try:
        with pytest.raises(RuntimeError, match="dispatch gate"):
            controller.stop(wait=False, timeout=0.05)
    finally:
        controller.dispatch_lock.release()

    assert time.monotonic() - started < 0.5


def test_repeated_stop_before_process_start_is_safe_and_does_not_enqueue(
    constructed_controller,
):
    controller, input_queue, *_ = constructed_controller

    controller.stop(wait=True)
    controller.stop(wait=True)

    assert input_queue.puts == []


def test_startup_crash_can_be_stopped_and_reaped_without_gate_deadlock(
    constructed_controller, monkeypatch
):
    controller, input_queue, *_ = constructed_controller

    def crash_during_startup(self):
        error = RuntimeError("synthetic startup crash")
        self._record_fatal_error(error)
        self.startup_event.set()
        raise error

    monkeypatch.setattr(controller_module.Controller, "run", crash_during_startup)
    controller.start(wait=False)
    controller.join(timeout=2.0)
    assert not controller.is_alive()

    controller.stop(wait=True)

    assert controller.exitcode != 0
    assert "synthetic startup crash" in controller.fatal_error_summary
    assert input_queue.puts == []

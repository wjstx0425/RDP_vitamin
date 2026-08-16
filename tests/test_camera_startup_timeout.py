import time

import pytest

from real_world.bimanual_umi_env import BimanualUmiEnv
from real_world.multi_uvc_camera import MultiUvcCamera
from real_world.robot_api.arm.Controller import Controller
from real_world.uvc_camera import UvcCamera
from real_world.video_recorder import VideoRecorder


class FakeEvent:
    def __init__(self, *, ready: bool, delay: float = 0.0) -> None:
        self.ready = ready
        self.delay = delay
        self.wait_timeouts = []

    def wait(self, timeout: float | None = None) -> bool:
        self.wait_timeouts.append(timeout)
        sleep_for = self.delay if timeout is None else min(self.delay, max(timeout, 0.0))
        time.sleep(sleep_for)
        return self.ready and (timeout is None or self.delay <= timeout)


def test_uvc_camera_start_wait_times_out_with_device_details() -> None:
    ready_event = FakeEvent(ready=False)
    camera = UvcCamera.__new__(UvcCamera)
    camera.ready_event = ready_event
    camera.dev_video_path = "/dev/video2"
    camera.is_alive = lambda: True

    started_at = time.monotonic()
    with pytest.raises(TimeoutError) as exc_info:
        camera.start_wait(timeout=0.001)

    assert time.monotonic() - started_at < 0.1
    assert 0.0 <= ready_event.wait_timeouts[0] <= 0.001
    assert "device=/dev/video2" in str(exc_info.value)
    assert "alive=True" in str(exc_info.value)


def test_uvc_camera_start_wait_shares_deadline_with_video_recorder() -> None:
    camera_event = FakeEvent(ready=True, delay=0.006)
    recorder_event = FakeEvent(ready=False, delay=0.05)
    recorder = VideoRecorder.__new__(VideoRecorder)
    recorder.ready_event = recorder_event
    camera = UvcCamera.__new__(UvcCamera)
    camera.ready_event = camera_event
    camera.dev_video_path = "/dev/video2"
    camera.is_alive = lambda: True
    camera.video_recorder = recorder

    started_at = time.monotonic()
    with pytest.raises(TimeoutError) as exc_info:
        camera.start_wait(timeout=0.015)

    assert time.monotonic() - started_at < 0.04
    assert recorder_event.wait_timeouts[0] is not None
    assert 0.0 <= recorder_event.wait_timeouts[0] < 0.015
    assert "device=/dev/video2" in str(exc_info.value)
    assert "alive=True" in str(exc_info.value)


def test_multi_uvc_camera_shares_timeout_in_device_order() -> None:
    calls = []

    class FakeCamera:
        def __init__(self, device: str, *, delay: float = 0.0) -> None:
            self.device = device
            self.delay = delay

        def start_wait(self, timeout: float) -> None:
            calls.append((self.device, timeout))
            time.sleep(min(self.delay, max(timeout, 0.0)))

    cameras = MultiUvcCamera.__new__(MultiUvcCamera)
    cameras.cameras = {
        "/dev/video2": FakeCamera("/dev/video2", delay=0.006),
        "/dev/video0": FakeCamera("/dev/video0"),
    }

    cameras.start_wait(timeout=0.015)

    assert [device for device, _ in calls] == ["/dev/video2", "/dev/video0"]
    assert 0.0 <= calls[1][1] < calls[0][1] <= 0.015


def test_multi_uvc_stop_wait_does_not_join_unstarted_processes() -> None:
    joins = []

    class FakeProcess:
        def __init__(self, device: str, *, alive: bool) -> None:
            self.device = device
            self.alive = alive

        def is_alive(self) -> bool:
            return self.alive

        def join(self) -> None:
            if not self.alive:
                raise AssertionError("an unstarted process must not be joined")
            joins.append(self.device)

        def end_wait(self, timeout: float | None = None) -> None:
            self.join()

    cameras = MultiUvcCamera.__new__(MultiUvcCamera)
    cameras.cameras = {
        "/dev/video2": FakeProcess("/dev/video2", alive=False),
        "/dev/video0": FakeProcess("/dev/video0", alive=True),
    }

    cameras.stop_wait()

    assert joins == ["/dev/video0"]


def test_video_recorder_end_wait_uses_finite_join_timeout() -> None:
    join_timeouts = []
    recorder = VideoRecorder.__new__(VideoRecorder)
    recorder.join = lambda timeout=None: join_timeouts.append(timeout)
    recorder.is_alive = lambda: True

    with pytest.raises(TimeoutError):
        recorder.end_wait(timeout=0.01)

    assert join_timeouts == [0.01]


def test_uvc_camera_end_wait_shares_deadline_with_recorder() -> None:
    calls = []

    class FakeRecorder:
        def end_wait(self, timeout: float | None = None) -> None:
            calls.append(("recorder", timeout))

    camera = UvcCamera.__new__(UvcCamera)

    def join(timeout=None) -> None:
        calls.append(("camera", timeout))
        time.sleep(0.006)

    camera.join = join
    camera.is_alive = lambda: False
    camera.dev_video_path = "/dev/video2"
    camera.video_recorder = FakeRecorder()

    camera.end_wait(timeout=0.015)

    assert calls[0] == ("camera", pytest.approx(0.015, abs=0.003))
    assert calls[1][0] == "recorder"
    assert calls[1][1] is not None
    assert 0.0 <= calls[1][1] < calls[0][1]


def test_multi_uvc_stop_wait_shares_finite_deadline() -> None:
    calls = []

    class FakeCamera:
        pid = 123

        def __init__(self, device: str, *, delay: float = 0.0) -> None:
            self.device = device
            self.delay = delay

        def is_alive(self) -> bool:
            return True

        def end_wait(self, timeout: float | None = None) -> None:
            calls.append((self.device, timeout))
            time.sleep(min(self.delay, timeout or self.delay))

    cameras = MultiUvcCamera.__new__(MultiUvcCamera)
    cameras.cameras = {
        "/dev/video2": FakeCamera("/dev/video2", delay=0.006),
        "/dev/video0": FakeCamera("/dev/video0"),
    }

    cameras.stop_wait(timeout=0.015)

    assert [device for device, _ in calls] == ["/dev/video2", "/dev/video0"]
    assert calls[0][1] is not None
    assert calls[1][1] is not None
    assert 0.0 <= calls[1][1] < calls[0][1] <= 0.015


def test_controller_stop_wait_uses_finite_join_timeout() -> None:
    join_timeouts = []
    controller = Controller.__new__(Controller)
    controller.join = lambda timeout=None: join_timeouts.append(timeout)
    controller.is_alive = lambda: True

    with pytest.raises(TimeoutError):
        controller.stop_wait(timeout=0.01)

    assert join_timeouts == [0.01]


def test_bimanual_stop_controller_waits_without_stopping_camera() -> None:
    calls = []

    class FakeController:
        def stop(self, *, wait: bool) -> None:
            calls.append(("controller", "stop", wait))

    class FakeCamera:
        def stop(self, *, wait: bool) -> None:
            calls.append(("camera", "stop", wait))

    env = BimanualUmiEnv.__new__(BimanualUmiEnv)
    env.controller = FakeController()
    env.camera = FakeCamera()

    env.stop_controller(wait=True)

    assert calls == [("controller", "stop", True)]


def test_bimanual_stop_failure_is_not_retried_during_context_cleanup() -> None:
    calls = []

    class FakeController:
        def stop(self, *, wait: bool, timeout=None) -> None:
            calls.append(("controller", "stop", wait, timeout))
            raise TimeoutError("controller dispatch gate did not drain")

        def is_alive(self) -> bool:
            return True

        def stop_wait(self, timeout=None) -> None:
            raise AssertionError("failed Controller stop must not be retried")

    class FakeCamera:
        def stop(self, *, wait: bool) -> None:
            calls.append(("camera", "stop", wait))

        def is_alive(self) -> bool:
            return True

        def stop_wait(self, timeout=None) -> None:
            calls.append(("camera", "stop_wait", timeout))

    env = BimanualUmiEnv.__new__(BimanualUmiEnv)
    env.controller = FakeController()
    env.camera = FakeCamera()

    with pytest.raises(TimeoutError, match="dispatch gate"):
        env.stop_controller(wait=True)
    env.stop(wait=True)

    assert [call for call in calls if call[:2] == ("controller", "stop")] == [
        ("controller", "stop", True, None)
    ]
    assert ("camera", "stop", False) in calls
    assert any(call[:2] == ("camera", "stop_wait") for call in calls)


def test_bimanual_start_cleans_up_both_groups_and_preserves_timeout() -> None:
    calls = []

    class FakeCameraGroup:
        def start(self, *, wait: bool) -> None:
            calls.append(("camera", "start", wait))

        def start_wait(self) -> None:
            raise TimeoutError("camera readiness failed")

        def stop(self, *, wait: bool) -> None:
            calls.append(("camera", "stop", wait))

        def stop_wait(self, timeout: float | None = None) -> None:
            calls.append(("camera", "stop_wait"))
            raise RuntimeError("cleanup failed")

    class FakeControllerGroup:
        def start(self, *, wait: bool) -> None:
            calls.append(("controller", "start", wait))

        def stop(self, *, wait: bool) -> None:
            calls.append(("controller", "stop", wait))

        def is_alive(self) -> bool:
            return False

        def stop_wait(self, timeout: float | None = None) -> None:
            calls.append(("controller", "stop_wait"))
            raise AssertionError("an unstarted process must not be joined")

    env = BimanualUmiEnv.__new__(BimanualUmiEnv)
    env.camera = FakeCameraGroup()
    env.controller = FakeControllerGroup()

    with pytest.raises(TimeoutError, match="camera readiness failed"):
        env.start()

    assert ("camera", "stop", False) in calls
    assert ("controller", "stop", False) in calls
    assert ("camera", "stop_wait") in calls
    assert ("controller", "stop_wait") not in calls


@pytest.mark.parametrize("failing_child", ["camera", "controller"])
def test_bimanual_start_cleans_up_when_child_start_raises(failing_child) -> None:
    calls = []

    class FakeChildGroup:
        def __init__(self, name: str):
            self.name = name

        def start(self, *, wait: bool) -> None:
            calls.append((self.name, "start", wait))
            if self.name == failing_child:
                raise RuntimeError(f"{self.name} start failed")

        def stop(self, *, wait: bool) -> None:
            calls.append((self.name, "stop", wait))

        def is_alive(self) -> bool:
            return False

        def stop_wait(self, timeout: float | None = None) -> None:
            raise AssertionError("inactive child must not be joined")

    env = BimanualUmiEnv.__new__(BimanualUmiEnv)
    env.camera = FakeChildGroup("camera")
    env.controller = FakeChildGroup("controller")
    env.STARTUP_FAILURE_CLEANUP_TIMEOUT = 0.01

    with pytest.raises(RuntimeError, match=f"{failing_child} start failed"):
        env.start()

    assert ("camera", "stop", False) in calls
    assert ("controller", "stop", False) in calls


def test_bimanual_start_bounds_cleanup_and_still_attempts_controller() -> None:
    stop_wait_calls = []

    class FakeCameraGroup:
        def start(self, *, wait: bool) -> None:
            pass

        def start_wait(self) -> None:
            raise TimeoutError("original startup timeout")

        def stop(self, *, wait: bool) -> None:
            pass

        def stop_wait(self, timeout: float | None = None) -> None:
            stop_wait_calls.append(("camera", timeout))
            time.sleep(0.05 if timeout is None else max(timeout, 0.0))

    class FakeControllerGroup:
        def start(self, *, wait: bool) -> None:
            pass

        def stop(self, *, wait: bool) -> None:
            pass

        def is_alive(self) -> bool:
            return True

        def stop_wait(self, timeout: float | None = None) -> None:
            stop_wait_calls.append(("controller", timeout))
            time.sleep(0.05 if timeout is None else max(timeout, 0.0))

    env = BimanualUmiEnv.__new__(BimanualUmiEnv)
    env.camera = FakeCameraGroup()
    env.controller = FakeControllerGroup()
    env.STARTUP_FAILURE_CLEANUP_TIMEOUT = 0.01

    started_at = time.monotonic()
    with pytest.raises(TimeoutError, match="original startup timeout"):
        env.start()

    assert time.monotonic() - started_at < 0.04
    assert [name for name, _ in stop_wait_calls] == ["camera", "controller"]
    assert all(timeout is not None for _, timeout in stop_wait_calls)

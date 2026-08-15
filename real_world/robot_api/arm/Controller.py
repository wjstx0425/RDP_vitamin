import gc
import sys
import os
dir = os.getcwd()
sys.path.append(dir)

import time
import enum
import multiprocessing as mp
import ctypes
from multiprocessing.managers import SharedMemoryManager

import numpy as np
# from scipy.spatial.transform import Rotation

from utils.shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from utils.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from real_world.robot_api.arm.RobotControl_pykin import RobotControl

from utils.pose_trajectory_interpolator import PoseTrajectoryInterpolator
from utils.pose_util import mat_to_pose, pose_to_mat, pose_to_pos_quat, pos_quat_to_pose
from utils.precise_sleep import precise_wait

COMMAND_QUEUE_CAPACITY = 512
DEBUG_QUEUE_CAPACITY = 4096

class CustomError(Exception):
    def __init__(self, message):
        self.message = message


class ControllerProcessError(RuntimeError):
    """Raised in the parent when the controller child is not healthy."""

# 用于控制机器人状态的指令类，包括停机、行动和 SCHEDULE_WAYPOINT（？）三种状态
class Command(enum.Enum):
    STOP = 0
    SERVOL = 1
    SCHEDULE_WAYPOINT = 2

def interpolate_gripper_target(current, target, max_speed: float, dt: float) -> np.ndarray:
    """Move gripper command toward target with a per-cycle speed limit."""
    current = np.asarray(current, dtype=np.float64).reshape(1)
    target = np.asarray(target, dtype=np.float64).reshape(1)
    max_delta = max(float(max_speed), 0.0) * float(dt)
    if max_delta == 0.0:
        return current.copy()
    return current + np.clip(target - current, -max_delta, max_delta)


class Controller(mp.Process):
    def __init__(self,
                shm_manager: SharedMemoryManager, # 多进程控制
                launch_timeout = 3,
                verbose = False,

                frequency : int = 100,
                get_max_k : int = None,
                max_pos_speed : float = 0.25,
                max_rot_speed : float = 0.16,
                max_gripper_speed : float = 0.01,
                receive_latency : float = 0.0,
                single_arm_mode: bool = False,
                shutdown_timeout_s: float = 3.0,
                ):

        super().__init__(name="arm_controller") # 直接调用父类 mp.Process 的初始化函数，初始化该进程

        # 进程参数初始化
        self.verbose = verbose # 用来控制是否输出调试信息的变量
        try:
            self.launch_timeout = float(launch_timeout)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "launch_timeout must be finite and positive"
            ) from exc
        if not np.isfinite(self.launch_timeout) or self.launch_timeout <= 0.0:
            raise ValueError("launch_timeout must be finite and positive")
        self.single_arm_mode = single_arm_mode
        try:
            self.shutdown_timeout_s = float(shutdown_timeout_s)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "shutdown_timeout_s must be finite and positive"
            ) from exc
        if not np.isfinite(self.shutdown_timeout_s) or self.shutdown_timeout_s <= 0:
            raise ValueError("shutdown_timeout_s must be finite and positive")
        self._dispatch_poll_interval_s = min(0.05, self.shutdown_timeout_s)

        # 控制参数初始化
        self.frequency = frequency
        if get_max_k is None:
            get_max_k = int(frequency * 5)
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self.max_gripper_speed = max_gripper_speed
        self.receive_latency = receive_latency

        # 创建用于储存控制信息的队列，控制信息格式如example所示
        example = {
            'cmd': Command.SERVOL.value,
            'target_pose_left': np.zeros((6,), dtype=np.float64),
            'target_pose_right': np.zeros((6,), dtype=np.float64),
            'target_gripper_left': np.zeros((1,), dtype=np.float64),
            'target_gripper_right': np.zeros((1,), dtype=np.float64),
            'duration': 0.0,
            'target_time': 0.0
        }

        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=COMMAND_QUEUE_CAPACITY,
        )

        # 创建用于储存反馈信息的队列，反馈信息内容及格式如example所示
        example = {
            'ee_pose_left': np.zeros((6,), dtype=np.float64),
            'ee_pose_right': np.zeros((6,), dtype=np.float64),
            'gripper_pose_left': np.zeros((1,), dtype=np.float64),
            'gripper_pose_right': np.zeros((1,), dtype=np.float64),
            'robot_receive_timestamp': time.time(),
            'robot_timestamp': time.time()
        }

        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )

        # DEBUG BUFFER
        if not self.single_arm_mode:
            self.side = ["left", "right"]
        else:
            self.side = ["left"]
        self.para = ["x", "y", "z", "rx", "ry", "rz", "g"]
        example = dict()
        for side in self.side:
            for para in self.para:
                example[f"ee_pose_{side}_{para}"] = 0.0
                example[f"target_pose_{side}_{para}"] = 0.0
        example["time"] = 0.0

        self.input_queue_debug = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=DEBUG_QUEUE_CAPACITY,
        )

        # Pre-compute debug keys to avoid string formatting in the hot loop
        self._debug_keys_per_side = {
            side: [(f"ee_pose_{side}_{para}", f"target_pose_{side}_{para}") for para in self.para]
            for side in self.side
        }
        self._debug_message = {"time": 0.0}
        for side in self.side:
            for ee_key, tgt_key in self._debug_keys_per_side[side]:
                self._debug_message[ee_key] = 0.0
                self._debug_message[tgt_key] = 0.0

        # 一些变量赋值
        self.ready_event = mp.Event()
        self.startup_event = mp.Event()
        self.stop_event = mp.Event()
        self.dispatch_lock = mp.Lock()
        # Parent stop callers serialize here. The child never acquires this
        # lock, so child owner-death cannot poison stop-call coordination.
        self._stop_lock = mp.Lock()
        self._fatal_error = mp.Array(ctypes.c_char, 4096, lock=True)
        self._stop_requested = False
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer
        self.single_arm_mode = single_arm_mode

    '''===进程控制==='''

    # 进程控制函数（开始/结束）
    def start(self, wait=True):
        self._stop_requested = False
        self.stop_event.clear()
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[ArmController] Controller process spawned at {self.pid}")

    def _child_state(self) -> tuple[bool, bool, int | None]:
        try:
            child_alive = bool(self.is_alive())
        except (AssertionError, ValueError):
            child_alive = False
        try:
            child_pid = self.pid
        except (AssertionError, ValueError):
            child_pid = None
        started = child_pid is not None or child_alive
        try:
            child_exitcode = self.exitcode
        except (AssertionError, ValueError):
            child_exitcode = None
        return started, child_alive, child_exitcode

    def _drain_dispatch_gate(self, deadline: float, timeout_s: float) -> bool:
        """Drain one in-flight dispatch or detect an owner-dead child.

        Returns ``True`` after acquiring and releasing the gate. Returns
        ``False`` when the child has exited and may have poisoned the gate.
        """
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                _started, child_alive, child_exitcode = self._child_state()
                if not child_alive:
                    return False
                raise ControllerProcessError(
                    "Controller dispatch gate did not drain within "
                    f"{timeout_s}s while child remained alive "
                    f"(exitcode={child_exitcode})"
                )
            acquired = self.dispatch_lock.acquire(
                timeout=min(self._dispatch_poll_interval_s, remaining)
            )
            if acquired:
                self.dispatch_lock.release()
                return True
            _started, child_alive, _exitcode = self._child_state()
            if not child_alive:
                return False

    def stop(self, wait=True, timeout: float | None = None):
        try:
            checked_timeout = (
                self.shutdown_timeout_s if timeout is None else float(timeout)
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("stop timeout must be finite and positive") from exc
        if not np.isfinite(checked_timeout) or checked_timeout <= 0.0:
            raise ValueError("stop timeout must be finite and positive")
        deadline = time.monotonic() + checked_timeout

        # Repeated parent callers wait for the same drain/reap result instead
        # of observing _stop_requested and returning ahead of the first caller.
        with self._stop_lock:
            enqueue_stop = False
            if not self._stop_requested:
                self._stop_requested = True
                self.stop_event.set()
                enqueue_stop = True

            started, child_alive, _exitcode = self._child_state()
            if not started:
                return

            # Queue operations are intentionally outside dispatch_lock.
            if enqueue_stop and child_alive:
                self.input_queue.put({'cmd': Command.STOP.value})

            self._drain_dispatch_gate(deadline, checked_timeout)

            if wait:
                remaining = max(0.0, deadline - time.monotonic())
                self.stop_wait(timeout=remaining)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # 进程确认函数（等待本进程启动/等待子进程结束）
    def start_wait(self):
        signaled = self.startup_event.wait(self.launch_timeout)
        if not signaled:
            self.check_health(require_ready=False)
            raise TimeoutError(
                f"Controller did not become ready within {self.launch_timeout}s"
            )
        self.check_health(require_ready=True)

    def stop_wait(self, timeout: float | None = None) -> None:
        try:
            child_alive = bool(self.is_alive())
        except (AssertionError, ValueError):
            child_alive = False
        if not child_alive and getattr(self, "_popen", None) is None:
            return
        self.join(timeout)
        if timeout is not None and self.is_alive():
            raise TimeoutError(f"Controller shutdown timed out after {timeout}s")

    # 返回该进程是否已准备好
    @property
    def is_ready(self):
        if not self.ready_event.is_set() or self.fatal_error_summary:
            return False
        try:
            return bool(self.is_alive()) and self.exitcode is None
        except (AssertionError, ValueError):
            return False

    @property
    def fatal_error_summary(self) -> str:
        with self._fatal_error.get_lock():
            encoded = bytes(self._fatal_error.get_obj()).split(b"\0", 1)[0]
        return encoded.decode("utf-8", errors="replace")

    def _record_fatal_error(self, error: BaseException) -> None:
        summary = f"{type(error).__name__}: {error}"
        encoded = summary.encode("utf-8", errors="replace")[:4095]
        with self._fatal_error.get_lock():
            self._fatal_error.get_obj().value = encoded

    def check_health(self, *, require_ready: bool = True) -> None:
        fatal = self.fatal_error_summary
        if fatal:
            raise ControllerProcessError(
                f"Controller child failed: {fatal}"
            )
        try:
            alive = bool(self.is_alive())
        except (AssertionError, ValueError):
            alive = False
        if not alive:
            exitcode = self.exitcode
            if self.pid is None:
                detail = "has not been started"
            else:
                detail = f"exited with code {exitcode}"
            raise ControllerProcessError(f"Controller child {detail}")
        if require_ready and not self.ready_event.is_set():
            raise ControllerProcessError("Controller child is not ready")

    '''===主要功能API：控制机器人运动==='''

    # 安排经过点？
    def schedule_waypoint(self, pose_left:list, pose_right:list, gripper_left:list, gripper_right:list, target_time:float):
        self.check_health()
        pose_left = np.asarray(pose_left, dtype=np.float64)
        if pose_left.shape != (6,) or not np.all(np.isfinite(pose_left)):
            raise ValueError("pose_left must be a finite array with shape (6,)")
        gripper_left = np.asarray(gripper_left, dtype=np.float64)
        if gripper_left.shape != (1,) or not np.all(np.isfinite(gripper_left)):
            raise ValueError("gripper_left must be a finite array with shape (1,)")

        if not self.single_arm_mode:
            pose_right = np.asarray(pose_right, dtype=np.float64)
            if pose_right.shape != (6,) or not np.all(np.isfinite(pose_right)):
                raise ValueError("pose_right must be a finite array with shape (6,)")
            gripper_right = np.asarray(gripper_right, dtype=np.float64)
            if gripper_right.shape != (1,) or not np.all(np.isfinite(gripper_right)):
                raise ValueError(
                    "gripper_right must be a finite array with shape (1,)"
                )
        else:
            # only for occupancy check, never used
            pose_right = np.zeros((6,))
            gripper_right = np.zeros((1,))

        checked_target_time = float(target_time)
        if not np.isfinite(checked_target_time):
            raise ValueError("target_time must be finite")
        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pose_left': pose_left,
            'target_pose_right': pose_right,
            'target_gripper_left': gripper_left,
            'target_gripper_right': gripper_right,
            'target_time': checked_target_time
        }

        self.input_queue.put(message)

    def renew_debug_buffer(self, message):
        self.input_queue_debug.put(message)

    '''===主要功能API：获取机器人状态；状态将在控制主循环中上传==='''

    def get_state(self, k=None, out=None):
        self.check_health()
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k,out=out)

    def get_all_state(self):
        self.check_health()
        return self.ring_buffer.get_all()

    def get_debug_info(self):
        # Debug data remains drainable after an intentional stop/join so slow
        # plotting can happen only after robot motion has ceased.
        return self.input_queue_debug.get_all()

    def run(self):
        robot_control = None
        try:
            robot_control = RobotControl()
            if self.verbose:
                print(f"[PositionalController] Connect to robot")

            dt_controller = 1. / self.frequency

            '''INITIALIZE'''
            # initialize time and last waypoint time
            curr_t = time.monotonic()
            last_waypoint_time = curr_t
            curr_ee_pose = robot_control.get_ee_pose()

            # get initial ee pose and gripper pose
            curr_ee_pos_quat_left = curr_ee_pose["left_arm_ee2rb"]
            curr_ee_pose_left = pos_quat_to_pose(curr_ee_pos_quat_left[:3], curr_ee_pos_quat_left[3:])
            target_gripper_left = curr_ee_pose["left_gripper"] # pre-set gripper pose, because we won't use interpolator for grippers

            # initialize pose interpolator
            pose_interp_left = PoseTrajectoryInterpolator(
                times=[curr_t],
                poses=[curr_ee_pose_left]
            )

            if not self.single_arm_mode:
                curr_ee_pos_quat_right = curr_ee_pose["right_arm_ee2rb"]
                curr_ee_pose_right = pos_quat_to_pose(curr_ee_pos_quat_right[:3], curr_ee_pos_quat_right[3:])
                target_gripper_right = curr_ee_pose["right_gripper"]
                pose_interp_right = PoseTrajectoryInterpolator(
                times=[curr_t],
                poses=[curr_ee_pose_right]
            )
            else:
                curr_ee_pos_quat_right = None
                curr_ee_pose_right = None
                target_gripper_right = None
                pose_interp_right = None

            '''MAIN LOOP'''
            t_start = time.monotonic()
            iter_idx = 0
            keep_running = True
            cmd_pose = None
            gripper_goal_left = np.asarray(target_gripper_left, dtype=np.float64).reshape(1)
            gripper_goal_right = (
                np.asarray(target_gripper_right, dtype=np.float64).reshape(1)
                if target_gripper_right is not None else None
            )
            gc_interval = self.frequency  # collect once per second
            gc.disable()

            while keep_running and not self.stop_event.is_set():
                t_now = time.monotonic()

                # DEBUG
                # get current ee pose and gripper pose
                curr_ee_pose = robot_control.get_ee_pose()

                curr_ee_pos_quat_left = curr_ee_pose["left_arm_ee2rb"]
                curr_gripper_left = curr_ee_pose["left_gripper"]
                curr_ee_pose_left = pos_quat_to_pose(curr_ee_pos_quat_left[:3], curr_ee_pos_quat_left[3:])
                if not self.single_arm_mode:
                    curr_ee_pos_quat_right = curr_ee_pose["right_arm_ee2rb"]
                    curr_gripper_right = curr_ee_pose["right_gripper"]
                    curr_ee_pose_right = pos_quat_to_pose(curr_ee_pos_quat_right[:3], curr_ee_pos_quat_right[3:])
                else:
                    curr_ee_pos_quat_right = None
                    curr_gripper_right = None
                    curr_ee_pose_right = None

                # update robot state to ring buffer
                t_recv = time.time()
                state = {
                    'ee_pose_left': curr_ee_pose_left,
                    'ee_pose_right': curr_ee_pose_right,
                    'gripper_pose_left': curr_gripper_left,
                    'gripper_pose_right': curr_gripper_right,
                    'robot_receive_timestamp': t_recv,
                    'robot_timestamp': t_recv - self.receive_latency,
                }
                self.ring_buffer.put(state)

                # get target ee mat (cache interpolation result for reuse in debug)
                target_pose_interp_left = pose_interp_left(t_now)
                if not self.single_arm_mode:
                    target_pose_interp_right = pose_interp_right(t_now)
                else:
                    target_pose_interp_right = None

                if cmd_pose is not None:
                    gripper_goal_left = np.asarray(cmd_pose["left_gripper"], dtype=np.float64).reshape(1)
                    target_gripper_left = interpolate_gripper_target(
                        target_gripper_left,
                        gripper_goal_left,
                        self.max_gripper_speed,
                        dt_controller,
                    )
                    if not self.single_arm_mode:
                        gripper_goal_right = np.asarray(cmd_pose["right_gripper"], dtype=np.float64).reshape(1)
                        target_gripper_right = interpolate_gripper_target(
                            target_gripper_right,
                            gripper_goal_right,
                            self.max_gripper_speed,
                            dt_controller,
                        )
                    else:
                        target_gripper_right = None

                target_ee_pos_quat_left = pose_to_pos_quat(target_pose_interp_left)
                if not self.single_arm_mode:
                    target_ee_pos_quat_right = pose_to_pos_quat(
                        target_pose_interp_right
                    )
                else:
                    target_ee_pos_quat_right = None
                    target_gripper_right = None

                # set target pose and execute robot
                target_pose = {
                    "left_arm_ee2rb": target_ee_pos_quat_left,
                    "right_arm_ee2rb": target_ee_pos_quat_right,
                    "left_gripper": target_gripper_left,
                    "right_gripper": target_gripper_right,
                }

                with self.dispatch_lock:
                    if self.stop_event.is_set():
                        break
                    robot_control.set_target_CP(
                        target_pose, single_arm_mode=self.single_arm_mode
                    )
                    robot_control.execute()

                # DEBUG: record every controller cycle so the plotted trace covers
                # the whole deploy without hidden downsampling.
                for side in self.side:
                    if side == "left":
                        _ee_pose = curr_ee_pose_left
                        _gripper = curr_gripper_left
                        _interp_result = target_pose_interp_left
                        _gripper_target = target_gripper_left
                    else:
                        _ee_pose = curr_ee_pose_right
                        _gripper = curr_gripper_right
                        _interp_result = target_pose_interp_right
                        _gripper_target = target_gripper_right

                    keys = self._debug_keys_per_side[side]
                    for para_idx, para in enumerate(self.para):
                        ee_key, tgt_key = keys[para_idx]
                        if para == "g":
                            self._debug_message[ee_key] = _gripper
                            self._debug_message[tgt_key] = _gripper_target
                        else:
                            self._debug_message[ee_key] = _ee_pose[para_idx]
                            self._debug_message[tgt_key] = _interp_result[para_idx]

                self._debug_message["time"] = t_now - t_start
                self.renew_debug_buffer(self._debug_message)

                # Get a command from input queue. The period of getting command is dt_controller
                try:
                    commands = self.input_queue.get_all()
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0
                except Exception:
                    n_cmd = 0

                # If commands are received, put it into the interpolator
                for i in range(n_cmd):
                    command = {key: value[i] for key, value in commands.items()}
                    cmd = command['cmd']

                    if cmd == Command.SCHEDULE_WAYPOINT.value:
                        cmd_pose = {
                            "left_arm_ee2rb": command['target_pose_left'],
                            "right_arm_ee2rb": command['target_pose_right'],
                            "left_gripper": command['target_gripper_left'],
                            "right_gripper": command['target_gripper_right'],
                        }

                        # The timestamp of the received single frame target action is global time
                        target_time = float(command['target_time'])

                        # Convert global time to monotonic time, for subsequent interpolation
                        target_time = time.monotonic() - time.time() + target_time
                        # The time at the end of the loop (start time + single loop duration),
                        curr_time = t_now + dt_controller

                        if target_time <= curr_time:
                            print("[controller] action is too late")
                        else:
                            pass
                            # print("[controller] target_time, curr_time:", target_time, curr_time)
                            # print("[controller] time :", target_time - curr_time)

                        # Interpolator: If the target time is behind the current time, no action is executed
                        pose_interp_left = pose_interp_left.schedule_waypoint(
                            pose=cmd_pose["left_arm_ee2rb"],
                            time=target_time,
                            max_pos_speed=self.max_pos_speed,
                            max_rot_speed=self.max_rot_speed,
                            curr_time=curr_time,
                            last_waypoint_time=last_waypoint_time,
                        )
                        if not self.single_arm_mode:
                            pose_interp_right = pose_interp_right.schedule_waypoint(
                                pose=cmd_pose["right_arm_ee2rb"],
                                time=target_time, #ACTION TIME
                            max_pos_speed=self.max_pos_speed,
                            max_rot_speed=self.max_rot_speed,
                                curr_time=curr_time, # CURRENT TIME
                                last_waypoint_time=last_waypoint_time,
                            )
                        # Update the latest target time
                        last_waypoint_time = target_time

                    else:
                        keep_running = False
                        break

                # regulate frequency with absolute time grid + precise sleep
                t_cycle_end = t_start + (iter_idx + 1) * dt_controller
                if time.monotonic() < t_cycle_end:
                    precise_wait(t_cycle_end)
                else:
                    # print("[controller] loop speed error, please slow down the controller frequency")
                    t_start = time.monotonic() - (iter_idx + 1) * dt_controller

                # first loop successful, ready to receive command
                if iter_idx == 0:
                    self.ready_event.set()
                    self.startup_event.set()
                iter_idx += 1

                # deterministic GC: collect during the sleep window to avoid random pauses
                if iter_idx % gc_interval == 0:
                    gc.collect()

        except BaseException as e:
            self._record_fatal_error(e)
            self.startup_event.set()
            print(f"Exception occurred: {e}")
            raise
        finally:
            gc.enable()
            print('\n\n\n\nterminate_current_policy\n\n\n\n\n')

            if robot_control is not None:
                robot_control.stop()
                del robot_control
            self.startup_event.set()

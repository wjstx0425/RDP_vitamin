from typing import Optional, Callable, Dict
import enum
import time
import cv2
import numpy as np
import multiprocessing as mp
from threadpoolctl import threadpool_limits
from multiprocessing.managers import SharedMemoryManager
from configs.camera_config import DEFAULT_CAMERA_CONFIG
from utils.timestamp_accumulator import get_accumulate_timestamp_idxs
from utils.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from utils.shared_memory.shared_memory_queue import SharedMemoryQueue, Full, Empty
from real_world.video_recorder import VideoRecorder
from utils.camera_device import V4L2Camera

class Command(enum.Enum):
    RESTART_PUT = 0
    START_RECORDING = 1
    STOP_RECORDING = 2

class UvcCamera(mp.Process):
    """
    Call policy.common.usb_util.reset_all_elgato_devices
    if you are using Elgato capture cards.
    Required to workaround firmware bugs.
    """
    MAX_PATH_LENGTH = 4096 # linux path has a limit of 4096 bytes
    
    def __init__(
            self,
            shm_manager: SharedMemoryManager,
            dev_video_path,
            resolution=(1280, 720),
            capture_fps=20,
            put_fps=None,
            put_downsample=True,
            get_max_k=30,
            receive_latency=0.0,
            cap_buffer_size=1,
            num_threads=2,
            transform: Optional[Callable[[Dict], Dict]] = None,
            vis_transform: Optional[Callable[[Dict], Dict]] = None,
            recording_transform: Optional[Callable[[Dict], Dict]] = None,
            video_recorder: Optional[VideoRecorder] = None,
            verbose=False,
            # Camera parameters for V4L2Camera
            camera_format: str = DEFAULT_CAMERA_CONFIG.pixel_format,
            auto_exposure: int = DEFAULT_CAMERA_CONFIG.auto_exposure,  # 3=manual, 1=auto
            exposure: int | None = DEFAULT_CAMERA_CONFIG.exposure,
            auto_white_balance: int = DEFAULT_CAMERA_CONFIG.auto_white_balance,  # 0=manual, 1=auto
            wb_temperature: int | None = DEFAULT_CAMERA_CONFIG.white_balance_temperature,
            brightness: int = DEFAULT_CAMERA_CONFIG.brightness,
            gain: int = DEFAULT_CAMERA_CONFIG.gain,
            capture_timestamp_delay: float = DEFAULT_CAMERA_CONFIG.capture_timestamp_delay,
            # tactile point cloud params (保留但不使用)
            enable_tactile_pc: bool = False,
            fps_num_points: int = 256,
            tactile_lower_bound: int = 10,
            gamma: int = DEFAULT_CAMERA_CONFIG.gamma,
        ):
        super().__init__()

        if put_fps is None:
            put_fps = capture_fps
        
        # create ring buffer
        resolution = tuple(resolution)
        shape = resolution[::-1]
        examples = {
            'color': np.empty(
                shape=shape+(3,), dtype=np.uint8)
        }
        examples['camera_capture_timestamp'] = 0.0
        examples['camera_receive_timestamp'] = 0.0
        examples['timestamp'] = 0.0
        examples['step_idx'] = 0
        
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=examples if transform is None
                else transform(dict(examples)),
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=put_fps
        )

        # create command queue
        examples = {
            'cmd': Command.RESTART_PUT.value,
            'put_start_time': 0.0,
            'video_path': np.array('a'*self.MAX_PATH_LENGTH),
            'recording_start_time': 0.0,
        }

        command_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=examples,
            buffer_size=1024
        )

        # create video recorder
        if video_recorder is None:
            # default to nvenc GPU encoder
            video_recorder = VideoRecorder.create_hevc_nvenc(
                shm_manager=shm_manager,
                fps=capture_fps, 
                input_pix_fmt='bgr24', 
                bit_rate=6000*1000)
        assert video_recorder.fps == capture_fps

        # copied variables
        self.shm_manager = shm_manager
        self.dev_video_path = dev_video_path
        self.resolution = resolution
        self.capture_fps = capture_fps
        self.put_fps = put_fps
        self.put_downsample = put_downsample
        self.receive_latency = receive_latency
        self.cap_buffer_size = cap_buffer_size
        self.transform = transform
        self.vis_transform = vis_transform
        self.recording_transform = recording_transform
        self.video_recorder = video_recorder
        self.verbose = verbose
        self.put_start_time = None
        self.num_threads = num_threads
        # Camera parameters
        self.camera_format = camera_format
        self.auto_exposure = auto_exposure
        self.exposure = exposure
        self.auto_white_balance = auto_white_balance
        self.wb_temperature = wb_temperature
        self.brightness = brightness
        self.gain = gain
        self.gamma = gamma
        self.capture_timestamp_delay = capture_timestamp_delay
        # tactile point cloud params (保留但不使用)
        self.enable_tactile_pc = enable_tactile_pc
        self.fps_num_points = fps_num_points
        self.tactile_lower_bound = tactile_lower_bound
        # Create independent calibration state for each camera instance
        self.tactile_calibration_state = {
            'is_calibrated': False,
            'x_center_err': 0.0,
            'y_center_err': 0.0
        }

        # shared variables
        self.stop_event = mp.Event()
        self.ready_event = mp.Event()
        self.ring_buffer = ring_buffer
        self.command_queue = command_queue

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= user API ===========
    def start(self, wait=True, put_start_time=None):
        self.put_start_time = put_start_time
        shape = self.resolution[::-1]
        data_example = np.empty(shape=shape+(3,), dtype=np.uint8)
        self.video_recorder.start(
            shm_manager=self.shm_manager, 
            data_example=data_example)
        # must start video recorder first to create share memories
        super().start()
        if wait:
            self.start_wait()
    
    def stop(self, wait=True):
        self.video_recorder.stop()
        self.stop_event.set()
        if wait:
            self.end_wait()

    def start_wait(self, timeout: float = 15.0) -> None:
        deadline = time.monotonic() + max(timeout, 0.0)
        if not self.ready_event.wait(max(0.0, deadline - time.monotonic())):
            raise TimeoutError(
                f"Camera startup timed out after {timeout}s: "
                f"device={self.dev_video_path}, alive={self.is_alive()}"
            )
        try:
            self.video_recorder.start_wait(timeout=max(0.0, deadline - time.monotonic()))
        except TimeoutError as exc:
            raise TimeoutError(
                f"Camera startup timed out after {timeout}s: "
                f"device={self.dev_video_path}, alive={self.is_alive()}"
            ) from exc
    
    def end_wait(self, timeout: Optional[float] = None) -> None:
        deadline = None if timeout is None else time.monotonic() + max(timeout, 0.0)
        error = None
        try:
            join_timeout = None if deadline is None else max(0.0, deadline - time.monotonic())
            self.join(join_timeout)
            if timeout is not None and self.is_alive():
                raise TimeoutError(
                    f"Camera shutdown timed out after {timeout}s: device={self.dev_video_path}"
                )
        except Exception as exc:
            error = exc
        try:
            recorder_timeout = None if deadline is None else max(0.0, deadline - time.monotonic())
            self.video_recorder.end_wait(timeout=recorder_timeout)
        except Exception as exc:
            if error is None:
                error = exc
        if error is not None:
            raise error

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    def get(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k, out=out)


    def start_recording(self, video_path: str, start_time: float=-1):
        path_len = len(video_path.encode('utf-8'))
        if path_len > self.MAX_PATH_LENGTH:
            raise RuntimeError('video_path too long.')
        self.command_queue.put({
            'cmd': Command.START_RECORDING.value,
            'video_path': video_path,
            'recording_start_time': start_time
        })
        
    def stop_recording(self):
        self.command_queue.put({
            'cmd': Command.STOP_RECORDING.value
        })
    
    def restart_put(self, start_time):
        self.command_queue.put({
            'cmd': Command.RESTART_PUT.value,
            'put_start_time': start_time
        })

    # ========= interval API ===========
    def run(self):
        # limit threads
        threadpool_limits(self.num_threads)
        cv2.setNumThreads(self.num_threads)

        # Initialize V4L2Camera
        w, h = self.resolution
        camera = V4L2Camera(
            device_path=self.dev_video_path,
            format=self.camera_format,
            width=w,
            height=h,
            capture_fps=self.capture_fps,
            buffer_count=self.cap_buffer_size,
        )

        # Configure camera parameters
        camera.set_white_balance(
            auto=(self.auto_white_balance == 1),
            temperature=self.wb_temperature if self.auto_white_balance == 0 else None
        )
        camera.set_exposure(
            auto=(self.auto_exposure == 1),
            exposure_time=self.exposure if self.auto_exposure == 3 else None
        )
        camera.set_brightness(brightness=self.brightness)
        camera.set_gain(self.gain)
        camera.set_gamma(gamma=self.gamma)
        
        try:

            # put frequency regulation
            put_idx = None
            put_start_time = self.put_start_time
            if put_start_time is None:
                put_start_time = time.time()

            # reuse frame buffer
            iter_idx = 0
            t_start = time.time()

            while not self.stop_event.is_set():
                # Read frame from V4L2Camera
                t_recv = time.time()  # 接收时间戳
                ret, frame_rgb = camera.read()
                # print(ret)
                if not ret or frame_rgb is None:
                    continue

                # Convert RGB to BGR (OpenCV uses BGR format)
                # frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                # Timestamp handling (选项A)
                # receive_timestamp: 使用 time.time()（读取帧时）
                # capture_timestamp: 使用 receive_timestamp - 固定延迟
                t_cap = t_recv - self.capture_timestamp_delay  # capture_timestamp = receive_timestamp - 0.101
                t_cal = t_recv - self.receive_latency# 0.1  # calibrated timestamp

                data = dict()
                data['camera_receive_timestamp'] = t_recv
                data['camera_capture_timestamp'] = t_cap
                data['color'] = frame_rgb  # 使用BGR格式
                
                # apply transform
                put_data = data
                if self.transform is not None:
                    put_data = self.transform(dict(data))
                if put_data is None:
                    continue

                if self.put_downsample:                
                    # put frequency regulation
                    local_idxs, global_idxs, put_idx \
                        = get_accumulate_timestamp_idxs(
                            timestamps=[t_cal],
                            start_time=put_start_time,
                            dt=1/self.put_fps,
                            # this is non in first iteration
                            # and then replaced with a concrete number
                            next_global_idx=put_idx,
                            # continue to pump frames even if not started.
                            # start_time is simply used to align timestamps.
                            allow_negative=True
                        )

                    for step_idx in global_idxs:
                        put_data['step_idx'] = step_idx
                        put_data['timestamp'] = t_cal
                        self.ring_buffer.put(put_data, wait=False)
                else:
                    step_idx = int((t_cal - put_start_time) * self.put_fps)
                    put_data['step_idx'] = step_idx
                    put_data['timestamp'] = t_cal
                    self.ring_buffer.put(put_data, wait=False)

                # signal ready
                if iter_idx == 0:
                    self.ready_event.set()    

                # perf
                t_end = time.time()
                duration = t_end - t_start
                frequency = np.round(1 / duration, 1)
                t_start = t_end
                if self.verbose:
                    print(f'[UvcCamera {self.dev_video_path}] FPS {frequency}')

                # fetch command from queue
                try:
                    commands = self.command_queue.get_all()
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0

                # execute commands
                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command['cmd']
                    if cmd == Command.RESTART_PUT.value:
                        put_idx = None
                        put_start_time = command['put_start_time']
                    elif cmd == Command.START_RECORDING.value:
                        video_path = str(command['video_path'])
                        start_time = command['recording_start_time']
                        if start_time < 0:
                            start_time = None
                        self.video_recorder.start_recording(video_path, start_time=start_time)
                    elif cmd == Command.STOP_RECORDING.value:
                        self.video_recorder.stop_recording()

                iter_idx += 1
        finally:
            self.video_recorder.stop()
            # When everything done, release the camera
            if 'camera' in locals():
                camera.release()

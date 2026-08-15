import time
from typing import Dict, List, Optional

import v4l2
import fcntl
import cv2
import numpy as np
import os
import mmap
import array
import select
import datetime
from fractions import Fraction
import math
import tempfile
import threading
import pyudev

class CameraHandler:
    @staticmethod
    def _parse(cam_dev: Dict[str, str]) -> Dict[str, str]:
        try:
            cam_info = {
                "dev_name": cam_dev.properties["DEVNAME"],
                "model": cam_dev.properties["ID_V4L_PRODUCT"],
                "serial": cam_dev.properties["ID_SERIAL_SHORT"],
            }
        except Exception as e:
            print(e)
            cam_info = {}
        # for k, v in cam_dev.properties.items():
        #     print(k, v)
        return cam_info

    @staticmethod
    def list_cams() -> List[Dict[str, str]]:
        context = pyudev.Context()
        print(
            "Finding udev devices with subsystem=video4linux, id_model=USB_2.0_Camera"
        )
        cams = context.list_devices(subsystem="video4linux")
        # cams = context.list_devices(subsystem="video4linux", ID_MODEL="USB_2.0_Camera")
        print("Following udev devices found: ")
        for device in cams:
            print(device)
        cams = [dict(CameraHandler._parse(_)) for _ in cams]
        if not cams:
            print("Could not find any udev devices matching parameters")
        print(cams)
        return cams

    @staticmethod
    def find_cam(model: str, serial: str) -> Optional[Dict[str, str]]:
        cams = CameraHandler.list_cams()
        print(f"Searching for camera with model {model}, serial {serial}")
        for cam in cams:
            if cam["model"] == model:
                if cam["serial"] == serial:
                    return cam
        print(f"No camera with model {model}, serial {serial} found")
        return None


class V4L2Camera:
    _bad_mjpg_run_dir = None
    _decode_warning_lock = threading.Lock()

    def __init__(
            self,
            device_path="/dev/video0",
            format="YUYV",
            width=640,
            height=480,
            save_bad_mjpg_frames=True,
            capture_fps=None,
            buffer_count=8,
            reuse_last_mjpg_frame=True,
    ):
        if (
            capture_fps is not None
            and (
                isinstance(capture_fps, bool)
                or not isinstance(capture_fps, (int, float))
                or not math.isfinite(capture_fps)
                or capture_fps <= 0
            )
        ):
            raise ValueError("capture_fps must be a positive finite number or None")
        if isinstance(buffer_count, bool) or not isinstance(buffer_count, int):
            raise TypeError("buffer_count must be an integer")
        if buffer_count <= 0:
            raise ValueError("buffer_count must be positive")
        if not isinstance(reuse_last_mjpg_frame, bool):
            raise TypeError("reuse_last_mjpg_frame must be a boolean")
        self.device_path = device_path
        self.fd = None
        self.width = width
        self.height = height
        self.frame_id = 0
        self.format = format
        self.string_info = {}
        self._last_frame = None
        self._bad_mjpg_count = 0
        self.save_bad_mjpg_frames = save_bad_mjpg_frames
        self.capture_fps = capture_fps
        self.actual_capture_fps = None
        self.requested_buffer_count = buffer_count
        self.buffer_count = None
        self.reuse_last_mjpg_frame = reuse_last_mjpg_frame
        device_tag = self.device_path.strip(os.sep).replace(os.sep, "_") or "camera"
        if self.save_bad_mjpg_frames:
            if V4L2Camera._bad_mjpg_run_dir is None:
                run_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                V4L2Camera._bad_mjpg_run_dir = os.path.join(
                    os.getcwd(), "bad_mjpg_frames", f"{run_tag}_{os.getpid()}"
                )
            self._bad_mjpg_dir = os.path.join(V4L2Camera._bad_mjpg_run_dir, device_tag)
        else:
            self._bad_mjpg_dir = None
        try:
            self.init_camera()
            self.start_video()
        except BaseException:
            try:
                self.release()
            except Exception as cleanup_exc:
                print(f"相机初始化失败后的清理也失败: {cleanup_exc}")
            raise

    @classmethod
    def _decode_mjpg(cls, jpg):
        with cls._decode_warning_lock:
            with tempfile.TemporaryFile() as stderr_capture:
                saved_stderr_fd = os.dup(2)
                try:
                    os.dup2(stderr_capture.fileno(), 2)
                    decoded = cv2.imdecode(jpg, cv2.IMREAD_COLOR)
                finally:
                    os.dup2(saved_stderr_fd, 2)
                    os.close(saved_stderr_fd)

                stderr_capture.seek(0)
                decode_warning = stderr_capture.read().decode(
                    "utf-8",
                    errors="replace",
                )

        return decoded, decode_warning

    def _save_bad_mjpg(
            self,
            raw,
            ts_2_datetime_str,
            in_ram_time_str,
            buf,
            reason,
            decode_warning,
            decoded=None,
    ):
        self._bad_mjpg_count += 1
        if not self.save_bad_mjpg_frames:
            return None
        os.makedirs(self._bad_mjpg_dir, exist_ok=True)
        base_name = (
            f"{self._bad_mjpg_count:06d}_frame_{self.frame_id:06d}_"
            f"{ts_2_datetime_str}_buf{buf.index}"
        )
        jpg_path = os.path.join(self._bad_mjpg_dir, f"{base_name}.jpg")
        meta_path = os.path.join(self._bad_mjpg_dir, f"{base_name}.txt")
        decoded_path = os.path.join(self._bad_mjpg_dir, f"{base_name}_decoded.png")
        with open(jpg_path, "wb") as f:
            f.write(raw)
        if decoded is not None:
            cv2.imwrite(decoded_path, decoded)
        with open(meta_path, "w", encoding="utf-8") as f:
            f.write(f"device_path: {self.device_path}\n")
            f.write(f"fourcc: {self.fourcc}\n")
            f.write(f"expected_size: {self.width}x{self.height}\n")
            f.write(f"frame_id: {self.frame_id}\n")
            f.write(f"buffer_index: {buf.index}\n")
            f.write(f"bytesused: {buf.bytesused}\n")
            f.write(f"v4l2_timestamp: {ts_2_datetime_str}\n")
            f.write(f"in_ram_timestamp: {in_ram_time_str}\n")
            f.write(f"reason: {reason}\n")
            f.write(f"raw_path: {jpg_path}\n")
            if decoded is not None:
                f.write(f"decoded_path: {decoded_path}\n")
            if decode_warning:
                f.write("decode_warning:\n")
                f.write(decode_warning)
                if not decode_warning.endswith("\n"):
                    f.write("\n")
        return jpg_path

    def _get_string_ctrl(self, ctrl_id):
        """
        内部函数：查询指定ID的字符串控制项
        :param ctrl_id: 控制项ID（如V4L2_CID_MANUFACTURER）
        :return: 字符串值（None表示不支持该控制项）
        """
        if self.fd is None:
            return None

        # 1. 先查询控制项是否存在及属性
        qctrl = v4l2.v4l2_queryctrl()
        qctrl.id = ctrl_id
        try:
            fcntl.ioctl(self.fd, v4l2.VIDIOC_QUERYCTRL, qctrl)
        except Exception as e:
            # 设备不支持该控制项，返回None
            return None

        # 2. 检查控制项是否为字符串类型
        if (qctrl.type & v4l2.V4L2_CTRL_TYPE_STRING) == 0:
            return None

        # 3. 读取字符串值（通过v4l2_control结构体）
        ctrl = v4l2.v4l2_control()
        ctrl.id = ctrl_id
        try:
            fcntl.ioctl(self.fd, v4l2.VIDIOC_G_CTRL, ctrl)
        except Exception as e:
            return None

        # 4. 解码字符串（C语言char数组转Python字符串，去除空字符）
        return ctrl.string.decode("utf-8").rstrip('\x00')

    def init_camera(self):
        """初始化相机设备"""
        try:
            self.fd = os.open(self.device_path, os.O_RDWR)
            print(f"成功打开相机设备: {self.device_path}")
        except Exception as e:
            raise IOError(f"无法打开相机设备: {e}")

        # 配置相机格式（YUYV格式，常用的原始图像格式）
        fmt = v4l2.v4l2_format()
        fmt.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
        fmt.fmt.pix.width = self.width
        fmt.fmt.pix.height = self.height
        if self.format == "YUYV" or self.format == "YUY2":
            fmt.fmt.pix.pixelformat = v4l2.V4L2_PIX_FMT_YUYV  # 像素格式
        elif self.format == "MJPG" or self.format == "JPEG":
            fmt.fmt.pix.pixelformat = v4l2.V4L2_PIX_FMT_MJPEG  # 像素格式
        else:
            fmt.fmt.pix.pixelformat = v4l2.V4L2_PIX_FMT_YUYV  # 像素格式
        fmt.fmt.pix.field = v4l2.V4L2_FIELD_NONE
        fcntl.ioctl(self.fd, v4l2.VIDIOC_S_FMT, fmt)  # 设置格式
        print(f"相机格式配置: {self.width}x{self.height}, {self.format}")

        with open('/proc/uptime', 'r') as f:
            uptime_seconds = float(f.readline().split()[0])
        realtime = time.clock_gettime(time.CLOCK_REALTIME)
        monotonic_raw = time.clock_gettime(time.CLOCK_MONOTONIC_RAW)
        # 计算系统启动时间戳
        monotonic = time.clock_gettime(time.CLOCK_MONOTONIC)
        self.time_offset = time.time() - monotonic

    def set_brightness(self, brightness=None):
        ctrl = v4l2.v4l2_control()
        ctrl.id = v4l2.V4L2_CID_BRIGHTNESS  # brightness控制ID
        ctrl.value = brightness
        try:
            fcntl.ioctl(self.fd, v4l2.VIDIOC_S_CTRL, ctrl)
            print(f"手动brightness设置为: {brightness}K")
        except Exception as e:
            print(f"设置brightness失败（设备可能不支持）: {e}")

    def set_gain(self, gain=None):
        ctrl = v4l2.v4l2_control()
        ctrl.id = v4l2.V4L2_CID_GAIN  # brightness控制ID
        ctrl.value = gain
        try:
            fcntl.ioctl(self.fd, v4l2.VIDIOC_S_CTRL, ctrl)
            print(f"手动gain设置为: {gain}K")
        except Exception as e:
            print(f"设置gain失败（设备可能不支持）: {e}")

    def set_gamma(self, gamma=None):
        ctrl = v4l2.v4l2_control()
        ctrl.id = v4l2.V4L2_CID_GAMMA  # gamma控制ID
        ctrl.value = gamma
        try:
            fcntl.ioctl(self.fd, v4l2.VIDIOC_S_CTRL, ctrl)
            print(f"手动gamma设置为: {gamma}K")
        except Exception as e:
            print(f"设置gamma失败（设备可能不支持）: {e}")

    def set_white_balance(self, auto=True, temperature=None):
        """
        设置白平衡参数
        :param auto: 是否自动白平衡（True/False）
        :param temperature: 手动白平衡时的色温（单位K，如4000-6500），仅auto=False时有效
        """
        # 1. 设置自动白平衡开关
        ctrl = v4l2.v4l2_control()
        ctrl.id = v4l2.V4L2_CID_AUTO_WHITE_BALANCE  # 自动白平衡控制ID
        ctrl.value = 1 if auto else 0  # 1=自动，0=手动
        try:
            fcntl.ioctl(self.fd, v4l2.VIDIOC_S_CTRL, ctrl)
            print(f"自动白平衡: {'开启' if auto else '关闭'}")
        except Exception as e:
            print(f"设置自动白平衡失败（设备可能不支持）: {e}")
            return

        # 2. 手动设置色温（仅当关闭自动白平衡时）
        if not auto and temperature is not None:
            ctrl = v4l2.v4l2_control()
            ctrl.id = v4l2.V4L2_CID_WHITE_BALANCE_TEMPERATURE  # 色温控制ID
            ctrl.value = temperature
            try:
                fcntl.ioctl(self.fd, v4l2.VIDIOC_S_CTRL, ctrl)
                print(f"手动白平衡色温设置为: {temperature}K")
            except Exception as e:
                print(f"设置手动色温失败（设备可能不支持）: {e}")

    def set_exposure(self, auto=True, exposure_time=None):
        """
        设置曝光时间
        :param auto: 是否自动曝光（True/False）
        :param exposure_time: 手动曝光时间（单位通常为微秒，具体取决于相机），仅auto=False时有效
        """
        # 1. 设置自动曝光开关（不同相机可能用不同的控制ID，此处以通用为例）
        ctrl = v4l2.v4l2_control()
        ctrl.id = v4l2.V4L2_CID_EXPOSURE_AUTO  # 自动曝光控制ID
        # 注意：部分相机的AUTO_EXPOSURE值定义不同，需参考相机文档
        # 例如：1=自动，0=手动（或其他值，需根据设备调整）
        ctrl.value = 3 if auto else 1
        try:
            fcntl.ioctl(self.fd, v4l2.VIDIOC_S_CTRL, ctrl)
            print(f"自动曝光: {'开启' if auto else '关闭'}")
        except Exception as e:
            print(f"设置自动曝光失败（设备可能不支持）: {e}")
            return

        # 2. 手动设置曝光时间（仅当关闭自动曝光时）
        if not auto and exposure_time is not None:
            ctrl = v4l2.v4l2_control()
            ctrl.id = v4l2.V4L2_CID_EXPOSURE_ABSOLUTE  # 绝对曝光时间控制ID
            ctrl.value = exposure_time
            try:
                fcntl.ioctl(self.fd, v4l2.VIDIOC_S_CTRL, ctrl)
                print(f"手动曝光时间设置为: {exposure_time}")
            except Exception as e:
                print(f"设置手动曝光失败（设备可能不支持）: {e}")

    def start_video(self):
        if self.capture_fps is not None:
            requested_fps = Fraction(str(self.capture_fps))
            parm = v4l2.v4l2_streamparm()
            parm.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
            parm.parm.capture.timeperframe.numerator = requested_fps.denominator
            parm.parm.capture.timeperframe.denominator = requested_fps.numerator
            try:
                fcntl.ioctl(self.fd, v4l2.VIDIOC_S_PARM, parm)
            except OSError as exc:
                raise OSError(
                    exc.errno,
                    f"{self.device_path}: failed to configure capture FPS "
                    f"to {self.capture_fps}: {exc.strerror or exc}",
                ) from exc

            numerator = parm.parm.capture.timeperframe.numerator
            denominator = parm.parm.capture.timeperframe.denominator
            if numerator == 0 or denominator == 0:
                raise RuntimeError(
                    f"{self.device_path}: driver returned an invalid frame interval "
                    f"{numerator}/{denominator}"
                )
            actual_fps = Fraction(denominator, numerator)
            self.actual_capture_fps = float(actual_fps)
            if actual_fps != requested_fps:
                print(
                    f"[WARN] {self.device_path}: requested capture FPS "
                    f"{float(requested_fps):g}, but driver configured "
                    f"{self.actual_capture_fps:g}; continuing with the driver value"
                )
            else:
                print(
                    f"{self.device_path}: requested capture FPS {float(requested_fps):g}, "
                    f"driver configured {self.actual_capture_fps:g}"
                )

        # 获取当前格式
        fmt = v4l2.v4l2_format()
        fmt.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
        fcntl.ioctl(self.fd, v4l2.VIDIOC_G_FMT, fmt)
        width = fmt.fmt.pix.width
        height = fmt.fmt.pix.height
        pixelformat = fmt.fmt.pix.pixelformat
        fourcc = ''.join([chr((pixelformat >> (8 * i)) & 0xFF) for i in range(4)])
        print(f"{self.device_path}: {width}x{height}, format={fourcc}")
        self.fourcc = fourcc

        # 申请缓冲区
        req = v4l2.v4l2_requestbuffers()
        req.count = self.requested_buffer_count
        req.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
        req.memory = v4l2.V4L2_MEMORY_MMAP
        try:
            fcntl.ioctl(self.fd, v4l2.VIDIOC_REQBUFS, req)
        except OSError as exc:
            raise OSError(
                exc.errno,
                f"{self.device_path}: failed to request "
                f"{self.requested_buffer_count} capture buffers: {exc.strerror or exc}",
            ) from exc
        self.buffer_count = int(req.count)
        if self.buffer_count <= 0:
            raise RuntimeError(
                f"{self.device_path}: driver allocated no capture buffers "
                f"(requested {self.requested_buffer_count})"
            )
        print(
            f"{self.device_path}: allocated {self.buffer_count} capture buffers "
            f"(requested {self.requested_buffer_count})"
        )

        # 映射缓冲区
        self.bufs = []
        for i in range(self.buffer_count):
            buf = v4l2.v4l2_buffer()
            buf.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
            buf.memory = v4l2.V4L2_MEMORY_MMAP
            buf.index = i
            fcntl.ioctl(self.fd, v4l2.VIDIOC_QUERYBUF, buf)
            mm = mmap.mmap(self.fd, buf.length, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=buf.m.offset)
            self.bufs.append((buf, mm))
            fcntl.ioctl(self.fd, v4l2.VIDIOC_QBUF, buf)

        # 启动视频流
        buf_type = array.array('i', [v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE])
        fcntl.ioctl(self.fd, v4l2.VIDIOC_STREAMON, buf_type)

    def read(self):
        """读取一帧图像并转换为RGB格式（用于显示）"""
        # 读取原始YUYV数据（每2个像素占4字节，因此总大小=width*height*2）
        r, _, _ = select.select([self.fd], [], [], 2)
        if self.fd not in r:
            print(f"{self.device_path}: timeout")
            return False, None

        # 取帧
        buf = v4l2.v4l2_buffer()
        buf.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
        buf.memory = v4l2.V4L2_MEMORY_MMAP
        fcntl.ioctl(self.fd, v4l2.VIDIOC_DQBUF, buf)

        ts = buf.timestamp.secs + buf.timestamp.usecs / 1e6
        raw = self.bufs[buf.index][1][:buf.bytesused]
        # 获取当前时间（包含微秒）
        ts += self.time_offset
        ts_2_datetime = datetime.datetime.fromtimestamp(ts)

        # 4. 格式化输出（保留毫秒，格式：年-月-日 时:分:秒.毫秒）
        ts_2_datetime_str = ts_2_datetime.strftime("%Y%m%d_%H%M%S_%f")
        # 计算系统启动时间戳

        now = time.time()

        in_ram_time_str = datetime.datetime.fromtimestamp(now).strftime("%Y%m%d_%H%M%S_%f")
        # frame_id += 1

        # 根据格式保存
        # fname = f"with_stress_15min/{os.path.basename(os.path.basename(self.device_path))}_{frame_id}_{ts_2_datetime_str}_{time_str}.jpg"

        if self.fourcc in ("YUYV", "YUY2"):
            yuv = np.frombuffer(raw, dtype=np.uint8).reshape((self.height, self.width, 2))
            rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_YUYV)
            # cv2.imwrite(f"frame_{self.frame_id}.jpg", rgb)
        elif self.fourcc in ("UYVY",):
            yuv = np.frombuffer(raw, dtype=np.uint8).reshape((self.height, self.width, 2))
            rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_UYVY)
        elif self.fourcc in ("MJPG", "JPEG"):
            jpg = np.frombuffer(raw, dtype=np.uint8)
            is_complete_jpeg = (
                len(jpg) >= 4
                and jpg[0] == 0xFF
                and jpg[1] == 0xD8
                and jpg[-2] == 0xFF
                and jpg[-1] == 0xD9
            )
            if is_complete_jpeg:
                decoded, decode_warning = self._decode_mjpg(jpg)
            else:
                decoded = None
                decode_warning = ""
            bad_reason = None
            if not is_complete_jpeg:
                bad_reason = "missing jpeg soi/eoi marker"
            elif decoded is None:
                bad_reason = "cv2.imdecode returned None"
            elif decode_warning:
                bad_reason = decode_warning.strip().splitlines()[0]
            elif decoded.shape[:2] != (self.height, self.width):
                bad_reason = (
                    f"decoded shape {decoded.shape[:2]} does not match "
                    f"{(self.height, self.width)}"
                )
            if bad_reason is not None:
                saved_path = self._save_bad_mjpg(
                    raw,
                    ts_2_datetime_str,
                    in_ram_time_str,
                    buf,
                    bad_reason,
                    decode_warning,
                    decoded=decoded,
                )
                saved_detail = f", saved={saved_path}" if saved_path is not None else ""
                if self._last_frame is not None and self.reuse_last_mjpg_frame:
                    print(
                        f"{self.device_path}: bad MJPG frame "
                        f"#{self._bad_mjpg_count} ({bad_reason}), "
                        f"bytes={buf.bytesused}{saved_detail}, "
                        "using last valid frame"
                    )
                    rgb = self._last_frame
                else:
                    fallback_detail = (
                        "stale-frame reuse disabled"
                        if self._last_frame is not None
                        else "no last valid frame"
                    )
                    print(
                        f"{self.device_path}: bad MJPG frame "
                        f"#{self._bad_mjpg_count} ({bad_reason}), "
                        f"bytes={buf.bytesused}{saved_detail}, "
                        f"{fallback_detail}"
                    )
                    fcntl.ioctl(self.fd, v4l2.VIDIOC_QBUF, buf)
                    return False, None
            else:
                rgb = decoded
                self._last_frame = rgb
        else:
            return False, None



        # print(f"{ts_2_datetime_str}, {in_ram_time_str}")
        fcntl.ioctl(self.fd, v4l2.VIDIOC_QBUF, buf)
        self.frame_id += 1

        # 将YUYV转换为RGB（使用OpenCV）
        # print((ts_2_datetime_str, in_ram_time_str), rgb)
        return (ts_2_datetime_str, in_ram_time_str), rgb

    def isOpened(self):
        """检查相机是否成功打开"""
        return self.fd is not None

    def release(self):
        """释放相机资源"""
        if self.fd is not None:
            fd = self.fd
            self.fd = None
            buf_type = array.array('i', [v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE])
            try:
                fcntl.ioctl(fd, v4l2.VIDIOC_STREAMOFF, buf_type)
            finally:
                os.close(fd)
            print("相机设备已关闭")


def init_camera(
        cfg_dict: Dict,
):
    cam = CameraHandler.find_cam(cfg_dict["model"], cfg_dict["serial"])
    if cam is None:
        raise Exception(f"Cannot find camera with serial {cfg_dict['serial']}")

    dev_name = cam["dev_name"]

    print(dev_name)

    cap = V4L2Camera(dev_name, format=cfg_dict["data_format"], width=cfg_dict["width"], height=cfg_dict["height"])
    cap.set_white_balance(auto=False, temperature=cfg_dict["wb_temperature"])  # 关闭自动白平衡，设置为6000K
    cap.set_exposure(auto=False, exposure_time=cfg_dict["exposure"])  # 关闭自动曝光，设置为300

    ret, frame = cap.read()
    for i in range(30):
        ret, frame = cap.read()
    return cap


if __name__ == "__main__":
    # 初始化相机
    expo_value_list = [10, 15, 20, 25]
    record_time = 5 * 60
    CameraHandler.list_cams()
    cv2.namedWindow("V4L2 Camera", cv2.WINDOW_NORMAL)
    camera_dev_path = "/dev/video4"
    camera = V4L2Camera(camera_dev_path, format="MJPG", width=1920, height=1080)

    for expo_value in expo_value_list:
        save_folder = f"latency_test_images_96flow_xense_1080p_expo_{expo_value}"
        os.makedirs(save_folder, exist_ok=True)

        # 配置参数（根据相机支持情况调整）
        camera.set_white_balance(auto=False, temperature=4500)  # 关闭自动白平衡，设置为6000K
        camera.set_exposure(auto=False, exposure_time=int(expo_value * 10))  # 关闭自动曝光，设置为300
        camera.set_brightness(brightness=64)
        camera.set_gain(100)
        camera.set_gamma(gamma=200)

        # 读取并显示视频流
        save_num = 0
        start_time = time.time()
        try:
            while True:
                ret, frame = camera.read()
                if not ret:
                    continue

                v4l2_time_str, in_ram_time_str = ret

                cv2.imshow("V4L2 Camera", frame)
                key = cv2.waitKey(1) & 0xFF
                # if key == ord('s'):
                cv2.imwrite(os.path.join(save_folder, f"{os.path.basename(camera_dev_path)}_{save_num}_{v4l2_time_str}_{in_ram_time_str}.jpg"), frame)
                save_num += 1
                # 按'q'退出
                if key == ord('q'):
                    break
                if time.time() - start_time > record_time:
                    break
        except:
            camera.release()
            cv2.destroyAllWindows()
    camera.release()
    cv2.destroyAllWindows()

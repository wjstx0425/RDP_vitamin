#!/usr/bin/env python3
# ruff: noqa: E402
from __future__ import annotations

import argparse
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
import sys
import threading
import time

import cv2

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from configs.camera_config import DEFAULT_CAMERA_CONFIG
from configs.camera_config import CameraConfig
from configs.camera_config import CameraDeviceConfig
from configs.camera_config import CameraSide
from utils.camera_device import V4L2Camera

MAX_DISPLAY_WIDTH = 3000
READER_JOIN_TIMEOUT_S = 2.5
NO_FRAME_TIMEOUT_S = 6.0


@dataclass(frozen=True)
class FrameSnapshot:
    frame: object | None
    fps: float
    error: str | None


class CameraReader:
    def __init__(self, name: str, camera: V4L2Camera):
        self.name = name
        self.camera = camera
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._frame = None
        self._fps = 0.0
        self._error = None
        self._thread = threading.Thread(
            target=self._run,
            name=f"preview-{name}",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        frame_count = 0
        started_at = time.monotonic()
        last_frame_at = started_at
        while not self._stop_event.is_set():
            try:
                ok, frame = self.camera.read()
            except Exception as exc:
                if not self._stop_event.is_set():
                    error = str(exc)
                    with self._lock:
                        self._error = error
                    print(f"[ERROR] {self.name} reader failed: {error}")
                break
            if not ok or frame is None:
                if time.monotonic() - last_frame_at >= NO_FRAME_TIMEOUT_S:
                    error = f"no valid frame received for {NO_FRAME_TIMEOUT_S:.1f} seconds"
                    with self._lock:
                        self._error = error
                    print(f"[ERROR] {self.name} reader failed: {error}")
                    break
                continue
            frame_count += 1
            now = time.monotonic()
            last_frame_at = now
            elapsed = now - started_at
            with self._lock:
                self._frame = frame
                if elapsed >= 1.0:
                    self._fps = frame_count / elapsed
            if elapsed >= 1.0:
                frame_count = 0
                started_at = now

    def snapshot(self) -> FrameSnapshot:
        with self._lock:
            frame = None if self._frame is None else self._frame.copy()
            return FrameSnapshot(frame=frame, fps=self._fps, error=self._error)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread.ident is not None:
            self._thread.join(timeout=READER_JOIN_TIMEOUT_S)
        timed_out = self._thread.is_alive()
        if timed_out:
            print(f"[ERROR] {self.name} reader did not stop before timeout")
        try:
            self.camera.release()
        finally:
            if timed_out:
                self._thread.join(timeout=READER_JOIN_TIMEOUT_S)
                if self._thread.is_alive():
                    print(f"[ERROR] {self.name} reader is still running")


def open_camera(device: CameraDeviceConfig, config: CameraConfig) -> V4L2Camera:
    camera = V4L2Camera(
        device_path=device.path,
        format=config.pixel_format,
        width=config.width,
        height=config.height,
        save_bad_mjpg_frames=False,
        capture_fps=config.capture_fps,
        buffer_count=config.buffer_size,
        reuse_last_mjpg_frame=False,
    )
    try:
        camera.set_white_balance(
            auto=config.auto_white_balance == 1,
            temperature=(
                config.white_balance_temperature
                if config.auto_white_balance == 0
                else None
            ),
        )
        camera.set_exposure(
            auto=config.auto_exposure == 1,
            exposure_time=config.exposure if config.auto_exposure == 3 else None,
        )
        camera.set_brightness(brightness=config.brightness)
        camera.set_gain(config.gain)
        camera.set_gamma(gamma=config.gamma)
    except BaseException:
        try:
            camera.release()
        except Exception as exc:
            print(f"[ERROR] failed to release {device.name}: {exc}")
        raise
    return camera


def release_failed_initialization(
    name: str,
    camera: V4L2Camera | None,
    reader: CameraReader | None,
) -> None:
    try:
        if reader is not None:
            reader.stop()
        elif camera is not None:
            camera.release()
    except Exception as exc:
        print(f"[ERROR] failed to release {name}: {exc}")


def display_frame(name: str, snapshot: FrameSnapshot) -> None:
    if snapshot.frame is None:
        return
    frame = snapshot.frame
    cv2.putText(
        frame,
        f"FPS: {snapshot.fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )
    height, width = frame.shape[:2]
    if width > MAX_DISPLAY_WIDTH:
        scale = MAX_DISPLAY_WIDTH / width
        frame = cv2.resize(frame, (MAX_DISPLAY_WIDTH, int(height * scale)))
    cv2.imshow(name, frame)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preview VB3 V4L2 cameras without recording"
    )
    parser.add_argument("--side", choices=("left", "right", "both"), default="both")
    parser.add_argument(
        "--left-device",
        default=None,
        help="temporarily override the configured left camera path",
    )
    parser.add_argument(
        "--right-device",
        default=None,
        help="temporarily override the configured right camera path",
    )
    return parser


def build_config(args: argparse.Namespace) -> CameraConfig:
    paths = list(DEFAULT_CAMERA_CONFIG.device_paths)
    if args.left_device is not None:
        paths[0] = args.left_device
    if args.right_device is not None:
        paths[1] = args.right_device
    return DEFAULT_CAMERA_CONFIG.with_device_paths(paths)


def run_preview(config: CameraConfig, side: CameraSide) -> int:
    readers: list[CameraReader] = []
    try:
        for device in config.select_devices(side):
            camera = None
            reader = None
            try:
                camera = open_camera(device, config)
                reader = CameraReader(device.name, camera)
                readers.append(reader)
                reader.start()
            except KeyboardInterrupt:
                if reader not in readers:
                    release_failed_initialization(device.name, camera, reader)
                raise
            except Exception as exc:
                if reader in readers:
                    readers.remove(reader)
                release_failed_initialization(device.name, camera, reader)
                print(f"[ERROR] {device.name} initialization failed for {device.path}: {exc}")
                continue
            print(f"[CAM] {device.name}: {device.path} ({config.width}x{config.height})")

        if not readers:
            print("[ERROR] no requested camera could be opened")
            return 1

        print("Press Q in a preview window or Ctrl+C in the terminal to exit.")
        while True:
            for reader in readers.copy():
                snapshot = reader.snapshot()
                if snapshot.error is not None:
                    try:
                        reader.stop()
                    except Exception as exc:
                        print(f"[ERROR] failed to release {reader.name}: {exc}")
                    with suppress(Exception):
                        cv2.destroyWindow(reader.name)
                    readers.remove(reader)
                    continue
                display_frame(reader.name, snapshot)
            if not readers:
                print("[ERROR] all active camera readers failed")
                return 1
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                return 0
    except KeyboardInterrupt:
        return 0
    finally:
        for reader in readers:
            try:
                reader.stop()
            except Exception as exc:
                print(f"[ERROR] failed to release {reader.name}: {exc}")
        try:
            cv2.destroyAllWindows()
        except Exception as exc:
            print(f"[ERROR] failed to close OpenCV preview windows: {exc}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run_preview(build_config(args), args.side)
    except (TypeError, ValueError) as exc:
        print(f"[ERROR] invalid camera configuration: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

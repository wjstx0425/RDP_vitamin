from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import replace
import math
from typing import Literal

CameraSide = Literal["left", "right", "both"]
SUPPORTED_PIXEL_FORMATS = frozenset({"YUYV", "YUY2", "MJPG", "JPEG"})


def _require_int(name: str, value: object, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")


def _require_optional_int(
    name: str,
    value: object,
    *,
    minimum: int,
) -> None:
    if value is not None:
        _require_int(name, value, minimum=minimum)


@dataclass(frozen=True)
class CameraDeviceConfig:
    name: str
    path: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError("camera name must be a string")
        if not self.name.strip():
            raise ValueError("camera name must not be empty or whitespace")
        if not isinstance(self.path, str):
            raise TypeError(f"camera path for {self.name} must be a string")
        if not self.path.strip():
            raise ValueError(f"camera path for {self.name} must not be empty")


#Camera Configuration left/right hand camera config
@dataclass(frozen=True)
class CameraConfig:
    devices: tuple[CameraDeviceConfig, ...] = (
        CameraDeviceConfig(name="left_hand", path="/dev/video0"),
        CameraDeviceConfig(name="right_hand", path="/dev/video2"),
    )
    pixel_format: str = "MJPG"
    width: int = 3840
    height: int = 800
    capture_fps: int = 20
    buffer_size: int = 3
    auto_exposure: int = 3
    exposure: int | None = 170
    auto_white_balance: int = 0
    white_balance_temperature: int | None = 4600
    brightness: int = 0
    gain: int = 40
    gamma: int = 50
    capture_timestamp_delay: float = 0.101

    def __post_init__(self) -> None:
        if not isinstance(self.devices, tuple):
            raise TypeError("devices must be a tuple")
        if not all(isinstance(device, CameraDeviceConfig) for device in self.devices):
            raise TypeError("devices must contain CameraDeviceConfig values")
        names = [device.name for device in self.devices]
        if names != ["left_hand", "right_hand"]:
            raise ValueError("devices must contain left_hand then right_hand")
        if not isinstance(self.pixel_format, str):
            raise TypeError("pixel_format must be a string")
        if self.pixel_format not in SUPPORTED_PIXEL_FORMATS:
            supported = ", ".join(sorted(SUPPORTED_PIXEL_FORMATS))
            raise ValueError(
                f"unsupported pixel_format {self.pixel_format!r}; expected one of {supported}"
            )
        _require_int("width", self.width, minimum=1)
        _require_int("height", self.height, minimum=1)
        _require_int("capture_fps", self.capture_fps, minimum=1)
        _require_int("buffer_size", self.buffer_size, minimum=1)
        _require_int("auto_exposure", self.auto_exposure, minimum=0)
        if self.auto_exposure not in (1, 3):
            raise ValueError("auto_exposure must be 1 (auto) or 3 (manual)")
        _require_optional_int("exposure", self.exposure, minimum=1)
        if self.auto_exposure == 3 and self.exposure is None:
            raise ValueError("manual exposure requires an exposure value")
        _require_int("auto_white_balance", self.auto_white_balance, minimum=0)
        if self.auto_white_balance not in (0, 1):
            raise ValueError("auto_white_balance must be 0 (manual) or 1 (auto)")
        _require_optional_int(
            "white_balance_temperature",
            self.white_balance_temperature,
            minimum=1,
        )
        if self.auto_white_balance == 0 and self.white_balance_temperature is None:
            raise ValueError("manual white balance requires a temperature")
        _require_int("brightness", self.brightness, minimum=0)
        _require_int("gain", self.gain, minimum=0)
        _require_int("gamma", self.gamma, minimum=0)
        if (
            isinstance(self.capture_timestamp_delay, bool)
            or not isinstance(self.capture_timestamp_delay, int | float)
        ):
            raise TypeError("capture_timestamp_delay must be a number")
        if not math.isfinite(self.capture_timestamp_delay):
            raise ValueError("capture_timestamp_delay must be finite")
        if self.capture_timestamp_delay < 0:
            raise ValueError("capture_timestamp_delay must not be negative")

    @property
    def resolution(self) -> tuple[int, int]:
        return self.width, self.height

    @property
    def device_paths(self) -> tuple[str, ...]:
        return tuple(device.path for device in self.devices)

    def select_devices(self, side: CameraSide) -> tuple[CameraDeviceConfig, ...]:
        if side not in ("left", "right", "both"):
            raise ValueError(f"unsupported camera side: {side}")
        if side == "both":
            return self.devices
        target = f"{side}_hand"
        return tuple(device for device in self.devices if device.name == target)

    def with_device_paths(self, paths: Sequence[str]) -> CameraConfig:
        if len(paths) != len(self.devices):
            raise ValueError(f"expected {len(self.devices)} camera paths, got {len(paths)}")
        devices = tuple(replace(device, path=path) for device, path in zip(self.devices, paths, strict=True))
        return replace(self, devices=devices)


DEFAULT_CAMERA_CONFIG = CameraConfig()

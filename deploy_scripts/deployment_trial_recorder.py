"""Deployment trial artifacts for server-side RDP evaluation."""

from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from utils.pose_util import mat_to_pose, pose10d_to_pose_col


RDP_IMAGE_KEYS = (
    "observation.images.camera0",
    "observation.images.camera1",
    "observation.images.tactile_left_0",
    "observation.images.tactile_right_0",
    "observation.images.tactile_left_1",
    "observation.images.tactile_right_1",
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _copy_policy_images(observation: dict[str, Any]) -> dict[str, np.ndarray]:
    missing = [key for key in RDP_IMAGE_KEYS if key not in observation]
    if missing:
        raise ValueError(f"trial observation is missing image keys: {missing}")

    images = {}
    for key in RDP_IMAGE_KEYS:
        image = np.asarray(observation[key])
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"{key} must be HWC RGB, got {image.shape}")
        if image.dtype != np.uint8:
            raise ValueError(f"{key} must be uint8 RGB, got {image.dtype}")
        images[key] = np.ascontiguousarray(image.copy())
    return images


def _write_image_batch(directory: Path, images: dict[str, np.ndarray]) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    for key, image in images.items():
        output_path = directory / f"{key}.png"
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(output_path), image_bgr):
            raise OSError(f"failed to write trial image: {output_path.resolve()}")


def decode_relative_actions(
    raw_action: np.ndarray,
    *,
    sides: list[str],
) -> dict[str, np.ndarray]:
    """Decode each arm's pose10d delta into xyz, axis-angle, and gripper."""
    actions = np.asarray(raw_action)
    if actions.ndim != 2 or actions.shape[1] != len(sides) * 10:
        raise ValueError(
            f"raw action must have shape (N, {len(sides) * 10}), got {actions.shape}"
        )

    decoded = {}
    for robot_idx, side in enumerate(sides):
        start = robot_idx * 10
        relative_pose = mat_to_pose(
            pose10d_to_pose_col(actions[:, start : start + 9])
        )
        gripper = actions[:, start + 9 : start + 10]
        decoded[side] = np.concatenate([relative_pose, gripper], axis=-1)
    return decoded


def extract_robot_poses(
    env_obs: dict[str, Any],
    *,
    sides: list[str],
) -> dict[str, dict[str, list[float]]]:
    """Extract the latest Quest-frame arm pose and measured gripper width."""
    poses = {}
    for robot_idx, side in enumerate(sides):
        poses[side] = {
            "position": np.asarray(
                env_obs[f"robot{robot_idx}_eef_pos"][-1]
            ).reshape(-1).tolist(),
            "rotation_axis_angle": np.asarray(
                env_obs[f"robot{robot_idx}_eef_rot_axis_angle"][-1]
            ).reshape(-1).tolist(),
            "gripper_width": np.asarray(
                env_obs[f"robot{robot_idx}_gripper_width"][-1]
            ).reshape(-1).tolist(),
        }
    return poses


class DeploymentTrialRecorder:
    """Write small step records synchronously and PNG batches off-thread."""

    def __init__(
        self,
        output_root: Path,
        *,
        policy_type: str,
        data_type: str,
        image_interval: int = 5,
        now: datetime | None = None,
    ) -> None:
        if image_interval < 1:
            raise ValueError("image_interval must be positive")
        started_at = now or datetime.now()
        self.trial_id = "trial_" + started_at.strftime("%Y%m%d_%H%M%S_%f")
        self.trial_dir = Path(output_root) / self.trial_id
        self.trial_dir.mkdir(parents=True, exist_ok=False)
        self.image_interval = int(image_interval)
        self._steps_file = (self.trial_dir / "steps.jsonl").open(
            "x", encoding="utf-8", buffering=1
        )
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=f"{self.trial_id}-images",
        )
        self._image_futures: list[Future] = []
        self._image_directories: set[str] = set()
        self._finished = False
        self._manifest = {
            "trial_id": self.trial_id,
            "started_at": started_at.isoformat(),
            "ended_at": None,
            "policy_type": policy_type,
            "data_type": data_type,
            "image_interval": self.image_interval,
            "status": "running",
            "result_label": None,
            "termination_reason": None,
            "failure_step": None,
            "failure": None,
            "step_count": 0,
            "image_batches": [],
        }
        self._write_manifest()

    def _write_manifest(self) -> None:
        manifest_path = self.trial_dir / "manifest.json"
        temporary_path = self.trial_dir / "manifest.json.tmp"
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(self._manifest, file, indent=2, sort_keys=True)
        temporary_path.replace(manifest_path)

    def should_save_periodic_images(self, iter_idx: int) -> bool:
        return int(iter_idx) > 0 and int(iter_idx) % self.image_interval == 0

    def save_images(
        self,
        observation: dict[str, Any],
        *,
        reason: str,
        iter_idx: int,
    ) -> None:
        if reason == "initial":
            directory_name = "initial"
        elif reason == "step":
            directory_name = f"step_{int(iter_idx):06d}"
        elif reason == "failure":
            directory_name = f"failure_step_{int(iter_idx):06d}"
        else:
            raise ValueError(f"unsupported trial image reason: {reason}")
        relative_directory = f"images/{directory_name}"
        if relative_directory in self._image_directories:
            raise ValueError(f"trial image batch already exists: {relative_directory}")

        images = _copy_policy_images(observation)
        self._image_directories.add(relative_directory)
        self._manifest["image_batches"].append(
            {
                "reason": reason,
                "iter_idx": int(iter_idx),
                "directory": relative_directory,
            }
        )
        self._image_futures.append(
            self._executor.submit(
                _write_image_batch,
                self.trial_dir / relative_directory,
                images,
            )
        )

    def log_step(self, record: dict[str, Any]) -> None:
        if self._finished:
            raise RuntimeError("trial recorder is already finished")
        self._steps_file.write(
            json.dumps(_jsonable(record), separators=(",", ":")) + "\n"
        )
        self._manifest["step_count"] += 1

    def record_failure(
        self,
        error: BaseException,
        *,
        failure_step: int,
        observation: dict[str, Any] | None,
        stage: str,
    ) -> None:
        if self._manifest["failure"] is not None:
            return
        self._manifest["failure_step"] = int(failure_step)
        self._manifest["failure"] = {
            "stage": str(stage),
            "type": type(error).__name__,
            "message": str(error),
        }
        if observation is not None:
            self.save_images(
                observation,
                reason="failure",
                iter_idx=failure_step,
            )

    def finish(self, *, result_label: str, termination_reason: str) -> None:
        if result_label not in ("success", "failure"):
            raise ValueError("result_label must be 'success' or 'failure'")
        if self._finished:
            raise RuntimeError("trial recorder is already finished")
        self._finished = True

        self._executor.shutdown(wait=True)
        image_error = None
        for future in self._image_futures:
            try:
                future.result()
            except BaseException as exc:
                if image_error is None:
                    image_error = exc
        self._steps_file.close()

        if image_error is not None:
            self._manifest["failure"] = {
                "stage": "image_write",
                "type": type(image_error).__name__,
                "message": str(image_error),
            }
            self._manifest["status"] = "failed"
            self._manifest["result_label"] = "failure"
            self._manifest["termination_reason"] = "image_write_failure"
        else:
            self._manifest["status"] = (
                "failed" if result_label == "failure" else "completed"
            )
            self._manifest["result_label"] = result_label
            self._manifest["termination_reason"] = str(termination_reason)
        self._manifest["ended_at"] = datetime.now().isoformat()
        self._write_manifest()

        if image_error is not None:
            raise image_error

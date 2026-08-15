"""Audit ordered pick-tube training actions and states without robot access."""

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np


ACTION_DIM = 20
ARM_ACTION_DIM = 10
STATE_XYZ_OFFSETS = {"left": 0, "right": 7}
IDENTITY_ROT6D = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
ACTION_ARM_MAPPING = {"left": "[0:10]", "right": "[10:20]"}


@dataclass(frozen=True)
class DatasetArrays:
    """The action and observation arrays of an ordered robot dataset."""

    action: np.ndarray
    state: np.ndarray
    episode_ends: np.ndarray


def validate_arrays(action: np.ndarray, state: np.ndarray, episode_ends: np.ndarray, source: Path) -> DatasetArrays:
    """Validate dataset array shape, finite values, and episode boundaries."""
    action_value = np.asarray(action, dtype=np.float64)
    state_value = np.asarray(state, dtype=np.float64)
    ends_value = np.asarray(episode_ends)
    if action_value.ndim != 2 or action_value.shape[1] != ACTION_DIM:
        raise ValueError(f"{source}: expected action shape (frames, {ACTION_DIM}), got {action_value.shape}")
    if state_value.ndim != 2 or state_value.shape != action_value.shape:
        raise ValueError(f"{source}: expected state shape (frames, {ACTION_DIM}), got {state_value.shape}")
    if not np.isfinite(action_value).all() or not np.isfinite(state_value).all():
        raise ValueError(f"{source}: action and state must contain only finite values")
    if ends_value.ndim != 1 or not ends_value.size:
        raise ValueError(f"{source}: expected non-empty one-dimensional episode_ends")
    if not np.issubdtype(ends_value.dtype, np.integer):
        raise ValueError(f"{source}: episode_ends must be integers")
    ends_value = ends_value.astype(np.int64, copy=False)
    if np.any(ends_value <= 0) or np.any(np.diff(ends_value) <= 0) or ends_value[-1] != len(action_value):
        raise ValueError(f"{source}: episode_ends must strictly end at frame count {len(action_value)}")
    return DatasetArrays(action=action_value, state=state_value, episode_ends=ends_value)


def load_zarr(path: Path) -> DatasetArrays:
    """Load a replay-buffer Zarr dataset, importing Zarr only when needed."""
    import zarr

    root_path = path / "replay_buffer.zarr" if (path / "replay_buffer.zarr").is_dir() else path
    root = zarr.open_group(str(root_path), mode="r")
    return validate_arrays(
        np.asarray(root["data/action"]),
        np.asarray(root["data/observation_state"]),
        np.asarray(root["meta/episode_ends"]),
        root_path,
    )


def load_lerobot(path: Path) -> DatasetArrays:
    """Load a LeRobot episode collection, importing PyArrow only when needed."""
    import pyarrow.parquet as pq

    metadata = path / "meta" / "episodes.jsonl"
    if not metadata.is_file():
        raise FileNotFoundError(f"{path}: missing meta/episodes.jsonl")
    actions: list[np.ndarray] = []
    states: list[np.ndarray] = []
    ends: list[int] = []
    total = 0
    for parquet in sorted((path / "data").glob("chunk-*/episode_*.parquet"), key=_lerobot_episode_key):
        table = pq.read_table(parquet, columns=["actions", "observation.state"])
        action = np.asarray(table["actions"].to_pylist(), dtype=np.float64)
        state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)
        actions.append(action)
        states.append(state)
        total += len(action)
        ends.append(total)
    if not actions:
        raise FileNotFoundError(f"{path}: no data/chunk-*/episode_*.parquet files")
    return validate_arrays(np.concatenate(actions), np.concatenate(states), np.asarray(ends), path)


def _lerobot_episode_key(parquet: Path) -> tuple[int, int]:
    chunk_match = re.fullmatch(r"chunk-(\d+)", parquet.parent.name)
    episode_match = re.fullmatch(r"episode_(\d+)\.parquet", parquet.name)
    if chunk_match is None or episode_match is None:
        raise ValueError(f"{parquet}: expected chunk-<number>/episode_<number>.parquet")
    return int(chunk_match.group(1)), int(episode_match.group(1))


def load_dataset(path: Path, source_format: str = "auto") -> DatasetArrays:
    """Load a supported dataset, inferring its layout only when requested."""
    if source_format == "zarr":
        return load_zarr(path)
    if source_format == "lerobot":
        return load_lerobot(path)
    if source_format != "auto":
        raise ValueError(f"source_format must be one of auto, zarr, lerobot; got {source_format!r}")
    if path.name == "replay_buffer.zarr" or (path / "replay_buffer.zarr").is_dir():
        return load_zarr(path)
    if (path / "meta" / "episodes.jsonl").is_file():
        return load_lerobot(path)
    raise FileNotFoundError(f"{path}: could not identify Zarr replay buffer or LeRobot dataset")


def xyz_norms(action: np.ndarray, side: str) -> np.ndarray:
    """Return per-frame action position magnitudes for one arm."""
    if side not in ("left", "right"):
        raise ValueError(f"side must be left or right, got {side!r}")
    value = np.asarray(action, dtype=np.float64)
    offset = 0 if side == "left" else ARM_ACTION_DIM
    return np.linalg.norm(value[:, offset : offset + 3], axis=1)


def _state_xyz_motion(state: np.ndarray, side: str) -> np.ndarray:
    offset = STATE_XYZ_OFFSETS[side]
    positions = np.asarray(state, dtype=np.float64)[:, offset : offset + 3]
    return np.concatenate((np.zeros(1), np.linalg.norm(np.diff(positions, axis=0), axis=1)))


def first_moving_side(action: np.ndarray, threshold: float) -> str:
    """Classify the arm whose action first reaches the movement threshold."""
    left_hits = np.flatnonzero(xyz_norms(action, "left") >= threshold)
    right_hits = np.flatnonzero(xyz_norms(action, "right") >= threshold)
    left_index = int(left_hits[0]) if left_hits.size else math.inf
    right_index = int(right_hits[0]) if right_hits.size else math.inf
    if left_index == right_index:
        return "simultaneous" if left_index != math.inf else "none"
    return "left" if left_index < right_index else "right"


def _correlation(first: np.ndarray, second: np.ndarray) -> float | None:
    if len(first) < 2:
        return None
    first_centered = first - np.mean(first)
    second_centered = second - np.mean(second)
    denominator = float(np.linalg.norm(first_centered) * np.linalg.norm(second_centered))
    if denominator == 0.0:
        return None
    return float(np.dot(first_centered, second_centered) / denominator)


def scan_lag(action_motion: np.ndarray, state_motion: np.ndarray, max_lag: int) -> dict[str, float | int | None]:
    """Find the action-to-state offset with the strongest Pearson correlation."""
    action_value = np.asarray(action_motion, dtype=np.float64)
    state_value = np.asarray(state_motion, dtype=np.float64)
    if action_value.ndim != 1 or state_value.ndim != 1 or len(action_value) != len(state_value):
        raise ValueError("action_motion and state_motion must be equally sized one-dimensional arrays")
    if not np.isfinite(action_value).all() or not np.isfinite(state_value).all():
        raise ValueError("action_motion and state_motion must be finite")
    if isinstance(max_lag, bool) or not isinstance(max_lag, int) or max_lag < 0:
        raise ValueError("max_lag must be a non-negative integer")

    candidates: list[tuple[float, int]] = []
    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:
            first, second = action_value[:-lag], state_value[lag:]
        elif lag < 0:
            first, second = action_value[-lag:], state_value[:lag]
        else:
            first, second = action_value, state_value
        correlation = _correlation(first, second)
        if correlation is not None:
            candidates.append((correlation, lag))
    if not candidates:
        return {"best_lag_frames": 0, "correlation": None}
    correlation, lag = max(candidates, key=lambda item: (item[0], -abs(item[1]), -item[1]))
    return {"best_lag_frames": lag, "correlation": correlation}


def _side_metrics(action: np.ndarray, side: str) -> dict[str, Any]:
    offset = 0 if side == "left" else ARM_ACTION_DIM
    position = action[:, offset : offset + 3]
    rotation = action[:, offset + 3 : offset + 9]
    gripper = action[:, offset + 9]
    return {
        "position_rms_mm_per_step": float(np.sqrt(np.mean(np.sum(np.square(position), axis=1))) * 1000.0),
        "rot6d_residual_to_identity_rms": float(np.sqrt(np.mean(np.square(rotation - IDENTITY_ROT6D)))),
        "gripper": {
            "min": float(np.min(gripper)),
            "mean": float(np.mean(gripper)),
            "std": float(np.std(gripper)),
            "max": float(np.max(gripper)),
        },
    }


def _start_frames(data: DatasetArrays, window: int) -> np.ndarray:
    parts = [data.action[start : min(start + window, end)] for start, end in _episode_ranges(data.episode_ends)]
    return np.concatenate(parts)


def _episode_ranges(episode_ends: np.ndarray) -> list[tuple[int, int]]:
    starts = np.concatenate((np.array([0]), episode_ends[:-1]))
    return [(int(start), int(end)) for start, end in zip(starts, episode_ends, strict=True)]


def audit_dataset(
    data: DatasetArrays,
    start_windows: tuple[int, ...] = (30, 60),
    movement_threshold: float = 0.001,
    max_lag: int = 10,
) -> dict[str, Any]:
    """Return schema, action-scale, first-motion, and action/state-lag diagnostics."""
    data = validate_arrays(data.action, data.state, data.episode_ends, Path("dataset"))
    if not start_windows or any(isinstance(window, bool) or not isinstance(window, int) or window < 1 for window in start_windows):
        raise ValueError("start_windows must contain positive integers")
    if isinstance(movement_threshold, (bool, np.bool_)) or not np.isfinite(movement_threshold) or movement_threshold < 0.0:
        raise ValueError("movement_threshold must be a finite non-negative number")

    episodes: list[dict[str, Any]] = []
    counts = {side: 0 for side in ("left", "right", "simultaneous", "none")}
    for index, (start, end) in enumerate(_episode_ranges(data.episode_ends)):
        first_side = first_moving_side(data.action[start:end], movement_threshold)
        counts[first_side] += 1
        episodes.append({
            "index": index,
            "start_frame": start,
            "end_frame": end,
            "frames": end - start,
            "first_moving_side": first_side,
        })

    return {
        "frames": int(len(data.action)),
        "action_arm_mapping": dict(ACTION_ARM_MAPPING),
        "schema": {
            "action_shape": list(data.action.shape),
            "state_shape": list(data.state.shape),
            "episode_ends": data.episode_ends.tolist(),
            "finite": True,
        },
        "all_frames": {side: _side_metrics(data.action, side) for side in ("left", "right")},
        "start_windows": {
            str(window): {side: _side_metrics(_start_frames(data, window), side) for side in ("left", "right")}
            for window in start_windows
        },
        "episodes": episodes,
        "first_moving_side_counts": counts,
        "lag_scans": {
            side: scan_lag(xyz_norms(data.action, side), _state_xyz_motion(data.state, side), max_lag)
            for side in ("left", "right")
        },
    }


def emit_json(report: dict[str, Any], output: Path | None) -> None:
    """Render an audit to stdout or create a new JSON report at ``output``."""
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if output is None:
        print(rendered, end="")
        return
    with output.open("x", encoding="utf-8") as stream:
        stream.write(rendered)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit ordered pick-tube training actions and states")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--format", choices=("auto", "zarr", "lerobot"), default="auto")
    parser.add_argument("--start-windows", nargs="+", type=int, default=[30, 60])
    parser.add_argument("--movement-threshold-m", type=float, default=0.001)
    parser.add_argument("--max-lag", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    data = load_dataset(args.dataset.resolve(), args.format)
    report = audit_dataset(data, tuple(args.start_windows), args.movement_threshold_m, args.max_lag)
    emit_json(report, args.output)


if __name__ == "__main__":
    main()

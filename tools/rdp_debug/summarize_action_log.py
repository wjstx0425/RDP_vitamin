"""Summarize offline RDP action-debug JSONL files."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


def percentiles(values: np.ndarray | list[float]) -> dict[str, float]:
    """Return a compact percentile summary for a finite numeric series."""
    series = np.asarray(values, dtype=np.float64)
    if series.size == 0:
        return {}
    if not np.isfinite(series).all():
        raise ValueError("percentile values must be finite")
    return {
        "min": float(np.min(series)),
        "p50": float(np.percentile(series, 50)),
        "p95": float(np.percentile(series, 95)),
        "p99": float(np.percentile(series, 99)),
        "max": float(np.max(series)),
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a strict JSONL action log, retaining source context on errors."""
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error.msg}") from error
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    if len(rows) < 2:
        raise ValueError(f"{path}: expected at least two records")
    return rows


def _finite_scalar(value: Any, context: str) -> float:
    try:
        scalar = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context}: expected a finite number") from error
    if not np.isfinite(scalar):
        raise ValueError(f"{context}: expected a finite number")
    return scalar


def _strictly_increasing(records: list[dict[str, Any]], key: str) -> None:
    previous: float | None = None
    for index, record in enumerate(records):
        value = _finite_scalar(record.get(key), f"record {index} {key}")
        if previous is not None and value <= previous:
            raise ValueError(f"record {index} {key}: expected strict monotonic increase")
        previous = value


def _pose(value: Any, context: str) -> np.ndarray:
    pose = np.asarray(value, dtype=np.float64)
    if pose.shape != (6,) or not np.isfinite(pose).all():
        raise ValueError(f"{context}: expected finite pose shape (6,)")
    return pose


def _gripper(value: Any, context: str) -> float:
    gripper = np.asarray(value, dtype=np.float64)
    if gripper.shape != (1,) or not np.isfinite(gripper).all():
        raise ValueError(f"{context}: expected finite gripper shape (1,)")
    return float(gripper[0])


def _jump_summary(
    jumps: dict[str, dict[str, list[float]]],
) -> dict[str, dict[str, float]]:
    return {
        side: {
            f"{group}_mean": float(np.mean(values)) if values else 0.0
            for group, values in by_group.items()
        }
        for side, by_group in jumps.items()
    }


def summarize_records(records: list[dict[str, Any]], replan_interval: int) -> dict[str, Any]:
    """Summarize timing and consecutive scheduled controller-target changes."""
    if len(records) < 2:
        raise ValueError("expected at least two records")
    if isinstance(replan_interval, bool) or not isinstance(replan_interval, int) or replan_interval < 1:
        raise ValueError("replan_interval must be a positive integer")

    _strictly_increasing(records, "iter_idx")
    _strictly_increasing(records, "obs_seq")
    times = np.asarray(
        [_finite_scalar(record.get("time"), f"record {index} time") for index, record in enumerate(records)],
        dtype=np.float64,
    )
    periods = np.diff(times)
    if np.any(periods <= 0.0):
        raise ValueError("record time: expected strict monotonic increase")

    position_jumps = {side: {"within": [], "boundary": []} for side in ("left", "right")}
    rotation_jumps = {side: {"within": [], "boundary": []} for side in ("left", "right")}
    gripper_jumps = {side: {"within": [], "boundary": []} for side in ("left", "right")}
    previous: dict[str, tuple[np.ndarray, float]] = {}
    target_leads: list[float] = []
    scheduled = 0

    for record_index, record in enumerate(records):
        record_time = times[record_index]
        controller_records = record.get("controller_records", [])
        if not isinstance(controller_records, list):
            raise ValueError(f"record {record_index} controller_records: expected a list")
        for controller_index, controller_record in enumerate(controller_records):
            if not isinstance(controller_record, dict):
                raise ValueError(
                    f"record {record_index} controller_records[{controller_index}]: expected an object"
                )
            if not controller_record.get("scheduled", False):
                continue
            scheduled += 1
            target_time = _finite_scalar(
                controller_record.get("target_time"),
                f"record {record_index} controller_records[{controller_index}] target_time",
            )
            target_leads.append(target_time - record_time)
            iteration = int(record["iter_idx"])
            group = "boundary" if record_index and iteration % replan_interval == 0 else "within"
            for side in ("left", "right"):
                context = f"record {record_index} controller_records[{controller_index}] {side}"
                pose = _pose(controller_record.get(f"{side}_target_pose"), f"{context}_target_pose")
                gripper = _gripper(controller_record.get(f"{side}_gripper"), f"{context}_gripper")
                if side in previous:
                    previous_pose, previous_gripper = previous[side]
                    position_jumps[side][group].append(float(np.linalg.norm(pose[:3] - previous_pose[:3])))
                    rotation_jumps[side][group].append(
                        float(
                            (Rotation.from_rotvec(previous_pose[3:]).inv()
                            * Rotation.from_rotvec(pose[3:])).magnitude()
                        )
                    )
                    gripper_jumps[side][group].append(abs(gripper - previous_gripper))
                previous[side] = (pose, gripper)

    duration = float(times[-1] - times[0])
    return {
        "frames": len(records),
        "duration_s": duration,
        "effective_hz": float((len(records) - 1) / duration),
        "scheduled": scheduled,
        "period_s": percentiles(periods),
        "target_lead_s": percentiles(target_leads),
        "position_jump_m": _jump_summary(position_jumps),
        "rotation_jump_rad": _jump_summary(rotation_jumps),
        "gripper_jump_m": _jump_summary(gripper_jumps),
    }


def write_new_text(path: Path, text: str) -> None:
    """Write CLI output without silently replacing an existing report."""
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize offline RDP action-debug JSONL files")
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--replan-interval", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {str(path): summarize_records(load_jsonl(path), args.replan_interval) for path in args.logs}
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        write_new_text(args.output, rendered + "\n")
    else:
        print(rendered)


if __name__ == "__main__":
    main()

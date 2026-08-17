#!/usr/bin/env python3
"""Replay paired deployment-trial observations without opening robot connections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

from tools.rdp_debug.replay_saved_observations import EXPECTED_IMAGE_KEYS
from tools.rdp_debug.replay_saved_observations import _load_config
from tools.rdp_debug.replay_saved_observations import _resolve_vitamin_path
from tools.rdp_debug.replay_saved_observations import _validated_action
from tools.rdp_debug.replay_saved_observations import import_vitamin_runtime
from tools.rdp_debug.replay_saved_observations import seed_everything
from tools.rdp_debug.replay_saved_observations import write_new_text


ACTION_DIM = 20


def axis_angle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:
    """Convert a finite axis-angle vector to a 3x3 rotation matrix."""
    vector = np.asarray(axis_angle, dtype=np.float64)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"axis_angle: expected finite shape (3,), got {vector.shape}")
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z = vector / angle
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def image_directory(trial: Path, iteration: int) -> Path:
    """Return the image batch associated with an image-sampled iteration."""
    if iteration < 0:
        raise ValueError("iteration must be non-negative")
    name = "initial" if iteration == 0 else f"step_{iteration:06d}"
    return trial / "images" / name


def load_trial_rows(trial: Path) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    """Load one manifest and index its finite, contiguous JSONL records by iteration."""
    manifest_path = trial / "manifest.json"
    steps_path = trial / "steps.jsonl"
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    rows: dict[int, dict[str, Any]] = {}
    with steps_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            iteration = int(row["iter_idx"])
            if iteration in rows:
                raise ValueError(f"{steps_path}:{line_number}: duplicate iter_idx {iteration}")
            rows[iteration] = row
    if sorted(rows) != list(range(len(rows))):
        raise ValueError(f"{steps_path}: iter_idx must be contiguous from zero")
    if int(manifest.get("step_count", -1)) != len(rows):
        raise ValueError(f"{manifest_path}: step_count does not match steps.jsonl")
    return manifest, rows


def load_trial_observation(
    trial: Path,
    row: dict[str, Any],
    cv2_module: Any,
) -> dict[str, np.ndarray]:
    """Load the exact state and six processed RGB images for one saved iteration."""
    iteration = int(row["iter_idx"])
    directory = image_directory(trial, iteration)
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    state = np.asarray(row["state"], dtype=np.float32)
    if state.shape != (ACTION_DIM,) or not np.isfinite(state).all():
        raise ValueError(f"iter {iteration}: expected finite 20D state, got {state.shape}")
    observation: dict[str, np.ndarray] = {"observation.state": state}
    for key in EXPECTED_IMAGE_KEYS:
        path = directory / f"{key}.png"
        bgr = cv2_module.imread(str(path), cv2_module.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(path)
        observation[key] = cv2_module.cvtColor(bgr, cv2_module.COLOR_BGR2RGB)
    return observation


def action_metrics(action: np.ndarray, conversion_pose: dict[str, Any]) -> dict[str, Any]:
    """Decode local translations and rotate them into the robot-base frame."""
    value = np.asarray(action, dtype=np.float64)
    if value.shape != (ACTION_DIM,) or not np.isfinite(value).all():
        raise ValueError(f"action: expected finite shape (20,), got {value.shape}")
    result: dict[str, Any] = {}
    for side, start in (("left", 0), ("right", 10)):
        local = value[start : start + 3]
        pose = conversion_pose[side]
        rotation = axis_angle_to_matrix(np.asarray(pose["rotation_axis_angle"], dtype=np.float64))
        base = rotation @ local
        result[side] = {
            "local_translation_mm": (local * 1000.0).tolist(),
            "base_translation_mm": (base * 1000.0).tolist(),
            "translation_norm_mm": float(np.linalg.norm(local) * 1000.0),
            "gripper_m": float(value[start + 9]),
        }
    return result


def _vector_summary(values: np.ndarray) -> dict[str, list[float]]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or not np.isfinite(array).all():
        raise ValueError(f"expected finite vector array [N,3], got {array.shape}")
    return {
        "mean": np.mean(array, axis=0).tolist(),
        "std": np.std(array, axis=0).tolist(),
        "min": np.min(array, axis=0).tolist(),
        "max": np.max(array, axis=0).tolist(),
    }


def summarize_samples(samples: list[dict[str, Any]], recorded: dict[str, Any]) -> dict[str, Any]:
    """Summarize per-seed local/base translations and retain the online sample."""
    if not samples:
        raise ValueError("at least one seed sample is required")
    report: dict[str, Any] = {"recorded_online_action": recorded, "sides": {}}
    for side in ("left", "right"):
        local = np.asarray([sample["metrics"][side]["local_translation_mm"] for sample in samples])
        base = np.asarray([sample["metrics"][side]["base_translation_mm"] for sample in samples])
        norms = np.asarray([sample["metrics"][side]["translation_norm_mm"] for sample in samples])
        grippers = np.asarray([sample["metrics"][side]["gripper_m"] for sample in samples])
        report["sides"][side] = {
            "local_translation_mm": _vector_summary(local),
            "base_translation_mm": _vector_summary(base),
            "translation_norm_mm": {
                "mean": float(np.mean(norms)),
                "std": float(np.std(norms)),
                "min": float(np.min(norms)),
                "max": float(np.max(norms)),
            },
            "gripper_m": {
                "mean": float(np.mean(grippers)),
                "std": float(np.std(grippers)),
                "min": float(np.min(grippers)),
                "max": float(np.max(grippers)),
            },
            "positive_base_z_fraction": float(np.mean(base[:, 2] > 0.0)),
            "base_z_above_0_1mm_fraction": float(np.mean(base[:, 2] > 0.1)),
        }
    report["seed_samples"] = samples
    return report


def _parse_iterations(text: str) -> list[int]:
    values = [int(value) for value in text.split(",") if value.strip()]
    if not values or any(value < 0 for value in values) or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("iterations must be unique non-negative comma-separated integers")
    return values


def replay(args: argparse.Namespace) -> dict[str, Any]:
    """Load checkpoints once, then replay selected paired observations for all seeds."""
    import cv2
    import torch

    vitamin_repo = args.vitamin_repo.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    config = _load_config(config_path)
    model = config["model"]
    control = config["control"]
    device = torch.device(args.device or str(model.get("device", "cuda:0")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    ldp_checkpoint = _resolve_vitamin_path(vitamin_repo, model["ldp_checkpoint"])
    at_checkpoint = _resolve_vitamin_path(vitamin_repo, model["at_checkpoint"])
    encoder_dir = _resolve_vitamin_path(vitamin_repo, model["tactile_encoder_dir"])
    missing = [path for path in (ldp_checkpoint, at_checkpoint) if not path.is_file()]
    if not encoder_dir.is_dir():
        missing.append(encoder_dir)
    if missing:
        raise FileNotFoundError("missing RDP deployment files:\n" + "\n".join(map(str, missing)))

    PickTubeRDPRuntime, load_policy, load_tactile_resnet18 = import_vitamin_runtime(vitamin_repo)
    policy, checkpoint_config = load_policy(
        ldp_checkpoint, at_checkpoint, device, int(model.get("num_inference_steps", 8))
    )
    tactile_encoder = load_tactile_resnet18(encoder_dir, device=device)
    runtime = PickTubeRDPRuntime(
        policy,
        tactile_encoder,
        device,
        slow_update_interval=int(control.get("slow_update_interval", 5)),
        dataset_obs_temporal_downsample_ratio=int(checkpoint_config.dataset_obs_temporal_downsample_ratio),
        n_obs_steps=int(checkpoint_config.n_obs_steps),
    )

    reports: dict[str, Any] = {}
    for trial_argument in args.trials:
        trial = trial_argument.expanduser().resolve()
        manifest, rows = load_trial_rows(trial)
        interval = int(manifest["image_interval"])
        requested = args.iterations or list(range(0, len(rows), interval))
        invalid = [value for value in requested if value not in rows or value % interval != 0]
        if invalid:
            raise ValueError(f"{trial}: iterations are unavailable image steps: {invalid}")
        trial_report: dict[str, Any] = {
            "manifest": {key: value for key, value in manifest.items() if key != "image_batches"},
            "iterations": {},
        }
        for iteration in requested:
            row = rows[iteration]
            observation = load_trial_observation(trial, row, cv2)
            recorded_action = _validated_action(row["raw_action"])
            conversion_pose = row.get("conversion_pose", row.get("observation_pose"))
            if conversion_pose is None:
                raise ValueError(
                    f"{trial}: iter {iteration} has neither conversion_pose nor observation_pose"
                )
            recorded_metrics = action_metrics(recorded_action, conversion_pose)
            samples: list[dict[str, Any]] = []
            for seed in args.seeds:
                seed_everything(seed, torch)
                runtime.reset()
                started = time.perf_counter()
                action, slow_update = runtime.predict(observation)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                predicted = _validated_action(action)
                samples.append(
                    {
                        "seed": int(seed),
                        "inference_ms": float((time.perf_counter() - started) * 1000.0),
                        "slow_update": bool(slow_update),
                        "raw_action": predicted.tolist(),
                        "metrics": action_metrics(predicted, conversion_pose),
                    }
                )
            iteration_report = summarize_samples(samples, recorded_metrics)
            iteration_report.update(
                {
                    "iter_idx": int(row["iter_idx"]),
                    "obs_seq": int(row["obs_seq"]),
                    "state": row["state"],
                    "conversion_pose": conversion_pose,
                }
            )
            trial_report["iterations"][str(iteration)] = iteration_report
        reports[trial.name] = trial_report
    return {
        "config": str(config_path),
        "vitamin_repo": str(vitamin_repo),
        "device": str(device),
        "seeds": args.seeds,
        "trials": reports,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vitamin-repo", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--trials", required=True, nargs="+", type=Path)
    parser.add_argument("--iterations", type=_parse_iterations)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(32)))
    parser.add_argument("--device")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = replay(args)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        write_new_text(args.output.expanduser().resolve(), rendered)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Replay saved RDP observations locally without opening robot connections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np


EXPECTED_IMAGE_KEYS = (
    "observation.images.camera0",
    "observation.images.camera1",
    "observation.images.tactile_left_0",
    "observation.images.tactile_right_0",
    "observation.images.tactile_left_1",
    "observation.images.tactile_right_1",
)
ACTION_DIM = 20


def discover_steps(root: Path) -> list[Path]:
    """Return saved observation directories in strict numerical order."""
    steps = sorted(path for path in root.glob("step_*") if path.is_dir())
    if not steps:
        raise FileNotFoundError(f"{root}: no step_* directories")
    try:
        numbers = [int(path.name.removeprefix("step_")) for path in steps]
    except ValueError as error:
        raise ValueError(f"{root}: step directories must end in an integer") from error
    ordered = [path for _, path in sorted(zip(numbers, steps, strict=True))]
    for expected, actual in enumerate(sorted(numbers), min(numbers)):
        if actual != expected:
            raise ValueError(f"{root}: missing saved step {expected:06d}")
    return ordered


def load_saved_observation(step: Path, cv2_module: Any) -> dict[str, np.ndarray]:
    """Load one finite 20D state and the required on-disk JPEG images as RGB."""
    state_path = step / "observation.state.npy"
    state = np.load(state_path, allow_pickle=False).astype(np.float32, copy=False)
    if state.shape != (ACTION_DIM,) or not np.isfinite(state).all():
        raise ValueError(f"{state_path}: expected finite shape (20,), got {state.shape}")
    observation: dict[str, np.ndarray] = {"observation.state": state}
    for key in EXPECTED_IMAGE_KEYS:
        path = step / f"{key}.jpg"
        bgr = cv2_module.imread(str(path), cv2_module.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(path)
        observation[key] = cv2_module.cvtColor(bgr, cv2_module.COLOR_BGR2RGB)
    return observation


def _summary(values: np.ndarray | list[float]) -> dict[str, float]:
    series = np.asarray(values, dtype=np.float64)
    if series.size == 0:
        return {"mean": 0.0, "max": 0.0}
    if not np.isfinite(series).all():
        raise ValueError("summary values must be finite")
    return {"mean": float(np.mean(series)), "max": float(np.max(series))}


def summarize_actions(
    actions: np.ndarray,
    slow_flags: list[bool],
    timings_ms: list[float],
    replan_interval: int,
) -> dict[str, Any]:
    """Summarize local inference timings and gripper jumps at replan boundaries."""
    values = np.asarray(actions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != ACTION_DIM or not np.isfinite(values).all():
        raise ValueError(f"actions: expected finite shape (frames, 20), got {values.shape}")
    if len(values) != len(slow_flags) or len(values) != len(timings_ms):
        raise ValueError("actions, slow_flags, and timings_ms must have matching lengths")
    if isinstance(replan_interval, bool) or not isinstance(replan_interval, int) or replan_interval < 1:
        raise ValueError("replan_interval must be a positive integer")
    timing_values = np.asarray(timings_ms, dtype=np.float64)
    if not np.isfinite(timing_values).all():
        raise ValueError("timings_ms must be finite")

    replan_frames = [index for index, slow in enumerate(slow_flags) if slow]
    jumps = {side: {"within": [], "boundary": []} for side in ("left", "right")}
    for index in range(1, len(values)):
        group = "boundary" if index in replan_frames else "within"
        for side, gripper_index in (("left", 9), ("right", 19)):
            jumps[side][group].append(abs(float(values[index, gripper_index] - values[index - 1, gripper_index])))
    return {
        "frames": int(len(values)),
        "replan_frames": replan_frames,
        "slow_update_frames": len(replan_frames),
        "inference_ms": _summary(timing_values),
        "gripper_boundary_jump_m": {side: _summary(groups["boundary"]) for side, groups in jumps.items()},
        "gripper_within_jump_m": {side: _summary(groups["within"]) for side, groups in jumps.items()},
    }


def write_new_text(path: Path, text: str) -> None:
    """Create an output file without silently replacing an existing report."""
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)


def import_vitamin_runtime(vitamin_repo: Path) -> tuple[Any, Any, Any]:
    """Import only local policy runtime helpers after CLI arguments are validated."""
    sys.path.insert(0, str(vitamin_repo))
    from deploy_pick_tube_rdp import PickTubeRDPRuntime
    from deploy_pick_tube_rdp import load_policy
    from reactive_diffusion_policy.deploy.tactile_encoder_torch import load_tactile_resnet18

    return PickTubeRDPRuntime, load_policy, load_tactile_resnet18


def seed_everything(seed: int, torch_module: Any) -> None:
    """Seed all policy randomness used by the local replay process."""
    random.seed(seed)
    np.random.seed(seed)
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)


def _load_config(path: Path) -> dict[str, Any]:
    import yaml

    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    for section in ("model", "control"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"{path}: missing config section {section}")
    return config


def _resolve_vitamin_path(vitamin_repo: Path, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return (path if path.is_absolute() else vitamin_repo / path).resolve()


def _validated_action(action: Any) -> np.ndarray:
    value = np.asarray(action, dtype=np.float32)
    if value.shape == (1, ACTION_DIM):
        value = value[0]
    if value.shape != (ACTION_DIM,) or not np.isfinite(value).all():
        raise RuntimeError(f"runtime returned expected finite shape (20,), got {value.shape}")
    return value


def _predict_frames(runtime: Any, observations: list[dict[str, np.ndarray]], torch_module: Any, device: Any) -> dict[str, Any]:
    runtime.reset()
    actions: list[np.ndarray] = []
    slow_flags: list[bool] = []
    timings_ms: list[float] = []
    for observation in observations:
        started = time.perf_counter()
        action, slow_update = runtime.predict(observation)
        if device.type == "cuda":
            torch_module.cuda.synchronize(device)
        timings_ms.append((time.perf_counter() - started) * 1000.0)
        actions.append(_validated_action(action))
        slow_flags.append(bool(slow_update))
    return {"actions": np.stack(actions), "slow_flags": slow_flags, "timings_ms": timings_ms}


def replay(args: argparse.Namespace) -> dict[str, Any]:
    """Load local checkpoints and observations, then run offline policy inference."""
    import cv2
    import torch

    vitamin_repo = args.vitamin_repo.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    observations_root = args.observations.expanduser().resolve()
    if not vitamin_repo.is_dir():
        raise FileNotFoundError(f"{vitamin_repo}: Vitamin repository not found")
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
    )
    steps = discover_steps(observations_root)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        steps = steps[: args.limit]
    observations = [load_saved_observation(step, cv2) for step in steps]
    if args.seeds:
        reports: dict[str, Any] = {}
        for seed in args.seeds:
            seed_everything(seed, torch)
            result = _predict_frames(runtime, [observations[0]], torch, device)
            reports[str(seed)] = summarize_actions(result["actions"], result["slow_flags"], result["timings_ms"], int(control.get("slow_update_interval", 5)))
        return {"seeds": reports}
    result = _predict_frames(runtime, observations, torch, device)
    return summarize_actions(result["actions"], result["slow_flags"], result["timings_ms"], int(control.get("slow_update_interval", 5)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vitamin-repo", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--observations", required=True, type=Path)
    parser.add_argument("--device")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = replay(args)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        write_new_text(args.output, rendered)


if __name__ == "__main__":
    main()

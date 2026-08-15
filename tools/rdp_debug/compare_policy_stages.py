"""Compare an offline Vitamin policy at source, AT, and LDP boundaries.

This tool deliberately contains no deployment or device integrations.  Its
runtime dependencies are imported only after command-line inputs have been
validated, so the pure comparison helpers remain useful in the server image.
"""

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


_SIDES = {"left": 0, "right": 10}


def select_episode_window(
    episode_ends: np.ndarray, episode_index: int, start_frame: int, horizon: int,
) -> slice:
    """Return an absolute window that is contained entirely in one episode."""
    ends = np.asarray(episode_ends)
    if ends.ndim != 1 or not len(ends) or not np.issubdtype(ends.dtype, np.integer):
        raise ValueError("episode_ends must be a non-empty one-dimensional integer array")
    if np.any(ends <= 0) or np.any(np.diff(ends) <= 0):
        raise ValueError("episode_ends must be strictly increasing positive values")
    if isinstance(episode_index, bool) or not isinstance(episode_index, (int, np.integer)):
        raise ValueError("episode_index must be an integer")
    if episode_index < 0 or episode_index >= len(ends):
        raise ValueError(f"episode_index {episode_index} is outside [0, {len(ends)})")
    if isinstance(start_frame, bool) or not isinstance(start_frame, (int, np.integer)) or start_frame < 0:
        raise ValueError("start_frame must be a non-negative integer")
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) or horizon <= 0:
        raise ValueError("horizon must be a positive integer")

    start = 0 if episode_index == 0 else int(ends[episode_index - 1])
    end = int(ends[episode_index])
    absolute_start = start + int(start_frame)
    if absolute_start < start or absolute_start + horizon > end:
        raise ValueError(
            f"episode {episode_index}: [{start_frame}:{start_frame + horizon}] "
            f"exceeds episode length {end - start}"
        )
    return slice(absolute_start, absolute_start + int(horizon))


def _actions(value: np.ndarray, name: str) -> np.ndarray:
    actions = np.asarray(value, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[0] == 0 or actions.shape[1] != 20 or not np.isfinite(actions).all():
        raise ValueError(f"{name}: expected equal finite [T, 20] arrays")
    return actions


def _rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Convert the two-vector rotation representation to orthonormal frames."""
    first = rot6d[..., :3]
    second = rot6d[..., 3:]
    first_norm = np.linalg.norm(first, axis=-1, keepdims=True)
    first_unit = np.divide(first, first_norm, out=np.zeros_like(first), where=first_norm > 1e-12)
    first_unit = np.where(first_norm > 1e-12, first_unit, np.array((1.0, 0.0, 0.0)))
    projected = second - np.sum(first_unit * second, axis=-1, keepdims=True) * first_unit
    second_norm = np.linalg.norm(projected, axis=-1, keepdims=True)
    second_unit = np.divide(projected, second_norm, out=np.zeros_like(projected), where=second_norm > 1e-12)
    fallback = np.where(
        np.abs(first_unit[..., :1]) < 0.9,
        np.array((1.0, 0.0, 0.0)),
        np.array((0.0, 1.0, 0.0)),
    )
    fallback -= np.sum(fallback * first_unit, axis=-1, keepdims=True) * first_unit
    fallback /= np.linalg.norm(fallback, axis=-1, keepdims=True)
    second_unit = np.where(second_norm > 1e-12, second_unit, fallback)
    third_unit = np.cross(first_unit, second_unit)
    return np.stack((first_unit, second_unit, third_unit), axis=-1)


def _rotation_errors(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    truth_matrix = _rot6d_to_matrix(truth)
    prediction_matrix = _rot6d_to_matrix(prediction)
    relative = np.swapaxes(truth_matrix, -1, -2) @ prediction_matrix
    trace = np.trace(relative, axis1=-2, axis2=-1)
    return np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))


def stage_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, dict[str, float]]:
    """Measure bimanual action error, preserving the left/right action split."""
    reference = _actions(truth, "truth")
    candidate = _actions(prediction, "prediction")
    if reference.shape != candidate.shape:
        raise ValueError("truth and prediction: expected equal finite [T, 20] arrays")

    report: dict[str, dict[str, float]] = {}
    for side, offset in _SIDES.items():
        position_delta_mm = (candidate[:, offset:offset + 3] - reference[:, offset:offset + 3]) * 1000.0
        rotation = _rotation_errors(reference[:, offset + 3:offset + 9], candidate[:, offset + 3:offset + 9])
        gripper_delta_mm = np.abs(candidate[:, offset + 9] - reference[:, offset + 9]) * 1000.0
        report[side] = {
            "position_rmse_mm": float(np.sqrt(np.mean(np.square(position_delta_mm)))),
            "rotation_geodesic_mean_rad": float(np.mean(rotation)),
            "rotation_geodesic_rmse_rad": float(np.sqrt(np.mean(np.square(rotation)))),
            "gripper_mae_mm": float(np.mean(gripper_delta_mm)),
            "gripper_max_mm": float(np.max(gripper_delta_mm)),
        }
    return report


def classify_stage(stage_a_valid: bool, at_error: float, ldp_error: float, threshold: float) -> str:
    """Attribute the first boundary whose maximum error exceeds ``threshold``."""
    values = np.asarray((at_error, ldp_error, threshold), dtype=np.float64)
    if not np.isfinite(values).all() or threshold < 0.0:
        raise ValueError("errors and threshold must be finite and threshold must be non-negative")
    if not stage_a_valid:
        return "source_or_conversion"
    if at_error > threshold:
        return "at_or_at_checkpoint"
    if ldp_error > threshold:
        return "ldp_or_observation_conditioning"
    return "training_path_consistent"


def max_stage_error(report: dict[str, dict[str, float]]) -> float:
    """Collapse position/gripper metres and rotation radians to one gate value."""
    values: list[float] = []
    for side in ("left", "right"):
        metrics = report[side]
        values.extend(
            (
                float(metrics["position_rmse_mm"]) / 1000.0,
                float(metrics["rotation_geodesic_rmse_rad"]),
                float(metrics["gripper_mae_mm"]) / 1000.0,
            )
        )
    if not np.isfinite(values).all():
        raise ValueError("stage metrics must be finite")
    return max(values)


def run_stages(policy: Any, sample: dict[str, Any], horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the checkpoint's recorded action through its AT and LDP paths.

    ``torch`` is intentionally imported here rather than at module import time.
    The method calls match the checkpoint VAE interface and run under inference
    mode so this function has no training-side effects.
    """
    torch = importlib.import_module("torch")
    with torch.inference_mode():
        truth = sample["action"][:horizon]
        tactile = sample["extended_obs"]["tactile_embedding"][:horizon].unsqueeze(0)
        normalized_action = policy.normalizer["action"].normalize(truth.unsqueeze(0))
        encoded = policy.at.encoder(policy.at.preprocess(normalized_action / policy.at.act_scale))
        if policy.at.use_vq:
            latent, _, _ = policy.at.quant_state_with_vq(encoded)
        else:
            latent, _ = policy.at.quant_state_without_vq(encoded)
            latent = policy.at.postprocess_quant_state_without_vq(latent)
        temporal_cond = policy.at.get_temporal_cond({"tactile_embedding": tactile}).to(latent.device)
        reconstructed = policy.at.get_action_from_latent_with_temporal_cond(latent, temporal_cond)
        reconstructed = policy.normalizer["action"].unnormalize(reconstructed)[0]
        predicted_latent = policy.predict_action(
            sample["obs"],
            dataset_obs_temporal_downsample_ratio=1,
            return_latent_action=True,
        )["action"][:, 0]
        predicted = policy.predict_from_latent_action(
            predicted_latent,
            {"tactile_embedding": tactile},
            extended_obs_last_step=horizon,
            dataset_obs_temporal_downsample_ratio=1,
        )["action_pred"][0, :horizon]
        return tuple(value.detach().float().cpu().numpy() for value in (truth, reconstructed, predicted))


def _existing_path(path: Path, label: str, *, directory: bool = False) -> Path:
    if (path.is_dir() if directory else path.is_file()):
        return path.resolve()
    expected = "directory" if directory else "file"
    raise ValueError(f"{label} must be an existing {expected}: {path}")


def _resolve_repo_path(vitamin_repo: Path, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return (path if path.is_absolute() else vitamin_repo / path).resolve()


def _load_policy(vitamin_repo: Path, config_path: Path, device_name: str | None) -> tuple[Any, Any, Any]:
    """Use the deployed checkpoint loader without constructing its bridge."""
    sys.path.insert(0, str(vitamin_repo))
    torch = importlib.import_module("torch")
    deploy = importlib.import_module("deploy_pick_tube_rdp")
    config = deploy.load_config(config_path)
    model = config["model"]
    device = torch.device(device_name or str(model.get("device", "cuda:0")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    ldp_checkpoint = _resolve_repo_path(vitamin_repo, model["ldp_checkpoint"])
    at_checkpoint = _resolve_repo_path(vitamin_repo, model["at_checkpoint"])
    for path in (ldp_checkpoint, at_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    policy, checkpoint_config = deploy.load_policy(
        ldp_checkpoint,
        at_checkpoint,
        device,
        int(model.get("num_inference_steps", 8)),
    )
    return policy, checkpoint_config, device


def _open_replay_buffer(dataset_path: Path) -> Any:
    zarr = importlib.import_module("zarr")
    replay_path = dataset_path / "replay_buffer.zarr" if (dataset_path / "replay_buffer.zarr").is_dir() else dataset_path
    root = zarr.open_group(str(replay_path), mode="r")
    for key in (
        "data/action",
        "data/camera1",
        "data/camera2",
        "data/observation_state",
        "data/tactile_embedding",
        "meta/episode_ends",
    ):
        if key not in root:
            raise ValueError(f"{replay_path}: missing {key}")
    return root


def _load_sample(root: Any, window: slice, device: Any) -> dict[str, Any]:
    """Build the exact tensor contract used by the deployed policy."""
    torch = importlib.import_module("torch")
    start = int(window.start)
    action = np.asarray(root["data/action"][window], dtype=np.float32)
    tactile = np.asarray(root["data/tactile_embedding"][window], dtype=np.float32)
    state = np.asarray(root["data/observation_state"][start], dtype=np.float32)
    camera1 = np.asarray(root["data/camera1"][start], dtype=np.uint8)
    camera2 = np.asarray(root["data/camera2"][start], dtype=np.uint8)
    expected_horizon = int(window.stop - window.start)
    if action.shape != (expected_horizon, 20):
        raise ValueError(f"action window must be ({expected_horizon}, 20), got {action.shape}")
    if tactile.shape != (expected_horizon, 2048):
        raise ValueError(f"tactile window must be ({expected_horizon}, 2048), got {tactile.shape}")
    if state.shape != (20,) or camera1.shape != (224, 224, 3) or camera2.shape != camera1.shape:
        raise ValueError("dataset must contain state (20,) and two RGB images (224,224,3)")
    if not np.isfinite(action).all() or not np.isfinite(tactile).all() or not np.isfinite(state).all():
        raise ValueError("dataset action, tactile, and state values must be finite")

    def camera_tensor(image: np.ndarray) -> Any:
        return torch.from_numpy(np.ascontiguousarray(image)).to(device).permute(2, 0, 1).float().div_(255.0).reshape(1, 1, 3, 224, 224)

    tactile_tensor = torch.from_numpy(tactile).to(device).unsqueeze(0)
    return {
        "action": torch.from_numpy(action).to(device),
        "extended_obs": {"tactile_embedding": tactile_tensor[0]},
        "obs": {
            "camera1": camera_tensor(camera1),
            "camera2": camera_tensor(camera2),
            "observation_state": torch.from_numpy(state).to(device).reshape(1, 1, 20),
            "tactile_embedding": tactile_tensor[:, :1],
        },
    }


def _write_report(output: Path | None, report: dict[str, Any]) -> None:
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if output is None:
        print(rendered, end="")
        return
    with output.open("x", encoding="utf-8") as stream:
        stream.write(rendered)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare offline Vitamin source, AT, and LDP policy stages")
    parser.add_argument("--vitamin-repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.001)
    parser.add_argument("--device")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    vitamin_repo = _existing_path(args.vitamin_repo, "vitamin repository", directory=True)
    config_path = _existing_path(args.config, "deployment config")
    dataset_path = _existing_path(args.dataset, "converted dataset", directory=True)
    if args.horizon != 20:
        raise ValueError("this comparator requires the recorded 20-step horizon")
    if not np.isfinite(args.threshold) or args.threshold < 0.0:
        raise ValueError("threshold must be a finite non-negative value")

    group = _open_replay_buffer(dataset_path)
    episode_ends = np.asarray(group["meta/episode_ends"])
    window = select_episode_window(episode_ends, args.episode, args.start_frame, args.horizon)
    policy, checkpoint_config, device = _load_policy(vitamin_repo, config_path, args.device)
    if int(checkpoint_config.dataset_obs_temporal_downsample_ratio) != 1:
        raise ValueError("comparator requires dataset_obs_temporal_downsample_ratio=1")
    sample = _load_sample(group, window, device)
    truth, reconstructed, predicted = run_stages(policy, sample, args.horizon)
    at_report = stage_metrics(truth, reconstructed)
    ldp_report = stage_metrics(truth, predicted)
    at_error = max_stage_error(at_report)
    ldp_error = max_stage_error(ldp_report)
    _write_report(args.output, {
        "window": {"start": window.start, "stop": window.stop},
        "action_layout": {"left": "[0:10]", "right": "[10:20]"},
        "at": at_report,
        "ldp": ldp_report,
        "scalar_error": {"at": at_error, "ldp": ldp_error, "threshold": args.threshold},
        "classification": classify_stage(True, at_error, ldp_error, args.threshold),
    })


if __name__ == "__main__":
    main()

"""Episode splitting and physical validation metrics for pick-tube v2."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import einops
import numpy as np
import torch
from omegaconf import OmegaConf

from reactive_diffusion_policy.common.artifact_manifest import stable_json_digest
from reactive_diffusion_policy.common.pick_tube_action_contract import (
    HIGH_ROTATION_DELTA_DEG,
    HIGH_TRANSLATION_DELTA_M,
    LOW_ROTATION_DELTA_DEG,
    LOW_TRANSLATION_DELTA_M,
)


_ARM_LAYOUT = {
    "left": (slice(0, 3), slice(3, 9)),
    "right": (slice(10, 13), slice(13, 19)),
}


def build_episode_split_manifest(
    episode_sources: Sequence[int] | np.ndarray,
    *,
    val_ratio: float,
    seed: int,
) -> dict:
    """Return a deterministic source-stratified episode split."""
    sources = np.asarray(episode_sources)
    if sources.ndim != 1 or sources.size == 0:
        raise ValueError("episode_sources must be a non-empty one-dimensional array")
    if not 0.0 <= float(val_ratio) < 1.0:
        raise ValueError("val_ratio must be in [0, 1)")

    validation_ids: list[int] = []
    rng = np.random.default_rng(int(seed))
    if val_ratio > 0:
        for source in sorted(np.unique(sources).tolist()):
            source_ids = np.flatnonzero(sources == source)
            if source_ids.size < 2:
                raise ValueError(
                    f"source {source!r} needs at least two episodes for a held-out split"
                )
            requested = max(1, int(round(source_ids.size * float(val_ratio))))
            count = min(requested, source_ids.size - 1)
            validation_ids.extend(
                int(value)
                for value in np.sort(rng.choice(source_ids, size=count, replace=False))
            )

    validation_ids = sorted(validation_ids)
    validation_set = set(validation_ids)
    train_ids = [
        episode_id
        for episode_id in range(int(sources.size))
        if episode_id not in validation_set
    ]
    identity = {
        "seed": int(seed),
        "val_ratio": float(val_ratio),
        "episode_sources": sources.tolist(),
        "train_episode_ids": train_ids,
        "validation_episode_ids": validation_ids,
    }
    return {**identity, "split_digest": stable_json_digest(identity)}


def _as_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.float()
        return tensor.numpy()
    return np.asarray(value)


def _rotation_matrix(rotation_6d: np.ndarray) -> np.ndarray:
    first = np.asarray(rotation_6d[:3], dtype=np.float64)
    second = np.asarray(rotation_6d[3:], dtype=np.float64)
    first_norm = np.linalg.norm(first)
    if not np.isfinite(first).all() or first_norm < 1e-12:
        first = np.asarray([1.0, 0.0, 0.0])
    else:
        first = first / first_norm
    second = second - np.dot(first, second) * first
    second_norm = np.linalg.norm(second)
    if not np.isfinite(second).all() or second_norm < 1e-12:
        axis = np.eye(3)[int(np.argmin(np.abs(first)))]
        second = axis - np.dot(first, axis) * first
        second_norm = np.linalg.norm(second)
    second = second / second_norm
    third = np.cross(first, second)
    return np.stack((first, second, third), axis=-2)


def _geodesic_degrees(first: np.ndarray, second: np.ndarray) -> float:
    relative = first.T @ second
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _mean_or_nan(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _p95_or_nan(values: Sequence[float]) -> float:
    return float(np.percentile(values, 95)) if values else float("nan")


def compute_idle_rollout_metrics(
    target,
    prediction,
    idle_mask,
    horizon: int = 29,
    *,
    valid_mask=None,
) -> dict[str, float]:
    """Measure bimanual physical errors from unnormalized 20D actions."""
    target_array = _as_numpy(target).astype(np.float64, copy=False)
    prediction_array = _as_numpy(prediction).astype(np.float64, copy=False)
    idle_array = _as_numpy(idle_mask).astype(bool, copy=False)
    if target_array.shape != prediction_array.shape or target_array.shape[-1] != 20:
        raise ValueError("target and prediction must have matching [..., T, 20] shapes")
    if target_array.ndim == 2:
        target_array = target_array[None]
        prediction_array = prediction_array[None]
    if target_array.ndim != 3:
        raise ValueError("target and prediction must have shape [B, T, 20]")
    if idle_array.ndim == 2:
        idle_array = idle_array[None]
    if idle_array.shape != (*target_array.shape[:-1], 2):
        raise ValueError("idle_mask must have shape [B, T, 2]")
    if horizon < 1 or horizon > target_array.shape[-2]:
        raise ValueError("horizon must be positive and no longer than the action sequence")

    if valid_mask is None:
        valid_array = np.ones(target_array.shape[:-1], dtype=bool)
    else:
        valid_array = _as_numpy(valid_mask).astype(bool, copy=False)
        if valid_array.ndim == 1:
            valid_array = valid_array[None]
        if valid_array.shape != target_array.shape[:-1]:
            raise ValueError("valid_mask must have shape [B, T]")

    start = target_array.shape[-2] - int(horizon)
    target_array = target_array[:, start:]
    prediction_array = prediction_array[:, start:]
    idle_array = idle_array[:, start:]
    valid_array = valid_array[:, start:]

    integrated_translation: dict[str, list[float]] = {
        arm: [] for arm in _ARM_LAYOUT
    }
    integrated_rotation: dict[str, list[float]] = {
        arm: [] for arm in _ARM_LAYOUT
    }
    idle_step_translation: dict[str, list[float]] = {
        arm: [] for arm in _ARM_LAYOUT
    }
    idle_step_rotation: dict[str, list[float]] = {
        arm: [] for arm in _ARM_LAYOUT
    }
    active_translation: dict[str, list[float]] = {
        arm: [] for arm in _ARM_LAYOUT
    }
    active_rotation: dict[str, list[float]] = {
        arm: [] for arm in _ARM_LAYOUT
    }
    micro_target_count = 0
    micro_predicted_count = 0
    predicted_micro_count = 0
    true_predicted_micro_count = 0

    for arm_index, (arm, (position_slice, rotation_slice)) in enumerate(
        _ARM_LAYOUT.items()
    ):
        for batch_index in range(target_array.shape[0]):
            target_total_rotation = np.eye(3)
            prediction_total_rotation = np.eye(3)
            target_total_translation = np.zeros(3)
            prediction_total_translation = np.zeros(3)
            has_idle = False
            is_idle_rollout = bool(
                np.all(
                    valid_array[batch_index]
                    & idle_array[batch_index, :, arm_index]
                )
            )
            for time_index in range(target_array.shape[1]):
                if not valid_array[batch_index, time_index]:
                    continue
                target_position = target_array[batch_index, time_index, position_slice]
                predicted_position = prediction_array[
                    batch_index, time_index, position_slice
                ]
                target_rotation = _rotation_matrix(
                    target_array[batch_index, time_index, rotation_slice]
                )
                predicted_rotation = _rotation_matrix(
                    prediction_array[batch_index, time_index, rotation_slice]
                )
                translation_error = float(
                    np.linalg.norm(predicted_position - target_position)
                )
                rotation_error = _geodesic_degrees(target_rotation, predicted_rotation)
                if idle_array[batch_index, time_index, arm_index]:
                    has_idle = True
                    target_total_translation += target_position
                    prediction_total_translation += predicted_position
                    target_total_rotation = target_total_rotation @ target_rotation
                    prediction_total_rotation = (
                        prediction_total_rotation @ predicted_rotation
                    )
                    idle_step_translation[arm].append(translation_error * 1000.0)
                    idle_step_rotation[arm].append(rotation_error)
                else:
                    active_translation[arm].append(translation_error * 1000.0)
                    active_rotation[arm].append(rotation_error)
                    target_motion_translation = float(np.linalg.norm(target_position))
                    target_motion_rotation = _geodesic_degrees(
                        np.eye(3), target_rotation
                    )
                    predicted_motion_translation = float(
                        np.linalg.norm(predicted_position)
                    )
                    predicted_motion_rotation = _geodesic_degrees(
                        np.eye(3), predicted_rotation
                    )
                    target_is_micro = (
                        (
                            target_motion_translation >= LOW_TRANSLATION_DELTA_M
                            or target_motion_rotation >= LOW_ROTATION_DELTA_DEG
                        )
                        and target_motion_translation <= HIGH_TRANSLATION_DELTA_M
                        and target_motion_rotation <= HIGH_ROTATION_DELTA_DEG
                    )
                    predicted_is_micro = (
                        predicted_motion_translation >= LOW_TRANSLATION_DELTA_M
                        or predicted_motion_rotation >= LOW_ROTATION_DELTA_DEG
                    )
                    if target_is_micro:
                        micro_target_count += 1
                        micro_predicted_count += int(predicted_is_micro)
                    if predicted_is_micro:
                        predicted_micro_count += 1
                        true_predicted_micro_count += int(target_is_micro)
            if has_idle and is_idle_rollout:
                integrated_translation[arm].append(
                    float(
                        np.linalg.norm(
                            prediction_total_translation - target_total_translation
                        )
                        * 1000.0
                    )
                )
                integrated_rotation[arm].append(
                    _geodesic_degrees(
                        target_total_rotation, prediction_total_rotation
                    )
                )

    all_integrated_translation = sum(integrated_translation.values(), [])
    all_integrated_rotation = sum(integrated_rotation.values(), [])
    all_idle_step_translation = sum(idle_step_translation.values(), [])
    all_idle_step_rotation = sum(idle_step_rotation.values(), [])
    metrics: dict[str, float] = {
        "val_idle_translation_29_mm": _mean_or_nan(all_integrated_translation),
        "val_idle_rotation_29_deg": _mean_or_nan(all_integrated_rotation),
        "val_idle_translation_step_p95_mm": _p95_or_nan(
            all_idle_step_translation
        ),
        "val_idle_rotation_step_p95_deg": _p95_or_nan(all_idle_step_rotation),
        "val_micro_motion_recall": (
            float(micro_predicted_count / micro_target_count)
            if micro_target_count
            else float("nan")
        ),
        "val_micro_motion_precision": (
            float(true_predicted_micro_count / predicted_micro_count)
            if predicted_micro_count
            else float("nan")
        ),
    }
    metrics["val_idle_translation_p95_mm"] = metrics[
        "val_idle_translation_step_p95_mm"
    ]
    metrics["val_idle_rotation_p95_deg"] = metrics[
        "val_idle_rotation_step_p95_deg"
    ]
    for arm in _ARM_LAYOUT:
        metrics.update(
            {
                f"val_idle_{arm}_translation_29_mm": _mean_or_nan(
                    integrated_translation[arm]
                ),
                f"val_idle_{arm}_rotation_29_deg": _mean_or_nan(
                    integrated_rotation[arm]
                ),
                f"val_idle_{arm}_translation_step_p95_mm": _p95_or_nan(
                    idle_step_translation[arm]
                ),
                f"val_idle_{arm}_rotation_step_p95_deg": _p95_or_nan(
                    idle_step_rotation[arm]
                ),
                f"val_active_{arm}_translation_mae_mm": _mean_or_nan(
                    active_translation[arm]
                ),
                f"val_active_{arm}_rotation_mae_deg": _mean_or_nan(
                    active_rotation[arm]
                ),
            }
        )
    metrics["val_active_translation_mae_mm"] = _mean_or_nan(
        sum(active_translation.values(), [])
    )
    metrics["val_active_rotation_mae_deg"] = _mean_or_nan(
        sum(active_rotation.values(), [])
    )
    return metrics


def evaluate_checkpoint_feasibility(
    *,
    idle_translation_29_mm: float,
    idle_rotation_29_deg: float,
    active_metric: float,
    active_baseline: float | None,
    micro_motion_recall: float,
    max_active_degradation: float = 0.05,
    min_micro_motion_recall: float = 0.95,
) -> dict[str, float | bool]:
    """Apply hard gates and calculate the idle score used by top-k selection."""
    score = float(idle_translation_29_mm) + float(idle_rotation_29_deg) / 0.5
    usable_baseline = (
        active_baseline is not None
        and math.isfinite(float(active_baseline))
        and float(active_baseline) > 0
    )
    degradation = (
        float(active_metric) / float(active_baseline) - 1.0
        if usable_baseline
        else float("inf")
    )
    feasible = bool(
        usable_baseline
        and math.isfinite(score)
        and math.isfinite(float(micro_motion_recall))
        and degradation <= float(max_active_degradation) + 1e-12
        and float(micro_motion_recall) >= float(min_micro_motion_recall)
    )
    return {
        "val_active_degradation": degradation,
        "val_micro_motion_recall": float(micro_motion_recall),
        "val_idle_score": score,
        "val_checkpoint_feasible": feasible,
        "val_deployable": feasible,
    }


def _select(config, key: str):
    if OmegaConf.is_config(config):
        return OmegaConf.select(config, key)
    current = config
    for part in key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _action_contract_identity(config) -> tuple[object, object]:
    version = _select(config, "task.action_representation_version")
    contract = _select(config, "task.action_contract")
    if version is None:
        version = _select(config, "artifacts.action_representation_version")
    if contract is None:
        contract = _select(config, "artifacts.action_contract")
    return version, contract


def validate_resume_action_contract(current_config, checkpoint_config) -> None:
    """Reject a checkpoint whose action contract differs from this run."""
    current = _action_contract_identity(current_config)
    checkpoint = _action_contract_identity(checkpoint_config)
    if current == (None, None) and checkpoint == (None, None):
        return
    if current != checkpoint:
        raise ValueError(
            "cannot resume across action-contract version boundaries: "
            f"current={current!r}, checkpoint={checkpoint!r}"
        )


def reconstruct_at_actions(model, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Decode deterministic posterior-mode AT reconstructions in physical units."""
    normalized = model.normalizer["action"].normalize(batch["action"])
    state_representation = model.encoder(model.preprocess(normalized / model.act_scale))
    if model.use_vq:
        latent, _, _ = model.quant_state_with_vq(state_representation)
    else:
        latent, _ = model.quant_state_without_vq(
            state_representation,
            sample=False,
        )
        latent = model.postprocess_quant_state_without_vq(latent)
    if model.use_rnn_decoder:
        temporal = model.get_temporal_cond(batch["extended_obs"]).to(model.device)
        decoded = model.decoder(latent, temporal)
    else:
        decoded = model.decoder(latent)
    normalized_prediction = einops.rearrange(
        decoded,
        "N (T A) -> N T A",
        T=model.input_dim_h,
        A=model.input_dim_w,
    ) * model.act_scale
    return model.normalizer["action"].unnormalize(normalized_prediction)

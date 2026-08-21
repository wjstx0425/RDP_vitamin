import math
from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from reactive_diffusion_policy.common.pick_tube_validation import (
    build_episode_split_manifest,
    compute_idle_rollout_metrics,
    evaluate_checkpoint_feasibility,
    validate_resume_action_contract,
)


IDENTITY_6D = np.asarray([1, 0, 0, 0, 1, 0], dtype=np.float64)


def _rotation_6d_z(degrees):
    angle = math.radians(degrees)
    return np.asarray(
        [math.cos(angle), math.sin(angle), 0, -math.sin(angle), math.cos(angle), 0],
        dtype=np.float64,
    )


def _neutral_actions(batch=1, horizon=29):
    actions = np.zeros((batch, horizon, 20), dtype=np.float64)
    actions[..., 3:9] = IDENTITY_6D
    actions[..., 13:19] = IDENTITY_6D
    return actions


def test_episode_split_is_deterministic_stratified_and_disjoint():
    sources = np.repeat(np.arange(6), 10)

    first = build_episode_split_manifest(sources, val_ratio=0.1, seed=42)
    second = build_episode_split_manifest(sources, val_ratio=0.1, seed=42)

    assert first == second
    assert len(first["validation_episode_ids"]) == 6
    assert set(first["train_episode_ids"]).isdisjoint(first["validation_episode_ids"])
    assert sorted(first["train_episode_ids"] + first["validation_episode_ids"]) == list(
        range(60)
    )
    assert {sources[index] for index in first["validation_episode_ids"]} == set(range(6))
    assert len(first["split_digest"]) == 64


def test_idle_metrics_integrate_translation_and_compose_so3_rotations():
    target = _neutral_actions()
    prediction = target.copy()
    prediction[..., 10] = 0.001
    prediction[..., 13:19] = _rotation_6d_z(1.0)
    idle_mask = np.zeros((1, 29, 2), dtype=bool)
    idle_mask[..., 1] = True

    metrics = compute_idle_rollout_metrics(target, prediction, idle_mask, horizon=29)

    assert metrics["val_idle_translation_29_mm"] == pytest.approx(29.0)
    assert metrics["val_idle_rotation_29_deg"] == pytest.approx(29.0, abs=1e-5)
    assert metrics["val_idle_translation_step_p95_mm"] == pytest.approx(1.0)
    assert metrics["val_idle_rotation_step_p95_deg"] == pytest.approx(1.0)
    assert metrics["val_idle_right_translation_29_mm"] == pytest.approx(29.0)
    assert math.isnan(metrics["val_idle_left_translation_29_mm"])


def test_metrics_report_active_errors_and_micro_motion_recall():
    target = _neutral_actions(horizon=4)
    prediction = target.copy()
    idle_mask = np.ones((1, 4, 2), dtype=bool)
    idle_mask[..., 0] = False
    target[..., 0] = 0.0006
    prediction[:, :3, 0] = 0.0006
    prediction[:, 3, 0] = 0.0

    metrics = compute_idle_rollout_metrics(target, prediction, idle_mask, horizon=4)

    assert metrics["val_active_left_translation_mae_mm"] == pytest.approx(0.15)
    assert metrics["val_micro_motion_recall"] == pytest.approx(0.75)


@pytest.mark.parametrize(
    ("active", "baseline", "recall", "feasible"),
    [(1.05, 1.0, 0.95, True), (1.051, 1.0, 0.95, False), (1.0, 1.0, 0.949, False)],
)
def test_checkpoint_feasibility_enforces_active_and_micro_motion_limits(
    active, baseline, recall, feasible
):
    result = evaluate_checkpoint_feasibility(
        idle_translation_29_mm=0.4,
        idle_rotation_29_deg=0.1,
        active_metric=active,
        active_baseline=baseline,
        micro_motion_recall=recall,
    )

    assert result["val_active_degradation"] == pytest.approx(active / baseline - 1.0)
    assert result["val_idle_score"] == pytest.approx(0.6)
    assert result["val_checkpoint_feasible"] is feasible
    assert result["val_deployable"] is feasible


def test_missing_active_baseline_is_not_deployable():
    result = evaluate_checkpoint_feasibility(
        idle_translation_29_mm=0.1,
        idle_rotation_29_deg=0.1,
        active_metric=0.0,
        active_baseline=None,
        micro_motion_recall=1.0,
    )

    assert result["val_checkpoint_feasible"] is False
    assert result["val_deployable"] is False


def test_resume_rejects_action_contract_version_mismatch():
    current = OmegaConf.create(
        {"task": {"action_representation_version": 2, "action_contract": "v2"}}
    )
    legacy = OmegaConf.create(
        {"task": {"action_representation_version": 1, "action_contract": "v1"}}
    )

    validate_resume_action_contract(current, current)
    with pytest.raises(ValueError, match="action-contract version"):
        validate_resume_action_contract(current, legacy)


def test_pick_tube_v2_configs_select_feasible_idle_score():
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    config_dir = str(
        Path(__file__).resolve().parents[1] / "reactive_diffusion_policy" / "config"
    )
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        at_cfg = compose(config_name="train_pick_tube_at_workspace")
        ldp_cfg = compose(config_name="train_pick_tube_ldp_workspace")

    for cfg in (at_cfg, ldp_cfg):
        assert cfg.task.dataset.val_ratio == 0.1
        assert cfg.task.action_representation_version == 2
        assert cfg.task.action_contract == "bimanual_relative_pose20d_v2"
        assert cfg.checkpoint.topk.monitor_key == "val_idle_score"
        assert cfg.checkpoint.topk.mode == "min"
        assert cfg.validation.max_active_degradation == 0.05
        assert cfg.validation.min_micro_motion_recall == 0.95


def test_v2_experiment_launcher_uses_fresh_20_epoch_runs():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_pick_tube_rdp_experiments.sh"
    ).read_text(encoding="utf-8")

    assert '"AT_EPOCHS=20"' in script
    assert '"LDP_EPOCHS=20"' in script
    assert '"RESUME=false"' in script
    assert "pca%d_armwise_rdp_zarr_v2" in script

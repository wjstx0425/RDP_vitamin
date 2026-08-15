import numpy as np
import pytest
from pathlib import Path
import subprocess
import sys

from tools.rdp_debug.compare_policy_stages import classify_stage
from tools.rdp_debug.compare_policy_stages import max_stage_error
from tools.rdp_debug.compare_policy_stages import select_episode_window
from tools.rdp_debug.compare_policy_stages import stage_metrics


def test_episode_window_never_crosses_episode_boundary() -> None:
    ends = np.array([25, 50])
    window = select_episode_window(ends, episode_index=1, start_frame=3, horizon=20)
    assert window == slice(28, 48)
    with pytest.raises(ValueError, match="exceeds episode"):
        select_episode_window(ends, episode_index=0, start_frame=10, horizon=20)


def test_stage_metrics_preserve_left_right_slices() -> None:
    truth = np.zeros((20, 20), dtype=np.float64)
    prediction = truth.copy()
    prediction[:, 10] = 0.002
    report = stage_metrics(truth, prediction)
    assert report["left"]["position_rmse_mm"] == 0.0
    assert report["right"]["position_rmse_mm"] == pytest.approx(2.0 / np.sqrt(3.0))
    assert report["left"]["rotation_geodesic_mean_rad"] == 0.0
    assert report["right"]["rotation_geodesic_rmse_rad"] == 0.0


def test_stage_metrics_rejects_nonfinite_or_wrong_shapes() -> None:
    valid = np.zeros((20, 20), dtype=np.float64)
    with pytest.raises(ValueError, match="equal finite \\[T, 20\\] arrays"):
        stage_metrics(valid, valid[:, :-1])
    valid[0, 0] = np.nan
    with pytest.raises(ValueError, match="equal finite \\[T, 20\\] arrays"):
        stage_metrics(valid, valid)


def test_classification_assigns_first_failed_boundary() -> None:
    assert classify_stage(False, 0.0, 0.0, 0.001) == "source_or_conversion"
    assert classify_stage(True, 0.01, 0.0, 0.001) == "at_or_at_checkpoint"
    assert classify_stage(True, 0.0, 0.01, 0.001) == "ldp_or_observation_conditioning"
    assert classify_stage(True, 0.0, 0.0, 0.001) == "training_path_consistent"


def test_max_stage_error_includes_position_rotation_and_gripper() -> None:
    report = {
        "left": {
            "position_rmse_mm": 2.0,
            "rotation_geodesic_rmse_rad": 0.003,
            "gripper_mae_mm": 4.0,
        },
        "right": {
            "position_rmse_mm": 1.0,
            "rotation_geodesic_rmse_rad": 0.002,
            "gripper_mae_mm": 5.0,
        },
    }
    assert max_stage_error(report) == pytest.approx(0.005)


def test_comparator_source_is_hardware_isolated() -> None:
    source = Path("tools/rdp_debug/compare_policy_stages.py").read_text(encoding="utf-8")
    for forbidden in ("RobotBridgeClient", "websockets", "requests", "BimanualUmiEnv", "/dev/video"):
        assert forbidden not in source


def test_cli_help_matches_runbook_contract() -> None:
    result = subprocess.run(
        [sys.executable, "tools/rdp_debug/compare_policy_stages.py", "--help"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    for option in ("--vitamin-repo", "--config", "--dataset", "--episode", "--output"):
        assert option in result.stdout

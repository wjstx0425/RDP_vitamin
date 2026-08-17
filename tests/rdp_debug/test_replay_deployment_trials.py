import json
from pathlib import Path

import numpy as np
import pytest

from tools.rdp_debug.replay_deployment_trials import action_metrics
from tools.rdp_debug.replay_deployment_trials import axis_angle_to_matrix
from tools.rdp_debug.replay_deployment_trials import image_directory
from tools.rdp_debug.replay_deployment_trials import load_trial_rows


def test_axis_angle_rotation_maps_local_x_to_base_y() -> None:
    rotation = axis_angle_to_matrix(np.array([0.0, 0.0, np.pi / 2.0]))
    assert rotation @ np.array([1.0, 0.0, 0.0]) == pytest.approx([0.0, 1.0, 0.0])


def test_action_metrics_reports_local_and_base_directions() -> None:
    action = np.zeros(20)
    action[0] = 0.002
    action[9] = 0.12
    action[10:13] = [0.0, 0.0, -0.003]
    pose = {
        "left": {"rotation_axis_angle": [0.0, 0.0, np.pi / 2.0]},
        "right": {"rotation_axis_angle": [0.0, 0.0, 0.0]},
    }

    metrics = action_metrics(action, pose)

    assert metrics["left"]["local_translation_mm"] == pytest.approx([2.0, 0.0, 0.0])
    assert metrics["left"]["base_translation_mm"] == pytest.approx([0.0, 2.0, 0.0])
    assert metrics["right"]["base_translation_mm"] == pytest.approx([0.0, 0.0, -3.0])


def test_trial_rows_require_contiguous_steps_and_matching_manifest(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text(json.dumps({"step_count": 2}), encoding="utf-8")
    with (tmp_path / "steps.jsonl").open("w", encoding="utf-8") as stream:
        stream.write(json.dumps({"iter_idx": 0}) + "\n")
        stream.write(json.dumps({"iter_idx": 1}) + "\n")

    manifest, rows = load_trial_rows(tmp_path)

    assert manifest["step_count"] == 2
    assert list(rows) == [0, 1]


def test_image_directory_uses_initial_then_zero_padded_steps(tmp_path: Path) -> None:
    assert image_directory(tmp_path, 0) == tmp_path / "images" / "initial"
    assert image_directory(tmp_path, 15) == tmp_path / "images" / "step_000015"

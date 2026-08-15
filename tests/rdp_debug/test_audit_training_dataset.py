from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from tools.rdp_debug.audit_training_dataset import DatasetArrays
from tools.rdp_debug.audit_training_dataset import audit_dataset
from tools.rdp_debug.audit_training_dataset import load_dataset
from tools.rdp_debug.audit_training_dataset import scan_lag


def synthetic_data() -> DatasetArrays:
    action = np.zeros((12, 20), dtype=np.float64)
    state = np.zeros((12, 20), dtype=np.float64)
    action[1:4, 0] = 0.004
    action[7:10, 10] = 0.003
    state[2:5, 0] = np.cumsum(action[1:4, 0])
    state[8:11, 7] = np.cumsum(action[7:10, 10])
    return DatasetArrays(action=action, state=state, episode_ends=np.array([6, 12]))


def test_audit_identifies_first_moving_side_per_episode() -> None:
    report = audit_dataset(synthetic_data(), start_windows=(3, 6), movement_threshold=0.001, max_lag=3)
    assert report["episodes"][0]["first_moving_side"] == "left"
    assert report["episodes"][1]["first_moving_side"] == "right"


def test_lag_scan_recovers_one_frame_state_response() -> None:
    motion = np.array([0.0, 1.0, 2.0, 0.0, 0.0])
    response = np.array([0.0, 0.0, 1.0, 2.0, 0.0])
    result = scan_lag(motion, response, max_lag=2)
    assert result["best_lag_frames"] == 1
    assert result["correlation"] == pytest.approx(1.0)


def test_load_dataset_rejects_unknown_format(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="source_format"):
        load_dataset(tmp_path, "unknown")


def test_cli_help_does_not_require_optional_readers() -> None:
    result = subprocess.run(
        [sys.executable, "tools/rdp_debug/audit_training_dataset.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Audit ordered pick-tube training actions and states" in result.stdout

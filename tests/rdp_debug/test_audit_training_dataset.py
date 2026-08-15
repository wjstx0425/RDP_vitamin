from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import pytest

from tools.rdp_debug.audit_training_dataset import DatasetArrays
from tools.rdp_debug.audit_training_dataset import audit_dataset
from tools.rdp_debug.audit_training_dataset import load_dataset
from tools.rdp_debug.audit_training_dataset import load_lerobot
from tools.rdp_debug.audit_training_dataset import scan_lag
from tools.rdp_debug.audit_training_dataset import validate_arrays


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


@pytest.mark.parametrize("state_columns", [10, 21])
def test_validate_arrays_requires_exactly_twenty_state_columns(state_columns: int) -> None:
    with pytest.raises(ValueError, match=r"state shape \(frames, 20\)"):
        validate_arrays(
            np.zeros((2, 20)),
            np.zeros((2, state_columns)),
            np.array([2]),
            Path("dataset"),
        )


def test_audit_reports_stable_action_arm_mapping() -> None:
    report = audit_dataset(synthetic_data(), start_windows=(3,), movement_threshold=0.001, max_lag=3)
    assert report["action_arm_mapping"] == {"left": "[0:10]", "right": "[10:20]"}


def test_load_lerobot_orders_numeric_chunk_and_episode_indices(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "episodes.jsonl").write_text("{}\n", encoding="utf-8")
    paths = [
        root / "data" / "chunk-10" / "episode_1.parquet",
        root / "data" / "chunk-2" / "episode_12.parquet",
        root / "data" / "chunk-2" / "episode_3.parquet",
    ]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    values = {str(path): value for path, value in zip(paths, (101.0, 212.0, 203.0), strict=True)}

    class Column:
        def __init__(self, rows: list[list[float]]) -> None:
            self.rows = rows

        def to_pylist(self) -> list[list[float]]:
            return self.rows

    class Table:
        def __init__(self, marker: float) -> None:
            self.marker = marker

        def __getitem__(self, column: str) -> Column:
            return Column([[self.marker] + [0.0] * 19])

    parquet_module = types.ModuleType("pyarrow.parquet")
    parquet_module.read_table = lambda path, columns: Table(values[str(path)])  # type: ignore[attr-defined]
    pyarrow_module = types.ModuleType("pyarrow")
    pyarrow_module.parquet = parquet_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyarrow", pyarrow_module)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", parquet_module)

    data = load_lerobot(root)
    assert data.action[:, 0].tolist() == [203.0, 212.0, 101.0]
    assert data.episode_ends.tolist() == [1, 2, 3]


def test_start_window_aggregates_each_episode_prefix() -> None:
    action = np.zeros((6, 20), dtype=np.float64)
    action[0, 0] = 0.001
    action[3, 0] = 0.003
    data = DatasetArrays(action=action, state=np.zeros((6, 20)), episode_ends=np.array([3, 6]))
    report = audit_dataset(data, start_windows=(1,), movement_threshold=0.001, max_lag=1)
    assert report["start_windows"]["1"]["left"]["position_rms_mm_per_step"] == pytest.approx(np.sqrt(5.0))


def test_audit_rejects_boolean_movement_threshold() -> None:
    with pytest.raises(ValueError, match="movement_threshold"):
        audit_dataset(synthetic_data(), start_windows=(3,), movement_threshold=True, max_lag=3)

import numpy as np
import pytest

from real_world.bimanual_umi_env import _select_alignment_camera_idx


def _camera_data(*timestamp_sequences: list[float]) -> dict[int, dict[str, np.ndarray]]:
    return {
        camera_idx: {"timestamp": np.asarray(timestamps, dtype=np.float64)}
        for camera_idx, timestamps in enumerate(timestamp_sequences)
    }


def test_alignment_skips_candidate_older_than_other_camera_buffer() -> None:
    camera_data = _camera_data([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])

    assert _select_alignment_camera_idx(camera_data, num_obs_cameras=2) == 1


def test_alignment_accepts_equal_timestamps() -> None:
    camera_data = _camera_data([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])

    assert _select_alignment_camera_idx(camera_data, num_obs_cameras=2) == 0


def test_alignment_preserves_latest_not_after_candidate_policy() -> None:
    camera_data = _camera_data([1.0, 2.0, 3.0], [2.95, 3.05, 3.10])

    assert _select_alignment_camera_idx(camera_data, num_obs_cameras=2) == 0


def test_alignment_rejects_empty_timestamp_buffer_clearly() -> None:
    camera_data = _camera_data([1.0, 2.0, 3.0], [])

    with pytest.raises(RuntimeError, match="empty timestamp buffer"):
        _select_alignment_camera_idx(camera_data, num_obs_cameras=2)

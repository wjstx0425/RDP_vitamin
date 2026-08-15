from pathlib import Path

import numpy as np
import pytest

from tools.rdp_debug.replay_saved_observations import EXPECTED_IMAGE_KEYS
from tools.rdp_debug.replay_saved_observations import discover_steps
from tools.rdp_debug.replay_saved_observations import load_saved_observation
from tools.rdp_debug.replay_saved_observations import summarize_actions
from tools.rdp_debug.replay_saved_observations import write_new_text


class FakeCv2:
    IMREAD_COLOR = 1
    COLOR_BGR2RGB = 2

    def imread(self, path: str, mode: int) -> np.ndarray | None:
        assert mode == self.IMREAD_COLOR
        return np.full((2, 3, 3), len(Path(path).stem), dtype=np.uint8)

    def cvtColor(self, image: np.ndarray, code: int) -> np.ndarray:
        assert code == self.COLOR_BGR2RGB
        return image[..., ::-1]


def test_discover_steps_requires_contiguous_numbering(tmp_path: Path) -> None:
    (tmp_path / "step_000001").mkdir()
    (tmp_path / "step_000003").mkdir()
    with pytest.raises(ValueError, match="missing saved step 000002"):
        discover_steps(tmp_path)


def test_load_saved_observation_loads_float32_state_and_rgb_images(tmp_path: Path) -> None:
    state = np.arange(20, dtype=np.float64)
    np.save(tmp_path / "observation.state.npy", state)
    for key in EXPECTED_IMAGE_KEYS:
        (tmp_path / f"{key}.jpg").write_bytes(b"jpeg")

    observation = load_saved_observation(tmp_path, FakeCv2())

    assert observation["observation.state"].dtype == np.float32
    assert np.array_equal(observation["observation.state"], state.astype(np.float32))
    assert tuple(observation) == ("observation.state", *EXPECTED_IMAGE_KEYS)
    assert observation[EXPECTED_IMAGE_KEYS[0]].shape == (2, 3, 3)


def test_source_has_no_online_robot_dependencies() -> None:
    source = Path("tools/rdp_debug/replay_saved_observations.py").read_text(encoding="utf-8")
    for forbidden in ("RobotBridgeClient", "websockets", "requests", "BimanualUmiEnv", "/dev/video"):
        assert forbidden not in source


def test_replay_summary_exposes_five_step_boundary_jump() -> None:
    actions = np.zeros((10, 20), dtype=np.float64)
    actions[:5, 9] = 0.07
    actions[5:, 9] = 0.11
    report = summarize_actions(actions, [True, False, False, False, False] * 2, [3.0] * 10, 5)

    assert report["frames"] == 10
    assert report["replan_frames"] == [0, 5]
    assert report["gripper_boundary_jump_m"]["left"]["mean"] == pytest.approx(0.04)


def test_output_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    output.write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError):
        write_new_text(output, "replacement")

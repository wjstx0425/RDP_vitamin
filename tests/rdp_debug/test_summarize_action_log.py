from pathlib import Path

import pytest

from tools.rdp_debug.summarize_action_log import load_jsonl
from tools.rdp_debug.summarize_action_log import summarize_records


def record(index: int, time_s: float, left_grip: float) -> dict:
    return {
        "time": time_s,
        "iter_idx": index,
        "obs_seq": index + 1,
        "raw_action_len": 1,
        "new_action_len": 1,
        "controller_records": [{
            "scheduled": True,
            "target_time": time_s + 0.05,
            "left_target_pose": [0, 0, 0, 0, 0, 0],
            "right_target_pose": [0, 0, 0, 0, 0, 0],
            "left_gripper": [left_grip],
            "right_gripper": [0.03],
        }],
    }


def test_summary_separates_replan_boundary_gripper_jumps() -> None:
    rows = [record(i, i * 0.05, 0.01 if i < 5 else 0.03) for i in range(10)]
    report = summarize_records(rows, replan_interval=5)
    assert report["frames"] == 10
    assert report["effective_hz"] == pytest.approx(20.0)
    assert report["scheduled"] == 10
    assert report["gripper_jump_m"]["left"]["boundary_mean"] == pytest.approx(0.02)
    assert report["gripper_jump_m"]["left"]["within_mean"] == pytest.approx(0.0)


def test_load_jsonl_reports_source_line(tmp_path: Path) -> None:
    source = tmp_path / "bad.jsonl"
    source.write_text('{"time": 1}\nnot-json\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad.jsonl:2"):
        load_jsonl(source)

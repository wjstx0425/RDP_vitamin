import json
from datetime import datetime

import cv2
import numpy as np

from deploy_scripts.deployment_trial_recorder import (
    DeploymentTrialRecorder,
    RDP_IMAGE_KEYS,
    decode_relative_actions,
    extract_robot_poses,
)


def make_policy_observation() -> dict:
    observation = {
        key: np.full((2, 3, 3), index * 20, dtype=np.uint8)
        for index, key in enumerate(RDP_IMAGE_KEYS, start=1)
    }
    observation["observation.images.camera0"][0, 0] = [255, 0, 0]
    observation["observation.state"] = np.arange(20, dtype=np.float32)
    observation["task"] = "pick up two tubes"
    return observation


def test_trial_recorder_saves_steps_images_and_failure_manifest(tmp_path):
    observation = make_policy_observation()
    recorder = DeploymentTrialRecorder(
        tmp_path,
        policy_type="rdp",
        data_type="vitac",
        image_interval=5,
        now=datetime(2026, 8, 16, 16, 1, 2, 345678),
    )

    assert recorder.trial_id == "trial_20260816_160102_345678"
    assert recorder.should_save_periodic_images(0) is False
    assert recorder.should_save_periodic_images(4) is False
    assert recorder.should_save_periodic_images(5) is True

    recorder.save_images(observation, reason="initial", iter_idx=0)
    recorder.save_images(observation, reason="step", iter_idx=5)
    recorder.log_step(
        {
            "iter_idx": 0,
            "obs_seq": 17,
            "observation_timestamp": np.float64(100.25),
            "state": observation["observation.state"],
            "raw_action": np.zeros((1, 20), dtype=np.float32),
        }
    )
    recorder.record_failure(
        RuntimeError("controller failed"),
        failure_step=7,
        observation=observation,
        stage="action_execution",
    )
    recorder.finish(result_label="failure", termination_reason="exception")

    trial_dir = tmp_path / recorder.trial_id
    records = [
        json.loads(line)
        for line in (trial_dir / "steps.jsonl").read_text().splitlines()
    ]
    assert records == [
        {
            "iter_idx": 0,
            "obs_seq": 17,
            "observation_timestamp": 100.25,
            "state": list(range(20)),
            "raw_action": [[0.0] * 20],
        }
    ]

    initial_bgr = cv2.imread(
        str(
            trial_dir
            / "images"
            / "initial"
            / "observation.images.camera0.png"
        ),
        cv2.IMREAD_COLOR,
    )
    initial_rgb = cv2.cvtColor(initial_bgr, cv2.COLOR_BGR2RGB)
    np.testing.assert_array_equal(
        initial_rgb,
        observation["observation.images.camera0"],
    )
    for image_key in RDP_IMAGE_KEYS:
        filename = f"{image_key}.png"
        assert (trial_dir / "images" / "initial" / filename).is_file()
        assert (trial_dir / "images" / "step_000005" / filename).is_file()
        assert (
            trial_dir / "images" / "failure_step_000007" / filename
        ).is_file()

    manifest = json.loads((trial_dir / "manifest.json").read_text())
    assert manifest["trial_id"] == recorder.trial_id
    assert manifest["policy_type"] == "rdp"
    assert manifest["data_type"] == "vitac"
    assert manifest["image_interval"] == 5
    assert manifest["step_count"] == 1
    assert manifest["status"] == "failed"
    assert manifest["result_label"] == "failure"
    assert manifest["termination_reason"] == "exception"
    assert manifest["failure_step"] == 7
    assert manifest["failure"] == {
        "stage": "action_execution",
        "type": "RuntimeError",
        "message": "controller failed",
    }
    assert [entry["directory"] for entry in manifest["image_batches"]] == [
        "images/initial",
        "images/step_000005",
        "images/failure_step_000007",
    ]


def test_decode_relative_actions_and_extract_robot_poses():
    raw_action = np.zeros((1, 20), dtype=np.float32)
    raw_action[:, 3:9] = [1, 0, 0, 0, 1, 0]
    raw_action[:, 13:19] = [1, 0, 0, 0, 1, 0]
    raw_action[0, :3] = [0.01, 0.02, 0.03]
    raw_action[0, 9] = 0.04
    raw_action[0, 10:13] = [-0.01, -0.02, -0.03]
    raw_action[0, 19] = 0.05

    decoded = decode_relative_actions(raw_action, sides=["left", "right"])

    np.testing.assert_allclose(
        decoded["left"],
        [[0.01, 0.02, 0.03, 0.0, 0.0, 0.0, 0.04]],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        decoded["right"],
        [[-0.01, -0.02, -0.03, 0.0, 0.0, 0.0, 0.05]],
        atol=1e-6,
    )

    env_obs = {
        "robot0_eef_pos": np.array([[1.0, 2.0, 3.0]]),
        "robot0_eef_rot_axis_angle": np.array([[0.1, 0.2, 0.3]]),
        "robot0_gripper_width": np.array([[0.04]]),
        "robot1_eef_pos": np.array([[4.0, 5.0, 6.0]]),
        "robot1_eef_rot_axis_angle": np.array([[0.4, 0.5, 0.6]]),
        "robot1_gripper_width": np.array([[0.05]]),
    }

    poses = extract_robot_poses(env_obs, sides=["left", "right"])

    assert poses == {
        "left": {
            "position": [1.0, 2.0, 3.0],
            "rotation_axis_angle": [0.1, 0.2, 0.3],
            "gripper_width": [0.04],
        },
        "right": {
            "position": [4.0, 5.0, 6.0],
            "rotation_axis_angle": [0.4, 0.5, 0.6],
            "gripper_width": [0.05],
        },
    }

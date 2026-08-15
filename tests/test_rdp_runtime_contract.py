import numpy as np

from configs.server_config import SERVER_CONFIG
from deploy_scripts import bimanual_smolvla_online as online
from real_world.real_inference_util import get_real_umi_action


class FakeEnv:
    def __init__(self) -> None:
        self.observation = {
            "robot0_eef_pos": np.zeros((1, 3), dtype=np.float64),
            "robot0_eef_rot_axis_angle": np.zeros((1, 3), dtype=np.float64),
            "robot1_eef_pos": np.zeros((1, 3), dtype=np.float64),
            "robot1_eef_rot_axis_angle": np.zeros((1, 3), dtype=np.float64),
        }
        self.exec_calls = []

    def get_obs(self):
        return self.observation

    def exec_actions(self, *, actions, timestamps):
        self.exec_calls.append((actions.copy(), timestamps.copy()))
        return [{"scheduled": True}]


def valid_config(**updates):
    config = {
        "policy_type": "rdp",
        "data_type": "vitac",
        "language_prompt": "pick up two tubes",
        "control_frequency": 30.0,
        "controller_frequency": 80.0,
        "single_arm_mode": False,
        "no_state_obs_mode": False,
        "steps_per_inference": 1,
        "action_horizon": 1,
    }
    return config | updates


def identity_action() -> np.ndarray:
    action = np.zeros((1, 20), dtype=np.float32)
    for robot_index in range(2):
        start = robot_index * 10
        action[0, start + 3 : start + 9] = [1, 0, 0, 0, 1, 0]
        action[0, start + 9] = 0.03
    return action


def test_rdp_contract_requires_vitac_one_step_control() -> None:
    validated = online.validate_smolvla_config(valid_config())
    assert validated["policy_type"] == "rdp"

    for invalid in (
        {"data_type": "vision"},
        {"action_horizon": 2, "steps_per_inference": 1},
        {"action_horizon": 2, "steps_per_inference": 2},
    ):
        try:
            online.validate_smolvla_config(valid_config(**invalid))
        except ValueError as error:
            assert "RDP requires" in str(error)
        else:
            raise AssertionError(f"RDP config should have failed: {invalid}")


def test_smolvla_config_keeps_backward_compatible_default() -> None:
    config = valid_config()
    config.pop("policy_type")
    config.update(data_type="vision", action_horizon=20, steps_per_inference=5)
    assert online.validate_smolvla_config(config)["policy_type"] == "smolvla"


def test_rdp_receive_time_schedule_keeps_single_action_fresh() -> None:
    env = FakeEnv()
    now = 200.0
    result = online.execute_action_chunk(
        identity_action(),
        action_horizon=1,
        n_robots=2,
        max_pos_delta=SERVER_CONFIG.max_action_pos_delta,
        max_rot_delta=SERVER_CONFIG.max_action_rot_delta,
        min_gripper=SERVER_CONFIG.min_gripper,
        max_gripper=SERVER_CONFIG.max_gripper,
        obs_timestamp=now - SERVER_CONFIG.camera.capture_timestamp_delay,
        now=lambda: now,
        dt=1.0 / SERVER_CONFIG.control_frequency,
        exec_mode="rtc",
        env=env,
        converter=get_real_umi_action,
        action_pose_repr="relative",
        max_executed_actions=1,
        schedule_from_receive=True,
        command_lead_s=SERVER_CONFIG.rdp_command_lead_s,
    )

    np.testing.assert_allclose(result.fresh.timestamps, [now + SERVER_CONFIG.rdp_command_lead_s])
    np.testing.assert_array_equal(result.fresh.mask, [True])
    assert len(env.exec_calls) == 1

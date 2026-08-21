import json

import numpy as np
import pytest

from convert_pick_tube_lerobot_to_rdp_zarr import build_v2_manifest, create_output
from reactive_diffusion_policy.common.pick_tube_action_contract import (
    ACTION_CONTRACT,
    ACTION_REPRESENTATION_VERSION,
    HIGH_GRIPPER_DELTA_M,
    HIGH_ROTATION_DELTA_DEG,
    HIGH_TRANSLATION_DELTA_M,
    IDLE_ENTRY_FRAMES,
    IDLE_EXIT_FRAMES,
    LOW_GRIPPER_DELTA_M,
    LOW_ROTATION_DELTA_DEG,
    LOW_TRANSLATION_DELTA_M,
    TERMINAL_ACTION_POLICY,
    canonical_noop_from_state,
    canonicalize_episode_actions,
)


IDENTITY_6D = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)


def _episode(length=10):
    state = np.zeros((length, 20), dtype=np.float32)
    state[:, 6] = np.linspace(0.02, 0.03, length)
    state[:, 13] = np.linspace(0.04, 0.05, length)
    action = np.zeros((length, 20), dtype=np.float32)
    action[:, 3:9] = IDENTITY_6D
    action[:, 13:19] = IDENTITY_6D
    return state, action


def test_terminal_action_is_repaired_without_mutating_source_arrays():
    state, action = _episode()
    action[-1, 3:9] = 0
    action[-1, 13:19] = 0
    source_action = action.copy()

    result = canonicalize_episode_actions(state, action)

    np.testing.assert_array_equal(result.action_raw, action)
    np.testing.assert_array_equal(action, source_action)
    assert not result.action_valid[-1]
    np.testing.assert_allclose(result.action[-1, :3], 0.0)
    np.testing.assert_allclose(result.action[-1, 3:9], IDENTITY_6D)
    assert result.action[-1, 9] == state[-1, 6]
    np.testing.assert_allclose(result.action[-1, 10:13], 0.0)
    np.testing.assert_allclose(result.action[-1, 13:19], IDENTITY_6D)
    assert result.action[-1, 19] == state[-1, 13]


def test_canonical_noop_is_bimanual_identity_with_current_gripper_widths():
    state, _ = _episode(length=1)

    noop = canonical_noop_from_state(state[0])

    np.testing.assert_allclose(noop[:3], 0.0)
    np.testing.assert_allclose(noop[3:9], IDENTITY_6D)
    assert noop[9] == state[0, 6]
    np.testing.assert_allclose(noop[10:13], 0.0)
    np.testing.assert_allclose(noop[13:19], IDENTITY_6D)
    assert noop[19] == state[0, 13]


def test_degenerate_nonterminal_rotation_is_rejected():
    state, action = _episode()
    action[0, 3:9] = 0

    with pytest.raises(ValueError, match="degenerate rotation"):
        canonicalize_episode_actions(state, action)


def test_idle_hysteresis_uses_accepted_thresholds_and_frame_counts():
    state, action = _episode(length=12)
    # The right arm is active while the left arm remains under every low limit.
    action[:, 10] = 0.001
    # Two high-motion left rows exit the idle state only on the second row.
    action[9:11, 0] = 0.00081

    result = canonicalize_episode_actions(state, action)

    assert IDLE_ENTRY_FRAMES == 8
    assert IDLE_EXIT_FRAMES == 2
    assert LOW_TRANSLATION_DELTA_M == 0.0005
    assert LOW_ROTATION_DELTA_DEG == 0.25
    assert LOW_GRIPPER_DELTA_M == 0.0005
    assert HIGH_TRANSLATION_DELTA_M == 0.0008
    assert HIGH_ROTATION_DELTA_DEG == 0.4
    assert HIGH_GRIPPER_DELTA_M == 0.0008
    assert not result.idle_arm_mask[:7, 0].any()
    assert result.idle_arm_mask[7:10, 0].all()
    assert not result.idle_arm_mask[10:, 0].any()


def test_idle_labels_are_canonicalized_from_original_physical_motion():
    state, action = _episode(length=10)
    action[:, 9] = 0.03
    action[:, 10] = 0.001

    result = canonicalize_episode_actions(state, action)

    np.testing.assert_array_equal(result.action_raw, action)
    np.testing.assert_allclose(result.action[7:9, :3], 0.0)
    np.testing.assert_allclose(result.action[7:9, 3:9], np.tile(IDENTITY_6D, (2, 1)))
    np.testing.assert_allclose(result.action[7:9, 9], state[7:9, 6])


def test_converter_v2_schema_and_manifest_are_json_serializable(tmp_path):
    zarr_path = tmp_path / "replay_buffer.zarr"
    pca_path = tmp_path / "pca.npz"
    pca_path.write_bytes(b"stable-pca")
    root, arrays = create_output(zarr_path, tactile_embedding_dim=30)

    assert arrays["action_raw"].shape == (0, 20)
    assert arrays["action_raw"].dtype == np.dtype("float32")
    assert arrays["action_valid"].shape == (0,)
    assert arrays["action_valid"].dtype == np.dtype("bool")
    assert arrays["idle_arm_mask"].shape == (0, 2)
    assert arrays["idle_arm_mask"].dtype == np.dtype("bool")

    episode_manifest = [
        {"dataset": "pick_tube_01", "episode_index": 0, "length": 10, "repeat": 1}
    ]
    manifest = build_v2_manifest(
        arrays=arrays,
        pca_path=pca_path,
        tactile_embedding_dim=30,
        episode_manifest=episode_manifest,
        repair_counts={"terminal_actions": 1, "invalid_nonterminal_actions": 0},
        idle_coverage_by_source={"pick_tube_01": {"left": 0.2, "right": 0.3}},
        git_commit="deadbeef",
    )
    root["meta"].attrs["v2_manifest_json"] = json.dumps(manifest, sort_keys=True)

    assert manifest["action_representation_version"] == ACTION_REPRESENTATION_VERSION
    assert manifest["action_contract"] == ACTION_CONTRACT
    assert manifest["terminal_action_policy"] == TERMINAL_ACTION_POLICY
    assert manifest["repair_counts"]["terminal_actions"] == 1
    assert manifest["arrays"]["action"]["shape"] == [0, 20]
    assert len(manifest["pca_sha256"]) == 64
    assert len(manifest["dataset_digest"]) == 64
    json.loads(root["meta"].attrs["v2_manifest_json"])

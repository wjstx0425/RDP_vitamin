import pytest

from reactive_diffusion_policy.workspace.base_workspace import BaseWorkspace


def test_resume_advances_saved_end_of_epoch_state():
    workspace = BaseWorkspace(cfg=None)
    workspace.epoch = 4
    workspace.global_step = 116789

    workspace.advance_training_state_for_resume()

    assert workspace.epoch == 5
    assert workspace.global_step == 116790
    assert workspace.get_remaining_epochs(10) == 5


def test_epoch_target_is_total_for_fresh_and_completed_runs():
    workspace = BaseWorkspace(cfg=None)
    workspace.epoch = 0
    workspace.global_step = 0

    assert workspace.get_remaining_epochs(10) == 10

    workspace.epoch = 10
    assert workspace.get_remaining_epochs(10) == 0
    assert workspace.get_remaining_epochs(5) == 0


def test_negative_epoch_target_is_rejected():
    workspace = BaseWorkspace(cfg=None)
    workspace.epoch = 0

    with pytest.raises(ValueError, match="non-negative"):
        workspace.get_remaining_epochs(-1)

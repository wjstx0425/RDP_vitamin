from pathlib import Path
import numpy as np
import torch
import torchvision
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from convert_pick_tube_lerobot_to_rdp_zarr import (
    DEFAULT_DATASETS,
    DEFAULT_DATASET_REPEATS,
    parse_dataset_repeats,
)
from reactive_diffusion_policy.common.replay_buffer import ReplayBuffer
from reactive_diffusion_policy.common.sampler import SequenceSampler
from reactive_diffusion_policy.dataset.real_image_tactile_dataset import RealImageTactileDataset
from reactive_diffusion_policy.model.vision.multi_image_obs_encoder import TrainOnlyTransform


def test_pick_tube_conversion_defaults_include_new_diverse_data():
    assert DEFAULT_DATASETS[-2:] == ("pick_tube_05", "pick_tube_06")
    assert parse_dataset_repeats(list(DEFAULT_DATASET_REPEATS)) == {
        "pick_tube_05": 2,
        "pick_tube_06": 2,
    }


def test_sequence_sampler_repeats_selected_episodes_only():
    replay_buffer = ReplayBuffer.create_empty_numpy()
    replay_buffer.add_episode({"value": np.arange(3, dtype=np.float32)[:, None]})
    replay_buffer.add_episode({"value": np.arange(2, dtype=np.float32)[:, None]})

    sampler = SequenceSampler(
        replay_buffer=replay_buffer,
        sequence_length=1,
        episode_mask=np.array([True, True]),
        episode_repeats=np.array([1, 2]),
    )

    assert sampler.indices[:, 0].tolist() == [0, 1, 2, 3, 4, 3, 4]


def test_color_jitter_is_bypassed_in_eval_mode():
    transform = TrainOnlyTransform(torchvision.transforms.ColorJitter(brightness=0.5))
    value = torch.rand(2, 3, 16, 16)
    transform.eval()

    output = transform(value)

    assert output is value


def test_validation_sampler_keeps_observation_read_limit_without_oversampling():
    replay_buffer = ReplayBuffer.create_empty_numpy()
    replay_buffer.add_episode({"value": np.arange(3, dtype=np.float32)[:, None]})
    replay_buffer.add_episode({"value": np.arange(2, dtype=np.float32)[:, None]})

    dataset = object.__new__(RealImageTactileDataset)
    dataset.replay_buffer = replay_buffer
    dataset.horizon = 1
    dataset.n_latency_steps = 0
    dataset.pad_before = 0
    dataset.pad_after = 0
    dataset.val_mask = np.array([False, True])
    dataset.key_first_k = {"value": 1}

    val_dataset = dataset.get_validation_dataset()

    assert val_dataset.sampler.key_first_k == {"value": 1}
    assert val_dataset.sampler.indices[:, 0].tolist() == [3, 4]


def test_pick_tube_configs_match_official_rdp_temporal_and_model_defaults():
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    config_dir = str(
        (Path(__file__).resolve().parents[1] / "reactive_diffusion_policy" / "config")
    )
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        at_cfg = compose(config_name="train_pick_tube_at_workspace")
        ldp_cfg = compose(config_name="train_pick_tube_ldp_workspace")

    for cfg in (at_cfg, ldp_cfg):
        assert cfg.horizon == 32
        assert cfg.n_obs_steps == 2
        assert cfg.dataset_obs_steps == 4
        assert cfg.dataset_obs_temporal_downsample_ratio == 2
        assert cfg.n_action_steps == 29
        assert cfg.task.dataset.pad_before == 3
        assert cfg.task.dataset.pad_after == 28
        assert cfg.task.dataset.val_ratio == 0.0
        assert cfg.task.dataset.use_episode_repeats is False
        assert cfg.checkpoint.topk.monitor_key == "train_loss"
        assert cfg.training.checkpoint_every == 10
        assert cfg.training.val_every == 1
        assert list(cfg.task.shape_meta.obs.tactile_embedding.shape) == [30]
        assert list(cfg.task.shape_meta.extended_obs.tactile_embedding.shape) == [30]
        assert "pca30" in cfg.task.dataset_path

    assert at_cfg.at.policy.n_latent_dims == 16
    assert at_cfg.at.policy.conv_latent_dims == 32
    assert at_cfg.at.policy.rnn_latent_dims == 64
    assert at_cfg.at.policy.n_embed == 32
    assert ldp_cfg.policy.at.n_latent_dims == 16
    assert ldp_cfg.policy.at.conv_latent_dims == 32
    assert ldp_cfg.policy.at.rnn_latent_dims == 64
    assert ldp_cfg.policy.at.n_embed == 32
    assert ldp_cfg.task.dataset.at.n_latent_dims == 16
    assert ldp_cfg.task.dataset.at.conv_latent_dims == 32
    assert ldp_cfg.task.dataset.at.rnn_latent_dims == 64
    assert ldp_cfg.task.dataset.at.n_embed == 32
    assert at_cfg.dataloader.batch_size == 64
    assert at_cfg.training.num_epochs == 20
    assert at_cfg.training.checkpoint_every == 10

    assert list(ldp_cfg.policy.down_dims) == [512, 1024, 2048]
    assert len(ldp_cfg.policy.obs_encoder.random_transforms) == 1
    assert ldp_cfg.policy.obs_encoder.random_transforms[0].type == "RandomCrop"
    assert list(ldp_cfg.policy.obs_encoder.resize_shape) == [224, 224]
    assert ldp_cfg.dataloader.batch_size == 64
    assert ldp_cfg.dataloader.num_workers == 8
    assert ldp_cfg.training.num_epochs == 10
    assert ldp_cfg.training.checkpoint_every == 10

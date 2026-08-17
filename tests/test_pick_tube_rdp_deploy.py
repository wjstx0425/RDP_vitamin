import importlib.util
from pathlib import Path

import numpy as np
import torch
from torch import nn
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "deploy_pick_tube_rdp", ROOT / "deploy_pick_tube_rdp.py"
)
deploy = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(deploy)


class FakeTactileEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_means = None

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        self.last_means = images.mean(dim=(1, 2, 3)).detach().cpu()
        return self.last_means.to(images.device)[:, None].repeat(1, 512)


class FakePolicy:
    def __init__(self) -> None:
        self.slow_calls = 0
        self.fast_history_lengths = []
        self.slow_observation_states = []

    def predict_action(self, obs_dict, **kwargs):
        assert tuple(obs_dict["camera1"].shape) == (1, 2, 3, 224, 224)
        assert tuple(obs_dict["camera2"].shape) == (1, 2, 3, 224, 224)
        assert tuple(obs_dict["observation_state"].shape) == (1, 2, 20)
        assert tuple(obs_dict["tactile_embedding"].shape) == (1, 2, 30)
        torch.testing.assert_close(
            obs_dict["tactile_embedding"][0, :, :15],
            torch.full((2, 15), 1.0 / 255.0),
        )
        torch.testing.assert_close(
            obs_dict["tactile_embedding"][0, :, 15:],
            torch.full((2, 15), 2.0 / 255.0),
        )
        assert kwargs["return_latent_action"] is True
        self.slow_observation_states.append(
            obs_dict["observation_state"][0, :, 0].detach().cpu().tolist()
        )
        self.slow_calls += 1
        return {"action": torch.zeros((1, 29, 128))}

    def predict_from_latent_action(
        self,
        latent_action,
        extended_obs,
        extended_obs_last_step,
        dataset_obs_temporal_downsample_ratio,
    ):
        assert tuple(latent_action.shape) == (1, 128)
        assert dataset_obs_temporal_downsample_ratio == 2
        history_length = extended_obs["tactile_embedding"].shape[1]
        assert extended_obs_last_step == history_length
        self.fast_history_lengths.append(history_length)
        return {"action": torch.full((1, history_length, 20), float(history_length))}


def observation(step: int = 0) -> dict:
    result = {
        "observation.images.camera0": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation.images.camera1": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation.state": np.full(20, step, dtype=np.float32),
    }
    for value, key in enumerate(deploy.TACTILE_KEYS, start=1):
        result[key] = np.full((224, 224, 3), value, dtype=np.uint8)
    return result


def test_prepare_inference_config_drops_training_only_color_jitter() -> None:
    config = OmegaConf.create(
        {
            "policy": {
                "obs_encoder": {
                    "random_transforms": [
                        {"type": "RandomCrop", "ratio": 0.9},
                        {
                            "type": "ColorJitter",
                            "brightness": 0.25,
                            "contrast": 0.25,
                            "saturation": 0.15,
                            "hue": 0.03,
                        },
                    ]
                }
            }
        }
    )

    deploy.prepare_inference_config(config)

    assert OmegaConf.to_container(
        config.policy.obs_encoder.random_transforms,
        resolve=True,
    ) == [{"type": "RandomCrop", "ratio": 0.9}]


def test_runtime_updates_slow_plan_every_five_steps_and_decodes_every_step() -> None:
    policy = FakePolicy()
    encoder = FakeTactileEncoder()
    means = np.zeros((2, 1024), dtype=np.float32)
    components = np.zeros((2, 15, 1024), dtype=np.float32)
    components[:, np.arange(15), np.arange(15)] = 1.0
    tactile_pca = deploy.BimanualTactilePCA(means, components)
    runtime = deploy.PickTubeRDPRuntime(
        policy,
        encoder,
        torch.device("cpu"),
        tactile_pca,
        slow_update_interval=5,
        dataset_obs_temporal_downsample_ratio=2,
        n_obs_steps=2,
    )

    slow_updates = []
    actions = []
    for step in range(7):
        action, slow_update = runtime.predict(observation(step))
        actions.append(action)
        slow_updates.append(slow_update)

    assert slow_updates == [True, False, False, False, False, True, False]
    assert policy.slow_calls == 2
    assert policy.slow_observation_states == [[0.0, 0.0], [3.0, 5.0]]
    assert policy.fast_history_lengths == [4, 5, 6, 7, 8, 4, 5]
    assert all(action.shape == (1, 20) and action.dtype == np.float32 for action in actions)
    np.testing.assert_allclose(encoder.last_means, np.arange(1, 5) / 255.0)

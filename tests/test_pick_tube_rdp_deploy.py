import importlib.util
from pathlib import Path

import numpy as np
import pytest
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
    def __init__(self, components_per_arm: int) -> None:
        self.components_per_arm = components_per_arm
        self.slow_calls = 0
        self.fast_history_lengths = []
        self.slow_observation_states = []

    def predict_action(self, obs_dict, **kwargs):
        assert tuple(obs_dict["camera1"].shape) == (1, 2, 3, 224, 224)
        assert tuple(obs_dict["camera2"].shape) == (1, 2, 3, 224, 224)
        assert tuple(obs_dict["observation_state"].shape) == (1, 2, 20)
        tactile_dim = self.components_per_arm * 2
        assert tuple(obs_dict["tactile_embedding"].shape) == (1, 2, tactile_dim)
        torch.testing.assert_close(
            obs_dict["tactile_embedding"][0, :, : self.components_per_arm],
            torch.full((2, self.components_per_arm), 1.0 / 255.0),
        )
        torch.testing.assert_close(
            obs_dict["tactile_embedding"][0, :, self.components_per_arm :],
            torch.full((2, self.components_per_arm), 3.0 / 255.0),
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
        assert extended_obs["tactile_embedding"].shape[2] == self.components_per_arm * 2
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


def tactile_cfg(obs_dim: int, extended_dim: int | None = None):
    return OmegaConf.create(
        {
            "shape_meta": {
                "obs": {"tactile_embedding": {"shape": [obs_dim]}},
                "extended_obs": {
                    "tactile_embedding": {
                        "shape": [obs_dim if extended_dim is None else extended_dim]
                    }
                },
            }
        }
    )


class FakeAT:
    def __init__(self) -> None:
        self.normalizer = None

    def set_normalizer(self, normalizer) -> None:
        self.normalizer = normalizer


class LoadedFakePolicy:
    def __init__(self) -> None:
        self.at = FakeAT()
        self.normalizer = object()
        self.num_inference_steps = None
        self.eval_called = False
        self.device = None

    def eval(self):
        self.eval_called = True
        return self

    def to(self, device):
        self.device = device
        return self


def policy_cfg(tactile_dim: int):
    cfg = tactile_cfg(tactile_dim)
    cfg._target_ = "tests.FakeWorkspace"
    cfg.policy = {
        "at": {},
        "obs_encoder": {"random_transforms": []},
    }
    cfg.training = {"use_ema": True}
    return cfg


def payload(cfg):
    return {"cfg": cfg}


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


@pytest.mark.parametrize("tactile_dim", [16, 30, 60])
def test_validate_tactile_dimensions_accepts_matching_artifacts(
    tactile_dim: int,
) -> None:
    deploy.validate_tactile_dimensions(
        tactile_dim,
        tactile_cfg(tactile_dim),
        tactile_cfg(tactile_dim),
        Path("ldp.ckpt"),
        Path("at.ckpt"),
    )


def test_validate_tactile_dimensions_reports_every_source() -> None:
    with pytest.raises(ValueError) as error:
        deploy.validate_tactile_dimensions(
            16,
            tactile_cfg(16, extended_dim=30),
            tactile_cfg(60),
            Path("ldp.ckpt"),
            Path("at.ckpt"),
        )

    message = str(error.value)
    assert "PCA output=16D" in message
    assert "LDP obs (ldp.ckpt)=16D" in message
    assert "LDP extended_obs (ldp.ckpt)=30D" in message
    assert "AT obs (at.ckpt)=60D" in message
    assert "AT extended_obs (at.ckpt)=60D" in message


def test_tactile_dim_reports_missing_checkpoint_field() -> None:
    with pytest.raises(ValueError, match="LDP checkpoint is missing"):
        deploy._tactile_dim(OmegaConf.create({}), "LDP", "obs")


@pytest.mark.parametrize("role", ["LDP", "AT"])
@pytest.mark.parametrize("invalid_payload", [None, {}, {"cfg": None}])
def test_load_policy_reports_checkpoint_payload_cfg_errors(role, invalid_payload, monkeypatch) -> None:
    ldp_checkpoint = Path("ldp.ckpt")
    at_checkpoint = Path("at.ckpt")
    valid_payload = payload(policy_cfg(16))

    def load_payload(path, current_role):
        if current_role == role:
            return invalid_payload
        return valid_payload

    monkeypatch.setattr(deploy, "_load_checkpoint_payload", load_payload)

    with pytest.raises(ValueError) as error:
        deploy.load_policy(
            ldp_checkpoint,
            at_checkpoint,
            torch.device("cpu"),
            num_inference_steps=8,
            tactile_embedding_dim=16,
        )

    message = str(error.value)
    assert role in message
    assert str(ldp_checkpoint if role == "LDP" else at_checkpoint) in message
    assert "cfg" in message


def test_load_policy_reports_unresolved_checkpoint_metadata(monkeypatch) -> None:
    ldp_checkpoint = Path("ldp.ckpt")
    at_checkpoint = Path("at.ckpt")
    ldp_cfg = policy_cfg(16)
    ldp_cfg.shape_meta.obs.tactile_embedding.shape = ["${missing_dimension}"]

    monkeypatch.setattr(
        deploy,
        "_load_checkpoint_payload",
        lambda path, role: payload(ldp_cfg if role == "LDP" else policy_cfg(16)),
    )

    with pytest.raises(ValueError) as error:
        deploy.load_policy(
            ldp_checkpoint,
            at_checkpoint,
            torch.device("cpu"),
            num_inference_steps=8,
            tactile_embedding_dim=16,
        )

    message = str(error.value)
    assert "LDP" in message
    assert str(ldp_checkpoint) in message
    assert "shape_meta.obs.tactile_embedding.shape" in message
    assert error.value.__cause__ is not None


def test_load_policy_validates_matching_payloads_before_workspace_construction(monkeypatch) -> None:
    ldp_checkpoint = Path("ldp.ckpt")
    at_checkpoint = Path("at.ckpt")
    ldp_payload = payload(policy_cfg(30))
    at_payload = payload(tactile_cfg(30))
    payload_calls = []
    validated = False
    workspace_instances = []
    original_validate = deploy.validate_tactile_dimensions

    def load_payload(path, role):
        payload_calls.append((path, role))
        return ldp_payload if role == "LDP" else at_payload

    def validate(*args):
        nonlocal validated
        assert args[1] == ldp_payload["cfg"]
        assert args[2] is at_payload["cfg"]
        original_validate(*args)
        validated = True

    class FakeWorkspace:
        def __init__(self, cfg):
            assert validated
            self.cfg = cfg
            self.ema_model = LoadedFakePolicy()
            self.model = LoadedFakePolicy()
            self.normalizer = object()
            self.loaded_payload = None
            workspace_instances.append(self)

        def load_payload(self, loaded_payload):
            self.loaded_payload = loaded_payload

    monkeypatch.setattr(deploy, "_load_checkpoint_payload", load_payload)
    monkeypatch.setattr(deploy, "validate_tactile_dimensions", validate)
    monkeypatch.setattr(deploy.hydra.utils, "get_class", lambda target: FakeWorkspace)

    policy, cfg = deploy.load_policy(
        ldp_checkpoint,
        at_checkpoint,
        torch.device("cpu"),
        num_inference_steps=7,
        tactile_embedding_dim=30,
    )

    assert payload_calls == [(ldp_checkpoint, "LDP"), (at_checkpoint, "AT")]
    assert len(workspace_instances) == 1
    assert workspace_instances[0].loaded_payload is ldp_payload
    assert policy is workspace_instances[0].ema_model
    assert cfg.at_load_dir == str(at_checkpoint)
    assert policy.at.normalizer is policy.normalizer
    assert policy.num_inference_steps == 7
    assert policy.eval_called
    assert policy.device == torch.device("cpu")


@pytest.mark.parametrize(
    ("pca_dim", "ldp_obs", "ldp_extended_obs", "at_obs", "at_extended_obs", "source"),
    [
        pytest.param(16, 30, 30, 30, 30, "PCA output", id="pca"),
        pytest.param(30, 16, 30, 30, 30, "LDP obs", id="ldp-obs"),
        pytest.param(30, 30, 16, 30, 30, "LDP extended_obs", id="ldp-extended-obs"),
        pytest.param(30, 30, 30, 16, 30, "AT obs", id="at-obs"),
        pytest.param(30, 30, 30, 30, 16, "AT extended_obs", id="at-extended-obs"),
    ],
)
def test_load_policy_rejects_each_mismatched_tactile_dimension_before_workspace(
    monkeypatch,
    pca_dim,
    ldp_obs,
    ldp_extended_obs,
    at_obs,
    at_extended_obs,
    source,
) -> None:
    ldp_checkpoint = Path("ldp.ckpt")
    at_checkpoint = Path("at.ckpt")
    ldp_payload = payload(policy_cfg(ldp_obs))
    ldp_payload["cfg"].shape_meta.extended_obs.tactile_embedding.shape = [ldp_extended_obs]
    at_payload = payload(tactile_cfg(at_obs, at_extended_obs))

    monkeypatch.setattr(
        deploy,
        "_load_checkpoint_payload",
        lambda path, role: ldp_payload if role == "LDP" else at_payload,
    )
    monkeypatch.setattr(
        deploy.hydra.utils,
        "get_class",
        lambda target: pytest.fail("workspace construction must follow validation"),
    )

    with pytest.raises(ValueError, match=source):
        deploy.load_policy(
            ldp_checkpoint,
            at_checkpoint,
            torch.device("cpu"),
            num_inference_steps=8,
            tactile_embedding_dim=pca_dim,
        )


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param(16, id="scalar"),
        pytest.param("7", id="string"),
        pytest.param([16.9], id="fractional"),
        pytest.param([True], id="boolean"),
        pytest.param([16, 30], id="multiple_items"),
        pytest.param([0], id="zero"),
        pytest.param([-1], id="negative"),
    ],
)
def test_tactile_dim_rejects_malformed_shapes(shape) -> None:
    cfg = OmegaConf.create(
        {"shape_meta": {"obs": {"tactile_embedding": {"shape": shape}}}}
    )

    with pytest.raises(ValueError):
        deploy._tactile_dim(cfg, "LDP", "obs")


@pytest.mark.parametrize("components_per_arm", [8, 15, 30])
def test_runtime_updates_slow_plan_every_five_steps_and_decodes_every_step(
    components_per_arm: int,
) -> None:
    policy = FakePolicy(components_per_arm)
    encoder = FakeTactileEncoder()
    means = np.zeros((2, 1024), dtype=np.float32)
    components = np.zeros((2, components_per_arm, 1024), dtype=np.float32)
    components[:, np.arange(components_per_arm), np.arange(components_per_arm)] = 1.0
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

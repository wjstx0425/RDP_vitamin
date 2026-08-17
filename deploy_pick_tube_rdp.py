#!/usr/bin/env python3
"""Run a pick-tube RDP checkpoint against the VB3 robot bridge."""

from __future__ import annotations

import argparse
import copy
import os
import time
from pathlib import Path
from typing import Any

import cv2
import dill
import hydra
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

from reactive_diffusion_policy.deploy.bridge_client import RobotBridgeClient
from reactive_diffusion_policy.deploy.tactile_encoder_torch import load_tactile_resnet18


CAMERA_KEYS = ("observation.images.camera0", "observation.images.camera1")
TACTILE_KEYS = (
    "observation.images.tactile_left_0",
    "observation.images.tactile_right_0",
    "observation.images.tactile_left_1",
    "observation.images.tactile_right_1",
)
STATE_KEY = "observation.state"
IMAGE_SIZE = 224
STATE_DIM = 20
ACTION_DIM = 20


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    for section in ("model", "connection", "control", "runtime"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing config section: {section}")
    return config


def resolve_path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def load_policy(
    ldp_checkpoint: Path,
    at_checkpoint: Path,
    device: torch.device,
    num_inference_steps: int,
):
    with ldp_checkpoint.open("rb") as file:
        payload = torch.load(
            file,
            pickle_module=dill,
            weights_only=False,
            map_location="cpu",
        )
    cfg = copy.deepcopy(payload["cfg"])
    OmegaConf.set_struct(cfg, False)
    cfg.at_load_dir = str(at_checkpoint)
    cfg.policy.at.load_dir = str(at_checkpoint)
    cfg.policy.at.device = str(device)

    workspace_class = hydra.utils.get_class(cfg._target_)
    workspace = workspace_class(cfg)
    workspace.load_payload(payload)
    policy = workspace.ema_model if bool(cfg.training.use_ema) else workspace.model
    policy.at.set_normalizer(policy.normalizer)
    policy.num_inference_steps = int(num_inference_steps)
    policy.eval().to(device)
    return policy, cfg


def _rgb_image(value: Any, key: str) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"{key} must be HWC RGB, got {image.shape}")
    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating) and float(image.max(initial=0.0)) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE):
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(image)


class PickTubeRDPRuntime:
    def __init__(
        self,
        policy,
        tactile_encoder,
        device: torch.device,
        slow_update_interval: int,
        dataset_obs_temporal_downsample_ratio: int,
        n_obs_steps: int,
    ) -> None:
        if slow_update_interval < 1:
            raise ValueError("slow_update_interval must be positive")
        if dataset_obs_temporal_downsample_ratio < 1 or n_obs_steps < 1:
            raise ValueError("Observation steps and temporal downsample ratio must be positive")
        self.policy = policy
        self.tactile_encoder = tactile_encoder
        self.device = device
        self.slow_update_interval = slow_update_interval
        self.temporal_downsample_ratio = dataset_obs_temporal_downsample_ratio
        self.n_obs_steps = n_obs_steps
        self.reset()

    def reset(self) -> None:
        self.step = 0
        self.latent_action: torch.Tensor | None = None
        self.tactile_history: list[torch.Tensor] = []
        self.observation_history: list[dict[str, torch.Tensor]] = []

    def _prepare_observation(
        self, observation: dict[str, Any]
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        missing = [
            key for key in (*CAMERA_KEYS, *TACTILE_KEYS, STATE_KEY) if key not in observation
        ]
        if missing:
            raise ValueError(f"Robot observation is missing keys: {missing}")

        state = np.asarray(observation[STATE_KEY], dtype=np.float32)
        if state.shape != (STATE_DIM,) or not np.isfinite(state).all():
            raise ValueError(f"Expected finite {STATE_DIM}D state, got {state.shape}")

        camera_images = [_rgb_image(observation[key], key) for key in CAMERA_KEYS]
        tactile_images = [_rgb_image(observation[key], key) for key in TACTILE_KEYS]
        camera_tensor = torch.from_numpy(np.stack(camera_images)).to(self.device)
        camera_tensor = camera_tensor.permute(0, 3, 1, 2).float().mul_(1.0 / 255.0)
        tactile_tensor = torch.from_numpy(np.stack(tactile_images)).to(self.device)
        tactile_tensor = tactile_tensor.permute(0, 3, 1, 2).float().mul_(1.0 / 255.0)
        tactile_embedding = self.tactile_encoder(tactile_tensor).reshape(1, 1, -1)
        if tactile_embedding.shape[-1] != 2048:
            raise RuntimeError(f"Expected 2048D tactile embedding, got {tactile_embedding.shape}")

        obs_dict = {
            "camera1": camera_tensor[0].reshape(1, 1, 3, IMAGE_SIZE, IMAGE_SIZE),
            "camera2": camera_tensor[1].reshape(1, 1, 3, IMAGE_SIZE, IMAGE_SIZE),
            "observation_state": torch.from_numpy(state).to(self.device).reshape(1, 1, -1),
            "tactile_embedding": tactile_embedding,
        }
        return obs_dict, tactile_embedding

    def _padded_observation_history(self) -> list[dict[str, torch.Tensor]]:
        raw_steps = self.n_obs_steps * self.temporal_downsample_ratio
        history = self.observation_history[-raw_steps:]
        if not history:
            raise RuntimeError("Observation history is empty")

        if len(history) < raw_steps:
            history = [history[0]] * (raw_steps - len(history)) + history
        return history

    def _slow_policy_observation(self) -> dict[str, torch.Tensor]:
        """Build the same temporally-downsampled history used by the dataset."""
        history = self._padded_observation_history()

        # The dataset selects [1, 3] from a four-frame window when ratio=2.
        selected = history[self.temporal_downsample_ratio - 1::self.temporal_downsample_ratio]
        if len(selected) != self.n_obs_steps:
            raise RuntimeError(
                f"Expected {self.n_obs_steps} slow observations, got {len(selected)}"
            )
        return {
            key: torch.cat([frame[key] for frame in selected], dim=1)
            for key in selected[0]
        }

    @torch.inference_mode()
    def predict(self, observation: dict[str, Any]) -> tuple[np.ndarray, bool]:
        current_obs, tactile_embedding = self._prepare_observation(observation)
        self.observation_history.append(current_obs)
        raw_steps = self.n_obs_steps * self.temporal_downsample_ratio
        if len(self.observation_history) > raw_steps:
            self.observation_history = self.observation_history[-raw_steps:]

        slow_update = self.latent_action is None or self.step % self.slow_update_interval == 0
        if slow_update:
            obs_dict = self._slow_policy_observation()
            result = self.policy.predict_action(
                obs_dict,
                dataset_obs_temporal_downsample_ratio=self.temporal_downsample_ratio,
                return_latent_action=True,
            )
            self.latent_action = result["action"][:, 0].detach()
            # Match the official runner: start decoding with the complete raw
            # observation window, then extend it once per control step.
            self.tactile_history = [
                frame["tactile_embedding"]
                for frame in self._padded_observation_history()
            ]
        else:
            self.tactile_history.append(tactile_embedding)

        extended_obs = {"tactile_embedding": torch.cat(self.tactile_history, dim=1)}
        result = self.policy.predict_from_latent_action(
            self.latent_action,
            extended_obs,
            extended_obs_last_step=len(self.tactile_history),
            dataset_obs_temporal_downsample_ratio=self.temporal_downsample_ratio,
        )
        action = result["action"][0, -1].detach().float().cpu().numpy()
        if action.shape != (ACTION_DIM,) or not np.isfinite(action).all():
            raise RuntimeError(f"Expected finite {ACTION_DIM}D action, got {action.shape}")
        self.step += 1
        return action[None].astype(np.float32, copy=False), slow_update


def _token(connection: dict[str, Any]) -> str | None:
    token_env = str(connection.get("token_env", "VB_ROBOT_TOKEN"))
    token = os.environ.get(token_env)
    if bool(connection.get("require_token", True)) and not token:
        raise ValueError(f"Required authentication variable is not set: {token_env}")
    return token


def run(config_path: Path, device_override: str | None = None) -> None:
    config = load_config(config_path)
    model_config = config["model"]
    connection = config["connection"]
    control = config["control"]
    runtime_config = config["runtime"]
    device = torch.device(device_override or str(model_config.get("device", "cuda:0")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    ldp_checkpoint = resolve_path(str(model_config["ldp_checkpoint"]))
    at_checkpoint = resolve_path(str(model_config["at_checkpoint"]))
    encoder_dir = resolve_path(str(model_config["tactile_encoder_dir"]))
    missing = [path for path in (ldp_checkpoint, at_checkpoint) if not path.is_file()]
    if not encoder_dir.is_dir():
        missing.append(encoder_dir)
    if missing:
        paths = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing RDP deployment files:\n{paths}")
    print(f"[rdp] Loading LDP: {ldp_checkpoint}")
    print(f"[rdp] Loading AT: {at_checkpoint}")
    policy, checkpoint_cfg = load_policy(
        ldp_checkpoint,
        at_checkpoint,
        device,
        int(model_config.get("num_inference_steps", 8)),
    )
    tactile_encoder = load_tactile_resnet18(encoder_dir, device=device)
    rdp = PickTubeRDPRuntime(
        policy,
        tactile_encoder,
        device,
        slow_update_interval=int(control.get("slow_update_interval", 5)),
        dataset_obs_temporal_downsample_ratio=int(
            checkpoint_cfg.dataset_obs_temporal_downsample_ratio
        ),
        n_obs_steps=int(checkpoint_cfg.n_obs_steps),
    )

    bridge = RobotBridgeClient(
        address=str(connection["address"]),
        port=int(connection.get("port", 26421)),
        token=_token(connection),
        add_port=connection.get("add_port"),
        retry_interval_s=float(connection.get("retry_interval_s", 1.0)),
    )
    bridge.send_config(
        {
            "policy_type": "rdp",
            "data_type": "vitac",
            "language_prompt": str(config.get("task", "pick up two tubes")),
            "control_frequency": float(control.get("control_frequency", 30.0)),
            "controller_frequency": float(control.get("controller_frequency", 80.0)),
            "single_arm_mode": False,
            "no_state_obs_mode": False,
            "steps_per_inference": 1,
            "action_horizon": 1,
        }
    )

    ack_timeout = float(connection.get("action_ack_timeout_s", 3.0))
    status_interval = float(runtime_config.get("status_interval_s", 2.0))
    max_iterations = int(runtime_config.get("max_iterations", 0))
    try:
        print("[rdp] Waiting for robot warmup observation")
        _, warmup_observation = bridge.receive_observation()
        for index in range(int(runtime_config.get("warmup_runs", 2))):
            rdp.reset()
            started = time.perf_counter()
            rdp.predict(warmup_observation)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            print(f"[rdp] Warmup {index + 1}: {(time.perf_counter() - started) * 1000:.1f}ms")
        rdp.reset()
        if not bool(runtime_config.get("auto_start", False)):
            input("[rdp] Ready. Press Enter to start the robot... ")
        bridge.send_state("start")

        iteration = 0
        last_status = time.monotonic()
        while max_iterations <= 0 or iteration < max_iterations:
            obs_seq, observation = bridge.receive_observation()
            started = time.perf_counter()
            action, slow_update = rdp.predict(observation)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_ms = (time.perf_counter() - started) * 1000.0
            bridge.send_action(action, obs_seq)
            bridge.receive_action_ack(obs_seq, timeout=ack_timeout)
            iteration += 1
            now = time.monotonic()
            if now - last_status >= status_interval:
                print(
                    f"[rdp] iter={iteration} obs_seq={obs_seq} "
                    f"slow={slow_update} inference_ms={inference_ms:.1f}"
                )
                last_status = now
    except KeyboardInterrupt:
        print("[rdp] Interrupted")
    finally:
        try:
            bridge.send_state("stop")
        finally:
            bridge.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("configs") / "deploy_pick_tube_rdp.yaml",
    )
    parser.add_argument("--device")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.config.expanduser().resolve(), arguments.device)

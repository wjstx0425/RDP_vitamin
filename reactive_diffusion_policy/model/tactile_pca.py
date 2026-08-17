"""Arm-wise PCA projection for the four pick-tube tactile embeddings."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn


TACTILE_SENSOR_ORDER = (
    "observation.images.tactile_left_0",
    "observation.images.tactile_right_0",
    "observation.images.tactile_left_1",
    "observation.images.tactile_right_1",
)
SENSOR_EMBEDDING_DIM = 512
SENSORS_PER_ARM = 2
ARM_COUNT = 2
ARM_INPUT_DIM = SENSOR_EMBEDDING_DIM * SENSORS_PER_ARM
COMPONENTS_PER_ARM = 15
RAW_TACTILE_DIM = ARM_INPUT_DIM * ARM_COUNT
REDUCED_TACTILE_DIM = COMPONENTS_PER_ARM * ARM_COUNT
PCA_FORMAT_VERSION = 1


def group_tactile_embeddings(values: np.ndarray) -> np.ndarray:
    """Return ``[..., 2, 1024]`` values grouped by robot arm.

    The input follows ``TACTILE_SENSOR_ORDER``. ``left_0 + left_1`` form the
    left-arm vector and ``right_0 + right_1`` form the right-arm vector.
    """
    values = np.asarray(values)
    if values.shape[-2:] == (len(TACTILE_SENSOR_ORDER), SENSOR_EMBEDDING_DIM):
        arm_0 = np.concatenate((values[..., 0, :], values[..., 2, :]), axis=-1)
        arm_1 = np.concatenate((values[..., 1, :], values[..., 3, :]), axis=-1)
    elif values.shape[-1:] == (RAW_TACTILE_DIM,):
        reshaped = values.reshape(
            *values.shape[:-1], len(TACTILE_SENSOR_ORDER), SENSOR_EMBEDDING_DIM
        )
        return group_tactile_embeddings(reshaped)
    else:
        raise ValueError(
            "Expected tactile embeddings ending in [4,512] or [2048], "
            f"got {values.shape}"
        )
    return np.stack((arm_0, arm_1), axis=-2)


class BimanualTactilePCA(nn.Module):
    """Apply two independent 1024-to-15 PCA projections."""

    def __init__(self, means: np.ndarray, components: np.ndarray) -> None:
        super().__init__()
        means = np.asarray(means, dtype=np.float32)
        components = np.asarray(components, dtype=np.float32)
        if means.shape != (ARM_COUNT, ARM_INPUT_DIM):
            raise ValueError(
                f"PCA means must be [{ARM_COUNT},{ARM_INPUT_DIM}], got {means.shape}"
            )
        if components.shape != (ARM_COUNT, COMPONENTS_PER_ARM, ARM_INPUT_DIM):
            raise ValueError(
                "PCA components must be "
                f"[{ARM_COUNT},{COMPONENTS_PER_ARM},{ARM_INPUT_DIM}], "
                f"got {components.shape}"
            )
        self.register_buffer("means", torch.from_numpy(means))
        self.register_buffer("components", torch.from_numpy(components))

    @classmethod
    def from_npz(
        cls,
        path: str | Path,
        *,
        device: str | torch.device = "cpu",
    ) -> "BimanualTactilePCA":
        path = Path(path)
        with np.load(path, allow_pickle=False) as artifact:
            version = int(np.asarray(artifact["format_version"]).item())
            if version != PCA_FORMAT_VERSION:
                raise ValueError(
                    f"Unsupported tactile PCA format {version}; expected {PCA_FORMAT_VERSION}"
                )
            sensor_order = tuple(str(value) for value in artifact["sensor_order"].tolist())
            if sensor_order != TACTILE_SENSOR_ORDER:
                raise ValueError(
                    f"Tactile PCA sensor order {sensor_order} does not match {TACTILE_SENSOR_ORDER}"
                )
            model = cls(artifact["means"], artifact["components"])
        return model.eval().to(device)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.shape[-2:] == (len(TACTILE_SENSOR_ORDER), SENSOR_EMBEDDING_DIM):
            arm_0 = torch.cat((values[..., 0, :], values[..., 2, :]), dim=-1)
            arm_1 = torch.cat((values[..., 1, :], values[..., 3, :]), dim=-1)
            grouped = torch.stack((arm_0, arm_1), dim=-2)
        elif values.shape[-1:] == (RAW_TACTILE_DIM,):
            reshaped = values.reshape(
                *values.shape[:-1], len(TACTILE_SENSOR_ORDER), SENSOR_EMBEDDING_DIM
            )
            arm_0 = torch.cat((reshaped[..., 0, :], reshaped[..., 2, :]), dim=-1)
            arm_1 = torch.cat((reshaped[..., 1, :], reshaped[..., 3, :]), dim=-1)
            grouped = torch.stack((arm_0, arm_1), dim=-2)
        else:
            raise ValueError(
                "Expected tactile embeddings ending in [4,512] or [2048], "
                f"got {tuple(values.shape)}"
            )
        grouped = grouped.to(device=self.means.device, dtype=self.means.dtype)
        reduced = torch.einsum(
            "...ad,acd->...ac",
            grouped - self.means,
            self.components,
        )
        return reduced.flatten(start_dim=-2)

    def transform_numpy(self, values: np.ndarray) -> np.ndarray:
        grouped = group_tactile_embeddings(values).astype(np.float32, copy=False)
        means = self.means.detach().cpu().numpy()
        components = self.components.detach().cpu().numpy()
        reduced = np.einsum(
            "...ad,acd->...ac",
            grouped - means,
            components,
            optimize=True,
        )
        return reduced.reshape(*reduced.shape[:-2], REDUCED_TACTILE_DIM)


def save_tactile_pca(
    path: str | Path,
    *,
    means: np.ndarray,
    components: np.ndarray,
    explained_variance_ratio: np.ndarray,
    sample_count: int,
) -> None:
    """Save a portable PCA artifact consumed by conversion and deployment."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        format_version=np.asarray(PCA_FORMAT_VERSION, dtype=np.int64),
        sensor_order=np.asarray(TACTILE_SENSOR_ORDER),
        means=np.asarray(means, dtype=np.float32),
        components=np.asarray(components, dtype=np.float32),
        explained_variance_ratio=np.asarray(explained_variance_ratio, dtype=np.float32),
        sample_count=np.asarray(sample_count, dtype=np.int64),
    )

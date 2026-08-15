"""Shared RDP action and rotation contracts."""

import numpy as np


ACTION_DIM = 20
ARM_ACTION_DIM = 10


def split_action(action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a finite bimanual action into contiguous left and right arms."""
    value = np.asarray(action, dtype=np.float64)
    if value.shape != (ACTION_DIM,) or not np.isfinite(value).all():
        raise ValueError(f"expected finite action shape ({ACTION_DIM},), got {value.shape}")
    return value[:ARM_ACTION_DIM], value[ARM_ACTION_DIM:]


def rot6d_columns_to_matrix(values: np.ndarray) -> np.ndarray:
    """Turn the two-column 6D rotation representation into an SO(3) matrix."""
    pair = np.asarray(values, dtype=np.float64).reshape(2, 3)
    first = pair[0] / np.linalg.norm(pair[0])
    second = pair[1] - np.dot(first, pair[1]) * first
    second = second / np.linalg.norm(second)
    return np.column_stack((first, second, np.cross(first, second)))


def rotation_geodesic(first: np.ndarray, second: np.ndarray) -> float:
    """Return the shortest angular distance between two rotation matrices."""
    relative = np.asarray(first).T @ np.asarray(second)
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(cosine))

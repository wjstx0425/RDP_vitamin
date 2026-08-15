import numpy as np
import pytest

from tools.rdp_debug.common import rotation_geodesic
from tools.rdp_debug.common import split_action


def test_split_action_uses_contiguous_left_right_layout() -> None:
    action = np.arange(20, dtype=np.float64)
    left, right = split_action(action)
    np.testing.assert_array_equal(left, np.arange(10))
    np.testing.assert_array_equal(right, np.arange(10, 20))


def test_rotation_geodesic_handles_axis_angle_branch_wrap() -> None:
    from scipy.spatial.transform import Rotation

    first = Rotation.from_rotvec([np.pi - 1e-4, 0.0, 0.0]).as_matrix()
    second = Rotation.from_rotvec([-np.pi + 1e-4, 0.0, 0.0]).as_matrix()
    assert rotation_geodesic(first, second) == pytest.approx(2e-4, abs=1e-8)

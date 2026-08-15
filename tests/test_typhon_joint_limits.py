import inspect
from pathlib import Path
from typing import ClassVar
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import yaml

from real_world.robot_api.arm import RobotControl_pykin as robot_control_module
from real_world.robot_api.arm.RobotControl_pykin import RobotControl

REPO_ROOT = Path(__file__).resolve().parents[1]
LEFT_JOINT_NAMES = [f"left-joint_arm_{index}" for index in range(1, 8)]
RIGHT_JOINT_NAMES = [f"right-joint_arm_{index}" for index in range(1, 8)]


class FakeTyphonWrapper:
    def __init__(self, **kwargs) -> None:
        del kwargs


class FakeSingleArm:
    setup_link_name_calls: ClassVar[list[tuple[str, str]]] = []

    def __init__(self, urdf_path) -> None:
        del urdf_path

    def setup_link_name(self, base_link, ee_link) -> None:
        type(self).setup_link_name_calls.append((base_link, ee_link))


@pytest.fixture
def typhon_init_without_hardware(monkeypatch):
    FakeSingleArm.setup_link_name_calls.clear()
    monkeypatch.setattr(robot_control_module, "RobotWrapperTyphon", FakeTyphonWrapper)
    monkeypatch.setattr(
        robot_control_module,
        "prepare_urdf_for_pykin",
        lambda urdf_path: ("unused.urdf", "/tmp/nonexistent-test-urdf"),
    )
    monkeypatch.setattr(robot_control_module, "SingleArm", FakeSingleArm)


@pytest.mark.parametrize(
    ("urdf_name", "joint_names"),
    [
        pytest.param("AM2_left.urdf", LEFT_JOINT_NAMES, id="left"),
        pytest.param("AM2_right.urdf", RIGHT_JOINT_NAMES, id="right"),
    ],
)
def test_real_am2_joint_limits_are_complete_and_finite(urdf_name, joint_names) -> None:
    urdf_path = REPO_ROOT / "real_world" / "robot_api" / "assets" / urdf_name

    limits = RobotControl._extract_joint_limits(str(urdf_path), joint_names)

    assert len(limits) == 7
    assert all(np.isfinite(lower) and np.isfinite(upper) for lower, upper in limits)
    assert all(lower < upper for lower, upper in limits)


def test_unknown_joint_limit_fails_closed() -> None:
    unknown_joint = "left-joint_arm_unknown"
    urdf_path = REPO_ROOT / "real_world" / "robot_api" / "assets" / "AM2_left.urdf"

    with pytest.raises(RuntimeError, match=unknown_joint):
        RobotControl._extract_joint_limits(str(urdf_path), [unknown_joint])


@pytest.mark.parametrize(
    ("arm_name", "joint_names"),
    [
        pytest.param("left_arm", [], id="zero"),
        pytest.param("left_arm", LEFT_JOINT_NAMES[:6], id="six"),
        pytest.param(
            "right_arm",
            [*RIGHT_JOINT_NAMES, "right-joint_arm_8"],
            id="eight",
        ),
        pytest.param(
            "left_arm",
            [*LEFT_JOINT_NAMES[:6], LEFT_JOINT_NAMES[0]],
            id="duplicate",
        ),
    ],
)
def test_typhon_init_rejects_invalid_joint_layout_before_hardware_use(
    typhon_init_without_hardware,
    arm_name,
    joint_names,
) -> None:
    del typhon_init_without_hardware
    kwargs = {
        "left_joint_names": LEFT_JOINT_NAMES,
        "right_joint_names": RIGHT_JOINT_NAMES,
    }
    kwargs[f"{arm_name.removesuffix('_arm')}_joint_names"] = joint_names

    with pytest.raises(
        RuntimeError,
        match=rf"{arm_name}.*expected=7.*actual={len(joint_names)}",
    ) as exc_info:
        RobotControl(config_path=None, **kwargs)

    if len(set(joint_names)) != len(joint_names):
        assert f"unique={len(set(joint_names))}" in str(exc_info.value)


@pytest.mark.parametrize(
    ("target_dimension", "limit_dimension"),
    [
        pytest.param(7, 6, id="too-few-limits"),
        pytest.param(6, 7, id="too-many-limits"),
    ],
)
def test_sanitize_joint_targets_rejects_dimension_mismatch(
    target_dimension,
    limit_dimension,
) -> None:
    control = object.__new__(RobotControl)
    control._joint_limits = {"left_arm": [(-1.0, 1.0)] * limit_dimension}

    with pytest.raises(
        RuntimeError,
        match=rf"left_arm.*target={target_dimension}.*limits={limit_dimension}",
    ):
        control.sanitize_joint_targets("left_arm", [0.0] * target_dimension)


def test_robot_control_exposes_only_typhon_backend() -> None:
    parameters = inspect.signature(RobotControl).parameters

    assert parameters["config_path"].default == "configs/typhon_am2.yaml"
    assert "robot_type" not in parameters
    assert "vel_max" not in parameters
    assert not hasattr(robot_control_module, "RobotWrapperEyou")
    assert not hasattr(robot_control_module, "EYOU_AVAILABLE")


def test_robot_control_uses_injected_wrapper(
    typhon_init_without_hardware,
) -> None:
    del typhon_init_without_hardware
    injected = object()

    control = RobotControl(config_path=None, robot_wrapper=injected)

    assert control.robot is injected


def test_robot_control_without_config_uses_arm_7_ee_links(
    typhon_init_without_hardware,
) -> None:
    del typhon_init_without_hardware

    RobotControl(config_path=None, robot_wrapper=object())

    assert FakeSingleArm.setup_link_name_calls == [
        ("base_link", "left-link_arm_7"),
        ("base_link", "right-link_arm_7"),
    ]


def test_typhon_am2_config_uses_flat_urdfs_and_ordered_joint_names() -> None:
    config_path = REPO_ROOT / "configs" / "typhon_am2.yaml"

    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    assert config["urdf_path_left"] == "real_world/robot_api/assets/AM2_left.urdf"
    assert config["urdf_path_right"] == "real_world/robot_api/assets/AM2_right.urdf"
    assert (config["ee_link_left"], config["ee_link_right"]) == (
        "left-link_arm_7",
        "right-link_arm_7",
    )
    for side in ("left", "right"):
        urdf_path = REPO_ROOT / config[f"urdf_path_{side}"]
        link_names = {
            link.attrib["name"]
            for link in ET.parse(urdf_path).getroot().findall("link")
        }
        assert config[f"ee_link_{side}"] in link_names
    assert config["left_joint_names"] == LEFT_JOINT_NAMES
    assert config["right_joint_names"] == RIGHT_JOINT_NAMES
    assert config["typhon"] == {
        "base_url": "http://192.168.100.100:8081",
        "timeout": 5.0,
        "auto_enter_control_mode": True,
    }
    assert "robot_type" not in config
    assert "vel_max" not in config
    assert not (REPO_ROOT / "configs" / "am2.yaml").exists()

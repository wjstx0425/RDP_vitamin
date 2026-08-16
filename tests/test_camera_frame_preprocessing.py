import cv2
import numpy as np
import pytest

from real_world import bimanual_umi_env as env_module


def _training_resize_rgb(panel: np.ndarray) -> np.ndarray:
    resized = cv2.resize(
        panel,
        (224, 224),
        interpolation=cv2.INTER_LINEAR,
    )
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)


def _capture_deploy_transform(
    monkeypatch,
    *,
    obs_float32: bool,
    camera_idx: int = 0,
):
    captured = {}

    class FakeMultiUvcCamera:
        def __init__(self, **kwargs) -> None:
            captured["transforms"] = kwargs["transform"]

    monkeypatch.setattr(
        env_module.VideoRecorder,
        "create_hevc_nvenc",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(env_module, "MultiUvcCamera", FakeMultiUvcCamera)
    monkeypatch.setattr(env_module, "Controller", lambda **_kwargs: object())

    env_module.BimanualUmiEnv(
        cam_path=["/dev/video-test-0", "/dev/video-test-1"],
        data_type="vitac",
        obs_image_resolution=(224, 224),
        obs_float32=obs_float32,
        shm_manager=object(),
    )
    return captured["transforms"][camera_idx]


@pytest.fixture
def triptych_bgr_frame() -> np.ndarray:
    height, panel_width = 800, 1280
    x = np.broadcast_to(
        np.linspace(0, 255, panel_width, dtype=np.uint8)[None, :],
        (height, panel_width),
    )
    y = np.broadcast_to(
        np.linspace(0, 255, height, dtype=np.uint8)[:, None],
        (height, panel_width),
    )

    left_tactile = np.stack((x, y, np.full_like(x, 17)), axis=-1)
    visual = np.stack((x, np.full_like(x, 80), 255 - x), axis=-1)
    right_tactile = np.stack((np.full_like(x, 33), x, y), axis=-1)

    visual[:, :96] = (10, 20, 200)
    visual[:, -96:] = (30, 220, 40)
    left_tactile[:96, :96] = (7, 31, 211)
    left_tactile[-96:, -96:] = (171, 61, 19)

    return np.concatenate((left_tactile, visual, right_tactile), axis=1)


@pytest.fixture(params=[0, 1])
def deploy_transform(monkeypatch, request):
    return _capture_deploy_transform(
        monkeypatch,
        obs_float32=False,
        camera_idx=request.param,
    )


def test_vitac_transform_preserves_left_tactile_panel_orientation(
    deploy_transform,
    triptych_bgr_frame: np.ndarray,
) -> None:
    left_tactile, visual, right_tactile = np.split(
        triptych_bgr_frame,
        3,
        axis=1,
    )

    expected = {
        "color": _training_resize_rgb(visual),
        "left_tactile": _training_resize_rgb(left_tactile),
        "right_tactile": _training_resize_rgb(right_tactile),
    }

    actual = deploy_transform({"color": triptych_bgr_frame.copy()})

    assert set(actual) == set(expected)
    for key, expected_image in expected.items():
        assert actual[key].shape == (224, 224, 3)
        assert actual[key].dtype == np.uint8
        np.testing.assert_array_equal(actual[key], expected_image)

    assert actual["color"][112, 0].tolist() == [200, 20, 10]
    assert actual["color"][112, -1].tolist() == [40, 220, 30]
    assert actual["left_tactile"][0, 0].tolist() == [211, 31, 7]
    assert actual["left_tactile"][-1, -1].tolist() == [19, 61, 171]


def test_vitac_transform_preserves_float32_output_contract(
    monkeypatch,
    triptych_bgr_frame: np.ndarray,
) -> None:
    deploy_transform = _capture_deploy_transform(
        monkeypatch,
        obs_float32=True,
    )
    left_tactile, visual, right_tactile = np.split(
        triptych_bgr_frame,
        3,
        axis=1,
    )
    expected = {
        "color": _training_resize_rgb(visual).astype(np.float32) / 255,
        "left_tactile": _training_resize_rgb(left_tactile).astype(np.float32)
        / 255,
        "right_tactile": _training_resize_rgb(right_tactile).astype(np.float32)
        / 255,
    }

    actual = deploy_transform({"color": triptych_bgr_frame.copy()})

    assert set(actual) == set(expected)
    for key, expected_image in expected.items():
        assert actual[key].shape == (224, 224, 3)
        assert actual[key].dtype == np.float32
        assert 0.0 <= actual[key].min() <= actual[key].max() <= 1.0
        np.testing.assert_array_equal(actual[key], expected_image)

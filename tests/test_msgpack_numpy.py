from pathlib import Path
import tomllib

import numpy as np
import pytest

from client import msgpack_numpy
from client.interface_client import InterfaceClient


def _check(expected, actual):
    if isinstance(expected, np.ndarray):
        assert expected.shape == actual.shape
        assert expected.dtype == actual.dtype
        assert np.array_equal(expected, actual, equal_nan=expected.dtype.kind == "f")
    elif isinstance(expected, dict):
        assert expected.keys() == actual.keys()
        for key in expected:
            _check(expected[key], actual[key])
    elif isinstance(expected, (list, tuple)):
        assert len(expected) == len(actual)
        for expected_item, actual_item in zip(expected, actual, strict=True):
            _check(expected_item, actual_item)
    else:
        assert expected == actual


@pytest.mark.parametrize(
    "data",
    [
        1,
        1.0,
        "hello",
        np.bool_(True),
        np.array([1, 2, 3])[0],
        np.str_("asdf"),
        [1, 2, 3],
        {"key": [1, 2, 3]},
        np.array(1.0),
        np.array([1, 2, 3], dtype=np.int32),
        np.array(["asdf", "qwer"]),
        np.array([True, False]),
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        np.array([np.nan, np.inf, -np.inf]),
        {"arr": np.array([1, 2, 3]), "nested": {"arr": np.array([4, 5, 6])}},
    ],
)
def test_pack_unpack(data):
    _check(data, msgpack_numpy.unpackb(msgpack_numpy.packb(data)))


@pytest.mark.parametrize(
    "data",
    [
        np.array([(1,)], dtype=[("value", "i4")]),
        np.array([object()], dtype=object),
        np.array([1 + 2j], dtype=np.complex64),
    ],
)
def test_pack_rejects_unsafe_numpy_dtypes(data):
    with pytest.raises(ValueError, match="Unsupported dtype"):
        msgpack_numpy.packb(data)


def test_interface_client_uses_shared_packer(monkeypatch):
    class FakePacker:
        pass

    websocket = object()
    monkeypatch.setattr(msgpack_numpy, "Packer", FakePacker)
    monkeypatch.setattr(InterfaceClient, "_connect", lambda self: websocket)
    monkeypatch.setattr(InterfaceClient, "_expect_hello", lambda self: None)

    client = InterfaceClient()

    assert isinstance(client._packer, FakePacker)
    assert client._ws is websocket


def test_project_has_minimal_server_dependencies():
    root = Path(__file__).resolve().parents[1]
    with (root / "pyproject.toml").open("rb") as project_file:
        project = tomllib.load(project_file)
    dependencies = project["project"]["dependencies"]
    sources = project["tool"]["uv"].get("sources", {})
    lock_text = (root / "uv.lock").read_text(encoding="utf-8")

    assert project["project"]["name"] == "vb3-robot-server"
    assert not any(dependency.startswith("openpi-client") for dependency in dependencies)
    assert any(dependency.startswith("msgpack>=") for dependency in dependencies)
    assert any(dependency.startswith("websockets>=") for dependency in dependencies)
    assert "openpi-client" not in sources
    assert "workspace" not in project["tool"]["uv"]
    assert "openpi-client" not in lock_text
    assert 'editable = "packages/openpi-client"' not in lock_text
    assert not (root / "packages").exists()

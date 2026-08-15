import json
import os
from pathlib import Path
import stat
import subprocess

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPOSITORY_ROOT / "scripts" / "bimanual_smolvla.sh"
SETUP_SCRIPT = REPOSITORY_ROOT / "scripts" / "setup_environment.sh"


def test_environment_setup_uses_managed_uv_without_conda():
    source = SETUP_SCRIPT.read_text(encoding="utf-8")

    assert "uv/install.sh" in source
    assert 'PYTHON_VERSION="3.11"' in source
    assert "--locked" in source
    assert "--managed-python" in source
    assert "conda" not in source.lower()


def test_environment_setup_reinstalls_opencv_after_removing_conflicting_variants():
    source = SETUP_SCRIPT.read_text(encoding="utf-8")

    assert "--reinstall-package" in source
    assert "opencv-python" in source


def test_smolvla_launcher_runs_from_repository_root_and_forwards_arguments(tmp_path):
    """The wrapper is executable, relocates to the repository, and preserves Click args."""
    assert LAUNCHER.exists(), "SmolVLA launcher does not exist"
    assert stat.S_IMODE(LAUNCHER.stat().st_mode) & stat.S_IXUSR

    captured_invocation = tmp_path / "python-invocation.json"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/usr/bin/env python3\n"
        "import json\n"
        "import os\n"
        "import stat\n"
        "import sys\n"
        "from pathlib import Path\n"
        "args = sys.argv[1:]\n"
        "token_path = Path(args[args.index('--token-file') + 1])\n"
        "Path(os.environ['PYTHON_CAPTURE']).write_text(\n"
        "    json.dumps({\n"
        "        'cwd': os.getcwd(),\n"
        "        'argv': args,\n"
        "        'token': token_path.read_text().strip(),\n"
        "        'token_mode': stat.S_IMODE(token_path.stat().st_mode),\n"
        "    })\n"
        ")\n"
    )
    fake_python.chmod(0o755)

    outside_repository = tmp_path / "outside repository"
    outside_repository.mkdir()
    argument_path = tmp_path / "model checkpoints" / "latest checkpoint"
    arguments = ["--checkpoint", str(argument_path), "--task", "pick-and-place"]
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHON_CAPTURE": str(captured_invocation),
            "VB3_SERVER_PYTHON": str(fake_python),
            "VB3_TOKEN_FILE": str(tmp_path / "missing-token-list.txt"),
            "VB_ROBOT_TOKEN": "launcher-test-token",
        }
    )

    subprocess.run(
        [str(LAUNCHER), *arguments],
        cwd=outside_repository,
        env=environment,
        check=True,
    )

    invocation = json.loads(captured_invocation.read_text())
    assert invocation["cwd"] == str(REPOSITORY_ROOT)
    assert invocation["argv"][: 1 + len(arguments)] == [
        "deploy_scripts/bimanual_smolvla_online.py",
        *arguments,
    ]
    assert invocation["argv"][-2] == "--token-file"
    assert invocation["token"] == "launcher-test-token"
    assert invocation["token_mode"] == 0o600
    assert not Path(invocation["argv"][-1]).exists()


def test_smolvla_launcher_has_no_machine_specific_runtime_paths():
    source = LAUNCHER.read_text(encoding="utf-8")

    assert '.venv/bin/python' in source
    assert "/home/" not in source
    assert "CONDA_PREFIX" not in source
    assert "LD_LIBRARY_PATH" not in source

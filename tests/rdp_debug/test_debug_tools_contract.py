from pathlib import Path
import subprocess
import sys


SCRIPT_NAMES = (
    "summarize_action_log.py",
    "audit_training_dataset.py",
    "replay_saved_observations.py",
    "compare_policy_stages.py",
)


def test_all_debug_clis_have_hardware_free_help() -> None:
    for name in SCRIPT_NAMES:
        result = subprocess.run(
            [sys.executable, f"tools/rdp_debug/{name}", "--help"],
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout.lower()


def test_debug_modules_do_not_import_online_components() -> None:
    forbidden = (
        "RobotBridgeClient",
        "websockets",
        "requests",
        "BimanualUmiEnv",
        "RobotWrapper",
        "/dev/video",
    )
    for name in SCRIPT_NAMES:
        source = Path(f"tools/rdp_debug/{name}").read_text(encoding="utf-8")
        assert not any(token in source for token in forbidden)

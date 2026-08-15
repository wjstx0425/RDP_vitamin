from pathlib import Path

from configs.server_config import SERVER_CONFIG
from configs.server_config import SmolVLAServerConfig
from deploy_scripts.bimanual_smolvla_online import SMOLVLA_OBSERVATION_RESOLUTION
from real_world.robot_api.arm.Controller import COMMAND_QUEUE_CAPACITY
from real_world.robot_api.arm.Controller import DEBUG_QUEUE_CAPACITY


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_visual_runtime_matches_training_rate_and_resolution() -> None:
    assert SERVER_CONFIG.camera.capture_fps == 30
    assert SMOLVLA_OBSERVATION_RESOLUTION == (256, 256)


def test_parent_opencv_is_single_threaded_before_camera_processes_fork() -> None:
    source = (
        REPOSITORY_ROOT / "deploy_scripts" / "bimanual_smolvla_online.py"
    ).read_text(encoding="utf-8")

    configure_opencv = source.index("cv2.setNumThreads(1)")
    start_shared_memory = source.index("with SharedMemoryManager() as shm_manager:")

    assert configure_opencv < start_shared_memory


def test_server_config_collects_runtime_defaults() -> None:
    config = SmolVLAServerConfig()

    assert config.observation_resolution == (256, 256)
    assert config.obs_float32 is False
    assert config.control_frequency == 30.0
    assert config.controller_frequency == 80.0
    assert config.effective_camera_obs_latency == config.camera.capture_timestamp_delay
    assert config.action_horizon == 20
    assert config.n_robots == 2
    assert config.action_dim == 20
    assert config.steps_per_inference == 5
    assert config.max_executed_actions == 5


def test_controller_queues_are_bounded_for_runtime_traffic() -> None:
    assert 50 <= COMMAND_QUEUE_CAPACITY <= 4096
    assert 1000 <= DEBUG_QUEUE_CAPACITY <= 100_000

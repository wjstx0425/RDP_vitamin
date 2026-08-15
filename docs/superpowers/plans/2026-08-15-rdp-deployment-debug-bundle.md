# RDP Deployment Debug Bundle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a hardware-isolated Chinese debug runbook and four executable offline tools that localize pick-tube failures from recorded robot actions back through deployment observations, LDP conditioning, AT reconstruction, and source demonstrations.

**Architecture:** Put dependency-light parsing and metrics in `tools/rdp_debug/common.py`; keep each CLI in a focused module whose model- or dataset-heavy imports occur only after argument validation. The server repository owns the diagnostic entry points and synthetic tests, while model execution is delegated explicitly to a user-supplied Vitamin repository path.

**Tech Stack:** Python 3.11 for server-side tests, NumPy/SciPy, argparse, pytest, optional Zarr/PyArrow readers on the training machine, and the Python 3.12 Vitamin environment for Torch checkpoint replay.

## Global Constraints

- All diagnostics are offline and read-only: no WebSocket, HTTP, camera device, Typhon, `RobotBridgeClient`, `BimanualUmiEnv`, or controller construction.
- Preserve the user's uncommitted `configs/server_config.py`; no task may stage or modify it.
- State is exactly `[L rel-start xyz+axis-angle(6), L grip, R rel-start xyz+axis-angle(6), R grip, L wrt R xyz+axis-angle(6)]` with shape `(20,)`.
- Action is exactly `[L xyz(3), L rot6d columns(6), L grip, R xyz(3), R rot6d columns(6), R grip]` with shape `(20,)`.
- Camera mapping is `camera0 -> policy camera1`, `camera1 -> policy camera2`; tactile order is `left_0, right_0, left_1, right_1`, four 512D embeddings flattened to 2048D.
- Rotation jump metrics use SO(3) geodesic distance and must not subtract axis-angle vectors directly.
- Scripts write nothing unless `--output` is supplied; output paths must not already exist.
- Model tools load only local trusted checkpoints and never download data or weights.
- Every new production function follows a witnessed RED-GREEN TDD cycle.

---

### Task 1: Shared contracts and action-log summarizer

**Files:**
- Create: `tools/__init__.py`
- Create: `tools/rdp_debug/__init__.py`
- Create: `tools/rdp_debug/common.py`
- Create: `tools/rdp_debug/summarize_action_log.py`
- Create: `tests/rdp_debug/test_common.py`
- Create: `tests/rdp_debug/test_summarize_action_log.py`

**Interfaces:**
- Consumes: newline-delimited action-debug records produced by `deploy_scripts/bimanual_smolvla_online.py`.
- Produces: `split_action(action) -> tuple[np.ndarray, np.ndarray]`, `rot6d_columns_to_matrix(values) -> np.ndarray`, `rotation_geodesic(a, b) -> float`, `percentiles(values) -> dict[str, float]`, `load_jsonl(path) -> list[dict]`, and `summarize_records(records, replan_interval) -> dict`.

- [ ] **Step 1: Write failing common-math tests**

```python
import numpy as np

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
```

- [ ] **Step 2: Verify the common-math tests fail for the missing module**

Run: `uv run pytest tests/rdp_debug/test_common.py -q`

Expected: collection fails with `ModuleNotFoundError: No module named 'tools.rdp_debug'`.

- [ ] **Step 3: Implement the shared action and rotation contracts**

```python
ACTION_DIM = 20
ARM_ACTION_DIM = 10


def split_action(action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    value = np.asarray(action, dtype=np.float64)
    if value.shape != (ACTION_DIM,) or not np.isfinite(value).all():
        raise ValueError(f"expected finite action shape ({ACTION_DIM},), got {value.shape}")
    return value[:ARM_ACTION_DIM], value[ARM_ACTION_DIM:]


def rot6d_columns_to_matrix(values: np.ndarray) -> np.ndarray:
    pair = np.asarray(values, dtype=np.float64).reshape(2, 3)
    first = pair[0] / np.linalg.norm(pair[0])
    second = pair[1] - np.dot(first, pair[1]) * first
    second = second / np.linalg.norm(second)
    return np.column_stack((first, second, np.cross(first, second)))


def rotation_geodesic(first: np.ndarray, second: np.ndarray) -> float:
    relative = np.asarray(first).T @ np.asarray(second)
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(cosine))
```

- [ ] **Step 4: Verify shared tests pass**

Run: `uv run pytest tests/rdp_debug/test_common.py -q`

Expected: `2 passed`.

- [ ] **Step 5: Write failing JSONL summary tests**

```python
def record(index: int, time_s: float, left_grip: float) -> dict:
    return {
        "time": time_s,
        "iter_idx": index,
        "obs_seq": index + 1,
        "raw_action_len": 1,
        "new_action_len": 1,
        "controller_records": [{
            "scheduled": True,
            "target_time": time_s + 0.05,
            "left_target_pose": [0, 0, 0, 0, 0, 0],
            "right_target_pose": [0, 0, 0, 0, 0, 0],
            "left_gripper": [left_grip],
            "right_gripper": [0.03],
        }],
    }


def test_summary_separates_replan_boundary_gripper_jumps() -> None:
    rows = [record(i, i * 0.05, 0.01 if i < 5 else 0.03) for i in range(10)]
    report = summarize_records(rows, replan_interval=5)
    assert report["frames"] == 10
    assert report["effective_hz"] == pytest.approx(20.0)
    assert report["scheduled"] == 10
    assert report["gripper_jump_m"]["left"]["boundary_mean"] == pytest.approx(0.02)
    assert report["gripper_jump_m"]["left"]["within_mean"] == pytest.approx(0.0)


def test_load_jsonl_reports_source_line(tmp_path: Path) -> None:
    source = tmp_path / "bad.jsonl"
    source.write_text('{"time": 1}\nnot-json\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad.jsonl:2"):
        load_jsonl(source)
```

- [ ] **Step 6: Verify summary tests fail because functions are absent**

Run: `uv run pytest tests/rdp_debug/test_summarize_action_log.py -q`

Expected: collection fails on missing `summarize_records` or `load_jsonl`.

- [ ] **Step 7: Implement strict JSONL loading, metrics, and CLI**

```python
def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error.msg}") from error
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    if len(rows) < 2:
        raise ValueError(f"{path}: expected at least two records")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize offline RDP action-debug JSONL files")
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--replan-interval", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {str(path): summarize_records(load_jsonl(path), args.replan_interval) for path in args.logs}
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        write_new_text(args.output, rendered + "\n")
    else:
        print(rendered)
```

The implementation validates monotonic `iter_idx`/`obs_seq`, computes duration as `last.time - first.time`, frequency as `(frames - 1) / duration`, percentile summaries for adjacent periods and target lead, and SO(3) distances from each consecutive six-value controller target pose via `scipy.spatial.transform.Rotation.from_rotvec`.

- [ ] **Step 8: Run focused and real-log verification**

Run: `uv run pytest tests/rdp_debug/test_common.py tests/rdp_debug/test_summarize_action_log.py -q`

Expected: all focused tests pass.

Run: `uv run python tools/rdp_debug/summarize_action_log.py action_debug_logs/20260815_145643/action_debug.jsonl action_debug_logs/20260815_160855/action_debug.jsonl`

Expected: JSON reports approximately `9.34 Hz` and `19.76 Hz`, with 112 and 249 scheduled records respectively.

- [ ] **Step 9: Commit Task 1**

```bash
git add tools/__init__.py tools/rdp_debug/__init__.py tools/rdp_debug/common.py \
  tools/rdp_debug/summarize_action_log.py tests/rdp_debug/test_common.py \
  tests/rdp_debug/test_summarize_action_log.py
git commit -m "feat: add offline RDP action log diagnostics"
```

### Task 2: Training-dataset auditor

**Files:**
- Create: `tools/rdp_debug/audit_training_dataset.py`
- Create: `tests/rdp_debug/test_audit_training_dataset.py`

**Interfaces:**
- Consumes: either `<dataset>/replay_buffer.zarr/{data/action,data/observation_state,meta/episode_ends}` or a LeRobot root containing `meta/episodes.jsonl` and `data/chunk-*/episode_*.parquet` with `actions` and `observation.state`.
- Produces: `DatasetArrays(action, state, episode_ends)`, `load_dataset(path, source_format)`, `scan_lag(action_motion, state_motion, max_lag)`, and `audit_dataset(data, start_windows, movement_threshold, max_lag)`.

- [ ] **Step 1: Write failing tests for layout, episode-start energy, and lag recovery**

```python
def synthetic_data() -> DatasetArrays:
    action = np.zeros((12, 20), dtype=np.float64)
    state = np.zeros((12, 20), dtype=np.float64)
    action[1:4, 0] = 0.004
    action[7:10, 10] = 0.003
    state[2:5, 0] = np.cumsum(action[1:4, 0])
    state[8:11, 7] = np.cumsum(action[7:10, 10])
    return DatasetArrays(action=action, state=state, episode_ends=np.array([6, 12]))


def test_audit_identifies_first_moving_side_per_episode() -> None:
    report = audit_dataset(synthetic_data(), start_windows=(3, 6), movement_threshold=0.001, max_lag=3)
    assert report["episodes"][0]["first_moving_side"] == "left"
    assert report["episodes"][1]["first_moving_side"] == "right"


def test_lag_scan_recovers_one_frame_state_response() -> None:
    motion = np.array([0.0, 1.0, 2.0, 0.0, 0.0])
    response = np.array([0.0, 0.0, 1.0, 2.0, 0.0])
    result = scan_lag(motion, response, max_lag=2)
    assert result["best_lag_frames"] == 1
    assert result["correlation"] == pytest.approx(1.0)
```

- [ ] **Step 2: Verify auditor tests fail for the missing module**

Run: `uv run pytest tests/rdp_debug/test_audit_training_dataset.py -q`

Expected: collection fails with missing `audit_training_dataset`.

- [ ] **Step 3: Implement format readers with lazy optional imports**

```python
@dataclass(frozen=True)
class DatasetArrays:
    action: np.ndarray
    state: np.ndarray
    episode_ends: np.ndarray


def load_zarr(path: Path) -> DatasetArrays:
    import zarr

    root_path = path / "replay_buffer.zarr" if (path / "replay_buffer.zarr").is_dir() else path
    root = zarr.open_group(str(root_path), mode="r")
    return validate_arrays(
        np.asarray(root["data/action"]),
        np.asarray(root["data/observation_state"]),
        np.asarray(root["meta/episode_ends"]),
        root_path,
    )


def load_lerobot(path: Path) -> DatasetArrays:
    import pyarrow.parquet as pq

    actions: list[np.ndarray] = []
    states: list[np.ndarray] = []
    ends: list[int] = []
    total = 0
    for parquet in sorted((path / "data").glob("chunk-*/episode_*.parquet")):
        table = pq.read_table(parquet, columns=["actions", "observation.state"])
        action = np.asarray(table["actions"].to_pylist(), dtype=np.float64)
        state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)
        actions.append(action)
        states.append(state)
        total += len(action)
        ends.append(total)
    if not actions:
        raise FileNotFoundError(f"{path}: no data/chunk-*/episode_*.parquet files")
    return validate_arrays(np.concatenate(actions), np.concatenate(states), np.asarray(ends), path)
```

- [ ] **Step 4: Implement per-side motion, first-motion, and lag reports**

```python
def xyz_norms(action: np.ndarray, side: str) -> np.ndarray:
    offset = 0 if side == "left" else 10
    return np.linalg.norm(action[:, offset : offset + 3], axis=1)


def first_moving_side(action: np.ndarray, threshold: float) -> str:
    left_hits = np.flatnonzero(xyz_norms(action, "left") >= threshold)
    right_hits = np.flatnonzero(xyz_norms(action, "right") >= threshold)
    left_index = int(left_hits[0]) if left_hits.size else math.inf
    right_index = int(right_hits[0]) if right_hits.size else math.inf
    if left_index == right_index:
        return "simultaneous" if left_index != math.inf else "none"
    return "left" if left_index < right_index else "right"
```

The aggregate report includes finite/schema checks, all-frame and first-30/60-frame position RMS in millimetres per step, rot6d residual-to-identity RMS, gripper min/mean/std/max, per-episode first-moving side counts, and lag scans for each arm using action xyz norm versus the matching state xyz consecutive-difference norm.

- [ ] **Step 5: Implement CLI and explicit output behavior**

```python
def main() -> None:
    parser = argparse.ArgumentParser(description="Audit ordered pick-tube training actions and states")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--format", choices=("auto", "zarr", "lerobot"), default="auto")
    parser.add_argument("--start-windows", nargs="+", type=int, default=[30, 60])
    parser.add_argument("--movement-threshold-m", type=float, default=0.001)
    parser.add_argument("--max-lag", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    data = load_dataset(args.dataset.resolve(), args.format)
    report = audit_dataset(data, tuple(args.start_windows), args.movement_threshold_m, args.max_lag)
    emit_json(report, args.output)
```

- [ ] **Step 6: Verify tests and CLI help pass without Zarr/PyArrow installed**

Run: `uv run pytest tests/rdp_debug/test_audit_training_dataset.py -q`

Expected: all auditor tests pass.

Run: `uv run python tools/rdp_debug/audit_training_dataset.py --help`

Expected: exit 0; optional dataset libraries are not imported.

- [ ] **Step 7: Commit Task 2**

```bash
git add tools/rdp_debug/audit_training_dataset.py tests/rdp_debug/test_audit_training_dataset.py
git commit -m "feat: add ordered training dataset audit"
```

### Task 3: Saved-observation replay

**Files:**
- Create: `tools/rdp_debug/replay_saved_observations.py`
- Create: `tests/rdp_debug/test_replay_saved_observations.py`

**Interfaces:**
- Consumes: `--vitamin-repo`, `--config`, and saved `step_*` directories with state NPY plus two camera and four tactile JPEG files.
- Produces: `discover_steps(root) -> list[Path]`, `load_saved_observation(step, cv2_module) -> dict[str, np.ndarray]`, `summarize_actions(actions, slow_flags, timings_ms, replan_interval) -> dict`, and a subprocess-safe CLI executed in the Vitamin environment.

- [ ] **Step 1: Write failing discovery and input-contract tests**

```python
EXPECTED_IMAGE_KEYS = (
    "observation.images.camera0",
    "observation.images.camera1",
    "observation.images.tactile_left_0",
    "observation.images.tactile_right_0",
    "observation.images.tactile_left_1",
    "observation.images.tactile_right_1",
)


def test_discover_steps_requires_contiguous_numbering(tmp_path: Path) -> None:
    (tmp_path / "step_000001").mkdir()
    (tmp_path / "step_000003").mkdir()
    with pytest.raises(ValueError, match="missing saved step 000002"):
        discover_steps(tmp_path)


def test_source_has_no_online_robot_dependencies() -> None:
    source = Path("tools/rdp_debug/replay_saved_observations.py").read_text(encoding="utf-8")
    for forbidden in ("RobotBridgeClient", "websockets", "requests", "BimanualUmiEnv", "/dev/video"):
        assert forbidden not in source
```

- [ ] **Step 2: Verify replay tests fail for the missing module**

Run: `uv run pytest tests/rdp_debug/test_replay_saved_observations.py -q`

Expected: collection fails with missing `replay_saved_observations`.

- [ ] **Step 3: Implement saved-step validation and RGB loading**

```python
def discover_steps(root: Path) -> list[Path]:
    steps = sorted(path for path in root.glob("step_*") if path.is_dir())
    if not steps:
        raise FileNotFoundError(f"{root}: no step_* directories")
    numbers = [int(path.name.removeprefix("step_")) for path in steps]
    for expected, actual in enumerate(numbers, numbers[0]):
        if actual != expected:
            raise ValueError(f"{root}: missing saved step {expected:06d}")
    return steps


def load_saved_observation(step: Path, cv2_module: Any) -> dict[str, np.ndarray]:
    state_path = step / "observation.state.npy"
    state = np.load(state_path, allow_pickle=False).astype(np.float32, copy=False)
    if state.shape != (20,) or not np.isfinite(state).all():
        raise ValueError(f"{state_path}: expected finite shape (20,), got {state.shape}")
    observation: dict[str, np.ndarray] = {"observation.state": state}
    for key in IMAGE_KEYS:
        path = step / f"{key}.jpg"
        bgr = cv2_module.imread(str(path), cv2_module.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(path)
        observation[key] = cv2_module.cvtColor(bgr, cv2_module.COLOR_BGR2RGB)
    return observation
```

- [ ] **Step 4: Write failing action-summary and overwrite tests**

```python
def test_replay_summary_exposes_five_step_boundary_jump() -> None:
    actions = np.zeros((10, 20), dtype=np.float64)
    actions[:5, 9] = 0.07
    actions[5:, 9] = 0.11
    report = summarize_actions(actions, [True, False, False, False, False] * 2, [3.0] * 10, 5)
    assert report["frames"] == 10
    assert report["replan_frames"] == [0, 5]
    assert report["gripper_boundary_jump_m"]["left"]["mean"] == pytest.approx(0.04)


def test_output_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    output.write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError):
        write_new_text(output, "replacement")
```

- [ ] **Step 5: Verify new replay tests fail before implementation**

Run: `uv run pytest tests/rdp_debug/test_replay_saved_observations.py -q`

Expected: failures name the missing summary/output behavior.

- [ ] **Step 6: Implement direct runtime loading, sequential replay, and multi-seed mode**

```python
def import_vitamin_runtime(vitamin_repo: Path) -> tuple[Any, Any, Any]:
    sys.path.insert(0, str(vitamin_repo))
    from deploy_pick_tube_rdp import PickTubeRDPRuntime
    from deploy_pick_tube_rdp import load_policy
    from reactive_diffusion_policy.deploy.tactile_encoder_torch import load_tactile_resnet18
    return PickTubeRDPRuntime, load_policy, load_tactile_resnet18


def seed_everything(seed: int, torch_module: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)
```

The CLI parses the deployment YAML itself, resolves checkpoint paths relative to the Vitamin repository, calls `load_policy`, `load_tactile_resnet18`, and `PickTubeRDPRuntime` directly, performs one `reset()` followed by ordered `predict()` calls, records wall-clock inference time and raw `(20,)` actions, and optionally repeats the first saved frame for every integer in `--seeds`. It never imports `bridge_client.py` or calls `run()` from the deployment module.

- [ ] **Step 7: Verify replay tests and contract-only CLI help**

Run: `uv run pytest tests/rdp_debug/test_replay_saved_observations.py -q`

Expected: all replay tests pass.

Run: `uv run python tools/rdp_debug/replay_saved_observations.py --help`

Expected: exit 0 without importing Torch, OpenCV, or Vitamin.

- [ ] **Step 8: Run trusted-checkpoint offline smoke replay in the Vitamin environment**

Run:

```bash
cd /home/typhon/RDP_vitamin
.venv/bin/python /home/typhon/RDP_vb3_robot_server/vb3_robot_server/tools/rdp_debug/replay_saved_observations.py \
  --vitamin-repo /home/typhon/RDP_vitamin \
  --config /home/typhon/RDP_vitamin/configs/deploy_pick_tube_rdp.yaml \
  --observations /home/typhon/FRS_Tact/outputs/frs_remote_observations/20260813_162558 \
  --device cuda:0 --limit 10
```

Expected: ten finite `(20,)` actions, slow-update frames 0 and 5, and no bridge connection message.

- [ ] **Step 9: Commit Task 3**

```bash
git add tools/rdp_debug/replay_saved_observations.py tests/rdp_debug/test_replay_saved_observations.py
git commit -m "feat: add hardware-isolated saved observation replay"
```

### Task 4: Policy-stage comparator

**Files:**
- Create: `tools/rdp_debug/compare_policy_stages.py`
- Create: `tests/rdp_debug/test_compare_policy_stages.py`

**Interfaces:**
- Consumes: Vitamin repository, deployment YAML, converted Zarr, episode index, start frame, and a 20-step horizon.
- Produces: `select_episode_window(...)`, `stage_metrics(truth, prediction)`, `classify_stage(stage_a_valid, at_error, ldp_error, threshold)`, and a model-heavy CLI that executes only after all paths and shapes pass validation.

- [ ] **Step 1: Write failing stage-window and classification tests**

```python
def test_episode_window_never_crosses_episode_boundary() -> None:
    ends = np.array([25, 50])
    window = select_episode_window(ends, episode_index=1, start_frame=3, horizon=20)
    assert window == slice(28, 48)
    with pytest.raises(ValueError, match="exceeds episode"):
        select_episode_window(ends, episode_index=0, start_frame=10, horizon=20)


def test_stage_metrics_preserve_left_right_slices() -> None:
    truth = np.zeros((20, 20), dtype=np.float64)
    prediction = truth.copy()
    prediction[:, 10] = 0.002
    report = stage_metrics(truth, prediction)
    assert report["left"]["position_rmse_mm"] == 0.0
    assert report["right"]["position_rmse_mm"] == pytest.approx(2.0 / np.sqrt(3.0))


def test_classification_assigns_first_failed_boundary() -> None:
    assert classify_stage(False, 0.0, 0.0, 0.001) == "source_or_conversion"
    assert classify_stage(True, 0.01, 0.0, 0.001) == "at_or_at_checkpoint"
    assert classify_stage(True, 0.0, 0.01, 0.001) == "ldp_or_observation_conditioning"
    assert classify_stage(True, 0.0, 0.0, 0.001) == "training_path_consistent"
```

- [ ] **Step 2: Verify comparator tests fail for the missing module**

Run: `uv run pytest tests/rdp_debug/test_compare_policy_stages.py -q`

Expected: collection fails with missing `compare_policy_stages`.

- [ ] **Step 3: Implement pure window, metric, and classification functions**

```python
def select_episode_window(episode_ends: np.ndarray, episode_index: int, start_frame: int, horizon: int) -> slice:
    start = 0 if episode_index == 0 else int(episode_ends[episode_index - 1])
    end = int(episode_ends[episode_index])
    absolute_start = start + start_frame
    if absolute_start < start or absolute_start + horizon > end:
        raise ValueError(f"episode {episode_index}: [{start_frame}:{start_frame + horizon}] exceeds episode length {end - start}")
    return slice(absolute_start, absolute_start + horizon)


def classify_stage(stage_a_valid: bool, at_error: float, ldp_error: float, threshold: float) -> str:
    if not stage_a_valid:
        return "source_or_conversion"
    if at_error > threshold:
        return "at_or_at_checkpoint"
    if ldp_error > threshold:
        return "ldp_or_observation_conditioning"
    return "training_path_consistent"
```

`stage_metrics` reports per-side xyz vector RMSE in millimetres, SO(3) geodesic mean/RMSE from rot6d columns, and gripper MAE/max in millimetres. It validates equal finite `[T,20]` arrays before computing any score.

- [ ] **Step 4: Write a failing forbidden-dependency source test**

```python
def test_comparator_source_is_hardware_isolated() -> None:
    source = Path("tools/rdp_debug/compare_policy_stages.py").read_text(encoding="utf-8")
    for forbidden in ("RobotBridgeClient", "websockets", "requests", "BimanualUmiEnv", "/dev/video"):
        assert forbidden not in source
```

- [ ] **Step 5: Implement the three model stages using the Vitamin checkpoint API**

```python
@torch.inference_mode()
def run_stages(policy: Any, sample: dict[str, torch.Tensor], horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    truth = sample["action"][:horizon]
    tactile = sample["extended_obs"]["tactile_embedding"][:horizon].unsqueeze(0)
    normalized_action = policy.normalizer["action"].normalize(truth.unsqueeze(0))
    encoded = policy.at.encoder(policy.at.preprocess(normalized_action / policy.at.act_scale))
    if policy.at.use_vq:
        latent, _, _ = policy.at.quant_state_with_vq(encoded)
    else:
        latent, _ = policy.at.quant_state_without_vq(encoded)
        latent = policy.at.postprocess_quant_state_without_vq(latent)
    temporal_cond = policy.at.get_temporal_cond({"tactile_embedding": tactile})
    reconstructed = policy.at.get_action_from_latent_with_temporal_cond(latent, temporal_cond)
    reconstructed = policy.normalizer["action"].unnormalize(reconstructed)[0]
    predicted_latent = policy.predict_action(
        sample["obs"],
        dataset_obs_temporal_downsample_ratio=1,
        return_latent_action=True,
    )["action"][:, 0]
    predicted = policy.predict_from_latent_action(
        predicted_latent,
        {"tactile_embedding": tactile},
        extended_obs_last_step=horizon,
        dataset_obs_temporal_downsample_ratio=1,
    )["action"][0, :horizon]
    return tuple(value.detach().float().cpu().numpy() for value in (truth, reconstructed, predicted))
```

Before calling this function, the CLI constructs `RealImageTactileDataset` from the checkpoint Hydra configuration with `load_to_memory=False`, obtains the requested absolute sequence through its replay buffer, creates CHW float camera tensors scaled by `1/255`, state/tactile tensors, and the exact 20-step truth/tactile chunk. The reconstruction path above intentionally uses the checkpoint's existing `encoder`, `quant_state_with_vq` or `quant_state_without_vq`, `postprocess_quant_state_without_vq`, `get_temporal_cond`, and `get_action_from_latent_with_temporal_cond` methods from `model/vae/model.py`.

- [ ] **Step 6: Verify pure tests and CLI import behavior**

Run: `uv run pytest tests/rdp_debug/test_compare_policy_stages.py -q`

Expected: all comparator tests pass without Torch, Zarr, or Vitamin imports during collection.

Run: `uv run python tools/rdp_debug/compare_policy_stages.py --help`

Expected: exit 0 in the server environment.

- [ ] **Step 7: Commit Task 4**

```bash
git add tools/rdp_debug/compare_policy_stages.py tests/rdp_debug/test_compare_policy_stages.py
git commit -m "feat: add AT and LDP stage responsibility comparator"
```

### Task 5: Chinese runbook, reference outputs, and full safety verification

**Files:**
- Create: `docs/rdp_pick_tube_debug_runbook_20260815.md`
- Create: `tests/rdp_debug/test_debug_tools_contract.py`
- Modify: `RDP_DEPLOYMENT.md`

**Interfaces:**
- Consumes: all four CLIs and the evidence recorded in the approved design spec.
- Produces: a self-contained Chinese responsibility guide, exact copyable commands, result-return checklist, and a cross-tool hardware-isolation test.

- [ ] **Step 1: Write failing whole-bundle contract tests**

```python
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
    forbidden = ("RobotBridgeClient", "websockets", "requests", "BimanualUmiEnv", "RobotWrapper", "/dev/video")
    for name in SCRIPT_NAMES:
        source = Path(f"tools/rdp_debug/{name}").read_text(encoding="utf-8")
        assert not any(token in source for token in forbidden)
```

- [ ] **Step 2: Verify bundle tests fail until all CLIs satisfy the contract**

Run: `uv run pytest tests/rdp_debug/test_debug_tools_contract.py -q`

Expected: at least one failure identifies an absent CLI, nonzero help command, or forbidden dependency.

- [ ] **Step 3: Write the Chinese runbook with evidence labels and responsibility matrix**

The runbook must use the following fixed section structure:

```markdown
# Pick-tube RDP 真机部署定责与离线排障手册（2026-08-15）

## 1. 当前结论与真机安全门
## 2. 仓库、模型和日志清单
## 3. 观测与动作数据契约
## 4. 部署数据流与时序
## 5. 两次真机运行对比
## 6. FRS 172 帧离线回放结果
## 7. 已解决问题
## 8. 已排除问题
## 9. 未解决问题与证据等级
## 10. 四阶段定责矩阵
## 11. 训练机执行步骤
## 12. 如何回传结果
## 13. 再次上真机前的检查清单
```

Every conclusion starts with `【已确认】`, `【推断】`, or `【未知】`. The document records both run counts/rates, the event-driven bridge fix, five-step gripper boundary statistics, the SO(3) logging artifact, Typhon HTTP 400 shutdown chain, FRS state/gripper OOD values, 32-seed asymmetry, training normalizer RMS, the absence of ordered training rows locally, and the non-atomic four-POST controller risk. It explicitly states that globally scaling action magnitude is unsafe because the right arm is already more active on the saved input.

- [ ] **Step 4: Add exact training-machine commands and output bundle names**

```bash
python tools/rdp_debug/audit_training_dataset.py \
  /absolute/path/to/pick_tube_01_04_rdp_zarr \
  --format zarr --start-windows 30 60 --max-lag 10 \
  --output debug_outputs/training_dataset_audit.json

python tools/rdp_debug/compare_policy_stages.py \
  --vitamin-repo /absolute/path/to/RDP_vitamin \
  --config /absolute/path/to/configs/deploy_pick_tube_rdp.yaml \
  --dataset /absolute/path/to/pick_tube_01_04_rdp_zarr \
  --episode 0 --start-frame 0 --horizon 20 \
  --output debug_outputs/policy_stages_ep000000_f000000.json
```

The return checklist requests the two JSON files, exact Git SHAs, SHA256 for LDP/AT/encoder artifacts, the checkpoint Hydra config, conversion command, and the first failing stack trace if a script exits nonzero. It forbids returning full proprietary image/video data unless needed later.

- [ ] **Step 5: Link the runbook from deployment documentation**

Add this sentence to `RDP_DEPLOYMENT.md`:

```markdown
For offline responsibility assignment, training-data checks, and safe replay commands, see [the 2026-08-15 pick-tube RDP debug runbook](docs/rdp_pick_tube_debug_runbook_20260815.md).
```

- [ ] **Step 6: Run focused, full, lint, and diff verification**

Run: `uv run pytest tests/rdp_debug -q`

Expected: all debug-tool tests pass.

Run: `uv run pytest -q`

Expected: the complete existing server test suite plus new tests passes with zero failures.

Run: `uv run ruff check tools/rdp_debug tests/rdp_debug`

Expected: `All checks passed!`.

Run: `git diff --check`

Expected: no output and exit 0.

- [ ] **Step 7: Verify the runbook against the approved design spec**

Run:

```bash
rg -n "【已确认】|【推断】|【未知】" docs/rdp_pick_tube_debug_runbook_20260815.md
rg -n "9\.34|19\.76|HTTP 400|32.*seed|2\.38|2\.54|五步|非原子" \
  docs/rdp_pick_tube_debug_runbook_20260815.md
```

Expected: all three evidence labels and every fixed reference value/topic are present.

- [ ] **Step 8: Commit Task 5**

```bash
git add docs/rdp_pick_tube_debug_runbook_20260815.md RDP_DEPLOYMENT.md \
  tests/rdp_debug/test_debug_tools_contract.py
git commit -m "docs: add RDP deployment responsibility runbook"
```

- [ ] **Step 9: Perform final branch review and fresh verification**

Generate a review package from the implementation branch base through `HEAD`, dispatch the requesting-code-review reviewer, fix all Critical and Important findings in one fix wave, then rerun:

```bash
uv run pytest -q
uv run ruff check tools/rdp_debug tests/rdp_debug
git diff --check
git status --short
```

Expected: zero test failures, clean Ruff and diff checks, and no unrelated change. The original checkout must still show only the user's pre-existing `M configs/server_config.py` outside the committed implementation history.

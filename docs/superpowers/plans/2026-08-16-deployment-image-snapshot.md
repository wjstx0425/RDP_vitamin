# Deployment Image Snapshot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a one-shot deployment mode that saves the exact pre-publication policy images as lossless, visually correct PNG files and exits before any start signal or action scheduling.

**Architecture:** Keep the feature in `bimanual_smolvla_online.py` so it reuses negotiated policy settings and the real `BimanualUmiEnv` transform. A focused snapshot writer validates HWC RGB `uint8` arrays, writes BGR-converted PNG files plus a manifest, while the CLI branch calls it after warmup observation construction and returns through existing context-manager cleanup.

**Tech Stack:** Python 3.11, Click, NumPy, OpenCV, pytest, uv

## Global Constraints

- Snapshot images must come from the exact `obs_dict` prepared by `get_real_umi_obs_dict(...)`.
- Snapshot mode must return before `publish_obs`, `wait_for_start`, action receipt, or command scheduling.
- Save lossless PNG with correct visible colors by converting in-memory RGB to OpenCV BGR only for writing.
- Reject `--save-image-snapshot` combined with `--dry-run` or `--save_obs` before hardware initialization.
- Preserve normal deployment and continuous `--save_obs` behavior.
- Do not modify or commit the user's `configs/server_config.py` changes or the untracked `VB-VLA/` tree.

---

### Task 1: Lossless deployment snapshot writer

**Files:**
- Modify: `deploy_scripts/bimanual_smolvla_online.py:1-410`
- Test: `deploy_scripts/bimanual_smolvla_online_test.py`

**Interfaces:**
- Consumes: `observation: dict`, `output_root: Path`, `policy_type: str`, `data_type: str`, and optional deterministic `now: datetime`.
- Produces: `save_deployment_image_snapshot(...) -> Path`, a unique directory of PNG files and `manifest.json`.

- [x] **Step 1: Write failing writer tests**

Add a test with RGB primary-color pixels and a fixed timestamp:

```python
def test_deployment_snapshot_saves_lossless_rgb_png_and_manifest(tmp_path):
    image = np.array(
        [[[255, 0, 0], [0, 255, 0], [0, 0, 255]]],
        dtype=np.uint8,
    )
    observation = {"observation.images.camera0": image}

    snapshot_dir = smolvla.save_deployment_image_snapshot(
        observation,
        tmp_path,
        policy_type="rdp",
        data_type="vitac",
        now=smolvla.datetime(2026, 8, 16, 12, 34, 56, 789000),
    )

    saved_bgr = smolvla.cv2.imread(
        str(snapshot_dir / "observation.images.camera0.png"),
        smolvla.cv2.IMREAD_COLOR,
    )
    saved_rgb = smolvla.cv2.cvtColor(saved_bgr, smolvla.cv2.COLOR_BGR2RGB)
    np.testing.assert_array_equal(saved_rgb, image)
    manifest = json.loads((snapshot_dir / "manifest.json").read_text())
    assert manifest["policy_type"] == "rdp"
    assert manifest["data_type"] == "vitac"
    assert manifest["images"]["observation.images.camera0"] == {
        "filename": "observation.images.camera0.png",
        "shape": [1, 3, 3],
        "dtype": "uint8",
        "min": 0,
        "max": 255,
    }
```

Add parameterized validation tests for float dtype, non-HWC shape, and missing image keys:

```python
@pytest.mark.parametrize(
    ("observation", "match"),
    [
        (
            {"observation.images.camera0": np.zeros((2, 3, 3), dtype=np.float32)},
            r"camera0.*uint8",
        ),
        (
            {"observation.images.camera0": np.zeros((2, 3), dtype=np.uint8)},
            r"camera0.*HWC",
        ),
        ({"observation.state": np.zeros(20)}, "no image keys"),
    ],
)
def test_deployment_snapshot_rejects_invalid_observations(
    tmp_path,
    observation,
    match,
):
    with pytest.raises(ValueError, match=match):
        smolvla.save_deployment_image_snapshot(
            observation,
            tmp_path,
            policy_type="rdp",
            data_type="vitac",
        )
```

- [x] **Step 2: Run writer tests and verify RED**

Run:

```bash
uv run --no-sync pytest -q deploy_scripts/bimanual_smolvla_online_test.py -k 'deployment_snapshot' -x
```

Expected: failure because `save_deployment_image_snapshot` does not exist.

- [x] **Step 3: Implement the snapshot writer**

Add this focused function before `ObsSaver`:

```python
def save_deployment_image_snapshot(
    observation: dict,
    output_root: Path,
    *,
    policy_type: str,
    data_type: str,
    now: datetime | None = None,
) -> Path:
    image_keys = sorted(
        key for key in observation if key.startswith("observation.images.")
    )
    if not image_keys:
        raise ValueError("deployment snapshot observation has no image keys")

    captured_at = now or datetime.now()
    snapshot_dir = Path(output_root) / (
        "deploy_snapshot_" + captured_at.strftime("%Y%m%d_%H%M%S_%f")
    )
    snapshot_dir.mkdir(parents=True, exist_ok=False)
    image_manifest = {}

    for key in image_keys:
        image = np.asarray(observation[key])
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"{key} must be HWC RGB, got {image.shape}")
        if image.dtype != np.uint8:
            raise ValueError(f"{key} must be uint8 RGB, got {image.dtype}")

        filename = f"{key}.png"
        output_path = snapshot_dir / filename
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(output_path), image_bgr):
            raise OSError(f"failed to write deployment snapshot: {output_path.resolve()}")
        image_manifest[key] = {
            "filename": filename,
            "shape": list(image.shape),
            "dtype": str(image.dtype),
            "min": int(image.min()),
            "max": int(image.max()),
        }

    manifest = {
        "captured_at": captured_at.isoformat(),
        "policy_type": policy_type,
        "data_type": data_type,
        "images": image_manifest,
    }
    with (snapshot_dir / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, sort_keys=True)
    return snapshot_dir.resolve()
```

- [x] **Step 4: Run writer tests and verify GREEN**

Run:

```bash
uv run --no-sync pytest -q deploy_scripts/bimanual_smolvla_online_test.py -k 'deployment_snapshot' -x
```

Expected: all snapshot-writer tests pass.

- [x] **Step 5: Commit the writer**

```bash
git add deploy_scripts/bimanual_smolvla_online.py deploy_scripts/bimanual_smolvla_online_test.py
git commit -m "feat: save lossless deployment image snapshots"
```

### Task 2: Pre-action snapshot CLI branch

**Files:**
- Modify: `deploy_scripts/bimanual_smolvla_online.py:783-1036`
- Modify: `real_world/bimanual_umi_env.py:323-340`
- Test: `deploy_scripts/bimanual_smolvla_online_test.py`
- Test: `tests/test_camera_startup_timeout.py`
- Test: `tests/test_smolvla_runtime_contract.py`

**Interfaces:**
- Consumes: Click flag `--save-image-snapshot` and the warmup `obs_dict` returned by `get_real_umi_obs_dict(...)`.
- Produces: the snapshot `Path` as the callback result, a printed absolute directory, and cleanup without policy publication or action execution.

- [x] **Step 1: Write failing CLI safety and flow tests**

Extend `call_main` with `"save_image_snapshot": False`. Add CLI validation tests asserting both forbidden combinations fail before `RobotClient` construction:

```python
@pytest.mark.parametrize("other_args", [["--dry-run"], ["--save_obs", "true"]])
def test_snapshot_rejects_incompatible_modes_before_client(monkeypatch, other_args):
    monkeypatch.setattr(
        smolvla,
        "RobotClient",
        lambda **_kwargs: pytest.fail("invalid snapshot mode constructed RobotClient"),
    )
    result = CliRunner().invoke(
        smolvla.main,
        ["--save-image-snapshot", *other_args],
    )
    assert result.exit_code == 2
    assert "cannot be combined" in result.output
```

Add a fake-client/fake-environment flow test that proves the snapshot branch is
before publication and start waiting:

```python
def test_snapshot_saves_warmup_obs_and_exits_before_publish_or_start(
    monkeypatch,
    tmp_path,
):
    events = []

    class SnapshotClient(FakeStartupClient):
        def publish_obs(self, _obs):
            pytest.fail("snapshot mode published an observation")

        def stop(self):
            super().stop()
            events.append("client_stop")

    class FakeSharedMemoryManager:
        def __enter__(self):
            return object()

        def __exit__(self, *_args):
            return False

    class SnapshotEnv:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            events.append("env_enter")
            return self

        def __exit__(self, *_args):
            events.append("env_exit")
            return False

        def get_obs(self):
            events.append("get_obs")
            return {"timestamp": np.array([100.0]), **zero_robot_obs()}

    client = SnapshotClient(
        config=valid_config(
            policy_type="rdp",
            data_type="vitac",
            steps_per_inference=1,
            action_horizon=1,
        )
    )
    expected_observation = {
        key: np.zeros((224, 224, 3), dtype=np.uint8)
        for key in (
            "observation.images.camera0",
            "observation.images.camera1",
            "observation.images.tactile_left_0",
            "observation.images.tactile_right_0",
            "observation.images.tactile_left_1",
            "observation.images.tactile_right_1",
        )
    }
    captured = {}
    expected_path = tmp_path / "snapshot"
    fake_env_module = types.ModuleType("real_world.bimanual_umi_env")
    fake_env_module.BimanualUmiEnv = SnapshotEnv

    monkeypatch.setattr(smolvla, "load_token_list", lambda _: ["token"])
    monkeypatch.setattr(smolvla, "RobotClient", lambda **_: client)
    monkeypatch.setattr(smolvla, "SharedMemoryManager", FakeSharedMemoryManager)
    monkeypatch.setattr(
        smolvla,
        "wait_for_smolvla_config",
        lambda _client, _deadline: client.config,
    )
    monkeypatch.setattr(
        smolvla,
        "get_real_umi_obs_dict",
        lambda **_kwargs: expected_observation,
    )

    def save_snapshot(observation, *_args, **_kwargs):
        captured["observation"] = observation
        events.append("save_snapshot")
        return expected_path

    monkeypatch.setattr(smolvla, "save_deployment_image_snapshot", save_snapshot)
    monkeypatch.setattr(
        smolvla,
        "wait_for_start",
        lambda *_args, **_kwargs: pytest.fail("snapshot mode waited for start"),
    )
    monkeypatch.setattr(smolvla.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(smolvla.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(smolvla.cv2, "setNumThreads", lambda _count: None)
    monkeypatch.setitem(sys.modules, "real_world.bimanual_umi_env", fake_env_module)

    result = call_main(save_image_snapshot=True)

    assert result == expected_path
    assert captured["observation"] is expected_observation
    assert events == [
        "env_enter",
        "get_obs",
        "get_obs",
        "save_snapshot",
        "env_exit",
        "client_stop",
    ]
    assert client.join_timeouts == [1.0]
```

- [x] **Step 2: Run CLI snapshot tests and verify RED**

Run:

```bash
uv run --no-sync pytest -q deploy_scripts/bimanual_smolvla_online_test.py -k 'snapshot' -x
```

Expected: failure because the Click option and callback argument do not exist.

- [x] **Step 3: Add option validation and the pre-action return branch**

Add the Click option and callback argument:

```python
@click.option(
    "--save-image-snapshot",
    is_flag=True,
    help="Save one lossless policy-input image set before start, then exit",
)
```

Before `load_token_list(...)`, reject invalid combinations:

```python
if save_image_snapshot and (dry_run or save_obs):
    raise click.UsageError(
        "--save-image-snapshot cannot be combined with --dry-run or --save_obs"
    )
```

Wrap snapshot-mode shared-memory/camera lifetime so the client is stopped only
after those resources have exited, including environment construction, warmup,
observation conversion, and writer exceptions:

```python
@contextmanager
def _shared_memory_manager_with_client_cleanup(client=None):
    try:
        with SharedMemoryManager() as shm_manager:
            yield shm_manager
    finally:
        if client is not None:
            _stop_robot_client(client)


with _shared_memory_manager_with_client_cleanup(
    client if save_image_snapshot else None
) as shm_manager:
    ...
```

If calibration loading fails before that context is entered, stop the snapshot
client and re-raise. Immediately after warmup `obs_dict` construction and before
`publish_obs`, add:

```python
if save_image_snapshot:
    snapshot_dir = save_deployment_image_snapshot(
        obs_dict,
        Path(ROOT_DIR) / "eval_obs_data",
        policy_type=policy_type,
        data_type=data_type,
    )
    print(f"[ImageSnapshot] Saved deployment images to: {snapshot_dir}")
    return snapshot_dir
```

Add fault-injection tests for calibration loading, environment import, OpenCV
setup, child-process startup, observation conversion, and snapshot writing.
Assert that clients, environments, and shared-memory managers are cleaned up,
and that failure paths never publish or wait for start. Extend
`BimanualUmiEnv.start()` startup rollback to cover exceptions from
`camera.start()` and `controller.start()`, not only readiness waiting.

- [x] **Step 4: Run CLI snapshot tests and verify GREEN**

Run:

```bash
uv run --no-sync pytest -q deploy_scripts/bimanual_smolvla_online_test.py -k 'snapshot' -x
```

Expected: all snapshot tests pass.

- [x] **Step 5: Run focused and full regression suites**

Run:

```bash
uv run --no-sync pytest -q \
  deploy_scripts/bimanual_smolvla_online_test.py \
  tests/test_camera_frame_preprocessing.py
uv run --no-sync pytest -q
git diff --check
```

Expected: all tests pass and `git diff --check` emits no output.

- [x] **Step 6: Commit CLI integration and plan**

```bash
git add \
  docs/superpowers/plans/2026-08-16-deployment-image-snapshot.md \
  deploy_scripts/bimanual_smolvla_online.py \
  deploy_scripts/bimanual_smolvla_online_test.py
git commit -m "feat: add pre-action image snapshot mode"
```

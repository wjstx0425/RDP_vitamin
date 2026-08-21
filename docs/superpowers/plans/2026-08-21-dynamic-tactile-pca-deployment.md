# Dynamic Tactile PCA Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make deployment preflight and runtime behavior explicitly support matching 16D, 30D, and 60D PCA/LDP/AT artifact sets.

**Architecture:** Keep PCA output dimension derived solely from the selected artifact. Add small, independently testable checkpoint metadata helpers before workspace construction, then parameterize the existing CPU runtime test across all supported dimensions.

**Tech Stack:** Python 3.12, PyTorch, NumPy, OmegaConf, pytest.

## Global Constraints

- Scope is deployment inference only; do not modify training entrypoints.
- The YAML continues to specify LDP, AT, and PCA paths explicitly.
- Do not add a separate tactile dimension setting or automatic path construction.
- Accept only artifacts whose PCA, LDP observation, LDP extended observation, AT observation, and AT extended observation dimensions are identical.
- Do not pad, truncate, coerce, or reshape mismatched tactile embeddings.
- Preserve the fixed tactile encoder output contract `[4, 512]`.
- The truncated local 60D LDP remains an external artifact blocker and must not be treated as a successful load.

---

### Task 1: Validate PCA values and all supported projection widths

**Files:**
- Modify: `reactive_diffusion_policy/model/tactile_pca.py:54-79`
- Test: `tests/test_tactile_pca.py`

**Interfaces:**
- Consumes: PCA `means: np.ndarray` and `components: np.ndarray` supplied to `BimanualTactilePCA`.
- Produces: `BimanualTactilePCA.output_dim: int` for 2x8, 2x15, and 2x30 artifacts; construction rejects non-finite PCA arrays.

- [ ] **Step 1: Ensure the deployment environment has pytest**

Run:

```bash
uv pip install --python .venv/bin/python pytest
```

Expected: pytest is installed into the existing deployment virtual environment without changing repository dependency files.

- [ ] **Step 2: Write failing finite-value and projection-matrix tests**

Update the dynamic projection test to use pytest parameterization and add finite-value rejection:

```python
import pytest


@pytest.mark.parametrize("components_per_arm", [8, 15, 30])
def test_projection_dimension_is_inferred_from_components(
    components_per_arm: int,
) -> None:
    means = np.zeros((2, 1024), dtype=np.float32)
    components = np.zeros((2, components_per_arm, 1024), dtype=np.float32)
    model = BimanualTactilePCA(means, components)

    assert model.components_per_arm == components_per_arm
    assert model.output_dim == components_per_arm * 2
    assert model(torch.from_numpy(sensor_values())).shape == (model.output_dim,)
    assert model.transform_numpy(sensor_values()).shape == (model.output_dim,)
    flat_batch = np.stack([sensor_values().reshape(-1)] * 2)
    assert model(torch.from_numpy(flat_batch)).shape == (2, model.output_dim)
    assert model.transform_numpy(flat_batch).shape == (2, model.output_dim)


@pytest.mark.parametrize("field", ["means", "components"])
def test_pca_rejects_non_finite_values(field: str) -> None:
    means = np.zeros((2, 1024), dtype=np.float32)
    components = np.zeros((2, 8, 1024), dtype=np.float32)
    if field == "means":
        means[0, 0] = np.nan
    else:
        components[0, 0, 0] = np.inf

    with pytest.raises(ValueError, match=f"PCA {field} must contain only finite values"):
        BimanualTactilePCA(means, components)
```

- [ ] **Step 3: Run the tests to verify RED**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_tactile_pca.py
```

Expected: the non-finite-value cases fail because the constructor currently accepts NaN and infinity.

- [ ] **Step 4: Add minimal finite-value checks**

After the existing shape checks in `BimanualTactilePCA.__init__`, add:

```python
if not np.isfinite(means).all():
    raise ValueError("PCA means must contain only finite values")
if not np.isfinite(components).all():
    raise ValueError("PCA components must contain only finite values")
```

- [ ] **Step 5: Run the focused tests to verify GREEN**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_tactile_pca.py
```

Expected: all tactile PCA tests pass, including 16D, 30D, and 60D projection widths.

- [ ] **Step 6: Commit Task 1**

```bash
git add reactive_diffusion_policy/model/tactile_pca.py tests/test_tactile_pca.py
git commit -m "test: validate dynamic tactile PCA artifacts"
```

---

### Task 2: Preflight PCA, LDP, and AT dimensions before policy construction

**Files:**
- Modify: `deploy_pick_tube_rdp.py:52-94`
- Test: `tests/test_pick_tube_rdp_deploy.py`

**Interfaces:**
- Consumes: `Path` objects for LDP and AT checkpoints, each payload's `cfg`, and the selected PCA `output_dim`.
- Produces: `_load_checkpoint_payload(path: Path, role: str) -> dict`, `_tactile_dim(cfg: Any, role: str, field: str) -> int`, and `validate_tactile_dimensions(...) -> None`; `load_policy` invokes them before Hydra workspace construction.

- [ ] **Step 1: Write failing metadata-validation tests**

Add these helpers and tests to `tests/test_pick_tube_rdp_deploy.py`:

```python
import pytest


def tactile_cfg(obs_dim: int, extended_dim: int | None = None):
    return OmegaConf.create(
        {
            "shape_meta": {
                "obs": {"tactile_embedding": {"shape": [obs_dim]}},
                "extended_obs": {
                    "tactile_embedding": {
                        "shape": [obs_dim if extended_dim is None else extended_dim]
                    }
                },
            }
        }
    )


@pytest.mark.parametrize("tactile_dim", [16, 30, 60])
def test_validate_tactile_dimensions_accepts_matching_artifacts(
    tactile_dim: int,
) -> None:
    deploy.validate_tactile_dimensions(
        tactile_dim,
        tactile_cfg(tactile_dim),
        tactile_cfg(tactile_dim),
        Path("ldp.ckpt"),
        Path("at.ckpt"),
    )


def test_validate_tactile_dimensions_reports_every_source() -> None:
    with pytest.raises(ValueError) as error:
        deploy.validate_tactile_dimensions(
            16,
            tactile_cfg(16, extended_dim=30),
            tactile_cfg(60),
            Path("ldp.ckpt"),
            Path("at.ckpt"),
        )

    message = str(error.value)
    assert "PCA output=16D" in message
    assert "LDP obs (ldp.ckpt)=16D" in message
    assert "LDP extended_obs (ldp.ckpt)=30D" in message
    assert "AT obs (at.ckpt)=60D" in message
    assert "AT extended_obs (at.ckpt)=60D" in message


def test_tactile_dim_reports_missing_checkpoint_field() -> None:
    with pytest.raises(ValueError, match="LDP checkpoint is missing"):
        deploy._tactile_dim(OmegaConf.create({}), "LDP", "obs")
```

- [ ] **Step 2: Parameterize the fake runtime across 16D, 30D, and 60D**

Replace `FakePolicy` with a dimension-aware fake:

```python
class FakePolicy:
    def __init__(self, components_per_arm: int) -> None:
        self.components_per_arm = components_per_arm
        self.slow_calls = 0
        self.fast_history_lengths = []
        self.slow_observation_states = []

    def predict_action(self, obs_dict, **kwargs):
        assert tuple(obs_dict["camera1"].shape) == (1, 2, 3, 224, 224)
        assert tuple(obs_dict["camera2"].shape) == (1, 2, 3, 224, 224)
        assert tuple(obs_dict["observation_state"].shape) == (1, 2, 20)
        tactile_dim = self.components_per_arm * 2
        assert tuple(obs_dict["tactile_embedding"].shape) == (1, 2, tactile_dim)
        torch.testing.assert_close(
            obs_dict["tactile_embedding"][0, :, : self.components_per_arm],
            torch.full((2, self.components_per_arm), 1.0 / 255.0),
        )
        torch.testing.assert_close(
            obs_dict["tactile_embedding"][0, :, self.components_per_arm :],
            torch.full((2, self.components_per_arm), 3.0 / 255.0),
        )
        assert kwargs["return_latent_action"] is True
        self.slow_observation_states.append(
            obs_dict["observation_state"][0, :, 0].detach().cpu().tolist()
        )
        self.slow_calls += 1
        return {"action": torch.zeros((1, 29, 128))}

    def predict_from_latent_action(
        self,
        latent_action,
        extended_obs,
        extended_obs_last_step,
        dataset_obs_temporal_downsample_ratio,
    ):
        assert tuple(latent_action.shape) == (1, 128)
        assert dataset_obs_temporal_downsample_ratio == 2
        history_length = extended_obs["tactile_embedding"].shape[1]
        assert extended_obs["tactile_embedding"].shape[2] == self.components_per_arm * 2
        assert extended_obs_last_step == history_length
        self.fast_history_lengths.append(history_length)
        return {"action": torch.full((1, history_length, 20), float(history_length))}
```

Replace the runtime test with:

```python
@pytest.mark.parametrize("components_per_arm", [8, 15, 30])
def test_runtime_updates_slow_plan_every_five_steps_and_decodes_every_step(
    components_per_arm: int,
) -> None:
    policy = FakePolicy(components_per_arm)
    encoder = FakeTactileEncoder()
    means = np.zeros((2, 1024), dtype=np.float32)
    components = np.zeros((2, components_per_arm, 1024), dtype=np.float32)
    components[:, np.arange(components_per_arm), np.arange(components_per_arm)] = 1.0
    tactile_pca = deploy.BimanualTactilePCA(means, components)
    runtime = deploy.PickTubeRDPRuntime(
        policy,
        encoder,
        torch.device("cpu"),
        tactile_pca,
        slow_update_interval=5,
        dataset_obs_temporal_downsample_ratio=2,
        n_obs_steps=2,
    )

    slow_updates = []
    actions = []
    for step in range(7):
        action, slow_update = runtime.predict(observation(step))
        actions.append(action)
        slow_updates.append(slow_update)

    assert slow_updates == [True, False, False, False, False, True, False]
    assert policy.slow_calls == 2
    assert policy.slow_observation_states == [[0.0, 0.0], [3.0, 5.0]]
    assert policy.fast_history_lengths == [4, 5, 6, 7, 8, 4, 5]
    assert all(action.shape == (1, 20) and action.dtype == np.float32 for action in actions)
    np.testing.assert_allclose(encoder.last_means, np.arange(1, 5) / 255.0)
```

- [ ] **Step 3: Run deployment tests to verify RED**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_pick_tube_rdp_deploy.py
```

Expected: metadata-validation tests fail because the validation helpers do not exist; the runtime parameterization documents 16D/30D/60D behavior.

- [ ] **Step 4: Implement checkpoint loading and dimension extraction**

Add before `load_policy`:

```python
def _load_checkpoint_payload(path: Path, role: str) -> dict[str, Any]:
    try:
        with path.open("rb") as file:
            return torch.load(
                file,
                pickle_module=dill,
                weights_only=False,
                map_location="cpu",
            )
    except Exception as exc:
        raise RuntimeError(f"Failed to inspect {role} checkpoint {path}: {exc}") from exc


def _tactile_dim(cfg: Any, role: str, field: str) -> int:
    key = f"shape_meta.{field}.tactile_embedding.shape"
    shape = OmegaConf.select(cfg, key)
    if shape is None or len(shape) != 1:
        raise ValueError(f"{role} checkpoint is missing a valid {key}")
    dimension = int(shape[0])
    if dimension < 1:
        raise ValueError(f"{role} checkpoint has invalid {key}: {list(shape)}")
    return dimension


def validate_tactile_dimensions(
    pca_dim: int,
    ldp_cfg: Any,
    at_cfg: Any,
    ldp_checkpoint: Path,
    at_checkpoint: Path,
) -> None:
    dimensions = {
        "PCA output": pca_dim,
        f"LDP obs ({ldp_checkpoint})": _tactile_dim(ldp_cfg, "LDP", "obs"),
        f"LDP extended_obs ({ldp_checkpoint})": _tactile_dim(
            ldp_cfg, "LDP", "extended_obs"
        ),
        f"AT obs ({at_checkpoint})": _tactile_dim(at_cfg, "AT", "obs"),
        f"AT extended_obs ({at_checkpoint})": _tactile_dim(
            at_cfg, "AT", "extended_obs"
        ),
    }
    if len(set(dimensions.values())) != 1:
        details = ", ".join(f"{source}={dimension}D" for source, dimension in dimensions.items())
        raise ValueError(
            f"Tactile embedding dimension mismatch: {details}. "
            "Use matching PCA, AT, and LDP artifacts."
        )
```

- [ ] **Step 5: Use preflight in `load_policy`**

Replace the inline LDP load and single-dimension check with:

```python
payload = _load_checkpoint_payload(ldp_checkpoint, "LDP")
at_payload = _load_checkpoint_payload(at_checkpoint, "AT")
cfg = copy.deepcopy(payload["cfg"])
at_cfg = at_payload["cfg"]
OmegaConf.set_struct(cfg, False)
validate_tactile_dimensions(
    tactile_embedding_dim,
    cfg,
    at_cfg,
    ldp_checkpoint,
    at_checkpoint,
)
```

The next existing statement remains `cfg.at_load_dir = str(at_checkpoint)`;
workspace construction and policy setup continue unchanged after preflight.

- [ ] **Step 6: Run the focused tests to verify GREEN**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_pick_tube_rdp_deploy.py tests/test_tactile_pca.py
```

Expected: all tests pass for the 16D, 30D, and 60D metadata and fake-runtime matrix.

- [ ] **Step 7: Run deployment-related regression tests**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_pick_tube_rdp_deploy.py tests/test_tactile_pca.py tests/test_pick_tube_training_data.py
```

Expected: all selected tests pass with zero failures.

- [ ] **Step 8: Run real artifact smoke checks**

Run:

```bash
.venv/bin/python -c "from pathlib import Path; import dill, torch; from reactive_diffusion_policy.model.tactile_pca import BimanualTactilePCA; base=Path('data/weights/wjstx_rdp/tactile/rdp'); pcas=[base/'tactile_pca_armwise_01_02_03_04_05_06_2x8.npz',base/'tactile_pca_2x15.npz',base/'tactile_pca_armwise_01_02_03_04_05_06_2x30.npz']; print('PCA', [BimanualTactilePCA.from_npz(path).output_dim for path in pcas]); print('AT', [int(torch.load((base/str(dim)/'at/latest.ckpt').open('rb'),pickle_module=dill,weights_only=False,map_location='cpu')['cfg'].shape_meta.obs.tactile_embedding.shape[0]) for dim in (16,30,60)]); print('LDP16', int(torch.load((base/'16/ldp/latest.ckpt').open('rb'),pickle_module=dill,weights_only=False,map_location='cpu')['cfg'].shape_meta.obs.tactile_embedding.shape[0]))"
```

Expected: `PCA [16, 30, 60]`, `AT [16, 30, 60]`, and `LDP16 16`.

Then run:

```bash
.venv/bin/python -c "from pathlib import Path; import deploy_pick_tube_rdp as deploy; deploy._load_checkpoint_payload(Path('data/weights/wjstx_rdp/tactile/rdp/60/ldp/latest.ckpt'),'LDP')"
```

Expected: non-zero exit with `Failed to inspect LDP checkpoint` and the
underlying `failed finding central directory` cause, proving the current 60D
artifact is reported as corrupt rather than as a dimension mismatch.

- [ ] **Step 9: Commit Task 2**

```bash
git add deploy_pick_tube_rdp.py tests/test_pick_tube_rdp_deploy.py
git commit -m "feat: preflight tactile deployment artifacts"
```

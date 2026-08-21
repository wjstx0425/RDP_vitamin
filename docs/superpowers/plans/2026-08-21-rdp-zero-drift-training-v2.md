# RDP Zero-Drift Training v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved 20D v2 pick-tube training pipeline so physical no-op maps to network zero, invalid terminal actions cannot contaminate AT/LDP targets, idle-arm reconstruction is explicitly optimized, and v2 artifacts cannot be silently mismatched.

**Architecture:** Add a pure pick-tube action-contract module and emit auditable v2 arrays during conversion. Extend the existing dataset/normalizer and VAE loss without changing the external 20D layout or temporal indexing. Make LDP targets deterministic behind a frozen AT boundary, then add episode validation and strict artifact identity checks to training and deployment.

**Tech Stack:** Python 3, NumPy, PyTorch, Zarr, Hydra/OmegaConf, Accelerate, diffusers, pytest.

## Global Constraints

- Preserve the 20D contiguous layout: left `[0:10]`, right `[10:20]`, with each arm `[delta_position(3), rotation_6d(6), gripper(1)]`.
- Preserve horizon `32`, observation positions `[1, 3]`, action start position `3`, raw observation count `4`, and temporal downsample ratio `2`.
- Do not overwrite source parquet actions; store them as `action_raw` and store canonical v2 labels separately as `action`.
- Physical zero translation and identity rotation must normalize to exact zero under `zero_centered_v2`.
- New v2 AT/LDP/normalizer artifacts must hard-fail on version or identity mismatch; old artifacts remain available only through explicit `legacy-compatible` loading.
- AT validation and all LDP latent targets use `posterior.mode()`; AT training continues to use `posterior.sample()`.
- Use episode-level validation with ratio `0.1`, fixed seed, and no window-level leakage.
- Release thresholds remain: 29-step idle drift below `1 mm / 0.5 deg`, idle per-step p95 below `0.05 mm / 0.03 deg`, active degradation below `5%`, micro-motion recall at least `95%`, and 60-second live drift below `2 mm / 1 deg`.
- Follow test-driven development: add each regression test and observe the expected failure before editing production code.

---

### Task 1: V2 action contract, converter output, and idle masks

**Files:**
- Create: `reactive_diffusion_policy/common/pick_tube_action_contract.py`
- Modify: `convert_pick_tube_lerobot_to_rdp_zarr.py`
- Modify: `reactive_diffusion_policy/common/sampler.py`
- Modify: `reactive_diffusion_policy/dataset/real_image_tactile_dataset.py`
- Test: `tests/test_pick_tube_action_contract_v2.py`
- Test: `tests/test_pick_tube_training_data.py`

**Interfaces:**
- Produces `canonical_noop_from_state(state: np.ndarray) -> np.ndarray`.
- Produces `canonicalize_episode_actions(state, action) -> CanonicalEpisodeActions` with `action_raw`, `action`, `action_valid`, and `idle_arm_mask` arrays.
- Produces constants `ACTION_REPRESENTATION_VERSION = 2`, `ACTION_CONTRACT = "bimanual_relative_pose20d_v2"`, and `TERMINAL_ACTION_POLICY = "canonical_relative_noop_v2"`.
- Extends Zarr data arrays with `action_raw`, `action_valid`, and `idle_arm_mask`, and writes a JSON-serializable v2 manifest into metadata attributes.
- Extends dataset samples with top-level `valid_mask: Tensor[T]` and `idle_arm_mask: Tensor[T,2]` when the arrays exist.

- [ ] **Step 1: Write failing action-contract tests**

Add tests that construct a short synthetic two-arm episode and assert:

```python
result = canonicalize_episode_actions(state, action)
np.testing.assert_array_equal(result.action_raw, action)
assert not result.action_valid[-1]
np.testing.assert_allclose(result.action[-1, :3], 0.0)
np.testing.assert_allclose(result.action[-1, 3:9], [1, 0, 0, 0, 1, 0])
assert result.action[-1, 9] == state[-1, 6]
np.testing.assert_allclose(result.action[-1, 10:13], 0.0)
np.testing.assert_allclose(result.action[-1, 13:19], [1, 0, 0, 0, 1, 0])
assert result.action[-1, 19] == state[-1, 13]
```

Also assert that a degenerate nonterminal rotation raises `ValueError`, and that the entry/exit idle hysteresis uses 8/2 frames and the accepted `0.5 mm, 0.25 deg, 0.5 mm` low thresholds plus `0.8 mm, 0.4 deg, 0.8 mm` high thresholds.

- [ ] **Step 2: Run the focused tests and confirm RED**

Run:

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q tests/test_pick_tube_action_contract_v2.py
```

Expected: collection fails because `pick_tube_action_contract` does not exist.

- [ ] **Step 3: Implement the pure action contract**

Use a frozen dataclass:

```python
@dataclass(frozen=True)
class CanonicalEpisodeActions:
    action_raw: np.ndarray
    action: np.ndarray
    action_valid: np.ndarray
    idle_arm_mask: np.ndarray
```

Convert each rotation-6D to a matrix by normalized Gram-Schmidt for angle calculation, reject basis norms/cross-product norms below `1e-6`, generate idle masks from the original labels, canonicalize idle arm labels, and replace only the source terminal target with a canonical no-op. Never mutate caller-owned arrays.

- [ ] **Step 4: Extend the converter and manifest**

Create Zarr arrays with exact shapes/dtypes:

```text
action_raw     [N,20] float32
action         [N,20] float32
action_valid   [N]    bool
idle_arm_mask  [N,2]  bool
```

Append the canonicalization results episode-by-episode. Compute SHA256 with a streaming helper over the PCA file and a stable JSON episode manifest. Store action version, terminal policy, thresholds, repair counts, array schema, PCA hash, dataset digest, and git commit under `root["meta"].attrs["v2_manifest_json"]`.

- [ ] **Step 5: Make padding/sample metadata available to training**

Load optional v2 arrays in `RealImageTactileDataset`. Return sample-aligned boolean `valid_mask` and `[T,2] idle_arm_mask`. For legacy buffers, return all-true validity and all-false idle masks only when an explicit `allow_legacy_action_contract=True`; v2 task configs must leave this false.

When `SequenceSampler` pads an action sequence, fill the action suffix with a canonical no-op using the final gripper widths instead of repeating a nonzero relative action. Preserve existing repeated-edge behavior for observations and tactile features.

- [ ] **Step 6: Run Task 1 tests and the temporal regression suite**

Run:

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q tests/test_pick_tube_action_contract_v2.py tests/test_pick_tube_training_data.py
```

Expected: all tests pass and existing `[1,3]` observation selection assertions remain unchanged.

- [ ] **Step 7: Commit Task 1**

```bash
git add reactive_diffusion_policy/common/pick_tube_action_contract.py convert_pick_tube_lerobot_to_rdp_zarr.py reactive_diffusion_policy/common/sampler.py reactive_diffusion_policy/dataset/real_image_tactile_dataset.py tests/test_pick_tube_action_contract_v2.py tests/test_pick_tube_training_data.py
git commit -m "feat: add pick-tube v2 action contract"
```

---

### Task 2: Zero-preserving normalizer and physical AT loss

**Files:**
- Modify: `reactive_diffusion_policy/common/normalize_util.py`
- Create: `reactive_diffusion_policy/model/vae/physical_action_loss.py`
- Modify: `reactive_diffusion_policy/model/vae/model.py`
- Modify: `reactive_diffusion_policy/dataset/real_image_tactile_dataset.py`
- Modify: `reactive_diffusion_policy/config/task/pick_tube_image_tactile_emb_at_30fps.yaml`
- Modify: `reactive_diffusion_policy/config/task/pick_tube_image_tactile_emb_ldp_30fps.yaml`
- Test: `tests/test_pick_tube_action_normalizer_v2.py`
- Test: `tests/test_pick_tube_at_physical_loss.py`

**Interfaces:**
- Extends `get_action_normalizer(..., version: str = "legacy_v1")` with `zero_centered_v2`.
- Adds dataset constructor option `action_normalizer_version: str = "legacy_v1"` and configures pick-tube v2 tasks with `zero_centered_v2`.
- Produces `project_rotation_6d(rotation_6d: Tensor) -> tuple[Tensor, Tensor]`, returning a valid matrix and degeneracy penalty.
- Produces `compute_bimanual_physical_loss(target, prediction, valid_mask, idle_arm_mask, weights) -> dict[str, Tensor]`.

- [ ] **Step 1: Write failing normalizer tests**

Assert for both arm slices that physical zero translation and identity rotation normalize to exact zero, round-trip maximum error is below `1e-7`, grippers remain range-normalized, and per-axis translation scale equals `1 / max(Q99.5(abs(x)), epsilon)` with zero offset.

- [ ] **Step 2: Write failing physical-loss tests**

Test identity projection, nonorthogonal finite projection, determinant/orthogonality bounds, masking of invalid/padded timesteps, independent left/right averaging, geodesic rotation response, idle loss response, and finite gradients for nearly collinear inputs.

- [ ] **Step 3: Run tests and confirm RED**

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q tests/test_pick_tube_action_normalizer_v2.py tests/test_pick_tube_at_physical_loss.py
```

Expected: failures identify missing v2 normalizer and physical loss module.

- [ ] **Step 4: Implement zero-centered v2 normalization**

For position slices `[0:3]` and `[10:13]`, create manual normalizers with zero offset and robust per-axis symmetric scale. For rotation slices, create manual normalizers with unit scale and offset equal to negative identity-6D. Keep gripper range normalization. Retain `legacy_v1` behavior for old configurations.

- [ ] **Step 5: Implement differentiable physical losses**

Use stable normalized Gram-Schmidt and clamp geodesic cosine to `[-1 + 1e-7, 1 - 1e-7]`. Compute Huber losses at physical scales `1 mm`, `1 degree`, and `5 mm`; compute idle error at `0.1 mm` and `0.05 degree`; add degeneracy loss and optional raw residual auxiliary loss capped by configuration at `0.1`.

- [ ] **Step 6: Integrate physical loss into VAE**

Keep posterior sampling in AT training. Reshape decoder output to `[B,T,20]`, differentiably unnormalize target/prediction, read `valid_mask`/`idle_arm_mask`, and return scalar plus named metrics:

```text
position_loss rotation_loss gripper_loss idle_loss
degenerate_loss rot6_aux_loss kl_loss rep_loss
```

Preserve legacy scalar L1 when `action_loss_version=legacy_v1`. Configure pick-tube v2 AT with `physical_v2`, `idle_weight=1.0`, and the accepted physical scales.

- [ ] **Step 7: Run Task 2 tests and AT workspace regressions**

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q tests/test_pick_tube_action_normalizer_v2.py tests/test_pick_tube_at_physical_loss.py tests/test_workspace_resume.py
```

Expected: all pass.

- [ ] **Step 8: Commit Task 2**

```bash
git add reactive_diffusion_policy/common/normalize_util.py reactive_diffusion_policy/model/vae/physical_action_loss.py reactive_diffusion_policy/model/vae/model.py reactive_diffusion_policy/dataset/real_image_tactile_dataset.py reactive_diffusion_policy/config/task/pick_tube_image_tactile_emb_at_30fps.yaml reactive_diffusion_policy/config/task/pick_tube_image_tactile_emb_ldp_30fps.yaml tests/test_pick_tube_action_normalizer_v2.py tests/test_pick_tube_at_physical_loss.py
git commit -m "feat: train AT with zero-preserving physical loss"
```

---

### Task 3: Deterministic LDP targets, frozen AT, and correct optimizer steps

**Files:**
- Modify: `reactive_diffusion_policy/model/vae/model.py`
- Modify: `reactive_diffusion_policy/dataset/real_image_tactile_latent_diffusion_dataset.py`
- Modify: `reactive_diffusion_policy/policy/latent_diffusion_unet_image_policy.py`
- Modify: `reactive_diffusion_policy/workspace/train_at_workspace.py`
- Modify: `reactive_diffusion_policy/workspace/train_diffusion_unet_image_workspace.py`
- Test: `tests/test_pick_tube_ldp_training_v2.py`
- Test: `tests/test_gradient_accumulation.py`

**Interfaces:**
- Extends `VAE.quant_state_without_vq(state, *, sample: bool = True)`; `sample=False` uses `posterior.mode()`.
- Adds `VAE.parameters()` and `VAE.requires_grad_(bool)` helpers over encoder, decoder, quant/post-quant or VQ modules.
- Adds `LatentDiffusionUnetImagePolicy.freeze_action_tokenizer()` and `encode_latent_target(nactions)`.
- Adds a small pure helper `should_optimizer_step(batch_idx, num_batches, accumulate_every) -> bool` shared by AT/LDP workspaces.

- [ ] **Step 1: Write failing frozen/deterministic tests**

Assert identical input produces bitwise/numerically identical `sample=False` latent targets, LDP construction freezes every AT parameter, a backward pass leaves all AT gradients `None`, AT remains in eval mode after `policy.train()`, and only LDP parameters appear in the optimizer.

- [ ] **Step 2: Write failing accumulation tests**

For accumulation 2 and 3, assert optimizer/scheduler/EMA counts equal `ceil(num_batches / accumulation)` and include the final partial group. Assert accumulation 1 preserves current counts.

- [ ] **Step 3: Run focused tests and confirm RED**

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q tests/test_pick_tube_ldp_training_v2.py tests/test_gradient_accumulation.py
```

- [ ] **Step 4: Add deterministic VAE encoding and frozen AT boundary**

Implement the `sample` switch without changing the default AT behavior. Freeze all AT modules during LDP initialization, override policy training-mode propagation so AT returns to eval, encode targets under `torch.inference_mode()`, detach them, and use `sample=False` in both latent normalizer fitting and LDP loss.

- [ ] **Step 5: Correct accumulation, scheduler, and EMA stepping**

Use `(batch_idx + 1) % accumulate_every == 0 or batch_idx + 1 == num_batches`. Step optimizer, scheduler, and EMA together; zero gradients only after those steps. Scheduler length uses `ceil`. Keep global logging counters compatible with checkpoint resume, while introducing an optimizer-step counter when needed for EMA resume.

- [ ] **Step 6: Run Task 3 and existing resume/EMA tests**

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q tests/test_pick_tube_ldp_training_v2.py tests/test_gradient_accumulation.py tests/test_workspace_resume.py
```

Expected: all pass.

- [ ] **Step 7: Commit Task 3**

```bash
git add reactive_diffusion_policy/model/vae/model.py reactive_diffusion_policy/dataset/real_image_tactile_latent_diffusion_dataset.py reactive_diffusion_policy/policy/latent_diffusion_unet_image_policy.py reactive_diffusion_policy/workspace/train_at_workspace.py reactive_diffusion_policy/workspace/train_diffusion_unet_image_workspace.py tests/test_pick_tube_ldp_training_v2.py tests/test_gradient_accumulation.py
git commit -m "fix: make LDP latent training deterministic"
```

---

### Task 4: Artifact manifests, normalizer cache signatures, and strict deployment

**Files:**
- Create: `reactive_diffusion_policy/common/artifact_manifest.py`
- Modify: `reactive_diffusion_policy/workspace/train_diffusion_unet_image_workspace.py`
- Modify: `reactive_diffusion_policy/workspace/base_workspace.py`
- Modify: `deploy_pick_tube_rdp.py`
- Modify: `configs/deploy_pick_tube_rdp.yaml`
- Test: `tests/test_pick_tube_artifact_manifest.py`
- Modify: `tests/test_pick_tube_rdp_deploy.py`

**Interfaces:**
- Produces `sha256_file(path: Path) -> str`, `stable_json_digest(value) -> str`, and `ArtifactManifest` serialization/validation.
- Produces `build_normalizer_cache_signature(cfg, dataset, at_path) -> dict` and exact-match cache reuse.
- Extends `load_policy(..., artifact_verification: str)` with `strict` and `legacy-compatible`.
- V2 checkpoint configuration carries `artifacts` containing dataset/action/normalizer/AT/PCA/tactile identities.

- [ ] **Step 1: Write failing artifact tests**

Test stable hashing, exact cache-signature reuse, recomputation on dataset/split/AT/PCA/action-version changes, and rejection of missing v2 fields.

- [ ] **Step 2: Write failing deployment pairing tests**

Assert matching v2 bundles load, same-dimensional different AT fails, same-dimensional different PCA fails, v1/v2 mixing fails, missing v2 metadata fails in strict mode, and legacy metadata is accepted only under explicit `legacy-compatible` with a warning.

- [ ] **Step 3: Run focused tests and confirm RED**

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q tests/test_pick_tube_artifact_manifest.py tests/test_pick_tube_rdp_deploy.py
```

- [ ] **Step 4: Implement manifest and cache contract**

Write `normalizer.meta.json` atomically beside `normalizer.pkl`. Reuse only when the full canonical signature matches. Include dataset/split/action/normalizer versions, AT/PCA/tactile hashes, temporal configuration, latent target mode, and git commit. Missing or mismatched cache metadata triggers recomputation, not reuse.

- [ ] **Step 5: Persist expected artifacts in checkpoints**

Populate `cfg.artifacts` before checkpoint serialization. AT records dataset/PCA/action/normalizer identities; LDP additionally records the AT hash and deterministic latent mode.

- [ ] **Step 6: Enforce deployment artifact modes**

Add `model.artifact_verification: strict` to the v2 deployment config. Validate identities before policy use. Preserve the existing dimension checks as an early error. Require an explicit `legacy-compatible` value for old checkpoints.

- [ ] **Step 7: Run Task 4 tests**

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q tests/test_pick_tube_artifact_manifest.py tests/test_pick_tube_rdp_deploy.py
```

Expected: all pass.

- [ ] **Step 8: Commit Task 4**

```bash
git add reactive_diffusion_policy/common/artifact_manifest.py reactive_diffusion_policy/workspace/train_diffusion_unet_image_workspace.py reactive_diffusion_policy/workspace/base_workspace.py deploy_pick_tube_rdp.py configs/deploy_pick_tube_rdp.yaml tests/test_pick_tube_artifact_manifest.py tests/test_pick_tube_rdp_deploy.py
git commit -m "feat: bind RDP deployment artifacts"
```

---

### Task 5: Episode validation, checkpoint metrics, and end-to-end verification

**Files:**
- Create: `reactive_diffusion_policy/common/pick_tube_validation.py`
- Modify: `reactive_diffusion_policy/dataset/real_image_tactile_dataset.py`
- Modify: `reactive_diffusion_policy/workspace/train_at_workspace.py`
- Modify: `reactive_diffusion_policy/workspace/train_diffusion_unet_image_workspace.py`
- Modify: `reactive_diffusion_policy/config/task/pick_tube_image_tactile_emb_at_30fps.yaml`
- Modify: `reactive_diffusion_policy/config/task/pick_tube_image_tactile_emb_ldp_30fps.yaml`
- Modify: `reactive_diffusion_policy/config/train_at_workspace.yaml`
- Modify: `reactive_diffusion_policy/config/train_latent_diffusion_unet_real_image_workspace.yaml`
- Modify: `scripts/run_pick_tube_rdp_experiments.sh`
- Create: `tests/test_pick_tube_validation_v2.py`
- Modify: `tests/test_pick_tube_training_data.py`

**Interfaces:**
- Produces `compute_idle_rollout_metrics(target, prediction, idle_mask, horizon=29) -> dict[str,float]`.
- Produces a deterministic episode split manifest and validation feasibility fields.
- Logs `val_idle_translation_29_mm`, `val_idle_rotation_29_deg`, per-step p95, active metrics, micro-motion recall, and `val_idle_score`.

- [ ] **Step 1: Write failing metric/split tests**

Assert deterministic episode-level split at ratio `0.1`, source stratification, no episode overlap, correct integrated translation/rotation for synthetic sequences, correct p95, and feasibility rejection when active degradation exceeds 5% or micro-motion recall is below 95%.

- [ ] **Step 2: Run focused tests and confirm RED**

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q tests/test_pick_tube_validation_v2.py tests/test_pick_tube_training_data.py
```

- [ ] **Step 3: Implement validation metrics and split manifest**

Calculate physical per-arm metrics from unnormalized action, use SO(3) composition for integrated rotation, and report active/idle plus left/right components. Store exact train/validation episode IDs and split digest.

- [ ] **Step 4: Integrate validation and checkpoint selection**

Configure `val_ratio=0.1`. A checkpoint is feasible only when active degradation is at most 5% and micro-motion recall is at least 95%; among feasible checkpoints use:

```text
val_idle_score = val_idle_translation_29_mm / 1.0
               + val_idle_rotation_29_deg / 0.5
```

Continue logging train/val loss, but use `val_idle_score` for v2 top-k selection. If no checkpoint is feasible, save latest for diagnosis but do not label it deployable.

- [ ] **Step 5: Correct launcher epochs and v2 resume behavior**

Make experiment labels match actual epoch values, default new v2 runs to fresh output directories, and reject resume across action-contract versions. Retain explicit 16D/30D/60D experiment selection.

- [ ] **Step 6: Run the complete relevant test suite**

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q \
  tests/test_pick_tube_action_contract_v2.py \
  tests/test_pick_tube_action_normalizer_v2.py \
  tests/test_pick_tube_at_physical_loss.py \
  tests/test_pick_tube_ldp_training_v2.py \
  tests/test_gradient_accumulation.py \
  tests/test_pick_tube_artifact_manifest.py \
  tests/test_pick_tube_validation_v2.py \
  tests/test_pick_tube_training_data.py \
  tests/test_workspace_resume.py \
  tests/test_pick_tube_rdp_deploy.py
```

Expected: zero failures.

- [ ] **Step 7: Run static and config checks**

```bash
python -m compileall -q convert_pick_tube_lerobot_to_rdp_zarr.py deploy_pick_tube_rdp.py reactive_diffusion_policy
git diff --check
```

Expected: both exit zero.

- [ ] **Step 8: Commit Task 5**

```bash
git add reactive_diffusion_policy/common/pick_tube_validation.py reactive_diffusion_policy/dataset/real_image_tactile_dataset.py reactive_diffusion_policy/workspace/train_at_workspace.py reactive_diffusion_policy/workspace/train_diffusion_unet_image_workspace.py reactive_diffusion_policy/config/task/pick_tube_image_tactile_emb_at_30fps.yaml reactive_diffusion_policy/config/task/pick_tube_image_tactile_emb_ldp_30fps.yaml reactive_diffusion_policy/config/train_at_workspace.yaml reactive_diffusion_policy/config/train_latent_diffusion_unet_real_image_workspace.yaml scripts/run_pick_tube_rdp_experiments.sh tests/test_pick_tube_validation_v2.py tests/test_pick_tube_training_data.py
git commit -m "feat: select RDP checkpoints by idle drift"
```

## Post-implementation training and deployment commands

Code completion does not claim that new checkpoints meet physical release thresholds. After implementation, regenerate all PCA16/PCA30/PCA60 Zarr buffers into new v2 directories, train each AT from scratch, select passing AT checkpoints, train paired LDP checkpoints from scratch, then run fixed-observation DDIM `8/20/50/100` and slow-update `1/4/8/16` ablations before a watchdog-protected robot canary.

The implementation handoff must report the exact commands supported by the existing launchers after their v2 changes, but it must not launch multi-hour training or move a robot without a separate explicit run request.

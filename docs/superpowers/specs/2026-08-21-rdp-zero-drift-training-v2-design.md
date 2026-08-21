# RDP Zero-Drift Training v2 Design

## Goal

Retrain the 16D, 30D, and 60D pick-tube RDP policies so a physically idle arm is represented, reconstructed, predicted, and deployed as an unbiased no-op while preserving active-arm task performance.

The accepted release thresholds are:

- AT posterior-mode reconstruction over 29 idle steps: less than 1 mm accumulated translation and less than 0.5 degrees accumulated rotation.
- Idle per-step action p95: less than 0.05 mm translation and less than 0.03 degrees rotation.
- A 60-second live left-active/right-idle trial: less than 2 mm right-end-effector net translation and less than 1 degree net rotation.
- Active left-arm metrics may degrade by no more than 5% relative to the frozen v1 baseline.
- Deliberate right-arm micro-motion recall must be at least 95%.

## Scope and non-goals

This design introduces a new 20D action-training contract and requires rebuilding the training dataset plus retraining every AT and LDP checkpoint. The external action layout remains unchanged:

```text
left  = [delta_position(3), rotation_6d(6), gripper(1)]
right = [delta_position(3), rotation_6d(6), gripper(1)]
```

The v2 contract is not weight-compatible with v1 even though both use 20 values. A v1 AT, v2 normalizer, and v2 LDP must never be mixed.

The following are outside this first implementation:

- Replacing the 20D action with a 14D Lie-algebra representation.
- Treating a deployment deadband or inactive-arm pose lock as the primary fix.
- Changing the established observation positions `[1, 3]`, action start position `3`, horizon `32`, or raw-observation downsample ratio `2`.
- Rewriting the entire VAE hierarchy as registered `torch.nn.Module` objects.

A 14D representation remains the fallback only if the accepted 20D v2 design cannot meet the release thresholds.

## Evidence driving the design

The design addresses confirmed behavior rather than a suspected left/right wiring error:

- All inspected episode terminal actions contain an invalid all-zero rotation-6D. With `horizon=32` and `pad_after=28`, each terminal row appears 435 times across terminal windows, about 13.59 times the exposure of an interior row.
- Current AT checkpoints reconstruct an exactly neutral right-arm chunk with approximately 4.8 to 6.0 mm accumulated translation over 29 decoded steps. The 30D reconstruction rotation bias is the same order as the live deployment drift.
- Current relative-position normalization does not map physical zero to normalized zero.
- Current AT loss is a scalar L1 over raw 20D values and does not measure physical SO(3) rotation error or idle-arm stationarity.
- Current LDP latent normalizer and training target use independent posterior samples.
- Validation is disabled and checkpoints are selected without an episode-held-out idle metric.
- No left/right slice, temporal index, PCA arm grouping, or normalize/unnormalize round-trip error was found.

## Architecture

The v2 pipeline has five explicit contracts:

```text
source parquet and tactile cache
        |
        v
auditable v2 conversion
  action_raw + action + masks + manifest
        |
        v
zero-preserving action normalizer
        |
        v
AT physical reconstruction objective
        |
        v
deterministic frozen-AT latent targets for LDP
        |
        v
strictly bound AT/LDP/PCA deployment artifacts
```

Every boundary carries a version and content fingerprint. New training fails closed on a missing or mismatched v2 contract.

## Data contract and conversion

### Stored arrays

Original parquet files are never overwritten. The converted v2 Zarr stores:

```text
action_raw       float array [N, 20], byte-faithful source action
action           float array [N, 20], canonical v2 training action
action_valid     bool array  [N],     valid source-action target
idle_arm_mask    bool array  [N, 2],  left/right physical-idle label
```

The existing observation, state, episode boundary, image, and tactile arrays retain their present layout.

### Terminal action

An invalid source terminal row remains visible in `action_raw`, is marked `action_valid=false`, and is replaced in `action` by a canonical relative no-op:

```text
delta_position = [0, 0, 0]
rotation_6d    = [1, 0, 0, 0, 1, 0]
gripper        = current corresponding state gripper width
```

Nonterminal rotation-6D rows must be finite, have two nondegenerate basis vectors, and not be collinear. Conversion fails if a nonterminal invalid rotation is found.

Sequence padding never repeats a nonzero relative action. It fills action suffixes with the same canonical no-op and repeats only the final observation/tactile values. The sampler exposes a timestep validity mask so reconstruction metrics and losses can distinguish source targets from padding.

The AT reconstruction objective excludes invalid source terminal rows and padded timesteps. Canonical no-op suffixes may still enter the AT encoder so the encoded chunk is finite and semantically meaningful. LDP targets are generated only from canonicalized chunks; an invalid all-zero rotation never enters the AT latent target path.

### Data-driven idle mask

Idle labels are generated from the original physical per-arm deltas before canonicalization. A candidate arm enters idle when all of the following hold for at least eight consecutive frames:

- Translation norm is below 0.5 mm per step.
- SO(3) rotation angle is below 0.25 degrees per step.
- Absolute gripper change is below 0.5 mm per step.
- The other arm is active.

The arm exits idle when any of the following holds for at least two consecutive frames:

- Translation norm exceeds 0.8 mm per step.
- Rotation angle exceeds 0.4 degrees per step.
- Absolute gripper change exceeds 0.8 mm per step.

When an arm is idle, its v2 training target is canonicalized to zero translation, identity rotation, and the preceding gripper width. The original action remains available in `action_raw`.

The converter records the thresholds, hysteresis lengths, frame counts changed per arm, and per-source mask coverage. The held-out micro-motion recall metric prevents these thresholds from silently erasing deliberate fine motion.

### Dataset manifest

The v2 manifest contains at least:

- Dataset fingerprint and source episode list/order/length digest.
- Action representation and normalizer versions.
- Terminal repair count and invalid nonterminal count.
- Idle thresholds, hysteresis parameters, and left/right coverage.
- Tactile cache source/revision, tensor order, and SHA256.
- PCA SHA256, output dimension, and sensor-to-arm order.
- Converter git commit.
- Array shapes and dtypes.

Training refuses a missing v2 manifest, a fingerprint mismatch, an invalid nonterminal rotation, a tactile-cache identity mismatch, or an unexpected sensor order.

## Zero-preserving action normalizer

The action representation is identified as `bimanual_relative_pose20d_v2`, and its normalizer as `zero_centered_v2`.

### Translation

Each relative-translation component uses a robust symmetric training-set scale and zero offset:

```text
scale_j = max(quantile_99.5(abs(delta_position_j)), epsilon)
normalized_j = delta_position_j / scale_j
```

The transform does not silently clamp outliers; out-of-range rates are reported. This guarantees that physical zero maps exactly to normalized zero.

### Rotation

Rotation is represented to the network as a residual from identity:

```text
identity_6d = [1, 0, 0, 0, 1, 0]
rotation_residual = rotation_6d - identity_6d
```

Identity therefore maps exactly to zero. Decoder residuals are added back to identity and projected to SO(3) with differentiable Gram-Schmidt before physical loss computation or action output.

### Gripper

Gripper width remains an absolute quantity and retains range normalization. It is not forced to have a zero-centered physical interpretation.

Normalizer fitting uses only the training split, valid source rows, and canonical v2 actions. The serialized normalizer includes its parameter hash, dataset fingerprint, action schema, and split digest.

## AT objective

AT decoder output is differentiably unnormalized before physical loss calculation. Losses are first averaged within each arm and then combined so dimension count or activity frequency cannot cause one arm to dominate.

The objective is:

```text
L = L_position
  + L_rotation
  + L_gripper
  + lambda_idle * L_idle
  + lambda_degenerate * L_degenerate
  + beta * L_KL
  + lambda_rot6_aux * L_rot6_aux
```

Definitions:

- `L_position`: Huber error in physical translation, divided by 1 mm.
- `L_rotation`: Huber error of SO(3) geodesic angle, divided by 1 degree.
- `L_gripper`: Huber physical-width error, divided by 5 mm.
- `L_idle`: idle-mask translation error relative to zero plus geodesic rotation error relative to identity, scaled by 0.1 mm and 0.05 degrees respectively.
- `L_degenerate`: penalty when the two predicted rotation-6D basis vectors are too short or nearly collinear.
- `L_KL`: existing Gaussian posterior regularizer, initially retaining beta `1e-6`.
- `L_rot6_aux`: optional raw residual-space Huber auxiliary term with weight no greater than 0.1; it is not the primary rotation objective.

The first controlled sweep uses `lambda_idle` in `{0.5, 1.0, 2.0}` while holding the data split and seeds fixed. All terms honor `action_valid`, padding validity, and per-arm idle masks.

Every decoded rotation must be finite and satisfy:

```text
Frobenius norm(R^T R - I) < 1e-5
abs(det(R) - 1) < 1e-5
```

## LDP latent contract

AT remains stochastic during AT training but becomes deterministic at the LDP boundary:

```text
AT training                  posterior.sample()
AT validation                posterior.mode()
LDP latent normalizer fit    posterior.mode()
LDP training target          posterior.mode()
LDP validation target        posterior.mode()
```

This preserves AT decoder robustness while ensuring that identical action chunks have identical LDP targets and normalizer statistics.

### Frozen AT boundary

After loading AT for LDP training:

- AT is placed in evaluation mode.
- Every AT parameter has `requires_grad=false`.
- Target encoding runs under inference mode.
- Encoded latent targets are detached.
- AT is excluded from the LDP optimizer.

Tests assert that AT gradients remain `None`, AT parameters are bitwise unchanged across an LDP step, and AT stays in evaluation mode. The first implementation uses an explicit frozen-tokenizer interface rather than a broad VAE class hierarchy rewrite.

### Gradient accumulation and EMA

Training tracks microsteps separately from optimizer steps. Optimizer, scheduler, zero-grad, and EMA updates occur only after a complete accumulation group or the final partial group of an epoch. Scheduler length is:

```text
ceil(number_of_batches / gradient_accumulate_every) * number_of_epochs
```

EMA follows optimizer steps, never raw microsteps.

## Validation and checkpoint selection

The fixed validation split is 10% at episode granularity, stratified so every source dataset contributes at least one validation episode. Window-level random splitting is prohibited. The split seed and exact episode lists are part of the manifest and checkpoint.

AT and end-to-end LDP validation report metrics by:

- Left and right arm.
- Active and idle phase.
- Interior and terminal region.
- Translation signed bias, MAE, p50, and p95.
- SO(3) rotation geodesic error.
- Gripper MAE.
- 29-step and 300-step integrated target drift.
- Posterior mean and standard deviation.
- Deliberate micro-motion precision and recall.

Checkpoint selection uses hard feasibility constraints rather than hiding regressions in a weighted sum:

1. Active left-arm metrics degrade by no more than 5% relative to the frozen v1 baseline.
2. Deliberate micro-motion recall is at least 95%.
3. Among feasible checkpoints, minimize normalized idle score:

```text
idle_score = translation_29_steps_mm / 1.0
           + rotation_29_steps_degrees / 0.5
```

Training loss remains observable but is not the top-k selection target.

## Artifact integrity

AT checkpoints record the dataset, action contract, normalizer, train/validation split, PCA, and tactile-cache fingerprints. LDP checkpoints additionally record the exact AT SHA256 and latent-target mode.

The LDP normalizer cache has a sidecar manifest containing:

- Dataset and split fingerprints.
- Action representation and normalizer versions.
- AT and PCA SHA256 values.
- Tactile encoder/cache identity.
- Horizon, padding, temporal ratio, and latent shape.
- Posterior target mode.
- Normalizer parameter hash.
- Git commit.

An exact manifest match permits cache reuse. A missing or changed field forces recomputation; path existence alone is never sufficient.

Deployment supports two explicit loading modes:

- `strict`: required for v2; missing metadata or any AT/PCA/action/normalizer mismatch is fatal.
- `legacy-compatible`: explicitly permits v1 checkpoints with a prominent warning for offline comparison only.

Dimension equality is retained as a preliminary check but cannot substitute for identity verification.

## Deployment inference selection

Inference settings are selected by staged offline ablation rather than inheriting the current DDIM-8/slow-update-16 values.

With identical recorded observations and at least 20 fixed diffusion seeds:

1. Hold slow-update interval at 1 and compare DDIM steps 8, 20, 50, and 100.
2. Select the smallest DDIM count that meets idle and active thresholds within the real-time budget.
3. Hold that DDIM count and compare slow-update intervals 1, 4, 8, and 16.

Each run records raw latent, decoded 20D action, right-arm physical relative action, 300-step integrated target, latency, GPU memory, and 30 Hz deadline-miss rate.

No deployment setting is accepted if it meets latency by violating the agreed idle, active, or micro-motion thresholds.

## Testing

### Unit tests

- Physical zero translation and identity rotation normalize to exact zero and round-trip within `1e-7`.
- Terminal all-zero rotation is converted to a canonical no-op while `action_raw` is unchanged.
- Invalid nonterminal rotation is rejected.
- Idle-mask entry/exit hysteresis matches the specified thresholds and frame counts.
- Padding action is canonical no-op and observation padding retains the last observation.
- AT masked physical losses exclude invalid and padded targets.
- All projected rotations satisfy orthogonality/determinant bounds and remain finite.
- LDP posterior-mode targets are repeatable for identical input.
- LDP backward leaves every AT gradient as `None` and parameters unchanged.
- Accumulation factors 2 and 3 produce correct optimizer, scheduler, and EMA step counts, including final partial groups.
- Any dataset, split, AT, PCA, action-contract, or normalizer hash change invalidates the normalizer cache.
- Same-dimensional but different AT or PCA artifacts fail strict deployment loading.
- Existing observation `[1, 3]` and action-start `3` temporal alignment remains unchanged.

### Integration tests

- AT v2 encode/decode maintains `[batch, 32, 20]` externally and produces valid physical rotations.
- LDP maintains its expected `[batch, 8, 32]` latent shape.
- Matched 16D, 30D, and 60D artifact bundles pass strict loading.
- Cross-pairing same-dimensional AT artifacts fails.
- Episode-held-out metrics are deterministic for fixed seeds.

### Experiment gates

- Reproduce and archive the v1 AT neutral-reconstruction and LDP command-only baselines.
- Run one-variable AT ablations in this order: terminal/mask, zero-centered normalizer, physical rotation loss, idle canonicalization, and idle-loss weight.
- Only AT candidates passing the 29-step thresholds proceed to LDP training.
- Run posterior-sample versus posterior-mode LDP ablation with the selected AT.
- Run DDIM and slow-update ablations on fixed observations/seeds.
- Pass command-only replay before any live canary.

## Live canary and release

The live canary uses an independent watchdog, not a learned hold rule. During a left-active/right-idle test, it stops the trial if the right arm moves more than 5 mm or 3 degrees from its initial pose, or exceeds configured velocity safety limits.

Release requires:

- All unit and integration tests passing.
- Strict artifact verification passing.
- The accepted offline AT, end-to-end LDP, active-performance, and micro-motion thresholds passing.
- A 60-second live trial with right-arm net drift below 2 mm and 1 degree.
- No controller saturation, robot error, NaN, or deadline safety violation.

## Rollout sequence

1. Freeze the current v1 datasets, checkpoints, recorded observations, seeds, and baseline reports.
2. Add the v2 action contract, auditable conversion, masks, and manifests.
3. Add zero-preserving normalization and AT physical losses.
4. Retrain and select AT checkpoints independently for 16D, 30D, and 60D.
5. Add deterministic latent targets, frozen AT, accumulation/EMA corrections, and cache signatures.
6. Retrain and select paired LDP checkpoints.
7. Add strict artifact checks and generate deployment bundles.
8. Run fixed-seed DDIM and slow-update ablations.
9. Pass command-only replay and then the watchdog-protected robot canary.
10. Promote only artifact bundles meeting every release gate.

## Compatibility and migration

Existing v1 checkpoints remain usable only through an explicit legacy comparison path. They do not gain the v2 fixes. New runs use fresh output directories with resume disabled across the v1/v2 boundary.

The original parquet files, tactile embeddings, tactile encoder, and PCA artifacts may be reused only when their manifest identities match. Action and latent normalizers are always recomputed for v2. All AT and LDP model weights are retrained from the beginning.

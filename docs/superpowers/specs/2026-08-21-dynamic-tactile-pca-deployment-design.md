# Dynamic Tactile PCA Deployment Design

## Goal

Make the pick-tube deployment inference path run checkpoints trained with
16D, 30D, or 60D tactile embeddings when the deployment YAML explicitly
selects a matching LDP checkpoint, AT checkpoint, and arm-wise PCA artifact.

This work covers deployment inference only. It does not change training
entrypoints or infer artifact paths from a dimension setting.

## Current State

`BimanualTactilePCA` already accepts component arrays shaped
`[2, N, 1024]` and exposes an output dimension of `2 * N`. The deployment
runtime also uses that output dimension when it projects tactile encoder
features. Consequently, the core projection supports the existing 2x8,
2x15, and 2x30 artifacts.

The remaining deployment gap is checkpoint preflight validation. The current
loader validates only the LDP observation dimension against the PCA output.
It does not validate the LDP extended-observation declaration or the AT
checkpoint before model construction. A mismatched AT checkpoint therefore
fails later with an opaque state-dictionary shape error.

The local 60D LDP checkpoint is also truncated and cannot be deserialized.
That artifact must be replaced independently of this code change.

## Artifact Selection

The deployment YAML remains the only artifact-selection interface:

```yaml
model:
  ldp_checkpoint: /path/to/D/ldp/latest.ckpt
  at_checkpoint: /path/to/D/at/latest.ckpt
  tactile_pca_path: /path/to/tactile_pca_2xN.npz
```

The user selects all three paths explicitly. The code does not add a
`tactile_dim` setting and does not construct paths automatically. This avoids
mixing PCA statistics from a different dataset with otherwise shape-compatible
checkpoints.

## Architecture and Data Flow

The deployment startup sequence will be:

1. Resolve the three configured artifact paths and verify that they exist.
2. Load the PCA artifact on the requested device and derive `output_dim` from
   its component shape.
3. Read the LDP checkpoint payload on CPU and extract every declared tactile
   dimension used by deployment.
4. Read the small AT checkpoint payload on CPU and extract the corresponding
   declarations.
5. Validate that PCA output, LDP observation, LDP extended observation, AT
   observation, and AT extended observation dimensions are identical.
6. Only after preflight succeeds, construct the workspace, load weights, move
   the policy to the target device, and begin robot initialization.

Runtime tactile data continues to flow as:

```text
4 RGB tactile images
  -> tactile encoder [4, 512]
  -> arm grouping [2, 1024]
  -> PCA [2, N]
  -> flattened tactile observation [2N]
  -> LDP observation and AT decoder
```

The tactile encoder contract remains fixed at four 512D vectors. Only the PCA
component count and downstream tactile dimension are dynamic.

## Validation Interface

Checkpoint dimension extraction will be isolated from policy construction so
it can be tested without instantiating the multi-gigabyte LDP workspace.
Validation will report all dimension sources together, including their
artifact roles, rather than exposing a later PyTorch tensor-size mismatch.

The accepted invariant is:

```text
PCA output
  == LDP shape_meta.obs.tactile_embedding
  == LDP shape_meta.extended_obs.tactile_embedding
  == AT shape_meta.obs.tactile_embedding
  == AT shape_meta.extended_obs.tactile_embedding
```

Missing or malformed dimension declarations will fail during preflight with
the checkpoint role and field name in the error. A corrupt checkpoint will
retain the underlying deserialization cause while adding which configured
artifact could not be inspected.

No fallback, silent coercion, padding, truncation, or projection reshaping is
allowed.

## PCA Artifact Validation

The existing dynamic `[2, N, 1024]` component contract is retained for
`N >= 1`. Deployment tests will establish 2x8, 2x15, and 2x30 as the supported
artifact matrix corresponding to 16D, 30D, and 60D checkpoints.

The implementation will also reject non-finite PCA means or components before
robot startup. This prevents NaN or infinity from propagating into every
predicted action.

## Testing

Lightweight CPU tests will cover:

- PCA construction and projection for 2x8, 2x15, and 2x30 components.
- Both accepted encoder input layouts, including batched inputs.
- Runtime slow and fast inference paths parameterized over 16D, 30D, and 60D.
- Matching PCA/LDP/AT metadata for all three dimensions.
- PCA versus LDP mismatch, LDP observation versus extended-observation
  mismatch, and AT versus LDP/PCA mismatch.
- Malformed or non-finite PCA artifacts.

Artifact smoke checks may inspect real checkpoint metadata on CPU, but normal
unit tests will not instantiate every multi-gigabyte LDP model. Final GPU
acceptance requires:

1. The current complete 16D three-artifact set loads and reaches robot warmup.
2. A replacement complete 60D LDP, the existing 60D AT, and the 2x30 PCA load
   and reach robot warmup.

The current truncated 60D LDP is an external acceptance blocker, not a code
condition that can be handled as success.

## Non-Goals

- Changing AT or LDP training scripts.
- Automatically choosing files from a tactile dimension.
- Converting checkpoints between dimensions.
- Supporting mismatched artifacts through padding or truncation.
- Changing robot observation, state, action, or control-frequency contracts.

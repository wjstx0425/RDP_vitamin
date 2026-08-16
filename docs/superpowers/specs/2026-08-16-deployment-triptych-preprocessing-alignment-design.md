# Deployment Triptych Preprocessing Alignment Design

## Problem

The collection pipeline splits each 3840x800 camera frame into three
1280x800 panels, rotates the left tactile panel by 180 degrees, and directly
resizes every panel to the model resolution with OpenCV's default
`INTER_LINEAR` interpolation. The deployment path instead preserves aspect
ratio and center-crops each panel to a square. For a 1280x800 panel this drops
roughly 37.5 percent of the horizontal field of view, so deployment does not
match the training-data geometry.

## Decision

Align the shared deployment preprocessing with the collection pipeline. In
`real_world/bimanual_umi_env.py`, all visual and tactile panels will be resized
directly to `obs_image_resolution` with `cv2.INTER_LINEAR`, without aspect-ratio
preservation or center cropping. The existing triptych split, left-tactile
180-degree rotation, BGR-to-RGB conversion, optional float32 scaling, bad-frame
handling, and fisheye-mask order remain unchanged.

This change applies to every policy using the shared environment: RDP at
224x224 and SmolVLA at 256x256, in both vision and vitac modes. That is
intentional because the collection pipeline defines the image contract.

## Considered Approaches

1. **Direct resize in the triptych path (selected).** This exactly matches the
   collection pipeline and keeps the generic image-transform helper unchanged.
2. Add a selectable legacy center-crop mode. This eases rollback but preserves
   two image contracts and permits accidental training/deployment mismatch.
3. Change the generic `get_image_transform` helper. This is broader than the
   bug and risks changing unrelated future callers.

## Implementation Boundary

- Add a small module-level pure helper for panel resizing and BGR-to-RGB
  conversion so geometry can be tested without camera hardware.
- Wire the existing camera transform to that helper for visual and, in vitac
  mode, both tactile panels.
- Remove the now-unused `get_image_transform` import from this environment.
- Do not change the untracked `VB-VLA/` deployment copy, generic image helper,
  camera protocol, policy configuration, or saved-observation tooling.

## Error Handling

Existing behavior is preserved: missing, malformed, or incorrectly sized
triptych frames are dropped. OpenCV errors for an invalid configured output
resolution remain fail-fast during transform execution.

## Testing

A hardware-free regression test will feed a synthetic 3840x800 BGR triptych
containing coordinate gradients and edge/corner sentinels through the actual
transform passed to `MultiUvcCamera`. It will compare all three outputs exactly
against the collection-pipeline reference operations:

1. split the three panels;
2. rotate the left tactile panel by 180 degrees;
3. call `cv2.resize(..., interpolation=cv2.INTER_LINEAR)`;
4. convert BGR to RGB.

The test also checks output shape, dtype, full-width edge retention, and the
left-tactile rotation. It must fail against the current center-crop behavior
before production code changes, then pass after the minimal implementation.
Related camera and action-scheduling tests, followed by the full test suite,
provide regression coverage.

## Acceptance Criteria

- Deployment output is pixel-identical to the stated collection-pipeline
  reference for the synthetic triptych.
- No horizontal center crop remains in the shared deployment triptych path.
- Visual and both tactile outputs retain RGB channel order and expected
  rotation semantics.
- Existing user changes in `configs/server_config.py` and the untracked
  `VB-VLA/` tree are not modified.

# Deployment Triptych Preprocessing Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the shared deployment camera transform pixel-equivalent to the collection pipeline's direct-resize geometry.

**Architecture:** Keep frame validation, masking, triptych splitting, and tactile rotation in `BimanualUmiEnv`. Add one module-level pure panel helper that performs explicit `INTER_LINEAR` resize, BGR-to-RGB conversion, and optional float scaling, then wire the existing camera closure to it.

**Tech Stack:** Python 3.11, NumPy, OpenCV, pytest, uv

## Global Constraints

- Directly resize every 1280x800 panel to `obs_image_resolution`; do not preserve aspect ratio or center crop.
- Use explicit `cv2.INTER_LINEAR` to match collection-time `cv2.resize` defaults.
- Preserve left-tactile 180-degree rotation, BGR-to-RGB conversion, optional float32 scaling, bad-frame handling, and fisheye-mask ordering.
- Apply the shared contract to RDP and SmolVLA, in both vision and vitac modes.
- Do not modify `utils/common/cv2_util.py`, the untracked `VB-VLA/` copy, or the user's `configs/server_config.py` changes.

---

### Task 1: Add and fix the triptych preprocessing regression

**Files:**
- Create: `tests/test_camera_frame_preprocessing.py`
- Modify: `real_world/bimanual_umi_env.py:1-220`

**Interfaces:**
- Consumes: a decoded BGR `np.ndarray` panel and `(width, height)` output resolution.
- Produces: `_resize_panel_for_model(panel: np.ndarray, output_resolution: tuple[int, int], obs_float32: bool) -> np.ndarray` and unchanged camera-transform output keys.

- [ ] **Step 1: Write the failing deploy-path regression test**

Create a synthetic 3840x800 BGR triptych with distinct visual edge sentinels and left-tactile corner sentinels. Monkeypatch only hardware constructors, capture the real transform passed to `MultiUvcCamera`, and assert exact equality with this collection-pipeline reference:

```python
def training_resize_rgb(panel: np.ndarray) -> np.ndarray:
    resized = cv2.resize(panel, (224, 224), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

expected = {
    "color": training_resize_rgb(visual),
    "left_tactile": training_resize_rgb(cv2.rotate(left, cv2.ROTATE_180)),
    "right_tactile": training_resize_rgb(right),
}
actual = deploy_transform({"color": triptych_bgr_frame.copy()})

for key, expected_image in expected.items():
    assert actual[key].shape == (224, 224, 3)
    assert actual[key].dtype == np.uint8
    np.testing.assert_array_equal(actual[key], expected_image)
```

Also assert the visual left/right edge RGB values and the rotated left-tactile output corners explicitly.

- [ ] **Step 2: Run the new test and verify RED**

Run:

```bash
uv run --no-sync pytest -q tests/test_camera_frame_preprocessing.py::test_vitac_transform_matches_training_triptych_preprocessing -x
```

Expected: one assertion failure showing the deploy output differs from the direct-resize reference because the current transform center-crops the panel.

- [ ] **Step 3: Implement the minimal direct-resize helper**

In `real_world/bimanual_umi_env.py`, remove the unused `get_image_transform` import and add:

```python
def _resize_panel_for_model(
    panel: np.ndarray,
    output_resolution: tuple[int, int],
    obs_float32: bool,
) -> np.ndarray:
    resized = cv2.resize(
        panel,
        output_resolution,
        interpolation=cv2.INTER_LINEAR,
    )
    resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    if obs_float32:
        resized = resized.astype(np.float32) / 255
    return resized
```

Replace all three uses of the center-cropping transform in the camera closure with `_resize_panel_for_model(panel, obs_image_resolution, obs_float32)`. Do not change split, rotation, mask, or bad-frame logic.

- [ ] **Step 4: Run the focused test and verify GREEN**

Run:

```bash
uv run --no-sync pytest -q tests/test_camera_frame_preprocessing.py::test_vitac_transform_matches_training_triptych_preprocessing -x
```

Expected: `1 passed`.

- [ ] **Step 5: Run related regression tests**

Run:

```bash
uv run --no-sync pytest -q \
  tests/test_camera_frame_preprocessing.py \
  tests/test_camera_timestamp_alignment.py \
  tests/test_bimanual_action_scheduling.py \
  tests/test_camera_startup_timeout.py
```

Expected: all selected tests pass with no failures.

- [ ] **Step 6: Run repository-wide verification**

Run:

```bash
uv run --no-sync pytest -q
git diff --check
git status --short
```

Expected: pytest exits zero; `git diff --check` emits no output; status includes only the planned test/code/plan changes plus the pre-existing `configs/server_config.py` and `VB-VLA/` entries.

- [ ] **Step 7: Commit the implementation**

```bash
git add \
  docs/superpowers/plans/2026-08-16-deployment-triptych-preprocessing-alignment.md \
  real_world/bimanual_umi_env.py \
  tests/test_camera_frame_preprocessing.py
git commit -m "fix: align deployment image preprocessing with data"
```

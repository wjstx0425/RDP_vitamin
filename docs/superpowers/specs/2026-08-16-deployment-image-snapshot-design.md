# Deployment Image Snapshot Design

## Problem

The operator needs to inspect the exact images produced by the deployment
observation path without allowing the robot to enter its action loop. The
existing `--save_obs` option starts only after the policy client sends the
`start` signal, saves every step, writes lossy JPEG files, and passes in-memory
RGB arrays directly to OpenCV without converting them to BGR. Its files are
therefore unsafe as evidence for a pre-motion visual check.

## Decision

Add a separate `--save-image-snapshot` flag to
`deploy_scripts/bimanual_smolvla_online.py`. The mode will use the negotiated
policy configuration and the real `BimanualUmiEnv` camera transform. After the
normal camera warmup, it will build the same `obs_dict` that would be published
to the policy client, save its `observation.images.*` values, and return before
publishing a warmup observation, waiting for a `start` signal, receiving an
action, or scheduling a robot command.

The environment still starts the existing controller process because
`BimanualUmiEnv.get_obs()` needs robot state for the complete observation, but
snapshot mode never sends an action command. Context-manager shutdown and
robot-client shutdown remain responsible for stopping all child processes.

## Alternatives Considered

1. **Pre-action snapshot in the real deployment path (selected).** This proves
   the actual negotiated resolution, camera mapping, and triptych transform
   while keeping action scheduling unreachable.
2. Repair and reuse continuous `--save_obs`. This requires passing the `start`
   gate and can write an unbounded number of files, which is unnecessary for a
   visual check.
3. Add a standalone camera script. This avoids a policy client but duplicates
   deployment configuration and preprocessing, so it could show images from a
   path different from the one being audited.

## Output Contract

Each run creates a unique directory under `eval_obs_data/`:

```text
deploy_snapshot_YYYYMMDD_HHMMSS_microseconds/
  observation.images.camera0.png
  observation.images.camera1.png
  observation.images.tactile_left_0.png
  observation.images.tactile_right_0.png
  observation.images.tactile_left_1.png
  observation.images.tactile_right_1.png
  manifest.json
```

Dual-arm RDP mode must produce all six PNG files. Single-arm or vision modes
save only the image keys present in the negotiated observation.

The source arrays are HWC RGB `uint8` values. Before `cv2.imwrite`, each image
is converted with `cv2.COLOR_RGB2BGR`. PNG is used to avoid additional lossy
compression. `manifest.json` records policy type, data type, filename, shape,
dtype, minimum, and maximum for each saved image.

## Components and Data Flow

Introduce a focused pure-to-filesystem helper that:

1. selects keys beginning with `observation.images.`;
2. validates each image as HWC, three-channel, `uint8` data;
3. converts RGB to BGR and writes a PNG;
4. checks the boolean result from `cv2.imwrite`; and
5. writes the manifest only after all images succeed.

The main deployment flow calls this helper immediately after
`get_real_umi_obs_dict(...)` constructs the warmup observation. Snapshot mode
prints the absolute output path, stops the robot client, and returns. Normal
deployment and the existing continuous `--save_obs` behavior remain unchanged.

`--save-image-snapshot` is incompatible with `--dry-run`, because dry-run does
not initialize cameras, and with `--save_obs`, because snapshot mode exits
before continuous saving can start. Invalid combinations fail before hardware
initialization with a clear Click error.

## Error Handling

- An observation with no `observation.images.*` keys fails explicitly.
- Invalid image shape or dtype reports the offending observation key.
- A failed PNG write raises an error containing the resolved output path.
- Failures propagate through the existing context managers so cameras,
  controller, shared memory, and the robot client are stopped.
- Partially written snapshot directories are retained as diagnostic evidence;
  a manifest is absent unless the complete image set was written.

## Testing

Add focused tests that:

- save RGB arrays containing unambiguous red, green, and blue pixels, reload
  the PNG through OpenCV, convert it back to RGB, and assert exact equality;
- verify filenames and manifest shape/dtype/range metadata;
- reject invalid dtype, invalid shape, and observations without image keys;
- run the CLI snapshot branch with fake client/environment objects and assert
  that it captures the post-`get_real_umi_obs_dict` observation, returns before
  `publish_obs`, never calls `wait_for_start`, and never enters the action loop;
  and
- preserve existing normal and dry-run behavior tests.

The complete repository pytest suite is the final regression gate.

## Usage

Start the normal RDP policy client so it sends the negotiated RDP/vitac
configuration, then start the server with:

```bash
scripts/bimanual_rdp.sh --save-image-snapshot
```

The server prints the snapshot directory and exits before the operator can
send the robot start signal. The policy client will observe the intentional
server disconnect and should then be stopped normally.

## Acceptance Criteria

- A snapshot run creates lossless, visually correct PNG files from the exact
  observation dictionary prepared for policy publication.
- RDP dual-arm output contains two visual and four tactile images at 224x224.
- Snapshot mode cannot reach start waiting, action receipt, or command
  scheduling.
- Existing continuous observation saving and normal deployment are unchanged.
- User changes in `configs/server_config.py` and the untracked `VB-VLA/` tree
  are not modified or committed.

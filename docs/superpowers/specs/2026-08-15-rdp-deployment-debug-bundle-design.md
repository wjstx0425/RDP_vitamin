# RDP Deployment Debug Bundle Design

## Problem

The pick-tube RDP deployment spans two repositories and a separate training
machine. Current evidence is distributed across robot action logs, terminal
output, checkpoint metadata, offline replay commands, and source-code audits.
This makes it difficult to reproduce the investigation on the machine that
contains the original LeRobot and RDP Zarr data, and it encourages conclusions
that mix confirmed facts with hypotheses.

The debug bundle must provide one reproducible path from raw demonstrations to
the command accepted by the robot controller. It must also preserve the safety
boundary established during the current investigation: all diagnostic tools
are offline and read-only, and none may connect to the robot bridge or Typhon.

## Goals

- Produce a self-contained Chinese runbook covering the deployed data contract,
  checkpoint configuration, two real-robot runs, offline replay, resolved
  defects, remaining defects, and the evidence supporting each conclusion.
- Provide standalone command-line scripts that can be copied to the training
  machine and run against explicit paths.
- Separate observations into `confirmed`, `inference`, and `unknown` categories.
- Localize failures across four boundaries: raw dataset, AT reconstruction,
  LDP-to-AT prediction, and deployed observation input.
- Give every experiment an input contract, exact command, machine-readable or
  tabular output, and an interpretation rule.
- Prevent all tools from importing or constructing `RobotBridgeClient`, making
  HTTP requests to Typhon, opening cameras, or starting controller processes.
- Preserve the user's existing uncommitted `configs/server_config.py` changes.

## Non-goals

- Fixing the policy, retraining checkpoints, smoothing actions, or changing
  controller behavior in this bundle.
- Claiming whether AT, LDP, the demonstration labels, or input-domain shift is
  the final root cause before the training-machine experiments are run.
- Replacing the existing deployment scripts or changing the wire protocol.
- Automating a real-robot acceptance test.
- Downloading the original training dataset as part of normal script execution.

## Considered Approaches

### One Markdown file with inline scripts

This is easy to transfer but makes scripts hard to test, reuse, and version.
Long Python here-documents also make the runbook difficult to scan. It is not
selected.

### Runbook plus standalone read-only scripts

The runbook explains evidence and decision rules, while focused scripts produce
repeatable measurements. This is selected because it is portable, testable,
and lets the training machine run only the checks it needs.

### Jupyter notebook

A notebook would make plots convenient but introduces environment and execution
order dependencies. The deployment machines are command-line oriented, so it
is not selected.

## Deliverables

### Runbook

Create `docs/rdp_pick_tube_debug_runbook_20260815.md`. It contains:

1. a short executive conclusion and safety warning;
2. repository, checkpoint, encoder, configuration, and log inventory;
3. the exact 20D state and action layouts, image/tactile ordering, units,
   coordinate composition, normalization, and gripper calibration;
4. the observation-to-action-to-controller data flow;
5. a comparison of the `20260815_145643` and `20260815_160855` runs;
6. the 172-frame FRS offline replay and repeated-identical-frame experiments;
7. confirmed fixes, confirmed exclusions, unresolved defects, and hypotheses;
8. a four-stage responsibility matrix and interpretation table;
9. training-machine prerequisites and copyable commands for every tool;
10. artifact collection instructions for returning results to the deployment
    machine; and
11. safety gates that must pass before another real-robot RDP run.

The runbook records exact values already established, including the bridge
frequency improvement, the remaining 19.76 Hz loop rate, five-step gripper
discontinuity, Typhon HTTP 400 termination, training action RMS, 32-seed
left/right comparison, and out-of-distribution gripper state.

### Action-log summarizer

Create `tools/rdp_debug/summarize_action_log.py`.

- Input: one or more `action_debug.jsonl` paths.
- Output: frame continuity, duration, effective frequency, period percentiles,
  scheduling counts, command lead, per-side raw pose magnitude, gripper ranges,
  and within-plan versus replan-boundary jump statistics.
- Rotation comparisons use SO(3) geodesic distance. The script must not repeat
  the existing axis-angle vector-subtraction artifact that produced values near
  `2*pi`.
- Invalid or incomplete JSONL records fail with a line-numbered error.

### Training-dataset auditor

Create `tools/rdp_debug/audit_training_dataset.py`.

- Input: either the converted RDP Zarr path or a LeRobot dataset root with
  episode Parquet files.
- Required data: 20D action, 20D state, and episode boundaries. Images are not
  loaded.
- Output: schema validation, per-side xyz/rotation/gripper distributions,
  first 30/60-frame action energy, detected first-moving side, and a bounded lag
  scan comparing action motion with consecutive state changes.
- The report distinguishes all-episode statistics from episode-start statistics.
- Left/right mapping is always reported explicitly as `[0:10]` and `[10:20]`.

### Saved-observation replay

Create `tools/rdp_debug/replay_saved_observations.py`.

- Input: RDP Vitamin repository, deployment YAML, and a directory containing
  `step_XXXXXX` saved observations.
- It loads local trusted checkpoints and invokes `PickTubeRDPRuntime` directly.
- It never creates `RobotBridgeClient` and never imports server controller code.
- Output: input shape/dtype/range checks, normalized state range, inference
  timing, finite-action checks, action statistics, five-step boundary metrics,
  and optional repeated-frame multi-seed measurements.
- The command defaults to inspection only and writes JSON only when an explicit
  `--output` path is supplied.

### Policy-stage comparator

Create `tools/rdp_debug/compare_policy_stages.py`.

- Input: RDP Vitamin repository, AT/LDP checkpoints, a converted Zarr dataset,
  episode index, and start frame.
- Stage A reads the ground-truth 20-step action chunk.
- Stage B performs AT encode/decode reconstruction with the corresponding
  extended tactile sequence.
- Stage C performs LDP latent prediction from the initial observation followed
  by AT decoding.
- Output compares per-step and aggregate left/right position, rotation, and
  gripper errors at each stage.
- Interpretation is explicit:
  - wrong ground truth means source data or conversion;
  - correct ground truth but wrong AT reconstruction means AT/checkpoint;
  - correct AT but wrong LDP-to-AT output means LDP/observation conditioning;
  - correct training samples but wrong saved-observation replay means input
    domain, normalization, calibration, or physical sensor mapping.

## Safety Architecture

All tools operate on paths supplied on the command line. They do not contain a
default bridge address, robot IP, camera device, or controller command. The
replay and stage-comparison tools import only model and dataset modules from the
Vitamin repository. Tests scan the diagnostic source for forbidden bridge,
camera, HTTP, and controller imports in addition to exercising behavior.

Scripts do not modify checkpoints or datasets. Optional outputs use a new,
explicit path and refuse to overwrite an existing file unless the user removes
it first. Dataset and checkpoint loading errors include the resolved path and
the missing contract.

## Testing

Create focused tests under `tests/rdp_debug/` using synthetic JSONL, NumPy, and
minimal temporary Zarr/Parquet fixtures where the repository dependencies make
those formats available.

Tests cover:

- valid and malformed action-log records;
- exact frequency and replan-boundary statistics;
- SO(3) wraparound without a false `2*pi` jump;
- left/right dataset slicing and episode-start energy;
- known action/state lag recovery from synthetic trajectories;
- saved-observation discovery and missing-key diagnostics;
- forbidden online dependency checks; and
- CLI help and import behavior without CUDA, cameras, bridge, or robot access.

Model-heavy replay is verified with a dry-run contract test locally. The
runbook also includes the already executed real-checkpoint offline results as
reference output; CI does not require the 673 MB LDP checkpoint.

## Acceptance Criteria

- A user on the training machine can start from the runbook and execute each
  responsibility stage without reading source code first.
- Every reported conclusion is labeled confirmed, inferred, or unknown and
  links to a log, checkpoint statistic, code path, or command output.
- The synthetic test suite passes without hardware, network access, or CUDA.
- `--help` works for all four scripts in the server repository environment.
- No diagnostic script can connect to WebSocket, HTTP, cameras, or controller
  processes through its normal or test path.
- The existing server and Vitamin working trees have no new unrelated changes;
  only the user's pre-existing server configuration modification remains.

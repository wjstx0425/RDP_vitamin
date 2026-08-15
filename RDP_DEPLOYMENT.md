# RDP client support

The existing SmolVLA server also accepts the pick-tube RDP client. Start it with:

```bash
export VB_ROBOT_TOKEN='shared-token'
bash scripts/bimanual_rdp.sh
```

For a protocol-only check that does not initialize robot hardware:

```bash
bash scripts/bimanual_rdp.sh --dry-run --dry-run-iterations 6
```

The RDP client must negotiate the following values:

```text
policy_type=rdp
data_type=vitac
action_horizon=1
steps_per_inference=1
```

For offline responsibility assignment, training-data checks, and safe replay commands, see [the 2026-08-15 pick-tube RDP debug runbook](docs/rdp_pick_tube_debug_runbook_20260815.md).

Compared with the SmolVLA path, the server publishes 224×224 observations and
schedules each returned action from receive time with a 50 ms lead. It still
uses the existing 20D state constructor, per-arm relative-action integration,
action limits, controller health check, authentication, observation sequence,
and action acknowledgement logic.

The policy-side setup and full launch order are documented in
`reactive_diffusion_policy-main/DEPLOY_PICK_TUBE_RDP.md` in the shared workspace.

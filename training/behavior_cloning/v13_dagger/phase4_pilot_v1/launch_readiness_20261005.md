# Phase 4 launch readiness

Status: **ready_for_explicit_launch_review**. No live launch. Separate approval required.

## Isolated runner and control isolation

StudentPilot inherits the RLBot transport base, not the original teacher. Its only returned ControllerState comes from the pinned 13D student. SDK PlayerInput is checked against that exact student output. OriginalShadow has no game socket and rejects control/state messages; the validated shadow, worker, decoder, client and immutable summary code are byte-identical to Phase 3.

The temporary 1v1 configuration uses Epic, Blue human and Orange student. State-setting is disabled, native replay/countdown retained, and no probe planner or receive-loop sleeps exist. Exactly one launch reservation is allowed; additional attempts require review. A recorded separate launch approval is required before --run can connect to RLBot.

Model/observation/prediction/reset contracts are unchanged. Both paths consume one actual callback stream, native RLBot prediction and canonical elapsed time. Shadow state never follows or resets to student actions. Only columns 0–12 reach the model; private labels are diagnostic. Empty-prediction errors remain unsuppressed in the original lookup branch.

## Non-live checks

`{
  "status": "code_preflight_passed",
  "worker": {
    "status": "ready",
    "checkpoint_sha256": "579fa26010140514d2e9b7f0e2871bee569c594b7da8648e31ee1489007cd5b7",
    "parameters": 26181,
    "torch": "2.11.0+cu128",
    "device": "cpu",
    "hidden_resets": 1,
    "role": "diagnostic_only"
  },
  "shadow_transport_rejected": true,
  "byte_identical_shadow_worker_decoder_client_and_publication": true,
  "student_is_only_returned_controller": true,
  "causal_hidden_chain": true,
  "native_float_and_button_conversion_exact": true,
  "unexpected_controls_wrong_agent_and_unset_expected_rejected": true,
  "no_state_setting_or_intentional_pauses": true,
  "no_dataset_or_training_reader_in_live_path": true,
  "immutable_summary_and_injected_denial_checks": true,
  "game_packet_or_prediction_constructed": false,
  "actual_live_callbacks": 0,
  "preflight_outputs_diagnostic_only": true
}`

43 protected hashes unchanged: True; V10/V12/V13 completed artifacts and both prior Phase 3 versions unchanged: True.

## Train-only descriptive reference

Status: `verified_train_only_description`.

The builder requires only four training matches: pilot_01, v10_001, v10_004, v10_006. It hashes eight required raw/observation files before any sample content is read; it never opens validation/test samples. It checks raw/float32 tensor parity and chronological submission links, then saves only phase/mask-conditioned quantiles/histograms/occupancy, never training tensors. Primary baseline is Orange-only pilot_01/v10_006; secondary is all four training matches. These descriptions cannot tune or change the model.

One pilot_01 raw header was inspected before hashing during schema discovery. Its pinned hash was immediately verified; the production reference builder hashes all required training inputs before semantic reads.

## Before launch review

If reference statistics are pending, run the following non-live commands. The first scans about 107,400 training rows and may take tens of seconds; progress lists hashes/matches. The second verifies reference integrity and repeats non-live runner checks. Neither launches Rocket League.

```powershell
& '.\training\venv\Scripts\python.exe' -u -B '.\training\behavior_cloning\v13_dagger\phase4_pilot_v1\runner\reference_statistics.py' --build
& '.\python-example\venv\Scripts\python.exe' -u -B '.\training\behavior_cloning\v13_dagger\phase4_pilot_v1\runner\launch.py' --check
```

Paste their output. Full readiness is not claimed until both succeed. Do not run a live command yet.

## Future pilot/report boundaries

After separate approval: one 90-second monotonic wall window starts at the first verified student submission, with at most 180 seconds startup. Natural match end/user stop/error can end earlier. No fallback controller or fabricated repair is returned on failure. Records remain quarantined diagnostics without a dataset split, DAgger aggregation, fitting or test evaluation.

The report separately checks final student submission/clean shutdown, native transport, model continuity, actual learner-driven states, original shadow sequences after missed student jumps, potentially unsupported expert labels, train-reference occupancy, unobserved categories and queue/latency/callback limits. No competence, causal distribution-shift or DAgger-benefit claim follows from one partial match.

Prepared configs: `C:\Users\shreyas\Desktop\model wars\training\behavior_cloning\v13_dagger\phase4_pilot_v1\prepared\20261005_061646_065073`. This preparation creates no live session.

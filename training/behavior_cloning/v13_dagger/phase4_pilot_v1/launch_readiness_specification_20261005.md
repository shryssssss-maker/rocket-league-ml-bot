# Phase 4 student-controlled pilot — launch-readiness specification

**Proposal for explicit review. No Phase 4 live runner exists yet; no live launch or preflight has occurred.**

Phase 3 is formally accepted with scope: faithful independent shadow on processed callbacks under teacher control. Lossless delivery, learner-induced states, expert-label usefulness, student competence and DAgger benefit remain unproven.

## Proposed run — preference pending

One natural 1v1 pilot: Orange 13D student against Blue human, 90 seconds of wall time after the first verified student submission, plus at most 180 seconds startup. Keep replay/countdown; stop earlier if the match ends. Confirm who plays and record their actual opponent ID. No controlled state-setting, deliberate receive pauses or extra collection rounds. This is a partial-match diagnostic, not a win-rate test.

The 90-second duration is a proposal awaiting the user preference response; no dependent runner is implemented before that choice and the specification are approved.

## Control, observation and memory contract

Frozen checkpoint: `C:\Users\shreyas\Desktop\model wars\training\behavior_cloning\v13_dagger\phase2_bootstrap_v1\B_best.pt`; SHA256 `579fa26010140514d2e9b7f0e2871bee569c594b7da8648e31ee1489007cd5b7`. Architecture and decoder remain unchanged (26,181 parameters). Existing training Python performs CPU inference; existing RLBot Python handles the game.

Only the student ControllerState reaches RLBot PlayerInput. Original teacher is shadow-only, using the byte-identical validated original-source actor with independent sequence/boost-tracker state and a non-game sink rejecting control/state messages. No teacher takeover or hybrid controller is permitted, including on error.

13D features: `ball_forward, ball_right, ball_up, prediction_forward, prediction_right, prediction_up, ball_distance, car_speed, ball_present, prediction_valid, prediction_horizon, prediction_first_offset, callback_dt`. Same frozen builder/scales/masks, columns 0–12 only, no previous-action inputs and no private sequence labels. First callback dt is zero. Duplicate/backward clock is an explicit contract failure, not an invented reset.

Use the exact actual delivered native packet, prediction and canonical elapsed time. Preserve selector `int((elapsed + 2 - first_slice_time) * 120)`. Preserve the original empty-prediction exception in its gameplay lookup branch. The independent observation probe keeps the existing recorder guard and is labeled separately. No interpolation, timestamp repair, provider substitution or simulator.

Student GRU and original shadow sequence reset once at match/session initialization, never on goal/replay/kickoff, missing ball, long gaps or disagreement. Advance each once per actual processed callback, without copying student actions into shadow sequence state.

## Diagnostic evidence and action-to-state links

- session/callback/frame IDs
- canonical elapsed time and callback/frame gaps
- match phase and agent/team identity
- packet/prediction receipt counters, monotonic time and object identities
- exact 13D float32 input
- student logits, mode, continuous steering and exact eight controls
- actual student controller submission and hidden-state hash chain
- shadow eight controls, branch, selected index/timestamp/position and attempted selection
- shadow sequence ID, before/after phase/index/start/done state
- observation selector probe separately from shadow gameplay selection
- current actual car/ball/opponent physics, scores and transition events
- native air_state, has_jumped, has_double_jumped, has_dodged and demolition fields
- previous submitted student action and link to next actually received state

Link each submitted student action to the next actually received state, retaining elapsed/frame gaps and last delivered prediction receipt. That next packet is an observed result after an action hold and intervening game physics/human play, not an exact isolated-action counterfactual. Stop/end leaves the final hold without a later observed state; report it as censored.

Store isolated raw diagnostic JSONL and metadata only, with integrity hashes and immutable summary snapshots. Mark `not_training_dataset`; create no processed training tensors, train/validation/test assignment, DAgger aggregation or training job.

## Learner-induced state distribution

Pre-register a train-only descriptive comparison. Primary reference: Orange-teacher natural training matches `pilot_01` and `v10_006`; secondary: all four training matches. Validate required artifact hashes before reading samples. No validation or test samples are read/evaluated. Do not compare the natural pilot to Phase 3 controlled-probe occupancy as if those distributions were equivalent.

- Counts/exposure by phase, masks, near/far, car-speed bands and relative geometry sectors
- Per-feature min, median, p05/p95 and p01/p99 under original normalized representation
- Fraction outside reference min/max and p01/p99 bounds
- Fixed training-reference quantile-bin occupancy and per-feature Jensen-Shannon divergence
- Both callback-count-weighted and actual elapsed-hold-time-weighted occupancy

Condition masks and phases: ball-relative features require ball-present, predicted features require prediction-valid, and Kickoff/Active are separated from nonplayable phases. Keep original normalized features; reference quantile bins are reporting only. Separate initial dt and crossed-phase hold intervals. Publish denominators and masks to prevent missing-ball zeros/replay exposure from masquerading as physical novelty.

One short match, different opponents/sessions, human actions, phases and callback delivery confound attribution. No significance or competence claim from marginal divergence alone

## Stateful shadow under actual learner control

- Verified student-only submissions before the states labeled learner-driven
- Original shadow starts sequence while student does not produce the corresponding Jump
- Subsequent actual learner-controlled callbacks show persistent original shadow sequence state
- Shadow Front dodge after student omitted preceding jump, grouped by native air_state and has_jumped
- Actual measured timing, ground/air status, ball-relative geometry and prediction selection at those disagreements

Native physical-state diagnostics use the installed RLBot fields `air_state`, `has_jumped`, `has_double_jumped`, `has_dodged`, `dodge_elapsed` and demolition status. These are diagnostic labels only, never new policy inputs. Classify shadow Front-dodge commands issued while the student remained grounded/did not jump; do not automatically call them valid expert targets.

A stateful original expert can emit committed sequence actions in a physical state where the student did not perform the prerequisite. Report such cases; do not silently reset/relabel or assume these labels are suitable for training

If rare sequence-divergence cases do not occur naturally, report them as unobserved and keep the research evidence incomplete. Do not manufacture coverage through probes or secretly run another match.

## Bounds and failure policy

- user Ctrl+C
- source/shadow selector exception
- nonfinite/malformed/unsupported model output
- missing controlled car
- observation/timing contract violation
- worker exit or 2-second timeout
- unexpected controller sender or native transport mismatch
- diagnostic integrity failure
- server disconnect

On failure stop/disconnect and record the actual last student action hold and missing submissions. No substitute teacher/neutral/scripted action is introduced. Keep existing two-second worker timeout; measure inference/IPC/end-to-end control latency and actual callback timing. A final report requires verified last student submission and clean agent/worker shutdown; diagnostic publication cannot suppress a valid policy output.

## Required preflight before live approval

- 43 protected hashes and frozen artifact/source pins match; prior phases preserved
- Shadow source/worker/decoder match validated code and student memory continuity survives phases
- No teacher/shadow PlayerInput transport exists; only student outputs can be submitted
- Natural 1v1 config: Epic, rendering optional, state-setting disabled, no receive-pause planner
- Immutable summary publication, error accounting and final student submission gating
- First callback/history/dt and exact 13D feature projection checks
- Phase4-only diagnostics paths and no dataset/test/training imports in live path
- Read-only training-reference metrics verified before any sample reads
- Bounded timeout/user-stop/worker-failure paths do not trigger teacher fallback

## Review criteria and current status

Separate technical control/transport integrity from learner-state coverage, shadow-label behavior and gameplay competence. Report action agreement/error by mode/phase, rare subsets and sequence boundaries; undefined precision/recall stays null. Use existing 1e-6 analog agreement tolerance and ±0.02 steering-sign deadband for diagnostics, not controller quantization.

Observed novelty or shadow disagreement does not establish that labels will improve a policy. No model-selection, retraining or test tuning follows. Formal pilot review must classify missing evidence and timing/queue constraints separately.

Protected hashes checked: 43; prior Phase 3 files and completed artifacts unchanged. Only this preparation script plus Markdown/JSON specification were created. No reference tensors were read, no distribution statistics computed, no launcher created and no game launched.

Approve/revise the run configuration, distribution-reference/metrics and fail-stop protocol first. Then prepare the isolated runner and non-live preflight, and present their readiness evidence before explicit live-launch review. No executable launch command is supplied at this specification-only stage.

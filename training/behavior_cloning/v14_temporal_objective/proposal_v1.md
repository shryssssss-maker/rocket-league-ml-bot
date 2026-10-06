# V14 temporal objective — proposal v1

Status: design only; awaiting objective/metric approval. No fitting, checkpoint evaluation, dataset sample reads, or live launch performed for this proposal.

## One scientific change

Use **causal core-phase weighting of mode cross-entropy**, with weight 8 on Jump, intervening release, and Front-dodge callbacks, and weight 1 elsewhere. No new head, transition loss, architecture, sampling, input, or action change.

For a chronological chunk of N actual callbacks:

`L = sum_t [w_t * CE(mode_logits_t, teacher_mode_t) + I(teacher_mode_t=Chase) * (steer_t-teacher_steer_t)^2] / N`

Normalize by callback count, not sum of weights. The Chase-only steering coefficient remains 1. This deliberately changes effective gradient emphasis, while keeping each callback exactly once per epoch. Weight 8 is a single proposed preregistered value; no weight sweep or validation-driven adjustment.

### Causal supervision tracker

Derive weights from current/prior native teacher actions and the existing ball-present mask only. Check that action-mode labels agree with the unchanged native decoder. Track across chunks; initialize idle at match boundaries only. Never feed this tracker to the GRU.

- Ball absent: weight 1 and leave the supervision tracker pending. Missing-ball Neutral is never classified as release/coast.
- Ball present, Jump: weight 8; enter Jump. Repeated Jump stays Jump. A Jump following Coast/idle starts a new core.
- Ball present, Neutral after Jump: enter Release, weight 8. Repeated Neutral in Release stays weighted.
- Ball present, Front dodge after Release: enter Dodge, weight 8. Repeated Dodge stays weighted.
- Ball present, Neutral after Dodge: enter Coast, weight 1. Repeated Coast Neutral stays unweighted.
- Ball present, Chase: enter idle, weight 1.
- Other transitions (for example idle to Dodge or Jump directly to Dodge): stop preflight with a diagnostic. Do not repair targets, invent a release, drop callbacks, or silently widen this proposal. First callback is included under the same rules.

The tracker labels the observed action core, not the teacher's private sequence phase. It has no duration heuristic, nominal-time target, or future lookahead. A missing-ball interruption does not reset model memory or tracker memory.

This increases supervision for the small initial Jump interval, rare release Neutral, and subsequent Dodge. Matching these labels at their original callback times encourages correct order and timing. It does not mathematically enforce them. In particular, it cannot prove 13D observability or live robustness.

## Preserved model and training

Same V13 Policy: Linear(13,64), ReLU, one causal GRU(64,64), Linear(64,5), four mode logits and tanh steer: **26,181 parameters**. Float32 inputs/parameters, int64 class targets, hidden state [1,1,64]. Unchanged deterministic eight-channel native decoder: throttle, steer, pitch, yaw, roll, jump, boost, handbrake. Constant yaw/roll/boost/handbrake remain zero/false.

Inputs, in order: ball_forward, ball_right, ball_up, prediction_forward, prediction_right, prediction_up, ball_distance, car_speed, ball_present, prediction_valid, prediction_horizon, prediction_first_offset, callback_dt. Same projection of frozen columns 0–12; same normalization and selector. No previous actions or private sequence features.

User approved repeating V13 fresh initialization and schedule, rather than fine-tuning its checkpoint. Seed 42 for Python/NumPy/PyTorch/CUDA; Adam lr=0.001, betas=(0.9,0.999), eps=1e-8, weight_decay=0, amsgrad=false; at most 20 epochs, patience 4. Chronological chunks of 512; fixed frozen match order; no clipping/scheduler/resampling/augmentation. Hidden state resets only at match boundaries and is detached after each training chunk, preserving V13's carried numerical state across optimizer updates. No resets at goal/replay/kickoff/missing ball.

Train: pilot_01, v10_001, v10_004, v10_006 (107,400 callbacks). Validation: pilot_02, v10_002 (47,271). No test reads. Verify only authorized train/validation raw/tensor files against the manifest before content reads; do not call V13's broad prepared()/V11 reader that hashes test artifacts too.

Preserve checkpoint selection: minimum **unweighted original V13 validation loss**, strict improvement, earliest tie; same patience. Report both weighted training/objective diagnostics and unweighted comparable validation losses. Sequence metrics determine final research acceptance, not opportunistic checkpoint selection. This isolates training objective, although unweighted selection could reject temporally better epochs; report that limitation without changing selection afterward.

Record runtime/device/dependency versions, source/config hashes, seeded initial state hash, checkpoint hashes, optimization history and deterministic settings. Seed equality alone does not guarantee bit-identical GPU execution. A later implementation preflight must verify same-seed initial parameter equality with V13's Policy. Existing V13 checkpoint is a baseline reference only, never the initialization or an output target.

## Baseline

Saved V13 validation: overall mode 96.2260%; Chase mode 99.4623%; native jump precision 83.4906%, recall 23.4748%; Front-dodge precision 76.2376%, recall 27.1127%; pure Jump mode accuracy 4.3011%; sequence-boundary mode accuracy 48.6952%; all-callback steering MAE 0.1406502; Chase steering MAE 0.1495617; exact and 1e-6-tolerant native agreement 17.9603%; unweighted loss 0.1653862.

Native jump precision/recall includes both Jump and Front dodge because both press the button. Report pure Jump mode precision/recall separately to avoid conflating these measures. Boundary accuracy keeps the existing V13 diagnostic subset (radius five callbacks); private labels may define this retrospective metric, never training weights/model inputs.

Saved aggregate results do not include the proposed ordered/timed core metric. After approval, perform **evaluation only** of the pinned V13 checkpoint on validation to establish it, with new outputs under V14. Do not refit V13. Freeze this baseline result and denominator before V14 fitting. No old artifacts are overwritten.

## Primary temporal metric, fixed before fitting

Run-length encode teacher modes on actual chronological callbacks. A complete observed core is four consecutive runs: Jump(2), Neutral release(0), Front dodge(3), Neutral coast(0). Native labels must agree with the decoder. Include only cores whose four edge callbacks and intervening callbacks have ball present, no callback dt >0.1 s, strictly increasing canonical time, and no match boundary. Interrupted, long-gap and censored cores are reported separately, not discarded invisibly or counted as successes. A completed core here means the observed start of Coast; it does not assert full 0.8-second coast completion.

For each eligible teacher core, record four **observed** run-start times: Jump, release, Dodge, Coast. Extract the same four-run patterns from chronological model predictions. Evaluate teacher cores in order; pair to the unused predicted core with closest Jump onset within 2/120 s (earliest onset breaks ties). Missing match is failure. Success requires all four edge-time errors <=2/120 s + 1e-6 s, exact four-run order, and at least one actual native jump=false release callback. No interpolation or nominal-duration repair. A half-second held jump will fail the release/Dodge/Coast edge checks.

Primary score: successfully timed, ordered cores / all eligible teacher cores. Report numerator/denominator, unmatched/extra predicted cores, per-match results, ordering-only rate, signed/absolute onset and edge errors, observed/predicted Jump/release/Dodge duration errors (median/p95/max). Timing comparisons are retrospective metrics only, never future supervision. Unmatched predictions anywhere in validation contribute to extra-core rate; no best-window search around a reference event.

At least 20 eligible validation cores are required; otherwise evidence is inconclusive. If baseline score b >0.85, this improvement gate has insufficient headroom and must be reviewed **before fitting**, not relaxed after V14 results.

**Primary success:** V14 timed-core score >= max(0.25, b+0.15), where b is the newly measured pinned V13 validation baseline. This is an absolute 15 percentage-point improvement, with a 25% minimum. Extra complete predicted cores per validation minute must not exceed the baseline rate by more than 0.5/minute. These are provisional research acceptance thresholds, not live-competence criteria.

**Chase regression guard:** Chase accuracy >= saved baseline minus 0.5 percentage points (>=98.9623%); Chase steering MAE <= baseline +0.02 (<=0.1695617). Overall mode accuracy >= saved baseline minus 1 percentage point (>=95.2260%). All must pass together with the primary criterion. Improved aggregate accuracy without temporal improvement is unsuccessful for this purpose.

Also report all requested metrics: total/per-channel loss, native throttle/steer MAE, pitch error/accuracy, jump precision/recall/accuracy, Front-dodge precision/recall, pure Jump accuracy, native exact/tolerant agreement, per-mode and Jump/Dodge subsets, sequence boundaries, missing-ball and >0.1 s callback gaps. Report precision tradeoffs and interrupted-core behavior even if mandatory gates pass. No teacher-forced vs student-forced action-feedback robustness claim: 13D has no previous-action inputs; recorded physics/time remain fixed.

## Limitations and next step

This tests objective sensitivity on teacher-controlled recorded physics only. It does not resolve recovery representation, native maneuver prerequisites on learner-induced states, DAgger label suitability, or closed-loop improvement. Train/validation are match-disjoint but human_a overlaps; human_b's two train matches share a session. No test access or new collection.

Validation results, temporal scores, regression verdict and final A/B/C/D research recommendation are **not available: experiment not run**. The one next step is review/approve this exact objective and metric specification, then implement one offline experiment. No live launch, DAgger, PPO, or deployment follows automatically.

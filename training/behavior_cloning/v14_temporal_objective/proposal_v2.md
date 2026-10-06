# V14 boundary-aware temporal objective — proposal v2

**Design only; awaiting review. No implementation, fitting, model evaluation, dataset sample reads, or live launch.** This replaces v1 as the proposed main experiment. `proposal_v1.md/json` remain unchanged as a documented 8× core-phase-weighting alternative/ablation; that alternative is not scheduled to run.

## Exact single objective

Keep V13's original loss and add one differentiable, boundary-balanced binary cross-entropy term. There is no class weighting in the original mode loss, no second auxiliary term, and no new output head.

Let y_t be the existing teacher mode (N=0, C=1, J=2, D=3), z_t the four model logits, p_t=softmax(z_t), s_t the unchanged tanh steering output, and s*_t the teacher steer. For a chronological training chunk containing n callbacks:

`L_original = (1/n) sum_t [CE(z_t,y_t) + I(y_t=C)*(s_t-s*_t)^2]`

Define four soft adjacent-boundary scores and corresponding binary targets:

| Boundary k | Model score q[t,k] | Teacher target b[t,k] |
|---|---|---|
| Jump onset | (1-p[t-1,J])*p[t,J] | I(y[t-1] != J and y[t] = J) |
| Jump → release | p[t-1,J]*p[t,N] | I(y[t-1] = J and y[t] = N) |
| release → Front dodge | p[t-1,N]*p[t,D] | I(y[t-1] = N and y[t] = D) |
| Front dodge → coast | p[t-1,D]*p[t,N] | I(y[t-1] = D and y[t] = N) |

These are differentiable **transition scores**, not a claim of independent callbacks or a calibrated joint probability. Neutral release/coast meanings are operational adjacent-action definitions; no hidden phase inference or teacher-private state is used. A Neutral→Dodge edge after an interruption is labeled as the observed adjacent edge, not certified as a physically valid maneuver. Jump onset includes any non-Jump predecessor; no unsupported teacher transition is fabricated or repaired.

Let E be eligible adjacent pairs whose current callback is in this chunk. For each k, let P_k={t in E:b[t,k]=1}, Z_k={t in E:b[t,k]=0}. Set qbar=clamp(q,epsilon,1-epsilon), epsilon=1e-7, and:

`A_k = mean_{t in P_k}[-log(qbar[t,k])]` when P_k exists.

`B_k = mean_{t in Z_k}[-log(1-qbar[t,k])]` when Z_k exists.

`L_k = (A_k+B_k)/2` if both sets exist; `L_k=A_k` or `B_k` if only one exists. If E is empty, the boundary loss is exactly zero.

`L_boundary = (1/4) sum_{k=1..4} L_k`

**`L_V14 = L_original + 0.1 * L_boundary`**

The sole added mechanism is boundary-balanced transition BCE. Positive/negative normalization is part of that term, not dataset resampling or a second loss. Every actual callback remains in the original loss once per epoch; every eligible adjacent pair is assigned once to its current callback's chunk. No temporal smoothing, duration target, window search, nominal-time repair, or future label is used in supervision.

### Why this targets the failure

Positive edge loss rewards the correct preceding/current modes at the teacher's observed boundary callback. A missing or late Jump, skipped release, or collapsed Dodge increases the relevant positive loss. Negative edge loss penalizes misplaced/extra boundaries, including a sequence shifted to later callbacks. Equal boundary-type averaging prevents common onset evidence from obscuring release/Dodge edges; positive/negative averaging prevents non-boundaries from overwhelming rare events.

It encourages ordering through correct adjacent edges, but does not guarantee a complete maneuver or physical dodge. Local boundaries can still be individually correct while a whole sequence is wrong; the existing timed-core metric remains the primary acceptance criterion. No private teacher phase, sequence ID, or future callback enters the GRU or this term.

## λ and numerical stability, fixed before fitting

**λ=0.1 once; no sweep or adjustment after baseline/new results.** At uniform mode probabilities, the three product scores are 1/16: positive BCE≈2.773 and negative BCE≈0.0645; balanced loss≈1.419. Onset score is 3/16: positive≈1.674, negative≈0.208; balanced≈0.941. With both classes present for all four types, L_boundary≈1.300, so its initial contribution≈0.130 versus original mode CE≈1.386, before steering error. This is an analytic scale justification, not a claim that gradient magnitudes or all chunks match those values. Rare positive edges receive focused gradients while the original loss remains intact.

- Compute softmax, scores and reductions in float32, with no new mixed precision. Use `log1p(-qbar)` for negative BCE and `log(qbar)` for positive BCE.
- Clamp only the loss's transition score; never alter probabilities used for argmax/decoder, labels, input, steering, timestamps, or metrics. Report clamp counts. Clamp saturation can zero boundary gradients at extreme scores; disclose this limitation rather than suppressing it. Original CE continues to supply mode gradients.
- Do not divide by absent sets. Empty eligible chunks add zero; record their count and eligible/positive/negative counts by edge type.
- Assert finite logits, probabilities, both loss components and gradients; on nonfinite values stop, with no skipped update, clipping, label repair, or optimizer substitution.
- The BCE is numerically bounded by approximately -log(epsilon)=16.118 per score; the added term is therefore at most approximately 1.612 per chunk (float32 endpoint rounding reported). This is not a gradient norm bound.
- Positive counts vary between chunks; report boundary loss and support counts. Do not change chunking, weighting or λ to compensate retrospectively.

## Eligibility, gaps and chunk edges

An adjacent pair is eligible only if: both callbacks belong to the same match; both actual frozen ball-present masks are true; canonical elapsed time strictly increases; and actual callback dt is >0 and <=0.1 s. Verify stored dt against canonical-time differences under the frozen contract before fitting; inconsistencies stop preflight. No additional prediction-valid restriction is introduced. Phase changes alone do not exclude a pair or reset anything.

- First match callback: original loss included, no adjacent boundary term. Initial prior probability/label absent; preserve original dt input.
- Missing ball on either side: exclude only that adjacent pair. Never bridge over missing callbacks to find an earlier present ball. Missing-ball rows remain unchanged in original loss and validation.
- Gap >0.1 s, zero/nonincreasing time: no inferred boundary across the gap; report separately. No time correction, extrapolation, target replacement, or GRU reset.
- Match boundary: no pair across matches; reset GRU and adjacent-pair cache there only.
- Chunk boundary: include the first callback's pair using its actual preceding callback label/mask/time and **detached cached previous mode probabilities** from the immediately preceding training chunk. This preserves V13's truncated gradient graph and optimizer schedule. Do not replay the preceding callback or recompute its probability under updated parameters. The preceding probability was obtained before that chunk's optimizer update, as was V13's carried hidden state. Report these cross-chunk pairs separately. Within a chunk, both adjacent probabilities retain gradients.

This is a causal loss at t using t-1 and t. Caches are training bookkeeping, not inputs or private teacher memory. The GRU still runs through every callback and never resets on goal, replay, kickoff, missing ball or gaps.

## Everything else unchanged

V13 architecture: Linear(13,64), ReLU, one unidirectional GRU(64,64), Linear(64,5): **26,181 parameters**; four logits plus tanh steer; identical native decoder. Inputs float32 [callbacks,13], hidden [1,1,64], teacher class targets int64. Exact features: ball_forward, ball_right, ball_up, prediction_forward, prediction_right, prediction_up, ball_distance, car_speed, ball_present, prediction_valid, prediction_horizon, prediction_first_offset, callback_dt. Same frozen column projection/scales, predictions and action representation.

Repeat V13 fresh initialization, seed 42 for Python/NumPy/PyTorch/CUDA. Adam lr=0.001, betas=(0.9,0.999), eps=1e-8, weight_decay=0, amsgrad=false; at most 20 epochs, patience 4; chronological 512-callback chunks, fixed match order, numerical hidden continuity and detach after each training chunk. No clipping/scheduler/augmentation/resampling. Compare seeded initial parameter hashes with the existing Policy before fitting; record runtime/device/deterministic settings, since identical seeds do not guarantee identical GPU execution.

Train IDs: pilot_01, v10_001, v10_004, v10_006 (107,400 callbacks). Validation IDs: pilot_02, v10_002 (47,271). No test artifact/sample reads; do not invoke the broad V13 prepared()/V11 reader. Verify only authorized manifest-listed train/validation artifacts before content reads. Frozen data and old checkpoint remain read-only; new artifacts belong to V14.

Checkpoint selection and early stopping remain **minimum original unweighted V13 validation loss**, strict improvement, earliest tie. Do not select by weighted loss or temporal metric. Report the boundary component separately. The unchanged selector can prefer a less temporally successful checkpoint; disclose that outcome without retrospective selection changes.

## Baseline and unchanged timed-core metric

Reuse saved V13 validation results: mode accuracy 96.2260%; Chase accuracy 99.4623%; native jump precision/recall 83.4906%/23.4748%; Front-dodge precision/recall 76.2376%/27.1127%; pure Jump accuracy 4.3011%; sequence-boundary accuracy 48.6952%; steering MAE 0.1406502 overall and 0.1495617 on Chase; exact and 1e-6-tolerant native agreement 17.9603%; original loss 0.1653862. Native jump includes both Jump and Dodge; report pure-Jump precision/recall separately.

The timed-core metric remains as specified in v1:

1. Run-length encode teacher modes; eligible complete observed core is four consecutive runs J→N→D→N, with native controls consistent. Its four edge callbacks/intervening callbacks must have ball present, no dt>0.1 s, strictly increasing canonical time, and no match boundary. A core is complete at Coast onset, not full coast completion. Report interrupted/gap/censored cores separately.
2. Extract the same patterns from model predictions. Process teacher cores chronologically; pair to the unused predicted core with closest Jump onset within 2/120 s, tie earliest. Pairing cannot cross matches. Missing match fails. Success requires exact order, at least one native jump=false release callback, and every edge-time error <=2/120+1e-6 s. Compare observed times, never nominal durations/interpolation. Count all unmatched predicted cores as extra predictions.
3. Report successful/eligible cores, per-match results, ordering-only rate, signed/absolute edge errors, Jump/release/Dodge duration errors (median/p95/max), missed/extra cores, and gap/missing-ball strata. Retrospective evaluation may inspect completed runs; supervision remains adjacent and causal.

Saved aggregate V13 results lack this metric. After design approval, evaluate the pinned V13 checkpoint on validation only, without refitting; save/freeze new baseline evidence under V14 before fitting. If fewer than 20 eligible cores or baseline b>0.85, stop for review before fitting. Do not change gates after seeing V14 outcomes.

## Preregistered decision rule and regression guards

Temporal success requires timed-core score >=max(0.25,b+0.15), and extra complete predicted cores/minute <=baseline+0.5/minute. Same provisional research thresholds as v1.

All unchanged regression guards must also hold: Chase accuracy >=0.9896232223028739; Chase steering MAE <=0.1695616946664056; overall mode accuracy >=0.9522601595058281.

- Temporal gate plus all guards pass: offline objective success with scope; recommend seeking approval for one small live validation, never launch automatically.
- Temporal gate passes but a regression guard fails: partial temporal improvement with unacceptable tradeoff; no deployment. Recommend review for one further scoped offline experiment, not opportunistic checkpoint selection.
- Temporal gate fails, regardless of aggregate accuracy: unsuccessful for its intended purpose; recommend investigation of representation, without changing it here.
- Integrity/numerical failure, insufficient baseline cores, or missing metrics: incomplete/invalid experiment, no success claim; report the blocker before further work.

Report all requested aggregate/per-mode/native-channel losses/errors, Jump/Dodge subsets, boundary accuracy, missing-ball/gap metrics and precision tradeoffs. Existing private diagnostics may define the unchanged retrospective boundary-accuracy subset only; they do not build the boundary training term. No action-feedback robustness claim, physical dodge claim, DAgger benefit or closed-loop competence follows from offline improvement.

## Status and limitations

Teacher-controlled recorded physics remain fixed. Adjacent scores cannot prove global sequence consistency, physical maneuver prerequisites or 13D sufficiency. Excluded gap/missing-ball pairs narrow boundary supervision, but those callbacks remain model inputs/base-loss targets. Match-disjoint splits retain human_a overlap; human_b's two train matches share a session.

All new validation scores and pass/fail decisions are **not run**. No test reads, DAgger, model fitting, live game, teacher modification or deployment is authorized by this proposal. **One next step: review the exact v2 formula, λ, eligibility/chunk handling and unchanged gates before implementation.**

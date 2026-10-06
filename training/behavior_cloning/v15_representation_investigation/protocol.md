# V15 representation-sufficiency diagnostic protocol — 2026-10-06

Investigation only. No policy fitting/checkpoint loading, test samples, live launch, DAgger, or contract change. Data analysis runs only through the user-run command after source audit and artifact verification. Outputs are new diagnostics, never training tensors.

## Phase 0: audited sources and availability

The official 13D remains the exact V13 projection of frozen V10 columns 0–12. `training/v2_observation_test/observation_contract.py:build,basis` defines car-native forward/right/up dot products, without team inversion. Relative XYZ and 3D distance divide by 6000; speed magnitude by 2300; masks are 0/1; selected horizon by 2 seconds; first-slice offset by 1/120 second; callback dt by 1/60 second. Missing ball zeros features 0–6 and 8–11; car speed/dt remain. No clipping/padding/interpolation. Prior callback actions are not 13D inputs.

`python-example-original/src/util/ball_prediction_analysis.py:find_slice_at_time` uses `int((elapsed+2-first_slice_time)*120)`, returning None outside range and preserving empty-slice IndexError. Original teacher invokes prediction only at 3D distance >1500 when no pending sequence; the shared observation diagnostic may invoke a selector probe independently. Actual received packets/canonical elapsed and actual delivered RLBot prediction produced frozen observations; no alternate provider.

Native authority is the installed `python-example/venv/Lib/site-packages/rlbot_flatbuffers/__init__.pyi`, version 0.19.0, used by rlbot 2.0.0b55. `Physics` starts at line 2409; `Rotator` at 2132; `PlayerInfo` at 5580; `AirState` at 47; `ControllerState` at 1994. `training/behavior_cloning/recorder.py` writes only car physics, not all PlayerInfo fields. `common.py:physics` records position/velocity/angular_velocity/rotation_pyr. The later V13 full-match `runner/live_agent.py:player` records additional native flags, but those are absent from frozen demonstration training/validation records.

| Field | Actual source / units | Frozen V10 / reconstruction | Live student availability |
|---|---|---|---|
| Euler orientation | packet.players[index].physics.rotation.pitch/yaw/roll; radians, installed Rotator documentation | car.rotation_pyr, exactly retained native numbers | Yes |
| Forward/right/up basis | derived by the existing basis(rotation_pyr) formula; dimensionless, car-local axes in world coordinates | Exactly reproducible by the same formula; numerical trig precision disclosed | Yes; derived, not a native basis field |
| world-up alignment | up.z = cos(pitch)*cos(roll); dimensionless [-1,1] | Exactly reconstructible from recorded angles using the contract formula | Yes; derived |
| Angular velocity | physics.angular_velocity.x/y/z; world-axis components, radians/s | car.angular_velocity retained | Yes |
| World position/altitude | physics.location.x/y/z, UU | car.position retained | Yes |
| Linear velocity | physics.velocity.x/y/z, UU/s; 13D only keeps magnitude | car.velocity retained | Yes |
| Surface/air state | PlayerInfo.air_state enum: OnGround, Jumping, DoubleJumping, Dodging, InAir | NOT recorded in V10; height/orientation cannot reconstruct surface contact exactly | Yes |
| Jump/double jump/dodge | PlayerInfo.has_jumped/has_double_jumped/has_dodged; booleans with reset semantics described in stub | NOT recorded; cannot derive reliably from commands/physics | Yes |
| Dodge timing/direction | PlayerInfo.dodge_timeout, dodge_elapsed (seconds), dodge_dir (unit 2D direction) | NOT recorded | Yes |
| Current/latest native input | PlayerInfo.last_input, native ControllerState channels | NOT recorded; cannot equate to previous submission | Yes; packet timing relative to submissions still needs future validation |
| Previous submitted controls | recorder.previous_submission.controls/mode/callback/T, eight native channels; input fields verified against preceding row | Recorded; exactly previous confirmed submission, not proof of successful engine maneuver | Student can track its own confirmed submissions; teacher history differs from student history |
| Boost | PlayerInfo.boost, 0–100 | NOT recorded; boost button=false does not reconstruct amount | Yes |
| Supersonic/demolition/hitbox/touch | PlayerInfo fields is_supersonic, demolished_timeout, hitbox, latest_touch | Not fully retained in V10; no proxy promoted to native truth | Available in SDK |

Physical unit/coordinate reference: [RLBot v5 useful game values](https://wiki.rlbot.org/v5/botmaking/useful-game-values/) documents UU, world-up Z, car speed and angular velocity (5.5 rad/s). [RLBot's coordinate rotation code](https://github.com/RLBot/RLBot/blob/master/src/main/python/rlbot/utils/structures/game_data_struct.py) explicitly documents global angular-velocity axes; it is cross-version corroboration, not proof of installed v5 binary provenance. No installed build revision is inferred from those links. The installed AirState stub has an inconsistent Jumping duration/tick parenthetical; do not use it as a timing contract.

## Fixed data and query construction

Installed runtime class descriptors are also verified without a connection or instantiated packet in `native_sdk_audit.json`. `AirState.OnGround` describes all-wheel surface contact/wall adhesion, not necessarily an upright car on the arena floor. World-up alignment is never substituted for this flag. `last_input` reports native controller input, not a guarantee that its intended maneuver executed. Source teacher behavior has no explicit inversion/recovery strategy: orientation influences chasing, speed triggers flips, and independently pending sequence state controls continuation. Consequently teacher-mode predictability alone does not measure optimal recovery behavior.

Train: pilot_01, v10_001, v10_004, v10_006. Validation: pilot_02, v10_002. Verify manifest SHA256 and every actually used artifact before content reads; no test artifact hashing/opening. Preserve all 43 hashes and old V13/V14 artifacts.

Validation query endpoints are a fixed union: every 128th callback; all pure Jump and Front-dodge rows; up to 256 evenly spaced sequence-boundary callbacks per match (existing radius-five diagnostic subset); and up to 64 evenly spaced rows per match in each kickoff, inverted, tilted, long-Neutral-Chase, corner/back-region and missing-ball subset. No future observation input. The latter selection is diagnostic stratification, not dataset resampling or policy fitting. Report overall performance on the every-128 subset separately from the targeted union; neither is falsely called an exact population statistic.

All history comparisons use common endpoints with at least 16 actual callbacks available. Training candidate endpoints obey the same rule. No match crossing or padding; first 15 endpoints excluded from this diagnostic only and counted. Histories H=1,2,4,8,16 contain ordered past/current observations ending at t, never t+1. Missing-ball/gap rows remain in histories, and actual span seconds are reported. Histories are finite diagnostics, not a test of the GRU's unlimited causal state.

Previously documented V13 query anchors are callbacks 456,4279,9815,19942 (kickoff), 579,4380,8300,13200,13500,16500,20250 (recovery), 10335 (other). Their full-match records are diagnostic queries only; expert modes are explicitly stateful shadow labels on learner-induced physics, not approved training targets. No aggregation. Never substitute teacher-private phases for model inputs.

## Neighbor and nonparametric probes

Use exhaustive float64 nearest-neighbor search against all authorized training endpoints, in bounded query/training blocks. Distance for H is RMS of the 13H unchanged normalized feature differences. Deterministic ties use training-match order and callback index. No whitening, learned projection, tuned distance threshold or future features. Report exact feature equality separately from approximate closeness; near neighbors do not prove an information-theoretic alias. Recompute returned distances directly to expose numerical cancellation.

For k=1,8,32,128 report neighbor mode probabilities, target-label agreement, majority purity/entropy, distinct matches, native jump/pitch frequencies and continuous steering distribution/error. Fixed distance bins: <=0.005, (0.005,0.02], (0.02,0.05], (0.05,0.1], >0.1. These are descriptive normalized-space bins, not validated reliability thresholds. Material native differences: different mode, jump/pitch difference, or Chase steer difference >0.2. Report physical differences in up.z, angular velocity, position, speed and history span. Neighbor callbacks are correlated; no independence/significance claim.

The single lightweight probe is uniform **32-neighbor majority classification** (argmax ties choose lowest mode index) and mean teacher Chase steering among Chase neighbors. No model fitting or architecture sweep. k=1/8/128 are ambiguity diagnostics only; no k is selected after results. Report native jump recall separately from pure Jump recall/accuracy, Front-dodge recall, Chase/overall accuracy, boundary/subset results, and local purity. Train rows serve only as a reference library; validation labels never enter distances or vote fitting.

## Candidate groups, individually

- A: one scalar `world_up_alignment = cos(pitch)*cos(roll)`, unscaled. Minimal upright/inversion descriptor; does not encode full yaw or contact state.
- B: three angular components `angular_velocity_xyz / 5.5`, no clipping; world axes retained. Recorded quantities rather than an invented native maneuver state.
- C: unavailable from frozen train/validation. No fabricated grounded/jump flags or statistical probe. Later V13 flags are descriptive query evidence only.
- D: four one-hot **previous submitted** modes plus previous steer, exactly the frozen old columns 13–17. This explicitly tests a teacher-forced context association, not an executed-engine-input contract or student-feedback robustness. Cannot be automatically recommended on high purity alone after V12's known shortcut failure.
- E: recorded world XYZ /6000, no team inversion. Only probe it if the baseline finds at least 20 close (RMS<=0.02) mismatched neighbors with >=1000 UU position difference in kickoff/recovery/corner/back subsets. Position is not added merely because it exists. If no evidence meets this predeclared condition, report E not justified and do not run it.
- No Group F or feature combinations unless a separate review authorizes them.

Compare each available A/B/D and conditionally E as 13D+current group, plus the same group at H=16 to compare added state against causal history. Use squared distance `RMS(history13_difference)^2 + RMS(current_group_difference)^2`: equal group-block coefficient 1, fixed before results. This avoids declaring new-space distances directly comparable to 13D distances; retain base13 distance to each returned neighbor. Do not change the metric to improve a result. Group dimensions/scales can affect neighbor ranking; report this sensitivity limitation rather than tune it.

Candidate values are rounded to float32 after their specified transform, then promoted to float64 for statistical distance arithmetic. Thus no candidate exploits extra precision that would be absent from a future float32 policy input. A future candidate ordering, if justified, is original frozen13 followed only by that group's listed scalar/XYZ/features; this investigation creates no official contract.

## Predeclared evidence interpretation

History improvement: H16 versus H1 gains >=0.10 boundary accuracy and >=0.05 pure-Jump recall and >=0.05 Front-dodge recall, without >0.01 loss in representative overall accuracy. Candidate physical feature evidence: same gains over H16 baseline at H16, plus >=0.10 improvement in mean true-label neighbor probability among close ambiguous maneuver/recovery queries; at least 30 boundary rows and 20 close ambiguous targeted rows; gains positive in each validation match. No thresholds revised after results. These are exploratory research criteria, not policy-success gates or statistical proof.

Classification A: history helps and no physical group meets criteria (representation appears adequate over tested context; modeling/horizon merits study). B: a physical group meets criteria while history alone does not. C: both help. D: neither or support inadequate/incomparable. No automatic claim that the representation hypothesis is rejected when criteria fail.

If multiple physical groups qualify, recommend the smallest dimension; tie priority A then B then E, not a validation score sweep. D-only association stays inconclusive for representation repair because execution/teacher-forcing semantics are unresolved. Any proposed future feature ordering is original13 followed by the qualifying group only, creating a NEW candidate contract, never changing the official 13D here.

H16 commonly covers only about a quarter-second, less than the full original flip/coast sequence. Even a B result is limited evidence for an added variable over this finite context, not proof that the existing GRU's longer history cannot resolve the ambiguity. A and D do not prove that no feature could help.

One next recommendation follows classification; no training/live/test/DAgger launches automatically. Final reports include all 15 requested sections, unavailable fields, unmatched/OOD queries and known leakage/correlation limits. If results remain inconclusive, recommend one focused observability investigation rather than manufacturing a feature addition.

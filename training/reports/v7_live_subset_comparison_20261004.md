# Bounded V7 live snapshot comparison

**Comparison complete; V7 pass is not declared. User review required.**

Only the unchanged-policy 115 verified fixtures / 345 eligible snapshots were used. The four boost-failed fixtures and corner_04 were excluded. Capture artifacts and all 43 protected hashes remained unchanged.

The captured packet fields and all delivered prediction slice values were restored without numerical changes. Both paths received the same shared replay objects and actual canonical elapsed time; mutation checks passed. This is replay of recorded live data, not a synthetic forecast. Original raw wire packets and unused packet fields were not saved, so full packet fidelity is not claimed.

Reference: unchanged original find_slice_at_time; selected index observed from the returned slice object. Candidate: isolated exact-expression selector wrapper plus the unchanged V2 live_adapter. No production adapter selector previously existed; this diagnostic wrapper is not deployed.

Reference 18D oracle independently writes the declared schema using original Orientation/relative_location/Vec3. Features are compared after their declared float32 cast. Original Vec3/basis intermediates and the candidate builder can differ in floating-point precision; all differences are recorded, never corrected.

Induced steering compares original steer_toward_target with steering computed from candidate normalized float32 local coordinates under the same Chase formula. No student/model runs. Chase is conditional on no pending sequence and uses the actual current-ball distance; this does not establish teacher gameplay decisions or temporal state.

Previous-action fields come only from the actual prior diagnostic control submission. Every retained prior action is Neutral, so other modes and teacher memory are outside this comparison.

## Overall numerical differences

| Quantity | Maximum absolute difference | Median absolute difference |
|---|---:|---:|
| index | 0 | 0 |
| timestamp | 0 | 0 |
| target_position | 0 | 0 |
| steering | 5.35754686171e-08 | 0 |

Position is Euclidean error in UU; timestamps are seconds; steering is controller magnitude. Feature tolerance is the existing V2 1e-5; masks/previous-action values and selector outputs require exact agreement. The prior provisional V7 0.1 steering limit is shown only as a diagnostic threshold, not automatic gate approval.

## Scenario groups

| Group | Fixtures | Snapshots | Exceptions | Failed snapshots | Sign disagreements | Max time error (s) | Max target error (UU) | Max steer error | Median steer error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| car_near_path | 12 | 36 | 0 | 0 | 0 | 0 | 0 | 2.03567823e-08 | 0 |
| ceiling | 12 | 36 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| corner | 10 | 30 | 0 | 0 | 0 | 0 | 0 | 2.83737238e-08 | 0 |
| floor | 12 | 36 | 0 | 0 | 0 | 0 | 0 | 5.17548066e-08 | 0 |
| free_flight | 12 | 36 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| goal_mouth | 11 | 33 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| rolling | 12 | 36 | 0 | 0 | 0 | 0 | 0 | 3.77957666e-08 | 0 |
| spin | 12 | 36 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| stationary | 11 | 33 | 0 | 0 | 0 | 0 | 0 | 5.35754686e-08 | 0 |
| wall | 11 | 33 | 0 | 0 | 0 | 0 | 0 | 2.23321693e-08 | 0 |

Every group's maximum/median for all metrics and all 18 features, plus every individual snapshot, is in the accompanying JSON report.

## Each 18D feature

| Index | Feature | Maximum difference | Median difference |
|---:|---|---:|---:|
| 0 | ball_forward | 0 | 0 |
| 1 | ball_right | 0 | 0 |
| 2 | ball_up | 0 | 0 |
| 3 | prediction_forward | 0 | 0 |
| 4 | prediction_right | 0 | 0 |
| 5 | prediction_up | 0 | 0 |
| 6 | ball_distance | 0 | 0 |
| 7 | car_speed | 0 | 0 |
| 8 | ball_present | 0 | 0 |
| 9 | prediction_valid | 0 | 0 |
| 10 | prediction_horizon | 0 | 0 |
| 11 | prediction_first_offset | 0 | 0 |
| 12 | callback_dt | 0 | 0 |
| 13 | previous_neutral | 0 | 0 |
| 14 | previous_chase | 0 | 0 |
| 15 | previous_jump | 0 | 0 |
| 16 | previous_dodge | 0 | 0 |
| 17 | previous_steer | 0 | 0 |

## All failures and steering-sign disagreements

Exceptions: 0. Failed snapshots: 0. Sign disagreements outside ±0.02: 0.


## Limits of the evidence

- All eligible snapshots have ball present and valid prediction; missing/invalid masks not exercised here.
- Previous action is Neutral only; Chase/Jump/Front-dodge history and recurrent memory are not exercised.
- Human controls may be nonneutral; no fixed-input rollout, collision forecast accuracy or temporal-teacher equivalence claim.
- No live provider replacement, generated forecasts, prediction timestamp correction, production integration or ML.

No V7 pass declaration, recapture, training or deployment followed this comparison.

## Exact sources

- Original selector: python-example-original/src/util/ball_prediction_analysis.py:find_slice_at_time.
- Original geometry/steering: src/util/orientation.py:Orientation/relative_location; src/util/vec.py:Vec3; src/util/drive.py:steer_toward_target.
- Candidate schema: training/v2_observation_test/observation_contract.py:FEATURES/build/live_adapter.
- Diagnostic wrapper and oracle: training/v7_prediction_test/compare_live_subset.py:candidate_select/oracle_features/compare.
- Input capture hashes, dependency versions and exact code hashes are included in JSON.

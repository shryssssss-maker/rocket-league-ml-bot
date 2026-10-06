# Phase 3 corrected rerun contract review

**Recommended result: pass_with_scope_awaiting_review. Formal gate approval remains with the user.**

## 1. Verified teacher/shadow contract

1,453 unique actual processed callbacks and 1,453 raw lines; no duplicate callbacks. Recomputed teacher/shadow failures: 0; maximum analog difference: 0.0. Native eight-channel outputs, buttons, sequence state/index/done/start times, branches and live selector index/timestamp/position agreed. Packet/prediction identities and canonical elapsed time matched, with independent sequence memory.

1,453 original-teacher controller submissions verified. Missing submissions: []; transport disagreements: []; final callback 1453 submitted and verified: True. No student/shadow control reached the transport. Student GRU memory had one initial reset and no hidden-hash discontinuities.

## 2. Actual observed student disagreements

Student/teacher modes differed on 99 callbacks. Teacher sequence starts: 3. Student missed 3 starts at callbacks [465, 611, 668]; each predicted Chase instead of teacher Jump. There were 4 teacher Front-dodge callbacks following a student prediction with jump=False. No student output was forced or changed.

## 3. Independent shadow continuation / off-policy scope

Actual shadow continuation after student action disagreements was observed on teacher-controlled physics. No learner-induced off-policy physical-state continuation or student-controlled gameplay was tested.

913 pending-sequence callbacks followed the observed missed starts. 2 sequences completed; sequence 3 persisted across goal, replay, countdown and kickoff. Same pending transition sequence IDs: [3]. Student disagreements did not overwrite/reset shadow memory.

## 4. Required actual coverage

| Category | Unique callbacks |
|---|---:|
| neutral_to_chase | 1 |
| chase_to_jump | 2 |
| release | 12 |
| front_dodge | 18 |
| coast | 122 |
| completion | 2 |
| missing_ball_pending | 753 |
| goal_pending | 197 |
| replay_pending | 556 |
| kickoff_pending | 97 |
| long_gap | 1 |
| near_ball | 74 |
| far_ball | 466 |
| live_selector | 466 |
| student_missed_sequence_start | 3 |
| student_missed_prior_jump | 4 |
| shadow_continues_after_student_divergence | 913 |

Unobserved required categories: none. Probe requests alone were not counted. Learner-controlled physical-state behavior remains outside this gate and untested.

## 5. Diagnostic completion and delivery limitations

Diagnostic errors: 0; publication errors: 0; duplicates: {}; hidden-chain errors: []. Agent/worker shutdown confirmed. The v1 summary-publication and duplicate-accounting failure did not recur.

Intentional receive pauses: 3. Callback dt: `{'count': 1452, 'minimum': 0.008333206176757812, 'median': 0.016666412353515625, 'p95': 0.016668319702148438, 'maximum': 4.17500114440918}`. Frame gaps: `{'count': 1452, 'minimum': 1, 'median': 2.0, 'p95': 2, 'maximum': 501}`. CPU worker IPC latency: `{'count': 1453, 'minimum': 0.0005739999905927107, 'median': 0.0008088000031420961, 'p95': 0.0011595999967539683, 'maximum': 0.004013099998701364}`.

User terminal reports 1/10/100 missed-message warning thresholds; total losses cannot be reconstructed from these thresholds.
Only actual processed callbacks are covered. No lossless-delivery or undelivered-physics-tick equivalence is claimed. Instrumentation, IPC, filesystem writes and intentional pauses can influence callback delivery. Native predictions may be reused between callbacks; no timestamps, slices or source packets were changed.

## 6. Integrity and review decision

All 43 protected source hashes, pinned frozen manifest/checkpoint/projection/V12 anchors, capture-code/config hashes and saved session-file hashes passed verification. The capture reported full immutable-artifact preservation, and this review verified its own before/after immutable snapshot. Original bots, previous runs, V10/V12/V13 completed artifacts and this capture remain byte-for-byte unchanged.

No V10 test evaluation, student control, DAgger training dataset/aggregation, fitting or PPO occurred. Only a new read-only review script and this review folder were created.

Submit Phase 3 for explicit review. Stop; no Phase 4, learner controls, DAgger collection/aggregation or training.

Original session: `C:\Users\shreyas\Desktop\model wars\training\behavior_cloning\v13_dagger\phase3_shadow_v2\sessions\20261005_044629_296920`. JSON report includes every coverage example, disagreement, receipt/latency statistic and input SHA256.

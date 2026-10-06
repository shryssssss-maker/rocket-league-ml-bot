# V13 13D GRU bootstrap

Status: **phase2_bootstrap_complete_awaiting_review**.

Checkpoint: `C:\Users\shreyas\Desktop\model wars\training\behavior_cloning\v13_dagger\phase2_bootstrap_v1\B_best.pt`; SHA256 `579fa26010140514d2e9b7f0e2871bee569c594b7da8648e31ee1489007cd5b7`; best epoch 20; 26,181 parameters.

Train/validation only; no test evaluation. Exact retained 13 features, identical scaling/masks/dt. Mode/continuous steering head and V12 training family/settings preserved. No private state inputs, balancing, resampling or new normalization.

| Validation condition | Loss | Mode accuracy | Steer MAE | Jump precision | Jump recall |
|---|---:|---:|---:|---:|---:|
| teacher_forced_13d | 0.165386 | 0.962260 | 0.140650 | 0.8349056603773585 | 0.23474801061007958 |
| student_forced_13d_no_action_feedback | 0.165386 | 0.962260 | 0.140650 | 0.8349056603773585 | 0.23474801061007958 |

## Interpretation

Removing previous-action inputs makes both conditions identical by construction. Differences from chunked versus stepwise floating-point evaluation are numerical only; similarity is not closed-loop robustness.

Numerical comparison: `{"input_bits_identical": true, "max_logit_difference": 2.9802322387695312e-06, "max_steering_difference": 8.009374141693115e-07, "mode_disagreements": 0}`.

The callback-by-callback pass tracks its own predicted action but does not feed it into any feature. Physics, prediction and timing remain recorded. No real-game robustness or learner-induced state coverage is established.

Full per-channel/action/phase/boundary/missing-ball/long-gap evidence is in report.json. Private labels are retrospective metric groupings only.

Original V10/V12/source contracts unchanged; source hashes verified before/after. No live launch or DAgger collection. Phase3 requires this gate review and remains original-teacher-controlled.

# Phase 4 natural student-controlled pilot

Status: **pilot_complete_awaiting_review**. Stop for review.

Session: `C:\Users\shreyas\Desktop\model wars\training\behavior_cloning\v13_dagger\phase4_pilot_v1\sessions\20261005_063422_960966`. 5238 unique processed callbacks; 5238 verified student submissions. Final callback submitted: True.

The 13D student alone submitted controls. The unchanged original teacher ran independently in shadow on actual delivered packets/predictions. No state-setting, receive pauses, replay manipulation, fallback or repair was used. Diagnostics remain quarantined.

## Observed learner trajectory and shadow behavior

```json
{
  "actual_learner_induced_callbacks": 5237,
  "shadow_sequence_starts": 9,
  "shadow_completions": 9,
  "student_missed_shadow_starts": 9,
  "offpolicy_shadow_continuation_callbacks": 631,
  "missing_ball_pending": 0,
  "pending_match_phases": {
    "GoalScored": 0,
    "Replay": 0,
    "Countdown": 0,
    "Kickoff": 71
  },
  "long_callback_gaps_over_five_ticks": 0,
  "actual_shadow_prediction_selections": 1064,
  "near_far": {
    "far": 1399,
    "near": 3839
  }
}
```

Missed shadow sequence-start callbacks: [353, 630, 730, 1191, 1784, 1860, 3395, 4099, 4590]. Possibly unexecutable shadow dodge callbacks: [1200, 1201, 1202, 1203, 1204, 1205, 1206, 1207, 1208, 1209, 1210, 1211, 1870, 1871, 1872, 1873, 1874, 1875, 1876, 1877, 1878, 1879, 1880, 1881, 1882, 3404, 3405, 3406, 3407, 3408, 3409, 3410, 3411, 3412, 3413, 3414, 3415, 3416, 4600, 4601, 4602, 4603, 4604, 4605, 4606, 4607, 4608, 4609, 4610, 4611, 4612].

Zero counts identify unobserved categories; this bounded run must not be extended to force coverage. Linked prior submissions identify actual student-controlled successor callbacks. They do not establish what teacher-controlled physics would have been.

## Disagreement and timing

Native action agreement within analog tolerance: 212/5238; mode agreement: 4012/5238. Steering-sign disagreements outside ±0.02: 379.

Per-channel errors, mode counts, callback/inference/submission latency, all disagreement IDs, receipt reuse, phase transitions and failures are preserved in the JSON companion and raw diagnostics.

## Train-only descriptive comparison

Primary reference: Orange train matches pilot_01/v10_006. Secondary: all four train matches. Phase/mask-conditioned feature quantiles, excursions, fixed-bin Jensen–Shannon differences and actual-hold occupancy are descriptive only. No validation/test samples were read; no policy selection or fitting occurred. Cross-phase and final-censored holds are excluded.

## Integrity, failures and limits

Failures: 0; error records: 0; duplicate callbacks: {}; publication errors: [].

Processed callbacks only; receipt counters cannot prove lossless physics-tick delivery.
Natural 90-second wall window includes native countdown/replay; no coverage forcing.
Shadow equivalence was validated in Phase 3; this pilot has no executed-teacher physics counterfactual.
Expert temporal labels may assume actions the learner never executed; they are not approved learning targets.
Marginal train-reference differences describe one learner trajectory; they do not establish causal distribution shift.
No competence gate, aggregation, fitting or additional run is authorized by report completion.

Exact evidence and file SHA256s: `C:\Users\shreyas\Desktop\model wars\training\behavior_cloning\v13_dagger\phase4_pilot_v1\sessions\20261005_063422_960966\contract_report.json` and `C:\Users\shreyas\Desktop\model wars\training\behavior_cloning\v13_dagger\phase4_pilot_v1\sessions\20261005_063422_960966\diagnostic_integrity.json`.

No second pilot, DAgger aggregation, training, or progression follows automatically.

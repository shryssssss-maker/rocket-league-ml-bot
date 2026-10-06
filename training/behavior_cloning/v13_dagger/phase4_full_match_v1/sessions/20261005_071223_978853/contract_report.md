# Natural five-minute student-controlled match

Status: **natural_match_complete_awaiting_review**. Stop for review.

Session: `C:\Users\shreyas\Desktop\model wars\training\behavior_cloning\v13_dagger\phase4_full_match_v1\sessions\20261005_071223_978853`. 26635 unique processed callbacks; 26635 verified student submissions. Final callback submitted: True.

The 13D student alone submitted controls. The unchanged original teacher ran independently in shadow on actual delivered packets/predictions. No state-setting, receive pauses, replay manipulation, fallback or repair was used. Diagnostics remain quarantined.

## Observed learner trajectory and shadow behavior

```json
{
  "actual_learner_induced_callbacks": 26634,
  "shadow_sequence_starts": 33,
  "shadow_completions": 33,
  "student_missed_shadow_starts": 33,
  "offpolicy_shadow_continuation_callbacks": 2311,
  "missing_ball_pending": 0,
  "pending_match_phases": {
    "GoalScored": 0,
    "Replay": 0,
    "Countdown": 57,
    "Kickoff": 313
  },
  "long_callback_gaps_over_five_ticks": 0,
  "actual_shadow_prediction_selections": 7320,
  "near_far": {
    "far": 8783,
    "near": 16644,
    "missing": 1208
  }
}
```

Missed shadow sequence-start callbacks: [453, 715, 787, 1231, 1435, 1566, 1775, 1871, 2025, 2803, 4261, 4345, 4510, 4993, 6822, 8001, 8990, 9808, 10370, 11296, 11548, 13065, 14305, 15957, 16909, 19650, 19938, 21070, 21206, 21311, 23308, 24082, 24231]. Possibly unexecutable shadow dodge callbacks: [1444, 1445, 1446, 1447, 1448, 1449, 1450, 1451, 1452, 1453, 1454, 1455, 1456, 1575, 1576, 2034, 2035, 2036, 2037, 2038, 2039, 2040, 2041, 2042, 2043, 2044, 2045, 2046, 2812, 2813, 2814, 2815, 2816, 2817, 2818, 2819, 2820, 2821, 2822, 2823, 2824, 4270, 4271, 4272, 4273, 4274, 4275, 4276, 4277, 4278, 4279, 5001, 5002, 5003, 5004, 5005, 5006, 5007, 5008, 5009, 5010, 5011, 5012, 5013, 8998, 8999, 9000, 9001, 9002, 9003, 9004, 9005, 9006, 9007, 9008, 9009, 9010, 9011, 10378, 10379, 10380, 10381, 10382, 10383, 10384, 10385, 10386, 10387, 10388, 10389, 10390, 10391, 11304, 11305, 11306, 11307, 11308, 11309, 11310, 11311, 11312, 11313, 11314, 11315, 11316, 11317, 11556, 11557, 11558, 11559, 11560, 11561, 11562, 11563, 11564, 11565, 11566, 11567, 11568, 14313, 14314, 14315, 14316, 14317, 14318, 14319, 14320, 14321, 14322, 14323, 14324, 16918, 16919, 16920, 16921, 16922, 16923, 16924, 16925, 16926, 16927, 16928, 16929, 16930, 16931, 19659, 19660, 19661, 19662, 21079, 21080, 21081, 21082, 21083, 21084, 21085, 21086, 21087, 21088, 21089, 21090, 21091, 21319, 21320, 21321, 21322, 21323, 21324, 21325, 21326, 21327, 21328, 21329, 21330, 21331, 23317, 23318, 23319, 23320, 23321, 23322, 23323, 23324, 23325, 23326, 23327, 23328, 23329, 23330, 24091, 24092, 24093, 24094, 24095, 24096, 24097, 24098, 24099, 24100, 24101, 24102, 24103, 24104, 24240, 24241, 24242, 24243, 24244, 24245, 24246, 24247, 24248, 24249, 24250, 24251, 24252].

Zero counts identify unobserved categories; this bounded run must not be extended to force coverage. Linked prior submissions identify actual student-controlled successor callbacks. They do not establish what teacher-controlled physics would have been.

## Disagreement and timing

Native action agreement within analog tolerance: 2028/26635; mode agreement: 19621/26635. Steering-sign disagreements outside ±0.02: 1615.

Per-channel errors, mode counts, callback/inference/submission latency, all disagreement IDs, receipt reuse, phase transitions and failures are preserved in the JSON companion and raw diagnostics.

## Train-only descriptive comparison

Use the secondary all-four-train reference as the relevant descriptive comparison for this Blue student. The inherited primary is Orange-only and is retained as explicitly side-mismatched context, not a Blue-side matched baseline. No new reference samples were read. Phase/mask-conditioned feature comparisons do not tune the policy. Cross-phase and final-censored holds are excluded.

## Integrity, failures and limits

Failures: 0; error records: 0; duplicate callbacks: {}; publication errors: [].

Processed callbacks only; receipt counters cannot prove lossless physics-tick delivery.
One native five-minute match with countdown/replay and overtime; no wall-clock deadline or coverage forcing.
Shadow equivalence was validated in Phase 3; this pilot has no executed-teacher physics counterfactual.
Expert temporal labels may assume actions the learner never executed; they are not approved learning targets.
Marginal train-reference differences describe one learner trajectory; they do not establish causal distribution shift.
No competence gate, aggregation, fitting or additional run is authorized by report completion.

Exact evidence and file SHA256s: `C:\Users\shreyas\Desktop\model wars\training\behavior_cloning\v13_dagger\phase4_full_match_v1\sessions\20261005_071223_978853\contract_report.json` and `C:\Users\shreyas\Desktop\model wars\training\behavior_cloning\v13_dagger\phase4_full_match_v1\sessions\20261005_071223_978853\diagnostic_integrity.json`.

No second pilot, DAgger aggregation, training, or progression follows automatically.

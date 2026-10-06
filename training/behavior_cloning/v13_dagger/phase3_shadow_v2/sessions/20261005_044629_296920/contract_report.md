# Phase 3 shadow-teacher contract

Recommended verdict: **pass_with_scope_awaiting_review**. Final agent status: coverage_complete_awaiting_review; supervisor status: coverage_complete_awaiting_review. Review required; no automatic pass.

Callbacks: 1453; all categories observed: True; comparison/error rows: 0; transport discrepancies: 0; hidden-chain discrepancies: 0.

Raw lines: 1453; duplicate callback records: {}; diagnostic errors: 0; missing submissions: []; final callback submitted and verified: True.

Actual student sequence-start disagreements: [465, 611, 668]; shadow continuation callbacks after those starts: 913.

Independent shadow continuation after real student disagreements on teacher-controlled physics only. Learner-induced off-policy physics was not tested.

Unobserved categories: none.

## Actual evidence

- neutral_to_chase: 1 retained callback examples
- chase_to_jump: 2 retained callback examples
- release: 12 retained callback examples
- front_dodge: 12 retained callback examples
- coast: 12 retained callback examples
- completion: 2 retained callback examples
- missing_ball_pending: 12 retained callback examples
- goal_pending: 12 retained callback examples
- replay_pending: 12 retained callback examples
- kickoff_pending: 12 retained callback examples
- long_gap: 1 retained callback examples
- near_ball: 12 retained callback examples
- far_ball: 12 retained callback examples
- live_selector: 12 retained callback examples
- student_missed_sequence_start: 3 retained callback examples
- student_missed_prior_jump: 4 retained callback examples
- shadow_continues_after_student_divergence: 12 retained callback examples

Same sequence across goal/replay/kickoff: [3].

43 protected hashes unchanged: True; completed immutable artifacts unchanged: True.

## Scope

- Only actual processed RLBot callbacks; not lossless delivery or equivalence on undelivered physics ticks.
- Receive pauses can cause queue-full warnings; IPC/profiling/logging can change delivered callback scheduling.
- Teacher drives; student outputs never reach the game. No learner-induced physical-state coverage is established.
- Shadow is an independent original-source actor; exact equivalence does not prove learned sequence competence.
- Requested probe states do not count as evidence; only actual later packets/sequence transitions do.
- Observation prediction probe is distinct from whether the original gameplay branch consulted prediction.
- Diagnostic callback logs are not a DAgger training dataset; no tensor export, aggregation, fitting or test evaluation.

Full native controls, source/shadow temporal states, live selector traces, student diagnostics, packet/prediction receipts and intervention events are retained in session JSONL. See JSON report for every discrepancy. Stop for review; no student-controlled gameplay or DAgger collection follows.

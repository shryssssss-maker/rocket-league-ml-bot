# V10 natural demonstration collection review

Status: **coverage_targets_met_awaiting_review**. Dataset not frozen; no training.

Eligible matches: 9 (6 new); callbacks: 239,205.
Cumulative Jump: 1,108; Front dodge: 3,359; sequence starts/completions: 250/249.

## Coverage decisions

| Question | Current answer |
|---|---|
| jump_coverage_sufficient | True |
| front_dodge_coverage_sufficient | True |
| teacher_side_diversity_sufficient | True |
| opponent_session_diversity_sufficient | True |
| temporal_sequence_coverage_sufficient | True |
| dataset_ready_to_freeze_for_bc | Yes under the approved collection criteria, subject to user freeze review; no automatic freeze or guarantee of BC success |

## Match assignments

| Match | Split | Teacher side | Opponent | Play session | Callbacks | Jump | Dodge | Completed sequences |
|---|---|---|---|---|---:|---:|---:|---:|
| pilot_01 | train | Orange | human_a | pilot_20261004_session_identity_not_separately_recorded | 32258 | 169 | 519 | 39 |
| pilot_02 | validation | Orange | human_a | pilot_20261004_session_identity_not_separately_recorded | 27401 | 94 | 285 | 21 |
| pilot_03 | test | Orange | human_a | pilot_20261004_session_identity_not_separately_recorded | 36825 | 165 | 488 | 35 |
| v10_001 | train | Blue | human_a | human_a_v10_session_01 | 25376 | 121 | 377 | 28 |
| v10_002 | validation | Blue | human_a | human_a_v10_session_01 | 19870 | 92 | 283 | 21 |
| v10_003 | test | Blue | human_a | human_a_v10_session_01 | 23982 | 96 | 279 | 21 |
| v10_004 | train | Blue | human_b | human_b_v10_session_01 | 24230 | 115 | 351 | 26 |
| v10_005 | test | Orange | human_c | human_c_v10_session_01 | 23727 | 157 | 482 | 36 |
| v10_006 | train | Orange | human_b | human_b_v10_session_01 | 25536 | 99 | 295 | 22 |

## Integrity and distributions

Every eligible raw row was audited: contiguous callback index, actual side, native action support, canonical elapsed/dt, bitwise 18D reconstruction and exact raw/tensor round-trip. Raw/tensor SHA256 and protected sources were checked. Original pilot files were not rewritten.
The JSON includes side/opponent/session/split grouping, all native channels, rare modes, diagnostic per-sequence phase coverage, near/far, missing-ball, match phases, speed bands and distributions, car-local ball geometry, ball speed/spin/height, prediction horizon/reuse and invalid/incomplete match evidence.

## Remaining gaps

- Coverage thresholds are provisional heuristics, not proof of learned recurrent behavior or generalization.
- Natural collection does not guarantee extreme states or very long pending-sequence interruptions.
- Match-level splitting does not establish opponent/session-disjoint evaluation; report identities explicitly.

Stop after coverage review. No synthetic trajectories, controlled probes, prediction substitutions, additional matches, dataset freeze or training start automatically.

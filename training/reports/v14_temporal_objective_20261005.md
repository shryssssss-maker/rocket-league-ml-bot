# V14 v2 temporal objective results

Status: **unsuccessful_for_temporal_purpose**.

Original V13 loss + 0.1 × approved adjacent-boundary BCE; one fresh seed-42 GRU. No v1/test/live/DAgger.

## Baseline and validation comparison

| Metric | Pinned V13 | V14 v2 |
|---|---:|---:|
| Original total loss | 0.16538622975349426 | 0.18274301290512085 |
| Overall mode accuracy | 0.9622601595058281 | 0.9559772376298364 |
| Chase accuracy | 0.9946232223028739 | 0.9876065274081243 |
| Native Jump precision | 0.8349056603773585 | 0.848780487804878 |
| Native Jump recall | 0.23474801061007958 | 0.23076923076923078 |
| Front-dodge precision | 0.7623762376237624 | 0.8449197860962567 |
| Front-dodge recall | 0.2711267605633803 | 0.27816901408450706 |
| Pure Jump-mode accuracy | 0.043010752688172046 | 0.03763440860215054 |
| Boundary mode accuracy | 0.4869516935036091 | 0.4575235980011105 |
| Steer MAE, all callbacks | 0.14065021202979325 | 0.16547069151738905 |
| Steer MAE, Chase | 0.14956169466640562 | 0.1795164698673304 |
| Exact native agreement | 0.17960271625309387 | 0.17884114996509487 |
| Timed-core score | 0.0 | 0.0 |

## Objective and training

The exact formula, edge construction, numerical/masking rules and lambda justification are retained in proposal_v2.md/json. Same 26,181-parameter model, input projection, native decoder, optimizer, seed and schedule. Selection uses original unweighted validation loss, not the boundary or temporal score.
Selected checkpoint: `C:\Users\shreyas\Desktop\model wars\training\behavior_cloning\v14_temporal_objective\run_v2\B_best.pt`; SHA256 `fe25a208d3fedaef02db27481c40b6eb7d771331c67109d9d7e22a29c65fff82`.

## Temporal sequence and regression gates

Checks: `{"temporal_score": false, "extra_core_rate": true, "chase_accuracy": false, "chase_steering": false, "overall_accuracy": true}`.
Timed cores: V13 0/42; V14 0/42. Extra cores/minute: V13 0.22765838770390806; V14 0.07588612923463603.

Complete per-mode/channel, temporal edge/duration, missing-ball/gap and regression evidence is in report.json.

## Limitations

Recorded teacher-controlled physics only; no closed-loop or DAgger benefit claim
No previous-action feedback exists in 13D. Local edge supervision does not ensure global/physical maneuver validity. Censored/interrupted cores are reported separately. Match-disjoint validation retains opponent overlap; checkpoint selection may reject temporally better epochs. One seed does not establish statistical significance.

## One recommendation

Investigate representation; do not change it automatically. Stop for review.

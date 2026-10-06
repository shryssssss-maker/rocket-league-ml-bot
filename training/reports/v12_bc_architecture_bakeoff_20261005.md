# V12 Behavior Cloning architecture bakeoff

Status: **training_and_offline_evaluation_complete_awaiting_review**. Offline only; no deployment or PPO.

Frozen manifest: `080943fd8f9a81ada800e0a2f9ff3593b67f66d32272211d67030c03cf433a83`. Inputs verified before and after; unchanged.

## Fixed experiment

A: 18→128→64 ReLU MLP. B: 18→64 ReLU→GRU(64), one causal layer. Both use four mode logits and continuous tanh steering, decoded in throttle/steer/pitch/yaw/roll/jump/boost/handbrake order.

Unweighted mode cross-entropy plus Chase-only steering squared error averaged over all callbacks. Adam 0.001, seed 42, max 20 epochs, patience 4. Chronological 512-callback chunks; GRU values carried across chunks, gradients detached. Reset only at match boundaries. No balancing or new normalization.

Checkpoints selected on teacher-forced validation loss. Both selections completed before test tensor/label loading. Test evaluated once as a final suite with both conditions; no tuning from test.

## Overall comparison

| Model | Split | Context | Loss | Throttle MAE | Steer MAE | Jump precision | Jump recall | Exact native agreement |
|---|---|---|---:|---:|---:|---:|---:|---:|
| A | validation | teacher_forced | 0.026817 | 0.001967 | 0.026690 | 0.8885941644562334 | 0.8885941644562334 | 0.209558 |
| A | validation | student_forced_offline | 3.002808 | 0.735313 | 0.360407 | None | 0.0 | 0.191449 |
| A | test | teacher_forced | 0.031051 | 0.002449 | 0.024109 | 0.8888221153846154 | 0.8872225554889022 | 0.282797 |
| A | test | student_forced_offline | 2.544235 | 0.615658 | 0.326554 | None | 0.0 | 0.257376 |
| B | validation | teacher_forced | 0.020937 | 0.002010 | 0.030883 | 0.9314516129032258 | 0.9190981432360743 | 0.210785 |
| B | validation | student_forced_offline | 2.595244 | 0.786888 | 0.358724 | 0.028142589118198873 | 0.01989389920424403 | 0.197288 |
| B | test | teacher_forced | 0.023845 | 0.002496 | 0.028807 | 0.931265206812652 | 0.9184163167366527 | 0.284276 |
| B | test | student_forced_offline | 2.432597 | 0.712790 | 0.322393 | 0.028169014084507043 | 0.017996400719856028 | 0.267620 |

## Review interpretation from saved results

Both fits completed, but neither model demonstrates adequate offline autoregressive robustness for deployment. This conclusion uses the already-completed evaluation evidence only; no training or test evaluation was repeated.

| Model | Split | Teacher-forced mode accuracy | Student-forced mode accuracy | Teacher-forced steer MAE | Student-forced steer MAE |
|---|---|---:|---:|---:|---:|
| MLP A | validation | 99.54% | 25.07% | 0.02669 | 0.36041 |
| GRU B | validation | 99.66% | 19.73% | 0.03088 | 0.35872 |
| MLP A | test | 99.42% | 36.80% | 0.02411 | 0.32655 |
| GRU B | test | 99.57% | 26.76% | 0.02881 | 0.32239 |

The GRU has lower teacher-forced loss, higher Front-dodge recall (97.12% versus 92.39% on test), and higher sequence-boundary mode accuracy (91.58% versus 88.48%). The MLP has lower steering MAE. Pure Jump-mode accuracy is slightly better for the MLP (77.75% versus 76.08%); jump-button recall includes both Jump and Front-dodge and should not be confused with pure Jump-mode accuracy.

Under student-forced test context, jump-button recall falls to 0% for A and 1.80% for B; Front-dodge recall falls to 0% and 1.60%. Chase-mode accuracy falls to 15.51% for A and 0% for B. High overall jump-button accuracy under this condition is dominated by negative examples and does not establish jump competence. Exact native agreement is also strongly influenced by Neutral controls and exact continuous-steering comparisons.

These findings are consistent with reliance on teacher-forced previous-action context and unstable autoregressive behavior, but the precise failure mechanism has not been established by a dedicated diagnostic. Recorded physics/predictions remain fixed in this evaluation; no live gameplay conclusion is proven.

All missing-ball callbacks have correct Neutral-mode outputs in both conditions. That does not establish survival of learned pending sequence memory. Only two validation callbacks exceed 100 ms, and none do in test; no broad long-gap robustness claim is supported.

No architecture is approved as a deployable winner. Stop for review of the context-dependence failure before any further experiment; do not repeat test evaluation or tune using these test results.

## Selected checkpoints

- A: best epoch 20; 11013 parameters; `C:\Users\shreyas\Desktop\model wars\training\behavior_cloning\v12\runs\v12_bc_mlp_gru_20261005_v1_20261005_033125_751956\A_best.pt`; SHA256 `80cd51065035b601a3d9bd4d6f9c8c60a594a95f339f00bee72282465e567a24`.
- B: best epoch 9; 26501 parameters; `C:\Users\shreyas\Desktop\model wars\training\behavior_cloning\v12\runs\v12_bc_mlp_gru_20261005_v1_20261005_033125_751956\B_best.pt`; SHA256 `0eaab20bd2860db2a981b4ca99a0a2256a7f95bc2dbdfd446b0354ffcce55295`.

## Detailed evidence

JSON records every requested per-channel loss/MAE, throttle/steer MAE, pitch accuracy/error, jump precision/recall, mode confusion, Front-dodge precision/recall, Jump/Dodge subsets, sequence-phase-stratified action agreement, ±5-callback sequence-boundary neighborhoods, missing-ball neighborhoods and callbacks after dt >100 ms. Empty subsets and undefined precision/recall are reported as null, not success.

Exact native agreement uses no tolerance. A separate analog ≤1e-6/buttons-exact agreement is also reported. Private phase identity has no output head: phase-stratified action accuracy is not phase reconstruction.

Teacher-forced validation preserves recorded prior-action features. Student-forced evaluation copies each observation and replaces only prior-mode one-hot and prior steering with the preceding decoded prediction; all recorded physics/prediction/dt remain unchanged. JSON explicitly reports student-minus-teacher-forced degradation separately for validation and test.

Yaw=0, roll=0, boost=false and handbrake=false for every teacher target. The decoded heads enforce that same support; no variation or ability outside teacher support is claimed.

## Limitations

- Splits are match-disjoint, NOT fully opponent/session-disjoint: human_a spans train/validation/test. human_b has two train matches only (v10_004 and v10_006), sharing human_b_v10_session_01. human_c has one test match (v10_005). No records or split assignments were changed.
- Opponent identities were declared by the user; pilot play-session identities were not separately recorded and remain unknown.
- Coverage targets are provisional heuristics, not proof of recurrent learning or generalization; natural collection does not guarantee extreme states or long pending-sequence interruptions.
- Counts describe actual processed RLBot callbacks, not lossless network delivery or undelivered physics ticks.
- Hidden teacher sequence state is diagnostic only and excluded from the unchanged 18D observations.
- This is an immutable hash-identified reference artifact, not an OS write lock on the original files. Referenced files must remain available and pass verification; changes require a new dataset version.
- Historical code hashes are retained from collection metadata. Current code snapshots are labeled separately and do not claim to reconstruct unavailable historical revisions.
- One seed and bounded fixed architecture/loss comparison, not a hyperparameter sweep or generalization guarantee.
- Student-forced diagnostic uses recorded physics/prediction, not physics resulting from student controls; not closed-loop evaluation.
- Final exact native agreement is with frozen float32 analog targets; raw double-precision controller values remain preserved in raw records.
- Private sequence labels used only for retrospective metric subsets, never model inputs, fitting weights or selection.
- Both modes use identical CPU evaluation. Training on GPU and CPU evaluation may have small floating-point differences.

Stop for review. No live deployment, teacher replacement or RL fine-tuning.

# V13 context audit

Phase 1: **passed with scope**. Unchanged V12 inspected; no fixes, fitting or live launch.

## Exact 18D ordering

0: ball_forward | 1: ball_right | 2: ball_up | 3: prediction_forward | 4: prediction_right | 5: prediction_up | 6: ball_distance | 7: car_speed | 8: ball_present | 9: prediction_valid | 10: prediction_horizon | 11: prediction_first_offset | 12: callback_dt | 13: previous_neutral | 14: previous_chase | 15: previous_jump | 16: previous_dodge | 17: previous_steer

## Tensor contract

{
  "callback_input": "[L,18] float32, L <=512 during chunks; [1,18] during autoregressive evaluation",
  "gru_internal_input": "[1,L,64] float32, batch_first=True",
  "gru_hidden": "[1,1,64] float32; one layer, one direction, one match at a time",
  "output": "mode logits [L,4] float32 and tanh continuous steering [L] float32; decoded native controls [L,8] float32",
  "target": "mode int64 for CE; analog float32; frozen buttons uint8 converted to float32 only for native metric arrays"
}

## Hidden reset

None at each match start in train/validation/final evaluation and each independent context pass. No goal/replay/kickoff/missing-ball resets. Training h.detach() at chunk boundaries truncates gradients without resetting values.

## Chronology

Fixed frozen match order, consecutive slices start:start+512; tail retained. Every train callback once per epoch. Optimizer updates each chunk; hidden values computed under earlier weights carry into the next chunk.

## Teacher forced

Recorded 18D arrays passed unchanged; GRU hidden carries across consecutive chunks.

## Student forced

Recorded input copied callback-by-callback; replace indices 13:17 with previous argmax predicted mode one-hot and index17 with previous decoded steering (zero outside Chase). Initial previous action Neutral/zero. Recorded indices0:13 remain bitwise unchanged. No ground-truth prior actions used after initialization.

## Teacher information

Previous-action fields are intentionally teacher-forced in standard inputs. No private sequence ID/index/phase is read by Policy/train/teacher_loss. Final diagnostics use private labels only as retrospective metric masks. Current prediction is a service forecast supplied at the callback, not a later recorded ball outcome.

## Reproducibility

Code/config/checkpoint hashes match recorded V12 run; safe weights_only state-dict loading succeeds. Seed42/runtime/device/selection history recorded. Exact rerun bit identity is not guaranteed because GPU determinism was not enforced. Optimizer state and per-chunk recurrent state were not saved, so checkpoints do not support exact interrupted-run continuation.

## Diagnosis

No context mutation bug found in inspected predict implementation/numerical probe. Evidence supports sensitivity to previous-action context. Physical-state distribution shift and the precise learned shortcut mechanism are not established.

## Evidence and limits

Numerical capture of the original V12 evaluation function confirms teacher-forced input identity, unchanged physical/prediction/dt indices0:13, exactly previous predicted mode/steering in indices13:18, and no in-place modification. This is an implementation probe, not another V10 test evaluation.

Checkpoint/config/code SHA256s and every checkpoint tensor shape/dtype are in the JSON companion. MLP has 11,013 parameters; GRU has 26,501. Hidden shape is [1,1,64].

The V11 frozen row audit remains the evidence for prior-action provenance in the recorded data. Phase1 does not repeat the dataset sweep or claim a comprehensive root-cause explanation.

Original code and saved artifacts unchanged. Stop before subsequent gates until their specifications are resolved.

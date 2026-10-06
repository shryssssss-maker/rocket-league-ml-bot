# V11 frozen dataset preprocessing and leakage audit

Status: **passed_with_scope**.

Manifest SHA256: `080943fd8f9a81ada800e0a2f9ff3593b67f66d32272211d67030c03cf433a83`.

Verified all 171 manifest-listed external artifact hashes and the frozen version seals before reading dataset contents. Tensor shape/dtype/byte-size/order checks passed; every raw row exactly reconstructs its 18D float32 tensor and native action tensors.

All 239,205 rows audited. Prior submitted controls/mode/time/callback agree with the immediately preceding processed teacher callback. Each of nine matches starts with null raw dt/history, processed dt zero and Neutral/zero previous action. No match crosses a split boundary.

Missing-ball masks and zeroed ball/prediction features match frozen data. Car-speed, dt and prior-action context are retained. Teacher-private state stays diagnostic only. No phase-based resets or added input features.

All per-match statistics, aggregate counts/distributions, native channel distributions, split/side/opponent/session groups and sequence/prediction-reuse totals exactly reproduce the frozen manifest. This uses the verified frozen audit arithmetic, preserving quantile and floating-point accumulation conventions.

## Required checks

| # | Check | Result |
|---|---|---|
| 1 | Pinned manifest and all frozen/artifact SHA256 integrity | pass |
| 2 | Exact tensor shapes/dtypes/channel ordering | pass |
| 3 | 18D feature ordering against frozen manifest | pass |
| 4 | Native eight-channel ordering against frozen manifest | pass |
| 5 | Previous-action features from immediately prior actual callback | pass |
| 6 | History initialization at every match boundary | pass |
| 7 | No split/sample/index recipe crosses a match boundary | pass |
| 8 | Causal index recipes exclude future callbacks; no windows exported | pass |
| 9 | Private sequence state remains diagnostic, excluded from model-input recipes | pass |
| 10 | Missing-ball features exactly preserved | pass |
| 11 | Initial raw null dt/history and tensor/observation zero dt explicit | pass |
| 12 | All frozen per-match and aggregate distributions exactly reproduced | pass |
## Leakage and scope

- Existing frozen tensors audited callback-by-callback; no sequence/window preprocessing implementation exists yet.
- Causal index recipes exhaustively checked for lengths 2/4/8/16/32/64, without choosing a length or constructing data.
- Frozen observation reconstruction depends on current physics/prediction and strictly prior submitted action only; prediction is a contemporaneous forecast, not a later recorded outcome.
- Private sequence diagnostics only affect statistics; they are excluded from 18D and causal input recipes.
- Match-disjoint splits are not fully opponent/session-disjoint. human_a spans all splits; human_b has two train matches sharing one session; human_c has one test match.
- Historical protection evidence is hash-pinned in freeze metadata; current bot/source files are deliberately not read.

## single callback feedforward

Advantages:

- Simple, low latency, no window assembly; consumes only frozen 18D.

Limitations:

- 18D includes previous mode/steering and elapsed dt, but not sequence elapsed time, pending sequence identity or recurrent memory. Release and Coast both emit Neutral; private timing survives missing-ball/replay callbacks. A single callback is therefore not a sufficient exact temporal teacher state.
- Shared previous-action inputs are teacher-forced in this dataset. Live student uses its own previous emitted action; errors can compound. Offline agreement does not prove live equivalence.

Preprocessing requirements:

- Keep 18D/native targets unchanged; reset previous history at each match; preserve raw dt null versus processed zero; no scaling fit on validation/test. Any future learned scaling must use train only.


## short causal history

Advantages:

- Can expose prior mode changes and delivered callback timing without private labels.

Limitations:

- Fixed callback count spans variable elapsed time. Finite history may lose a pending sequence origin across missing-ball/replay/long gaps; no tested length is approved or guaranteed sufficient.
- Teacher-forced previous actions versus student-generated actions remains a deployment issue.

Preprocessing requirements:

- Use only rows <= current callback from the same match/split. Include the current observation; target is current action, not a future action.
- At match start use explicit shorter history or separately reviewed validity masks; do not silently pad, repeat, or drop callbacks.
- Do not reset at goals/replays/missing ball unless a future design explicitly approves a change; teacher memory can survive them.
- No centered windows, backward interpolation from future packets, shuffled recurrent state, or private sequence labels as inputs.


Neither architecture selected; no history tensors, training examples, preprocessing exports or models created.

Frozen inputs unchanged. Stop for review; no training.

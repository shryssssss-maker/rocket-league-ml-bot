# V10 frozen dataset

Dataset version: **v10_live_1v1_18d_20261005_v1**. User-approved freeze; no training.

Manifest SHA256: `080943fd8f9a81ada800e0a2f9ff3593b67f66d32272211d67030c03cf433a83`.

Manifest: `C:\Users\shreyas\Desktop\model wars\training\behavior_cloning\datasets\v10_live_1v1_18d_20261005_v1\manifest.json`. All per-file paths, SHA256 hashes, tensor shapes/dtypes, identities and collection provenance are recorded there.

## Split sizes

| Split | Matches | Callbacks | Starts | Completions |
|---|---:|---:|---:|---:|
| train | 4 | 107,400 | 115 | 115 |
| validation | 2 | 47,271 | 42 | 42 |
| test | 3 | 84,534 | 93 | 92 |

## Action and sequence coverage

| Mode | Callbacks | Percent |
|---|---:|---:|
| chase | 177,174 | 74.0678% |
| jump | 1,108 | 0.4632% |
| neutral | 57,564 | 24.0647% |
| front-dodge | 3,359 | 1.4042% |

Total callbacks: 239,205. Sequence starts/completions: 250/249. All-four-phase completions: 249. Jump and Front-dodge counts exceed approved 1,000/3,000 targets.

## Ball availability and timing

Near/far/missing-ball counts: {"far": 54614, "near": 140086, "missing_ball": 44505}.

Callback dt in seconds (first callback of each match excluded): `{"count": 239196, "minimum": 0.00830078125, "maximum": 0.241668701171875, "mean": 0.016676038467140886, "p50": 0.01666259765625, "p90": 0.0166778564453125, "p95": 0.016693115234375, "p99": 0.024993896484375}`.

All processed callbacks retained in original order. No resampling, balancing, chunks, augmentation, synthetic trajectories or changes to the 18D/native-control contracts.

## Integrity and paths

Verified 171 artifact files against collection/tensor manifests. Zero invalid/dropped processed rows in accepted audit; all 43 protected hashes and 57 pilot files unchanged.

Raw and tensor artifacts remain at their original paths. Code/config/report snapshots are under the versioned `snapshots/` folder. File SHA256 values and tensor shape/dtype/order are enumerated in manifest.json. The accepted collection review remains historical and is not rewritten.

## Limitations

- Splits are match-disjoint, NOT fully opponent/session-disjoint: human_a spans train/validation/test. human_b has two train matches only (v10_004 and v10_006), sharing human_b_v10_session_01. human_c has one test match (v10_005). No records or split assignments were changed.
- Opponent identities were declared by the user; pilot play-session identities were not separately recorded and remain unknown.
- Coverage targets are provisional heuristics, not proof of recurrent learning or generalization; natural collection does not guarantee extreme states or long pending-sequence interruptions.
- Counts describe actual processed RLBot callbacks, not lossless network delivery or undelivered physics ticks.
- Hidden teacher sequence state is diagnostic only and excluded from the unchanged 18D observations.
- This is an immutable hash-identified reference artifact, not an OS write lock on the original files. Referenced files must remain available and pass verification; changes require a new dataset version.
- Historical code hashes are retained from collection metadata. Current code snapshots are labeled separately and do not claim to reconstruct unavailable historical revisions.

Frozen; stop for review before preprocessing or training.

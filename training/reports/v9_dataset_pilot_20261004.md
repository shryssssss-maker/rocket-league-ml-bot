# V9 teacher dataset pilot



Status: **PILOT ACCEPTED — DATASET NOT FROZEN**. No model training.



Complete accepted matches: 3; callbacks: 96484.



Real Rocket League / RLBot / unmodified original teacher. No controlled resets, receive pauses, synthetic trajectories or RocketSim.

Every processed callback remains in raw records. Invalid/incomplete matches are rejected explicitly, never silently filtered into training tensors.

Teacher private sequence state is diagnostic-only. The unchanged 18D features contain previous actually submitted controls, not private sequence labels.

Processed tensors are little-endian typed binary arrays with explicit shapes/dtypes/hashes, separate from raw JSONL. No sequences/chunks are constructed yet.



## Match splits



| Match | Split | Callbacks | Flips | Missing ball |

|---|---|---:|---:|---:|

| pilot_01 | train | 32258 | 39 | 9435 |

| pilot_02 | validation | 27401 | 21 | 6372 |

| pilot_03 | test | 36825 | 36 | 12358 |



Reserved split: pilot_01=train, pilot_02=validation, pilot_03=test. Incomplete matches are not counted.



## Distributions and integrity



The accompanying JSON records per-match action-channel distributions (min/max/mean/percentiles and zero/±1 counts), output and source-branch distributions, near/far geometry, phases, flips/completions, missing-ball counts, callback-dt distribution, prediction reuse, invalid/dropped rows and all collection failures.

All accepted raw inputs and exported tensors are hash-verified. All 43 protected sources remain unchanged. Packet/receipt counters document provenance; this does not claim lossless transport or coverage of undelivered ticks.



## Larger collection decision



The complete pilot supports a larger collection experiment: all four action modes and both near/far cases were observed, and integrity/reconstruction passed. User review is still required; three matches do not establish adequate coverage, class balance or statistical generalization.

Stop for pilot review; no additional collection or training is automatically approved.


## Completed pilot evidence review

All three matches ended naturally. 96,484 processed callbacks were retained: train=32,258 (`pilot_01`), validation=27,401 (`pilot_02`), test=36,825 (`pilot_03`). All exported native controller combinations and tensor shapes were checked. Raw/tensor hashes were reverified; manifest SHA256 values are in the JSON report. Zero invalid rows, zero omitted processed rows and zero collection failures. All 43 protected hashes remain unchanged.

| Output mode | Callbacks | Fraction |
|---|---:|---:|
| chase | 61,718 | 63.97% |
| jump | 428 | 0.44% |
| neutral | 33,046 | 34.25% |
| front-dodge | 1,292 | 1.34% |

Native channels remain exact: throttle 0/1, continuous steering in [-1,1], pitch 0/-1, exact jump buttons, yaw/roll zero and boost/handbrake false. Aggregate channel distributions are in the JSON. No quantization or private sequence-state input was introduced.

Source branches: far-ball=15,475; near-ball=46,339; pending sequence=6,505; missing ball=28,165. Geometric near/far counts with a ball present are 48,741/19,578, including pending sequences; these differ from decision-branch denominators.

Phases: Countdown=9,076; Kickoff=7,168; Active=52,065; GoalScored=6,797; Replay=21,358; Ended=20. Missing-ball fraction: 29.19%. There were 96 sequence starts and 95 completions. Match 3 sequence 36 remained pending at Coast/index 3 when the natural match ended. Its pending state is retained; no completion is invented.

| Callback delta statistic | Milliseconds |
|---|---:|
| minimum | 8.301 |
| p50 | 16.663 |
| p95 | 16.693 |
| p99 | 25.002 |
| maximum | 241.669 |

There are 96,481 noninitial intervals: three exceeded 100 ms; none were nonpositive or exceeded one second. No intentional receive pauses were used. Record-path maxima were 250 ms, 219 ms and 16 ms across the matches. These include teacher execution/profiling, observation construction and file logging; they do not isolate the cause of stalls.

The pilot supports a larger collection experiment after review: all action modes, near/far states and ordinary phase transitions are observed, with successful reconstruction and integrity checks. It does not establish sufficient training coverage: Jump accounts for only 0.44% of callbacks, all matches use one human opponent and one teacher side, and validation/test each contain one match. Preserve callback order and match-level splits. Zero omitted processed rows does not prove lossless packet delivery or equivalence on undelivered ticks.

Status: **pilot accepted; dataset not frozen**. No further collection, sequence construction, resampling, rebalancing, BC, PPO or model training was performed.

## User review — 5 October 2026

Pilot technically accepted and recorder approved for larger natural collection. Rare-mode, teacher-side and independent opponent/session coverage remain insufficient to freeze the BC dataset. The original pilot matches remain unchanged. No training is approved.

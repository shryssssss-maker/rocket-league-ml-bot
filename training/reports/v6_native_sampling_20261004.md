# V6 native RocketSim sampling investigation

Date: 4 October 2026. **Native sampling offsets established for the tested parameter combinations. Overall V6 remains INCOMPLETE:** exact installed-binary source provenance is not established, and relevant API limitations require review before choosing a provider/configuration. V7 and ML work were not started.

## Source revision and installed artifact

Source checkout: `training/reports/v6_upstream_source_2da51b1dac7b8127127613a5ff30e490bdd70dd8/`.

- Repository: `mtheall/RocketSim`, branch `python-dev`.
- Resolved HEAD: **`2da51b1dac7b8127127613a5ff30e490bdd70dd8`**.
- Commit: `update abi compatibility`, 24 April 2026.
- `pyproject.toml` declares version **2.2.1**.
- Relevant checked-out files were verified against their pinned Git blob hashes; SHA256 hashes are also saved in the JSON.

Installed artifact: `training/venv/Lib/site-packages/RocketSim.pyd`, version metadata **2.2.1**, size **867,840 bytes**, SHA256:

```text
e3ee24ca82445b4bfcc754583f6778d7b0d8b7a7f7d64f872be8c65e621a63d0
```

The binary matches its installed distribution `RECORD` hash/size. `direct_url.json` is absent; installed metadata contains no recorded source commit. The wheel tag is `cp311-cp311-win_amd64`, and the package is currently imported under Python 3.12.10. Matching version numbers and matching observed behaviors do **not** prove that this binary was built from the retrieved commit. No rebuild, upgrade, installation or binary modification was performed.

## Exact checked-out source findings

The statements in this section describe the pinned source. Installed-binary observations are distinguished below.

| Topic | Source authority | Finding |
|---|---|---|
| Arena argument naming/count | `python-mtheall/Arena.cpp:1954`, `Arena::GetBallPrediction` | Despite the `num_ticks` docstring, the parser uses **`num_states`**. It allocates a list of N returned states and sets tracker horizon to N × stride. |
| Returned sampling stride | Same function, output loop | Reads tracker entries 0, stride, 2×stride, …, (N−1)×stride. It does not return N/stride states. |
| Predictor argument/count | `python-mtheall/BallPredictor.cpp:280`, `BallPredictor::GetBallPrediction` | Accepts `num_states`; sets horizon N × stride; returns exactly N entries at the same stride indices. |
| Initial sample | `src/Sim/BallPredTracker/BallPredTracker.cpp:63`, `ForceUpdateAllPred` | Assigns `predData[0] = initialBallState`; steps once for each subsequent tracker entry. Fresh/full predictions include the current state at offset zero. |
| Arena elapsed ticks | Same file, `UpdatePredFromArena` | Computes caller arena tick count minus tracker `lastUpdateTickCount`, then calls `UpdatePredManual`. |
| Manual elapsed ticks | Same file, `UpdatePredManual` | `ticks_since_last_update` is the number of physics ticks since the prior tracker update, not a requested prediction offset or decision count. |
| Cache reuse | Same function | If the elapsed-tick cached entry matches the supplied state, removes elapsed entries and extends the tail when elapsed ticks are positive. Otherwise fully regenerates from the supplied current state. |
| Matching tolerance | `src/Sim/Ball/Ball.cpp:12`, `BallState::Matches` | Strict position distance <0.8 UU, velocity distance <0.4 UU/s and angular-velocity distance <0.02 are required. It is not exact state equality. |

Cache reuse can therefore return a sample-zero state close to, but not numerically identical to, the supplied current state. Offset zero describes the prediction's time origin; it does not guarantee exact geometry after a tolerated input change.

### Source-supported cache-growth hazard

Both wrappers change `numPredTicks` and reserve capacity. In `UpdatePredManual`, an unchanged matching state with **zero elapsed ticks** takes the `No change, no update needed` branch. That branch does not extend `predData.size()` for a larger horizon. The wrappers then index the requested entries without validating that the actual vector size covers them.

This source permits reads beyond the existing data when the requested horizon grows at an unchanged zero-tick update. Reserving capacity is not populating states. This finding comes from source inspection; **the old suspicious interval-8 output was not used to prove its exact failure mechanism**, and the unsafe horizon-growth configuration was not repeated. No trimming or silent refresh workaround was implemented.

### Source-supported Python-field discrepancy

`python-mtheall/BallState.cpp:359/410`, `Setpos/Setvel`, replace Python-facing vector references. `BallState::ToBallState` at line 129 converts those references into the C++ state. `python-mtheall/Ball.cpp:Ball::SetState` uses that conversion.

By contrast, `python-mtheall/BallPredictor.cpp:330` passes `PyCast<BallState>(ballState)->state` directly to the tracker. Post-construction vector assignments need not be reflected in that stored C++ state. This explains the checked-out implementation's different handling of an assigned-field object versus a constructor-initialized object; empirical observations are consistent with it. No copy/conversion workaround or replacement initialization policy was added to production code.

## Controlled installed-binary experiments

Runner: `training/v6_prediction_test/probe_native_sampling.py`. Existing RocketSim collision meshes were used with car-free Soccar arenas. Initial ball state: position (0,0,1200), velocity (600,200,100), angular velocity (0.1,0.2,0.3).

Each Arena count/stride combination gets a **fresh arena/cache**, so the probe does not enlarge an unchanged cache. Its predicted states are compared with a separate arena stepped one physics tick at a time. Position, velocity and angular velocity are compared; integer sample offsets are identified independently. Caller state/tick count are checked before and after prediction.

| N returned states | Stride | Verified sample offsets, in physics ticks | Fresh-state maximum difference | After caller advances two ticks: maximum difference |
|---:|---:|---|---:|---:|
| 1 | 1 | 0 | 0 | 0 |
| 12 | 1 | 0…11 | 0 | 0.0001220703125 |
| 12 | 2 | 0,2,…22 | 0 | 0 |
| 12 | 3 | 0,3,…33 | 0 | 0 |
| 12 | 8 | 0,8,…88 | 0 | 0 |
| 241 | 1 | 0…240 | 0 | 0.0001220703125 |

All fresh predictions matched the separate reference exactly. Same-size repeated calls at unchanged time also matched exactly. Predictions did not advance the caller arena or change its current ball state. After two caller ticks, returned offsets matched 2+i×stride on the original reference timeline. Cache-tail numerical differences are explicitly retained rather than labeled exact trajectory identity.

The offset-disambiguation bound was 0.01 per measured component, with each returned sample additionally required to be nearest to the declared reference tick. This is a diagnostic tick-identification bound for the selected moving-ball fixture, **not** a proposed V7 trajectory-parity acceptance threshold. The initial overly strong assertion of exact cached-state identity failed; the diagnostic was corrected to measure actual cache differences and separately establish offsets. That discrepancy remains reported above.

Installed Arena keyword parsing accepted `num_states=12` and rejected `num_ticks=12`, consistent with the checked-out parser and contrary to the bound method's docstring spelling.

### BallPredictor initialization and updates

- Four fresh constructor-initialized cases, strides 1/2/3/8 with N=12 and elapsed ticks=0, matched independently stepped states exactly at i×stride.
- Three further fresh instances with elapsed ticks 1/2/12 and N=12, stride 1, also matched current-input-origin samples exactly. On first full regeneration, the elapsed-tick argument did not shift the output origin ahead of the supplied current state.
- A fixed-size predictor was updated with elapsed ticks 0, then 2, then 12. All returned sample offsets were 0…11 relative to each supplied current state. Maximum reference differences were 0, 0.0001220703125 and 0.0000019073486328125 respectively.
- A large input-state replacement at zero elapsed ticks fully regenerated prediction and matched exactly.
- A subsequent 0.1-UU input-position change at zero elapsed ticks reused the previous cache. Supplied x was 100.0999984741 while sample-zero x remained 100.0: difference **0.0999984741 UU**. This is consistent with the source's matching margins; no automatic input correction was made.
- A keyword-constructed zero-spin state matched exactly. A default state followed by `.pos`/`.vel` assignments did **not**: Python getters showed position z=1200 and nonzero velocity, but the predictor's first state had z≈93.1500015 and zero velocity. This reproduces the earlier default-position discrepancy in a separate safe test without an unsafe interval call. Maximum reference difference was about **1113.0261**. Its mismatch is saved, not hidden behind an initialization workaround.

Thus basic predictor initialization is not universally unreliable; its input construction/mutation path matters. These tests do not certify all constructors, all cache histories or all arena configurations.

## Sampling versus timestamp contract

Native `BallState` results do not supply RLBot-style absolute `game_seconds`. The measured output contract is a sequence of **integer physics-tick offsets**: 0, stride, …, (N−1)×stride, relative to the supplied/caller current state on full prediction.

The arena configured for 120 Hz reports a float32 `tick_time` of **0.008333333767950535** and reciprocal `tick_rate` of **119.99999237060547**. Those are measured native values. An external absolute timestamp origin and a declared clock-labeling rule remain necessary; the runner does not invent or attach a production time array. Integer offsets are established, while floating-point clock labeling must be reviewed against the existing canonical 120-Hz design before the V7 fixture adapter is frozen.

The original teacher assumes 120-Hz list indexing. A stride>1 list cannot simply be passed to that unchanged selector as though every element were one tick apart. At stride 1 with offset-zero sample, an exact two-second index of 240 needs at least 241 samples. This is a representation constraint, not a production-provider choice, and floating-point first-time arithmetic still follows the already established teacher contract.

No interpolation, lookup modification, timestamps copied from future realized trajectories, stale cutoff, output trimming or substituted state was introduced. No RLBot/RocketSim predictor-accuracy comparison was made.

## Preservation, files and verdict

Created:

- `training/v6_prediction_test/probe_native_sampling.py`
- `training/reports/v6_native_sampling_20261004.json`
- `training/reports/v6_native_sampling_20261004.md`

The established teacher selector code and JSON results were hashed before/after and remained unchanged. All 43 protected hashes remained unchanged. The installed native binary's hash remained unchanged. Retrieved upstream files were read and verified, not edited. Neither bot directory was modified. The user performed the source checkout; no additional download or install was performed by the diagnostic.

Final bounded probe measured approximately **3.29 seconds** internally. Reproduction, if wanted:

```powershell
& '.\training\venv\Scripts\python.exe' -B '.\training\v6_prediction_test\probe_native_sampling.py'
```

The runner reports measurements and remaining limitations, **not an overall V6 PASS**. See [JSON evidence](v6_native_sampling_20261004.json).

**Verdict:** tested fresh/fixed-size sample offsets are established, and significant initialization/cache limitations are now characterized. Exact build-commit provenance remains unknown. Under the user's explicit instruction to stop if provenance cannot be established or native API limitations remain unresolved, **V6 remains incomplete**. No production API/configuration/workaround was chosen. The next decision is whether this pinned binary plus empirical evidence is acceptable provenance and which limitations may be explicitly constrained; those choices require user review. V7, demonstrations, BC, PPO and the student remain unstarted.

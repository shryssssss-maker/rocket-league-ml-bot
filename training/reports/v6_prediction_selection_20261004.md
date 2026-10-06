# V6 prediction selection and timestamp investigation

Date: 4 October 2026. **Teacher-selector portion: PASS. Overall V6: INCOMPLETE — native sampling investigation pending.** Do not mark the entire V6 gate passed or proceed to V7.

Specification: [student contract design](student_contract_design_20261004.md), section 7, V6. Recovered reference revision verified: `fd061f457bf19175b4a9b3b3d7811a987044c64d`.

## Files created so far

- `training/v6_prediction_test/run_v6.py`: independent selector and original-source tests.
- `training/reports/v6_prediction_selection_20261004.json`: all selector cases and measured outcomes.
- `training/reports/v6_prediction_selection_20261004.md`: this interim report.

Both bot directories remain untouched; all 43 protected source/configuration hashes matched. Earlier gates were not rerun. No match, student, demonstrations, dataset generator, BC, PPO, installation, upgrade or production adapter was created. A separate short native sampling discovery command did instantiate RocketSim arenas/predictors; its results are discussed below. No provider-accuracy comparison was made.

## Exact original selection contract

Source: `python-example-original/src/bot.py:MyBot.get_output` and `src/util/ball_prediction_analysis.py:find_slice_at_time`.

1. The normal far-ball branch requests `packet.match_info.seconds_elapsed + 2` only when full 3D car-to-ball distance is strictly greater than 1500.
2. The helper reads `ball_prediction.slices[0].game_seconds`.
3. It calculates `index = int((requested_time - first_slice_time) * 120)` using Python integer conversion, which truncates toward zero.
4. It returns `slices[index]` if `0 <= index < len(slices)`; otherwise it returns `None`.
5. It does not interpolate, search actual slice timestamps, round to nearest, use floor, clamp an out-of-range index, or reject a message on age alone.
6. The selected slice's own `game_seconds` is its actual declared timestamp. It is not overwritten with the requested time. The selected position comes directly from that slice.
7. `None` causes the original teacher to retain the current ball as its Chase target. An empty slice list instead raises `IndexError` at the initial `[0]` access when the helper is reached.

The original missing-ball and pending-sequence early returns bypass prediction lookup. A near-ball normal callback also bypasses it. An empty prediction therefore does not necessarily crash every callback, but it must not be silently relabeled as a successful fallback on a far-ball lookup.

## Tests and results

The independent `run_v6.py:reconstructed_select` is compared directly with the original helper. Numbered positions identify the selected index independently of timestamp equality. Predictions and teacher packets are tested as both native objects and actual RLBot `CorePacket` round trips.

| Test group | Cases | Result |
|---|---:|---|
| Helper: first times 0/12/500 s, lengths 0/1/239/240/241/720, positive/negative and one-ULP boundary requests | 1,152 | PASS |
| Helper: deliberately irregular slice timestamps | 2 | PASS; confirms nominal indexing rather than timestamp search |
| Teacher: both teams, clocks 0/12/500 s, first-slice offsets 0/1/2 ticks and stale offsets -1 tick/-1 s/-5 s, lengths 0/240/241/720 | 288 | PASS |
| Empty-prediction lookup bypass: near ball, missing ball, pending sequence; both representations | 6 | PASS |

Total helper cases: **1,154**. Helper outcomes: 658 selected slices, 304 `None` results, 192 expected `IndexError` cases. Teacher outcomes: 128 selected slices, 88 current-ball fallbacks, 72 expected `IndexError` cases. Expected exceptions are explicitly compared and retained; they are not suppressed as successful selections.

Every selected index, timestamp and position agrees exactly with the independent selector. Teacher cases also verify the exact requested time, actual selected target/current-ball fallback, and resulting steering. The protected-file count is **43**, all unchanged.

### Important edge cases

- A request half a tick before the first slice has a negative fractional scaled index. `int` returns 0, whereas `floor` would return -1. Both original and reconstruction select the first slice. Do not add a negative-time cutoff while claiming exact source behavior.
- At current time 12 s with first-slice offset two ticks, the direct-object case selected index **237**, while the wire-rounded case selected **238**. Each reconstructed selector matched its original path exactly. Nominal arithmetic such as `240 - offset_ticks` is not a substitute for the actual expression on the delivered timestamps.
- In a first-slice offset of minus one tick at current time 12 s, the wire case selected a slice timestamp of approximately **13.9916667938 s**, giving an actual horizon of approximately **1.9916667938 s**. Feature 10 must encode selected time minus current elapsed time, not a hard-coded 2.
- A deliberately irregular message still selected index 240 for a request at 14 s with first time 12 s, even though that slice's timestamp was 17 s. This diagnostic establishes what the helper does; it does not approve irregular provider output.
- A stale message can still yield a valid selection if its range covers the computed index. There is no source age threshold. Stale messages beyond range produce `None` and current-ball fallback.

For the proposed observation, first-slice offset means `(first_slice_time - current_elapsed)/(1/120)`, and prediction horizon means `(selected_slice.game_seconds - current_elapsed)/2`. These retain the actual supplied time values. Missing/invalid prediction uses the previously declared masks and zeroing rules; an original teacher exception remains an exception case, not a valid demonstration.

## Native RocketSim sampling discovery — unresolved

Installed authorities inspected: `training/venv/Lib/site-packages/RocketSim.pyi`, bound method docstrings, and `rlgym/rocket_league/sim/rocketsim_engine.py` mesh initialization. Version: RocketSim 2.2.1. The methods expose:

```text
Arena.get_ball_prediction(num_ticks=120, tick_interval=1)
BallPredictor.get_ball_prediction(ball_state, ticks_since_last_update,
                                  num_states=120, tick_interval=1)
```

A short native command used a Soccar arena and ball position (0,0,1200), velocity (600,200,100), without cars. Observations:

- `Arena.get_ball_prediction(12,1)` returned 12 states; its first position matched the current ball position. Subsequent positions advanced, and the caller arena's tick count remained 0. A reference-step comparison is still needed before declaring all sample offsets.
- For intervals 2 and 3, the returned length was still 12 while early samples appeared farther apart in time.
- For interval 8, a trailing early inspected sample contained implausible subnormal coordinates, rather than a plausible continuation. The returned length cannot currently be treated as proof that every entry is usable. The cause has not been established from source.
- A fresh `BallPredictor` called with the supplied standalone `BallState`, `ticks_since_last_update=0`, 12 states and interval 1 returned default-position samples near z=93.15, not the supplied z=1200. Initialization/cache/update-counter semantics must be understood before assigning blame or selecting this API.

These are observations from a discovery command, not certified provider output or a diagnosis of an exact C++ defect. No subsequent unsafe interval sweep was performed. **Do not use these results to fabricate timestamps, silently trim unknown tails or choose a production predictor.**

The package identifies the [mtheall RocketSim binding repository](https://github.com/mtheall/RocketSim) as its Python binding reference. Browser retrieval of the relevant C++ file contents failed. The user was supplied a source-only PowerShell download command that pins the resolved repository revision and retrieves the Arena/BallPredictor implementations into an isolated `training/reports/v6_upstream_source_<revision>/` folder. No dependency build or installation is requested. A current repository snapshot is explanatory evidence, not automatically proof of the installed binary's exact build revision.

## Remaining V6 work and stop point

After the requested source download:

1. Read binding/core implementations for list length, sampling stride and cache initialization/update rules.
2. Compare only verified-safe native sampling calls with separately stepped car-free RocketSim reference states, establishing first sample and subsequent tick offsets empirically.
3. Establish how to attach supplied canonical game-time timestamps to verified native states without claiming native `BallState` provides timestamps.
4. Report any installed binding limitation and ask before making a production API/workaround decision.

The completed teacher portion reproduces the recovered source exactly. Native first-sample and `tick_interval` semantics have not yet passed their gate. **V6 remains incomplete.** V7 and all ML work remain unstarted.

Teacher-only diagnostic reproduction (already executed):

```powershell
& '.\python-example\venv\Scripts\python.exe' -B '.\training\v6_prediction_test\run_v6.py'
```

It reports teacher-selector PASS and explicitly leaves native sampling pending. See [measured JSON evidence](v6_prediction_selection_20261004.json).

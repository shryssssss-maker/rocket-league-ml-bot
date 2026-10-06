# V4 temporal sequence equivalence report

Date: 4 October 2026. **Result: PASS**, with the explicitly approved mixed-schedule limitation below. V4 is safe to mark passed for the tested schedules and independent temporal reconstruction. V5 and later gates were not run.

Specification: [student design report](student_contract_design_20261004.md), section 7, V4. Reference revision verified: `fd061f457bf19175b4a9b3b3d7811a987044c64d`.

## Files created and isolation

- `training/v4_temporal_test/run_v4.py`: independent temporal reconstruction and diagnostic runner.
- `training/reports/v4_temporal_equivalence_20261004.json`: numerical results, bounded transition examples, state snapshots, provenance and hashes.
- `training/reports/v4_temporal_equivalence_20261004.md`: this report.

Neither bot directory was modified. All 43 protected source/configuration hashes matched before and after execution. The isolated V3 stubs and geometry convention were imported without running V3. V0 was not rerun. No model, demonstrations, dataset generator, training, RocketSim stepping, dependency installation or Rocket League launch was involved.

## What was compared

The source path directly executes the unchanged recovered `MyBot.get_output`, `begin_front_flip`, `Sequence.tick` and `ControlStep.tick`. A fresh original instance is created per schedule, and its state persists across that schedule's callbacks. The socket-owning constructor is bypassed; rendering, boost-pad bookkeeping and outgoing quick chat are stubbed. Gameplay decisions and temporal methods execute from original source.

The candidate's `run_v4.py:Step`, `Sequence` and `Reconstruction` independently implement temporal state. **They do not call the original sequence/step methods.** The four durations and controller combinations are reconstructed from `python-example-original/src/bot.py:MyBot.begin_front_flip`:

| Phase / index | Nominal duration | Controller differences from Neutral |
|---|---:|---|
| Jump / 0 | 0.05 s | jump=true |
| Release / 1 | 0.05 s | None |
| Front dodge / 2 | 0.20 s | jump=true, pitch=-1 |
| Coast / 3 | 0.80 s | None |

For every scheduled callback, assertions compare:

1. All eight controller channels, with finite analog outputs and exact Boolean button types/values.
2. Sequence presence, index, completion flag, all four step start times and durations.
3. Flip-start counts, including suppression of new flip triggers while a sequence is pending.
4. The strict measured predicate `elapsed_since_step_start > duration`.
5. Returning the old phase's controls on its finishing callback.
6. Leaving the next step's start time unset until its first callback.
7. Returning to ordinary Chase or starting a newly triggered flip on the callback after sequence completion.

Physics and velocity change between synthetic callbacks, including new speeds inside the flip-trigger range while a sequence is active. Ball presence remains true and distance remains below 1500, keeping missing-ball transitions and prediction lookup outside V4.

## Schedule coverage

There were **138,494 scheduled callback comparisons across 29 runs**. Each of the seven regular schedules was tested with both teams and two representations: direct native objects and actual RLBot `CorePacket` round-trip packets. The extra explicit equality trace used direct objects.

| Schedule family | Runs | Callback comparisons |
|---|---:|---:|
| Constant 1-, 2-, 3-, 4-, 5-frame gaps; 721 callbacks per run | 20 | 14,420 |
| Concatenated retained live intervals; 545 callbacks per run | 4 | 2,180 |
| Synthetic schedule matching the complete live gap histogram; 30,470 callbacks per run | 4 | 121,880 |
| Explicit equality / finishing / next-callback trace | 1 | 14 |
| **Total** | **29** | **138,494** |

Regular frame gaps are represented through elapsed game time at a 120 Hz physics base. No wall-clock sleeps or fixed 60 Hz decision loop are used. The explicit duration tests additionally use precisely constructed times; their purpose is strict equality, rather than a physical callback cadence claim.

Additional checks, outside the scheduled callback count:

- **12 strict-duration anchors**: for each of the four steps, test the representable time immediately below duration, exact duration, and the representable time immediately above duration, with a start time of zero. Expected completion is false, false, true. Returned controls and start times must also match. These equality anchors use direct objects so float32 transport cannot erase the one-ULP distinctions.
- A same-current-packet/different-history demonstration: a fresh teacher chases while a teacher with a pending Jump returns Jump controls.
- A stronger **identical-18D observation** demonstration, described below, with both histories also checked against the independent reconstruction.

## Mixed schedule provenance and approved limitation

Source session: `training/live_reference_test/sessions/20261004_055830_563928/`, using `bot_summary.json` and `bot_events.jsonl`. Their SHA256 hashes are recorded in the V4 JSON.

The original live diagnostics did **not** save every callback's chronological interval. The user explicitly approved using both available proxies with this limitation documented:

1. **Retained intervals:** 544 logged individual intervals, taken in event-log order and concatenated. Their histogram is gap 1: 11, gap 2: 523, gap 3: 9, gap 4: 1. Actual retained `game_dt` values are used, preserving their measured numerical precision. Concatenation does not reconstruct adjacency or original absolute times between logged events.
2. **Full histogram surrogate:** all 30,469 gaps from the live summary, shuffled deterministically with seed `20261004`. Counts are gap 1: 874, gap 2: 28,996, gap 3: 518, gap 4: 67, gap 5: 14. The runner asserts exact histogram agreement and uses gap/120 for its synthetic time deltas.

Neither proxy is a replay of the complete original callback order. V4 passes the agreed proxy coverage; it does not claim recovery of evidence that was never logged.

## Results

| V4 requirement | Result | Evidence |
|---|---|---|
| Eight controller outputs agree | PASS | Maximum analog error **0.0**; exact buttons on every scheduled callback |
| Sequence index and presence agree | PASS | Exact state comparison on every callback |
| Completion flags agree | PASS | Exact Boolean/state agreement |
| All step start times and durations agree | PASS | Exact values, including unset future start times |
| Strict elapsed > duration | PASS | Per-callback invariants plus all 12 equality anchors |
| Finishing callback retains old controls | PASS | Checked at every transition |
| Next phase starts on its next callback | PASS | Future timer remains unset on prior phase completion |
| Normal decision resumes after completion | PASS | Checked in every run |
| Pending sequence suppresses fresh trigger/Chase | PASS | Varying speed/target with exact controls and flip counts |
| Constant 1–5-frame schedules | PASS | All 20 representation/team runs |
| Approved mixed proxies | PASS | All eight runs, provenance retained |
| Protected source/configuration files unchanged | PASS | 43 hashes |

Across scheduled runs there were **1,911 flip starts and 1,883 completions**. These are synthetic test counters, not physical flips or gameplay metrics. Twenty-eight regular runs intentionally stop after a fixed callback count with a final sequence still pending; the explicit equality trace finishes its sequence. No ending reset or forced completion was introduced.

The JSON stores at most the first 24 state-changing examples per run, plus the last transition and final state. It records total transition counts and a SHA256 digest over every state-changing event. **Every callback was asserted**, even when its example was not saved. This keeps diagnostics bounded rather than retaining a large imitation dataset.

## Exact temporal behavior established

`ControlStep.tick` starts a timer on that step's first delivered callback, not on the prior phase's completion. A step completes only when measured elapsed time is strictly greater than its duration. `Sequence.tick` increments its index but returns that old step's controls on the same callback. The following callback initializes the next phase. Coast completion similarly returns Neutral; normal decision logic is reached on the next callback.

Consequently the 1.10-second sum of nominal phase durations does not define a fixed total sequence wall/game-time duration. In the first direct-object sequence of each constant-gap run:

| Gap | First Coast completion time, starting at t=0 |
|---:|---:|
| 1 frame | 1.1583333333333334 s |
| 2 frames | 1.2166666666666666 s |
| 3 frames | 1.275 s |
| 4 frames | 1.3 s |
| 5 frames | 1.3333333333333333 s |

These are measurements for the supplied schedules, not universal upper bounds.

Float32 timestamp transport can also move a strict transition relative to an unrounded synthetic clock. For example, the first 3-frame-gap sequence completed at 1.275 s with direct timestamps and at approximately 1.2000000477 s after wire conversion. **Each original/reconstruction pair matched exactly on its shared actual packet timestamps.** V4 does not require or claim identical transition timing between differently rounded clocks. A future simulation adapter must preserve the declared delivered timestamp semantics; rounding decimal durations, counting an assumed number of 60 Hz decisions, or smoothing callback time would change reference behavior.

## Why temporal memory is required

At current elapsed time **1.0 s**, two source histories have the same current packet, previous callback time **0.99 s**, previous transmitted **Neutral**, and previous steering **0**. Their current 18D observations, built by the isolated V2 builder with prediction invalid, are identical.

- History A's Release has just finished. The current callback starts Front dodge: pitch=-1 and jump=true.
- History B is still in Coast. The current callback returns Neutral.

The histories, identical vector, distinct controls and sequence states are recorded in `identical_observation_witness` in the JSON. The independent reconstruction agrees on both histories.

Thus the current 18D vector, even with previous-action and callback-delta fields, does not uniquely determine the teacher's action. The necessary history includes progress through Jump/Release/Dodge/Coast and elapsed time since a phase's first-use callback. No private teacher sequence index or timer was added to the observation. This supports evaluating recurrent memory or sufficient history later; **it does not prove the proposed GRU will learn or retain that state**.

## Discrepancies and reproduction

No final control or temporal-state discrepancies remained. An initial harness setup attempt tried to assign an installed read-only `MatchInfo` field; the fixture construction was corrected to create a new native `MatchInfo` object. This was a test setup error, not a teacher behavior discrepancy; no reference source or acceptance tolerance changed.

Executed using the existing Python 3.12.10 / RLBot 2.0.0b55 environment. The final bounded diagnostic measured approximately **2.71 seconds** internally:

```powershell
& '.\python-example\venv\Scripts\python.exe' -B '.\training\v4_temporal_test\run_v4.py' --approved-mixed-proxies
```

The flag records the user's approval of proxy coverage. Without it, the runner performs only fixed schedules and reports partial status rather than claiming the full agreed V4 gate. Reproduction is optional; the approved final command already passed.

See [machine-readable results](v4_temporal_equivalence_20261004.json) and [isolated runner](../v4_temporal_test/run_v4.py).

**Stop at V4.** Missing ball, replay/kickoff/goal transitions and interrupted-sequence long-gap behavior remain V5. Prediction indexing/providers, latency, physical execution and learned memory remain later gates. No demonstration pipeline or ML implementation has started.

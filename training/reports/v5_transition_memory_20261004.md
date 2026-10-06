# V5 missing-ball and transition-memory report

Date: 4 October 2026. **Result: PASS.** V5 is safe to mark passed for the tested synthetic transitions and interrupted sequences. Stop at V5; V6 and later gates were not run.

Specification: [student contract design](student_contract_design_20261004.md), section 7, V5. Recovered reference Git HEAD verified: `fd061f457bf19175b4a9b3b3d7811a987044c64d`.

## Files created

- `training/v5_transition_test/run_v5.py`: isolated V5 diagnostic and missing-ball extension to the independent V4 reconstruction.
- `training/reports/v5_transition_memory_20261004.json`: measured results, interrupted/resumed states, bounded examples, trace hashes and code hash.
- `training/reports/v5_transition_memory_20261004.md`: this report.

Neither `python-example/` nor `python-example-original/` was modified. All 43 protected source/configuration hashes matched before and after execution. Existing V3/V4 helpers were imported without running their experiments or changing their files. No match, physics simulation, model, dataset generator, BC, PPO, dependency installation or deployment integration was performed.

## Source basis and independent comparison

The original path executes unchanged `python-example-original/src/bot.py:MyBot.get_output` and `begin_front_flip`, plus `src/util/sequence.py:Sequence.tick/ControlStep.tick`. A source instance persists throughout each trial. Its socket-owning constructor is bypassed; rendering, boost bookkeeping and outgoing communication are diagnostic stubs.

The source checks `len(packet.balls) == 0` before the pending-sequence branch. It returns a default Neutral controller without calling `Sequence.tick`. It does not consult match phase or score. `ControlStep.tick` preserves its first-use timestamp and uses the strict predicate `current_elapsed - start_time > duration`. `Sequence.tick` advances at most one phase per callback and returns that phase's controls even when it finishes.

`run_v5.py:Reconstruction` adds only the matching explicit missing-ball early return to the independent V4 temporal reconstruction. V4's reconstructed `Step` and `Sequence` remain independent of the original temporal methods. No goal, replay, kickoff or position-reset rule was invented.

## Exact coverage

**472 trials and 26,260 paired callback comparisons**, using both teams and both direct native objects and actual RLBot `CorePacket` round-trip packets:

| Trial group | Cases |
|---|---:|
| Seven pending-state prefixes × four interruption durations × four interruption types × two teams × two representations | 448 |
| Fresh Chase or completed sequence × three ball-availability patterns × two teams × two representations | 24 |
| **Total** | **472** |

The seven prefixes represent Jump already started; Release, Dodge and Coast each before and after its own timer starts. An unstarted Jump is not manufactured: the original starts Jump immediately when triggering a flip.

Interruption durations: **0.1, 1, 5 and 10 seconds**. Each prefix/duration tests:

1. Missing-ball callbacks during Active play, followed by a present ball.
2. GoalScored → Replay → Countdown → Kickoff callbacks with ball absent, followed by an Active callback with ball present.
3. The same phase progression with ball present throughout.
4. No delivered callbacks during the interval, followed by a ball-present callback after the long gap.

Four intermediate interruption callbacks are supplied at 10%, 40%, 70% and 100% of the duration where applicable. The returning callback is one physics-tick interval after the specified duration. Position/orientation changes model kickoff repositioning, and score metadata increments to represent a goal. The ball remains within 1500 UU whenever present; prediction lookup and provider behavior are excluded.

Every interrupted trial subsequently drains the pending sequence with ball-present callbacks, checks normal Chase on the callback after completion, and checks a new speed-triggered flip replacing the completed sequence. The new flip is a separate decision after completion, not a reset during the interruption.

The 24 phase-matrix cases exercise GoalScored, Replay, Countdown, Kickoff, Active, Paused, Ended and Inactive with all-present, all-missing and alternating ball presence. They start either without a sequence or with a completed sequence retained. These synthetic combinations test the source's response to packet metadata, not which combinations the live server normally emits.

## Results by requirement

| Requirement | Result | Evidence |
|---|---|---|
| Ball unavailable → Neutral | PASS | All **992 missing-ball callbacks** returned exact Neutral |
| Missing-ball callbacks preserve state | PASS | Sequence presence/index/done, every start time, flip count and sequence object identity unchanged |
| Ball return uses surviving sequence | PASS | Exact resumed controls and state; started/unstarted timers distinguished |
| Goal and immediate subsequent callbacks | PASS | Goal/replay progression tested with and without ball, pending sequence and ordinary Chase |
| Kickoff and repositioning | PASS | Phase/spawn changes caused no invented sequence reset |
| Replay/post-goal state survival | PASS | Exact comparison across all tested phases and availability patterns |
| Long callback gaps in every phase | PASS | 0.1/1/5/10-second interruptions; at most one phase advanced on return |
| Same eight controller outputs | PASS | Maximum analog difference **0.0**; button values/types exact |
| Same sequence state and transitions | PASS | Exact per-callback snapshots and explicit transition invariants |
| Normal decisions after completion | PASS | Chase followed by an independently fresh flip trigger in all 448 interrupted trials |
| Protected source/configuration unchanged | PASS | All **43 hashes** matched |

Other synthetic counters: 908 flip starts, 460 sequence completions and 1,840 step finishes. These include setup and deliberate post-completion triggers; they are not gameplay statistics. Interrupted trials deliberately end after validating a new Jump, rather than pretending that final new sequence has completed.

All callbacks are asserted. The JSON retains up to 12 representative interruption/return/completion events per trial, the interrupted and resumed states, final state and a SHA256 digest over the entire trace. No demonstration dataset is saved.

## What survives and what resumes normal decisions

**Missing ball does not reset or tick the sequence.** It leaves the index, completion flag and start times untouched while returning Neutral. The game clock still advances in the supplied callbacks; elapsed time is not subtracted from an already-started step's timer.

On the first ball-present callback:

- An already-started step uses its old start time. If elapsed time now exceeds its duration, it advances one index while still returning its old controls.
- An unstarted step begins its timer on this returning callback. Time spent waiting before first use does not consume that step's duration.
- Even a very long interruption does not skip all remaining phases or immediately return to Chase.
- If Coast finishes, that callback still returns Neutral. Normal Chase or a fresh flip decision happens on the next ball-present callback.

Representative direct-object results after a **10-second no-callback gap**:

| Interrupted state | Index after returning callback | Returned controls |
|---|---:|---|
| Jump started | 1 | Jump |
| Release unstarted | 1 | Neutral; Release timer starts now |
| Release started | 2 | Neutral |
| Dodge unstarted | 2 | Front dodge; Dodge timer starts now |
| Dodge started | 3 | Front dodge |
| Coast unstarted | 3 | Neutral; Coast timer starts now |
| Coast started | 4, done=true | Neutral; normal decision deferred until next present callback |

Goal, replay, countdown, kickoff, score changes and position changes do not clear the source sequence. With the ball present, even a Replay callback ticks the sequence; with the ball absent, even an Active callback returns Neutral without ticking it. Availability and sequence completion, rather than phase names, determine these source behaviors.

A completed sequence object also remains attached during ordinary Chase and missing-ball callbacks. A subsequent legitimate speed trigger creates a new sequence. This is distinct from clearing temporal state at a goal or kickoff.

## Implications for the later contract

The reference temporal state must survive ordinary goals, replays and kickoff repositioning within the same controlled-agent lifetime. A future data/teacher adapter must not reset the sequence or replace a stale first-use timestamp merely because a phase changes. Missing-ball Neutral controls do not mean the teacher has forgotten its pending phase.

This confirms the reconstruction behavior, not the reliability of learned recurrent memory. No GRU or student is implemented. How a learned hidden state tracks these histories remains a later pilot/evaluation question.

## Limits, discrepancies and reproduction

No final action or state discrepancies were observed. Direct-object and wire-represented packets are each compared against the reconstruction using identical actual timestamps; the two representations are not asserted to have identical boundary timing after rounding.

All tests retain one agent lifetime and monotonically increasing game time. Repositioning means synthetic post-goal/kickoff physics changes, **not** a new agent, a process restart, controlled-car identity replacement or backward clock. Those lifecycle conditions are not silently interpreted as source reset rules. No live transition timing, collision behavior, absent-car handling or prediction-service compatibility is established by this diagnostic.

The short diagnostic ran in the existing Python 3.12.10 / RLBot 2.0.0b55 environment, measuring approximately **0.70 seconds** internally:

```powershell
& '.\python-example\venv\Scripts\python.exe' -B '.\training\v5_transition_test\run_v5.py'
```

Expected result: `V5 PASS: 472 trials, 26260 callbacks; maximum analog difference 0.0.` Reproduction is optional; the corrected bounded-output runner has already passed.

See [JSON evidence](v5_transition_memory_20261004.json) and [isolated V5 runner](../v5_transition_test/run_v5.py).

**Stop at V5.** V6 prediction indexing/timestamps and all later gates remain unexecuted. Demonstrations and all ML work remain unstarted.

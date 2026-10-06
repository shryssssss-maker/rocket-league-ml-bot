# Phase 3 launch readiness

**Prepared only. No live launch. Phase 3 has not passed.**

Non-live checks: passed_nonlive. All 43 protected hashes unchanged; completed V10/V12/V13 artifacts unchanged: True.

## Control and memory contract

The live wrapper returns only original MyBot.get_output. The independent shadow invokes the same original methods on the same native packet/prediction/time, with its own sequence and boost tracker. It has no socket; its sink discards cosmetic render/quickchat and rejects controls/state requests.

The existing training Python hosts the accepted checkpoint on CPU. Its only input is callback index plus 13 float32 features. No private teacher state/history is sent. GRU memory persists across all phases and missing-ball callbacks, and resets only at session/match initialization.

The unchanged frozen observation builder supplies columns 0–12. The observation selector probe follows the frozen recorder’s nonempty-prediction guard and original helper; the teacher’s far-ball branch retains its original empty-prediction IndexError. No interpolation, shifting, clamping, timestamp correction or simulator is used.

## Bounded real-game probes

Orange original teacher, blue human. Keep hands off during probes: both cars and the ball can move following state-setting requests. At most 24 requests, 6 explicitly logged pauses, 300 seconds after the first callback plus 180 seconds startup. Two phase-specific pauses use 1.25 seconds; pending countdown probes use 5 seconds. Pauses occur only after teacher controls are submitted. Requests alone never count as coverage.

Completion requires every explicit callback evidence category and two sequence completions, including an actual student disagreement at a teacher sequence start and subsequent independent shadow continuation. If such cases do not occur, the run remains incomplete; no forced student output or fabricated evidence is allowed.

## Required evidence

- Neutral->Chase
- Chase->Jump
- Release
- Front dodge
- Coast/completion
- At least two starts/two completions
- Missing ball with pending sequence
- Same pending sequence across goal/replay/kickoff
- Actual long gap >=1 second
- Near/far branches
- Actual selector index/timestamp/position
- Exact native eight-channel submission
- Student misses a teacher sequence start
- Teacher dodge after student prior jump=False
- Independent shadow continuation after student divergence

## Limits

- Only actual processed RLBot callbacks; not lossless delivery or equivalence on undelivered physics ticks.
- Receive pauses can cause queue-full warnings; IPC/profiling/logging can change delivered callback scheduling.
- Teacher drives; student outputs never reach the game. No learner-induced physical-state coverage is established.
- Shadow is an independent original-source actor; exact equivalence does not prove learned sequence competence.
- Requested probe states do not count as evidence; only actual later packets/sequence transitions do.
- Observation prediction probe is distinct from whether the original gameplay branch consulted prediction.
- Diagnostic callback logs are not a DAgger training dataset; no tensor export, aggregation, fitting or test evaluation.

## Launch after review

```powershell
& '.\python-example\venv\Scripts\python.exe' -u -B '.\training\behavior_cloning\v13_dagger\phase3_shadow_v1\launch.py' --run
```

Expected: session path, original-teacher-only notice, periodic callbacks/flip/missing-category counts, bounded stop status and diagnostic contract-report paths. Ctrl+C produces an incomplete report. No DAgger dataset, fitting, test-set evaluation or student control follows.

Prepared configs: `C:\Users\shreyas\Desktop\model wars\training\behavior_cloning\v13_dagger\phase3_shadow_v1\prepared\20261005_042717_381938`. Every later launch creates a new session; existing outputs are never overwritten.

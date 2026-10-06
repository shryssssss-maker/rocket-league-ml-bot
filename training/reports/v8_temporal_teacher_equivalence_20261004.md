# V8 temporal teacher equivalence

**Status: PASSED WITH SCOPE — accepted by the user for temporal equivalence on actual processed RLBot callbacks.**

Live callbacks: 1374. Comparison failures/errors: 0. Original/candidate sequence starts: 3/3.
Maximum analog difference: 0.0; established tolerance: 1e-6. Sequence phase/index, done/start times, branch, selected index/timestamp/position and buttons require exact equality.

| Required category | Actual live evidence present |
|---|---|
| 1 Neutral -> Chase | Yes |
| 2 Chase -> Jump | Yes |
| 3 Jump release | Yes |
| 4 Front dodge | Yes |
| 5 Coast and completion | Yes |
| 6 Multiple sequences | Yes |
| 7 Missing ball while sequence pending | Yes |
| 8 Goal/replay/kickoff while sequence pending | Yes |
| 9 Long callback gaps while pending | Yes |
| 10 Near and far branches | Yes |
| 11 Actual live selector | Yes |
| 12 Native eight-channel controllers | Yes |

All categories observed: True. Protected hashes unchanged: 43.

## Method

The source MyBot initialization/get_output and original Sequence/helper functions execute unmodified on actual RLBot native callbacks and the current delivered BallPrediction. A passive sys.setprofile observer records source helper return locals and consumed phase; no source function/global is replaced. The independent candidate sees the same native objects sequentially and keeps its own temporal state. Only source ControllerState is returned to RLBot.

Both sequence snapshots are logged before/after every comparison callback, with emitted output modes and consumed phases kept distinct. The finishing callback returns old-phase controls while index/done may already advance. Missing-ball callbacks preserve both pending states. Previous-action observation provenance is the actual prior source control submission, not candidate output.

Approved controlled interventions request near/far positions, speed-triggered flips and a real goal while a sequence is pending. Actual phase/ball callbacks determine coverage, never requested state. Two labeled 1.25-second pauses and a five-second pending-sequence pause over real countdown occur after source controls are sent; no intermediate ticks/packets are invented. Replay and countdown are retained.

Long gap evidence requires at least one second of actual canonical callback delta with an already-pending sequence. Multiple-sequence coverage requires two observed starts and two completions. Goal, replay and kickoff each need separate pending-sequence evidence linked to the same observed sequence identity.

## Evidence and limitations

Session: C:\Users\shreyas\Desktop\model wars\training\v8_temporal_test\sessions\20261004_205108_820323. Full per-callback evidence: callbacks.jsonl; interventions: events.jsonl; metadata/config hashes: metadata.json. Required-category examples and every failure are included in the JSON report.

Coverage can remain incomplete if real transitions do not overlap the sequence as requested. Do not fabricate missing-ball/phase states, search for synthetic substitutions, or infer success from requests. Mismatch/error, natural match end, user interruption or the bounded supervisor timeout stops the test. No candidate deployment or ML follows.

## Exact implementation

- training/v8_temporal_test/candidate.py: independent Candidate/Sequence.
- training/v8_temporal_test/live_agent.py: LiveHarness.observe_original/get_output/cover/plan.
- Unmodified source: python-example-original/src/bot.py and src/util/sequence.py, ball_prediction_analysis.py.
- Auxiliary declared observation: training/v2_observation_test/observation_contract.py:live_adapter.
- Both protected bot directories remain untouched.

## Post-capture evidence review

All 1,374 callback rows were audited: native eight-channel controls and before/after sequence snapshots agree exactly. Both paths received the same actual packet and prediction objects. There were 437 far-ball selector calls, 75 near-ball branches, 131 pending-sequence branches and 731 missing-ball callbacks. All 731 missing-ball callbacks occurred with pending sequence memory and left that memory unchanged. Three sequences started; two completed. The third remained pending when coverage stopped.

Sequence 3 links the required real transitions:

| Phase | Callback | Frame | Canonical elapsed seconds |
|---|---:|---:|---:|
| GoalScored | 641 | 1650 | 13.75 |
| Replay | 840 | 2011 | 16.758333206176758 |
| Countdown | 1372 | 3062 | 25.516666412353516 |
| Kickoff | 1374 | 3644 | 30.366666793823242 |

Actual long callback gaps were 1.0249996185302734 seconds / 123 frames at callback 583 and 4.116666793823242 seconds / 494 frames at callback 1374. These are measured received-callback deltas, not the requested wall-clock pause durations.

The user-supplied terminal log contains outbound queue-full warnings during intentional receive pauses. Therefore this establishes equivalence on processed actual callbacks; it does not establish lossless packet delivery or behavior on unreceived physics ticks. Controlled probes are distinct from natural gameplay. No candidate controls were sent to the game. All 43 protected hashes were rechecked unchanged after the run.

Every required category has explicit evidence. V8 was accepted by the user with the processed-callback scope stated above. This report does not claim lossless packet delivery or equivalence on undelivered physics ticks. Subsequent work is separately authorized; no ML training is implied.

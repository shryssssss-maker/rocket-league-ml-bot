# V9: natural live teacher demonstration pilot

This directory records the **unmodified recovered Python Example teacher** playing real Rocket League through RLBot. It contains no model, optimizer, trainer, candidate policy, RocketSim import, state-setting calls or intentional receive pauses.

## Pilot

Approved opponent: blue human versus orange original teacher. Three complete natural five-minute 1v1 matches, including ordinary replay/countdown and overtime. Run each separately with the existing live interpreter:

```powershell
& '.\python-example\venv\Scripts\python.exe' -u -B '.\training\behavior_cloning\launch.py' --match-id pilot_01
& '.\python-example\venv\Scripts\python.exe' -u -B '.\training\behavior_cloning\launch.py' --match-id pilot_02
& '.\python-example\venv\Scripts\python.exe' -u -B '.\training\behavior_cloning\launch.py' --match-id pilot_03
```

Play normally. The source teacher is the only bot supplying controls. Each command exits after a received natural match-end packet; interrupted matches remain incomplete and are not exported. A 180-second startup bound and 20-minute collection bound prevent indefinite waits; overtime beyond that bound remains incomplete. Existing match directories cannot be overwritten. If a match fails, review its evidence before retrying or deleting anything.

The split is assigned **before collection**: `pilot_01=train`, `pilot_02=validation`, `pilot_03=test`. Every callback belonging to a match stays in that split. Three matches demonstrate pipeline functionality; they cannot establish strong held-out statistical generalization.

## Raw records

`pilots/pilot_20261004/<match>/raw/trajectory.jsonl` contains every `get_output` callback. Fields include actual packet state/time/frame/phase, receipt counters/monotonic times/object identity, original source controls, previous actually submitted controls, the unchanged 18D float32 vector, source selector attempts and selected native slice, and separately the prospective observation prediction probe. Hidden sequence before/after state and consumed phase are **diagnostic labels only**. Match transitions and irregular callback gaps remain explicit.

Prediction data comes exclusively from the actually delivered RLBot `BallPrediction`. No interpolation, timestamp adjustment, clamping, car-free prediction or synthetic slices. The source helper remains unchanged. An empty prediction retains the validated observation invalid-mask behavior; if the original teacher's far-ball branch reaches its empty lookup, its original `IndexError` is logged and propagated. Do not relabel that failed callback as valid training evidence.

Invalid rows remain in raw records and errors remain separately in `raw/errors.jsonl`. No row is silently resampled, repaired or skipped. A match containing invalid rows is rejected from tensor export. Missing-ball rows remain valid when the declared observation/teacher contracts succeed. Private sequence memory is never added to the 18D vector.

Raw state includes car/ball location, velocity, angular velocity and orientation, first prediction time and selected observation position/time. This supports exact feature reconstruction. Full raw prediction trajectories are not stored on each callback; neither selection nor the 18D vector requires the other slices. Receipt identity and teacher/observation selections distinguish reuse and branch eligibility.

## Processed tensors and integrity

`processed/` is separate from raw JSONL. Little-endian typed `.bin` arrays contain `[N,18]` observations, `[N,5]` native analog controls, `[N,3]` exact Boolean buttons stored as uint8 0/1, mode labels, callback/frame indices and float64 times/deltas. Each manifest declares shape, dtype, channel/feature order and SHA256. Initial dt is 0 only for the first callback; raw initial dt remains null. No sequence chunks, history resets or train batches are built yet. Analog action values are not quantized.

Export requires natural match completion, recorder-close acknowledgement, contiguous callback indices, exact bitwise observation reconstruction, supported native actions, matching recorder row counts and matching integrity hashes. Source hashes, configuration, interpreter/packages, recorder and observation-builder hashes are retained. All 43 protected files are checked before/after collection and during reporting.

Configured packet rates and receipt counts do not prove lossless transport. RLBot may coalesce incoming packets before calling `get_output`; the dataset preserves actual **processed callbacks**, not undelivered physics ticks. Logger overhead is measured in the agent summary and must be reviewed after the pilot.

## Non-live preflight and reporting

```powershell
& '.\python-example\venv\Scripts\python.exe' -B '.\training\behavior_cloning\launch.py' --check
& '.\python-example\venv\Scripts\python.exe' -B '.\training\behavior_cloning\launch.py' --report
```

Preflight imports the live stack and compares feature reconstruction against archived actual V8 evidence; it never adds those archived rows to the pilot. Reporting verifies raw/tensor hashes and creates the requested Markdown/JSON pilot reports. Completed user-run matches automatically update the reports. Review all action/branch/phase/timing distributions before approving larger collection. No training or larger run starts automatically.

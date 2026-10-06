# V2 observation parity report

Date: 4 October 2026. **Result: PASS.** V2 is safe to mark passed for the documented 18D candidate and tested adapter paths. This does not freeze the complete student contract or authorize V3 or training.

Specification: [student contract design](student_contract_design_20261004.md), sections 3, 4, 6 and V2 in section 7. Reference: recovered Python Example revision `fd061f457bf19175b4a9b3b3d7811a987044c64d`.

## Files created

- `training/v2_observation_test/observation_contract.py`: isolated shared feature builder and native live/simulation adapters.
- `training/v2_observation_test/run_v2.py`: deterministic diagnostic runner, native-object workers, independent geometry oracle and assertions.
- `training/reports/v2_observation_parity_20261004.json`: measured results, dependency versions, code/fixture hashes and rejected cases.
- `training/reports/v2_observation_parity_20261004.md`: this report.

Neither `python-example/` nor `python-example-original/` was modified. All 43 protected source/configuration hashes in `training/live_reference_test/source_hashes.json` matched before and after execution. No model, dataset generator, training/deployment integration, dependency installation, match launch or physics stepping was performed. Fixtures exist transiently for diagnostics; no demonstration dataset was produced.

## Exact tested observation

Output is 18 finite float32 values, in this order. Local axes are forward/right/up, using full 3D geometry and the original teacher's orientation convention. No team inversion, clipping, running statistics, truncation or padding is applied.

| Index | Feature | Scale / rule |
|---:|---|---|
| 0–2 | Current ball relative forward/right/up | Displacement dotted with own orientation basis, divided by 6000 UU |
| 3–5 | Selected predicted point relative forward/right/up | Same transform and scale |
| 6 | Current 3D car-to-ball distance | Divide by 6000 UU |
| 7 | Own 3D speed | Divide by 2300 UU/s |
| 8 | Ball present | Exact 0 or 1 |
| 9 | Effective selected-prediction validity | Exact 0 or 1; zero if ball absent |
| 10 | Selected prediction time minus current elapsed time | Divide by 2 seconds |
| 11 | First prediction slice time minus current elapsed time | Divide by 1/120 second |
| 12 | Current elapsed time minus previous processed callback time | Divide by 1/60 second; zero on initialization |
| 13–16 | Previously transmitted mode | Exact one-hot: Neutral, Chase, Jump, Front dodge |
| 17 | Previously transmitted steering | Continuous; zero on initialization and non-Chase modes |

Missing ball zeros features 0–6 and 8–11, retaining speed, callback delta and previous-action context. Missing/invalid prediction zeros features 3–5 and 9–11. A valid predicted point at the car position remains distinguishable from missing prediction by its validity mask. Initialization uses previous Neutral and zero steering. Scale factors are not bounds: analytic cases verify distance and speed features can equal 2.

## Adapter and oracle paths

The live worker constructs installed RLBot `GamePacket` and `BallPrediction` objects, serializes/deserializes each through its actual `CorePacket` envelope, and passes the unpacked native objects to `live_adapter`. Own physics and elapsed time come from the packet; selected prediction position/time come from the supplied native slice. The first-slice timestamp comes from the native prediction message.

The simulation worker constructs installed RLGym `GameState`, `Car` and `PhysicsObject` objects. Native RocketSim `Angle(yaw, pitch, roll).as_rot_mat()` is converted using the installed engine's `as_numpy().reshape(3, 3).transpose()` convention. `PhysicsObject.forward/right/up` supply the basis columns. A supplied native RocketSim `BallState` supplies the predicted position. A separate check uses the installed RLGym Euler-to-matrix path for every paired case.

Both adapters call `observation_contract.build`. An independently written oracle in `run_v2.py` additionally uses the recovered teacher's original `Orientation`, `relative_location` and `Vec3` helpers, with separately expressed feature assembly. Eleven analytical anchors independently establish signs, scaling, ordering and missing-value behavior; adapter agreement alone is not the sole correctness check.

Positions, velocities, Euler inputs and supplied timestamps are canonicalized to float32 before constructing matched native objects, reflecting live wire precision. Geometry is calculated before casting the completed feature vector to float32. Native orientation-matrix rounding can therefore produce small relative-coordinate differences; bitwise identity of all geometry features is not claimed.

The simulation path receives explicit ball-presence and clock metadata: active-play RLGym states do not themselves reproduce missing-ball phases or live callback clocks. Previous mode/steering mean previously transmitted controls, not a car's current input echo. These supplied contexts are matched between paths; a production phase/clock adapter has not been validated here.

## Exact tests performed

Seed: `20261004`. There are 500 distinct physical arrangements, each tested with both team labels, giving **1,000 paired states**:

- 400 structured paired cases: eight yaw headings, five pitch/roll combinations, and five displacement sizes. Orientations include pitched, vertical, rolled wall poses and upside-down poses. The displacement sizes cover coincident, near, intermediate and far geometry; these are not strict teacher branch-boundary tests.
- 600 seeded varied paired cases: different positions, heights, full 3D car/ball velocities, rotations, selected prediction points and timestamps.
- Balanced availability coverage: 600 ball-present cases, 400 ball-missing cases and 200 effective valid predictions. Cases include absent prediction, unavailable selection, stale supplied prediction masked by absent ball, and valid prediction at zero relative position.
- Both team labels produce identical vectors for otherwise identical cases; controlled-car selection alternates between native player indices with an opponent present.
- Supplied callback gaps cover 1, 2, 3 and 5 physics ticks, plus initialization. Previous-action modes and continuous Chase steering are represented.
- All 1,000 cases are checked against the original geometry oracle and the alternate RLGym Euler path.
- Eleven analytical anchors: full known 18D vector; missing ball; missing prediction; initialization; valid zero-relative prediction; five orientation/axis sign checks; and an unclipped distance/speed check.
- Sixteen malformed shared-builder inputs must raise errors: malformed/nonfinite geometry, non-orthonormal basis, invalid masks, incomplete valid predictions, unsupported previous actions, and duplicate/backward/nonfinite callback times.
- Four adapter guard checks must raise errors: missing live/simulation controlled car and missing live/simulation selected prediction object when declared valid.
- Six deliberate output mutations must be detected by the comparator: right-axis sign, coordinate scale, validity mask, mode order, time scale and distance clipping. These are comparator sensitivity probes, not a claim of exhaustive mutation testing.

## Results and requirement gates

| Requirement | Result | Evidence |
|---|---|---|
| Exactly 18 float32 features, declared order, finite values | PASS | Shape/dtype/finiteness assertions and full-vector analytical anchor |
| Current and predicted car-local geometry | PASS | Native paths, original-helper oracle and axis/sign anchors |
| Yaw, pitch/roll, wall and upside-down poses | PASS | Structured pose coverage and seeded cases; static geometry only |
| Moving balls and varied distances | PASS | Native ball velocities and varied positions; no forecasting/dynamics claim |
| Missing ball/prediction and genuine zero distinction | PASS | Masks, zeroing rules and dedicated anchors |
| Distance and speed scaling without clipping | PASS | Native comparisons plus values exceeding 1 |
| Supplied time features | PASS | Exact agreement for horizon, first-slice offset and callback delta |
| Previous mode order and continuous steering | PASS | Exact agreement; unsupported contexts rejected |
| Both teams, correct own-car extraction | PASS | 500 cases per team and alternating player indices |
| Masks and mode fields exact | PASS | Zero difference and exact comparisons |
| Maximum normalized native-path difference ≤ 1e-5 | PASS | **1.1920928955078125e-7** |
| Malformed inputs fail loudly | PASS | 16 builder and four adapter rejection cases |
| Protected source/configuration unchanged | PASS | 43 hashes unchanged |

Maximum live-versus-original-helper oracle difference: **0**. Maximum alternate RLGym Euler-path difference: **2.384185791015625e-7**, also below the 1e-5 gate.

Only the six relative-coordinate channels differ between native live and native simulation paths. Distance, speed, availability masks, all three time features, previous-mode one-hot and previous steering have maximum difference **0**. The small coordinate differences are consistent with native rotation-matrix floating-point rounding; no sign, ordering, scale or missing-data discrepancy was observed in this corpus.

Machine-readable evidence and per-feature differences: [V2 JSON results](v2_observation_parity_20261004.json). It also records exact code and fixture SHA256 hashes.

## Reproduction

Executed using the existing two environments, with no installation or upgrade:

- Live: Python 3.12.10, `rlbot==2.0.0b55`, `rlbot-flatbuffers==0.19.0`.
- Simulation representations: `RocketSim==2.2.1`, `rlgym-rocket-league==2.0.1`, `numpy==1.26.4`.

From the project root in PowerShell:

```powershell
& '.\python-example\venv\Scripts\python.exe' -B '.\training\v2_observation_test\run_v2.py'
```

The diagnostic measured 0.71 seconds internally and completed as a short local test. The runner invokes the existing training environment for the simulation-representation worker. Expected final status is `V2 PASS`, followed by maximum differences and protected-hash confirmation. Reproduction is optional; V2 has already been executed.

## Scope limits and stop point

Prediction points are deliberately preselected fixture inputs. No prediction indexing, forecasting, empty-message source behavior or RLBot/RocketSim provider compatibility was established; those remain V6/V7. Identical supplied timestamps test feature encoding, not independent simulator-clock generation, callback delivery, latency, action holds or sequence timing. Static wall/upside-down poses test coordinate transforms, not physical recovery behavior. No recurrent memory or learned behavior was tested.

**V2 passed. Stop here.** V3 and all later experiments remain unexecuted; BC, demonstration generation, PPO and the student remain unimplemented.

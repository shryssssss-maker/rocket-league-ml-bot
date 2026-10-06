# V3 strict source-boundary report

Date: 4 October 2026. **Final result: PASS.** V3 is safe to mark passed for fresh-callback branch behavior. Stop here; V4 and later experiments were not run.

Specification: [student design report](student_contract_design_20261004.md), section 7, V3. Reference Git HEAD was checked against `fd061f457bf19175b4a9b3b3d7811a987044c64d`. All 43 protected source/configuration hashes matched the saved manifest before and after execution.

## Files created

1. `training/v3_boundary_test/run_v3.py`: isolated branch reconstruction, direct original-source execution, matched fixtures and assertions.
2. `training/reports/v3_source_boundaries_20261004.json`: full per-case evidence and summary, including test-code SHA256.
3. `training/reports/v3_source_boundaries_20261004.md`: this report.

Neither bot directory nor V0/V1/V2 code/reports were changed. No training, model, demonstration generator, physics stepping, installation or live match was involved.

## Source and adapter comparison

The original path calls unchanged `python-example-original/src/bot.py:MyBot.get_output` on a fresh object for every fixture. It executes the original `begin_front_flip`, `util.drive.steer_toward_target`, `util.ball_prediction_analysis.find_slice_at_time`, and `util.sequence.Sequence.tick/ControlStep.tick` functions.

The socket-owning constructor is bypassed. Rendering, boost-pad bookkeeping and outgoing quick chat use diagnostic stubs; no connection is established. Transparent runtime wrappers trace prediction lookup and Chase steering while calling their original functions. Quick-chat invocation records the original flip trigger, and the created sequence is checked independently. The render stub captures the actual target selected by the original source. No source file is patched on disk.

`run_v3.py:adapted_result` reconstructs only the current raw-state decision: full 3D distance, strict prediction branch, full 3D speed, strict flip branch, and continuous steering. It reuses the original prediction-selection helper and sequence primitives intentionally. This is an isolated source-equivalence diagnostic, not a new scripted expert or a production simulation teacher adapter. Prediction indexing and temporal sequence equivalence remain V6 and V4 respectively.

The reconstruction uses raw positions and velocities, **not** distance/speed recovered from normalized float32 observations. The normalized 18D features must not determine exact reference branch predicates.

## Exact tests

There were **1,012 paired source/reconstruction cases**, with a fresh original instance and fresh reconstruction per case:

| Group | Cases | Coverage |
|---|---:|---|
| Boundary cross-product | 864 | Three distances × six speeds × two prediction availability cases × two teams × six poses × two input representations |
| Local-axis targets | 144 | Six poses × three local axes × both signs × both teams × two input representations |
| Exact mixed-component norms | 4 | Distance components (900,1200,0) give exactly 1500; speed components (450,600,0) and (480,640,0) give exactly 750 and 800; both input representations |

Distance fixtures: **1499.9, 1500.0, 1500.1**. Speed fixtures: **749.9, 750.0, 750.1, 799.9, 800.0, 800.1**.

The boundary cross-product uses vertical distance and velocity components, ensuring that an accidental 2D norm would fail. Poses include identity, quarter-turn yaw, nonzero combined pitch/yaw/roll, vertical pitch, rolled wall pose and upside-down pose. Positions include the origin and elevated wall-adjacent coordinates. Local-axis cases include forward, behind, left/right and above/below targets with nontrivial own positions. These are static packets, not wall/recovery dynamics.

Every fixture is tested once as directly constructed native Python objects and once after actual RLBot `CorePacket` serialization/deserialization of both `GamePacket` and `BallPrediction`. Equality cases are exact in the raw fixtures; wire-rounded near-boundary metrics are recorded separately in the JSON. Every boundary case additionally asserts that wire precision did not change its expected nominal branch.

Prediction fixtures provide either a fixed valid target distinct from the current ball, or a nonempty prediction with insufficient range so the original helper returns `None`. The latter verifies current-ball fallback. No empty-message exception, index-boundary sweep, prediction provider or forecast is tested here.

## Boundary results

| Raw metric | Expected behavior | Result |
|---|---|---|
| Distance 1499.9 | No prediction lookup; current ball target | PASS |
| Distance 1500.0 | No prediction lookup; current ball target | PASS |
| Distance 1500.1 | Prediction lookup; selected point if available, otherwise current ball | PASS |
| Speed 749.9 | Chase | PASS |
| Speed 750.0 | Chase | PASS |
| Speed 750.1 | Begin front flip; first Jump controls | PASS |
| Speed 799.9 | Begin front flip; first Jump controls | PASS |
| Speed 800.0 | Chase | PASS |
| Speed 800.1 | Chase | PASS |

In **96 simultaneous far-ball/flip cases**, both paths performed prediction lookup **before** initiating the flip. The selected target matched even though the first sequence control superseded Chase controls. The first flip output was exactly `[0,0,0,0,0,true,false,false]`; no throttle or steering was carried into it.

| V3 requirement | Result | Measurement |
|---|---|---|
| Same strict branch and execution order | PASS | Exact event-list agreement in all 1,012 cases |
| Same selected target/fallback | PASS | Exact coordinate agreement |
| Same eight controller channels | PASS | Maximum analog error **0.0**; Boolean channels exact |
| Same flip creation decision | PASS | Exact agreement |
| Full 3D distance and speed | PASS | Vertical and mixed-component fixtures |
| Continuous steering, no quantization | PASS | Direct original formula comparison |
| Yaw/roll/boost/handbrake remain zero/false | PASS | Asserted in both paths for every case |
| Correct controlled car and team-independent behavior | PASS | Both teams and alternating own-player indices |
| Protected files unchanged | PASS | All 43 hashes matched |

The analog comparator limit was `1e-12`; the observed maximum difference was exactly **zero**, including steering. Controller order is throttle, steer, pitch, yaw, roll, jump, boost, handbrake. Buttons are required to be actual Boolean values, not merely numerically equal to them.

## Discrepancy found and corrected

The first run failed on a wire-round-tripped rolled pose with a target directly behind the car. The initial reconstruction used Python `sum()` for dot products. Its initial positive zero erased a negative-zero local-right result that the original `Vec3.dot` preserves through its three-term expression.

This matters because `atan2(-0, negative)` and `atan2(+0, negative)` select opposite signed angles. The original returned steering **-1**; the initial reconstruction returned **+1**. The correction uses the original arithmetic order:

```python
delta[0]*axis[0] + delta[1]*axis[1] + delta[2]*axis[2]
```

The final runner contains an explicit regression check: wire case 480 must reproduce original steering -1 and must demonstrate that the sum-based variant would produce +1. The corrected reconstruction passes the entire corpus with zero analog difference. No tolerance was relaxed to hide the defect.

This finding also documents a limit of V2: small coordinate-error comparisons do not distinguish positive and negative zero, and its isolated feature builder uses `sum()` for geometry. V2's declared numeric gate remains satisfied, but it does not establish exact downstream steering at the behind-car angular discontinuity. **V2 was not modified or silently promoted into an exact teacher controller.** Any future reference-control adaptation must preserve source arithmetic; the neural observation contract remains a candidate subject to later review. No student action accuracy is established here.

## Reproduction and stop point

Executed in the existing Python 3.12.10 / RLBot 2.0.0b55 environment. The final test measured approximately **0.15 seconds** internally and was a short local command:

```powershell
& '.\python-example\venv\Scripts\python.exe' -B '.\training\v3_boundary_test\run_v3.py'
```

Expected output: `V3 PASS: 1012 paired cases; maximum analog difference 0.0; 43 protected hashes unchanged.` Followed by the 96 prediction-before-flip cases and report path. Reproduction is optional; the final corrected V3 test has already run successfully.

Full raw/post-wire metrics, original and reconstructed controls, branch events, targets and regression evidence are saved in [the JSON report](v3_source_boundaries_20261004.json).

V3 tests fresh callbacks only. Continuing/completing a pending sequence, exact phase durations, no-ball transitions, prediction indexing, provider compatibility and action application timing remain unverified by V3. **V4 and later gates remain unexecuted.** No demonstration or ML work is authorized by this result.

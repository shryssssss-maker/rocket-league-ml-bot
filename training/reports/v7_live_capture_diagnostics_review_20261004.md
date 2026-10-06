# Live-only capture diagnostic changes

Update, 2026-10-04: the user approved `bounded_first_received_packet_proposal_v1` exactly as documented below. The runner now sets `approved=True`; numeric tolerances and the first-eligible-processed-callback rule are unchanged. The approval-aware non-live preflight passed under sessions/20261004_192604_840291, including verified-only coverage eligibility and all 43 protected hashes. Live capture is authorized for the user to launch from PowerShell. No live capture, comparison or ML work has been executed by the assistant. The remaining proposal-status wording below describes the earlier preparation state.

Status: non-live preflight passed. No live launch or V7 comparison was performed. Fixture verification tolerances below are proposed and remain disabled pending explicit user approval.

## Source and timing

The diagnostic consumes the actual packet passed to RLBot Bot.get_output and the actual Bot.ball_prediction assigned by the installed framework. It never constructs populated prediction slices, imports RocketSim, rewrites prediction timestamps or modifies the original selector. Canonical time and all comparison inputs are received packet data. Requested fixture data are separately named application-evidence inputs only.

Overridden receipt handlers record packet/prediction counters, a combined delivery-order counter, monotonic receipt time and session-local Python object identity, then forward to the original framework handlers. Delivery counters identify reuse even if Python object IDs are recycled. Each callback records the latest identities and reuse since the previous callback; sampled predictions additionally have a payload SHA256 and reuse-since-probe flag. These counters are not server-side frame IDs. The framework still supplies its latest available prediction asynchronously; no atomic same-frame pairing is claimed or manufactured.

## Proposed application verification (not approved)

Use the first processed callback with an actual ball and both cars, whose packet receipt counter/time follows the state-setting request. Record actual current physics, requested physics, receipt metadata, elapsed game time and frame difference. Do not search later callbacks for a better match, extrapolate the fixture or rewrite actual physics. Messages coalesced by the existing Bot framework remain visible through receipt counters; this rule refers to the first eligible processed callback, not every transport packet.

Proposed `verified` limits:

| Field | Ball | Each car |
|---|---:|---:|
| Position error, Euclidean, Unreal units | 100 | 35 |
| Velocity error, Euclidean, Unreal units/second | 100 | 100 |
| Angular velocity error, Euclidean, radians/second | 0.25 | 0.25 |
| Rotation error, maximum wrapped pitch/yaw/roll component, radians | 0.30 | 0.05 |
| Boost error from requested zero, boost units | N/A | 0.5 |

Also require positive frame advance of at most five frames and nonnegative elapsed time of at most 5/120 + 0.0001 seconds (approximately 41.77 ms). The 0.1 ms allowance accounts for packet timestamp representation; it does not alter any timestamp. Compare request numbers at their float32 transport precision. Circular angle differences are diagnostic error calculations only; they do not change input angles or predictions.

`near_match` permits twice the field tolerances and at most twelve frames/100.1 ms. Near matches never count toward coverage. Missing required fields, larger errors, invalid numbers, rollback or a later first relevant callback classify the decision as `failed`. Missing-ball callbacks before such a decision are logged and leave application unverified. Ten-second pending-fixture timeout is failed. A first relevant callback can arrive before the server applies a request; this conservative rule can therefore fail an eventual reset. It supplies bounded evidence, not a server acknowledgment or proof of causality. Ball/car impacts within the window can also cause failure, without any compensating physics correction.

The numerical tolerances are proposals rather than empirical results. The position allowances accommodate short physics advancement (for example, 2,200 units/second travels about 91.7 units over five ticks); velocity bounds exceed gravity's approximately 27-unit/second change over that interval. They intentionally do not accommodate arbitrary bounce impulses. Car bounds keep placement and heading evidence substantially tighter. Ball rotation is recorded and conservatively checked even though spherical ball orientation is not a predictor-selection input.

Only approved-policy `verified` fixtures with three valid selector snapshots count toward verified coverage. Completion requires all 120 fixtures verified, all 360 scheduled probes valid and no errors. Requested scenario names alone are never evidence that a reset occurred, nor proof that a particular bounce or goal outcome occurred. Such trajectory outcomes remain for later validation.

## Failures and teacher provenance

Missing-ball callbacks are written explicitly without invoking the selector. Each scheduled diagnostic probe retains complete available prediction physics and original timestamps. Absent optional physics fields are recorded as null, rather than synthesized as zeros. Before a selector exception is re-raised, its received packet/prediction inputs and exception type are flushed to captures.jsonl and errors.jsonl. Empty predictions preserve the original IndexError. Out-of-range selection is marked unsuccessful, without clamping or fallback. Error halts capture and the runner stops through its error summary; framework-level exception handling is unchanged.

The probe is explicitly unconditional diagnostic selection, not teacher gameplay. Recorded teacher branch evidence includes ball presence, full 3D distance and strict distance > 1500, plus eligibility conditional on no pending sequence. Actual teacher sequence state is not observed. Full branch eligibility is therefore unknown for far-ball states and false for missing/near-ball states; no invented sequence reset is used.

Diagnostic control history is separate from teacher memory. Successful PlayerInput submissions to SocketRelay are logged with their actual eight controls, callback counter, index and monotonic time; the callback records the preceding submission. Packet last_input is retained separately in probe player records. Submission does not claim that Rocket League applied the input. There is no recovered teacher temporal-state evidence in this neutral-agent capture.

## Metadata and artifacts

Session metadata records config hashes, runner hash, protected hashes, session ID, Python and rlbot/rlbot-flatbuffers/psutil versions, source provenance and proposed policy. Agent metadata records initialized index/team/name. Outputs are captures.jsonl, delivery_events.jsonl, errors.jsonl, capture_summary.json, agent_metadata.json, metadata.json and run_summary.json. Each diagnostic log has a 256 MiB bound that fails explicitly instead of silently truncating. Startup receipt buffering is bounded to 10,000 events.

Changed source: training/v7_prediction_test/capture.py. Added report: this file. Preflight created isolated session configs/results under training/v7_prediction_test/sessions/20261004_191929_077790 and 20261004_192036_270122. Neither bot directory nor any of the 43 protected files changed.

## Non-live validation

Executed capture.py --check using the existing live venv. Passed: versions/imports/config parsing, state-setting object construction, proposed verified/near_match/failed classifications, missing-ball evidence, original empty-prediction exception with evidence written before propagation, failed-fixture exclusion, receipt counter distinction for repeated objects, and all 43 protected hashes. No populated synthetic prediction slices were created. This does not validate real asynchronous delivery, state-setting success, live input submission or compatibility results.

Future command after tolerance approval: `& '.\python-example\venv\Scripts\python.exe' -B '.\training\v7_prediction_test\capture.py'`. It is deliberately blocked while VERIFICATION_POLICY.approved is false. No capture will be launched by the assistant without explicit approval.

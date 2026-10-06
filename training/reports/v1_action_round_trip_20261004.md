# V1: teacher action round-trip

Date: 4 October 2026. **PASS for the specified teacher contract through the installed RLBot send envelope and RocketSim engine mapping. Safe to mark V1 passed within this scope.** A supplementary standalone RLBot serialization helper failed and must not be used as the transport codec; details are below.

V0 was not rerun. V2 or later, BC, dataset generation, PPO and the student were not implemented or run. Neither protected bot directory was modified.

## Files created

1. `training/v1_action_test/run_v1.py` — isolated diagnostic, including strict label/vector codecs and two worker interpreters.
2. `training/reports/v1_action_round_trip_20261004.json` — machine-readable measurements, rejected-case messages, versions and diagnostic-code SHA-256.
3. `training/reports/v1_action_round_trip_20261004.md` — this report.

No other files were changed by V1. All 43 recorded protected source/configuration hashes and the pinned reference revision were checked before and after execution.

## Specification and source paths

Specification: `student_contract_design_20261004.md`, section 1 and V1 row. Mode IDs are 0 Neutral, 1 Chase, 2 Jump, 3 Front dodge.

Reference revision: `fd061f457bf19175b4a9b3b3d7811a987044c64d`.

- `python-example-original/src/bot.py:MyBot.get_output` — Neutral from the missing-ball branch and Chase throttle/steering assignments.
- `MyBot.begin_front_flip` — Jump, Neutral/release, Front dodge, Neutral/coast controller objects.
- Installed live `rlbot/managers/bot.py:Bot._packet_processor` — sends `PlayerInput(index, controller)`.
- Installed live `rlbot/interface.py:SocketRelay.send_msg` — serializes `InterfacePacket(msg).pack()`.
- Installed training `rlgym/rocket_league/sim/rocketsim_engine.py:RocketSimEngine.step` — assigns numeric columns 0–7 to native `RocketSim.CarControls`.

The eight-channel order is exactly:

```text
[throttle, steer, pitch, yaw, roll, jump, boost, handbrake]
```

RLBot's additional `use_item` remains false and is checked separately.

## Exact tests performed

### Supported teacher commands

Tested 1,004 teacher cases:

- One Neutral controller returned by the original no-ball branch.
- One Jump controller returned by the original `begin_front_flip` call.
- One Front-dodge controller obtained from the original sequence's declared step.
- 1,001 Chase commands with steering `(i - 500) / 500`, for integer `i = 0..1000`: endpoints -1 and +1, zero, and spacing 0.002.

The four declared sequence controller combinations were independently checked against the original objects. The original `get_output` Chase branch constructed all 1,001 Chase controls. Only its steering helper was replaced temporarily in memory to supply each test scalar, then restored; this isolates action assignment from geometry. No source file was patched and no geometry, observation or sequence-timing experiment was conducted.

For each supported case:

1. Strictly encode the eight fields into the supported mode/steering label.
2. Reconstruct the eight fields and compare against the original case.
3. Construct/read RLBot named controller fields.
4. Pack and unpack the actual `InterfacePacket(PlayerInput(7, controller))` envelope. Check player index and all controller fields, including false `use_item`.
5. Feed both original numeric vectors and RLBot-wire-returned vectors into the installed `RocketSimEngine.step` using float32 `(1,8)` input, for both values of its delay option.
6. Capture the actual native `CarControls` fields delivered to `set_controls`; compare with original fields and re-encode/reconstruct the returned teacher command.

The engine method was executed with inert arena/car sinks, without calling its constructor. Therefore, no arena was created, no physics ran, no observations were built, and no live connection was opened. Both delay branches were tested for mapping only; timing equivalence is outside V1.

### Independent channel-routing sentinels

Teacher yaw/roll/boost/handbrake are constant, so supported teacher commands alone cannot detect every permutation among those fields. Four transport-only sentinels used distinct analog values `[0.11,-0.22,0.33,-0.44,0.55]` and independent button activation to check each channel's routing.

These deliberately unsupported controller vectors were tested through the transport layer only and rejected by the teacher codec. They do not expand the approved action support or introduce a new controller policy.

Total installed engine mapping calls: **4,032** = 1,008 cases including sentinels × two input paths × two delay branches.

### Loud rejection tests

All **28 named invalid cases** raised `ValueError` with an explicit message. Cases cover:

- Nonzero yaw/roll; boost/handbrake on; RLBot `use_item` on.
- Reverse/partial throttle, unsupported pitch, pitch without jump.
- Steering during Neutral/Jump/Front dodge; Chase combined with jump/dodge pitch.
- Nonbinary buttons, NaN, infinity, out-of-range analog data, Boolean analog input.
- Seven- or nine-channel vectors.
- Unknown/Boolean mode IDs, non-Chase steering labels and out-of-range steering labels.

The four transport-only sentinel vectors were additionally rejected as teacher commands. No nearest-mode assignment, rounding into steering bins, or silent unsupported-action correction was used.

## Requirement results

| V1 requirement | Result | Evidence |
|---|---|---|
| Neutral | PASS | Original neutral controls preserved through all paths |
| Chase | PASS | All 1,001 source-constructed Chase commands round-tripped |
| Jump | PASS | Original first-jump controls preserved |
| Front dodge | PASS | Original pitch -1 / jump true combination preserved |
| 1,001 evenly spaced steering values across [-1,1] | PASS | Exactly 1,001 distinct values remained after RLBot serialization |
| Exact eight-channel transport mapping | PASS | Actual RLBot envelope and installed engine method; independent sentinels distinguish all fields |
| Analog round-trip error ≤ 1e-6 | PASS | Maximum **2.95639037695139e-08** |
| Exact button values | PASS | Returned jump/boost/handbrake values match exactly and are native Python Booleans |
| Teacher yaw/roll/boost/handbrake constants | PASS | Zero/false on every supported case and every return path |
| Unsupported teacher combinations fail loudly | PASS | 28 named rejection cases plus four unsupported sentinel vectors |
| Protected directories unchanged | PASS | 43 hashes and reference revision verified before/after |

The small analog error is ordinary float32 representation loss in RLBot wire/native simulator controls, not steering quantization. All 1,001 steering values remain distinct. The error is approximately 34 times smaller than the specified limit. The specified supported modes showed no unintended field changes, mode changes or button changes.

## Discrepancy: standalone ControllerState serialization helper

In installed `rlbot-flatbuffers==0.19.0`, the supplementary call:

```text
ControllerState.unpack(ControllerState(throttle=1, steer=0.123456789).pack())
```

did not reconstruct the original fields. It returned approximately:

```text
throttle = 5.605193857299268e-45
steer    = 1.0
pitch    = 0.12345679104328156
```

This helper probe **FAILS** and is not safe as a standalone controller codec in the current installation. Its underlying cause was not investigated or repaired as part of V1.

The actual RLBot send envelope instead returned throttle 1, steer 0.12345679104328156, pitch 0, with the declared buttons/constants intact. The installed `Bot._packet_processor` and `SocketRelay.send_msg` use this envelope. All required V1 round-trips use that actual route and pass. Therefore, this discrepancy does not block V1 for the specified real transport, but must remain documented to prevent future use of the failing shortcut.

No dependency was upgraded and no bot source was changed to work around the helper.

## Reproduction and limits

The final diagnostic completed in about one second. Both existing environments were used; no dependency installation or long-running command was needed.

```powershell
& '.\python-example\venv\Scripts\python.exe' -B '.\training\v1_action_test\run_v1.py'
```

The machine-readable report records the exact live/simulation dependency versions, test-script SHA-256, counts, source-derived controls and standalone helper probe. This is transport/representation validation, not a live match, physical maneuver test, latency test, observation test, training run or demonstration collection.

**V1 is safe to mark passed for the approved teacher action support using the actual InterfacePacket transport. Stop here; V2 needs separate authorization.**

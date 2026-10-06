# Live capture application-failure diagnostic

Session: `training/v7_prediction_test/sessions/20261004_192953_891979`.

This is a diagnostic review of saved application evidence and source dependencies, not V7 comparison. No prediction provider, selector, policy, capture source or captured state was changed. No later matching packet was sought for corner_04. No live capture, RocketSim, demonstrations or ML work was run.

## 1. Four boost-only failures

All four were classified by the approved first-callback rule two frames (approximately 16.67 ms) after their requests. Every recorded non-boost application error was within the approved verified bounds. The only failing values were:

| Fixture | Controlled-car boost error | Other-car boost error | Largest ball-position error | Other-car angular-velocity error |
|---|---:|---:|---:|---:|
| stationary_00 | 24 | 0 | 0 UU | 0.217916 rad/s |
| wall_03 | 0 | 12 | 24.8102 UU | 0.240491 rad/s |
| corner_07 | 0 | 12 | 27.3685 UU | 0.240491 rad/s |
| goal_mouth_03 | 0 | 12 | 21.7345 UU | 0.240491 rad/s |

The approved boost limit is 0.5 and the requested boost is zero for both cars. Thus these are genuine failures of the **current full verification policy**, even though their physics application evidence otherwise passes. Their three selector snapshots each remain numerically valid; the raw classification remains failed and coverage-ineligible.

Before stationary_00 both cars had boost 34; after its first verification callback the controlled car had 24 and the other had zero. Before the other three requests both cars had zero; the first verification callback reports the other car at 12. These observations establish the mismatch, not its engine-side cause.

A short installed-binding diagnostic round-tripped DesiredGameState through `flat.InterfacePacket.pack/unpack`: requested boost values None, 0, 1, 12 and 24 were retained exactly. This rules out loss of explicit zero in that tested Python serialization path. An initial attempted CorePacket round-trip was invalid because CorePacket is the inbound message union; the corrected InterfacePacket test used the actual outbound message type. Neither test connected to RLBot or the game.

The capture does not retain boost-pad activity, engine-side state-setting acknowledgment or intra-frame boost updates. Therefore it cannot conclusively distinguish reset/update ordering, pickup behavior or another live state-setting effect. It would be unjustified to label these as confirmed pad pickups or silently repair their values.

### Relevance to this particular contract

- `python-example-original/src/util/ball_prediction_analysis.py:find_slice_at_time` reads the requested game time, first slice timestamp, slice count and selected slice. It never reads a car or boost amount.
- `python-example-original/src/bot.py:MyBot.get_output` uses ball availability, pending sequence, own position/velocity, ball position and prediction. It updates BoostPadTracker but never consults the stored pad data for decisions. It does not read car boost or output boost=true.
- `python-example-original/src/util/drive.py:steer_toward_target` uses own location/orientation and target location. Boost is absent.
- `training/v2_observation_test/observation_contract.py:FEATURES/build/live_adapter` use own position/velocity/orientation, current ball position, selected prediction position/timestamps, callback timing and previous mode/steer. There is no boost feature or other-car input.
- [Official RLBot prediction documentation](https://wiki.rlbot.org/v5/botmaking/ball-path-prediction/) describes ball physics predictions excluding future car collisions. Car boost amount is not a predictor input under that documented contract; car collisions that already happened are represented by the actual current ball physics.

At fixed actually received physics, prediction and timing, boost amount has **no direct dependency** in the selector, Chase steering or any of the 18D features. It is not equivalent to the controller's boost button. In normal gameplay, applying boost can change subsequent own velocity/position or cause an actual ball contact; those changed physics are relevant. This conclusion is limited to the current recorded-state diagnostic contract, not a claim that boost is irrelevant to all bots or future rollouts.

All 2,312 diagnostic agent control submissions were neutral, including boost=false. The four fixtures' twelve retained selector snapshots also record boost=false for both players. However, the human player's later snapshots for wall_03, corner_07 and goal_mouth_03 contain throttle=1, steer=1 and yaw=1. Thus the capture must not be described as two neutral controllers or a controlled future rollout. Their actual packet states remain legitimate inputs for a bounded snapshot adapter comparison; later closed-loop tests would need different control discipline. These samples do not reveal every human input at every intermediate frame.

**Recommendation:** retain boost as recorded telemetry, but propose removing it from *coverage eligibility for this specific live selector/18D snapshot diagnostic*. This is a policy amendment requiring user approval. Do not remove own speed/orientation/position checks, alter predictions or retroactively relabel these four failures under the existing policy. No amendment was implemented.

## 2. corner_04: major application failure

The state-setting request returned from the Python send path and its event was logged. The installed `rlbot/interface.py:SocketRelay.send_msg/send_bytes` serializes InterfacePacket and calls socket.sendall. This proves local transmission-path completion, not engine application acknowledgment.

Request: frame **3009**, canonical time **25.075000762939453**, receipt counter **1459**, monotonic time **50826.937**, phase **MatchPhase.Active**. First relevant processed callback: frame **3011**, canonical time **25.09166717529297**, receipt counter **1460**, monotonic time **50826.953**, phase **MatchPhase.Active**. Difference: **two frames / 0.016666412353515625 seconds**. There is no recorded goal/kickoff phase transition between these two callbacks.

| Body | Requested position | Actual immediately before request | Actual first verification packet |
|---|---|---|---|
| Ball | (3200, 4300, 470) | (-3364.710, -4377.240, 447.460) | (-3355.100, -4364.100, 445.530) |
| Controlled car, index 0/orange | (-2200, -2500, 17) | (2200, 2500.160, 17) | (2200, 2500.160, 17) |
| Other car, index 1/blue | (3200, 3000, 17) | (-3125.490, -2971.980, 18.340) | (-3118.830, -2968.280, 18.340) |

Requested ball velocity was (1300, 1000, 100), angular velocity zero and rotation zero. Actual before-request ball velocity was (576.811, 789.071, -107.971); first verification velocity was (576.511, 788.671, -118.741). Ball angular velocity remains (-0.167510, 0.167510, 1.651110) across these two recorded states. Controlled-car requested yaw was 2.094395 rad; both received states retain approximately 1.570796 rad. Other-car requested velocity was zero; the first verification packet reports (401.861, 227.281, 0.261).

Position errors against the request were **10,864.463 UU** for the ball, **6,660.450 UU** for the controlled car and **8,691.834 UU** for the other car. In contrast, actual pre-request to first-packet movement is only (9.610, 13.140, -1.930) for the ball; the controlled car's position does not change; the other moves roughly (6.660, 3.700, 0). These are recorded-state differences, not replacement/extrapolated states.

**Supported diagnosis:** the first verification packet still contains the preceding fixture's opposite-side configuration, with small continuing motion. It does not show corner_04 applied and then ordinary physics producing the requested-vs-actual error. This is consistent with a state-setting delivery/application race or delay. Whether the engine subsequently applied or rejected the message, and the exact server/bridge ordering, remain unknown: there is no acknowledgment log. No later packet was searched to recover coverage. All three corner_04 selector snapshots remain excluded from coverage despite their individually valid selectors.

### Geometry and collision assessment

The requested ball point is inside the standard Stadium bounds. [Official field geometry](https://wiki.rlbot.org/v5/botmaking/useful-game-values/) gives walls at x=4096/y=5120, ceiling z=2048 and the approximate 45-degree corner plane x+y=8064. At (3200,4300,470), x+y=7500, approximately 398.8 UU inside that corner plane. This supplies a substantial margin for the documented approximately 91–93 UU ball radius, with 470 UU floor height and 1578 UU ceiling separation. This is a conservative gross-intersection check, not certification of the exact curved mesh.

The requested ball-to-other-car center separation is approximately 1,376.7 UU; ball-to-controlled-car separation is approximately 8,695.1 UU. There is no requested ball/car overlap suggested by these positions. Car centers are well inside the lateral field bounds. The human car's observed resting height is about 18.34 rather than requested 17, which can cause small floor settling; it cannot explain all three bodies staying thousands of units away on the opposite side.

Saved first-packet evidence verifies corner_00 at the same positive x/y with ball z=450 and corner_08 at z=490. Ten of the twelve corner fixtures verified overall. These support that the family is not grossly invalid. They are separate existing fixtures, not later matches searched for corner_04.

**Recommendation:** no corner-position redesign is justified by this failure. Keep corner_04 invalid. If a targeted recapture is later requested, retain its intended state initially and diagnose request/receipt ordering. Moving the fixture inward would not resolve evidence that the first packet still reflects the previous state. A safer parked-human configuration could be discussed for future controlled-rollout tests, but was not created here.

## 3. Existing evidence sufficiency and next decision

Under the unchanged policy, the eligible subset is **115 fixtures / 345 selector snapshots**, not 360 eligible snapshots. The remaining 15 valid-selector snapshots belong to five coverage-ineligible fixtures. Verified fixtures by group: stationary 11, free flight 12, rolling 12, spin 12, floor 12, wall 11, corner 10, ceiling 12, goal-mouth 11 and car-near-path 12.

**The existing 115 are sufficient in count and group diversity for a bounded live snapshot selector/adapter comparison**, subject to explicit approval to use that subset. They exceed the earlier minimum of 100 matched states and retain every scenario group. This does not prove actual bounce/goal outcome coverage or teacher temporal memory, and no comparison or pass determination was performed. Human input also prevents interpreting these captures as fixed-input future rollouts.

A targeted recapture is **not necessary solely to begin that bounded subset comparison**. It would be needed to claim the originally prescribed all-120 verified capture completion, or to resolve corner_04's application-order issue. If the user approves a boost-verification amendment later, the four boost-only fixtures could be audited as an explicitly revised-policy subset; that would not retroactively change their original labels. Do not amend the policy or rerun capture automatically.

The session stays `capture_incomplete`. Recommended next decision is whether to authorize bounded comparison on the unchanged-policy 115-fixture subset, or instead require a separately approved targeted recapture. Until review, comparison and capture remain paused.

## Evidence and preservation

Primary saved evidence: capture_summary.json (approved classifications/errors), delivery_events.jsonl (fixture_request, fixture_application_evidence, callback_state and control submissions), captures.jsonl (actual states, original predictions, selected timestamps and transmitted input observations), and run_summary.json. Existing files were read only. Report is the only new project file. The protected-source check passed all 43 hashes after investigation. Neither bot directory was modified.

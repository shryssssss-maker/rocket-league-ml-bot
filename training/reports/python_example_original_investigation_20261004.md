# Original Python example: recovery and teacher investigation

Date: 4 October 2026. Scope: source recovery and inspection only. No gameplay, simulation, dataset generation, training, or deployment changes were performed.

The current user instruction supersedes the earlier teacher plan: investigate the official Python example as a reference for a **1v1** demonstration → behavior cloning → early live validation pipeline. Architecture choices remain open until the reference is verified.

## 1. Recovery and version identity

- The user cloned `https://github.com/RLBot/python-example.git` into `python-example-original/`.
- Clean reference HEAD: `fd061f457bf19175b4a9b3b3d7811a987044c64d`, dated 27 February 2026, message `Add link to python.org into README`.
- `git -C python-example-original status --porcelain` returned no changes after inspection. The reference directory remains untouched.
- The existing modified `python-example/` also has that HEAD. This establishes the same upstream Git base, not proof that every historical local file or dependency matched the user's remembered working session.
- All eight `.py` files in `src/util/` have identical SHA-256 hashes between the two directories. `run.py` has no content difference. The tracked bot implementation is replaced locally; the current match config changes `launcher = "Steam"` to `"Epic"`.
- The most recent upstream commit affecting `src/bot.py` is `c845ee4`, dated 22 September 2025, message `Lints`.

Pinned upstream source: [bot.py](https://github.com/RLBot/python-example/blob/fd061f457bf19175b4a9b3b3d7811a987044c64d/src/bot.py).

## 2. Exact behavior

The bot is a predictive ball chaser with a timed front-flip sequence. It is not a goal-aware or opponent-aware controller.

Execution order in `MyBot.get_output()` ([source](../../python-example-original/src/bot.py)):

1. Update the boost-pad tracker.
2. If the packet contains no ball, return a neutral `ControllerState`.
3. If an unfinished sequence exists, tick it and return its controls immediately.
4. Read the controlled car's position and velocity, and ball 0's position.
5. Start with the current ball position as the driving target.
6. If **3D car-to-ball distance > 1500**, query ball prediction at `seconds_elapsed + 2`. If a slice is returned, use its position as the target.
7. Render target/speed diagnostics.
8. If **750 < 3D car speed < 800**, begin the front-flip sequence immediately.
9. Otherwise return forward throttle `1.0` and calculated steering.

Sources: `src/bot.py:24` (`get_output`), `:54` (distance/prediction), `:90` (speed trigger), `:95` (steering), `:101` (`begin_front_flip`). Thresholds are strict. Speed and distance include Z; neither is flattened to the ground plane.

### Steering

`steer_toward_target()` transforms target displacement into the car's frame using the forward/right/up basis computed from pitch, yaw, and roll. It then returns:

```text
steer = clamp(5 × atan2(target_right, target_forward), -1, 1)
```

This is continuous steering. It saturates at approximately ±0.2 radians of local heading error. Local target Z is calculated but is not used in the steering formula.

Sources: [drive.py:22](../../python-example-original/src/util/drive.py), [orientation.py:26](../../python-example-original/src/util/orientation.py), `relative_location()` at `orientation.py:47`.

### What it does not command

The bot never requests boost, handbrake, yaw, roll, or an item. It has no explicit defensive positioning, shot alignment, opponent response, boost routing, wall driving, aerial targeting, recovery routine, or kickoff-specific strategy. This describes the source logic; it does not establish how successfully incidental contacts or recoveries occur in gameplay.

## 3. Jumping, flipping, and hidden state

`begin_front_flip()` creates these four steps:

| Phase | Nominal duration | Explicit command | Other eight-channel controls |
|---|---:|---|---|
| First jump | 0.05 s | `jump=True` | All zero/false |
| Release jump | 0.05 s | `jump=False` | All zero/false |
| Front dodge | 0.20 s | `jump=True`, `pitch=-1` | All zero/false |
| Coast | 0.80 s | Neutral controller | All zero/false |

The nominal durations sum to **1.10 s**. During the entire sequence, throttle is zero and steering is zero. The bot ignores target, speed, and ball geometry after entering the sequence, except for the earlier no-ball return.

The actual timing is callback-dependent:

- `ControlStep.start_time` is initialized on that step's first callback.
- Completion uses **`elapsed_time > duration`**, not `>=`.
- `Sequence.tick()` advances its index when a step finishes but returns that finishing step's controls for the current callback. The next step starts on a later callback.
- Consequently, simply scheduling changes at fixed cumulative times 0.05/0.10/0.30/1.10 would not reproduce the source exactly.

Source: [sequence.py](../../python-example-original/src/util/sequence.py), `ControlStep.tick():39`, `Sequence.tick():53`. The stored phase/index/done flag and step start times are genuine policy memory.

No grounded, jump-available, or dodge-available check guards the speed trigger. There is no explicit sequence reset on kickoff, goal, or clock rollback. The installed `MatchInfo.seconds_elapsed` documentation says that this clock also advances during kickoffs, replays, and pauses.

**Student implication:** a single current-state vector is not established as sufficient to reproduce every action. Sequence history must be represented through an agreed history/recurrent design or explicit memory mechanism. Supplying the teacher's private sequence state during training without a way to obtain it at student inference would create a mismatch.

## 4. Native observation/input contract

The original teacher has **no fixed-dimensional neural observation vector** and no observation normalization. Its native interface is an RLBot packet, external ball prediction, controlled-player index, field information, and its own sequence state.

| Information | Actual use | Source |
|---|---|---|
| Controlled-player index | Select `packet.players[self.index]` | `bot.py:44` |
| Own position XYZ | Distance to ball and target-relative geometry | `bot.py:45`, `drive.py:24` |
| Own velocity XYZ | 3D speed for flip trigger and rendering | `bot.py:46`, `:90` |
| Own pitch/yaw/roll | Construct the local orientation basis | `orientation.py:26` |
| Ball 0 position XYZ | Current target and distance branch | `bot.py:47`, `:54` |
| Game elapsed seconds | Future prediction query and sequence timing | `bot.py:58`, `sequence.py:41` |
| Prediction slice 0 timestamp and slice positions | Select the position near two seconds ahead | `ball_prediction_analysis.py:21` |
| Field boost-pad locations/types; packet pad active/timer values | Tracker bookkeeping, unused by controller decisions | `boost_pad_tracker.py` |
| Sequence index/done/start time/durations | Determine which flip controls to return | `sequence.py` |

The controller does not directly read opponents, own boost amount, car angular velocity, ball velocity, ball angular velocity, last controls, or scores. Ball velocity/spin and field collisions can affect its external prediction indirectly.

Coordinates remain in the native global Rocket League frame; the teacher performs **no team inversion**. Steering is calculated in the car-local frame: X forward, Y right, Z up. Position is in Unreal units, velocity in units/second, and Euler angles in radians. Global Z points upward and negative Y points toward the blue goal; yaw zero points along positive X and increases clockwise. Sources: `Vec3`/`Orientation`, installed `rlbot_flatbuffers.Rotator`, and [official RLBot game values](https://wiki.rlbot.org/v5/botmaking/useful-game-values/).

Missing input behavior:

- No ball: return neutral before ticking sequence.
- Prediction request outside available indices: helper returns `None`, so target remains the current ball.
- **Empty prediction list: helper accesses `slices[0]` before checking bounds, so it can raise `IndexError`.** The comment about handling inadequate prediction does not cover this case.
- Boost-pad tracker assumes initialized field information and matching packet pad indexing.

The installed framework ignores packets where the bot's player index is absent before calling `get_output` (`rlbot.managers.bot.Bot._packet_processor`).

## 5. Native action contract

The source returns `rlbot.flat.ControllerState`, re-exported from the installed `rlbot_flatbuffers` package. Its constructor defaults all analog channels to zero and buttons to false (`__init__.pyi:1994`).

An exact eight-channel recording order can be declared as:

```text
[throttle, steer, pitch, yaw, roll, jump, boost, handbrake]
```

This would be our recording convention; the original sends named fields and has no action index.

| Channel | Original outputs | Meaning |
|---|---|---|
| Throttle | 1 while chasing; 0 during sequence/no-ball return | -1 reverse, +1 forward |
| Steer | Continuous [-1,1] while chasing; 0 otherwise | -1 left, +1 right |
| Pitch | -1 during front-dodge step; 0 otherwise | -1 nose down, +1 nose up |
| Yaw | 0 | -1 left, +1 right |
| Roll | 0 | -1 left, +1 right |
| Jump | Boolean per sequence | Press/release jump |
| Boost | False | Press boost |
| Handbrake | False | Press handbrake |

RLBot also exposes `use_item`, which remains false here and is outside the ordinary eight-channel soccar interface.

### Why the existing 90-action table is not an exact representation

The installed `LookupTableAction.make_lookup_table()` (`rlgym-rocket-league==2.0.1`, [source](../venv/Lib/site-packages/rlgym/rocket_league/action_parsers/lookup_table_action.py)):

- Restricts steering to -1, 0, or +1, losing intermediate teacher steering.
- Copies steer into yaw in its ground rows. Original chasing returns yaw zero even when steer is nonzero.
- Sets handbrake true for the jump-plus-nonzero-pitch aerial row. Original front dodge has handbrake false.

Thus nearest-row conversion would be a behavior approximation, not an exact action mapping. Even some saturated-steer/flip commands differ in other fields.

The installed `RocketSimEngine.step()` already accepts numeric controls of shape `(ticks, 8)` and assigns these eight channels individually. The current project environment places `RepeatAction(LookupTableAction(), repeats=8)` in front of that engine; that parser is specific to the old policy, not mandatory for RocketSim itself.

## 6. Timestep and prediction contract

The original bot has **no repeat-8 setting**. It makes decisions per packet that the framework processes. The installed `Bot._run()` processes the latest packet after draining incoming messages, so callback frequency must not be assumed to equal every physics tick.

`MatchInfo.frame_num` documents physics-frame counts and allows gaps between consecutive packets. The prediction helper assumes 120 prediction slices/second and selects:

```text
index = int((seconds_elapsed + 2 - first_slice.game_seconds) × 120)
```

It does not interpolate or search by each slice's timestamp. The installed prediction type describes a six-second, 120 Hz trajectory without car collisions. Source: `ball_prediction_analysis.find_slice_at_time()` and installed `rlbot_flatbuffers.BallPrediction` (`__init__.pyi:3681`).

Repeat 8 at 120 Hz is 66.7 ms, longer than either 50 ms jump/release phase. Keeping the old cadence without a comparison would alter this teacher's sequence timing. A fixed simulation cadence, prediction timestamp alignment, action latency, and live callback/hold behavior must be agreed and verified before collecting labels.

Prediction feasibility is supported by installed RocketSim 2.2.1 type stubs:

- `Arena.get_ball_prediction(num_ticks, tick_interval)` at `RocketSim.pyi:178`.
- `BallPredictor.get_ball_prediction(ball_state, ticks_since_last_update, num_states, tick_interval)` at `:512`.

These APIs were inspected, not run. Their output timing, first-sample offset, car-collision behavior, and bounce parity with RLBot's prediction have not been established. RLGym's current engine does not expose this prediction as a ready student observation.

The engine defaults to `rlbot_delay=True`, documented as a one-tick input delay. Actual live latency still needs measurement; that option alone does not prove equality.

Rotation ordering is another adapter trap: RLGym `PhysicsObject.euler_angles` is pitch/yaw/roll, while `RocketSim.Angle.as_numpy()` documents yaw/pitch/roll. Do not pass one array to the other unchanged.

## 7. 1v1 suitability and dependency compatibility

- The default `rlbot.toml` has exactly a blue human and orange example bot: a 1v1 configuration (`:95–101`).
- The controller itself does not enforce two cars or read an opponent. Its metadata has `tags=[]`; `dev.toml` instead lists one bot for development. Therefore, it supports use in our 1v1 experiment, but is not explicitly restricted to 1v1 by its implementation.
- Original `requirements.txt` is only `rlbot>=2.0.0.beta`; no exact dependency lock is published there.
- The README requests Python 3.12+, consistent with `typing.override` in the source.
- Existing live environment: `rlbot==2.0.0b55`, `rlbot_flatbuffers==0.19.0`, with RLBot metadata requiring `psutil==7.*` and `rlbot_flatbuffers~=0.19.0`. Source: installed distribution `METADATA` files.
- Teacher APIs: `Bot.initialize/get_output/run`, controlled-player mapping, `field_info`, prediction subscriptions, `ControllerState`, `GamePacket`, renderer/anchors, and `send_match_comm`.
- `Bot.run()` defaults `wants_ball_predictions=True`; `_packet_processor()` assigns the latest prediction and sends `PlayerInput(index, controller)`.
- No PyTorch, RLGym, or ML model is needed to execute the original live teacher.
- Original launch config uses Steam and expects `python-example-original/venv/`; our working launch config uses Epic. No original environment was installed or launch configuration changed. These differences must be handled in a separate live test setup.
- Upstream source is MIT licensed, copyright 2025 RLBot. Retain its copyright and permission notice in copies/adaptations. Source: `python-example-original/LICENSE`.

## 8. Reuse and necessary adaptations

| Component | Assessment |
|---|---|
| Vec3 arithmetic, orientation basis, steering function | Reusable mathematical logic; verify adapter axes/signs |
| Sequence/ControlStep and flip thresholds | Reuse exact semantics for the reference; reset/timing policy needs explicit treatment in simulation |
| Prediction target selection | Reuse threshold/look-ahead/indexing logic once equivalent timestamped predictions are supplied |
| Boost tracker | Bookkeeping only; can satisfy its packet/field requirements or exclude it from an isolated control adapter after proving output equivalence |
| Rendering and match communication | Live side effects, not action-selection inputs; simulation can use documented no-op interfaces without changing decisions |
| RLBot packet/self fields | Need an adapter from simulated state, not a new controller |
| Existing launch shell | Proven launch infrastructure, but `src/bot.py` now runs the frozen ML actor and is not the original teacher |
| Existing 92D builder | Historical reference only; lacks prediction and private sequence state and includes reconstructed live state fields the original does not need |
| Existing LookupTable/repeat parser | Cannot exactly represent this teacher's analog actions/timing |

No custom expert or simulation adapter was implemented.

## 9. Student contract: findings and decisions still required

**Established:** the native teacher uses unnormalized state/prediction plus temporal memory and outputs mixed continuous/button controls. It is not a 92D → 90-index policy.

**Recommendation for discussion:** preserve its raw eight-channel actions first. A student that retains continuous steering and models button/control modes can imitate these commands without forcing them into the old table. BC would then need a suitable mixed continuous/categorical loss rather than a single 90-class cross-entropy. Alternatively, adopting a discrete table requires an explicit quantization rule and a live comparison of that quantized reference against the original.

For observations, the eventual shared builder must supply the teacher-relevant car/ball geometry, speed, time information, and a consistent two-second prediction input with an availability/age policy. Sequence history must be learnable/available during live student inference. History stacking or a small recurrent model is a possible route; the current single-vector feedforward proposal is not yet sufficient by proof.

**No student observation dimension, feature order, scale constants, architecture, prediction fallback, or action repeat is frozen in this investigation.** Doing so before agreeing on history, prediction, action representation, and live behavior would invent a contract. Opponent features are not needed to reproduce the reference's decisions; no teammate features are proposed.

## 10. Conclusion and next gate

The clean official source is recovered and pinned, and its behavior is now understood. It is a plausible modest teacher/reference with specific limitations, but **known-good live gameplay has not been re-established**.

Before designing demonstrations, confirm that this predictive ball chase and speed-triggered flip is the behavior the user remembers. If it matches, the next gate is a short 1v1 live validation of the pinned teacher using a separate launch configuration and the known-working live dependency stack, preserving both source directories. Observe approach/contact, steering, flip execution, prediction availability, callback/frame intervals, and errors.

Only after that gate should we choose the shared observation/history and action contracts, then prove simulator/reference consistency, then collect a small dataset.

Questions for review:

1. Does this source behavior match the remembered original, or did the known-good bot also boost, defend, or perform other maneuvers?
2. After live reference validation, should the first student preserve continuous steering/buttons, or should we evaluate a deliberately quantized reference first?
3. How should student temporal memory and the simulation/live decision cadence be represented? This must be discussed before fixing a feedforward observation dimension.

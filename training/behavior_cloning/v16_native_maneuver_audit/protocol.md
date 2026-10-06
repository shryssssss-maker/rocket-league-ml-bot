# V16 native maneuver-state audit — preregistration v1, 2026-10-06

Stage 1 is infrastructure/source/preflight only. Live collection requires separate explicit authorization. Nothing here changes the official 13D, teacher, decoder, a checkpoint or frozen artifacts. This is a teacher-label observability study, not policy-success evaluation. Current user instructions supersede the historical CanoPy/2v2 direction still present in the root history documents.

## Installed authority and callback lifecycle

Live interpreter: `python-example/venv/Scripts/python.exe`; installed `rlbot==2.0.0b55`, `rlbot-flatbuffers==0.19.0`. `source_audit.json` contains AST-extracted exact datatype, source line and documentation for every requested field, plus runtime class descriptor checks. No native GamePacket/PlayerInfo instance or connection is constructed in this audit.

Installed `rlbot/managers/bot.py` `_handle_packet` / `_handle_ball_prediction` assign separate latest objects. `_run` drains queued messages, then processes only the latest packet; it does not promise a callback for each received message or physics tick. `_packet_processor` selects `_latest_prediction`, calls `get_output(packet)`, and subsequently sends `PlayerInput`. `rlbot/interface.py` unpacks messages before dispatching handlers; `send_msg` packs a message and calls socket `sendall`. A return from send confirms local submission, not engine execution. Installed sources and the binding binary are hashed.

All native primitives and car/ball physics are copied from `packet.players[index]` / `packet.balls[0]` before invoking `original.MyBot.get_output`. The unchanged source receives that actual packet and SDK-selected actual delivered BallPrediction. No atomic packet/prediction timestamp pairing is claimed. Receipt counters/order, object IDs, elapsed time, frame number, monotonic timestamps and prediction reuse are retained. Object IDs alone are not durable identities; counters distinguish deliveries. No timestamp correction, synthetic slices or replacement provider.

| Field | Native type / units | State / reset semantics supported by installed documentation | Timing classification |
|---|---|---|---|
| air_state | AirState enum: 0 OnGround, 1 Jumping, 2 DoubleJumping, 3 Dodging, 4 InAir | Surface contact / current force state; OnGround includes wall adhesion. No orientation proxy. Documentation contains inconsistent Jumping seconds/ticks; no inferred timing. | Received packet state, copied pre-current-action. Exact engine sampling point unknown. |
| has_jumped | bool | Indicates first jump; falling from a surface may leave flags false. No invented phase reset; exact reset producer not available. | Historical maneuver state in pre-action packet. |
| has_double_jumped | bool | Since leaving ground; documentation says false on ground. | Historical maneuver state in pre-action packet. |
| has_dodged | bool | Since leaving ground; documentation says false on ground. | Historical maneuver state in pre-action packet. |
| dodge_timeout | float seconds | Remaining availability; sentinel -1 on ground or expiry; fall-off/reset case is described in stub. | Received timer, no deadline reconstructed. |
| dodge_elapsed | float seconds | Time since last dodge; documented reset to zero on landing. | Historical timer copied pre-action. |
| dodge_dir | Vector2, unit direction or zero stall | Last dodge direction. Frame/reset lifetime not specified in installed documentation; record raw X/Y with no invented frame. | Historical packet value, not current command. |
| last_input | ControllerState, native analog [-1,1] and buttons | SDK/schema only says last input from player. Producer phase, correspondence to transmission and successful execution are unproven. | Available pre-current-decision; possibly previous engine-applied/received input. Remains timing-ambiguous until observational characterization. |
| boost | float 0–100 | Current amount; reset behavior not established here. | Auxiliary raw packet value, excluded from primary probe. |
| demolished_timeout | float seconds, -1 not demolished | Remaining demolition duration; not converted to an invented Boolean. | Auxiliary raw, excluded from primary probe. |
| is_supersonic | bool | Current speed classification. | Auxiliary raw, excluded from primary probe. |
| hitbox / hitbox_offset / latest_touch | BoxShape / Vector3 / Touch-or-null | Exact installed definitions exposed in source audit; not maneuver-state candidates or extra physics inputs. | Audited only; not promoted or collected as extra policy features. |

[Upstream schema](https://github.com/RLBot/flatbuffers-schema/blob/main/schema/gamedata.fbs) corroborates the field definitions but does not prove the installed engine/build provenance. Stage 1 cannot prove reset behavior or `last_input` engine semantics through class descriptors. Live records must retain that uncertainty; no field is advertised as post-current-action data because capture is before that submission. If ordering validation detects a future/post-action snapshot, analysis stops instead of using it as a causal feature.

## Dataset and natural protocol

New isolated `matches/v16_001` through `v16_006`. Six natural five-minute 1v1 games, normal overtime/replays/countdowns. One explicit command launches one match; nothing launches the next match automatically. A 180-second startup and 20-minute supervisor safety limit mark timeouts incomplete, rather than treating truncated games as complete. Ctrl+C also preserves an incomplete attempt. No state setting, forced maneuvers, intentional agent pauses, student, fallback or synthetic repair. Original teacher is sole control source, including native rendering/quickchat behavior. Temporary Epic configs alone differ from upstream launch configuration.

User-supplied identities in `collection_plan.json` are literal:

| Match | Teacher | Opponent | Shared session |
|---|---|---|---|
| v16_001 | Blue | human_a | human_a_v10_session_01 |
| v16_002 | Orange | human_b | human_b_v10_session_01 |
| v16_003 | Blue | human_c | human_c_v10_session_01 |
| v16_004 | Orange | human_a | human_a_v10_session_01 |
| v16_005 | Blue | human_b | human_b_v10_session_01 |
| v16_006 | Orange | human_c | human_c_v10_session_01 |

These are three people and three supplied session IDs, not six independent sessions. The `v10` substring in the supplied names does not authorize any V10 sample read. No match folder or failed run is overwritten/resumed. Failed/missing/incomplete matches stop analysis for review; replacement collection is not automatic.

Raw callback records and confirmed-submission journals are separate. Each processed callback has one callback record, with actual pre-action native state, exact float32 official 13D, actual native teacher controls, sequence/phase/ID diagnostic labels, car/ball position/velocity/angular velocity/rotation, actual selector trace and an independent observation selector probe, receipt provenance, phase transitions and previous confirmed submission. The submission journal joins by callback index and preserves before/after-send monotonic times, exact controls and local success/failure. No current submission is claimed before transport. Missing-ball/error callbacks are recorded, not silently omitted. Original selector exceptions, including empty-prediction IndexError in its far branch, propagate through the unchanged SDK error handling; no fallback controls. Successful send return does not acknowledge engine application.

Summary publications use unique immutable snapshot files, avoiding overwrite-of-open-file races. Diagnostic I/O errors mark the attempt unusable without substituting teacher output; no duplicate callback/error record is inserted. Integrity contains every match artifact, excluding the hash manifest itself. Processing uses only verified complete matches, no raw-to-training tensors.

## Exact existing 13D

Original columns 0–12 in `contracts.FEATURES`; no previous actions/private teacher phase in these inputs. Shared pinned builder remains car-native noninverted forward/right/up, relative XYZ and distance /6000, car speed /2300, masks 0/1, selected horizon /2, first-slice offset /(1/120), callback dt /(1/60). First callback dt=0 in input and null in raw timing. Missing ball zeros 0–6 and 8–11; speed and dt remain. No resets on goals/replay/kickoff/missing ball, no resampling. Histories reset only by keeping each match in a separate array. Exact selector `int((elapsed + 2 - first_slice_time) * 120)` remains unchanged; neither floor/round nor interpolation/clamping. Observation-probe selection is explicitly separate from teacher gameplay selection.

## Preregistered inputs and distance

Primary baseline: causal 13D histories H=1,2,4,8,16,32,64. Flatten actual past/current callbacks ending at t, divide by sqrt(13H); distance is RMS over normalized frozen features. No padding, future inputs or crossing match boundaries. Common probe endpoints have at least 64 callbacks; first 63 are excluded from this diagnostic and counted, but remain in raw capture/history-span statistics. Missing balls/gaps remain in histories. No reset/compensation for gaps.

Primary native block **C excludes last_input**: AirState one-hot 5, three Boolean flags, dodge_timeout/1.45 seconds, dodge_elapsed/1 second, native dodge_dir X/Y unscaled. Twelve dimensions, in that order, rounded float32. Preserve sentinel -1, no clipping or learned scaling. Direction frame is not inferred; same native component convention used throughout. The block is a diagnostic encoding, not an official observation extension.

Secondary blocks, only at H1 and H64: A=previous confirmed native teacher submission (eight native channels); B=native last_input (eight native channels). Analog values unchanged, buttons encoded 0/1. These do not become primary evidence for a representation repair. A is teacher-controlled context, not a future student's own-history robustness test.

Augmented distance squared = RMS(history13 difference)^2 + RMS(current diagnostic block difference)^2, equal block weight 1. Only native C is tested against every H baseline. No combinations, tuned weights/k/scales, projection fitting, classifier training, architecture sweep or test reads. Float64 arithmetic for search; original features/diagnostic transforms round float32. Exhaustive block search, deterministic distance/reference-order ties; returned distances recomputed directly (roundoff still a limitation). Exactly 32 uniform neighbors; majority ties select lowest mode index. Chase steering is mean native Chase-neighbor steering when predicted mode is Chase, zero otherwise.

## Reference/query and subsets

Leave-one-complete-match-out across all six **new audit** matches, five references and one query per fold. No V10 train/validation/test samples read, and no student logs used as labels. This is match-disjoint, not opponent/session-disjoint.

Fixed query union per match: every 128th endpoint starting at index 63; all pure Jump/Front-dodge callbacks; at most 256 evenly spaced eligible endpoints in each boundary, kickoff, inverted, tilted, missing-ball and long-gap subset. Representative metrics use only the every-128 grid; targeted union results are not population estimates. No label-driven changes to collected data. Label-based diagnostic query/subset selection never supplies labels to distance inputs.

Sequence boundaries: action-mode changes with ball present on both adjacent callbacks and dt<=5/120+0.0001 seconds; evaluate the fixed ±5-callback neighborhood. This is a diagnostic subset definition, not altered teacher timing. Missing-ball/gap subsets are reported separately. Long gap >=1 second. Kickoff is actual MatchPhase.Kickoff. Inverted up.z<0, tilted up.z<.5, derived from recorded orientation for strata only; neither asserts ground contact or optimal recovery.

Report overall/representative mode accuracy, Pure Jump accuracy/recall (same class-conditional mode measure), native Jump-button recall, Front-dodge recall, boundary accuracy, Chase accuracy and steering MAE, true-label probability, close-neighbor probability and ambiguity rate; each fold and each subset separately. Pool rates with corresponding denominators; report counts. No silent tuning on results.

## Close conflicts, purity and `last_input`

Close means current13 RMS<=.02. Material difference is different mode/jump/pitch, or both Chase and steer difference>.2. For each query retain the nearest material conflicting returned neighbor (32-neighbor scope, not every possible close pair), plus modes/actions/native flags/timers/direction/last_input, orientation/angular velocity and native callback times. Cross-match elapsed differences are differences between independent clocks, not causal delay.

Independent/usable support uses conservative event deduplication: original sequence IDs for consumed maneuver phases; otherwise fixed two-second clock bins. Greedily retain event-disjoint pairs across all folds, with at least one maneuver endpoint; neither event may be reused. Private sequence IDs only deduplicate diagnostic support, never enter inputs. These event cases are less correlated, not proven statistically independent. Systematic usefulness requires the preregistered probe gains across held-out matches; merely having different native states is not evidence.

For each native field report exact categorical P(mode|state) and P(state|mode). Private consumed phase splits Neutral release/coast **only as diagnostic labels**, never inputs. Timer bins fixed: sentinel -1; other negative; [0,.05), [.05,.1), [.1,.2), [.2,.8), [.8,1.45), [1.45,+inf). Retain raw distributions. Native direction uses fixed component-sign categories, without claiming a coordinate frame. last_input analog descriptive bins use exact zero, [-1,-.5), [-.5,-.02), [-.02,.02] nonzero, (.02,.5], (.5,1]; buttons exact. No optimized thresholds.

last_input timing: compare exact buttons and analog tolerance 1e-6 with prior confirmed submissions at lags 1..8; retain eligibility denominators, changed-native-input cases and current-label coincidence separately. No best lag is used to rewrite/align values. Flat action runs are uninformative for unique lag identification. Equality does not prove successful execution. Report native flag transitions and documented ground-reset observations, including contradictions; do not overwrite/reset values. Sources do not identify the installed engine producer commit. Where observations cannot distinguish engine-applied vs last-received input, keep that question unresolved.

## Evidence criteria — fixed before collection

Primary native evidence compares H64+C to H64 13D, the longest available predefined causal context, not only V15's H16. Also expose all shorter histories to show potential redundancy and timing support.

Require at least 30 usable event-disjoint close maneuver conflicts; >=10 percentage-point boundary gain, >=5-point Pure Jump gain, >=5-point Front-dodge gain; each of these gains strictly positive in every held-out match; representative mode regression <=1 point. C excludes last_input, and every used value must pass pre-action provenance validation. No alternate ambiguity-reduction gate is used. Insufficient rare-mode support in any fold is inconclusive, not filled with a synthetic value. Thresholds never revised after results.

History evidence uses the same gains/guards comparing H64 13D versus H1 13D. Classification: A if native evidence meets all gates but longer history alone does not; C if both meet gates; B only if history meets gates, close support exists and C's three gains over H64 each have absolute magnitude<2 points (predefined descriptive small-effect rule); D otherwise. All classifications are scoped finite observational evidence, not information-theoretic necessity/absence or policy success. No good probe authorizes training.

## Temporal horizon reporting

H1/2/4/8/16/32/64 span stats use all endpoints with enough actual callbacks: count, median, p10/p90, min/max and distinct maneuver-event endpoints. Report fraction of spans>=1.1 seconds as a nominal-duration descriptor, not proof of full sequence coverage or GRU sufficiency. The source's strict elapsed>duration behavior and callback finish step can lengthen real sequences; irregular delivery and pending memory across missing ball/phase transitions matter. No artificial clock/tick padding.

## Outputs / hard stop

After live capture and separately launched analysis: raw match journals/config/metadata/integrity in this directory; `results_v1` neighbor evidence, lag timing evidence, conditional distributions, physical conflict examples, report and hashes. Publish final `training/reports/v16_native_maneuver_observability_20261006.md/.json` with all twenty requested sections. Final report does not exist at preparation time and cannot imply collection/pass. Source seal and all match hashes checked before/after analysis. No frozen artifacts/test samples opened as data. Stop for review; no feature change, policy fitting, DAgger, student launch or additional matches.

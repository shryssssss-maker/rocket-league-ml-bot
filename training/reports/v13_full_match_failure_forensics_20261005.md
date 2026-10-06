# V13 full-match failure forensics — 2026-10-05

## 1. Executive summary

Read-only review of the complete Blue-student/Orange-human_b match. **26,635 callbacks**, 33 completed shadow sequences, two pure student Jump callbacks. The 5,238/zero-Jump/nine-sequence figures in the request refer to the earlier pilot. Final score Blue 2–Orange 1. Overtime is inferred from the five-minute configuration, a tied 1–1 countdown at T=330.317, and subsequent play until T=447.117; an explicit overtime flag was not captured. One result does not establish competence.

**Four out of four detected kickoffs contain pre-contact orientation loss.** Ground-origin jump+pitch −1 outputs precede airborne pitching and inversion; no native dodge is recorded by orientation-loss onset. Three have a human first touch before the student. In the fourth, the student eventually records first contact at T=358.308, about 24 seconds after kickoff begins. Thus the initial failure is not supported as collision-induced flipping.

Two distinct prolonged pause patterns exist: inverted/Neutral near the side region, and mostly upright/Neutral near the back region. Recovery is intermittent. The current capture cannot identify a unique video-described corner episode or prove actual wall constraint. **Do not blindly aggregate these labels into DAgger.**

## 2. Exact kickoff events

| Event | Kickoff callbacks / canonical T | First student jump / bad orientation | First recorded human touch | Stable recovery in window | Outcome |
|---|---|---|---|---|---|
| K1 | 415–579; 7.058–9.833s | 456 / 484 | 579 | 672 | pre-contact airborne pitch rotation/inversion; human first recorded touch |
| K2 | 4223–4380; 70.733–73.392s | 4279 / 4307 | 4380 | 4465 | pre-contact airborne pitch rotation/inversion; human first recorded touch |
| K3 | 9773–9928; 163.833–166.458s | 9815 / 9843 | 9928 | 10327 | pre-contact airborne pitch rotation/inversion; human first recorded touch |
| K4 | 19902–20199; 334.333–339.342s | 19942 / 19970 | 21963 | not in window | pre-contact airborne pitch rotation/inversion; student eventually first recorded touch |

| Event | Student first touch callback / T | Human first touch callback / T |
|---|---|---|
| K1 | 2015 / 33.933334s | 579 / 9.833333s |
| K2 | 5033 / 84.358330s | 4380 / 73.391670s |
| K3 | 9974 / 167.225006s | 9928 / 166.458328s |
| K4 | 21323 / 358.308319s | 21963 / 369.049988s |

Touch searches cover the interval until the next kickoff/end, not only the short orientation timeline. Each touch’s original repr, first observed callback, and immediate subsequent packet states are preserved in JSON. These identify recorded ball touches, not every collision or the game application time of a submitted control.

Orientation criterion: body-up dot world-up = cos(pitch)×cos(roll). Values below −0.5 support inversion; below +0.5 indicate severe tilt. This is a physical orientation measure, not a semantic dodge/flip or collision detector. Native has_jumped/has_dodged are reported separately.

At initial K1, callback 456 outputs [0,0,−1,0,0,true,false,false] from OnGround; callback 484 has up_z≈0.48 at [222,−3157,181], and 494 has up_z≈−0.27. At callback 579 (human touch), up_z≈−1 with student Neutral. The shadow changes Jump→release→Front dodge→coast based on its independent sequence, whereas the student holds jump while pitching. K2/K3/K4 show the same ground-jump/airborne-pitch pattern. K3 alone has two pure Jump callbacks (9815–9816), followed by Front dodge without an intervening release.

A. Kickoff orientation loss is systematic in these four observations, not proven systematic over all matches. B. The temporal action error starts before first recorded contact. Steering/throttle differences are also recorded; their independent contribution is unproven. C. The shadow differs at relevant pre-contact callbacks, particularly sequence timing. D. Initial inversion precedes contact; later collisions or surfaces may prolong it. E. The original teacher also flips on a speed band, but this run cannot establish that it would suffer the same outcome under its own physics.

## 3. Exact side/back-region pause candidates

| Candidate | Callbacks | T / duration | Up-z range | Student modes | Shadow Chase while student Neutral | Confidence |
|---|---|---|---|---|---:|---|
| W1 | 8187–8752 | 137.425s / 9.500s | -1.000…-0.998 | {'Neutral': 566} | 566 | high: sustained inverted side-region standstill |
| W2 | 8850–8917 | 148.567s / 1.117s | -0.999…-0.979 | {'Neutral': 68} | 68 | lower: brief low-speed continuation near back/side region |
| W3 | 13070–13902 | 219.350s / 14.008s | 0.087…1.000 | {'Neutral': 833} | 768 | high: prolonged mostly upright back-region Neutral pause |
| W4 | 13944–14012 | 234.058s / 1.133s | 0.978…0.994 | {'Neutral': 69} | 69 | lower: brief low-speed continuation near back/side region |
| W5 | 14046–14302 | 235.758s / 4.300s | 0.985…1.000 | {'Neutral': 257} | 257 | lower: brief low-speed continuation near back/side region |
| W6 | 14378–14424 | 241.350s / 0.767s | 0.999…0.999 | {'Neutral': 47} | 47 | lower: brief low-speed continuation near back/side region |

W1 is the strongest inverted-side pause candidate: 8187–8752, 9.500 s; at 8300/8500/8752 the car is essentially fixed around [−3807,391,42], upside down, with Neutral controls. Shadow Chase differs on those same callbacks. W2 is a brief continuation before eventual recovery. Neither proves contact with a wall: x≈−3807 is only a regional proxy.

W3 is the strongest back-region pause candidate: 13070–13902, 14.008 s. It begins tilted, then is upright by 13200 at [1433,−4949,18], speed≈1 UU/s, still Neutral while shadow Chase. W4–W6 are briefer nearby pauses. The prolonged pause is not wholly explained by inversion. Repeated steering into a wall is not supported during all-zero Neutral portions; no effective recovery attempt is recorded there. Later motion can include gravity or human/ball interaction and is not credited solely to the model.

Every candidate has a callback-by-callback lead-up/pause/recovery window in callback_windows.json, including exact physics, orientations, both controls, sequence states, observations, prediction selections and callback dt. No rows were resampled or corrected.

## 4. Successful versus prolonged recovery

| Bad-orientation onset | End | Duration | Stable recovery | Student modes during episode |
|---:|---:|---:|---:|---|
| 484 | 665 | 3.058s | 672 | {'Front dodge': 2, 'Neutral': 135, 'Chase': 45} |
| 4307 | 4459 | 2.575s | 4465 | {'Front dodge': 2, 'Neutral': 105, 'Chase': 46} |
| 7705 | 8960 | 21.125s | 8965 | {'Chase': 341, 'Neutral': 915} |
| 15667 | 16824 | 19.483s | 16830 | {'Chase': 337, 'Neutral': 821} |
| 19970 | 21019 | 17.625s | 21024 | {'Front dodge': 2, 'Neutral': 1026, 'Chase': 22} |

The initial kickoff episode recovers near callback 666 and upright ground motion is visible by 9000 in the long side episode. In contrast, the episode 7705–8960 lasts 21.125 s and includes the 9.5 s inverted standstill. Another inversion 15667–16824 lasts 19.483 s, including a zero-speed inverted Neutral state around 16500. The overtime episode lasts 17.625 s. These are not permanent failures: prolonged cases eventually become upright without a goal-reset being counted as recovery.

Milestone snapshots preserve ball geometry/prediction, actual speed/orientation, controls and shadow labels. GRU continuity is verified through hidden hashes, but hashes cannot explain what state the GRU encoded. Actions use deterministic argmax/tanh inference, not stochastic sampling. Different recovery durations correlate with position, motion and Neutral/Chase periods; sparse examples cannot separate gravity/collision dynamics from action effects causally.

## 5. Teacher-versus-student failure classification

| Category | Evidence-based classification |
|---|---|
| Kickoff inversion | Primarily student temporal-action learning failure: premature jump+pitch and missing release precede inversion. Teacher speed-only flip trigger is also limited; its own outcome is untested. |
| Upright back-region pause | Likely student-learning failure: prolonged Neutral while original shadow Chase on actual state. Wall constraint and hidden-state cause remain ambiguous. |
| Inverted side pause / recovery | Mixed learning and representation limitation; inherited teacher limitation also matters because it has no explicit orientation-aware recovery routine. |
| Missed jump timing / front-dodge sequence | Strong student-learning evidence: all 33 shadow sequence starts missed; only two pure Jump callbacks, and explicit no-release kickoff patterns. |
| Later shadow dodge labels on learner physics | Stateful-label/action-prerequisite mismatch, not proof of helpful expert behavior. |

Source: python-example-original/src/bot.py, MyBot.get_output and begin_front_flip. Teacher ignores other logic while pending; at 750<speed<800 starts 0.05s Jump, 0.05s release, 0.2s jump+pitch−1, 0.8s coast. It has no dedicated kickoff strategy, wall navigation or inversion recovery; this is an inherited reference limitation, not a newly introduced bug.

## 6. 13D observability

Exact features remain ball-relative XYZ, predicted-relative XYZ, distance, speed, two availability masks, selected horizon, first-slice offset and callback dt. Positions are car-relative, so orientation influences them implicitly. It is incorrect to say there is no orientation information whatsoever. However, no world-up vector, absolute car position, angular velocity, velocity direction, native air/jump/dodge state, wall distance or executed action history is supplied.

A relative target vector does not uniquely determine orientation relative to gravity: multiple world/car configurations can produce the same relative features. Speed discards direction. Thus inversion versus upright or wall contact versus ordinary low-speed chasing is not generally identifiable from a single 13D observation. Temporal changes can offer clues, but do not guarantee identification: stationary periods, unobserved forces/contact and absent executed-action inputs can remain ambiguous. GRU state hashes do not prove reliable inference.

Approximate opposite-orientation feature neighbors from this match are included in JSON. They are explicitly not exact identical-input collisions, and neither prove nor disprove history-level ambiguity. No feature change or architecture is proposed/implemented in this audit.

## 7. DAgger implications

Blind aggregation is **not technically justified**: 216 shadow Front-dodge labels occur while the actual car is grounded with has_jumped=false. A shadow sequence follows its own prior expert actions even when the student did not execute them. The full-match evidence therefore requires an expert-label/prerequisite qualification decision before treating these as imitation targets. Observation identifiability and temporal supervision also deserve tests, but this audit does not choose a repair, filter, objective or altered teacher.

Dropping labels, resetting the teacher, adding recovery rules or changing input features would each alter a contract and must be separately reviewed. No such alteration is made here.

## 8. Evidence limits and integrity

- Actual packets contain rotation and jump/dodge flags, but no collision impulse/contact pair.
- Touch was logged as repr; parse only explicit game_seconds, preserve original strings. Null/missing new touch is not proof of no collision.
- No video timestamp identifies which wall episode the human meant. Rank candidates, do not assert a unique match.
- No collision mesh/field geometry captured; location thresholds are regional proxies.
- Only GRU hidden-state hashes and outputs captured, not hidden vectors; no identifiability/causal memory proof.
- Teacher never controlled physics in this match; matching/different shadow labels are not outcome counterfactuals.
- Processed callbacks only, no proof of lossless delivery; no new validation/test samples accessed.

Verified 235 capture-file hashes and 43 protected sources before analysis. Rehashed all existing session files after report construction; no existing artifacts were written. No training/validation/test samples, fitting, aggregation, simulation or live launch used.

## 9. Recommended next experiment only

**E — collect targeted diagnostic evidence before deciding**, subject to separate approval. Focus on naturally occurring kickoff pitch/jump/release timing and paired fast/prolonged recovery states, with time-aligned video and structured native touch/contact-prerequisite evidence. Use the unchanged student and faithful shadow; do not force expert-sequence state resets or treat labels as a dataset. Specify label qualification/observability hypotheses and success criteria before capture.

Why E: the capture establishes action-sequence failure and Neutral pauses, but cannot uniquely locate the human’s corner event, attribute wall impulses, explain GRU hidden state, or establish which counterfactual shadow labels are appropriate. This uncertainty makes direct DAgger (A) premature and does not yet justify selecting B, C or D as the sole repair. No next experiment is launched.

Full callback evidence: `C:\Users\shreyas\Desktop\model wars\training\reports\v13_full_match_forensics_20261005\callback_windows.json`. JSON companion: `C:\Users\shreyas\Desktop\model wars\training\reports\v13_full_match_failure_forensics_20261005.json`. Stop for review.

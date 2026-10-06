# Reconstructed: Original Python Example Live Validation

This document reconstructs the missing live-validation report from surviving harness documentation, saved session summaries, and the recorded handoff. It is not the original report.

Reconstruction date: 4 October 2026. Session: `20261004_043919_979748`.

No passage is presented as verbatim from the missing report. Exact field names and values below come from the identified surviving summaries. The user acceptance quotation is preserved from the harness README and handoff, not from the missing report.

## Evidence sources and attribution

Only these sources were used:

- [Harness README](../live_reference_test/README.md): surviving documentation of isolation, preparation, acceptance, interruption, and subsequent reader correction.
- [bot_summary.json](../live_reference_test/sessions/20261004_043919_979748/bot_summary.json): recorded controller counters, callback timing, phases, and latest sample.
- [run_summary.json](../live_reference_test/sessions/20261004_043919_979748/run_summary.json): recorded readiness, launcher outcome, scores, roster, and protected-source integrity result.
- The project handoff supplied in this conversation: current strategy, interpretation of the accepted reference gate, and work that remains unvalidated. This is conversational evidence, not a recovered report or independent runtime measurement.

The existing `python_example_original_live_validation_20261004.md` is left untouched. This new document does not replace it or repair references to it.

## Direct evidence: statements in the harness README

These are attributed statements from the surviving documentation; their underlying implementation or preparation tests were not independently rerun for this reconstruction.

- The original reference revision is `fd061f457bf19175b4a9b3b3d7811a987044c64d`.
- The harness preserves `python-example-original/` and the modified `python-example/`. It documents a manifest of 43 protected source/configuration files checked before and after a match.
- The harness uses the existing `python-example/venv/Scripts/python.exe`, installs nothing, and imports no ML policy.
- The default configuration is blue human versus orange recovered bot in five-minute standard soccer, with normal boost/mutators, countdowns, and replays. The documented launch changes are Steam to Epic, rendering enabled, and an isolated agent configuration path.
- The diagnostic wrapper calls the original gameplay functions once per callback and returns the same `ControllerState` object without field changes. Prediction instrumentation is in memory; original files are not patched.
- Fourteen synthetic callbacks reportedly matched all nine RLBot controller fields, including `use_item`. The original and instrumented paths both raised `IndexError` for empty prediction input. These are preparation checks, not simulator-equivalence validation.
- The README explicitly records practical reference acceptance for this session and the user's observation: "ran perfect acc to me".
- It records the launcher interruption while reading the periodically replaced diagnostics summary, with no complete match result.
- It documents a subsequent `SummaryReader` correction: retain the last complete snapshot across brief read failures, retry on later polls, and stop after ten seconds of persistent read failure. Short mocked read/recovery/timeout checks reportedly passed with protected hashes unchanged.
- It explicitly states that the corrected reader has not yet been tested in another live match and does not alter teacher decisions.

The README's observation instructions are instructions, not proof that each requested maneuver or interaction occurred during this session. A detailed per-maneuver visual record is not established by surviving artifacts.

## Direct evidence: exact bot summary counters

| Field | Recorded value |
|---|---:|
| `initialized` | `true` |
| `callbacks` | 18301 |
| `original_errors` | 0 |
| `last_error` | `null` |
| `diagnostic_errors` | 0 |
| `invalid_outputs` | 0 |
| `prediction_requests` | 3412 |
| `prediction_slices_selected` | 3412 |
| `prediction_steering_callbacks` | 3400 |
| `flip_starts` | 25 |
| `sequence_completions` | 24 |
| `observed_teacher_touch_events` | 47 |
| `duplicate_frames` | 0 |
| `frame_rollbacks` | 0 |
| `game_clock_rollbacks` | 0 |
| `events_log_capped` | `false` |

The summary explicitly defines touch counting as distinct observed latest-touch timestamps; skipped packets can hide touches. Therefore 47 is an observed timestamp count, not a guaranteed exhaustive physical-contact count.

### Recorded phase callback counts

| Phase | Callbacks |
|---|---:|
| `MatchPhase.Countdown` | 2177 |
| `MatchPhase.Kickoff` | 1585 |
| `MatchPhase.Active` | 11428 |
| `MatchPhase.GoalScored` | 1470 |
| `MatchPhase.Replay` | 1641 |

### Recorded callback timing

| Physics-frame gap | Count |
|---|---:|
| 1 | 336 |
| 2 | 17369 |
| 3 | 354 |
| 4 | 115 |
| 5 | 126 |

| Field | Recorded value |
|---|---:|
| `callback_wall_ms_mean` | 17.918753808738327 |
| `callback_wall_ms_max` | 249.53010003082454 |
| `game_dt_max` | 0.04168701171875 |

The summary states that timing includes instrumentation overhead and that observed frame gaps are not automatic failures. These values do not establish fixed 60 Hz delivery or actual controller application latency.

### Latest recorded sample

The latest sample records physics frame `37276`, elapsed game time `310.6333312988281`, and `MatchPhase.Kickoff`. It contains two players, one ball, `sequence_active = true`, sequence index after the callback `3`, and returned sequence phase `3`.

All eight recorded controls are zero/false. The returned sequence step records start time `309.8500061035156`, nominal duration `0.8`, and elapsed duration `0.7833251953125`. Scores are blue 3, orange 5.

This establishes an active sequence at the latest recorded sample. Its exact state at every instant between that sample and launcher termination is not established by surviving artifacts.

## Direct evidence: run summary

| Field | Recorded value |
|---|---|
| `status` | `error` |
| `teacher_ready` | `true` |
| `teacher_initialized` | `true` |
| `teacher_controls_observed` | `true` |
| `protected_sources_unchanged` | `true` |
| `final_scores` | Team 0: 3; team 1: 5 |
| `error_type` | `PermissionError` |

The recorded error is permission denied while reading this session's `bot_summary.json`. The recorded roster identifies `Recovered Python Example` on team 1 and `Player` on team 0.

The recorded phases are Countdown, Kickoff, Active, GoalScored, and Replay. `MatchPhase.Ended` is absent. The field name `final_scores` does not establish a finalized match result: it is the last score retained by the interrupted launcher.

This summary does not contain an `experiment = V0` marker, post-fix reader-failure counters, or an explicit clean cleanup result. Those details are not established by surviving artifacts for this session.

## Reconstructed narrative and gate interpretation

The counters support sustained execution of the recovered reference in live Rocket League: it initialized, delivered controllers, observed touches, started and completed timed sequences, and selected predictions used for steering. Recorded original-controller errors and invalid outputs were zero. The README and handoff record the user's practical acceptance of the observed behavior.

The distinction between 25 flip starts and 24 completions is consistent with the latest sample showing an active sequence. It does not establish that all 25 starts produced successful physical front flips; per-flip physical success is not established by surviving artifacts.

The run reached scoring and replay phases, with orange leading 5–3 when interrupted. It did not establish a completed match, a tournament win, or competitive strength.

The permission-denied launcher error, zero recorded original errors, and README explanation support classifying the interruption as a harness diagnostics-reader failure rather than an original-controller failure. The periodically replaced-file race explanation is attributed to the README and handoff; an independent low-level reconstruction of the race is not established by surviving artifacts.

**Reconstructed gate conclusion:** the practical known-good reference-behavior gate passed according to the recorded user acceptance. Clean post-fix full-match harness validation did not pass in this session and remains pending. This conclusion reconstructs the current project interpretation; it is not quoted from the missing report.

## What this run does not establish

The following are not established by surviving artifacts:

- Competitive teacher strength or a finalized 5–3 match outcome.
- Exact equivalence to the user's earliest historical working installation beyond practical acceptance of this recovered reference.
- Exhaustive physical-touch counts, touch usefulness, or success of every physical flip.
- Exact live action latency, hold behavior, or simulator/live timing equivalence.
- RLBot/RocketSim prediction compatibility, bounce parity, or timestamp equivalence.
- A validated RocketSim teacher adapter or a frozen student observation/action/timing contract.
- Correct learned GRU memory, behavior-cloning success, closed-loop student stability, or successful student live deployment.
- A teacher demonstration dataset, student training, PPO from BC, or an accepted final ML baseline. The handoff explicitly states these have not started or been completed as applicable.
- Another live validation of the corrected diagnostics reader.

## Pending V0, as documented and handed off

V0 is one isolated live reference run after the diagnostics-reader correction, using the original teacher without changing gameplay. The harness README requires natural match completion (`status = match_ended`), an initialized/ready teacher with observed controls, unchanged protected hashes, and no original or diagnostic errors. Invalid outputs and cleanup/read failures must also be inspected.

A successful reconstruction of documentation does not complete V0. No live run, source change, dataset collection, simulation validation, or training was performed to create this document. V0 remains the immediate next gate; later validation and demonstrations must not be treated as authorized or complete by this report.

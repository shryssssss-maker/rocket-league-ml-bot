# Constrained provider validation and V7 capture preparation

Status: V6 latency measurement pending. V7 capture prepared but not launched; no provider comparison performed.

## Option A

`training/v6_prediction_test/constrained_provider.py` checks RocketSim 2.2.1 and binary SHA256 `e3ee24ca82445b4bfcc754583f6778d7b0d8b7a7f7d64f872be8c65e621a63d0`. Exact build commit remains unknown. Collision mesh hashes are recorded separately.

Each call constructs BallState with constructor arguments, creates a fresh car-free SOCCAR arena, sets the state, calls `get_ball_prediction(num_states=241, tick_interval=1)`, retains all 241 states, assigns `T + i / 120`, and releases the arena. No BallPredictor, arena/cache reuse, stride, trimming, horizon change, or timestamp correction is used.

The short validator passed 18 cases across six ball states and three canonical times. It verified caller dictionaries unchanged, sample-zero position/velocity, all timestamps, original teacher selected index/time, and creation/prediction/release of 18 distinct arenas. All 43 protected hashes remained unchanged. See `v6_option_a_contract_20261004.json` for results. This is not a latency result or RLBot prediction parity claim.

Run `validate_option_a.py --samples 2000` using the training venv for 20 warmups and 2,000 varied ball-state predictions. The timed operation includes construction, prediction, returned-state finite checks, timestamp attachment and arena release; initialization and input generation are excluded. It reports min/median/mean/p90/p95/p99/max and counts exceeding one 120 Hz tick and one 60 Hz interval, with raw measurements. These intervals are diagnostics, not invented acceptance thresholds. Every sampled caller state is checked for mutation.

## Capture runner

`training/v7_prediction_test/capture.py` is capture-only, using the existing live interpreter. `--check` passed installed imports, state-setting message construction, temporary config parsing and all 43 protected hashes without launching RLBot or Rocket League.

The runner generates temporary session configs for Epic, standard Stadium soccer and a 1v1 roster: a neutral diagnostic agent and a human car. Replays/countdown are skipped and match length is unlimited only in that temporary config. Do not supply human controller inputs during capture. Neither recovered nor modified bot source is executed as gameplay logic or edited.

There are 120 requested fixtures: twelve signed/orientation variations each for stationary, free flight, rolling, spin, floor, wall, corner, ceiling, goal-mouth and car-near-path groups. It requests ball and both car states and retains three delivered snapshots per fixture. It records the requested state, injection time/frame, actual packet physics, full unmodified RLBot prediction slices, canonical elapsed time, callback delta inputs, and the original helper's selected index/time/target. It does not replace actual packet states with fixture requests or align predictions artificially.

Capture readiness does not prove that a state-setting request was applied: that must be examined from the recorded request/actual-state data before comparison. Prediction service age, phase interruptions and initial-state differences are retained, not filtered to force agreement. A ten-second pending-fixture timeout is recorded as a failure; the runner has an overall six-minute bound. Empty prediction lookup errors are reported and propagated rather than hidden.

The installed `rlbot/managers/bot.py:Bot.set_game_state` invokes `fill_desired_game_state` with four positional arguments, while installed `rlbot/utils/__init__.py:fill_desired_game_state` accepts three. The diagnostic directly builds that same DesiredGameState message with the helper's actual signature and sends it through the existing SocketRelay. No installed package changes were made.

Expected capture outputs: `captures.jsonl`, `capture_summary.json`, `run_summary.json`, and metadata plus temporary launch configs under `training/v7_prediction_test/sessions/<timestamp>/`. A complete capture has 120 fixtures and 360 snapshots; failures or missing snapshots mean capture completeness needs review. Capture completion is not a V7 pass.

## Next gate

Review the Option A latency report before launching capture. After capture, review completeness and evaluate the previously declared V7 timestamp, target-position, Chase-steering and observation-feature gates by scenario, including bounce, goal and sign failures. No comparison, demonstrations, ML training or production integration has started.

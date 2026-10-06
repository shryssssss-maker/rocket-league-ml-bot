# Recovered reference: isolated live 1v1 check

Purpose: verify the recovered original's live behavior before designing an ML student or demonstrations. The practical live reference gate was **accepted by the user** in session `20261004_043919_979748`. See the [live validation report](../reports/python_example_original_live_validation_20261004.md). The launcher interrupted that run on a diagnostics-file read failure; a complete match result was not recorded.

## Source and isolation

- Original revision: `fd061f457bf19175b4a9b3b3d7811a987044c64d`.
- `python-example-original/` and `python-example/` are not modified by this harness.
- `source_hashes.json` records 43 protected source/configuration files across both directories. The launcher checks them before and after the match.
- The launcher uses the existing `python-example/venv/Scripts/python.exe`; it installs nothing and imports no ML policy.
- All temporary launch configurations and runtime diagnostics go in timestamped `sessions/` directories, ignored by this folder's `.gitignore`.
- `-B`/`PYTHONDONTWRITEBYTECODE` prevents imported original source from generating bytecode files.

The isolated match config preserves the original match/mutators and changes only Steam to Epic, rendering to enabled, and the agent configuration path. The isolated agent config reads the original loadout and uses the known-working interpreter through a session-local `run_agent.cmd`. This avoids stripping quotes from executable paths containing spaces in RLBotServer's command-shell layer. Its working directory is the new session folder.

The default match is **blue human vs orange recovered bot**, five-minute standard soccer, normal boost/mutators, kickoff countdown and replays retained. No state setting, scripted opponent, action conversion, timestep throttling, dataset collection, or training occurs.

## Run in PowerShell

For the V0 full-match harness verification only:

```powershell
& '.\python-example\venv\Scripts\python.exe' -B '.\training\live_reference_test\launch.py' --v0
```

Let the match end naturally; five minutes of match time plus startup, countdowns, replays and possible overtime can take substantially longer in real time. Do not press Ctrl+C to finish a successful V0 run. The launcher prints `MATCH ENDED` when it observes the natural ended phase and records `experiment = V0` in the session metadata and run summary. Ctrl+C leaves this experiment incomplete. No later experiment runs automatically.

V0 review requires natural end (`status = match_ended`), initialized/ready teacher with observed controls, unchanged protected source hashes, and no original or diagnostic errors. Invalid outputs and any cleanup/read failures must also be inspected. Share both summaries and any traceback before declaring V0 passed.

**V0 passed in repeat session `20261004_055830_563928`:** natural match end, orange 8–blue 2, zero original/diagnostic errors or invalid outputs, unchanged protected hashes, and one successfully handled writer replacement retry. No later experiment was started. See the [V0 report](../reports/v0_live_reference_harness_20261004.md).

The first V0 session, `20261004_054339_069322`, ended naturally, orange 4–blue 2, with zero original errors and invalid outputs, but one summary-file replacement error; it did not pass. The isolated writer correction retries transient replacement failures without callback sleeps, records retry/deferred-publication counters, and preserves persistent failures as diagnostic errors. Retirement permits a bounded 200 ms final flush. The passing repeat validated that recovery in live play. Original session evidence remains unchanged.

From the project root:

```powershell
& '.\python-example\venv\Scripts\python.exe' -B '.\training\live_reference_test\launch.py'
```

This starts/reuses RLBotServer and launches Epic Rocket League through the same `MatchManager.start_match(Path)` route used by the existing shell. Startup may take a minute or more. It waits up to 180 seconds for the named orange teacher to initialize and produce callbacks, in a two-player blue/orange match. In plain mode, it requires the named car to show non-neutral input. Game activity alone does not satisfy startup. A match, including replays and possible overtime, takes longer; Ctrl+C stops it early.

The launcher prints the session directory, `WAITING FOR TEACHER` while the agent is unready, then `TEACHER READY` and periodic `LIVE` lines with phase/frame/scores. Agent diagnostics print `OG reference ready` and approximately one `OG` line every two seconds. Depending on server process output routing, agent messages may appear in another RLBot output window; the session files remain the reliable diagnostics location. An original exception recorded by the wrapper stops this verification run and is retained in its diagnostics.

Ending the match or pressing Ctrl+C saves `run_summary.json`, stops the test match, and disconnects the launcher. RLBotServer is left available; the launcher does not forcibly terminate an existing shared server.

## Observe

1. Initially leave the blue car still and observe approach, steering, acceleration, ball contact, and front flips.
2. Then interact with the ball so that the bot encounters moving balls and longer approaches. Its original white line shows the driving target; the cyan line from the ball indicates a selected prediction target.
3. Observe goals/kickoffs/replays, recovery after collisions, and any circling/stuck behavior. Note whether the behavior matches the remembered reference.
4. If exceptions repeat or the bot fails to start, stop with Ctrl+C and share the first traceback. The wrapper does not repair or hide the original's errors.

The sequence nominally lasts 1.10 seconds, with callback-dependent boundaries. Do not infer successful physical flips solely from `flip_starts`; confirm visually. Likewise, selected prediction slices establish source execution, while meaningful influence on long-distance steering needs the logged target and gameplay observation.

## Diagnostics

`diagnostic_bot.py` imports the original `MyBot` class and calls its gameplay functions once per callback. It returns the **same ControllerState object** with no field changes. A temporary in-memory forwarding wrapper observes the original prediction helper's actual return value; the original files are not patched.

The wrapper records:

- Physics frame, elapsed game time, match phase, speed, distance, car/ball locations.
- Actual prediction lookup request, slice count/index/times, selected target, and whether the prediction was used for chasing instead of superseded by a flip.
- Flip starts, sequence phase transitions, consumed phase vs index after the callback, and step start time/elapsed/duration.
- All eight controller fields, output validity, sampled original-logic runtime.
- Every-callback frame-gap histogram, duplicate/rollback counts, wall interval mean/max, game-time gaps, and phase callback counts.
- Observed teacher touch timestamps and scores. Touch counts are observations of distinct latest-touch timestamps, not guaranteed exhaustive physical contact counts.
- Original errors rethrown unchanged and separate diagnostic errors.

`bot_events.jsonl` is capped at 1 MiB. `bot_summary.json` is refreshed periodically and on retirement; counters continue after the event cap. Timing includes instrumentation overhead. Original exceptions can still be repeatedly printed by RLBot itself, so stop a failing run promptly.

## Share results

After the launcher exits:

```powershell
$liveRun = Get-ChildItem '.\training\live_reference_test\sessions' -Directory |
    Where-Object { Test-Path (Join-Path $_.FullName 'run_summary.json') } |
    Sort-Object Name -Descending | Select-Object -First 1
Get-Content (Join-Path $liveRun.FullName 'run_summary.json')
if (Test-Path (Join-Path $liveRun.FullName 'bot_summary.json')) {
    Get-Content (Join-Path $liveRun.FullName 'bot_summary.json')
}
```

Share both summaries, the first traceback if present, and observations about remembered behavior, useful contacts, scoring, flips, stuck/circling behavior, and transitions. The large event log is not needed initially. If no summary exists, share the startup terminal output and printed session path.

## Optional plain comparison

If instrumentation appears to affect behavior or cadence, a separate run can execute the original `src/bot.py` directly, with the same isolated Epic configuration:

```powershell
& '.\python-example\venv\Scripts\python.exe' -B '.\training\live_reference_test\launch.py' --plain
```

Plain mode has the original rendering and launcher phase/score reporting, but no internal prediction/sequence diagnostics or `bot_summary.json`.

For a quick configuration/source/import check only, add `--check`. It creates isolated config files but launches no game, bot connection, or server. The Python helper `load_match_config` rejects the upstream `Soccer` token; this harness preserves the proven server-side TOML path route rather than rewriting game-mode semantics.

## Preparation checks completed

- Protected source hashes and original revision verified.
- Match TOML parsed as blue human/orange bot; isolated agent/loadout configuration parsed by the installed RLBot helper; original imports resolved using the live interpreter.
- Fourteen synthetic callbacks compared all nine RLBot controller fields (including `use_item`) against the original: exact agreement across prediction/chasing, flip phases/completion, out-of-range prediction fallback, and missing-ball handling.
- Empty prediction raised the same `IndexError` in the original and instrumented paths. The harness deliberately retains that original failure behavior.
- No synthetic check instantiated RocketSim, connected a bot, started RLBotServer, or launched Rocket League.

These checks establish harness preparation only. They do not establish live success, physical flip execution, useful gameplay, or the remembered bot's identity.

## Accepted live run and diagnostics correction

Session `20261004_043919_979748` verified agent startup and sustained controller callbacks. The user reported “ran perfect acc to me.” Saved counters show 18,301 callbacks, 47 observed teacher touch timestamps, 25 flip starts, 24 sequence completions, and zero original errors or invalid outputs. Orange led 5–3 at interruption; the match did not reach a recorded ended phase.

The launcher encountered `PermissionError` reading the periodically replaced `bot_summary.json`. Its new `SummaryReader` retains the last complete snapshot across brief read failures and retries at the next poll. It stops after ten seconds of persistent read failure and records read-failure counts. Short mocked read/recovery/timeout checks passed, and protected hashes remained unchanged. This reader correction has not yet been tested in another live match. It does not alter teacher decisions.

## First live attempt and launcher correction

Session `20261004_043030_840777` reached game activity but did not start the reference agent. RLBotServer reported `'C:\Users\shreyas\Desktop\model' is not recognized`, agent exit code 1, and ready agents 0/1. No bot diagnostics were created. This was a launcher quoting failure, not a gameplay result. The source directories remained unchanged.

The corrected launch uses a space-free local batch filename and retains quoting inside the batch. `--check` now exercises that actual shell/batch path in help-only mode, without starting an agent connection or a game. Readiness now requires teacher evidence instead of only an active match; the new `run_summary.json` records teacher readiness/initialization and observed controls separately.

# V0: full-match reference harness verification

Date: 4 October 2026. Scope: V0 only. No V1 or later experiment was implemented or run.

## Verdict

**V0 passed on the repeat session `20261004_055830_563928`.** The full match ended naturally with zero original errors, zero diagnostic errors, zero invalid outputs, and unchanged protected source/configuration hashes. One transient summary replacement failure was recovered and recorded as a retry. No V1 or later experiment was started.

## Passing repeat

Session: `training/live_reference_test/sessions/20261004_055830_563928/`.

Evidence: session `run_summary.json`, `bot_summary.json`, `session_metadata.json`, and the user's terminal output. The terminal printed `MATCH ENDED`; the run summary recorded `status=match_ended` and `MatchPhase.Ended`. Final score: **orange reference 8, blue human 2**. This informal human match verifies the reference/harness, not student quality or general competitive strength.

| V0 requirement | Recorded result |
|---|---|
| Correct experiment and pinned reference | `experiment=V0`; revision `fd061f457bf19175b4a9b3b3d7811a987044c64d` |
| Teacher ready / initialized / controls observed | All true; two-player orange teacher versus blue human roster |
| Natural match end | `status=match_ended`; ended phase in launcher and four ended callbacks in teacher counters |
| Source/configuration protection | `protected_sources_unchanged=true`; all 43 hashes independently rechecked after the run |
| Original errors / last error | 0 / null |
| Diagnostic errors | 0 |
| Invalid outputs | 0 |
| Launcher diagnostic read failures | 0 |
| Writer replacement retries / deferred saves | 1 / 0 |
| Cleanup/disconnect errors | None recorded |

Additional counters: 30,470 callbacks; 98 observed teacher touch timestamps; 31 flip starts and 31 sequence completions; 5,346 prediction requests, all selecting slices, with 5,327 prediction-steering callbacks. Counts establish execution and observed contacts, not successful physical execution of every flip or contact usefulness.

Countdown, kickoff, active play, goals, replays and match end were observed. Frame gaps ranged from one to five, with no duplicate frames, frame rollbacks or game-clock rollbacks. Maximum game-time gap was 0.041687 seconds. The event log did not reach its cap.

The final launcher frame was 60,808 at elapsed game time 506.733337 seconds. The periodic latest sample was frame 60,765, only 43 physics frames earlier; its score agrees with the final score. The four ended callbacks are retained in the counters even though the latest periodic sample still identifies active play. This is consistent with the sampling cadence, not evidence that the match-end summary is stale.

The single recorded writer retry demonstrates that the sharing-race recovery path was exercised during the live run without an unhandled diagnostic error. The first attempt's failure remains recorded below and in its original artifacts.

**Stop point:** report V0 passed and await explicit authorization for any later experiment. No model, BC, data collection pipeline, V1, or later validation was implemented as part of this repeat review.

## First live attempt — completed but failed the clean-diagnostics gate

Session: `training/live_reference_test/sessions/20261004_054339_069322/`.

Sources: the session's `run_summary.json`, `bot_summary.json`, and the user's pasted terminal output. The terminal output was supplied as `Pasted text.txt`; the session artifacts remain the primary saved numerical evidence. The original session files were not changed to remove the failure.

| V0 requirement | Finding | Result |
|---|---|---|
| Experiment identity | `experiment=V0` | Pass |
| Teacher initialized, ready and producing controls | All corresponding run-summary flags true; orange named teacher versus blue human | Pass |
| Natural match end | `status=match_ended`; `MatchPhase.Ended` observed | Pass |
| Protected source/configuration files unchanged | Recorded true; all 43 hashes checked again after the correction | Pass |
| No original controller errors | `original_errors=0`, `last_error=null` | Pass |
| Valid controller outputs | `invalid_outputs=0` | Pass |
| Clean diagnostics | `diagnostic_errors=1` | **Fail** |
| Launcher read/cleanup failures | `diagnostic_read_failures=0`; no cleanup/disconnect error recorded | Pass for recorded evidence |

The final score was **orange reference 4, blue human 2**. This is a completed informal human match, not a benchmark of student performance.

Other recorded counters:

- 24,435 callbacks.
- 96 observed teacher touch timestamps, which are not guaranteed exhaustive physical-contact counts.
- 24 flip starts and 24 sequence completions; counts do not prove every physical flip succeeded.
- 3,649 prediction requests, all obtaining slices; 3,637 prediction-steering callbacks.
- Countdown, kickoff, active play, goal-scored, replay and ended callbacks all recorded.
- Zero duplicate frames, frame rollbacks or game-clock rollbacks.
- Frame gaps of 1–5 and one eight-frame gap; maximum game-time gap 0.0666666 seconds. This is timing evidence, not an automatic gameplay failure.
- Event log below its cap.

The last launcher frame was 48,980, with game elapsed time 408.166656 seconds. This clock includes transitions and is not the five-minute scoring clock.

## Failure and isolated correction

At approximately game elapsed time 266.37 seconds, the terminal reported:

```text
OG diagnostics error (controls unchanged): [WinError 5] Access is denied:
bot_summary.tmp -> bot_summary.json
```

The failure occurred in `training/live_reference_test/diagnostic_bot.py:Diagnostics.save`, while replacing the periodic summary file. This is consistent with a Windows file-sharing race, including a reader temporarily holding the destination; the exact lock owner was not measured. The prior reader-side correction prevented a launcher crash but did not protect this writer operation. Gameplay continued and the match ended normally.

Only the isolated writer was changed:

1. On a replacement `PermissionError`, try up to three immediate replacements, without sleeping in a control callback.
2. If still locked, keep the previous complete JSON file and defer publication to the next periodic save. Track `summary_replace_retries` and `summary_deferred_writes` explicitly.
3. A sharing failure persisting for ten seconds remains a diagnostic error. Other serialization/write/replace errors remain errors immediately.
4. On retirement only, allow a bounded 200 ms retry window for final publication. Failure remains reported.
5. Never clear original/diagnostic error counters to manufacture a clean result.

The wrapper still returns the original controller object unchanged. Neither `python-example-original/` nor `python-example/` was edited.

## Checks after correction

Short local checks passed for:

- One failed replace followed by immediate success, with retry counters persisted.
- Repeated sharing failures deferred without any callback sleep.
- Last-good JSON preservation and subsequent fresh publication.
- Persistent sharing failure remaining a diagnostic error, retained after recovery.
- Retirement retry and successful final flush.
- All 43 protected source/configuration hashes.
- `launch.py --v0 --check`: imports, source revision, isolated 1v1 config, and actual batch-command quoting. No game was launched by these checks.

One diagnostic-error print during the local checks was deliberately induced to test persistent-failure visibility; it is not a second live failure.

## Repeat instructions issued after the first attempt — now completed

The user runs the long live command:

```powershell
& '.\python-example\venv\Scripts\python.exe' -B '.\training\live_reference_test\launch.py' --v0
```

Allow natural match completion and share the terminal ending plus `run_summary.json` and `bot_summary.json`. Handled retry counters may be nonzero, but original errors, unhandled diagnostic errors and invalid outputs must remain zero. Any persistent publication failure or stale final evidence must be investigated before accepting V0.

The repeat described above passed V0. These instructions are preserved as the first-attempt record; no additional run is required for V0, and no later experiment starts automatically.

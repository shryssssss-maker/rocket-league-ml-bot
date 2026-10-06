# V16 Stage 1 launch readiness — 2026-10-06

**Preparation passed with scope. Live collection has NOT started. Await explicit authorization.**

Source authority: installed RLBot 2.0.0b55 / rlbot-flatbuffers 0.19.0, original commit fd061f457bf19175b4a9b3b3d7811a987044c64d. All requested native descriptors exist. See source_audit.json for exact source lines, datatypes and documentation, and protocol.md for field units/reset/timing qualifications.

Native values are copied from the actual controlled player's packet before calling the unchanged teacher. The original teacher alone returns controls. Actual RLBot prediction deliveries are used unchanged; packet/prediction receipt counters and monotonic/message ordering expose reuse/coalescing. No atomic pair or engine input-application timing is assumed. last_input remains unresolved beyond its documented last-input description until live lag/transition evidence is reviewed.

## Non-live checks

- Final both-side Epic configs parsed; original teacher module imported without constructing a bot/socket.
- Session path bound inside the individual batch file, including warm-server cases; no environment-only routing.
- Unique immutable summary publication verified; earlier preflight/seal evidence preserved.
- Exact 13D names, finite native encoding, negative timer sentinel and causal array indexing checked.
- Fourteen independent exhaustive-distance comparisons (H1/2/4/8/16/32/64, baseline/native block), plus deterministic exact ties, passed using algebra-only matrices.
- All 43 protected hashes and selected V13/V14 source/artifact checksum anchors verified after preflight. No frozen dataset sample read.

## Collection roster / commands after explicit approval

| Match | Original teacher | Human | Supplied session |
|---|---|---|---|
| v16_001 | Blue | human_a (Orange) | human_a_v10_session_01 |
| v16_002 | Orange | human_b (Blue) | human_b_v10_session_01 |
| v16_003 | Blue | human_c (Orange) | human_c_v10_session_01 |
| v16_004 | Orange | human_a (Blue) | human_a_v10_session_01 |
| v16_005 | Blue | human_b (Orange) | human_b_v10_session_01 |
| v16_006 | Orange | human_c (Blue) | human_c_v10_session_01 |

These literal IDs identify three shared sessions, not six independent sessions. A leave-one-match-out diagnostic is not opponent/session-disjoint. No old V10 sample access follows from the supplied names.

Run exactly one approved match per command, from project root. No command starts the next match automatically:

```powershell
& '.\python-example\venv\Scripts\python.exe' -u -B '.\training\behavior_cloning\v16_native_maneuver_audit\collector\launch.py' --run --match-id v16_001
```

For the other assigned matches use the identical command with v16_002, v16_003, v16_004, v16_005, or v16_006. The companion JSON enumerates all six exact commands. Do not rerun an existing ID: folders/data cannot be overwritten or resumed. Failed attempts stop for review.

Expected output: assignment and human side, periodic callbacks/sequence counts, then V16 STOPPED: natural_match_ended and result path. Five minutes is game clock, with natural goals/replays/countdowns/overtime; startup can also take time. A three-minute startup or twenty-minute live supervisor timeout preserves an incomplete attempt. Ctrl+C is incomplete. No fixed flip-count claim.

Paste the final terminal output/error/queue observations after each match. Raw records live under matches/<ID>/raw/. Submission journal joins to callbacks by exact index; local send return never claims engine execution. Final teacher submission and agent closure must be confirmed. Per-match integrity seals all match artifacts.

## Separate diagnostic analysis after complete collection/review

```powershell
& '.\training\venv\Scripts\python.exe' -u -B '.\training\behavior_cloning\v16_native_maneuver_audit\analysis.py' --run
```

This hashes all six new matches before raw reads; missing, incomplete, invalid, duplicate or hash-mismatched evidence stops rather than being silently excluded. Fixed 32-neighbor leave-one-match-out histories through H64 are CPU-intensive and print row/fold/query progress. Do not launch analysis during collection. There is no model fitting or training tensor creation.

Outputs: results_v1/ diagnostic evidence/integrity, then training/reports/v16_native_maneuver_observability_20261006.md and .json with all twenty requested sections. These scientific reports do not exist during preparation; there is no V16 evidence classification or scientific gate pass yet.

## Limits and stop

SDK queue draining can coalesce received packets. Callback/frame gaps do not establish lossless tick delivery; no deliberate receive-loop pauses are added. Natural coverage may be sparse. Ground/reset and input lag behavior remain empirical; source comments alone do not establish engine producer semantics. No field is chosen for an official policy contract.

No game, student, policy training, test, DAgger, teacher/official-input change or frozen-artifact modification occurred. Stop pending explicit live authorization.

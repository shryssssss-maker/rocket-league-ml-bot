# V16 native maneuver-state observability audit

Preparation only until separate live approval. The original recovered teacher controls natural 1v1 games; no student, training, RocketSim, state setting or deliberate agent pauses.

Read `protocol.md`, `source_audit.json`, `collection_plan.json` and the launch-readiness report before collection. User-supplied opponent/session IDs are persisted exactly. Repeated IDs are shared sessions. No existing match folder can be overwritten/resumed.

From project root, non-live checks:

```powershell
& '.\python-example\venv\Scripts\python.exe' -u -B '.\training\behavior_cloning\v16_native_maneuver_audit\collector\launch.py' --check
& '.\training\venv\Scripts\python.exe' -u -B '.\training\behavior_cloning\v16_native_maneuver_audit\analysis.py' --check
```

After explicit authorization, one match per command (change only to the next approved ID after reviewing the previous outcome):

```powershell
& '.\python-example\venv\Scripts\python.exe' -u -B '.\training\behavior_cloning\v16_native_maneuver_audit\collector\launch.py' --run --match-id v16_001
```

IDs/teacher/opponent: 001 Blue/human_a; 002 Orange/human_b; 003 Blue/human_c; 004 Orange/human_a; 005 Blue/human_b; 006 Orange/human_c. Human plays the opposite side. One natural five-minute match can take longer due to startup, goals/replays and overtime. Ctrl+C is incomplete. Expected terminal progress every ten seconds, then `V16 STOPPED: natural_match_ended`. Paste final output and gameplay/queue/error observations. Do not launch a replacement for a failed attempt without review.

After all six complete matches, separately authorized diagnostic analysis:

```powershell
& '.\training\venv\Scripts\python.exe' -u -B '.\training\behavior_cloning\v16_native_maneuver_audit\analysis.py' --run
```

Analysis hashes every artifact before reading raw callback contents, validates synchronization/submissions/13D, then runs fixed 32-neighbor leave-one-match-out diagnostics. Exhaustive H64 neighbor searches are CPU-intensive and may take substantial time; periodic row/fold/query progress is printed. Do not run analysis while collection is active. Incomplete/missing/hash-mismatched data stops analysis rather than silently dropping it. No V10 samples are used.

Paths: `matches/<ID>/raw/callbacks.jsonl`, `submissions.jsonl`, `errors.jsonl`; immutable `summary_snapshots`; metadata/config/run summary/closure/integrity. Diagnostic outputs under `results_v1/`. Final reports under `training/reports/v16_native_maneuver_observability_20261006.md/.json` only after analysis. No tensors for model training are created.

Important: local send return is not engine acknowledgement. Packet/prediction deliveries are asynchronous; SDK coalesces queued packets. `last_input` remains an observed pre-current-action packet value; exact engine semantics are not assumed. All private teacher state is diagnostic metadata, excluded from probe inputs. Good observability diagnostics do not authorize any training or official observation changes.

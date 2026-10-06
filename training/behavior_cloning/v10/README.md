# V10: larger natural demonstration collection

This extends collection only. The approved V9 recorder, both bot directories, the original teacher and the 18D observation builder remain unchanged. The three pilot directories are read-only inputs. `pilot_preservation.json` fingerprints all 57 existing pilot files; every V10 preflight, collection and review verifies that snapshot plus all 43 protected source hashes.

## Before collecting

`collection_plan.json` identifies actual anonymous human opponents, which ID played the pilot, natural play-session IDs, style descriptions, and every match's split and teacher side. Do not relabel repeat sessions by one person as independent opponents. Each planned entry must contain `match_id` (`v10_001`, etc.), `split`, `teacher_side`, `opponent_id`, `play_session_id` and `play_style`. The first assigned match is train / Blue teacher / human_a. human_a played the pilot; no additional independent person is yet available. Further assignments must preserve recorded identities honestly.

User-approved provisional coverage targets:

- At least 1,000 Jump and 3,000 Front-dodge processed callbacks overall.
- At least 200 completed sequences with all four consumed phases observed; at least 100/30/30 in train/validation/test.
- Both teacher sides in every split, with at least three complete matches per side overall.
- At least three actual human opponents. Report real play-session diversity separately; no additional numeric session stopping threshold was approved.

These are engineering coverage heuristics, not validated guarantees of BC success. Count distinct sequences and per-split coverage alongside callback counts, because successive frames from one maneuver are correlated. A coverage target never authorizes training or automatic dataset freeze. Existing pilot match assignments remain exactly train/validation/test as collected.

A possible first batch is nine additional complete matches: three per available human, covering train/validation/test and both teacher sides. The real roster and assignments must be approved before this becomes a plan. Continue or stop based on observed coverage, not a fixed callback budget. Extra natural matches may be required; no resets or forced rare-action probes are permitted.

## Preflight and user-run commands

```powershell
& '.\python-example\venv\Scripts\python.exe' -B '.\training\behavior_cloning\v10\collection.py' --check
```

Preflight parses isolated Blue/Orange-teacher configs, imports the existing approved recorder and checks preservation. It does not launch Rocket League. Blue-side config parsing is not a claim of successful live blue-side operation; actual received team fields will be verified on the first natural blue match.

Once a real approved plan exists, the human runs one assigned match at a time:

```powershell
& '.\python-example\venv\Scripts\python.exe' -u -B '.\training\behavior_cloning\v10\collection.py' --match-id v10_001
```

Play as the team opposite the teacher. Observe startup/progress output and wait for natural match end. Startup is bounded at 180 seconds, and collection at 20 minutes including normal replay/countdown/overtime. Ctrl+C or timeout retains an incomplete match; it cannot count toward eligible coverage. There are no controlled state requests or receive pauses. Each match is preserved in a new `matches/<id>/` directory; an existing directory cannot be overwritten.

After a match, run the offline audit/export/review:

```powershell
& '.\python-example\venv\Scripts\python.exe' -u -B '.\training\behavior_cloning\v10\collection.py' --review
```

Review may take time as the dataset grows: it rereads actual raw callbacks, reconstructs features bitwise, validates native controls and canonical time/history, checks SHA256 values, and compares raw values to each tensor. It prints each match before auditing it. It never launches Rocket League, trains anything, changes pilot files or resamples callbacks. Use the user terminal for this potentially longer operation.

## Data and reporting

Raw JSONL and separately processed typed binary tensors retain the V9 schema: 18 float32 features; five exact native analog channels; three exact Boolean channels stored as uint8; callback/frame indices; float64 canonical times/deltas. All processed callbacks remain in original order. Invalid or incomplete matches are explicitly excluded, with evidence retained. Hidden sequence state remains diagnostic-only.

Actual source-selector results and the prospective observation probe remain separately labeled, and both consume only delivered RLBot predictions. No timestamps, slices or teacher branch behavior are altered. Teacher-side changes affect match configuration only; no observation inversion is added.

The V10 Markdown/JSON review reports include:

- Counts by split, teacher side, declared opponent, and actual declared play session.
- Native channels, rare modes, sequence starts/completions, all-phase sequence coverage and missing-ball interruptions.
- Near/far geometry, car-local ball XYZ and distance, speed bands/distributions, ball speed/spin/height and prediction horizons.
- Match phases/transitions, canonical callback-dt distributions, receipt reuse and record-path maxima.
- Invalid/incomplete match evidence, raw/tensor round-trip checks, manifest hashes and preservation results.

Output paths are `training/reports/v10_collection_review_20261005.md` and `.json`. Interim status stays collection-in-progress; meeting the approved targets means a candidate for user review, not an automatically frozen dataset. Match-level splitting is preserved, but it does not establish opponent/session-disjoint evaluation. Identities/styles are user declarations, not independently detected. No lossless-delivery or undelivered-physics-tick claim is made.

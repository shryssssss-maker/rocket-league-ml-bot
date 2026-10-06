# V15 investigation runner

Source/contract audit and preregistered diagnostic plan: `protocol.md`. Installed SDK descriptor evidence: `native_sdk_audit.json`. No official observations or model artifacts change.

The investigation uses six authorized frozen train/validation matches and the previously recorded V13 full-match anchors as quarantined queries. Native maneuver flags are absent from V10, so Group C cannot be probed statistically without new evidence; no flag is inferred. Group D means prior submitted controls, not packet last_input or physical execution.

Numerical/source check (no dataset samples):

```powershell
& '.\training\venv\Scripts\python.exe' -u -B '.\training\behavior_cloning\v15_representation_investigation\analysis.py' --check
```

User-run investigation:

```powershell
& '.\training\venv\Scripts\python.exe' -u -B '.\training\behavior_cloning\v15_representation_investigation\analysis.py' --run
```

Expect authorized tensor/raw hash checks, per-match parsing progress, fixed history searches and individual feature-group searches. Exhaustive CPU neighbor searches across >100,000 training endpoints can take several minutes or longer; CPU load is expected. Memory is bounded by match arrays plus block-distance searches; no all-query/all-training distance matrix is allocated. There is no training epoch or optimizer. The fixed 32-neighbor probe is statistical only, with k=1/8/32/128 ambiguity diagnostics and no tuning/sweeps.

Completion produces `training/reports/v15_representation_investigation_20261006.md/json`, plus full neighbor/query evidence under this directory's exclusively created `results_v1/`. It reports A/B/C/D classification and one recommendation, then stops. Classification has explicit finite-context/observational limitations; it is not an authorization to fit or change inputs.

Paste the final output after completion. If it errors, retain `failure.json` and paste the traceback; do not delete outputs/retry or alter the fixed protocol silently. No test/live/DAgger/full-policy fitting occurs. Frozen V10/V13/V14 inputs and all 43 protected files are hash-checked before and after.

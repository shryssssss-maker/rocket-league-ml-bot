# V14 v2 offline implementation

One approved boundary-aware objective, implemented only under this directory. `proposal_v2.md/json` are the specification. `proposal_v1.md/json` and its report snapshots are inactive alternatives; no CLI executes v1.

## Commands

From the project root, numerical preflight (no dataset sample reads or fitting):

```powershell
& '.\training\venv\Scripts\python.exe' -u -B '.\training\behavior_cloning\v14_temporal_objective\experiment.py' --check
```

Then **user-run** validation baseline evaluation, not fitting:

```powershell
& '.\training\venv\Scripts\python.exe' -u -B '.\training\behavior_cloning\v14_temporal_objective\experiment.py' --baseline
```

This hashes authorized validation tensors/raw diagnostics before reading them, evaluates the pinned V13 checkpoint, reproduces its saved aggregate metrics, and freezes the new timed-core baseline under `baseline_v2/`. Expect `V14 AUTHORIZED HASH`, `V14 VALIDATION`, then `V14 BASELINE: baseline_ready_before_fitting` or `baseline_requires_review_no_fitting`. It may take tens of seconds or a few minutes for hashing/raw parsing/inference, depending on disk/runtime. No test artifacts are opened. If it fails or requires review, **stop**; do not train, delete the partial baseline, or change gates.

Only after a ready baseline, the separate user-run single V14 v2 experiment:

```powershell
& '.\training\venv\Scripts\python.exe' -u -B '.\training\behavior_cloning\v14_temporal_objective\experiment.py' --train
```

This verifies authorized train/validation artifacts, initializes the exact fresh V13 GRU at seed 42, fits up to 20 epochs with the approved objective, chooses the minimum original unweighted validation loss, evaluates once selected, and reports the preregistered verdict. Expect bounded hash/progress/epoch output and `V14 v2 COMPLETE` followed by a report path. GPU fitting should be relatively short; file verification and validation can dominate. Paste the baseline's final status before fitting if review is needed, and the complete final training output afterward. No live/test/DAgger/PPO follows.

## Files and safeguards

- `boundary.py`: exact lambda, transition formulas, reduction, eligibility and timed-core metric.
- `experiment.py`: restricted artifact reader, pinned model/decoder/metric extraction, baseline gate, one-run fitting and validation/reporting.
- `preflight.py`: independent scalar loss oracle, numerical gradient/mask/chunk/temporal checks. These are numerical fixtures, never training examples.
- `implementation_seal_v2.json`: SHA256s of implementation/specification; checked at entry and completion. Old V13 checkpoint/config/source and all 43 protected hashes are also checked. No old launcher or broad V11 dataset verifier is imported.
- `preflight_v2.json`: numerical check evidence; no dataset evaluation.

Baseline and run directories are created exclusively and cannot be overwritten/resumed. Errors preserve partial evidence and require review. Raw data/tensors/teacher/private labels are never rewritten. Private validation sequence labels are used solely for unchanged retrospective metric subsets, never inputs or the boundary loss. Model hidden state and detached prior-probability cache reset at match boundaries only. Missing/gap pairs are excluded solely from the added term; callbacks remain in the base loss.

Successful final evidence is under `run_v2/report.md/json`, with predictions, per-chunk support/clamp diagnostics, source/runtime pins and output integrity hashes. The explicitly requested `training/reports/v14_temporal_objective_20261005.md/json` receive the final result only after successful post-run integrity checks; prior design reports are retained inside the new run. This publication is a report update, not model deployment.

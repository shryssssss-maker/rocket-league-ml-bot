# V14 v2 implementation readiness

Status: implementation ready; numerical preflight passed.

Exact approved lambda=0.1, boundary formulas/reductions/eligibility, original architecture/seed/optimizer/schedule/selection and regression guards are preserved. No v1 execution. Eight numerical checks passed, including an independent scalar oracle, gradients, absent-group reductions, detached cross-chunk pairing, masks/gaps, exact native decoder, causal model continuity and timed-core failures.

All 43 protected hashes and pinned V13 source/checkpoint/config hashes match. Manifest metadata binds 48 authorized train/validation tensor paths. No dataset samples, validation baseline, fitting, test, live, or DAgger executed.

The user-run baseline command verifies actual validation artifact hashes and establishes/freeze-checks the timed-core baseline before fitting. A ready baseline is mandatory; errors/partial runs are preserved and do not authorize a rerun or changed gates. The separate --train command in README.md is the single fresh V14 v2 experiment. No further experiment or deployment follows it.

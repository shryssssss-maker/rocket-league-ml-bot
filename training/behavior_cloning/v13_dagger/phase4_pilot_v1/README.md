# Phase 4 preparation — specification review

The user approved exactly one **90-second natural 1v1** run: **Blue human versus Orange 13D student**. Original teacher is shadow-only. No state-setting, intentional receive pauses, replay manipulation or additional runs until review. Retain native replay/countdown behavior.

This run configuration overrides the earlier pending preference in the [full specification](launch_readiness_specification_20261005.md); the [approval record](run_configuration_approval_20261005.json) preserves that correction explicitly.

**No live launcher has been created or run. The full protocol and non-live preflight still require review before launch.**

Review these proposed details:

- Preserve the accepted 13D model, input semantics, native action decoder, original selector and independent original shadow memory. Only student controls reach RLBot.
- Log actual received physics, predictions, callback timing, student submissions and shadow expert actions/sequence state. Keep the records quarantined as diagnostic evidence, without training tensors, aggregation or fitting.
- Measure state occupancy against frozen **training data only**: primarily Orange-teacher matches `pilot_01` and `v10_006`, secondarily all training matches. Hash required artifacts before reading them. No validation/test sample reads or evaluations.
- Report phase/mask-conditioned feature distributions and reference-bound excursions, plus actual shadow-sequence continuation when student actions omit required preceding jumps. These are descriptive observations; one short match cannot establish causal distribution shift, expert-label usefulness or DAgger benefit.
- Stop at 90 seconds after the first verified student submission, natural match end, user stop, worker/contract failure or server disconnect. No teacher fallback, invented reset or synthetic action/state repair. Preserve existing two-second inference timeout and native action-hold accounting.
- Require verified final student submission, clean shutdown, unchanged protected artifacts and a report explicitly listing missing coverage and callback/queue/timing limitations.

After specification approval, prepare the isolated runner/reference-statistics checks and run only non-live preflight. Present the resulting readiness evidence before live launch. No Phase 4 execution or training follows automatically from this preparation.

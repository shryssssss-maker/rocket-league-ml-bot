# Model Wars: policy audit and proposed V2 bootstrap

Date: 2026-10-01. Status: diagnostic complete; V2 design only, not implemented or trained.

## Recommendation

For a deliberately weak but functional organizer starter, weak teacher -> behavior cloning (BC) -> measured PPO fine-tuning is the better next hypothesis to test. It is not yet a demonstrated improvement. First validate the teacher in closed-loop simulation, then test a small BC student with the existing architecture/action table. Do not spend another large PPO budget before that evidence exists.

Keep the 1M checkpoint as a frozen behavioral benchmark. Keep the historical 65k reference and every previous report. Neither is an accepted organizer baseline.

## 1. Current root-cause assessment

The evidence establishes narrow learned contact, not ordinary match competence. Controlled contacts improved substantially; grounded contact remained uncommon; early jumping dominated; kickoff-v2 produced no learner touches. Initial ground propulsion exists, so “cannot drive at all” is the wrong diagnosis. The missing demonstrated skills are sustained directional control, approach across varied states, and reliable grounded contact.

The live behavior is consistent with that failure rather than inherently contradictory to the offline results. Offline fixtures start near a stationary ball with a limited heading distribution and an idle opponent. A full match visits very different orientations, positions, speeds, recoveries and opponent interactions. Bringing the live ball nearby did not recreate all of those fixture variables. Distance alone cannot explain the failure.

The newest continuous live session, spanning JSONL segments 006 and 007, contains 14,646 policy decisions, including 14,370 active-play decisions. Action 51 occurred 8,317 times; action 12 occurred 4,358 times. Together they account for 86.54% of decisions. The session records no policy_error events. This supports an actual policy control pattern, not an apparent pattern caused solely by kickoff countdown or a missing inference loop. Absence of logged errors is not proof of complete adapter equivalence.

Physical disruption changing the action regime is compatible with state-conditioned behavior. It does not establish that ball geometry is controlling steering appropriately. A fatal feature-order/action-table bug remains unconfirmed; internal timer/grounded approximations and simulator/live dynamics remain unresolved contributors.

Confidence: high that the current policy is brittle and insufficient for the starter; moderate that narrow distribution/objective explain much of it; unproven that changing observations or BC alone will solve it. There is no evidence yet that a larger network is necessary.

## 2. Controlled ball-sensitivity result

Executed `training/policy_ball_sensitivity.py`, using the installed training DefaultObs and existing checkpoint loader. No simulator was instantiated or advanced; no optimizer, download or training run occurred. CPU inference used eval/no_grad.

Report: `training/logs/guide_early_1m_v1/policy_sensitivity/ball_sensitivity_001.json`.

Checkpoint SHA-256 before and after:

`f1a6b40950f79045a7a13d50c6a38935834e83dd4ad5c9333120338bed0d7281`

There are 126 interventions: seven fixed state templates, each with six directions at horizontal distances 250, 1,000 and 3,500 uu. Four templates use synthetic GameStates passed through the exact training builder: blue goal-facing stationary, mirrored orange, blue yaw zero, and blue moving forward at 500 uu/s. Three retain recorded live vectors from active-play states producing actions 51, 12 and 41. In every sweep, only ball-position slots 0–2 change; slots 3–91 are asserted identical. Reports include all 92 values, 90 logits, 90 probabilities and all eight controller fields.

Actions below are ordered: ahead, ahead-left, ahead-right, left, right, behind.

| Fixed template | Close | Medium | Far |
|---|---|---|---|
| Blue, grounded, goal-facing | 41,41,41,41,41,41 | 41,41,41,41,41,41 | 51,51,41,51,41,41 |
| Orange mirrored equivalent | 41,41,41,41,41,41 | 41,41,41,41,41,41 | 51,51,41,51,41,41 |
| Blue, grounded, yaw zero | 41,41,41,41,41,41 | 41,41,41,41,41,41 | 41,41,41,41,41,41 |
| Blue, grounded, forward speed 500 | 41,41,41,41,41,41 | 41,41,41,41,41,41 | 51,51,41,41,41,41 |
| Recorded grounded action-51 state | 41,41,41,41,41,41 | 41,41,41,41,41,41 | 51,41,15,41,15,41 |
| Recorded airborne action-41 state | 41,41,41,41,41,41 | 41,41,41,41,41,41 | 41,41,41,41,41,41 |
| Recorded grounded action-12 state | 12,12,12,12,12,12 | 12,12,12,12,12,12 | 12,12,12,12,51,12 |

No tested intervention selects positive steer. Action 41 means throttle +1, steer 0, pitch -1, yaw 0, **roll +1**, jump 1, boost 1, handbrake 1. Thus “41” is not plain forward driving. Actions 12 and 51 both steer -1; 51 also boosts and has roll +1.

Probabilities DO change. In the stationary blue template, mean total-variation distance between paired left/right distributions is about 0.244, with maximum 0.582. For a far ball directly right, action 41 has probability approximately 0.825; for a far ball ahead, action 51 is the argmax at approximately 0.320. In the recorded action-12 state, mean paired variation is only approximately 0.055 and expected steer stays approximately -0.85 to -0.93.

Conclusion: the model is not literally insensitive to ball position. It often fails to translate changing ball position into appropriate directional actions, and has strong state-dependent jump/left-steer preferences. Sampling is not an automatic cure: a distribution can remain directionally biased even when its argmax is unchanged.

Limitations: these are fixed-state interventions, not rollouts; there are only three selected live exemplars, not a population sample. Some far live-position interventions can lie outside field bounds and are mathematical sensitivity probes, not valid match fixtures. Synthetic opponent/ball timer states are controlled, not representative of every live context. This experiment does not establish causality for all circling or prove the representation is unusable.

### Diagnostic-label correction

The previous live diagnostic derived driver-right by negating `PhysicsObject.right`. That was incorrect. The installed API exposes right as rotation-matrix column 1; Rocket League's coordinate conventions require care. The new test uses that convention. Earlier derived right values and controlled left/right labels should not be used unchanged for directional conclusions. This label error did NOT enter DefaultObs or the action/controller conversion. No existing live file was changed in this audit. See [RLBot coordinate documentation](https://wiki.rlbot.org/v5/botmaking/useful-game-values/).

## 3. DefaultObs limitations and exact feature coverage

The authoritative installed builder is `training/venv/Lib/site-packages/rlgym/rocket_league/obs_builders/default_obs.py`; the project factory/config/preprocessing are in `training/observations/__init__.py`. Preprocessing converts to float32, checks shape/finite values, and does not add normalization layers.

All 92 slots are covered here. Vector triples are x,y,z, respectively.

| Indices | Existing DefaultObs | Candidate relative semantics |
|---|---|---|
| 0–2 | Team-world ball position /2300 | Car-local ball displacement /2300 |
| 3–5 | Team-world ball velocity /2300 | Car-local ball-minus-car velocity /2300 |
| 6–8 | Team-world ball angular velocity /pi | Car-local ball angular velocity /pi |
| 9–42 | 34 team-ordered pad cooldowns /10 | Unchanged |
| 43 | holding jump | Unchanged |
| 44 | handbrake | Unchanged |
| 45 | has jumped | Unchanged |
| 46 | is jumping | Unchanged |
| 47 | has flipped | Unchanged |
| 48 | is flipping | Unchanged |
| 49 | has double-jumped | Unchanged |
| 50 | can flip | Unchanged |
| 51 | air time since jump, raw | Unchanged |
| 52–54 | Own team-world position /2300 | Unchanged |
| 55–57 | Own team-world forward vector | Unchanged |
| 58–60 | Own team-world up vector | Unchanged |
| 61–63 | Own team-world velocity /2300 | Unchanged |
| 64–66 | Own team-world angular velocity /pi | Unchanged |
| 67 | Own boost /100 | Unchanged |
| 68 | Own demolition timer, raw | Unchanged |
| 69 | Own grounded flag | Unchanged |
| 70 | Own boosting flag | Unchanged |
| 71 | Own supersonic flag | Unchanged |
| 72–74 | Opponent team-world position /2300 | Car-local displacement /2300 |
| 75–77 | Opponent team-world forward | Relative orientation, subject to test below |
| 78–80 | Opponent team-world up | Relative orientation, subject to test below |
| 81–83 | Opponent team-world velocity /2300 | Car-local opponent-minus-car velocity /2300 |
| 84–86 | Opponent team-world angular velocity /pi | Car-local angular velocity /pi |
| 87 | Opponent boost /100 | Unchanged |
| 88 | Opponent demolition timer, raw | Unchanged |
| 89 | Opponent grounded flag | Unchanged |
| 90 | Opponent boosting flag | Unchanged |
| 91 | Opponent supersonic flag | Unchanged |

DefaultObs is not missing the information needed to locate the ball. It requires the network to learn subtraction and orientation-dependent projection. The current small feed-forward actor has no explicit relative displacement/closing velocity or history; stateful simulator fields do not substitute for a complete dynamics history. These are plausible sample-efficiency limitations, not evidence of impossibility.

## 4. Is RelativeDefaultObs worth testing?

Yes, as a separate BC comparison, not a replacement for this checkpoint. The inspected upstream builder keeps the own-car block absolute/team-oriented and makes ball/other-car physics relative. Its default padding is three cars per team (172 values); matching our 1v1 requires explicit padding configuration to produce 92 values. Preserve our coefficients. Orange pad ordering and own-car inversion remain; relative geometry uses world car physics. Source: [RelativeDefaultObs](https://github.com/RLGym/rlgym-tools/blob/main/rlgym_tools/rocket_league/obs_builders/relative_default_obs.py).

The position/linear-velocity transformation explicitly subtracts the origin and projects onto its basis. Angular velocity is projected without subtracting own angular velocity. That is not identical to the time derivative of rotating-frame displacement. Source: [relative physics implementation](https://github.com/RLGym/rlgym-tools/blob/main/rlgym_tools/rocket_league/math/relative.py).

Two source-review caveats need resolving before use. The inspected main-branch builder calls `relative_physics` directly despite storing a selectable relative function. More importantly, its orientation expression is `target_rotation @ origin_rotation.T`; installed PhysicsObject exposes basis vectors as columns. A numeric noncommuting pitch/yaw/roll test against direct basis projection `origin_rotation.T @ target_rotation` differs by max absolute 0.2335. This is an arithmetic/convention discrepancy to investigate with the exact pinned candidate, not an executed package-wide validation or a reason to silently patch dependencies. Ball displacement/velocity do not use that orientation expression.

Relative input should simplify reach/steering supervision, but own position/orientation still encode field/goal context; it does not add prediction or eliminate live timer approximations. Shared builder, raw-state conversion, mirrored-team tests and tilted-car tests are mandatory before deployment. Never load the current 1M actor with relative input just because both have 92 values. Contracts must identify feature semantics, not only dimensions.

## 5. Why teacher + BC + PPO is justified

Our problem is obtaining a useful behavioral prior quickly, not proving that enough random-initialized PPO could never work. A teacher gives dense, direct supervision of turning, propulsion and strike setup without waiting for sparse contact discovery. We can control skill weakness and collect exact action labels using our simulator/table.

Rocket League replay-pretraining demonstrates imitation followed by RL as a bootstrap route, and describes both compounding errors and direct bot-demonstration labels. Its reported success does not prove this proposed simple teacher will work. We do not need replay downloads or inverse dynamics for simulator teacher labels. See [replay-pretraining](https://github.com/Rolv-Arild/replay-pretraining).

Main risks: a bad teacher teaches bad behavior; BC can look accurate offline yet fail after visiting unfamiliar states; teacher logic can depend on hidden memory the student cannot observe; class imbalance can hide constant-action collapse. Validate each rather than assuming BC is easier by definition.

## 6. Deliberately weak teacher design

Use the existing v2 behind-ball opponent as a prototype to inspect, not automatically as a trusted teacher and not by modifying the benchmark. Give the teacher a separate version identifier.

For grounded low-ball situations, let b be ball horizontal position and G the opponent goal center. Compute d = normalized(G-b), then a setup target q = b - s*d. The provisional standoff s is a tunable teacher parameter, not a promotion threshold. First drive to the behind-ball setup region; once suitably aligned, approach through the ball along d. Use forward/right projections to calculate target angle, discretize steering, choose neutral/forward/reverse throttle to handle large angles and excessive speed, and boost only when reasonably aligned with room to approach. Quantize to an existing 90-action row and apply it for the same eight ticks.

Keep target selection mostly stateless and derived from observable geometry/velocity. If hysteresis or remembered phases become necessary, establish whether identical observations receive inconsistent labels; do not silently give the teacher inaccessible history. Handle target overlap/zero distances safely. A small constant-velocity moving-ball lead can be a later isolated extension; no complex predictor.

The initial teacher demonstrates ground control rather than advanced jumps. This is a data distribution choice, not a permanent student action mask. All 90 student outputs remain available. Validate recovery from simple bad headings and modest perturbations; do not add aerials, flicks, wall strategy or elaborate defense. Simple defensive examples clear away from own goal, without claiming competent defense.

Teacher acceptance comes before data generation: does it actually reach, make grounded contacts and send the ball in intended directions over randomized closed-loop fixtures? Does it avoid circling even when the ball is close? If not, fix the teaching task/controller before cloning it.

## 7. Demonstration generation

Collect observation immediately BEFORE the teacher action, label the exact applied lookup index, then advance eight ticks. Store episode/state-family IDs, seeds, teacher version, action-table and observation-contract hashes, package versions and termination metadata.

Use legal randomized state families: close/medium/far distances; both lateral signs; wide car headings; varied initial speed/boost; stationary and moving low balls; multiple field regions; behind-ball and wrong-side setups; offensive and simple clearing states; standard kickoffs; idle then weak opponents; both teams. Include failure/recovery states rather than retaining only successful contact clips. Make the mixture explicit and balance families/actions where practical.

Generate raw-state records or a reproducible common-state stream so both observation arms receive identical states and teacher labels. Split by episode/fixture family, never adjacent frames randomly. Hold out heading/distance/velocity combinations as well as seeds. Keep meaningful standard-kickoff tests separate from the training mixture.

Proposed first budget after approval: 100k–250k labels, not an assumed sufficient amount. At 92 float32 values plus one uint8 action, compact arrays take 36.9–92.25 MB before metadata. Two observation arms double those arrays; raw-state retention adds more. Profile generation first; no reliable elapsed-time estimate exists yet. Small numbered shards avoid huge JSON datasets. No public gameplay dataset is required.

## 8. BC design

Keep actor 92 -> 128 -> 128 -> 90 with existing activations. Start a fresh actor for the primary test rather than importing the 1M jump/circling prior; retain 1M as benchmark. Train categorical cross-entropy against teacher indices. No critic target is required for BC; prepare a fresh critic when PPO begins. No architecture enlargement, action mask or live heuristic.

Use the same labels, split, model initialization seeds, optimizer budget and evaluation fixtures for two small arms: existing DefaultObs versus validated 92D relative input. This isolates representation from teaching approach. First implement/validate teacher + DefaultObs BC; add the relative arm only after its convention tests pass. Do not combine reward changes or worker changes with this comparison.

Measure held-out cross-entropy, action accuracy against a majority-action baseline, steering confusion by target-angle bin, and propulsion/boost agreement. Some lookup entries are behaviorally equivalent in grounded states; report control agreement too. Above all, roll out the student independently. High imitation accuracy is not a pass if it circles or never reaches.

If the student fails on its own visited states, query the teacher on those states and test a small dataset-aggregation round. This addresses sequential distribution shift rather than adding arbitrary model size. It is a later option, not a requirement for the first experiment. See the original [DAgger paper](https://arxiv.org/abs/1011.0686).

The verified GPU has 6 GiB VRAM. The current model is small; modest minibatches (initially e.g. 512, measured rather than assumed optimal) are realistic. No optimization infrastructure or new large dependency stack is justified yet.

## 9. PPO fine-tuning design

Only after a BC policy shows useful independent behavior: initialize the actor from BC, use a new critic/optimizer, preserve the selected observation contract/table/cadence, and freeze the BC checkpoint as a regression reference. Do not resume an unrelated 1M optimizer/critic as if this were the same experiment.

Initial fine-tuning should test retention/improvement of approach and grounded contact. Introduce directional touch, attack, scoring and simple defense only when corresponding evaluation supports it. Existing guide rewards can be a control, not an automatic final objective. Positive air reward may conflict with desired early grounded behavior, but change it only in a separate measured ablation. Do not prescribe new weights now.

If PPO erases BC behavior, investigate update size/critic initialization and then a separately tested temporary BC or reference-policy regularizer. Do not silently bolt a new loss onto the current PPO implementation. Each change must have a named experiment.

Important local-code finding: this PPO standardizes advantages but does NOT standardize returns/rewards. Its value loss operates on raw return scale. Therefore “RLGym normalizes rewards so copied guide weights are equivalent” is not true for this implementation. Large finite value loss alone is neither proof of failure nor proof of good value fitting; future runs should also report explained variance/return scale.

## 10. Measurable curriculum

Start with a mixed ground-control distribution, not a single nearby aligned trajectory. Maintain earlier-task regression fixtures while broadening distance, heading and ball velocity. Add strike setup, useful contact, kickoff, weak-opponent play and short matches as measurable task families, not automatically after N steps.

Each stage requires repeated held-out improvement in relevant physical behavior. Moving-ball interception follows competent stationary approach; opponent pressure follows competent unopposed contact. Include controlled grounded recovery states before full chaotic randomization. Keep some old-state replay in later data/training mixtures to test forgetting.

Do not advance on reward/entropy alone. Choose numeric promotion thresholds after teacher and initial BC curves establish attainable behavior; report uncertainty and tradeoffs rather than inventing success cutoffs now.

## 11. Keeping the final bot genuinely ML-driven

Teacher code is offline data collection/evaluation only. Live runtime is packet -> shared observation builder -> trained actor -> unchanged action table -> controller, with argmax initially. There is no teacher call, PID, ball-chaser fallback, hardcoded kickoff or steering correction in the live gameplay path. Safety-neutral handling for invalid packets is distinct from gameplay assistance and must be logged.

Students/teams receive an actual learned checkpoint and trainable code. BC provides the behavioral prior; PPO can improve beyond teacher habits. Deliberate weakness comes from limited demonstrated skills and verified behavior, not inserting artificial live failures.

## 12. Reuse

Reuse simulator/GPU setup, exact table/cadence, small actor, checkpoint loading and contract checks, reproducible fixtures/seeds, contact callbacks, diagnostic logging, live deployment shell, and existing reports/checkpoints as benchmarks. Reuse learned lessons: no-touch wins are not competence; grounded contact must be classified at contact; full matches need separate evaluation.

Keep v2 benchmark opponent frozen. A teacher may share reviewed ideas, but independently version its implementation so opponent comparisons stay meaningful.

## 13. Retire assumptions, not files

Do not delete the 98k branches or 1M/65k artifacts. Retire them as automatic baseline candidates or default next resume points. Retire training reward/raw win rate as promotion criteria; retire the assumption that a nearby ball means fixture-equivalent state; retire uncorrected derived left/right labels. Do not discard DefaultObs, PPO or the 90-action table as fundamentally broken on current evidence.

## 14. Minimum viable V2 experiment, after review

1. Unit-test/version the weak ground teacher and benchmark it on mirrored directional, randomized approach, ground-contact and kickoff fixtures. No BC if teacher fails.
2. Freeze one diverse 100k-label dataset/split and its raw-state reconstruction contract. Verify label cadence and held-out boundaries.
3. Train one small DefaultObs BC actor, evaluate closed-loop against the teacher and frozen 1M policy. This is the first test of the bootstrap strategy.
4. Separately validate and test the relative-input BC arm with matched data/budget. Do not adapt the current checkpoint's input semantics.
5. Recheck simulator/live observation parity and watch the selected pure ML BC actor before adding PPO.
6. Only then authorize one controlled PPO fine-tune, with regression evaluations before increasing scale.

This sequence can be stopped early if teacher quality, student control or live parity fails. None of these training/data-generation stages has been started.

## 15. Exact decision metrics

Use identical fixture/opponent versions and episode limits for teacher, fresh policy, frozen 1M and BC candidates; evaluate argmax and seeded sampling separately. Start with the existing 40 episodes per fixture/mode and multiple held-out seed sets; aggregate only identical definitions. Report paired differences and uncertainty, not model rankings from a handful of goals.

| Question | Measurement |
|---|---|
| Correct conditional control? | Action/probability/expected-steer sweeps with only ball/target angle varied; signed heading error and steering agreement with teacher target |
| Sustained ground approach? | Grounded duration before jump/contact, horizontal speed/closing velocity, actual distance reduction, heading error over time; not throttle fraction alone |
| Reliable reaching? | Touch-episode rate, first-touch rate, total contacts, time-to-first-touch conditional on contact AND timeout/miss rate |
| Grounded rather than aerial shortcut? | Ground/air/unknown classification at physics contact, first-contact class, ground-contact episode rate and contact fractions |
| Circling? | Accumulated yaw/turns plus net approach progress and contact count over the same time window; log as a behavior diagnostic, not a new reward |
| Useful contact? | Controlled learner-contact ball-velocity change projected toward intended goal/clear direction, with interfering opponent contacts excluded or flagged |
| Transfer? | Held-out distance/heading/velocity/field families, both teams, and kickoff-v2 learner touches/first touches/episode duration |
| Match behavior? | Goals for/against, own-goal/attribution caveats, wins/draws/losses, interaction-free wins flagged; no raw win-only conclusion |
| BC learned rather than majority label? | Held-out cross-entropy, balanced steering/action/control agreement versus majority baseline, then independent closed-loop behavior |
| PPO preserved bootstrap? | All behavioral metrics against frozen BC; KL/entropy/value explained variance as optimization diagnostics only |
| Deployable ML? | Synthetic and recorded-state parity, table/controller parity, input/state approximation audit, live inference timing and actual contact/control behavior |

Goalward movement over a whole episode is not proof that a learner touch was useful. Touch timing/physics provenance and controlled post-contact measurements are required. Likewise pre/post-step ground flags can miss ground contact followed by immediate bounce; use the existing physics callback definition for contact comparisons. Earlier evaluator ground rates and callback ground rates are not interchangeable.

V2 is better only if it reproducibly improves directional ground approach/contact across held-out families without sacrificing deployment correctness, and then shows basic kickoff/short-match interaction. A functional weak starter need not win reliably, but the current zero-touch kickoff behavior is not acceptable. No candidate is promoted to baseline.pt in this review.

## Files and commands

Created only:

- `training/policy_ball_sensitivity.py`
- `training/logs/guide_early_1m_v1/policy_sensitivity/ball_sensitivity_001.json`
- `training/reports/v2_bootstrap_review_001.md`

Existing training/deployment source, checkpoints, rewards, environment and experiment reports were not modified. The short inference diagnostic was executed; no training or live match was launched.

Optional repeat from the project root (creates the next numbered JSON report; scans existing live logs):

```powershell
& '.\training\venv\Scripts\python.exe' '.\training\policy_ball_sensitivity.py'
```

There is intentionally no training command yet. Review the teacher specification and V2 experimental sequence first.

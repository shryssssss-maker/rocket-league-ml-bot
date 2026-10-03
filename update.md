# MODEL WARS — complete project history and current status

Last updated: **4 October 2026**.

This document records the project from the working RLBot example through simulator setup, every saved training experiment, checkpoint evaluations, reward/contact diagnostics, live deployment, historical teacher/BC proposals, and the CanoPy feasibility investigation. It is a historical record, not an instruction to launch another training run. [MODEL_WARS_DECISION_LOG.md](MODEL_WARS_DECISION_LOG.md) records current decisions; older proposals below do not override it. The latest completed investigation and its blocker are recorded in Section 28.

## How to read this record

Saved configuration, CSV and JSON files are the primary evidence. Earlier terminal output and the user's observations supply additional history where a standalone report was not saved. Proposed work is explicitly distinguished from completed work. Intermediate checkpoints are listed even when no independent behavioral evaluation exists. Duplicate numbered evaluations are retained but are not additional training experiments.

The appendices inventory **11 saved training runs, 58 checkpoint files, 82 ordinary evaluation reports and 16 detailed contact reports** present at the **2 October 2026 inventory snapshot**. The CanoPy downloads and investigation artifacts added on 4 October are recorded separately in Section 28; they are not new Model Wars training runs or additions to that historical checkpoint count. Other reward, profiling, live and sensitivity diagnostics are described separately. These are artifact counts, not counts of independently successful models.

Older README/CURRICULUM paragraphs sometimes still say a now-completed experiment is pending. Interpret them as historical snapshots; use the saved results and the latest-status section below.

## 1. Project purpose

MODEL WARS is an overnight Rocket League AI competition. Four-person teams receive exactly the same deliberately weak pretrained ML bot, improve their own copy, submit it, and eventually compete on an organizer tournament machine.

The starter must be genuinely ML-driven and functional enough to drive, orient, approach and make basic useful ground contacts, with obvious room for improvement. We are not trying to reproduce Nexto or build a strong competitive bot. Advanced aerials, flicks, sophisticated defense and positioning are outside the initial target.

Intended pipeline:

```text
RocketSim + RLGym -> learning -> PyTorch checkpoint
                                      |
Rocket League + RLBot -> observation -> actor -> lookup action -> controller
```

Future validation, participant packaging and tournament automation have not been built. No checkpoint has passed the organizer-baseline promotion gate.

## 2. Workspace and hardware

Project root: `C:\Users\shreyas\Desktop\model wars`.

```text
model wars/
  python-example/      existing RLBot v5 deployment shell and its venv
  training/            simulator, PPO, checkpoints, evaluations and its own venv
  venv/                existing root environment; not our designated training venv
  TEACHER_TASK.md       requested scope/specification for the next teacher phase
  update.md            this consolidated history
```

Actual development laptop: Lenovo LOQ, Intel i7 HX, NVIDIA RTX 4050 Laptop GPU, 24 GB RAM, 512 GB SSD, Windows 11. . GPU diagnostics measured approximately **6 GiB VRAM**, not the initially assumed 8 GiB. Planning must use the measured capacity.

Simulation is CPU-side; PPO tensor operations use CUDA. Four environments in the experiments are independent simulator instances in one process, not four multiprocessing workers.

## 3. First milestone: working live RLBot shell

Before the ML pipeline was built, the official RLBot Python example already launched Rocket League successfully through Epic.

Important setup/debugging history:

- Rocket League is installed through **Epic Games**, not Steam.
- RLBot v5 was installed through the official launcher.
- Server executable: `C:\Users\shreyas\AppData\Local\RLBot5\bin\RLBotServer.exe`.
- Server address: `127.0.0.1:23234`.
- Official Python example repository: `https://github.com/RLBot/python-example`.
- Its original Python 3.11 environment failed because `typing.override` was unavailable.
- Python 3.12.10 was installed and the example's venv was recreated with Python 3.12.
- Verified live packages included `rlbot==2.0.0b55`, `rlbot_flatbuffers==0.19.0`, `psutil==7.2.2`.
- `rlbot.toml` was corrected from `launcher = "Steam"` to `launcher = "Epic"`.
- RLBot detected `C:\Program Files\Epic Games\rocketleague\Binaries\Win64\RocketLeague.exe`.
- Manual server startup helped isolate an early launcher/server-lifetime issue.
- Running `python run.py` inside `python-example` successfully started the server, Rocket League, a match and the example bot.

Achievement: a known-good live deployment shell. Lesson: preserve this launcher and environment; do not confuse training problems with already-solved Epic/RLBot integration.

Initially we kept example gameplay untouched. Later the user explicitly authorized replacing its gameplay decisions with the trained ML actor while retaining the same project/launcher. The current example is therefore ML-controlled, not the untouched original scripted example.

## 4. Isolated headless training environment

Created `training/` with its own Python 3.12.10 venv, separate from `python-example/venv`.

Verified stack:

| Component | Version |
|---|---|
| Python | 3.12.10 |
| rlgym | 2.0.1 |
| rlgym-api | 2.0.0 |
| rlgym-rocket-league | 2.0.1 |
| rocketsim | 2.2.1 |
| numpy | 1.26.4 |
| cmeel | 0.61.0 |
| torch, after GPU validation | 2.11.0+cu128 |

Runtime versions and dependencies are recorded in [training/requirements.txt](training/requirements.txt).

[smoke_test.py](training/smoke_test.py) created a headless 1v1 RLGym/RocketSim environment with DefaultObs, 90 lookup actions, eight ticks per action, `10 * GoalReward + 0.1 * TouchReward`, and a two-second timeout. Verified imports, creation, reset, finite observations, valid actions, advancing physics, car movement, reward dictionaries, timeout truncation, repeated reset and clean shutdown. Raw observations were `(92,)`, float64. The learning pipeline subsequently casts them to float32 without changing semantics.

### Packaging caveat

`pip check` reports `rocketsim 2.2.1 is not supported on this platform` under Python 3.12. The wheel filename and internal compatibility tag disagree; runtime imports/simulation nevertheless passed. We documented the discrepancy, did not patch package metadata, and did not randomly downgrade the working stack. This remains a reproducibility caveat.

Achievement: a functioning headless simulator independent of Rocket League/RLBot. This was not yet evidence of learning.

## 5. Reward and CUDA validation

[reward_test.py](training/reward_test.py) uses controlled simulator states rather than waiting for random goals. Historical successful checks:

- Real car/ball collision produced touch reward +0.1 for the touching agent and zero for the other.
- Controlled shots into both goals verified scoring +10, conceding -10 and correct agent IDs.
- Goals terminated episodes; idle kickoff timed out and truncated.

[gpu_test.py](training/gpu_test.py) confirmed Python 3.12.10, PyTorch 2.11.0+cu128, bundled CUDA runtime 12.8, `torch.cuda.is_available() == True`, NVIDIA GeForce RTX 4050 Laptop GPU, approximately 6.0 GiB VRAM, and GPU matrix multiplication matching CPU results. Historical driver inspection reported driver 581.86 and CUDA driver support 13.0. Driver capability and bundled runtime are different values.

Only the necessary PyTorch stack was installed; no separate manual CUDA toolkit was needed. Downloads and long-running training/evaluations were delegated to the user's terminal.

Lesson: reward events and GPU execution were validated independently before interpreting PPO output.

## 6. The PPO implementation and shared contract

Core files: [train.py](training/train.py), [ppo.py](training/ppo.py), [config.py](training/config.py), [checkpoint.py](training/checkpoint.py), [environment.py](training/environment.py), [policies](training/policies/__init__.py), [observations](training/observations/__init__.py), [evaluate.py](training/evaluate.py).

Separate networks:

```text
Actor:  92 -> 128 tanh -> 128 tanh -> 90 logits
Critic: 92 -> 128 tanh -> 128 tanh -> 1 value
Total actor + critic parameters: 68,571
```

Shared action contract: RLGym LookupTableAction, 90 entries, eight controls in order `throttle, steer, pitch, yaw, roll, jump, boost, handbrake`; RepeatAction holds each choice for eight physics ticks. RocketSim runs at 120 ticks/s, yielding nominally 15 decisions/s. The contract includes `rlbot_delay=True`.

Common experimental defaults: seed 42; rollout 256 steps per environment; learning rate 0.0003; gamma 0.99; GAE lambda 0.95; PPO clip 0.2; four epochs; minibatch 256; entropy coefficient 0.01; value coefficient 0.5; gradient clipping 0.5; approximate-KL stopping target 0.03; periodic checkpoint interval 32,768 transitions.

PPO uses GAE, distinguishes goal termination from timeout bootstrapping, stops advantages across resets, normalizes advantages and checks finite loss/gradients. **It does not normalize rewards or returns.** Therefore reward scale affects the critic's raw squared loss and large losses cannot be compared blindly across reward recipes.

Checkpoints contain model/optimizer state, configuration, versions and observation/action contract. Resume adds transitions to the checkpoint count but starts fresh simulator episodes and RNG streams; it is not a bit-exact continuation of a saved physics world. Run directories and report filenames are protected against silent overwrites.

### Fixture definitions

| Fixture | Initial task |
|---|---|
| touch | Stationary nearby ball; learner roughly 550 units behind it with lateral variation; distant idle opponent |
| bridge | Distance 550–1100, lateral offset up to 300, heading variation up to 0.3 rad; stationary ball, idle opponent |
| approach | Distance 900–1700, lateral offset up to 600, heading variation up to 0.65 rad; stationary ball, idle opponent |
| kickoff | Standard kickoff setup; fixed scripted opponent v1 or v2 |

Both teams are exercised by alternating learner sides. Evaluation seeds and sampled-action seeds are recorded separately. Most later comparisons use 40 episodes and action seed 3000. Early match tests are short sudden-death episodes, not full five-minute live matches.

## 7. Experiment ledger: initial sparse-reward PPO

### E01 — `ppo_smoke`: 0 -> 4,096

One environment, 30-second episode limit, original goal10/touch0.1 reward. Saved step 0 and 4,096 checkpoints.

At completion: rolling reward 3.0, policy loss -0.00364, value loss 0.02746, entropy 4.495, approximately 365 transitions/s.

Ten-game trained evaluation: four wins, zero losses, six draws, **mean learner touches 0**. Separate 20-game controls: untrained policy seven wins and mean touches 0.1; random actions eight wins and mean touches 0.1.

Achievement: rollout, updates, checkpoint saving/loading and evaluation worked. Lesson: even random/untrained behavior could win against the first opponent; these wins were not evidence of improved skill.

### E02 — `ppo_scale`: 4,096 -> 69,632

Continuation of the smoke checkpoint with four environments and 65,536 additional transitions. Saved 4,096, 36,864 and 69,632.

Final rolling reward 1.913, policy loss -0.01247, value loss 0.06075, entropy 4.319, approximately 582 transitions/s. Twenty-game evaluation: five wins, zero losses, fifteen draws; goals 5–0; mean touches **0**; all five wins were recorded as wins without learner touch.

A repeated evaluation printed results but raised `FileExistsError` when its output path already existed. This was report-overwrite protection, not a failed simulator/model evaluation. Later scripts saved numbered alternatives instead of discarding previous reports.

Lesson: apparent 25% win rate was a false positive for learner competence. Opponent own goals and reward attribution required investigation.

## 8. E03 — `touch_probe`: 0 -> 65,536

Fresh four-environment nearby-ball task, four-second episodes, original goal10/touch0.1 reward, all 90 actions. Saved 0, 32,768 and 65,536.

Training rolling reward started 0.01875, briefly increased, then ended 0; entropy fell from approximately 4.499 to 3.468. This looked poor in stochastic training, but deterministic evaluation revealed a narrow contact maneuver.

| Seed / mode | Initial mean touches | Trained mean touches | Trained touch episodes | Ground proxy | Ball progress |
|---|---:|---:|---:|---:|---:|
| 1000 argmax | 0 | 1.150 | 95% | 0% | 844 |
| 2000 argmax | 0 | 1.275 | 97.5% | 0% | 899 |
| 1000 sampled diagnostic | — | 0 | 0% | 0% | 0 |
| 2000 sampled diagnostic | — | 0 | 0% | 0% | 0 |

Initial and trained close-ball evaluations were all draws. Mean argmax first-touch time was approximately 1.79 s. Jump-action fraction was approximately 69% argmax versus 29.5% sampled. Historical action analysis found action 78 on 1,661/2,400 seed-1000 argmax intervals and action 7 on 659/2,400; these did not establish normal propelled ground approach.

Kickoff transfer against v1 appeared to win nine of twenty games, but all nine wins occurred without learner touch. Mean touches was only 0.1. Against improved opponent v2, both initial and trained policies lost all twenty games, conceded twenty goals and made zero touches.

Lesson: PPO learned something narrow, and argmax/sampling can produce dramatically different trajectories. Falling stochastic reward does not contradict a deterministic shortcut. Neither proves convergence or useful match skill.

## 9. E04 — `ground_probe_v1`: 0 -> 32,768

Same short touch task and sparse reward, but optional ground-only mask selected among the first 24 lookup entries while retaining 90 actor outputs. Saved 0 and 32,768.

| Mode, seed 1000 | Initial touch episodes | Trained touch episodes | Initial mean touches | Trained mean touches |
|---|---:|---:|---:|---:|
| Argmax | 0% | 0% | 0 | 0 |
| Sample | 52.5% | 12.5% | 0.650 | 0.125 |

Jump fraction was zero by construction. Sampled ball progress fell from 665 to 263. Final rolling reward 0.04, entropy 2.880, finite losses.

Lesson: removing jump did not teach ground control in this short experiment. It exposed sparse-signal/control problems but did not establish that PPO cannot learn driving at meaningful scale. This was not retained as the default action contract.

## 10. E05 — `ground_approach_v1`: 0 -> 32,768

This saved run is present even though it was not fully described in the earlier conversation. Configuration: touch fixture, four seconds, four environments, ground-only mask, original goal10/touch0.1 plus signed approach weight 0.02. Saved 0 and 32,768.

Training evidence: rolling reward approximately -0.000084 at step 1,024 and -0.188692 at completion; final policy loss -0.00549, value loss 0.000524, entropy 2.370, approximately 491 transitions/s; final rollout recorded zero touches.

No standalone before/after evaluation JSON exists under this run directory at this update. Therefore no behavioral improvement or held-out contact rate can be claimed. Lesson: record this run as executed, but do not fill gaps with guessed results.

## 11. E06 — `early_reward_probe_v1`: 0 -> 65,536

Fresh full-action experiment: eight-second touch episodes; weights goal0, touch1, signed approach0.02, moving-facing0.01, air0.001. Saved 0, 32,768 and 65,536.

The custom ApproachReward projects car displacement toward the prior ball position over the interval, normalizes by max-speed travel distance and clips to [-1,1]. It is **not** the instantaneous tutorial velocity reward. The older FacingBallReward is speed-gated, not the later guide orientation-only term.

Final rolling reward 0.813; policy loss -0.00834; value loss 0.07317; entropy 4.219; approximately 597 transitions/s.

| Seed / mode | Initial touch rate | Trained touch rate | Trained mean touches | Trained ground proxy |
|---|---:|---:|---:|---:|
| 1000 argmax | 0% | 87.5% | 1.550 | 87.5% |
| 2000 argmax | 0% | 87.5% | 1.475 | 87.5% |
| 1000 sample | 30% | 62.5% | 0.650 | 25% |
| 2000 sample | 25% | 57.5% | 0.700 | 25% |

Argmax first touch approximately 0.67 s, almost no jump actions, mean goalward ball movement around 3,900 units. Three and two argmax goals occurred across the two 40-episode sets, all after learner contact. These are narrow task results, not match competence.

Wider approach seed 3000: argmax touch rate 10% -> 17.5%; sample 15% -> 22.5%. Trained mean touches 0.250/0.275 and ball progress 711/940. Transfer was much weaker than close-ball performance.

This became the frozen historical reference:

`training/checkpoints/early_reward_probe_v1/baseline_step_65536.pt`

SHA-256: `484499AD7EFF3EA944E5E93A321C24E58BBA4C02392715DEDADF0D142F9939EF`.

Lesson: dense shaping helped a narrow grounded contact behavior. Success was brittle across a broader reset distribution, and 65k transitions remained a diagnostic-scale budget.

## 12. E07 — `approach_stage2_v1`: 65,536 -> 196,608

Resumed the frozen E06 reference for 131,072 additional transitions on the wider approach fixture, ten-second episodes, same architecture/actions/reward. Saved 65,536, 98,304, 131,072, 163,840 and 196,608.

Rolling reward rose approximately -0.14 -> +0.256, but behavioral evaluation regressed.

| Checkpoint / seed | Argmax touch rate | Sample touch rate | Sample mean touches |
|---|---:|---:|---:|
| Reference 65,536 / 4000 | 22.5% | 25% | 0.300 |
| 98,304 / 4000 | 0% | 7.5% | 0.100 |
| 131,072 / 4000 | 0% | 20% | 0.250 |
| 196,608 / 4000 | 0% | 20% | 0.350 |
| 196,608 / 3000 | 0% | 10% | 0.125 |
| 196,608 / touch seed1000 | 0% | 42.5% | 0.600 |

Final argmax reports selected action 52 throughout: an effectively idle ground action with roll -1 and no throttle/steer/jump/boost. Intermediate checkpoints did not demonstrate a recovery; step 163,840 has no separate saved evaluation.

Lesson: improving shaped reward did not imply improving approach. A signed movement penalty plus sparse success could favor inactivity, but that causal explanation remained a hypothesis. Preserve the branch; do not promote/resume it as the next baseline.

## 13. Bridge transfer and E08 — `bridge_stage2_v1`: 65,536 -> 98,304

Before another continuation, evaluated the reference on gentler bridge fixtures:

| Seed | Argmax touch / ground proxy | Sample touch / ground proxy | Argmax ball progress |
|---|---|---|---:|
| 5000 | 50% / 50% | 42.5% / 17.5% | 2,047 |
| 6000 | 42.5% / 42.5% | 22.5% / 5% | 1,810 |

The reference scored three/four argmax goals after contact. The bridge distribution overlapped the proven close task without the full approach variation.

E08 resumed the same reference for 32,768 additional bridge transitions with unchanged signed reward. Saved 65,536 and 98,304. Final rolling reward 0.310, entropy 3.706, approximately 455 transitions/s.

Final argmax touch rate: **0%** on bridge seeds 5000 and 6000 and touch seed1000. Action 38 dominated every argmax interval—effectively idle on ground. Sample bridge touch rates were 32.5% and 27.5%, ground proxies 2.5% each; close-task sample contact 47.5% versus reference 62.5%.

Lesson: a gentler fixture alone did not preserve the deterministic skill. Another positive reward curve was not a successful curriculum advance.

## 14. E09 — `bridge_positive_reward_v1`: 65,536 -> 98,304

Controlled ablation resumed the original reference, not the failed approach/bridge branch. Same bridge horizon, additional steps and model; only negative custom approach values were clipped to zero. Saved 65,536 and 98,304.

Final rolling reward 0.650, entropy 4.146, approximately 646 transitions/s.

| Fixture | Argmax touch rate | Sample touch rate | Argmax ground proxy | Sample ground proxy |
|---|---:|---:|---:|---:|
| bridge5000 | 35% | 42.5% | 0% | 0% |
| bridge6000 | 22.5% | 37.5% | 0% | 0% |
| touch1000 | 47.5% | 65% | 0% | 2.5% |

This avoided total argmax inactivity but did not recover the reference's reliable ground contacts. Higher shaped reward and occasional goals did not justify baseline promotion. Both 98,304 bridge branches were retained as diagnostic artifacts, not continuing training sources.

## 15. Reward-accounting diagnostics and untrained proposals

[diagnose_steps.py](training/diagnose_steps.py) records each interval's weighted rewards, exact action, speed, radial velocity, distance/progress and ground/contact flags. Three-episode bridge seed5000 comparison:

| Metric | Frozen reference | Positive-only branch |
|---|---:|---:|
| Intervals | 450 | 396 |
| Positive approach intervals | 151 | 200 |
| Positive approach without throttle/boost | 16 | 145 |
| Fraction of positive intervals without propulsion command | 10.6% | 72.5% |
| Approach reward on those intervals | 0.0374 | 0.5516 |
| Total signed/original approach reward | -0.9751 | 0.7662 |
| Total touch reward | 1 | 6 |

No throttle/boost command is a proxy, not proof of passive coasting: gravity, collisions, momentum and steering also affect motion. Of the candidate's 145 no-propulsion positive intervals, 144 were airborne.

[compare_approach_rewards.py](training/compare_approach_rewards.py) rescored saved trajectories with old recorded reward, pre-touch-gated old positive reward, gated distance progress and gated closing-velocity improvement. Candidate airborne no-propulsion positive intervals changed from 144 under recorded reward to 20 under gated old/distance reward and 86 under closing-velocity improvement. Candidate corresponding total scores were 0.7662, 0.2165, 0.2139 and 0.0303; these scales were not equivalent.

Lesson: first-touch gating changed reward accounting on fixed trajectories, but did not establish what PPO would learn. Closing-velocity change could still credit physics without active propulsion. Contact intervals could not be split precisely by an eight-tick trace.

Status: a first-touch-gate flag and tests were added, but **no first-touch-gated training run exists**. No closing-velocity or alternate distance-progress training branch was performed. The plan shifted to a conventional velocity reward before these proposals were trained.

Short saved diagnostic smoke reports (`diagnostic_smoke_step65536*.json`, `raw_reward_smoke_3steps.json`) verified tracer/report plumbing, not new learned skills.

## 16. Conventional reward scale measurements

The installed RLGym Rocket League 2.0.1 reward package did not ship SpeedTowardBallReward. A local implementation mirrored the documented tutorial formula with a zero-distance guard. It was first measured separately, then used in the guide recipe. The nonnegative tutorial velocity term is distinct from our optional positive-only custom progress ablation.

Forty episodes per configuration, frozen reference:

| Fixture / mode | Touch mean | Velocity-to-ball mean / p95 | Old face mean / p95 | Air mean |
|---|---:|---|---|---:|
| bridge7000 argmax | 0.00500 | 0.09389 / 0.50410 | -0.10097 / 0.57138 | 0.02501 |
| touch3000 argmax | 0.01889 | 0.09069 / 0.46376 | 0.05400 / 0.63100 | 0.03496 |
| bridge7000 sample | 0.00302 | 0.06450 / 0.33844 | -0.04182 / 0.33415 | 0.71515 |
| touch3000 sample | 0.00750 | 0.06398 / 0.30207 | -0.02434 / 0.32503 | 0.75042 |

Touch p95 was zero because contact was sparse; touch max was one. Sampled trajectories spent far more time airborne. These old face statistics measure the speed-gated FacingBallReward, **not** the subsequently introduced GuideFaceBallReward.

Lesson: distribution and term definition matter, not merely copied weights. Raw statistics do not prove a good learning objective.

## 17. Pipeline throughput profile

[profile_pipeline.py](training/profile_pipeline.py), bridge, four serial environments, 256 rollout steps, 1,024 transitions:

| Component | Seconds | Wall fraction |
|---|---:|---:|
| Policy inference | 0.4579 | 38.5% |
| Simulator stepping | 0.2907 | 24.5% |
| PPO update | 0.4197 | 35.3% |
| Bootstrap/GAE | 0.0155 | 1.3% |
| Batching/bookkeeping | small | below 1% combined |

Total 1.1879 s, approximately 862 transitions/s. This was one disposable rollout plus one in-memory optimizer update; no checkpoint was altered or saved by profiling. It was not strictly a zero-gradient diagnostic and not a learning experiment.

Lesson: simulator stepping was not the only bottleneck. Multiprocessing was discussed but **not implemented**; reward and worker changes were not combined.

## 18. E10 — `guide_early_preflight_v1`: 0 -> 8,192

Fresh bridge run, ten seconds, four environments, full action space, conventional `guide_early_v1`:

| Term | Weight |
|---|---:|
| TouchReward | 50 |
| SpeedTowardBallReward | 5 |
| GuideFaceBallReward, orientation-only | 1 |
| InAirReward | 0.15 |
| GoalReward | 0 |

No custom progress, first-touch gate or closing-velocity delta. Saved 0 and 8,192. Rolling reward 49.03 at 1,024 -> 66.07 at 8,192; final policy loss -0.00710, value loss 105.23, entropy 4.453. Larger reward weights produced larger value-loss scale. All shown losses were finite.

Achievement: this configuration completed a short preflight. No standalone behavioral evaluation exists for its 8,192 checkpoint; do not claim ground skill from it.

## 19. E11 — `guide_early_1m_v1`: 0 -> 1,000,000

Fresh actor/critic, not a continuation of the 65k reference or 8,192 checkpoint. Same guide recipe and bridge distribution. The matching early preflight output reflects the same seeded setup, not proof that its saved checkpoint was resumed. Repeated copies of the same terminal transcript are not counted as separate saved runs.

Thirty-two checkpoint files are preserved, including initial, periodic and final weights. Final:

`training/checkpoints/guide_early_1m_v1/baseline_step_1000000.pt`

SHA-256 verified unchanged in the later sensitivity audit:

`f1a6b40950f79045a7a13d50c6a38935834e83dd4ad5c9333120338bed0d7281`

| Training point | Rolling reward | Entropy |
|---|---:|---:|
| 1,024 | 49.03 | 4.499 |
| 65,536 | 68.27 | 4.186 |
| 262,144 | 124.63 | 3.561 |
| 524,288 | 162.24 | 3.342 |
| 1,000,000 | 208.99 | 2.649 |

Final policy loss -0.01463, value loss 255.106, approximate KL 0.000833, cumulative throughput approximately 559 transitions/s. This implies roughly 30 minutes at the recorded average, not a guaranteed future runtime.

### Paired held-out evaluations, same fixtures and action seed

Ground values below are the **older conservative evaluator proxy**.

| Fixture / mode | Touch rate 0 -> 1M | Mean touches 0 -> 1M | Ground proxy 0 -> 1M | Ball progress 0 -> 1M | First-touch s 0 -> 1M |
|---|---|---|---|---|---|
| bridge7000 argmax | 5% -> 65% | 0.050 -> 0.825 | 0% -> 2.5% | 235 -> 1,131 | 1.00 -> 1.86 |
| bridge7000 sample | 17.5% -> 47.5% | 0.175 -> 0.750 | 2.5% -> 2.5% | 596 -> 1,332 | 2.48 -> 1.74 |
| bridge8000 argmax | 2.5% -> 65% | 0.025 -> 0.875 | 0% -> 2.5% | 113 -> 903 | 1.07 -> 1.94 |
| bridge8000 sample | 20% -> 62.5% | 0.200 -> 0.900 | 2.5% -> 0% | 793 -> 1,176 | 1.49 -> 2.73 |
| touch3000 argmax | 0% -> 37.5% | 0 -> 0.375 | 0% -> 0% | 0 -> 923 | n/a -> 0.87 |
| touch3000 sample | 27.5% -> 55% | 0.300 -> 0.875 | 5% -> 2.5% | 970 -> 1,457 | 1.15 -> 1.52 |

First-touch time is conditional on touching; more successful episodes can increase that mean by including slower successes. It is not directly a speed regression across unmatched successful subsets.

Harder approach seed9000 at 1M: argmax touch 10%, mean touches0.1, first touch3.22 s, ground proxy0%, ball progress196; sample touch35%, mean0.525, first touch3.58 s, ground proxy0%, progress853.

Kickoff-v2 seed9000, twenty argmax games: **0 wins, 20 losses, 0 learner touches**, opponent first-touch rate100%, goals0–20, mean duration4.75 s, ball progress approximately -5,291. Positive shaped reward53.93 did not signify useful play.

Achievement: stronger transferable contact evidence than the tiny early probes. Lesson: broad ball-contact improvement is not reliable grounded approach or full match competence. A million transitions still does not automatically validate the starter.

## 20. Contact classification, first-touch sequences and grounded startup

[contact_diagnostics.py](training/contact_diagnostics.py) generated sixteen reports: initial/1M, argmax/sample, bridge7000/bridge8000/touch3000/approach3000, forty episodes each.

It uses RocketSim contact-time wheel state rather than an arbitrary car-height threshold. Older evaluate.py ground proxy required grounded state across interval boundaries; contact-time classification can detect a ground hit followed by bounce. Their percentages are **not interchangeable**.

At 1M:

| Fixture | Argmax ground-touch episodes | Argmax airborne contact fraction | Sample ground-touch episodes | Sample airborne contact fraction |
|---|---:|---:|---:|---:|
| bridge7000 | 10% | 85.5% | 10% | 83.7% |
| bridge8000 | 12.5% | 83.1% | 12.5% | 85.2% |
| touch3000 | 0% | 100% | 5% | 93.7% |
| approach3000 | 7.5% | 72% | 5% | 89.2% |

Approach3000 argmax touch episodes improved10% ->25%, mean touches0.125 ->0.275; sample15% ->35%, mean0.2 ->0.5. Most contacts remained airborne.

Historical first-touch window analysis reported 157 touch episodes and 1,565 preceding intervals: forward throttle58%, steer52.7%, jump28.9%, nonzero pitch64%, boost44%, grounded7.7%. Argmax subset: throttle58.3%, steer57%, jump32.7%, pitch67.1%, boost41.4%, grounded5.3%. Jump was selected within the preceding ten intervals in156/157 episodes. Many cars were already airborne at the beginning of the window, so this did not establish a simple ground-drive -> jump sequence.

Subsequent grounded-start analysis, reported in the project discussion, found:

- Across160 1M argmax episode starts, all160 selected forward throttle/boost and had positive horizontal closing velocity afterward.
- Across320 episode samples, first jump occurred on interval1 in220, by interval3 in305 and by interval5 inall320.
- For100 episodes with delayed jumping, initial grounded phase averaged1.56 intervals (~0.10 s), forward throttle92%, boost91%, end speed189 uu/s and closing velocity176 uu/s; distance decreased in99/100.

These sequence/startup aggregates are historical analysis results reported in the discussion; no separate standalone summary JSON for them was found in the current inventory. The underlying contact reports remain available.

Lesson: initial propulsion exists. The policy rapidly transitions to aerial contact; we should not misdiagnose all failure as inability to apply throttle. Sustained orientation/ground control is the unproven capability.

## 21. Deploying the frozen 1M policy through the existing example

The user explicitly requested integration into **the existing** `python-example`, not a second bot/project/launcher. Existing bot class is `MyBot` in [src/bot.py](python-example/src/bot.py).

Implemented [ml_policy.py](python-example/src/ml_policy.py) for checkpoint loading, exact action table, CPU eval/no_grad inference and RLBot-state-to-GameState adapter. The existing bot receives GamePacket in get_output, constructs observations, chooses an action every eight physics frames and holds its controller between decisions. Model loads once. Argmax is default; sampling is configurable. Periodic debug prints report action, controls, positions and inference time.

The live RLBot venv remains separate. Deployment exposes needed training-library imports from the training venv rather than reinstalling/replacing the known-good RLBot stack. This is workable local integration, not yet a portable participant package.

Self-test facilities check checkpoint/network shapes, finite inference, valid table index/controller conversion and bot initialization. Subsequent live recordings prove the policy produced controllers in Rocket League. Inference observed around0.3–0.6 ms. No RocketSim world, rewards or optimizer runs in live inference.

No ball-chasing, heuristic steering, scripted jump timing or teacher fallback was inserted into ML gameplay. Neutral handling of unavailable replay/invalid packets is not gameplay assistance.

### Watched live result

User watched approximately five minutes: barely any ball interaction, frequent acceleration/turning/circling, poor maneuvering. Manually disrupting the car changed action regime; returning to normal ground states often restored the hard-left cycle. Physically pushing the ball nearby still did not reliably elicit approach/contact.

Lesson: deployment was technically functional, but the model had not become a playable starter. Distance-to-ball alone cannot explain the live failure. Do not declare a model defect or adapter defect without isolating the contract and policy behavior.

## 22. Live observation/action audit and continuous recording

[observation_diagnostics.py](python-example/src/observation_diagnostics.py) added machine-readable feature labels, action mapping reports, controlled synthetic cases, optional live state probes and continuous JSONL recording. Existing artifacts are under [observation_diagnostics](python-example/logs/observation_diagnostics/).

Exact training DefaultObs layout:

| Slots | Meaning |
|---|---|
| 0–2 | Ball team-world position /2300 |
| 3–5 | Ball team-world linear velocity /2300 |
| 6–8 | Ball team-world angular velocity /pi |
| 9–42 | 34 team-ordered boost-pad cooldowns /10 |
| 43–51 | Holding-jump, handbrake, jumped/jumping/flipped/flipping/double-jump/can-flip flags and raw air-time |
| 52–71 | Own position, forward, up, linear/angular velocity, boost, demo timer, ground/boost/supersonic flags |
| 72–91 | Opponent same20 fields |

Own/opponent position and linear velocity use /2300; angular velocity /pi; boost /100; vectors/flags and demo timer are not rescaled like positions. Orange inversion rotates world coordinates180 degrees and reorders pads. There is no explicit car-local ball-left/forward feature. Relative geometry in logs is diagnostic-only.

Known live approximations: analog handbrake ramp reconstructed from controls; boost/air/flip timer proxies; grounded state differs from simulator wheel-contact definition. They remain potential material contributors, not proven fatal causes.

Saved action audit:90 rows, `all_rows_equal=true`, including controller fields. Synthetic blue/orange mirrored scene: all92 slots equal within approximately8.74e-8. This does not prove all real live packet states are equivalent. No confirmed fatal feature-order/scaling/team inversion or numeric table/controller sign mismatch was found.

### Continuous logging

Recording was changed from a small frame sample to continuous action-decision JSONL with flushing, heartbeat/phase/error records and approximately64 MiB rotation, so data is written while the match is running. The current match config has five-minute regulation and unlimited overtime; recording must not assume it ends after five minutes. JSONL is append-friendly, unlike a single unfinished JSON array.

Earlier recording004 primarily captured countdown/kickoff/other phases and could not establish sustained active-play behavior. Newer segments006/007 contain one continuous session with14,646 decisions,14,370 active-play decisions and no policy_error events. Segment timestamps extend roughly0.15–1075.39 game seconds; this is not a claim that the user watched every minute.

Action51:8,317 decisions; action12:4,358; together86.54%. Recorded grounded fraction approximately89.8%, jump-command fraction8.58%, boost-command fraction62.34%. Action51 includes throttle+1, steer-1, yaw-1, roll+1, boost1; action12 includes throttle+1, steer-1, yaw-1, boost0. The omitted roll field matters when airborne.

### Important correction discovered later

The old diagnostic incorrectly negated PhysicsObject.right when labeling driver-right. Earlier derived local lateral signs/left-right probe labels need reinterpretation. **That error was not used in the model's92D observation or controller conversion.** It is a confirmed diagnostic-label bug, not a confirmed gameplay-adapter bug. Existing historical files are preserved with this caveat.

## 23. Latest completed experiment: frozen-policy ball sensitivity

[policy_ball_sensitivity.py](training/policy_ball_sensitivity.py) and [numbered report001](training/logs/guide_early_1m_v1/policy_sensitivity/ball_sensitivity_001.json).

Seven fixed templates x six directions x three distances =126 CPU inference interventions. Four synthetic templates use the installed exact DefaultObs builder; three use recorded live states and alter only slots0–2. All89 remaining features are asserted identical. Full92D inputs,90 logits/probabilities, argmax IDs and eight controls are saved. Checkpoint hash before/after matched. No optimizer, simulator stepping, training, reward modification or new live match occurred.

Direction order: ahead, ahead-left, ahead-right, left, right, behind. Distances250/1000/3500 uu.

| Template | Close | Medium | Far |
|---|---|---|---|
| Ground blue facing goal | 41,41,41,41,41,41 | 41,41,41,41,41,41 | 51,51,41,51,41,41 |
| Mirrored orange | Same | Same | Same |
| Ground blue yaw0 | All41 | All41 | All41 |
| Ground blue speed500 | All41 | All41 | 51,51,41,41,41,41 |
| Recorded ground action51 | All41 | All41 | 51,41,15,41,15,41 |
| Recorded airborne action41 | All41 | All41 | All41 |
| Recorded ground action12 | All12 | All12 | 12,12,12,12,51,12 |

No tested case selected positive steer. Action41 includes throttle1, steer0, pitch-1, yaw0, roll1, jump1, boost1, handbrake1—not ordinary neutral ground acceleration.

Probabilities changed: stationary-blue paired left/right mean total-variation distance~0.244, maximum~0.582. Recorded ground-action12 state mean~0.055; expected steer remained around-0.85 to-0.93.

Conclusion: the policy is **not literally ball-position insensitive**, but often fails to turn changed geometry into useful actions. This supports brittle state-conditioned control, not proof that92D lacks ball information or the architecture is incapable. Fixed interventions are not rollout proof; selected live exemplars are few and some far-ball mathematical probes may be outside legal field bounds.

## 24. Research and design review: teacher -> BC -> PPO

Full review: [v2_bootstrap_review_001.md](training/reports/v2_bootstrap_review_001.md).

Research considered public replay/dataset options, replay-pretraining, richer relative observations and imitation followed by RL. A separate requested public-dataset investigation was canceled before completion. No public dataset was downloaded or converted, and no replay/inverse-dynamics model was trained. Do not present a dataset selection as completed work.

RelativeDefaultObs was inspected as a **candidate**, not installed/adopted. Configured1v1 can remain92D while changing semantics: relative ball/other-car physics, absolute own-car fields. Default padding would instead yield172 values. A reviewed orientation formula differed from direct basis projection in a mixed-rotation numeric test (maximum absolute difference0.2335); convention tests/pinning are required before adoption. No existing checkpoint was fed relative observations.

Recommendation: weak scripted teacher -> diverse simulator demonstrations -> small BC actor -> measured PPO fine-tune is a justified next hypothesis for a functional weak starter. It is not a proven fix. BC can clone bad habits or fail on student-visited states; validate teacher and closed-loop student separately.

Proposed teacher: identify opponent goal, choose desired strike direction, target a point behind the ball, approach/align, then drive through it. Keep simple steering/throttle/braking/reverse/boost decisions, exact lookup indices/eight-tick cadence, no advanced aerials/prediction. Teacher is offline supervision, **not final live gameplay control**.

Proposed dataset: varied distances/headings/velocities/field positions, both teams, offense/clearing/kickoff/recovery states; label observation BEFORE exact applied action; split by episode/family, not neighboring frames. A small initial100k–250k-label budget was discussed, not generated.

Proposed BC: retain small architecture and90 outputs; supervised categorical action labels; test existing DefaultObs first, relative variant separately with matched states/budget. Keep fresh actor as primary BC experiment and1M as benchmark. Add PPO only after independent approach/ground-contact/live evidence; no new reward weights selected yet.

## 25. Latest pending task and what has NOT happened

Root [TEACHER_TASK.md](TEACHER_TASK.md) specifies a teacher-only build/validation phase: controlled directional scenarios, randomized states, geometry/target/action traces, contact/ground/useful-direction/circling/recovery metrics, then stop/report before BC. Its presence is a specification, not proof of an implemented teacher.

At this documentation snapshot no standalone teacher implementation or teacher-validation reports were found. Existing scripted benchmark opponent v2 is not automatically a validated demonstration teacher. Any future temporary live teacher diagnostic must be explicit and separate from the ML gameplay path; this update does not authorize or perform it.

Not completed: teacher pilot validation; demonstration collection; behavior cloning; relative-input training; BC/PPO fine-tuning; multiprocessing; participant ZIP/validator; tournament runner; accepted organizer baseline.pt.

Current work is measurement/design, not another training run. Current deployed ML checkpoint remains the frozen1M experiment.

## 26. What we have achieved, and what each claim means

| Claim | Status |
|---|---|
| Epic/RLBot server/example launches | Verified historically and used live |
| Headless RocketSim/RLGym reset/step/reward lifecycle | Verified |
| CUDA/PyTorch on actual laptop | Verified |
| PPO updates, checkpoints, reproducible evaluations | Implemented and exercised |
| Narrow contact learning | Demonstrated in multiple controlled experiments |
| Sustained useful ground play | Not demonstrated reliably |
| Standard kickoff-v2 interaction | Failed in recorded1M test: zero learner touches |
| Live ML inference/controller output | Works technically |
| Perfect simulator/live observation equivalence | Not established; approximations remain |
| Why circling happens | Strong evidence of biased brittle policy; complete causal decomposition unresolved |
| Teacher/BC pivot | Designed and technically motivated, not trained/validated |
| Ready organizer baseline | No |

Most important lessons: preserve negative results; win without interaction is not competence; action buttons are not physical progress; contact is not useful contact; reward increase is not skill increase; argmax and sampled behavior differ; evaluate both controlled and full-play distributions; check feature semantics rather than only dimensions; classify contacts at contact time; distinguish diagnostic-label errors from gameplay errors; never promote on optimizer metrics alone.

## 27. Source and script index

Setup/runtime: `smoke_test.py`, `reward_test.py`, `gpu_test.py`, requirements/README. Learning: config/environment/observations/rewards/policies/checkpoint/ppo/train. Behavioral evaluation: evaluate/opponent_test/test_match_metrics. Detailed measurement: diagnose_steps/compare_approach_rewards/profile_pipeline/contact_diagnostics/policy_ball_sensitivity. Deployment: python-example bot/ml_policy/observation_diagnostics and existing run.py/config.

Regression-test files include `test_ppo.py`, `test_ground_policy.py`, `test_approach_reward.py`, `test_approach_start.py`, `test_compare_approach_rewards.py`, `test_early_rewards.py`, `test_speed_toward_ball.py`, `test_guide_reward.py`, `test_match_metrics.py` and `opponent_test.py`. These cover analytical GAE behavior, masking, controlled rewards/starts, counterfactual accounting and metrics/opponent behavior. Historical development ran short checks, but file existence alone is not a complete dated execution record. No tests/training/evaluations were rerun to write this document; the artifact inventory was read.

The following appendices give exact checkpoint and evaluation coverage so a future contributor can trace any summarized result back to the repository.


## 28. CanoPy feasibility investigation — 4 October 2026

### Scope and current trajectory

Both `MODEL_WARS_DECISION_LOG.md` and this historical record were read completely before the investigation. The current objective is a coherent, deliberately weaker **2v2 ML student**, using a verified CanoPy teacher and **online policy distillation** by default, with bounded optional PPO refinement later. Older scripted-teacher/offline-BC proposals in Sections 24–25 are historical. Tournament infrastructure remains parked.

The authorized task was **Step 1 only: investigate CanoPy provenance, checkpoint, configuration, and feasibility**. No student implementation, BC, PPO changes, training run, teacher gameplay evaluation, or live deployment occurred. Existing project source, old checkpoints, and the deployed bot were preserved. Only downloaded research artifacts and investigation documentation/evidence were added.

### What we downloaded and checked

References: [CanoPy model](https://huggingface.co/FlameF0X/CanoPy), [published model files](https://huggingface.co/FlameF0X/CanoPy/tree/main/model), RLGym DefaultObs/action-parser source, upstream Python `rlgym-ppo` policy/critic definitions, PyTorch serialization documentation, and the supplied Rust transfer-learning/RLBot references.

The initial shell download attempt was blocked by sandbox network access. The user requested that downloads and long-running commands be supplied for execution in their terminal, and ran the supplied PowerShell download command. It fetched the complete published revision:

`c171a3f6235d134556ff315c087bbea60056d727`

Saved directory: `training/reports/canopy_investigation_20261004_013501/`.

The revision contains `.gitattributes`, `README.md`, and:

| Model file | Bytes | Inspected contents |
|---|---:|---|
| `model/PPO_POLICY.pt` | 275,942 | Actor state dictionary |
| `model/PPO_POLICY_OPTIMIZER.pt` | 552,318 | Adam-compatible optimizer state |
| `model/PPO_VALUE_NET.pt` | 230,084 | Critic state dictionary |
| `model/PPO_VALUE_NET_OPTIMIZER.pt` | 460,672 | Adam-compatible optimizer state |
| `model/BOOK_KEEPING_VARS.json` | 410 | Training bookkeeping and reward statistics |

The download command's hashing pipeline also enumerated the `SHA256.csv` output while writing it, causing a file-lock error. This was a command defect, not evidence of a failed download. Subsequent independent inspection verified **all downloaded file sizes** and **all four checkpoint SHA-256 hashes** against the saved Hugging Face metadata. Actor hash:

`54162458cc17b8e21530befd717796c41b41a58545b9fafd5bfd10f6fbc4208c`

### Safe checkpoint inspection and actual architecture

Inspection used ZIP member checks and static `pickletools.genops()` analysis before restricted loading. The only detected pickle globals were `collections.OrderedDict`, `torch._utils._rebuild_tensor_v2`, and `torch.FloatStorage`. PyTorch's `get_unsafe_globals_in_checkpoint()` returned an empty list for all four files. All loaded with `torch.load(..., map_location='cpu', weights_only=True)` in the existing training venv. No unrestricted pickle loading or added safe-global allowlist was used. These are bounded deserialization checks, not a universal safety guarantee.

The actor's exact learned tensor shapes are:

| Tensor | Shape |
|---|---|
| `model.0.weight`, `model.0.bias` | `[256,92]`, `[256]` |
| `model.2.weight`, `model.2.bias` | `[128,256]`, `[128]` |
| `model.4.weight`, `model.4.bias` | `[90,128]`, `[90]` |

**Published actor: 92 → 256 → 128 → 90, 68,314 parameters.** Published critic: **92 → 256 → 128 → 1, 56,833 parameters**. All learned tensors were finite float32. The earlier expected **172-input CanoPy architecture is not supported by this export**.

A temporary in-memory reconstruction following upstream `rlgym_ppo.ppo.discrete_policy.DiscreteFF`—ReLU hidden layers and final Softmax—loaded all actor keys with `strict=True`. A zero 92D vector produced finite `[1,90]` probabilities, sum approximately `0.99999994`, argmax action 17. Softmax of the extracted final linear logits exactly matched the reconstructed output. This confirms tensor compatibility and access to all 90 scores **under that reconstruction**, not exact CanoPy inference parity or useful gameplay. Parameter-free activation classes cannot be proven from state-dictionary tensors alone; CanoPy's own source/version was not published.

### Blocking observation mismatch

Short state-construction probes using installed RLGym 2.0.1 DefaultObs confirmed:

| Mode | `zero_padding` | Observation size |
|---|---|---:|
| 1v1 | `None` | 92 |
| 2v2 | `None` | 132 |
| 2v2 | `2` | 132 |
| 2v2 | `3` (constructor default) | 172 |

The builder's formula is `52 + 20 * number_of_cars` without padding, or `52 + 20 * zero_padding * 2` with padding. Thus **172D is the padded contract, not the dimension of every 2v2 DefaultObs configuration**. Feeding either 132D or 172D to the reconstructed published actor raised a matrix-dimension error against its 92-input first layer.

Installed DefaultObs has exact known feature/scaling/team-inversion semantics, preserves insertion order within ally/opponent lists, and appends zero blocks after actual cars. With padded 2v2, indices 92–111 and 152–171 are zero blocks. The complete candidate layout is documented in the feasibility report. However, **these installed-library semantics cannot be asserted as CanoPy's exact contract**: the publication provides no observation-builder instantiation or dependency pin explaining its 92D inputs.

The numeric action table in our stack is `[90,8]`, with eight-tick RepeatAction producing `(8,8)` controls. CanoPy's README declares lookup-table actions and repeat 8, but exact teacher action-row ordering and control-delay settings remain unverified without its source/config. Our existing environment uses `rlbot_delay=True`.

**Do not infer that the published weights were necessarily trained in 1v1.** The tensor establishes a 92D input requirement, not the training game mode or feature meaning. Truncating 2v2 observations, dropping cars, or inventing input-layer weights would not reproduce a verified teacher.

### Configuration, provenance, and compatibility findings

- The complete published inventory has **no `config.json`, training/inference source, requirements lock, license file, or evaluation report**, despite README references to configuration and a training-script evaluation function. Exact CanoPy preprocessing, feature ordering, padding, activation classes, versions, and deployment behavior remain unknown.
- `BOOK_KEEPING_VARS.json` records **19,000,020 cumulative timesteps**, **114 model updates**, and epoch 18; both optimizers record step 114. The model card's **1B-step setting is not proof that this checkpoint completed 1B steps**.
- The README/model metadata declares **Apache-2.0 for the model repository**. This is positive licensing evidence, but there is no weight-specific provenance/ownership manifest or description of third-party initialization. Exact weight rights and upstream coverage were not independently established. Dependencies retain their own licenses; the report records Apache/MIT/BSD and relevant bundled-library terms separately.
- The actual training stack remains Python **3.12.10**, torch **2.11.0+cu128**, RLGym **2.0.1**, rlgym-api **2.0.0**, rlgym-rocket-league **2.0.1**, RocketSim **2.2.1**, NumPy **1.26.4**, cmeel **0.61.0**. Restricted loading and temporary CPU forward inference work without package changes. `rlgym-ppo` is not installed; its learner stack is not needed merely to reconstruct an actor.
- The sidecar contains reward running statistics, not observation statistics; the card says observation standardization was disabled. Reward statistics, critic, and optimizers are unnecessary for an actor-only forward pass. Unpublished preprocessing still cannot be ruled out.
- Our existing `training/checkpoint.py.load_checkpoint()` expects the local checkpoint wrapper and tanh ActorCritic, so it cannot directly load CanoPy's bare state dictionary. A separate verified teacher loader would be necessary after the contract blocker is resolved.

### Verdict and next gate

**Feasibility answer: C. NO for the artifact currently published and inspected.** The weights load and expose 90 outputs under a plausible upstream-compatible reconstruction, but a **verified frozen 2v2 teacher distribution is blocked** by the 92D checkpoint versus 132D/172D DefaultObs mismatch and missing exact source/configuration. No RocketSim 2v2 teacher rollout or live behavior result is claimed.

The smallest next step is to obtain the matching CanoPy 2v2 checkpoint, its hash, observation-builder configuration/source, policy class/version, exact action table, repeat/delay settings, dependency pins, reference outputs, and confirmation of weight-license coverage from the publisher. No message was sent to the publisher. If these cannot be obtained, investigate another permissively licensed, source-verifiable 2v2 teacher. **Hard-label BC does not solve missing observation semantics**; it is a fallback only when teacher actions are reliable but probabilities cannot be recovered.

Evidence and deliverable:

- [CanoPy Feasibility Report](training/reports/canopy_investigation_20261004_013501/CANOPY_FEASIBILITY_REPORT.md): complete 12-section report, source attribution, risks, and future integration points.
- [Checkpoint inspection](training/reports/canopy_investigation_20261004_013501/checkpoint_inspection.json): hashes, tensor shapes, restricted-load results, versions, and short probe outcomes.
- [Repository metadata](training/reports/canopy_investigation_20261004_013501/repository_metadata.json), [downloaded model card](training/reports/canopy_investigation_20261004_013501/README.md), and [bookkeeping](training/reports/canopy_investigation_20261004_013501/model/BOOK_KEEPING_VARS.json): exact downloaded provenance/configuration evidence.

Step 1 ended at the report. Student implementation/training, BC, PPO refinement, teacher gameplay validation, baseline promotion, and tournament work remain unperformed.

## Appendix A — every saved checkpoint and its training metrics

Paths are `training/checkpoints/<run>/baseline_step_<step>.pt`. A dash means no CSV update row exists at that exact step; initial/resume snapshots are not new trained/evaluated policies. Periodic weights are retained even where no separate behavioral report exists. All values below come from saved metrics.csv, not a new training execution.

### approach_stage2_v1

| Saved step | Rolling reward | Policy loss | Value loss | Entropy | Approx KL | Transitions/s |
|---|---:|---:|---:|---:|---:|---:|
| 65,536 | — | — | — | — | — | — |
| 98,304 | 0.043 | -0.01357 | 0.00320 | 4.259 | 0.002093 | 423.0 |
| 131,072 | 0.182 | -0.01297 | 0.00420 | 4.213 | 0.002094 | 400.1 |
| 163,840 | 0.218 | -0.01282 | 0.00435 | 3.930 | 0.002197 | 364.3 |
| 196,608 | 0.256 | -0.01388 | 0.02507 | 4.004 | 0.002451 | 345.9 |

### bridge_positive_reward_v1

| Saved step | Rolling reward | Policy loss | Value loss | Entropy | Approx KL | Transitions/s |
|---|---:|---:|---:|---:|---:|---:|
| 65,536 | — | — | — | — | — | — |
| 98,304 | 0.650 | -0.01097 | 0.02262 | 4.146 | 0.002317 | 646.2 |

### bridge_stage2_v1

| Saved step | Rolling reward | Policy loss | Value loss | Entropy | Approx KL | Transitions/s |
|---|---:|---:|---:|---:|---:|---:|
| 65,536 | — | — | — | — | — | — |
| 98,304 | 0.310 | -0.01132 | 0.01347 | 3.706 | 0.008008 | 454.6 |

### early_reward_probe_v1

| Saved step | Rolling reward | Policy loss | Value loss | Entropy | Approx KL | Transitions/s |
|---|---:|---:|---:|---:|---:|---:|
| 0 | — | — | — | — | — | — |
| 32,768 | 0.657 | -0.00958 | 0.09522 | 4.022 | 0.003080 | 747.3 |
| 65,536 | 0.813 | -0.00834 | 0.07317 | 4.219 | 0.001363 | 596.7 |

### ground_approach_v1

| Saved step | Rolling reward | Policy loss | Value loss | Entropy | Approx KL | Transitions/s |
|---|---:|---:|---:|---:|---:|---:|
| 0 | — | — | — | — | — | — |
| 32,768 | -0.189 | -0.00549 | 0.00052 | 2.370 | 0.003698 | 490.9 |

### ground_probe_v1

| Saved step | Rolling reward | Policy loss | Value loss | Entropy | Approx KL | Transitions/s |
|---|---:|---:|---:|---:|---:|---:|
| 0 | — | — | — | — | — | — |
| 32,768 | 0.040 | -0.00992 | 0.00044 | 2.880 | 0.006193 | 720.0 |

### guide_early_1m_v1

| Saved step | Rolling reward | Policy loss | Value loss | Entropy | Approx KL | Transitions/s |
|---|---:|---:|---:|---:|---:|---:|
| 0 | — | — | — | — | — | — |
| 32,768 | 62.251 | -0.00381 | 138.14430 | 4.354 | 0.000882 | 704.4 |
| 65,536 | 68.272 | -0.01051 | 131.14265 | 4.186 | 0.002489 | 561.1 |
| 98,304 | 76.537 | -0.00759 | 128.74940 | 4.093 | 0.001623 | 513.9 |
| 131,072 | 93.329 | -0.00630 | 172.44475 | 3.962 | 0.000736 | 501.1 |
| 163,840 | 78.749 | -0.00751 | 124.96700 | 3.973 | 0.001484 | 499.6 |
| 196,608 | 106.227 | -0.00588 | 227.13551 | 3.956 | 0.000549 | 510.4 |
| 229,376 | 110.520 | -0.00685 | 328.09067 | 4.003 | 0.000677 | 502.5 |
| 262,144 | 124.627 | -0.00604 | 161.90795 | 3.561 | 0.000675 | 487.7 |
| 294,912 | 132.001 | -0.01056 | 209.71237 | 3.533 | 0.002603 | 487.4 |
| 327,680 | 115.658 | -0.00502 | 345.61491 | 3.856 | 0.000993 | 492.9 |
| 360,448 | 136.570 | -0.00424 | 406.47263 | 3.608 | 0.001453 | 498.6 |
| 393,216 | 154.299 | -0.01231 | 150.43017 | 3.490 | 0.004650 | 502.9 |
| 425,984 | 146.923 | -0.00632 | 198.57380 | 3.505 | 0.001422 | 507.2 |
| 458,752 | 153.310 | -0.01148 | 179.46411 | 3.689 | 0.003045 | 509.5 |
| 491,520 | 157.660 | -0.01278 | 62.00490 | 3.240 | 0.002439 | 514.7 |
| 524,288 | 162.241 | -0.01657 | 85.52630 | 3.342 | 0.004853 | 515.0 |
| 557,056 | 169.906 | -0.01293 | 104.11159 | 3.238 | 0.002481 | 516.9 |
| 589,824 | 162.113 | -0.01280 | 74.71019 | 3.083 | 0.002581 | 521.7 |
| 622,592 | 170.139 | -0.01380 | 94.30180 | 2.946 | 0.003877 | 524.9 |
| 655,360 | 179.192 | -0.01243 | 87.83722 | 2.911 | 0.002900 | 524.8 |
| 688,128 | 181.424 | -0.01227 | 101.23732 | 2.716 | 0.002590 | 525.0 |
| 720,896 | 186.129 | -0.00552 | 57.13185 | 2.334 | 0.003011 | 530.8 |
| 753,664 | 202.473 | -0.01188 | 287.20465 | 2.829 | 0.001827 | 536.1 |
| 786,432 | 203.064 | -0.00927 | 215.82947 | 2.670 | 0.001873 | 537.1 |
| 819,200 | 200.110 | -0.01444 | 77.00087 | 2.512 | 0.004550 | 536.7 |
| 851,968 | 203.409 | -0.01471 | 70.48558 | 2.650 | 0.003100 | 539.8 |
| 884,736 | 206.838 | -0.00722 | 266.98986 | 2.718 | 0.001703 | 544.6 |
| 917,504 | 211.388 | -0.00834 | 251.81054 | 2.741 | 0.001351 | 548.8 |
| 950,272 | 224.273 | -0.00924 | 157.88536 | 2.503 | 0.002768 | 553.5 |
| 983,040 | 221.235 | -0.00698 | 171.27499 | 2.726 | 0.001770 | 557.5 |
| 1,000,000 | 208.990 | -0.01463 | 255.10556 | 2.649 | 0.000833 | 558.6 |

### guide_early_preflight_v1

| Saved step | Rolling reward | Policy loss | Value loss | Entropy | Approx KL | Transitions/s |
|---|---:|---:|---:|---:|---:|---:|
| 0 | — | — | — | — | — | — |
| 8,192 | 66.074 | -0.00710 | 105.23060 | 4.453 | 0.001451 | 1609.5 |

### ppo_scale

| Saved step | Rolling reward | Policy loss | Value loss | Entropy | Approx KL | Transitions/s |
|---|---:|---:|---:|---:|---:|---:|
| 4,096 | — | — | — | — | — | — |
| 36,864 | 1.696 | -0.01182 | 0.02838 | 4.340 | 0.001257 | 614.7 |
| 69,632 | 1.913 | -0.01247 | 0.06075 | 4.319 | 0.001167 | 581.5 |

### ppo_smoke

| Saved step | Rolling reward | Policy loss | Value loss | Entropy | Approx KL | Transitions/s |
|---|---:|---:|---:|---:|---:|---:|
| 0 | — | — | — | — | — | — |
| 4,096 | 3.000 | -0.00364 | 0.02746 | 4.495 | 0.000023 | 364.7 |

### touch_probe

| Saved step | Rolling reward | Policy loss | Value loss | Entropy | Approx KL | Transitions/s |
|---|---:|---:|---:|---:|---:|---:|
| 0 | — | — | — | — | — | — |
| 32,768 | 0.012 | -0.01296 | 0.00070 | 3.897 | 0.002895 | 841.9 |
| 65,536 | 0.000 | -0.01162 | 0.00067 | 3.468 | 0.002083 | 764.4 |

## Appendix B — every ordinary saved evaluation report

All paths below are relative to `training/logs/`. Each filename is retained, including numbered repeat reports. Repeats do not imply additional training or independent seed sets. `legacy` means the report predates the explicit mode field; consult its evaluator/history rather than inventing a missing field. Missing fields are shown as dashes, not zero. Ground percentages in this appendix are the evaluator proxy, not the contact-time callback classification. Ball progress is total episode displacement toward goal, not proof of learner-caused useful contact.

### approach_stage2_v1

| Report filename | Games | Step | Mode | Touch episodes | Mean touches | Ground proxy | First-touch s | Ball progress | W/L/D | No-touch wins |
|---|---:|---:|---|---:|---:|---:|---:|---:|---|---:|
| [eval_approach_seed3000_new_argmax.json](training/logs/approach_stage2_v1/eval_approach_seed3000_new_argmax.json) | 40 | 196608 | argmax | 0.0% | 0.000 | 0.0% | — | 0.0 | 0/0/40 | 0 |
| [eval_approach_seed3000_new_argmax_2.json](training/logs/approach_stage2_v1/eval_approach_seed3000_new_argmax_2.json) | 40 | 196608 | argmax | 0.0% | 0.000 | 0.0% | — | 0.0 | 0/0/40 | 0 |
| [eval_approach_seed3000_new_sample.json](training/logs/approach_stage2_v1/eval_approach_seed3000_new_sample.json) | 40 | 196608 | sample | 10.0% | 0.125 | 0.0% | 5.40 | 162.5 | 0/0/40 | 0 |
| [eval_approach_seed3000_new_sample_2.json](training/logs/approach_stage2_v1/eval_approach_seed3000_new_sample_2.json) | 40 | 196608 | sample | 10.0% | 0.125 | 0.0% | 5.40 | 162.5 | 0/0/40 | 0 |
| [eval_approach_seed4000_new_argmax.json](training/logs/approach_stage2_v1/eval_approach_seed4000_new_argmax.json) | 40 | 196608 | argmax | 0.0% | 0.000 | 0.0% | — | 0.0 | 0/0/40 | 0 |
| [eval_approach_seed4000_new_argmax_2.json](training/logs/approach_stage2_v1/eval_approach_seed4000_new_argmax_2.json) | 40 | 196608 | argmax | 0.0% | 0.000 | 0.0% | — | 0.0 | 0/0/40 | 0 |
| [eval_approach_seed4000_new_sample.json](training/logs/approach_stage2_v1/eval_approach_seed4000_new_sample.json) | 40 | 196608 | sample | 20.0% | 0.350 | 2.5% | 2.83 | 630.2 | 1/0/39 | 0 |
| [eval_approach_seed4000_new_sample_2.json](training/logs/approach_stage2_v1/eval_approach_seed4000_new_sample_2.json) | 40 | 196608 | sample | 20.0% | 0.350 | 2.5% | 2.83 | 630.2 | 1/0/39 | 0 |
| [eval_approach_seed4000_old_argmax.json](training/logs/approach_stage2_v1/eval_approach_seed4000_old_argmax.json) | 40 | 65536 | argmax | 22.5% | 0.250 | 22.5% | 1.12 | 884.8 | 3/0/37 | 0 |
| [eval_approach_seed4000_old_argmax_2.json](training/logs/approach_stage2_v1/eval_approach_seed4000_old_argmax_2.json) | 40 | 65536 | argmax | 22.5% | 0.250 | 22.5% | 1.12 | 884.8 | 3/0/37 | 0 |
| [eval_approach_seed4000_old_sample.json](training/logs/approach_stage2_v1/eval_approach_seed4000_old_sample.json) | 40 | 65536 | sample | 25.0% | 0.300 | 7.5% | 1.77 | 922.6 | 0/0/40 | 0 |
| [eval_approach_seed4000_old_sample_2.json](training/logs/approach_stage2_v1/eval_approach_seed4000_old_sample_2.json) | 40 | 65536 | sample | 25.0% | 0.300 | 7.5% | 1.77 | 922.6 | 0/0/40 | 0 |
| [eval_touch_seed1000_new_argmax.json](training/logs/approach_stage2_v1/eval_touch_seed1000_new_argmax.json) | 40 | 196608 | argmax | 0.0% | 0.000 | 0.0% | — | 0.0 | 0/0/40 | 0 |
| [eval_touch_seed1000_new_argmax_2.json](training/logs/approach_stage2_v1/eval_touch_seed1000_new_argmax_2.json) | 40 | 196608 | argmax | 0.0% | 0.000 | 0.0% | — | 0.0 | 0/0/40 | 0 |
| [eval_touch_seed1000_new_sample.json](training/logs/approach_stage2_v1/eval_touch_seed1000_new_sample.json) | 40 | 196608 | sample | 42.5% | 0.600 | 0.0% | 1.20 | 1244.4 | 0/0/40 | 0 |
| [eval_touch_seed1000_new_sample_2.json](training/logs/approach_stage2_v1/eval_touch_seed1000_new_sample_2.json) | 40 | 196608 | sample | 42.5% | 0.600 | 0.0% | 1.20 | 1244.4 | 0/0/40 | 0 |
| [intermediate_step131072_argmax_seed4000.json](training/logs/approach_stage2_v1/intermediate_step131072_argmax_seed4000.json) | 40 | 131072 | argmax | 0.0% | 0.000 | 0.0% | — | 0.0 | 0/0/40 | 0 |
| [intermediate_step131072_sample_seed4000.json](training/logs/approach_stage2_v1/intermediate_step131072_sample_seed4000.json) | 40 | 131072 | sample | 20.0% | 0.250 | 0.0% | 4.38 | 377.9 | 0/0/40 | 0 |
| [intermediate_step98304_argmax_seed4000.json](training/logs/approach_stage2_v1/intermediate_step98304_argmax_seed4000.json) | 40 | 98304 | argmax | 0.0% | 0.000 | 0.0% | — | 0.0 | 0/0/40 | 0 |
| [intermediate_step98304_sample_seed4000.json](training/logs/approach_stage2_v1/intermediate_step98304_sample_seed4000.json) | 40 | 98304 | sample | 7.5% | 0.100 | 0.0% | 1.89 | 237.8 | 0/0/40 | 0 |

### bridge_positive_reward_v1

| Report filename | Games | Step | Mode | Touch episodes | Mean touches | Ground proxy | First-touch s | Ball progress | W/L/D | No-touch wins |
|---|---:|---:|---|---:|---:|---:|---:|---:|---|---:|
| [eval_bridge_seed5000_argmax.json](training/logs/bridge_positive_reward_v1/eval_bridge_seed5000_argmax.json) | 40 | 98304 | argmax | 35.0% | 0.650 | 0.0% | 1.20 | 1574.9 | 2/0/38 | 0 |
| [eval_bridge_seed5000_sample.json](training/logs/bridge_positive_reward_v1/eval_bridge_seed5000_sample.json) | 40 | 98304 | sample | 42.5% | 0.525 | 0.0% | 1.39 | 1670.3 | 1/0/39 | 0 |
| [eval_bridge_seed6000_argmax.json](training/logs/bridge_positive_reward_v1/eval_bridge_seed6000_argmax.json) | 40 | 98304 | argmax | 22.5% | 0.375 | 0.0% | 1.30 | 968.9 | 2/0/38 | 0 |
| [eval_bridge_seed6000_sample.json](training/logs/bridge_positive_reward_v1/eval_bridge_seed6000_sample.json) | 40 | 98304 | sample | 37.5% | 0.550 | 0.0% | 1.34 | 1410.5 | 2/0/38 | 0 |
| [eval_touch_seed1000_argmax.json](training/logs/bridge_positive_reward_v1/eval_touch_seed1000_argmax.json) | 40 | 98304 | argmax | 47.5% | 1.025 | 0.0% | 0.94 | 1937.3 | 1/0/39 | 0 |
| [eval_touch_seed1000_sample.json](training/logs/bridge_positive_reward_v1/eval_touch_seed1000_sample.json) | 40 | 98304 | sample | 65.0% | 0.750 | 2.5% | 1.05 | 2183.1 | 3/0/37 | 0 |

### bridge_probe

| Report filename | Games | Step | Mode | Touch episodes | Mean touches | Ground proxy | First-touch s | Ball progress | W/L/D | No-touch wins |
|---|---:|---:|---|---:|---:|---:|---:|---:|---|---:|
| [step65536_seed5000_argmax.json](training/logs/bridge_probe/step65536_seed5000_argmax.json) | 40 | 65536 | argmax | 50.0% | 0.675 | 50.0% | 0.86 | 2046.6 | 3/0/37 | 0 |
| [step65536_seed5000_sample.json](training/logs/bridge_probe/step65536_seed5000_sample.json) | 40 | 65536 | sample | 42.5% | 0.650 | 17.5% | 1.24 | 1670.2 | 2/0/38 | 0 |
| [step65536_seed6000_argmax.json](training/logs/bridge_probe/step65536_seed6000_argmax.json) | 40 | 65536 | argmax | 42.5% | 0.550 | 42.5% | 0.89 | 1809.8 | 4/0/36 | 0 |
| [step65536_seed6000_sample.json](training/logs/bridge_probe/step65536_seed6000_sample.json) | 40 | 65536 | sample | 22.5% | 0.275 | 5.0% | 1.33 | 923.1 | 2/0/38 | 0 |

### bridge_stage2_v1

| Report filename | Games | Step | Mode | Touch episodes | Mean touches | Ground proxy | First-touch s | Ball progress | W/L/D | No-touch wins |
|---|---:|---:|---|---:|---:|---:|---:|---:|---|---:|
| [eval_bridge_seed5000_argmax.json](training/logs/bridge_stage2_v1/eval_bridge_seed5000_argmax.json) | 40 | 98304 | argmax | 0.0% | 0.000 | 0.0% | — | 0.0 | 0/0/40 | 0 |
| [eval_bridge_seed5000_sample.json](training/logs/bridge_stage2_v1/eval_bridge_seed5000_sample.json) | 40 | 98304 | sample | 32.5% | 0.375 | 2.5% | 2.25 | 876.6 | 0/0/40 | 0 |
| [eval_bridge_seed6000_argmax.json](training/logs/bridge_stage2_v1/eval_bridge_seed6000_argmax.json) | 40 | 98304 | argmax | 0.0% | 0.000 | 0.0% | — | 0.0 | 0/0/40 | 0 |
| [eval_bridge_seed6000_sample.json](training/logs/bridge_stage2_v1/eval_bridge_seed6000_sample.json) | 40 | 98304 | sample | 27.5% | 0.350 | 2.5% | 1.73 | 783.0 | 0/0/40 | 0 |
| [eval_touch_seed1000_argmax.json](training/logs/bridge_stage2_v1/eval_touch_seed1000_argmax.json) | 40 | 98304 | argmax | 0.0% | 0.000 | 0.0% | — | 0.0 | 0/0/40 | 0 |
| [eval_touch_seed1000_sample.json](training/logs/bridge_stage2_v1/eval_touch_seed1000_sample.json) | 40 | 98304 | sample | 47.5% | 0.600 | 7.5% | 1.27 | 1488.5 | 0/0/40 | 0 |

### early_reward_probe_v1

| Report filename | Games | Step | Mode | Touch episodes | Mean touches | Ground proxy | First-touch s | Ball progress | W/L/D | No-touch wins |
|---|---:|---:|---|---:|---:|---:|---:|---:|---|---:|
| [approach_argmax_seed3000_step0.json](training/logs/early_reward_probe_v1/approach_argmax_seed3000_step0.json) | 40 | 0 | argmax | 10.0% | 0.125 | 2.5% | 1.40 | 376.4 | 1/0/39 | 0 |
| [approach_argmax_seed3000_step65536.json](training/logs/early_reward_probe_v1/approach_argmax_seed3000_step65536.json) | 40 | 65536 | argmax | 17.5% | 0.250 | 17.5% | 1.16 | 710.5 | 3/0/37 | 0 |
| [approach_sample_seed3000_step0.json](training/logs/early_reward_probe_v1/approach_sample_seed3000_step0.json) | 40 | 0 | sample | 15.0% | 0.200 | 5.0% | 3.28 | 571.1 | 1/0/39 | 0 |
| [approach_sample_seed3000_step65536.json](training/logs/early_reward_probe_v1/approach_sample_seed3000_step65536.json) | 40 | 65536 | sample | 22.5% | 0.275 | 12.5% | 1.84 | 939.9 | 3/0/37 | 0 |
| [diag_argmax_seed1000_step0.json](training/logs/early_reward_probe_v1/diag_argmax_seed1000_step0.json) | 40 | 0 | argmax | 0.0% | 0.000 | 0.0% | — | 0.0 | 0/0/40 | 0 |
| [diag_argmax_seed1000_step65536.json](training/logs/early_reward_probe_v1/diag_argmax_seed1000_step65536.json) | 40 | 65536 | argmax | 87.5% | 1.550 | 87.5% | 0.67 | 3939.8 | 3/0/37 | 0 |
| [diag_argmax_seed2000_step0.json](training/logs/early_reward_probe_v1/diag_argmax_seed2000_step0.json) | 40 | 0 | argmax | 0.0% | 0.000 | 0.0% | — | 0.0 | 0/0/40 | 0 |
| [diag_argmax_seed2000_step65536.json](training/logs/early_reward_probe_v1/diag_argmax_seed2000_step65536.json) | 40 | 65536 | argmax | 87.5% | 1.475 | 87.5% | 0.67 | 3901.8 | 2/0/38 | 0 |
| [diag_sample_seed1000_step0.json](training/logs/early_reward_probe_v1/diag_sample_seed1000_step0.json) | 40 | 0 | sample | 30.0% | 0.300 | 7.5% | 1.13 | 1129.6 | 1/0/39 | 0 |
| [diag_sample_seed1000_step65536.json](training/logs/early_reward_probe_v1/diag_sample_seed1000_step65536.json) | 40 | 65536 | sample | 62.5% | 0.650 | 25.0% | 1.00 | 2388.4 | 0/0/40 | 0 |
| [diag_sample_seed2000_step0.json](training/logs/early_reward_probe_v1/diag_sample_seed2000_step0.json) | 40 | 0 | sample | 25.0% | 0.250 | 10.0% | 1.21 | 809.1 | 1/0/39 | 0 |
| [diag_sample_seed2000_step65536.json](training/logs/early_reward_probe_v1/diag_sample_seed2000_step65536.json) | 40 | 65536 | sample | 57.5% | 0.700 | 25.0% | 1.10 | 2043.1 | 1/0/39 | 0 |

### ground_probe_v1

| Report filename | Games | Step | Mode | Touch episodes | Mean touches | Ground proxy | First-touch s | Ball progress | W/L/D | No-touch wins |
|---|---:|---:|---|---:|---:|---:|---:|---:|---|---:|
| [after_argmax.json](training/logs/ground_probe_v1/after_argmax.json) | 40 | 32768 | argmax | 0.0% | 0.000 | 0.0% | — | 0.0 | 0/0/40 | 0 |
| [after_sample.json](training/logs/ground_probe_v1/after_sample.json) | 40 | 32768 | sample | 12.5% | 0.125 | 12.5% | 1.09 | 263.3 | 0/0/40 | 0 |
| [before_argmax.json](training/logs/ground_probe_v1/before_argmax.json) | 40 | 0 | argmax | 0.0% | 0.000 | 0.0% | — | 0.0 | 0/0/40 | 0 |
| [before_sample.json](training/logs/ground_probe_v1/before_sample.json) | 40 | 0 | sample | 52.5% | 0.650 | 52.5% | 1.66 | 664.6 | 0/0/40 | 0 |

### guide_early_1m_v1

| Report filename | Games | Step | Mode | Touch episodes | Mean touches | Ground proxy | First-touch s | Ball progress | W/L/D | No-touch wins |
|---|---:|---:|---|---:|---:|---:|---:|---:|---|---:|
| [eval_approach_seed9000_argmax_step1000000.json](training/logs/guide_early_1m_v1/eval_approach_seed9000_argmax_step1000000.json) | 40 | 1000000 | argmax | 10.0% | 0.100 | 0.0% | 3.22 | 195.8 | 0/0/40 | 0 |
| [eval_approach_seed9000_sample_step1000000.json](training/logs/guide_early_1m_v1/eval_approach_seed9000_sample_step1000000.json) | 40 | 1000000 | sample | 35.0% | 0.525 | 0.0% | 3.58 | 853.3 | 0/0/40 | 0 |
| [eval_bridge_seed7000_argmax_step0.json](training/logs/guide_early_1m_v1/eval_bridge_seed7000_argmax_step0.json) | 40 | 0 | argmax | 5.0% | 0.050 | 0.0% | 1.00 | 234.6 | 0/0/40 | 0 |
| [eval_bridge_seed7000_argmax_step1000000.json](training/logs/guide_early_1m_v1/eval_bridge_seed7000_argmax_step1000000.json) | 40 | 1000000 | argmax | 65.0% | 0.825 | 2.5% | 1.86 | 1131.4 | 0/0/40 | 0 |
| [eval_bridge_seed7000_sample_step0.json](training/logs/guide_early_1m_v1/eval_bridge_seed7000_sample_step0.json) | 40 | 0 | sample | 17.5% | 0.175 | 2.5% | 2.48 | 595.6 | 1/0/39 | 0 |
| [eval_bridge_seed7000_sample_step1000000.json](training/logs/guide_early_1m_v1/eval_bridge_seed7000_sample_step1000000.json) | 40 | 1000000 | sample | 47.5% | 0.750 | 2.5% | 1.74 | 1332.4 | 0/0/40 | 0 |
| [eval_bridge_seed8000_argmax_step0.json](training/logs/guide_early_1m_v1/eval_bridge_seed8000_argmax_step0.json) | 40 | 0 | argmax | 2.5% | 0.025 | 0.0% | 1.07 | 113.4 | 0/0/40 | 0 |
| [eval_bridge_seed8000_argmax_step1000000.json](training/logs/guide_early_1m_v1/eval_bridge_seed8000_argmax_step1000000.json) | 40 | 1000000 | argmax | 65.0% | 0.875 | 2.5% | 1.94 | 903.3 | 0/0/40 | 0 |
| [eval_bridge_seed8000_sample_step0.json](training/logs/guide_early_1m_v1/eval_bridge_seed8000_sample_step0.json) | 40 | 0 | sample | 20.0% | 0.200 | 2.5% | 1.49 | 792.5 | 0/0/40 | 0 |
| [eval_bridge_seed8000_sample_step1000000.json](training/logs/guide_early_1m_v1/eval_bridge_seed8000_sample_step1000000.json) | 40 | 1000000 | sample | 62.5% | 0.900 | 0.0% | 2.73 | 1175.5 | 0/0/40 | 0 |
| [eval_kickoff_v2_seed9000_argmax_step1000000.json](training/logs/guide_early_1m_v1/eval_kickoff_v2_seed9000_argmax_step1000000.json) | 20 | 1000000 | argmax | 0.0% | 0.000 | 0.0% | — | -5291.4 | 0/20/0 | 0 |
| [eval_touch_seed3000_argmax_step0.json](training/logs/guide_early_1m_v1/eval_touch_seed3000_argmax_step0.json) | 40 | 0 | argmax | 0.0% | 0.000 | 0.0% | — | 0.0 | 0/0/40 | 0 |
| [eval_touch_seed3000_argmax_step1000000.json](training/logs/guide_early_1m_v1/eval_touch_seed3000_argmax_step1000000.json) | 40 | 1000000 | argmax | 37.5% | 0.375 | 0.0% | 0.87 | 922.8 | 0/0/40 | 0 |
| [eval_touch_seed3000_sample_step0.json](training/logs/guide_early_1m_v1/eval_touch_seed3000_sample_step0.json) | 40 | 0 | sample | 27.5% | 0.300 | 5.0% | 1.15 | 969.7 | 0/0/40 | 0 |
| [eval_touch_seed3000_sample_step1000000.json](training/logs/guide_early_1m_v1/eval_touch_seed3000_sample_step1000000.json) | 40 | 1000000 | sample | 55.0% | 0.875 | 2.5% | 1.52 | 1457.4 | 0/0/40 | 0 |

### ppo_scale

| Report filename | Games | Step | Mode | Touch episodes | Mean touches | Ground proxy | First-touch s | Ball progress | W/L/D | No-touch wins |
|---|---:|---:|---|---:|---:|---:|---:|---:|---|---:|
| [evaluation.json](training/logs/ppo_scale/evaluation.json) | 20 | 69632 | legacy | — | 0.000 | — | — | — | 5/0/15 | 5 |

### ppo_smoke

| Report filename | Games | Step | Mode | Touch episodes | Mean touches | Ground proxy | First-touch s | Ball progress | W/L/D | No-touch wins |
|---|---:|---:|---|---:|---:|---:|---:|---:|---|---:|
| [evaluation.json](training/logs/ppo_smoke/evaluation.json) | 10 | 4096 | legacy | — | 0.000 | — | — | — | 4/0/6 | — |
| [random_evaluation.json](training/logs/ppo_smoke/random_evaluation.json) | 20 | — | legacy | — | 0.100 | — | — | — | 8/0/12 | — |
| [untrained_evaluation.json](training/logs/ppo_smoke/untrained_evaluation.json) | 20 | 0 | legacy | — | 0.100 | — | — | — | 7/0/13 | — |

### touch_probe

| Report filename | Games | Step | Mode | Touch episodes | Mean touches | Ground proxy | First-touch s | Ball progress | W/L/D | No-touch wins |
|---|---:|---:|---|---:|---:|---:|---:|---:|---|---:|
| [after.json](training/logs/touch_probe/after.json) | 40 | 65536 | legacy | — | 1.150 | — | — | — | 0/0/40 | 0 |
| [before.json](training/logs/touch_probe/before.json) | 40 | 0 | legacy | — | 0.000 | — | — | — | 0/0/40 | 0 |
| [diag_argmax_seed1000.json](training/logs/touch_probe/diag_argmax_seed1000.json) | 40 | 65536 | argmax | 95.0% | 1.150 | 0.0% | 1.79 | 844.0 | 0/0/40 | 0 |
| [diag_argmax_seed2000.json](training/logs/touch_probe/diag_argmax_seed2000.json) | 40 | 65536 | argmax | 97.5% | 1.275 | 0.0% | 1.79 | 899.1 | 0/0/40 | 0 |
| [diag_sample_seed1000.json](training/logs/touch_probe/diag_sample_seed1000.json) | 40 | 65536 | sample | 0.0% | 0.000 | 0.0% | — | 0.0 | 0/0/40 | 0 |
| [diag_sample_seed2000.json](training/logs/touch_probe/diag_sample_seed2000.json) | 40 | 65536 | sample | 0.0% | 0.000 | 0.0% | — | 0.0 | 0/0/40 | 0 |
| [holdout_after.json](training/logs/touch_probe/holdout_after.json) | 40 | 65536 | legacy | — | 1.275 | — | — | — | 0/0/40 | 0 |
| [holdout_before.json](training/logs/touch_probe/holdout_before.json) | 40 | 0 | legacy | — | 0.000 | — | — | — | 0/0/40 | 0 |
| [kickoff_transfer.json](training/logs/touch_probe/kickoff_transfer.json) | 20 | 65536 | legacy | — | 0.100 | — | — | — | 9/0/11 | 9 |
| [v2_after.json](training/logs/touch_probe/v2_after.json) | 20 | 65536 | legacy | — | 0.000 | — | — | — | 0/20/0 | 0 |
| [v2_before.json](training/logs/touch_probe/v2_before.json) | 20 | 0 | legacy | — | 0.000 | — | — | — | 0/20/0 | 0 |

## Appendix C — all sixteen contact-time classification reports

Source directory: `training/logs/guide_early_1m_v1/contact_diagnostics/`. Forty episodes per report. Ground/air episode sets may overlap: an episode can contain both. Air-contact fraction is based on raw contact callbacks, whereas mean touches follows the report's touch-step counting; do not treat those denominators as identical. First-touch time is conditional on an episode touching. These percentages intentionally differ from the more conservative boundary-based proxy in Appendix B.

| Report | Touch episodes | Mean touches | Ground episodes | Air episodes | Air contact fraction | Jump | Throttle | Boost | First-touch s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [contact_approach_seed3000_argmax_step0.json](training/logs/guide_early_1m_v1/contact_diagnostics/contact_approach_seed3000_argmax_step0.json) | 10.0% | 0.125 | 2.5% | 7.5% | 75.0% | 57.0% | 63.7% | 62.9% | 1.40 |
| [contact_approach_seed3000_argmax_step1000000.json](training/logs/guide_early_1m_v1/contact_diagnostics/contact_approach_seed3000_argmax_step1000000.json) | 25.0% | 0.275 | 7.5% | 20.0% | 72.0% | 29.3% | 66.5% | 52.3% | 3.59 |
| [contact_approach_seed3000_sample_step0.json](training/logs/guide_early_1m_v1/contact_diagnostics/contact_approach_seed3000_sample_step0.json) | 15.0% | 0.200 | 5.0% | 10.0% | 85.2% | 20.2% | 49.4% | 42.8% | 3.28 |
| [contact_approach_seed3000_sample_step1000000.json](training/logs/guide_early_1m_v1/contact_diagnostics/contact_approach_seed3000_sample_step1000000.json) | 35.0% | 0.500 | 5.0% | 30.0% | 89.2% | 24.4% | 56.8% | 45.9% | 3.97 |
| [contact_bridge_seed7000_argmax_step0.json](training/logs/guide_early_1m_v1/contact_diagnostics/contact_bridge_seed7000_argmax_step0.json) | 5.0% | 0.050 | 0.0% | 5.0% | 100.0% | 62.1% | 63.4% | 63.1% | 1.00 |
| [contact_bridge_seed7000_argmax_step1000000.json](training/logs/guide_early_1m_v1/contact_diagnostics/contact_bridge_seed7000_argmax_step1000000.json) | 65.0% | 0.825 | 10.0% | 62.5% | 85.5% | 26.1% | 72.2% | 59.4% | 1.86 |
| [contact_bridge_seed7000_sample_step0.json](training/logs/guide_early_1m_v1/contact_diagnostics/contact_bridge_seed7000_sample_step0.json) | 17.5% | 0.175 | 2.5% | 15.0% | 84.6% | 20.2% | 49.3% | 42.7% | 2.48 |
| [contact_bridge_seed7000_sample_step1000000.json](training/logs/guide_early_1m_v1/contact_diagnostics/contact_bridge_seed7000_sample_step1000000.json) | 47.5% | 0.750 | 10.0% | 40.0% | 83.7% | 22.7% | 57.8% | 47.2% | 1.74 |
| [contact_bridge_seed8000_argmax_step0.json](training/logs/guide_early_1m_v1/contact_diagnostics/contact_bridge_seed8000_argmax_step0.json) | 2.5% | 0.025 | 0.0% | 2.5% | 100.0% | 56.0% | 60.2% | 59.9% | 1.07 |
| [contact_bridge_seed8000_argmax_step1000000.json](training/logs/guide_early_1m_v1/contact_diagnostics/contact_bridge_seed8000_argmax_step1000000.json) | 65.0% | 0.875 | 12.5% | 60.0% | 83.1% | 27.2% | 69.3% | 56.1% | 1.94 |
| [contact_bridge_seed8000_sample_step0.json](training/logs/guide_early_1m_v1/contact_diagnostics/contact_bridge_seed8000_sample_step0.json) | 20.0% | 0.200 | 5.0% | 15.0% | 75.0% | 20.3% | 49.4% | 42.8% | 1.49 |
| [contact_bridge_seed8000_sample_step1000000.json](training/logs/guide_early_1m_v1/contact_diagnostics/contact_bridge_seed8000_sample_step1000000.json) | 62.5% | 0.900 | 12.5% | 55.0% | 85.2% | 22.8% | 58.8% | 48.2% | 2.73 |
| [contact_touch_seed3000_argmax_step0.json](training/logs/guide_early_1m_v1/contact_diagnostics/contact_touch_seed3000_argmax_step0.json) | 0.0% | 0.000 | 0.0% | 0.0% | n/a | 63.7% | 68.5% | 68.3% | — |
| [contact_touch_seed3000_argmax_step1000000.json](training/logs/guide_early_1m_v1/contact_diagnostics/contact_touch_seed3000_argmax_step1000000.json) | 37.5% | 0.375 | 0.0% | 37.5% | 100.0% | 26.1% | 67.0% | 55.0% | 0.87 |
| [contact_touch_seed3000_sample_step0.json](training/logs/guide_early_1m_v1/contact_diagnostics/contact_touch_seed3000_sample_step0.json) | 27.5% | 0.300 | 10.0% | 17.5% | 69.2% | 19.8% | 49.3% | 42.5% | 1.15 |
| [contact_touch_seed3000_sample_step1000000.json](training/logs/guide_early_1m_v1/contact_diagnostics/contact_touch_seed3000_sample_step1000000.json) | 55.0% | 0.875 | 5.0% | 52.5% | 93.7% | 22.2% | 57.3% | 47.3% | 1.52 |

## Appendix D — other saved diagnostic families

| Artifact family | Existing files / scope | What it establishes |
|---|---|---|
| Tracer smoke | `diagnostic_smoke_step65536.json`, `_v2.json`; `diagnostics/raw_reward_smoke_3steps.json` | Short schema/runtime checks, not new trained behavior |
| Per-step rewards | `diagnostics/reference_bridge_seed5000.json`, `positive_bridge_seed5000.json` | Recorded weighted terms, physical state and controls |
| Counterfactual reward | `diagnostics/approach_counterfactual_seed5000.json` | Alternative scoring of fixed trajectories only |
| Raw scales | Four `reward_scales/{bridge_seed7000,touch_seed3000}_{argmax,sample}.json` reports | Distribution-dependent term magnitudes |
| Pipeline profile | `profiling/bridge_4env_1024transitions.json` | One temporary rollout/update timing breakdown |
| Live feature mapping | `python-example/logs/observation_diagnostics/feature_mapping_001.json` through `_004.json` | Indexed feature contract and known reconstructions |
| Live action mapping | `action_mapping_001.json` through `_004.json` in the same directory | Ninety numeric table/controller rows verified |
| Controlled adapter cases | `controlled_cases_001.json` through `_003.json` | Synthetic directional cases; lateral labels require the documented correction |
| Team mirror | `team_mirror_001.json` | Synthetic parity, not comprehensive live equivalence |
| Live streams | `live_observations_001.jsonl` through `_007.jsonl` | Packet/state/observation/action evidence; 006/007 are rotated segments of one continuous session |
| Policy sensitivity | `policy_sensitivity/ball_sensitivity_001.json` | 126 fixed-state ball-position interventions, checkpoint unchanged |
| V2 review | `training/reports/v2_bootstrap_review_001.md` | Research/design recommendation, not executed V2 learning |
| Teacher specification | `TEACHER_TASK.md` | Pending validation scope, not teacher results |

## Final handoff

We have a working end-to-end technical ML pipeline and credible evidence of narrow learning, but not an accepted functional competition starter. The current trajectory is the 2v2 teacher → online-distilled smaller student plan in `MODEL_WARS_DECISION_LOG.md`; historical scripted-teacher/BC proposals are not current instructions. The 4 October CanoPy investigation found that the published actor requires 92D inputs and does not establish the expected 2v2 observation contract. Resolve that checkpoint/source mismatch or verify another suitable teacher before implementation or training. Current checkpoints/configurations are preserved. No BC or new PPO training should be implied by this document. Later updates should add new run IDs, artifact paths, before/after behavior and limitations instead of rewriting earlier negative results.

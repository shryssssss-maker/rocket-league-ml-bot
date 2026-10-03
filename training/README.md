# Model Wars: first PPO learning pipeline

**Current experimental direction:** See `CURRICULUM.md`. Return to the full
90-action space and test dense early rewards with no goal reward before long
training. Older "next run" commands lower in this README are historical
diagnostics, not current recommendations. The evaluation script now uses each
checkpoint's recorded reward weights; compare behavioral metrics, not raw
reward, across different reward configurations.

The first full-action early-reward probe improved close-ball contact in both
argmax and sampled evaluation on seeds 1000 and 2000. Stage-two `approach`
fixtures now vary distance, lateral position, and heading; evaluate transfer
before training on that scenario. See `CURRICULUM.md` for measured results.

Later continuations on both `approach` and the gentler `bridge` fixture lost
the useful argmax contact behavior. `--approach-positive-only` is an optional
reward ablation for fresh runs resumed from the earlier step-65,536 checkpoint;
it removes negative retreat reward without altering existing run defaults.

## Frozen reference and per-step diagnostic

Keep `checkpoints/early_reward_probe_v1/baseline_step_65536.pt` as the current
reference. Neither step-98,304 bridge continuation is a replacement. Do not
promote a `baseline.pt` from these runs.

Run the evaluation-only tracer from the project root in PowerShell (one 10-second
episode; it does not train or modify the checkpoint):

```powershell
& '.\training\venv\Scripts\python.exe' '.\training\diagnose_steps.py' --model '.\training\checkpoints\early_reward_probe_v1\baseline_step_65536.pt' --scenario bridge --seconds 10 --episodes 1 --seed 5000 --mode argmax --output '.\training\logs\diagnostics\reference_bridge_seed5000.json'
```

The JSON contains one record per 8-tick action interval: weighted `touch`,
`approach`, `face`, `air`, `goal`, and `total` rewards; the exact eight controls;
and speed, radial velocity toward the ball, distance, interval progress, and
contact/ground flags. Its summary counts positive approach rewards earned while
neither throttle nor boost was commanded. This is a **no-propulsion-command
proxy**, not proof of passive coasting: momentum, gravity, collisions, and
steering can also change the car's motion. Reward terms use the checkpoint's
original weights and installed RLGym reward-function call order. Existing
reports are never overwritten; choose a new `--output` for each comparison.

Do not change the approach reward until these traces identify which behavior
earns it. A distance-before minus distance-after reward is a candidate for a
separate, controlled experiment, not part of this diagnostic.

Compare two saved traces offline, without running RocketSim or training:

```powershell
& '.\training\venv\Scripts\python.exe' '.\training\compare_approach_rewards.py' --reference '.\training\logs\diagnostics\reference_bridge_seed5000.json' --candidate '.\training\logs\diagnostics\positive_bridge_seed5000.json' --output '.\training\logs\diagnostics\approach_counterfactual_seed5000.json'
```

The comparison scores each existing 8-tick row with (1) the original recorded
approach reward, (2) pre-touch-gated positive original reward, (3) gated
positive distance change, and (4) gated positive improvement in closing
velocity. Candidate values use the checkpoint's 0.02 approach weight and are
normalized/clipped to `[0, 1]`; reward scales are not automatically equivalent.
The contact interval is excluded because one 8-tick row cannot be split at
the exact touch tick. These are **counterfactual scores on fixed trajectories**,
not evidence of how a retrained policy would behave.

### Controlled first-touch gate branch

**Paused, not a current training recommendation.** The flag remains available
for a later ablation, but first measure conventional early-reward scales and
training throughput. No first-touch-gated training has been run.

`--approach-until-first-touch` keeps the **signed** approach formula and all
other reward weights unchanged. Only the approach term becomes zero on the
learner's first touch interval and every later interval in that episode. It
does **not** terminate the episode, suppress TouchReward, or alter the action
and observation contracts. The gate resets each episode and is tracked per
agent. Do not combine this flag with `--approach-positive-only`.

The following branch is designed to compare with the preserved signed
`bridge_stage2_v1` continuation: both resume the untouched step-65,536
checkpoint, use the same bridge fixture, 10-second episodes, seed 42, four
environments, and 32,768 additional transitions. The only intended difference
is the first-touch gate. Run training yourself; this command is **not** run by
the coding agent, and is retained below only as a historical proposed command:

```powershell
& '.\training\venv\Scripts\python.exe' '.\training\train.py' --resume '.\training\checkpoints\early_reward_probe_v1\baseline_step_65536.pt' --scenario bridge --episode-seconds 10 --steps 32768 --num-envs 4 --approach-until-first-touch --run-name bridge_first_touch_gate_v1
```

Do not infer improvement from training reward, which now has different episode
accounting. Evaluate the resulting step-98,304 checkpoint against the signed
control and the frozen reference on identical bridge and near-ball seeds,
using both argmax and sampled actions.

### Raw early-reward scale measurement (before any new training)

The installed `rlgym-rocket-league==2.0.1` reward package contains
`TouchReward`, `GoalReward`, and `CombinedReward`; it does **not** ship a
velocity-to-ball reward. `rewards/speed_toward_ball.py` mirrors the
`SpeedTowardBallReward` example in the [current RLGym training tutorial](https://rlgym.org/Rocket%20League/training_an_agent/),
with only a zero-distance safety guard. It is **not connected to PPO** yet.

`diagnose_steps.py` now reports unweighted raw `touch`, `velocity_to_ball`,
`face`, and `air` values at each simulated action interval and their mean,
95th percentile, maximum, positive rate, and conditional positive mean.
The existing weighted reward trace remains unchanged. Sample both argmax and
stochastic action modes because their state distributions differ:

```powershell
foreach ($mode in @('argmax', 'sample')) {
    & '.\training\venv\Scripts\python.exe' '.\training\diagnose_steps.py' --model '.\training\checkpoints\early_reward_probe_v1\baseline_step_65536.pt' --scenario bridge --seconds 10 --episodes 40 --seed 7000 --mode $mode --action-seed 3000 --output ".\training\logs\reward_scales\bridge_seed7000_${mode}.json"
    & '.\training\venv\Scripts\python.exe' '.\training\diagnose_steps.py' --model '.\training\checkpoints\early_reward_probe_v1\baseline_step_65536.pt' --scenario touch --seconds 8 --episodes 40 --seed 3000 --mode $mode --action-seed 3000 --output ".\training\logs\reward_scales\touch_seed3000_${mode}.json"
}
```

These are simulator sampling runs, not training. They may take several minutes;
run them yourself. Reports refuse to overwrite existing files. Inspect the
`raw_reward_stats` tables before choosing any new weights. In particular,
TouchReward is sparse, so its 95th percentile may be zero even if its maximum
and positive-event rate are meaningful.

### Separate single-process throughput profile

`profile_pipeline.py` measures one disposable 1,024-transition rollout and
one in-memory PPO update using the **existing** reward and single-process
architecture. It times observation batching, policy inference/host transfer,
simulator stepping, rollout bookkeeping, bootstrap/GAE, final batching, and
PPO update. It does not save model weights or launch a long training run, but
it **does** perform one optimizer update on the temporary in-memory model.
Run this yourself when ready:

```powershell
& '.\training\venv\Scripts\python.exe' '.\training\profile_pipeline.py' --model '.\training\checkpoints\early_reward_probe_v1\baseline_step_65536.pt' --scenario bridge --episode-seconds 10 --num-envs 4 --rollout-steps 256 --output '.\training\logs\profiling\bridge_4env_1024transitions.json'
```

This is a baseline timing measurement, not evidence of learning or a test of
multiprocessing. The profiler does not alter `train.py` or the frozen checkpoint.

### Conventional early-contact reward experiment

`--reward-recipe guide_early_v1` selects a separate fixed reward:

| Term | Weight |
| --- | ---: |
| TouchReward | 50 |
| SpeedTowardBallReward | 5 |
| GuideFaceBallReward | 1 |
| InAirReward | 0.15 |
| GoalReward | 0 (absent) |

This ports the [ZealanL early-stage recipe](https://github.com/ZealanL/RLGym-PPO-Guide/blob/main/making_a_good_bot.md)
to the installed RLGym 2.x API. Velocity uses the positive-only tutorial
formula; `GuideFaceBallReward` is orientation-only and is **separate from** the
older speed-weighted `FacingBallReward`. The earlier raw face statistics were
for that older term, so they are not measurements of this new face definition.
The recipe does not use custom approach progress, closing-velocity change,
first-touch gating, positive-only approach, or scoring reward. It leaves the
92D observation, 90 actions, and 128-unit MLP unchanged. The existing reward
remains the default for old checkpoints and scripts.

Before training, inspect the resolved fresh-run configuration without writing:

```powershell
& '.\training\venv\Scripts\python.exe' '.\training\train.py' --reward-recipe guide_early_v1 --scenario bridge --episode-seconds 10 --steps 1048576 --num-envs 4 --run-name guide_early_v1 --dry-run
```

Do not start the million-step run until the separate throughput profile has
been measured. Any actual training command must be run by the user. Because
these weights are much larger than the previous recipe and this PPO code does
not normalize rewards automatically, a short preflight and finite-loss check
are prudent before committing to a long run.

This directory owns the training environment. The working live bot and its
environment remain in `../python-example/`. Do not install these dependencies
into the live bot's venv.

## Run the smoke test

From `C:\Users\shreyas\Desktop\model wars` in PowerShell:

```powershell
& '.\training\venv\Scripts\python.exe' '.\training\smoke_test.py'
```

No activation, Rocket League launch, RLBot server, GPU, or rendering is required.
The script runs two seeded 1v1 episodes, checks finite observations and rewards,
checks advancing physics ticks and car movement, prints diagnostic output, and
closes the environment even if an assertion fails. A successful run exits with
code 0 and prints `PASS`.

## Verified stack

Verified on Windows with Python 3.12.10:

| Package | Version |
| --- | --- |
| rlgym | 2.0.1 |
| rlgym-api | 2.0.0 |
| rlgym-rocket-league | 2.0.1 |
| rocketsim | 2.2.1 |
| numpy | 1.26.4 |
| cmeel | 0.61.0 |
| torch | 2.11.0+cu128 |

Runtime dependencies, including transitives, are pinned in `requirements.txt`.
To reproduce in a fresh directory with Python 3.12 installed:

```powershell
py -3.12 -m venv training\venv
& '.\training\venv\Scripts\python.exe' -m pip install -r '.\training\requirements.txt'
& '.\training\venv\Scripts\python.exe' '.\training\smoke_test.py'
```

Known upstream packaging issue: the RocketSim 2.2.1 Windows download is named
`rocketsim-2.2.1-0-cp36-abi3-win_amd64.whl`, but its internal `WHEEL` metadata
contains `Tag: cp311-cp311-win_amd64`. Consequently `pip check` reports
`rocketsim 2.2.1 is not supported on this platform` on Python 3.12.10.
Importing RocketSim, creating an arena, and both full smoke-test episodes pass.
The metadata has not been patched locally; this discrepancy remains an upstream
reproducibility caveat.

## Environment behavior

The implementation follows the [official RLGym quickstart](https://rlgym.org/Getting%20Started/quickstart/)
and the actual installed source, using the raw RLGym v2 API:

- `reset()` returns observations keyed by agent ID.
- `step(actions)` returns `(observations, rewards, terminated, truncated)`,
  each keyed by agent ID; there is no fifth `info` return value.
- `DefaultObs(zero_padding=None)` produces NumPy arrays of shape `(92,)` and
  dtype `float64` for this fixed 1v1 setup.
- `LookupTableAction` offers 90 discrete choices. Each input is a one-element
  integer array; `RepeatAction` repeats the decoded eight controls for 8 physics
  ticks. The simulator runs at 120 ticks per simulated second.
- Decoded control order is throttle, steer, pitch, yaw, roll, jump, boost,
  handbrake. This order will matter for future live inference.
- Rewards are `10 * GoalReward + 0.1 * TouchReward`. Zero rewards are normal
  during this short random-action test.
- A goal sets `terminated`; the two-second time limit sets `truncated`.
  Both verified episodes completed via timeout at 30 steps / 240 ticks.
- A second episode verifies reset works after completion. Seeds cover kickoff
  selection and actions; this does not promise identical physics across platforms.

## Reward validation

```powershell
& '.\training\venv\Scripts\python.exe' '.\training\reward_test.py'
```

This test uses real simulator transitions, without overwriting reward event flags.
A moving blue car collides with the ball and earns +0.1; orange earns 0. Controlled
shots into both goals verify +10 for the scoring team, -10 for the conceding team,
correct agent IDs, and goal termination. An idle kickoff verifies timeout truncation.
All assertions passed on the installed stack.

## Hardware and CUDA validation

Hardware: Lenovo LOQ, Intel i7 HX, RTX 4050 Laptop GPU, 24 GB RAM, 512 GB SSD.
`nvidia-smi` reports driver 581.86, CUDA driver support 13.0, and **6141 MiB VRAM**
(approximately 6 GB rather than the initially stated 8 GB). Training must fit
this measured capacity. Driver CUDA support is distinct from the CUDA runtime
bundled with PyTorch.

```powershell
& '.\training\venv\Scripts\python.exe' '.\training\gpu_test.py'
```

The GPU test checks CUDA availability, the device name, and a real matrix
multiplication against its CPU result. It raises an error if CUDA is unavailable.

Verified: PyTorch 2.11.0+cu128, CUDA available, RTX 4050 Laptop GPU, 6.0 GiB
VRAM, and matrix multiplication matches the CPU result. Only torch and its
required dependencies were installed, using the official CUDA 12.8 wheel index;
no separate CUDA toolkit, torchvision, or torchaudio was installed.

## PPO baseline

`config.py` contains the experiment defaults. `policies/` contains separate
128-by-128 tanh actor and critic networks (68,571 parameters total). Simulation
runs on CPU; batched sampling and PPO updates run on CUDA. The learner alternates
blue/orange sides and plays a fixed scripted ball chaser. Default training
episodes end on a goal or after 30 simulated seconds.

`observations/` owns the original DefaultObs coefficients and only casts the
92-dimensional result to float32. `environment.py` owns the 90-action table,
8-tick action repetition, and opponent. The default reward remains
`10 * GoalReward + 0.1 * TouchReward`; optional, independently weighted
approach, moving-face-ball, and air terms are available for new experiments.

`ppo.py` implements PPO-Clip, GAE, normalized advantages, entropy regularization,
gradient clipping, and an approximate-KL stopping threshold. Goal termination
zeros bootstrap value; a timeout bootstraps the final pre-reset observation.
GAE stops across either type of reset. Three analytical regression tests in
`test_ppo.py` passed for goals, timeouts, and rollout boundaries.

### Training commands

Run from the project root. Choose a new run name each time; existing experiment
directories are protected from overwrite.

```powershell
& '.\training\venv\Scripts\python.exe' '.\training\train.py' --steps 4096 --num-envs 1 --run-name my_smoke
```

`--steps` counts learner transitions, each representing eight physics ticks.
With multiple environments, the count rounds up to a multiple of `--num-envs`.
The current implementation batches independent simulator instances in one
process; it does **not** yet run CPU worker processes. Measure the current
learner before adding process parallelism.

Resume weights and Adam state with:

```powershell
& '.\training\venv\Scripts\python.exe' '.\training\train.py' --resume '.\training\checkpoints\ppo_scale\baseline_step_69632.pt' --steps 65536 --num-envs 4 --run-name next_run
```

Resume steps are additional to the checkpoint count. Resume starts fresh
simulator episodes and seeded RNG streams; it is not a bit-exact continuation
of simulator or random-generator state.

Each run writes `logs/<run>/config.json` and `metrics.csv`. CSV columns include
training step, last completed episode reward, rolling mean reward and length
(up to 100 completed episodes), completed episodes and touches in the rollout,
policy/value loss, entropy, approximate KL, checkpoint count, throughput, and
peak PyTorch allocated GPU memory. Empty episode fields mean no episode finished
yet. Peak allocated memory excludes the CUDA context and other applications.

Checkpoints are stored in `checkpoints/<run>/baseline_step_<count>.pt` at the
start, periodically, and after the final update. They contain model/optimizer
state, configuration, versions, and an explicit observation/action contract.
Loading uses `weights_only=True` and checks the contract and simulator versions.
The contract is the future live-deployment specification; a live RLBot adapter
has not been built or validated yet.

### Evaluation commands

```powershell
& '.\training\venv\Scripts\python.exe' '.\training\evaluate.py' --model '.\training\checkpoints\ppo_scale\baseline_step_69632.pt' --games 20 --output '.\training\logs\ppo_scale\evaluation.json'
```

Evaluation uses CPU inference, argmax actions, fixed kickoff seeds, alternating
sides, and the same scripted opponent. Each game is **sudden death**, ending on
one goal or a 30-second draw, rather than a full five-minute Rocket League match.
Reports include games, wins/losses/draws, goals, win rate, mean episode reward,
mean touch events, wins without any learner touch, and an action histogram in
the JSON report. Fixed seeds make same-machine comparisons repeatable; physics
results are not guaranteed bit-identical across platforms. The scripted opponent
can score own goals, so win rate alone is weak evidence of skill.

For a random learner against the same opponent:

```powershell
& '.\training\venv\Scripts\python.exe' '.\training\evaluate.py' --random --games 20
```

For an untrained-policy comparison use the initial checkpoint
`checkpoints/ppo_smoke/baseline_step_0.pt` with the same games, seed, and timeout.
Use another `--seed` (for example 2000) for a holdout comparison. When an output
file exists, evaluation chooses a numbered filename before simulation (for
example `evaluation_2.json`), preserving the earlier report.

### Results achieved so far

- CUDA training smoke: 4,096 transitions, finite losses, saved and loaded
  checkpoint. Actor weight change L2 from initialization: approximately 0.973.
- Larger resumed experiment: 65,536 additional transitions, finishing at 69,632.
  Final rolling reward: 1.913; policy loss: -0.0125; value loss: 0.0608;
  entropy: 4.319 (near-uniform initialization was approximately 4.500).
  Overall throughput: approximately 582 learner transitions/second.
- First 4,096-step checkpoint, 10 evaluation games: 4 wins, 0 losses, 6 draws,
  **zero learner touches**. These wins do not establish learned ball control.
- Untrained policy, 20 games: 7 wins, 0 losses, 13 draws; mean touches 0.1.
- Random learner, 20 games: 8 wins, 0 losses, 12 draws; mean touches 0.1.
- Final 69,632-step checkpoint, 20 games at seed 1000: 5 wins, 0 losses,
  15 draws; mean reward 2.5, mean learner touches 0.0. All five wins occurred
  without a learner touch. This result does not establish improvement over the
  untrained policy or random learner. The saved report already contains this
  result; a repeated run previously failed only at the report-saving stage.

Long-running downloads, training, and evaluations are now run by the user in
their own terminal. Controlled touch learning and a second-seed comparison now
pass. Next: measure transfer to standard kickoffs before more training or
parallel CPU workers. No `baseline.pt` has been promoted as the
organizer baseline; no live neural bot or event infrastructure has been added.

## Controlled touch-learning probe

Sparse kickoff rewards and opponent own goals make the first experiment poor
evidence of useful learning. An optional `touch` scenario places the learner
550 units behind the ball with a seeded lateral offset; the other car waits far
away. Learner sides still alternate. This is a controlled 1v1 experiment, not
the standard kickoff benchmark. Observations, controls, and rewards are unchanged.
Use short four-second episodes to focus this probe on contacting the ball.

Start from fresh weights so the comparison has a clear untrained control:

```powershell
& '.\training\venv\Scripts\python.exe' '.\training\train.py' --scenario touch --episode-seconds 4 --steps 65536 --num-envs 4 --run-name touch_probe
```

Compare initial and final checkpoints on the exact same controlled fixtures:

```powershell
& '.\training\venv\Scripts\python.exe' '.\training\evaluate.py' --scenario touch --seconds 4 --games 40 --model '.\training\checkpoints\touch_probe\baseline_step_0.pt' --output '.\training\logs\touch_probe\before.json'
& '.\training\venv\Scripts\python.exe' '.\training\evaluate.py' --scenario touch --seconds 4 --games 40 --model '.\training\checkpoints\touch_probe\baseline_step_65536.pt' --output '.\training\logs\touch_probe\after.json'
```

Compare `mean_touches` and `mean_reward`, then repeat at a holdout seed. Gains
on this probe establish only controlled ball-touch learning; they do not establish
kickoff skill, scoring skill, or tournament readiness. Lengthy commands are
delegated to the user's terminal.

The user completed 65,536 CUDA training transitions and both deterministic
comparisons. Each evaluation used 40 four-second episodes with alternating sides:

| Evaluation seed | Untrained mean touches | Trained mean touches | Untrained mean reward | Trained mean reward |
| --- | --- | --- | --- | --- |
| 1000 | 0.0 | 1.15 | 0.0 | 0.115 |
| 2000 (second-seed check) | 0.0 | 1.275 | 0.0 | 0.1275 |

All games timed out without goals. This establishes improved deterministic ball
contact on new offsets within this controlled reset distribution. Touch counts
are rewarded contact events, not the percentage of episodes with a touch.
Stochastic training reward declined toward zero even though argmax evaluation
improved, so the evaluation result should not be presented as stable training
convergence. Full-match skill and transfer to normal kickoffs remain unverified.
The first checkpoint with measured touch improvement is
`checkpoints/touch_probe/baseline_step_65536.pt`; retain its initial checkpoint
and the four before/after JSON reports as controls.

Kickoff transfer evaluation (20 games, seed 1000): 9 wins, 0 losses, 11 draws,
mean reward 4.51, mean touches 0.1. All nine wins occurred without learner
touches. Mean contact matched the untrained kickoff control; transfer has not
been demonstrated.

## Goal-aware scripted opponent

The original `v1` chaser remains available and is the default so previous
fixtures remain reproducible. `--opponent v2` selects an opponent that first
approaches behind the ball relative to the enemy goal, then drives through it.
The steering sign was verified with real simulator transitions. The quick
`opponent_test.py` check passed for both teams: v2 touched and scored into the
correct goal at step 62 in each controlled straight-line setup. This does not
establish strength or elimination of own goals in full kickoff matches.

Train and evaluate with an explicit opponent version; do not compare v1 and v2
win rates as if they were the same benchmark. New checkpoint configurations
record `opponent_version`; earlier checkpoints default to v1 when resumed.
The touch probe still uses its idle opponent regardless of this flag.

Next compare initial and touch-trained policies against v2 on the same fixtures:

```powershell
& '.\training\venv\Scripts\python.exe' '.\training\evaluate.py' --opponent v2 --games 20 --seed 1000 --model '.\training\checkpoints\touch_probe\baseline_step_0.pt' --output '.\training\logs\touch_probe\v2_before.json'
& '.\training\venv\Scripts\python.exe' '.\training\evaluate.py' --opponent v2 --games 20 --seed 1000 --model '.\training\checkpoints\touch_probe\baseline_step_65536.pt' --output '.\training\logs\touch_probe\v2_after.json'
```

The user ran those v2 evaluations. Both the untrained and touch-trained
policies lost all 20 games, with zero learner touches and mean reward -10.
See `CURRICULUM.md` for the organizer baseline target, the next measured
diagnostic, later curriculum stages, and the checkpoint promotion gate.

## Evaluation metrics, version 2

`evaluate.py` supports `--mode argmax` (default), `--mode sample`, and
`--mode random` (`--random` remains an alias). `--seed` determines simulator
fixtures and `--action-seed` controls sampled or random actions. Action RNG
starts from `action_seed + episode_index`, making repeated runs reproducible
and pairing action streams across checkpoints on matching episode indexes.
Argmax is the deterministic deployed-policy approximation; sampled mode
matches the stochastic action choice used during PPO collection.

Version-2 reports retain the old score, reward, and touch fields and add:

- `touch_episode_rate`: fraction of games with at least one learner touch step.
- `mean_raw_contacts`: RocketSim callback count, which can exceed rewarded
  touch steps; `mean_touches` still matches TouchReward's one credit per step.
- `first_touch_rate`, opponent and same-step rates: only the first action step
  with a contact is known; two contacts within one 8-tick step are ambiguous.
- Mean time to first learner touch, conditional on a learner touch. `null`
  means no learner touch in the evaluated games.
- `mean_ground_touch_steps`: learner touched and was grounded at both action
  boundaries. This is a conservative proxy, not the exact contact posture.
- `mean_goalward_touch_steps`: ball moved toward the opponent goal during an
  exclusive learner-touch step. This is a direction proxy, not a causal shot
  quality score. Overall ball progress is final minus initial ball Y, signed
  toward the opponent goal, including movement without learner contact.
- Own-goal proxies use the last recorded toucher at an eight-tick boundary.
  Unknown or same-step attribution is counted separately. They are not an
  official own-goal event from Rocket League.
- Episode duration, jump-action fraction, action histogram, per-episode
  records, reward/contract metadata, package versions, and checkpoint config.

The user ran the upgraded evaluator's first diagnostic. On seed-1000 and seed-2000 sets
of 40 nearby-ball episodes, the trained checkpoint's argmax mode touched in
95% and 97.5% of episodes, whereas sampled mode touched in 0% on both sets.
Argmax ball progress toward goal averaged approximately 844 and 899 units;
sampled mode had zero. Neither mode registered a ground-touch step. This
explains the earlier evaluation/training-reward discrepancy at the behavioral
level. It does not establish which exact sampled action interrupts contact.

The next experiment holds reward, scenario, model size, and opponent fixed,
while using `--ground-only` to mask actions 24 through 89. Actions 0 through
23 are verified ground controls from the same 90-action LookupTableAction.
The existing checkpoints load without this mask; new checkpoint config records
`ground_only: true`, and loading it recreates the mask for inference. Ground
masking is optional and can be removed by teams when improving the baseline.
This is a **new experiment**, not a re-interpretation of old checkpoints.

User-run command for the next short training experiment:

```powershell
& '.\training\venv\Scripts\python.exe' '.\training\train.py' --scenario touch --ground-only --episode-seconds 4 --steps 32768 --num-envs 4 --run-name ground_probe_v1
```

Evaluate its step-0 and step-32768 checkpoints in both modes on the same
fixtures before deciding whether this is progress. The metric test, old
checkpoint loading check, PPO tests, and ground-mask action tests passed. No
new training was launched by the assistant.

The user completed that run and evaluation. On 40 seed-1000 four-second
fixtures, the untrained ground policy touched in 52.5% of episodes under
sampling, but the 32,768-step policy touched in only 12.5%. Both argmax
policies touched in 0%. Sampled mean goalward ball progress declined from
about 665 to 263 units. The trained action distribution favored boosted
turns. Ground masking alone did not yield a useful policy.

## One optional approach reward experiment

`rewards/approach.py` implements signed car movement toward the previous ball
position, normalized by max-speed travel over one 8-tick action and clipped
to [-1, 1]. The existing 10-goal plus 0.1-touch reward remains the default.
`--approach-weight 0.02` adds at most 0.02 per action for useful approach,
with a negative value for moving away. It does not reward a stationary car
when the ball moves toward it. Six isolated reward tests passed.

For a clean comparison, the next user-run training command uses the same
ground-only mask, reset, 32,768 transitions, four environments, and seed as
`ground_probe_v1`; only the approach term is added:

```powershell
& '.\training\venv\Scripts\python.exe' '.\training\train.py' --scenario touch --ground-only --approach-weight 0.02 --episode-seconds 4 --steps 32768 --num-envs 4 --run-name ground_approach_v1
```

Evaluation now reconstructs reward weights from the checkpoint configuration,
so episode reward is **not** comparable across differently weighted runs.
Inspect sampled and argmax
contact, ground contact, ball movement, and reward on paired fixture seeds
before claiming progress or extending this training run. The assistant has
not started this longer run.

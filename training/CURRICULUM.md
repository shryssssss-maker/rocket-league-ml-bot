# Model Wars organizer baseline: evaluation-led curriculum

## Current direction (supersedes older diagnostic proposals below)

The 32k ground-mask and 65k touch runs are **diagnostics**, not evidence that
PPO cannot learn ground driving. Ground-only masking did not help in its first
short test; do not extend that branch by default. Return to the full 90-action
space. The early reward can now be set per run: goal, touch, signed approach,
facing while moving, and air. Existing runs still default to 10 goal + 0.1
touch. For a fresh early-stage run, explicitly set goal weight to zero.

`ApproachReward` measures signed car progress toward the prior ball position.
It is not identical to the official example's nonnegative instantaneous
`SpeedTowardBallReward`, but provides the missing dense directional signal.
`FacingBallReward` is speed-gated so standing still while facing the ball earns
zero. `InAirReward` is intentionally tiny and optional: the prior full-action
touch policy already overused jump. These are experimental choices, not proven
optimal weights. Isolated unit tests and a one-step RocketSim check passed;
no long run has been made using this revised stack.

First compare step-0 and trained checkpoints on *the same* close-ball fixtures,
both argmax and sampled, then expand the start distribution to longer, angled
approaches if actual contact improves. Evaluate across held-out seeds. Do not
compare raw reward between runs with different reward weights; compare contact,
first touch, time to contact, ground proxy, and ball movement. A few dozen
million transitions is an external guide's order-of-magnitude suggestion,
not a guarantee or an automatic next command for our serial simulator.

The sections below record earlier experiments and their rationale; commands
described there as "next" are historical, not the current recommendation.

### Early-reward probe result and stage-two transfer check

The user ran `early_reward_probe_v1` for 65,536 transitions with full 90
actions, eight-second near-ball episodes, and goal/touch/approach/face/air
weights 0/1/0.02/0.01/0.001. On 40 episodes per seed (1000 and 2000):

| Mode | Step 0 touch rate | Step 65,536 touch rate | Step 0 ground proxy | Step 65,536 ground proxy |
| --- | ---: | ---: | ---: | ---: |
| Argmax, seed 1000 | 0% | 87.5% | 0% | 87.5% |
| Argmax, seed 2000 | 0% | 87.5% | 0% | 87.5% |
| Sample, seed 1000 | 30% | 62.5% | 7.5% | 25% |
| Sample, seed 2000 | 25% | 57.5% | 10% | 25% |

The trained argmax policy almost never selected jump and moved the ball about
3,900 units toward goal on average, with 3 and 2 goals respectively and no
no-touch wins. Sampled contact and ball progress improved on both seeds too,
but sampled ground contact is only 25%. This is evidence for a narrow contact
skill on the close-ball fixture, not evidence of kickoff or 1v1 competence.

The `approach` reset scenario now implements stage two: stationary center ball,
learner 900–1700 units away, lateral offset up to 600 units, heading offset up
to 0.65 radians, alternating sides, idle distant opponent. Reward, observation,
and all 90 actions remain unchanged. Test the existing step-0 and trained
checkpoints on this wider fixture **before** resuming training. If transfer is
weak, training on this stage becomes a controlled curriculum intervention.
All long evaluation or training commands are run by the user.

The user completed that transfer check on 40 seed-3000 stage-two episodes.
Argmax contact rose only from 10% at initialization to 17.5% for the trained
near-ball checkpoint; sampled contact rose from 15% to 22.5%. The trained
ground-contact proxy was 17.5% argmax and 12.5% sampled, with average ball
progress about 711 and 940 units respectively. All three goals in each trained
mode followed learner contact, but three goals among 40 episodes are not a
reliable attack rate. Compared with 87.5% near-ball argmax contact, transfer
is weak. This supports stage-two training rather than kickoff training or
baseline promotion. Preserve this checkpoint as the pre-stage-two control.

The user then resumed that checkpoint for 131,072 additional transitions on
`approach`, preserving the same reward and full action space. The resulting
`approach_stage2_v1/baseline_step_196608.pt` checkpoint exists. Rolling shaped
reward rose from roughly -0.14 near the start to +0.26 at completion; that is
an optimizer signal, **not yet evidence of better stage-two contact**. Compare
it against step 65,536 on identical approach fixtures in both argmax and
sampled modes, including a new held-out seed, then check near-ball regression.
Do not advance to moving-ball or kickoff training solely from this reward curve.

The user ran the paired follow-up. The final step-196,608 policy regressed:
argmax made **zero touches in 40 episodes** on each of approach seeds 3000 and
4000, and zero in 40 near-ball seed-1000 episodes. On approach seed 4000,
sampled contact fell from 25% (step 65,536) to 20%; on seed 3000 it fell from
22.5% to 10%. On the near-ball seed-1000 fixture, sampled contact fell from
62.5% to 42.5%. Ground-contact proxy fell to zero argmax everywhere tested;
sampled ground contact was zero on approach seed 3000 and 2.5% on seed 4000.
The final argmax policy chose lookup action 52 on every evaluated step:
throttle/steer/jump/boost all zero, only roll=-1. On a grounded car this is
effectively idle. This is decisive regression despite higher training reward.
Do not continue from the final checkpoint or promote it as a baseline.

Reward and action analysis suggests a plausible local optimum: signed approach
can be negative when moving away, while idling receives approximately zero;
the wide approach fixture makes sparse touch success rarer. This remains a
hypothesis, not a proven causal explanation. High policy entropy means the
sampled policy still varies even when argmax selects an inert action. Inspect
intermediate checkpoints before changing rewards or scheduling another run.

The intermediate check on 40 seed-4000 approach episodes also failed to find
an improvement: step 98,304 had 0% argmax and 7.5% sampled contact; step
131,072 had 0% argmax and 20% sampled contact. The pre-stage-two checkpoint
had 22.5% argmax and 25% sampled contact on that same fixture. Step 98,304's
argmax policy mostly reversed/turned, and step 131,072 likewise did not make
contact. Keep the original step-65,536 checkpoint; do not resume this failed
branch. The evidence does not yet separate an overly wide start distribution
from the signed reward's possible idle incentive.

The next controlled fixture is `bridge`: learner distance 550–1100, lateral
offset up to 300, heading offset up to 0.3 radians, stationary center ball,
and idle distant opponent. It overlaps the proven close start and the easiest
portion of `approach`, while leaving the old `touch` and `approach` fixtures
unchanged. Evaluate the original step-65,536 checkpoint on bridge **before**
any more training. If bridge is learnable, a continuation there changes only
the reset distribution compared with the failed stage-two run. If bridge is
also poor, investigate reward/action behavior before spending more steps.

The user evaluated the preserved step-65,536 policy on bridge seeds 5000 and
6000 (40 episodes each). Argmax touch/ground-proxy rates were 50% and 42.5%;
sampled touch rates were 42.5% and 22.5%, with ground proxies 17.5% and 5%.
The policy scored 3 and 4 argmax goals, all after learner contact, while
average goalward ball progress was about 2,047 and 1,810 units. This fixture
is intermediate in difficulty between the close-ball and wider approach
tasks, though sampled performance varies substantially between seed sets.
The next controlled intervention is a **short** continuation from the original
step-65,536 checkpoint on bridge, retaining reward weights, model, 90 actions,
optimizer state, and 10-second horizon. Compare its step-98,304 checkpoint
against step 65,536 on identical bridge fixtures and a new held-out seed,
then check close-ball retention. Do not continue the failed `approach_stage2_v1`
branch or infer success from its training reward.

The user completed that 32,768-transition bridge continuation. The final
step-98,304 checkpoint regressed despite a finite, positive rolling reward:
argmax touch rate was 0/40 on both bridge seed sets and 0/40 on the original
near-ball seed-1000 set. Argmax selected lookup action 38 on every action step
in those reports: no throttle/steer/jump/boost, only pitch/roll (effectively
idle while grounded). Sampled bridge touch was 32.5% and 27.5% versus the
pre-bridge checkpoint's 42.5% and 22.5%; sampled ground-contact proxy fell
to 2.5% on each seed. Sampled near-ball touch fell from 62.5% to 47.5%.
Do **not** resume or promote this checkpoint. Two distinct continuation
distributions have now produced an inert argmax action, so difficulty alone
is not an adequate explanation.

The next reward ablation is optional `--approach-positive-only`. It clips
negative directional car progress to zero while keeping the same positive
progress scale, touch/facing/air weights, fixture, network, 90-action table,
optimizer state, and additional transition count. The old signed reward
remains the default and all existing checkpoint configurations load with it.
The current RLGym example also uses a nonnegative speed-toward-ball term, but
this ablation is **not** mathematically identical: ours measures progress over
the prior eight-tick action toward the prior ball position. This isolates
whether removing the retreat penalty reduces the apparent idle incentive.
If it fails, investigate action aliasing or other training dynamics before
more curriculum scaling. Compare behavioral metrics, not raw reward, across
the two reward configurations.

The target is an intentionally weak but functional ground bot: it can drive from
kickoff, orient toward the ball, reach it, touch it, and sometimes move it toward
the other goal. Aerial and advanced mechanics are outside this baseline target.
No checkpoint has yet met the promotion gate below.

## Evidence and immediate experiment

The 65,536-step touch checkpoint increased mean rewarded touch steps from zero
to 1.15 and 1.275 on two sets of 40 close-ball episodes. It did not increase
kickoff touch frequency, and both it and the untrained policy lost all 20
recorded games to v2. In the seed-1000 touch evaluation, its argmax action 78
was selected 1,661/2,400 times (69.2%): jump, pitch forward, and handbrake
with no throttle. Action 7, also with no throttle, was selected 659/2,400 times
(27.5%). Contact in that setup does not yet show the desired ground approach.

The first diagnostic is complete. On 40 four-second fixtures at seed 1000,
argmax touched in 38 episodes (95%) and sampled actions touched in none. At
seed 2000, argmax touched in 39 (97.5%) and sampled actions again touched in
none. Both modes had zero grounded touch steps. Argmax advanced the ball by
about 844 and 899 units toward the goal, respectively; sampled actions had
zero ball movement. The sampled jump-action fraction was about 29.5% versus
69% for argmax. This matches the collapse of sampled training reward, but the
exact causal action sequence has not been reconstructed from per-step traces.
The current policy is not yet a functional ground baseline.

The next controlled intervention is optional `--ground-only` action masking.
It retains the 90 output slots and action IDs but allows selection only from
LookupTableAction's first 24 ground actions. Existing checkpoints remain
unmasked. Train a fresh small run on the **same** four-second touch scenario,
with the **same** reward, model size, and idle opponent. Compare initial and
trained ground-mask checkpoints using sampled and argmax modes at seeds 1000
and 2000. Examine ground-touch episode rate, learner first-touch rate, time to
contact, jump-action fraction (expected zero), and goalward ball progress.
This isolates one action-availability change. Do not infer success from a low
loss or a raw win rate. Long runs are launched by the user in their terminal.

The current code uses `DefaultObs` with 92 float32 inputs to the network, the
same 90 discrete actions, and `10 * GoalReward + 0.1 * TouchReward`. Keep this
contract and the 128-by-128 actor/critic while diagnosing basic behavior.

The user completed the first ground-mask run at 32,768 transitions. On 40
seed-1000 fixtures, sampled contact fell from 52.5% at initialization to 12.5%
after training. Argmax contact stayed at 0% before and after. The trained
policy concentrated on boosted left/right turns, and mean goalward ball
progress fell from about 665 to 263 units under sampling. Ground action
availability alone did not produce learning. This checkpoint is a control,
not a baseline candidate.

## Curriculum stages

Each stage begins from a preserved checkpoint, uses a newly named run, and has
paired before/after reports on fixed evaluation fixtures. Training and held-out
evaluation seeds are recorded separately. Advancement requires repeatable
improvement in the behavior named for that stage, including a second held-out
seed. Choose numerical promotion thresholds after seeing comparable learning
curves and a simple reference policy. Do not advance on steps, loss, entropy,
training reward, or win rate alone.

1. **Near ball:** Existing `touch` scenario, 550 units behind a stationary
   ball, small lateral offset, idle opponent, four-second horizon. Measure
   episode contact rate, first touch, time to touch, action use, and the ground
   proxy. The unmasked checkpoint learned jumping contact; the fresh masked
   comparison is the next test of actual ground approach.
2. **Longer and angled approach:** Implemented `bridge` and `approach` scenarios. Increase
   starting distance, lateral offset, and heading variation in controlled
   ranges while the opponent remains idle. Keep the same reward first. Evaluate
   both the old close-ball and new longer-start fixtures so adaptation does
   not erase contact at the earlier stage.
3. **Moving ball:** Add seeded moderate starting ball velocities across and
   toward/away from the learner. Measure contact on held-out speeds/directions,
   time to touch, and whether contact changes ball movement toward goal.
4. **Non-aggressive opponent:** Keep the ball approach task but let a fixed
   opponent move without an immediate scoring threat. Distinguish learner
   first touches from opponent first touches and ambiguous same-step events.
5. **Standard kickoff challenge:** Use the actual kickoff mutator and v2
   opponent. Measure reaching/contacting the ball, grounded contact, kickoff
   first touch, conceding time, and goal attribution. A candidate that wins
   without contact does not satisfy this stage.
6. **Short 1v1:** Evaluate complete 30-second sudden-death episodes against v2
   across several held-out seeds and both learner sides. Examine positive
   ball movement and scoring in addition to survival and contact.

These are design stages, not yet implemented training commands. Implement one
stage at a time only after the preceding measurements make the next change
concrete. Retain `v1` and `v2` opponent versions for comparable old reports.

## Reward changes and checks

The first optional shaping term is now implemented: `ApproachReward`. It
projects the car's movement during one eight-tick action onto the direction
from its previous position to the ball's previous position. It divides by the
distance a max-speed car could travel in eight ticks and clips to [-1, 1].
Training weight 0.02 means its weighted magnitude cannot exceed 0.02 per
action. Moving toward the ball is positive, moving away negative, and moving
sideways or waiting for the ball to arrive earns zero. Signed progress
discourages back-and-forth distance farming across complete cycles, though
discounting can still make early gains more valuable than later penalties.
Six isolated tests check toward, away, sideways, stationary, ball-motion,
and clipping cases. It adds no new observation or action semantics.

The next experiment uses the same fresh initialization seed, scenario,
ground-action mask, model, steps, and idle opponent as `ground_probe_v1`,
changing **only** approach weight from 0 to 0.02. Evaluation now reconstructs
each checkpoint's training reward; raw reward is not comparable across those
variants, but contact metrics are. Evaluate on seed
1000 and a held-out seed, both sampled and argmax, before keeping the term.
The new experiment has not been trained yet. If contact becomes reliable,
then consider directional ball-velocity feedback as a separate later change.
Keep only reward terms that improve the intended behavior metric without
creating a new exploit.

## Promotion gate for `baseline.pt`

Keep candidate checkpoints named by run and step until all of these are
demonstrated across held-out kickoff fixtures against the same v2 opponent:

- The car spawns, drives, orients, and interacts with the ball more reliably
  than the untrained and random controls.
- A measurable portion of episodes contain learner contact; ground-contact
  and first-touch measures support the intended simple ground behavior.
- Some exclusive learner contacts move the ball toward the opponent goal, and
  scored goals are inspected for last-touch attribution and no-touch wins.
- Conceding time and match results show the candidate is not merely waiting
  for v2 to score. Gains repeat across fixture seeds and both sides.
- Reward, observation, action, package, and opponent metadata accompany the
  checkpoint. The eventual RLBot adapter implements the exact observation
  and action contract and passes a live behavior check.

Numerical thresholds remain unset until this candidate/reference comparison
exists. Training speed and CPU process parallelism follow behavioral validation.
The working `python-example/` remains untouched throughout.

# Model Wars — Architecture & Decision Log

> **Purpose:** Living document for the Model Wars bot pipeline.  
> Every major discussion gets recorded here with the decision reached, the reason, what remains unresolved, and what we do next.

**Last updated:** 2026-10-02  
**Current phase:** Baseline bot implementation planning

---

## 0. Project Goal

Model Wars is an overnight student AI/RL competition.

### Event concept
- Teams of 4.
- Organizers provide the same pretrained starter bot to every team.
- Teams improve the bot overnight.
- Teams submit their bot.
- Submitted bots are automatically validated.
- Bots compete in an organizer-controlled Rocket League tournament.
- The submitted bot must materially use ML to control gameplay; purely hard-coded gameplay is not acceptable.
- The starter should be **coherent but deliberately weaker than a competitive/high-GC bot**, leaving meaningful room for teams to improve it.

---

# Decision 1 — Do we train the starter entirely from scratch?

### Options discussed
1. Continue cold-start PPO from random initialization.
2. Use a strong pretrained bot as a teacher and train a smaller student from demonstrations.
3. Use a combination of imitation learning and later PPO fine-tuning.

### Evidence from our current experiments

Our cold-start PPO work reached a 1M-step checkpoint and clearly learned some useful behavior, including substantially higher ball-contact rates and better ball progress than the initial policy.

However, live RLBot testing showed the resulting policy was still poor at spatial navigation and directional control. The policy often entered repetitive action patterns and struggled to approach the ball accurately.

We also performed policy sensitivity tests showing that the network was responding to changes in ball position in some ways, but frequently failed to translate those changes into useful steering decisions.

### Decision

**Pivot away from relying on cold-start PPO as the primary way to create the starter.**

The new primary pipeline is:

**Strong pretrained teacher → demonstrations → smaller student → validation → optional PPO fine-tuning → freeze starter**

### Reason

A teacher gives the student access to already-coherent game behavior instead of asking PPO to discover basic spatial control from scratch.

### Status

**DECIDED**

---

# Decision 2 — Use Nexto/Necto as the teacher?

### Candidate considered

**Nexto / Necto**

The public Necto repository uses **CC BY-NC-SA 4.0**.

That creates additional questions around:
- NonCommercial use.
- Attribution.
- ShareAlike.
- Whether a student trained from the teacher's actions/trajectories could be considered adapted material.
- Whether the exact pretrained weights have the permissions needed for our event and distribution model.

Our event appears substantially noncommercial from the facts discussed (student ACM/MUJ event, no sale of the model, no commercial exploitation), but that does **not** turn the legal question into an automatic yes. The AI-training/derivative-work question also does not have a universally simple answer.

### Decision

**Do not build Model Wars around Nexto unless we obtain clear permission / legal clearance.**

Instead, search for a teacher with a clearly permissive license and publicly usable weights.

### Status

**DECIDED**

---

# Decision 3 — Find a permissively licensed pretrained teacher

## Candidate: CanoPy

We identified **CanoPy** as the current leading candidate.

Public model repository:

https://huggingface.co/FlameF0X/CanoPy

The model repository states:
- Pretrained Rocket League RL agent.
- PPO-based training.
- RLGym-based setup.
- `DefaultObs`.
- Lookup Table action space.
- Action repeat of 8.
- 2v2 training.
- Public policy/value weights.
- Apache-2.0 license stated by the model repository.

### Why CanoPy is attractive

It gives us a teacher whose publicly stated license is much cleaner for our use case than the CC BY-NC-SA terms attached to Nexto/Necto.

### Important unresolved issue

We still need to verify:
1. Exact checkpoint/weight provenance.
2. Exact observation contract.
3. Exact action contract.
4. Whether the published inference code reproduces the model correctly.
5. How well it behaves in our specific tournament environment.

### Decision

**Use CanoPy as the primary teacher candidate, subject to technical validation and final license/provenance verification.**

### Status

**TENTATIVE / VALIDATION REQUIRED**

Source:
https://huggingface.co/FlameF0X/CanoPy

---

# Decision 4 — 1v1 or 2v2?

### Initial plan

We originally designed our own starter around a 1v1 setup.

### New information

CanoPy is explicitly trained for **2v2**.

Instead of forcing a 2v2 teacher into a 1v1 student problem, we considered changing the Model Wars competition itself to 2v2.

### Decision

**Model Wars will move to 2v2.**

This aligns:
- Teacher training environment.
- Student training environment.
- Tournament environment.

It also introduces a richer task involving:
- Positioning.
- Rotation.
- Challenges.
- Team interaction.
- Passing/support behavior.

### Consequence

We need to redesign the student around the correct **2v2 observation contract** rather than reusing the current 1v1 92D observation blindly.

### Status

**DECIDED**

---

# Decision 5 — How will the student be trained?

### Current plan

Start with **Behavior Cloning (BC)** rather than jumping directly into more reward engineering.

Pipeline:

```text
CanoPy teacher
      ↓
teacher gameplay / state-action demonstrations
      ↓
Behavior Cloning
      ↓
small student policy
      ↓
offline + simulation validation
      ↓
optional PPO fine-tuning
      ↓
freeze Model Wars starter
```

### Why

First prove that the student can imitate a competent teacher.

Only after imitation is working do we add RL fine-tuning.

### Not decided yet

We still need to compare:
- Plain Behavior Cloning.
- True policy distillation.
- BC followed by PPO.
- How much we intentionally reduce the student's capacity/training so it remains a starter rather than a teacher clone.

### Status

**PROVISIONAL**

---

# Decision 5 — Student Observation Contract

### Candidate approaches

1. Reuse the current 1v1 92D observation.
2. Build a custom relative-position observation.
3. Reproduce CanoPy's 2v2 `DefaultObs` contract exactly.

### Discussion result

CanoPy is documented as using RLGym `DefaultObs` for 2v2.

For the standard RLGym configuration discussed, the observation is **172-dimensional**. The observation contains global ball/boost/state features plus fixed-size car-state blocks for the relevant teammates/opponents, with zero-padding used for unused slots.

Using the same observation contract for the teacher and student has an important advantage:

```text
CanoPy teacher
      ↓
172D DefaultObs
      ↓
teacher action
      ↓
demonstration
      ↓
172D DefaultObs
      ↓
student action
```

This avoids introducing an additional observation translation layer during behavior cloning.

### Decision

**Use the same 172D RLGym `DefaultObs` contract for the student, with the same normalization and padding semantics used by the teacher.**

### Required validation

Before collecting the main dataset, implement and test our own observation builder against RLGym's implementation on generated 2v2 states.

Required checks:
- Exact feature ordering.
- Scaling/normalization.
- Team/orientation handling.
- Ally/opponent ordering.
- Zero-padding behavior.
- Numerical agreement within a defined tolerance.

### Status

**DECIDED — implementation validation required**

# Decision 6 — Student Network Architecture

### Initial proposal

The first proposed student was:

```text
172 → 128 → 64 → 90
```

The idea was to make the student substantially smaller than CanoPy's documented:

```text
172 → 256 → 128 → 90
```

### Validation

Research into Rocket League/RLGym transfer learning and general policy distillation supports the overall teacher → smaller-student approach, but it does **not** establish that `128 → 64` is the uniquely correct size for this task.

The current `rlgymppo_rs` project has an explicit transfer-learning implementation that distills a larger teacher into a smaller student and exposes student `policy_layer_sizes` as a training choice. It also reports transfer loss and teacher/student action agreement. citeturn929094search0

The RLGym PPO guide notes that policy layer size is a tunable capacity/performance trade-off and that larger policies can learn better, although those recommendations are for PPO training rather than specifically for our distillation setup. citeturn929094search1

General policy-distillation literature supports transferring a capable policy into lower-capacity students, including cases where a distilled low-capacity policy outperforms a low-capacity agent trained directly from scratch; however, the correct compression level is task-dependent. citeturn929094academia26turn929094academia27

### Decision

**Do not hard-lock one student width yet.**

Run a controlled capacity sweep using the same teacher demonstrations and evaluation protocol:

```text
A: 172 → 128 → 64 → 90
B: 172 → 192 → 96 → 90
C: 172 → 256 → 128 → 90
```

Measure both imitation quality and actual 2v2 gameplay before selecting the final architecture.

### Selection criteria

- Teacher action agreement.
- Action-distribution divergence / transfer loss.
- Ball-touch rate.
- Useful/goalward contacts.
- Goals scored/conceded.
- Recovery and anti-circling behavior.
- Teammate positioning / basic rotation behavior.
- Live RLBot stability.
- Inference cost.

The final architecture should be **the smallest student that produces coherent and stable gameplay**, not simply the smallest network.

### Status

**DECIDED — empirical architecture selection required**

# Decision 7 — Behavior Cloning vs Policy Distillation

### Options

**A. Hard-label Behavior Cloning**

Train on:

```text
observation → teacher's selected action
```

using a supervised classification loss.

**B. Policy Distillation**

Train on:

```text
student observation → teacher action distribution
```

using KL/cross-entropy-style distribution matching.

**C. Hybrid**

Use distribution matching as the primary objective, while also tracking hard-label action agreement and optionally using hard-label BC as a baseline experiment.

### Research / ecosystem validation

The `rlgymppo_rs` project includes a dedicated `transfer_learn(...)` path for distilling a larger Rocket League policy into a smaller policy. It trains the student to match the teacher's action distribution and exposes transfer loss plus teacher/student action agreement. It requires the same action space, while allowing different observation spaces. citeturn929094search0

Policy-distillation research supports transferring a trained RL policy into a smaller student, and work specifically studying PPO distillation reports that low-capacity distilled students can outperform low-capacity agents trained directly in the environment; distillation followed by environment fine-tuning can also recover or approach teacher-level performance in some settings. citeturn929094academia26turn929094academia27

CanoPy is a PPO policy with a discrete Lookup Table action space and a documented [256, 128] policy architecture, making distribution-level transfer conceptually compatible with our student setup. citeturn193525search0turn796599search0

### Important implementation constraint

We have not yet verified that the published CanoPy checkpoint exposes its full action logits through an existing inference implementation. The public model repository clearly provides the policy weights and model configuration, but its visible model card does not provide a complete distillation/data-collection implementation. citeturn193525search2turn193525search3

Therefore the plan is:

1. Reconstruct and verify CanoPy inference.
2. Confirm we can obtain the 90-way action logits/probabilities.
3. Use **policy distillation as the primary student-training objective**.
4. Keep hard-label BC as a baseline/sanity-check experiment.
5. If exact teacher probabilities cannot be recovered reliably, fall back to hard-label BC rather than introducing an unverified teacher signal.

### Decision

**Primary method: offline policy distillation.**

**Fallback: hard-label Behavior Cloning.**

The student will learn from a fixed dataset of 2v2 teacher states and teacher policy outputs. This keeps the first training stage supervised and reproducible; PPO fine-tuning remains a separate later decision.

### Status

**DECIDED — teacher-logit extraction must be validated before dataset generation**

# Decision 8 — CanoPy Teacher Integration / Feasibility Gate

### What we verified

The public CanoPy model repository states that CanoPy is:
- RLGym + RLBot v5.
- PPO.
- 2v2.
- 8-action-repeat.
- `DefaultObs`.
- Lookup-table action space.
- Policy layers `[256, 128]`.
- Apache-2.0 licensed.

The repository contains a PyTorch policy checkpoint at:

```text
model/PPO_POLICY.pt
```

along with the value network and optimizer checkpoints.

The `rlgymppo_rs` project has a mature online teacher→student distillation implementation, but its documented transfer path is built around its own checkpoint/model representation. CanoPy's published checkpoint is a PyTorch `.pt` model, so **we should not immediately port the whole project to Rust just to use its transfer-learning implementation**.

### Decision

**Use a Python/PyTorch integration spike first.**

The spike must:

1. Reconstruct CanoPy's policy network from its documented configuration.
2. Load `PPO_POLICY.pt`.
3. Reproduce the exact 2v2 observation contract.
4. Reproduce the exact 90-action LookupTable.
5. Produce the teacher's 90-way logits/probabilities.
6. Compare deterministic actions against CanoPy's published/inference behavior where a reference is available.
7. Run CanoPy inside our RocketSim environment for multiple 2v2 states.
8. Confirm the output is stable enough to act as a frozen online teacher.

### Important consequence

We **do not yet start student training**.

First we prove:

```text
RocketSim state
      ↓
CanoPy observation
      ↓
CanoPy PyTorch checkpoint
      ↓
90-way teacher distribution
```

Once that works, online distillation becomes straightforward.

### Fallback

If we cannot reliably reproduce the teacher's probability distribution, use **hard-label teacher actions** and Behavior Cloning rather than building an unverified teacher representation.

### Status

**DECIDED — implementation spike is the next engineering gate**

### Sources

- CanoPy model: https://huggingface.co/FlameF0X/CanoPy
- CanoPy model files: https://huggingface.co/FlameF0X/CanoPy/tree/main/model
- rlgymppo-rs transfer learning: https://github.com/VirxEC/rlgymppo_rs

# Decision 9 — How to Deliberately Weaken the Model Wars Starter

### Goal

The starter should be:

> **Competent enough to play coherent 2v2 Rocket League, but clearly improvable during an overnight competition.**

We do not want artificial crippling such as random controls, disabled mechanics, or deliberately broken inference.

### Validation

The teacher→student transfer literature supports the idea that a smaller student can retain substantial teacher behavior, and Rocket League transfer tooling explicitly expects distillation to converge and then optionally continue with normal PPO fine-tuning. citeturn786506search0turn786506academia25turn786506academia27

That means **network capacity alone should not be our only weakness control**. A student can still become surprisingly strong if allowed to fully converge.

### Controls we will use

#### 1. Student capacity
Select the smallest architecture from the planned capacity sweep that remains coherent.

Candidates:

```text
172 → 128 → 64 → 90
172 → 192 → 96 → 90
172 → 256 → 128 → 90
```

#### 2. Distillation stopping point
Do not automatically train until maximum teacher agreement.

Stop once the student reaches the target baseline quality.

#### 3. Initial PPO stage
Do **not** run an unlimited PPO phase before distributing the starter.

PPO, if used, will be a bounded refinement stage rather than another full training campaign.

#### 4. Behavior scope
The baseline does not need to reproduce every advanced teacher behavior.

The intended starter should primarily demonstrate:
- basic movement and approach,
- useful ball contacts,
- basic 2v2 positioning,
- recoveries,
- ordinary defensive/offensive play.

Advanced aerial consistency, sophisticated passing, high-level mechanics, and other difficult behaviors can remain areas for competitors to improve.

### Decision

**Use controlled under-training rather than artificial handicapping.**

The default freeze pipeline is:

```text
CanoPy
   ↓
online distillation
   ↓
small coherent student
   ↓
stop before full convergence
   ↓
(optional bounded PPO refinement)
   ↓
freeze
   ↓
Model Wars
```

The exact stopping point will be selected through evaluation rather than a fixed arbitrary number of steps.

### Status

**DECIDED — exact PPO refinement policy is the next discussion**

# Decision 10 — PPO Fine-Tuning Policy

### Recommendation

Use PPO as a **short, bounded refinement experiment**, not as a mandatory long training stage.

Final intended pipeline:

```text
CanoPy
   ↓
online policy distillation
   ↓
small coherent student
   ↓
evaluate
   ↓
short PPO refinement
   ↓
evaluate again
   ↓
keep PPO only if it demonstrably improves the starter
   ↓
freeze
```

### Why

Distillation gives the student a strong behavioral starting point. PPO can then correct weaknesses that appear when the smaller student interacts with the actual environment rather than simply matching the teacher.

The key difference from the old project is:

```text
OLD:
random policy → PPO → discover basic gameplay

NEW:
competent student → PPO → refine/robustify
```

Therefore the PPO stage should be substantially smaller and more controlled than the old cold-start PPO campaign.

### PPO objective

The primary objective should remain **actual gameplay quality**, not maximizing a shaping metric.

Start with:
- GoalReward as the primary signal.
- Small ball/goal progression or approach shaping only where needed.
- Avoid making TouchReward a dominant objective.

This follows the lesson from our previous training: a policy can optimize contact-related behavior without becoming genuinely effective at gameplay.

### Stop / keep rule

Train two versions from the same distilled checkpoint:

```text
A = distilled-only
B = distilled + bounded PPO
```

Evaluate both using the same held-out 2v2 suite.

Keep PPO only if it produces a meaningful improvement in:
- actual game performance,
- useful ball progression/contacts,
- recovery,
- defensive/offensive behavior,
- teammate positioning/coordination,
- robustness across opponents and initial states,

without making the starter too close to the teacher.

If PPO does not produce a clear improvement, **ship the distilled-only student**.

### Decision

**PPO is an optional refinement stage with a bounded budget and an evidence-based keep/drop gate.**

We will **not** automatically run an unlimited PPO campaign before freezing the Model Wars baseline.

### Status

**DECIDED — exact PPO budget and evaluation protocol must be defined together**

# Decision 11 — Evaluation Must Include Controlled Fixtures

### Discussion result

A normal 2v2 match alone is not enough to evaluate the starter.

Full matches measure overall gameplay, but they make it difficult to diagnose a specific capability. Controlled fixtures let us test exact situations repeatedly and deterministically.

### Decision

Use **two mandatory evaluation layers**:

#### Layer A — Held-out controlled fixtures

Create deterministic scenarios covering areas such as:

```text
Movement / approach
- straight approach
- angled approach
- moving ball
- ball moving toward/away from car

Offense
- open goal
- attacking corner
- teammate possession
- teammate challenge

Defense
- ball near own goal
- opponent possession
- emergency recovery
- corner clear

Interaction
- teammate competing for ball
- opponent competing for ball
- different teammate locations

Recovery / mechanics
- low boost
- awkward orientation
- airborne recovery
- wall-adjacent ball
```

The final evaluation fixtures must be **held out from training**.

#### Layer B — Full 2v2 matches

Run normal matches against a fixed, reproducible opponent set and evaluate:
- goals,
- conceded goals,
- useful contacts,
- progression,
- recovery,
- teammate behavior,
- overall robustness.

### Why both are required

```text
              Candidate bot
                   │
          ┌────────┴────────┐
          ↓                 ↓
   Controlled fixtures   Full 2v2
          ↓                 ↓
   capability diagnosis   real gameplay
          └────────┬────────┘
                   ↓
              freeze gate
```

Fixtures tell us **why** the bot succeeds/fails. Full matches tell us whether those capabilities combine into useful gameplay.

### Comparison protocol

Distilled-only and distilled+PPO candidates must be evaluated on the **same fixture seeds and same match conditions** so that PPO's contribution can be measured directly.

### Important anti-leakage rule

Training, development fixtures, and final evaluation fixtures must be separated. A capability test that the student repeatedly sees during training cannot be treated as independent final evidence.

### Status

**DECIDED — controlled fixtures are a core evaluation pillar, not an optional diagnostic**

# Decision 12 — Resource Limits and Runtime Isolation

### Discussion result

Resource limits are required, but they should be **measured from the final starter/runtime and set generously**, rather than invented arbitrarily.

The purpose is tournament stability and fairness, not to make the models artificially weaker.

### Required limits

Apply the same organizer-controlled limits to every submission:

- CPU usage
- RAM usage
- inference latency
- startup time
- GPU memory, if a shared GPU is used
- disk/storage usage
- file-system access
- network access

### Runtime policy

All submissions run on the **same organizer hardware and software environment**.

Teams do not receive different limits based on their development machines.

Recommended isolation:

```text
Team submission
      ↓
isolated process/container
      ├── isolated state directory
      ├── approved filesystem access
      ├── controlled resources
      └── restricted/no external network
```

### Network

Do not permit arbitrary internet access during tournament execution.

This prevents submissions from:
- downloading new models,
- calling external inference APIs,
- depending on unstable network services,
- accessing other teams' resources.

### Inference latency

Latency is a particularly important limit for an RL bot because excessively slow inference directly affects gameplay.

Measure inference-time distributions rather than judging from one slow call. Use a reasonable allowance above the frozen starter's measured worst-case/percentile behavior.

### How exact numbers are chosen

Do **not** pick the final numeric thresholds yet.

First benchmark:
1. One frozen starter instance.
2. Two simultaneous instances.
3. Worst-case controlled fixtures.
4. Full 2v2 matches.
5. Expected tournament concurrency.

Then set limits comfortably above normal usage with enough headroom for participant improvements.

### Crashes/timeouts

A submission that crashes, hangs during startup, or repeatedly exceeds runtime limits should fail validation or receive a match-level technical loss according to the tournament rules.

A single failing bot must not stop the tournament.

### Decision

**Use measured, generous, identical resource limits with isolated execution.**

Exact numerical limits will be selected only after the final starter and tournament hardware are benchmarked.

### Status

**DECIDED — numerical thresholds remain an implementation/benchmarking task**

# Decision 13 — Validator and Rejection Rules

### Submission pipeline

Every team submission follows the same automated validation flow:

```text
team ZIP
   ↓
manifest / file validation
   ↓
dependency validation
   ↓
startup + RLBot smoke test
   ↓
ML inference check
   ↓
controlled fixture tests
   ↓
resource / latency checks
   ↓
ACCEPT / REJECT
```

### Automatic rejection

A submission should be automatically rejected when it:

- does not contain the required files/entry point,
- cannot start within the allowed startup time,
- cannot connect to the organizer RLBot environment,
- repeatedly crashes,
- cannot perform model inference,
- uses forbidden packages/dependencies,
- attempts forbidden network or filesystem access,
- exceeds hard runtime/resource limits,
- fails mandatory smoke/fixture requirements.

### Manual review

Use manual review for cases that are not safely decidable by automation, such as:

- ambiguous ML usage,
- unusual but potentially valid architecture,
- borderline resource usage,
- unexpected dependency behavior,
- suspicious attempts to bypass competition rules.

### ML requirement

The rule should be:

> **Gameplay decisions must materially depend on a trained ML model included with the submission.**

The validator should verify that:
- a model/checkpoint is present,
- the submitted program loads it,
- inference actually executes,
- the model output influences controller actions.

The validator does not need to prove that every line of code is machine learning.

### Validated artifact

Once a submission passes validation:

1. Calculate a cryptographic hash of the accepted ZIP/artifact.
2. Store the accepted artifact immutably.
3. Use that exact artifact in the tournament.
4. Do not allow silent replacement after validation.

### Resubmissions

Rejected teams should receive:
- the failed test name,
- a concise failure reason,
- relevant logs/metrics,
- the deadline for resubmission.

A resubmission goes through the complete validator again.

### Decision

**Use automated rejection for objective failures, manual review for ambiguous cases, and cryptographically freeze the exact validated artifact used in the tournament.**

### Status

**DECIDED — implementation details remain**

# Previous Cold-Start PPO Work — Reference Only

This section records work that led to the pivot. It is not the current primary path.

### Environment
- RLGym 2.x.
- RocketSim.
- 92D DefaultObs.
- 90-action LookupTableAction.
- 8 physics ticks per action.
- PPO.
- 4 simulator instances.
- RTX 4050 Laptop GPU.

### Major result

A guide-style reward recipe using:
- Touch reward.
- Speed-toward-ball reward.
- Face-ball reward.
- Small in-air reward.

produced a 1M-step checkpoint with substantial improvement in ball-contact and ball-progress metrics.

Checkpoint:

`training/checkpoints/guide_early_1m_v1/baseline_step_1000000.pt`

However, live RLBot behavior remained inadequate for our intended starter quality, particularly in spatial navigation and accurate approach/steering.

### Conclusion

The work is useful as research/reference, but it should not be the main path for constructing the event starter.

---

# Current Architecture Direction

## Teacher

**Primary candidate:** CanoPy

Expected role:

> Generate high-quality state → action demonstrations.

The teacher is **not** the bot that will be distributed to participants.

---

## Student

The student should be:
- Smaller than the teacher.
- Designed for 2v2.
- Easy for participants to modify.
- Strong enough to exhibit coherent gameplay.
- Weak enough to leave meaningful room for improvement.

The student should **not** simply become a full-strength CanoPy clone.

---

# Constraints / Design Principles

1. **License clarity matters.**
2. **Teacher and student should operate under the same game mode whenever practical.**
3. **Validate exact observation/action contracts before training.**
4. **Do not assume a repository's top-level license automatically covers every model/checkpoint dependency.**
5. **Do not start another large RL run until the student demonstrates coherent behavior.**
6. **Use held-out evaluation scenarios to detect memorization or narrow shortcuts.**
7. **The final starter should be intentionally weaker than the teacher.**
8. **The final starter must work reliably through the same RLBot deployment shell used by the competition.**

---

## Baseline-First Priority Board

| Priority | Item | Status |
|---|---|---|
| **P0** | CanoPy checkpoint/inference validation | 🔴 Next implementation gate |
| **P0** | 172D observation + 90-action exact-contract tests | 🔴 Required |
| **P0** | Student capacity sweep | 🔴 Required |
| **P0** | Online teacher→student distillation | 🔴 Required |
| **P0** | Distilled-only vs bounded-PPO comparison | 🔴 Required |
| **P0** | Fixture + full-match baseline evaluation | 🔴 Required |
| **P0** | Freeze final starter | 🔴 Required |
| **P2** | Tournament/validator infrastructure | ⏸ Parked |

# Decision Status Board

| Decision | Status | Current result |
|---|---|---|
| Cold-start PPO as primary starter-generation method | ✅ Decided | No |
| Nexto/Necto as teacher | ✅ Decided | No, unless separately cleared |
| Permissive pretrained teacher | ✅ Decided | Yes |
| CanoPy as teacher candidate | 🟡 Pending validation | Current primary candidate |
| Competition format | ✅ Decided | 2v2 |
| Student training | ✅ Decided | Offline policy distillation; BC fallback |
| Student observation | ✅ Decided | 172D RLGym `DefaultObs`, matching teacher semantics |
| CanoPy teacher integration | 🔴 Open | Python/PyTorch feasibility spike |
| Student architecture | ✅ Decided | Run capacity sweep; select smallest coherent student |
| Action space | 🔴 Open | Likely LookupTable, needs validation |
| Demonstration collection | ✅ Decided | Online distillation; no permanent dataset by default |
| Student weakening strategy | ✅ Decided | Controlled under-training; bounded PPO only if needed |
| PPO fine-tuning | ✅ Decided | Optional short refinement with keep/drop evaluation gate |
| Evaluation + freeze criteria | 🟡 Open | Two-layer suite: held-out fixtures + full 2v2 matches |
| Controlled fixtures | ✅ Decided | Mandatory held-out capability tests |
| Tournament validator | ✅ Decided | Automated validation + manual review + frozen artifact |
| Resource limits | ✅ Decided | Measured, generous, identical, isolated |

---

# Baseline Bot Is the Primary Project

The tournament infrastructure discussions are **parked for later**.

The major engineering objective is:

> **Produce a reliable, coherent, deliberately weaker 2v2 ML baseline bot that teams can improve overnight.**

Nothing about tournament brackets, scoring formats, or submission logistics should distract from this until the baseline itself is working.

## Current baseline pipeline

```text
CanoPy teacher
      ↓
verified 2v2 inference
      ↓
online policy distillation
      ↓
small student capacity sweep
      ↓
distilled student
      ↓
controlled PPO refinement experiment
      ↓
fixtures + full 2v2 evaluation
      ↓
freeze starter
```

## Immediate implementation priorities

1. **CanoPy feasibility spike**
   - Load the published checkpoint.
   - Reproduce the exact observation contract.
   - Reproduce the 90-action contract.
   - Obtain the teacher's action distribution/logits.
   - Run the teacher reliably inside our RocketSim stack.

2. **Student implementation**
   - 172D `DefaultObs`.
   - Same 90-action LookupTable.
   - Compare the three candidate policy capacities.
   - Keep the smallest architecture that remains coherent.

3. **Online distillation**
   - Frozen CanoPy teacher.
   - Train the student directly from RocketSim states.
   - No permanent dataset unless teacher integration forces an offline fallback.

4. **PPO refinement experiment**
   - Compare distilled-only vs distilled+bounded PPO.
   - Keep PPO only if it produces a measurable robustness/gameplay improvement.

5. **Baseline evaluation**
   - Held-out controlled fixtures.
   - Full 2v2 matches.
   - Blue and Orange.
   - Teacher-gap check.
   - Freeze only after all gates pass.

## Parked for later

The following decisions remain documented, but they are **not current engineering priorities**:

- tournament bracket/match format,
- final submission packaging,
- validator implementation,
- runtime sandboxing,
- tournament concurrency/hardware allocation,
- scoring and event operations.

These should be revisited only after the baseline bot is successfully built and frozen.

**This is now the primary project track.**


# Research Sources

### CanoPy
https://huggingface.co/FlameF0X/CanoPy

### Necto / Nexto
https://github.com/Rolv-Arild/Necto

### RLGym
https://github.com/RLGym/rlgym

---

# Change Log

## 2026-10-02
- Re-centered the entire project around the **baseline bot as the primary task**.
- Parked tournament format, validator, runtime, and event-operations discussions until after the baseline is built and frozen.
- Made CanoPy inference validation the immediate engineering gate.
- Locked validator behavior: objective failures are automatic rejection, ambiguous cases go to manual review, and the exact accepted artifact is cryptographically frozen for tournament use.
- Added resubmission diagnostics and complete revalidation requirements.
- Moved the next discussion to **tournament match format and live runtime protocol**.
- Locked resource limits as a competition requirement: measured, generous, identical, and enforced in isolated organizer-controlled runtime.
- Added network/filesystem restrictions and inference-latency monitoring to the runtime policy.
- Deferred exact numeric thresholds until the frozen starter and tournament hardware are benchmarked.
- Moved the next discussion to **validator and rejection rules**.
- Added controlled held-out fixtures as a mandatory evaluation pillar alongside full 2v2 matches.
- Required identical fixture seeds/conditions when comparing distilled-only vs PPO-refined students.
- Added train/development/final-fixture separation to prevent evaluation leakage.
- Moved the next discussion to **submission, validation, sandboxing, and tournament runtime**.
- Finalized the PPO decision: **bounded optional refinement**, with distilled-only vs distilled+PPO evaluated from the same checkpoint and PPO kept only when it demonstrably improves the baseline.
- Explicitly rejected an unlimited PPO campaign for the starter.
- Moved the next discussion to **evaluation and freeze criteria**.
- Validated the weakening strategy against current distillation/PPO references: avoid artificial handicaps; control strength through student capacity, distillation stopping, and a bounded optional PPO phase.
- Decided not to automatically distill to maximum teacher agreement or run unlimited PPO before the starter is frozen.
- Moved the next discussion to the exact **PPO fine-tuning policy**.
- Confirmed the practical CanoPy integration path should start with a **Python/PyTorch feasibility spike**, because the published teacher checkpoint is a PyTorch `.pt` model while `rlgymppo_rs` uses its own model/checkpoint system.
- Added a hard teacher-integration gate before any student training: exact observation, action, checkpoint loading, and 90-way teacher distribution must work first.
- Confirmed online distillation remains the default; hard-label BC is the fallback if teacher probability extraction cannot be validated.
- Moved the next discussion to **how to deliberately weaken the final Model Wars starter**.
- Chose **offline policy distillation** as the primary student-training method, with hard-label Behavior Cloning as a fallback/sanity-check path.
- Confirmed that the distillation method should only proceed after CanoPy's 90-way action logits/probabilities are reproduced and validated.
- Moved the next discussion to **demonstration collection and dataset design**.
- Validated the teacher→smaller-student architecture concept against Rocket League transfer-learning tooling and policy-distillation research.
- Decided to select the final student width empirically rather than locking `172 → 128 → 64 → 90` in advance.
- Added a 3-model capacity sweep: `128/64`, `192/96`, and `256/128` hidden layers.
- Moved the next discussion to Behavior Cloning vs policy distillation.
- Decided to use the same **172D RLGym `DefaultObs` contract** for the 2v2 student as the teacher, pending implementation-level equivalence tests.
- Moved the next open discussion to **student network architecture/capacity**.
- Pivoted from cold-start PPO toward teacher → student training.
- Decided not to rely on Nexto/Necto without separate license clearance.
- Selected a permissively licensed pretrained teacher as the new direction.
- Identified CanoPy as the current teacher candidate.
- Changed competition direction from 1v1 to **2v2** to align with CanoPy.
- Chose Behavior Cloning as the first student-training method.
- Established the next decision: verify the exact CanoPy observation/action contract.

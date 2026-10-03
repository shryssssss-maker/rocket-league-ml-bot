# CanoPy Feasibility Report

Investigation date: 4 October 2026. Scope: Step 1 only. Both project records were read completely. No project source, old checkpoint, PPO configuration, or deployed bot was changed. No training or gameplay evaluation was launched. Short, in-memory checkpoint and observation diagnostics were performed. Investigation evidence is in [checkpoint_inspection.json](checkpoint_inspection.json); downloaded provenance is in [repository_metadata.json](repository_metadata.json).

**Verdict: C. NO — the currently published artifact does not establish a usable, verified 2v2 teacher under the requested observation contract.** It loads successfully in our PyTorch stack, but its actor and critic accept **92 inputs**, whereas installed RLGym DefaultObs produces **132 inputs for unpadded 2v2**, or **172 inputs with the default three-cars-per-team padding**. The missing teacher source/config prevents verifying a different 92D contract. This does not prove the weights were trained in 1v1; it proves that the advertised 2v2 DefaultObs contract and downloaded checkpoint are unresolved.

## 1. Checkpoint

Publisher: `FlameF0X/CanoPy`. Downloaded revision: `c171a3f6235d134556ff315c087bbea60056d727`. Metadata records last modification as `2025-09-05T15:03:59Z`. [Published repository](https://huggingface.co/FlameF0X/CanoPy/tree/main), [downloaded metadata](repository_metadata.json).

The complete revision contains `.gitattributes`, `README.md`, and these five model files:

| File under `model/` | Bytes | Contents confirmed by inspection |
|---|---:|---|
| `PPO_POLICY.pt` | 275,942 | Actor state dictionary; six float32 tensors |
| `PPO_POLICY_OPTIMIZER.pt` | 552,318 | Adam-compatible optimizer state and parameter group |
| `PPO_VALUE_NET.pt` | 230,084 | Critic state dictionary; six float32 tensors |
| `PPO_VALUE_NET_OPTIMIZER.pt` | 460,672 | Adam-compatible optimizer state and parameter group |
| `BOOK_KEEPING_VARS.json` | 410 | Training bookkeeping and reward running statistics |

All downloaded file sizes match metadata. All four `.pt` SHA-256 hashes match Hugging Face's recorded LFS hashes. Actor SHA-256:

`54162458cc17b8e21530befd717796c41b41a58545b9fafd5bfd10f6fbc4208c`

The original CSV error came from enumerating its own output during hashing. Independent verification above succeeded; the checkpoints are intact. Full hashes and tensors are recorded in [checkpoint_inspection.json](checkpoint_inspection.json).

The [bookkeeping file](model/BOOK_KEEPING_VARS.json) reports `cumulative_timesteps=19000020`, `cumulative_model_updates=114`, and `epoch=18`. Both optimizer files have step 114 for every parameter. The README's one-billion-step figure is therefore **not evidence that this exported checkpoint completed one billion steps**. The sidecar is evidence of the accompanying recorded training progress, not independent proof of the run's history.

There is **no `config.json`, training script, inference script, requirements file, license file, or evaluation report** in this revision's file inventory, despite the README referring to configuration and a training-script evaluation function.

## 2. License / provenance

The precise available license declaration is `license: apache-2.0` in the downloaded [README model-card metadata](README.md), also reflected in `repository_metadata.json.cardData.license`. This is a **model-repository license declaration**, not merely a license inherited from an unrelated code repository. It is evidence that the publisher presents CanoPy under Apache-2.0. However, the publication contains no separate license text, weight-specific grant, training provenance manifest, or description of third-party initialization. Consequently, independently verified ownership/coverage of the exact weights remains unresolved; the declaration alone cannot establish every upstream right. [Published model card](https://huggingface.co/FlameF0X/CanoPy).

The card names FlameF0X and gives Discord contact `@flame_f0x`. The Hugging Face account/revision and exact file hashes are verified. Original training code, run configuration, evaluation results, and a CanoPy-specific reference inference implementation were not located in the publication or targeted source searches. This is a bounded search result, not a claim that no such source exists anywhere.

Relevant dependencies have separate licenses:

| Component | Evidence / scope |
|---|---|
| RLGym 2.0.1, rlgym-api 2.0.0, rlgym-rocket-league 2.0.1 | Installed distribution license files are Apache-2.0; [upstream LICENSE](https://github.com/RLGym/rlgym/blob/main/LICENSE) |
| `rlgym-ppo` upstream Python source | [Apache-2.0 LICENSE](https://github.com/AechPro/rlgym-ppo/blob/main/LICENSE); this covers its source, not automatically CanoPy weights |
| RocketSim 2.2.1 | Installed `rocketsim-2.2.1.dist-info/license/LICENSE` is MIT, copyright ZealanL |
| PyTorch 2.11.0+cu128 | Installed metadata identifies BSD-3-Clause; distribution has `LICENSE` and `NOTICE` for additional components |
| NumPy 1.26.4 | Installed `numpy-1.26.4.dist-info/LICENSE.txt`: BSD license plus bundled-library terms, including OpenBLAS/LAPACK, GCC runtime exception, and libquadmath LGPL terms |
| cmeel 0.61.0 | Installed `cmeel-0.61.0.dist-info/licenses/LICENSE`: BSD-2-Clause |
| Supplied RLBot/RLBot reference | [LICENSE.txt](https://github.com/RLBot/RLBot/blob/master/LICENSE.txt) is MIT; that repository explicitly describes v4 and is not a version pin or complete license inventory for our installed v5 runtime |
| `rlgymppo_rs` reference | Repository reports GPL-2.0; it is a research reference, not a dependency installed or adopted by this investigation |

These declarations do not establish license coverage for game binaries, simulator collision assets, CUDA components, or every transitive library. No claim of blanket Apache licensing or completed redistribution clearance is made. No Rust port is needed to investigate these Python weights. [Rust reference](https://github.com/VirxEC/rlgymppo_rs).

## 3. Policy architecture

The actor's **exact tensor-defined linear dimensions** are:

| Actor tensor | Shape |
|---|---|
| `model.0.weight` / `model.0.bias` | `[256,92]` / `[256]` |
| `model.2.weight` / `model.2.bias` | `[128,256]` / `[128]` |
| `model.4.weight` / `model.4.bias` | `[90,128]` / `[90]` |

Thus the published actor is **92 → 256 → 128 → 90**, with 68,314 learned parameters. All tensors are finite float32. Critic dimensions are **92 → 256 → 128 → 1**, with 56,833 parameters. Evidence: the respective checkpoint tensors, recorded in [checkpoint_inspection.json](checkpoint_inspection.json).

Upstream `rlgym_ppo.ppo.discrete_policy.DiscreteFF.__init__` builds Linear → ReLU → Linear → ReLU → Linear → Softmax for hidden sizes `[256,128]`; `get_output()` returns probabilities. `ValueEstimator` uses ReLU hidden layers and a scalar linear output. These definitions match the published tensor names and module metadata. [DiscreteFF source](https://github.com/AechPro/rlgym-ppo/blob/main/rlgym_ppo/ppo/discrete_policy.py), [ValueEstimator source](https://github.com/AechPro/rlgym-ppo/blob/main/rlgym_ppo/ppo/value_estimator.py).

**ReLU/Softmax are a well-supported reconstruction, not an independently confirmed exact CanoPy implementation.** A state dictionary does not identify parameter-free activation classes; CanoPy does not pin its `rlgym-ppo` version or publish its policy class.

## 4. Observation contract

**CanoPy's exact feature ordering, coefficients, ally/opponent ordering, and padding configuration cannot be established from the published files.** The README names DefaultObs and says observation standardization was disabled, but provides no builder instantiation or version. The first-layer tensor proves a 92D input requirement; it does not prove feature semantics.

The installed RLGym 2.0.1 implementation provides a precise **candidate contract**, which must not be labeled as CanoPy's verified contract. Source: `training/venv/Lib/site-packages/rlgym/rocket_league/obs_builders/default_obs.py`, class `DefaultObs`, methods `__init__`, `get_obs_space`, `_build_obs`, and `_generate_car_obs`.

`get_obs_space()` computes `52 + 20 * len(state.cars)` without padding, or `52 + 20 * zero_padding * 2` with padding. Short state-construction probes confirmed:

| Mode | Padding | Actual size |
|---|---|---:|
| 1v1 | `None` | 92 |
| 2v2 | `None` | 132 |
| 2v2 | `2` | 132 |
| 2v2 | `3` (constructor default) | 172 |

Therefore **172D is padded 2v2, not an intrinsic dimension for every 2v2 DefaultObs configuration**. The current [RLGym example](https://github.com/RLGym/rlgym/blob/main/example.py) uses 2v2 with `zero_padding=None`. Evidence for the probe is in `checkpoint_inspection.json`.

For installed DefaultObs with `zero_padding=3`, the 2v2 layout is:

| Inclusive indices | Features |
|---|---|
| 0–2 | Ball position × 1/2300 |
| 3–5 | Ball linear velocity × 1/2300 |
| 6–8 | Ball angular velocity × 1/pi |
| 9–42 | 34 boost-pad timers × 1/10 |
| 43–51 | Own holding-jump, handbrake, has-jumped, is-jumping, has-flipped, is-flipping, has-double-jumped, can-flip, air-time-since-jump |
| 52–71 | Own car block |
| 72–91 | Actual ally block |
| 92–111 | Zero ally block |
| 112–131 | First opponent block |
| 132–151 | Second opponent block |
| 152–171 | Zero opponent block |

Each 20-field car block is: position (3), forward (3), up (3), linear velocity (3), angular velocity (3), boost amount, demo respawn timer, on-ground flag, is-boosting flag, is-supersonic flag. Position/linear velocity use 1/2300; angular velocity uses 1/pi; boost uses 1/100. Forward/up, timers, handbrake, and flags are not additionally scaled. `ang_coef` is stored but not used to emit Euler-angle features.

Orange agents use `state.inverted_ball`, `state.inverted_boost_pad_timers`, and each car's `inverted_physics`. Other cars retain `state.cars.items()` insertion order within the ally/enemy lists. There is **no sorting by distance**, and padding follows the real entries. These facts are established for the installed builder only. [Upstream DefaultObs source](https://github.com/RLGym/rlgym/blob/main/rlgym/rocket_league/obs_builders/default_obs.py).

Our existing `training/observations/__init__.py` explicitly uses `zero_padding=None` and requires a finite `(92,)` float32 result. That is the old 1v1 contract. Its dimensional match to this checkpoint does not establish semantic compatibility with CanoPy.

## 5. Action contract

**Confirmed:** the actor contains 90 output coordinates. **Not confirmed:** the exact CanoPy mapping from those coordinates to controls. The card says lookup-table actions but publishes no parser or numeric table.

Installed `LookupTableAction.make_lookup_table()` constructs a `[90,8]` table in control order `throttle, steer, pitch, yaw, roll, jump, boost, handbrake`. Ground rows 0–23 enumerate throttle/steer in `(-1,0,1)`, boost/handbrake in `(0,1)`, skipping boost with throttle other than +1; rows are `[throttle or boost, steer, 0, steer, 0, 0, boost, handbrake]`. Aerial rows enumerate pitch/yaw/roll, jump, boost in source loop order; omit jump with nonzero yaw and ground duplicates; rows are `[boost,yaw,pitch,yaw,roll,jump,boost,handbrake]`, with handbrake enabled for nonneutral jump rotations.

Probe: row 0 is `[-1,-1,0,-1,0,0,0,0]`; row 89 is `[1,1,1,1,1,0,1,0]`. Source: installed `action_parsers/lookup_table_action.py`, methods `make_lookup_table` and `parse_actions`; [upstream source](https://github.com/RLGym/rlgym/blob/main/rlgym/rocket_league/action_parsers/lookup_table_action.py). Our `training/environment.py` already uses this table. Exact teacher parity still needs its source/version or published table.

## 6. Action repeat

The CanoPy README explicitly declares repeat **8**. Our `training/environment.py.ACTION_REPEAT` is also 8. Installed `RepeatAction.parse_actions()` repeats a single parsed control row into `(8,8)`; the short probe confirmed that shape. At 120 physics ticks/s this means nominally 15 decisions/s.

Our `RocketSimEngine.__init__(rlbot_delay=True)` defaults to a one-tick control delay; `step()` advances physics before applying controls in that mode. CanoPy's exact delay setting and deployment cadence are **not published**. Matching repeat alone does not verify latency semantics. Sources: CanoPy README, installed `repeat_action.py`, installed `sim/rocketsim_engine.py`, and our environment contract.

## 7. Inference path

Safe inspection was completed in this order: file size/hash verification; ZIP member inspection; static `pickletools.genops()` inspection of `data.pkl`; `torch.serialization.get_unsafe_globals_in_checkpoint()`; finally `torch.load(..., map_location='cpu', weights_only=True)`. No `weights_only=False`, custom pickle execution, or added safe-global allowlist was used.

All four files reference only `collections.OrderedDict`, `torch._utils._rebuild_tensor_v2`, and `torch.FloatStorage` in static inspection. PyTorch's unsafe-global scanner returned an empty list, and restricted loading succeeded. These checks reduce deserialization risk; they are not proof that arbitrary checkpoint files are universally safe. [PyTorch 2.11 serialization documentation](https://docs.pytorch.org/docs/2.11/notes/serialization.html).

An in-memory network following upstream DiscreteFF, in eval mode with gradients disabled, loaded the actor with `strict=True`: all keys matched. A zero 92D input produced finite `[1,90]` probabilities, sum `0.9999999403953552`, argmax 17. **This synthetic probe is not a gameplay result or reference-parity result.** Inputs of 132D and 172D both failed matrix multiplication against the 92-input layer. Evidence: `checkpoint_inspection.json.network_probe` and the actor's first-layer tensor.

The existing `training/checkpoint.py.load_checkpoint()` cannot load CanoPy directly: it expects our wrapper fields `format_version`, `contract`, `config`, `model`, and `versions`, then creates our tanh ActorCritic. CanoPy publishes a bare upstream-style state dictionary. A separate verified teacher loader would be needed, rather than changing the old checkpoint loader.

## 8. Teacher logits/probabilities

**The checkpoint has enough actor weights to calculate all 90 output scores; there is no sampled-action-only export limitation.** Under the upstream-compatible reconstruction, the final linear layer provides raw logits and its Softmax provides the 90-way probability vector. A probe verified exact parity between Softmax of those logits and the reconstructed model output.

Upstream `DiscreteFF.get_output()` returns Softmax probabilities; `get_action()` clamps them to `[1e-11,1]` before argmax or multinomial selection. For exact stochastic action semantics, any normalized categorical representation must account for that clamp. Source: [DiscreteFF.get_output/get_action](https://github.com/AechPro/rlgym-ppo/blob/main/rlgym_ppo/ppo/discrete_policy.py).

**What is still missing is a verified teacher distribution for actual 2v2 states**, because observation semantics and the exact parameter-free operations are unresolved. Recovering scores from synthetic 92D vectors does not clear the distillation gate. Hard-label BC would not resolve this blocker either: valid teacher actions still require valid inputs and action mapping.

## 9. Dependency compatibility

Actual training environment versions were queried: Python **3.12.10**, torch **2.11.0+cu128**, rlgym **2.0.1**, rlgym-api **2.0.0**, rlgym-rocket-league **2.0.1**, rocketsim **2.2.1**, numpy **1.26.4**, cmeel **0.61.0**. They match `training/requirements.txt`. Restricted tensor loading and the temporary network forward pass work on CPU without installing another package. `rlgym-ppo` is not installed.

CanoPy's exact original Python/PyTorch/RLGym/RocketSim/rlgym-ppo versions are **unknown**. Its archive format and state dictionary do not encode a dependency lock. Upstream `rlgym-ppo/setup.py` currently identifies version 1.3.13, Python >=3.7, torch >1.13, NumPy >1.21, plus training dependencies; that is evidence about current upstream requirements, **not CanoPy's original environment**. No need to install its learner/wandb/gym stack merely to run the reconstructed actor. [Upstream setup.py](https://github.com/AechPro/rlgym-ppo/blob/main/setup.py).

Special statistics: the card reports observation standardization disabled; the downloaded sidecar contains only `reward_running_stats` (mean, variance, count), not observation statistics. Those reward stats and optimizer/critic state are unnecessary for an actor-only forward pass. It remains unverified whether unpublished observation preprocessing was used. Upstream `PPOLearner.save_to()` explains the separate actor/value/optimizer files. [PPOLearner source](https://github.com/AechPro/rlgym-ppo/blob/main/rlgym_ppo/ppo/ppo_learner.py).

The historical RocketSim wheel-tag/pip-check discrepancy remains a reproducibility caveat recorded in `update.md`; it does not explain the actor's 92-input tensor. Current upstream RLGym example APIs also differ from the installed release, so copying main-branch code is not an exact version pin.

## 10. Risks / unknowns

1. **Blocking mismatch:** published actor requires 92D, while requested padded 2v2 contract is 172D and installed unpadded 2v2 is 132D.
2. **No exact teacher builder/config:** feature order, normalization, car ordering, padding, delay, and any custom filtering are unverified.
3. **No exact policy/inference source or dependency lock:** ReLU/Softmax reconstruction fits upstream and tensors but lacks CanoPy reference parity.
4. **No exact action-table evidence:** 90 outputs and a lookup-table description do not prove row ordering.
5. **Training progress discrepancy:** bookkeeping records roughly 19M steps; the card's 1B setting is not a confirmed completed checkpoint budget.
6. **No verified teacher gameplay:** neither RocketSim 2v2 rollouts nor live RLBot behavior was measured. With an unresolved input contract, performance claims would be unreliable.
7. **Model rights/provenance incomplete:** Apache model-card declaration exists; weight-specific ownership and third-party provenance are not independently established.

Do not truncate observations to 92, remove teammate/opponent blocks, append invented weights, or reinterpret the checkpoint as a verified 1v1 model. Each would introduce an unsupported contract assumption.

## 11. Exact files that would need modification

**None now.** The current blocker needs corrected artifacts or missing source, not changes to PPO. Only investigation documentation/evidence was added.

Once a genuine teacher contract is established, the concrete local integration points are:

| Existing file / function | Future requirement |
|---|---|
| `training/observations/__init__.py`: `OBS_SIZE`, `OBS_CONFIG`, `obs_builder`, `preprocess` | Add a separate versioned teacher/2v2 contract; current implementation hard-requires 92D |
| `training/environment.py`: `build_env`, `contract`, `Match` | Separate 2v2 construction and four-agent handling; current mutator is `(1,1)` and Match assumes one learner/one opponent |
| `training/policies/__init__.py`: `ActorCritic` | Keep old model intact; provide a separate teacher architecture and, later, separately approved student widths |
| `training/checkpoint.py`: `load_checkpoint` | Keep old loader intact; add a separate bare-state-dictionary teacher loader with source/hash/contract checks |
| `training/requirements.txt` | Add/pin dependencies only if corrected teacher source demonstrates a need; no current upgrade required for tensor loading |

New teacher module and contract-test filenames have not been adopted. They should be separate from old PPO and deployment paths. `training/ppo.py`, existing old checkpoints, and `python-example/src/bot.py` require no Step 1 changes. RLBot adapter work is deferred until simulator teacher inference is verified.

## 12. Recommended next step

**Resolve the publication mismatch before any teacher integration or student training.** Request from the publisher:

1. The actor checkpoint actually used for 2v2, its SHA-256, and matching run configuration.
2. The complete observation-builder implementation/instantiation and feature-index contract; specifically an explanation of this export's `[256,92]` first layer.
3. The policy class/version, exact action table/parser, repeat and delay settings, and reference inference outputs on shared states.
4. Dependency pins and explicit confirmation that the supplied weights are covered by the declared Apache-2.0 terms, including any third-party initialization.

No message was sent to the publisher. The smallest fallback is to obtain the corrected CanoPy export and missing contract. If those cannot be obtained, select another permissively licensed, source-verifiable 2v2 teacher through a separate investigation. **Hard-label BC is not the fallback for missing observation semantics; it only helps when teacher actions are reliable but full probabilities cannot be recovered.**

Answer to the requested success question: **C. NO for the artifact currently published and inspected.** PyTorch compatibility and 90-output extraction are feasible, but verified 2v2 teacher inference is blocked by the checkpoint/observation mismatch and missing source. Stop at this report; do not implement or train automatically.

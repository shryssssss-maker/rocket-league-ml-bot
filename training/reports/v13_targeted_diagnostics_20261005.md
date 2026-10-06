# V13 targeted diagnostics — 2026-10-05

Read-only analysis of the existing complete match. No new collection, fitting, model/contract change or DAgger aggregation. Training-only nearest neighbors are descriptive.

## 1. Kickoff temporal hypothesis

| Kickoff | Shadow start | Student first jump button | Jump delay versus shadow | First jump mode | Continuous hold | Jump-false callbacks during reference release | Native dodge before release |
|---|---:|---:|---:|---|---:|---:|---|
| K1 | 453 | 456 | 0.050000s | Front dodge | 0.500000s | 0/4 | False |
| K2 | 4261 | 4279 | 0.299995s | Front dodge | 0.500000s | 4/4 | False |
| K3 | 9808 | 9815 | 0.116669s | Jump | 0.500000s | 3/4 | False |
| K4 | 19938 | 19942 | 0.066650s | Front dodge | 0.500000s | 0/4 | False |

The table distinguishes pure Jump mode from the native jump button in Front-dodge output. Actual first-jump timing must be interpreted relative to shadow sequence start; a late jump can still carry pitch prematurely. A semantic Front-dodge label does not prove a physical dodge. Phase consumption and exact native controls are preserved in JSON.

All four first jump-button presses occur after the corresponding shadow start, not earlier. Each first press remains held for about 0.5s. Three start with pitch -1; the fourth starts with two pure Jump callbacks then adds pitch without an intervening release. Jump-false during a shadow release is not a correctly placed student release if the student has not yet pressed jump. These distinguish delayed initiation from missing own-sequence release.

Reference nominal durations are 0.05s Jump, 0.05s release, 0.20s jump+pitch, 0.80s coast. The original ControlStep uses strict elapsed>duration and returns finishing-step controls before the next callback consumes the following step. Therefore delivered phase spans need not equal nominal durations. No optimal-kickoff claim follows.

Milestones include pre-kickoff, first jump, first release, severe tilt, inversion, first actual touch and stable recovery. Full actual callback windows retain previous submitted actions, all native physical/jump fields, angular velocity, prediction selections and original sequence state.

## 2. Paired recovery evidence

| Pair category | Callbacks | Duration | Student jump-button callbacks | Movement resumes within 2s after interval | Nearby ball touches |
|---|---|---:|---:|---|---:|
| fast | 484–665 | 3.058s | 2 | True | 1 |
| fast | 4307–4459 | 2.575s | 2 | True | 1 |
| prolonged | 7705–8960 | 21.125s | 0 | True | 3 |
| prolonged | 15667–16824 | 19.483s | 0 | True | 3 |
| prolonged_neutral | 8187–8752 | 9.500s | 0 | True | 0 |
| prolonged_neutral | 13070–13902 | 14.008s | 0 | True | 0 |

Two fast orientation episodes, two prolonged episodes and two prolonged Neutral episodes were selected from the existing forensic criteria, without changing those criteria. Onset/midpoint/offset/after states and aligned causal windows are preserved. No regional proximity is called proven wall collision. Nearby touch records may explain opportunity for interaction, not causal impulse attribution.

## 3. Single-frame and causal-history observability

Twelve relevant query anchors are compared against the four frozen training matches at 1, 16 and 64 callbacks. Current normalized features are unaltered; 16/64 callback durations vary with actual dt and are reported. Search candidates never cross match boundaries or use future callbacks.

| Query | History callbacks | Best train match / ending callback | RMS feature difference | Exact? | Query / neighbor history seconds | Neighbor terminal orientation |
|---:|---:|---|---:|---|---|---|
| 456 | 1 | v10_006 / 17844 | 0.00577409 | False | 0.000 / 0.000 | upright |
| 456 | 16 | v10_001 / 9103 | 0.00280415 | False | 0.250 / 0.250 | upright |
| 456 | 64 | v10_006 / 5216 | 0.00160162 | False | 1.050 / 1.050 | upright |
| 4279 | 1 | pilot_01 / 3442 | 0.01073955 | False | 0.000 / 0.000 | upright |
| 4279 | 16 | pilot_01 / 3442 | 0.01314656 | False | 0.250 / 0.250 | upright |
| 4279 | 64 | v10_006 / 21460 | 0.01874053 | False | 1.050 / 1.050 | upright |
| 9815 | 1 | v10_004 / 18977 | 0.00545794 | False | 0.000 / 0.000 | upright |
| 9815 | 16 | v10_006 / 442 | 0.00244532 | False | 0.250 / 0.250 | upright |
| 9815 | 64 | v10_004 / 18977 | 0.05203800 | False | 1.092 / 1.050 | upright |
| 19942 | 1 | v10_006 / 17844 | 0.00387182 | False | 0.000 / 0.000 | upright |
| 19942 | 16 | pilot_01 / 9282 | 0.00230261 | False | 0.250 / 0.250 | upright |
| 19942 | 64 | pilot_01 / 9287 | 0.01222720 | False | 1.075 / 1.075 | upright |
| 579 | 1 | v10_001 / 23285 | 0.09347899 | False | 0.000 / 0.000 | upright |
| 579 | 16 | v10_001 / 4701 | 0.04862790 | False | 0.250 / 0.250 | upright |
| 579 | 64 | v10_001 / 4713 | 0.07967411 | False | 1.092 / 1.050 | upright |
| 4380 | 1 | v10_006 / 20087 | 0.08537494 | False | 0.000 / 0.000 | inverted |
| 4380 | 16 | v10_001 / 19477 | 0.05028348 | False | 0.250 / 0.250 | upright |
| 4380 | 64 | v10_006 / 6677 | 0.07048483 | False | 1.092 / 1.050 | upright |
| 8300 | 1 | pilot_01 / 4837 | 0.08290972 | False | 0.000 / 0.000 | upright |
| 8300 | 16 | pilot_01 / 4841 | 0.08674954 | False | 0.250 / 0.250 | upright |
| 8300 | 64 | pilot_01 / 4854 | 0.11349762 | False | 1.050 / 1.050 | upright |
| 13200 | 1 | v10_006 / 6693 | 0.04162258 | False | 0.000 / 0.000 | upright |
| 13200 | 16 | v10_006 / 6700 | 0.04363337 | False | 0.250 / 0.250 | upright |
| 13200 | 64 | v10_006 / 6700 | 0.07113050 | False | 1.083 / 1.083 | upright |
| 13500 | 1 | v10_006 / 6695 | 0.07966461 | False | 0.000 / 0.000 | upright |
| 13500 | 16 | v10_006 / 6701 | 0.07989393 | False | 0.250 / 0.250 | upright |
| 13500 | 64 | v10_006 / 6716 | 0.09527978 | False | 1.058 / 1.083 | tilted |
| 16500 | 1 | pilot_01 / 4837 | 0.07575108 | False | 0.000 / 0.000 | upright |
| 16500 | 16 | pilot_01 / 4842 | 0.07964794 | False | 0.250 / 0.250 | upright |
| 16500 | 64 | pilot_01 / 4858 | 0.10810286 | False | 1.050 / 1.050 | upright |
| 20250 | 1 | v10_001 / 7744 | 0.05603896 | False | 0.000 / 0.000 | inverted |
| 20250 | 16 | v10_001 / 7754 | 0.05692336 | False | 0.250 / 0.250 | tilted |
| 20250 | 64 | v10_001 / 7768 | 0.06018049 | False | 1.050 / 1.050 | tilted |
| 10335 | 1 | v10_004 / 7025 | 0.10372162 | False | 0.000 / 0.000 | upright |
| 10335 | 16 | v10_004 / 7029 | 0.10994573 | False | 0.250 / 0.250 | upright |
| 10335 | 64 | pilot_01 / 4827 | 0.17560289 | False | 1.050 / 1.050 | upright |

Per-feature deltas, history MAEs, terminal orientation/geometry and top-five neighbors are in JSON. Distances have no validated reliability threshold and are not classification accuracy. Exact equality and approximate resemblance are explicitly separated. A finite search cannot establish reliable 13D recovery-state identification.

Orientation influences relative target coordinates implicitly. Single-frame 13D does not uniquely specify world-up, contact constraints, angular motion or jump availability. A GRU can exploit temporal clues, but neither nearest histories nor hidden-state hashes prove that it can infer these hidden facts reliably. This audit does not claim orientation is completely absent or propose new features.

## 4. Expert-label qualification

- executable now: 0
- prerequisite missing: 216
- ambiguous: 0
- clearly not a meaningful direct correction: 0

Scope is the previously flagged 216 grounded/no-prior-jump shadow Front-dodge callbacks, not all expert labels. Each record includes prior student command evidence, native availability flags, air state, sequence phase and elapsed sequence time. Executed command does not establish successful physical first jump; current has_jumped=false remains the observed state.

Grounded jump+pitch commands are legal and may initiate a first jump on a rising button edge. They cannot be assumed to realize the shadow’s intended second-jump/dodge. Thus missing prerequisites are a maneuver-semantic classification, not proof the direct control is useless. No label was filtered, repaired or used as a target.

## 5. Video alignment

No synchronized video was supplied or found in known session/forensic evidence folders. No mapping of the human’s corner observation to one episode is invented. A future capture would require video presentation timestamps mapped to callback/frame/canonical-time markers with logged clock offset/latency; goal/replay events alone provide only coarse anchors. No future capture launched.

## 6. Decision framework

| Finding | Evidence | Implication |
|---|---|---|
| Temporal kickoff error | Native button-edge/release traces above; rotation before ball contact | Direct reason to test temporal supervision, not an assertion that missing features are the sole cause |
| Recovery distinguishability | Exact versus approximate train-history neighbors, original orientation metadata | No validated information-sufficiency claim; representation remains an open question |
| Prolonged Neutral | W1/W3 aligned windows; shadow often Chase on the same actual state | Student policy/temporal learning issue exists; physical constraint and hidden-state cause unproven |
| Stateful prerequisite mismatch | 216 qualified labels with actual native state and prior submissions | Blind DAgger must remain blocked pending a label-semantics decision |
| Teacher limitation | MyBot has speed-triggered sequence and no explicit recovery routine | Do not assume the shadow is an optimal recovery oracle |
| Student-only discrepancy | Different phase/action timing from unchanged source | An isolated temporal experiment is supported; teacher-controlled outcome counterfactual absent |

**Exactly one recommended next research direction: B — a bounded temporal training/objective experiment**, requiring separate approval and a specification before fitting. Keep 13D/checkpoint/reference immutable during this audit. A future experiment would test sequence timing and rare-mode supervision with the same input contract, rather than declaring a feature repair from nearest-neighbor resemblance.

A is premature as a mandatory repair because the finite neighbor search does not prove causal-history indistinguishability. C is premature because maneuver prerequisites and label usefulness remain unresolved. D changes the reference before testing the directly measured imitation failure. E is not required to reproduce the native timing failure already present; missing video still limits wall-causality claims. F is not required as the next intervention: questions can be tested sequentially without changing multiple factors. B is a proposed experiment, not a claimed cure or authorization to train.

## 7. Integrity, limits and stop

All eight permitted training raw/observation artifacts were hash-verified before full sample reads, with manifest ordering/dtype checks. No validation/test sample was opened. No training tensors were persisted; only diagnostic neighbor metadata and callback windows were written. Source/capture hashes were checked before and after analysis.

Evidence is observational, packets may omit physics ticks, contacts are repr-based, GRU hidden vectors were not captured, and the teacher never controlled these physical trajectories. Report neither competency, causal distribution-shift diagnosis nor DAgger benefit.

Callback windows: `C:\Users\shreyas\Desktop\model wars\training\reports\v13_targeted_diagnostics_20261005_v1\callback_windows.json`. JSON companion: `C:\Users\shreyas\Desktop\model wars\training\reports\v13_targeted_diagnostics_20261005.json`. STOP for human review.

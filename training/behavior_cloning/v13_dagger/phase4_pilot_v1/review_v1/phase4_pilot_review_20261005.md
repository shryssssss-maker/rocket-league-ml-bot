# Phase 4 pilot review — 2026-10-05

**Classification: failed the strict 90-second runner limit; the single diagnostic pilot otherwise completed with scope.**

The automatic report says `pilot_complete_awaiting_review`. Manual evidence review found two controls submitted after the approved cutoff: the final submission was at +90.031 seconds. This is a runner timing-contract violation, not a model-training result. No rerun or fix was performed.

Session: `C:\Users\shreyas\Desktop\model wars\training\behavior_cloning\v13_dagger\phase4_pilot_v1\sessions\20261005_063422_960966`. Blue human_b versus Orange pinned 13D GRU student. One natural live 1v1; original teacher shadow-only.

### 1. Actual learner-induced physical states

5,238 unique callbacks and 5,238 exact student-only transport submissions. 5,237 successor callbacks link to the preceding verified student submission. First/last frames: 13/10,645; canonical elapsed span: 88.600003 seconds. The first callback is not labeled learner-induced. All captured observations use the actual received packet; no requested or synthetic states were used.

Near ball: 3,839 callbacks; far ball: 1,399. Every snapshot had a ball and valid prediction. Scores remained Blue 0 / Orange 0; scoring ability, touch rate and useful contact were not independently measured.

### 2. Student versus independent shadow labels

| Mode | Student callbacks | Shadow callbacks |
|---|---:|---:|
| Neutral | 1012 | 480 |
| Chase | 4168 | 4598 |
| Jump | 0 | 41 |
| Front dodge | 58 | 119 |

Mode agreement: 4,012/5,238 (76.59%). Native eight-channel agreement within 1e-6 analog tolerance: 212/5,238 (4.05%). Steering-sign disagreements outside ±0.02: 379.

| Channel | MAE versus shadow |
|---|---:|
| throttle | 0.222604 |
| steer | 0.354692 |
| pitch | 0.023100 |
| yaw | 0.000000 |
| roll | 0.000000 |
| jump | 0.030546 |
| boost | 0.000000 |
| handbrake | 0.000000 |

Yaw/roll remained exactly zero; boost/handbrake remained false in both paths. Steering absolute error: median 0.206024, p95 1.0, maximum 1.402245.

The shadow started and completed nine original sequences; the student missed all nine Jump-mode starts and emitted no pure Jump mode anywhere. Of 41 shadow Jump callbacks, 40 had no student jump button. The shadow continued independently through 631 pending callbacks after missed starts. There were 51 shadow Front-dodge callbacks while the actual learner car was grounded and had not jumped. These are potentially incompatible action prerequisites, not proof that the labels form useful learning targets.

Missed starts: [353, 630, 730, 1191, 1784, 1860, 3395, 4099, 4590]. Per-sequence evidence is in the JSON review.

The shadow made 1,064 actual prediction selections; all matched the observation probe index/timestamp/position on those callbacks. Probe selection on other callbacks does not imply the original teacher selected a slice while a sequence was pending. This run verifies input provenance and independent continuation, not a new teacher-controlled-physics counterfactual.

### 3. Train-only descriptive distribution comparisons

Primary reference: 57,794 Orange-side train callbacks (pilot_01/v10_006). Secondary: all 107,400 train callbacks. No validation/test samples were accessed. Comparisons did not choose, modify or fit the policy.

| Feature (primary reference, ALL phase aggregate) | Outside train min/max | Outside p01–p99 | Callback JS bits | Hold-weighted JS bits |
|---|---:|---:|---:|---:|
| ball_forward | 0 | 57 | 0.083695 | 0.083914 |
| ball_right | 0 | 14 | 0.090699 | 0.091606 |
| ball_up | 0 | 22 | 0.039646 | 0.038577 |
| prediction_forward | 0 | 128 | 0.041190 | 0.040460 |
| prediction_right | 0 | 0 | 0.029740 | 0.029580 |
| prediction_up | 0 | 22 | 0.010732 | 0.010924 |
| ball_distance | 0 | 17 | 0.028939 | 0.029100 |
| car_speed | 0 | 8 | 0.086038 | 0.083336 |
| ball_present | 0 | 0 | 0.000000 | 0.000000 |
| prediction_valid | 0 | 0 | 0.000000 | 0.000000 |
| prediction_horizon | 0 | 395 | 0.239111 | 0.240642 |
| prediction_first_offset | 0 | 0 | 0.075215 | 0.074426 |
| callback_dt | 0 | 82 | 0.289587 | 0.283648 |

All-feature phase/mask-conditioned quantiles, excursions and primary/secondary histograms are preserved in the JSON report. Aggregate speed median was about 1,199.6 UU/s; median car-to-ball distance about 784.3 UU. These marginal comparisons describe the observed learner trajectory and do not establish causal distribution shift, joint-state coverage, or DAgger benefit.

Actual same-phase hold occupancy: Countdown 5.741667 s; Kickoff 2.958333 s; Active 79.866670 s. Two intervals crossing phase boundaries were excluded; final hold is censored. These are canonical-game-time weights, not proof that the wall window equaled active play.

### 4. Unobserved categories

No missing-ball callbacks, goal events, replay, pending-sequence goal/replay/countdown overlap, or gaps greater than five physics ticks were observed. Pending shadow sequences did overlap Kickoff for 71 callbacks. No probes, pauses, replay manipulation or additional run were used to force missing coverage.

### 5. Callback, queue and shutdown limits

Callback dt: median 16.6664 ms; p95 16.6702 ms; p99 25.0015 ms; max 41.6679 ms. Model inference: median 0.4394 ms; p95 0.6772 ms; p99 0.8162 ms; max 30.6122 ms.

Callback-to-submission clock: median 0 ms, p95/p99 16 ms, max 47 ms. The coarse monotonic readings must not be interpreted as zero processing cost. This metric includes shadow work, IPC/inference and diagnostic logging, and is not game-application latency.

5,238 distinct packet receipt counters and prediction receipt counters; no consecutive prediction-receipt reuse. These counts do not prove lossless physics-tick delivery. The pasted terminal excerpt contains no queue-full warning, but no complete server log is available for a global queue claim.

The supervisor stopped at its bounded completion condition; final callback 5,238 was verified submitted, worker closed, and agent_closed.json exists. Cleanup errors were empty. The SDK logged “SocketRelay disconnected unexpectedly!” during session shutdown; shutdown therefore cannot be called warning-free. agent_closed.status=running describes the internal agent before external stop, not evidence it kept running afterward.

Strict cutoff failures: [{'callback': 5237, 'offset_seconds': 90.01599999998871}, {'callback': 5238, 'offset_seconds': 90.03099999998813}]. The supervisor polls every 50 ms and stops/disconnects outside the receive loop; this implementation did not enforce a hard transport cutoff. No timing semantics were changed after the run.

### 6. Integrity and final disposition

57 diagnostic-file hashes verified; all runner code matches session metadata; all 43 protected hashes verified. Saved supervisor reports immutable V10/V12/V13 artifacts unchanged. Frozen dataset, original teacher, bot directories, input contract and prediction selector were untouched.

Zero recorded callback/transport/hidden-chain/provenance/publication errors or duplicate callback records. The timing-limit failure is additional to these automatic counts and is explicitly recorded in this review.

The pilot produced real learner-induced trajectory evidence and faithfully independent shadow behavior within the validated processed-callback scope. It does not prove competence, label usefulness, lossless delivery, or DAgger improvement. Diagnostics remain quarantined. Stop for review: no aggregation, retraining, PPO or second run.

Full machine evidence: `C:\Users\shreyas\Desktop\model wars\training\behavior_cloning\v13_dagger\phase4_pilot_v1\review_v1\phase4_pilot_review_20261005.json`. Original report and capture are retained unchanged.

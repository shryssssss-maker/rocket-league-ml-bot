# V12 offline BC architecture bakeoff

The user approved the choices in `experiment.json`. Its SHA256 is pinned in
`bakeoff.py`; changes require a new reviewed experiment. Input is the immutable
V10 manifest `080943fd8f9a81ada800e0a2f9ff3593b67f66d32272211d67030c03cf433a83`.

Run from the repository root using the existing training environment:

```powershell
& '.\training\venv\Scripts\python.exe' -u -B '.\training\behavior_cloning\v12\bakeoff.py' --check
& '.\training\venv\Scripts\python.exe' -u -B '.\training\behavior_cloning\v12\bakeoff.py' --run
```

`--check` performs only numerical probes: shapes, finite outputs, supported
eight-channel decoding, GRU chunk continuity and causality, and metric arithmetic.
It does not generate trajectories or train. `--run` verifies every frozen input
hash before fitting and after evaluation, and writes only new V12 run artifacts
and the requested V12 reports. It imports no teacher, RLBot or RocketSim.

A is a ReLU MLP, 18→128→64. B is a ReLU projection 18→64 and a one-layer causal
64-unit GRU. Both emit four mode logits and continuous tanh steering. Decoding
preserves throttle/steer/pitch/yaw/roll/jump/boost/handbrake ordering and the
teacher's supported control combinations. Frozen native target files are read
unchanged; frozen mode labels are the equivalent factorization of those targets.
Yaw/roll/boost/handbrake are constant throughout the source data and output support.

Training uses each train callback once per epoch, in frozen match order and
chronological 512-callback chunks. Short final chunks retain every callback.
The GRU resets only at match boundaries, including at the start of a new epoch's
match traversal. Chunk boundaries detach gradients without resetting hidden
values. As in stateful truncated backpropagation, hidden values computed under
the previous chunk's weights are carried after an optimizer update. There is
no phase reset, balancing, new normalization, history tensor export or augmentation.

Validation teacher-forced loss selects checkpoints, with at most 20 epochs and
patience 4. Both models' selections are persisted before test arrays or private
diagnostic labels are loaded. Test is a single final evaluation suite for each
selected checkpoint, including both context conditions. Results never tune the
configuration. Checkpoints are state dictionaries; loading uses `weights_only=True`.

Final evaluation uses CPU with one thread for both models and conditions.
Teacher-forced context remains untouched. Student-forced evaluation copies each
recorded 18D observation, replaces only prior-mode one-hot and prior steering
with the preceding predicted native action, and preserves current recorded
physics/prediction/time features. It is an offline autoregressive robustness
diagnostic, not a game played by the student.

Private sequence phase/index and missing-ball boundaries are diagnostic subset
labels only. Sequence-phase results report action agreement within each labeled
phase; there is no hidden-phase output or hidden-phase recovery claim. Boundary
neighborhoods use ±5 callbacks retrospectively for metrics, not training/input.
Long-gap metrics use dt >100 ms. Empty subsets and undefined precision are null.

Exact native agreement compares all eight decoded channels to frozen float32
analog/uint8 targets without a tolerance. A separate analog ≤1e-6/buttons-exact
metric is included. Per-channel diagnostic loss is native decoded MSE; the
optimization loss is the approved mode CE plus Chase-only steering squared error
averaged over all callbacks, with equal coefficient 1 and no class weights.

Runs are exclusively created under `runs/`; partial runs are preserved and not
resumed. If interrupted, stop for review before rerunning, particularly after
test evaluation has begun. Final report existence prevents an additional test
evaluation without review. No live deployment, teacher replacement or PPO follows.

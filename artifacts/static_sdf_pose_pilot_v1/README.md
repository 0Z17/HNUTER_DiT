# Frozen static SDF pose pilot

The selected deployable experiment weight is `models/sdf/best.pt` (Git LFS),
SHA-256 `6e0da29b562b433bc1813b2f632cdd1f0a420705b003767ae32ce31e94745192`.
It is the EMA inference state selected at step 2,000 from the **background-randomized**
seed-919 expert dataset, training seed 31. It contains no optimizer/RNG resume state.

`models/initial_sdf/best.pt` retains the earlier fixed-background experiment (data seed 811,
training seed 29), including its failed industrial migration evidence. The two experiments
are adaptive small pilots, not a statistically controlled comparison of architectures.

Both models train all parameters from random initialization, using measured nvblox TSDF/ESDF
history and the project's existing DiT block implementation. Neither loads old v003,
box, point or occupancy-voxel weights. The output schema is `hnuter_static_sdf_pose_v1`:
12 cubic controls over a fixed 20-second horizon, including position and local XYZ Euler
coordinates, with analytic world translation and body angular derivatives.

Use the SDF runtime (`--mode sdf --sdf-checkpoint ...`), not the legacy/temporal runtime.
The default WebUI model remains v003. Full reproduction, scope, negative results and timing
interpretation are in [STATIC_SDF_POSE.md](../../docs/STATIC_SDF_POSE.md).

Archived JSONs contain collection/mapping provenance, training curves, frozen test results,
native actuator-driven contact checks and industrial reports. Large per-scan lists remain in
ignored local results, with raw report hashes recorded in
[the aggregate record](../../docs/static_sdf_pose_validation.json). Both initial failures and
successful runs are retained. `final_freeze.json` binds the final checkpoint, dataset and
runtime source hashes; `verification.json` records final tests and archive checks.

The stored map tensors have six channels: TSDF, weight, ESDF, ESDF-known, TSDF-observed and
latest measured hits. Four causal snapshots use a 32³ local query grid at 0.25 m spacing;
native mapping uses 0.15 m voxels. During flight, mapping and neural inference share the
same isolated CUDA process. Only scans and spline coefficients cross runtime IPC.

Unknown map cells are not certified free. The current pilot has a narrow stationary-start,
small-attitude, static-obstacle curriculum and does not establish arbitrary industrial
generalization, moving-start replanning, dynamic-obstacle prediction or contact control.

# HNUTER DiT v003 smooth-guidance evaluation report

## Configuration

- Dataset: frozen `free_flight_159103traj_dit_v001`
- Model: canonical-frame standard Cross-Attention DiT
- Parameters: 11,964,169
- Training: 48,000 steps, batch 64, FP16 AMP, EMA
- Training collision auxiliary: weight 0.002, safe distance 0.04 m
- Best validation total loss: 0.01998806287977664 at step 48,000
- Training wall time: 2,024.5 s for the optimization loop
- Checkpoint SHA-256: `301070cd16c847d749da548bca4d66e409e7ed698346c9b938f3ddf782936dad`

## Guidance design and selection

The legacy method independently normalized and displaced all colliding waypoints.
The replacement optimizes a 16-control-point displacement field and interpolates
it across the 128-waypoint trajectory with endpoint tapering. Its objective is:

`collision + 0.10 prior + 0.25 smoothness + 0.05 jerk + 0.25 stretch + 1.00 tangent`.

Collision distance uses a 0.02 m soft minimum over the 42 robot spheres and OBBs,
which avoids discontinuously switching the single active sphere/box pair.

Guidance parameters were selected only on 30 family-stratified validation
conditions after moving the closest collision box 0.30 m toward the start-goal
segment. Each condition used four candidates and 25 DDIM steps with fixed seeds.

Selected configuration:

- final 8 DDIM steps;
- maximum smooth displacement 0.12 m per guided step;
- 0.04 m proxy safe distance and 0.05 m softplus scale;
- one final temporal smoothing pass and three inner optimization steps;
- orientation and endpoints are not modified.

Validation collision-free rate increased from 10.83% unguided to 52.50% smooth;
best-of-4 solved rate increased from 30.0% to 66.67%. Legacy pointwise guidance
reached 63.33% and 76.67%, respectively, but its mean P95 turn angle was 8.90 deg
versus 5.56 deg for smooth guidance and 4.58 deg unguided.

## Frozen test results

The test uses 100 family-stratified conditions, eight candidates per condition,
25 DDIM steps, identical latent seeds, and full HNUTER URDF + COAL audit at all
128 waypoints.

| Scenario | Variant | Collision-free | Best-of-8 | P95 turn | Max turn | Batch time |
|---|---|---:|---:|---:|---:|---:|
| Original | unguided | 39.38% | 74% | 4.58 deg | 8.10 deg | 102.1 ms |
| Original | legacy pointwise | **90.63%** | **100%** | 6.25 deg | 12.47 deg | 130.7 ms |
| Original | smooth multi-cost | 83.00% | 98% | **5.60 deg** | **9.69 deg** | 200.6 ms |
| Yaw +90 deg | unguided | 39.38% | 74% | 4.58 deg | 8.10 deg | 101.4 ms |
| Yaw +90 deg | legacy pointwise | **90.63%** | **100%** | 6.25 deg | 12.47 deg | 129.3 ms |
| Yaw +90 deg | smooth multi-cost | 83.00% | 98% | **5.60 deg** | **9.69 deg** | 200.3 ms |
| Obstacle shift 0.30 m | unguided | 14.88% | 40% | 4.71 deg | 8.28 deg | 101.5 ms |
| Obstacle shift 0.30 m | legacy pointwise | **72.88%** | **89%** | 8.28 deg | 17.40 deg | 129.7 ms |
| Obstacle shift 0.30 m | smooth multi-cost | 55.63% | 79% | **5.94 deg** | **10.24 deg** | 200.0 ms |

For obstacle shift, smooth minus unguided collision-free rate is +40.75 percentage
points with paired condition-bootstrap 95% CI `[34.38, 47.13]`; best-of-8 is
+39 points with CI `[30, 49]`. Smooth versus legacy pointwise loses 17.25 points
of collision-free rate, but reduces P95 turn by 2.34 deg, maximum turn by 7.17 deg,
acceleration RMS by 0.000820, jerk RMS by 0.000718, and backtracking fraction by
0.00109. All five shape improvements have paired bootstrap intervals excluding zero.

## Interpretation and limitations

Smooth guidance fixes the high-frequency local failure visible in the legacy
pointwise result, but it is a deliberate trade-off rather than a strict dominance:
it gives up some discrete collision-free rate and costs about 98.5 ms more per
eight-path batch than unguided inference. The three differentiable inner updates
per guided DDIM step are the main extra cost.

The displacement remains local to the DiT prior. It cannot reliably switch to a
distant homotopy when the prior goes through a deep collision or a newly closed
corridor. Such cases still require diverse decision-changing training data, a
global planner/proposal stage, or multi-start candidate selection.

Guidance uses differentiable collision spheres against OBB tokens. Final metrics
use full URDF + COAL, but only at 128 discrete waypoints. The result is still a
geometric candidate, not a continuous collision-free or dynamically feasible
guarantee. Continuous collision checking, retiming, and closed-loop validation
remain mandatory before deployment.

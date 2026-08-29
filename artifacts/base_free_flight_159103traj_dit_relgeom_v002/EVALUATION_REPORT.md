# HNUTER DiT v001 vs v002 evaluation

## Controlled setup

- Dataset: `free_flight_159103traj_dit_v001`, frozen test split.
- 100 family-stratified start/goal conditions.
- 8 samples per condition, 25 DDIM steps.
- Identical latent seeds for v001 and v002.
- Acceptance: full HNUTER URDF + COAL at all 128 generated waypoints.
- v001: absolute obstacle-token Cross-Attention.
- v002: yaw canonical frame + pairwise relative-geometry Cross-Attention +
  timestep-weighted clean-trajectory collision auxiliary loss.

## Results

| Scenario | Model | Collision-free | Best-of-8 solved | Mean minimum clearance | Mean length |
|---|---|---:|---:|---:|---:|
| Original test | v001 | 38.00% | 75.00% | -0.0462 m | 6.420 m |
| Original test | v002 | 38.38% | 68.00% | -0.0506 m | 6.387 m |
| Whole scene yaw +90° | v001 | 11.50% | 21.00% | -0.3297 m | 8.440 m |
| Whole scene yaw +90° | v002 | 38.38% | 68.00% | -0.0506 m | 6.387 m |
| Relevant obstacle shifted 0.30 m | v001 | 13.38% | 41.00% | -0.1262 m | — |
| Relevant obstacle shifted 0.30 m | v002 | 14.13% | 39.00% | -0.1386 m | — |

Yaw-equivariance error:

| Model | Position RMS | Rotation RMS |
|---|---:|---:|
| v001 | 0.51371 m | 25.575° |
| v002 | 0.00030 m | 0.020° |

Paired, condition-level bootstrap differences for v002 minus v001:

| Scenario | Candidate collision-free delta | 95% CI | Best-of-8 delta | 95% CI |
|---|---:|---:|---:|---:|
| Original test | +0.375 pp | [-2.25, +3.125] pp | -7.0 pp | [-15.0, +1.0] pp |
| Whole scene yaw +90° | +26.875 pp | [+19.75, +34.375] pp | +47.0 pp | [+36.0, +57.0] pp |
| Obstacle shifted 0.30 m | +0.75 pp | [-1.75, +3.50] pp | -2.0 pp | [-10.0, +6.0] pp |

## Interpretation

The canonical frame is successful: v002 is effectively yaw-equivariant and
retains its original test performance after a 90° world rotation, while v001
fails sharply.

The obstacle-understanding result is not yet successful. On both the original
test split and the controlled obstacle-shift stress test, v002's collision-free
rate is statistically indistinguishable from v001. Its mean trajectory response
to the moved box is also smaller (`0.0483 m` RMS versus `0.0588 m`), so the
current result does not support a claim that pairwise attention learned stronger
decision-changing obstacle behavior.

This matches the data limitation: the frozen demonstrations do not provide the
same start/goal with counterfactual obstacle placements that force another
homotopy. The new architecture removes a coordinate shortcut and the auxiliary
loss teaches local geometric consistency, but neither supplies missing
counterfactual route decisions. The next safety-oriented step should therefore
be inference-time collision guidance (or candidate optimization/ranking), and
the next learning-oriented step should be decision-changing paired data when it
becomes possible to collect it.

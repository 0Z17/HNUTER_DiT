# Geometry encoder pilot, 2026-10-04

Three freshly trained temporal DiT checkpoints, each trained for 2,000 steps on
the same seed-107 dataset: 128 training, 32 validation, 32 test episodes, split
by 96 complete paired groups. The `models/{box,point,voxel}/best.pt` files are
validation-selected EMA inference weights tracked through Git LFS. They do not
contain optimizer or RNG state for training resumption.

All models generate 16 cubic position-spline controls over eight fixed physical
seconds. They hold measured attitude; they are not six-DoF contact policies.
Point and voxel inputs use four measured LiDAR scans. Voxels encode occupied,
ray-free evidence and unknown cells, not a fused nvblox TSDF map.

Validation selected **voxel** before the independent test. On 32 test episodes,
the feasible/safe/progress rates were box 68.75%, point 0%, voxel 65.625%.
Voxel with repeated latest-frame history scored 21.875%. The voxel-minus-box
95% whole-group bootstrap interval spans zero; superiority is not established.
The point model is retained as a failed experimental result, not a deployment
recommendation. Full methods, physical validation, video and limitations are in
[the experiment report](../../docs/GEOMETRY_ENCODER_PILOT.md).

The selected voxel model completed both 30-second industrial simulations with
zero actual contacts. Incoming reached the target with guard intervention;
outgoing stopped 0.52484 m short and did not reach it. Both runs rejected plans.
The full outgoing video retains the failed completion. No scale-up or
replacement of the existing box baseline is supported by this pilot.

Training metadata, data freeze, source hashes and checkpoint hashes are retained
here. Original source-location strings are provenance from the experiment
worktree. Raw data, intermediate checkpoints, logs and rendered frames remain
local under `results/geometry_pilot/`; they are intentionally excluded from Git.
Existing v003 and the previous temporal pilot weights are unchanged.

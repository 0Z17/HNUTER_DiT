# Geometry latency and nvblox conditioning analysis

Measured 2026-10-04. This supplements the frozen [geometry pilot](GEOMETRY_ENCODER_PILOT.md).
No training, checkpoint selection, planner replacement, or industrial flight was performed in this follow-up.

## Fair latency comparison

The previous comparison did not measure the complete cost of constructing each obstacle representation.
Previously saved box tokens cannot be treated as free preprocessing while point/voxel inputs are rebuilt.
This benchmark rebuilds **all three representations from the same resident measured-return windows**.

RTX 4080 Laptop GPU, Torch 2.13.0+cu130, batch 1, 20 DDIM steps, four historical scans,
32 held-out windows / 16 paired groups, three repeats, five warm-up windows per representation.
Representations are interleaved in randomized order. Each scan has a median 7,528 valid returns
from the 20,000 emitted rays. CUDA synchronization bounds timing; file reads and reconstruction
of XYZ returns from stored ranges happen outside timing. Models and dataset remain frozen.
The workstation was not reserved exclusively; desktop/background GPU load can affect results.

| Representation | CPU construction, median ms | Public model runtime, median ms | Measured total, median ms | Total P95 ms |
| --- | ---: | ---: | ---: | ---: |
| Box | 57.81 | 34.15 | 92.13 | 95.29 |
| Point | 10.62 | 34.33 | 44.85 | 46.78 |
| Voxel | 118.60 | 38.52 | 156.99 | 161.96 |

The public runtime includes input validation, transfer, history encoding, diffusion and CPU trajectory
export. Total percentiles are measured directly; sums of stage medians/P95s are not total percentiles.
Excluded: ray acquisition, IPC, nvblox mapping, guard checks, controller, physics and rendering.
These results therefore do **not** establish the complete system's wall-clock planning frequency.
Against a hypothetical 100 ms / 10 Hz planning budget, voxel already exceeds the budget and box
leaves very little room for the omitted work. Point is fastest here but failed the pilot's quality
evaluation and cannot be recommended for deployment on latency alone.

Separate synchronized diagnostic timings:

| Stage, median ms | Box | Point | Voxel |
| --- | ---: | ---: | ---: |
| CPU validation | 0.121 | 0.149 | 3.253 |
| Host-to-device transfer | 0.192 | 0.139 | 0.643 |
| Complete history encoder | 0.877 | 0.901 | 1.096 |
| Condition MLP | 0.065 | 0.063 | 0.073 |
| Cached 20-step DDIM + spline | 32.950 | 32.963 | 33.038 |
| Device-to-host trajectory | 0.057 | 0.056 | 0.057 |

The history encoder includes the spatial encoder, position/time embeddings and temporal Transformer.
It runs **once per sample**, with context reused across DDIM steps. The diagnostic denoising loop
was checked against the production sampler: maximum absolute output error was zero for all three
representations. CPU input payloads are 5,328 / 26,704 / 4,194,384 bytes respectively; these are not
peak GPU memory measurements. The dense voxel input also incurs full-array CPU validation.

**The current voxel bottleneck is CPU construction of four occupancy/free/unknown grids, not the
3D neural encoder.** Voxel representation alone confers no real-time advantage. Box construction
also has a material cost. GPU voxelization, incremental integration and reusing geometry/features
could change this comparison; none is enabled by merely substituting a representation name.

The existing voxel model uses independent measured-scan occupancy/free/unknown grids, not nvblox
TSDF or ESDF. Its weights cannot be interpreted as a trained TSDF encoder.

## Measured nvblox GPU interface probe

A separate experiment uses the project's pinned official nvblox wheel and the existing NativeMapper
calibration workaround, without changing Isaac's Torch. Environment: nvblox
0.0.10.dev1+cu12ubuntu24, Torch 2.9.1+cu128, same GPU. This is **not a fourth trained policy**.

Eight held-out windows, four chronological frames per window, three repeats, eight warm-up frames;
96 measured updates. A map is cleared before each window, then incrementally fused across its four
frames. Resolution 0.25 m, range limit 20 m, ESDF maximum distance 5 m, spherical image 720×128
with padded vertical FOV 104 degrees. Each update queries TSDF and ESDF at all 262,144 positions
of a 64³ local grid and packs GPU distance/weight/known features. Grid coordinates and buffers
are prepared outside timing. Output remains on the GPU.

| Operation | Median ms | P95 ms |
| --- | ---: | ---: |
| CPU spherical projection | 0.493 | 0.576 |
| Transfer + TSDF integration | 0.283 | 0.987 |
| ESDF update | 0.638 | 1.361 |
| Full-grid TSDF + ESDF query | 0.055 | 0.076 |
| GPU feature packing | 0.129 | 0.301 |
| **Measured per-frame total** | **1.647** | **2.566** |

Maximum measured frame total: 8.377 ms. Every output was finite; the ESDF-known count inside the
8 m spherical ROI ranged from 2,888 to 10,029. Most grid locations remain unknown and must not
be interpreted as free space. This checks execution and nonempty observations, **not distance
accuracy, dynamic-obstacle handling, or downstream policy quality**.

This is a short, small-map replay with no competing Isaac render/physics work. It excludes startup,
map clearing, query-grid generation, CPU export, IPC, neural encoding and diffusion. A continually
growing industrial map may cost more. The four fused snapshots have different semantics from four
independent occupancy scans. In particular, **1.65 ms per update cannot be directly divided into
118.60 ms per four-frame reconstruction to claim an equivalent-policy speedup**. The two experiments
also use different Torch/CUDA environments. The result supports investigating GPU map reuse,
not claiming that a complete nvblox-conditioned DiT already runs in that time.

## Existing nvblox-compatible encoding routes

1. **Local distance-field tensor + small 3D encoder.** Query a bounded local region into GPU tensors;
   retain TSDF distance/weight, ESDF distance and explicit observation masks. Add timestamps only
   if actually tracked. This fits the existing DiT cross-attention interface after a new encoder
   and controlled retraining. TSDF is a truncated projective surface distance, not a substitute for
   Euclidean collision clearance. The probe above demonstrates the query interface, not training.
2. **Sparse block tokens.** Official nvblox provides direct Torch views of allocated 8³ voxel blocks.
   A shared block encoder plus spatial coordinates could produce sparse tokens for DiT. Creating
   views avoids copying the blocks, but stacking/encoding may still allocate; this is not automatic
   cross-process zero-copy. Reacquire views after map clearing/decay. Avoid converting the entire
   global map AABB to a dense tensor. [Official voxel access documentation](https://nvidia-isaac.github.io/nvblox/v0.0.10/pages/torch_examples_voxel_access.html).
3. **Feature-map tokens + diffusion policy.** NVIDIA's *mindmap* combines nvblox reconstruction and
   fused visual features with a 3D diffusion action policy. It encodes reconstructed surface points
   with features alongside current observations; it does not feed every dense voxel directly into
   a DiT. Its simulation manipulation results support architectural compatibility, not our UAV's
   real-time or contact-control performance. [Primary paper](https://arxiv.org/abs/2509.20297).

nvblox supplies map integration and access; it is not itself a pretrained learned obstacle encoder.
Its official deep-feature example uses AM-RADIO features and reports more than 10 GB GPU memory,
so that complete example is a poor first fit for a 12 GB GPU shared with Isaac. LiDAR-only geometry
conditioning is the smaller first experiment; visual features would additionally require synchronized
camera/depth association. [Official feature example](https://nvidia-isaac.github.io/nvblox/v0.0.10/pages/torch_examples_deep_features.html).

nvblox also supports differentiable ESDF queries for trajectory optimization. This offers a separate
collision-cost/gradient path in addition to learned conditioning; adding it would require measuring
its cost and validating constraints. [Official gradient example](https://nvidia-isaac.github.io/nvblox/v0.0.10/pages/torch_examples_gradients.html).

## Recommended next controlled change

First optimize the **mapping-to-encoder data path**, retaining the present DiT as the backbone:
keep bounded local queries and the encoder in the nvblox-compatible GPU process, avoid exporting
large dense grids through NumPy/file IPC, and reuse unchanged map blocks. This requires dependency
and runtime verification in the pinned worker environment; the present Torch environments differ.
It is a proposal, not an implemented production path.

Keep short-term time-stamped scan/occupancy history alongside background TSDF features. The current
worker uses one static TSDF mapper and does not separate moving objects; accumulated surfaces can
leave stale evidence. TSDF weight is neither an observation timestamp nor obstacle velocity. Do not
erase the dynamic history under the assumption that fusion alone encodes motion. nvblox core has
dynamic mapping facilities, but they are not enabled in this project's worker.

Train a new small comparison when the input semantics change, keeping the box baseline, fixed
splits/budgets, actual contacts and reference-limit failures. Future tuning needs fresh held-out
evaluation rather than repeatedly selecting on this test set. Measure P50/P95 latency for acquisition,
mapping, local extraction, encoding, diffusion, IPC and guard checks under concurrent simulation;
also report observation age, dropped frames and planning deadline misses.

For later contact tasks, local geometry/features can remain useful conditioning. The present 0.25 m
grid and position-only trajectory output are insufficient for precise contact: add finer near-tool
geometry, full pose/action representations, intended-contact handling and force/impedance feedback,
with separate validation. Existing manipulation diffusion research does not establish those abilities
for this drone.

## Evidence and reproduction

- [Frozen-policy latency summary](geometry_latency_validation.json), including source/checkpoint hashes.
- [nvblox interface probe summary](nvblox_roi_latency_probe.json), including worker/script hashes.
- Raw measurements and successful/failed startup logs are kept locally under
  `results/geometry_latency_20261004/`, excluded from Git.
- [Policy benchmark](../tools/benchmark_geometry_latency.py) and
  [nvblox probe](../tools/benchmark_nvblox_roi.py) preserve the executed script bytes.

Run from the model repository with a **new** output directory. The original local pilot dataset is
required; it is not contained in the Git clone.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 \
  /home/z017/research/curobo_env/bin/python tools/benchmark_geometry_latency.py \
  --output results/geometry_latency_repeat
```

The interface probe intentionally uses container paths `/sim`, `/data`, `/out` and the project's
existing pinned perception image. Create a fresh output directory before running:

```bash
mkdir results/nvblox_roi_latency_repeat
docker run --rm --gpus all \
  --mount type=bind,source=/home/z017/research/isaacsim-industrial-reactor,target=/sim,readonly \
  --mount "type=bind,source=$PWD,target=/data,readonly" \
  --mount "type=bind,source=$PWD/tools/benchmark_nvblox_roi.py,target=/probe.py,readonly" \
  --mount "type=bind,source=$PWD/results/nvblox_roi_latency_repeat,target=/out" \
  -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=2 -e PYTHONDONTWRITEBYTECODE=1 \
  -e LD_LIBRARY_PATH=/opt/nvblox-venv/lib/python3.12/site-packages/nvidia/npp/lib \
  --entrypoint /opt/nvblox-venv/bin/python isaaclab-sim:6.0.0-nvblox /probe.py
```

Initial probe launches exposed an import-path omission and a missing NPP library path; both were
corrected using the existing worker launch configuration before the successful measurement. No
dependency upgrade was performed. Only benchmark scripts and documentation are added by this
follow-up; production code, trained weights and the simulator repository are unchanged.

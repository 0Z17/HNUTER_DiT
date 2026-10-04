# Static SDF flight demo and loop profiling

本轮已完成完整工业测试录像：**24,000 个真实物理步，零碰撞，位置误差 1.57 cm、姿态误差约 1°**。完整 0–24 秒仿真对应 241 帧，MP4 为 24.1 秒、1920×1200、10 fps，约 4.2 MiB。带双相机录像的飞行循环用时 **92.96 秒墙钟**，其中 Fabric/渲染/图像保存累计 **44.68 秒**；初始化不在该数值内。与无录像的 50.93 秒测试分开报告。

[播放/下载 MP4](../results/static_sdf_demo/export/hnuter_static_sdf_demo.mp4) · [验证记录](../artifacts/static_sdf_demo_v1/video_qa.json)

瓶颈主要在**仿真接口、GPU/CPU 同步和 Python 闭环开销**，不能简单等同于纯物理求解器，也不能把参考样条求值和 SE(3) 控制耗时算成 DiT 网络推理。下面给出逐段实测与边界。


This follow-up records the frozen `6e0da29b...45192` checkpoint on industrial fixture 100. No retraining, controller gain changes, timestep changes, pose replay, or relaxed perception timeout is used.

## What limits real time?

A new 24-second, 24,000-step headless run with phase profiling and no in-loop rendering completed in **50.927 seconds**. It reached the requested pose with zero contacts. After the first simulation second, host wall-time attribution was:

| Phase | Measured time | Share |
| --- | ---: | ---: |
| Contact readback and collision evaluation | 16.892 s | 37.97% |
| Reference sampling/polling, controller and actuator writes | 13.368 s | 30.05% |
| State readback and loop checks | 7.509 s | 16.88% |
| Physics step call and timestep validation | 3.850 s | 8.65% |
| LiDAR capture and asynchronous submission | 2.469 s | 5.55% |
| Telemetry, progress and pacing | 0.370 s | 0.83% |
| Render branch bookkeeping (no rendering) | 0.030 s | 0.07% |

These are **host wall-clock phases**, not CUDA kernel profiling: physics is asynchronous, so a later `.numpy()` readback can wait for work launched by an earlier physics call. It is incorrect to interpret the 8.65% as the total GPU physics cost. It is also incorrect to attribute the full 30.05% control phase to diffusion inference. The DiT/nvblox worker runs concurrently; the main control loop mostly evaluates the accepted spline and drives the existing SE(3) controller.

The first physics step spent approximately **4.45 s** loading CUDA work. That cold cost is retained separately from the steady comparison. Without that first simulation second, 23 simulation seconds still required about **44.49 s** of profiled host time. Cold initialization therefore does not explain the ongoing slowdown.

The source explains the substantial synchronization/readback workload: `NewtonHnuterBridge.read_state()` reads body transforms and velocities to NumPy each millisecond, `apply()` reads and rewrites control arrays, and the independent collision evaluator reads contact counts and arrays every physics step. Preserving collision verification and controller conventions is essential; reducing these transfers or performing compact contact reductions on GPU should be benchmarked before changing physics parameters. The existing 1 ms integration and original gains are unchanged.

The prior frozen trials measured DiT inference at 51–56 ms and first complete planning response at 97–100 ms. This pilot requests one stationary-start plan and continues sensing/ESDF monitoring. **It does not measure sustained 10 Hz replanning**, and rendering shares GPU resources with the mapper/model. The defensible conclusion is that the current simulator/control/readback loop is the dominant bottleneck, rather than the once-per-flight diffusion call; pure solver cost has not been isolated.

## Recording and reproducibility

```bash
./run_industrial.sh --headless --mode sdf \
  --sdf-checkpoint artifacts/static_sdf_pose_pilot_v1/models/sdf/best.pt \
  --sdf-scene-index 100 --duration 24 --max-speed .65 --max-acceleration .9 \
  --seed 23 --real-time --profile-loop --record-video --video-fps 10 \
  --demo-perception --output results/static_sdf_demo/recording_final

python tools/export_sdf_demo.py --run results/static_sdf_demo/recording_final \
  --output results/static_sdf_demo/export/hnuter_static_sdf_demo.mp4 \
  --ffmpeg /path/to/ffmpeg
```

For the no-video profile, omit `--record-video --video-fps 10 --demo-perception` and use a new output directory.

The MP4 has four synchronized panels: actual follow camera, actual FPV camera, measured MID-360 approximation point returns, and an actual nvblox ESDF horizontal slice. The point panel uses local XY and colors measured returns by world height. The ESDF grid remains at its fixed planning anchor with an explicit unknown mask; distances displayed above 3 m are clipped. Neither panel loads evaluator obstacle boxes. The blue line is the accepted model plan and is only shown after its actual activation; the white line is the measured trajectory.

An optional diagnostic flag copies a 32×32 ESDF slice from GPU to disk. It never becomes a model input, and its additional export cost is recorded by the worker. Normal execution retains the existing GPU-local map path. Per-frame manifests bind RGB, physics steps, measured states, scan timestamps and the exact latest map response available to the controller; export rejects timestamp/hash mismatches. A cloud can be slightly older than its image, and its actual age is displayed.

All frames, including initial history accumulation and terminal hold, are retained. **One video second equals one simulation second, not one recording wall-clock second.** Rendering and PNG capture add overhead; the recording is not a real-time benchmark. Raw frames, MP4, scan data and logs are ignored by Git and retained under `results/static_sdf_demo/`.

The original follow camera was visibly occluded by a test obstacle in the first recording; that complete successful flight is retained as `results/static_sdf_demo/recording`. The final demo uses a session-only rear/upper camera at body-local (-3, 1.5, 2.2) m, looking at the body origin with a 24 mm lens. This changes neither physical motion nor saved simulator assets.

Final verification: 22 targeted tests passed; both recording runs and the independent no-video profile reached the pose without contact. All 241 primary camera state records equal the physics log exactly, all displayed sensor/map timestamps are causal, and the entire MP4 decoded successfully (241 frames). The final views at simulation 0, 5, 8, 15 and 24 seconds were visually checked.

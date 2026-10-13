# 交互执行的冻结验证资料

本目录随 Git 发布。测量完成于 2026-10-12，整理于 2026-10-13；系统构建与限制见
[交互系统文档](../../docs/INTERACTIVE_SYSTEM.md)。这些是固定输入的完整物理仿真结果，
不能外推为所有随机场景的无碰撞保证。

| 文件 | 用途 |
|---|---|
| `comparison.json` | 同场景、同候选、同 MPPI seed 的前后对照，以及默认 UI 场景结果 |
| `plan_seed_42.json` / `plan_seed_123.json` | 完整场景、端点和 8 条候选；用于复现两个对照条件 |
| `plan_default_ui_seed_42.json` | 随机场景族、scene seed 42、inference seed 20260828 的默认 UI 条件 |
| `before_seed_*.json` / `after_seed_*.json` | 指标摘要；控制周期原始记录留在本地 smoke 归档 |
| `after_default_ui_seed_42.json` | 默认 UI 自动选择候选后的闭环结果 |
| `legacy_mujoco_execution.py` | 旧版逐 waypoint 停顿参考和控制流程的测量快照，仅用于基准复现 |
| `webui_toppra.jpg` | 新版 UI 完成执行的截图 |
| `manifest.json` | 文件校验哈希、依赖版本、外部仓库固定 revision 和数据整理说明 |

当前代码的复现命令（在仓库根目录执行；路径可替换为自己的外部仓库）：

```bash
MPLCONFIGDIR=/tmp/hnuter_benchmark_matplotlib \
/home/z017/research/curobo_env/bin/python -m hnuter_dit.benchmark_interactive_flight \
  --plan artifacts/interactive_execution_validation/plan_seed_42.json \
  --baseline artifacts/interactive_execution_validation/before_seed_42.json \
  --collector-root /home/z017/research/HNUTER_trajectory_collector \
  --mujoco-root /home/z017/research/MuJoCo-for-HNUTER \
  --output /tmp/hnuter_after_seed_42.json
```

重跑旧版，可加 `--module-source artifacts/interactive_execution_validation/legacy_mujoco_execution.py`，
省略 `--baseline`，并输出至另一个文件。普通交互执行始终使用新版 TOPP-RA。
基准运行不加载 DiT checkpoint，也不重新随机采样；输入规划中的 `final_candidate_id`、完整
plan hash、scene hash 和 MPPI seed 必须匹配。外部 collector、URDF、MuJoCo 模型、meshes
及控制库仍须可用。新流程会对实际平滑曲线使用 COAL 真实距离后端复检。

速度上限由旧版 0.6 m/s 改为新版 1.0 m/s，参考和执行器控制也同时改变；因此对照反映整条
执行流程的改善，不能只归因于 TOPP-RA。飞行时长使用仿真时间，CPU 墙钟时间会随机器负载变化。

原始逐步数据位于被 Git 忽略的 `artifacts/interactive_flight_smoke/`，不是克隆后运行所需的文件。
后续实验应写新输出，不覆盖本目录的冻结资料。

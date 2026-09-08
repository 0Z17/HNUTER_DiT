# HNUTER DiT Trajectory Lab

本仓库提供 HNUTER free-flight Base DiT 的模型文件、训练实现、对照评测与本地 WebUI。
界面可以随机生成训练分布内的十类 OBB 场景和起终点位姿，也支持手工增删、
移动和缩放障碍物，编辑起终点 XYZ/roll/pitch/yaw，并一次生成多条 SE(3)
轨迹进行三维对比。

## 模型

当前默认模型为 v003：

```text
artifacts/base_free_flight_159103traj_dit_canonical_cross_v003/models/dit_cross/best.pt
```

- `dit_cross` / canonical standard Cross-Attention
- 11,964,169 parameters
- 128 × pose9：`xyz + rotation-6D`
- canonical frame 中的 32 × 10 OBB environment tokens
- 48,000 training steps
- validation diffusion loss `0.0199700`
- validation collision loss `0.00901`
- validation total loss `0.0199881`
- checkpoint SHA-256
  `301070cd16c847d749da548bca4d66e409e7ed698346c9b938f3ddf782936dad`

v003 保留了 v002 中有效的：

- 起点为原点、起点到终点水平投影为 +x、世界竖直方向为 +z 的 yaw-aligned canonical frame；
- 在恢复的 clean trajectory `x0_hat` 上计算 42-sphere/OBB collision auxiliary loss；
- 由 `alpha_bar(t)` 调度的碰撞损失权重，高噪声 timestep 自动减弱。

v003 移除了未证明有效且开销较高的 pairwise relative-geometry attention，恢复为
PyTorch 标准 Cross-Attention。推理时可在最后若干 DDIM step 对 `x0_hat` 使用
42-sphere/OBB 可微碰撞 guidance。当前默认是 validation split 选出的新版平滑
多目标 guidance：最后 8 步、每步最大平滑位移 0.12 m，由 16 个控制点定义整条
位移场，每步做 3 次优化。代价同时包含碰撞、偏离 prior、二阶平滑度、jerk、
路径段拉伸和切向反转。姿态和起终点不会被修改，最终仍由完整 URDF + COAL 验碰。

v002 实验模型仍保留在：

```text
artifacts/base_free_flight_159103traj_dit_relgeom_v002/models/dit_rel_cross/best.pt
```

旧 v001 基线仍保留在：

```text
artifacts/base_free_flight_159103traj_dit_v001/models/dit_cross/best.pt
```

`best.pt` 是按 validation loss 选择的 EMA 推理模型；`last.pt` 保存 optimizer、
scheduler、EMA、sampler 和 RNG 状态，可用于精确恢复训练。

### v001 / v002 / v003 对照结果

评测使用冻结数据集的同一 test split：100 个按场景族分层抽取的条件，每个条件
8 条候选、25 DDIM steps、相同 latent seed，并使用完整 HNUTER URDF + COAL
检查全部 128 个 waypoint。

| 场景 | 模型 | Collision-free | Best-of-8 | P95 转角 | 最大转角 | Batch time |
|---|---|---:|---:|---:|---:|---:|
| 原始 test | v003 unguided | 39.38% | 74% | 4.58° | 8.10° | 102.1 ms |
| 原始 test | v003 legacy pointwise | **90.63%** | **100%** | 6.25° | 12.47° | 130.7 ms |
| 原始 test | v003 smooth multi-cost | 83.00% | 98% | **5.60°** | **9.69°** | 200.6 ms |
| 整体 yaw +90° | v003 unguided | 39.38% | 74% | 4.58° | 8.10° | 101.4 ms |
| 整体 yaw +90° | v003 legacy pointwise | **90.63%** | **100%** | 6.25° | 12.47° | 129.3 ms |
| 整体 yaw +90° | v003 smooth multi-cost | 83.00% | 98% | **5.60°** | **9.69°** | 200.3 ms |
| 障碍物平移 0.30 m | v003 unguided | 14.88% | 40% | 4.71° | 8.28° | 101.5 ms |
| 障碍物平移 0.30 m | v003 legacy pointwise | **72.88%** | **89%** | 8.28° | 17.40° | 129.7 ms |
| 障碍物平移 0.30 m | v003 smooth multi-cost | 55.63% | 79% | **5.94°** | **10.24°** | 200.0 ms |

表中的 batch time 是一次并行生成 8 条、25 DDIM steps 的神经采样时间，包含
对应 guidance，不包含随后执行的 COAL 验碰。新版约增加 98.5 ms/batch，相对
unguided 约 2 倍；这是每个 guided step 内 3 次可微几何优化的代价。

障碍物平移场景中，smooth guidance 相对 unguided 的 collision-free 增益为
`+40.75 pp`，95% CI `[+34.38, +47.13] pp`。相对旧逐点 guidance，它牺牲
`17.25 pp` collision-free，但 P95 转角下降 `2.34°`、最大转角下降 `7.17°`，
二阶差分 RMS 下降 `0.000820`、jerk RMS 下降 `0.000718`，回退比例也显著降低。
因此新版默认优先解决截图中的局部尖折、环绕和往返振荡；若只追求离散点碰撞率，
旧版数值更高，但不再作为 WebUI 默认值。

v002 的 yaw 等变位置 RMS 为 `0.00030 m`，v001 为 `0.514 m`；因此 canonical
frame 明确解决了世界横向/纵向旋转导致的失效。但 v002 在原始 IID 和障碍物偏移
测试中的 collision-free 改善不显著。障碍物偏移候选率配对差为 `+0.75 pp`，
95% bootstrap CI 为 `[-1.75, +3.50] pp`。这说明 relative-geometry attention 与
训练期 collision loss 还不能替代 decision-changing data 或 inference guidance。

v003 完整记录见：

```text
artifacts/base_free_flight_159103traj_dit_canonical_cross_v003/guidance_tuning_smooth_validation.json
artifacts/base_free_flight_159103traj_dit_canonical_cross_v003/comparison_v003_smooth_guidance.json
artifacts/base_free_flight_159103traj_dit_canonical_cross_v003/EVALUATION_REPORT.md
```

### 模型版本与仓库体积

仓库只通过 Git LFS 版本化当前可部署的 `best.pt`。最新 EMA 快照
`ema.pt` 和包含优化器、调度器与 RNG 状态的 `last.pt` 默认留在本地训练
存储中，不进入 Git 历史。训练配置、验证指标、checkpoint 选择记录和 SHA-256
仍作为普通文本文件提交，以便追踪与复现每次训练。

长期迭代时，不要把每个 epoch 或 step 的 checkpoint 都提交到 Git。只保留少量
正式发布模型，并优先通过 GitHub Release、Hugging Face Hub、对象存储或 DVC
管理历史权重；仓库中保存对应下载地址、版本、指标和校验哈希。

## 启动 WebUI

使用已经验证过 PyTorch/CUDA、NumPy、COAL 的环境：

```bash
cd /home/z017/research/HNUTER_DiT
./run_webui.sh --device cuda
```

浏览器打开 <http://127.0.0.1:8790>。

也可以使用 CPU 验证功能，推理速度会明显更慢：

```bash
./run_webui.sh --device cpu
```

默认场景逻辑来自：

```text
/home/z017/research/HNUTER_trajectory_collector
```

可通过 `--collector-root` 和 `--checkpoint` 指向其他兼容位置。

要临时切回 v001：

```bash
./run_webui.sh --device cuda \
  --checkpoint artifacts/base_free_flight_159103traj_dit_v001/models/dit_cross/best.pt
```

要关闭 guidance，在 WebUI 中把 `Guidance 后期步数` 设为 `0`。

## 训练与对照评测

重新训练或从 `last.pt` 恢复 v003：

```bash
./run_training_v003.sh
```

在 validation split 上重新选择 guidance 参数：

```bash
/home/z017/research/curobo_env/bin/python -m hnuter_dit.tune_guidance --device cuda
```

运行 v001/v002/v003/guidance 的完整统一评测：

```bash
/home/z017/research/curobo_env/bin/python -m hnuter_dit.evaluate_v003 --device cuda
```

运行原始 test + 90° yaw 等变性对照：

```bash
/home/z017/research/curobo_env/bin/python -m hnuter_dit.evaluate_models \
  --condition-count 100 --samples-per-condition 8 --device cuda
```

运行关键障碍物 0.30 m 偏移测试：

```bash
/home/z017/research/curobo_env/bin/python -m hnuter_dit.evaluate_obstacle_shift \
  --condition-count 100 --samples-per-condition 8 --device cuda
```

## WebUI 功能

- 随机选择或指定十类 free-flight 场景族；
- 使用可复现 seed 随机生成 8–18 个 OBB 障碍物；
- 在采样区域内随机生成碰撞自由的起终点位置和姿态；
- 直接编辑起终点 XYZ 和 roll/pitch/yaw；
- 添加、删除、移动、缩放或旋转 box 障碍物；
- 在三维画布上直接拖动障碍物和 START/GOAL，`Alt+拖动` 调整高度；
- 双击画布空白位置直接放置新的 box 障碍物；
- 选择 10/25/50/100 DDIM steps，一次生成 1–16 条候选；
- 可交互关闭或调整最后若干 DDIM step 的碰撞 guidance 与单步位移上限；
- 透视、顶视和前视三维显示，支持旋转、平移、缩放与轨迹动画；
- 对每条轨迹显示长度、最小净空、碰撞点、越界点和姿态超限点；
- 使用完整 HNUTER URDF + COAL 检查 128 个生成 waypoint；
- 导出当前场景/端点和全部预测结果 JSON。

绿色轨迹表示当前 128 个 waypoint 通过完整 URDF/COAL、采样边界和飞行姿态
限制检查。它仍然是无时间参数的几何候选；部署前必须进行连续碰撞复检、
时间参数化和闭环控制验证。

## HTTP API

```text
GET  /api/catalog
POST /api/scene/random
POST /api/endpoints/random
POST /api/validate
POST /api/sample
```

静态页面和 API 都只使用 Python 标准库 HTTP server，不依赖 npm 或 CDN。

## 验证

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/z017/research/curobo_env/bin/python -m pytest -q
node --check hnuter_dit/web_assets/app.js
```

本项目及其参考实现采用 GPL-3.0 许可证。

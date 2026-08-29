# HNUTER DiT Trajectory Lab

本仓库提供 HNUTER free-flight Base DiT 的模型文件、训练实现、对照评测与本地 WebUI。
界面可以随机生成训练分布内的十类 OBB 场景和起终点位姿，也支持手工增删、
移动和缩放障碍物，编辑起终点 XYZ/roll/pitch/yaw，并一次生成多条 SE(3)
轨迹进行三维对比。

## 模型

当前默认模型为 v002：

```text
artifacts/base_free_flight_159103traj_dit_relgeom_v002/models/dit_rel_cross/best.pt
```

- `dit_rel_cross` / `small_environment_relative_geometry`
- 12,002,121 parameters
- 128 × pose9：`xyz + rotation-6D`
- 32 × 10 OBB environment tokens
- 48,000 training steps
- validation diffusion loss `0.0200399`
- validation collision loss `0.0105369`
- validation total loss `0.0200610`
- checkpoint SHA-256
  `c47f2a1e2ae962a766efb6bb5153b380c51b9c74557efc5cb181f92a1da6a502`

v002 相对 v001 增加了：

- 起点为原点、起点到终点水平投影为 +x、世界竖直方向为 +z 的 yaw-aligned canonical frame；
- trajectory token 与 obstacle token 之间的 13 维 pairwise relative geometry；
- relative geometry 对 attention logits 和 value message 的共同调制；
- 在恢复的 clean trajectory `x0_hat` 上计算 42-sphere/OBB collision auxiliary loss；
- 由 `alpha_bar(t)` 调度的碰撞损失权重，高噪声 timestep 自动减弱。

旧 v001 基线仍保留在：

```text
artifacts/base_free_flight_159103traj_dit_v001/models/dit_cross/best.pt
```

`best.pt` 是按 validation loss 选择的 EMA 推理模型；`last.pt` 保存 optimizer、
scheduler、EMA、sampler 和 RNG 状态，可用于精确恢复训练。

### v001 / v002 对照结果

评测使用冻结数据集的同一 test split：100 个按场景族分层抽取的条件，每个条件
8 条候选、25 DDIM steps、相同 latent seed，并使用完整 HNUTER URDF + COAL
检查全部 128 个 waypoint。

| 场景 | 模型 | Collision-free | Best-of-8 solved | Mean min clearance |
|---|---:|---:|---:|---:|
| 原始 test | v001 | 38.00% | 75.00% | -0.0462 m |
| 原始 test | v002 | 38.38% | 68.00% | -0.0506 m |
| 整体 yaw +90° | v001 | 11.50% | 21.00% | -0.3297 m |
| 整体 yaw +90° | v002 | 38.38% | 68.00% | -0.0506 m |
| 关键障碍物向路线移动 0.30 m | v001 | 13.38% | 41.00% | -0.1262 m |
| 关键障碍物向路线移动 0.30 m | v002 | 14.13% | 39.00% | -0.1386 m |

v002 的 yaw 等变位置 RMS 为 `0.00030 m`，v001 为 `0.514 m`；因此 canonical
frame 明确解决了世界横向/纵向旋转导致的失效。但 v002 在原始 IID 和障碍物偏移
测试中的 collision-free 改善不显著。障碍物偏移候选率配对差为 `+0.75 pp`，
95% bootstrap CI 为 `[-1.75, +3.50] pp`。这说明 relative-geometry attention 与
训练期 collision loss 还不能替代 decision-changing data 或 inference guidance。

完整记录见：

```text
artifacts/base_free_flight_159103traj_dit_relgeom_v002/comparison_v001_vs_v002.json
artifacts/base_free_flight_159103traj_dit_relgeom_v002/obstacle_shift_comparison.json
artifacts/base_free_flight_159103traj_dit_relgeom_v002/EVALUATION_REPORT.md
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

## 训练与对照评测

重新训练或从 `last.pt` 恢复 v002：

```bash
./run_training_v002.sh
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

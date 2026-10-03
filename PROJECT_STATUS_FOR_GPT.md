> **2026-10-04 分支补充**：以下正文保留原历史模型研究记录。当前实验分支基于 `094eeab`，新增工业 Isaac Sim 真实飞行与当前 MID-360 条件的在线适配，详见 [DYNAMIC_PLANNING.md](docs/DYNAMIC_PLANNING.md)。模型仍是原 v003，新增的是低噪声神经修正、轨迹约束和感知/执行接口，没有动态场景重训练；请勿将历史测试集指标当成工业动态飞行成功率。

# HNUTER DiT 项目现状与后续讨论上下文

> 更新时间：2026-09-08
>
> 用途：将本文档直接提供给新的 GPT/研究合作者，使其在讨论下一阶段方案前准确了解当前实现、实验结果与限制。
>
> 项目阶段：free-flight Base DiT v003 已完成训练；平滑多目标 inference guidance 已完成实现和冻结测试；v003 尚未提交到 GitHub。

## 1. 一句话摘要

当前系统是一个面向 UAM 自由飞行几何轨迹生成的 environment-conditioned DiT：输入起点、终点和最多 32 个 OBB 障碍物，生成 128 个 SE(3) waypoint。坐标系对齐已经解决世界方向旋转导致的失效，但仅靠 Cross-Attention、relative geometry 和训练期 collision loss，尚未让模型学会在 OOD 障碍变化下可靠切换同伦路径。后期可微碰撞 guidance 能显著提高避障率；新版平滑 guidance 又明显缓解了旧版 guidance 在障碍物附近产生尖折、抖动和回退的问题，但仍存在避障率、平滑度和推理耗时之间的权衡。

## 2. 最终目标

希望训练一个可用于 UAM/无人机自由飞行规划的 diffusion/DiT 模型，具备以下能力：

1. 根据起点、终点、位姿和障碍物几何生成多条 SE(3) 候选轨迹；
2. 障碍物移动、删除或新增后，轨迹能够根据当前几何重新决策，而不是复现训练模板；
3. 对训练分布外的障碍布局具有一定泛化能力；
4. 轨迹不仅离散 waypoint 无碰撞，还应平滑、少回退、便于后续时间参数化和控制；
5. 通过 WebUI 交互编辑障碍物与起终点，直观看到模型响应。

当前只处理 `free_flight`。未来的 `inspection` 和 `surface` 任务必须使用各自的 task plugin 和 validity checker，不能直接复用 free-flight 规则。

## 3. 仓库、数据和运行环境

### 3.1 路径

```text
DiT 代码和模型：
/home/z017/research/HNUTER_DiT

专家轨迹采集器与场景/COAL 实现：
/home/z017/research/HNUTER_trajectory_collector

冻结训练数据：
/home/z017/research/HNUTER_datasets/free_flight_159103traj_dit_v001

最初参考的 diffusion 仓库：
/home/z017/research/HNUTER-diffusion

验证过的 Python/OMPL/COAL ABI：
/home/z017/research/curobo_env/bin/python
```

### 3.2 Git 状态

- GitHub：`git@github.com:0Z17/HNUTER_DiT.git`
- 当前分支：`main`
- GitHub/本地 HEAD：`52378bb Add canonical relative-geometry DiT v002`
- v001 和 v002 的 `best.pt` 已通过 Git LFS 管理。
- **v003 模型、guidance、最新评测、WebUI 和文档当前仍是本地未提交改动。**
- Hugging Face 发布状态没有记录在仓库元数据中，继续操作前应先核对远端仓库。

这意味着：新的 GPT 如果只查看 GitHub，会看到 v002，而不是本文所述的完整 v003 工作区。

## 4. 冻结数据集

数据集 ID：`free_flight_159103traj_dit_v001`，状态为 frozen。

| Split | Environments | Conditions | Trajectories |
|---|---:|---:|---:|
| Train | 3,211 | 54,763 | 127,360 |
| Validation | 400 | 6,834 | 15,845 |
| Test | 400 | 6,978 | 15,898 |
| 总计 | 4,011 | 68,575 | 159,103 |

环境按 environment ID 分组后再划分，比例为 80/10/10，因此同一环境不会跨 train/validation/test 泄漏。共有十个场景族，每个 validation/test split 中每族各 40 个环境：

- `central_block`
- `frame_doorway`
- `mixed_industrial`
- `multi_homotopy`
- `narrow_passage`
- `orientation_sensitive_passage`
- `pillar_wall`
- `sparse_obb_clutter`
- `staggered_corridor`
- `wall_protrusion_bracket`

### 4.1 轨迹表示

```text
trajectory: [128, 9] float32
features: xyz + rotation-6D
parameterization: normalized SE(3) arc length
physical time: 无
condition: start pose9 + goal pose9 = [18]
```

推理输出会把 rotation-6D 转回 `xyz + quaternion_wxyz` 的 pose7。

### 4.2 障碍物表示

```text
environment tokens: [32, 10]
features: center xyz + extent xyz + quaternion wxyz
mask: active obstacle token
floor: 不作为 obstacle token 输入
```

目前只建模 OBB。训练碰撞 proxy 使用 UAM 的 42 个 collision spheres 对 OBB 的 signed distance；最终评测使用完整 HNUTER URDF + COAL。

### 4.3 与泛化直接相关的数据问题

当前用户无法立即重新采集专家轨迹，因此 v002/v003 均使用同一个冻结数据集。数据没有系统性提供如下 counterfactual supervision：

```text
相同/相近 start、goal
环境 A -> 专家必须从障碍物上方绕行
环境 B -> 只移动关键障碍物后，专家必须从下方或另一侧绕行
```

已有拓扑标签也高度不均衡，例如：

| 标签 | 数量 |
|---|---:|
| `above` | 29,514 |
| `below` | 753 |
| `primary` | 47,380 |
| `free_space_alternative` | 6,689 |
| `dogleg_inner` | 15,685 |
| `dogleg_outer` | 355 |
| `winding_over` | 691 |
| `winding_under` | 8,291 |

因此网络可以通过场景族、起终点和常见专家轨迹形状预测一个高概率模板，而不必真正学习“关键障碍变化后应切换哪一个 homotopy”。这是目前 obstacle shortcut 的主要解释之一。

## 5. 当前默认模型：v003

Checkpoint：

```text
/home/z017/research/HNUTER_DiT/artifacts/
base_free_flight_159103traj_dit_canonical_cross_v003/
models/dit_cross/best.pt
```

### 5.1 模型规格

| 项目 | 配置 |
|---|---|
| 类型 | environment-conditioned DiT |
| Model type | `dit_cross` |
| 参数量 | 11,964,169 |
| checkpoint 大小 | 约 46 MiB |
| Sequence length | 128 |
| Pose dimension | 9 |
| Hidden dimension | 256 |
| DiT depth | 8 |
| Attention heads | 8 |
| Diffusion training steps | 100 |
| Prediction target | velocity / v-prediction |
| Environment conditioning | standard Cross-Attention |
| Endpoint handling | DDIM 每步 hard clamp |
| Sampling default | 25-step DDIM |
| CFG | **当前未实现** |

模型每个 block 包含 trajectory self-attention、trajectory 对 obstacle tokens 的 Cross-Attention、MLP，以及由 timestep embedding 与 start/goal embedding 控制的 modulation。

### 5.2 Canonical frame

v002/v003 会在训练和推理前进行 `start_goal_yaw_z_up_v001` 坐标变换：

- 起点平移到原点；
- 起点到终点的水平投影对齐 canonical `+x`；
- `+z` 始终保持世界竖直向上；
- 起点、终点、整条轨迹、障碍物中心和障碍物朝向使用同一刚体变换；
- 生成后再变换回世界坐标。

这只消除了 yaw/平移坐标捷径，不会自动赋予模型障碍物拓扑推理能力。

### 5.3 训练目标

基础目标是 v-prediction diffusion loss，只计算内部 waypoint：

```math
L_{diff}=\operatorname{MSE}(\hat v_t,v_t).
```

从预测 velocity 恢复 clean trajectory：

```math
\hat x_0=\sqrt{\bar\alpha_t}x_t-\sqrt{1-\bar\alpha_t}\hat v_t.
```

在 `x0_hat` 上计算 collision auxiliary loss：

```math
L=L_{diff}+0.002L_{col}.
```

```math
L_{col}=\bar\alpha_t\,\operatorname{mean}\left[
\operatorname{softplus}\left(\frac{0.04-d(\hat x_0,E)}{0.05}\right)^2
\right].
```

其中 `d` 是 42 robot spheres 对 OBB 的 signed distance；训练时每 4 个 waypoint 取一个碰撞采样点。乘以 `alpha_bar(t)` 后，高噪声 timestep 的 collision loss 会自动减弱，避免在几何意义很弱的噪声状态上施加强碰撞梯度。

### 5.4 训练配置和结果

```text
steps: 48,000
batch size: 64
optimizer: AdamW
learning rate: 2e-4
warmup: 2,000
min LR ratio: 0.05
weight decay: 1e-4
gradient clip: 1.0
EMA decay: 0.995
AMP: FP16 enabled
seed: 20260827
GPU: NVIDIA GeForce RTX 4080 Laptop GPU
training loop wall time: 2,024.5 s（约 33.7 min）
```

最佳 checkpoint 在 step 48,000：

```text
validation diffusion loss: 0.0199700363
validation collision loss: 0.0090133025
validation total loss: 0.0199880629
SHA-256: 301070cd16c847d749da548bca4d66e409e7ed698346c9b938f3ddf782936dad
```

`best.pt` 是 validation total loss 最低的 EMA 权重。`last.pt` 保留 optimizer、scheduler、EMA、sampler 和 RNG 状态，但默认不进入 Git。

## 6. 模型版本演化与已得到的结论

### v001：absolute standard Cross-Attention

- 世界坐标输入；
- 标准 obstacle-token Cross-Attention；
- 无 canonical frame；
- 无训练期 collision auxiliary；
- 11,964,169 参数；
- best validation loss：`0.02108097`。

问题：专家轨迹在数据中有明显方向先验，整体将场景和端点 yaw 旋转 90° 后性能严重下降。

### v002：canonical + pairwise relative-geometry Cross-Attention

- canonical frame；
- collision auxiliary loss；
- pairwise trajectory-obstacle relative geometry；
- relative feature 包括 body-frame obstacle delta、distance、extent 和 relative rotation；
- 12,002,121 参数；
- best validation total loss：`0.02006100`。

结果：yaw 等变性问题基本解决，但 obstacle-shift OOD 避障没有显著改善，而且推理更慢。

### v003：canonical + standard Cross-Attention

- 保留 canonical frame 与 collision auxiliary；
- 移除未证明有效的 pairwise relative-geometry attention；
- 恢复标准 Cross-Attention；
- 参数量回到 11,964,169；
- unguided 推理比 v002 更快；
- obstacle-shift 性能仍没有显著改善。

关键结论：

1. **Canonical frame 有效。** v001 的 90° yaw 等变位置 RMS 为 `0.51371 m`，v002 为 `0.00030 m`，v003 为约 `0.000003 m`。
2. **当前 pairwise relative geometry 没有证明有效。** v002 相对 v001 的 obstacle-shift collision-free 仅 `+0.75 pp`，95% CI `[-1.75,+3.50] pp`。
3. **标准 Cross-Attention 与训练期 collision loss 不足以让模型切换 homotopy。** v003 obstacle-shift unguided collision-free 仅 14.88%。
4. **不能用 validation diffusion loss 代替规划指标。** 较低 denoising loss 不代表更高 collision-free 或更强 OOD 响应。

## 7. 当前 inference guidance

### 7.1 旧版 pointwise guidance

旧版在最后 8 个 DDIM step 中：

- 对每个 waypoint 独立计算最近 sphere/OBB 的 hard-min 梯度；
- 将每个非零梯度归一化；
- 单 waypoint 每 guided step 最大移动 0.20 m；
- 做两次 `[0.25,0.5,0.25]` 局部滤波。

它能迅速把离散点推出障碍物，所以 collision-free 很高；但由于 waypoint 独立响应、hard-min 活跃对象切换、单步位移相对 waypoint 间距过大，容易在障碍物附近形成尖折、锯齿、局部环和回退。

### 7.2 当前默认：smooth multi-objective guidance

新版不直接独立优化 128 个 waypoint，而是优化 16 个三维控制点，经线性插值和 endpoint taper 得到完整位移场：

```math
\Delta p_i=\sin^2(\pi i/(L-1))\;\operatorname{Interp}(u)_i.
```

总 cost：

```math
J=
J_{collision}
+0.10J_{reference}
+0.25J_{smooth}
+0.05J_{jerk}
+0.25J_{stretch}
+1.00J_{tangent}.
```

- `collision`：42 spheres/OBBs，安全距离 0.04 m，softplus scale 0.05 m；
- `reference`：限制 displacement，避免远离 DiT prior；
- `smooth`：惩罚 displacement 二阶差分；
- `jerk`：惩罚 displacement 三阶差分；
- `stretch`：限制相邻 waypoint 段长相对 prior 的变化；
- `tangent`：强烈惩罚相对 prior 的局部切向反转；
- sphere/obstacle 聚合使用 0.02 m soft-min，减少最近碰撞对象切换的不连续；
- 每个 guided DDIM step 做 3 次低维 Adam-like 优化；
- 最后做一次局部平滑；
- 起终点和 orientation 不变。

Validation 选择出的默认参数：

```text
mode: smooth
late DDIM steps: 8 / 25
maximum smooth displacement per guided step: 0.12 m
control points: 16
inner optimization steps: 3
safe distance: 0.04 m
collision sigma: 0.05 m
softmin temperature: 0.02 m
smoothing passes: 1
```

这些参数在 30 个 family-stratified validation conditions、每条件 4 个候选、关键障碍物向 start-goal segment 平移 0.30 m 的设置上选择。参数选择同时限制路径长度、P95 转角、二阶差分 RMS、jerk RMS 和回退比例，不只最大化碰撞率。

### 7.3 Guidance 不是 CFG

当前实现是对预测 clean trajectory 的 scene-specific differentiable cost 求梯度，属于 cost/energy guidance。项目目前没有 condition dropout、unconditional branch 或 conditional/unconditional prediction mixing，因此**没有实现 classifier-free guidance（CFG）**。

CFG 可以用于 DiT，但它主要加强模型对条件分布的服从程度；如果模型从训练数据中没有学到障碍变化对应的正确决策，增大 CFG scale 也可能只是更强地复现错误模板，不能替代碰撞 cost 或 counterfactual data。

## 8. 冻结测试结果

测试协议：

- frozen test split；
- 100 个按十类场景族分层抽取的 conditions；
- 每条件 8 个候选；
- 25 DDIM steps；
- 各版本使用相同 latent seeds；
- 所有 128 个 waypoint 使用完整 HNUTER URDF + COAL 验碰；
- `valid` 还要求采样边界有效、roll 不超过 40°、pitch 不超过 70°。

### 8.1 v003 guidance 对照

| 场景 | 方法 | Collision-free | Best-of-8 | Mean clearance | P95 turn | Max turn | 8-path batch |
|---|---|---:|---:|---:|---:|---:|---:|
| 原始 test | unguided | 39.38% | 74% | -0.0506 m | 4.58° | 8.10° | 102.1 ms |
| 原始 test | legacy pointwise | **90.63%** | **100%** | +0.0381 m | 6.25° | 12.47° | 130.7 ms |
| 原始 test | smooth multi-cost | 83.00% | 98% | +0.0382 m | **5.60°** | **9.69°** | 200.6 ms |
| yaw +90° | unguided | 39.38% | 74% | -0.0506 m | 4.58° | 8.10° | 101.4 ms |
| yaw +90° | legacy pointwise | **90.63%** | **100%** | +0.0381 m | 6.25° | 12.47° | 129.3 ms |
| yaw +90° | smooth multi-cost | 83.00% | 98% | +0.0382 m | **5.60°** | **9.69°** | 200.3 ms |
| obstacle shift 0.30 m | unguided | 14.88% | 40% | -0.1360 m | 4.71° | 8.28° | 101.5 ms |
| obstacle shift 0.30 m | legacy pointwise | **72.88%** | **89%** | -0.0129 m | 8.28° | 17.40° | 129.7 ms |
| obstacle shift 0.30 m | smooth multi-cost | 55.63% | 79% | -0.0170 m | **5.94°** | **10.24°** | 200.0 ms |

这里的 batch time 在 NVIDIA GeForce RTX 4080 Laptop GPU 上测得，包含 DiT DDIM
生成和对应的 guidance，但不包含随后完整 COAL 验碰。

### 8.2 统计解释

障碍物平移场景中：

- smooth 相对 unguided collision-free：`+40.75 pp`，95% CI `[+34.38,+47.13] pp`；
- smooth 相对 unguided best-of-8：`+39 pp`，95% CI `[+30,+49] pp`；
- smooth 相对 legacy pointwise collision-free：`-17.25 pp`；
- smooth 相对 legacy 的 P95 转角：`-2.34°`；
- smooth 相对 legacy 的最大转角：`-7.17°`；
- smooth 相对 legacy 的二阶差分 RMS：`-0.000820`；
- smooth 相对 legacy 的 jerk RMS：`-0.000718`；
- smooth 相对 legacy 的回退比例：`-0.00109`。

后五个轨迹形状指标的 paired bootstrap CI 均不包含 0。因此新版确实减少了旧版的局部紊乱，但它是明确的安全率—平滑度—延迟折中，而不是对旧版严格全面占优。

## 9. 如何理解目前的失败模式

### 9.1 障碍物移走后仍按 S 形模板绕行

这更像 learned prior / dataset shortcut，而不是数值上的 diffusion “训练不稳定”。同类场景和起终点在数据中往往对应固定的常见 homotopy；如果没有反事实样本告诉网络“框移走后 C 形更短”，denoising objective 不会主动发现这条规则。

### 9.2 障碍物移动后轨迹仍碰撞

说明 Base DiT 对障碍变化的响应量太小，或者只做局部形状调整而没有改变拓扑。v002 的 pairwise relative geometry 并未显著改善这个问题，所以不能简单归因于 standard Cross-Attention 缺少相对坐标。

### 9.3 旧 guidance 在碰撞处紊乱

原因包括 hard-min 梯度切换、逐 waypoint 独立归一化、大步长、多次局部推开，以及缺少对曲率、jerk、路径拉伸和反向段的约束。新版 smooth guidance 已针对这些问题修改。

### 9.4 新 smooth guidance 仍无法通过某些 OOD 场景

新版 guidance 是围绕 DiT prior 的局部连续形变。若 prior 深入大障碍物、通道已经闭合或必须切换到远处同伦路径，局部优化可能没有足够的搜索范围。此时需要全局拓扑 proposal、更多不同 homotopy 候选、搜索/优化混合，或者重新收集 decision-changing data。

## 10. 当前 WebUI

启动：

```bash
cd /home/z017/research/HNUTER_DiT
./run_webui.sh --device cuda
```

默认地址：<http://127.0.0.1:8790>。如果端口被占用：

```bash
./run_webui.sh --device cuda --port 8791
```

主要能力：

- 随机选择或指定十类 free-flight 场景；
- 随机生成 8–18 个 OBB；
- 随机起点、终点和姿态；
- 编辑 start/goal XYZ 与 roll/pitch/yaw；
- 添加、删除、平移、缩放、旋转障碍物；
- 在三维画布拖动障碍物和 start/goal；
- 一次生成 1–16 条候选；
- 选择 10/25/50/100 DDIM steps；
- 调整或关闭最后若干步 guidance；
- 显示最小净空、长度、碰撞点、越界、姿态超限、P95/最大转角、二阶差分 RMS、jerk RMS 和回退比例；
- 最终使用完整 URDF + COAL 审核 128 个 waypoint。

WebUI 当前默认是 smooth guidance：8 steps、0.12 m。

## 11. 关键文件索引

```text
hnuter_dit/model.py
  DiT、standard/relative Cross-Attention、cosine schedule、DDIM sampling

hnuter_dit/geometry.py
  canonical frame 与世界/canonical 坐标互换

hnuter_dit/collision_loss.py
  42-sphere/OBB signed distance 与训练 collision auxiliary

hnuter_dit/guidance.py
  legacy pointwise 与 smooth multi-objective inference guidance

hnuter_dit/training.py
  v002/v003 数据变换、训练、验证、EMA 和 checkpoint

hnuter_dit/tune_guidance.py
  validation-only guidance 参数选择

hnuter_dit/evaluate_v003.py
  v001/v002/v003、unguided/legacy/smooth 的统一冻结测试

hnuter_dit/service.py
  模型服务、随机场景、完整 COAL 审核、WebUI API

hnuter_dit/web_assets/
  可交互 WebUI

artifacts/base_free_flight_159103traj_dit_canonical_cross_v003/
  v003 checkpoint、配置、validation tuning、完整测试结果和报告
```

最重要的结果文件：

```text
artifacts/base_free_flight_159103traj_dit_canonical_cross_v003/
  EVALUATION_REPORT.md
  guidance_tuning_smooth_validation.json
  comparison_v003_smooth_guidance.json
  resolved_training_config.json
  run_status.json
```

## 12. 当前已经验证的内容

- v003 训练完成，48,000/48,000 steps；
- checkpoint SHA 与 `run_status.json` 一致；
- frozen validation 上完成 guidance 参数选择；
- frozen test 上完成 100 conditions × 8 candidates × 3 scenarios 的统一对照；
- Python 模块可编译；
- WebUI JavaScript 语法检查通过；
- 当前 11 项 pytest 全部通过；
- v003 尚未提交/推送，因此继续开发前必须保护当前 dirty worktree。

## 13. 不能过度声称的内容

1. 当前 Base DiT **没有**证明已学会通用障碍物拓扑推理。
2. Relative-geometry Cross-Attention 在本数据和实验中**没有**证明提升 OOD 避障。
3. Training collision loss 改善的是几何一致性信号，不提供 OOD collision-free guarantee。
4. Guidance 只对当前 OBB proxy 可微，最终 COAL 结果可能与 proxy 不完全一致。
5. 128 个离散 waypoint 全部通过 COAL，不等于相邻 waypoint 之间连续无碰撞。
6. 当前轨迹按归一化弧长参数化，没有速度、加速度或真实时间，不能直接执行。
7. WebUI 中的绿色轨迹仍需连续碰撞检测、时间参数化、动力学和闭环控制验证。
8. 当前结果只针对 free-flight，不能外推到 inspection 或 surface 任务。

## 14. 后续最值得讨论的方向

### 优先级 A：在不重新采集数据的条件下

1. **两阶段或三阶段规划**

   DiT 负责生成多个 prior/proposal；对每个候选做 differentiable refinement；最后用完整 COAL 排序并选优。不要要求单次局部 guidance 同时承担全局同伦搜索。

2. **分层/拓扑 latent**

   先预测或枚举 `left/right/above/below/through` 等 route class 或粗 waypoint corridor，再由 trajectory DiT 生成连续轨迹。需要处理当前标签不平衡，并避免把错误的现有 topology 标签当作绝对真值。

3. **更高效的 smooth guidance**

   当前三次 inner optimization 使 8-path batch 从约 102 ms 增至约 200 ms。可以测试：一次或两次更新、解析/preconditioned gradient、8–12 个 control points、仅对 active collision window 优化、CUDA graph 或融合 distance kernel。

4. **连续段碰撞代价**

   训练和 guidance 都主要检查 waypoint。可对相邻段自适应插值，尤其在高曲率或障碍附近加密，降低 waypoint 间穿障风险。

5. **候选级选择目标**

   评估不应只看 collision-free。建议统一 score：完整 COAL clearance、长度、最大曲率、jerk、回退、边界、姿态和候选多样性。

6. **受控消融**

   分别消融 canonical frame、collision auxiliary、standard/relative attention、hard/soft min、smooth/jerk/stretch/tangent cost、control point 数与 inner steps。所有调参只用 validation，最终只报告 frozen test。

### 优先级 B：未来允许扩充专家数据时

1. 构造 paired/counterfactual scenes：固定 start/goal，仅移动关键障碍物，并让专家路径必须改变 homotopy；
2. 对同一条件保存多个高质量不同 homotopy 专家解，避免单模板监督；
3. 主动挖掘当前模型失败场景，进行 hard-negative / DAgger-like 数据闭环；
4. 平衡 `above/below`、`inner/outer` 等拓扑标签；
5. 将路径质量和安全裕度纳入专家选择，而不是只保存第一个可行解。

### 对 CFG 的建议

若后续加入 CFG，需要在训练时随机丢弃 environment 或 condition，显式训练 unconditional/partially conditional branch，再在采样时组合预测。建议把它作为“条件服从强度”消融，而不是把它当成解决 OOD collision 的主要手段。

## 15. 希望 GPT 重点帮助回答的问题

在提出方案前，请先接受以下事实：当前没有新的 counterfactual expert data；canonical frame 已证明有效；relative attention 已做过但没有显著改善；smooth guidance 已改善形状但降低了部分碰撞率并增加延迟。

希望进一步讨论：

1. 在不重新收集专家数据时，怎样让局部 guidance 具备一定的全局 homotopy 切换能力？
2. 是采用粗路径/拓扑 proposal + DiT refinement，还是采用 diffusion 内部的 energy guidance 更合理？
3. 如何设计连续碰撞、曲率和动力学可行性的统一 differentiable cost，并避免各项梯度尺度冲突？
4. 如何把当前 200 ms/8 paths 的 smooth guidance 降到接近 100–130 ms，同时保留平滑性？
5. 如何构造严格的 OOD benchmark，区分坐标旋转泛化、几何位移泛化和真正的拓扑决策泛化？
6. 如果未来只允许少量补采数据，哪些 counterfactual 场景最有信息量？
7. 是否值得引入 route-class latent、constraint diffusion、trajectory optimizer 或 sampling-based planner hybrid？

## 16. 可直接发给 GPT 的开场提示

```text
请阅读随附的 PROJECT_STATUS_FOR_GPT.md，并把它视为当前项目的事实基线。

我的目标是提升 UAM free-flight DiT 在障碍物移动、新增或删除后的 OOD 避障和
同伦切换能力，同时保持轨迹平滑与较低推理延迟。请不要重复建议已经验证无明显
收益的“只加入 relative-geometry Cross-Attention”，也不要把 CFG 当作碰撞保证。

当前暂时不能重新采集大规模专家数据。请基于现有 v003、smooth guidance、完整
URDF+COAL 评测和已有 WebUI，提出下一阶段可验证的技术方案。方案需要说明：
1) 它具体解决哪个已观察失败模式；
2) 需要修改的数据、模型、loss 或 sampler；
3) 训练/推理开销；
4) 必须做的消融；
5) validation/test 协议；
6) 失败标准和停止标准。

请先给出不超过三个优先级明确的方案，再详细展开你最推荐的一个。
```

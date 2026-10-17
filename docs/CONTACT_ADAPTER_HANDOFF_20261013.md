# Contact/Surface adapter：对话、数据与实验交接

更新日期：2026-10-13，Asia/Singapore（UTC+8）。完整实验报告于当天 13:19:46 交付。

本文件用于让新的 Codex 会话接续本项目，不是完整聊天逐字稿。它整理已确认的研究目标、用户决策、当前文件状态、训练和验证协议、最终结果及尚未验证的问题。数值依据本机已交付报告和完成证据；建议、历史讨论和正式结果在文中分别标明。

## 2026-10-17：源码提交归档补记

本次在 `codex/v003-task-adapters` 提交 Contact/Surface adapter 的源码、测试、实验协议、研究报告、图表与可视化验证记录，并包含 Contact 闭环执行所需的 `flight_control.py`、`retiming.py` 和 `mujoco_execution.py`。完整自由飞行交互界面的独立改动未纳入本次提交。

下文第 3 节及接续提示中“未提交”“HEAD 为 094eeab”的描述是 2026-10-13 的历史快照；后续以实际 `git status` / `git log` 为准。本次提交不合并 `main`，也不推送远端。权重、数据、原始预测和完整统计附录仍留本地，Git 归档范围见 [实验归档说明](../artifacts/contact_adapter_multifamily_v001/README.md)。

## 1. 新会话首先需要知道的状态

**本轮实验已经全部完成。当前没有需要继续等待的训练或正式评测，也不应因为旧状态文件而重新启动流水线。**

- 21 个 adapter 均完成 8,000 次有效 optimizer 更新；Base v003 全程冻结。
- Base 加 adapter 共 22 个模型，完成原始/移障两场景、`off/on/on_smooth` 三种 decoder 的 K1 和固定子集 K8 验证。
- 全部 748,000 条原始预测已生成；三种 decoder 合计 2,244,000 个候选评分。
- 完整附录包含 **2,263,800 个汇总指标行**。这是统计表行数，与候选评分行数不同。
- 交付审核覆盖 K1 132 个、K8 132 个单模型总体成功率单元格：22 模型 × 2 场景 × 3 decoder。
- 主流水线、平滑补充、汇总恢复、交付守护均正常退出，exit code 为 0；交付时核验了完整进程身份和无剩余所属进程。
- 定时 heartbeat `contact-adapter` 已在完成交付后删除；不要自动重新创建。

有几处历史文件容易误导新会话：

| 文件或描述 | 当前记录 | 正确解释 |
|---|---|---|
| `experiment_status.json` | `training_complete_evaluation_pending` | 训练矩阵驱动留下的阶段记录；完整评测状态看主 `pipeline_status.json` |
| `consolidated_reports/pipeline_status.json` | `failed`，`KeyError: 'input_files'` | 故意保留的 v001 汇总代码故障证据；恢复版和交付已经完成 |
| `report_followup.json` | 旧 `ACTIVE` 值 | 历史创建记录；实际 automation 已删除 |
| `docs/CONTACT_ADAPTER_ABLATION_V001.md` | 含“运行中”“尚未生成报告”等措辞 | 原设计/执行阶段文档；完成事实以本交接和实际完成证据为准 |

本文件的编写没有重新训练、采样、评分、调整阈值、提交或推送代码。

## 2. 研究意图与本对话的主要决策

用户的研究主线是：先在 canonical Base DiT v003 上验证 **task adaptive**，将它作为主要创新点；之后再考虑动态雷达输入、不同环境特征表达，以及工厂飞行。

对话中的主要推进过程：

1. 检查 `main`、动态雷达、静态 SDF、Contact/Inspection 任务分支的关系和 GitHub 分支可见性。
2. 用户明确关注静态环境中的 task-adaptive Inspection/Contact。早期旧任务实现与小数据、控制点/样条基座有关，不能直接视为目前 canonical v003 上已经成立的 adapter。
3. 决定回到 `main` 的 v003 基线，打 tag，再创建独立开发分支。
4. 用户询问旧 Contact/Surface 约 2,000 条专家数据是否覆盖多环境族，随后要求基于专家采集器重新收集约 10,000 条多场景接触轨迹。
5. 新采集要求包含不同角度的平面与圆柱、起终点位姿、接触面参数，并给 adapter 的 cross attention 提供明确的任务几何输入。
6. 用户要求一定比例的接触任务必须同时绕障，保留利用 Base 避障和 diffusion 候选多样性的必要性。
7. 用户要求采集/实验期间有温控守护，过热暂停，降温后继续。
8. 用户明确采用 500/1,000/2,000/4,000/8,000 条嵌套子集、每档 3 个 seed，并要求独立 10,000 条测试条件、移障泛化和投影消融。
9. 增加两类 8k 负控；普通投影出现 shape 全失败后，保持原协议和阈值，增加独立、参数预先固定的 `on_smooth` 补充消融。
10. 用户多次要求继续完整实验；一次“4”已澄清为误发，并非缩减实验规模。
11. 用户要求可视化零成功率轨迹，并追问起点曲率与曲率阈值；完成了实际保存轨迹的诊断，未据此放宽正式门限。
12. 全部实验与报告已交付。用户现在要求将上下文写成 Markdown，供其他会话读取。

**当前正式实验只验证连续 Contact/Surface 几何接触段，没有训练或验证 Inspection adapter，也没有把动态雷达接入本轮模型。**

## 3. 仓库、分支与本地工作树

仓库：[HNUTER_DiT](https://github.com/0Z17/HNUTER_DiT)。本地工作目录为 `/home/z017/research/HNUTER_DiT`。

2026-10-13 只读检查的本地 Git 状态：

| ref | 提交 | 含义 |
|---|---|---|
| 当前 `codex/v003-task-adapters` | `094eeab5f1740659da43663b3b236e39167e27c7` | 从 v003 基线开始的 adapter 开发工作树 |
| `main` / 本地 `origin/main` | `094eeab` | canonical v003 基线 |
| tag `v003-adapter-baseline` | 指向 `094eeab` | 供 adapter 开发保留的基线 tag |
| `codex/dynamic-lidar-planning` | `1153440` | 旧动态雷达实验分支 |
| `codex/static-sdf-pose-planning` | `9e1ca71` | 旧静态 SDF 位姿规划分支 |
| `codex/task-adaptive-contact-inspection` | `0783486` | 旧 Contact/Inspection 实验分支 |

旧三分支的本地 remote-tracking refs 也存在。本次没有 fetch，以上不是对 GitHub 最新实时状态的验证。

**Contact 实验源码中有大量未跟踪/未提交文件，当前分支名和 HEAD 本身不能携带这些实现。**新会话如果在同一台电脑，应使用现有工作树；只从 GitHub clone 或只切换到该 ref，可能没有本次 adapter 代码和产物。工作树还存在交互飞行、MuJoCo、Web UI 等其他改动，不能将它们全部归因于本轮 Contact 实验，也不要擅自清理或统一提交。

## 4. 优先阅读的文件

以下是实际存在的绝对路径，可直接交给新会话：

| 优先级 | 文件 | 用途 |
|---|---|---|
| 1 | [完整中文报告](/home/z017/research/HNUTER_DiT/artifacts/contact_adapter_multifamily_v001/consolidated_reports/research_report_zh.md) | 最终科学结果、全部消融、口径与限制 |
| 2 | [交付证据](/home/z017/research/HNUTER_DiT/artifacts/contact_adapter_multifamily_v001/consolidated_reports/delivery_commit.json) | 交付文件 SHA、覆盖检查、上游退出与失败记录保留证据 |
| 3 | [冻结主协议](/home/z017/research/HNUTER_DiT/artifacts/contact_adapter_multifamily_v001/protocol.json) | 21 个 adapter 的固定矩阵与实验定义 |
| 4 | [恢复报告 commit](/home/z017/research/HNUTER_DiT/artifacts/contact_adapter_multifamily_v001/consolidated_reports/recovery_v002/report_commit.json) | 正式恢复输出的哈希与指标行数 |
| 5 | [原协议说明](/home/z017/research/HNUTER_DiT/docs/CONTACT_ADAPTER_ABLATION_V001.md) | 设计细节；其中执行状态有历史措辞 |

数据和产物目录：

| 目录 | 内容 |
|---|---|
| `/home/z017/research/HNUTER_trajectory_collector` | 专家轨迹采集器仓库 |
| `/home/z017/research/HNUTER_datasets/surface_multifamily_contact_10000traj_v001` | 已冻结的 10k 接触专家数据 |
| `/home/z017/research/HNUTER_datasets/surface_contact_adapter_benchmark_10000cond_v001` | 与训练隔离的新地图/新条件测试集 |
| `/home/z017/research/HNUTER_DiT/artifacts/contact_adapter_multifamily_v001` | 本轮实验根目录，下文简称 `EXP` |
| `EXP/models/<model_id>/` | `best.pt`、`last.pt`、`resolved_training_config.json`、子集与历史 |
| `EXP/evaluation/<model_id>/` | 原始预测、off/on 评分与各分片契约 |
| `EXP/smooth_projection_supplement/` | 平滑投影评分、保存路径和完整补充报告 |
| `EXP/consolidated_reports/recovery_v002/` | 成功完成的严格汇总恢复产物 |
| `EXP/source_snapshot/` 及各补充/恢复快照 | 冻结源码、测试和运行环境证据 |

根交付附件是指向恢复附件的符号链接：

- [CSV.gz](/home/z017/research/HNUTER_DiT/artifacts/contact_adapter_multifamily_v001/consolidated_reports/full_statistics.csv.gz)
- [JSONL.gz](/home/z017/research/HNUTER_DiT/artifacts/contact_adapter_multifamily_v001/consolidated_reports/full_statistics.jsonl.gz)

先阅读小的 Markdown 和 commit/status。父报告 JSON 约 370 MB，平滑补充 JSON 约 1.1 GB；不必为了解上下文同时完整加载它们。恢复版 `report.json` 约 40 MB，详细数值也可通过压缩附录筛选。

## 5. Base DiT v003 的身份及已有水平

Base checkpoint：

`/home/z017/research/HNUTER_DiT/artifacts/base_free_flight_159103traj_dit_canonical_cross_v003/models/dit_cross/best.pt`

Base 是 canonical-frame Cross-Attention DiT，11,964,169 参数；原自由飞行数据集为 `free_flight_159103traj_dit_v001`。Base 训练为 48,000 steps、batch 64、AMP/EMA，最佳验证损失为 0.0199881。Contact adapter 不更新 Base 参数，也不重新拟合其归一化统计。

历史自由飞行评估见 [Base EVALUATION_REPORT.md](/home/z017/research/HNUTER_DiT/artifacts/base_free_flight_159103traj_dit_canonical_cross_v003/EVALUATION_REPORT.md)。该测试是 100 条件、每条件 8 候选、128 个位姿的完整 URDF 节点碰撞审核：

| 场景 | 推理方法 | 128 节点审核下的候选无碰撞率 | Best-of-8 条件解决率 |
|---|---|---:|---:|
| 原始 | unguided | 39.38% | 74% |
| 原始 | legacy pointwise guidance | 90.63% | 100% |
| 原始 | smooth multi-cost guidance | 83.00% | 98% |
| 障碍移动 0.30 m | unguided | 14.88% | 40% |
| 障碍移动 0.30 m | legacy pointwise guidance | 72.88% | 89% |
| 障碍移动 0.30 m | smooth multi-cost guidance | 55.63% | 79% |

候选无碰撞率指一条候选路径的全部 128 位姿通过审核的候选占比，不是将所有节点混合后计算逐节点通过比例。这些数值不是 Contact 严格成功率。自由飞行目标、数据、条件数、guidance、成功判据与本轮实验不同；旧节点审核结果也不是连续路径或动力学保证。本轮 Contact 的 Base 对照使用统一 25 步 DDIM，**没有额外碰撞推理引导**。

对工厂场景的 OOD 是用户关切；本轮结果不能证明 factory semantic OOD 泛化，也没有正式量化一个真实工厂飞行成功率。

## 6. 新专家数据与 cross-attention 输入

当前新数据取代了早期约 2k 数据作为本实验输入。正式统计：

- 10,000 条专家轨迹，500 张源地图，7,000 个不同任务条件。
- 平面 5,000 条、圆柱 5,000 条；绕障 `detour` 6,000 条、简单 `easy` 4,000 条。
- 3,000 个绕障条件各有两条配对专家，加 4,000 个简单条件；因此路径数不等于独立条件数。
- 每地图 20 条专家、14 个不同条件。按源地图隔离为 train 8,000 / validation 1,000 / test 1,000 条路径。
- 10 个场景族各 1,000 条路径：`sparse_obb_clutter`、`central_block`、`multi_homotopy`、`narrow_passage`、`orientation_sensitive_passage`、`wall_protrusion_bracket`、`frame_doorway`、`pillar_wall`、`staggered_corridor`、`mixed_industrial`。

数据与 Base 数据同在 `/home/z017/research/HNUTER_datasets`，保持 Base-compatible 路径/环境格式，并扩展任务字段。

| 输入 | 格式与含义 |
|---|---|
| 路径 | float32 `[128,9]`，世界坐标的 XYZ + rotation6d；rotation6d 为旋转矩阵前两列拼接 |
| 参数化 | 归一化加权弧长，**无真实物理时间**；旋转权重 0.22 m/rad |
| 起终点 | 两个 pose9，共 18 维 |
| 环境 | OBB `[32,10]` 和 mask `[32]` |
| 任务几何 | 32 维曲面/工具/约束特征，schema `surface_cross_attention_features_v001` |
| 支撑物 | `[8,10]` 及 mask |
| 禁入图域 | `[8,4]` 及 mask；当前冻结数据未启用 forbidden 案例 |

32 维任务特征记录：plane/cylinder 类型、世界原点、3×3 曲面 frame、半径/厚度、`u/v/psi` 范围、工具在 body 坐标中的位置与 quaternion、位置/法向容差和碰撞 margin。世界路径、端点、OBB、曲面原点与轴按同一 start-goal yaw canonical frame 变换；半径、图域坐标、body 工具外参不能错误地按世界点处理。

归一化引用原 Base 的 `canonical_statistics.json`，没有在 Contact 数据上重新拟合。路线模式标签、专家路径、planner certificate/guide 不得作为模型推理输入。

采集器关键源文件在其 `expert_trajectory_collector/surface/` 下，包括 `dataset_v1.py`、`campaign_planner.py`、`campaign_sampling.py`、`geometry.py`。应遵守采集器仓库自己的说明与 AGENTS，而不要在已有冻结数据里原地改变定义。

## 7. Adapter 与训练协议

架构为冻结 Base + task cross-attention 残差：在 blocks 0/3/7 加零初始化任务残差，bottleneck 64，物理长度尺度 2 m；adapter 训练参数 1,376,960。任务 token 包括曲面、工具、约束、起终点、支撑物和禁入图域，保留 Base 的环境 attention。

| 专家路径预算 | 独立源地图 | 不同任务条件 | seeds |
|---:|---:|---:|---|
| 500 | 25 | 350 | 17 / 29 / 43 |
| 1,000 | 50 | 700 | 17 / 29 / 43 |
| 2,000 | 100 | 1,400 | 17 / 29 / 43 |
| 4,000 | 200 | 2,800 | 17 / 29 / 43 |
| 8,000 | 400 | 5,600 | 17 / 29 / 43 |

子集 seed 为 20261011，按完整地图嵌套，配对专家不能拆散。主模型 15 个，加两种负控各 3 个，共 21 个 adapter。

每个模型的冻结训练设置：

- 8,000 次**成功** optimizer 更新，batch 64，共 512,000 次样本呈现；AMP overflow 跳步记录并重试。
- LR 1e-4、weight decay 1e-4、warmup 500、EMA 0.995；每 500 更新做验证，每 100 更新保存恢复 checkpoint。
- 固定全部 1,000 条验证专家与固定验证噪声 seed `202610110077`，用共同验证目标选择 `best.pt`。
- Loss：扩散 v-prediction + contact 5.0 + normal 0.5 + tool-position smooth 0.1 + collision 0.002；辅助项按 alpha_bar 加权。
- `velocity/v-prediction` 是扩散训练目标，不是无人机真实速度。
- 可微 collision surrogate 是 42 个 body 球与环境/支撑 OBB，stride 4；目标实体不在该 surrogate 中，完整 URDF 独立评测仍然必要。
- 使用反向增强；增强和重复更新不增加独立专家数或条件数。

这是**固定 compute 的数据量实验，不是等 epoch 实验**。五档路径平均呈现次数约为 1024/512/256/128/64。

负控的准确含义：

| 名称 | 改动 | 保留 |
|---|---|---|
| `no_contact_loss` | 同时去掉 contact、normal、smooth 三个辅助项 | 去噪目标与 collision 项 |
| `no_geometry_tokens` | adapter attention 仅激活起终点 task tokens | 相同 adapter、真实环境输入、原几何损失 |

验证损失均值约为 0.045758 / 0.044046 / 0.043477 / 0.043983 / 0.043815（五档依序）；这是模型选择损失，不能解释成任务成功率。

模型文件示例：

`/home/z017/research/HNUTER_DiT/artifacts/contact_adapter_multifamily_v001/models/geometry_n8000_seed17/`

该目录包含 `best.pt`（adapter checkpoint）、`last.pt`（恢复状态）、`run_status.json`、`resolved_training_config.json`、`training_history.json`、`training_subset.json`。Adapter 不能脱离绑定的 Base checkpoint 与统计单独推理。

## 8. 独立验证与三种 decoder

测试集为 1,000 张新源地图、10,000 个新任务条件，排除 Contact 全部 500 地图和 Base 源地图，并检查几何指纹。10 家族均衡，每地图 plane/cylinder 各 5 条件，每种 3 detour + 2 easy。

| 评测项 | 冻结定义 |
|---|---|
| K1 | 每场景全部 10,000 条件、1,000 地图，candidate_index=0 |
| K8 | 推理前固定 1,000 条件、100 地图，各 8 候选；每家族 10 地图 |
| 两场景 | `original` / `obstacle_shift`，共 20,000 个场景 case |
| 移障 | 选择靠近起终点身体位置弦线的非 floor OBB，朝弦线移动 0.30 m；沿用 Base shift function |
| 采样 | 25 DDIM steps / 100-step diffusion schedule；同条件/候选的模型与两场景共享初始 latent，seed 20261011 |
| 原始路径数 | 每模型 34,000：20,000 K1 case + 14,000 K8 的额外候选；22 模型合计 748,000 |

`off/on/on_smooth` 全部复用相同保存 raw，没有为 decoder 重新采样。K1 与 K8 分母不同，不能将两张表相减称为 K8 增益，也不能将八个候选视为八个独立任务。

| decoder | 实现 | 连续边解释 |
|---|---|---|
| `off` | 原始模型位姿输出 | 世界 XYZ 线性 + 最短 SO(3) SLERP |
| `on` | 有界 plane/cylinder 图域解析接触投影，重建法向/twist，保留端点 | 未包裹 intrinsic 状态线性插值后解析 lift |
| `on_smooth` | 在 `[u,v,psi]` 域固定 16 控制点 clamped cubic 有界拟合，正则 lambda=1e-6，再输出 128 接触节点 | 仍按 intrinsic 线性/lift；不是把连续样条本身当作碰撞证书 |

平滑补充参数在正式补充评测前固定，未按 benchmark 结果调参。它不做障碍修复或全局绕障，也不读专家、oracle 或路线模式。KKT 容差 1e-10、最大 512 sweeps；拟合异常、未收敛和失败候选保留在分母。

完整成功需同时通过：接触 ≤1 mm、法向 ≤1°、有效姿态、workspace、给定端点、接触图域/twist/禁入规则、shape quality、完整 URDF 物理碰撞和有限步长密集边审核。Numeric success 另用 ≤2e-6 m/rad。

Shape 门限固定为最大曲率 **8 m^-1**、累计回退 ≤0.25 m、16 interval 局部 chord/arc ≥0.65。完整 URDF 审核包含七个 active collision primitives；先检查 128 节点，通过前置 gates 后才做 ≤2 cm / 0.02 rad 密集边检查。未做密集检查的失败候选不能改称完整成功；节点净空已扣 margin，不能当成整段密集最小净空。

原始 10,000 条件都有推理前物理见证。移障后见证可行子集为 8,921，另有 755 assembly-invalid、324 endpoint-invalid；主 `all_stress` 分母仍保留全部 10,000。预算内未找到见证一般应记 unknown，而不是不可行证明；配对可行分析使用两场景见证交集。

统计使用 family 分层、以地图为 cluster 的 1,000 次 percentile bootstrap，seed 20261011。三训练 seed 的样本 SD 与地图 CI 分开；地图 CI 条件于已训练的这些 seeds，不替代训练随机性的置信区间，也未做多重比较校正。

## 9. 最终结果摘要

**所有 22 模型的 `off/on`，两场景的 K1 与 Any@8 严格成功率均为 0。`on_smooth` 获得非零成功率；原来的零结果没有被删除或重写。**

以下为 `on_smooth` K1，单位 %，格式“3 seed mean ± sample SD [95% map-cluster CI]”：

| 专家数 | 原始场景 | 移障场景 |
|---:|---:|---:|
| 500 | 17.81 ± 0.77 [17.17, 18.42] | 18.62 ± 1.38 [18.00, 19.28] |
| 1,000 | 19.38 ± 0.82 [18.69, 20.08] | 20.53 ± 0.81 [19.81, 21.26] |
| 2,000 | 19.63 ± 0.30 [18.96, 20.32] | 21.30 ± 0.75 [20.58, 22.01] |
| 4,000 | 20.12 ± 0.89 [19.47, 20.79] | 21.27 ± 1.64 [20.62, 21.95] |
| 8,000 | 19.31 ± 0.82 [18.63, 19.98] | 20.87 ± 0.73 [20.16, 21.59] |

4k 原始均值最高，数据量收益不单调；不要据此重新挑 checkpoint 或给 decoder 调参。

8k 和对照在相同平滑 decoder 下：

| 模型 | 原始 K1 | 移障 K1 |
|---|---:|---:|
| Base（单个冻结模型） | 0.91 [0.73, 1.11] | 0.82 [0.65, 1.01] |
| geometry 8k | 19.31 ± 0.82 | 20.87 ± 0.73 |
| no_contact_loss 8k | 18.35 ± 2.40 | 20.00 ± 1.90 |
| no_geometry_tokens 8k | 8.79 ± 0.67 | 9.68 ± 0.75 |

原始场景 matched-seed 配对差（百分点）：geometry−Base **+18.40 [17.69,19.10]**；geometry−no_contact_loss **+0.96 [0.54,1.38]**；geometry−no_geometry_tokens **+10.52 [9.97,11.08]**。辅助损失对照差的 seed SD 为 2.29 pp，优势较小、跨训练 seed 有变化；不能仅凭地图 CI 宣称训练随机性结论已稳健。

8k 原始场景的 gates：

| 指标 | off | on | on_smooth |
|---|---:|---:|---:|
| 接触通过率 | 0% | 100% | 100% |
| 法向通过率 | 1.28% | 100% | 100% |
| Shape 通过率 | 0% | 0% | 60.85% |
| 128 节点完整 URDF 审核下的候选无碰撞率 | 46.63% | 46.59% | 47.53% |
| 最终完整成功率 | 0% | 0% | 19.31% |

8k raw 最大接触误差均值约 119.77 mm，Base 约 1,308.64 mm。Adapter 明显改善接触几何，但 raw 仍不满足 1 mm 门限。普通投影满足接触/法向后仍全因 shape 失败；平滑改进后仍受到形状和全身碰撞约束。节点无碰撞率不能代替完整成功率。

移障配对分析（8k smooth）：

- 成功率 shift−original：**+1.57 pp [1.08,2.09]**。
- 相对 `on` 的移动效果 DID：**+1.57 pp [1.07,2.03]**，受 `on` 成功恒为 0 的 floor 限制。
- 节点无碰撞率下降 **1.64 pp**，CI [-2.25,-1.05]；该指标 DID **-0.11 pp [-0.29,0.07]**，跨零。

8k 原始 smooth 分层：plane **21.63%**、cylinder **16.99%**；easy **31.77%**、detour **11.00%**；家族最低 `wall_protrusion_bracket` **12.23%**、最高 `multi_homotopy` **23.60%**。需绕障的接触任务仍是明显瓶颈。

8k Any@8 为原始 **52.83 ±2.53% [50.50,55.17]**、移障 **61.53 ±0.85% [59.20,63.94]**；Base 为 4.2% / 4.7%。这是固定 1,000 条件的 K8 指标，不能与全部 10,000 条件 K1 相减作配对增益。

成功候选成对几何多样性：原始位置 pair RMS 均值 **0.340916 m [0.296938,0.386527]**，移障 **0.284684 m [0.256988,0.312611]**。至少一个 seed 有可计算成功 pair 的条件为 445/1000 与 519/1000；各 seed 有效条件可能不同。少于两个成功候选的单 seed 条件为 null，不能填零；off/on 因零成功而没有成功 pair 诊断。该统计是成功条件中的几何差异，不是所有请求的覆盖或同伦类完备证明。

## 10. 可视化与曲率问题的实际诊断

用户要求看最接近成功的接触轨迹，解释零成功率。诊断在已保存数据上扫描 15 个 geometry 模型（五规模 × 三 seed）的原始 K1；先要求所有非 shape 节点 gates 合格，再按曲率等质量指标挑条件。它是条件化最佳案例，不是总体成功率或最终 smooth 最优模型的证明。

选中：`geometry_n8000_seed17`，`map_000527/condition_002`，`original`，candidate 0，plane / detour / `pillar_wall`，runtime input row 10544。

- 普通投影 `on` 唯一保存的失败类型为 `shape_quality`；最大曲率 15.422555 m^-1，回退约 0.04119 m，chord16 ratio 0.94860。
- 接触最大误差约 1.76e-7 m、法向误差约 7.86e-9 rad；节点 margin 后最小净空 +0.07241 m。
- `dense_checked=false`：因为 shape 前置 gate 失败，不能仅放宽门限就把它标成完整成功。
- Raw 最大曲率约 14.9696 m^-1，接触最大误差约 0.27454 m、法向最大误差约 8.56°，还失败 contact/normal/chart。
- 图中 body XYZ 与 tool-tip 位置要分清；接触约束验证的是工具点/法向，身体中心可以因工具外参而远离曲面。

诊断文件：

- [轨迹与局部曲率图](/home/z017/research/HNUTER_DiT/artifacts/contact_adapter_multifamily_v001/trajectory_diagnostics_20261013/best_8k/trajectory_overview.png)
- [接触、法向、曲率和节点净空](/home/z017/research/HNUTER_DiT/artifacts/contact_adapter_multifamily_v001/trajectory_diagnostics_20261013/best_8k/failure_diagnostics.png)
- [实际路径 NPZ](/home/z017/research/HNUTER_DiT/artifacts/contact_adapter_multifamily_v001/trajectory_diagnostics_20261013/best_8k/paths.npz)
- [诊断与来源哈希](/home/z017/research/HNUTER_DiT/artifacts/contact_adapter_multifamily_v001/trajectory_diagnostics_20261013/best_8k/diagnosis.json)
- [全 geometry 条件化选择记录](/home/z017/research/HNUTER_DiT/artifacts/contact_adapter_multifamily_v001/trajectory_diagnostics_20261013/selection.json)

用户追问“最大曲率在起点，是否没有前继或启动导致”。实际计算不含端点 0/127。选中 `on` 最大值在**节点 2**，使用节点 1/2/3；两段长度约 24.87 mm，转角约 21.98°。全路径有 22 个内点超过 8 m^-1；即使诊断性忽略首尾 5 或 10 点，仍有约 14.5615 m^-1 的内部峰值（节点 26）。因此不能把全部 shape 失败解释为首点没有前继或“启动加速度”。红色标记仅表示全局 argmax，并不代表只有该点超阈值。

用户也询问是否可放宽阈值。8 m^-1 来自采集器的几何自然度/质量筛选，不是已从推力、控制带宽、速度或接触力推导出的硬物理极限。对真正光滑、有时间参数的路径，法向加速度满足 `a_n=v²κ`，但当前归一化弧长轨迹没有物理时间。离散 shape proxy、每节点的 acceleration/jerk 也不能直接解释成实际飞行加速度。

**正式阈值没有改变，8/12/16/24/32 的敏感性实验也尚未执行。**若未来授权该补充，应保留 v001 原结果、另建协议；对新通过 shape 的候选还需补做以前被跳过的密集审核，不能只重标 shape 后直接统计完整成功。

## 11. 执行过程、温控与报告恢复

全部大型实验使用 `/home/z017/research/curobo_env/bin/python`。记录环境包括 Python 3.10.20、Torch 2.13.0+cu130、NumPy 2.2.6、CUDA 13.0、COAL 3、OMPL 2、SciPy 1.15.3、Matplotlib 3.10.9。机器为 RTX 4080 Laptop 12GB、AMD 16C/32T，正式任务 affinity 8,9，数值库线程 1。

温控规则：CPU ≥90°C 或 GPU ≥83°C 时暂停本守护创建的专用子组；CPU ≤80°C 且 GPU ≤75°C 稳定 10 秒后恢复，poll 1 秒。这是反应式控制，不能宣称温度从未短暂超过阈值；暂停计入 wall time。只能在核对 PID/start_ticks/UID/PGID/SID/完整命令后处理本实验进程，不得影响其他任务。

主要完成时间（Singapore）：

| 阶段 | 时间/已保存 wall time |
|---|---|
| 主流水线启动 | 2026-10-11 10:58:02 |
| 主流水线完成 | 2026-10-13 05:44:46；整体约 42.78 h，含等待训练和温控 |
| 新测试地图/条件构建 | 约 20.47 h |
| 全模型原始预测生成 | 约 12.98 h |
| off/on 评分 | 约 5.82 h |
| 主报告 | 约 8.05 min |
| 平滑评分 | 约 6.06 h；复用 raw，没有重训 |
| 平滑报告完成 | 2026-10-13 12:25:56；报告阶段约 37.14 min |
| v002 汇总恢复完成 | 2026-10-13 13:10:46；约 9.22 min，含约 351 s 温控暂停 |
| 最终交付完成 | 2026-10-13 13:19:46 |

训练过程早期曾因前台守护退出被身份审核阻止下游继续，保存了已有完整模型与恢复证据，再转独立后台串行流水线；没有通过删除结果或缩减预算继续。不能把未确认的旧退出原因自动归因于过热。

测试比训练耗时长的原因可由实际阶段时间和规模解释：训练使用缓存和可微近似；验证需要新地图/可行条件构建、全 22 模型多步扩散、两场景、多候选、完整 URDF/密集边与三 decoder，另含温控暂停。它不是只对一个 adapter 测一次。

最后的汇总故障是确定的代码 schema 问题：原 `report_contact_ablation_appendix.py` 对所有源码契约强取 `input_files`；合法的 `contact_smooth_projection_sources_v001` 本来只有 `source_files` 和 `frozen_dependency_sources`。因此 v001 汇总在 2026-10-13 12:26:38 抛 `KeyError: 'input_files'`，未进入正式导出。

恢复采取新文件和新目录，保留旧源码/failed 状态/日志/快照：

- `report_contact_ablation_appendix_v002.py` 仅对已知 code-only schema 接受缺省 `input_files`，仍核验完整源码/依赖 SHA；其他 pipeline 的 `input_files` 继续强制必需。
- `contact_ablation_report_recovery_v002.py` 在 `consolidated_reports/recovery_v002` 只做一次完整输入审核与汇总；没有训练、采样、decoder 重放、碰撞重评分或新 bootstrap。
- `EXP/report_delivery_v002.py` 审核完成状态、哈希和全部附录覆盖，再只改 Markdown 本地链接为绝对路径，创建根交付文档与附件链接；统计正文不变。
- 恢复层联合回归 26 项、交付 helper 回归 16 项通过。这些是完整性代码测试，不是额外科学实验样本。
- 正式恢复审核了 177,262 个输入文件绑定，交付另外流式计数 2,263,800 附录行并核对总体 K1/K8 矩阵。

## 12. 实现入口地图

下列文件均位于 `/home/z017/research/HNUTER_DiT/hnuter_dit/`：

| 模块 | 职责 |
|---|---|
| `contact_adapter.py` | task encoder、冻结 Base 残差、canonicalize、adapter 加载 |
| `train_contact_adapter.py` | 训练、EMA、共同验证与 checkpoint 恢复 |
| `contact_experiment.py` | 固定模型矩阵、数据子集与协议 |
| `contact_benchmark.py` | 独立地图/条件与 paired 移障输入 |
| `contact_inference.py` | 统一 DDIM、共享 latent 与推理输入白名单 |
| `evaluate_contact_experiment.py` | 分离 generate/score、按地图原子分片 |
| `contact_metrics.py` | 实际 pose、完整 URDF、gates、节点和密集边审核 |
| `contact_projection.py` | 普通解析接触投影 |
| `contact_smooth_projection.py` | 固定平滑拟合 decoder |
| `evaluate_contact_smooth_projection.py` | 保存 raw 的平滑补充评分与 projected 分片 |
| `contact_pipeline.py` | 主流水线 |
| `contact_smooth_pipeline.py` | 平滑补充流水线 |
| `report_contact_experiment.py` / `report_contact_smooth_projection.py` | 两条上游的正式统计与科学图 |
| `contact_ablation_report_pipeline.py` / `report_contact_ablation_appendix.py` | 原 v001 汇总；保留故障现场，勿直接重跑 |
| `contact_ablation_report_recovery_v002.py` / `report_contact_ablation_appendix_v002.py` | 成功的报告专用恢复 |
| `thermal_run.py` | 身份绑定与本组温控 |

## 13. 关键身份哈希

下列为算法/数据身份和交付核验值，而非 Git commit：

| 对象 | SHA-256 |
|---|---|
| Base checkpoint | `301070cd16c847d749da548bca4d66e409e7ed698346c9b938f3ddf782936dad` |
| Base statistics | `c68607d14de96626d83f95012fd68ccabcd23c12983c6755749b17628aeec3c8` |
| 训练 dataset_freeze.json | `60140df62fe2b8c13f84289770ba9b65730007535e8049888661056822936d37` |
| 训练 manifest.json | `222df415be520add71bae725746998e5d85a5f104212cbb158d7fc9b927a922b` |
| 训练 prepared cache | `0be26388dd787752140c24f20a788585a4b28d2aeba4352254eb9a9b22df058f` |
| Benchmark manifest.json | `418d2fb08dca2289fba7c1d0cbf43cf4a6c073ec03694a944146c113022feb97` |
| 主 pipeline 源契约内部 hash | `6b203bdd895a402604a25e0a78b036b68f7c1d0e0d3c7855ddae3cd12a739845` |
| 平滑 pipeline 源契约内部 hash | `d8422cba4411a5e0e59975f3a9ecdc05b011e79477c81467abc0feb339a839da` |
| v002 恢复源契约内部 hash | `adee8cb81f2e4f5ef485841552c6d7807939190406660f4de356ebb83b2bba78` |
| 父 report.json | `024270233a72825cff646464261b5b4ea2f5f32d854db70490f31c9899d9893d` |
| 平滑 report.json | `d15469eb25a15d36da13458413f42f73e01b60aff380cf1a0270eb4a764147cd` |
| v002 恢复 report.json | `822003b4dd0951046ea568c61d01f2ff0c03d11abecd0304f2fdd6e0ea51a3b9` |
| 根交付 Markdown | `d1292a34444a219c31416bbd3249b2d5f21ff67928947010ca0edba2689272b1` |
| CSV.gz | `c906f74b38c5650c69b9ce4e105e4552ca3ba1ae7e2d77ca82b4c6aea6130ca2` |
| JSONL.gz | `e0e1e0090ee6d8217498e72ad24075c626ac780ab64c456c31a15eb57f53b723` |

契约内部 `contract_sha256` 不等于该 JSON 文件字节 SHA；复核时不要混淆。全部逐模型 checkpoint、分片与源文件哈希见 report/contract/commit，不需将本交接当作完整替代。

## 14. 可直接执行的轻量只读检查

以下脚本仅读取小状态文件，不启动任务：

```bash
cd /home/z017/research/HNUTER_DiT
/home/z017/research/curobo_env/bin/python - <<'PY'
import json
from pathlib import Path
r = Path('/home/z017/research/HNUTER_DiT/artifacts/contact_adapter_multifamily_v001')
for name in (
    'pipeline_status.json',
    'thermal/pipeline/status.json',
    'smooth_projection_supplement/pipeline_status.json',
    'smooth_projection_supplement/thermal/pipeline/status.json',
    'consolidated_reports/recovery_v002/pipeline_status.json',
    'consolidated_reports/recovery_v002/thermal/pipeline/status.json',
    'consolidated_reports/recovery_v002/thermal/delivery/status.json',
    'consolidated_reports/delivery_commit.json',
):
    d = json.loads((r / name).read_text())
    print(name, d.get('status'), d.get('exit_code'), d.get('current_stage'))
PY
```

本轮预期输出均为 complete；pipeline 状态不一定有 exit_code，守护状态有 0。旧根 `consolidated_reports/pipeline_status.json` 的 failed 不是此清单中的权威完成状态。

## 15. 后续方向与尚未完成事项

本轮采集、训练、完整消融、泛化验证与报告交付已经完成。下面是供未来讨论的研究方向，不是读取本文件就授权自动启动的新实验：

1. 针对 raw 的高频形状误差、端点处理和任务条件表达做诊断；完整 adapter 的 raw 仍未达严格接触门限。
2. 考虑训练中更直接的平滑/intrinsic 轨迹表达与质量损失；当前结果体现“adapter + decoder”的作用，不能把投影构造的精确接触全部归因于神经网络输出。
3. 强化 detour 和完整机体碰撞学习，核对 42 球 surrogate、目标实体和完整 URDF 之间的差异；绕障条件当前仅约 11% K1。
4. 如果用户确认，另做曲率阈值敏感性和分层质量/几何可行/动力学评价，保留原始 8 m^-1 结果，并补足新的密集审核。
5. 之后再研究动态雷达、工厂语义 OOD、approach/retreat、接触力与闭环动力学执行。

仍未验证：真实飞行/接触力/摩擦/真实工具形状，工厂语义 OOD，圆柱全周任意法向，启用 forbidden 区域的数据行为，阈值放宽实验，Inspection adapter。

已冻结 v001 的源码、参数、数据、模型选择与结果均应保持可追溯。后续修改使用明确的新版本/实验目录与新契约，避免把新结果混入旧报告；保留当前源码快照与已有未提交工作。禁止用删除结果、减少种子/条件/更新/候选或修改成功定义来掩盖失败。

## 16. 给新会话的启动消息

可以将下面这段直接粘贴给新会话：

> 请先阅读 `/home/z017/research/HNUTER_DiT/docs/CONTACT_ADAPTER_HANDOFF_20261013.md` 和其中链接的完整中文报告，理解当前 Contact/Surface adapter 实验上下文。工作目录是 `/home/z017/research/HNUTER_DiT`，当前开发分支是 `codex/v003-task-adapters`。本轮 21 个 adapter 和 22 模型的两场景三 decoder 验证已全部完成，后续检查已删除。旧 consolidated 根 failed 状态、训练 pending 状态、旧 ACTIVE 记录都是已解释的历史文件，请不要据此重复启动或覆盖冻结实验。先用轻量只读检查确认关键文件可访问，再根据我接下来的具体任务继续；不要仅凭读取交接文档自动启动新的训练或评测。

Codex 新会话不会因同一个项目名自动拥有完整旧聊天；让新会话明确读取本文件即可恢复主要上下文。若在其他机器或新 checkout，需要同时提供本文件及所需源码、模型和数据，绝对路径也需要按实际位置重新对应。

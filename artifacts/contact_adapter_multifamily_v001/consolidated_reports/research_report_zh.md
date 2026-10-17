# Contact/Surface 完整科研消融报告

本报告整合已经严格完成且文件校验通过的父实验与平滑投影补充实验。范围是 plane/cylinder 曲面上的连续几何接触路径；不属于 Inspection，也未验证接触力或飞行动力学。

## 实验矩阵与公平性

冻结 Base 加 21 个 adapter：500、1000、2000、4000、8000 条专家路径各三个种子（17/29/43），另有 8k 的两个三种子负控。Base 权重未更新。每个 adapter 为 8000 次 optimizer 更新、batch 64，共 512000 次样本呈现；每条专家近似呈现次数随五档数据量分别为 1024/512/256/128/64；小数据预算获得更多重复呈现，这是固定 compute 的数据量比较，并非等 epoch 实验。

主 K1：1000 张未见源地图、10000 个条件，各在 original 与 obstacle_shift 场景评测 candidate_index=0。K8：推理前固定的同一 1000 条条件，两场景均保留八个候选。统一 DDIM 25 步、同一 latent seed=20261011、同一条件/候选与相同冻结 Base。

本次 Base 是 frozen v003 的统一 DDIM、无额外碰撞引导先验；adapter 也使用相同统一采样。旧 Base free-flight 报告中的 pointwise/smooth collision-guided 成功率不能直接与此 Contact 成功率比较。沿用同一个障碍 shift function 只保证障碍移动定义一致，不表示旧 guided 协议的整个采样与评测一致。

三 decoder：off 为原始输出，on 为冻结解析接触投影，on_smooth 为固定 16 控制点的有界三次拟合后 128 个接触节点。正则 λ=1e-6，边仍按 v001 线性 intrinsic/lift 解释；没有障碍修复、专家或 oracle 模式判别，也没有重新拟合保存轨迹。

no_contact_loss 的准确含义是同时移除 contact、normal、smooth 三个辅助损失，保留轨迹去噪与实际配置中的 collision 项；no_geometry_tokens 保留 adapter 与损失，但注意力输入仅起终点 token。两项控制检验不同假设。

95% CI 是按 family 分层、以源地图为 cluster 的 1000 次 percentile bootstrap（seed 20261011）；± 后数值是三个训练种子的样本 SD，不能将地图 CI 当作训练 seed 方差的 CI。所有区间沿用已审计报告，不重新 bootstrap。单模型/Base 不展示不存在的三 seed SD。

失败、投影异常和不收敛候选留在请求分母内。诊断缺失保留空值，不能填 0。frozen_feasible 只表示推理前物理见证；unknown 不等于不可行，见证也不保证本实验 128 节点的 shape-quality gate 可满足。移动差值的可行子集采用两场景见证交集。

## 有效训练数据与完成证据

| 专家路径 | 独立源地图 | 不同任务条件 |
|---:|---:|---:|
| 500 | 25 | 350 |
| 1000 | 50 | 700 |
| 2000 | 100 | 1400 |
| 4000 | 200 | 2800 |
| 8000 | 400 | 5600 |

每张地图 20 条专家路径、14 个不同任务条件；成对绕行专家、反向增强与重复更新不增加独立地图或条件。

| 模型 | updates / presentations | best_step | common validation_total | selected adapter SHA256 |
|---|---:|---:|---:|---|
| geometry_n0500_seed17 | 8000 / 512000 | 8000 | 0.045938001 | `3ce9d4dfa5509f9fb1c37111c4c5255470ca4cbf87bd21b6859c3c2c955e9f38` |
| geometry_n0500_seed29 | 8000 / 512000 | 8000 | 0.04556473 | `77391ceb743f0309827dee9a67cf541f6a546d09b4c8d288c41531189e6f43f6` |
| geometry_n0500_seed43 | 8000 / 512000 | 7500 | 0.045772483 | `172b4df7796d8a75c34cf442949e265281822a6d7d95c646c1ec7739e0340215` |
| geometry_n1000_seed17 | 8000 / 512000 | 8000 | 0.044029956 | `f3cc1930424b37828c9ed2295200904212d25c84d32ca1610905220eec7a89fe` |
| geometry_n1000_seed29 | 8000 / 512000 | 7500 | 0.043934682 | `54b721ab5ef4c550f389e691b4e90694fc4852668c4a0b121546723375abfb3f` |
| geometry_n1000_seed43 | 8000 / 512000 | 8000 | 0.044174353 | `40c635918c2473541b6dd3853649b371e96889dc79ad025b7cba2fcc71211798` |
| geometry_n2000_seed17 | 8000 / 512000 | 8000 | 0.04389208 | `3f58941347ae90653dbaebd508cad2f3da6699a4e081aaac94726fe9ebff3ac1` |
| geometry_n2000_seed29 | 8000 / 512000 | 8000 | 0.042950742 | `b59bfcc90b2cb0f8749feb35775edd620984dcb221a770685d1d12535d3962a8` |
| geometry_n2000_seed43 | 8000 / 512000 | 8000 | 0.043589311 | `e21b1de0b31c8b8ed70cdedf63e2a5b9760c36d00b24a6977c495d2adaf86238` |
| geometry_n4000_seed17 | 8000 / 512000 | 8000 | 0.044931974 | `938bc1ca029dd1ecffca726bbaf3c58d11bff83f5c8494273b72d0f14b5548ce` |
| geometry_n4000_seed29 | 8000 / 512000 | 8000 | 0.043312155 | `26cae7c3cf652ea6ba1084842ec8ff83cd8b6888892c1cf8701c8be69054dc6e` |
| geometry_n4000_seed43 | 8000 / 512000 | 7500 | 0.043703507 | `fae1a0bef8355a87c5a6f795a928d51a0b3ea506dda3fd1c9ecdff1a0e20302b` |
| geometry_n8000_seed17 | 8000 / 512000 | 8000 | 0.045162093 | `36ad5a59ce400531af12132250c65a8b152faa537908611fda830f68a2cbcd63` |
| geometry_n8000_seed29 | 8000 / 512000 | 8000 | 0.043001074 | `7a91c53637fa62d187cfaf2112296f11aadc3ea83612163568fa5c9960038b13` |
| geometry_n8000_seed43 | 8000 / 512000 | 8000 | 0.043280733 | `5f4a8560ee1586b1a9790167394e59a2f1a3e2b4f2d30fcbdad13437639ba00f` |
| no_contact_loss_n8000_seed17 | 8000 / 512000 | 8000 | 0.047118258 | `3008f7f9a1726b69bc27b4b808c0139ca8690960607fa677e6154a3c008514d4` |
| no_geometry_tokens_n8000_seed17 | 8000 / 512000 | 8000 | 0.058820101 | `bf26fa8b40c5ac85822781bfea17fac577630e3e2ec2c820ab817281d619645e` |
| no_contact_loss_n8000_seed29 | 8000 / 512000 | 8000 | 0.046453058 | `5b07b762554e9eac29b329091c4e2dd9bd48331e46c14c28277da03a3e026ae0` |
| no_geometry_tokens_n8000_seed29 | 8000 / 512000 | 8000 | 0.060226246 | `4a5b23a33a62abfc762d38d33731a1029c15964659c6285eba91dbd116d736a2` |
| no_contact_loss_n8000_seed43 | 8000 / 512000 | 8000 | 0.04510557 | `5388aa285ac0dd41bf65df15009ff76185e6a261688081d9953ec266626f80e0` |
| no_geometry_tokens_n8000_seed43 | 8000 / 512000 | 8000 | 0.060156086 | `8eb2250d684e99b7f7a0e612e03f3b2b101f0b7bface50a6b8c1f0435add83e4` |

冻结 Base SHA256：`301070cd16c847d749da548bca4d66e409e7ed698346c9b938f3ddf782936dad`。上述 validation_total 是同一固定噪声验证目标的模型选择损失，**不是任务成功率**。

## 主 K1 成功率

表格为 mean ± seed SD [95% map-cluster CI]，单位 %。

### 全部压力条件

| 变体 / 专家数 | 场景 | 条件数 | off | on | on_smooth |
|---|---|---:|---:|---:|---:|
| base / 0 | original | 10000 | 0.00% [0.00, 0.00] | 0.00% [0.00, 0.00] | 0.91% [0.73, 1.11] |
| base / 0 | obstacle_shift | 10000 | 0.00% [0.00, 0.00] | 0.00% [0.00, 0.00] | 0.82% [0.65, 1.01] |
| geometry / 500 | original | 10000 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 17.81 ± 0.77% [17.17, 18.42] |
| geometry / 500 | obstacle_shift | 10000 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 18.62 ± 1.38% [18.00, 19.28] |
| geometry / 1000 | original | 10000 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 19.38 ± 0.82% [18.69, 20.08] |
| geometry / 1000 | obstacle_shift | 10000 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 20.53 ± 0.81% [19.81, 21.26] |
| geometry / 2000 | original | 10000 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 19.63 ± 0.30% [18.96, 20.32] |
| geometry / 2000 | obstacle_shift | 10000 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 21.30 ± 0.75% [20.58, 22.01] |
| geometry / 4000 | original | 10000 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 20.12 ± 0.89% [19.47, 20.79] |
| geometry / 4000 | obstacle_shift | 10000 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 21.27 ± 1.64% [20.62, 21.95] |
| geometry / 8000 | original | 10000 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 19.31 ± 0.82% [18.63, 19.98] |
| geometry / 8000 | obstacle_shift | 10000 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 20.87 ± 0.73% [20.16, 21.59] |
| no_contact_loss / 8000 | original | 10000 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 18.35 ± 2.40% [17.66, 18.99] |
| no_contact_loss / 8000 | obstacle_shift | 10000 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 20.00 ± 1.90% [19.34, 20.69] |
| no_geometry_tokens / 8000 | original | 10000 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 8.79 ± 0.67% [8.32, 9.23] |
| no_geometry_tokens / 8000 | obstacle_shift | 10000 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 9.68 ± 0.75% [9.15, 10.21] |

### 冻结物理见证子集

| 变体 / 专家数 | 场景 | 条件数 | off | on | on_smooth |
|---|---|---:|---:|---:|---:|
| base / 0 | original | 10000 | 0.00% [0.00, 0.00] | 0.00% [0.00, 0.00] | 0.91% [0.72, 1.11] |
| base / 0 | obstacle_shift | 8921 | 0.00% [0.00, 0.00] | 0.00% [0.00, 0.00] | 0.92% [0.73, 1.14] |
| geometry / 500 | original | 10000 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 17.81 ± 0.77% [17.22, 18.42] |
| geometry / 500 | obstacle_shift | 8921 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 19.06 ± 1.62% [18.40, 19.78] |
| geometry / 1000 | original | 10000 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 19.38 ± 0.82% [18.74, 20.03] |
| geometry / 1000 | obstacle_shift | 8921 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 20.91 ± 0.80% [20.25, 21.63] |
| geometry / 2000 | original | 10000 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 19.63 ± 0.30% [19.03, 20.25] |
| geometry / 2000 | obstacle_shift | 8921 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 21.75 ± 0.91% [21.03, 22.49] |
| geometry / 4000 | original | 10000 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 20.12 ± 0.89% [19.52, 20.77] |
| geometry / 4000 | obstacle_shift | 8921 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 21.59 ± 1.76% [20.82, 22.31] |
| geometry / 8000 | original | 10000 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 19.31 ± 0.82% [18.72, 19.91] |
| geometry / 8000 | obstacle_shift | 8921 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 21.26 ± 0.78% [20.50, 22.02] |
| no_contact_loss / 8000 | original | 10000 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 18.35 ± 2.40% [17.76, 19.00] |
| no_contact_loss / 8000 | obstacle_shift | 8921 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 20.68 ± 1.82% [19.99, 21.38] |
| no_geometry_tokens / 8000 | original | 10000 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 8.79 ± 0.67% [8.33, 9.25] |
| no_geometry_tokens / 8000 | obstacle_shift | 8921 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 10.20 ± 0.82% [9.65, 10.80] |

## 固定子集 K8 Any@8

表格为 mean ± seed SD [95% map-cluster CI]，单位 %。

### 全部压力条件

| 变体 / 专家数 | 场景 | 条件数 | off | on | on_smooth |
|---|---|---:|---:|---:|---:|
| base / 0 | original | 1000 | 0.00% [0.00, 0.00] | 0.00% [0.00, 0.00] | 4.20% [3.10, 5.50] |
| base / 0 | obstacle_shift | 1000 | 0.00% [0.00, 0.00] | 0.00% [0.00, 0.00] | 4.70% [3.60, 5.80] |
| geometry / 500 | original | 1000 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 51.50 ± 2.27% [49.10, 54.00] |
| geometry / 500 | obstacle_shift | 1000 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 57.23 ± 3.96% [54.90, 59.57] |
| geometry / 1000 | original | 1000 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 53.30 ± 1.01% [50.67, 56.00] |
| geometry / 1000 | obstacle_shift | 1000 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 60.83 ± 0.50% [58.50, 63.17] |
| geometry / 2000 | original | 1000 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 53.17 ± 0.76% [50.57, 55.74] |
| geometry / 2000 | obstacle_shift | 1000 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 61.60 ± 1.83% [59.27, 63.90] |
| geometry / 4000 | original | 1000 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 54.50 ± 2.13% [51.96, 56.97] |
| geometry / 4000 | obstacle_shift | 1000 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 61.73 ± 3.24% [59.23, 64.07] |
| geometry / 8000 | original | 1000 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 52.83 ± 2.53% [50.50, 55.17] |
| geometry / 8000 | obstacle_shift | 1000 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 61.53 ± 0.85% [59.20, 63.94] |
| no_contact_loss / 8000 | original | 1000 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 55.40 ± 2.67% [52.77, 57.90] |
| no_contact_loss / 8000 | obstacle_shift | 1000 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 61.80 ± 0.46% [59.60, 64.20] |
| no_geometry_tokens / 8000 | original | 1000 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 33.90 ± 0.95% [31.53, 36.30] |
| no_geometry_tokens / 8000 | obstacle_shift | 1000 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 37.80 ± 1.68% [35.60, 39.97] |

### 冻结物理见证子集

| 变体 / 专家数 | 场景 | 条件数 | off | on | on_smooth |
|---|---|---:|---:|---:|---:|
| base / 0 | original | 1000 | 0.00% [0.00, 0.00] | 0.00% [0.00, 0.00] | 4.20% [3.10, 5.40] |
| base / 0 | obstacle_shift | 893 | 0.00% [0.00, 0.00] | 0.00% [0.00, 0.00] | 5.26% [3.95, 6.63] |
| geometry / 500 | original | 1000 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 51.50 ± 2.27% [48.90, 54.07] |
| geometry / 500 | obstacle_shift | 893 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 59.91 ± 4.54% [57.63, 62.13] |
| geometry / 1000 | original | 1000 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 53.30 ± 1.01% [50.70, 55.80] |
| geometry / 1000 | obstacle_shift | 893 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 63.79 ± 0.76% [61.40, 66.22] |
| geometry / 2000 | original | 1000 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 53.17 ± 0.76% [50.70, 55.63] |
| geometry / 2000 | obstacle_shift | 893 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 64.43 ± 2.31% [61.96, 66.85] |
| geometry / 4000 | original | 1000 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 54.50 ± 2.13% [52.10, 56.90] |
| geometry / 4000 | obstacle_shift | 893 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 64.58 ± 3.66% [62.24, 66.96] |
| geometry / 8000 | original | 1000 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 52.83 ± 2.53% [50.30, 55.17] |
| geometry / 8000 | obstacle_shift | 893 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 64.39 ± 1.03% [61.90, 66.70] |
| no_contact_loss / 8000 | original | 1000 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 55.40 ± 2.67% [52.87, 57.80] |
| no_contact_loss / 8000 | obstacle_shift | 893 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 65.43 ± 0.13% [63.06, 67.97] |
| no_geometry_tokens / 8000 | original | 1000 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 33.90 ± 0.95% [31.60, 36.30] |
| no_geometry_tokens / 8000 | obstacle_shift | 893 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 39.87 ± 1.57% [37.58, 42.19] |

## 22 模型逐种子 K1 / K8

以下逐模型区间均为源地图 CI；完整 frozen-feasible 与全部分层逐种子指标见机器可读附录。

| 模型 / seed | 场景 | K1 off / on / smooth | K8 off / on / smooth |
|---|---|---|---|
| base / None | original | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 0.91% [0.73, 1.11] | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 4.20% [3.10, 5.50] |
| base / None | obstacle_shift | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 0.82% [0.65, 1.01] | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 4.70% [3.60, 5.80] |
| geometry_n0500_seed17 / 17 | original | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 16.97% [16.29, 17.63] | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 48.90% [46.00, 51.80] |
| geometry_n0500_seed17 / 17 | obstacle_shift | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 17.17% [16.48, 17.91] | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 53.40% [50.50, 56.20] |
| geometry_n0500_seed29 / 29 | original | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 17.99% [17.23, 18.74] | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 53.10% [50.00, 56.30] |
| geometry_n0500_seed29 / 29 | obstacle_shift | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 19.92% [19.19, 20.72] | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 61.30% [58.60, 63.80] |
| geometry_n0500_seed43 / 43 | original | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 18.48% [17.71, 19.26] | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 52.50% [49.60, 55.20] |
| geometry_n0500_seed43 / 43 | obstacle_shift | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 18.77% [18.00, 19.54] | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 57.00% [54.30, 59.80] |
| geometry_n1000_seed17 / 17 | original | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 19.52% [18.80, 20.28] | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 53.10% [50.10, 56.10] |
| geometry_n1000_seed17 / 17 | obstacle_shift | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 20.28% [19.49, 21.05] | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 60.30% [57.60, 63.20] |
| geometry_n1000_seed29 / 29 | original | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 18.50% [17.75, 19.32] | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 52.40% [49.70, 55.20] |
| geometry_n1000_seed29 / 29 | obstacle_shift | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 19.87% [19.08, 20.68] | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 60.90% [58.40, 63.30] |
| geometry_n1000_seed43 / 43 | original | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 20.12% [19.39, 20.85] | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 54.40% [51.10, 57.70] |
| geometry_n1000_seed43 / 43 | obstacle_shift | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 21.43% [20.63, 22.24] | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 61.30% [58.60, 64.00] |
| geometry_n2000_seed17 / 17 | original | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 19.55% [18.76, 20.35] | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 52.30% [49.50, 55.20] |
| geometry_n2000_seed17 / 17 | obstacle_shift | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 20.60% [19.77, 21.40] | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 60.00% [57.10, 63.10] |
| geometry_n2000_seed29 / 29 | original | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 19.97% [19.20, 20.77] | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 53.50% [50.40, 56.60] |
| geometry_n2000_seed29 / 29 | obstacle_shift | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 22.10% [21.26, 22.93] | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 63.60% [61.10, 66.00] |
| geometry_n2000_seed43 / 43 | original | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 19.38% [18.59, 20.10] | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 53.70% [50.40, 56.90] |
| geometry_n2000_seed43 / 43 | obstacle_shift | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 21.21% [20.42, 21.99] | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 61.20% [58.50, 64.00] |
| geometry_n4000_seed17 / 17 | original | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 19.20% [18.50, 19.94] | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 52.20% [49.80, 54.90] |
| geometry_n4000_seed17 / 17 | obstacle_shift | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 19.39% [18.68, 20.11] | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 58.00% [55.30, 60.70] |
| geometry_n4000_seed29 / 29 | original | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 20.18% [19.39, 20.94] | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 54.90% [52.10, 57.60] |
| geometry_n4000_seed29 / 29 | obstacle_shift | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 22.08% [21.28, 22.87] | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 63.40% [60.70, 66.10] |
| geometry_n4000_seed43 / 43 | original | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 20.98% [20.22, 21.80] | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 56.40% [53.20, 59.70] |
| geometry_n4000_seed43 / 43 | obstacle_shift | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 22.35% [21.53, 23.18] | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 63.80% [61.00, 66.50] |
| geometry_n8000_seed17 / 17 | original | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 18.89% [18.09, 19.69] | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 51.90% [49.10, 54.60] |
| geometry_n8000_seed17 / 17 | obstacle_shift | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 20.07% [19.25, 20.94] | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 61.50% [59.10, 64.20] |
| geometry_n8000_seed29 / 29 | original | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 18.78% [18.00, 19.54] | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 50.90% [48.20, 53.50] |
| geometry_n8000_seed29 / 29 | obstacle_shift | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 21.49% [20.65, 22.32] | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 62.40% [59.60, 65.10] |
| geometry_n8000_seed43 / 43 | original | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 20.25% [19.44, 21.02] | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 55.70% [52.80, 58.70] |
| geometry_n8000_seed43 / 43 | obstacle_shift | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 21.06% [20.27, 21.84] | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 60.70% [57.90, 63.50] |
| no_contact_loss_n8000_seed17 / 17 | original | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 20.36% [19.56, 21.17] | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 57.50% [54.40, 60.70] |
| no_contact_loss_n8000_seed17 / 17 | obstacle_shift | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 21.65% [20.88, 22.47] | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 62.30% [59.40, 65.10] |
| no_geometry_tokens_n8000_seed17 / 17 | original | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 9.53% [8.95, 10.07] | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 34.80% [32.20, 37.30] |
| no_geometry_tokens_n8000_seed17 / 17 | obstacle_shift | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 10.52% [9.90, 11.16] | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 39.70% [37.10, 42.20] |
| no_contact_loss_n8000_seed29 / 29 | original | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 15.70% [14.98, 16.44] | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 52.40% [49.50, 55.30] |
| no_contact_loss_n8000_seed29 / 29 | obstacle_shift | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 17.92% [17.19, 18.64] | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 61.70% [59.20, 64.20] |
| no_geometry_tokens_n8000_seed29 / 29 | original | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 8.62% [8.10, 9.17] | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 32.90% [30.30, 35.50] |
| no_geometry_tokens_n8000_seed29 / 29 | obstacle_shift | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 9.41% [8.81, 9.99] | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 36.50% [33.90, 39.20] |
| no_contact_loss_n8000_seed43 / 43 | original | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 18.99% [18.23, 19.75] | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 56.30% [53.30, 59.10] |
| no_contact_loss_n8000_seed43 / 43 | obstacle_shift | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 20.43% [19.66, 21.22] | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 61.40% [58.60, 64.20] |
| no_geometry_tokens_n8000_seed43 / 43 | original | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 8.22% [7.69, 8.72] | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 34.00% [31.10, 36.90] |
| no_geometry_tokens_n8000_seed43 / 43 | obstacle_shift | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 9.10% [8.55, 9.65] | 0.00% [0.00, 0.00] / 0.00% [0.00, 0.00] / 37.20% [34.70, 39.70] |

## K1 decoder、移动与差分配对效果

Δ 为相同条件/候选/训练 seed 内的差，再按条件聚合三个 seed。表为成功率差（pp），± seed SD [95% map CI]。K8 不在此表混用；其固定子集汇总 CI 已在上表报告。

| 变体 / 专家数 | 分母 | 场景/decoder | 对比 | Δ成功率 |
|---|---|---|---|---:|
| base / 0 | all_stress (n=10000) | original | on−off | +0.00 pp [0.00, 0.00] |
| base / 0 | all_stress (n=10000) | original | on_smooth−on | +0.91 pp [0.73, 1.11] |
| base / 0 | frozen_feasible (n=10000) | original | on−off | +0.00 pp [0.00, 0.00] |
| base / 0 | frozen_feasible (n=10000) | original | on_smooth−on | +0.91 pp [0.73, 1.10] |
| base / 0 | all_stress (n=10000) | obstacle_shift | on−off | +0.00 pp [0.00, 0.00] |
| base / 0 | all_stress (n=10000) | obstacle_shift | on_smooth−on | +0.82 pp [0.65, 1.01] |
| base / 0 | frozen_feasible (n=8921) | obstacle_shift | on−off | +0.00 pp [0.00, 0.00] |
| base / 0 | frozen_feasible (n=8921) | obstacle_shift | on_smooth−on | +0.92 pp [0.73, 1.12] |
| base / 0 | all_stress (n=10000) | off | shift−original | +0.00 pp [0.00, 0.00] |
| base / 0 | all_stress (n=10000) | on | shift−original | +0.00 pp [0.00, 0.00] |
| base / 0 | all_stress (n=10000) | on_smooth | shift−original | -0.09 pp [-0.26, 0.08] |
| base / 0 | all_stress (n=10000) | 两场景 | (smooth shift效应)−(on shift效应) | -0.09 pp [-0.27, 0.08] |
| base / 0 | both_scenarios_frozen_feasible (n=8921) | off | shift−original | +0.00 pp [0.00, 0.00] |
| base / 0 | both_scenarios_frozen_feasible (n=8921) | on | shift−original | +0.00 pp [0.00, 0.00] |
| base / 0 | both_scenarios_frozen_feasible (n=8921) | on_smooth | shift−original | -0.10 pp [-0.29, 0.07] |
| base / 0 | both_scenarios_frozen_feasible (n=8921) | 两场景 | (smooth shift效应)−(on shift效应) | -0.10 pp [-0.28, 0.07] |
| geometry / 500 | all_stress (n=10000) | original | on−off | +0.00 ± 0.00 pp [0.00, 0.00] |
| geometry / 500 | all_stress (n=10000) | original | on_smooth−on | +17.81 ± 0.77 pp [17.19, 18.43] |
| geometry / 500 | frozen_feasible (n=10000) | original | on−off | +0.00 ± 0.00 pp [0.00, 0.00] |
| geometry / 500 | frozen_feasible (n=10000) | original | on_smooth−on | +17.81 ± 0.77 pp [17.17, 18.39] |
| geometry / 500 | all_stress (n=10000) | obstacle_shift | on−off | +0.00 ± 0.00 pp [0.00, 0.00] |
| geometry / 500 | all_stress (n=10000) | obstacle_shift | on_smooth−on | +18.62 ± 1.38 pp [18.02, 19.31] |
| geometry / 500 | frozen_feasible (n=8921) | obstacle_shift | on−off | +0.00 ± 0.00 pp [0.00, 0.00] |
| geometry / 500 | frozen_feasible (n=8921) | obstacle_shift | on_smooth−on | +19.06 ± 1.62 pp [18.34, 19.69] |
| geometry / 500 | all_stress (n=10000) | off | shift−original | +0.00 ± 0.00 pp [0.00, 0.00] |
| geometry / 500 | all_stress (n=10000) | on | shift−original | +0.00 ± 0.00 pp [0.00, 0.00] |
| geometry / 500 | all_stress (n=10000) | on_smooth | shift−original | +0.81 ± 0.97 pp [0.36, 1.29] |
| geometry / 500 | all_stress (n=10000) | 两场景 | (smooth shift效应)−(on shift效应) | +0.81 ± 0.97 pp [0.34, 1.22] |
| geometry / 500 | both_scenarios_frozen_feasible (n=8921) | off | shift−original | +0.00 ± 0.00 pp [0.00, 0.00] |
| geometry / 500 | both_scenarios_frozen_feasible (n=8921) | on | shift−original | +0.00 ± 0.00 pp [0.00, 0.00] |
| geometry / 500 | both_scenarios_frozen_feasible (n=8921) | on_smooth | shift−original | +2.57 ± 0.99 pp [2.09, 3.01] |
| geometry / 500 | both_scenarios_frozen_feasible (n=8921) | 两场景 | (smooth shift效应)−(on shift效应) | +2.57 ± 0.99 pp [2.08, 3.02] |
| geometry / 1000 | all_stress (n=10000) | original | on−off | +0.00 ± 0.00 pp [0.00, 0.00] |
| geometry / 1000 | all_stress (n=10000) | original | on_smooth−on | +19.38 ± 0.82 pp [18.72, 20.08] |
| geometry / 1000 | frozen_feasible (n=10000) | original | on−off | +0.00 ± 0.00 pp [0.00, 0.00] |
| geometry / 1000 | frozen_feasible (n=10000) | original | on_smooth−on | +19.38 ± 0.82 pp [18.73, 20.01] |
| geometry / 1000 | all_stress (n=10000) | obstacle_shift | on−off | +0.00 ± 0.00 pp [0.00, 0.00] |
| geometry / 1000 | all_stress (n=10000) | obstacle_shift | on_smooth−on | +20.53 ± 0.81 pp [19.83, 21.23] |
| geometry / 1000 | frozen_feasible (n=8921) | obstacle_shift | on−off | +0.00 ± 0.00 pp [0.00, 0.00] |
| geometry / 1000 | frozen_feasible (n=8921) | obstacle_shift | on_smooth−on | +20.91 ± 0.80 pp [20.17, 21.62] |
| geometry / 1000 | all_stress (n=10000) | off | shift−original | +0.00 ± 0.00 pp [0.00, 0.00] |
| geometry / 1000 | all_stress (n=10000) | on | shift−original | +0.00 ± 0.00 pp [0.00, 0.00] |
| geometry / 1000 | all_stress (n=10000) | on_smooth | shift−original | +1.15 ± 0.34 pp [0.70, 1.65] |
| geometry / 1000 | all_stress (n=10000) | 两场景 | (smooth shift效应)−(on shift效应) | +1.15 ± 0.34 pp [0.68, 1.61] |
| geometry / 1000 | both_scenarios_frozen_feasible (n=8921) | off | shift−original | +0.00 ± 0.00 pp [0.00, 0.00] |
| geometry / 1000 | both_scenarios_frozen_feasible (n=8921) | on | shift−original | +0.00 ± 0.00 pp [0.00, 0.00] |
| geometry / 1000 | both_scenarios_frozen_feasible (n=8921) | on_smooth | shift−original | +3.17 ± 0.30 pp [2.69, 3.65] |
| geometry / 1000 | both_scenarios_frozen_feasible (n=8921) | 两场景 | (smooth shift效应)−(on shift效应) | +3.17 ± 0.30 pp [2.65, 3.66] |
| geometry / 2000 | all_stress (n=10000) | original | on−off | +0.00 ± 0.00 pp [0.00, 0.00] |
| geometry / 2000 | all_stress (n=10000) | original | on_smooth−on | +19.63 ± 0.30 pp [18.99, 20.26] |
| geometry / 2000 | frozen_feasible (n=10000) | original | on−off | +0.00 ± 0.00 pp [0.00, 0.00] |
| geometry / 2000 | frozen_feasible (n=10000) | original | on_smooth−on | +19.63 ± 0.30 pp [18.98, 20.26] |
| geometry / 2000 | all_stress (n=10000) | obstacle_shift | on−off | +0.00 ± 0.00 pp [0.00, 0.00] |
| geometry / 2000 | all_stress (n=10000) | obstacle_shift | on_smooth−on | +21.30 ± 0.75 pp [20.60, 22.00] |
| geometry / 2000 | frozen_feasible (n=8921) | obstacle_shift | on−off | +0.00 ± 0.00 pp [0.00, 0.00] |
| geometry / 2000 | frozen_feasible (n=8921) | obstacle_shift | on_smooth−on | +21.75 ± 0.91 pp [21.00, 22.47] |
| geometry / 2000 | all_stress (n=10000) | off | shift−original | +0.00 ± 0.00 pp [0.00, 0.00] |
| geometry / 2000 | all_stress (n=10000) | on | shift−original | +0.00 ± 0.00 pp [0.00, 0.00] |
| geometry / 2000 | all_stress (n=10000) | on_smooth | shift−original | +1.67 ± 0.56 pp [1.21, 2.18] |
| geometry / 2000 | all_stress (n=10000) | 两场景 | (smooth shift效应)−(on shift效应) | +1.67 ± 0.56 pp [1.17, 2.16] |
| geometry / 2000 | both_scenarios_frozen_feasible (n=8921) | off | shift−original | +0.00 ± 0.00 pp [0.00, 0.00] |
| geometry / 2000 | both_scenarios_frozen_feasible (n=8921) | on | shift−original | +0.00 ± 0.00 pp [0.00, 0.00] |
| geometry / 2000 | both_scenarios_frozen_feasible (n=8921) | on_smooth | shift−original | +3.81 ± 0.72 pp [3.30, 4.32] |
| geometry / 2000 | both_scenarios_frozen_feasible (n=8921) | 两场景 | (smooth shift效应)−(on shift效应) | +3.81 ± 0.72 pp [3.31, 4.29] |
| geometry / 4000 | all_stress (n=10000) | original | on−off | +0.00 ± 0.00 pp [0.00, 0.00] |
| geometry / 4000 | all_stress (n=10000) | original | on_smooth−on | +20.12 ± 0.89 pp [19.49, 20.74] |
| geometry / 4000 | frozen_feasible (n=10000) | original | on−off | +0.00 ± 0.00 pp [0.00, 0.00] |
| geometry / 4000 | frozen_feasible (n=10000) | original | on_smooth−on | +20.12 ± 0.89 pp [19.45, 20.77] |
| geometry / 4000 | all_stress (n=10000) | obstacle_shift | on−off | +0.00 ± 0.00 pp [0.00, 0.00] |
| geometry / 4000 | all_stress (n=10000) | obstacle_shift | on_smooth−on | +21.27 ± 1.64 pp [20.61, 21.95] |
| geometry / 4000 | frozen_feasible (n=8921) | obstacle_shift | on−off | +0.00 ± 0.00 pp [0.00, 0.00] |
| geometry / 4000 | frozen_feasible (n=8921) | obstacle_shift | on_smooth−on | +21.59 ± 1.76 pp [20.85, 22.31] |
| geometry / 4000 | all_stress (n=10000) | off | shift−original | +0.00 ± 0.00 pp [0.00, 0.00] |
| geometry / 4000 | all_stress (n=10000) | on | shift−original | +0.00 ± 0.00 pp [0.00, 0.00] |
| geometry / 4000 | all_stress (n=10000) | on_smooth | shift−original | +1.15 ± 0.88 pp [0.71, 1.64] |
| geometry / 4000 | all_stress (n=10000) | 两场景 | (smooth shift效应)−(on shift效应) | +1.15 ± 0.88 pp [0.66, 1.62] |
| geometry / 4000 | both_scenarios_frozen_feasible (n=8921) | off | shift−original | +0.00 ± 0.00 pp [0.00, 0.00] |
| geometry / 4000 | both_scenarios_frozen_feasible (n=8921) | on | shift−original | +0.00 ± 0.00 pp [0.00, 0.00] |
| geometry / 4000 | both_scenarios_frozen_feasible (n=8921) | on_smooth | shift−original | +3.33 ± 1.00 pp [2.82, 3.81] |
| geometry / 4000 | both_scenarios_frozen_feasible (n=8921) | 两场景 | (smooth shift效应)−(on shift效应) | +3.33 ± 1.00 pp [2.81, 3.79] |
| geometry / 8000 | all_stress (n=10000) | original | on−off | +0.00 ± 0.00 pp [0.00, 0.00] |
| geometry / 8000 | all_stress (n=10000) | original | on_smooth−on | +19.31 ± 0.82 pp [18.66, 19.96] |
| geometry / 8000 | frozen_feasible (n=10000) | original | on−off | +0.00 ± 0.00 pp [0.00, 0.00] |
| geometry / 8000 | frozen_feasible (n=10000) | original | on_smooth−on | +19.31 ± 0.82 pp [18.63, 19.91] |
| geometry / 8000 | all_stress (n=10000) | obstacle_shift | on−off | +0.00 ± 0.00 pp [0.00, 0.00] |
| geometry / 8000 | all_stress (n=10000) | obstacle_shift | on_smooth−on | +20.87 ± 0.73 pp [20.19, 21.57] |
| geometry / 8000 | frozen_feasible (n=8921) | obstacle_shift | on−off | +0.00 ± 0.00 pp [0.00, 0.00] |
| geometry / 8000 | frozen_feasible (n=8921) | obstacle_shift | on_smooth−on | +21.26 ± 0.78 pp [20.45, 21.97] |
| geometry / 8000 | all_stress (n=10000) | off | shift−original | +0.00 ± 0.00 pp [0.00, 0.00] |
| geometry / 8000 | all_stress (n=10000) | on | shift−original | +0.00 ± 0.00 pp [0.00, 0.00] |
| geometry / 8000 | all_stress (n=10000) | on_smooth | shift−original | +1.57 ± 1.01 pp [1.08, 2.09] |
| geometry / 8000 | all_stress (n=10000) | 两场景 | (smooth shift效应)−(on shift效应) | +1.57 ± 1.01 pp [1.07, 2.03] |
| geometry / 8000 | both_scenarios_frozen_feasible (n=8921) | off | shift−original | +0.00 ± 0.00 pp [0.00, 0.00] |
| geometry / 8000 | both_scenarios_frozen_feasible (n=8921) | on | shift−original | +0.00 ± 0.00 pp [0.00, 0.00] |
| geometry / 8000 | both_scenarios_frozen_feasible (n=8921) | on_smooth | shift−original | +3.62 ± 1.11 pp [3.13, 4.11] |
| geometry / 8000 | both_scenarios_frozen_feasible (n=8921) | 两场景 | (smooth shift效应)−(on shift效应) | +3.62 ± 1.11 pp [3.09, 4.09] |
| no_contact_loss / 8000 | all_stress (n=10000) | original | on−off | +0.00 ± 0.00 pp [0.00, 0.00] |
| no_contact_loss / 8000 | all_stress (n=10000) | original | on_smooth−on | +18.35 ± 2.40 pp [17.69, 19.00] |
| no_contact_loss / 8000 | frozen_feasible (n=10000) | original | on−off | +0.00 ± 0.00 pp [0.00, 0.00] |
| no_contact_loss / 8000 | frozen_feasible (n=10000) | original | on_smooth−on | +18.35 ± 2.40 pp [17.65, 18.94] |
| no_contact_loss / 8000 | all_stress (n=10000) | obstacle_shift | on−off | +0.00 ± 0.00 pp [0.00, 0.00] |
| no_contact_loss / 8000 | all_stress (n=10000) | obstacle_shift | on_smooth−on | +20.00 ± 1.90 pp [19.35, 20.64] |
| no_contact_loss / 8000 | frozen_feasible (n=8921) | obstacle_shift | on−off | +0.00 ± 0.00 pp [0.00, 0.00] |
| no_contact_loss / 8000 | frozen_feasible (n=8921) | obstacle_shift | on_smooth−on | +20.68 ± 1.82 pp [19.94, 21.39] |
| no_contact_loss / 8000 | all_stress (n=10000) | off | shift−original | +0.00 ± 0.00 pp [0.00, 0.00] |
| no_contact_loss / 8000 | all_stress (n=10000) | on | shift−original | +0.00 ± 0.00 pp [0.00, 0.00] |
| no_contact_loss / 8000 | all_stress (n=10000) | on_smooth | shift−original | +1.65 ± 0.50 pp [1.19, 2.15] |
| no_contact_loss / 8000 | all_stress (n=10000) | 两场景 | (smooth shift效应)−(on shift效应) | +1.65 ± 0.50 pp [1.15, 2.12] |
| no_contact_loss / 8000 | both_scenarios_frozen_feasible (n=8921) | off | shift−original | +0.00 ± 0.00 pp [0.00, 0.00] |
| no_contact_loss / 8000 | both_scenarios_frozen_feasible (n=8921) | on | shift−original | +0.00 ± 0.00 pp [0.00, 0.00] |
| no_contact_loss / 8000 | both_scenarios_frozen_feasible (n=8921) | on_smooth | shift−original | +3.37 ± 0.22 pp [2.89, 3.87] |
| no_contact_loss / 8000 | both_scenarios_frozen_feasible (n=8921) | 两场景 | (smooth shift效应)−(on shift效应) | +3.37 ± 0.22 pp [2.87, 3.85] |
| no_geometry_tokens / 8000 | all_stress (n=10000) | original | on−off | +0.00 ± 0.00 pp [0.00, 0.00] |
| no_geometry_tokens / 8000 | all_stress (n=10000) | original | on_smooth−on | +8.79 ± 0.67 pp [8.29, 9.28] |
| no_geometry_tokens / 8000 | frozen_feasible (n=10000) | original | on−off | +0.00 ± 0.00 pp [0.00, 0.00] |
| no_geometry_tokens / 8000 | frozen_feasible (n=10000) | original | on_smooth−on | +8.79 ± 0.67 pp [8.32, 9.27] |
| no_geometry_tokens / 8000 | all_stress (n=10000) | obstacle_shift | on−off | +0.00 ± 0.00 pp [0.00, 0.00] |
| no_geometry_tokens / 8000 | all_stress (n=10000) | obstacle_shift | on_smooth−on | +9.68 ± 0.75 pp [9.18, 10.19] |
| no_geometry_tokens / 8000 | frozen_feasible (n=8921) | obstacle_shift | on−off | +0.00 ± 0.00 pp [0.00, 0.00] |
| no_geometry_tokens / 8000 | frozen_feasible (n=8921) | obstacle_shift | on_smooth−on | +10.20 ± 0.82 pp [9.67, 10.80] |
| no_geometry_tokens / 8000 | all_stress (n=10000) | off | shift−original | +0.00 ± 0.00 pp [0.00, 0.00] |
| no_geometry_tokens / 8000 | all_stress (n=10000) | on | shift−original | +0.00 ± 0.00 pp [0.00, 0.00] |
| no_geometry_tokens / 8000 | all_stress (n=10000) | on_smooth | shift−original | +0.89 ± 0.10 pp [0.57, 1.23] |
| no_geometry_tokens / 8000 | all_stress (n=10000) | 两场景 | (smooth shift效应)−(on shift效应) | +0.89 ± 0.10 pp [0.56, 1.20] |
| no_geometry_tokens / 8000 | both_scenarios_frozen_feasible (n=8921) | off | shift−original | +0.00 ± 0.00 pp [0.00, 0.00] |
| no_geometry_tokens / 8000 | both_scenarios_frozen_feasible (n=8921) | on | shift−original | +0.00 ± 0.00 pp [0.00, 0.00] |
| no_geometry_tokens / 8000 | both_scenarios_frozen_feasible (n=8921) | on_smooth | shift−original | +1.66 ± 0.15 pp [1.32, 2.00] |
| no_geometry_tokens / 8000 | both_scenarios_frozen_feasible (n=8921) | 两场景 | (smooth shift效应)−(on shift效应) | +1.66 ± 0.15 pp [1.31, 2.01] |

## K1 adapter−Base 与 8k 负控配对效果

| geometry 专家数 | 对照 | 分母 | 场景 | off Δ(pp) | on Δ(pp) | smooth Δ(pp) |
|---:|---|---|---|---:|---:|---:|
| 500 | base | all_stress | original | +0.00 ± 0.00 pp [0.00, 0.00] | +0.00 ± 0.00 pp [0.00, 0.00] | +16.90 ± 0.77 pp [16.20, 17.56] |
| 500 | base | all_stress | obstacle_shift | +0.00 ± 0.00 pp [0.00, 0.00] | +0.00 ± 0.00 pp [0.00, 0.00] | +17.80 ± 1.38 pp [17.14, 18.48] |
| 500 | base | frozen_feasible | original | +0.00 ± 0.00 pp [0.00, 0.00] | +0.00 ± 0.00 pp [0.00, 0.00] | +16.90 ± 0.77 pp [16.23, 17.54] |
| 500 | base | frozen_feasible | obstacle_shift | +0.00 ± 0.00 pp [0.00, 0.00] | +0.00 ± 0.00 pp [0.00, 0.00] | +18.14 ± 1.62 pp [17.44, 18.90] |
| 1000 | base | all_stress | original | +0.00 ± 0.00 pp [0.00, 0.00] | +0.00 ± 0.00 pp [0.00, 0.00] | +18.47 ± 0.82 pp [17.79, 19.19] |
| 1000 | base | all_stress | obstacle_shift | +0.00 ± 0.00 pp [0.00, 0.00] | +0.00 ± 0.00 pp [0.00, 0.00] | +19.71 ± 0.81 pp [18.94, 20.42] |
| 1000 | base | frozen_feasible | original | +0.00 ± 0.00 pp [0.00, 0.00] | +0.00 ± 0.00 pp [0.00, 0.00] | +18.47 ± 0.82 pp [17.82, 19.15] |
| 1000 | base | frozen_feasible | obstacle_shift | +0.00 ± 0.00 pp [0.00, 0.00] | +0.00 ± 0.00 pp [0.00, 0.00] | +19.99 ± 0.80 pp [19.26, 20.73] |
| 2000 | base | all_stress | original | +0.00 ± 0.00 pp [0.00, 0.00] | +0.00 ± 0.00 pp [0.00, 0.00] | +18.72 ± 0.30 pp [18.00, 19.43] |
| 2000 | base | all_stress | obstacle_shift | +0.00 ± 0.00 pp [0.00, 0.00] | +0.00 ± 0.00 pp [0.00, 0.00] | +20.48 ± 0.75 pp [19.73, 21.23] |
| 2000 | base | frozen_feasible | original | +0.00 ± 0.00 pp [0.00, 0.00] | +0.00 ± 0.00 pp [0.00, 0.00] | +18.72 ± 0.30 pp [18.11, 19.36] |
| 2000 | base | frozen_feasible | obstacle_shift | +0.00 ± 0.00 pp [0.00, 0.00] | +0.00 ± 0.00 pp [0.00, 0.00] | +20.83 ± 0.91 pp [20.07, 21.60] |
| 4000 | base | all_stress | original | +0.00 ± 0.00 pp [0.00, 0.00] | +0.00 ± 0.00 pp [0.00, 0.00] | +19.21 ± 0.89 pp [18.49, 19.88] |
| 4000 | base | all_stress | obstacle_shift | +0.00 ± 0.00 pp [0.00, 0.00] | +0.00 ± 0.00 pp [0.00, 0.00] | +20.45 ± 1.64 pp [19.78, 21.13] |
| 4000 | base | frozen_feasible | original | +0.00 ± 0.00 pp [0.00, 0.00] | +0.00 ± 0.00 pp [0.00, 0.00] | +19.21 ± 0.89 pp [18.57, 19.87] |
| 4000 | base | frozen_feasible | obstacle_shift | +0.00 ± 0.00 pp [0.00, 0.00] | +0.00 ± 0.00 pp [0.00, 0.00] | +20.67 ± 1.76 pp [19.88, 21.42] |
| 8000 | base | all_stress | original | +0.00 ± 0.00 pp [0.00, 0.00] | +0.00 ± 0.00 pp [0.00, 0.00] | +18.40 ± 0.82 pp [17.69, 19.10] |
| 8000 | base | all_stress | obstacle_shift | +0.00 ± 0.00 pp [0.00, 0.00] | +0.00 ± 0.00 pp [0.00, 0.00] | +20.05 ± 0.73 pp [19.34, 20.76] |
| 8000 | base | frozen_feasible | original | +0.00 ± 0.00 pp [0.00, 0.00] | +0.00 ± 0.00 pp [0.00, 0.00] | +18.40 ± 0.82 pp [17.79, 19.02] |
| 8000 | base | frozen_feasible | obstacle_shift | +0.00 ± 0.00 pp [0.00, 0.00] | +0.00 ± 0.00 pp [0.00, 0.00] | +20.34 ± 0.78 pp [19.60, 21.11] |
| 8000 | no_contact_loss | all_stress | original | +0.00 ± 0.00 pp [0.00, 0.00] | +0.00 ± 0.00 pp [0.00, 0.00] | +0.96 ± 2.29 pp [0.54, 1.38] |
| 8000 | no_contact_loss | all_stress | obstacle_shift | +0.00 ± 0.00 pp [0.00, 0.00] | +0.00 ± 0.00 pp [0.00, 0.00] | +0.87 ± 2.58 pp [0.46, 1.28] |
| 8000 | no_contact_loss | frozen_feasible | original | +0.00 ± 0.00 pp [0.00, 0.00] | +0.00 ± 0.00 pp [0.00, 0.00] | +0.96 ± 2.29 pp [0.54, 1.37] |
| 8000 | no_contact_loss | frozen_feasible | obstacle_shift | +0.00 ± 0.00 pp [0.00, 0.00] | +0.00 ± 0.00 pp [0.00, 0.00] | +0.58 ± 2.56 pp [0.17, 1.01] |
| 8000 | no_geometry_tokens | all_stress | original | +0.00 ± 0.00 pp [0.00, 0.00] | +0.00 ± 0.00 pp [0.00, 0.00] | +10.52 ± 1.37 pp [9.97, 11.08] |
| 8000 | no_geometry_tokens | all_stress | obstacle_shift | +0.00 ± 0.00 pp [0.00, 0.00] | +0.00 ± 0.00 pp [0.00, 0.00] | +11.20 ± 1.43 pp [10.62, 11.78] |
| 8000 | no_geometry_tokens | frozen_feasible | original | +0.00 ± 0.00 pp [0.00, 0.00] | +0.00 ± 0.00 pp [0.00, 0.00] | +10.52 ± 1.37 pp [10.02, 11.05] |
| 8000 | no_geometry_tokens | frozen_feasible | obstacle_shift | +0.00 ± 0.00 pp [0.00, 0.00] | +0.00 ± 0.00 pp [0.00, 0.00] | +11.06 ± 1.53 pp [10.44, 11.62] |

逐 seed 配对结果、所有质量指标的差值及分层 CI 同样完整导出，不能仅比较未配对点估计。

## 成功 gate 与关键失败分解

完整成功同时要求完整 URDF 物理碰撞通过、接触误差≤1 mm、法向误差≤1°、有效姿态/工作空间/两端/接触图域/twist、shape-quality 通过，并对成功候选做密集边审核。shape gate 含最大曲率≤8/m、累计回退≤0.25 m、chord16 ratio≥0.65。numeric success 另要求≤2e-6 m/rad，不能替代物理成功。单独 gate 的通过率不代表完整候选成功，各失败原因也可以同时发生。

| 变体 / 专家数 | 场景 / decoder | URDF节点碰撞自由 | contact≤1mm | normal≤1° | shape gate | dense_checked |
|---|---|---:|---:|---:|---:|---:|
| base / 0 | original / off | 7.70% [7.19, 8.19] | 0.00% [0.00, 0.00] | 0.00% [0.00, 0.00] | 0.00% [0.00, 0.00] | 0.00% [0.00, 0.00] |
| base / 0 | original / on | 32.96% [32.22, 33.71] | 100.00% [100.00, 100.00] | 100.00% [100.00, 100.00] | 0.00% [0.00, 0.00] | 0.00% [0.00, 0.00] |
| base / 0 | original / on_smooth | 34.45% [33.69, 35.20] | 100.00% [100.00, 100.00] | 100.00% [100.00, 100.00] | 3.01% [2.69, 3.34] | 0.91% [0.73, 1.11] |
| base / 0 | obstacle_shift / off | 6.42% [5.95, 6.91] | 0.00% [0.00, 0.00] | 0.00% [0.00, 0.00] | 0.00% [0.00, 0.00] | 0.00% [0.00, 0.00] |
| base / 0 | obstacle_shift / on | 27.29% [26.47, 28.15] | 100.00% [100.00, 100.00] | 100.00% [100.00, 100.00] | 0.00% [0.00, 0.00] | 0.00% [0.00, 0.00] |
| base / 0 | obstacle_shift / on_smooth | 28.62% [27.73, 29.44] | 100.00% [100.00, 100.00] | 100.00% [100.00, 100.00] | 2.92% [2.60, 3.21] | 0.82% [0.65, 1.01] |
| geometry / 500 | original / off | 46.80 ± 0.78% [46.35, 47.23] | 0.00 ± 0.00% [0.00, 0.00] | 1.00 ± 0.28% [0.87, 1.14] | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] |
| geometry / 500 | original / on | 46.72 ± 0.58% [46.27, 47.13] | 100.00 ± 0.00% [100.00, 100.00] | 100.00 ± 0.00% [100.00, 100.00] | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] |
| geometry / 500 | original / on_smooth | 47.75 ± 0.66% [47.28, 48.24] | 100.00 ± 0.00% [100.00, 100.00] | 100.00 ± 0.00% [100.00, 100.00] | 56.81 ± 3.60% [56.08, 57.47] | 17.82 ± 0.78% [17.18, 18.43] |
| geometry / 500 | obstacle_shift / off | 43.86 ± 1.50% [43.25, 44.48] | 0.00 ± 0.00% [0.00, 0.00] | 1.00 ± 0.27% [0.86, 1.13] | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] |
| geometry / 500 | obstacle_shift / on | 44.12 ± 1.28% [43.44, 44.75] | 100.00 ± 0.00% [100.00, 100.00] | 100.00 ± 0.00% [100.00, 100.00] | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] |
| geometry / 500 | obstacle_shift / on_smooth | 45.01 ± 1.46% [44.33, 45.69] | 100.00 ± 0.00% [100.00, 100.00] | 100.00 ± 0.00% [100.00, 100.00] | 56.94 ± 3.57% [56.28, 57.55] | 18.62 ± 1.38% [18.00, 19.28] |
| geometry / 1000 | original / off | 47.07 ± 0.53% [46.59, 47.54] | 0.00 ± 0.00% [0.00, 0.00] | 1.15 ± 0.12% [1.02, 1.31] | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] |
| geometry / 1000 | original / on | 47.02 ± 0.26% [46.58, 47.46] | 100.00 ± 0.00% [100.00, 100.00] | 100.00 ± 0.00% [100.00, 100.00] | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] |
| geometry / 1000 | original / on_smooth | 48.04 ± 0.24% [47.55, 48.52] | 100.00 ± 0.00% [100.00, 100.00] | 100.00 ± 0.00% [100.00, 100.00] | 59.65 ± 0.66% [58.96, 60.34] | 19.40 ± 0.81% [18.71, 20.09] |
| geometry / 1000 | obstacle_shift / off | 44.79 ± 0.59% [44.17, 45.37] | 0.00 ± 0.00% [0.00, 0.00] | 1.19 ± 0.13% [1.05, 1.34] | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] |
| geometry / 1000 | obstacle_shift / on | 44.97 ± 0.49% [44.28, 45.60] | 100.00 ± 0.00% [100.00, 100.00] | 100.00 ± 0.00% [100.00, 100.00] | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] |
| geometry / 1000 | obstacle_shift / on_smooth | 45.84 ± 0.47% [45.16, 46.54] | 100.00 ± 0.00% [100.00, 100.00] | 100.00 ± 0.00% [100.00, 100.00] | 59.84 ± 0.62% [59.20, 60.49] | 20.54 ± 0.81% [19.82, 21.28] |
| geometry / 2000 | original / off | 46.76 ± 0.33% [46.29, 47.18] | 0.00 ± 0.00% [0.00, 0.00] | 1.25 ± 0.13% [1.09, 1.41] | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] |
| geometry / 2000 | original / on | 46.82 ± 0.35% [46.40, 47.26] | 100.00 ± 0.00% [100.00, 100.00] | 100.00 ± 0.00% [100.00, 100.00] | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] |
| geometry / 2000 | original / on_smooth | 47.70 ± 0.22% [47.24, 48.18] | 100.00 ± 0.00% [100.00, 100.00] | 100.00 ± 0.00% [100.00, 100.00] | 61.09 ± 1.42% [60.37, 61.79] | 19.65 ± 0.31% [18.98, 20.34] |
| geometry / 2000 | obstacle_shift / off | 44.78 ± 0.55% [44.16, 45.40] | 0.00 ± 0.00% [0.00, 0.00] | 1.25 ± 0.12% [1.10, 1.42] | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] |
| geometry / 2000 | obstacle_shift / on | 45.27 ± 0.35% [44.55, 45.93] | 100.00 ± 0.00% [100.00, 100.00] | 100.00 ± 0.00% [100.00, 100.00] | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] |
| geometry / 2000 | obstacle_shift / on_smooth | 46.12 ± 0.54% [45.42, 46.82] | 100.00 ± 0.00% [100.00, 100.00] | 100.00 ± 0.00% [100.00, 100.00] | 61.21 ± 1.38% [60.53, 61.87] | 21.31 ± 0.75% [20.59, 22.02] |
| geometry / 4000 | original / off | 46.85 ± 0.03% [46.39, 47.31] | 0.00 ± 0.00% [0.00, 0.00] | 1.33 ± 0.27% [1.18, 1.49] | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] |
| geometry / 4000 | original / on | 46.78 ± 0.10% [46.36, 47.18] | 100.00 ± 0.00% [100.00, 100.00] | 100.00 ± 0.00% [100.00, 100.00] | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] |
| geometry / 4000 | original / on_smooth | 47.77 ± 0.21% [47.30, 48.23] | 100.00 ± 0.00% [100.00, 100.00] | 100.00 ± 0.00% [100.00, 100.00] | 61.13 ± 2.55% [60.44, 61.82] | 20.13 ± 0.89% [19.48, 20.80] |
| geometry / 4000 | obstacle_shift / off | 44.85 ± 0.82% [44.22, 45.50] | 0.00 ± 0.00% [0.00, 0.00] | 1.31 ± 0.27% [1.16, 1.48] | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] |
| geometry / 4000 | obstacle_shift / on | 44.86 ± 0.84% [44.15, 45.50] | 100.00 ± 0.00% [100.00, 100.00] | 100.00 ± 0.00% [100.00, 100.00] | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] |
| geometry / 4000 | obstacle_shift / on_smooth | 45.72 ± 0.83% [45.02, 46.41] | 100.00 ± 0.00% [100.00, 100.00] | 100.00 ± 0.00% [100.00, 100.00] | 61.31 ± 2.76% [60.66, 61.96] | 21.28 ± 1.64% [20.62, 21.95] |
| geometry / 8000 | original / off | 46.63 ± 0.43% [46.16, 47.05] | 0.00 ± 0.00% [0.00, 0.00] | 1.28 ± 0.06% [1.12, 1.44] | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] |
| geometry / 8000 | original / on | 46.59 ± 0.47% [46.17, 46.96] | 100.00 ± 0.00% [100.00, 100.00] | 100.00 ± 0.00% [100.00, 100.00] | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] |
| geometry / 8000 | original / on_smooth | 47.53 ± 0.43% [47.05, 47.97] | 100.00 ± 0.00% [100.00, 100.00] | 100.00 ± 0.00% [100.00, 100.00] | 60.85 ± 2.04% [60.11, 61.50] | 19.31 ± 0.82% [18.64, 19.99] |
| geometry / 8000 | obstacle_shift / off | 44.93 ± 0.55% [44.32, 45.57] | 0.00 ± 0.00% [0.00, 0.00] | 1.29 ± 0.01% [1.12, 1.46] | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] |
| geometry / 8000 | obstacle_shift / on | 45.06 ± 0.49% [44.34, 45.71] | 100.00 ± 0.00% [100.00, 100.00] | 100.00 ± 0.00% [100.00, 100.00] | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] |
| geometry / 8000 | obstacle_shift / on_smooth | 45.89 ± 0.57% [45.18, 46.58] | 100.00 ± 0.00% [100.00, 100.00] | 100.00 ± 0.00% [100.00, 100.00] | 60.88 ± 2.21% [60.22, 61.59] | 20.87 ± 0.73% [20.16, 21.59] |
| no_contact_loss / 8000 | original / off | 48.27 ± 0.50% [47.73, 48.74] | 0.00 ± 0.00% [0.00, 0.00] | 1.30 ± 0.13% [1.15, 1.45] | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] |
| no_contact_loss / 8000 | original / on | 48.33 ± 0.68% [47.86, 48.81] | 100.00 ± 0.00% [100.00, 100.00] | 100.00 ± 0.00% [100.00, 100.00] | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] |
| no_contact_loss / 8000 | original / on_smooth | 49.34 ± 0.68% [48.81, 49.81] | 100.00 ± 0.00% [100.00, 100.00] | 100.00 ± 0.00% [100.00, 100.00] | 55.36 ± 1.60% [54.68, 56.03] | 18.36 ± 2.40% [17.67, 19.00] |
| no_contact_loss / 8000 | obstacle_shift / off | 45.75 ± 0.49% [45.15, 46.40] | 0.00 ± 0.00% [0.00, 0.00] | 1.32 ± 0.20% [1.16, 1.48] | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] |
| no_contact_loss / 8000 | obstacle_shift / on | 46.41 ± 0.45% [45.66, 47.07] | 100.00 ± 0.00% [100.00, 100.00] | 100.00 ± 0.00% [100.00, 100.00] | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] |
| no_contact_loss / 8000 | obstacle_shift / on_smooth | 47.42 ± 0.46% [46.68, 48.14] | 100.00 ± 0.00% [100.00, 100.00] | 100.00 ± 0.00% [100.00, 100.00] | 55.51 ± 1.68% [54.81, 56.19] | 20.00 ± 1.90% [19.34, 20.69] |
| no_geometry_tokens / 8000 | original / off | 43.84 ± 0.26% [43.40, 44.26] | 0.00 ± 0.00% [0.00, 0.00] | 0.12 ± 0.05% [0.08, 0.18] | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] |
| no_geometry_tokens / 8000 | original / on | 44.97 ± 0.48% [44.51, 45.42] | 100.00 ± 0.00% [100.00, 100.00] | 100.00 ± 0.00% [100.00, 100.00] | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] |
| no_geometry_tokens / 8000 | original / on_smooth | 46.37 ± 0.39% [45.94, 46.82] | 100.00 ± 0.00% [100.00, 100.00] | 100.00 ± 0.00% [100.00, 100.00] | 37.40 ± 0.71% [36.71, 38.13] | 8.80 ± 0.67% [8.33, 9.24] |
| no_geometry_tokens / 8000 | obstacle_shift / off | 39.54 ± 0.47% [38.95, 40.11] | 0.00 ± 0.00% [0.00, 0.00] | 0.14 ± 0.05% [0.09, 0.20] | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] |
| no_geometry_tokens / 8000 | obstacle_shift / on | 41.43 ± 0.71% [40.79, 42.06] | 100.00 ± 0.00% [100.00, 100.00] | 100.00 ± 0.00% [100.00, 100.00] | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] |
| no_geometry_tokens / 8000 | obstacle_shift / on_smooth | 42.86 ± 0.56% [42.16, 43.51] | 100.00 ± 0.00% [100.00, 100.00] | 100.00 ± 0.00% [100.00, 100.00] | 37.43 ± 0.80% [36.77, 38.08] | 9.68 ± 0.75% [9.15, 10.21] |

dense_checked 是已进行密集检查的比例，评分流程仅对满足前置条件的候选进入密集审核。节点 clearance 是 128 节点处扣除 margin 后的最小值，不能解释成密集边最小 clearance。contact/normal/curvature、回退、chord16、clearance、path efficiency、图域/twist、拟合收敛与修正量以及失败类型均见完整附录和输入报告。

## 8k geometry 的曲面、难度与场景族分层

以下为三个种子聚合的 K1 成功率；全部预算、Base、负控、逐 seed、K8 和 family×plane/cylinder×easy/detour 交叉表完整保存于附录。

| 分层 | 场景 | 分母 | 条件数 | off | on | smooth |
|---|---|---|---:|---:|---:|---:|
| family=central_block | original | all_stress | 1000 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 22.03 ± 1.55% [20.10, 24.03] |
| family=central_block | original | frozen_feasible | 1000 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 22.03 ± 1.55% [20.07, 24.03] |
| family=central_block | obstacle_shift | all_stress | 1000 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 24.47 ± 0.31% [22.23, 26.57] |
| family=central_block | obstacle_shift | frozen_feasible | 893 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 25.12 ± 0.17% [22.70, 27.48] |
| family=frame_doorway | original | all_stress | 1000 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 23.53 ± 1.14% [21.33, 25.90] |
| family=frame_doorway | original | frozen_feasible | 1000 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 23.53 ± 1.14% [21.40, 25.60] |
| family=frame_doorway | obstacle_shift | all_stress | 1000 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 24.73 ± 1.35% [22.33, 27.20] |
| family=frame_doorway | obstacle_shift | frozen_feasible | 902 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 24.61 ± 1.44% [22.05, 27.22] |
| family=mixed_industrial | original | all_stress | 1000 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 14.37 ± 0.06% [12.70, 16.17] |
| family=mixed_industrial | original | frozen_feasible | 1000 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 14.37 ± 0.06% [12.63, 16.10] |
| family=mixed_industrial | obstacle_shift | all_stress | 1000 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 17.03 ± 0.74% [14.80, 19.10] |
| family=mixed_industrial | obstacle_shift | frozen_feasible | 882 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 17.54 ± 0.66% [15.21, 19.64] |
| family=multi_homotopy | original | all_stress | 1000 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 23.60 ± 1.56% [21.27, 26.10] |
| family=multi_homotopy | original | frozen_feasible | 1000 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 23.60 ± 1.56% [21.37, 25.83] |
| family=multi_homotopy | obstacle_shift | all_stress | 1000 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 26.13 ± 1.44% [23.70, 28.67] |
| family=multi_homotopy | obstacle_shift | frozen_feasible | 898 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 26.24 ± 1.33% [23.64, 28.93] |
| family=narrow_passage | original | all_stress | 1000 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 18.07 ± 0.71% [15.90, 20.03] |
| family=narrow_passage | original | frozen_feasible | 1000 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 18.07 ± 0.71% [16.00, 20.10] |
| family=narrow_passage | obstacle_shift | all_stress | 1000 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 19.80 ± 0.70% [17.57, 22.00] |
| family=narrow_passage | obstacle_shift | frozen_feasible | 892 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 20.63 ± 0.67% [18.45, 23.08] |
| family=orientation_sensitive_passage | original | all_stress | 1000 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 13.70 ± 1.81% [11.80, 15.60] |
| family=orientation_sensitive_passage | original | frozen_feasible | 1000 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 13.70 ± 1.81% [12.03, 15.43] |
| family=orientation_sensitive_passage | obstacle_shift | all_stress | 1000 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 15.97 ± 2.86% [14.07, 17.90] |
| family=orientation_sensitive_passage | obstacle_shift | frozen_feasible | 907 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 16.46 ± 2.84% [14.47, 18.76] |
| family=pillar_wall | original | all_stress | 1000 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 23.03 ± 1.61% [20.83, 25.33] |
| family=pillar_wall | original | frozen_feasible | 1000 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 23.03 ± 1.61% [20.87, 25.43] |
| family=pillar_wall | obstacle_shift | all_stress | 1000 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 23.80 ± 0.60% [21.40, 26.53] |
| family=pillar_wall | obstacle_shift | frozen_feasible | 900 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 24.37 ± 0.39% [21.72, 26.92] |
| family=sparse_obb_clutter | original | all_stress | 1000 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 20.63 ± 0.93% [18.50, 22.83] |
| family=sparse_obb_clutter | original | frozen_feasible | 1000 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 20.63 ± 0.93% [18.40, 22.87] |
| family=sparse_obb_clutter | obstacle_shift | all_stress | 1000 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 21.17 ± 0.51% [19.07, 23.43] |
| family=sparse_obb_clutter | obstacle_shift | frozen_feasible | 886 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 21.26 ± 0.91% [18.80, 23.82] |
| family=staggered_corridor | original | all_stress | 1000 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 21.87 ± 1.67% [19.73, 24.23] |
| family=staggered_corridor | original | frozen_feasible | 1000 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 21.87 ± 1.67% [19.63, 24.17] |
| family=staggered_corridor | obstacle_shift | all_stress | 1000 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 22.73 ± 0.75% [20.50, 25.00] |
| family=staggered_corridor | obstacle_shift | frozen_feasible | 864 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 23.61 ± 0.64% [20.94, 25.93] |
| family=wall_protrusion_bracket | original | all_stress | 1000 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 12.23 ± 1.92% [10.77, 13.87] |
| family=wall_protrusion_bracket | original | frozen_feasible | 1000 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 12.23 ± 1.92% [10.70, 13.83] |
| family=wall_protrusion_bracket | obstacle_shift | all_stress | 1000 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 12.90 ± 1.21% [11.23, 14.63] |
| family=wall_protrusion_bracket | obstacle_shift | frozen_feasible | 897 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 12.82 ± 1.36% [11.07, 14.61] |
| kind=cylinder | original | all_stress | 5000 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 16.99 ± 1.29% [16.14, 17.79] |
| kind=cylinder | original | frozen_feasible | 5000 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 16.99 ± 1.29% [16.17, 17.81] |
| kind=cylinder | obstacle_shift | all_stress | 5000 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 19.30 ± 0.74% [18.40, 20.29] |
| kind=cylinder | obstacle_shift | frozen_feasible | 4513 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 20.01 ± 0.73% [19.02, 21.02] |
| kind=plane | original | all_stress | 5000 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 21.63 ± 0.64% [20.66, 22.61] |
| kind=plane | original | frozen_feasible | 5000 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 21.63 ± 0.64% [20.71, 22.55] |
| kind=plane | obstacle_shift | all_stress | 5000 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 22.45 ± 1.32% [21.49, 23.42] |
| kind=plane | obstacle_shift | frozen_feasible | 4408 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 22.54 ± 1.43% [21.47, 23.58] |
| difficulty=detour | original | all_stress | 6000 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 11.00 ± 0.53% [10.33, 11.68] |
| difficulty=detour | original | frozen_feasible | 6000 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 11.00 ± 0.53% [10.33, 11.73] |
| difficulty=detour | obstacle_shift | all_stress | 6000 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 16.43 ± 1.07% [15.56, 17.38] |
| difficulty=detour | obstacle_shift | frozen_feasible | 5983 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 16.48 ± 1.07% [15.67, 17.40] |
| difficulty=easy | original | all_stress | 4000 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 31.77 ± 2.08% [30.50, 32.98] |
| difficulty=easy | original | frozen_feasible | 4000 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 31.77 ± 2.08% [30.57, 32.99] |
| difficulty=easy | obstacle_shift | all_stress | 4000 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 27.53 ± 2.04% [26.36, 28.68] |
| difficulty=easy | obstacle_shift | frozen_feasible | 2938 | 0.00 ± 0.00% [0.00, 0.00] | 0.00 ± 0.00% [0.00, 0.00] | 31.00 ± 2.45% [29.67, 32.44] |

## K8 成功候选对的几何多样性

只使用实际保存的成功 body XYZ / intrinsic UV，按相同节点进度计算每对候选的 MSD，再先对单条件候选对平均、后对条件平均。少于两个成功候选的条件为 null，不能填零；Any@8 仍包含这些条件。方向符号是相对端点弦的经验几何代理，不是同伦类证明。

| 变体 / 专家数 | 场景 / decoder | 可计算pair的条件数 | XYZ MSD(m²) | XYZ RMS(m) | UV MSD(m²) |
|---|---|---:|---:|---:|---:|
| base / 0 | original / off | 0 | 缺诊断（n=0） | 缺诊断（n=0） | 缺诊断（n=0） |
| base / 0 | original / on | 0 | 缺诊断（n=0） | 缺诊断（n=0） | 缺诊断（n=0） |
| base / 0 | original / on_smooth | 16 | 0.01 [0.01, 0.02] | 0.11 [0.09, 0.13] | 0.01 [0.01, 0.02] |
| base / 0 | obstacle_shift / off | 0 | 缺诊断（n=0） | 缺诊断（n=0） | 缺诊断（n=0） |
| base / 0 | obstacle_shift / on | 0 | 缺诊断（n=0） | 缺诊断（n=0） | 缺诊断（n=0） |
| base / 0 | obstacle_shift / on_smooth | 13 | 0.03 [0.02, 0.04] | 0.15 [0.12, 0.18] | 0.03 [0.02, 0.04] |
| geometry / 500 | original / off | 0 | 缺诊断（n=0） | 缺诊断（n=0） | 缺诊断（n=0） |
| geometry / 500 | original / on | 0 | 缺诊断（n=0） | 缺诊断（n=0） | 缺诊断（n=0） |
| geometry / 500 | original / on_smooth | 430 | 0.51 ± 0.01 [0.44, 0.58] | 0.40 ± 0.01 [0.36, 0.44] | 0.50 ± 0.01 [0.44, 0.57] |
| geometry / 500 | obstacle_shift / off | 0 | 缺诊断（n=0） | 缺诊断（n=0） | 缺诊断（n=0） |
| geometry / 500 | obstacle_shift / on | 0 | 缺诊断（n=0） | 缺诊断（n=0） | 缺诊断（n=0） |
| geometry / 500 | obstacle_shift / on_smooth | 479 | 0.31 ± 0.04 [0.25, 0.38] | 0.30 ± 0.02 [0.27, 0.34] | 0.30 ± 0.04 [0.24, 0.37] |
| geometry / 1000 | original / off | 0 | 缺诊断（n=0） | 缺诊断（n=0） | 缺诊断（n=0） |
| geometry / 1000 | original / on | 0 | 缺诊断（n=0） | 缺诊断（n=0） | 缺诊断（n=0） |
| geometry / 1000 | original / on_smooth | 440 | 0.48 ± 0.04 [0.40, 0.55] | 0.38 ± 0.03 [0.34, 0.42] | 0.47 ± 0.04 [0.40, 0.55] |
| geometry / 1000 | obstacle_shift / off | 0 | 缺诊断（n=0） | 缺诊断（n=0） | 缺诊断（n=0） |
| geometry / 1000 | obstacle_shift / on | 0 | 缺诊断（n=0） | 缺诊断（n=0） | 缺诊断（n=0） |
| geometry / 1000 | obstacle_shift / on_smooth | 498 | 0.31 ± 0.02 [0.25, 0.37] | 0.30 ± 0.01 [0.26, 0.34] | 0.30 ± 0.02 [0.24, 0.36] |
| geometry / 2000 | original / off | 0 | 缺诊断（n=0） | 缺诊断（n=0） | 缺诊断（n=0） |
| geometry / 2000 | original / on | 0 | 缺诊断（n=0） | 缺诊断（n=0） | 缺诊断（n=0） |
| geometry / 2000 | original / on_smooth | 450 | 0.41 ± 0.01 [0.35, 0.47] | 0.34 ± 0.01 [0.30, 0.37] | 0.40 ± 0.01 [0.34, 0.46] |
| geometry / 2000 | obstacle_shift / off | 0 | 缺诊断（n=0） | 缺诊断（n=0） | 缺诊断（n=0） |
| geometry / 2000 | obstacle_shift / on | 0 | 缺诊断（n=0） | 缺诊断（n=0） | 缺诊断（n=0） |
| geometry / 2000 | obstacle_shift / on_smooth | 510 | 0.25 ± 0.01 [0.20, 0.30] | 0.26 ± 0.00 [0.23, 0.29] | 0.25 ± 0.01 [0.19, 0.30] |
| geometry / 4000 | original / off | 0 | 缺诊断（n=0） | 缺诊断（n=0） | 缺诊断（n=0） |
| geometry / 4000 | original / on | 0 | 缺诊断（n=0） | 缺诊断（n=0） | 缺诊断（n=0） |
| geometry / 4000 | original / on_smooth | 459 | 0.43 ± 0.06 [0.37, 0.50] | 0.35 ± 0.03 [0.31, 0.40] | 0.43 ± 0.06 [0.36, 0.50] |
| geometry / 4000 | obstacle_shift / off | 0 | 缺诊断（n=0） | 缺诊断（n=0） | 缺诊断（n=0） |
| geometry / 4000 | obstacle_shift / on | 0 | 缺诊断（n=0） | 缺诊断（n=0） | 缺诊断（n=0） |
| geometry / 4000 | obstacle_shift / on_smooth | 523 | 0.30 ± 0.01 [0.24, 0.35] | 0.29 ± 0.00 [0.26, 0.32] | 0.29 ± 0.00 [0.23, 0.34] |
| geometry / 8000 | original / off | 0 | 缺诊断（n=0） | 缺诊断（n=0） | 缺诊断（n=0） |
| geometry / 8000 | original / on | 0 | 缺诊断（n=0） | 缺诊断（n=0） | 缺诊断（n=0） |
| geometry / 8000 | original / on_smooth | 445 | 0.41 ± 0.02 [0.34, 0.49] | 0.34 ± 0.01 [0.30, 0.39] | 0.41 ± 0.02 [0.33, 0.48] |
| geometry / 8000 | obstacle_shift / off | 0 | 缺诊断（n=0） | 缺诊断（n=0） | 缺诊断（n=0） |
| geometry / 8000 | obstacle_shift / on | 0 | 缺诊断（n=0） | 缺诊断（n=0） | 缺诊断（n=0） |
| geometry / 8000 | obstacle_shift / on_smooth | 519 | 0.29 ± 0.02 [0.24, 0.33] | 0.28 ± 0.01 [0.26, 0.31] | 0.28 ± 0.02 [0.23, 0.33] |
| no_contact_loss / 8000 | original / off | 0 | 缺诊断（n=0） | 缺诊断（n=0） | 缺诊断（n=0） |
| no_contact_loss / 8000 | original / on | 0 | 缺诊断（n=0） | 缺诊断（n=0） | 缺诊断（n=0） |
| no_contact_loss / 8000 | original / on_smooth | 468 | 0.55 ± 0.02 [0.47, 0.63] | 0.43 ± 0.02 [0.38, 0.47] | 0.54 ± 0.02 [0.46, 0.62] |
| no_contact_loss / 8000 | obstacle_shift / off | 0 | 缺诊断（n=0） | 缺诊断（n=0） | 缺诊断（n=0） |
| no_contact_loss / 8000 | obstacle_shift / on | 0 | 缺诊断（n=0） | 缺诊断（n=0） | 缺诊断（n=0） |
| no_contact_loss / 8000 | obstacle_shift / on_smooth | 532 | 0.34 ± 0.03 [0.28, 0.39] | 0.32 ± 0.01 [0.29, 0.36] | 0.33 ± 0.03 [0.27, 0.38] |
| no_geometry_tokens / 8000 | original / off | 0 | 缺诊断（n=0） | 缺诊断（n=0） | 缺诊断（n=0） |
| no_geometry_tokens / 8000 | original / on | 0 | 缺诊断（n=0） | 缺诊断（n=0） | 缺诊断（n=0） |
| no_geometry_tokens / 8000 | original / on_smooth | 220 | 0.63 ± 0.05 [0.51, 0.74] | 0.48 ± 0.03 [0.41, 0.54] | 0.62 ± 0.05 [0.49, 0.73] |
| no_geometry_tokens / 8000 | obstacle_shift / off | 0 | 缺诊断（n=0） | 缺诊断（n=0） | 缺诊断（n=0） |
| no_geometry_tokens / 8000 | obstacle_shift / on | 0 | 缺诊断（n=0） | 缺诊断（n=0） | 缺诊断（n=0） |
| no_geometry_tokens / 8000 | obstacle_shift / on_smooth | 238 | 0.47 ± 0.04 [0.34, 0.60] | 0.40 ± 0.03 [0.33, 0.47] | 0.45 ± 0.04 [0.33, 0.58] |

## 结果解读（仅依据冻结统计）

以下事实均以全部压力条件为分母；冻结物理见证子集见正文和完整附录。区间的正负只描述这些源地图上的不确定性，不是未见训练 seed 的泛化证明，也未做多重比较校正。

8k geometry / original：按 off/on/smooth，K1 为 0.00% / 0.00% / 19.31%，冻结 Base 为 0.00% / 0.00% / 0.91%；同条件 adapter−Base 配对 Δ 为 +0.00 ± 0.00 pp [0.00, 0.00] / +0.00 ± 0.00 pp [0.00, 0.00] / +18.40 ± 0.82 pp [17.69, 19.10]。各 decoder 的区间解释依次为：地图区间包含0，不能据此断言差值方向确定；地图区间包含0，不能据此断言差值方向确定；地图区间完全高于0。
no_contact_loss / original：off/on/smooth 的实际 K1 为 0.00% / 0.00% / 18.35%；8k geometry−该控制的 matched-seed Δ 为 +0.00 ± 0.00 pp [0.00, 0.00] / +0.00 ± 0.00 pp [0.00, 0.00] / +0.96 ± 2.29 pp [0.54, 1.38]。地图区间包含0，不能据此断言差值方向确定；地图区间包含0，不能据此断言差值方向确定；地图区间完全高于0。
no_geometry_tokens / original：off/on/smooth 的实际 K1 为 0.00% / 0.00% / 8.79%；8k geometry−该控制的 matched-seed Δ 为 +0.00 ± 0.00 pp [0.00, 0.00] / +0.00 ± 0.00 pp [0.00, 0.00] / +10.52 ± 1.37 pp [9.97, 11.08]。地图区间包含0，不能据此断言差值方向确定；地图区间包含0，不能据此断言差值方向确定；地图区间完全高于0。
8k geometry / obstacle_shift：按 off/on/smooth，K1 为 0.00% / 0.00% / 20.87%，冻结 Base 为 0.00% / 0.00% / 0.82%；同条件 adapter−Base 配对 Δ 为 +0.00 ± 0.00 pp [0.00, 0.00] / +0.00 ± 0.00 pp [0.00, 0.00] / +20.05 ± 0.73 pp [19.34, 20.76]。各 decoder 的区间解释依次为：地图区间包含0，不能据此断言差值方向确定；地图区间包含0，不能据此断言差值方向确定；地图区间完全高于0。
no_contact_loss / obstacle_shift：off/on/smooth 的实际 K1 为 0.00% / 0.00% / 20.00%；8k geometry−该控制的 matched-seed Δ 为 +0.00 ± 0.00 pp [0.00, 0.00] / +0.00 ± 0.00 pp [0.00, 0.00] / +0.87 ± 2.58 pp [0.46, 1.28]。地图区间包含0，不能据此断言差值方向确定；地图区间包含0，不能据此断言差值方向确定；地图区间完全高于0。
no_geometry_tokens / obstacle_shift：off/on/smooth 的实际 K1 为 0.00% / 0.00% / 9.68%；8k geometry−该控制的 matched-seed Δ 为 +0.00 ± 0.00 pp [0.00, 0.00] / +0.00 ± 0.00 pp [0.00, 0.00] / +11.20 ± 1.43 pp [10.62, 11.78]。地图区间包含0，不能据此断言差值方向确定；地图区间包含0，不能据此断言差值方向确定；地图区间完全高于0。

8k geometry，original 的 smooth−on：+19.31 ± 0.82 pp [18.66, 19.96]；地图区间完全高于0。
8k geometry，obstacle_shift 的 smooth−on：+20.87 ± 0.73 pp [20.19, 21.57]；地图区间完全高于0。
8k geometry，off 的 移动−原始：+0.00 ± 0.00 pp [0.00, 0.00]；地图区间包含0，不能据此断言差值方向确定。
8k geometry，on 的 移动−原始：+0.00 ± 0.00 pp [0.00, 0.00]；地图区间包含0，不能据此断言差值方向确定。
8k geometry，on_smooth 的 移动−原始：+1.57 ± 1.01 pp [1.08, 2.09]；地图区间完全高于0。
8k geometry，两场景 的 smooth移动效应−on移动效应（DID）：+1.57 ± 1.01 pp [1.07, 2.03]；地图区间完全高于0。

五档 geometry 学习曲线按 500→1000→2000→4000→8000 的实际 K1 均值如下；不强制单调，也不从 validation loss 推断任务成功，不据此选择 checkpoint 或调 decoder 参数。

original / off：0.00% → 0.00% → 0.00% → 0.00% → 0.00%。
original / on：0.00% → 0.00% → 0.00% → 0.00% → 0.00%。
original / on_smooth：17.81% → 19.38% → 19.63% → 20.12% → 19.31%。
obstacle_shift / off：0.00% → 0.00% → 0.00% → 0.00% → 0.00%。
obstacle_shift / on：0.00% → 0.00% → 0.00% → 0.00% → 0.00%。
obstacle_shift / on_smooth：18.62% → 20.53% → 21.30% → 21.27% → 20.87%。

K8 使用额外候选预算；固定子集 Any@8 与主 K1 的全部10000条件分母不同，不能把两表的差直接称为同条件 K8−K1 收益。冻结 JSON 未提供固定子集 candidate0 的汇总 CI，本报告不补造配对增益区间。

8k geometry / original / off：Any@8 0.00 ± 0.00% [0.00, 0.00]，0/1000 个条件至少在一个训练 seed 中有成功候选对，其余 1000 个条件的 pair 诊断缺失；XYZ MSD 缺诊断（n=0）。
8k geometry / original / on：Any@8 0.00 ± 0.00% [0.00, 0.00]，0/1000 个条件至少在一个训练 seed 中有成功候选对，其余 1000 个条件的 pair 诊断缺失；XYZ MSD 缺诊断（n=0）。
8k geometry / original / on_smooth：Any@8 52.83 ± 2.53% [50.50, 55.17]，445/1000 个条件至少在一个训练 seed 中有成功候选对，其余 555 个条件的 pair 诊断缺失；XYZ MSD 0.41 ± 0.02 [0.34, 0.49]。
8k geometry / obstacle_shift / off：Any@8 0.00 ± 0.00% [0.00, 0.00]，0/1000 个条件至少在一个训练 seed 中有成功候选对，其余 1000 个条件的 pair 诊断缺失；XYZ MSD 缺诊断（n=0）。
8k geometry / obstacle_shift / on：Any@8 0.00 ± 0.00% [0.00, 0.00]，0/1000 个条件至少在一个训练 seed 中有成功候选对，其余 1000 个条件的 pair 诊断缺失；XYZ MSD 缺诊断（n=0）。
8k geometry / obstacle_shift / on_smooth：Any@8 61.53 ± 0.85% [59.20, 63.94]，519/1000 个条件至少在一个训练 seed 中有成功候选对，其余 481 个条件的 pair 诊断缺失；XYZ MSD 0.29 ± 0.02 [0.24, 0.33]。

三seed聚合的 pair 观测计数是至少一个seed可计算的条件数；各seed有效pair条件可能不同。成功pair太少时，多样性只描述成功条件的条件分布，不能解释为全部请求的覆盖改善。


## 泛化范围、物理见证与计时限制

测试地图来自相同十种场景族的新地图；障碍移动是在原条件中将最近非地板 OBB 朝端点身体位置弦线移动 0.30 m。这支持地图/几何 shift 泛化评估，不证明新的工厂语义 OOD。完整接触装配可容纳的拒绝采样、角度/密度接受规则和重试记录随 manifest 冻结，不能假定等同任意实际场景分布。

条件物理见证是有限步长完整 URDF 审核的可行证据；求解器未找到见证不构成不可行证明，见证也不保证严格 shape gate。圆柱只覆盖有限上部接触带，并非全周或任意法向覆盖；forbidden 图域字段虽保留，但当前冻结输入没有启用案例，不声称该功能已经数据验证。未测试 approach/retreat、摩擦、接触力、真实工具形状或动力学可执行性；几何 jerk/acceleration 不能当作真实时间控制指标。端到端训练/守护 wall time 含温控暂停；保存的 evaluation_time_s 是评分调用 wall 计时，也可能包含暂停，均不是 flight/control latency。正式温控是 reactive 暂停：CPU≥90°C 或 GPU≥83°C，恢复需 CPU≤80°C 且 GPU≤75°C 稳定10秒；它不保证 CPU 从不越过90°C，已观察的97.125°C峰值与这一控制语义并不矛盾。

## 完整机器可读附录与复核

全部 2,263,800 个指标行： [CSV.gz](/home/z017/research/HNUTER_DiT/artifacts/contact_adapter_multifamily_v001/consolidated_reports/recovery_v002/full_statistics.csv.gz) / [JSONL.gz](/home/z017/research/HNUTER_DiT/artifacts/contact_adapter_multifamily_v001/consolidated_reports/recovery_v002/full_statistics.jsonl.gz)。每行包含表类、模型/预算/seed或三seed组、场景、decoder、分母、family/kind/difficulty/交叉scope、指标、观测数量、mean/sample SD/分位数/CI和seed均值/SD。CSV 缺值为空；JSONL 缺值为 null，seed_means 与 seeds 保留数组。按 table=K1/K8_any/K8_diversity/K1_paired/K1_adapter_control 筛选可复核所有表。

父报告 SHA256：`024270233a72825cff646464261b5b4ea2f5f32d854db70490f31c9899d9893d`。
补充报告 SHA256：`d15469eb25a15d36da13458413f42f73e01b60aff380cf1a0270eb4a764147cd`。

report.json 记录所有 22 模型、冻结参数、源文件、上游报告/输入/artifact哈希和本次三个输出文件 SHA256。生成前逐项验证完整状态、当前源码/输入/模型/raw/score/projected/meta/figure checksum与精确条件分母；缺项直接失败，CLI没有 partial/preview 模式。

训练 dataset_freeze 校验其 manifest 与 prepared_cache 字节，不加载训练轨迹。旧 freeze 没有保存 split-index/provenance 的单独 hash；本次额外记录其当前文件 hash，并交叉核对 Base、归一化与采集配置身份，不将这些新快照伪称为旧 freeze 中已有的 hash。

## 已审计的科研图

![learning_curve](/home/z017/research/HNUTER_DiT/artifacts/contact_adapter_multifamily_v001/reports/learning_curve.png)

![ablation_8000](/home/z017/research/HNUTER_DiT/artifacts/contact_adapter_multifamily_v001/reports/ablation_8000.png)

![family_kind_difficulty](/home/z017/research/HNUTER_DiT/artifacts/contact_adapter_multifamily_v001/reports/family_kind_difficulty.png)

![decoder_k1_success](/home/z017/research/HNUTER_DiT/artifacts/contact_adapter_multifamily_v001/smooth_projection_supplement/reports/decoder_k1_success.png)

![decoder_training_scale_and_shift](/home/z017/research/HNUTER_DiT/artifacts/contact_adapter_multifamily_v001/smooth_projection_supplement/reports/decoder_training_scale_and_shift.png)

![decoder_k8_success_and_saved_pair_diversity](/home/z017/research/HNUTER_DiT/artifacts/contact_adapter_multifamily_v001/smooth_projection_supplement/reports/decoder_k8_success_and_saved_pair_diversity.png)

![decoder_curvature_chord16_clearance](/home/z017/research/HNUTER_DiT/artifacts/contact_adapter_multifamily_v001/smooth_projection_supplement/reports/decoder_curvature_chord16_clearance.png)

![decoder_family_stress_and_frozen_feasible](/home/z017/research/HNUTER_DiT/artifacts/contact_adapter_multifamily_v001/smooth_projection_supplement/reports/decoder_family_stress_and_frozen_feasible.png)

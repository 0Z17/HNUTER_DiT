# 几何编码器小量对照实验

本轮比较盒子、点云和体素三种历史雷达编码。实验范围是固定预算的小量对照；
不因训练损失下降、离线几何通过或保护器拦截而扩大训练。预先固定的条件见
[geometry_encoder_protocol.json](geometry_encoder_protocol.json)，评测和最终报告均绑定其 SHA-256。
此前动态模型实验保留在 [TEMPORAL_DIT_PILOT.md](TEMPORAL_DIT_PILOT.md)。

## 不变项与比较项

三种模型使用同一份新采数据：seed 107，96 个独立组、192 个 episode，按完整组划分
64 / 16 / 16 组，对应 train / validation / test 的 128 / 32 / 32 个样本。
每组两个 episode 的最新射线观测相同、过去运动方向相反；未来障碍真值、运动速度、
episode ID 和教师等待时钟均不能进入模型。所有教师参考必须先通过原控制器、原 MJCF、
1 kHz 实际执行器飞行与接触检查，实际飞行轨迹独立保留。

正式数据已在 2026-10-04 冻结，位于本地
`results/geometry_pilot/collection96_seed107/`。独立加载检查确认 NPZ、4 份采集源文件快照
哈希及组间划分一致；采集使用 4 个 CPU worker，墙钟耗时 **166.220 秒**。
192 / 192 个教师 episode 均完成原生 MuJoCo 3.11.0 验收，无实际接触且在各自完整教师
时域最终达到目标；96 / 96 对均具有不同的等待时间。最大实际位置跟踪误差为
**0.017375 m**，最大实际速度为 **0.450212 m/s**。这些是教师数据验收结果，
不是新模型预测或工业迁移结果；完整教师时域可超过模型固定的 8 秒窗口。

| 冻结项 | 内容 |
| --- | --- |
| 数据集 SHA-256 | `548ba16e446391e800e1081825b714b84448af6683ad71022cc0fde551a8aa35` |
| 每个 episode 的决策窗口 | 1 个静止起步窗口，起步速度不超过 0.03 m/s |
| 雷达历史 | 4 帧，每帧 20,000 条近似 MID-360 射线；5 Hz 历史采样 |
| 训练目标 | 已通过物理验收的教师参考位置；`teacher_actual_positions` 单独保存实飞轨迹 |
| 未知回波 | 原始 range 为 NaN，不能当成可通行证据 |
| 来源与失败记录 | `freeze.json`、`source_snapshot/`、`collection_audit.json` |

每个样本包含过去 0.6 秒内 4 帧实际射线观测；历史先按采样时的传感器位姿变换到世界系，
再统一到当前机体位置和当前目标水平航向定义的坐标系。无回波或未观测位置保留未知含义。

| 表示 | 每帧输入 | 局部几何编码 |
| --- | --- | --- |
| box | 最多 32 个观测表面包围盒，完整尺寸和 wxyz 姿态 | 盒子 token 编码 |
| point | 最多 512 个实际观测点及有效 mask | 点云编码 |
| voxel | 64³ 网格，0.25 m 分辨率，16 m 立方范围内取 8 m 球形 ROI | 占据 / 已观测自由 / 未知三个通道的三维 CNN |

体素由各帧实际测量生成，**不是 nvblox TSDF 时域融合**。自由空间的射线证据、未知空间
与已命中表面分开编码；未来移动物体真值仅供教师和评估使用。编码器参数量和总参数量会分别
报告，不能将相同训练步数表述为完全相同计算量。

共享 DiT 配置为 dimension 128、4 个 block、4 个 attention head、2 个历史编码 block。
三个模型均从头训练 2,000 步、batch 32、seed 7、学习率 0.0002、EMA 0.99。
输入归一化只使用 train；每 100 步以 validation 的固定采样位置 RMSE 选择权重。
输出共享 16 个三次 B-spline 控制点、8 秒固定物理时间和 65 个输出时刻。速度及加速度
上限保持 0.5 m/s 和 0.6 m/s²，不通过重计时、端点钳制或未来真值挑选候选掩盖失败。

## 选型和独立测试

第一步仅使用 validation 比较三种表示。在 point / voxel 中按“可执行、无代理碰撞且
有足够进展”的整组平均比例选择工业验证模型；并列时依次使用较低位置 RMSE、point 优先。
选择结果先冻结数据、权重、预处理配置和协议哈希，再进入 test。选中的新编码器可能仍不如
box；选中不代表有效，也不会根据 test 更换工业演示模型。

test 使用全部 16 个完整组，固定 sampling seed 20404，并按 `20404 + group_id * 1000`
为同组条件使用同一随机种子。每个条件只有一个候选，不使用未来真值择优。
point 和 voxel 另做各自的历史消融：使用同一权重，把过去几何帧及 mask 全部替换为最新帧，
保留历史时间编码。差异的 bootstrap 以完整组为单位，报告 5,000 次重采样的区间。

可执行性检查原 B-spline 的速度、加速度解析极值；碰撞代理使用同一物理时刻的
42 球体 / 移动 OBB 距离，20 ms 采样并保留 0.04 m 余量。代理不是碰撞证书。
box / point / voxel 各自执行冻结 test 顺序中的前 8 个 episode，包含几何评估失败的输出，
按原控制点及解析导数进行 8,000 步实际回放。无接触的 8 秒前缀可能仍处于等待阶段，
不能将其写成完整任务到达。

工业验证只运行 validation 选中的新编码器，使用 constant_incoming / constant_outgoing
两个场景，各 30 秒、目标 `[-25,-33,2.2]`、seed 7、相同物理限幅。
分别报告真实接触、任务到达、参考接受、超限拒绝、观测保护拒绝和执行中干预。
保护器介入后的系统到达不能算作网络独立成功。outgoing 场景录制跟随相机视频，
10 fps 按仿真时间采样；视频播放速度不是系统墙钟实时性能。

## 复现入口

下面命令的路径应指向本轮同一份已冻结数据和三个训练结果。输出目录拒绝覆盖；
再次试验必须使用新的目录，不能覆盖选型或失败记录。

```bash
PYTHON="${PYTHON:-/home/z017/research/curobo_env/bin/python}"

"$PYTHON" -m hnuter_dit.temporal_data \
  --output results/geometry_pilot/collection96_seed107 \
  --groups 96 --seed 107 --rays 20000 --workers 4

for geometry_rep in box point voxel; do
  "$PYTHON" -m hnuter_dit.temporal_training \
    --dataset results/geometry_pilot/collection96_seed107 \
    --output "results/geometry_pilot/${geometry_rep}_s7_2000" \
    --input-representation "$geometry_rep" --history-mode temporal \
    --steps 2000 --batch-size 32 --seed 7 --validation-interval 100 \
    --sampling-steps 20 --device cuda
done

"$PYTHON" -m hnuter_dit.geometry_evaluation select \
  --dataset results/geometry_pilot/collection96_seed107 \
  --box results/geometry_pilot/box_s7_2000/best.pt \
  --point results/geometry_pilot/point_s7_2000/best.pt \
  --voxel results/geometry_pilot/voxel_s7_2000/best.pt \
  --output results/geometry_pilot/validation_selection --device cuda

"$PYTHON" -m hnuter_dit.geometry_evaluation test \
  --dataset results/geometry_pilot/collection96_seed107 \
  --selection results/geometry_pilot/validation_selection/report.json \
  --output results/geometry_pilot/heldout --native-count 8 --device cuda

# 从已经冻结的 validation 选型读取权重；不根据 test 更换演示模型。
geometry_checkpoint=$("$PYTHON" -c 'import json; r=json.load(open("results/geometry_pilot/validation_selection/report.json")); print(r["variants"][r["selected_new_encoder"]["representation"]]["checkpoint"])')

./run_industrial.sh --headless --mode temporal \
  --temporal-checkpoint "$geometry_checkpoint" \
  --dynamic-obstacle --dynamic-profile constant_incoming \
  --goal -25 -33 2.2 --duration 30 --max-speed .5 --max-acceleration .6 --seed 7 \
  --output results/geometry_pilot/industrial_incoming

./run_industrial.sh --headless --mode temporal \
  --temporal-checkpoint "$geometry_checkpoint" \
  --dynamic-obstacle --dynamic-profile constant_outgoing \
  --goal -25 -33 2.2 --duration 30 --max-speed .5 --max-acceleration .6 --seed 7 \
  --camera follow --record-video --video-fps 10 \
  --output results/geometry_pilot/industrial_outgoing

"$PYTHON" tools/export_flight_video.py \
  --manifest results/geometry_pilot/industrial_outgoing/video/manifest.json \
  --report results/geometry_pilot/industrial_outgoing/report.json \
  --output results/geometry_pilot/industrial_outgoing/flight.mp4

"$PYTHON" tools/report_geometry_pilot.py \
  --evaluation results/geometry_pilot/heldout/report.json \
  --industrial results/geometry_pilot/industrial_incoming/report.json \
  --industrial results/geometry_pilot/industrial_outgoing/report.json \
  --video results/geometry_pilot/industrial_outgoing/flight.mp4 \
  --output docs/geometry_encoder_validation.json
```

报告脚本生成紧凑 JSON 和同名 PNG。视频必须带有导出器生成的同名 JSON 证据文件；
报告按 MP4 哈希 → sidecar → 每帧 PNG / 时钟 manifest → 工业 report → validation 选中权重
核对完整链路，并检查 outgoing 场景、follow 相机和 10 fps。工业报告还核对 seed、目标、
请求时长和运动限幅。未到达、碰撞、超限或保护器干预的实际视频仍可保留，失败不会被裁掉。
物理回放中单个样本异常同样保留错误并继续其余固定样本；没有返回接触证据的样本标记
unknown，不能计作“已确认碰撞”或“已确认无碰撞”。

若结果连同源码已按原内容复制到正式仓库，而来源 JSON 仍记录临时工作树的绝对路径，
在报告命令增加 `--relocate-from /tmp/hnuter-dit-geometry-20261004`。工具只将该路径内的
证据映射到当前仓库，继续检查原 SHA-256，不修改冻结 JSON；相似前缀的其他路径不会被
替换。绘图的数据路径和视频 sidecar 的来源路径使用同一规则。也可以在正式仓库重新
执行视频导出命令，以新文件名生成记录正式路径的 sidecar，保留原导出证据。
推理延迟包含模型执行及张量传输，原始射线预处理的整批耗时单独报告；它不包含 IPC、
物理和渲染，也不能作为端到端实时保证。离线 box 直接加载已保存的 token，而 point / voxel
从原始历史射线重建；box 记录的离线加载耗时不包含采集时构建盒子的成本，不能将该列
当作三种表示完整预处理效率的公平排名。

## 本轮离线和原生结果

截至 2026-10-04，三种模型均完成预定的 2,000 步训练。下面的“选中 step”是仅由
validation 位置 RMSE 选出的 EMA 检查点，不代表其他模型少训练了步骤；看到 test 后
没有重新选择检查点。训练时间包含训练循环及其定期 validation，不包含此前原始射线
预处理、数据采集、原生回放或工业仿真。

| 表示 | 选中 step | 几何编码器参数 | 总参数 | 训练循环墙钟时间 |
| --- | ---: | ---: | ---: | ---: |
| box | 1,700 | 17,920 | 1,830,019 | 30.896 s |
| point | 300 | 9,088 | 1,821,187 | 38.203 s |
| voxel | 1,900 | 119,720 | 1,931,819 | 161.713 s |

point / voxel 的训练前射线预处理分别为 2.150 / 23.724 秒；box 使用已有 token，
因此不在该阶段重复构建盒子。这是固定训练步数下的比较，不是参数量或 FLOPs 完全匹配的
比较。point 结果只代表本轮点编码、训练预算和 validation 选优规则，不能推导出所有
点云编码方法都无效。

三个训练运行都使用 seed 7 和同一共享 head 架构，但几何编码器参数形状不同，构造时
消耗的随机数数量也不同，因此后续共享层的初始权重并非逐位相同。本轮只有一个训练
seed，没有多 seed 重复；即使出现小幅差异，也不能把它解释为严格隔离初始化因素后
证明的编码器因果优劣。

独立工业选型阶段使用 validation 的单一预定候选：box 为 62.5%（20/32），
point 为 0%，voxel 为 65.625%（21/32）。按预定规则在 point / voxel 中选择了
**voxel**，其选型记录在 test 之前冻结。后续工业实验继续使用该模型，不因为 test 的
box 更高而替换演示权重。

全部 32 个 test episode、16 个完整组的结果如下。组合通过率要求原时间可执行、
代理安全且相对同窗口教师具有足够进展；它不等于完整任务到达率。所有条件只采样一个
候选，没有未来真值择优或外部观测保护器干预。

| 条件 | 可执行 + 安全 + 足够进展 | 可执行 | 代理碰撞率 | 位置 RMSE | 采样 P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| box | **68.75%（22/32）** | 78.125% | 0% | 0.03044 m | 35.75 ms |
| point | 0%（0/32） | 0% | 34.375% | 0.76983 m | 35.63 ms |
| voxel | **65.625%（21/32）** | 75% | 0% | 0.02863 m | 37.41 ms |
| point，历史替换为最新帧 | 0%（0/32） | 0% | 34.375% | 0.76993 m | 35.47 ms |
| voxel，历史替换为最新帧 | 21.875%（7/32） | 68.75% | 28.125% | 0.78111 m | 35.87 ms |

voxel 相对 box 的组合通过率差为 **−3.125 个百分点**，按完整组 bootstrap 的 95%
区间为 **[−12.5, 6.25] 个百分点**。本轮没有显示 voxel 优于 box。
point 相对 box 为 −68.75 个百分点，区间 [−84.375, −50]。point 的 test 参考
平均最大速度为 1.966 m/s、平均最大加速度为 8.281 m/s²，明显超过预定限幅；
这些失败没有通过重计时修复或从统计中删去。

独立的 [point 训练诊断记录](../artifacts/geometry_encoder_pilot_v1/point_training_diagnostic.json)
检查了 4 个固定训练组，没有使用 test 指标或额外优化步骤。point 编码器、历史模块与输出层
均有有限非零梯度，参数也实际变化，但历史运动改变对采样结果的影响明显弱于 box。
因此不能只把失败归为 test 泛化问题；点采样、局部池化和特征缩放造成信息损失仍是待检验
假设，该诊断没有证明某个组件是唯一原因，也没有据此追加训练或放宽标准。

voxel 使用历史相对其自身冻结历史消融提高 **43.75 个百分点**，区间
**[28.125, 62.5] 个百分点**，支持该体素模型在这一有限课程中利用了历史。
这不能证明其优于盒子编码，也不能证明在工业背景或一般动态环境中有效。32 个 test
样本中 7 个教师在 8 秒内只需不超过 0.20 m 的进展：box / voxel 在这些样本的组合通过率
分别为 5/7、3/7，其余 25 个样本分别为 17/25、18/25；停留型样本与移动型样本均保留。

每种表示都回放相同的前 8 个固定 test episode，每次实际执行 8,000 步 / 8 秒，
共 24 次。离线代理失败的参考也照常进入这组真实物理对照，没有事后剔除。

| 表示 | 已完成回放 | 实际发生接触的 episode | 最大位置跟踪误差 | 8 秒末到达 |
| --- | ---: | ---: | ---: | ---: |
| box | 8/8 | 0/8 | 0.020842 m | 4/8 |
| point | 8/8 | **3/8** | **0.330056 m** | 0/8 |
| voxel | 8/8 | 0/8 | 0.020071 m | 4/8 |

box / voxel 通过本批原生接触和 0.12 m 跟踪误差检查；point 未通过。无接触的另一半
box / voxel 前缀仍在等待或延迟通行，不能表述为 8 个完整任务都已到达。

本轮已评检查点 SHA-256：

| 表示 | SHA-256 |
| --- | --- |
| box | `067b08e6cda8790599c13c05b9442ec8de91969ed5c452c67669076beaa9579e` |
| point | `aa3955a7dd2b7c7b5af3fbd911203a35fac11f89d34925b97d78150b2f006cbd` |
| voxel，validation 选中 | `524c71e0f959e76b162e07630587fc09d2bc20c7dc5b79e802f0e97ce04289d2` |

紧凑实测记录见 [geometry_encoder_validation.json](geometry_encoder_validation.json)，
比较图见 [geometry_encoder_validation.png](geometry_encoder_validation.png)。三个正式检查点
及冻结元信息归档在 [artifacts/geometry_encoder_pilot_v1](../artifacts/geometry_encoder_pilot_v1/README.md)，
权重分别为 `models/box/best.pt`、`models/point/best.pt`、`models/voxel/best.pt`，由 Git LFS
管理。它们是 validation 选出的 EMA 推理权重，不含恢复训练所需的 optimizer / RNG 状态。
重放本轮已归档的工业模型时，可设置
`geometry_checkpoint=artifacts/geometry_encoder_pilot_v1/models/voxel/best.pt`，再执行前述工业命令。

## 工业验证与完整视频

两场都执行 validation 预先选中的同一 voxel 权重，保持目标 `[-25,-33,2.2]`、seed 7、
0.5 m/s / 0.6 m/s² 上限。每场完成 30,000 个真实物理步 / 30 秒，实际接触检查覆盖每一步，
均为 0 次接触且没有接触缓冲溢出或无法映射的接触记录。

| 场景 | 系统结果 | 首次到达 | 最终距目标 | 接受参考 | 保护器拒绝 | 超限参考拒绝 | 执行中保护干预 | 循环墙钟时间 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| constant_incoming | 到达，`passed=true` | 16.599 s | 0.062661 m | 1 | 6 | 2 | 1 | 59.869 s |
| constant_outgoing | **未到达，`passed=false`** | — | **0.524841 m** | 1 | 2 | **19** | 1 | 108.201 s |

incoming 是观测保护器多次拒绝与一次执行干预之后的系统到达，不能计为模型独立完成。
outgoing 完成了规定的 30 秒仿真，但停在距目标约 0.525 米处，任务失败；这不是仿真
未运行完，也不能因“无碰撞”改判成功。两场的 `model_reference` 累计执行时间分别为
6.615 / 6.400 秒，outgoing 另有 0.700 秒记录为 `model_wait`。其余大量停等包含推理、
参考拒绝与保护制动，不能都解释成网络学会了等待。两场墙钟时间均超过 30 秒，不满足
完整系统墙钟实时运行。

原始报告分别为 [incoming](../results/geometry_pilot/industrial_incoming/report.json) 和
[outgoing](../results/geometry_pilot/industrial_outgoing/report.json)；可提交的报告副本见
[incoming 归档](../artifacts/geometry_encoder_pilot_v1/industrial_incoming.json) 与
[outgoing 归档](../artifacts/geometry_encoder_pilot_v1/industrial_outgoing.json)。
视频位于 [完整 outgoing 飞行视频](../results/geometry_pilot/industrial_outgoing/flight.mp4)，
使用实际跟随相机的 301 帧，覆盖仿真时间 0.0–30.0 秒，10 fps，包含首尾帧。
视频保留了参考拒绝、停等和未到达的负结果，没有裁剪成成功片段，也没有用位置回放替代
实际执行器飞行。视频文件及逐帧证据留在本地 `results/`，不作为模型权重提交到 Git。

**本轮 voxel 未显示优于 box，工业迁移也未通过无保护器干预的独立任务验证，不能据此
替换 box 基线；point 作为失败对照保留，不用于部署。本次不扩大数据或训练规模。**
后续若继续研究，应以新的受控实验检验具体问题，不能使用本轮 test 反复调参或放宽标准
来改判这些结果。

本轮完整回归为 **159 个测试通过（13.58 秒）**，命令和日志哈希见
[verification.json](../artifacts/geometry_encoder_pilot_v1/verification.json)。训练源码保持冻结，
原工业仿真仓库与旧模型权重未修改。

本课程仅覆盖静止起步和恒速穿越。移动起步、复杂动态交互、完整姿态学习、接触任务、
广泛工业分布泛化和系统墙钟实时性能均不在本轮结论内；保护器介入、拒绝与碰撞会单独
保留，不能将系统保护成功改写成模型自身成功。

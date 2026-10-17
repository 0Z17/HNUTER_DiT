# v003 Contact adapter 实验 v001

本文定义运行中的实验，不是完成结果。实际完成状态以
`artifacts/contact_adapter_multifamily_v001/` 中的状态、检查点和评分文件为准。

## 固定研究协议

基座为 canonical v003 `dit_cross`，11,964,169 个参数全部冻结。
保留已有环境 cross attention，在第 0、3、7 层加入零初始化的任务残差。
任务编码器使用曲面、参数域、工具外参、约束、起终点、支撑物与禁入区域的
不同 token；adapter 共训练 1,376,960 个参数。

训练数据是已经冻结的
`/home/z017/research/HNUTER_datasets/surface_multifamily_contact_10000traj_v001`。
其 8,000 条训练专家、1,000 条验证专家和 1,000 条测试专家按源地图隔离。
主实验规模为 500、1,000、2,000、4,000、8,000 条，每个规模使用随机种子
17、29、43。子集按完整地图嵌套抽取，每张地图 20 条轨迹；同条件的成对专家
始终在一起。轨迹数量不是独立条件数量，反向增强也不增加专家计数。

每个模型完成 8,000 次成功的 optimizer 更新，batch 64，学习率 1e-4，
EMA .995。AMP 跳步单独记录并重试。所有模型使用完全相同的固定验证噪声，
在全部 1,000 条验证轨迹上以同一评分选择检查点。接触数据不重新拟合 Base
归一化统计。检查点只保存 adapter 及可恢复状态，并绑定 Base、数据、子集和
训练源代码的 SHA-256。

另外在 8,000 条数据上各训练三个种子的负控：

- 去掉 contact、normal、smooth 几何损失，保留 velocity 和 .002 collision 损失。
- 屏蔽任务几何 token，保留起终点 token、真实环境输入和原几何损失。

这两项分别检查几何辅助损失与显式任务输入的作用，共 21 个 adapter。
完整矩阵与冻结参数见 `protocol.json`。

## 独立测试及障碍物移动

构建新的 1,000 张源地图、10,000 个任务条件，覆盖 10 个场景族。
排除 Contact 全部 500 张地图和 Base 的全部源地图；同时检查去 yaw/XY 变换后
的几何指纹。每张地图包括平面/圆柱各五个任务，每种包括三个需绕障条件和
两个简单条件。条件可行性先冻结，再运行神经模型。

每个模型在全部 10,000 条件上运行一次，主表报告单次采样成功率。
固定选择每个场景族 10 张地图，共 1,000 条件，各运行八个候选，附加报告
Any@8、候选成功率和经验路线多样性。重复采样不计为新的独立任务。

使用 25 步 DDIM/100 步训练 schedule。每个条件与候选编号派生独立的 CPU
生成器种子，所有模型与 original/shift 复用相同初始噪声。
投影开关直接复用保存的同一条原始轨迹，不重新采样。
所有模型均不附加碰撞推理引导，因此本次 Base 对照对应 v003 的无引导 prior。
旧 Base 报告另有 pointwise 和 smooth 引导结果；本次统一比较同一批 Contact
条件上的 Base、adapter 与 decoder 收益。

移动场景使用 Base 的 `shift_relevant_obstacle`：选择靠近起终点线段的
非 floor OBB，将其向线段方向移动 0.30 m。全部原始压力条件均保留；另在
模型推理前用完整 URDF 稠密证书检查与有预算的独立 OMPL 查询冻结可行子集。
端点无效、结构无效和没有找到证书分别统计；有预算的求解失败记为 unknown，
不当作不可行证明。模型、投影和指标输入均为物理条件白名单，不读取 oracle
或生成器路线标签。

## 约束投影与成功判据

解析 decoder 将预测工具点投影到有界平面/圆柱参数域，重建接触法向和最接近的
非周期 twist 姿态，保留给定起终点。它不规划绕障或修复机体碰撞。
projection off 的连续路径解释为世界 XYZ 线性插值与 SO(3) 最短 SLERP；
projection on 的解释为未包裹 intrinsic 状态线性插值后解析 lift。
这个插值区别是 decoder 定义的一部分，报告中应明确。

完整参考检查包括真实 URDF 的七个 active collision primitives、目标实体、
所有物理障碍和支撑物；训练用的 42 个球只是可微碰撞近似。
全部 128 个输出位姿测量 COAL 净空，净空已经扣过安全 margin。
候选还需满足完整机体 workspace、姿态、接触、法向、端点、参数域、禁入边、
twist 和路径质量规则。全部节点检查通过者继续按不超过 2 cm/.02 rad 的
插值步长检查；未进行密集检查的失败候选不能标为成功。

同时报告实际门限（接触 1 mm、法向 1°）与序列化数值门限（2e-6 m/rad）。
128 节点的最小净空不称为连续整段最小净空；密集检查是有限采样验证。
Base 本身也比较投影 off/on，以分离 decoder 收益和 adapter 收益。

主表按模型种子分别报告，再汇总三个种子的均值/标准差；置信区间按源地图
cluster bootstrap，而不是把同地图条件/八个候选当作独立样本。
另给场景族、曲面类型和绕障难度的分层，以及配对的投影和障碍移动差值。

## 温控、恢复与文件

`hnuter_dit.thermal_run` 只控制其创建并核对身份的专用进程组。
CPU >=90°C 或 GPU >=83°C 暂停；CPU <=80°C、GPU <=75°C 稳定十秒后恢复。
这是反应式温控，不是温度硬上限，传感器记录的峰值仍可能超过暂停门槛。
传感器缺失或守护异常采取暂停策略。状态包含 PID、start ticks、UID、PGID、SID，
恢复不得对复用后的 PID 发信号。强制 SIGKILL 无法由被杀的 Python 守护处理；
应检查守护状态，按记录身份恢复或终止本实验组。当前优先串行安排训练与
CPU 场景构建，降低散热争用；线程 affinity 调整只影响实验进程。

状态入口：

```bash
cat artifacts/contact_adapter_multifamily_v001/experiment_status.json
cat artifacts/contact_adapter_multifamily_v001/pipeline_status.json
cat artifacts/contact_adapter_multifamily_v001/thermal/pipeline/status.json
/home/z017/research/curobo_env/bin/python -m hnuter_dit.contact_benchmark \
  --output /home/z017/research/HNUTER_datasets/surface_contact_adapter_benchmark_10000cond_v001 \
  --status
```

已完成模型可在同目录恢复剩余矩阵；训练源文件或配置变化会拒绝混合恢复。
生成与评分分别按源地图原子落盘，校验 raw/input/score SHA。
完整报告只在全部 22 个模型（含 Base）和所有预定条件、候选、场景、投影组合
都通过完整性检查时生成。处理中不得把部分结果写成完整消融结论。

已经启动独立后台流水线，由同一个温控守护进程管理剩余训练、测试条件构建、
全部模型轨迹生成、两组投影评分和报告。评分使用两个单线程 worker；
构建场景使用一个 worker，各阶段串行执行。此前前台训练守护进程退出时，
下游身份检查拒绝继续；已保存两个完整模型，旧状态留在 `recovery_history/`，
现通过独立后台进程恢复第三个模型。退出原因尚未确认，不能将其归因于温度。
`thermal/training/` 是旧运行记录，当前温控状态应读取 `thermal/pipeline/`。
当前流水线命令为：

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
nohup setsid taskset -c 8,9 /home/z017/research/curobo_env/bin/python -m hnuter_dit.thermal_run \
  --state-dir /home/z017/research/HNUTER_DiT/artifacts/contact_adapter_multifamily_v001/thermal/pipeline \
  --gpu 0 -- /home/z017/research/curobo_env/bin/python -m hnuter_dit.contact_pipeline \
  --experiment /home/z017/research/HNUTER_DiT/artifacts/contact_adapter_multifamily_v001 \
  --benchmark /home/z017/research/HNUTER_datasets/surface_contact_adapter_benchmark_10000cond_v001 \
  >> artifacts/contact_adapter_multifamily_v001/pipeline_daemon.log 2>&1 < /dev/null &
```

锁会拒绝重复启动。`pipeline_source_contract.json` 固定全部执行模块、采集器物理
源文件与 native sampler，以及 Base、数据和协议哈希。运行期间修改这些文件
会拒绝接续。`source_snapshot/` 保存本次执行代码、测试和运行环境副本。
正式完成报告路径为 `reports/research_report_zh.md`，统计原始结构为
`reports/report.json`，图为 `reports/*.png`；现在这些正式结果尚未生成。

## 独立的平滑 decoder 补充消融

一张地图、10 个条件的流程试跑中，解析投影能够满足接触点与法向门限，
但预测路径的高频抖动导致曲率门限失败。该试跑不构成一万条件的成功率结果。
原始协议、训练损失、8/m 曲率门限及 off/on 评分不变；新增 `on_smooth`
作为独立的第三种 decoder，比较平滑处理对成功率、碰撞和多样性的影响。

新 decoder 先调用既有解析投影，再在有界、非周期的 `[u,v,psi]` 域内拟合
固定 16 控制点的 clamped cubic B-spline，采用域归一化的拟合误差与积分
二阶导数正则，固定权重 1e-6。起终点控制量固定，输入两端 pose9 逐位保留。
参数在正式补充评测前固定，不依据试跑成功率调参。输出仍为 128 个节点，
最终边仍使用原来的线性 intrinsic/lift 插值。

控制点有界保证数学样条处于参数矩形域内，但不保证曲率、禁入区域或机体
碰撞合格；不能将其称为连续样条碰撞证书。拟合只使用模型预测和任务参数，
不读取专家、oracle 或路线标签，也不调用绕障规划。拟合异常保留为失败，
未收敛标记单独统计，所有请求都保留在分母内。

补充评测复用 22 个模型已经保存的完全相同原始样本，每个模型对
original/shift 共 34,000 条路径评分，合计增加 748,000 个候选评分。
结果保存于 `smooth_projection_supplement/`；其源码契约与完整报告独立于
原协议。`contact_smooth_pipeline` 一次性等待父实验完整完成后执行补充评分
及报告，依然使用温控守护，并在缺模型、条件、候选或哈希不一致时拒绝完整结果。

补充状态入口为 `smooth_projection_supplement/pipeline_status.json` 与
`smooth_projection_supplement/thermal/pipeline/status.json`，完整结果输出到
`smooth_projection_supplement/reports/research_report_zh.md`。训练模型与新测试条件
仍由父流水线统一生成；补充流水线只重用父输出进行评分。

## 完整中文数值报告与附录

`report_contact_ablation_appendix` 只读取已经严格完成的父报告与平滑补充报告，
校验当前源码、输入、模型、分片、元数据和科研图的哈希，然后整合中文数值表。
它不重新采样、拟合、计算碰撞或 bootstrap。报告覆盖两种场景、三种 decoder、
五档训练数据、三个种子、Base 和两项负控，分别列出 K1、固定子集 Any@8、
K1 配对 decoder/移动/差分/adapter−对照效果、成功 gate 和地图/曲面/难度分层。
三个种子的样本标准差与源地图聚类 CI 分开报告；所有分层与诊断另完整导出
`full_statistics.csv.gz` 和 `full_statistics.jsonl.gz`，缺诊断保留空值。

默认输出目录为 `artifacts/contact_adapter_multifamily_v001/consolidated_reports/`。
完整中文报告为其中的 `research_report_zh.md`，`report.json` 记录完成状态、
全部输入/输出哈希和精确实验矩阵。结果尚未齐备时不生成正式报告。

`contact_ablation_report_pipeline` 是独立的一次性守候流程，等待两条上游流水线
完整结束和温控守护正常退出后才执行汇总。它不向上游发送信号；绑定上游的
PID、start ticks、UID、PGID、SID 与命令，身份改变或异常退出会拒绝继续。
汇总流程仍由同样的温控守护管理，状态位于
`consolidated_reports/pipeline_status.json` 和
`consolidated_reports/thermal/pipeline/status.json`。
独立 `source_contract.json` 固定新报告与守候源码；`report_commit.json` 保留
完成后的报告与附件哈希，防止恢复时覆盖已完成且校验不符的结果。

## 适用范围

数据和当前实验验证连续接触几何路径，尚不包括 approach/retreat、接触力、
动力学或实机闭环飞行。测试是相同场景族的新地图泛化，不能称为全新工业
语义环境 OOD。采集可行性筛选存在障碍密度与接触角度偏差；圆柱是有限上部
接触带，不代表整圈任意法向均有训练覆盖。

温控暂停会计入墙钟时间。有界 OMPL 见证搜索的实际运行状态可能影响是否找到
见证，因此 `unknown` 只表示预算内没有已确认见证，不能作为不可行证明。
所有模型使用同一份推理前冻结的实际测试输入与见证标签；全部 stress 分母保留
这些条件，可行子集另列，避免把搜索状态差异隐藏成模型性能提升。

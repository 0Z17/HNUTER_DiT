# 动态 DiT 小量训练实验

本实验将历史雷达直接作为扩散网络条件，并重新采集、训练独立权重。旧版 v003
及其工业场景实验保留，见 [DYNAMIC_PLANNING.md](DYNAMIC_PLANNING.md)。本实验不能
用外部保护器的制动或等待证明网络学到了动态避障。

## 输入和输出

- 输入为过去 0.6 秒的 4 帧实际射线观测，间隔 0.2 秒；每帧压缩为最多 32 个
  观测表面盒子。历史点云先变换到世界系，再统一到当前机体位置、当前起终点
  水平航向定义的 canonical 坐标系。不把各帧盒子序号当成同一物体的 ID。
- 网络条件还包括相对目标、当前线速度、当前姿态 rotation-6D。不提供障碍物真值
  速度、未来轨迹、场景时间表、episode ID 或 pair 标签。
- 小型 Temporal DiT 从头训练，通过带真实时间编码的历史 token 条件生成三维
  平移参考；姿态保持当前值。它不是新的完整六自由度学习策略。
- 输出为 8 秒、65 个固定时间点的位置参考，以 16 个三次 B-spline 控制点表示。
  前两个控制点确定当前起点和初速度，其余 14 个由扩散网络预测。不强制在 8 秒内
  到达目标，重复位置可以表示等待。运行时按原始物理时间执行，不按弧长重新计时。
- 计划以最新雷达帧的仿真时间为基准。推理完成后消耗已经过去的前缀，不把整个未来
  轨迹平移到激活时间；实测状态不匹配、参考超限或观测不充分时拒绝执行并单独统计。

## 数据和对照

初始采集使用原 HNUTER MJCF、原几何控制律和 1 kHz MuJoCo 积分。雷达使用
MuJoCo 实际碰撞几何射线求交、MID-360 近似视场，保留未知区域。教师的未来障碍
真值只用于离线生成标签、验碰和评估，不进入学生输入。
`positions` 标签是通过真实执行器飞行验收的教师参考指令，实际飞行轨迹另存为
`teacher_actual_positions`，不作为网络输入；这样不会把控制器的跟踪偏差再学习并施加一次。

同组的两个 episode 在决策时刻具有相同的最新雷达几何，但过去的障碍运动方向
相反。组内样本必须一起进入 train、validation 或 test，禁止按窗口随机拆分。
教师标签也必须实际产生不同的时序决策，才具有检验历史信息的价值。

首轮仅静止起步样本。移动中重规划、复杂非恒速运动、遮挡后突然出现、多障碍交互、
工业场景全分布覆盖均不由该小数据集保证。原始点云、数据集、训练日志及评估输出
保留在本地 `results/temporal_pilot/`；提交的冻结记录包含数据和权重 SHA-256。

同等训练预算的两个模型分别使用完整历史和仅最新帧。独立测试对照包括：

1. 训练后的历史模型；
2. 同架构、同预算的 latest-only 模型；
3. 历史模型输入的过去帧全部替换为最新帧；
4. 同初始化但未经训练的模型。

每个条件只使用一个预定 seed 的候选，同组使用相同随机 seed。禁止用未来真值从
多个预测中挑选安全路径后宣称是在线规划成功。样本的定时 42 球体/OBB 距离是几何
代理，另外用固定选取的 test episode 做实际执行器飞行与接触检查。

## 扩规模门槛

门槛预先写在 `hnuter_dit/temporal_evaluation.py` 的 `PILOT_GATES` 中，不根据 test
结果调整：至少 16 个独立测试组；原时间上可执行、无代理碰撞且有足够任务进展的
比例至少 80%；比 latest-only 高至少 10 个百分点，比历史消融高至少 5 个百分点，
比未训练模型高至少 10 个百分点；单次推理 P95 不超过 0.5 秒。还需至少 8 个固定
测试 episode 真实物理执行无碰撞且最大跟踪误差不超过 0.12 米，以及工业场景迎面/
远离两个场景实际无接触到达。工业结果必须绑定本次 temporal 权重 SHA，且没有保护器
干预、拒绝或计划拒绝造成的额外停等；系统保护成功与模型自身成功分别报告。

任一项未测或失败都不扩大数据/训练规模。训练损失下降、几何代理通过、保护器拦截
或悬停都不能单独满足该门槛。对照差异以整组 bootstrap 给出区间，有限样本不能
据此宣称一般动态环境有效或墙钟实时。

## 本轮结果

截至 2026-10-04，冻结数据含 96 个独立组、192 个 episode，每个 episode 一个静止
起步窗口。train / validation / test 分别为 128 / 32 / 32 个 episode，对应 64 / 16 / 16
个完整组。96 组均通过同组最新射线与 token 一致、历史运动相反、教师等待时间不同的检查。
目标约为 2 米平移，障碍以 0.22–0.28 m/s 匀速穿越；Y 向路线的交叉位置距起点约 1.3 米，
以避免障碍扫到静止机体。它是受限的等待/通行课程，并不覆盖任意动态航线。

模型有 1,830,019 个参数，hidden dimension 128、4 个 DiT block、4 个 attention head、
2 个历史编码 block。历史与 latest-only 版本架构相同，分别从头训练 2,000 步，batch 32、
seed 7、AdamW 学习率 0.0002、EMA 0.99。输入归一化只使用 train；每 100 步用 validation
的固定 seed、20 步 DDIM 采样位置 RMSE 选择权重。两次训练分别约 50.61 秒和 47.87 秒，
不含数据采集或物理验证。教师 B-spline 拟合 train / validation 位置 RMSE 均约 0.208 mm，
dense audit 最大速度不超过 0.451 m/s、最大加速度不超过 0.189 m/s²。

下表使用完整 16 组 test 的全部 32 个 episode，每个条件仅一个预定 seed 候选，
没有外部保护器干预或未来真值筛选。可执行性检查原 B-spline 的速度、加速度解析极值；
安全检查使用 20 ms 时间间隔的 42 球体/OBB 代理距离及 0.04 米余量。
“足够进展”相对于该条件的教师在同一 8 秒窗口内的进展计算，允许教师本来就需要等待。

| 条件 | 选用训练步数 | 可执行 + 安全 + 足够进展 | 可执行 | 代理碰撞率 |
| --- | ---: | ---: | ---: | ---: |
| 训练后的历史模型 | 2,000 | **87.5%（28/32）** | 87.5% | 0% |
| 相同历史模型，所有帧替换成最新帧 | 2,000 | 40.625%（13/32） | 81.25% | 31.25% |
| 同架构 latest-only 对照 | 300 | 0% | 0% | 43.75% |
| 未训练初始化，使用 train 归一化 | 0 | 0% | 0% | 56.25% |

latest-only 也完整训练了 2,000 步，但预定的 validation 位置 RMSE 规则选中了 step 300；
该快照在 test 上的速度和加速度超限，因此没有可执行候选。这个 0% 只描述本次预算、
训练和选优规则下的对照，不能推出所有单帧方法都无效，也没有在看到 test 后改选其他快照。
历史模型相对冻结历史消融的提升为 46.875 个百分点，按完整组 bootstrap 的 95% 区间为
[37.5, 56.25] 个百分点。这支持网络在该有限课程中利用了运动历史，尚不能证明广泛泛化。

历史模型的稳态单样本采样 P95 为 **0.03869 秒**，包含张量传输，不包含传感器捕获、
token 构造、进程通信、参考复检、物理或渲染，也不是完整系统墙钟实时率。8 个预先固定
test episode 的原执行器回放均完成 8,000 步 / 8 秒，无接触，最大位置跟踪误差
**0.020855 米**；其中 4 个在窗口末到达，另 4 个仍在执行等待/延迟通行参考，不能将无接触
回放表述为 8 个任务全部到达。

工业 `constant_incoming` 使用同一 temporal 权重完成 30,000 步 / 30 秒，真实接触检查
覆盖每一步，接触次数为零，首次到达时间 17.505 秒，最终距目标 0.079057 米。它是系统
层面的无接触到达，**未通过预注册的模型独立工业验证门槛，本次不扩大规模**。

该次运行进行了 10 次推理，仅接受 1 条参考；4 条被当前观测保护器拒绝，5 条因轨迹
超限被拒绝，执行中还发生 1 次保护器干预。4 次轨迹拒绝的最大加速度为 0.781–1.147 m/s²，
另 1 次最大速度为 0.529911 m/s，均超过预定的 0.6 m/s² / 0.5 m/s 上限；4 次保护器拒绝
均为 `observed_obstacle`，执行中的干预为 `unknown_centerline`。这些暴露了模型在真实工业
观测分布上的可执行性和覆盖不足，不能归咎于保护器或通过放宽限制解决。`model_reference` 累计执行
6.659 秒，没有记录为 `model_wait` 的参考等待。额外停等主要来自参考拒绝、
保护器制动及终点保持，不能归因于网络学会了对移动障碍等待。30 秒物理仿真的循环墙钟
时间为 117.019 秒，未达到完整系统墙钟实时。

`constant_outgoing` 远离障碍场景也完成 30,000 步 / 30 秒，每步接触检查均无接触，
首次到达时间 11.508 秒，最终距目标 0.068649 米。4 次推理仅接受 1 条参考，另有
1 次轨迹超限拒绝、2 次保护器拒绝和 1 次执行中保护干预；模型参考执行 6.661 秒，
没有 `model_wait`。循环墙钟时间为 114.626 秒。两场均为保护器介入后的系统到达，
均不满足模型独立迁移门槛；最终报告 `system_passed=true`、
`unassisted_model_passed=false`、`gate.expand=false`，本次不扩大数据或训练规模。

后续小量迭代应先覆盖更多决策时刻、工业背景与运动状态，并改善输出的物理可执行性；
再使用新的独立测试组复验，不能把本次 test 当作新的验证集或放宽限制来改判成功。

完整来源、权重哈希、分组统计和最终工业结果写入
[temporal_pilot_validation.json](temporal_pilot_validation.json)，图见
[temporal_pilot_validation.png](temporal_pilot_validation.png)。本轮已评权重及数据 SHA-256：

| 内容 | SHA-256 |
| --- | --- |
| 冻结数据 dataset.npz | `7c6e537e4cfba1ec70bddc9d14c2c4f807d235623ff74985b13ab189b4bdce0e` |
| temporal best.pt，step 2,000 | `327b2dcb97f10033a210f998822c69146214a978afbf5ca9a272fa47e8099b2f` |
| latest-only best.pt，step 300 | `2b6ba77c0de6311a54259aef091a783b0918a9b7ade6601aa71cf1693cdc87ad` |

本轮完整代码回归为 127 个测试通过（11.05 秒）；覆盖时间化表示、训练/归一化隔离、
检查点协议、真实独立推理进程、固定时间执行、输入观测和评估门槛。复现测试命令：

```bash
PYTHONPYCACHEPREFIX=/tmp/dit-pycache PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=2 \
  /home/z017/research/curobo_env/bin/python -m pytest -q
```

正式模型保存在 `artifacts/temporal_mid360_pilot_v1/models/temporal/best.pt` 和
`artifacts/temporal_mid360_pilot_v1/models/latest_only/best.pt`。旧 v003 仍是 WebUI 默认，
新 temporal 模式必须提供检查点，加载失败不会回退到旧模型。

## 复现

从仓库根目录执行。以下使用本机已安装 PyTorch/CUDA、NumPy、SciPy 和 MuJoCo 的环境；
其他机器可通过 `PYTHON` 指定具备相同依赖的解释器。工业场景仓库由 `HNUTER_SIM_ROOT`
或 `--sim-root` 指定，默认 `/home/z017/research/isaacsim-industrial-reactor`。
数据、训练和评估入口拒绝覆盖已有冻结结果；重复实验应把各命令中的输出目录一起换成新目录。

```bash
PYTHON="${PYTHON:-/home/z017/research/curobo_env/bin/python}"

"$PYTHON" -m hnuter_dit.temporal_data \
  --output results/temporal_pilot/collection96_seed73 \
  --groups 96 --seed 73 --rays 20000 --workers 4

"$PYTHON" -m hnuter_dit.temporal_training \
  --dataset results/temporal_pilot/collection96_seed73 \
  --output results/temporal_pilot/temporal_s7_2000 \
  --history-mode temporal --steps 2000 --batch-size 32 --seed 7 \
  --validation-interval 100 --sampling-steps 20 --device cuda

"$PYTHON" -m hnuter_dit.temporal_training \
  --dataset results/temporal_pilot/collection96_seed73 \
  --output results/temporal_pilot/latest_s7_2000 \
  --history-mode latest --steps 2000 --batch-size 32 --seed 7 \
  --validation-interval 100 --sampling-steps 20 --device cuda

"$PYTHON" -m hnuter_dit.temporal_evaluation \
  --dataset results/temporal_pilot/collection96_seed73 \
  --temporal results/temporal_pilot/temporal_s7_2000/best.pt \
  --latest-only results/temporal_pilot/latest_s7_2000/best.pt \
  --untrained results/temporal_pilot/temporal_s7_2000/initial.pt \
  --output results/temporal_pilot/evaluation_v1 \
  --seed 10403 --steps 20 --native-count 8 --device cuda
```

`initial.pt` 保存训练开始前的随机权重，但使用 train 归一化，因此并非完全无数据先验。
`best.pt` 是 validation 选取的 EMA 推理权重；**本 temporal 训练入口的 `last.pt` 仅保存
最终 EMA 快照，不含 optimizer、sampler 或 RNG 状态，不能恢复训练**。它与旧 v003 的
可恢复 `last.pt` 语义不同。原始日志、所有中间权重和数据留在被 Git 忽略的 `results/`。

工业飞行使用 `isaaclab-sim:6.0.0-nvblox` 镜像中的隔离推理进程和原 1 kHz 执行器控制。
下面的目标由初始约 `(-25,-35,2.2)` 沿世界 +Y 平移 2 米；限幅仍为 0.5 m/s 和 0.6 m/s²，
不会通过拉长预测时间来掩盖不可执行输出。去掉 `--headless` 可打开 GUI。

```bash
./run_industrial.sh --headless --mode temporal \
  --temporal-checkpoint artifacts/temporal_mid360_pilot_v1/models/temporal/best.pt \
  --dynamic-obstacle --dynamic-profile constant_incoming \
  --goal -25 -33 2.2 --duration 30 --max-speed .5 --max-acceleration .6 --seed 7 \
  --output results/temporal_pilot/industrial_incoming_seed7

./run_industrial.sh --headless --mode temporal \
  --temporal-checkpoint artifacts/temporal_mid360_pilot_v1/models/temporal/best.pt \
  --dynamic-obstacle --dynamic-profile constant_outgoing \
  --goal -25 -33 2.2 --duration 30 --max-speed .5 --max-acceleration .6 --seed 7 \
  --output results/temporal_pilot/industrial_outgoing_seed7
```

工业入口会校验检查点哈希，提前完成 CUDA 预热，并把原始计划、拒绝原因、保护器干预、
参考执行时间和每一步真实接触记录写入 report。当前训练只覆盖静止起步；运行时要求
当前与历史速度都不超过 0.03 m/s 后才提交新计划，不宣称已训练移动中重规划。
观察未知或条件不支持时停等是保护器行为，需要与模型自己生成的等待分开计数。

所有报告齐备后，生成紧凑记录与比较图：

```bash
"$PYTHON" tools/report_temporal_pilot.py \
  --evaluation results/temporal_pilot/evaluation_v1/report.json \
  --dataset results/temporal_pilot/collection96_seed73 \
  --temporal-training results/temporal_pilot/temporal_s7_2000 \
  --latest-training results/temporal_pilot/latest_s7_2000 \
  --industrial results/temporal_pilot/industrial_incoming_seed7/report.json \
  --industrial results/temporal_pilot/industrial_outgoing_seed7/report.json \
  --output docs/temporal_pilot_validation.json
```

# Task-adaptive Surface / Inspection 小量试验

日期：2026-10-04。分支：`codex/task-adaptive-contact-inspection`。

本轮完成 48 次专家数据采集、真实 MID-360/nvblox 编码、冻结先验的独立任务残差训练，以及 64 次执行器驱动的独立测试回放。**Surface 小样本显示任务条件能改变扫掠路线；Inspection 未达到任务成功门槛。暂不扩大数据规模。**

这是一项新的 SDF 扩展试验，不是论文 canonical v003 的直接复现，也不是工厂内接触作业部署。仿真使用本机 native MuJoCo 3.11.0 和未修改的 HNUTER 原控制器，1 ms 真实物理积分；没有通过修改机体位姿执行轨迹。

## 论文对应关系

已检查 `/home/z017/research/task_adaptive_diffusion`。当前编译入口 `ieeeconf/task_adaptive_diffusion.tex` 引入英文正文 `task_adaptive_diffusion_en.tex`；`MANUSCRIPT_GUIDE.md` 和 evidence manifest 对应较新的 Surface V0 实验。根目录 README、大纲及手工草稿还保留较早的 Surface+Inspection/组合任务设想，不能混用其模型哈希、数据量和成功率。

| 论文立意 | 本轮实现与边界 |
| --- | --- |
| 冻结运动先验，任务以可插拔残差适配 | 冻结当前 SDFPoseDiT 的所有权重、均值与标准差；Surface/Inspection 各一个独立适配器 |
| 任务几何与运动先验分离 | 24 维外部任务命令独立于实测 nvblox 地图；不输入专家绕行方向、未来状态、成功标签或场景 ID |
| 原始结果与几何修正分开 | 原始去噪结果直接计算训练几何损失；独立、无参数 decoder 只修平面/法向或控制点相机朝向 |
| 不能用投影掩盖任务无条件退化 | 同一起终点设置成对禁作业区域；直线+投影不能满足任务。固定地图和噪声，替换任务描述符做消融 |
| 几何接触不等于真实作业 | 增加有质量的探针、真实 solver 接触力、原执行器回放；分别报告参考几何、实际几何、力、失联、进度和非工具碰撞 |
| 观测与多任务设想 | 实现独立 Inspection 单点观测验收；未验证多适配器叠加、任务冲突处理或串行组合策略 |

当前正文的 Base 是 11,964,169 参数、128×pose9、SHA `301070cd…636dad`；本轮继续使用前一阶段的 **1,679,430 参数 SDF 六自由控制点先验**，SHA `6e0da29b562b433bc1813b2f632cdd1f0a420705b003767ae32ce31e94745192`。因此本轮数值不能放入原论文表格当作同协议结果。

## 任务、数据和感知

12 个布局，每布局 2 类任务×2 个条件，共 48 次固定尝试。按完整布局拆分为 8/2/2 个 train/validation/test 布局，即 32/8/8 条；成对条件不跨 split。每条轨迹 20 s，操作窗为 `[6.667, 13.333] s`。

- **Surface**：水平有限作业台，向下探针球面极点为机体系 `(0,0,-0.75)` m；在接触扫掠时绕开指定的禁作业区域。成对条件的场景、端点和传感器输入相同，仅禁区命令相反。禁区是作业语义区域，不是假造的实体障碍。
- **Inspection**：同一起终点，观察侧面目标点。使用机头 FPV 外参 `(0.24,0,0.025)` m，+X 光轴、90°×58.7°视场；实际 `mj_ray` 第一命中判断遮挡。要求角误差≤10°、距离 1.7–2.5 m，整个操作窗有效比例≥90%，连续有效≥5 s。它是单点可见性验收，不是图像识别、目标面积覆盖或视觉定位。

每次起飞前实际采集 0/0.2/0.4/0.6 s 四帧、每帧 20,000 条 MID-360 射线，同步真实物理位姿。实际 nvblox GPU 融合得到 4×6×32³ 局部字段，六通道为 TSDF、权重、ESDF、ESDF known、TSDF known、最新回波；未知区域保留独立标记，不视作空闲。

工作台、目标点和禁区来自**外部任务命令**，不是由当前雷达重建出来的任务识别结果。MID-360 有近距离向下盲区，不能声称它已完整感知正下方接触面。本轮是静止起步的四帧条件规划，未在接触飞行中在线重规划或更新任务几何。

采集结果：

| 任务 | 通过/尝试 | 有效 train / validation / test | 非许可碰撞 |
| --- | ---: | ---: | ---: |
| Surface 软接触稳定窗 | 18/24 | 10 / 4 / 4 | 0 |
| Inspection | 24/24 | 16 / 4 / 4 | 0 |

六条未过关的接触记录完整保留，不作为正专家训练。数据文件保留所有尝试；训练显式筛选 `accepted`，测试不筛选。采集耗时 75.77 s，使用 6 个独立 CPU 进程，不是在线工业仿真的实时性能测试。

## 接触模型和验收的实际含义

探针为固定子刚体，显式质量 20 g、正定惯量，含杆和半径 25 mm 的球；球心 z=-0.725 m，不能把球心误当论文的接触参考点。总物理质量 4.520024 kg，原控制器质量参数 4.5 kg、增益、倾转交叉映射和执行器限幅均未改变。

开发阶段保留 11 组调试记录，之后冻结：摩擦 `[0.2,0.005,0.0001]`、`solref=[5,1]`、`solimp=[0.9,0.95,0.001]`。这是**很软、未经实物标定的数值接触**。专家最大 solver 穿透 11.60–12.03 mm，不能称为真实刚性表面的稳定力控。

公共力外环读取 `mj_contactForce`，目标 0.8 N，Kp=0.002 m/N、Ki=0.004 m/(N·s)，位置参考补偿限幅±35 mm、速率≤25 mm/s；只改参考，不改实际状态。所有方法共用。接触仅允许 `probe_tip/task_pad` 顶面且处于 engage/operate/release 阶段；其他机体、杆、侧面或错误阶段接触判失败。

稳定窗为 OPERATE 开始后预先固定的 1 s 起至操作结束：接触占比≥95%，0.5–1.5 N 力区间占比≥90%，最长失联≤0.15 s，全阶段峰力≤3 N；另检查跟踪、终点、饱和、禁区和台面边界。

**18/24 只表示该稳定窗门槛通过。** 专家完整操作窗的力区间占比只有 83.25–83.73%，完整操作窗力指标 0/24；实际工具平面误差≤5 mm、法向≤2°的独立诊断也为 0/24。没有把这两项失败隐藏到“成功”中。

在学习模型的测试开始前，另固定了沿任务 X 方向扫掠进度≥总 X 位移40%的门槛，防止接触停留后才飞向终点。对已采专家只读追加审计，24/24 满足，原始采集报告未改写；见 `collection_task_audit.json`。

## 训练与输出容量

适配器使用 3 个任务 token，在第 0/2/3 个块注入零初始化 cross-attention 残差，瓶颈 32。六控制点采用 `[.35,.85,1,1,.85,.35]` 软支持权重；后续 attention/样条仍可传播变化，**不宣称严格的输出相位局部性**。

本次发现旧 SDF 的姿态 residual 标准差固定为 0.04，而冻结 LayerNorm+输出层将归一化 velocity 幅度限制在约4.54；Inspection 专家 yaw 约1.0–2.18 rad，超出旧小姿态分布。仅中间残差难以覆盖该输出范围。因此增加独立、零初始化的任务 velocity 输出残差头，基座仍逐位冻结。每任务 335,814 个参数；保留 block-only 的 327,392 参数对照。这是 SDF 扩展的新增设计，不冒称原论文完全相同的架构。

训练使用 100-step cosine v-prediction、20-step DDIM，seed=41；损失为 velocity MSE＋原始 clean estimate 上的任务几何损失＋物理控制量模仿误差，训练中不调用 decoder。

| 运行 | 训练步数 | 物理模仿权重 | 最低验证控制量 MSE |
| --- | ---: | ---: | ---: |
| Surface | 1500 | 0.5 | 0.001139 |
| Inspection 首轮 | 1500 | 0.5 | 0.207421 |
| Inspection block-only 首轮 | 1500 | 0.5 | 0.498061 |
| Inspection 调整后，选用 | 3000 | 20 | 0.101201 |
| Inspection block-only 同预算 | 3000 | 20 | 0.388273 |

首轮验证集显示朝向学到但距离不稳，所以只在原小数据上调整一次模仿权重并延长训练；所有选择先于测试推理。五次运行的基座/归一化逐位未变，基座梯度为 None。该验证 MSE 将米和弧度一起平均，是选点诊断，不能直接当任务成功率。

## 独立执行结果

每任务固定 4 条 held-out 条件，每方法每条件一个候选，seed=23；模型、错误任务条件和基线使用相同噪声、实测地图和初始 qpos/qvel/act/time。共 64 次原执行器回放，无非许可碰撞、无执行异常；几何和任务失败仍算失败。

| 方法 | Surface 软接触＋合法扫掠 | Inspection 全任务 |
| --- | ---: | ---: |
| 直线＋decoder | 0/4 | 0/4 |
| Base 原始 | 0/4 | 0/4 |
| Base＋decoder | 1/4 | 0/4 |
| Task adapter 原始 | **4/4** | **0/4** |
| Task adapter＋decoder | **4/4** | **0/4** |
| 错误任务描述符，原始/decoder | 0/4，0/4 | 0/4，0/4 |
| block-only，原始/decoder | 未运行 | 0/4，0/4 |

Surface 原始参考的平均平面误差中位数从 Base 的246.0 mm降至3.11 mm；直接 decoder 可将注册水平起始配置下的参考误差降至数值精度。但结果必须分层解释：

| Surface 诊断 | Adapter 原始 | Adapter＋decoder |
| --- | ---: | ---: |
| 参考几何≤5mm/2°＋合法区域 | 2/4 | 4/4 |
| 实际全操作窗几何≤5mm/2°＋合法区域 | 2/4 | 0/4 |
| 实际完整操作窗力区间占比≥90% | 4/4 | 0/4 |
| 稳定窗软接触＋合法扫掠 | 4/4 | 4/4 |

这表明**精确的参考平面投影并不保证更好的接触执行**。原始残差在这四条测试中碰巧更适合公共柔顺外环，不能据此声称已学习接触力策略：网络没有接收力反馈，力反馈只在公共执行外环。

Inspection 原始 adapter 的有效观测比例为37.05–63.48%，连续有效只有1.35–2.35 s，未达到90%/5 s。原始 block-only 为0%，但其 decoder 对照有效比例可达60.75–82.90%，仍未达标；因此不能因为验证误差降低就宣称新增输出头带来完整任务优势。错误任务描述符是固定真实场景下的干预消融，不是新有效目标的泛化测试。

单次缓存环境特征后的20步 GPU 去噪中位数：Base 34.8 ms，adapter 50.0 ms；仅8个条件，包含首次调用，没有专门预热基准，不含建图、编码、decoder或物理执行，不能外推为在线规划周期。

## 文件、证据和视频

- 代码：`hnuter_dit/task_{spec,physics,inspection,data,adapter,training,evaluation,video}.py`。
- 紧凑证据：[task_adaptive_pilot_validation.json](task_adaptive_pilot_validation.json)。完整原始数据、雷达、trace、失败、训练与评估报告保留在 `results/task_adaptive_pilot_v1/`，按仓库规则不提交。
- 权重与配置：`artifacts/task_adaptive_pilot_v1/models/`，每个 `best.pt` 只含该适配器；以 base SHA 绑定原 SDF checkpoint。未修改默认 WebUI 模型。
- 51 项任务单元/物理检查通过，包括禁区反事实、零初始化等价、基座冻结、原始几何梯度、真实接触白名单、传感器遮挡和连续停留中断反例。
- `results/task_adaptive_pilot_v1/export_final/task_expert_pilot.mp4`：固定首个 held-out 布局的040/042专家回放，40 s；Surface 明示 PARTIAL。
- `results/task_adaptive_pilot_v1/export_final/task_learned_pilot.mp4`：相同040/042的原始适配器执行，展示实际通过/失败。

视频是**真实物理遥测的示意回放，不是 Isaac RGB 录像**。雷达是实际采集的起飞前扫描，明确标记 frozen；没有伪装成飞行中更新。专家视频是采集上界，不代表模型效果。每个视频附来源哈希、海报和完整解码检查。

## 复现小量实验

CPU 需 numpy/scipy/mujoco/pytest；本机已验证 `/home/z017/research/curobo_env/bin/python`。GPU 使用既有 `isaaclab-sim:6.0.0-nvblox` 隔离环境，不安装或修改原 Isaac Torch。`HNUTER_SIM_ROOT` 可指向外部仿真仓库。使用新输出目录，已有目录会拒绝覆盖。

```bash
# 在本仓库根目录；python 必须有上述 CPU 依赖
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python -m pytest -q tests/test_task_adapter.py tests/test_task_spec.py \
  tests/test_task_physics.py tests/test_task_inspection.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m hnuter_dit.task_data --output results/task_pilot_new/collection --workers 6

./tools/run_task_gpu.sh hnuter_dit.task_training prepare \
  --source /dit/results/task_pilot_new/collection --output /dit/results/task_pilot_new/prepared --sim-root /sim
./tools/run_task_gpu.sh hnuter_dit.task_training train \
  --source /dit/results/task_pilot_new/prepared \
  --base /dit/artifacts/static_sdf_pose_pilot_v1/models/sdf/best.pt \
  --task surface --steps 1500 --output /dit/results/task_pilot_new/models/surface
./tools/run_task_gpu.sh hnuter_dit.task_training train \
  --source /dit/results/task_pilot_new/prepared \
  --base /dit/artifacts/static_sdf_pose_pilot_v1/models/sdf/best.pt \
  --task inspection --steps 3000 --physical-loss-weight 20 \
  --output /dit/results/task_pilot_new/models/inspection
./tools/run_task_gpu.sh hnuter_dit.task_training train \
  --source /dit/results/task_pilot_new/prepared \
  --base /dit/artifacts/static_sdf_pose_pilot_v1/models/sdf/best.pt \
  --task inspection --steps 3000 --physical-loss-weight 20 --block-only \
  --output /dit/results/task_pilot_new/models/inspection_block_only
./tools/run_task_gpu.sh hnuter_dit.task_evaluation predict \
  --source /dit/results/task_pilot_new/prepared \
  --base /dit/artifacts/static_sdf_pose_pilot_v1/models/sdf/best.pt \
  --adapters /dit/results/task_pilot_new/models \
  --block-only-adapter /dit/results/task_pilot_new/models/inspection_block_only/best.pt \
  --output /dit/results/task_pilot_new/predictions
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m hnuter_dit.task_evaluation execute \
  --source results/task_pilot_new/predictions --collection results/task_pilot_new/collection \
  --output results/task_pilot_new/evaluation --workers 6
```

GPU 启动器使用镜像默认用户；该镜像的隔离环境普通 UID 不可执行，生成目录可能属于 root。当前交付的本地结果已复制为当前用户所有。

下一轮应先改善专家全操作窗的力稳定性并标定接触柔顺模型，再处理相机连续朝向、距离和停留约束，验证通过后再考虑多任务串联、更多目标/障碍布局及工业环境部署。当前证据不足以开展大规模采集或宣称真实接触作业可用。

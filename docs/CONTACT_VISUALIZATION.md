# Contact adapter 三维可视化与 MuJoCo 闭环执行

入口为 `hnuter_dit/contact_visualization.py`、`run_contact_demo.sh`。本程序读取已完成的 Contact/Surface 实验，支持 Base 和 21 个 adapter、独立测试集的 10,000 个条件、原始/移障场景与三种 decoder。不启动训练或正式实验流水线，也不写回冻结实验目录。

## 2026-10-13：载入诊断与曲率警告修订

只读核查的 294 次真实回放全部读取成功，覆盖 22 个模型、十类任务族、平面/圆柱和原始/移障场景；其中 250 次没有通过原界面的严格执行筛选。原实现仅发送完全成功的 `on_smooth`，忽略当前选择的 decoder，容易将“不能发送仿真”误认为“轨迹没有载入”。现在分别显示读取结果、正式评测失败与执行阻断原因。保存协议只在预选 100/1000 张地图保存八候选，其余保存候选 0；可以用在线推理生成更多候选。

现在默认勾选“曲率超标允许诊断飞行”。超过 8 m⁻¹ 时显示醒目警告、超限节点数和原失败判定，同时允许通过其他检查的 `on` / `on_smooth` 进入闭环仿真。取消勾选恢复曲率阻断。碰撞、接触/法向、姿态、工作区、端点、禁区、twist 连续性、回退和 chord/arc 仍然拦截；原本因曲率失败跳过的密集检查会在实际执行参考上补做。

诊断飞行保留正式 `candidate_success=false`，不改写被冻结的实验结果。界面可调整参考平移速度/加速度上限，显示实际跟踪最大误差与 RMS、实际速度、加速度、jerk、角运动峰值和时间化方法。实际峰值来自 50 ms 控制周期的速度差分，不代表 1 ms 物理子步的全时域峰值。关闭已完成的物理窗口后保留已完成飞行的结果。

## 启动

本机已有环境：

```bash
cd /home/z017/research/HNUTER_DiT
./run_contact_demo.sh --device auto --open-browser
```

打开 `http://127.0.0.1:8793/`。默认回放 `geometry_n8000_seed17`、`map_000527/condition_002` 的原始场景。这是交接文档中的诊断案例，便于观察 raw、普通投影与平滑投影的区别；它不是总体成功率或模型最优性的代表。

无图形显示时仍可执行完整动力学：

```bash
./run_contact_demo.sh --device cpu --headless --no-realtime
```

跨机器可配置环境、数据与外部资产：

```bash
HNUTER_DIT_PYTHON=/path/to/python ./run_contact_demo.sh \
  --checkpoint /path/to/base/best.pt \
  --experiment /path/to/contact_adapter_multifamily_v001 \
  --benchmark /path/to/surface_contact_adapter_benchmark_10000cond_v001 \
  --collector-root /path/to/HNUTER_trajectory_collector \
  --mujoco-root /path/to/MuJoCo-for-HNUTER
```

Python 依赖沿用 `.[simulation]`：PyTorch、NumPy、SciPy、TOPP-RA、COAL、MuJoCo；外部 URDF、MuJoCo XML、meshes、控制器和 MPPI 包见 [交互系统交接](https://github.com/0Z17/HNUTER_DiT/blob/codex/interactive-mujoco-toppra/docs/INTERACTIVE_SYSTEM.md)。离线回放不加载神经网络权重，在线推理按所选模型延迟加载权重。浏览器画布无 CDN 依赖。

## 操作

1. 选择模型、场景、任务条件；按场景族、plane/cylinder 和 easy/detour 筛选或输入完整 condition ID。
2. “读取保存轨迹”使用正式实验的原始预测和评分；“在线推理”重新生成 1–8 个共享 latent 候选并做完整 URDF 审核。在线结果仅保存在服务内存，导出由浏览器下载。
3. 切换候选与 decoder，叠加三条机体/工具轨迹，拖动进度观察机体姿态、工具杆、工具点和曲面。查看节点接触误差、法向误差、局部曲率与净空。
4. 预览 MuJoCo 场景；选择当前候选的 `on` 或 `on_smooth`，必要时启用曲率诊断模式，再执行实际旋翼/倾转动力学闭环。服务会在执行前审核该时间参考，其他约束失败时显示具体原因。
5. 查看实际机体/工具轨迹、跟踪误差与任务误差；停止后等待执行器确认退出。导出 JSON 保存当前规划来源、条件、三种轨迹、原始评分与实际执行状态。

更换模型、条件或场景会使旧规划失效。执行中禁止改动规划；执行请求仅接受服务器的 `plan_id`、候选编号与 decoder，不读取客户端替换的曲线或环境。服务重启后需要重新载入规划。

## 三类轨迹与时间参考

| 输出 | 几何与指标含义 |
|---|---|
| `off` | 原始模型的 128 个世界 pose9；连续边为世界 XYZ 线性与最短 SO(3) SLERP。 |
| `on` | 原有解析投影，连续边为未包裹 `[u,v,psi]` 线性插值后解析 lift。 |
| `on_smooth` | 原有 16 控制点有界 cubic 拟合后输出 128 节点，正式连续边仍为 intrinsic 线性/lift。 |
| `on` MuJoCo 参考 | 精确保留全部 128 个 intrinsic 节点和线性/lift 边，各段使用 quintic 静止到静止计时；不进行几何平滑。 |
| `on_smooth` MuJoCo 参考 | 有界 intrinsic cubic 的解析 lift，经 TOPP-RA 生成物理时间参考；执行前对这条实际曲线独立复检。 |

圆柱上的投影轨迹由服务端插值 intrinsic 再 lift，浏览器不使用穿入圆柱的世界坐标弦线来播放。机体原点与工具点分别绘制，工具位置使用真实 `position_body` 与机体旋转计算。几何进度按节点索引归一化，不是物理秒；MuJoCo 仿真时间来自实际动力学积分。

离线评分保持冻结值，包括 1 mm / 1° 实用接触门限、8 m⁻¹ 曲率、0.25 m 回退、0.65 chord/arc 和原 collision margin。原始解码失败保留失败信息，缺失或未审核值显示为空，不填成零。净空来自 128 节点且已经扣除 margin，密集边检查不等于密集净空测量。

执行参考采用解析曲面几何，不使用自由飞行的独立 XYZ/rotation 平滑。约束与前馈使用世界平移和机体系角运动，包括工具偏置的影响。`on` 的空间折角处曲率没有光滑定义，所以每个节点停车以保持有限加速度；它保留原始节点的离散曲率警告，不把边内部的小曲率当作原始路径已合格。`on_smooth` 的实际 cubic 与冻结评测的 128 条边不同，必须复检实际曲线。两类参考都复检 full URDF、目标实体、支撑物、环境、工作区、姿态、图域、禁入区、接触/法向、端点和形状；诊断模式只把曲率转为警告。有限密集采样仍不是连续碰撞证书。

MuJoCo 场景包含同一条件的 OBB、支撑物、完整平面实体或圆柱实体。闭环复用既有残差 MPPI、配平、执行器分配与旋翼/倾转模型；飞行过程中不写入 free joint pose 来移动无人机。真实机体与环境接触仍会停止运行。

## 实际任务误差的解释

`flight_success` 表示仿真完成且满足既有终点跟踪容差；`contact_metrics.task_success` 还要求全部已记录实际工具点满足严格接触/法向和其他几何 gates。这两个结果独立展示，飞行到达终点不自动算接触任务成功。

当前接触工具是理想校准点与法向，MuJoCo 中没有新增实体工具的接触力、摩擦或阻抗控制。这里验证实际飞行动力学下的接触几何跟踪，不证明接触力任务已解决。实际任务诊断按控制周期采样，并明确记录采样范围；它不等于每个 1 ms 物理子步的工具接触证书。approach/retreat、Inspection adapter、真实工具接触与力控制仍需独立实现和验证。

## 接口

| API | 作用 |
|---|---|
| `GET /api/catalog` / `/api/health` | 模型、任务族、默认条件、执行器可用性。 |
| `POST /api/conditions` | 条件筛选与分页，最大 200 条/页。 |
| `POST /api/replay` | `{model_id, condition_id, scenario}`，读取并核验冻结结果。 |
| `POST /api/sample` | 同上，加 `count`（1–8）、`sampling_steps`（1–100）；固定共享 latent seed 20261011。 |
| `POST /api/plan/invalidate` | `{plan_id}`，撤销规划。 |
| `POST /api/execution/preview` | `{plan_id}`，同步完整任务场景。 |
| `POST /api/execution/start` | `{plan_id, candidate_index, decoder:"on"或"on_smooth", allow_curvature_failure:true, options:{speed_mps:0.35,max_acceleration_mps2:1.0}}`，独立审核后执行。 |
| `GET /api/execution/status` | 实际状态、机体/工具轨迹、跟踪误差、接触任务误差。 |
| `POST /api/execution/stop` | 请求停止，由工作线程确认后进入 stopped。 |

输入只从物理条件白名单读取。读取时核验 benchmark cache/index、条件分片、input SHA、模型契约、raw、评分和平滑分片的 hash 与身份绑定，不读取专家轨迹、oracle、路线标签或证书作为模型输入。

回归测试入口：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MPLCONFIGDIR=/tmp/hnuter-dit-mpl \
/home/z017/research/curobo_env/bin/python -m pytest -q \
  tests/test_contact_visualization.py tests/test_contact_execution.py \
  tests/test_retiming.py tests/test_mujoco_execution.py \
  tests/test_flight_control.py tests/test_execution_motion_metrics.py
```

本机的 ROS 全局 pytest 插件与项目 Python 不兼容，所以上述命令关闭无关插件自动加载。

## 已保存的程序验证

2026-10-13 使用真实权重完成默认条件的一候选 CPU 在线推理，并读取正式保存的八候选轨迹。两类任务均完成无图形显示的完整 MuJoCo 闭环（192 samples × 32 horizon、seed 1，速度上限 0.35 m/s）：

| 保存条件 / 候选 | 仿真时间 | 机体环境接触 | 终点位置误差 | 最大实际工具误差 | 最大实际法向误差 | 飞行成功 | 严格接触几何通过 |
|---|---:|---:|---:|---:|---:|---|---|
| plane / map_000527 / condition_002 / 7 | 14.65 s | 0 | 1.58 mm | 4.55 mm | 1.81° | 是 | 否 |
| cylinder / map_000001 / condition_009 / 0 | 7.80 s | 0 | 3.92 mm | 12.82 mm | 7.07° | 是 | 否 |

这两例是软件与执行链验证，不是新增总体成功率实验。时间参考自身通过独立精确曲面与完整 URDF 采样检查，实际工具几何失败来自闭环运动中的跟踪偏差。未改变正式实验阈值或结果。

完整规划、实际轨迹和审核记录在 [验证产物目录](../artifacts/contact_visualization_validation_v001)：`plane_plan.json`、`plane_trial.json`、`cylinder_plan.json`、`cylinder_trial.json`。

## 曲率诊断真实闭环对照

修订后使用同一 `geometry_n8000_seed17 / map_000527 / condition_002 / original / candidate 0`，复用保存的 raw。两次都是 seed 1、192×32 MPPI、速度上限 0.35 m/s、加速度上限 1 m/s²、角速度/角加速度上限 0.5 rad/s / 1 rad/s²；未重新采样或修改原始评分。

| 指标 | `on` 曲率诊断 | `on_smooth` |
|---|---:|---:|
| 保存源轨迹最大曲率 | 15.423 m⁻¹ | 2.592 m⁻¹ |
| 仿真时间（含保持/收敛） | 69.90 s | 15.80 s |
| 跟踪 RMS / 最大误差 | 7.45 / 31.46 mm | 5.44 / 20.52 mm |
| 实际平移加速度峰值 | 2.041 m/s² | 1.316 m/s² |
| 实际平移 jerk 峰值 | 14.949 m/s³ | 8.750 m/s³ |
| 实际角加速度 / 角 jerk 峰值 | 5.312 rad/s² / 83.001 rad/s³ | 1.126 rad/s² / 15.206 rad/s³ |
| 最大实际工具点误差 | 11.08 mm | 2.98 mm |
| 最终目标位置误差 | 1.027 mm | 0.824 mm |
| 环境接触数 | 0 | 0 |
| 飞行 / 严格接触几何 | 通过 / 失败 | 通过 / 失败 |

这例曲率超限轨迹在有界计时下能够完成飞行，动态跟踪和运动峰值较差。`on` 每节点停车，`on_smooth` 连续通过，比较同时包含几何与计时策略的差异，不能解释成同一固定速度下曲率的纯因果效应；实际运动峰值也不是参考上限的硬保证。原 8 m⁻¹ 是几何质量筛选门限，不是由本飞行试验确定的不可飞物理边界。

数据及输入/代码哈希在 [comparison.json](../artifacts/contact_visualization_validation_v002/curvature_plane_pair/comparison.json)，完整规划与两次实际状态在同目录 `plan.json`、`on_trial.json`、`on_smooth_trial.json`。载入核查记录见 [replay_loading_diagnosis.json](../artifacts/contact_visualization_validation_v002/replay_loading_diagnosis.json)。

# 工厂穿行 DiT 演示

本次演示使用完整工业场景的真实碰撞几何、原 HNUTER 执行器和 SE(3) 控制器，在罐体与厂房之间向北穿行，再向西转入设备间通道。**五段局部 DiT 计划全部完成，录像运行零接触、无执行保护中断。**

[播放 / 下载完整 MP4](../results/factory_weave/export/hnuter_factory_weave.mp4) · [录像物理记录](../artifacts/factory_weave_v1/recording_synchronized.json) · [逐帧核验](../artifacts/factory_weave_v1/video_qa.json)

## 实测结果

| 项目 | 无录像验证 | 正式录像 |
| --- | ---: | ---: |
| 仿真时间 | 109.2 s | 108.0 s |
| 真实物理步 | 109,200 | 108,000 |
| 飞行循环墙钟耗时（不含初始化） | 250.14 s | 480.25 s |
| 终点位置误差 | 1.67 cm | 1.67 cm |
| 终点姿态角误差 | 1.12° | 1.12° |
| 实际飞行距离 | 25.37 m | 25.51 m |
| 接触 / 执行保护干预 | 0 / 0 | 0 / 0 |

任务航点中心线长 22 m；模型生成的弯曲轨迹使实际飞行距离更长。正式 MP4 为 **108.1 秒、1920×1200、10 fps、1,081 帧**，覆盖初始化后的感知等待、五段飞行、分段停稳与终点悬停。一个视频秒对应一个仿真秒，不能据此宣称墙钟实时。

## 规划与感知

- 使用冻结权重 `artifacts/static_sdf_pose_pilot_v1/models/sdf/best.pt`，SHA-256 为 `6e0da29b562b433bc1813b2f632cdd1f0a420705b003767ae32ce31e94745192`；本次没有重新训练。
- 每段只给模型当前实测位姿、下一个目标位姿和实际雷达构建的局部距离场历史。每段约 4.4 m、20 秒，停稳后再生成下一段。航点位置来自离线任务设计；离线碰撞网格只用于航点中心线诊断，不进入网络或在线 nvblox 地图。
- 每段将目标方向旋转到局部 +X。世界坐标下保存 nvblox 实测地图，再查询旋转后的 32³ 局部网格；新的四帧历史在当前航点重新采集。原有六个距离场/已知标记/命中通道不变。保留未知标记，未降低观测支持比例、净空或时效阈值。
- 世界地图最多保留 120 秒，传感器距地图初始锚点不超过 28 m；分段时为下一段预留 30 秒，锚点超过 20 m 或时间不足即清空重建。单段局部视图的运行时限仍为 30 秒，网络仅输入约 0.6 秒的四帧历史。这是有界静态地图，不是永久全局 SLAM。
- 起点预设在 `(16.5, 7, 13)` m 的空中位置，设定发生在物理初始化前；录像不包含地面起飞。飞行过程中没有写入机器人位置或姿态、几何修路或专家轨迹回退。偏航随各段从 0 增至 1.5 rad，模型同时输出位置和姿态样条。

这是人工给定任务航点的局部 DiT 演示，尚不代表任意起终点自主全局导航、运动中连续重规划或未观测空间安全认证。此次路线经过调试选择；两次成功不能作为未见场景泛化成功率。

## 四路视频与核验

上方为实际跟随相机和机头相机；下方为实际 MID-360 参数近似雷达回波与 nvblox ESDF 切片。灰黑色网格明确表示未知。蓝线为已获准执行的 DiT 计划，白线为实测轨迹，黄色标记为目标。左上角小图为任务航点及累计实测轨迹。

图像通过真实渲染采集；相机、雷达和地图使用实际物理时间戳。导出阶段不加载工厂碰撞网格，不合成点云或重放机体位姿。两路相机的所有 1,081 帧状态与物理日志逐帧精确一致，感知时间戳因果性和源文件哈希已核验；完整 MP4 解码通过。原始录像、逐帧点云、地图切片及失败记录都保留在 `results/factory_weave/`，不进入 Git。

正式录像显式启用 `--recording-perception-barrier`：发生墙钟卡顿、最新观测龄超过 250 ms 时，保持仿真物理状态不动，等待**当前物理时刻新采集的观测**完成建图，最多等 300 ms；等待失败则明确结束，不能接受旧图或改写扫描时间。原有 400 ms 执行超时、未知标记和净空保护不变。此次触发 532 次同步等待，累计 7.79 秒。该开关仅允许 headless SDF 感知录像，普通闭环运行仍使用原有超时制动。它解决离线录制时墙钟停顿与仿真时钟不一致的问题，**不构成实时控制能力验证**。

## 调试记录

| 原始运行目录 | 结果与处理 |
| --- | --- |
| `preflight` | 最初 31 m 航线的第二段观测支持不足，三次拒绝，零接触；保留失败记录，改在设备更密集的通道布置任务。 |
| `preflight_observed` | 异步启动时经过 0.6 s 仍只有三帧历史，worker 明确报错；修正为等待四个真实历史快照，不补造历史。 |
| `preflight_history_ready` | 前两段完成，第三段约 3.6% 支持低于既有 5% 门槛，拒绝执行。检查发现每段切换清空了此前实测地图。 |
| `preflight_retained` | 改用有界世界地图并重采局部历史后五段完成。第一段有一次观测支持不足拒绝，继续感知后通过。 |
| `recording_final` | 未启用录像同步等待；65.108 s 时因感知超时停止，零接触。接触读回最长阻塞 598 ms，最后观测响应为 621 ms，而该帧建图为 8.3 ms。失败记录完整保留。 |
| `recording_synchronized` | 显式启用离线同步等待；0 次计划尝试被拒绝，五段最终通过，零接触。拒绝与停稳片段完整保留。 |

各次原始报告哈希与紧凑副本见 `artifacts/factory_weave_v1/`。保留地图同时改变了地图世界坐标离散化，不能将这些运行解释为严格单因素消融。代码通过 180 项全量测试，新增结束阶段错误传递和录像同步等待后，25 项相关回归测试通过。

`configs/factory_weave_v1.json` 保留最初失败的 31 m 任务供追溯；正式演示使用 `configs/factory_weave_observed_v1.json`。

## 复现

在 HNUTER_DiT 仓库运行；依赖模拟器仓库及已配置的 `isaaclab-sim:6.0.0-nvblox` 镜像。

```bash
./run_industrial.sh --headless --mode sdf \
  --sdf-checkpoint artifacts/static_sdf_pose_pilot_v1/models/sdf/best.pt \
  --factory-mission configs/factory_weave_observed_v1.json \
  --duration 130 --max-speed .65 --max-acceleration .9 --seed 23 \
  --real-time --profile-loop --record-video --video-fps 10 --demo-perception \
  --recording-perception-barrier \
  --output results/factory_weave/new_recording

python tools/export_sdf_demo.py --run results/factory_weave/new_recording \
  --output results/factory_weave/new_export/hnuter_factory_weave.mp4 \
  --ffmpeg /path/to/ffmpeg

python tools/validate_factory_video.py --run results/factory_weave/new_recording \
  --mp4 results/factory_weave/new_export/hnuter_factory_weave.mp4 --ffmpeg /path/to/ffmpeg \
  --output results/factory_weave/new_export/video_qa.json
```

无录像验证时去掉 `--record-video --video-fps 10 --demo-perception --recording-perception-barrier`。每次使用新的输出目录，避免覆盖先前证据。`tools/plan_factory_route.py` 可重新计算所用任务中心线的离线网格净空，但该结果不替代实际模型轨迹的物理验碰。

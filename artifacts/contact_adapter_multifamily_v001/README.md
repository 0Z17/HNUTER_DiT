# Contact/Surface v003 实验归档

此目录随 2026-10-17 的源码提交保存一份轻量归档：

- [冻结实验协议](protocol.json)：Base + 21 个 adapter、数据预算与训练/评测约定。
- [完整中文科研报告](consolidated_reports/research_report_zh.md)：两场景、三 decoder 的 K1/K8 和消融结果。
- [交付证据](consolidated_reports/delivery_commit.json) 与 [恢复报告证据](consolidated_reports/recovery_v002/report_commit.json)：原始文件哈希、输出覆盖和完成记录。
- [主实验图](reports/) 与 [平滑投影补充图](smooth_projection_supplement/reports/)。

冻结报告和证据保持生成时的内容与 SHA，不修改其中的本机绝对路径。跨机器浏览时，图像可通过上面的相对目录访问；证据记录的路径表示原实验文件位置，不表示所有对应文件均被 Git 跟踪。

训练/测试数据、21 个 adapter 权重、原始预测、逐候选评分、完整压缩统计附录、运行日志及其余实验文件继续保留在原机器，未包含在本提交中。仅克隆此分支即可检查源码和报告；回放、在线推理或完整复现实验还需要获取这些外部文件。

数据位置、训练协议与历史状态说明见 [实验交接](../../docs/CONTACT_ADAPTER_HANDOFF_20261013.md)，界面与执行说明见 [Contact 可视化](../../docs/CONTACT_VISUALIZATION.md)。

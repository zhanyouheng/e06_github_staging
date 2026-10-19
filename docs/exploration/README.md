本目录是 Turn 4 探索交付，原主实验保持在 ../training/，未被替代。

完整报告：../REPORT_turn4.md。验证实验全表：experiment_results.tsv；阶段汇总：stage_results.tsv。
最终方案：selection/final_selection.json；推理运行补充：selection/inference_runtime_v2.json。
唯一新的测试事件：revision/revision_evaluation_event.json（第二次查看测试集）。
权威预测：inference_v2/test/final_predictions.npz，以及 revision/scores.tsv / predictions.tsv。
完整每类、分层和 bootstrap 结果均在 revision/。NA 不是零；条件有效轮次区间不能替代普通 CI。

网络恢复只核验证书并完成交付，没有重启已完成训练、选择或测试。恢复盘点见 validation/recovery_inventory.json。
此前实验采用 class/seed/epoch/block 的原子检查点；最终选择已冻结，训练脚本的 no_test_guard 会阻止回到探索阶段。
本交付完成后的重复命令只运行校验：

```bash
cd /disk1/shengan_group/zyh
source gpu_handoff_pack_20261005/output/exploration/scripts/env.sh
/disk1/shengan_group/zyh/envs/seqscreen_train/bin/python -B gpu_handoff_pack_20261005/output/exploration/scripts/verify_delivery.py
```

模型文件依赖 schema.json 与 env/cpu_versions.json 指定的版本；Torch/CUDA 版本另在每个 GPU history.json 和 runtime_summary.json。
所有数据处理参数只拟合 train；32 个阈值只选于 val。后续扩展必须另立版本和评价计划，不得在本交付上按测试结果调参。
artifacts.tsv / SHA256SUMS / manifest.json / DONE.json 构成交付核验链。无需重新推理即可检查完整性。

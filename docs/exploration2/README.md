Turn 5 完整报告：../REPORT_turn5.md

本目录只含train/val研究，不含本轮测试评价。申请文件不是测试授权。

完整实验表：experiment_results.tsv；排行榜：selection/validation_leaderboard.tsv；模型冻结如有则见selection/final_selection.json。

只校验交付（不训练/推理）：source scripts/env.sh 后运行既有seqscreen_train/bin/python -B scripts/finalize_turn5.py；已有DONE时仅核验全部清单。所有路径及缺失值见schema.json。

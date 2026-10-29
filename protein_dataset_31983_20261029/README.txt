31,983 个蛋白编号的 32 类训练数据集
范围：positive_2019 13,784 + 背景重审 17,497 + 2026 新增 702。不含另外 720 和明确排除的 3 个编号。
proteins.fasta / proteins.tsv：序列及来源。sequence_groups.tsv：相同序列分组；划分训练/测试时同组不可拆开。
training_labels_wide.tsv：ac + 32 类，P=1、N=0、U=空白。training_mask_wide.tsv：P/N 为 1，U 为 0。
labels_32class.tsv：逐格训练标签、历史标签、现审标签、来源和批次。class_order.json 固定列顺序。
17,497 使用现审结果；13,784 保留历史正例 P，其余格使用现审结果，冲突见 label_conflicts.tsv。
702 的其他 31 类沿用历史 v2026 标签，未按 v3.2 重审。
702 中旧规则 A 排除的 471 个 VA 格按用户 2026-10-29 决定恢复为 P；未补审，其余 VA 格不变。逐格记录见 va_label_changes.tsv。
U 是缺失监督，不是 N。初审一致项是暂定一致，不等于人工金标。未生成训练/验证/测试划分。
原始审核结果和 2026-10-20 旧快照均未改动。summary.json 和 validation.json 记录统计与结构验收。

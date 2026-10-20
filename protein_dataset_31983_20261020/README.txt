蛋白审核结果合并快照：31,983 个编号
范围：背景重审 17,497 + positive_2019 13,784 + 后来确认的 702；不含另外 720 和明确排除的 3 个。
proteins.fasta：全部序列；proteins.tsv：编号、来源批次、序列版本及 SHA-256。
labels_32class.tsv：每个编号 32 行。historical_label 是历史标签，current_review_label 是本轮已完成结果；两者分列，未生成统一定稿标签。
current_review_pending=1 表示本轮未完成；not_scheduled 表示该批未纳入本轮。空白不等于 U，也不等于 N。
正例的新审核仍在进行。这里只复制快照，不自动覆盖原有正标签；下游仍需按用户决定合并非正类格并保留冲突记录。
702 的历史标签使用旧规则，当前规则适用性尚未统一确认。历史 P/N/U 不能自动当成当前规则下的金标准。
reviews/：按来源复制的判决、理由及可用状态记录，gzip JSONL；background/positive 的 resolved 分片同时包含相关初审及终审输出。
evidence/：对应注释材料。metadata/：协议、范围、进度和验收快照。未复制账号凭据、运行缓存或网络配置。
所有内容均为实际复制，不是软链接；原审核目录不改动。活跃审核按文件读取时保存，非全局同时刻快照，不会自动同步后续结果。
重复序列保留不同编号，用 sequence_sha256 关联；不要把相同序列拆到训练集和测试集两边。
summary.json/validation.json 为统计与结构验收；这不是完整新版训练标签发布。

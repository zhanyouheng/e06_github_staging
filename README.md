# E06 探索：32 类蛋白质功能多标签分类

本仓库整理 SeqScreen 复现/扩展项目的 E06 探索阶段配置、代码、报告和结果，面向 **32 类蛋白质功能多标签分类**。内容以 Turn 4 的集成与按类选择实验为主，并保留相关基础候选、Turn 5 后续探索及 v2a 训练材料，便于审阅实验设计、最终选择与结果来源。

这是按指定清单整理的研究归档。源文件保持原始内容，目录按文档、配置、脚本和结果重新组织；另附下述数据集审核快照，最终训练标签、模型和预测矩阵仍需另行准备。

## 数据集审核快照（2026-10-20）

[`protein_dataset_31983_20261020/`](protein_dataset_31983_20261020/) 保存本次上传的完整数据集文件夹：524 个文件，共 215,582,951 字节，包含 31,983 个蛋白编号、30,251 条不同序列，以及 32 类标签表（1,023,456 行）、审核记录、注释证据和元数据。

- 序列：[`proteins.fasta`](protein_dataset_31983_20261020/proteins.fasta)；编号与序列信息：[`proteins.tsv`](protein_dataset_31983_20261020/proteins.tsv)。
- 标签：[`labels_32class.tsv`](protein_dataset_31983_20261020/labels_32class.tsv)，历史标签和本轮审核标签分列保存。
- 说明与校验：[`README.txt`](protein_dataset_31983_20261020/README.txt)、[`summary.json`](protein_dataset_31983_20261020/summary.json)、[`validation.json`](protein_dataset_31983_20261020/validation.json) 和 [`SHA256SUMS`](protein_dataset_31983_20261020/SHA256SUMS)。

**此目录是审核结果快照，尚不是最终训练标签发布。** 未完成审核的空白不能直接当成 N 或 U；历史标签与当前标签的合并仍需明确规则。划分训练集和测试集时，应按 `sequence_sha256` 分组，避免相同序列跨集合。此快照与下文 E06 原始实验所用数据的对应关系尚未在本次上传中验证。

## E06 阶段与最终方案

E06 共评估 **133 个配置：5 个按类选择器 + 128 个加权集成网格配置**。计数见 [Turn 4 报告](docs/REPORT_turn4.md)、[E06 配置清单](configs/exploration/E06.json)和[阶段结果表](results/exploration/stage_results.tsv)。5 个按类选择器均未通过分量交叉检查的入选门槛，最终从合资格候选中按验证集 macro AP 选择全局集成。

最终冻结方案为：

```text
E06_mix_098 = 0.2 × E09_ensemble_A2B
            + 0.1 × E05_ensemble_C
            + 0.7 × E04_006_C
```

其中，`E09_ensemble_A2B` 与 `E05_ensemble_C` 分别为 A2B、C 特征上的固定三 seed MLP 均值，`E04_006_C` 为 C 特征上的 ExtraTrees。完整成员关系与配置见 [E06 配方图](configs/e06_recipe_graph.json)。

以下均为 **Turn 4 原始验证集**上的宏平均指标，按原报告保留六位小数：

| 指标 | 数值 |
| --- | ---: |
| macro AP | 0.902357 |
| macro F1（按类阈值） | 0.892310 |
| macro F1（共享阈值） | 0.857141 |

数值已与 [Turn 4 报告](docs/REPORT_turn4.md)、[最终候选摘要](candidates/E06_mix_098/summary.json)和[阶段结果表](results/exploration/stage_results.tsv)交叉核对。**32 个逐类阈值已冻结**，数值及类别顺序见 [final_selection.json](selection/final_selection.json) 的 `thresholds` 与 `class_order`；阈值在验证集上逐类最大化 F1，分数大于等于阈值判阳，并列时取较高阈值。阈值优化影响 F1，不改变 AP。

训练与评价屏蔽 unknown 标签，不将其视为负例。模型参数及数据驱动变换只拟合 train，验证集用于选择，未进行 train+val 重训。具体约定见[探索计划](docs/exploration/PLAN.md)及两份补充计划。

## 结果范围与后续材料

- `results/exploration/` 和 `selection/` 保留原探索的全阶段结果表与选择记录；`candidates/` 仅收录清单指定的 15 个候选摘要、逐类指标和完成记录。因此完整结果表中的候选多于本仓库收录的候选目录。
- Turn 5 后续探索的封存结论为“无可超越 E06_mix_098”，该轮没有新的测试评价，见 [STATE.md](docs/exploration2/STATE.md) 和[封存记录](configs/exploration2_e06_seal.json)。
- `results/v2a/` 包含在 v2a 标签口径下的训练结果及旧 E06 存档模型的重评指标。其 `E06_archived_metrics.json` 与上面的 Turn 4 验证指标属于不同评价口径，解读时应同时查阅 [v2a 训练报告](docs/REPORT_training_v2a.md)，不能直接混用。

## 目录结构

原 E06 研究归档为 171 个文件、约 1.38 MB，目录树如下（不含上面新增的 524 个数据集文件；标注每部分的用途与文件数）：

```text
.
├── README.md                        # 本文件：项目背景、E06 配方与指标、免责说明
├── .gitignore                       # 排除 npz/npy/pkl/pt 等大二进制
│
├── docs/                            # 叙述性文档（12）：讲"为什么"
│   ├── REPORT_turn4.md              # 探索阶段总报告：E00–E09、最终选择与修订评价
│   ├── REPORT_training_v2a.md       # v2a 重训报告：M00/M06、E06_archived 对照
│   ├── CHANGELOG_v4.md              # v4 标签变更记录
│   ├── exploration/                 # 探索阶段 PLAN.md + 两份 addendum + README + schema.json
│   └── exploration2/                # 第二轮探索的 PLAN / addendum / README / STATE
│
├── configs/                         # 全部实验配置（12）：讲"做什么"
│   ├── e06_recipe_graph.json        # E06_mix_098 配方图（最终方案的精确定义）
│   ├── exploration2_e06_seal.json   # exploration2「无可超越 E06_mix_098」封存结论
│   └── exploration/                 # E01–E09 各阶段配置（含 E06.json 全部 133 个配置）
│
├── scripts/                         # 全部代码（69）：讲"怎么做"
│   ├── exploration/                 # 探索阶段训练/推理/特征/选择脚本（23）
│   ├── exploration2_turn5/          # Turn 5 融合与审计脚本（29）
│   └── v2a_training/                # v2a 重训脚本（17）
│
├── selection/                       # 冻结的选择记录（7）
│   ├── final_selection.json         # 最终方案 + 32 个逐类冻结阈值
│   ├── validation_leaderboard.tsv   # 335 个候选完整排序
│   ├── crossfit_selector_checks.json    # 5 个按类选择器的交叉检查（被否决依据）
│   └── family_winners.json 等       # 家族胜者、MLP 结构记录
│
├── results/                         # 指标表（24）：讲"结果"
│   ├── exploration/                 # stage_results / experiment_results / data_flow / manifest 等（6）
│   └── v2a/                         # v2a 结果 TSV + E06_archived_metrics.json（18）
│
└── candidates/                      # 15 个关键候选 × 3 件证据（45）：讲"证据"
    ├── E06_mix_098/                 # 最终方案：summary.json + per_class.tsv + DONE.json
    ├── E06_class_*/                 # 5 个被交叉检查否决的按类选择器
    └── E04_006_C, E05_*, E09_*/     # 最终方案的 9 个组成成员
```

设计逻辑：docs 讲"为什么"，configs 讲"做什么"，scripts 讲"怎么做"，selection / results / candidates 讲"结果和证据"。所有模型权重与特征矩阵（npz/npy/pkl/pt 等）均未纳入，本仓库保留文本级证据链，可审阅完整实验依据，但不能直接端到端复现（见下文「未包含内容」）。

主要来源为 `gpu_handoff_pack_20261005/output/` 下的 `exploration/`、`exploration2/`、`v2/preregistration/` 和 `v2/training/`。Turn 5 脚本与封存记录来自 `source_review_pack_20261009/`，变更记录来自 `gpu_handoff_pack_20261008/CHANGELOG_v4.md`。

报告、脚本、manifest 和 DONE 记录保持原始内容，其中的路径、哈希和完整性声明对应原交付包。部分原始相对链接或运行路径在本仓库重新组织后需要适配；原清单也可能引用未收录的文件，不能据此认为本仓库包含完整交付包或可直接完成端到端复现。

## 未包含内容

- 大型二进制模型、权重、特征/embedding 矩阵、预测矩阵及其他 `*.npz`、`*.npy`、`*.pkl`、`*.pt`、`*.bin`、`*.ckpt` 文件未纳入；`.gitignore` 同时排除这些格式、`__pycache__/` 和 `.DS_Store`。
- 除上述数据集审核快照外，原实验的完整原始数据、运行环境、缓存、日志及清单之外的产物未纳入。`selection/` 的 NPZ 文件、候选模型文件和验证预测矩阵均不包含在本仓库中。完整产物说明见 [Turn 4 报告](docs/REPORT_turn4.md)与 [v2a 训练报告](docs/REPORT_training_v2a.md)。
- Turn 5 文档引用的 `REPORT_turn5.md` 及其完整结果、审计产物未在本次复制清单中；本仓库收录了指定的计划、状态、脚本和封存记录。
- 原 E06 研究归档的单文件大小检查上限为 **20 MB（20,000,000 字节）**，原清单中因超过该上限而排除的文件为零。新增数据集保留全部文件，其中 `labels_32class.tsv` 为 89,675,431 字节。

## 重要免责

**验证集被反复用于模型、早停、集成和阈值选择，存在选择偏差。** 验证指标不能视为独立泛化证据；按类选择器的分量交叉检查也不能消除此前模型选择和早停产生的影响。

**测试集非首次盲测。** 测试集已在 Turn 3 被查看，Turn 4 是冻结方案后的第二次查看，属于探索后的修订评价，不能称为首次盲测或独立预注册验证。原报告记录所有选择仅使用 train/val，且本轮测试评价后未继续训练、调整阈值或替换候选。完整实验限制与评价时间线见 [Turn 4 报告](docs/REPORT_turn4.md)。

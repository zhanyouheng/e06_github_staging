# E06 探索：32 类蛋白质功能多标签分类

本仓库整理 SeqScreen 复现/扩展项目的 E06 探索阶段配置、代码、报告和结果，面向 **32 类蛋白质功能多标签分类**。内容以 Turn 4 的集成与按类选择实验为主，并保留相关基础候选、Turn 5 后续探索及 v2a 训练材料，便于审阅实验设计、最终选择与结果来源。

这是按指定清单整理的研究归档。源文件保持原始内容，目录按文档、配置、脚本和结果重新组织；完整训练数据、模型和预测矩阵需另行准备。

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

```text
.
├── README.md
├── .gitignore
├── docs/
│   ├── REPORT_turn4.md              # 原始探索与修订评价报告
│   ├── REPORT_training_v2a.md       # v2a 训练报告
│   ├── CHANGELOG_v4.md              # v4 数据变更记录
│   ├── exploration/                # 计划、补充计划、README、schema
│   └── exploration2/               # Turn 5 计划、README 与状态
├── configs/
│   ├── exploration/                # 各阶段 JSON 配置（含 E06.json）
│   ├── e06_recipe_graph.json        # 最终 E06 成员与配方
│   └── exploration2_e06_seal.json   # 后续探索封存记录
├── scripts/
│   ├── exploration/                # 原探索 Python 脚本
│   ├── v2a_training/               # v2a 训练 Python 脚本
│   └── exploration2_turn5/          # 来源审阅包中的 Turn 5 脚本
├── selection/                      # JSON/TSV 选择记录与冻结阈值
├── results/
│   ├── exploration/                # 阶段/实验结果、数据流、运行与清单记录
│   └── v2a/                        # TSV 结果及 E06_archived_metrics.json
└── candidates/
    └── <原候选名>/                  # summary.json、per_class.tsv、DONE.json
```

主要来源为 `gpu_handoff_pack_20261005/output/` 下的 `exploration/`、`exploration2/`、`v2/preregistration/` 和 `v2/training/`。Turn 5 脚本与封存记录来自 `source_review_pack_20261009/`，变更记录来自 `gpu_handoff_pack_20261008/CHANGELOG_v4.md`。

报告、脚本、manifest 和 DONE 记录保持原始内容，其中的路径、哈希和完整性声明对应原交付包。部分原始相对链接或运行路径在本仓库重新组织后需要适配；原清单也可能引用未收录的文件，不能据此认为本仓库包含完整交付包或可直接完成端到端复现。

## 未包含内容

- 大型二进制模型、权重、特征/embedding 矩阵、预测矩阵及其他 `*.npz`、`*.npy`、`*.pkl`、`*.pt`、`*.bin`、`*.ckpt` 文件未纳入；`.gitignore` 同时排除这些格式、`__pycache__/` 和 `.DS_Store`。
- 完整原始数据、运行环境、缓存、日志及清单之外的产物未纳入。`selection/` 的 NPZ 文件、候选模型文件和验证预测矩阵均不包含在本仓库中。完整产物说明见 [Turn 4 报告](docs/REPORT_turn4.md)与 [v2a 训练报告](docs/REPORT_training_v2a.md)。
- Turn 5 文档引用的 `REPORT_turn5.md` 及其完整结果、审计产物未在本次复制清单中；本仓库收录了指定的计划、状态、脚本和封存记录。
- 单文件大小检查上限为 **20 MB（20,000,000 字节）**。本次指定清单中的所有文件均低于该上限，**因超过 20 MB 而排除的文件：无**。

## 重要免责

**验证集被反复用于模型、早停、集成和阈值选择，存在选择偏差。** 验证指标不能视为独立泛化证据；按类选择器的分量交叉检查也不能消除此前模型选择和早停产生的影响。

**测试集非首次盲测。** 测试集已在 Turn 3 被查看，Turn 4 是冻结方案后的第二次查看，属于探索后的修订评价，不能称为首次盲测或独立预注册验证。原报告记录所有选择仅使用 train/val，且本轮测试评价后未继续训练、调整阈值或替换候选。完整实验限制与评价时间线见 [Turn 4 报告](docs/REPORT_turn4.md)。

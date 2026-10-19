# v4 变更清单（2026-10-06）

基线：v3（`step2/curation/final_labels_v3.tsv` + `step3/background/` 定稿）。v3 全部保留不动。

## 变更 1：virulence_activity 兜底规则 A（用户拍板）

**规则**：VA 是严格残留桶——一个蛋白只要任何具体功能类成立，就不挂 VA 标签。
v3 该规则未全局执行（终审口径 5 只覆盖了矛盾清单批次）。

**改动**：

| 文件 | 改动 | 行数 |
|---|---|---|
| `final_labels_v4.tsv` | 41 个"VA+具体类"双标签蛋白的 VA 行：train_candidate → excluded，user_review 加 `remove_va_rule_a@v4` | 41 |
| `bg_removed_confirmed_risk_v4.tsv` | 679 个"VA+具体类"背景转入蛋白的 confirmed_classes 摘除 VA | 679 |
| `background_labels_final_v4.tsv` | 上述 679 蛋白的 (ac, virulence_activity) 格：pos → unknown（**mask，不是 neg**） | 679 |

**语义注意**：双标签蛋白的 VA 格一律 mask 而非判 neg——它们确实有毒力活性，
规则 A 只是标签归属约定，不能训练模型"不要预测 VA"。excluded 行照旧只是审核记录，
excluded ≠ 负例。

**规则 A 后的 VA 类**：25（v3 纯 VA）+ 457（背景纯 VA）= **482 个蛋白**
（2019 主宇宙 481 个，1 个无 2019 序列）。其中背景来源 457 个里 423 个来自病毒，
另有 71 个在其他类仍有 unknown 格，可能随后续补全退出 VA——来源构成变化需在
论文方法/局限中声明。

**对 GPU 侧的影响**：VA 类训练正例从"65 蛋白（含 41 双标签）+ 1,136 背景转入
（含 679 双标签）"变为 481 个纯 VA 蛋白；双标签蛋白的 VA 格从 pos 变 mask。
此变更与项 4（positive_labels_full.tsv）、项 3（同序列冲突裁决）一起进下一版
交接包，配套 v2 划分统一生效，不单独要求重训。

# 2026-10-07：positive_labels_full.tsv 建表（项 4 完成）

**产物**：`step3/v4/positive_labels_full.tsv`（ac, class, label, source），
宇宙 = final 18,144 蛋白 × 32 类 = **580,608 格**，CRLF。
脚本 `build_positive_labels_full.py` 可重跑（临时文件 + rename）。

**建表口径**（优先级从上到下）：

| 优先级 | 条件 | label | source | 格数 |
|---|---|---|---|---|
| 1 | train_candidate | pos | `train_candidate` | 23,761 |
| 2 | excluded 且 user_review 含 `remove_va` 标记 | unknown | `excluded_va_mask` | 252 |
| 3 | 其余 excluded（审核否决） | neg | `excluded` | 5,041 |
| 4 | 盲筛 not_fit | neg | `blindcheck_not_fit` | 422,910 |
| 5 | 盲筛 uncertain | unknown | `blindcheck_uncertain` | 5,982 |
| 6 | 盲筛 fit 但未进终稿 | unknown | `blindcheck_fit_not_final` | 10,334 |
| 7 | 蛋白无盲筛（3,632 个 final-only） | unknown | `no_blindcheck` | 112,328 |

合计：pos 23,761 / neg 427,951 / unknown 128,896。

**口径说明（两条需要明文化的决定）**：

1. **508 对盲筛冲突（train 优先判 pos）**：264 对盲筛 not_fit + 244 对盲筛 uncertain
   进了终稿。核查确认 508 对**全部**是 fullcheck 里 `is_known_positive=1` 的已知正例
   （其中 71 对还有 user_review `keep@1/keep@2` 人工保留记录），即"已知正例/人工复核
   翻盘盲筛"，判 pos 有据。清单落盘 `positive_labels_conflicts_508.tsv`（含
   blind_verdict / is_known_positive / user_review / step2_status），供 GPU 侧溯源。
2. **excluded 不一律当 neg**：252 对 user_review 带 `remove_va@3`/`remove_va_rule_a@v4`
   标记的全是 virulence_activity 类，蛋白本身确有其他具体类正例（已逐行核实 252/252），
   与变更 1 的 mask 语义一致 → unknown；其余 5,041 对（5,020 条 step2 审核否决 +
   21 条 user_review `drop@1/drop@2` 人工丢弃）是真正意义上的否决 → neg。

**注意**：neg 大头（42.3 万）来自盲筛 not_fit，盲筛本身是双引擎初筛、未经 astra 终审，
负例置信度低于 pos；`blindcheck_fit_not_final` 的 10,334 格是"初筛说像、但没进终稿"，
绝不能当 neg 用。

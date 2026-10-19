# v2a 保守范围训练报告

生成时间（UTC）：2026-10-07T20:43:14.756208+00:00。本轮仅开发集训练/评价；702全排除，未打开测试标签、分数、指标或划分载荷。

**结论：最终保留 `M06`。选择状态：`eligible_improvement`。V1 为 not_applicable；没有执行 unknown=neg 的非空处理组。**

## 1. 预登记与启动门

正式附录见 `preregistration/ADDENDUM_v2a.md`、`addendum.json` 和其DONE，先于任何新模型分数。原Turn12封存保持原样。阳性表17,442个唯一ID不是完整dev：本轮11,014个dev阳性侧ID均处于唯一32格块，14,013个dev背景ID在背景v4唯一；合计25,027。与702无交集，dev重复格0，review_2026/augmented正例污染0。

预检结果 `preflight/result.json`；A/A2冻结文件校验和全部一致，按冻结2019注释重新生成的最大绝对差均1.486e−8；B重建差3.280e−8，完整原始embedding分片SHA、ID与64位序列哈希逐行一致。原始注释seq_sha256实际是16位前缀，首轮审计错按64位比较导致失败；按字段实际格式修正后重新全检通过，原失败与修复依据保存在audit/preflight_attempt_01，不曾为此改变输入。

历史文件lstat检查 80822 项未改变；dev冻结特征另做SHA核验。旧组件划分不重切、旧ID不合并。纯dev残余同序列P/N为 53 个序列×类别格、44 个序列组，按授权保留；这不是把含702的57格原总数机械照搬。

## 2. 标签阶段账与变体

| stage | split | proteins | cells | P | N | U |
| --- | --- | --- | --- | --- | --- | --- |
| frozen_v1_development | all | 25027 | 800864 | 20549 | 445336 | 334979 |
| frozen_v1_development | train | 18770 | 600640 | 15388 | 333780 | 251472 |
| frozen_v1_development | val | 6257 | 200224 | 5161 | 111556 | 83507 |
| v2026_proposed_base | all | 25027 | 800864 | 19959 | 768221 | 12684 |
| v2026_proposed_base | train | 18770 | 600640 | 14943 | 575880 | 9817 |
| v2026_proposed_base | val | 6257 | 200224 | 5016 | 192341 | 2867 |
| after_sameseq | all | 25027 | 800864 | 20045 | 768137 | 12682 |
| after_sameseq | train | 18770 | 600640 | 14998 | 575828 | 9814 |
| after_sameseq | val | 6257 | 200224 | 5047 | 192309 | 2868 |
| ruleA_v2a | all | 25027 | 800864 | 20027 | 768137 | 12700 |
| ruleA_v2a | train | 18770 | 600640 | 14982 | 575828 | 9830 |
| ruleA_v2a | val | 6257 | 200224 | 5045 | 192309 | 2870 |

v3→v4：阳性侧33格、背景557格VA降级，共590个P→U；新增参考N来自源端已给定来源，非GPU把U自行改N。同序列明确裁决在旧dev触及169格、实际改变97格；随后18个VA格按用户批准规则A优先mask（train16、val2）。最终dev unknown为12,700；16,037是此前含702的base口径，不是本轮数。逐格流转在data/cell_changes.tsv，规则A前状态保留，未重审核标签。

V0用最终参考P/N、U全程mask。训练target、known_mask和权重另存data/training_loss_contract.npz；U的存储占位0具有严格零损失权重，绝非伪负例。V1允许的no_blindcheck格为0，M00–M06全部V1登记not_applicable，不重复拟合、不作V1−V0收益推断。审计返回件未进入监督。

## 3. 执行方法、独立性与资源

M00=0.2×A2B MLP三seed平均+0.1×C MLP三seed平均+0.7×C ExtraTrees。M01为A2B logistic，四个共享C由train分量OOF选；M02只增加当前annotation score独热块；M03只把blindcheck_not_fit负格损失降到0.25；M04为固定三seed[512,128] MLP。M05使用17个去重全局混合点；M06使用同网格逐类选择并固定50%向全局规则收缩。未增加额外方法/权重网格，所有候选与阴性结果均保留。

五折train按分量代表hash；每折内部独立命名空间hash留约10%分量做早停，最多120epoch、每1/5轮检查、patience6、改善>1e−6。然后在完整外层训练侧按最佳epoch重拟合。最终epoch取五折中位数，再在全train重拟合；所有折的词表/剪枝/均值/标准差只fit实际训练侧。M00是配方重拟合，不是复制旧E06训练过程。精确epoch表在results/neural_epochs.tsv。
五个内部早停集分别只有30/29/31/31/30个AP可评分类；它们由相同标签固定，所有候选共用，不为补齐而重新分折。稀有类不能在每个内部早停集直接影响epoch选择，这是限制，不把其NA当0。外层训练、外层留出和完整val均覆盖32个可评分类；每折支持见fold_support.tsv。

MLP为FP32，TF32关闭、确定性算法、AdamW lr0.001/wd0.0001/batch1024/clip5；alpha=min(10,sqrt(N/P))按本拟合侧参考已知格计算，损失逐类有效权重质量归一。树为每类256棵、max_features0.2、min_leaf1。logistic lbfgs/tol1e−6，3000不收敛才重试10000；警告见audit/convergence_warnings.tsv。

GPU2/3执行，避让GPU1的负载及GPU0驻留进程。CPU类任务从32提升到80并发；已完成DONE验证后保留，未完成同配置重跑。此资源调整、首次GPU脚本语法修复均在audit登记，无候选/seed/标签改变。每阶段临时文件写完验证后原子rename；checkpoint保存优化器与RNG，可同命令续跑。日志/PID见logs；精确环境版本和mamba包build见env。

## 4. 全候选验证排行榜与资格门

| candidate | full_macro_AP | crossfit_macro_AP | crossfit_delta_vs_M00 | eligible | primitive_cost |
| --- | --- | --- | --- | --- | --- |
| M00 | 0.718913 | 0.718913 | 0.000000 | False | 7 |
| M01 | 0.650201 | 0.650201 | -0.068712 | False | 1 |
| M02 | 0.721297 | 0.721297 | 0.002384 | True | 7 |
| M03 | 0.718986 | 0.718986 | 0.000073 | False | 7 |
| M04 | 0.658763 | 0.658763 | -0.060149 | False | 3 |
| M05 | 0.724161 | 0.721900 | 0.002987 | True | 14 |
| M06 | 0.739106 | 0.724921 | 0.006008 | True | 25 |

M01–M04仅通过train内部选择；M05/M06每个val留出折的权重/逐类规则只用另外四折选择。还对“资格过滤后选家族”的完整规则做嵌套分量交叉检查：资格判定也在外层另外四折内部重算，不先用全val挑家族再检查其留出成绩。

完整家族选择规则 pooled crossfit AP=0.724921，相对M00差=+0.006008，+0.002门结果=True。按单家族资格筛选后的全val优先方案为M06，最终为M06。+0.002只是开发资格门，不是统计显著性或新留出证明。

M01四个C的OOF记录见selection/M01_train_OOF_C.json；本次C=10被选中（OOF AP=0.613016），没有扩大至未登记的C。M01和M04单独明显弱于M00，M03单独的微小差值不达门槛；M02、M05、M06通过预定门。M06允许使用单支未过门的基成员，因为其整个逐类选择/收缩规则另行通过资格检查，并非把单支改名放行。

## 5. 同一v2a验证标签上的数字体系

| candidate | macro_AP | macro_F1 | crossfit_macro_AP | crossfit_macro_F1 | micro_AP | micro_precision | micro_recall | micro_F1 | any_label_FPR | K_AP | K_F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E06_archived | 0.534585 | 0.397357 | 0.534585 | 0.397357 | 0.701355 | 0.267781 | 0.923885 | 0.415215 | 0.020642 | 32 | 32 |
| M00 | 0.718913 | 0.738066 | 0.718913 | 0.574257 | 0.881409 | 0.832496 | 0.854113 | 0.843166 | 0.032765 | 32 | 32 |
| M06 | 0.739106 | 0.755553 | 0.724921 | 0.597540 | 0.888449 | 0.859325 | 0.858474 | 0.858899 | 0.027195 | 32 | 32 |

旧E06仅重评已存验证分数与原冻结逐类阈值，未重新选阈值。新模型阈值按各类已知val格F1最大化，>=判阳、并列高阈值；交叉F1用另外四折重新选择整个预测规则及阈值。旧E06的crossfit F1字段仅重复原冻结阈值分数（历史已依赖此val），不能解释为新交叉估计。AP与阈值无关。

**阈值稳定性是显著限制**：最终完整val F1=0.755553，交叉F1=0.597540，差0.158013。M00也有明显差距，因此不能把完整val最优阈值F1当稳定泛化性能。逐类固定预测算术对照见results/crossfit_per_class.tsv；不因发现差距而重新调阈值。完整val AP提升也大于交叉增量，保留这项选择乐观性。

改善并非全面：最终全负背景any-label FPR=0.027195，低于M00的0.032765，但高于旧E06的0.020642。已知格micro recall从旧E06的0.923885变为0.858474；较少报警伴随检出损失，不能说所有性能同时提升。

| candidate | TP | FP | unknown_cells | unknown_positive | unknown_alarm_rate | unknown_share_all_positive | all_positive | all_output_precision_reference_bound_low | all_output_precision_reference_bound_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E06_archived | 4661.000000 | 12745.000000 | 2870.000000 | 1818.000000 | 0.633449 | 0.094569 | 19224.000000 | 0.242457 | 0.337027 |
| M00 | 4309.000000 | 867.000000 | 2870.000000 | 318.000000 | 0.110801 | 0.057881 | 5494.000000 | 0.784310 | 0.842191 |
| M06 | 4331.000000 | 709.000000 | 2870.000000 | 257.000000 | 0.089547 | 0.048518 | 5297.000000 | 0.817633 | 0.866151 |

三个unknown口径依次为U格数、U内报警比例、U报警占全部报警比例。全输出precision只给“现有已知参考正确、U可任意”的代数识别上下界：[TP/所有阳性,(TP+U阳性)/所有阳性]，不是抽样置信区间、更不是真实部署precision。参考N覆盖扩大、重训压低分数与真实审核确认是不同事件；不能把unknown报警下降说成所有误报被消灭。旧1600格概率盲审只代表旧E06报警框，不能外推为本轮新模型完整输出precision。加权sigmoid或其混合分数也不是已校准的生物学概率。

| frame | U_cells | U_positive | all_positive | U_alarm_rate | U_share_all_positive | known_TP | known_FP |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E06_old_reference | 83507 | 14243 | 19224 | 0.170561 | 0.740897 | 4713 | 268 |
| E06_v2a_reference | 2870 | 1818 | 19224 | 0.633449 | 0.094569 | 4661 | 12745 |
| M00_v2a_reference | 2870 | 318 | 5494 | 0.110801 | 0.057881 | 4309 | 867 |
| selected_v2a_reference | 2870 | 257 | 5297 | 0.089547 | 0.048518 | 4331 | 709 |

该表前两行使用完全相同的E06分数和旧阈值，其差别只来自参考标签覆盖变化；旧口径U阳性14,243和TP4,713已重新对账。其余两行才加入重训/方法变化。E06→M00还同时改变监督与内部早停/损失归一协议，不能把全部差值单独归因于新增N；M00→最终是在共同新协议下的方法比较。旧U报警转入新P/N/U的逐项数量见archived_E06_old_U_positive_transitions.tsv。

每类P/N/U、正例分量数、AP、precision/recall/F1/FPR及阈值见results/per_class.tsv；32类全保留。全负背景必须32格全N且origin=background_sample，含U不入分母。全验证集、共同可用注释子集、长度四档、annotation score、来源/分类群/数据库、注释状态/fragment/映射标记见results/strata.tsv，各层给蛋白/分量数、P/N/U与固定可评分类数。source_status保持旧冻结来源层名称，不冒充新参考state。正例证据来源的召回在evidence_tiers.tsv；不为方便造一个未登记的来源限定AP。annotation_OOV.tsv区分提取状态、原始无注释与剪枝后零向量，并报告OOV；没有因此删样本。

## 6. 配对分量bootstrap及敏感性

| comparison | metric | point_delta | CI_status | main_CI_low | main_CI_high | valid_rounds | valid_fraction | conditional_CI_low | conditional_CI_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M06 - M00 | macro_AP | 0.020194 | NA_any_invalid_round | NA | NA | 1867 | 0.933500 | 0.011097 | 0.031276 |
| M06 - M00 | macro_F1 | 0.017487 | NA_any_invalid_round | NA | NA | 1867 | 0.933500 | -0.006917 | 0.035455 |
| M00 - E06_archived | macro_AP | 0.184328 | NA_any_invalid_round | NA | NA | 1867 | 0.933500 | 0.134079 | 0.210585 |
| M00 - E06_archived | macro_F1 | 0.340710 | NA_any_invalid_round | NA | NA | 1867 | 0.933500 | 0.275352 | 0.382279 |
| M06 - E06_archived | macro_AP | 0.204521 | NA_any_invalid_round | NA | NA | 1867 | 0.933500 | 0.152884 | 0.230353 |
| M06 - E06_archived | macro_F1 | 0.358196 | NA_any_invalid_round | NA | NA | 1867 | 0.933500 | 0.294662 | 0.392358 |

2000轮、seed20261012，同轮共享分量有放回抽样，重复分量按次数计权，原macro类别集合固定。任一无效轮次主CI即NA；条件分位区间仅作补充，不当有效主CI。涉及类别及次数见bootstrap/invalid_class_counts.tsv，完整索引sampling_indices.npy已封存。即使全部轮次有效，区间也只条件于这套开发数据、模型/阈值/选择流程，不能消除历轮验证选择偏差。V1政策差值not_applicable。

按批准18格规则A前标签重评固定分数/阈值，以及预登记去VA的31类描述性敏感性，见results/sensitivity.tsv。未用它们挑模型、阈值或修改主32类目标；不把删除难类后的宏指标当改善。

## 7. 失败、跳过与独立验收

模型失败文件：无；未完成家族见results/failed_families.tsv（空表表示全部基础家族完成）。702及其embedding全跳过；V1七家族全not_applicable。首轮哈希格式误判、GPU启动语法错误、CPU并发调整均有原记录与修复，不隐去失败。没有因验证分数差而删除候选。

独立数值核验：AP对sklearn、阈值穷举、重复分量权重、unknown分数不影响已知格指标、留出标签不改变选择规则均通过。所有21个完整训练神经模型重新从磁盘加载，对6,257条val重算逐位一致；完整CPU模型独立加载重算一致。最终所有候选AP/F1与sklearn误差<1e−12，10个bootstrap轮抽查一致。数据预处理11套独立重新核验通过。收敛/其他警告共0条，记录与处置在audit。

## 8. 不能宣称

- 不能宣称unknown=neg政策已有效：V1没有非空处理组。
- 不能把旧标签下AP与本轮AP直接相减当模型提升；唯一历史比较是同一新mask下的旧存档分数重评。
- 不能声称有新测试证据、部署性能或与原文严格同口径的超越；本轮没有新测试事件。
- 不能把交叉资格门、条件bootstrap区间或反复使用的val当作从未查看过的独立验证。
- 不能把审核来源/模型证据同源、弱负监督、残余P/N矛盾或规则A结构性mask自动“解决”；它们是版本化已知局限。
- 不能把annotation score当2019元数据；M02明确使用当前2026分数。

## 9. 产物与复算

输入契约和逐格变化=data/；启动证据=preflight/；逐折预处理=features/；基础模型与checkpoint=models/；全部候选=candidates/；连续分数/二值输出/阈值=predictions/；OOF选择及最终方案=selection/；完整数表=results/；配对抽样=bootstrap/；复算核验=audit/；日志/PID=logs/；最终只读文件=freeze/manifest.json。全文件清单、SHA与行数/条数/形状见artifacts.tsv及manifest.json，SHA256SUMS的路径统一相对output/。

从工作目录执行（全部已具备验证后跳过的检查点；不应重跑历史脚本）：

```bash
PYTHONDONTWRITEBYTECODE=1 envs/seqscreen_train/bin/python -B gpu_handoff_pack_20261005/output/v2/training/scripts/preflight.py
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 envs/seqscreen_train/bin/python -B gpu_handoff_pack_20261005/output/v2/training/scripts/prepare_features.py
PYTHONDONTWRITEBYTECODE=1 envs/seqscreen_train/bin/python -B gpu_handoff_pack_20261005/output/v2/training/scripts/launch.py
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 envs/seqscreen_train/bin/python -B gpu_handoff_pack_20261005/output/v2/training/scripts/select_evaluate.py
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 envs/seqscreen_train/bin/python -B gpu_handoff_pack_20261005/output/v2/training/scripts/bootstrap.py
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 envs/seqscreen_train/bin/python -B gpu_handoff_pack_20261005/output/v2/training/scripts/audit_cpu.py
```

GPU独立重算命令与固定设备见audit_gpu.log、env/gpu_worker_*.json；需先确认GPU空闲。封存后任何新改动必须另存版本。本轮到此停止。

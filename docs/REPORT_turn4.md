Turn 4 探索与修订评价报告

交付生成时间（UTC）：2026-10-05T19:38:30.182003+00:00。按 PLAN.md 及两份 addendum 完成 E00–E09。最终方案 E06_mix_098 在完整测试集上的 macro AP=0.886441、macro F1=0.783180，全负背景 any-label FPR=1.3509%。相对存档 C_weighted 的 Δmacro AP=+0.053798。这是用户授权的第二次查看测试集，属于探索后的修订评价；主实验 A/B/C 的既有结果和结论完整保留。

所有选择只使用 train/val；测试阶段之后没有训练、超参/阈值调整、候选替换或新的模型选择。该测试集已在 Turn 3 被查看，因此本结果不能称为首次盲测或独立预注册验证。验证集被反复用于模型、早停与集成探索，验证分数存在选择偏差。

网络恢复盘点：6,141 份 DONE 证书及 24,136 个被引用文件通过 SHA256 校验；335 条候选、20 个 GPU 配置全部完成，0 个中断配置。24 个已登记 PID 均已退出。磁盘已经包含最终选择、唯一修订评价和 bootstrap，因此本次续跑只完成证书核验和交付整理，没有重复测试事件。记录：exploration/validation/recovery_inventory.json、recovery_stages.tsv、recovery_certificates.tsv。

数据流转及不变项：总宇宙 31,284 个蛋白、1,001,088 个标签格，P/N/U=25,719/556,200/419,169。本轮没有额外排除蛋白，没有改变角色、标签、划分或序列。unknown 在训练损失和评价中均屏蔽，未生成伪标签。

| 集合 | 蛋白 | 分量 | P | N | U | 用途 |
| --- | --- | --- | --- | --- | --- | --- |
| train | 18770 | 4181 | 15388 | 333780 | 251472 | fit all data-driven transforms and model parameters |
| val | 6257 | 1398 | 5161 | 111556 | 83507 | selection only |
| test | 6257 | 1396 | 5170 | 110864 | 84190 | one frozen revised evaluation only |

开发输入包仅含 25,027 个 train+val 蛋白。原始文件、五个冻结文件、embedding 主产物和主实验训练产物共 2,733 个历史文件的哈希全部保持一致；主实验文件与冻结输入只读。五项既有问题保持原样：fragment 913、accession 映射差异、重复映射、同序列 P/N 不一致、缺源端 manifest。本轮不重新审核或修订其生物学标签。

验证基线（沿用主实验共享阈值）：

| 路线 | val macro AP | val macro F1 |
| --- | --- | --- |
| A_weighted | 0.778935 | 0.708407 |
| A_unweighted | 0.801183 | 0.743922 |
| B_weighted | 0.570745 | 0.551131 |
| B_unweighted | 0.571164 | 0.544325 |
| C_weighted | 0.847506 | 0.808101 |
| C_unweighted | 0.845270 | 0.795969 |

实验覆盖：E01 共同 C 的 30 个配置；E02 注释/文本的 60 个配置；E03 Nyström 的 18 个配置；E04 ExtraTrees 的 12 个配置；E05 12 个单 seed 拟合加 2 个固定三 seed 平均；E07 分量再加权 18 个配置；E08 归一化变体 40 个配置；E09 8 个单 seed 拟合加 2 个固定三 seed 平均；E06 为 5 个按类选择器加 128 个全局集成。合计 335 条候选记录，其中 20 个单 seed 仅诊断、5 个选择器被交叉检查排除，最终 310 个候选符合选择资格。所有逐项配置、seed、验证指标和状态列于 exploration/experiment_results.tsv，完整排序与资格列于 selection/validation_leaderboard.tsv。

| 阶段 | 候选数 | 合资格最佳候选 | val AP | val F1（按类阈值） | val F1（共享阈值） |
| --- | --- | --- | --- | --- | --- |
| E01 | 30 | E01_028_C | 0.848197 | 0.844607 | 0.809110 |
| E02 | 60 | E02_041_A5B | 0.848985 | 0.845876 | 0.810878 |
| E03 | 18 | E03_004_AN1 | 0.848996 | 0.836776 | 0.789477 |
| E04 | 12 | E04_006_C | 0.853273 | 0.855509 | 0.801084 |
| E05 | 14 | E05_ensemble_C | 0.860661 | 0.849214 | 0.805718 |
| E06 | 133 | E06_mix_098 | 0.902357 | 0.892310 | 0.857141 |
| E07 | 18 | E07_013_C | 0.847881 | 0.843396 | 0.809496 |
| E08 | 40 | E08_030_Cstd | 0.845739 | 0.843334 | 0.796431 |
| E09 | 10 | E09_ensemble_A2B | 0.862530 | 0.846433 | 0.801968 |

E00 发现 B_weighted 的 train AP=0.975791、val AP=0.570745，说明读出在已知训练单元上拟合很好但跨分量泛化差距较大；不能据此把原因归于线性欠拟合。B 在 6 个验证类别上胜过 A，按类、长度、注释状态及训练支持的诊断已保存。历史 GO/keyword/function 与标签构建可能共享证据，来源及注释质量也可能影响关联；这些现象不构成因果证据。没有根据测试短板再增加实验。

E01 的最佳 C 路线 AP=0.848197，较 C_weighted 验证基线仅增加 0.000692，未达到预登记 0.005 改善标准。E02 最佳 A5B=0.848985，同样未达到主要改善标准。E03 与 A 合并的最佳 AP=0.848996，未明显超过 C 基线；B 单独分支的结果完整保留。E04 C 树模型达到 0.853273，超过预定 0.005 改善门槛。E05 固定三 seed C 平均达到 0.860661，B 平均达到 0.615683：两者均达各自门槛，但 B 仍明显弱于带注释模型。E07 的最佳全局 C 为 0.847881，E08 的最佳 C 为 0.845739，没有形成对 C 的主要改善。

E09 A2B 三 seed 平均为 0.862530，相对 E05 C 平均只增加 0.001869，未达到该补充实验 0.005 门槛；FAB 三 seed 平均仅 0.832896，是负面结果。两种输入和各 seed 均保留，不挑最有利 seed。E06 的全局集成产生更高验证 AP，按类模型选择则未通过稳健性门槛。

| 按类选择器 | 分量交叉检查 AP | 预定全局参照 AP | ΔAP | 有资格入选 |
| --- | --- | --- | --- | --- |
| E06_class_C_A | 0.768266 | 0.801183 | -0.032916 | 否 |
| E06_class_C_B | 0.528110 | 0.576052 | -0.047942 | 否 |
| E06_class_C_C | 0.820805 | 0.848197 | -0.027393 | 否 |
| E06_class_family_free | 0.847935 | 0.862530 | -0.014595 | 否 |
| E06_class_family_guarded | 0.841378 | 0.862530 | -0.021152 | 否 |

按类选择器以固定分量 hash 分成 5 折，在另外 4 折选择候选、在留出折取分数；门槛为相对预指定全局参照 AP 至少增加 0.002。自由家族选择器在完整验证上的表观 AP=0.905572，虽然超过最终方案，但交叉检查 ΔAP<0，被规则排除。该检查仅约束选择器的额外乐观性；基础模型早停和家族选择仍用过验证集，不能称为完全独立验证。

特征与训练规则：A 保留原 train df≥10 的直接 GO/keyword 1,219 维；A2/A5 的 train df≥2/5 词表为 3,200/1,849 维，各词条区分 GO 和 keyword 前缀并逐行 L2。B 使用原 train 均值/标准差及逐行 L2 的 1,280 维冻结向量；C=[A,B]/sqrt(2)，A2B=[A2,B]/sqrt(2)。F/O 是冻结 function_cc/organism 的 train-fit word (1,2)-gram TF-IDF，min_df=3，实际 6,701/3,838 维。训练与验证 function_cc 空文本分别 2,003/638 条，均保留为零块。Nyström 仅从 train 选择 1,024 个 landmarks；归一化/长度变体统计也仅 fit train。最终方案未选择 function_cc 或 organism 特征。

每类独立 logistic/ExtraTrees 只用该类 known train 行；weighted 正例权重为 min(10,sqrt(N/P))，负例为 1。Logistic 使用 lbfgs、tol=1e-6、max_iter=3000，预登记收敛重试上限 10000；实际没有收敛警告或重试。MLP 使用逐类 known 数归一的 masked weighted BCE，不把 unknown 当负例；AdamW lr=0.001、weight_decay=0.0001、batch=1024、clip_grad_norm=5，最多 120 epoch，每 5 epoch 验证（含第 1 epoch），连续 6 次无改善早停。没有 train+val 重训。

最终唯一方案 E06_mix_098：score = 0.2×A2B 三 seed MLP 平均 + 0.1×C 三 seed MLP 平均 + 0.7×C ExtraTrees。MLP 均为一个 512 维隐层，GELU 与 LayerNorm，A2B dropout=0.2、C dropout=0.5；每个均使用固定 seed 20261006/20261007/20261008 的均值。ExtraTrees 每类 256 棵树，max_features=0.2、min_samples_leaf=1，seed=20261006+class_index。模型图涉及六个神经模型及一套 32 类树模型。

选择依据是合资格候选中验证 macro AP 最高（0.902356579），规则中的 1e-6 并列时先少成员后登记顺序未改变胜者。二值阈值为验证集上逐类 F1 最优阈值，候选为不同分数加全阴端点，>= 判阳、并列取较高阈值；32 个数值已冻结在 final_selection.json 和预测 NPZ 中。验证按类阈值 F1=0.892310259，共享阈值 F1=0.857141194。AP 改善来自分数排序，与阈值优化无关；新方案 F1/FPR 与主实验共享阈值的比较同时包含阈值策略变化。

修订评价事件与时间线（均为既有落盘记录）：

| 事件 | UTC | Asia/Shanghai |
| --- | --- | --- |
| 模型及阈值冻结 | 2026-10-05T18:59:16.788510+00:00 | 2026-10-06T02:59:16.788510+08:00 |
| 唯一修订事件登记 | 2026-10-05T19:10:10.433160+00:00 | 2026-10-06T03:10:10.433174+08:00 |
| 首次读取本轮测试标签 | 2026-10-05T19:11:25.806854+00:00 | 2026-10-06T03:11:25.806854+08:00 |
| 评价完成 | 2026-10-05T19:11:27.455377+00:00 | 2026-10-06T03:11:27.455377+08:00 |
| 独立最终审计 | 2026-10-05T19:18:53.398222+00:00 | 2026-10-06T03:18:53.398222+08:00 |
| 网络恢复证书盘点 | 2026-10-05T19:33:26.371182+00:00 | 2026-10-06T03:33:26.371182+08:00 |

事件 ID=turn4_revision_1，event_type=修订评价事件，overall_test_view_number=2，new_selected_schemes=1。原因是用户授权探索结束后评价一套预先冻结方案；与 Turn 3 的关系为后续探索扩展，不能覆盖原表示对照结论。选择文件 SHA256=1a7b22306a6e87048b4e2499dd4765eba8d89ca65277080890b29933e078d1e7。

测试前的推理一致性检查发现，三 seed 均值的“逐项乘 1/3 再求和”与开发时“先求和再除 3”有最大 1.11e-16 的浮点差异，影响 6 个恰在阈值上的验证决策。原 inference.py 和原验证产物保留；另存 inference_v2.py，严格复现开发求值顺序，并登记 inference_runtime_v2.json。模型权重、特征、集成权重和阈值均未变；修正发生于测试事件之前。v2 独立重算验证分数 max_abs_diff=0.0、二值预测完全一致。权威输出目录为 inference_v2/。

完整修订测试结果：每类 AP/F1 仅基于已知单元；micro 仅展平 known 单元。完整集与共同子集 K_AP=32/32、K_F1=32/32。全负背景分母要求该蛋白 32 类均为 neg；含任一 unknown 者不进入分母。以下存档 A/B/C 分数只作为预登记配对比较，没有重新选择或训练。

完整测试：6,257 蛋白、1,396 分量；P/N/U=5,170/110,864/84,190。

| 模型 | macro AP | macro F1 | micro P | micro R | micro F1 | any-label FPR | unknown 阳性占比 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A_weighted | 0.809550 | 0.756524 | 0.875171 | 0.865184 | 0.870149 | 7.6442% | 67.8553% |
| B_weighted | 0.524121 | 0.513176 | 0.638277 | 0.616054 | 0.626969 | 18.4843% | 50.1100% |
| C_weighted | 0.832643 | 0.757088 | 0.832320 | 0.873694 | 0.852505 | 8.5338% | 64.3383% |
| A_unweighted | 0.820259 | 0.751610 | 0.810626 | 0.894197 | 0.850363 | 11.3015% | 68.3184% |
| B_unweighted | 0.508343 | 0.467243 | 0.569949 | 0.649323 | 0.607052 | 25.6672% | 51.8909% |
| C_unweighted | 0.831882 | 0.748381 | 0.821188 | 0.874081 | 0.846810 | 9.1269% | 64.1965% |
| selected | 0.886441 | 0.783180 | 0.927118 | 0.863636 | 0.894252 | 1.3509% | 75.1637% |

具有可用注释共同子集：6,210 蛋白、1,385 分量；P/N/U=5,113/110,801/82,806。

| 模型 | macro AP | macro F1 | micro P | micro R | micro F1 | any-label FPR | unknown 阳性占比 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A_weighted | 0.810612 | 0.752051 | 0.875764 | 0.868570 | 0.872152 | 7.6137% | 67.7253% |
| B_weighted | 0.515159 | 0.503195 | 0.635831 | 0.615686 | 0.625596 | 18.4904% | 49.3711% |
| C_weighted | 0.826981 | 0.747822 | 0.831477 | 0.876198 | 0.853252 | 8.5036% | 64.1254% |
| A_unweighted | 0.824212 | 0.754334 | 0.810515 | 0.898494 | 0.852240 | 11.2722% | 68.2643% |
| B_unweighted | 0.498886 | 0.457070 | 0.567397 | 0.648739 | 0.605347 | 25.6757% | 51.2712% |
| C_unweighted | 0.826525 | 0.739412 | 0.820546 | 0.876394 | 0.847551 | 9.0969% | 64.0558% |
| selected | 0.884595 | 0.774100 | 0.927658 | 0.865245 | 0.895365 | 1.3184% | 74.7619% |

最终方案完整测试 TP/FP/FN/TN=4465/351/705/110513；全负背景为 41/3035，共同子集为 40/3034。完整测试共有 19,391 个预测阳性，其中 14,575 个落在 unknown，占 75.1637%；存档 C_weighted 的相应占比为 64.3383%。unknown 上的预测不算正确或错误，这一较高占比限制了对全部 32 类实际功能判别性能的外推。所有 32 类行与 NA 规则完整保存在 revision/per_class.tsv。共同子集沿用主实验的可用注释定义，未按新 A2 词表重新定义。

分量配对 bootstrap：复用主实验 seed=20261008 的 2,000 轮抽样索引，所有方法同轮配对；重复分量按次数计权，不改变原 macro 类别集合。两个子集均从原完整测试 1,396 分量的同一抽样宇宙取样，共同子集只屏蔽不属于该子集的蛋白。完整测试有效 1,604/2,000（80.2%），共同子集有效 1,585/2,000（79.25%）。预定保守规则下，存在任何无效轮次就把普通 95% CI 记为 NA，所以全部 24 个比较的普通区间都是 NA。

| 子集 | 指标 | 观测 Δ | 普通 95% CI | 有效轮数 | 仅有效轮次条件区间 |
| --- | --- | --- | --- | --- | --- |
| full_test | macro_AP | 0.053798 | NA | 1604/2000 | [0.028411, 0.083858] |
| full_test | macro_F1 | 0.026092 | NA | 1604/2000 | [-0.000323, 0.065765] |
| usable_annotations | macro_AP | 0.057613 | NA | 1585/2000 | [0.033749, 0.087347] |
| usable_annotations | macro_F1 | 0.026279 | NA | 1585/2000 | [-0.001383, 0.070141] |

上述条件区间只描述标签支持仍足够的轮次，不能替代普通 CI，也不支持按普通 CI 宣称显著性。完整测试无效轮次涉及 9 类，类别支持缺失次数（同一轮可重叠）：bacterial_counter_signaling 5、counter_immunoglobulin 51、degrade_ecm 5、host_gtpase 10、host_translation 94、host_xenophagy 51、resist_complement 1、resist_oxidative 4、suppress_detection 270。共同子集涉及 10 类，详见 support_summary.json。抽样索引、每 100 轮检查点、逐轮值/配对差值、全部六个存档比较均在 revision/bootstrap/，未丢弃无效轮次后重抽。

分层结果包含预定 27 层：长度四档、annotation score、注释状态、分类群、数据库来源、FASTA 来源、背景采样来源及证据层级。各层重新遵循原 NA/K 规则，不能把不同类别集合的层间 macro 数字视为控制其他因素的比较。以下长度表仅用于描述这次固定评价，未据此调整方案。

| 长度 | 蛋白 | K_AP/32 | K_F1/32 | macro AP | macro F1 |
| --- | --- | --- | --- | --- | --- |
| <=256 | 3007 | 29/32 | 29/32 | 0.8823450546877731 | 0.7921003452336737 |
| 257-512 | 1835 | 28/32 | 28/32 | 0.8930872645926881 | 0.8136741984279526 |
| 513-1022 | 1103 | 29/32 | 29/32 | 0.8308389347238097 | 0.657467671171252 |
| >1022 | 312 | 26/32 | 26/32 | 0.8481541492229852 | 0.6910567316503707 |

测试注释状态为 usable 6,210、成功读取但原注释为空 10、全部被裁剪或 OOV 37、提取失败 0，全部仍在完整测试中。annotation score 未在冻结元数据提供，因此 6,257 条记 missing，1–5 分层留零人数/NA；背景采样来源也沿用 missing。已知正例证据层级沿用原元数据的 unknown，未用新检索补填。完整层级 P/N/U、K、每类指标和汇总位于 revision/stratified_*.tsv。

运行与验证：使用工作目录下既有 mamba 环境 seqscreen_train / seqscreen_esm，无新增包安装或权重下载；Python 3.11.16、numpy 1.26.4、scipy 1.17.1、scikit-learn 1.7.2、joblib 1.6.0、numba 0.61.2；Torch 2.5.1、CUDA 12.6，四张 H100。CPU 类拟合并行、GPU 四 worker，缓存和临时路径全部由 scripts/env.sh 指向 exploration/cache。固定随机种子，MLP FP32、TF32 关闭且 deterministic algorithms 开启。

保存 5,696 个类拟合记录，其中复用主实验 576 个、新增 5,120 个，另有 20 个 MLP 拟合；复用模型保留原 seed=20261005 元数据。失败拟合 0、收敛重试 0、未处理收敛警告 0。GPU 拟合过程耗时之和 56.7 秒，单 worker 已记录最大 allocated/reserved 为 1.071/1.180 GiB；并行进程耗时之和不等于墙钟或核时。每类/每 seed 耗时、GPU峰值、PID与完整日志都已保存。

验证包含 26 项合成检查、固定模型重新推理与开发验证分数逐位一致、两子集独立 sklearn 指标复算、32 个 selected bootstrap macro 数值与重复分量 sample_weight 的独立核对、随机抽样索引完全再生、TSV 分数无损往返，以及 2,733 个旧产物哈希复核。最终审计结果为 passed=true。原子写临时文件验证后 rename，逐类、逐 seed、神经 epoch 与 bootstrap block 均有检查点；续跑只跳过已校验内容。

未开展的可选方向：历史 GO 祖先扩展（缺冻结历史本体）、其他 ESM 层或更大 PLM、classifier chains。unknown 伪标签未使用。没有因时间不足而省略任何已登记实验或验证步骤。后续如研究这些方向，应在新版本、预先定义的验证安排及独立评价数据上开展，当前测试结果不用于继续选择。paper_negatives 仍未提供，本轮仅使用原主背景口径。

主要交付文件如下，路径相对 output/exploration/。全量文件（含模型、特征、日志、缓存）的 SHA256、字节数与文本行数在 exploration/artifacts.tsv；SHA256SUMS 路径相对 output/，包含本 REPORT_turn4.md。清单自身、manifest 与 DONE 用外层校验链记录，避免自引用。

| 文件 | SHA256 |
| --- | --- |
| PLAN.md | 7f79d62ca0b980859d0021e0bb6840481f8ac20a981888b2cabd5ee0f64a6c4c |
| PLAN_addendum_v1.md | 29b005918d4ca3ec55d393142a9a56df2bd116a42776dc721540d327035336d9 |
| PLAN_addendum_v2.md | 4c177b493678a2208442ca96698c1506dca4fd84ba0633733226a363a87a8464 |
| experiment_results.tsv | 4bceb199c8a226bf14cdfa9353589df07ead963354783b556c57a89a2c4119af |
| stage_results.tsv | 98805f1da54124b7944acca76fe6821a18d16c3982c758bb3ba3e0eca31779c1 |
| selection/final_selection.json | 1a7b22306a6e87048b4e2499dd4765eba8d89ca65277080890b29933e078d1e7 |
| selection/inference_runtime_v2.json | e6be605f4ad1ba1374e90d6572a7340b3e4b6551026e3f96e31f8943c3d95fad |
| revision/revision_evaluation_event.json | 4c1c2ed43325b78a7593d1ad3827155daca47b302513991136abcf24ce02557a |
| inference_v2/test/final_predictions.npz | 98fc860fba4aae6c8f53c4b69bfd1f4d7a7ab8249496c47cf29c6e6074dc1886 |
| revision/scores.tsv | 93624a34a2b1b745efefa168dda23500cee4bd5fa48dcaab11f75b30ecd087c0 |
| revision/predictions.tsv | eacefc80ce2aa6f4d58bb22e0b6a5e88d374ee246ba04fe2605b18ed1589dc8f |
| revision/prediction_index.tsv | e782639a2fdd30b68987d921b5d5a5fdc0b200033356b5eef5c5aecf277c59e8 |
| revision/per_class.tsv | 719ae302e81a470eaae85ee9d025ec362ff179c724425c21f01244c656aec323 |
| revision/summary.tsv | f0abd67b530f7fa34c729ffc8280b4fd22491b816136268e1fca1d85f4e3065f |
| revision/stratified_summary.tsv | 303159975b0b1f4d0a5cfadcbaf3a768f6542891a44973bb709a0e674103ca9d |
| revision/stratified_per_class.tsv | 21a7b31cfd41315eed5fcc5163bcc248f44da65d50e3a23c8995c95ca06e1614 |
| revision/bootstrap/intervals.tsv | 1ff76e0e234049fcbd7c7eb4963d25da55ac6723e1851adba9d9fe39a22d3e1a |
| revision/bootstrap/draws.npz | f30c6bf51acce26bb8caffeb933e22c70887af723078386789b7f4636dab92be |
| validation/final_audit.json | c350bccd63beb500217a27eda2a54d49097afa4fa25a83c25798cae6439de9a5 |
| validation/recovery_inventory.json | 61392e4760e34f066708d8e4af91a98698b930aa0f38b70fc0749503c814253c |
| runtime_summary.json | 75899a43514ca991cf8b2b39d3d88d351e4486b16e8edf50b02bebe61bbb7aeb |
| schema.json | c77a23440de8e0ec89d4a126a747cd943b6c541318b4ba9a0a376caadd402838 |

已完成本轮授权工作。最终选择及修订评价保持冻结，训练和调优停止。

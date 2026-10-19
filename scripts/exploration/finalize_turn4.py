"""Package completed results. Does not train, select, infer, or evaluate models."""
from common import *
from collections import Counter
from datetime import datetime

def mdtable(headers,data):
 return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(str(v).replace('|','/') for v in r)+' |\n' for r in data)+'\n'
def f(x):return 'NA' if x is None else f'{float(x):.6f}'
def pct(x):return 'NA' if x is None else f'{100*float(x):.4f}%'
def local(t):return datetime.fromisoformat(t).astimezone(__import__('zoneinfo').ZoneInfo('Asia/Shanghai')).isoformat()

def main():
 if (E/'DONE.json').exists():
  from verify_delivery import verify
  verify();return
 start=time.monotonic();recovery=readj(E/'validation/recovery_inventory.json');audit=readj(E/'validation/final_audit.json')
 assert recovery['passed'] and audit['passed'] and not recovery['missing_or_interrupted_configurations']
 assert all(r['status']=='not_running' for r in recovery['recorded_pids'])
 selected=readj(E/'selection/final_selection.json');event=readj(E/'revision/revision_evaluation_event.json');record=readj(E/'revision/evaluation_record.json')
 assert verified(E/'revision/evaluation.DONE.json') and verified(E/'revision/bootstrap/DONE.json')
 summaries=[readj(p) for p in sorted((E/'candidates').glob('*/summary.json'))]
 ledger=[]
 for d in summaries:
  c=d['config'];eligible=d.get('selection_eligible',True)
  status='complete_eligible' if eligible else ('complete_seed_diagnostic' if c['algorithm']=='mlp' else 'complete_rejected_by_crossfit_guard')
  ledger.append({'candidate':d['name'],'stage':c['stage'],'algorithm':c['algorithm'],'seed':d['seed'],'validation_macro_AP':d['validation_macro_AP'],'validation_macro_F1_per_class':d['validation_macro_F1'],'validation_macro_F1_shared':d['validation_shared_macro_F1'],'eligible':eligible,'status':status,'configuration_json':json.dumps(c,sort_keys=True,separators=(',',':')),'summary_path':f"candidates/{d['name']}/summary.json",'summary_sha256':sha(E/f"candidates/{d['name']}/summary.json")})
 tsv(E/'experiment_results.tsv',list(ledger[0]),ledger)
 stages=[]
 for stage in sorted({d['config']['stage'] for d in summaries}):
  ds=[d for d in summaries if d['config']['stage']==stage];b=max([d for d in ds if d.get('selection_eligible',True)],key=lambda d:d['validation_macro_AP'])
  stages.append({'stage':stage,'candidate_records':len(ds),'eligible':sum(d.get('selection_eligible',True) for d in ds),'best_candidate':b['name'],'validation_macro_AP':b['validation_macro_AP'],'validation_macro_F1_per_class':b['validation_macro_F1'],'validation_macro_F1_shared':b['validation_shared_macro_F1'],'status':'all_registered_configurations_complete'})
 tsv(E/'stage_results.tsv',list(stages[0]),stages)
 fits=[readj(p) for p in sorted((E/'experiments').glob('*/class_??.json'))]
 gh=[readj(p) for p in sorted((E/'gpu').glob('*/history.json'))]
 timing={'at':now(),'class_records':len(fits),'reused_main_class_models':sum('source_model' in x for x in fits),'new_class_fits':sum('source_model' not in x for x in fits),'gpu_fits':len(gh),'class_fit_process_seconds_sum':sum(x.get('seconds',0) for x in fits),'gpu_fit_process_seconds_sum':sum(x['seconds'] for x in gh),'maximum_gpu_allocated_bytes':max(x['peak_allocated_bytes'] for x in gh),'maximum_gpu_reserved_bytes':max(x['peak_reserved_bytes'] for x in gh),'convergence_retry_count':sum(len(x.get('retries',[])) for x in fits),'failed_fit_count':sum(not x.get('success',True) for x in fits),'convergence_warnings':sum(len(x.get('warnings',[])) for x in fits),'incomplete_experiments':[],'process_seconds_note':'sum of overlapping per-fit elapsed times; not wall-clock time or CPU core-seconds','frozen_at':event['model_frozen_at_utc'],'evaluation_completed_at':record['completed_at_utc'],'recovery_inventory_at':recovery['at'],'network_recovery_retrained':False,'network_recovery_reevaluated_test':False,'torch_versions':sorted({x['torch'] for x in gh}),'torch_cuda_versions':sorted({x['cuda'] for x in gh})}
 jout(E/'runtime_summary.json',timing)
 failure_record={'at':now(),'runtime_training_failures':[],'interrupted_configurations_at_recovery':[],'rejected_selectors':[{k:d[k] for k in ['name','crossfit_macro_AP','reference_macro_AP','crossfit_delta','selection_eligible']} for d in readj(E/'selection/crossfit_selector_checks.json')],'pretest_inference_numerical_correction':readj(E/'selection/inference_runtime_v2.json'),'no_posttest_model_or_threshold_change':True,'unstarted_optional_branches':['GO ancestor expansion: frozen historical ontology not supplied','new PLM or other ESM layers: outside registered prioritized grid','classifier chains: not registered as mandatory','unknown pseudo-labels: not used'],'notes':'Statistical negative results are preserved in experiment_results.tsv and REPORT_turn4.md; no failed fit was silently scored.'}
 failure_record['delivery_retry']={'error':'KeyError: selection_eligible in first packaging attempt','resolution':'use get(selection_eligible, True), exactly matching immutable select_final.py','model_or_evaluation_effect':False,'at':now()}
 jout(E/'failures_and_deferred.json',failure_record)
 flow=[]
 for split,n,groups,P,N,U in [('train',18770,4181,15388,333780,251472),('val',6257,1398,5161,111556,83507),('test',6257,1396,5170,110864,84190)]:
  flow.append({'split':split,'proteins':n,'components':groups,'positive_cells':P,'negative_cells':N,'unknown_cells_masked':U,'excluded_by_exploration':0,'fitting_use':'fit all data-driven transforms and model parameters' if split=='train' else ('selection only' if split=='val' else 'one frozen revised evaluation only')})
 tsv(E/'data_flow.tsv',list(flow[0]),flow)

 schema={'version':'turn4_exploration_delivery_v1','encoding':'UTF-8; TSV has one header line; JSON null/TSV NA denotes undefined, never zero-filled','id_order':'explicit protein_id arrays; columns follow frozen class_order.json','labels':{'positive':1,'negative':0,'unknown':-1,'unknown_policy':'masked in all fitting and metrics'},'data/dev.npz':'train and val only, ID/group/label arrays and class order; no test labels','features':'train-fit definitions and parameters; scipy CSR npz or dense numpy npy. Float32 GPU copies are separately verified against canonical features. No role/source/evidence-level input features.','model_storage':{'linear':'experiments/<candidate>/class_XX.npz; intercept and coefficients','extra_trees':'experiments/<candidate>/class_XX.pkl; own trusted sklearn 1.7.2 artifact','neural':'gpu/<candidate>/best.pt; own torch 2.5.1 state dictionary plus config; last.pt retains optimizer and RNG for checkpoint audit','ensemble':'candidate summary config.members/config.weights; fixed three-seed means use numpy mean after stacking, as inference_v2.py'},'predictions':{'authoritative':'inference_v2/test/final_predictions.npz','arrays':{'protein_id':'6257 Unicode IDs','class_order':'32 Unicode class names','scores':'(6257,32) float64, probabilities','predictions':'(6257,32) uint8, scores>=thresholds','thresholds':'(32,) float64, selected on val'},'tsv':'revision/scores.tsv and predictions.tsv contain protein_id followed by fixed 32 classes; scores .17g roundtrip; index in revision/prediction_index.tsv','validation':'inference_v2/val/final_predictions.npz reproduces candidate scores and decisions exactly','legacy':'inference/val retained as pretest numerical-arithmetic diagnostic; never authoritative'},'bootstrap':{'draws':'revision/bootstrap/draws.npz; 2000x1396 component draws and multiplicities; seed20261008, same draws for all models and both subsets','models':['selected','A_weighted','B_weighted','C_weighted','A_unweighted','B_unweighted','C_unweighted'],'metrics':['macro_AP','macro_F1'],'blocks':'100 iterations per npz; values shape (100,7,2); paired delta shape (100,6,2)','ci_rule':'fixed original K; any invalid replicate makes ordinary CI NA; conditional valid-only intervals explicitly separate'},'selection':'selection/final_selection.json plus inference_runtime_v2.json, both frozen before test','certificates':'each DONE records hashes of completed files; verify before skipping; final DONE hashes inventory, manifest and checksum list','inventory':'artifacts.tsv paths relative to output; bytes, SHA256, text physical line counts (header included); binary line count NA. Inventory excludes only artifacts.tsv, SHA256SUMS, manifest.json, DONE.json to avoid self-reference; includes REPORT_turn4.md and all cache/log/PID files.'}
 jout(E/'schema.json',schema)
 readme='''本目录是 Turn 4 探索交付，原主实验保持在 ../training/，未被替代。

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
'''
 txt(E/'README.md',readme)

 baseline=[r for r in rows(E/'diagnostics/baseline_train_val.tsv') if r['split']=='val']
 full=readj(E/'revision/summary_full_test.json');common=readj(E/'revision/summary_usable_annotations.json');intervals=readj(E/'revision/bootstrap/intervals.json')
 chosen=readj(E/'candidates/E06_mix_098/summary.json');checks=readj(E/'selection/crossfit_selector_checks.json')
 ss=full['selected'];cs=common['selected'];strata=list(rows(E/'revision/stratified_summary.tsv'))
 lines=[]
 lines.append('Turn 4 探索与修订评价报告\n\n')
 lines.append(f'交付生成时间（UTC）：{now()}。按 PLAN.md 及两份 addendum 完成 E00–E09。最终方案 E06_mix_098 在完整测试集上的 macro AP={ss["macro_AP"]:.6f}、macro F1={ss["macro_F1"]:.6f}，全负背景 any-label FPR={pct(ss["any_label_FPR"])}。相对存档 C_weighted 的 Δmacro AP={ss["macro_AP"]-full["C_weighted"]["macro_AP"]:+.6f}。这是用户授权的第二次查看测试集，属于探索后的修订评价；主实验 A/B/C 的既有结果和结论完整保留。\n\n')
 lines.append('所有选择只使用 train/val；测试阶段之后没有训练、超参/阈值调整、候选替换或新的模型选择。该测试集已在 Turn 3 被查看，因此本结果不能称为首次盲测或独立预注册验证。验证集被反复用于模型、早停与集成探索，验证分数存在选择偏差。\n\n')
 lines.append(f'网络恢复盘点：{recovery["certificates_verified"]:,} 份 DONE 证书及 {recovery["unique_referenced_files_verified"]:,} 个被引用文件通过 SHA256 校验；335 条候选、20 个 GPU 配置全部完成，0 个中断配置。24 个已登记 PID 均已退出。磁盘已经包含最终选择、唯一修订评价和 bootstrap，因此本次续跑只完成证书核验和交付整理，没有重复测试事件。记录：exploration/validation/recovery_inventory.json、recovery_stages.tsv、recovery_certificates.tsv。\n\n')
 lines.append('数据流转及不变项：总宇宙 31,284 个蛋白、1,001,088 个标签格，P/N/U=25,719/556,200/419,169。本轮没有额外排除蛋白，没有改变角色、标签、划分或序列。unknown 在训练损失和评价中均屏蔽，未生成伪标签。\n\n')
 lines.append(mdtable(['集合','蛋白','分量','P','N','U','用途'],[[r['split'],r['proteins'],r['components'],r['positive_cells'],r['negative_cells'],r['unknown_cells_masked'],r['fitting_use']] for r in flow]))
 lines.append('开发输入包仅含 25,027 个 train+val 蛋白。原始文件、五个冻结文件、embedding 主产物和主实验训练产物共 2,733 个历史文件的哈希全部保持一致；主实验文件与冻结输入只读。五项既有问题保持原样：fragment 913、accession 映射差异、重复映射、同序列 P/N 不一致、缺源端 manifest。本轮不重新审核或修订其生物学标签。\n\n')
 lines.append('验证基线（沿用主实验共享阈值）：\n\n')
 lines.append(mdtable(['路线','val macro AP','val macro F1'],[[r['model'],f(r['macro_AP']),f(r['macro_F1'])] for r in baseline]))
 lines.append('实验覆盖：E01 共同 C 的 30 个配置；E02 注释/文本的 60 个配置；E03 Nyström 的 18 个配置；E04 ExtraTrees 的 12 个配置；E05 12 个单 seed 拟合加 2 个固定三 seed 平均；E07 分量再加权 18 个配置；E08 归一化变体 40 个配置；E09 8 个单 seed 拟合加 2 个固定三 seed 平均；E06 为 5 个按类选择器加 128 个全局集成。合计 335 条候选记录，其中 20 个单 seed 仅诊断、5 个选择器被交叉检查排除，最终 310 个候选符合选择资格。所有逐项配置、seed、验证指标和状态列于 exploration/experiment_results.tsv，完整排序与资格列于 selection/validation_leaderboard.tsv。\n\n')
 lines.append(mdtable(['阶段','候选数','合资格最佳候选','val AP','val F1（按类阈值）','val F1（共享阈值）'],[[r['stage'],r['candidate_records'],r['best_candidate'],f(r['validation_macro_AP']),f(r['validation_macro_F1_per_class']),f(r['validation_macro_F1_shared'])] for r in stages]))
 lines.append('E00 发现 B_weighted 的 train AP=0.975791、val AP=0.570745，说明读出在已知训练单元上拟合很好但跨分量泛化差距较大；不能据此把原因归于线性欠拟合。B 在 6 个验证类别上胜过 A，按类、长度、注释状态及训练支持的诊断已保存。历史 GO/keyword/function 与标签构建可能共享证据，来源及注释质量也可能影响关联；这些现象不构成因果证据。没有根据测试短板再增加实验。\n\n')
 lines.append('E01 的最佳 C 路线 AP=0.848197，较 C_weighted 验证基线仅增加 0.000692，未达到预登记 0.005 改善标准。E02 最佳 A5B=0.848985，同样未达到主要改善标准。E03 与 A 合并的最佳 AP=0.848996，未明显超过 C 基线；B 单独分支的结果完整保留。E04 C 树模型达到 0.853273，超过预定 0.005 改善门槛。E05 固定三 seed C 平均达到 0.860661，B 平均达到 0.615683：两者均达各自门槛，但 B 仍明显弱于带注释模型。E07 的最佳全局 C 为 0.847881，E08 的最佳 C 为 0.845739，没有形成对 C 的主要改善。\n\n')
 lines.append('E09 A2B 三 seed 平均为 0.862530，相对 E05 C 平均只增加 0.001869，未达到该补充实验 0.005 门槛；FAB 三 seed 平均仅 0.832896，是负面结果。两种输入和各 seed 均保留，不挑最有利 seed。E06 的全局集成产生更高验证 AP，按类模型选择则未通过稳健性门槛。\n\n')
 lines.append(mdtable(['按类选择器','分量交叉检查 AP','预定全局参照 AP','ΔAP','有资格入选'],[[d['name'],f(d['crossfit_macro_AP']),f(d['reference_macro_AP']),f(d['crossfit_delta']),'否'] for d in checks]))
 lines.append('按类选择器以固定分量 hash 分成 5 折，在另外 4 折选择候选、在留出折取分数；门槛为相对预指定全局参照 AP 至少增加 0.002。自由家族选择器在完整验证上的表观 AP=0.905572，虽然超过最终方案，但交叉检查 ΔAP<0，被规则排除。该检查仅约束选择器的额外乐观性；基础模型早停和家族选择仍用过验证集，不能称为完全独立验证。\n\n')
 lines.append('特征与训练规则：A 保留原 train df≥10 的直接 GO/keyword 1,219 维；A2/A5 的 train df≥2/5 词表为 3,200/1,849 维，各词条区分 GO 和 keyword 前缀并逐行 L2。B 使用原 train 均值/标准差及逐行 L2 的 1,280 维冻结向量；C=[A,B]/sqrt(2)，A2B=[A2,B]/sqrt(2)。F/O 是冻结 function_cc/organism 的 train-fit word (1,2)-gram TF-IDF，min_df=3，实际 6,701/3,838 维。训练与验证 function_cc 空文本分别 2,003/638 条，均保留为零块。Nyström 仅从 train 选择 1,024 个 landmarks；归一化/长度变体统计也仅 fit train。最终方案未选择 function_cc 或 organism 特征。\n\n')
 lines.append('每类独立 logistic/ExtraTrees 只用该类 known train 行；weighted 正例权重为 min(10,sqrt(N/P))，负例为 1。Logistic 使用 lbfgs、tol=1e-6、max_iter=3000，预登记收敛重试上限 10000；实际没有收敛警告或重试。MLP 使用逐类 known 数归一的 masked weighted BCE，不把 unknown 当负例；AdamW lr=0.001、weight_decay=0.0001、batch=1024、clip_grad_norm=5，最多 120 epoch，每 5 epoch 验证（含第 1 epoch），连续 6 次无改善早停。没有 train+val 重训。\n\n')
 lines.append('最终唯一方案 E06_mix_098：score = 0.2×A2B 三 seed MLP 平均 + 0.1×C 三 seed MLP 平均 + 0.7×C ExtraTrees。MLP 均为一个 512 维隐层，GELU 与 LayerNorm，A2B dropout=0.2、C dropout=0.5；每个均使用固定 seed 20261006/20261007/20261008 的均值。ExtraTrees 每类 256 棵树，max_features=0.2、min_samples_leaf=1，seed=20261006+class_index。模型图涉及六个神经模型及一套 32 类树模型。\n\n')
 lines.append(f'选择依据是合资格候选中验证 macro AP 最高（{chosen["validation_macro_AP"]:.9f}），规则中的 1e-6 并列时先少成员后登记顺序未改变胜者。二值阈值为验证集上逐类 F1 最优阈值，候选为不同分数加全阴端点，>= 判阳、并列取较高阈值；32 个数值已冻结在 final_selection.json 和预测 NPZ 中。验证按类阈值 F1={chosen["validation_macro_F1"]:.9f}，共享阈值 F1={chosen["validation_shared_macro_F1"]:.9f}。AP 改善来自分数排序，与阈值优化无关；新方案 F1/FPR 与主实验共享阈值的比较同时包含阈值策略变化。\n\n')
 lines.append('修订评价事件与时间线（均为既有落盘记录）：\n\n')
 lines.append(mdtable(['事件','UTC','Asia/Shanghai'],[['模型及阈值冻结',event['model_frozen_at_utc'],local(event['model_frozen_at_utc'])],['唯一修订事件登记',event['registered_at_utc'],event['registered_at_local']],['首次读取本轮测试标签',record['test_labels_read_at_utc'],local(record['test_labels_read_at_utc'])],['评价完成',record['completed_at_utc'],local(record['completed_at_utc'])],['独立最终审计',audit['at'],local(audit['at'])],['网络恢复证书盘点',recovery['at'],local(recovery['at'])]]))
 lines.append(f'事件 ID={event["event_id"]}，event_type=修订评价事件，overall_test_view_number=2，new_selected_schemes=1。原因是用户授权探索结束后评价一套预先冻结方案；与 Turn 3 的关系为后续探索扩展，不能覆盖原表示对照结论。选择文件 SHA256={sha(E/"selection/final_selection.json")}。\n\n')
 lines.append('测试前的推理一致性检查发现，三 seed 均值的“逐项乘 1/3 再求和”与开发时“先求和再除 3”有最大 1.11e-16 的浮点差异，影响 6 个恰在阈值上的验证决策。原 inference.py 和原验证产物保留；另存 inference_v2.py，严格复现开发求值顺序，并登记 inference_runtime_v2.json。模型权重、特征、集成权重和阈值均未变；修正发生于测试事件之前。v2 独立重算验证分数 max_abs_diff=0.0、二值预测完全一致。权威输出目录为 inference_v2/。\n\n')
 lines.append('完整修订测试结果：每类 AP/F1 仅基于已知单元；micro 仅展平 known 单元。完整集与共同子集 K_AP=32/32、K_F1=32/32。全负背景分母要求该蛋白 32 类均为 neg；含任一 unknown 者不进入分母。以下存档 A/B/C 分数只作为预登记配对比较，没有重新选择或训练。\n\n')
 for title,dd in [('完整测试：6,257 蛋白、1,396 分量；P/N/U=5,170/110,864/84,190。',full),('具有可用注释共同子集：6,210 蛋白、1,385 分量；P/N/U=5,113/110,801/82,806。',common)]:
  lines.append(title+'\n\n')
  order=['A_weighted','B_weighted','C_weighted','A_unweighted','B_unweighted','C_unweighted','selected']
  lines.append(mdtable(['模型','macro AP','macro F1','micro P','micro R','micro F1','any-label FPR','unknown 阳性占比'],[[n]+[f(dd[n][k]) for k in ['macro_AP','macro_F1','micro_precision','micro_recall','micro_F1']]+[pct(dd[n]['any_label_FPR']),pct(dd[n]['unknown_positive_fraction'])] for n in order]))
 lines.append(f'最终方案完整测试 TP/FP/FN/TN={ss["TP"]}/{ss["FP"]}/{ss["FN"]}/{ss["TN"]}；全负背景为 {ss["any_label_false_positives"]}/{ss["fully_negative_background_proteins"]}，共同子集为 {cs["any_label_false_positives"]}/{cs["fully_negative_background_proteins"]}。完整测试共有 {ss["all_predicted_positive"]:,} 个预测阳性，其中 {ss["unknown_predicted_positive"]:,} 个落在 unknown，占 {pct(ss["unknown_positive_fraction"])}；存档 C_weighted 的相应占比为 {pct(full["C_weighted"]["unknown_positive_fraction"])}。unknown 上的预测不算正确或错误，这一较高占比限制了对全部 32 类实际功能判别性能的外推。所有 32 类行与 NA 规则完整保存在 revision/per_class.tsv。共同子集沿用主实验的可用注释定义，未按新 A2 词表重新定义。\n\n')
 lines.append('分量配对 bootstrap：复用主实验 seed=20261008 的 2,000 轮抽样索引，所有方法同轮配对；重复分量按次数计权，不改变原 macro 类别集合。两个子集均从原完整测试 1,396 分量的同一抽样宇宙取样，共同子集只屏蔽不属于该子集的蛋白。完整测试有效 1,604/2,000（80.2%），共同子集有效 1,585/2,000（79.25%）。预定保守规则下，存在任何无效轮次就把普通 95% CI 记为 NA，所以全部 24 个比较的普通区间都是 NA。\n\n')
 primary=[r for r in intervals if r['comparison']=='selected-C_weighted']
 lines.append(mdtable(['子集','指标','观测 Δ','普通 95% CI','有效轮数','仅有效轮次条件区间'],[[r['subset'],r['metric'],f(r['observed_delta']),'NA',f"{r['valid_replicates']}/2000",f"[{r['conditional_valid_only_low']:.6f}, {r['conditional_valid_only_high']:.6f}]"] for r in primary]))
 lines.append('上述条件区间只描述标签支持仍足够的轮次，不能替代普通 CI，也不支持按普通 CI 宣称显著性。完整测试无效轮次涉及 9 类，类别支持缺失次数（同一轮可重叠）：bacterial_counter_signaling 5、counter_immunoglobulin 51、degrade_ecm 5、host_gtpase 10、host_translation 94、host_xenophagy 51、resist_complement 1、resist_oxidative 4、suppress_detection 270。共同子集涉及 10 类，详见 support_summary.json。抽样索引、每 100 轮检查点、逐轮值/配对差值、全部六个存档比较均在 revision/bootstrap/，未丢弃无效轮次后重抽。\n\n')
 lines.append('分层结果包含预定 27 层：长度四档、annotation score、注释状态、分类群、数据库来源、FASTA 来源、背景采样来源及证据层级。各层重新遵循原 NA/K 规则，不能把不同类别集合的层间 macro 数字视为控制其他因素的比较。以下长度表仅用于描述这次固定评价，未据此调整方案。\n\n')
 lr=[r for r in strata if r['stratum_type']=='length_bin'];assert len(lr)==4
 lines.append(mdtable(['长度','蛋白','K_AP/32','K_F1/32','macro AP','macro F1'],[[r['stratum'],r['proteins'],r['K_AP']+'/32',r['K_F1']+'/32',r['macro_AP'],r['macro_F1']] for r in lr]))
 lines.append('测试注释状态为 usable 6,210、成功读取但原注释为空 10、全部被裁剪或 OOV 37、提取失败 0，全部仍在完整测试中。annotation score 未在冻结元数据提供，因此 6,257 条记 missing，1–5 分层留零人数/NA；背景采样来源也沿用 missing。已知正例证据层级沿用原元数据的 unknown，未用新检索补填。完整层级 P/N/U、K、每类指标和汇总位于 revision/stratified_*.tsv。\n\n')
 lines.append(f'运行与验证：使用工作目录下既有 mamba 环境 seqscreen_train / seqscreen_esm，无新增包安装或权重下载；Python 3.11.16、numpy 1.26.4、scipy 1.17.1、scikit-learn 1.7.2、joblib 1.6.0、numba 0.61.2；Torch {", ".join(timing["torch_versions"])}、CUDA {", ".join(timing["torch_cuda_versions"])}，四张 H100。CPU 类拟合并行、GPU 四 worker，缓存和临时路径全部由 scripts/env.sh 指向 exploration/cache。固定随机种子，MLP FP32、TF32 关闭且 deterministic algorithms 开启。\n\n')
 lines.append(f'保存 {timing["class_records"]:,} 个类拟合记录，其中复用主实验 {timing["reused_main_class_models"]:,} 个、新增 {timing["new_class_fits"]:,} 个，另有 {timing["gpu_fits"]} 个 MLP 拟合；复用模型保留原 seed=20261005 元数据。失败拟合 {timing["failed_fit_count"]}、收敛重试 {timing["convergence_retry_count"]}、未处理收敛警告 0。GPU 拟合过程耗时之和 {timing["gpu_fit_process_seconds_sum"]:.1f} 秒，单 worker 已记录最大 allocated/reserved 为 {timing["maximum_gpu_allocated_bytes"]/2**30:.3f}/{timing["maximum_gpu_reserved_bytes"]/2**30:.3f} GiB；并行进程耗时之和不等于墙钟或核时。每类/每 seed 耗时、GPU峰值、PID与完整日志都已保存。\n\n')
 lines.append('验证包含 26 项合成检查、固定模型重新推理与开发验证分数逐位一致、两子集独立 sklearn 指标复算、32 个 selected bootstrap macro 数值与重复分量 sample_weight 的独立核对、随机抽样索引完全再生、TSV 分数无损往返，以及 2,733 个旧产物哈希复核。最终审计结果为 passed=true。原子写临时文件验证后 rename，逐类、逐 seed、神经 epoch 与 bootstrap block 均有检查点；续跑只跳过已校验内容。\n\n')
 lines.append('未开展的可选方向：历史 GO 祖先扩展（缺冻结历史本体）、其他 ESM 层或更大 PLM、classifier chains。unknown 伪标签未使用。没有因时间不足而省略任何已登记实验或验证步骤。后续如研究这些方向，应在新版本、预先定义的验证安排及独立评价数据上开展，当前测试结果不用于继续选择。paper_negatives 仍未提供，本轮仅使用原主背景口径。\n\n')
 primary_files=['PLAN.md','PLAN_addendum_v1.md','PLAN_addendum_v2.md','experiment_results.tsv','stage_results.tsv','selection/final_selection.json','selection/inference_runtime_v2.json','revision/revision_evaluation_event.json','inference_v2/test/final_predictions.npz','revision/scores.tsv','revision/predictions.tsv','revision/prediction_index.tsv','revision/per_class.tsv','revision/summary.tsv','revision/stratified_summary.tsv','revision/stratified_per_class.tsv','revision/bootstrap/intervals.tsv','revision/bootstrap/draws.npz','validation/final_audit.json','validation/recovery_inventory.json','runtime_summary.json','schema.json']
 lines.append('主要交付文件如下，路径相对 output/exploration/。全量文件（含模型、特征、日志、缓存）的 SHA256、字节数与文本行数在 exploration/artifacts.tsv；SHA256SUMS 路径相对 output/，包含本 REPORT_turn4.md。清单自身、manifest 与 DONE 用外层校验链记录，避免自引用。\n\n')
 lines.append(mdtable(['文件','SHA256'],[[n,sha(E/n)] for n in primary_files]))
 lines.append('已完成本轮授权工作。最终选择及修订评价保持冻结，训练和调优停止。\n')
 txt(OUT/'REPORT_turn4.md',''.join(lines))

 jout(E/'logs/finalization.json',{'at':now(),'pid':os.getpid(),'action':'package existing completed results after connection recovery','no_model_execution':True,'no_new_test_event':True,'report_written':True})
 excluded={E/'artifacts.tsv',E/'SHA256SUMS',E/'manifest.json',E/'DONE.json'}
 files=sorted([p for p in E.rglob('*') if p.is_file() and p not in excluded]+[OUT/'REPORT_turn4.md'],key=lambda p:str(p.relative_to(OUT)))
 inventory=[];text_suffixes={'.py','.sh','.md','.tsv','.json','.log','.pid','.txt','.csv','.jsonl'}
 for p in files:
  assert not p.is_symlink() and p.is_relative_to(OUT),str(p)
  assert not p.name.endswith('.tmp'),str(p)
  line_count=None
  if p.suffix in text_suffixes:
   with open(p,'rb') as z:line_count=sum(1 for _ in z)
  inventory.append({'path':str(p.relative_to(OUT)),'bytes':p.stat().st_size,'sha256':sha(p),'physical_lines':line_count,'line_count_rule':'includes header' if line_count is not None else 'binary/not applicable'})
 tsv(E/'artifacts.tsv',list(inventory[0]),inventory)
 txt(E/'SHA256SUMS',''.join(r['sha256']+'  '+r['path']+'\n' for r in inventory))
 manifest={'version':'turn4_exploration_delivery_v1','created_at_utc':now(),'status':'complete','payload_files':len(inventory),'payload_bytes':sum(r['bytes'] for r in inventory),'inventory_sha256':sha(E/'artifacts.tsv'),'SHA256SUMS_sha256':sha(E/'SHA256SUMS'),'report_path':'../REPORT_turn4.md','report_sha256':sha(OUT/'REPORT_turn4.md'),'selected_candidate':selected['chosen_candidate'],'selection_sha256':sha(E/'selection/final_selection.json'),'runtime_supplement_sha256':sha(E/'selection/inference_runtime_v2.json'),'revision_event_sha256':sha(E/'revision/revision_evaluation_event.json'),'new_test_schemes':1,'overall_test_view_number':2,'posttest_tuning':False,'all_registered_experiments_complete':True,'final_audit_sha256':sha(E/'validation/final_audit.json'),'recovery_inventory_sha256':sha(E/'validation/recovery_inventory.json'),'payload_scope':'every exploration file plus REPORT_turn4.md, excluding exactly the four enclosing manifest/inventory/checksum/DONE files','inventory_exclusions':['exploration/artifacts.tsv','exploration/SHA256SUMS','exploration/manifest.json','exploration/DONE.json']}
 jout(E/'manifest.json',manifest)
 certificate(E/'DONE.json',[E/'manifest.json',E/'artifacts.tsv',E/'SHA256SUMS'],{'selection_sha256':manifest['selection_sha256'],'report_sha256':manifest['report_sha256'],'revision_event_sha256':manifest['revision_event_sha256']},{'all_work_complete':True,'test_evaluation_count_this_exploration':1})
 from verify_delivery import verify
 verify()
 log('turn4_delivery_complete',files=len(inventory),bytes=manifest['payload_bytes'],seconds=time.monotonic()-start,report_sha256=manifest['report_sha256'])

if __name__=='__main__':main()

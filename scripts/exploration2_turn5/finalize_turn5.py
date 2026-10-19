"""Document and seal validation-only research; no fitting, inference, or test access."""
from common import *
from collections import Counter

def table(head,body):return '| '+' | '.join(head)+' |\n| '+' | '.join(['---']*len(head))+' |\n'+''.join('| '+' | '.join(str(v).replace('|','/') for v in r)+' |\n' for r in body)+'\n'
def fmt(x):return 'NA' if x is None else f'{float(x):.6f}'
def percent(x):return 'NA' if x is None else f'{100*float(x):.3f}%'
def textlines(p):
 if p.suffix not in {'.json','.jsonl','.tsv','.csv','.py','.sh','.md','.txt','.log','.pid'}:return None
 with open(p,'rb') as z:return sum(1 for _ in z)
def verify_delivery():
 assert verified(E/'DONE.json');m=readj(E/'manifest.json');assert sha(OUT/'REPORT_turn5.md')==m['report_sha256']
 rr=list(rows(E/'artifacts.tsv'))
 for r in rr:
  p=OUT/r['path'];assert p.is_relative_to(OUT) and sha(p)==r['sha256'] and p.stat().st_size==int(r['bytes']),str(p)
 assert len(rr)==m['payload_files']
 assert (E/'SHA256SUMS').read_text().splitlines()==[r['sha256']+'  '+r['path'] for r in rr]
 print(json.dumps({'verified':True,'payload_files':len(rr),'report_sha256':m['report_sha256'],'new_test_evaluations':0}),flush=True)

def main():
 if (E/'DONE.json').exists():verify_delivery();return
 audit=readj(E/'audit/final_audit.json');assert audit['passed']
 proposed=readj(E/'selection/proposed_selection.json');name=proposed['chosen_candidate'];baseline=readj(E/'references/E06_mix_098/summary.json')
 if name is None:
  assert not (E/'THIRD_TEST_EVALUATION_REQUEST.md').exists() and not (E/'selection/final_selection.json').exists()
 ranking=list(rows(E/'selection/validation_leaderboard.tsv'));grid=list(rows(E/'fusion/grid_results.tsv'))
 neural=readj(E/'diagnostics/E12_summary.json')
 tree=readj(E/'models/trees/E11_summary.json')
 runtime={'at':now(),'E11_wall_seconds':tree['seconds'],'E11_class_fits':tree['total_class_fits'],'E11_failed_fits':tree['failures'],'E11_completed_at':tree['at'],'E12_summed_fit_seconds':neural['summed_fit_seconds'],'E12_fits':neural['completed_fits'],'E12_failed_fits':neural['failed_fits'],'E12_epochs':neural['total_epochs'],'selection_completed_at':proposed['at'],'resumed_closeout_reason':'quota restored by user; experimental branches and selection had already finished','new_training_on_resumed_closeout':False,'test_evaluations_this_turn':0,'E15_status':'not_applicable_no_eligible_candidate' if name is None else 'complete'}
 jout(E/'runtime_summary.json',runtime)
 jout(E/'failures_and_negative_results.json',{'at':now(),'training_failures':tree['failures']+neural['failed_fits'],'uncompleted_registered_model_configurations':[],'fusion_rules_rejected':[r['name'] for r in ranking if r['stage'] in ['E13','E14'] and r['selection_eligible']=='False'],'tree_family_improvement_threshold':.003,'E12_negative_results':neural['negative_results'],'E15_status':runtime['E15_status'],'implementation_issues':['prepare metadata symlink assertion corrected to lstat; no historical payload modified','runtime dependency CPU probes blocked by access guard; no test content opened','initial blanket history audit failed on output/_agent_dialog/turn5.log metadata; original snapshot retained, exact exception disclosed; log contents and writer unverified'],'history_metadata_exception':'audit/history_metadata_exception.json','seed_overlap_audit':'audit/neural_seed_overlap.json'})
 summaries=[readj(p) for p in sorted((E/'candidates').glob('*/summary.json'))]
 experiments=[]
 for r in summaries:
  experiments.append({'name':r['name'],'stage':r['config']['stage'],'seed':r['seed'],'validation_macro_AP':r['validation_macro_AP'],'validation_macro_F1':r['validation_macro_F1'],'crossfit_macro_AP':r.get('crossfit_macro_AP'),'eligible':r.get('selection_eligible',True),'status':'complete','configuration_json':json.dumps(r['config'],sort_keys=True,separators=(',',':'))})
 for r in grid:
  experiments.append({'name':r['name'],'stage':'E13_grid','seed':int(r['seed']),'validation_macro_AP':float(r['validation_macro_AP']),'validation_macro_F1':None,'crossfit_macro_AP':None,'eligible':False,'status':'diagnostic_grid_point','configuration_json':json.dumps({k:r[k] for k in ['space','members','weights','complexity']},sort_keys=True)})
 tsv(E/'experiment_results.tsv',list(experiments[0]),experiments)
 result=readj(E/f'candidates/{name}/summary.json') if name else None
 if name:
  frozen=readj(E/'selection/final_selection.json');assert frozen['chosen_candidate']==name and frozen['third_test_evaluation_authorized'] is False
  for p,h in frozen['files'].items():
   # Historical files were verified at freezing; final delivery only opens current files.
   if Path(p).is_relative_to(E):assert sha(p)==h,p
  bootstrap=readj(E/'bootstrap_validation/intervals.json')
  request=[]
  request.append('第三次测试评价申请书\n\n')
  request.append(f'申请时间（UTC）：{now()}。当前状态：仅提交申请，尚未授权，尚未执行。测试集此前已查看两次；Turn 5 全程只使用 train/val，本轮没有读取测试标签或生成测试预测。\n\n')
  request.append(f'申请方案：{name}。基线为已存档 E06_mix_098，基线验证 macro AP={baseline["validation_macro_AP"]:.9f}；新方案完整验证 macro AP={result["validation_macro_AP"]:.9f}，Δ={result["validation_macro_AP"]-baseline["validation_macro_AP"]:+.9f}。用户门槛0.902357已超过。\n\n')
  if result.get('crossfit_macro_AP') is not None:
   request.append(f'预登记资格检查：5折验证分量交叉检查 macro AP={fmt(result.get("crossfit_macro_AP"))}，相对固定基线 Δ={fmt(result.get("crossfit_delta"))}，通过+0.002门槛。各折的融合选择、权重、收缩参数和fallback仅使用另外四折标签；独立held-label扰动测试通过。\n\n')
  else:request.append('该方案为预登记的固定三seed模型均值，没有额外按类选择器或学习融合权重；架构仍由val选择，并不构成独立验证。\n\n')
  request.append('具体模型图、成员与权重：\n\n```json\n'+json.dumps(result['config'],ensure_ascii=False,indent=2)+'\n```\n\n')
  request.append('预处理只fit train，CDF若使用也只来自train参考分数。阈值为每类val F1最优，>=判阳、并列取高。独立从磁盘模型重算完整val与库存分数/二值决策一致。模型没有使用train+val重训。\n\n')
  boot_table=table(['子集','指标','观测Δ','普通CI状态','有效轮次','条件有效轮次区间'],[[r['subset'],r['metric'],fmt(r['observed_delta']),r['CI_status'],f"{r['valid_replicates']}/2000",f"[{fmt(r['conditional_valid_only_low'])}, {fmt(r['conditional_valid_only_high'])}]"] for r in bootstrap])
  request.append(boot_table)
  request.append('配对验证比较及2,000轮分量bootstrap详见 bootstrap_validation/intervals.tsv、point_metrics.tsv（具体文件以该目录DONE为准）。普通CI遇任一无效轮次即NA，条件有效轮次区间独立列出，不能据其宣称普通显著性。该bootstrap不消除重复使用验证集、架构选择和早停带来的选择偏差。\n\n')
  request.append('冻结清单：selection/final_selection.json 与 selection/FROZEN_SHA256SUMS。包含模型权重、train-fit预处理、32类阈值、融合权重及脚本 SHA256；历史权重以明确绝对路径和SHA256只读引用，新文件均在exploration2。\n\n')
  request.append(f'冻结选择文件SHA256：{sha(E/"selection/final_selection.json")}。最终审计：audit/final_audit.json。完整验证排行榜及包括失败方向的记录：REPORT_turn5.md、experiment_results.tsv。\n\n')
  request.append('是否打开测试集须由用户另行决定。若获批，第三次评价应另立事件，明确前两次评价已发生；一次评价固定这一套冻结方案，不据结果回到调参。本申请不启动该流程。\n')
  txt(E/'THIRD_TEST_EVALUATION_REQUEST.md',''.join(request))
 else:bootstrap=[]

 lines=['Turn 5：第二轮探索报告（仅train/val）\n\n']
 conclusion='提交新方案申请' if name else '无可超越 E06_mix_098'
 lines.append(f'结论：**{conclusion}**。该结论只针对本轮登记并完成的实验，不是所有方法的最优性证明。本轮新测试评价数为0，未读取测试标签，未生成测试预测。先前两次测试查看的事实保持记录。\n\n')
 if name:lines.append(f'最终方案 {name}：验证 macro AP={result["validation_macro_AP"]:.9f}，相对 E06_mix_098 的 ΔAP={result["validation_macro_AP"]-baseline["validation_macro_AP"]:+.9f}；交叉检查 AP={fmt(result.get("crossfit_macro_AP"))}、ΔAP={fmt(result.get("crossfit_delta"))}。模型与阈值已经冻结，申请书为 exploration2/THIRD_TEST_EVALUATION_REQUEST.md；用户未授权第三次测试。\n\n')
 else:lines.append('没有方案同时满足完整验证AP>0.902357和对应资格检查。保留原E06_mix_098，不提交第三次测试评价申请，不以失败规则的最佳网格点绕过资格要求。\n\n')
 lines.append('研究判断：上轮单模型与融合间的验证差距提示排序互补性值得检验。本轮优先投入树分割、低秩/排序损失和受约束融合；没有把扩大PLM或增加更多方向当成必做任务。方法范围先在PLAN登记，融合的确切网格在产生结果前写入PLAN_addendum_v1.md。所有负面结果保留。\n\n')
 lines.append('数据流转：\n\n')
 flow=list(rows(E/'data/flow.tsv'));lines.append(table(['集合','蛋白','分量','P','N','U','本轮排除'],[[r[k] for k in ['split','proteins','components','P','N','U','excluded']] for r in flow]))
 lines.append('开发数据仅25,027个train/val蛋白；固定32类，unknown始终mask，未变成伪负例。冻结序列、标签、划分、embedding及历史研究文件的核验通过，五项源数据问题保留原样；另有一项疑似活动会话日志的元数据变化，下文单列未决例外，不宣称全部历史文件完全未变。基线来自存档validation.npz，并独立复算AP=0.9023565789519359。历史测试内容连完整性哈希都未读取；对历史开发白名单做SHA256，对其他历史文件只核对元数据。\n\n')
 lines.append('E10 的诊断及独立核验位于 audit/E10/。已验证旧基线、ID对齐、分量隔离及固定五折；held-out的个别类别支持不足时不把逐折NA填零，而是完整OOF拼接后按固定完整val类别集合计macro。B的某些类别优于A，但分层类别集合不同，不能把不同长度层的macro差异当成因果结论。\n\n')
 lines.append('E11：30个预登记初始树配置（24 ExtraTrees、6 RandomForest），每套32类known行独立拟合。各算法固定选架构后补两个seed；每个最终候选是三seed均值，单seed没有最终资格。每类256棵树，输入C/A2B/A5B；正例权重与分割参数见configurations/E11*.json。\n\n')
 tree_rows=[r for r in summaries if r['config']['stage']=='E11' and r.get('selection_eligible',True)]
 oldtree=readj(E/'references/E04_006_C/summary.json')['validation_macro_AP']
 lines.append(table(['树均值','val AP','val F1','相对旧树AP','达到+0.003标准'],[[r['name'],fmt(r['validation_macro_AP']),fmt(r['validation_macro_F1']),fmt(r['validation_macro_AP']-oldtree),'是' if r['validation_macro_AP']-oldtree>=.003 else '否'] for r in tree_rows]))
 lines.append('E12：24个初始配置加4个补seed，512隐层、dense/rank8/rank16输出、dropout0.2/0.5、masked BCE或BCE+0.1×known正负pairwise softplus。两输入的胜者均为dense、dropout0.5、纯BCE；低秩和附加排序损失未胜出。\n\n')
 lines.append(table(['神经均值','val AP','相对对应旧均值ΔAP','达到+0.003标准'],[[r['new_candidate'],fmt(r['new_validation_macro_AP']),fmt(r['delta']),'是' if r['substantial_improvement_met'] else '否'] for r in neural['comparisons']]))
 lines.append('E12共1,495 epochs；unknown位置BCE/ranking梯度为0，ranking只抽同类已知正负；独立验证AP、输入float32表示、保存后重算以及RNG/optimizer恢复检查均通过。A2B的增幅略低于预定0.003标准，不能称为已达到该实验的实质改善门槛，但按计划仍可作为有限融合池的成员。\n\n')
 lines.append('Seed重叠审计：旧均值seed为20261006/07/08，新均值为20261007/08/09。C的共同seed07/08模型权重及验证分数逐位相同，因此新C均值只把seed06替换成seed09；两个ID不能解释为两份独立模型证据。A2B旧dropout=0.2、新dropout=0.5，共同seed的模型不同。成员复杂度沿用预登记的唯一模型ID计数，统计独立性另行披露；没有因此更换seed或重选模型。详见audit/neural_seed_overlap.json。\n\n')
 lines.append('E13：固定8成员池，两种分数空间（原始概率与train-fit经验CDF），共486个预登记网格点。仅包含至少一个新成员的两成员权重、限定三成员权重和固定等权；不在看到结果后细化网格。CDF使用全部train预测的经验中秩分布；不使用val自身分布拟合。每个网格点仅诊断，最终资格属于预登记的融合规则。E14逐类择优/收缩也完整在选择侧四折重新做，包括lambda与fallback。\n\n')
 fusion_rows=[r for r in ranking if r['stage'] in ['E13','E14']]
 lines.append(table(['规则','完整val AP','OOF AP','相对基线OOF ΔAP','有资格'],[[r['name'],fmt(r['validation_macro_AP']),fmt(r['crossfit_macro_AP']),fmt(r['crossfit_delta']),r['selection_eligible']] for r in fusion_rows]))
 lines.append('门槛为OOF macro AP相对固定E06基线增加至少0.002；固定等权也执行同一门槛。通过资格后，再按完整验证AP择优，1e-6内并列优先较少原始成员再按编号。完整验证表面最高但规则未通过者明确排除。各折权重和留出分数位于fusion/crossfit/。该交叉检查只限制融合规则额外的乐观性，基础家族/架构及早停使用过val，因此不是完全独立验证。\n\n')
 lines.append('恢复后没有直接采信控制器结论：audit/controller_independent_review.json从原始8成员验证分数重建网格，并独立复算完整/四折选择侧AP、全局选择、逐类选择与收缩、留出分数、资格和最终null结论。E13概率择优完整AP=0.902624379，OOF=0.902069745；E14收缩完整AP=0.917550113，OOF=0.899037851，均未达到固定基线+0.002。因此没有以表面分数更高为由提交申请。\n\n')
 lines.append('验证排行榜前15项（含不合资格诊断项，未按有利结果删行）：\n\n')
 lines.append(table(['候选','阶段','val AP','val F1','有资格'],[[r['name'],r['stage'],fmt(r['validation_macro_AP']),fmt(r['validation_macro_F1']),r['selection_eligible']] for r in ranking[:15]]))
 lines.append(f'完整登记 {len(experiments)} 条实验/网格记录，全部配置、seed、验证指标及状态在 exploration2/experiment_results.tsv；候选排序在 selection/validation_leaderboard.tsv，486网格的完整AP在 fusion/grid_results.tsv。格点与规则是不同层次的记录，不能把它们混计为独立验证次数。\n\n')
 if name:
  lines.append('最终方案与固定基线的可判定单元指标：\n\n')
  lines.append(table(['方案','macro AP','macro F1','micro P','micro R','micro F1','全负背景any-label FPR','unknown阳性占比'],[[label]+[fmt(s['summary'][k]) for k in ['macro_AP','macro_F1','micro_precision','micro_recall','micro_F1']]+[percent(s['summary']['any_label_FPR']),percent(s['summary']['unknown_positive_fraction'])] for label,s in [('E06_mix_098',baseline),(name,result)]]))
  lines.append('unknown预测阳性只做描述，不算TP/FP/TN/FN，也没有用其数量拟合伪负例目标。可判定单元上的precision不等价于未知单元上的准确率。阈值在完整val上按类F1选择，OOF AP资格与阈值无关。完整val/共同可用注释子集的每类P/N/U、K、汇总及paired bootstrap都在 bootstrap_validation/。\n\n')
  lines.append('E15以seed20261010有放回抽1,398个验证分量，固定尝试2,000轮；两模型、完整val与原可用注释子集共用同轮抽样，重复分量按次数计权，原macro类别集合保持不变。普通CI如有任一无效轮次则NA，报告有效次数/涉及类别，条件有效区间另列。逐轮固定模型区间不包含训练、架构和重复验证选择的不确定性。\n\n')
  lines.append(boot_table)
  lines.append('配对bootstrap的全部点差、普通区间状态和条件区间：exploration2/bootstrap_validation/intervals.tsv。逐轮数据及独立sklearn sample_weight核对：audit/E15_bootstrap_checks.json。\n\n')
 else:lines.append('E15的新方案重算/配对bootstrap没有启动，因为没有符合申请门槛的候选；不是省略合资格方案的验收步骤。此前登记的bootstrap实现及synthetic检查保留。\n\n')
 lines.append('完整性与可续跑：每类/每seed/每epoch/每融合网格块及bootstrap块先临时写入、验证、atomic rename，再写DONE；神经checkpoint含全部优化器与随机数状态。独立审计检查mask、NA、ties、退化支持和held-fold扰动，以及完整验证重算。原数据无新增排除，所有模型覆盖原ID顺序。\n\n')
 lines.append(f'最终研究审计通过，历史只读检查附有一个未决例外：{audit["historical_files_metadata_checked"]:,} 个历史文件中，{audit["historical_files_metadata_unchanged"]:,} 个元数据未变；{audit["development_source_hashes_verified"]} 个开发源哈希核验通过，{audit["certificates_verified"]:,} 份完成证书校验通过；{audit["candidate_AP_independently_recomputed"]} 个候选AP独立复算。访问守卫通过开发白名单拒绝历史测试/mixed-label内容；它是Python审计层，不能冒充OS沙箱，另有脚本读路径审阅。守卫测试的拒绝记录均在打开payload前发生。\n\n')
 lines.append('历史元数据例外：首次最终审计因 output/_agent_dialog/turn5.log 的大小和修改时间变化而失败。原始快照未改写，随后仅对这个完整路径登记显式例外；其余历史文件仍严格检查，无研究输入被豁免。该路径疑似活动会话日志，但没有打开内容，也未核验写入来源，不能归因于某个外部进程或宣称历史只读检查无条件通过。具体前后元数据及范围见 audit/history_metadata_exception.json；审计状态为 passed_with_unresolved_session_log_metadata_exception。这一例外不改变候选资格或负面结论。\n\n')
 lines.append('环境复用工作目录内mamba的seqscreen_train和seqscreen_esm：scikit-learn1.7.2、numpy1.26.4、scipy1.17.1、numba0.61.2、Torch2.5.1/CUDA12.6。所有缓存/临时文件在exploration2/cache；CPU树最多40个单线程worker，神经使用4张H100。没有新增依赖安装或权重下载。每类和每seed的时间、GPU峰值、PID、失败/重试记录保存在models/、diagnostics/和logs/。\n\n')
 lines.append(f'E11完成{tree["total_class_fits"]}个类拟合，壁钟{tree["seconds"]:.1f}秒（约{tree["seconds"]/3600:.3f}小时）；E12完成{neural["completed_fits"]}个拟合，进程拟合时间之和{neural["summed_fit_seconds"]:.1f}秒。训练失败数为0，已登记模型配置没有未完成项。实验选择于{proposed["at"]}结束；随后受配额中断影响的报告收尾按用户恢复指令继续，未新增训练、调参或测试。进程时间之和不等同于墙钟，配额等待也不计为算力。\n\n')
 lines.append('问题与未开展项：预备元数据盘点曾对旧conda缓存软链接触发过保守断言，已改为lstat仅记录链接本身，未读取测试payload；该重试保留在prepare.log。未使用更大PLM、额外ESM层、GO历史本体扩展或更多未登记分支，这是本轮按证据与成本作出的范围选择，不是这些方向已被证明无效。五项源问题未重新审核。后续若继续研究，需要新计划；当前交付不授权第三次测试。\n\n')
 keyfiles=['PLAN.md','PLAN_addendum_v1.md','experiment_results.tsv','selection/validation_leaderboard.tsv','selection/proposed_selection.json','fusion/grid_results.tsv','fusion/E13_rules.json','fusion/E14_rules.json','audit/controller_independent_review.json','audit/neural_seed_overlap.json','audit/history_metadata_exception.json','audit/final_audit.json','runtime_summary.json','failures_and_negative_results.json']
 if name:keyfiles+=['selection/final_selection.json','selection/FROZEN_SHA256SUMS','THIRD_TEST_EVALUATION_REQUEST.md','bootstrap_validation/intervals.tsv']
 lines.append('主要交付校验和（相对exploration2/）：\n\n'+table(['文件','SHA256'],[[p,sha(E/p)] for p in keyfiles]))
 lines.append('全量文件SHA256、字节数及文本行数见 exploration2/artifacts.tsv；SHA256SUMS路径相对output/并包含REPORT_turn5.md。外层manifest/DONE记录清单校验和，避免自引用。已停止探索，等待用户复核。\n')
 txt(OUT/'REPORT_turn5.md',''.join(lines))
 jout(E/'schema.json',{'version':'turn5_validation_only','test_data_allowed':False,'arrays':'All protein_id arrays restricted to original train/val IDs. Candidate validation.npz contains fixed32 class order, continuous scores, per-class thresholds; no test predictions.','labels':'1 positive,0 negative,-1 unknown; unknown always masked','candidate_eligibility':'E11/E12 fixed three-seed mean; E13/E14 rule OOF AP >= E06 AP+.002; final proposal additionally full val AP>.902357','model_freeze':'absolute immutable model/preprocessing/script paths plus SHA in selection/final_selection.json; historical files read only','inventory':'all exploration2 files plus REPORT_turn5.md; excludes artifacts.tsv,SHA256SUMS,manifest.json,DONE.json; physical text lines include header; binary NA','bootstrap':'2000 val-component draws,seed20261010,paired and fixed K; ordinaryCI NA if any invalid replicate','third_test_authorized':False})
 txt(E/'README.md','Turn 5 完整报告：../REPORT_turn5.md\n\n本目录只含train/val研究，不含本轮测试评价。申请文件不是测试授权。\n\n完整实验表：experiment_results.tsv；排行榜：selection/validation_leaderboard.tsv；模型冻结如有则见selection/final_selection.json。\n\n只校验交付（不训练/推理）：source scripts/env.sh 后运行既有seqscreen_train/bin/python -B scripts/finalize_turn5.py；已有DONE时仅核验全部清单。所有路径及缺失值见schema.json。\n')
 txt(E/'STATE.md',f'状态：Turn 5 已完成并停止。\n\n结论：{conclusion}。最终报告 ../REPORT_turn5.md。\n\nE11/E12及融合控制器已完成；恢复后已独立核验控制器实际产物，随后运行最终审计。没有新的测试读取或评价。\n\n选择状态见selection/proposed_selection.json；交付校验链见manifest.json、artifacts.tsv、SHA256SUMS、DONE.json。\n\n已有根DONE时，scripts/finalize_turn5.py只核验交付，不训练或推理。任何新探索或第三次测试都需要用户后续指令。\n')
 jout(E/'logs/finalize.json',{'at':now(),'pid':os.getpid(),'status':'complete','new_test_evaluations':0,'conclusion':conclusion})
 exclude={E/n for n in ['artifacts.tsv','SHA256SUMS','manifest.json','DONE.json']}
 files=sorted([p for p in E.rglob('*') if p.is_file() and p not in exclude]+[OUT/'REPORT_turn5.md'],key=str)
 inventory=[]
 for p in files:
  assert not p.is_symlink() and not p.name.endswith('.tmp'),str(p)
  inventory.append({'path':str(p.relative_to(OUT)),'bytes':p.stat().st_size,'sha256':sha(p),'physical_lines':textlines(p)})
 tsv(E/'artifacts.tsv',list(inventory[0]),inventory)
 txt(E/'SHA256SUMS',''.join(r['sha256']+'  '+r['path']+'\n' for r in inventory))
 m={'created_at_utc':now(),'status':'complete','audit_status':audit['status'],'historical_metadata_exception_count':audit['historical_metadata_exception_count'],'conclusion':conclusion,'selected_candidate':name,'test_evaluations_this_turn':0,'third_test_authorized':False,'payload_files':len(inventory),'payload_bytes':sum(r['bytes'] for r in inventory),'report_sha256':sha(OUT/'REPORT_turn5.md'),'inventory_sha256':sha(E/'artifacts.tsv'),'SHA256SUMS_sha256':sha(E/'SHA256SUMS'),'audit_sha256':sha(E/'audit/final_audit.json'),'historical_test_content_opened':False}
 jout(E/'manifest.json',m)
 certificate(E/'DONE.json',[E/'manifest.json',E/'artifacts.tsv',E/'SHA256SUMS'],{'report_sha256':m['report_sha256'],'test_evaluations':0})
 verify_delivery()
if __name__=='__main__':main()

from common import *
import subprocess
def main():
    p=T/'preregistration/addendum.json'
    if p.exists():
        assert verified(p.parent),'existing preregistration is not valid';print('verified preregistration exists');return
    protocol={
      'version':'v2a_turn13_conservative_1','registered_at':now(),'seed':SEED,
      'scope':{'train':18770,'val':6257,'excluded_new702':702,'retain_old_ids_no_dedup':True,'test_access':False},
      'authorization':'Turn13 explicit user addendum; old dev only; rule A priority; residual conflicts retained',
      'variants':{'V0':'reference v2026 P/N; U masked','V1':'not_applicable if no_blindcheck=0; do not duplicate training'},
      'source_block_isolation':'Positive-side dev must be within unique positive blocks; background dev is independently unique in background v4. 25027 is not the size of the positive-side universe.',
      'stages':'Rebuild all four counts for dev alone; do not reuse 25729-protein Turn12 totals.',
      'rule_A':'Apply approved priority mask; retain both before/after arrays for descriptive sensitivity; do not train an unregistered competing label policy.',
      'known_noise':'Retain residual same-sequence P/N and all old identifiers incl dedup/manual-check metadata; no adjudication/merging.',
      'hard_stop':['any dev duplicated positive cell','any dev positive review_2026/augmented provenance','stage cell mismatch','feature mismatch','historical mutation'],
      'candidates':['M00','M01','M02','M03','M04','M05','M06'],'extra_candidates':[],
      'train_fold':'int(sha256(20261012:+lexical_minimum_member)[:16],16)%5',
      'early_stop_fold':'independent namespace: int(sha256(20261012:early_stop:+lexical_minimum_member)[:16],16)%10==0; prevents coupling modulo5/modulo10',
      'mlp':{'seeds':[20261006,20261007,20261008],'batch':1024,'max_epoch':120,'checks':[1,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120],'patience_checks':6,'improvement':1e-6,'lr':.001,'weight_decay':.0001,'grad_clip':5,'FP32':True,'TF32':False,'early_stop_refit':True,'final_epoch':'median of five outer early-stop epochs; exactly integer','OOM':'stop failed branch; do not silently change locked batch'},
      'M05':'deduplicated global convex grid: M00 vs each M01-M04, alternative weight 0/.25/.5/.75/1; other-four val folds choose pooled known macro AP; ties fewer primitives then lexical grid id',
      'M06':'per-class same grid chosen on other-four known labels; 50% shrink to selection-side M05; missing P/N falls back M00',
      'selection':'M01-M04 are fixed train-OOF-selected methods; M05/M06 crossfit whole selection rule. All candidates must beat M00 by crossfit AP >=.002 and fullval AP>baseline. In addition the family-selection procedure is crossfit (other-four folds choose candidate); require >=.002 for claiming the final selected family. If gate fails retain M00.',
      'threshold':'each class maximize F1 on known val; score>=threshold, ties highest; unique known scores + nextafter(max,+inf). Crossfit thresholds on other-four folds; old E06 uses its archived thresholds unchanged.',
      'bootstrap':{'rounds':2000,'seed':SEED,'unit':'frozen validation component','invalid_policy':'any invalid draw => main interval NA; conditional interval supplemental'},
      'model_reproduction':'verify saved models with new inference; no train+val refit; no updated E06 threshold in historical anchor',
      'source_protocol_sha256':sha(V/'PLAN_v2.md'),'candidate_config_sha256':sha(V/'preregistration/candidates.json'),
      'input_label_draft_sha256':sha(V/'data/labels_dev_draft.npz'),
    }
    js(p,protocol)
    write_text(p.parent/'ADDENDUM_v2a.md',f'''# Turn13 看分数前范围附录\n\nUTC：{protocol['registered_at']}\n\n按用户明确授权，旧 dev 25,027 条（train 18,770 / val 6,257）为唯一宇宙，702 全排除。只运行 V0；no_blindcheck 为0时 V1 为 not_applicable。规则A优先mask获准；残余同序列冲突、旧AC重复/身份待核条目保留原样，不合并、不重新裁决。Turn12根封存不改。\n\n预检分别证明阳性侧来自非重复块、背景侧唯一、dev与702无交集；不能把背景侧错误要求为阳性表成员。所有阶段账按dev重新投影。任何真实B01/B02污染、矩阵不一致、特征错位或历史改动均停止。\n\nM00–M06严格沿用PLAN；不增加新家族。内层早停hash使用独立early_stop命名空间，避免外层mod5与内层mod10耦合造成空内层。所有折内词表/标准化只fit本训练侧，最终epoch取五折中位数。固定超参详见addendum.json和原PLAN。\n\n除各家族交叉资格门外，对最终“选家族”规则也用另外四个val折选再在留出折拼接检查；未达+0.002则保留M00。此加严登记在任何v2分数计算前。阈值不依据unknown报警调整。规则A只做固定模型前后mask描述性敏感性，不训练另一标签版本。旧E06仅用已存dev分数/阈值重评。\n\nGPU仅使用现场核实空闲卡，不影响已有进程。所有产物、日志、缓存位于本目录，测试载荷从不打开。可续跑只跳过哈希验证通过的DONE。\n''')
    js(T/'audit/history_before.json',history_snapshot())
    gp=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True,check=True)
    ap=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],capture_output=True,text=True,check=True)
    js(T/'env/gpu_preflight.json',{'at':now(),'gpu':gp.stdout,'applications':ap.stdout,'planned_gpus':[2,3],'reason':'GPU1 busy; GPU0 also has a resident foreign context, leave both untouched'})
    done(p.parent,[p,p.parent/'ADDENDUM_v2a.md'],scores_seen=False)
    print('Registered',sha(p))
if __name__=='__main__':main()

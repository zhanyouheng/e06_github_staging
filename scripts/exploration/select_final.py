from common import *
from experiments import register_candidate
from metrics import class_sets
from sklearn.metrics import average_precision_score
import itertools

def all_summaries():return {p.parent.name:readj(p) for p in (E/'candidates').glob('*/summary.json')}
def family(cfg):
 st=cfg['stage'];f=cfg.get('feature','')
 if st=='E01':return 'linear_'+f
 if st=='E02':return 'rare_annotation' if f in ['A2','A5','A2B','A5B'] else 'historical_text'
 if st=='E03':return 'kernel_joint' if f.startswith('A') else 'kernel_B'
 if st=='E04':return 'trees_'+f
 if st=='E07':return 'component_weight_'+f
 if st=='E08':return 'joint_normalization' if f.startswith('C') else 'embedding_normalization'
 if st in ['E05','E09']:return 'MLP_'+f
 return st
def candidate_scores(name):
 with np.load(E/f'candidates/{name}/validation.npz',allow_pickle=False) as z:return z['scores']
def ap_columns(y,s):
 a=[]
 for j in range(y.shape[1]):
  k=y[:,j]!=-1
  a.append(float(average_precision_score(y[k,j],s[k,j])) if np.any(y[:,j]==1) and np.any(y[:,j]==0) else np.nan)
 return np.array(a)
def complexity(name,summaries):
 cfg=summaries[name]['config'];algorithm=cfg['algorithm']
 if algorithm=='ensemble':return sum(complexity(n,summaries) for n in cfg['members'])
 if algorithm=='class_selector':return sum(complexity(n,summaries) for n in set(cfg['members_by_class']))
 return 1
def pick_best(names,summaries):
 mx=max(summaries[n]['validation_macro_AP'] for n in names);ties=[n for n in names if mx-summaries[n]['validation_macro_AP']<=1e-6]
 return min(ties,key=lambda n:(complexity(n,summaries),n))
def selector(pool,fallback,guarded,fold,y,groups,scores):
 def choose(mask):
  aps=np.array([ap_columns(y[mask],scores[n][mask]) for n in pool]);choices=[];fi=pool.index(fallback)
  for j in range(32):
   valid=np.isfinite(aps[:,j]);best=fi
   if valid.any():
    mx=np.nanmax(aps[:,j]);best=next(i for i in range(len(pool)) if valid[i] and mx-aps[i,j]<=1e-6)
   if guarded:
    g=len(set(groups[mask&(y[:,j]==1)]))
    if g<5 or not np.isfinite(aps[fi,j]) or aps[best,j]<aps[fi,j]+0.01:best=fi
   choices.append(pool[best])
  return choices
 full=choose(np.ones(len(y),bool));score=np.column_stack([scores[n][:,j] for j,n in enumerate(full)])
 oof=np.empty_like(score);fold_choices=[]
 for k in range(5):
  held=fold==k;choices=choose(~held);fold_choices.append({'fold':k,'held_proteins':int(held.sum()),'members_by_class':choices})
  for j,n in enumerate(choices):oof[held,j]=scores[n][held,j]
 ap=float(np.mean(ap_columns(y,oof)));reference=float(np.mean(ap_columns(y,scores[fallback])));eligible=ap>=reference+0.002
 return full,score,oof,{'crossfit_macro_AP':ap,'reference_model':fallback,'reference_macro_AP':reference,'crossfit_delta':ap-reference,'eligibility_margin':0.002,'selection_eligible':eligible,'guarded':guarded,'fold_choices':fold_choices}
def main():
 no_test_guard();d=dev();summaries=all_summaries();base={n:s for n,s in summaries.items() if s['config']['stage']!='E06'}
 expected={'E01':30,'E02':60,'E03':18,'E04':12,'E07':18,'E08':40,'E05':14,'E09':10}
 for stage,count in expected.items():assert sum(s['config']['stage']==stage for s in base.values())==count,(stage,'incomplete experiment grid')
 eligible=[n for n,s in base.items() if s.get('selection_eligible',True)];assert eligible
 for n in base:assert verified(E/f'candidates/{n}/DONE.json')
 scores={n:candidate_scores(n) for n in eligible};fam={}
 for n in eligible:fam.setdefault(family(base[n]['config']),[]).append(n)
 winners={f:pick_best(ns,base) for f,ns in fam.items()};global_best=pick_best(list(winners.values()),base)
 jout(E/'selection/family_winners.json',{'winners':winners,'best_global_single_model':global_best,'test_used':False,'at':now()})
 # Component fold IDs are deterministic; the original train/val/test assignment is unchanged.
 group_fold={g:int.from_bytes(hashlib.sha256(f'{SEED}:selector:{g}'.encode()).digest()[:8],'big')%5 for g in set(d['val_groups'])}
 fold=np.array([group_fold[g] for g in d['val_groups']],np.int8);npz(E/'selection/validation_selector_folds.npz',protein_id=d['val_ids'],component_id=d['val_groups'],fold=fold)
 configs=[];crossfit_records=[]
 for f in ['A','B','C']:
  pool=sorted(n for n,s in base.items() if s['config']['stage']=='E01' and s['config']['feature']==f);fallback=pick_best(pool,base)
  full,sc,oof,check=selector(pool,fallback,False,fold,d['val_y'],d['val_groups'],scores)
  name=f'E06_class_C_{f}';cfg={'name':name,'stage':'E06','algorithm':'class_selector','members_by_class':full,'pool':pool,'seed':SEED,'order':len(configs)}
  if not verified(E/f'candidates/{name}/DONE.json'):register_candidate(name,cfg,sc,[E/f'candidates/{n}/DONE.json' for n in sorted(set(full))],check)
  npz(E/f'selection/{name}_crossfit.npz',protein_id=d['val_ids'],scores=oof,fold=fold);configs.append(cfg);crossfit_records.append({'name':name,**check})
 pool=sorted(set(winners.values())|{'E01_022_C'})
 for guarded in [False,True]:
  full,sc,oof,check=selector(pool,global_best,guarded,fold,d['val_y'],d['val_groups'],scores)
  name='E06_class_family_'+('guarded' if guarded else 'free');cfg={'name':name,'stage':'E06','algorithm':'class_selector','members_by_class':full,'pool':pool,'guarded':guarded,'seed':SEED,'order':len(configs)}
  if not verified(E/f'candidates/{name}/DONE.json'):register_candidate(name,cfg,sc,[E/f'candidates/{n}/DONE.json' for n in sorted(set(full))],check)
  npz(E/f'selection/{name}_crossfit.npz',protein_id=d['val_ids'],scores=oof,fold=fold);configs.append(cfg);crossfit_records.append({'name':name,**check})
 # Global mixtures use family winners, not the full set of hundreds of fits.
 top=sorted(set(winners.values()),key=lambda n:(-base[n]['validation_macro_AP'],complexity(n,base),n))[:5];mixtures=[]
 for a,b in itertools.combinations(top,2):
  for w in range(1,10):mixtures.append(([a,b],[w/10,1-w/10]))
 for i in range(1,9):
  for j in range(1,10-i):
   k=10-i-j
   if k>0:mixtures.append((top[:3],[i/10,j/10,k/10]))
 mixtures.extend([(top,[1/len(top)]*len(top)),(top[:3],[1/3]*3)])
 assert len(mixtures)<=250
 for index,(members,weights) in enumerate(mixtures):
  name=f'E06_mix_{index:03d}';cfg={'name':name,'stage':'E06','algorithm':'ensemble','members':members,'weights':weights,'seed':SEED,'order':len(configs)}
  if not verified(E/f'candidates/{name}/DONE.json'):
   sc=sum(w*scores[n] for n,w in zip(members,weights));register_candidate(name,cfg,sc,[E/f'candidates/{n}/DONE.json' for n in members],{'selection_eligible':True})
  configs.append(cfg)
 jout(E/'configurations/E06.json',configs);jout(E/'selection/crossfit_selector_checks.json',crossfit_records)
 summaries=all_summaries();eligible=[n for n,s in summaries.items() if s.get('selection_eligible',True)];chosen=pick_best(eligible,summaries)
 leaderboard=[]
 for n,s in sorted(summaries.items(),key=lambda q:(-q[1]['validation_macro_AP'],q[0])):
  leaderboard.append({'name':n,'stage':s['config']['stage'],'algorithm':s['config']['algorithm'],'feature':s['config'].get('feature','derived'),'seed':s['seed'],'macro_AP':s['validation_macro_AP'],
   'macro_F1_per_class_threshold':s['validation_macro_F1'],'macro_F1_shared_threshold':s['validation_shared_macro_F1'],'any_label_FPR_per_class_threshold':s['any_label_FPR'],'selection_eligible':s.get('selection_eligible',True),
   'crossfit_macro_AP':s.get('crossfit_macro_AP'),'complexity_members':complexity(n,summaries)})
 tsv(E/'selection/validation_leaderboard.tsv',list(leaderboard[0]),leaderboard)
 closure=set()
 def visit(n):
  if n in closure:return
  closure.add(n);cfg=summaries[n]['config']
  if cfg['algorithm']=='ensemble':
   for member in cfg['members']:visit(member)
  elif cfg['algorithm']=='class_selector':
   for member in cfg['members_by_class']:visit(member)
 visit(chosen);files=set()
 for n in closure:
  cp=E/f'candidates/{n}/DONE.json';assert verified(cp);files.add(cp)
  for rel in readj(cp)['files']:files.add(E/rel)
 # Freeze all train-fitted feature definitions/transforms needed by the chosen graph.
 used_features=set()
 def add_feature(name):
  if name in used_features:return
  used_features.add(name);fd=readj(E/f'features/{name}/definition.json');cp=E/f'features/{name}/DONE.json';assert verified(cp);files.add(cp)
  for rel in readj(cp)['files']:files.add(E/rel)
  for sub in fd.get('blocks',[]):add_feature(sub)
  if fd.get('type')=='nystroem':add_feature(fd['source'])
  if fd.get('type')=='embedding_length':add_feature(fd['base'])
 for n in closure:
  cfg=summaries[n]['config']
  if cfg['algorithm'] not in ['ensemble','class_selector']:add_feature(cfg['feature'])
 for p in [E/'PLAN.md',E/'PLAN_addendum_v1.md',E/'PLAN_addendum_v2.md',E/'data/dev.npz',E/'selection/validation_leaderboard.tsv',E/'selection/family_winners.json',E/'selection/crossfit_selector_checks.json',E/'selection/validation_selector_folds.npz']:
  files.add(p)
 # Core learning/inference definitions already exist; final evaluator is separate.
 for n in ['common.py','experiments.py','metrics.py','train_gpu.py','group_weighting.py','select_final.py','inference.py','infer_gpu.py']:files.add(S/n)
 decision={'frozen_at_utc':now(),'chosen_candidate':chosen,'reason':'highest eligible validation macro AP; ties within1e-6 prefer fewer primitive members then lexical registration ID; selector eligibility follows preregistered component crossfit guard',
  'validation_macro_AP':summaries[chosen]['validation_macro_AP'],'validation_macro_F1':summaries[chosen]['validation_macro_F1'],'thresholds':summaries[chosen]['thresholds'],
  'class_order':d['class_order'].tolist(),'threshold_policy':summaries[chosen]['threshold_policy'],'scheme':summaries[chosen]['config'],'candidate_closure':sorted(closure),'feature_closure':sorted(used_features),
  'candidate_count':len(summaries),'eligible_candidates':len(eligible),'ensemble_grid_size':len(mixtures),'files':{str(p.relative_to(E)):sha(p) for p in sorted(files)},
  'test_used_for_selection':False,'prior_test_was_evaluated_in_turn3':True,'new_test_evaluation_count':0,'source_labels_or_splits_modified':False,'post_freeze_training_forbidden':True}
 jout(E/'selection/final_selection.json',decision)
 for p in files|{E/'selection/final_selection.json'}:p.chmod(0o444)
 log('FINAL_MODEL_FROZEN',candidate=chosen,AP=decision['validation_macro_AP'],candidate_count=len(summaries),closure=sorted(closure))
if __name__=='__main__':main()

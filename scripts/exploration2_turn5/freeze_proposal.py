from access_guard import install
install()
from common import *
import shutil

def summary_path(n):
 for p in [E/f'candidates/{n}/summary.json',E/f'references/{n}/summary.json',OLD/f'candidates/{n}/summary.json']:
  if p.exists():return p
 raise ValueError(n)

def main():
 target=E/'selection/final_selection.json'
 if target.exists():
  z=readj(target)
  for p,h in z['files'].items():assert sha(Path(p))==h,p
  log('proposal_already_frozen',candidate=z['chosen_candidate']);return
 proposed=readj(E/'selection/proposed_selection.json');name=proposed['chosen_candidate'];assert name is not None
 summary=readj(E/f'candidates/{name}/summary.json');d=dev()
 rec=E/f'recompute/{name}/val.DONE.json';assert verified(rec),'independent model recompute not complete'
 boot=E/'bootstrap_validation/DONE.json';assert verified(boot),'paired validation bootstrap not complete'
 assert readj(boot)['context']['candidate']==name,'bootstrap belongs to another candidate'
 assert summary.get('selection_eligible',True) and summary['validation_macro_AP']>.902357
 files=set();graph={};specs=readj(E/'configurations/E13_grid.json')
 def add_file(p):
  p=Path(p);assert p.is_relative_to(E) or p.is_relative_to(OLD) or p==BASE/'features/B_preprocessing.npz' or p==BASE/'features/vocabulary.tsv'
  assert p.exists() and not p.is_symlink(),str(p);files.add(p)
 def node(n):
  if n in graph:return
  p=summary_path(n);s=readj(p);c=s['config'];graph[n]=c;add_file(p)
  if p.is_relative_to(OLD):
   base=OLD
  elif p.is_relative_to(E/'references'):
   base=OLD
  else:base=E
  if c['algorithm']=='global_fusion':members=c['members']
  elif c['algorithm']=='class_fusion':
   indices=set(c['class_grid_indices']+([c['global_grid_index']] if c['shrink'] else []));members=sorted(set(m for i in indices for m in specs[i]['members']))
  elif 'members' in c:members=c['members']
  else:members=[]
  for m in members:node(m)
  if not members:
   if c['algorithm'] in ['extra_trees','random_forest']:
    path=E/f'models/trees/{n}' if base==E else OLD/f'experiments/{n}'
    for j in range(32):add_file(path/f'class_{j:02d}.pkl');add_file(path/f'class_{j:02d}.json')
   elif c['algorithm'] in ['mlp','masked_mlp']:
    path=E/f'models/neural/{n}' if base==E else OLD/f'gpu/{n}'
    add_file(path/'best.pt');add_file(path/'history.json')
   else:raise ValueError(c['algorithm'])
 node(name)
 features=sorted({c['feature'] for c in graph.values() if 'feature' in c})
 for f in sorted(set(features)|{'A','A2','A5','B','C','A2B','A5B'}):
  add_file(E/f'features/{f}/definition.json');add_file(E/f'features/{f}/source_files.json')
 for source,dest in [(BASE/'features/B_preprocessing.npz',E/'preprocessing/B_preprocessing.npz'),(BASE/'features/vocabulary.tsv',E/'preprocessing/A_vocabulary.tsv')]:
  if not dest.exists():
   q=tmp(dest);shutil.copyfile(source,q);assert sha(q)==sha(source);commit(q,dest)
  assert sha(dest)==sha(source);add_file(dest)
 jout(E/'preprocessing/class_order.json',d['class_order'].tolist());add_file(E/'preprocessing/class_order.json')
 for p in (E/'fusion/cdf').glob('*.npz'):add_file(p)
 for f in ['PLAN.md','PLAN_addendum_v1.md','data/dev.npz','data/source_manifest.json','data/validation_selector_folds.npz','configurations/E13_grid.json','fusion/pool.json','fusion/E13_rules.json','fusion/E14_rules.json','selection/validation_leaderboard.tsv','selection/proposed_selection.json']:
  add_file(E/f)
 for p in S.glob('*.py'):
  if p.name!='finalize_turn5.py':add_file(p)
 add_file(S/'env.sh');add_file(rec);add_file(boot)
 for f in ['summary.json','validation.npz','per_class.tsv','DONE.json']:add_file(E/f'candidates/{name}/{f}')
 record={'version':'turn5_validation_only_proposal_v1','frozen_at_utc':now(),'chosen_candidate':name,'baseline':'E06_mix_098','validation_macro_AP':summary['validation_macro_AP'],'validation_macro_F1':summary['validation_macro_F1'],'crossfit_macro_AP':summary.get('crossfit_macro_AP'),'crossfit_delta':summary.get('crossfit_delta'),'class_order':d['class_order'].tolist(),'thresholds':summary['thresholds'],'threshold_policy':'per-class validation F1, ties higher, score>=threshold','model_graph':graph,'feature_names':features,'test_evaluated_this_turn':False,'third_test_evaluation_authorized':False,'status':'frozen_for_user_review_only','files':{str(p):sha(p) for p in sorted(files)},'validation_only_caveat':'repeated validation use; selector crossfit does not remove architecture/early-stop selection optimism','source_issues_unchanged':True}
 jout(target,record)
 checksum=E/'selection/FROZEN_SHA256SUMS';txt(checksum,''.join(h+'  '+p+'\n' for p,h in record['files'].items())+sha(target)+'  '+str(target)+'\n')
 # History remains untouched, including permissions. Only this round's selected files become readonly.
 for p in files|{target,checksum}:
  if p.is_relative_to(E):p.chmod(p.stat().st_mode & ~0o222)
 log('new_proposal_frozen',candidate=name,validation_AP=record['validation_macro_AP'],test_authorized=False)
if __name__=='__main__':main()

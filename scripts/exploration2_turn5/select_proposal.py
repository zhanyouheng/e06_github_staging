from access_guard import install
install()
from common import *
from fusion import leaves

def main():
 no_test_guard()
 for p in [E/'models/trees/DONE.json',E/'models/neural/DONE.json',E/'fusion/E13_rules.json',E/'fusion/E14_rules.json']:
  assert p.exists(),('registered stage incomplete',str(p))
 summaries=[readj(p) for p in sorted((E/'candidates').glob('*/summary.json'))]
 reference=readj(E/'references/E06_mix_098/summary.json');baseap=reference['validation_macro_AP'];records=[]
 for r in summaries:
  cfg=r['config'];eligible=r.get('selection_eligible',True)
  if cfg['stage'] in ['E13','E14']:
   assert eligible==(r['crossfit_macro_AP']>=baseap+.002)
  records.append({'name':r['name'],'stage':cfg['stage'],'algorithm':cfg['algorithm'],'seed':r['seed'],'validation_macro_AP':r['validation_macro_AP'],'validation_macro_F1':r['validation_macro_F1'],'shared_macro_F1':r['validation_shared_macro_F1'],'crossfit_macro_AP':r.get('crossfit_macro_AP'),'crossfit_delta':r.get('crossfit_delta'),'selection_eligible':eligible,'beats_user_threshold':r['validation_macro_AP']>.902357,'unknown_positive_fraction':r['summary']['unknown_positive_fraction'],'any_label_FPR':r['summary']['any_label_FPR'],'summary_sha256':sha(E/f"candidates/{r['name']}/summary.json")})
 records.sort(key=lambda r:(-r['validation_macro_AP'],r['name']))
 tsv(E/'selection/validation_leaderboard.tsv',list(records[0]),records)
 pool=[r for r in records if r['selection_eligible'] and r['beats_user_threshold']]
 if pool:
  best=max(r['validation_macro_AP'] for r in pool);ties=[r for r in pool if best-r['validation_macro_AP']<=1e-6]
  # E14 graph complexity is calculated from frozen grid dependencies.
  def complexity(r):
   c=readj(E/f"candidates/{r['name']}/summary.json")['config']
   if c['algorithm']=='class_fusion':
    specs=readj(E/'configurations/E13_grid.json');nodes=set(c['class_grid_indices']+([c['global_grid_index']] if c['shrink'] else []));members=set(n for i in nodes for n in specs[i]['members']);return len(set().union(*(leaves(n) for n in members)))
   if 'members' in c:return len(set().union(*(leaves(n) for n in c['members'])))
   return 1
  chosen=min(ties,key=lambda r:(complexity(r),r['name']))
  status='proposal_pending_validation_recompute_and_bootstrap'
 else:chosen=None;status='no_eligible_improvement'
 jout(E/'selection/proposed_selection.json',{'at':now(),'status':status,'baseline':'E06_mix_098','baseline_validation_macro_AP':baseap,'user_threshold':.902357,'candidate_records':len(records),'eligible_candidates':sum(r['selection_eligible'] for r in records),'qualifying_improvements':len(pool),'chosen_candidate':None if chosen is None else chosen['name'],'chosen_evidence':chosen,'selection_uses_test':False,'new_test_evaluation_authorized':False,'next_step':'independent val recompute, paired val bootstrap, freeze and request only' if chosen else 'report no eligible improvement and stop','tie_rule':'AP tolerance1e-6 then unique primitive members then candidate ID','leaderboard_sha256':sha(E/'selection/validation_leaderboard.tsv')})
 log('proposal_selected',status=status,candidate=None if chosen is None else chosen['name'])
if __name__=='__main__':main()

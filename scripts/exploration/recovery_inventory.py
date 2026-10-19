"""Read-only checkpoint audit after connection loss; never resumes model selection."""
from common import *
from collections import Counter

start=time.monotonic()
cache={}
def checked_sha(p):
 p=Path(p)
 if p not in cache:cache[p]=sha(p)
 return cache[p]

certs=[]
for p in sorted(E.rglob('*DONE.json')):
 d=readj(p)
 assert isinstance(d.get('files'),dict),str(p)
 for name,h in d['files'].items():
  q=E/name
  assert q.is_relative_to(E) and not q.is_symlink(),str(q)
  assert checked_sha(q)==h,('certificate mismatch',str(p),name)
 certs.append({'certificate':str(p.relative_to(E)),'files':len(d['files']),'status':'verified','original_completed_at':d.get('at')})
tsv(E/'validation/recovery_certificates.tsv',['certificate','files','status','original_completed_at'],certs)

expected=[]
for p in sorted((E/'configurations').glob('*.json')):
 expected.extend(readj(p))
# E05's initial grid is defined in the immutable training script, without importing torch.
for feature_name in ['B','C']:
 for hidden in [[512],[512,256]]:
  for dropout in [.2,.5]:
   order=len([x for x in expected if x.get('_initial_E05')])
   expected.append({'name':f'E05_{order:03d}_{feature_name}_s{SEED}','stage':'E05','_initial_E05':True})
for c in expected:
 assert (E/f"candidates/{c['name']}/DONE.json").exists(),('incomplete configuration',c['name'])
 if c['stage'] in ['E05','E09']:assert (E/f"gpu/{c['name']}/DONE.json").exists(),c['name']

summaries=[readj(p) for p in sorted((E/'candidates').glob('*/summary.json'))]
stages=Counter(d['config']['stage'] for d in summaries)
stage_rows=[{'stage':'E00','candidate_records':0,'expected_configurations':0,'status':'complete','evidence':'diagnostics/*.tsv and summary.json; descriptive analysis'}]
for stage in sorted(stages):
 stage_rows.append({'stage':stage,'candidate_records':stages[stage],'expected_configurations':sum(c['stage']==stage for c in expected),'status':'complete','evidence':'all expected candidate and model DONE certificates verified'})
assert len(summaries)==335 and len(list((E/'gpu').glob('*/DONE.json')))==20
assert all((E/'diagnostics'/name).exists() for name in ['baseline_per_class.tsv','baseline_train_val.tsv','B_vs_A_by_class.tsv','baseline_val_strata.tsv','summary.json'])
assert readj(E/'diagnostics/summary.json')['test_scores_read'] is False
tsv(E/'validation/recovery_stages.tsv',list(stage_rows[0]),stage_rows)

pids=[]
for p in sorted((E/'logs').glob('*.pid')):
 pid=int(p.read_text().strip())
 try:os.kill(pid,0);status='pid_exists_identity_not_assumed'
 except ProcessLookupError:status='not_running'
 except PermissionError:status='pid_exists_identity_not_assumed'
 pids.append({'pid_file':str(p.relative_to(E)),'pid':pid,'status':status})
tsv(E/'validation/recovery_pids.tsv',['pid_file','pid','status'],pids)
assert not list(E.rglob('*.tmp')),'uncommitted temporary files remain'
assert readj(E/'validation/final_audit.json')['passed'] is True
sel=readj(E/'selection/final_selection.json')
for name,h in sel['files'].items():assert checked_sha(E/name)==h,name
event=readj(E/'revision/revision_evaluation_event.json')
assert event['overall_test_view_number']==2 and event['new_selected_schemes']==1
assert event['context']['selection_sha256']==checked_sha(E/'selection/final_selection.json')
record={'at':now(),'seconds':time.monotonic()-start,'passed':True,'certificates_verified':len(certs),'unique_referenced_files_verified':len(cache),'candidate_records':len(summaries),'gpu_configurations_completed':20,'missing_or_interrupted_configurations':[], 'stages':stage_rows,'network_recovery_action':'verify existing completed work; finish documentation and delivery only','training_restarted':False,'test_evaluation_repeated':False,'existing_event_id':event['event_id'],'existing_test_view_number':2,'recorded_pids':pids}
jout(E/'validation/recovery_inventory.json',record)
log('recovery_inventory_passed',certificates=len(certs),candidates=len(summaries),gpu_completed=20,missing=0,seconds=record['seconds'])

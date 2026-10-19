from access_guard import install
install()
from common import *
from collections import Counter
from sklearn.metrics import average_precision_score
import argparse,stat

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--candidate');args=ap.parse_args();start=time.monotonic();d=dev();cache={}
 def h(p):
  p=Path(p)
  if p not in cache:cache[p]=sha(p)
  return cache[p]
 before=readj(E/'audit/history_metadata_before.json');changed=[]
 for name,old in before['files'].items():
  p=ROOT/name
  try:st=p.lstat()
  except FileNotFoundError:changed.append({'path':name,'reason':'missing'});continue
  actual={'size':st.st_size,'mtime_ns':st.st_mtime_ns,'mode':st.st_mode,'symlink_not_followed':stat.S_ISLNK(st.st_mode)}
  if actual!=old:changed.append({'path':name,'reason':'metadata changed','before':old,'after':actual})
 # Preserve the failed blanket read-only check; never silently rebaseline history.
 # This exact log path is the sole explicit closeout exception. Its contents and
 # writer are not inspected or attributed. All scientific artifacts remain strict.
 session_log='output/_agent_dialog/turn5.log'
 exceptions=[x for x in changed if x['path']==session_log and x['reason']=='metadata changed']
 unexpected=[x for x in changed if x not in exceptions]
 metadata_review={'at':now(),'snapshot_files':len(before['files']),'unchanged_files':len(before['files'])-len(changed),'all_historical_metadata_unchanged':not changed,'explicit_exceptions':exceptions,'unexpected_changes':unexpected,'exception_scope':'Only output/_agent_dialog/turn5.log; apparent session log based on path, contents and writer not verified','original_snapshot_preserved':True,'payload_opened':False,'first_blanket_audit_failed':bool(exceptions),'scientific_inputs_exempted':False}
 jout(E/'audit/history_metadata_exception.json',metadata_review)
 assert not unexpected,unexpected[:5]
 for p,expected in readj(E/'data/source_manifest.json')['sources'].items():assert h(p)==expected,p
 certs=list(E.rglob('*DONE.json'));ref_files=set()
 for p in certs:
  cert=readj(p)
  for path,expected in cert['files'].items():
   q=E/path
   assert q.is_relative_to(E) and h(q)==expected,('certificate mismatch',str(p),path);ref_files.add(q)
 summaries=[readj(p) for p in (E/'candidates').glob('*/summary.json')];expected_counts={'E11':36,'E12':30,'E13':4,'E14':2}
 actual=Counter(x['config']['stage'] for x in summaries);assert dict(actual)==expected_counts,(actual,expected_counts)
 for r in summaries:
  z=np.load(E/f"candidates/{r['name']}/validation.npz",allow_pickle=False)
  assert np.array_equal(z['protein_id'],d['val_ids']) and np.array_equal(z['class_order'],d['class_order'])
  sc=z['scores'];assert sc.shape==d['val_y'].shape and np.isfinite(sc).all()
  av=[]
  for j in range(32):
   k=d['val_y'][:,j]!=-1;av.append(average_precision_score(d['val_y'][k,j],sc[k,j]))
  assert abs(float(np.mean(av))-r['validation_macro_AP'])<1e-12,r['name']
  if r['config']['stage'] in ['E13','E14']:
   b=readj(E/'references/E06_mix_098/summary.json')['validation_macro_AP']
   assert r['selection_eligible']==(r['crossfit_macro_AP']>=b+.002)
 assert not list(E.rglob('*FAILURE.json')),'model failures require explicit disposition before delivery'
 assert not list(E.rglob('*.tmp')),'incomplete atomic files'
 proposed=readj(E/'selection/proposed_selection.json');assert proposed['chosen_candidate']==args.candidate
 independent=readj(E/'audit/controller_independent_review.json');assert independent['passed'] is True
 if args.candidate:
  assert verified(E/f'recompute/{args.candidate}/val.DONE.json')
  assert verified(E/'bootstrap_validation/DONE.json')
  assert proposed['chosen_evidence']['selection_eligible'] and proposed['chosen_evidence']['validation_macro_AP']>.902357
 # All sample IDs exposed in current prediction/data arrays must belong to the development universe.
 permitted=set(d['train_ids'])|set(d['val_ids']);arrays_checked=0
 for path in list((E/'candidates').glob('*/validation.npz'))+list((E/'references').glob('*/train_scores.npz'))+list((E/'recompute').glob('*/val_scores.npz')):
  with np.load(path,allow_pickle=False) as z:
   if 'protein_id' in z.files:assert set(z['protein_id'])<=permitted;arrays_checked+=1
 guard_records=[]
 for p in (E/'audit').glob('access_*.json'):
  x=readj(p)
  if 'historical_read_paths' not in x:continue
  assert not x['test_payload_reads_allowed']
  forbidden=[v for v in x['historical_read_paths'] if '/revision/' in v or '/inference/' in v or '/inference_v2/' in v or '/universe/labels' in v or '/test/' in v]
  assert not forbidden,forbidden
  guard_records.append({'pid':x['pid'],'historical_read_count':x['historical_read_count'],'denied_count':len(x['denied_accesses'])})
 result={'at':now(),'passed':True,'status':'passed_with_unresolved_session_log_metadata_exception' if exceptions else 'passed','historical_read_only_check_unqualified_pass':not changed,'historical_metadata_exception_count':len(exceptions),'history_metadata_review_sha256':sha(E/'audit/history_metadata_exception.json'),'seconds':time.monotonic()-start,'historical_files_metadata_unchanged':len(before['files'])-len(changed),'historical_files_metadata_checked':len(before['files']),'historical_test_payload_hashed_or_read':False,'development_source_hashes_verified':len(readj(E/'data/source_manifest.json')['sources']),'certificates_verified':len(certs),'unique_referenced_files_verified':len(ref_files),'candidate_counts':dict(actual),'candidate_AP_independently_recomputed':len(summaries),'sample_id_arrays_checked':arrays_checked,'guard_process_logs':guard_records,'guard_scope':'Python-audited I/O plus source-code review; not an OS/native-library sandbox','controller_independent_review_sha256':sha(E/'audit/controller_independent_review.json'),'test_evaluated_this_turn':False,'test_label_read_this_turn':False,'third_test_authorized':False,'selected_candidate':args.candidate,'frozen_source_issues_unchanged':True}
 jout(E/'audit/final_audit.json',result);log('turn5_final_audit_passed',candidate=args.candidate,old_metadata=len(before['files']),certificates=len(certs))
if __name__=='__main__':main()

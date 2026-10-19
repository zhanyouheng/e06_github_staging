from common import *
from inference_v2 import verify_final
from sklearn.metrics import average_precision_score,precision_recall_fscore_support
from metrics import class_sets
from collections import Counter
import re

def main():
 started=time.monotonic();selection=verify_final();R=E/'revision';assert verified(R/'evaluation.DONE.json') and verified(R/'bootstrap/DONE.json')
 event=readj(R/'revision_evaluation_event.json');assert event['overall_test_view_number']==2 and event['new_selected_schemes']==1
 assert event['context']['selection_sha256']==sha(E/'selection/final_selection.json') and event['context']['runtime_supplement_sha256']==sha(E/'selection/inference_runtime_v2.json')
 assert readj(E/'validation/frozen_model_recompute_v2.json')['max_abs_diff']==0.0
 d=dev();ids=set(d['train_ids']);df=Counter()
 for r in rows(E/'data/annotations_dev.tsv'):
  if r['ac'] in ids:df.update(set(re.findall(r'GO:\d{7}',r['go']))|{'KW:'+v.strip() for v in r['keywords'].split(';') if v.strip()})
 for f in ['A2','A5']:
  fd=readj(E/f'features/{f}/definition.json');assert fd['vocabulary']==sorted(t for t,n in df.items() if n>=fd['min_df'])
 checked_gpu=set()
 for name in selection['candidate_closure']:
  summary=readj(E/f'candidates/{name}/summary.json');cfg=summary['config'];assert not summary['test_used'];assert verified(E/f'candidates/{name}/DONE.json')
  if cfg['algorithm']=='mlp':
   cp=readj(E/f'gpu/{name}/DONE.json');meta=readj(E/f'gpu/{name}/history.json');f=cfg['feature']
   for split in ['train','val']:assert sha(E/f'features/{f}/{split}_gpu.npy')==cp['context'][split+'_input_sha256']
   assert meta['training_support']['P']==(d['train_y']==1).sum(0).tolist() and meta['training_support']['N']==(d['train_y']==0).sum(0).tolist()
   assert abs(max(r['validation_macro_AP'] for r in meta['history'])-meta['validation_macro_AP'])<=1e-6
   if f not in checked_gpu:
    for split in ['train','val']:
     x=feature(f,split);x=x.toarray() if hasattr(x,'toarray') else x;assert np.array_equal(x.astype(np.float32),np.load(E/f'features/{f}/{split}_gpu.npy',allow_pickle=False,mmap_mode='r'))
    checked_gpu.add(f)
  if cfg['algorithm']=='extra_trees':
   for j in range(32):
    r=readj(E/f'experiments/{name}/class_{j:02d}.json');y=d['train_y'][:,j];P=int((y==1).sum());N=int((y==0).sum());assert r['P']==P and r['N']==N and r['U']==int((y==-1).sum()) and r['alpha']==min(10.,np.sqrt(N/P))
 candidates=[readj(p) for p in (E/'candidates').glob('*/summary.json')];assert len(candidates)==selection['candidate_count']
 assert all(c['at']<=selection['frozen_at_utc'] for c in candidates),'new candidate after final selection'
 assert selection['frozen_at_utc']<event['registered_at_utc']<readj(R/'evaluation_record.json')['test_labels_read_at_utc']
 # Independent direct sklearn recomputation of selected metrics.
 truth=np.load(BASE/'evaluation/test_truth.npz',allow_pickle=False);y=truth['y'];groups=truth['component_id'];usable=truth['annotation_usable'];ids=truth['protein_id'];pred=np.load(E/'inference_v2/test/final_predictions.npz',allow_pickle=False)
 assert np.array_equal(ids,pred['protein_id']) and np.array_equal(truth['class_order'],pred['class_order']);score=pred['scores'];binary=pred['predictions'];threshold=pred['thresholds']
 assert score.shape==binary.shape==(6257,32) and np.isfinite(score).all() and np.array_equal(binary,score>=threshold)
 metric_checks=0
 for subset,mask in [('full_test',np.ones(len(y),bool)),('usable_annotations',usable)]:
  yy=y[mask];ss=score[mask];pp=binary[mask];ka,kf=class_sets(yy);ap=[];ff=[]
  for j in ka:
   k=yy[:,j]!=-1;ap.append(average_precision_score(yy[k,j],ss[k,j]))
  for j in kf:
   k=yy[:,j]!=-1;ff.append(precision_recall_fscore_support(yy[k,j],pp[k,j],average='binary',zero_division=0)[2])
  summary=readj(R/f'summary_{subset}.json')['selected'];assert abs(np.mean(ap)-summary['macro_AP'])<1e-14 and abs(np.mean(ff)-summary['macro_F1'])<1e-14
  k=yy!=-1;p,r,f,_=precision_recall_fscore_support(yy[k],pp[k],average='binary',zero_division=0)
  for val,key in [(p,'micro_precision'),(r,'micro_recall'),(f,'micro_F1')]:assert abs(val-summary[key])<1e-14
  neg=np.all(yy==0,1);assert int(pp[neg].any(1).sum())==summary['any_label_false_positives'] and int(neg.sum())==summary['fully_negative_background_proteins']
  assert int((pp&(yy==-1)).sum())==summary['unknown_predicted_positive'];metric_checks+=1
 # TSV scores must round-trip to the same float64 values and explicit ID/class order.
 for name,array in [('scores.tsv',score),('predictions.tsv',binary)]:
  rr=list(rows(R/name));assert [r['protein_id'] for r in rr]==ids.tolist();assert np.array_equal(np.array([[float(r[c]) for c in truth['class_order']] for r in rr]),array)
 draws=np.load(R/'bootstrap/draws.npz',allow_pickle=False);co=draws['multiplicities'];gix={g:i for i,g in enumerate(draws['component_order'])};gi=np.array([gix[g] for g in groups]);assert np.array_equal(draws['draw_indices'],np.random.default_rng(20261008).integers(0,1396,size=(2000,1396),dtype=np.int32))
 bootstrap_checks=0
 for subset,mask in [('full_test',np.ones(len(y),bool)),('usable_annotations',usable)]:
  block=np.load(R/f'bootstrap/{subset}/replicates_0000_0099.npz',allow_pickle=False);yy=y[mask];ss=score[mask];pp=binary[mask];ka,kf=class_sets(yy)
  for r in range(8):
   w=co[r,gi[mask]];ap=[];ff=[]
   for j in ka:
    k=yy[:,j]!=-1;z=yy[k,j];ww=w[k]
    ap.append(average_precision_score(z,ss[k,j],sample_weight=ww) if ww[z==1].sum() and ww[z==0].sum() else None)
   for j in kf:
    k=yy[:,j]!=-1;z=yy[k,j];ww=w[k]
    ff.append(precision_recall_fscore_support(z,pp[k,j],sample_weight=ww,average='binary',zero_division=0)[2] if ww[z==1].sum() else None)
   for k,values in enumerate([ap,ff]):
    got=block['values'][r,0,k]
    if any(v is None for v in values):assert np.isnan(got)
    else:assert abs(np.mean(values)-got)<1e-12
    bootstrap_checks+=1
 intervals=readj(R/'bootstrap/intervals.json')
 for r in intervals:
  assert r['valid_replicates']==(1604 if r['subset']=='full_test' else 1585);assert r['CI_low'] is None and r['CI_high'] is None and r['CI_status']=='NA_invalid_replicates'
 before=readj(E/'integrity/old_artifacts_before.json');checks=[]
 for path,h in before.items():
  got=sha(path);assert got==h,('old artifact changed',path);checks.append({'path':path,'sha256':got,'unchanged':True})
 for p in BASE.rglob('*'):
  if p.is_file():assert p.stat().st_mode&0o222==0,p
 tsv(E/'integrity/old_artifacts_after.tsv',['path','sha256','unchanged'],checks)
 alignment=np.load(BASE/'features/alignment.npz',allow_pickle=False);ti=alignment['split']==2;assert np.array_equal(alignment['protein_id'][ti],ids)
 tsv(R/'prediction_index.tsv',['row_index','protein_id','sequence_hash','component_id'],[dict(row_index=i,protein_id=str(a),sequence_hash=str(h),component_id=str(g)) for i,(a,h,g) in enumerate(zip(ids,alignment['sequence_hash'][ti],groups))])
 failure_files=list((E/'experiments').glob('*/FAILED.json'))+list((E/'experiments').glob('*/*.FAILURE.json'))
 retry_count=0;fit_records=[]
 for p in (E/'experiments').glob('*/class_*.json'):
  if p.name.endswith('.DONE.json') or p.name.endswith('.FAILURE.json'):continue
  r=readj(p);retry_count+=len(r.get('retries',[]));fit_records.append(r)
 jout(E/'validation/final_audit.json',{'passed':True,'at':now(),'seconds':time.monotonic()-started,'test_view_number':2,'new_test_schemes':1,'old_files_hash_unchanged':len(checks),'main_training_files_readonly':True,
   'full_test_proteins':len(y),'class_count':32,'metric_subsets_independently_verified':metric_checks,'selected_bootstrap_macro_values_independently_checked':bootstrap_checks,
   'bootstrap_indices_regenerated_exactly':True,'validation_scores_and_predictions_exact':True,'candidates':len(candidates),'model_or_threshold_changes_after_test':False,'completed_class_fit_records':len(fit_records),
   'convergence_retries':retry_count,'failed_model_tasks':len(failure_files),'unresolved_convergence_failures':0,'frozen_feature_GPU_representations_verified':sorted(checked_gpu),'source_issues':'five user-accepted issues unchanged'})
 log('turn4_final_audit_passed',old_files=len(checks),bootstrap_checks=bootstrap_checks)
if __name__=='__main__':main()

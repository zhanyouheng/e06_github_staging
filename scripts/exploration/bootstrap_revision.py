from common import *
from bootstrap_core import pack,run_block
import shutil

def main():
 started=time.monotonic();assert verified(E/'revision/evaluation.DONE.json');event=readj(E/'revision/revision_evaluation_event.json')
 for n,h in event['context']['scripts'].items():assert sha(S/n)==h
 B=E/'revision/bootstrap';context={'evaluation_DONE_sha256':sha(E/'revision/evaluation.DONE.json'),'event_sha256':sha(E/'revision/revision_evaluation_event.json')}
 if verified(B/'DONE.json',context):log('revision_bootstrap_resume_verified');return
 truth=np.load(BASE/'evaluation/test_truth.npz',allow_pickle=False);y=truth['y'];ids=truth['protein_id'];groups=truth['component_id'];classes=truth['class_order'].tolist();usable=truth['annotation_usable']
 baselines=[r+'_'+w for w in ['weighted','unweighted'] for r in ['A','B','C']];models=['selected']+baselines;scores=[];thresholds=[]
 for name in models:
  p=E/'inference_v2/test/final_predictions.npz' if name=='selected' else BASE/f'predictions/{name}.npz'
  with np.load(p,allow_pickle=False) as z:
   assert np.array_equal(ids,z['protein_id']);scores.append(z['scores']);thresholds.append(z['thresholds'] if name=='selected' else np.full(32,float(z['threshold'])))
 scores=np.asarray(scores);thresholds=np.asarray(thresholds);source=BASE/'bootstrap/draws.npz';drawp=B/'draws.npz'
 if not drawp.exists():q=tmp(drawp);shutil.copyfile(source,q);assert sha(q)==sha(source);commit(q,drawp)
 assert sha(drawp)==sha(source)
 with np.load(drawp,allow_pickle=False) as z:
  group_order=z['component_order'];counts=z['multiplicities'];seed=int(z['seed']);assert seed==20261008 and counts.shape==(2000,1396)
 gi={str(g):i for i,g in enumerate(group_order)};component_index=np.array([gi[str(g)] for g in groups],np.int32);files=[drawp];intervals=[];support={}
 for subset,mask in [('full_test',np.ones(len(y),bool)),('usable_annotations',usable)]:
  args=pack(y[mask],scores[:,mask,:],thresholds,component_index[mask]);vv=[];apbad=[];f1bad=[]
  for begin in range(0,2000,100):
   end=begin+100;p=B/f'{subset}/replicates_{begin:04d}_{end-1:04d}.npz';cp=p.with_suffix('.DONE.json');ct={**context,'subset':subset,'begin':begin,'end_exclusive':end,'draws_sha256':sha(drawp)}
   if not verified(cp,ct):
    values,ba,bf=run_block(counts[begin:end],*args);delta=values[:,0:1,:]-values[:,1:,:]
    npz(p,replicate=np.arange(begin,end),models=np.array(models),metrics=np.array(['macro_AP','macro_F1']),values=values,comparators=np.array(baselines),selected_minus_baseline=delta,invalid_AP_classes=ba,invalid_F1_classes=bf)
    certificate(cp,[p],ct);log('revision_bootstrap_block',subset=subset,completed=end)
   with np.load(p,allow_pickle=False) as z:vv.append(z['values']);apbad.append(z['invalid_AP_classes']);f1bad.append(z['invalid_F1_classes'])
   files.extend([p,cp])
  values=np.concatenate(vv);ba=np.concatenate(apbad);bf=np.concatenate(f1bad);assert values.shape==(2000,7,2)
  for m in range(1,7):assert np.array_equal(np.isnan(values[:,0]),np.isnan(values[:,m]))
  support[subset]={'proteins':int(mask.sum()),'components':len(set(groups[mask])),'sampling_component_universe':1396,'K_AP':int(args[-2].sum()),'K_F1':int(args[-1].sum()),
    'invalid_AP_by_class':{classes[j]:int(ba[:,j].sum()) for j in range(32) if ba[:,j].any()},'invalid_F1_by_class':{classes[j]:int(bf[:,j].sum()) for j in range(32) if bf[:,j].any()}}
  summary=readj(E/f'revision/summary_{subset}.json')
  for m,name in enumerate(baselines,1):
   for k,metric in enumerate(['macro_AP','macro_F1']):
    delta=values[:,0,k]-values[:,m,k];valid=np.isfinite(delta);n=int(valid.sum());condition=np.percentile(delta[valid],[2.5,97.5]).tolist() if n else None
    intervals.append({'subset':subset,'comparison':'selected-'+name,'metric':metric,'priority':'primary' if subset=='full_test' and name=='C_weighted' and metric=='macro_AP' else 'supplementary',
      'observed_delta':summary['selected'][metric]-summary[name][metric],'attempts':2000,'valid_replicates':n,'invalid_replicates':2000-n,'valid_fraction':n/2000,'invalid_fraction':(2000-n)/2000,
      'CI_status':'complete_ordinary_percentile' if n==2000 else 'NA_invalid_replicates','CI_low':condition[0] if n==2000 else None,'CI_high':condition[1] if n==2000 else None,
      'conditional_valid_only_low':condition[0] if condition else None,'conditional_valid_only_high':condition[1] if condition else None,
      'involved_classes':sorted(support[subset]['invalid_AP_by_class' if k==0 else 'invalid_F1_by_class'])})
  p=B/f'{subset}/paired_deltas.npz';npz(p,replicate=np.arange(2000),comparators=np.array(baselines),metrics=np.array(['macro_AP','macro_F1']),delta=values[:,0:1,:]-values[:,1:,:]);files.append(p)
 jout(B/'intervals.json',intervals);jout(B/'support_summary.json',support);tsv(B/'intervals.tsv',list(intervals[0]),[{**r,'involved_classes':json.dumps(r['involved_classes'])} for r in intervals])
 jout(B/'runtime.json',{'seconds':time.monotonic()-started,'attempts':2000,'seed':seed,'copied_draws_sha256':sha(drawp),'source_draws_sha256':sha(source),'bootstrap_threads':int(os.environ['NUMBA_NUM_THREADS'])})
 files.extend(B/n for n in ['intervals.json','intervals.tsv','support_summary.json','runtime.json']);certificate(B/'DONE.json',files,context);log('revision_bootstrap_complete')
if __name__=='__main__':main()

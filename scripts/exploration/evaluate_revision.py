from common import *
from inference_v2 import verify_final
from metrics import evaluate

def main():
 started=time.monotonic();selection=verify_final();event=readj(E/'revision/revision_evaluation_event.json');assert event['overall_test_view_number']==2
 for n,h in event['context']['scripts'].items():assert sha(S/n)==h
 assert verified(E/'inference_v2/test/final_predictions.DONE.json');R=E/'revision';context={'event_sha256':sha(R/'revision_evaluation_event.json'),'prediction_sha256':sha(E/'inference_v2/test/final_predictions.npz')}
 if verified(R/'evaluation.DONE.json',context):log('revision_evaluation_resume_verified');return
 # First new test-label/archived-score interpretation occurs here, after freeze and event.
 label_read=now();truth=np.load(BASE/'evaluation/test_truth.npz',allow_pickle=False);y=truth['y'];ids=truth['protein_id'];classes=truth['class_order'].tolist();groups=truth['component_id'];usable=truth['annotation_usable'];tiers=truth['positive_evidence_tier']
 pred=np.load(E/'inference_v2/test/final_predictions.npz',allow_pickle=False);assert np.array_equal(pred['protein_id'],ids) and pred['class_order'].tolist()==classes
 score=pred['scores'];thresholds=pred['thresholds'];assert np.array_equal(pred['predictions'],score>=thresholds);assert np.isfinite(score).all()
 baselines=[r+'_'+w for w in ['weighted','unweighted'] for r in ['A','B','C']];scores={'selected':score};ts={'selected':thresholds}
 for name in baselines:
  with np.load(BASE/f'predictions/{name}.npz',allow_pickle=False) as b:assert np.array_equal(b['protein_id'],ids);scores[name]=b['scores'];ts[name]=np.full(32,float(b['threshold']))
 # Only the chosen new model is evaluated. Other new experiment scores are never generated here.
 out_rows=[];sum_rows=[];files=[]
 for subset,mask in [('full_test',np.ones(len(ids),bool)),('usable_annotations',usable)]:
  summaries={}
  for name in ['selected']+baselines:
   per,sm=evaluate(y[mask],scores[name][mask],ts[name],classes,groups[mask]);summaries[name]=sm
   sum_rows.append({'subset':subset,'model':name,**{k:v for k,v in sm.items() if not isinstance(v,list)}})
   if name=='selected':out_rows.extend({'subset':subset,**r} for r in per)
   else:
    archived=readj(BASE/f'evaluation/summary_{subset}.json')[name]
    for k in ['macro_AP','macro_F1','micro_precision','micro_recall','micro_F1','any_label_FPR']:assert abs(archived[k]-sm[k])<1e-12
  f=R/f'summary_{subset}.json';jout(f,summaries);files.append(f)
 for n,rr in [('per_class.tsv',out_rows),('summary.tsv',sum_rows)]:
  f=R/n;tsv(f,list(rr[0]),rr);files.append(f)
 for suffix,array in [('scores.tsv',score),('predictions.tsv',pred['predictions'])]:
  f=R/suffix;tsv(f,['protein_id']+classes,[{'protein_id':str(pid),**{c:format(float(array[i,j]),'.17g') if suffix=='scores.tsv' else int(array[i,j]) for j,c in enumerate(classes)}} for i,pid in enumerate(ids)]);files.append(f)
 meta={r['protein_id']:r for r in rows(BASE/'features/protein_metadata.tsv')};prot={r['ac']:r for r in rows(OUT/'universe/proteins.tsv')};md=[meta[str(a)] for a in ids]
 levels={'length_bin':['<=256','257-512','513-1022','>1022'],'annotation_score':['1','2','3','4','5','missing'],'annotation_status':['usable','raw_empty','all_pruned_or_oov','extraction_failure']}
 for key in ['taxonomy','database','feature_source','background_sampling_source']:levels[key]=sorted({r[key] for r in md})
 strata=[]
 for key,values in levels.items():
  for v in values:strata.append((key,v,np.array([r[key]==v for r in md])))
 for v in sorted({prot[str(a)]['source'] for a in ids}):strata.append(('protein_origin',v,np.array([prot[str(a)]['source']==v for a in ids])))
 for v in sorted(set(tiers[y==1])):strata.append(('positive_evidence_tier',str(v),np.any((tiers==v)&(y==1),axis=1)))
 strata.append(('positive_evidence_tier','no_known_positive',np.all(y!=1,axis=1)))
 ss=[];pp=[]
 for kind,v,mask in strata:
  per,sm=evaluate(y[mask],score[mask],thresholds,classes,groups[mask]);ss.append({'stratum_type':kind,'stratum':v,**{k:x for k,x in sm.items() if not isinstance(x,list)}});pp.extend({'stratum_type':kind,'stratum':v,**r} for r in per)
 for n,rr in [('stratified_summary.tsv',ss),('stratified_per_class.tsv',pp)]:
  f=R/n;tsv(f,list(rr[0]),rr);files.append(f)
 f=R/'evaluation_record.json';jout(f,{'event_id':event['event_id'],'overall_test_view_number':2,'test_labels_read_at_utc':label_read,'completed_at_utc':now(),'seconds':time.monotonic()-started,
   'selected_candidate':selection['chosen_candidate'],'new_schemes_evaluated':1,'archived_comparators':baselines,'test_proteins':len(ids),'P':int((y==1).sum()),'N':int((y==0).sum()),'U':int((y==-1).sum()),'strata':len(strata),
   'archived_scores_recomputed_consistently':True,'thresholds_modified_on_test':False,'model_changed_on_test':False,'test_truth_source_sha256':sha(BASE/'evaluation/test_truth.npz')});files.append(f)
 certificate(R/'evaluation.DONE.json',files,context);log('revision_evaluation_complete',seconds=time.monotonic()-started)
if __name__=='__main__':main()

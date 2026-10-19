from common import *
from metrics import evaluate,choose_threshold

def per_class_thresholds(y,score):
 ts=[]
 for j in range(y.shape[1]):
  ts.append(choose_threshold(y[:,j:j+1],score[:,j:j+1])['threshold'] if np.any(y[:,j]==1) else float(np.nextafter(score[:,j].max(),np.inf)))
 return np.array(ts)
def register_candidate(name,config,score,model_files,extra=None):
 d=dev();assert score.shape==d['val_y'].shape and np.isfinite(score).all();p=E/f'candidates/{name}';shared=choose_threshold(d['val_y'],score);thresholds=per_class_thresholds(d['val_y'],score)
 per,sm=evaluate(d['val_y'],score,thresholds,d['class_order'].tolist(),d['val_groups']);_,ss=evaluate(d['val_y'],score,shared['threshold'],d['class_order'].tolist(),d['val_groups'])
 npz(p/'validation.npz',protein_id=d['val_ids'],class_order=d['class_order'],scores=score,thresholds=thresholds,shared_threshold=np.array(shared['threshold']))
 tsv(p/'per_class.tsv',list(per[0]),per)
 summary={'name':name,'config':config,'seed':config.get('seed',SEED),'validation_macro_AP':sm['macro_AP'],'validation_macro_F1':sm['macro_F1'],
   'validation_shared_macro_F1':ss['macro_F1'],'any_label_FPR':sm['any_label_FPR'],'shared_any_label_FPR':ss['any_label_FPR'],'threshold_policy':'per-class max validation F1, ties higher',
   'thresholds':thresholds.tolist(),'shared_threshold':shared['threshold'],'summary':sm,'shared_summary':ss,'at':now(),'test_used':False,
   'model_files':[str(Path(f).relative_to(E)) for f in model_files]}
 if extra:summary.update(extra)
 jout(p/'summary.json',summary);certificate(p/'DONE.json',[p/'validation.npz',p/'per_class.tsv',p/'summary.json']+model_files,
    {'config':config,'dev_sha256':sha(E/'data/dev.npz')})
 log('candidate_complete',name=name,AP=sm['macro_AP'],F1=sm['macro_F1'],shared_F1=ss['macro_F1']);return summary

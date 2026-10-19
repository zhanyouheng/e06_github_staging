from common import *
import numpy as np,pickle
from sklearn.metrics import average_precision_score,f1_score
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed

def replay(task):
    family,j,C=task;root=T/'models/cpu'/family/(('C'+str(C)) if C is not None else 'trees')/'full'/('class%02d'%j);assert verified(root)
    with open(root/'model.pkl','rb') as f:a=pickle.load(f)
    x=np.load(T/'features/full'/('A2B.npy' if family=='M01' else 'C.npy'),mmap_mode='r')[18770:]
    if family=='M02':x=np.concatenate([x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-30),np.load(T/'features/annotation_score.npy')[18770:]],axis=1)/np.float32(np.sqrt(2))
    s=np.full(len(x),a['constant']) if a['model'] is None else a['model'].predict_proba(x)[:,list(a['model'].classes_).index(1)]
    z=np.load(root/'predictions.npz');err=float(np.max(np.abs(s-z['score'])));assert err==0
    return {'family':family,'class_index':j,'C':C,'max_abs_difference':err,'model_sha256':sha(root/'model.pkl')}

def main():
    assert verified(T/'selection') and verified(T/'bootstrap')
    d=np.load(T/'data/development.npz');nt=len(d['train_fold']);y=d['y'][nt:];classes=d['class_order'];token=json.loads((T/'data/tokens.json').read_text());raw=np.load(T/'data/embedding_raw.npy',mmap_mode='r');feature_checks=[]
    for name in ['full']+['outer'+str(i) for i in range(5)]+['inner'+str(i) for i in range(5)]:
        p=T/'features'/name;assert verified(p);r=np.load(p/'fit_rows.npz')['row_index'];assert np.all(r<nt)
        if name!='full':
            k=int(name[-1]);assert np.all(d['train_fold'][r]!=k)
            if name.startswith('inner'):assert not d['inner_holdout'][r].any()
        df=Counter(v for i in r for v in set(token[i]));b=np.load(p/'B_parameters.npz');np.testing.assert_array_equal(b['mean'],raw[r].astype(float).mean(0));sd=raw[r].astype(float).std(0);sd[sd==0]=1;np.testing.assert_array_equal(b['std'],sd)
        for route,m in [('C',10),('A2B',2)]:assert json.loads((p/(route+'_vocab.json')).read_text())['vocabulary']==sorted(v for v,n in df.items() if n>=m)
        feature_checks.append({'feature_set':name,'fit_rows':len(r),'vocab_recomputed':True,'B_mean_std_bitwise_equal':True,'no_held_or_val_fitting':True})
    results=[]
    for p in sorted((T/'predictions').glob('*/validation.npz')):
        z=np.load(p);assert np.array_equal(z['protein_id'],d['protein_id'][nt:]);assert np.array_equal(z['class_order'],classes);s=z['scores'];pred=z['prediction'];np.testing.assert_array_equal(pred,s>=z['thresholds']);ap=[];f1=[]
        for j in range(32):
            m=y[:,j]>=0;assert ((y[m,j]==1).any() and (y[m,j]==0).any());ap.append(average_precision_score(y[m,j],s[m,j]));f1.append(f1_score(y[m,j],pred[m,j],zero_division=0))
        m=json.loads((p.parent/'metrics.json').read_text());err=abs(float(np.mean(ap))-m['macro_AP']);fe=abs(float(np.mean(f1))-m['macro_F1']);assert err<1e-12 and fe<1e-12
        results.append({'candidate':p.parent.name,'macro_AP_sklearn':float(np.mean(ap)),'macro_F1_sklearn':float(np.mean(f1)),'AP_abs_difference':err,'F1_abs_difference':fe})
    # Check ten bootstrap rounds against sklearn with explicit multiplicity weights.
    z=np.load(T/'bootstrap/replicates.npz');draw=np.load(T/'bootstrap/sampling_indices.npy',mmap_mode='r');frame=np.load(T/'bootstrap/sampling_frame.npz');gi=frame['protein_component_index'];checks=[]
    for b in np.linspace(0,1999,10,dtype=int):
        w=np.bincount(draw[b],minlength=len(frame['component_id']))[gi]
        for k,name in enumerate(z['names']):
            if not np.isfinite(z['macro_AP'][b,k]):continue
            s=np.load(T/'predictions'/str(name)/'validation.npz')['scores'];a=[]
            for j in range(32):
                m=y[:,j]>=0;a.append(average_precision_score(y[m,j],s[m,j],sample_weight=w[m]))
            err=abs(np.mean(a)-z['macro_AP'][b,k]);assert err<1e-12;checks.append({'round':int(b),'model':str(name),'AP_abs_difference':float(err)})
    tasks=[];C=json.loads((T/'selection/M01_train_OOF_C.json').read_text())['selected_C']
    for fam in ['M00','M01','M02','M03']:
        if not (T/'candidates'/fam/'DONE.json').exists():continue
        tasks.extend((fam,j,C if fam=='M01' else None) for j in range(32))
    with ProcessPoolExecutor(max_workers=16) as pool:modelrows=list(pool.map(replay,tasks))
    before=json.loads((T/'audit/history_before.json').read_text());after=history_snapshot();changed=[p for p,v in before.items() if after.get(p)!=v];assert not changed,changed
    warnings_rows=[]
    for p in (T/'models/cpu').rglob('result.json'):
        r=json.loads(p.read_text())
        for w in r.get('warnings',[]):warnings_rows.append({'path':str(p.relative_to(T)),**w,'attempts':json.dumps(r.get('attempts',[]))})
    tsv(T/'audit/convergence_warnings.tsv',warnings_rows,['path','category','message','attempts'])
    js(T/'audit/final_verification.json',{'at':now(),'status':'passed','features':feature_checks,'metric_checks':results,'bootstrap_checks':checks,'cpu_model_replay':modelrows,'historical_files_lstat_checked':len(before),'historical_changes':changed,'warning_count':len(warnings_rows),'no_test_payload_loaded':True,'all_existing_labels_unchanged':True})
    done(T/'audit/final',[T/'audit/final_verification.json',T/'audit/convergence_warnings.tsv',T/'audit/math_verification.json',T/'audit/gpu_model_replay.json'])
    print('final independent audit passed',flush=True)
if __name__=='__main__':main()

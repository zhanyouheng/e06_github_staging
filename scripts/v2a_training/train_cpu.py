from common import *
import numpy as np, pickle, warnings, traceback, argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from metrics import macro_ap

def one(task):
    family,stage,j,C=task;folder=T/'models/cpu'/family/(('C'+str(C)) if C is not None else 'trees')/stage/('class%02d'%j)
    if verified(folder):return {'family':family,'stage':stage,'class':j,'C':C,'status':'verified'}
    start=time.time();js(folder/'STARTED.json',{'at':now(),'pid':os.getpid(),'task':task,'script_sha256':sha(Path(__file__))});d=np.load(T/'data/development.npz',allow_pickle=False);nt=len(d['train_fold']);route='A2B' if family=='M01' else 'C';fset=stage
    fit=np.arange(nt) if stage=='full' else np.flatnonzero(d['train_fold']!=int(stage[5:]));held=np.arange(nt,len(d['y'])) if stage=='full' else np.flatnonzero(d['train_fold']==int(stage[5:]))
    x=np.load(T/'features'/fset/(route+'.npy'),mmap_mode='r')
    if family=='M02':x=np.concatenate([x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-30),np.load(T/'features/annotation_score.npy')],axis=1)/np.float32(np.sqrt(2))
    yy=d['y'][fit,j];P=int((yy==1).sum());N=int((yy==0).sum());alpha=min(10.,np.sqrt(N/max(P,1)));known=yy>=0;use=fit[known];target=yy[known];ww=np.where(target==1,alpha,1.)
    if family=='M03':
        src=d['source_names'][d['source_code'][use,j]];ww[(target==0)&(src=='blindcheck_not_fit')]*=.25
    warns=[];attempts=[];model=None;constant=None
    if P==0:constant=0.;status='no_known' if N==0 else 'P_zero'
    elif N==0:constant=1.;status='N_zero'
    else:
        status='regular'
        if family=='M01':
            for limit in [3000,10000]:
                model=LogisticRegression(C=C,penalty='l2',fit_intercept=True,solver='lbfgs',tol=1e-6,max_iter=limit,random_state=20261006+j)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always');model.fit(x[use],target,sample_weight=ww)
                wwlog=[{'category':w.category.__name__,'message':str(w.message)} for w in caught];warns.extend(wwlog);bad=any(issubclass(w.category,ConvergenceWarning) for w in caught);attempts.append({'max_iter':limit,'n_iter':model.n_iter_.tolist(),'convergence_warning':bad})
                if not bad:break
            if bad:raise RuntimeError('ConvergenceWarning persists after registered 10000 retry')
        else:
            model=ExtraTreesClassifier(n_estimators=256,max_features=.2,min_samples_leaf=1,random_state=20261006+j,n_jobs=1,bootstrap=False)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always');model.fit(x[use],target,sample_weight=ww)
            warns=[{'category':w.category.__name__,'message':str(w.message)} for w in caught]
    score=np.full(len(held),constant,dtype=float) if model is None else model.predict_proba(x[held])[:,list(model.classes_).index(1)]
    assert np.isfinite(score).all();artifact={'model':model,'constant':constant,'family':family,'stage':stage,'class_index':j,'route':route,'C':C,'training_support_status':status}
    atomic(folder/'model.pkl',lambda f:pickle.dump(artifact,f,protocol=5))
    with open(folder/'model.pkl','rb') as f:reloaded=pickle.load(f)
    again=np.full(len(held),reloaded['constant']) if reloaded['model'] is None else reloaded['model'].predict_proba(x[held])[:,list(reloaded['model'].classes_).index(1)]
    np.testing.assert_array_equal(score,again);npz(folder/'predictions.npz',protein_id=d['protein_id'][held],score=score)
    result={'family':family,'stage':stage,'class':str(d['class_order'][j]),'class_index':j,'C':C,'P':P,'N':N,'U':int((yy<0).sum()),'alpha':alpha,'negative_weight_mass':float(ww[target==0].sum()),'support_status':status,'seed':20261006+j,'warnings':warns,'attempts':attempts,'seconds':time.time()-start,'pid':os.getpid(),'saved_model_prediction_identical':True,'feature_done_sha':sha(T/'features'/fset/'DONE.json'),'code_sha':sha(Path(__file__)),'label_sha':sha(T/'data/development.npz')}
    js(folder/'result.json',result);done(folder,[folder/'model.pkl',folder/'predictions.npz',folder/'result.json']);return {'family':family,'stage':stage,'class':j,'C':C,'status':'complete','seconds':round(time.time()-start,2)}

def wrapped(task):
    try:return one(task)
    except Exception as e:
        family,stage,j,C=task;failure={'at':now(),'task':task,'exception':str(e),'traceback':traceback.format_exc(),'status':'failed_not_scored'};js(T/'failures'/f'cpu_{family}_{stage}_{j}_{C}.json',failure);return failure

def bundle(family,C):
    d=np.load(T/'data/development.npz');nt=len(d['train_fold']);root=T/'models/cpu'/family/(('C'+str(C)) if C is not None else 'trees');score=np.empty((nt,32),float)
    for k in range(5):
        idx=np.flatnonzero(d['train_fold']==k)
        for j in range(32):
            p=root/('outer'+str(k))/('class%02d'%j)
            if not verified(p):return None
            z=np.load(p/'predictions.npz');assert np.array_equal(z['protein_id'],d['protein_id'][idx]);score[idx,j]=z['score']
    npz(root/'oof.npz',protein_id=d['protein_id'][:nt],class_order=d['class_order'],scores=score)
    ap=macro_ap(d['y'][:nt],score);js(root/'oof_summary.json',{'macro_AP':ap,'selection_use':'train OOF only; optimistic for selecting shared C','C':C});return ap

def main():
    a=argparse.ArgumentParser();a.add_argument('--workers',type=int,default=24);args=a.parse_args();assert verified(T/'preflight') and verified(T/'features')
    js(T/'env/cpu_worker.json',{'at':now(),'pid':os.getpid(),'workers':args.workers,'threads_each':1,'script_sha':sha(Path(__file__))})
    tasks=[]
    for stage in ['full']+['outer'+str(k) for k in range(5)]:
        for family in ['M00','M02','M03']:
            for j in range(32):tasks.append((family,stage,j,None))
    for C in [.01,.1,1.,10.]:
        for k in range(5):
            for j in range(32):tasks.append(('M01','outer'+str(k),j,C))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for f in as_completed([pool.submit(wrapped,t) for t in tasks]):print(json.dumps(f.result(),ensure_ascii=False),flush=True)
    cs=[]
    for C in [.01,.1,1.,10.]:
        ap=bundle('M01',C);cs.append({'C':C,'OOF_macro_AP':ap,'status':'complete' if ap is not None else 'failed'})
    valid=[r for r in cs if r['OOF_macro_AP'] is not None]
    if valid:
        best=sorted(valid,key=lambda r:(-r['OOF_macro_AP'],r['C']))[0];js(T/'selection/M01_train_OOF_C.json',{'at':now(),'candidates':cs,'selected_C':best['C'],'test_or_val_selection':False})
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for f in as_completed([pool.submit(wrapped,('M01','full',j,best['C'])) for j in range(32)]):print(json.dumps(f.result()),flush=True)
    for family in ['M00','M02','M03']:bundle(family,None)
    js(T/'logs/cpu_finished.json',{'at':now(),'pid':os.getpid(),'M01_C_records':cs,'full_models_checked_at_evaluation':True})
if __name__=='__main__':main()

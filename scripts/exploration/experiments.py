from common import *
from metrics import evaluate,choose_threshold
from scipy import sparse
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.exceptions import ConvergenceWarning
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp,warnings,traceback,argparse

X=None;V=None;Y=None;D=None;CONFIG=None;CTX=None
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
def fit_one(j):
 cfg=CONFIG;p=E/f"experiments/{cfg['name']}/class_{j:02d}";cert=p.with_suffix('.DONE.json');ct={**CTX,'class':j}
 if verified(cert,ct):return readj(p.with_suffix('.json'))
 start=time.monotonic();yy=Y[:,j];known=yy!=-1;z=yy[known];P=int((z==1).sum());N=int((z==0).sum());alpha=min(10.,float(np.sqrt(N/P))) if P and cfg['weight']=='weighted' else 1.
 status='normal' if P and N else 'no_known_labels' if not P+N else 'no_positive' if not P else 'no_negative'
 result={'class_index':j,'P':P,'N':N,'U':int((yy==-1).sum()),'alpha':alpha,'status':status,'pid':os.getpid(),'retries':[],'warnings':[]}
 files=[];score=None
 try:
  if cfg['algorithm']=='logistic':
   coef=np.zeros(X.shape[1]);intercept=0.;constant=-1.;iterations=0
   if status!='normal':constant=float(P>0)
   elif cfg.get('reuse_baseline'):
    old=BASE/f"models/candidates/{cfg['feature']}_{cfg['weight']}/C{cfg['C']:g}/class_{j:02d}.npz"
    with np.load(old,allow_pickle=False) as m:coef=m['coef'];intercept=float(m['intercept']);constant=float(m['constant'])
    meta=readj(old.with_suffix('.json'));iterations=meta['iterations'];result.update(source_model=str(old),source_model_sha256=sha(old),source_seed=20261005)
   else:
    for maximum in [3000,10000]:
     model=LogisticRegression(C=cfg['C'],solver='lbfgs',tol=1e-6,max_iter=maximum,fit_intercept=True,penalty='l2',random_state=SEED)
     with warnings.catch_warnings(record=True) as ws:
      warnings.simplefilter('always');model.fit(X[known],z,sample_weight=np.where(z==1,alpha,1.))
     conv=[str(w.message) for w in ws if issubclass(w.category,ConvergenceWarning)];other=[str(w.message) for w in ws if not issubclass(w.category,ConvergenceWarning)]
     assert not other,other
     if conv:
      result['retries'].append({'max_iter':maximum,'warnings':conv})
      if maximum==10000:raise RuntimeError('unresolved convergence warning')
     else:break
    coef=model.coef_[0];intercept=float(model.intercept_[0]);iterations=int(model.n_iter_[0])
   f=p.with_suffix('.npz');npz(f,coef=coef,intercept=np.array(intercept),constant=np.array(constant));files.append(f)
   score=np.full(V.shape[0],constant) if constant>=0 else expit(V@coef+intercept);result['iterations']=iterations
  elif cfg['algorithm']=='extra_trees':
   if status=='normal':
    model=ExtraTreesClassifier(n_estimators=256,max_features=cfg['max_features'],min_samples_leaf=cfg['min_samples_leaf'],n_jobs=1,random_state=SEED+j,class_weight=None)
    with warnings.catch_warnings(record=True) as ws:
     warnings.simplefilter('always');model.fit(X[known],z,sample_weight=np.where(z==1,alpha,1.))
    assert not ws,[str(w.message) for w in ws];score=model.predict_proba(V)[:,list(model.classes_).index(1)]
   else:model={'constant':float(P>0)};score=np.full(V.shape[0],float(P>0))
   f=p.with_suffix('.pkl');pkl(f,model);files.append(f)
  else:raise ValueError(cfg['algorithm'])
  assert np.isfinite(score).all();sp=p.with_name(p.name+'_val.npy');npy(sp,score);files.append(sp)
  result.update(seconds=time.monotonic()-start,at=now(),model_files=[str(f.relative_to(E)) for f in files],success=True)
  jout(p.with_suffix('.json'),result);certificate(cert,files+[p.with_suffix('.json')],ct);return result
 except Exception as exc:
  result.update(success=False,error=repr(exc),traceback=traceback.format_exc(),seconds=time.monotonic()-start,at=now());jout(p.with_suffix('.FAILURE.json'),result);return result
def run(cfg):
 global X,V,Y,D,CONFIG,CTX
 no_test_guard();name=cfg['name'];done=E/f'candidates/{name}/DONE.json'
 if verified(done,{'config':cfg,'dev_sha256':sha(E/'data/dev.npz')}):log('candidate_resume_verified',name=name);return readj(done.parent/'summary.json')
 D=dev();Y=D['train_y'];CONFIG=cfg;X=feature(cfg['feature'],'train');V=feature(cfg['feature'],'val');start=time.monotonic()
 CTX={'config':cfg,'feature_DONE_sha256':sha(E/f"features/{cfg['feature']}/DONE.json"),'dev_sha256':sha(E/'data/dev.npz'),'code_sha256':sha(S/'experiments.py')}
 if cfg['algorithm']=='extra_trees':X=X.astype(np.float32);V=V.astype(np.float32)
 if sparse.issparse(X):X=X.tocsr();V=V.tocsr()
 with ProcessPoolExecutor(max_workers=24,mp_context=mp.get_context('fork')) as pool:
  jobs=[pool.submit(fit_one,j) for j in range(32)];rr=[f.result() for f in as_completed(jobs)]
 rr.sort(key=lambda r:r['class_index']);files=[]
 if not all(r['success'] for r in rr):
  jout(E/f'experiments/{name}/FAILED.json',{'config':cfg,'results':rr,'seconds':time.monotonic()-start});log('experiment_failed',name=name);return None
 score=np.column_stack([np.load(E/f'experiments/{name}/class_{j:02d}_val.npy',allow_pickle=False) for j in range(32)])
 for r in rr:
  j=r['class_index'];p=E/f'experiments/{name}/class_{j:02d}';files.extend(E/f for f in r['model_files']);files.extend([p.with_suffix('.json'),p.with_suffix('.DONE.json')])
 if cfg.get('reuse_baseline'):
  z=np.load(BASE/f"validation/{cfg['feature']}_{cfg['weight']}/C{cfg['C']:g}.npz",allow_pickle=False);assert np.array_equal(z['protein_id'],D['val_ids']);np.testing.assert_allclose(z['scores'],score,rtol=0,atol=1e-14)
 summary=register_candidate(name,cfg,score,files,{'fit_seconds':time.monotonic()-start,'class_fit_results':rr});X=V=None;return summary
def configurations(stage):
 configs=[]
 if stage=='E01':
  for f in ['A','B','C']:
   for w in ['weighted','unweighted']:
    for C in [0.1,1.,10.,100.,1000.]:configs.append(dict(feature=f,weight=w,C=C,reuse_baseline=C<=10,algorithm='logistic'))
 elif stage=='E02':
  for f in ['F','FA','FAB','A2','A5','A2B','A5B','O','FO','FABO']:
   for w in ['weighted','unweighted']:
    for C in [1.,10.,100.]:configs.append(dict(feature=f,weight=w,C=C,algorithm='logistic'))
 elif stage=='E03':
  for g in [1,4,16]:
   for f in [f'N{g}',f'AN{g}']:
    for C in [1.,10.,100.]:configs.append(dict(feature=f,weight='weighted',C=C,algorithm='logistic'))
 elif stage=='E04':
  for f in ['A','C','B']:
   for mf in ['sqrt',0.2]:
    for leaf in [1,3]:configs.append(dict(feature=f,weight='weighted',max_features=mf,min_samples_leaf=leaf,algorithm='extra_trees'))
 else:raise ValueError(stage)
 for i,c in enumerate(configs):c.update(name=f'{stage}_{i:03d}_{c["feature"]}',stage=stage,seed=SEED,order=i)
 return configs
def nystroem():
 for g in [1,4,16]:
  p=E/f'features/N{g}/DONE.json'
  if not verified(p):
   transform=Nystroem(kernel='rbf',gamma=g,n_components=1024,random_state=SEED,n_jobs=8)
   tr=transform.fit_transform(feature('B','train'));va=transform.transform(feature('B','val'))
   store_feature(f'N{g}',tr,va,{'type':'nystroem','source':'B','gamma':g,'n_components':1024,'seed':SEED,'normalization':'none','fit':'train only'},transform)
  f=f'AN{g}'
  if not verified(E/f'features/{f}/DONE.json'):
   x={s:sparse.hstack([feature('A',s),sparse.csr_matrix(feature(f'N{g}',s))],format='csr')/np.sqrt(2) for s in ['train','val']}
   store_feature(f,x['train'],x['val'],{'type':'concat','blocks':['A',f'N{g}'],'divide_sqrt_blocks':True})
def main():
 parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['E01','E02','E03','E04']);args=parser.parse_args();no_test_guard();assert verified(E/'data/DONE.json')
 configs=configurations(args.stage);jout(E/f'configurations/{args.stage}.json',configs)
 if args.stage=='E03':nystroem()
 for cfg in configs:
  if (E/'STOP_NEW_EXPERIMENTS').exists():log('stage_stopped_by_budget',stage=args.stage,next=cfg['name']);break
  run(cfg)
 log('stage_complete',stage=args.stage)
if __name__=='__main__':main()

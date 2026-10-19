from common import *
from experiments import register_candidate
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from scipy.special import expit
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp,warnings,traceback
X=V=Y=G=CFG=CTX=None
def fit(j):
 p=E/f"experiments/{CFG['name']}/class_{j:02d}";ct={**CTX,'class':j};cp=p.with_suffix('.DONE.json')
 if verified(cp,ct):return readj(p.with_suffix('.json'))
 started=time.monotonic();known=Y[:,j]!=-1;yy=Y[known,j];gg=G[known];uniq,inv,counts=np.unique(gg,return_inverse=True,return_counts=True);base=counts[inv].astype(float)**(-CFG['beta']);base/=base.mean()
 pw=float(base[yy==1].sum());nw=float(base[yy==0].sum());alpha=min(10.,np.sqrt(nw/pw)) if pw else 1.;weights=base*np.where(yy==1,alpha,1.);retry=[]
 assert pw>0 and nw>0
 for maxiter in [3000,10000]:
  m=LogisticRegression(C=CFG['C'],solver='lbfgs',tol=1e-6,max_iter=maxiter,random_state=SEED)
  with warnings.catch_warnings(record=True) as ws:
   warnings.simplefilter('always');m.fit(X[known],yy,sample_weight=weights)
  if ws:
   assert all(issubclass(w.category,ConvergenceWarning) for w in ws);retry.append({'max_iter':maxiter,'warnings':[str(w.message) for w in ws]})
   if maxiter==10000:raise RuntimeError('unresolved group-weighting convergence')
  else:break
 score=expit(V@m.coef_[0]+float(m.intercept_[0]));npz(p.with_suffix('.npz'),coef=m.coef_[0],intercept=np.array(float(m.intercept_[0])),constant=np.array(-1.));npy(p.with_name(p.name+'_val.npy'),score)
 meta={'class_index':j,'P':int((yy==1).sum()),'N':int((yy==0).sum()),'U':int((Y[:,j]==-1).sum()),'weighted_P':pw,'weighted_N':nw,'alpha':float(alpha),'component_count':len(uniq),
  'base_weight_min':float(base.min()),'base_weight_max':float(base.max()),'iterations':int(m.n_iter_[0]),'retries':retry,'seconds':time.monotonic()-started,'pid':os.getpid()}
 jout(p.with_suffix('.json'),meta);certificate(cp,[p.with_suffix('.npz'),p.with_name(p.name+'_val.npy'),p.with_suffix('.json')],ct);return meta
def main():
 global X,V,Y,G,CFG,CTX
 no_test_guard();d=dev();Y=d['train_y'];G=d['train_groups'];configs=[]
 for f in ['A','B','C']:
  for beta in [0.5,1.0]:
   for C in [1.,10.,100.]:configs.append({'name':f'E07_{len(configs):03d}_{f}','stage':'E07','algorithm':'group_logistic','feature':f,'beta':beta,'C':C,'seed':SEED,'order':len(configs)})
 jout(E/'configurations/E07.json',configs)
 for CFG in configs:
  name=CFG['name']
  if verified(E/f'candidates/{name}/DONE.json'):continue
  X=feature(CFG['feature'],'train');V=feature(CFG['feature'],'val');CTX={'config':CFG,'data_sha256':sha(E/'data/dev.npz'),'code_sha256':sha(S/'group_weighting.py'),'plan_addendum_sha256':sha(E/'PLAN_addendum_v1.md')}
  start=time.monotonic()
  with ProcessPoolExecutor(max_workers=16,mp_context=mp.get_context('fork')) as pool:rr=[f.result() for f in as_completed([pool.submit(fit,j) for j in range(32)])]
  scores=np.column_stack([np.load(E/f'experiments/{name}/class_{j:02d}_val.npy',allow_pickle=False) for j in range(32)]);files=[]
  for j in range(32):
   p=E/f'experiments/{name}/class_{j:02d}';files.extend([p.with_suffix('.npz'),p.with_suffix('.json'),p.with_suffix('.DONE.json')])
  register_candidate(name,CFG,scores,files,{'fit_seconds':time.monotonic()-start,'class_fit_results':sorted(rr,key=lambda r:r['class_index'])})
 log('group_weighting_complete')
if __name__=='__main__':main()

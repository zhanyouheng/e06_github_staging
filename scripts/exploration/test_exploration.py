from common import *
from bootstrap_core import pack,run_block
from metrics import evaluate,class_sets
from select_final import selector

def main():
 rng=np.random.default_rng(SEED);n=30;C=32;M=2;G=10;y=rng.choice([-1,0,1],size=(n,C));y[0]=1;y[1]=0;y[:,0]=0;y[0,0]=1
 score=np.round(rng.random((M,n,C)),1);thr=np.array([np.linspace(.1,.8,C),np.full(C,.4)]);group=np.arange(n)%G;counts=rng.integers(0,3,size=(6,G)).astype(np.uint16);counts[0]=1;counts[1]=0;counts[2,0]=0
 args=pack(y,score,thr,group);got,ba,bf=run_block(counts,*args);ka,kf=class_sets(y);checks=0
 for r,co in enumerate(counts):
  ix=np.repeat(np.arange(n),co[group])
  for m in range(M):
   records,sm=evaluate(y[ix],score[m,ix],thr[m],[str(j) for j in range(C)],group[ix])
   for k,key,active in [(0,'AP',ka),(1,'F1',kf)]:
    v=[records[j][key] for j in active]
    if any(z is None for z in v):assert np.isnan(got[r,m,k])
    else:assert abs(np.mean(v)-got[r,m,k])<1e-12
    checks+=1
 assert ba[1].all() and bf[1].all() and ba[2,0]
 pool=['first','second'];ss={name:score[i] for i,name in enumerate(pool)};fold=group%5
 for guarded in [False,True]:
  _,_,oof,meta=selector(pool,'first',guarded,fold,y,group,ss)
  changed=y.copy();changed[fold==0]=np.where(changed[fold==0]==1,0,1)
  _,_,oof2,meta2=selector(pool,'first',guarded,fold,changed,group,ss)
  assert np.array_equal(oof[fold==0],oof2[fold==0]),'held fold labels leaked into its selector';checks+=1
 jout(E/'validation/synthetic_tests.json',{'passed':True,'at':now(),'checks':checks,'tests':['per-class-threshold bootstrap matches sklearn with explicit component multiplicities','all-zero bootstrap counts and fixed macro NA support','selector held-fold labels cannot alter its held-fold predictions'],'seed':SEED,'test_set_used':False})
 log('exploration_synthetic_tests_passed',checks=checks)
if __name__=='__main__':main()

from common import *
from metrics import *
from select_evaluate import Selector
from sklearn.metrics import average_precision_score,f1_score
import numpy as np
def main():
    rng=np.random.default_rng(SEED);y=rng.choice([-1,0,1],size=(511,32),p=[.25,.6,.15]);s=np.round(rng.random(y.shape),1);w=rng.integers(0,5,len(y));ours=aps(y,s,w);errors=[]
    for j in range(32):
        m=y[:,j]>=0;v=average_precision_score(y[m,j],s[m,j],sample_weight=w[m]);errors.append(abs(v-ours[j]))
    assert max(errors)<1e-12
    th=thresholds(y,s)
    for j in range(32):
        m=y[:,j]>=0;candidates=np.r_[np.unique(s[:,j]),np.nextafter(s[:,j].max(),np.inf)];f=[f1_score(y[m,j],s[m,j]>=t,zero_division=0) for t in candidates];brute=max(t for t,a in zip(candidates,f) if abs(a-max(f))<1e-14);assert th[j]==brute
    # Weighted AP must exactly reproduce explicit repeated component/sample rows.
    expand=np.repeat(np.arange(len(y)),w);np.testing.assert_allclose(aps(y[expand],s[expand]),ours,atol=1e-14,rtol=0)
    # Changing UNKNOWN scores cannot change known-cell AP / confusion statistics.
    other=s.copy();other[y<0]=123;np.testing.assert_array_equal(aps(y,other),aps(y,s))
    # Selector weights for a held fold must be invariant to that fold's labels.
    fold=np.arange(len(y))%5;ss={'M00':s,'M01':np.round(rng.random(y.shape),2),'M04':np.round(rng.random(y.shape),2)};a=Selector(y,ss,fold);a1=a.build(fold!=0)[1];yy=y.copy();yy[fold==0]=rng.choice([-1,0,1],size=yy[fold==0].shape);b=Selector(yy,ss,fold);a2=b.build(fold!=0)[1];assert a1==a2
    n1=a.choose(fold!=0);n2=b.choose(fold!=0);assert n1[0]==n2[0] and n1[3]==n2[3]
    js(T/'audit/math_verification.json',{'at':now(),'status':'passed','AP_vs_sklearn_max_abs':max(errors),'thresholds_exact_bruteforce':True,'multiplicity_weights_equal_replication':True,'unknown_score_invariance':True,'held_fold_labels_cannot_change_selector_or_nested_family_choice':True,'seed':SEED,'synthetic_only':True})
    print('numeric and selection isolation checks passed')
if __name__=='__main__':main()

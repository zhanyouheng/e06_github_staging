"""Independent synthetic E13/E14 checks with no historical payload reads."""
from common import *
from sklearn.metrics import average_precision_score
from access_guard import install
install()
from fusion import ap_grid,pick,class_selector,BASELINE

def independent_ap(y,sc,fold):
    result=np.full((len(sc),6,32),np.nan)
    for m in range(len(sc)):
        for side in range(6):
            mask=np.ones(len(y),bool) if side==0 else fold!=side-1
            for j in range(32):
                k=mask&(y[:,j]!=-1)
                if np.any(y[k,j]==1) and np.any(y[k,j]==0):
                    result[m,side,j]=average_precision_score(y[k,j],sc[m,k,j])
    return result

def main():
    rng=np.random.default_rng(SEED)
    rows=130;fold=np.repeat(np.arange(5),rows//5)
    y=rng.choice([-1,0,1],size=(rows,32),p=[.45,.4,.15]).astype(np.int8)
    for k in range(5):y[k*26]=1;y[k*26+1]=0
    score=np.round(rng.random((6,rows,32)),1)
    score[5,:,:]=.5
    specs=[{'members':[BASELINE] if i==0 else [f'synthetic_{i}'],'space':'probability' if i<5 else 'cdf','complexity':i%3+1,'order':i} for i in range(6)]
    aps=ap_grid(y,score,fold);ref=independent_ap(y,score,fold)
    assert np.array_equal(np.isnan(aps),np.isnan(ref))
    err=float(np.nanmax(np.abs(aps-ref)));assert err<1e-12
    # An unknown cell cannot affect known-cell AP even if its score is extreme.
    changed=score.copy()
    for m in range(len(score)):changed[m][y==-1]=1e6
    assert np.allclose(ap_grid(y,changed,fold),aps,rtol=0,atol=1e-14,equal_nan=True)
    # Held-fold targets may change arbitrarily; that fold's selector output must not.
    options=list(range(5));macro=aps.mean(2)
    gchoices=[pick(macro[:,k+1],options,specs) for k in range(5)]
    selectors={shrink:class_selector(y,score,specs,aps,fold,shrink) for shrink in [False,True]}
    tested=0
    for k in range(5):
        altered=y.copy();held=fold==k
        altered[held]=rng.choice([-1,0,1],size=altered[held].shape)
        altered_aps=ap_grid(altered,score,fold)
        assert np.allclose(altered_aps[:,k+1],aps[:,k+1],rtol=0,atol=1e-14,equal_nan=True)
        assert pick(altered_aps.mean(2)[:,k+1],options,specs)==gchoices[k]
        for shrink in [False,True]:
            _,oof,_,trace=class_selector(altered,score,specs,altered_aps,fold,shrink)
            assert np.array_equal(oof[held],selectors[shrink][1][held])
            assert trace[k]==selectors[shrink][3][k]
            tested+=1
    # All positives of a class confined to held fold => selecting on other folds cannot fit that class.
    degenerate=y.copy();held=fold==2;degenerate[~held,0]=0;degenerate[held,0]=1
    degenerate[~held,1]=-1;degenerate[held,1]=np.arange(held.sum())%2
    da=ap_grid(degenerate,score,fold)
    assert np.isnan(da[:,3,0:2]).all()
    assert pick(da.mean(2)[:,3],options,specs) is None
    for shrink in [False,True]:
        _,oof,_,trace=class_selector(degenerate,score,specs,da,fold,shrink)
        assert np.array_equal(oof[held,:2],score[0,held,:2])
        assert trace[2]['class_grid_indices'][:2]==[0,0]
        assert trace[2]['global_grid_index']==0
    # Tie break is fixed complexity, then registered order, within the specified tolerance.
    assert pick(np.array([.7,.7000001,.69,.7,.6,.8]),[0,1,3],specs)==0
    result={'at':now(),'passed':True,'fusion_sha256':sha(S/'fusion.py'),'AP_comparisons':6*6*32,'AP_max_abs_diff_vs_sklearn':err,
            'ties_and_unknown_AP_checked':True,'heldfold_label_perturbation_global_checks':5,'heldfold_label_perturbation_class_checks':tested,
            'degenerate_selection_side_fallback_checks':4,'full_macro_classes':32,'test_set_read':False,'synthetic_only':True}
    jout(E/'audit/fusion_synthetic_tests.json',result)
    print(json.dumps(result))

if __name__=='__main__':main()

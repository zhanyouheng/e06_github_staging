"""Synthetic-only validation of paired bootstrap masking/counts/NA rules."""
from bootstrap_validation import *

def main():
    rng=np.random.default_rng(SEED);y=rng.choice([-1,0,1],size=(30,32)).astype(np.int8)
    y[:,0]=-1;y[:,1]=1;y[:,2]=0;y[:,3]=0;y[20:,3]=1
    groups=np.repeat(np.arange(3),10).astype(np.int32)
    scores=np.round(rng.random((2,30,32)),1)
    thresholds=np.stack([np.linspace(.1,.9,32),np.linspace(.2,.8,32)])
    counts=np.array([[3,0,0],[0,0,3],[1,1,1],[2,1,0]],dtype=np.uint16)
    kap,kfp=class_sets(y);args=pack(y,scores,thresholds,groups)
    values,ba,bf=run_block(counts,*args)
    checked=0
    for rep in range(len(counts)):
        ref=weighted_independent(y,scores,thresholds,counts[rep,groups].astype(np.float64),kap,kfp)
        assert np.allclose(values[rep],ref,rtol=0,atol=1e-12,equal_nan=True)
        weights=counts[rep,groups];p=((y==1)*weights[:,None]).sum(0);n=((y==0)*weights[:,None]).sum(0)
        assert np.array_equal(ba[rep],np.array([j in kap and (p[j]==0 or n[j]==0) for j in range(32)]))
        assert np.array_equal(bf[rep],np.array([j in kfp and p[j]==0 for j in range(32)]))
        checked+=4
    assert ba[0,3] and bf[0,3]
    assert ba[1,3] and not bf[1,3]
    assert not np.isnan(values[2]).any()
    assert 0 not in kap and 1 not in kap and 2 not in kap and 1 in kfp and 0 not in kfp and 2 not in kfp
    jout(E/'audit/bootstrap_synthetic_tests.json',{'at':now(),'passed':True,'macro_values_checked':checked,'ties':True,'duplicate_component_count_weights':True,'per_class_threshold_vectors':True,'fixed_K_AP_and_K_F1':True,'no_negative_AP_invalid_but_F1_valid_checked':True,'unknown_mask':True,'test_set_read':False,'synthetic_only':True,'core_sha256':sha(S/'bootstrap_core.py')})
    print('synthetic bootstrap checks passed')

if __name__=='__main__':main()

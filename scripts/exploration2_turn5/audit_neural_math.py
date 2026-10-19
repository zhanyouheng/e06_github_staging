"""Independent E12 arithmetic checks on synthetic arrays, never test-set data."""
import sys
from pathlib import Path
import numpy as np
from common import *

def prepare():
    from sklearn.metrics import average_precision_score
    rng=np.random.default_rng(SEED)
    labels=rng.choice([-1,0,1],size=(47,32),p=[.4,.4,.2]).astype(np.int8)
    labels[0,:]=1;labels[1,:]=0
    scores=np.round(rng.random(labels.shape),1)
    arrays={}
    for case in range(4):
        y=labels.copy();s=scores.copy()
        if case==1:s[:,:]=.5
        elif case==2:y[:,0]=-1;y[:,1]=0;y[:,2]=1
        elif case==3:s[y==-1]=rng.normal(100,20,size=(y==-1).sum())
        ap=[]
        for j in range(32):
            k=y[:,j]!=-1
            if np.any(y[:,j]==1) and np.any(y[:,j]==0):ap.append(float(average_precision_score(y[k,j],s[k,j])))
        arrays.update({f'y{case}':y,f's{case}':s,f'expected{case}':np.array(np.mean(ap))})
    npz(E/'audit/neural_synthetic_cases.npz',**arrays)

def check():
    import torch
    from neural import ap_numpy,pairwise
    with np.load(E/'audit/neural_synthetic_cases.npz',allow_pickle=False) as z:
        diffs=[abs(ap_numpy(z[f'y{k}'],z[f's{k}'])-float(z[f'expected{k}'])) for k in range(4)]
        y=z['y0'];score=z['s0']
    assert max(diffs)<1e-12
    # Fix generator state and known logits; vary all unknown logits substantially.
    rng=np.random.default_rng(SEED)
    initial=rng.normal(size=y.shape).astype(np.float32)
    modified=initial.copy();modified[y==-1]=rng.normal(1000,200,size=(y==-1).sum())
    a=torch.tensor(initial,requires_grad=True);b=torch.tensor(modified,requires_grad=True)
    la,ca=pairwise(a,y,np.random.default_rng(SEED))
    lb,cb=pairwise(b,y,np.random.default_rng(SEED))
    la.backward();lb.backward()
    assert la.item()==lb.item() and ca==cb
    assert torch.equal(a.grad,b.grad)
    assert np.all(a.grad.numpy()[y==-1]==0)
    # Full known-count-normalized BCE used by the trainer has the same mask property.
    for logits in [a,b]:logits.grad=None
    mask=torch.tensor(y!=-1,dtype=torch.float32);target=torch.tensor(y==1,dtype=torch.float32)
    losses=[]
    for logits in [a,b]:
        losses.append((torch.nn.functional.binary_cross_entropy_with_logits(logits,target,reduction='none')*mask).sum(0).div(mask.sum(0)).mean())
    for loss in losses:loss.backward()
    assert losses[0].item()==losses[1].item() and torch.equal(a.grad,b.grad)
    assert np.all(a.grad.numpy()[y==-1]==0)
    out={'at':now(),'passed':True,'AP_cases':4,'AP_max_abs_diff_vs_sklearn':max(diffs),'pairwise_unknown_logits_invariance':True,'pairwise_unknown_gradient_exact_zero':True,'BCE_unknown_logits_invariance':True,'BCE_unknown_gradient_exact_zero':True,'sampled_pairs':ca,'test_set_read':False,'neural_sha256':sha(S/'neural.py')}
    jout(E/'audit/neural_math.json',out)
    print(json.dumps(out))

if __name__=='__main__':
    from access_guard import install
    install()
    {'prepare':prepare,'check':check}[sys.argv[1]]()

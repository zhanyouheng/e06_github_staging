"""Independent synthetic checks of E12 known-mask normalization and ranking gradients."""
from neural import *

def main():
    no_test_guard();torch.set_num_threads(1)
    generator=np.random.default_rng(102)
    y=np.full((24,32),-1,dtype=np.int8)
    y[:2,0]=[1,0];y[:5,1]=[1,1,0,0,0];y[:20,2]=[1]*10+[0]*10
    y[:3,3]=1;y[:4,4]=0
    raw=generator.normal(size=y.shape).astype(np.float32)
    a=torch.tensor(raw,requires_grad=True)
    actual,n=pairwise(a,y,np.random.default_rng(42));actual.backward();grad=a.grad.clone()
    # Independent implementation: build per-class sampled means then mean of class means.
    b=torch.tensor(raw,requires_grad=True);rng=np.random.default_rng(42);class_losses=[];pair_count=0
    for j in range(32):
        positives=np.where(y[:,j]==1)[0];negatives=np.where(y[:,j]==0)[0]
        if len(positives)==0 or len(negatives)==0:continue
        count=min(64,len(positives)*len(negatives))
        ps=rng.choice(positives,size=count,replace=True);ns=rng.choice(negatives,size=count,replace=True)
        class_losses.append(torch.nn.functional.softplus(b[ns,j]-b[ps,j]).mean());pair_count+=count
    reference=torch.stack(class_losses).mean();reference.backward()
    torch.testing.assert_close(actual,reference,rtol=1e-6,atol=1e-7)
    torch.testing.assert_close(grad,b.grad,rtol=1e-6,atol=1e-7)
    assert n==pair_count==71
    assert torch.all(grad[torch.as_tensor(y==-1)]==0)
    assert torch.all(grad[torch.as_tensor(y==1)]<=0)
    assert torch.all(grad[torch.as_tensor(y==0)]>=0)
    assert torch.all(grad[:,3:]==0)
    y0=np.full_like(y,-1);c=torch.tensor(raw,requires_grad=True)
    loss0,n0=pairwise(c,y0,np.random.default_rng(42));loss0.backward()
    assert loss0.item()==0. and n0==0 and torch.all(c.grad==0)
    # BCE split weighted mean must recover full known-count-normalized loss and gradients.
    P,N=(y==1).sum(0),(y==0).sum(0)
    alpha=torch.tensor(np.minimum(10.,np.sqrt(N/np.maximum(P,1))).astype(np.float32))
    denom=torch.tensor(np.maximum(1,(y!=-1).sum(0)).astype(np.float32))
    targets=torch.tensor((y==1).astype(np.float32));mask=torch.tensor((y!=-1).astype(np.float32))
    full=torch.tensor(raw,requires_grad=True)
    r=torch.nn.functional.binary_cross_entropy_with_logits(full,targets,pos_weight=alpha,reduction='none')
    full_loss=((r*mask).sum(0)/denom).mean();full_loss.backward()
    split=torch.tensor(raw,requires_grad=True);losses=[]
    for inds in [np.arange(0,5),np.arange(5,17),np.arange(17,24)]:
        r=torch.nn.functional.binary_cross_entropy_with_logits(split[inds],targets[inds],pos_weight=alpha,reduction='none')
        mini=((r*mask[inds]).sum(0)/denom*(len(y)/len(inds))).mean()
        losses.append(mini*(len(inds)/len(y)))
    split_loss=torch.stack(losses).sum();split_loss.backward()
    torch.testing.assert_close(full_loss,split_loss,rtol=1e-6,atol=1e-7)
    torch.testing.assert_close(full.grad,split.grad,rtol=1e-6,atol=1e-7)
    assert torch.all(full.grad[torch.as_tensor(y==-1)]==0)
    record=dict(at=now(),passed=True,source_sha256=sha(S/'neural.py'),test_used=False,
        ranking_pair_count=n,ranking_loss=float(actual.detach()),ranking_reference=float(reference.detach()),
        ranking_gradient_max_abs_diff=float((grad-b.grad).abs().max()),
        BCE_full_loss=float(full_loss.detach()),BCE_partition_loss=float(split_loss.detach()),
        BCE_gradient_max_abs_diff=float((full.grad-split.grad).abs().max()),
        checks=['per-class sampled ranking mean then eligible-class mean matches independent reference',
                'at most 64 within-class known positive-negative pairs with replacement',
                'ranking positive gradient nonpositive and negative gradient nonnegative',
                'unknown ranking and BCE gradients exactly zero',
                'classes with no opposing known label have zero ranking contribution',
                'no eligible ranking classes returns differentiable zero',
                'size-weighted mini-batch BCE losses and gradients recover full known-count normalization'],
        conclusion='No implementation defect found in audited E12 ranking or BCE masking/normalization; class sampling and validation selection remain exploratory choices.')
    jout(E/'validation/E12_loss_audit.json',record)
    log('neural_loss_audit_passed',ranking_gradient_max_abs_diff=record['ranking_gradient_max_abs_diff'])

if __name__=='__main__':main()

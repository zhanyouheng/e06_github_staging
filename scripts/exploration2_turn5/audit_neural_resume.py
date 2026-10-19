"""Check deterministic recovery of one development-only ranking mini-batch."""
from neural import *

def main():
    no_test_guard()
    name=f'E12_001_C_s{SEED}'
    p=E/f'models/neural/{name}'
    cp=sorted((p/'checkpoints').glob('epoch_*.DONE.json'))[-1]
    assert verified(cp)
    ck=cp.with_name(cp.name.replace('.DONE.json','.pt'))
    device=configure(SEED,0)
    d=dev();y=d['train_y'];x=np.load(OLD/'features/C/train_gpu.npy',allow_pickle=False)
    results=[]
    for replicate in range(2):
        state=torch.load(ck,map_location='cpu',weights_only=False);cfg=state['config']
        net=Network(x.shape[1],cfg['output'],cfg['dropout']).to(device)
        net.load_state_dict(state['model'])
        opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001);opt.load_state_dict(state['optimizer'])
        gen=torch.Generator(device='cpu');gen.set_state(state['generator_state'])
        pairgen=np.random.default_rng();pairgen.bit_generator.state=state['pair_generator_state']
        torch.set_rng_state(state['cpu_rng']);torch.cuda.set_rng_state(state['cuda_rng'],device)
        random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng'])
        idx=torch.randperm(len(x),generator=gen)[:1024].numpy()
        xx=torch.as_tensor(x[idx],device=device)
        target=torch.as_tensor((y[idx]==1).astype(np.float32),device=device)
        mask=torch.as_tensor((y[idx]!=-1).astype(np.float32),device=device)
        P,N=(y==1).sum(0),(y==0).sum(0)
        pw=torch.as_tensor(np.minimum(10.,np.sqrt(N/np.maximum(P,1))).astype(np.float32),device=device)
        denom=torch.as_tensor(np.maximum(1,(y!=-1).sum(0)).astype(np.float32),device=device)
        net.train();opt.zero_grad(set_to_none=True)
        logits=net(xx)
        bce=(nn.functional.binary_cross_entropy_with_logits(logits,target,pos_weight=pw,reduction='none')*mask).sum(0)/denom*(len(x)/len(idx))
        rank,count=pairwise(logits,y[idx],pairgen)
        loss=bce.mean()+cfg['rank_weight']*rank
        loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5.);opt.step()
        results.append(dict(index=idx.copy(),loss=float(loss.detach()),pairs=count,
                            weights={k:v.detach().cpu().clone() for k,v in net.state_dict().items()},
                            generator=gen.get_state(),pair_generator=pairgen.bit_generator.state,
                            cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(device)))
        del net,opt
    a,b=results
    assert np.array_equal(a['index'],b['index']) and a['loss']==b['loss'] and a['pairs']==b['pairs']
    assert all(torch.equal(a['weights'][k],b['weights'][k]) for k in a['weights'])
    assert a['pair_generator']==b['pair_generator']
    for key in ['generator','cpu_rng','cuda_rng']:assert torch.equal(a[key],b[key])
    jout(E/'validation/E12_resume_check.json',dict(at=now(),passed=True,checkpoint=str(ck.relative_to(E)),
        checkpoint_sha256=sha(ck),checked='two independent disk checkpoint restores execute identical next shuffled ranking mini-batch',
        model_weights_bitwise_equal=True,loss_bitwise_equal=True,rng_states_bitwise_equal=True,test_used=False))
    log('neural_resume_check_passed',name=name)

if __name__=='__main__':main()

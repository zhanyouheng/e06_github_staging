from common import *
import numpy as np, torch, random, argparse, gc, traceback
from torch import nn
from metrics import macro_ap

class Network(nn.Module):
    def __init__(self,n,hidden,dropout):
        super().__init__();layers=[nn.LayerNorm(n,elementwise_affine=False)]
        for h in hidden:layers.extend([nn.Linear(n,h),nn.GELU(),nn.LayerNorm(h),nn.Dropout(dropout)]);n=h
        layers.append(nn.Linear(n,32));self.layers=nn.Sequential(*layers)
    def forward(self,x):return self.layers(x)

def configs():
    out=[]
    for family in ['M00','M02','M03','M04']:
        for route in (['A2B'] if family=='M04' else ['A2B','C']):
            for seed in [20261006,20261007,20261008]:out.append({'name':f'{family}_{route}_s{seed}','family':family,'route':route,'seed':seed,'hidden':[512,128] if family=='M04' else [512],'dropout':.5 if route=='C' else .2,'score_block':family=='M02','low_confidence':family=='M03'})
    return out

def save_pt(p,obj):
    atomic(p,lambda f:torch.save(obj,f))
    z=torch.load(p,map_location='cpu',weights_only=False);assert z['epoch']==obj['epoch']

def get_x(feature_set,route,score_block):
    x=np.load(T/'features'/feature_set/(route+'.npy'),mmap_mode='r')
    if score_block:
        norm=np.linalg.norm(x,axis=1,keepdims=True);x=np.concatenate([x/np.maximum(norm,1e-30),np.load(T/'features/annotation_score.npy')],axis=1)/np.float32(np.sqrt(2))
    return x

def fit(cfg,stage,feature_set,train_rows,held_rows,d,epochs=None):
    folder=T/'models/neural'/cfg['name']/stage
    if verified(folder):return json.loads((folder/'result.json').read_text())
    early=epochs is None;start=time.time();seed=cfg['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    x=get_x(feature_set,cfg['route'],cfg['score_block']);y=d['y'][train_rows];vy=d['y'][held_rows];P=(y==1).sum(0);N=(y==0).sum(0);alpha=np.minimum(10,np.sqrt(N/np.maximum(P,1))).astype(np.float32);alpha[(P>0)&(N==0)]=1
    ww=(y>=0).astype(np.float32);ww*=np.where(y==1,alpha[None,:],1.)
    if cfg['low_confidence']:
        sources=d['source_names'][d['source_code'][train_rows]];ww[(y==0)&(sources=='blindcheck_not_fit')]*=.25
    denom=ww.sum(0);constant=np.full(32,np.nan,np.float32);constant[P==0]=0;constant[(P>0)&(N==0)]=1
    xt=torch.as_tensor(np.array(x[train_rows]),device='cuda');vt=torch.as_tensor(np.array(x[held_rows]),device='cuda');yt=torch.as_tensor((y==1).astype(np.float32),device='cuda');wt=torch.as_tensor(ww,device='cuda');den=torch.as_tensor(np.maximum(denom,1),device='cuda')
    net=Network(x.shape[1],cfg['hidden'],cfg['dropout']).cuda();opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001);gen=torch.Generator().manual_seed(seed)
    start_epoch=0;hist=[];best=-1.;best_epoch=1;bad=0;elapsed=0.
    context={'config':cfg,'stage':stage,'feature_set':feature_set,'feature_done_sha':sha(T/'features'/feature_set/'DONE.json'),'labels_sha':sha(T/'data/development.npz'),'code_sha':sha(Path(__file__)),'epochs':epochs,'loss':'per-class effective-weight-mass normalized weighted BCE; unknown masked'}
    cj=folder/'last.certificate.json';cp=folder/'last.pt'
    if cj.exists():
        cert=json.loads(cj.read_text());assert cert['context']==context
        if cp.exists() and sha(cp)==cert['sha256']:
            q=torch.load(cp,map_location='cpu',weights_only=False);net.load_state_dict(q['model']);opt.load_state_dict(q['optimizer']);gen.set_state(q['generator']);torch.set_rng_state(q['cpu_rng']);torch.cuda.set_rng_state(q['cuda_rng']);start_epoch=q['epoch'];hist=q['history'];best=q['best'];best_epoch=q['best_epoch'];bad=q['bad'];elapsed=q['elapsed_seconds'];print('RESUME',cfg['name'],stage,start_epoch,flush=True)
    def infer():
        net.eval()
        with torch.inference_mode():s=torch.cat([net(vt[b:b+2048]).sigmoid().cpu() for b in range(0,len(vt),2048)]).numpy()
        s[:,np.isfinite(constant)]=constant[np.isfinite(constant)];assert np.isfinite(s).all();return s
    def cpu_state():return {k:v.detach().cpu() for k,v in net.state_dict().items()}
    maxepoch=120 if early else int(epochs)
    # If an interruption happened after an early-stop checkpoint, do not add epochs.
    iterator=[] if early and bad>=6 else range(start_epoch+1,maxepoch+1)
    for ep in iterator:
        net.train();order=torch.randperm(len(train_rows),generator=gen);loss_sum=0.
        for b in range(0,len(order),1024):
            idx=order[b:b+1024].cuda();opt.zero_grad(set_to_none=True)
            losses=nn.functional.binary_cross_entropy_with_logits(net(xt[idx]),yt[idx],reduction='none');loss=((losses*wt[idx]).sum(0)/den*(len(order)/len(idx))).mean()
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5.);opt.step();loss_sum+=float(loss.detach())*len(idx)/len(order)
        if ep==1 or ep%5==0 or ep==maxepoch:
            item={'epoch':ep,'loss':loss_sum,'seconds':elapsed+time.time()-start}
            if early:
                ap=macro_ap(vy,infer());assert ap is not None;item['inner_macro_AP']=ap
                if ap>best+1e-6:best=ap;best_epoch=ep;bad=0;save_pt(folder/'best.pt',{'epoch':ep,'model':cpu_state(),'config':cfg,'input_dim':x.shape[1],'constant':constant})
                else:bad+=1
            hist.append(item)
            save_pt(cp,{'epoch':ep,'model':cpu_state(),'optimizer':opt.state_dict(),'generator':gen.get_state(),'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),'history':hist,'best':best,'best_epoch':best_epoch,'bad':bad,'elapsed_seconds':elapsed+time.time()-start})
            js(cj,{'at':now(),'sha256':sha(cp),'context':context})
            print(json.dumps({'name':cfg['name'],'stage':stage,**item,'best_epoch':best_epoch if early else None,'bad':bad if early else None}),flush=True)
            if early and bad>=6:break
    if early:
        result={'chosen_epoch':best_epoch,'inner_macro_AP':best,'context':context,'seconds':elapsed+time.time()-start,'pid':os.getpid(),'history':hist}
        files=[folder/'best.pt',cp,cj]
    else:
        scores=infer();save_pt(folder/'model.pt',{'epoch':int(epochs),'model':cpu_state(),'config':cfg,'input_dim':x.shape[1],'constant':constant,'training_rows':train_rows,'feature_set':feature_set})
        z=torch.load(folder/'model.pt',map_location='cpu',weights_only=False);net.load_state_dict(z['model']);np.testing.assert_array_equal(infer(),scores)
        npz(folder/'predictions.npz',protein_id=d['protein_id'][held_rows],class_order=d['class_order'],scores=scores)
        result={'epochs':int(epochs),'context':context,'seconds':elapsed+time.time()-start,'pid':os.getpid(),'history':hist,'P':P.tolist(),'N':N.tolist(),'U':(y<0).sum(0).tolist(),'alpha':alpha.tolist(),'training_support_status':['no_known' if P[j]+N[j]==0 else 'P_zero' if P[j]==0 else 'N_zero' if N[j]==0 else 'regular' for j in range(32)],'negative_confidence_mass':(ww*(y==0)).sum(0).tolist(),'saved_model_predictions_bitwise_equal':True,'peak_gpu_allocated':torch.cuda.max_memory_allocated()};files=[folder/'model.pt',folder/'predictions.npz',cp,cj]
    js(folder/'result.json',result);done(folder,files+[folder/'result.json']);del net,opt,xt,vt,yt,wt;gc.collect();torch.cuda.empty_cache();return result

def run(cfg,d):
    f=d['train_fold'];ih=d['inner_holdout'];nt=len(f);chosen=[]
    for k in range(5):
        outer=f!=k;train=np.flatnonzero(outer&~ih);stop=np.flatnonzero(outer&ih)
        r=fit(cfg,'early'+str(k),'inner'+str(k),train,stop,d);chosen.append(r['chosen_epoch'])
        fit(cfg,'outer'+str(k),'outer'+str(k),np.flatnonzero(outer),np.flatnonzero(~outer),d,r['chosen_epoch'])
    ep=int(np.median(chosen));fit(cfg,'full','full',np.arange(nt),np.arange(nt,len(d['y'])),d,ep)
    folder=T/'models/neural'/cfg['name'];oof=np.zeros((nt,32),np.float32)
    for k in range(5):
        z=np.load(folder/('outer'+str(k))/'predictions.npz');idx=np.flatnonzero(f==k);assert np.array_equal(z['protein_id'],d['protein_id'][idx]);oof[idx]=z['scores']
    npz(folder/'oof.npz',protein_id=d['protein_id'][:nt],class_order=d['class_order'],scores=oof)
    js(folder/'recipe.json',{'config':cfg,'chosen_epochs':chosen,'full_epoch':ep,'preflight_sha256':sha(T/'preflight/DONE.json')})
    stages=['full']+['early'+str(k) for k in range(5)]+['outer'+str(k) for k in range(5)]
    done(folder,[folder/'recipe.json',folder/'oof.npz']+[folder/s/'DONE.json' for s in stages])

def main():
    a=argparse.ArgumentParser();a.add_argument('--worker',type=int,required=True);a.add_argument('--workers',type=int,default=2);args=a.parse_args()
    assert verified(T/'preflight') and verified(T/'features');assert torch.cuda.device_count()==1
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    js(T/f'env/gpu_worker_{args.worker}.json',{'at':now(),'pid':os.getpid(),'visible':os.environ.get('CUDA_VISIBLE_DEVICES'),'torch':torch.__version__,'CUDA':torch.version.cuda,'device':torch.cuda.get_device_name(),'architecture_list':torch.cuda.get_arch_list(),'numpy':np.__version__,'script_sha':sha(Path(__file__))})
    d=dict(np.load(T/'data/development.npz',allow_pickle=False));tasks=configs();failures=[]
    for i,cfg in enumerate(tasks):
        if i%args.workers!=args.worker:continue
        if verified(T/'models/neural'/cfg['name']):print('VERIFIED',cfg['name'],flush=True);continue
        try:run(cfg,d)
        except Exception as e:
            trace=traceback.format_exc();print(trace,flush=True);failures.append(cfg['name']);js(T/'failures'/(cfg['name']+'.json'),{'at':now(),'config':cfg,'exception':str(e),'trace':trace,'status':'failed_not_scored'});gc.collect();torch.cuda.empty_cache()
    js(T/f'logs/gpu_worker_{args.worker}_finished.json',{'at':now(),'pid':os.getpid(),'failures':failures})
if __name__=='__main__':main()

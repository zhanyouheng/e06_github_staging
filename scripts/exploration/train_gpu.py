from common import *
import torch,argparse,random,subprocess
from torch import nn

class Network(nn.Module):
 def __init__(self,dimensions,hidden,dropout):
  super().__init__();layers=[nn.LayerNorm(dimensions,elementwise_affine=False)];previous=dimensions
  for width in hidden:layers.extend([nn.Linear(previous,width),nn.GELU(),nn.LayerNorm(width),nn.Dropout(dropout)]);previous=width
  layers.append(nn.Linear(previous,32));self.layers=nn.Sequential(*layers)
 def forward(self,x):return self.layers(x)

def ap_numpy(y,score):
 out=[]
 for j in range(y.shape[1]):
  known=y[:,j]!=-1;target=y[known,j];values=score[known,j];P=(target==1).sum();N=(target==0).sum()
  if not P or not N:out.append(np.nan);continue
  order=np.argsort(-values,kind='stable');target=target[order];values=values[order]
  ends=np.r_[np.flatnonzero(values[:-1]!=values[1:]),len(values)-1];tp=np.cumsum(target==1)[ends];total=ends+1;increments=np.diff(np.r_[0,tp]);out.append(float(np.sum((increments/P)*(tp/total))))
 fixed=((y==1).sum(0)>0)&((y==0).sum(0)>0);return float(np.mean(np.array(out)[fixed]))
def save_torch(p,d):
 q=tmp(p);torch.save(d,q);r=torch.load(q,map_location='cpu',weights_only=False);assert r['epoch']==d['epoch'];commit(q,p)
def run(cfg,gpu):
 no_test_guard();name=cfg['name'];p=E/f'gpu/{name}';done=p/'DONE.json';context={'config':cfg,'dev_sha256':sha(E/'data/dev.npz'),'code_sha256':sha(S/'train_gpu.py'),
   'train_input_sha256':sha(E/f"features/{cfg['feature']}/train_gpu.npy"),'val_input_sha256':sha(E/f"features/{cfg['feature']}/val_gpu.npy")}
 if verified(done,context):log('gpu_resume_verified',name=name);return
 random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed']);torch.set_num_threads(2)
 torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 device=torch.device(f'cuda:{gpu}');torch.cuda.set_device(device);d=dev();y=d['train_y'];vy=d['val_y'];P=(y==1).sum(0);N=(y==0).sum(0)
 x=np.load(E/f"features/{cfg['feature']}/train_gpu.npy",allow_pickle=False);v=np.load(E/f"features/{cfg['feature']}/val_gpu.npy",allow_pickle=False)
 assert np.isfinite(x).all() and np.isfinite(v).all()
 xt=torch.as_tensor(x,device=device);vt=torch.as_tensor(v,device=device);yt=torch.as_tensor((y==1).astype(np.float32),device=device);mask=torch.as_tensor((y!=-1).astype(np.float32),device=device)
 alpha=np.minimum(10.,np.sqrt(N/np.maximum(P,1))).astype(np.float32);pw=torch.as_tensor(alpha,device=device);denom=mask.sum(0).clamp_min(1)
 net=Network(x.shape[1],cfg['hidden'],cfg['dropout']).to(device);opt=torch.optim.AdamW(net.parameters(),lr=0.001,weight_decay=0.0001)
 gen=torch.Generator(device='cpu').manual_seed(cfg['seed']);start_epoch=0;history=[];best=-1.;bad=0;best_epoch=0;start=time.monotonic();prior_elapsed=0.
 checkpoint=p/'last.pt';checkpoint_cert=p/'last.DONE.json'
 if verified(checkpoint_cert,context):
  state=torch.load(checkpoint,map_location='cpu',weights_only=False);net.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);gen.set_state(state['generator_state'])
  torch.set_rng_state(state['cpu_rng']);torch.cuda.set_rng_state(state['cuda_rng'],device);start_epoch=state['epoch'];history=state['history'];best=state['best'];bad=state['bad'];best_epoch=state['best_epoch'];prior_elapsed=state['elapsed_seconds']
 for epoch in range(start_epoch+1,121):
  net.train();order=torch.randperm(len(x),generator=gen);loss_sum=0.
  for begin in range(0,len(x),1024):
   idx=order[begin:begin+1024].to(device);opt.zero_grad(set_to_none=True)
   raw=nn.functional.binary_cross_entropy_with_logits(net(xt[idx]),yt[idx],pos_weight=pw,reduction='none')
   loss=((raw*mask[idx]).sum(0)/denom*(len(x)/len(idx))).mean();assert torch.isfinite(loss)
   loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5.);opt.step();loss_sum+=float(loss.detach())*len(idx)/len(x)
  if epoch==1 or epoch%5==0 or epoch==120:
   net.eval()
   with torch.inference_mode():scores=torch.cat([net(vt[b:b+2048]).sigmoid().cpu() for b in range(0,len(v),2048)]).numpy()
   assert np.isfinite(scores).all();ap=ap_numpy(vy,scores);history.append({'epoch':epoch,'loss':loss_sum,'validation_macro_AP':ap,'seconds':prior_elapsed+time.monotonic()-start})
   if ap>best+1e-6:
    best=ap;best_epoch=epoch;bad=0
    save_torch(p/'best.pt',{'epoch':epoch,'model':{k:t.detach().cpu() for k,t in net.state_dict().items()},'config':cfg,'input_dimensions':x.shape[1],'validation_macro_AP':ap})
    npz(p/'best_validation.npz',protein_id=d['val_ids'],class_order=d['class_order'],scores=scores)
   else:bad+=1
   state={'epoch':epoch,'model':{k:t.detach().cpu() for k,t in net.state_dict().items()},'optimizer':opt.state_dict(),'generator_state':gen.get_state(),'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(device),
     'history':history,'best':best,'best_epoch':best_epoch,'bad':bad,'elapsed_seconds':prior_elapsed+time.monotonic()-start,'config':cfg}
   save_torch(checkpoint,state);certificate(checkpoint_cert,[checkpoint,p/'best.pt',p/'best_validation.npz'],context)
   log('gpu_checkpoint',name=name,gpu=gpu,epoch=epoch,best_epoch=best_epoch,AP=ap,best_AP=best,bad_checks=bad)
   if bad>=6:break
 best_state=torch.load(p/'best.pt',map_location='cpu',weights_only=False);net.load_state_dict(best_state['model']);net.eval()
 with torch.inference_mode():scores=torch.cat([net(vt[b:b+2048]).sigmoid().cpu() for b in range(0,len(v),2048)]).numpy()
 with np.load(p/'best_validation.npz',allow_pickle=False) as z:np.testing.assert_allclose(scores,z['scores'],rtol=0,atol=0)
 meta={'config':cfg,'history':history,'best_epoch':best_epoch,'validation_macro_AP':best,'seconds':prior_elapsed+time.monotonic()-start,'gpu':gpu,'pid':os.getpid(),
   'seed':cfg['seed'],'peak_allocated_bytes':torch.cuda.max_memory_allocated(device),'peak_reserved_bytes':torch.cuda.max_memory_reserved(device),'torch':torch.__version__,'cuda':torch.version.cuda,
   'deterministic_algorithms':True,'tf32':False,'inference_dtype':'float32','early_stop_patience_checks':6,'max_epochs':120,'batch_size':1024,'loss':'per-class known-count-normalized weighted BCE; unknown multiplied by zero',
   'input_normalization':'LayerNorm across features within each protein, no data fit','optimizer':{'name':'AdamW','lr':0.001,'weight_decay':0.0001,'clip_grad_norm':5.0},'training_support':{'P':P.tolist(),'N':N.tolist(),'U':(y==-1).sum(0).tolist()},'test_used':False}
 jout(p/'history.json',meta);certificate(done,[p/'best.pt',p/'best_validation.npz',p/'history.json'],context);log('gpu_complete',name=name,best_AP=best,best_epoch=best_epoch)
 del net,opt,xt,vt,yt,mask;torch.cuda.empty_cache()

def configurations():
 cc=[]
 for feature_name in ['B','C']:
  for hidden in [[512],[512,256]]:
   for dropout in [0.2,0.5]:
    cc.append({'name':f'E05_{len(cc):03d}_{feature_name}_s{SEED}','stage':'E05','algorithm':'mlp','feature':feature_name,'hidden':hidden,'dropout':dropout,'seed':SEED,'weight':'weighted','order':len(cc)})
 return cc
def main():
 p=argparse.ArgumentParser();p.add_argument('--worker',type=int,default=0);p.add_argument('--config-file',type=Path);a=p.parse_args();assert torch.cuda.device_count()==4
 cfgs=readj(a.config_file) if a.config_file else configurations()
 for i,cfg in enumerate(cfgs):
  if i%4==a.worker:run(cfg,a.worker)
 log('gpu_worker_finished',worker=a.worker)
if __name__=='__main__':main()

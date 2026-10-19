from common import *
from train_gpu import Network,get_x,configs
import torch,numpy as np
def main():
    assert torch.cuda.device_count()==1;torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    d=np.load(T/'data/development.npz');nt=len(d['train_fold']);rows=[]
    for cfg in configs():
        root=T/'models/neural'/cfg['name'];assert verified(root);p=root/'full/model.pt';state=torch.load(p,map_location='cpu',weights_only=False);x=get_x('full',cfg['route'],cfg['score_block']);net=Network(state['input_dim'],cfg['hidden'],cfg['dropout']).cuda();net.load_state_dict(state['model']);net.eval();z=np.load(root/'full/predictions.npz');assert np.array_equal(z['protein_id'],d['protein_id'][nt:])
        xt=torch.as_tensor(np.array(x[nt:]),device='cuda')
        with torch.inference_mode():s=torch.cat([net(xt[b:b+2048]).sigmoid().cpu() for b in range(0,len(xt),2048)]).numpy()
        con=state['constant'];s[:,np.isfinite(con)]=con[np.isfinite(con)];err=float(np.max(np.abs(s-z['scores'])));assert err==0
        rows.append({'model':cfg['name'],'proteins':len(s),'output_dimensions':32,'max_abs_difference':err,'model_sha256':sha(p),'prediction_sha256':sha(root/'full/predictions.npz')});del net,xt;torch.cuda.empty_cache();print('replayed',cfg['name'],err,flush=True)
    js(T/'audit/gpu_model_replay.json',{'at':now(),'pid':os.getpid(),'gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'script_sha256':sha(Path(__file__)),'all_bitwise_equal':True,'models':rows})
if __name__=='__main__':main()

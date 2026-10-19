from common import *
from train_gpu import Network
import torch,argparse
def main():
 p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
 state=torch.load(E/f'gpu/{args.name}/best.pt',map_location='cpu',weights_only=False);cfg=state['config'];net=Network(state['input_dimensions'],cfg['hidden'],cfg['dropout']).cuda(0);net.load_state_dict(state['model']);net.eval()
 x=np.load(args.input,allow_pickle=False,mmap_mode='r');assert x.shape[1]==state['input_dimensions'];scores=[]
 with torch.inference_mode():
  for begin in range(0,len(x),2048):scores.append(net(torch.tensor(np.array(x[begin:begin+2048]),device='cuda:0')).sigmoid().cpu().numpy())
 out=np.concatenate(scores).astype(np.float64);assert np.isfinite(out).all();npy(args.output,out)
if __name__=='__main__':main()

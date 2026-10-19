"""Recompute only two fixed historical neural families on train/val development rows."""
from common import *
import argparse, random
import torch
from torch import nn

FAMILIES=('E05_ensemble_C','E09_ensemble_A2B')

class LegacyNetwork(nn.Module):
    def __init__(self, dimensions, hidden, dropout):
        super().__init__()
        layers=[nn.LayerNorm(dimensions,elementwise_affine=False)];previous=dimensions
        for width in hidden:
            layers.extend([nn.Linear(previous,width),nn.GELU(),nn.LayerNorm(width),nn.Dropout(dropout)])
            previous=width
        layers.append(nn.Linear(previous,32));self.layers=nn.Sequential(*layers)
    def forward(self,x):return self.layers(x)

def configs():
    all_configs=[]
    for family in FAMILIES:
        cfg=readj(OLD/f'candidates/{family}/summary.json')['config']
        assert cfg['algorithm']=='ensemble' and len(cfg['members'])==3
        for name in cfg['members']:
            member=readj(OLD/f'candidates/{name}/summary.json')['config']
            assert member['algorithm']=='mlp' and member['feature'] in ['C','A2B']
            assert member['name']==name and member['seed'] in [20261006,20261007,20261008]
            all_configs.append(member)
    return all_configs

def run(cfg,gpu):
    no_test_guard();name=cfg['name'];p=E/f'references/{name}'
    source=[OLD/f'candidates/{name}/summary.json',OLD/f'candidates/{name}/validation.npz',OLD/f'gpu/{name}/best.pt']
    source += [OLD/f"features/{cfg['feature']}/{split}_gpu.npy" for split in ['train','val']]
    context=dict(config=cfg,code_sha256=sha(S/'legacy_neural_inference.py'),
                 development_sha256=sha(E/'data/dev.npz'),source_sha256={str(f.relative_to(OLD)):sha(f) for f in source})
    if verified(p/'DONE.json',context):log('legacy_neural_resume_verified',name=name);return
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=torch.device(f'cuda:{gpu}');torch.cuda.set_device(device)
    state=torch.load(OLD/f'gpu/{name}/best.pt',map_location='cpu',weights_only=False)
    assert state['config']==cfg
    net=LegacyNetwork(state['input_dimensions'],cfg['hidden'],cfg['dropout']).to(device)
    net.load_state_dict(state['model']);net.eval();d=dev();start=time.monotonic();files=[]
    for split in ['train','val']:
        assert split in ['train','val']
        x=np.load(OLD/f"features/{cfg['feature']}/{split}_gpu.npy",allow_pickle=False)
        assert len(x)==len(d[f'{split}_ids']) and x.dtype==np.float32 and np.isfinite(x).all()
        xt=torch.as_tensor(x,device=device)
        with torch.inference_mode():score=torch.cat([net(xt[b:b+2048]).sigmoid().cpu() for b in range(0,len(x),2048)]).numpy().astype(np.float64)
        assert np.isfinite(score).all()
        if split=='val':
            with np.load(OLD/f'candidates/{name}/validation.npz',allow_pickle=False) as z:
                assert np.array_equal(z['protein_id'],d['val_ids']) and np.array_equal(z['class_order'],d['class_order'])
                diff=float(np.max(np.abs(score-z['scores'])))
                assert np.array_equal(score,z['scores']),diff
        f=p/('train_scores.npz' if split=='train' else 'validation.npz')
        npz(f,protein_id=d[f'{split}_ids'],class_order=d['class_order'],scores=score);files.append(f)
        del xt
    meta=dict(at=now(),name=name,source_sha256=context['source_sha256'],validation_max_abs_diff=0.,validation_bitwise_equal=True,
              seconds=time.monotonic()-start,gpu=gpu,pid=os.getpid(),torch=torch.__version__,cuda=torch.version.cuda,
              dtype='FP32 inference then exact float64 conversion',batch_size=2048,training_performed=False,test_used=False)
    jout(p/'inference.json',meta);files.append(p/'inference.json');certificate(p/'DONE.json',files,context)
    del net;torch.cuda.empty_cache();log('legacy_neural_member_complete',name=name,seconds=meta['seconds'])

def aggregate():
    no_test_guard();d=dev();files=[];records=[]
    for name in FAMILIES:
        cfg=readj(OLD/f'candidates/{name}/summary.json')['config'];p=E/f'references/{name}'
        sources=[OLD/f'candidates/{name}/summary.json',OLD/f'candidates/{name}/validation.npz']
        for split in ['train','val']:
            filename='train_scores.npz' if split=='train' else 'validation.npz';scores=[]
            for member in cfg['members']:
                mp=E/f'references/{member}';assert verified(mp/'DONE.json')
                with np.load(mp/filename,allow_pickle=False) as z:
                    assert np.array_equal(z['protein_id'],d[f'{split}_ids'])
                    assert np.array_equal(z['class_order'],d['class_order'])
                    assert z['scores'].dtype==np.float64
                    scores.append(z['scores'])
            score=np.mean(np.stack(scores),axis=0)
            if split=='val':
                with np.load(OLD/f'candidates/{name}/validation.npz',allow_pickle=False) as z:
                    assert np.array_equal(z['protein_id'],d['val_ids'])
                    diff=float(np.max(np.abs(score-z['scores'])))
                    assert np.array_equal(score,z['scores']),diff
            f=p/filename;npz(f,protein_id=d[f'{split}_ids'],class_order=d['class_order'],scores=score);files.append(f)
        record=dict(name=name,members=cfg['members'],aggregation='np.mean(np.stack(float64_member_scores),axis=0)',
                    validation_max_abs_diff=0.,validation_bitwise_equal=True,
                    source_sha256={str(f.relative_to(OLD)):sha(f) for f in sources},training_performed=False,test_used=False)
        jout(p/'inference.json',record);files.append(p/'inference.json');records.append(record)
        files += [E/f'references/{member}/DONE.json' for member in cfg['members']]
    certificate(E/'references/legacy_neural_DONE.json',files,dict(code_sha256=sha(S/'legacy_neural_inference.py'),
                development_sha256=sha(E/'data/dev.npz'),families=list(FAMILIES)),dict(records=records,training_performed=False,test_used=False))
    log('legacy_neural_inference_complete',families=list(FAMILIES),validation_max_abs_diff=0.)

def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',type=int,choices=range(4));p.add_argument('--aggregate',action='store_true');a=p.parse_args()
    if a.aggregate:aggregate();return
    assert a.worker is not None and torch.cuda.device_count()==4
    for i,cfg in enumerate(configs()):
        if i%4==a.worker:run(cfg,a.worker)

if __name__=='__main__':main()

"""Independent disk-model recomputation of a fixed eligible E11--E14 candidate.

Only train and val are accepted. No fitting, architecture choice or thresholds are
performed here. Frozen train-CDF references and exact registered arithmetic apply.
"""
from access_guard import install
install()
from common import *
import argparse, concurrent.futures, subprocess

ALLOWED_SPLITS=('train','val')

def summary_path(name):
    assert '/' not in name and '..' not in name
    return (E if name.startswith(('E11_','E12_','E13_','E14_')) else OLD)/f'candidates/{name}/summary.json'

def summary(name):return readj(summary_path(name))

def base_path(candidate):
    assert candidate.startswith(('E11_','E12_','E13_','E14_')) and '/' not in candidate and '..' not in candidate
    return E/f'recompute/{candidate}'

def grid_spec(index):return readj(E/'configurations/E13_grid.json')[index]

def children(cfg):
    if cfg['algorithm']=='class_fusion':
        indices=set(cfg['class_grid_indices'])
        if cfg['shrink']:indices.add(cfg['global_grid_index'])
        return sorted({n for i in indices for n in grid_spec(i)['members']})
    return cfg.get('members',[])

def closure(candidate):
    out={}
    def walk(name):
        if name in out:return
        cfg=summary(name)['config'];out[name]=cfg
        for child in children(cfg):walk(child)
    walk(candidate);return out

def feature_path(feature_name,split,gpu=False):
    assert split in ALLOWED_SPLITS
    if gpu:return OLD/f'features/{feature_name}/{split}_gpu.npy'
    for root in [E,OLD]:
        for suffix in ['npz','npy']:
            p=root/f'features/{feature_name}/{split}.{suffix}'
            if p.exists():return p
    raise FileNotFoundError((feature_name,split))

def leaf_sources(name,split):
    cfg=summary(name)['config'];algorithm=cfg['algorithm']
    files=[summary_path(name)]
    if algorithm in ['mlp','masked_mlp']:
        files.append((E/'models/neural' if name.startswith('E12_') else OLD/'gpu')/name/'best.pt')
        files.append(feature_path(cfg['feature'],split,gpu=True))
        if name.startswith('E12_'):files.append(E/f'models/neural/{name}/history.json')
    elif algorithm in ['extra_trees','random_forest']:
        parent=(E/'models/trees' if name.startswith('E11_') else OLD/'experiments')/name
        files += [parent/f'class_{j:02d}.pkl' for j in range(32)]
        files.append(feature_path(cfg['feature'],split))
    else:raise ValueError(('unsupported leaf',name,algorithm))
    return files

def leaf_context(candidate,name,split):
    return dict(candidate=candidate,name=name,split=split,code_sha256=sha(S/'recompute_dev.py'),
                dev_sha256=sha(E/'data/dev.npz'),source_sha256={str(f):sha(f) for f in leaf_sources(name,split)})

def node_paths(candidate,name,split):
    p=base_path(candidate)/'nodes'/name
    return p/f'{split}_scores.npz',p/f'{split}.DONE.json',p/f'{split}_recompute.json'

def save_node(candidate,name,split,scores,context,details):
    d=dev();assert split in ALLOWED_SPLITS
    assert scores.shape==(len(d[f'{split}_ids']),32) and np.isfinite(scores).all()
    p,cert,report=node_paths(candidate,name,split)
    record=dict(at=now(),name=name,split=split,rows=len(scores),dtype=str(scores.dtype),test_used=False,**details)
    if split=='val':
        reference=summary_path(name).parent/'validation.npz'
        with np.load(reference,allow_pickle=False) as z:
            assert np.array_equal(z['protein_id'],d['val_ids']) and np.array_equal(z['class_order'],d['class_order'])
            record['max_abs_diff']=float(np.max(np.abs(scores-z['scores'])))
            record['bitwise_equal']=bool(np.array_equal(scores,z['scores']))
        jout(report,record)
        assert record['bitwise_equal'],(name,record['max_abs_diff'])
    npz(p,protein_id=d[f'{split}_ids'],class_order=d['class_order'],scores=scores)
    jout(report,record);certificate(cert,[p,report],context)
    return scores

def cached_node(candidate,name,split,context):
    p,cert,_=node_paths(candidate,name,split)
    if verified(cert,context):
        with np.load(p,allow_pickle=False) as z:return z['scores']
    return None

def tree_leaf(candidate,name,split,workers):
    ctx=leaf_context(candidate,name,split)
    cached=cached_node(candidate,name,split,ctx)
    if cached is not None:return cached
    from scipy import sparse
    cfg=summary(name)['config'];x=feature(cfg['feature'],split)
    if name.startswith('E11_'):
        x=x.toarray() if sparse.issparse(x) else np.asarray(x)
        x=np.ascontiguousarray(x,dtype=np.float32)
    else:x=x.astype(np.float32)
    parent=(E/'models/trees' if name.startswith('E11_') else OLD/'experiments')/name
    def one(j):
        with open(parent/f'class_{j:02d}.pkl','rb') as f:model=pickle.load(f)
        if isinstance(model,dict):return np.full(len(dev()[f'{split}_ids']),model['constant'],dtype=np.float64)
        assert model.n_jobs==1
        return model.predict_proba(x)[:,list(model.classes_).index(1)]
    start=time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        scores=np.column_stack(list(executor.map(one,range(32))))
    return save_node(candidate,name,split,scores,ctx,dict(kind='tree_disk_models',workers=workers,seconds=time.monotonic()-start))

def gpu_leaf(candidate,name,split,gpu):
    import torch
    from torch import nn
    ctx=leaf_context(candidate,name,split)
    if cached_node(candidate,name,split,ctx) is not None:return
    cfg=summary(name)['config'];assert cfg['algorithm'] in ['mlp','masked_mlp']
    torch.set_num_threads(2);torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=torch.device(f'cuda:{gpu}');torch.cuda.set_device(device)
    model_file=(E/'models/neural' if name.startswith('E12_') else OLD/'gpu')/name/'best.pt'
    state=torch.load(model_file,map_location='cpu',weights_only=False)
    assert state['config']==cfg
    dimensions=state['input_dimensions']
    class NewNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.body=nn.Sequential(nn.LayerNorm(dimensions,elementwise_affine=False),nn.Linear(dimensions,512),
                nn.GELU(),nn.LayerNorm(512),nn.Dropout(cfg['dropout']))
            self.head=nn.Linear(512,32) if cfg['output']=='dense' else nn.Sequential(
                nn.Linear(512,int(cfg['output'].removeprefix('rank')),bias=False),nn.Linear(int(cfg['output'].removeprefix('rank')),32))
        def forward(self,x):return self.head(self.body(x))
    class LegacyNetwork(nn.Module):
        def __init__(self):
            super().__init__();layers=[nn.LayerNorm(dimensions,elementwise_affine=False)];previous=dimensions
            for width in cfg['hidden']:
                layers.extend([nn.Linear(previous,width),nn.GELU(),nn.LayerNorm(width),nn.Dropout(cfg['dropout'])]);previous=width
            layers.append(nn.Linear(previous,32));self.layers=nn.Sequential(*layers)
        def forward(self,x):return self.layers(x)
    net=(NewNetwork() if name.startswith('E12_') else LegacyNetwork()).to(device)
    net.load_state_dict(state['model']);net.eval()
    x=np.load(feature_path(cfg['feature'],split,gpu=True),allow_pickle=False)
    assert x.dtype==np.float32 and np.isfinite(x).all()
    xt=torch.as_tensor(x,device=device);start=time.monotonic()
    with torch.inference_mode():scores=torch.cat([net(xt[b:b+2048]).sigmoid().cpu() for b in range(0,len(x),2048)]).numpy()
    if name.startswith('E12_'):
        support=readj(E/f'models/neural/{name}/history.json')['training_support']
        P,N=np.array(support['P']),np.array(support['N'])
        scores[:,P==0]=0.;scores[:,(N==0)&(P>0)]=1.
    scores=scores.astype(np.float64)
    save_node(candidate,name,split,scores,ctx,dict(kind='neural_disk_model',gpu=gpu,inference_batch_size=2048,
        torch=torch.__version__,cuda=torch.version.cuda,tf32=False,seconds=time.monotonic()-start))
    log('recompute_neural_leaf_complete',name=name,split=split)

def run_gpu_leaves(candidate,split,names):
    if not names:return
    p=base_path(candidate)/'logs';p.mkdir(parents=True,exist_ok=True)
    python=ROOT.parent/'envs/seqscreen_esm/bin/python'
    def worker(gpu):
        for name in names[gpu::4]:
            if cached_node(candidate,name,split,leaf_context(candidate,name,split)) is not None:continue
            cmd=[str(python),str(S/'recompute_dev.py'),'--candidate',candidate,'--split',split,'--gpu-leaf',name,'--gpu',str(gpu)]
            with open(p/f'{name}_{split}.log','a',encoding='utf8') as output:
                proc=subprocess.Popen(cmd,cwd=str(ROOT.parent),stdout=output,stderr=subprocess.STDOUT)
                jout(p/f'{name}_{split}.pid.json',dict(at=now(),pid=proc.pid,command=cmd))
                result=proc.wait()
                assert result==0,(name,result,str(p/f'{name}_{split}.log'))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:list(executor.map(worker,range(4)))

def main_recompute(candidate,split,workers):
    assert split in ALLOWED_SPLITS
    graph=closure(candidate);leaf_names=[n for n,c in graph.items() if not children(c)]
    assert summary(candidate).get('selection_eligible',False),'candidate lacks final selection eligibility'
    assert graph[candidate]['algorithm'] in ['global_fusion','class_fusion','fixed_seed_mean','ensemble']
    neural=[n for n in leaf_names if graph[n]['algorithm'] in ['mlp','masked_mlp']]
    tree=[n for n in leaf_names if graph[n]['algorithm'] in ['extra_trees','random_forest']]
    assert set(neural+tree)==set(leaf_names)
    start=time.monotonic();run_gpu_leaves(candidate,split,neural)
    for name in tree:tree_leaf(candidate,name,split,workers)
    cache={};source_files={}
    def evaluate(name):
        if name in cache:return cache[name]
        cfg=graph[name];algorithm=cfg['algorithm']
        if name in leaf_names:
            result=cached_node(candidate,name,split,leaf_context(candidate,name,split));assert result is not None
            cache[name]=result;return result
        for child in children(cfg):evaluate(child)
        deps=[node_paths(candidate,child,split)[1] for child in children(cfg)]
        sources=[summary_path(name)]+deps
        if algorithm=='global_fusion' and cfg['space']=='cdf':sources += [E/f'fusion/cdf/{n}.npz' for n in cfg['members']]
        if algorithm=='class_fusion':sources.append(E/'configurations/E13_grid.json')
        ctx=dict(candidate=candidate,name=name,split=split,code_sha256=sha(S/'recompute_dev.py'),source_sha256={str(f):sha(f) for f in sources})
        existing=cached_node(candidate,name,split,ctx)
        if existing is not None:cache[name]=existing;return existing
        def member_value(member,space):
            values=evaluate(member)
            if space=='probability':return values
            assert space=='cdf'
            with np.load(E/f'fusion/cdf/{member}.npz',allow_pickle=False) as z:reference=z['sorted_train_scores']
            assert reference.shape==(len(dev()['train_ids']),32) and np.isfinite(reference).all()
            out=np.empty_like(values)
            for j in range(32):
                out[:,j]=(np.searchsorted(reference[:,j],values[:,j],side='left')+np.searchsorted(reference[:,j],values[:,j],side='right'))/(2*len(reference))
            return out
        def grid_value(index):
            spec=grid_spec(index)
            return sum(w*member_value(n,spec['space']) for n,w in zip(spec['members'],spec['weights']))
        if algorithm=='global_fusion':
            result=sum(w*member_value(n,cfg['space']) for n,w in zip(cfg['members'],cfg['weights']))
        elif algorithm=='class_fusion':
            gs={i:grid_value(i) for i in set(cfg['class_grid_indices'])}
            glob=grid_value(cfg['global_grid_index']) if cfg['shrink'] else None
            result=np.empty((len(dev()[f'{split}_ids']),32),dtype=np.float64)
            for j,i in enumerate(cfg['class_grid_indices']):
                lam=cfg['shrinkage_lambdas'][j]
                result[:,j]=lam*gs[i][:,j]+(1-lam)*glob[:,j] if cfg['shrink'] else gs[i][:,j]
        elif algorithm=='fixed_seed_mean' or summary(name).get('fixed_seed_mean',False):
            result=np.mean(np.stack([evaluate(child).astype(np.float64) for child in cfg['members']]),axis=0)
        elif algorithm=='ensemble':
            result=sum(w*evaluate(child) for child,w in zip(cfg['members'],cfg['weights']))
        else:raise ValueError(('unsupported graph node',name,algorithm))
        cache[name]=save_node(candidate,name,split,result,ctx,dict(kind=algorithm,arithmetic_order='registered member order, fixed means preserved'))
        return cache[name]
    final=evaluate(candidate);d=dev();output=base_path(candidate)
    final_file=output/f'{split}_scores.npz'
    details=dict(at=now(),candidate=candidate,split=split,leaf_models=leaf_names,leaf_count=len(leaf_names),graph_nodes=list(graph),
        recomputed_from_disk_models=True,grid_score_cache_read=False,CDF_reference='frozen train sorted scores only',
        seconds=time.monotonic()-start,code_sha256=sha(S/'recompute_dev.py'),training_performed=False,test_used=False)
    fields=dict(protein_id=d[f'{split}_ids'],class_order=d['class_order'],scores=final)
    if split=='val':
        with np.load(E/f'candidates/{candidate}/validation.npz',allow_pickle=False) as z:
            details['max_abs_diff']=float(np.max(np.abs(final-z['scores'])))
            details['score_bitwise_equal']=bool(np.array_equal(final,z['scores']))
            predictions=(final>=z['thresholds']).astype(np.uint8)
            details['prediction_bitwise_equal']=bool(np.array_equal(predictions,(z['scores']>=z['thresholds']).astype(np.uint8)))
            fields['thresholds']=z['thresholds'];fields['predictions']=predictions
        assert details['score_bitwise_equal'] and details['prediction_bitwise_equal']
    npz(final_file,**fields);jout(output/f'{split}_report.json',details)
    certificate(output/f'{split}.DONE.json',[final_file,output/f'{split}_report.json']+[node_paths(candidate,n,split)[1] for n in graph],
                dict(candidate=candidate,split=split,script_sha256=sha(S/'recompute_dev.py'),candidate_sha256=sha(summary_path(candidate))))
    log('independent_candidate_recompute_complete',**details)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--candidate',required=True)
    parser.add_argument('--split',choices=ALLOWED_SPLITS,default='val')
    parser.add_argument('--workers',type=int,default=16)
    parser.add_argument('--gpu-leaf')
    parser.add_argument('--gpu',type=int,choices=range(4),default=0)
    args=parser.parse_args()
    assert 1<=args.workers<=40
    if args.gpu_leaf:gpu_leaf(args.candidate,args.gpu_leaf,args.split,args.gpu)
    else:main_recompute(args.candidate,args.split,args.workers)

if __name__=='__main__':main()

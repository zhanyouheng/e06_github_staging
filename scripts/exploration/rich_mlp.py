from common import *
from experiments import register_candidate
from scipy import sparse
from sklearn.metrics import average_precision_score
import argparse

def main():
 no_test_guard();p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','initial','extra']);args=p.parse_args();d=dev()
 if args.mode=='prepare':
  configs=[]
  for feature_name in ['A2B','FAB']:
   for s in ['train','val']:
    x=feature(feature_name,s);x=x.toarray() if sparse.issparse(x) else x;npy(E/f'features/{feature_name}/{s}_gpu.npy',x.astype(np.float32))
   for dropout in [0.2,0.5]:configs.append({'name':f'E09_{len(configs):03d}_{feature_name}_s{SEED}','stage':'E09','algorithm':'mlp','feature':feature_name,'hidden':[512],'dropout':dropout,'seed':SEED,'weight':'weighted','order':len(configs)})
  jout(E/'configurations/E09_initial.json',configs);return
 metas=[]
 for cp in sorted((E/'gpu').glob('E09*/DONE.json')):
  assert verified(cp);meta=readj(cp.parent/'history.json');cfg=meta['config'];name=cfg['name'];metas.append(meta)
  if verified(E/f'candidates/{name}/DONE.json'):continue
  with np.load(cp.parent/'best_validation.npz',allow_pickle=False) as z:assert np.array_equal(z['protein_id'],d['val_ids']);score=z['scores'].astype(np.float64)
  ap=np.mean([average_precision_score(d['val_y'][d['val_y'][:,j]!=-1,j],score[d['val_y'][:,j]!=-1,j]) for j in range(32)]);assert abs(ap-meta['validation_macro_AP'])<1e-12
  register_candidate(name,cfg,score,[cp.parent/'best.pt',cp.parent/'best_validation.npz',cp.parent/'history.json',cp],{'selection_eligible':False,'reason':'single seed diagnostic; three fixed seed average only','fit_seconds':meta['seconds']})
 if args.mode=='initial':
  configs=[];selected={}
  for feature_name in ['A2B','FAB']:
   rr=sorted([m for m in metas if m['seed']==SEED and m['config']['feature']==feature_name],key=lambda m:(-m['validation_macro_AP'],m['config']['order']));assert len(rr)==2
   cfg=rr[0]['config'];selected[feature_name]={'first_seed_winner':cfg['name'],'config':cfg,'validation_macro_AP':rr[0]['validation_macro_AP']}
   for seed in [SEED+1,SEED+2]:configs.append({**cfg,'seed':seed,'name':cfg['name'].replace(f'_s{SEED}',f'_s{seed}')})
  jout(E/'selection/rich_mlp_architectures.json',selected);jout(E/'configurations/E09_extra.json',configs)
 else:
  selected=readj(E/'selection/rich_mlp_architectures.json')
  for f,sel in selected.items():
   base=sel['first_seed_winner'];members=[base.replace(f'_s{SEED}',f'_s{s}') for s in [SEED,SEED+1,SEED+2]];scores=[];files=[]
   for n in members:
    cp=E/f'candidates/{n}/DONE.json';assert verified(cp);files.append(cp)
    with np.load(cp.parent/'validation.npz',allow_pickle=False) as z:scores.append(z['scores'])
   name='E09_ensemble_'+f;cfg={'name':name,'stage':'E09','algorithm':'ensemble','members':members,'weights':[1/3]*3,'feature':f,'seed':SEED,'seeds':[SEED,SEED+1,SEED+2],'order':100 if f=='A2B' else 101}
   if not verified(E/f'candidates/{name}/DONE.json'):register_candidate(name,cfg,np.mean(scores,axis=0),files,{'selection_eligible':True,'fixed_seed_mean':True})
 log('rich_mlp_integration_complete',mode=args.mode)
if __name__=='__main__':main()

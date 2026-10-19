from common import *
from experiments import register_candidate
from sklearn.metrics import average_precision_score
import argparse

def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['initial','extra']);args=p.parse_args();no_test_guard();d=dev();available=[]
 for cp in sorted((E/'gpu').glob('*/DONE.json')):
  assert verified(cp);meta=readj(cp.parent/'history.json');cfg=meta['config'];name=cfg['name']
  with np.load(cp.parent/'best_validation.npz',allow_pickle=False) as z:
   assert np.array_equal(z['protein_id'],d['val_ids']);score=z['scores'].astype(np.float64)
  aps=[average_precision_score(d['val_y'][d['val_y'][:,j]!=-1,j],score[d['val_y'][:,j]!=-1,j]) for j in range(32)]
  assert abs(np.mean(aps)-meta['validation_macro_AP'])<1e-12
  done=E/f'candidates/{name}/DONE.json'
  if not verified(done):register_candidate(name,cfg,score,[cp.parent/'best.pt',cp.parent/'best_validation.npz',cp.parent/'history.json',cp],{'selection_eligible':False,'reason':'single-seed architecture/seed diagnostic; final MLP candidates are fixed three-seed means','fit_seconds':meta['seconds']})
  available.append(meta)
 if args.mode=='initial':
  assert len([m for m in available if m['seed']==SEED])==8;configs=[];selection={}
  for f in ['B','C']:
   records=[m for m in available if m['config']['feature']==f and m['seed']==SEED];records.sort(key=lambda m:(-m['validation_macro_AP'],sum(m['config']['hidden']),m['config']['order']))
   chosen=records[0];selection[f]={'first_seed_winner':chosen['config']['name'],'architecture':chosen['config'],'validation_macro_AP':chosen['validation_macro_AP'],'selection_uses_test':False}
   for seed in [SEED+1,SEED+2]:
    cfg={**chosen['config'],'name':chosen['config']['name'].replace(f'_s{SEED}',f'_s{seed}'),'seed':seed};configs.append(cfg)
  jout(E/'selection/mlp_architectures.json',selection);jout(E/'configurations/E05_extra_seeds.json',configs)
 else:
  selected=readj(E/'selection/mlp_architectures.json')
  for f in ['B','C']:
   base=selected[f]['first_seed_winner'];members=[base.replace(f'_s{SEED}',f'_s{s}') for s in [SEED,SEED+1,SEED+2]];ss=[];files=[]
   for n in members:
    cp=E/f'candidates/{n}/DONE.json';assert verified(cp);files.append(cp)
    with np.load(cp.parent/'validation.npz',allow_pickle=False) as z:ss.append(z['scores'])
   score=np.mean(ss,axis=0);name=f'E05_ensemble_{f}';cfg={'name':name,'stage':'E05','algorithm':'ensemble','members':members,'weights':[1/3]*3,'feature':f,'seed':SEED,'seeds':[SEED,SEED+1,SEED+2],'order':100 if f=='B' else 101}
   if not verified(E/f'candidates/{name}/DONE.json'):register_candidate(name,cfg,score,files,{'selection_eligible':True,'fixed_seed_mean':True})
 log('gpu_integration_complete',mode=args.mode)
if __name__=='__main__':main()

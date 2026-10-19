from access_guard import install
install()
from common import *
from registry import register_candidate
from metrics import class_sets,weighted_ap_sorted
from numba import njit,prange,set_num_threads
import itertools,argparse

BASELINE='E06_mix_098'
POOL=[BASELINE,'E04_006_C','E05_ensemble_C','E09_ensemble_A2B','E11_ensemble_extra_trees','E11_ensemble_random_forest','E12_ensemble_C','E12_ensemble_A2B']

@njit(cache=True,parallel=True)
def ap_grid(y,scores,fold):
 # Full validation and each four-fold selection side; held labels never used in its choice.
 nmodel,nrow,nclass=scores.shape
 out=np.empty((nmodel,6,nclass),np.float64)
 for task in prange(nmodel*nclass):
  m=task//nclass;j=task%nclass;idx=np.where(y[:,j]!=-1)[0]
  vals=scores[m,idx,j];order=np.argsort(-vals);idx=idx[order];vals=vals[order];lab=y[idx,j]
  for k in range(6):
   w=np.ones(idx.size,np.float64)
   if k>0:
    for i in range(idx.size):
     if fold[idx[i]]==k-1:w[i]=0.
   out[m,k,j]=weighted_ap_sorted(lab,vals,w)
 return out

def ap_columns(y,s):
 from sklearn.metrics import average_precision_score
 return np.array([float(average_precision_score(y[y[:,j]!=-1,j],s[y[:,j]!=-1,j])) if np.any(y[:,j]==1) and np.any(y[:,j]==0) else np.nan for j in range(32)])

def legacy_train():
 p=E/f'references/{BASELINE}/train_scores.npz'
 files=[E/f'references/{n}/train_scores.npz' for n in ['E09_ensemble_A2B','E05_ensemble_C','E04_006_C']]
 assert all(f.exists() for f in files),'old fixed member train inference incomplete'
 ctx={'member_hashes':[sha(f) for f in files]}
 if verified(p.with_suffix('.DONE.json'),ctx):return
 d=dev();arr=[]
 for f in files:
  with np.load(f,allow_pickle=False) as z:assert np.array_equal(z['protein_id'],d['train_ids']);arr.append(z['scores'])
 out=sum(w*s for w,s in zip([.2,.1,.7],arr))
 npz(p,protein_id=d['train_ids'],class_order=d['class_order'],scores=out)
 certificate(p.with_suffix('.DONE.json'),[p],ctx)

def train_score_path(name):
 if name.startswith('E11'):return E/f'models/trees/{name}/train_scores.npz'
 if name.startswith('E12'):return E/f'models/neural/{name}/train_scores.npz'
 return E/f'references/{name}/train_scores.npz'

def candidate_path(name):return E/('candidates' if name.startswith(('E11','E12')) else 'references')/name

def prepare_pool():
 if verified(E/'fusion/pool.DONE.json'):
  info=readj(E/'fusion/pool.json')
  for source in info['source_files'].values():
   assert sha(E/source['validation'])==source['validation_sha256']
   assert sha(E/source['train'])==source['train_sha256']
  with np.load(E/'fusion/pool_scores.npz',allow_pickle=False) as z:
   names=z['members'].tolist();features={n:{space:z[space][i] for space in ['probability','cdf']} for i,n in enumerate(names)}
  return names,features
 legacy_train();d=dev();names=[];sources={};features={}
 for name in POOL:
  v=candidate_path(name)/'validation.npz';t=train_score_path(name)
  if not v.exists() or not t.exists():
   raise RuntimeError(f'registered pool incomplete: {name}; mark branch stopped explicitly before excluding')
  with np.load(v,allow_pickle=False) as z:
   assert np.array_equal(z['protein_id'],d['val_ids']);vs=z['scores'].astype(np.float64)
  with np.load(t,allow_pickle=False) as z:
   assert np.array_equal(z['protein_id'],d['train_ids']);ts=z['scores'].astype(np.float64)
  assert vs.shape==(6257,32) and ts.shape==(18770,32) and np.isfinite(vs).all() and np.isfinite(ts).all()
  names.append(name);sources[name]={'validation':str(v.relative_to(E)),'validation_sha256':sha(v),'train':str(t.relative_to(E)),'train_sha256':sha(t)}
  reference=np.sort(ts,axis=0);cdf=np.empty_like(vs)
  for j in range(32):cdf[:,j]=(np.searchsorted(reference[:,j],vs[:,j],side='left')+np.searchsorted(reference[:,j],vs[:,j],side='right'))/(2*len(reference))
  p=E/f'fusion/cdf/{name}.npz';npz(p,sorted_train_scores=reference,val_scores=cdf,protein_id=d['val_ids'])
  features[name]={'probability':vs,'cdf':cdf}
 cfg={'at':now(),'members':names,'source_files':sources,'cdf_fit':'train only, all train proteins, label-independent empirical mid-CDF','test_used':False,'plan_addendum_sha256':sha(E/'PLAN_addendum_v1.md')}
 jout(E/'fusion/pool.json',cfg)
 npz(E/'fusion/pool_scores.npz',members=np.array(names),probability=np.stack([features[n]['probability'] for n in names]),cdf=np.stack([features[n]['cdf'] for n in names]),protein_id=d['val_ids'])
 certificate(E/'fusion/pool.DONE.json',[E/'fusion/pool.json',E/'fusion/pool_scores.npz']+[E/f'fusion/cdf/{n}.npz' for n in names],{'sources':sources})
 return names,features

def leaves(name):
 s=readj(candidate_path(name)/'summary.json') if name in POOL else (readj(E/f'candidates/{name}/summary.json') if (E/f'candidates/{name}/summary.json').exists() else readj(OLD/f'candidates/{name}/summary.json'))
 c=s['config']
 if 'members' not in c:return {name}
 result=set()
 for child in c['members']:result|=leaves(child)
 return result

def configurations(names):
 specs=[];new=[n for n in names if n.startswith(('E11','E12'))]
 for space in ['probability','cdf']:
  def add(members,weights,kind):
   leafset=set()
   for n in members:leafset|=leaves(n)
   specs.append({'name':f'E13_grid_{len(specs):03d}','space':space,'members':members,'weights':weights,'kind':kind,'complexity':len(leafset),'order':len(specs),'seed':SEED})
  for n in names:add([n],[1.],'single')
  for a,b in itertools.combinations(names,2):
   if a not in new and b not in new:continue
   for w in range(1,10):add([a,b],[w/10,1-w/10],'pair')
  tripleweights=sorted(set(itertools.permutations([.2,.2,.6]))|set(itertools.permutations([.2,.4,.4])))
  for a,b in itertools.combinations(new,2):
   for w in tripleweights:add([BASELINE,a,b],list(w),'triple')
  add(names,[1/len(names)]*len(names),'fixed_uniform')
 assert len(specs)<=500;return specs

def pick(values,indices,specs):
 good=[i for i in indices if np.isfinite(values[i])]
 if not good:return None
 best=max(values[i] for i in good)
 return min([i for i in good if best-values[i]<=1e-6],key=lambda i:(specs[i]['complexity'],specs[i]['order']))

def load_grid_scores():return np.load(E/'fusion/grid_scores.npy',mmap_mode='r',allow_pickle=False)

def grid():
 no_test_guard();done=E/'fusion/grid.DONE.json'
 if verified(done):return
 names,features=prepare_pool();specs=configurations(names);jout(E/'configurations/E13_grid.json',specs)
 d=dev();fold=np.load(E/'data/validation_selector_folds.npz',allow_pickle=False)['fold'];set_num_threads(24)
 allaps=[];blocks=[]
 for begin in range(0,len(specs),24):
  end=min(begin+24,len(specs));p=E/f'fusion/grid_blocks/{begin:03d}_{end-1:03d}.npz';ctx={'configs':specs[begin:end],'pool_sha256':sha(E/'fusion/pool.DONE.json'),'code_sha256':sha(S/'fusion.py')}
  if not verified(p.with_suffix('.DONE.json'),ctx):
   sc=np.stack([sum(w*features[n][c['space']] for n,w in zip(c['members'],c['weights'])) for c in specs[begin:end]])
   aps=ap_grid(d['val_y'],sc,fold);assert np.isfinite(aps).all(),'all selection sides must have fixed class support'
   npz(p,scores=sc,AP=aps);certificate(p.with_suffix('.DONE.json'),[p],ctx);log('fusion_grid_block',begin=begin,end=end,best_AP=float(aps[:,0,:].mean(1).max()))
  with np.load(p,allow_pickle=False) as z:allaps.append(z['AP'])
  blocks.append(p)
 aps=np.concatenate(allaps,axis=0);npz(E/'fusion/grid_AP.npz',AP=aps,macro_AP=aps.mean(2))
 target=E/'fusion/grid_scores.npy';temp=tmp(target);mm=np.lib.format.open_memmap(temp,mode='w+',dtype=np.float64,shape=(len(specs),len(d['val_ids']),32));offset=0
 for p in blocks:
  with np.load(p,allow_pickle=False) as z:mm[offset:offset+len(z['scores'])]=z['scores'];offset+=len(z['scores'])
 mm.flush();del mm;assert offset==len(specs);assert np.isfinite(np.load(temp,mmap_mode='r')).all();commit(temp,target)
 rec=[{'name':c['name'],'space':c['space'],'members':json.dumps(c['members']),'weights':json.dumps(c['weights']),'validation_macro_AP':float(aps[i,0].mean()),'complexity':c['complexity'],'selection_eligible':False,'status':'diagnostic_grid_point_only','seed':SEED} for i,c in enumerate(specs)]
 tsv(E/'fusion/grid_results.tsv',list(rec[0]),rec)
 certificate(done,[E/'configurations/E13_grid.json',E/'fusion/grid_AP.npz',target,E/'fusion/grid_results.tsv']+blocks,{'grid_points':len(specs),'code_sha256':sha(S/'fusion.py')})

def global_rules():
 no_test_guard();assert verified(E/'fusion/grid.DONE.json');d=dev();y=d['val_y'];specs=readj(E/'configurations/E13_grid.json');sc=load_grid_scores()
 aps=np.load(E/'fusion/grid_AP.npz')['AP'];macro=aps.mean(2);fold=np.load(E/'data/validation_selector_folds.npz')['fold']
 baseline=np.load(E/f'references/{BASELINE}/validation.npz')['scores'];baseap=float(ap_columns(y,baseline).mean());records=[]
 for space in ['probability','cdf']:
  indices=[i for i,c in enumerate(specs) if c['space']==space]
  for kind in ['learned_grid','fixed_uniform']:
   name=f'E13_{space}_{kind}';out=E/f'candidates/{name}'
   if verified(out/'DONE.json'):records.append(readj(out/'summary.json'));continue
   if kind=='fixed_uniform':chosen=next(i for i in indices if specs[i]['kind']=='fixed_uniform');choices=[chosen]*5
   else:chosen=pick(macro[:,0],indices,specs);choices=[pick(macro[:,k+1],indices,specs) for k in range(5)]
   assert chosen is not None;score=np.array(sc[chosen]);oof=np.empty_like(score);trace=[]
   for k,i in enumerate(choices):
    held=fold==k;oof[held]=baseline[held] if i is None else sc[i,held]
    trace.append({'fold':k,'selection_grid_index':i,'selected_configuration':None if i is None else specs[i],'held_proteins':int(held.sum()),'selection_proteins':int((~held).sum())})
   ap=float(ap_columns(y,oof).mean());eligible=ap>=baseap+.002
   p=E/f'fusion/crossfit/{name}.npz';npz(p,protein_id=d['val_ids'],scores=oof,fold=fold)
   check=E/f'fusion/crossfit/{name}.json';jout(check,{'name':name,'rule':kind,'space':space,'reference_model':BASELINE,'reference_macro_AP':baseap,'crossfit_macro_AP':ap,'crossfit_delta':ap-baseap,'eligibility_margin':.002,'selection_eligible':eligible,'fold_choices':trace,'full_configuration':specs[chosen],'held_labels_used_for_selection':False})
   c={'name':name,'stage':'E13','algorithm':'global_fusion','space':space,'members':specs[chosen]['members'],'weights':specs[chosen]['weights'],'grid_index':chosen,'rule':kind,'seed':SEED,'order':len(records),'complexity':specs[chosen]['complexity']}
   sources=[E/'fusion/grid.DONE.json',E/'fusion/pool.DONE.json',p,check]
   records.append(register_candidate(name,c,score,sources,{'selection_eligible':eligible,'crossfit_macro_AP':ap,'crossfit_delta':ap-baseap,'crossfit_reference':BASELINE,'eligibility_margin':.002}))
 jout(E/'fusion/E13_rules.json',[{k:r[k] for k in ['name','validation_macro_AP','selection_eligible','crossfit_macro_AP','crossfit_delta']} for r in records])

def class_selector(y,sc,specs,aps,fold,shrink=False):
 probability=[i for i,c in enumerate(specs) if c['space']=='probability'];base=next(i for i in probability if specs[i]['members']==[BASELINE]);macro=aps.mean(2)
 def choose(mask,column):
  glob=pick(macro[:,column],probability,specs);glob=base if glob is None else glob
  selected=[];lambdas=[];out=np.empty((len(y),32))
  for j in range(32):
   if not np.any(y[mask,j]==1) or not np.any(y[mask,j]==0):best=base;lam=1.;out[:,j]=sc[base,:,j]
   else:
    best=pick(aps[:,column,j],probability,specs);best=base if best is None else best;lam=1.
    if shrink:
     # All lambda selection uses only the training side of this validation fold.
     from sklearn.metrics import average_precision_score
     known=mask&(y[:,j]!=-1);vals=[]
     for x in [0.,.25,.5,.75,1.]:vals.append(float(average_precision_score(y[known,j],x*sc[best,known,j]+(1-x)*sc[glob,known,j])))
     mx=max(vals);lam=next(x for x,v in zip([0.,.25,.5,.75,1.],vals) if mx-v<=1e-6)
    out[:,j]=lam*sc[best,:,j]+(1-lam)*sc[glob,:,j] if shrink else sc[best,:,j]
   selected.append(best);lambdas.append(lam)
  return out,{'global_grid_index':glob,'class_grid_indices':selected,'shrinkage_lambdas':lambdas}
 full,fullcfg=choose(np.ones(len(y),bool),0);oof=np.empty_like(full);trace=[]
 for k in range(5):
  held=fold==k;score,cfg=choose(~held,k+1);oof[held]=score[held];trace.append({'fold':k,**cfg,'held_proteins':int(held.sum())})
 return full,oof,fullcfg,trace

def class_rules():
 no_test_guard();d=dev();y=d['val_y'];specs=readj(E/'configurations/E13_grid.json');sc=load_grid_scores();aps=np.load(E/'fusion/grid_AP.npz')['AP'];fold=np.load(E/'data/validation_selector_folds.npz')['fold']
 baseline=np.load(E/f'references/{BASELINE}/validation.npz')['scores'];baseap=float(ap_columns(y,baseline).mean());records=[]
 for shrink in [False,True]:
  name='E14_class_'+('shrunk' if shrink else 'free')
  if verified(E/f'candidates/{name}/DONE.json'):records.append(readj(E/f'candidates/{name}/summary.json'));continue
  score,oof,cfg,trace=class_selector(y,sc,specs,aps,fold,shrink);ap=float(ap_columns(y,oof).mean());eligible=ap>=baseap+.002
  check=E/f'fusion/crossfit/{name}.json';jout(check,{'name':name,'full_configuration':cfg,'fold_choices':trace,'reference_macro_AP':baseap,'crossfit_macro_AP':ap,'crossfit_delta':ap-baseap,'eligibility_margin':.002,'selection_eligible':eligible,'held_labels_used_for_selection':False})
  p=E/f'fusion/crossfit/{name}.npz';npz(p,protein_id=d['val_ids'],scores=oof,fold=fold)
  config={'name':name,'stage':'E14','algorithm':'class_fusion','space':'probability','shrink':shrink,'seed':SEED,'order':int(shrink),**cfg}
  records.append(register_candidate(name,config,score,[E/'fusion/grid.DONE.json',check,p],{'selection_eligible':eligible,'crossfit_macro_AP':ap,'crossfit_delta':ap-baseap,'eligibility_margin':.002}))
 jout(E/'fusion/E14_rules.json',[{k:r[k] for k in ['name','validation_macro_AP','selection_eligible','crossfit_macro_AP','crossfit_delta']} for r in records])

def main():
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['grid','global','class','all']);v=a.parse_args()
 if v.stage in ['grid','all']:grid()
 if v.stage in ['global','all']:global_rules()
 if v.stage in ['class','all']:class_rules()
if __name__=='__main__':main()

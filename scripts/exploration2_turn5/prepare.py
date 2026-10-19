from common import *
import shutil,sys,stat,importlib.metadata

def cp(source,target):
 h=sha(source)
 if target.exists():assert sha(target)==h;return
 q=tmp(target);shutil.copyfile(source,q);assert sha(q)==h;commit(q,target)

def metadata_snapshot():
 dest=E/'audit/history_metadata_before.json'
 if dest.exists():return
 records={}
 for p in sorted(ROOT.rglob('*')):
  if p.is_relative_to(E) or p==OUT/'REPORT_turn5.md':continue
  st=p.lstat()
  if not (stat.S_ISREG(st.st_mode) or stat.S_ISLNK(st.st_mode)):continue
  records[str(p.relative_to(ROOT))]={'size':st.st_size,'mtime_ns':st.st_mtime_ns,'mode':st.st_mode,'symlink_not_followed':stat.S_ISLNK(st.st_mode)}
 jout(dest,{'at':now(),'policy':'metadata only: no historical test or mixed-label content opened','files':records})

def main():
 no_test_guard();metadata_snapshot()
 sources={}
 for a,b in [('data/dev.npz','data/dev.npz'),('data/annotations_dev.tsv','data/annotations_dev.tsv'),('selection/validation_selector_folds.npz','data/validation_selector_folds.npz')]:
  cp(OLD/a,E/b);sources[str(OLD/a)]=sha(OLD/a)
 d=dev();assert not any('test' in k.lower() for k in d)
 assert len(d['train_ids'])==18770 and len(d['val_ids'])==6257
 assert len(set(d['train_ids'])&set(d['val_ids']))==0
 assert not set(d['train_groups'])&set(d['val_groups'])
 assert d['train_y'].shape==(18770,32) and d['val_y'].shape==(6257,32)
 assert set(np.unique(d['train_y']))=={-1,0,1}
 for name in ['A','B','C','A2','A5','A2B','A5B']:
  p=E/f'features/{name}';fs=[]
  cp(OLD/f'features/{name}/definition.json',p/'definition.json');fs.append(p/'definition.json')
  for split in ['train','val']:
   old=OLD/f'features/{name}/{split}.npz'
   if not old.exists():old=old.with_suffix('.npy')
   sources[str(old)]=sha(old)
   g=old.with_name(split+'_gpu.npy')
   if g.exists():sources[str(g)]=sha(g)
  these={k:v for k,v in sources.items() if f'/features/{name}/' in k}
  jout(p/'source_files.json',{'readonly_sources':these,'fit':'train only; inherited unchanged','no_test_data':True});fs.append(p/'source_files.json')
  certificate(p/'DONE.json',fs,{'feature':name,'dev_sha256':sha(E/'data/dev.npz')})
 for name in ['E06_mix_098','E04_006_C','E05_ensemble_C','E09_ensemble_A2B','E01_001_A','E01_011_B']:
  for f in ['summary.json','validation.npz']:
   src=OLD/f'candidates/{name}/{f}';cp(src,E/f'references/{name}/{f}');sources[str(src)]=sha(src)
 jout(E/'data/source_manifest.json',{'at':now(),'sources':sources,'test_read':False})
 rows_out=[]
 for split in ['train','val']:
  y=d[split+'_y'];rows_out.append({'split':split,'proteins':len(y),'components':len(set(d[split+'_groups'])),'P':int((y==1).sum()),'N':int((y==0).sum()),'U':int((y==-1).sum()),'excluded':0})
 tsv(E/'data/flow.tsv',list(rows_out[0]),rows_out)
 jout(E/'env/cpu_versions.json',{'python':sys.version,'environment':sys.prefix,'packages':{p:importlib.metadata.version(p) for p in ['numpy','scipy','scikit-learn','joblib','numba','threadpoolctl']},'cache_root':str(E/'cache')})
 certificate(E/'data/DONE.json',[E/'data'/n for n in ['dev.npz','annotations_dev.tsv','validation_selector_folds.npz','source_manifest.json','flow.tsv']],{'PLAN_sha256':sha(E/'PLAN.md')})
 log('turn5_prepared',train=18770,val=6257,test_opened=False)

if __name__=='__main__':main()

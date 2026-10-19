import csv,hashlib,json,os,time,pickle
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
ROOT=Path('/disk1/shengan_group/zyh/gpu_handoff_pack_20261005')
OUT=ROOT/'output';BASE=OUT/'training';E=OUT/'exploration';S=E/'scripts';SEED=20261006
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,allow_nan=False).encode()).hexdigest()
def readj(p):return json.loads(Path(p).read_text())
def rows(p):
 with open(p,newline='',encoding='utf8') as f:yield from csv.DictReader(f,delimiter='\t')
def tmp(p):
 p=Path(p);assert p.is_relative_to(E) or p==OUT/'REPORT_turn4.md',str(p);p.parent.mkdir(parents=True,exist_ok=True);return p.with_name(p.name+f'.{os.getpid()}.tmp')
def commit(q,p):
 with open(q,'rb') as f:os.fsync(f.fileno())
 os.replace(q,p);fd=os.open(Path(p).parent,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
def jout(p,d):
 q=tmp(p);q.write_text(json.dumps(d,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n');assert readj(q)==d;commit(q,p)
def txt(p,d):q=tmp(p);q.write_text(d,encoding='utf8');assert q.read_text()==d;commit(q,p)
def tsv(p,fields,data):
 q=tmp(p);n=0
 with open(q,'w',newline='',encoding='utf8') as f:
  w=csv.DictWriter(f,fields,delimiter='\t',lineterminator='\n');w.writeheader()
  for r in data:w.writerow({k:'NA' if r.get(k) is None else r[k] for k in fields});n+=1
 assert sum(1 for _ in rows(q))==n;commit(q,p)
def npz(p,**d):
 q=tmp(p)
 with open(q,'wb') as f:np.savez(f,**d)
 with np.load(q,allow_pickle=False) as z:
  for k,v in d.items():assert np.array_equal(z[k],np.asarray(v),equal_nan=True) if np.asarray(v).dtype.kind in 'fc' else np.array_equal(z[k],v)
 commit(q,p)
def npy(p,a):
 q=tmp(p)
 with open(q,'wb') as f:np.save(f,a,allow_pickle=False)
 b=np.load(q,mmap_mode='r',allow_pickle=False);assert np.array_equal(a,b);commit(q,p)
def pkl(p,d):
 q=tmp(p)
 with open(q,'wb') as f:pickle.dump(d,f,protocol=5)
 with open(q,'rb') as f:z=pickle.load(f)
 assert type(z)==type(d);commit(q,p)
def snpz(p,a):
 from scipy import sparse
 q=tmp(p)
 with open(q,'wb') as f:sparse.save_npz(f,a,compressed=False)
 b=sparse.load_npz(q);assert b.shape==a.shape and (b!=a).nnz==0;commit(q,p)
def certificate(p,files,context,extra=None):
 d={'at':now(),'context':context,'files':{str(Path(f).relative_to(E)):sha(f) for f in files},'status':'verified_done'}
 if extra:d.update(extra)
 jout(p,d)
def verified(p,context=None):
 if not Path(p).exists():return False
 d=readj(p)
 if context is not None:assert d['context']==context,('context mismatch',str(p))
 for f,h in d['files'].items():assert sha(E/f)==h,('checkpoint checksum mismatch',f)
 return True
def log(event,**kw):print(json.dumps({'time':now(),'pid':os.getpid(),'event':event,**kw},ensure_ascii=False,allow_nan=False),flush=True)
def dev():
 with np.load(E/'data/dev.npz',allow_pickle=False) as d:return {k:d[k] for k in d.files}
def feature(name,split):
 from scipy import sparse
 p=E/f'features/{name}/{split}.npz'
 return sparse.load_npz(p) if p.exists() else np.load(p.with_suffix('.npy'),allow_pickle=False,mmap_mode='r')
def store_feature(name,train,val,definition,transform=None):
 from scipy import sparse
 p=E/f'features/{name}';ct={'definition':definition,'dev_sha256':sha(E/'data/dev.npz')}
 if verified(p/'DONE.json',ct):return
 fs=[]
 for split,x in [('train',train),('val',val)]:
  f=p/(split+('.npz' if sparse.issparse(x) else '.npy'))
  if sparse.issparse(x):snpz(f,x)
  else:npy(f,x)
  fs.append(f)
 jout(p/'definition.json',definition);fs.append(p/'definition.json')
 if transform is not None:pkl(p/'transform.pkl',transform);fs.append(p/'transform.pkl')
 certificate(p/'DONE.json',fs,ct,{'train_shape':list(train.shape),'val_shape':list(val.shape)})
def no_test_guard():
 assert not (E/'selection/final_selection.json').exists(),'exploration has ended; model selection cannot resume after final freeze'


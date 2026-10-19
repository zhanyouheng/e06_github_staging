"""Turn13: all writes are confined to the new training tree; no legacy imports."""
import os, json, hashlib, csv, io, time
from pathlib import Path
from datetime import datetime, timezone
W=Path('/disk1/shengan_group/zyh')
O=W/'gpu_handoff_pack_20261005/output'
V=O/'v2'
T=V/'training'
N=W/'gpu_handoff_pack_20261008'
SEED=20261012
os.environ['PYTHONDONTWRITEBYTECODE']='1'
for key,sub in {'TMPDIR':'tmp','TMP':'tmp','TEMP':'tmp','XDG_CACHE_HOME':'xdg','TORCH_HOME':'torch','HF_HOME':'hf','PIP_CACHE_DIR':'pip','CONDA_PKGS_DIRS':'conda_pkgs','MPLCONFIGDIR':'mpl','NUMBA_CACHE_DIR':'numba','CUDA_CACHE_PATH':'cuda','TORCHINDUCTOR_CACHE_DIR':'inductor'}.items():
    p=T/'cache'/sub;p.mkdir(parents=True,exist_ok=True);os.environ[key]=str(p)
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def destination(p):
    p=Path(p).absolute()
    assert p.is_relative_to(T) and p.resolve().is_relative_to(T.resolve()),str(p)
    p.parent.mkdir(parents=True,exist_ok=True)
    return p
def atomic(p,writer,mode='wb'):
    p=destination(p);q=p.with_name(p.name+'.tmp.'+str(os.getpid()))
    with open(q,mode) as f:writer(f);f.flush();os.fsync(f.fileno())
    os.replace(q,p)
def write_text(p,s):atomic(p,lambda f:f.write(s), 'w')
def js(p,obj):write_text(p,json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def tsv(p,rows,fields=None):
    rows=list(rows);fields=fields or (list(rows[0]) if rows else [])
    def save(f):
        w=csv.DictWriter(f,fieldnames=fields,delimiter='\t',lineterminator='\n',extrasaction='ignore');w.writeheader();w.writerows(rows)
    atomic(p,save,'w')
def read_tsv(p):
    with open(p) as f:return list(csv.DictReader(f,delimiter='\t'))
def projected_tsv(p,allowed,id_col='ac'):
    """Decode label/source fields only AFTER the accession whitelist check."""
    with open(p) as f:
        head=f.readline().rstrip('\r\n').split('\t');j=head.index(id_col)
        for line in f:
            ac=line.split('\t',j+1)[j].rstrip('\r\n')
            if ac in allowed:yield dict(zip(head,line.rstrip('\r\n').split('\t')))
def npz(p,**arrays):
    import numpy as np
    atomic(p,lambda f:np.savez_compressed(f,**arrays))
def npy(p,a):
    import numpy as np
    atomic(p,lambda f:np.save(f,a,allow_pickle=False))
def done(folder,files,**info):
    folder=destination(Path(folder)/'DONE.json').parent
    js(folder/'DONE.json',dict(status='complete',at=now(),files={str(Path(p).relative_to(T)):sha(p) for p in files},**info))
def verified(folder):
    p=Path(folder)/'DONE.json'
    if not p.exists():return False
    d=json.loads(p.read_text())
    return d.get('status')=='complete' and all((T/k).is_file() and sha(T/k)==v for k,v in d['files'].items())
def history_snapshot():
    result={}
    # lstat only: test artifacts are NOT opened. No symlink traversal.
    for base in [W/'gpu_handoff_pack_20261005',N]:
        for root,dirs,files in os.walk(base,followlinks=False):
            dirs[:]=[d for d in dirs if Path(root)/d != T and d!='_agent_dialog']
            for name in files:
                p=Path(root)/name
                if p.is_symlink():continue
                s=p.lstat();result[str(p.relative_to(W))]=[s.st_size,s.st_mtime_ns,s.st_mode]
    return result

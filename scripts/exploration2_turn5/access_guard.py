"""Optional Turn 5 Python audit boundary, installed before any development IO.

This guards Python-audited opens and filesystem mutations; it is not an OS sandbox
and cannot claim to intercept native extension IO. Historical test/mixed payloads
are denied by a positive whitelist, including reads for checksum purposes.
"""
import atexit
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from datetime import datetime, timezone

WORK=Path('/disk1/shengan_group/zyh')
ROOT=WORK/'gpu_handoff_pack_20261005'
OUT=ROOT/'output'
OLD=OUT/'exploration'
E=OUT/'exploration2'
REPORT=OUT/'REPORT_turn5.md'
_installed=False
_history_reads=set()
_denied=[]
_dependency_reads=set()
_install_time=None


def _in(path, parent):
    return path == parent or parent in path.parents


def _historical_allowed(path):
    """Positive development whitelist. No historical manifest traversal."""
    if _in(path, OLD):
        r=path.relative_to(OLD)
        parts=r.parts
        if len(parts)==2 and parts[0]=='scripts' and r.suffix in {'.py','.sh'}:
            return True
        if str(r) in {'data/dev.npz','data/annotations_dev.tsv','selection/validation_selector_folds.npz'}:
            return True
        if len(parts)==3 and parts[0]=='features':
            return parts[2] in {'train.npz','train.npy','val.npz','val.npy','train_gpu.npy','val_gpu.npy','definition.json','transform.pkl','DONE.json'}
        if len(parts)==3 and parts[0]=='candidates':
            return parts[2] in {'summary.json','validation.npz','per_class.tsv','DONE.json'}
        if len(parts)==3 and parts[0]=='experiments':
            return bool(re.fullmatch(r'class_\d\d(?:\.npz|\.pkl|\.json|\.DONE\.json|_val\.npy|_train\.npy)',parts[2]))
        if len(parts)==3 and parts[0]=='gpu':
            return parts[2] in {'best.pt','history.json','best_validation.npz','DONE.json'}
        return False
    base=OUT/'training'
    if _in(path,base):
        r=path.relative_to(base);parts=r.parts
        if len(parts)==2 and parts[0]=='scripts' and r.suffix in {'.py','.sh'}:
            return True
        if str(r) in {'features/B_preprocessing.npz','features/vocabulary.tsv'}:
            return True
        if len(parts)>=3 and parts[0]=='models' and parts[1] in {'selected','candidates'}:
            return r.suffix in {'.npz','.pkl','.json'} and not any('test' in p.lower() for p in parts)
        return False
    return False


def _normalize(path):
    # Resolving links prevents an E2-named link from granting write access outside E2.
    if isinstance(path,int):
        return None
    if isinstance(path,bytes):
        path=os.fsdecode(path)
    return Path(os.path.realpath(os.path.abspath(os.fspath(path))))


def _write_allowed(path):
    return _in(path,E) or path==REPORT or (path.parent==REPORT.parent and re.fullmatch(r'REPORT_turn5\.md\.\d+\.tmp',path.name) is not None)


def _reject(event,path,reason):
    item={'event':event,'path':str(path),'reason':reason}
    _denied.append(item)
    raise PermissionError('Turn 5 access guard: '+reason+': '+str(path))


def _check(path,write,event):
    p=_normalize(path)
    if p is None:
        # os.open receives a path and is checked before obtaining a descriptor.
        # Existing standard streams and validated descriptors may be wrapped.
        return
    if write:
        if not _write_allowed(p):_reject(event,p,'writes restricted to exploration2 or REPORT_turn5')
        return
    if _in(p,E) or p==REPORT:return
    if _in(p,ROOT):
        if not _historical_allowed(p):_reject(event,p,'historical content absent from development whitelist')
        _history_reads.add(str(p));return
    # Normal interpreter/module resources are inside the already existing mamba environments.
    if _in(p,WORK/'envs'):
        _dependency_reads.add(str(p));return
    _reject(event,p,'read outside development roots and mamba environments')


def _hook(event,args):
    if event=='open':
        path,mode,flags=args
        writing=(isinstance(mode,str) and any(x in mode for x in 'wax+')) or bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
        _check(path,writing,event)
    elif event in {'os.remove','os.rmdir','os.mkdir','os.chmod','os.chown','os.truncate','os.utime'}:
        _check(args[0],True,event)
    elif event in {'os.rename','os.link'}:
        _check(args[0],True,event);_check(args[1],True,event)
    elif event=='os.symlink':
        # New symbolic links are unnecessary for this turn and obscure provenance.
        _reject(event,args[1],'symlink creation not permitted')


def flush():
    if not _installed:return
    dest=E/'audit'/f'access_{os.getpid()}.json'
    dest.parent.mkdir(parents=True,exist_ok=True)
    value={'pid':os.getpid(),'installed_at':_install_time,'at':datetime.now(timezone.utc).isoformat(),
           'historical_read_paths':sorted(_history_reads),'historical_read_count':len(_history_reads),
           'denied_accesses':_denied,'dependency_read_count':len(_dependency_reads),
           'test_payload_reads_allowed':False,'guard_scope':'Python-audited open/mutation events; native library IO is not universally intercepted'}
    tmp=dest.with_name(dest.name+'.tmp')
    with open(tmp,'w',encoding='utf8') as f:
        json.dump(value,f,indent=2,ensure_ascii=False,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
    with open(tmp,encoding='utf8') as f:assert json.load(f)==value
    os.replace(tmp,dest)


def install():
    global _installed,_install_time
    if _installed:return
    _install_time=datetime.now(timezone.utc).isoformat()
    _installed=True
    sys.addaudithook(_hook)
    atexit.register(flush)


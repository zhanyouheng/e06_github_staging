"""Detached continuation controller: never accesses a test payload."""
from common import *
import subprocess
def main():
    write_text(T/'logs/controller.pid',str(os.getpid())+'\n');js(T/'logs/controller_start.json',{'at':now(),'pid':os.getpid(),'script_sha256':sha(Path(__file__))})
    needed=['gpu_worker_0_finished.json','gpu_worker_1_finished.json','cpu_finished.json'];last=0
    while not all((T/'logs'/f).exists() for f in needed):
        if time.time()-last>60:
            print(now(),'awaiting',','.join(f for f in needed if not(T/'logs'/f).exists()),flush=True);last=time.time()
        time.sleep(30)
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
    phases=[('select_evaluate.py','selection'),('bootstrap.py','bootstrap'),('audit_cpu.py','audit/final')]
    for script,certificate in phases:
        if verified(T/certificate):print('verified phase',script,flush=True);continue
        cmd=[str(W/'envs/seqscreen_train/bin/python'),'-B','-u',str(T/'scripts'/script)];started=now()
        with open(T/'logs'/(script+'.log'),'a') as f:p=subprocess.Popen(cmd,cwd=W,env=env,stdout=f,stderr=subprocess.STDOUT)
        js(T/'logs'/(script+'.process.json'),{'started_at':started,'pid':p.pid,'command':cmd,'script_sha256':sha(T/'scripts'/script)});code=p.wait()
        js(T/'logs'/(script+'.exit.json'),{'started_at':started,'finished_at':now(),'returncode':code})
        if code or not verified(T/certificate):print('FAILED phase',script,code,flush=True);return
    # Final sealing is performed by the active agent after controller exit, so no
    # process can append to an included log after its checksum was recorded.
    js(T/'logs/controller_finished.json',{'at':now(),'pid':os.getpid(),'status':'ready_for_seal'});print('ready for final seal',flush=True)
if __name__=='__main__':main()

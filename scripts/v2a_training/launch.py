from common import *
import subprocess
def main():
    assert verified(T/'preflight') and verified(T/'features')
    gp=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True)
    ids=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
    mapping={int(r.split(',')[0]):r.split(',')[1].strip() for r in ids.splitlines()}
    jobs=[]
    for worker,gpu in enumerate([2,3]):jobs.append((f'gpu_{worker}',W/'envs/seqscreen_esm/bin/python',['--worker',str(worker),'--workers','2'],'train_gpu.py',gpu))
    jobs.append(('cpu',W/'envs/seqscreen_train/bin/python',['--workers','80'],'train_cpu.py',None))
    for name,python,args,script,gpu in jobs:
        p=T/'logs'/f'{name}.process.json'
        if p.exists():
            old=json.loads(p.read_text())
            try:os.kill(old['pid'],0);print('already alive',name,old['pid']);continue
            except ProcessLookupError:pass
        finished=T/'logs'/('cpu_finished.json' if name=='cpu' else f'gpu_worker_{name[-1]}_finished.json')
        if finished.exists() and not json.loads(finished.read_text()).get('failures',[]):
            print('worker completed',name);continue
        if gpu is not None:assert mapping[gpu] not in gp,'requested GPU now occupied; do not displace'
        env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
        if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
        else:env['CUDA_VISIBLE_DEVICES']=''
        cmd=[str(python),'-B','-u',str(T/'scripts'/script),*args];destination(T/'logs'/f'{name}.log')
        with open(T/'logs'/f'{name}.log','a') as f:q=subprocess.Popen(cmd,cwd=W,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        js(p,{'at':now(),'pid':q.pid,'command':cmd,'gpu':gpu,'script_sha256':sha(T/'scripts'/script)});print('launched',name,q.pid,flush=True)
    write_text(T/'STATE.md','''# Turn13 状态\n\n预检已通过，v2a数据已冻结为dev 25,027、702全排除，V1 not_applicable。\n\n训练已启动：logs/*.process.json记录独立会话PID和命令。scripts/launch.py可续跑，但先核对已有PID/GPU；单模型只跳过已验证DONE。GPU2/3使用，GPU0/1避让。CPU80类任务并发（从32增加的操作记录见audit/）。\n\n候选M00–M06；当前正在拟合基础模型，尚未进入val选方案。下一步：确认workers完成→聚合OOF/val→嵌套分量交叉选择→固定阈值/配对bootstrap2000→独立验收→中文报告和封存。禁止读测试载荷；不改变历史文件。\n''')
if __name__=='__main__':main()

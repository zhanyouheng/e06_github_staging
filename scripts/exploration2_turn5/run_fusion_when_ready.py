from access_guard import install
install()
from common import *
import subprocess

def main():
 no_test_guard();txt(E/'logs/fusion.pid',str(os.getpid())+'\n')
 needed=[E/'models/trees/DONE.json',E/'models/neural/DONE.json',E/'references/legacy_neural_DONE.json',E/'references/E04_006_C/train_DONE.json']
 last=-1
 while not all(p.exists() for p in needed):
  minute=int(time.monotonic()//60)
  if minute!=last:log('waiting_registered_model_branches',missing=[str(p.relative_to(E)) for p in needed if not p.exists()]);last=minute
  if (E/'STOP_NEW_EXPERIMENTS').exists():log('fusion_wait_stopped');return
  time.sleep(15)
 for p in needed:assert verified(p),str(p)
 py='/disk1/shengan_group/zyh/envs/seqscreen_train/bin/python'
 for args in [['audit_fusion.py'],['fusion.py','all'],['select_proposal.py']]:
  script=S/args[0]
  if not script.exists():assert args[0]=='audit_fusion.py';continue
  subprocess.run([py,'-B',str(script)]+args[1:],check=True,cwd='/disk1/shengan_group/zyh',env=dict(os.environ))
 log('fusion_and_proposal_finished')
if __name__=='__main__':main()

from common import *
from scipy import sparse
from scipy.special import expit
from sklearn.preprocessing import normalize
import re,subprocess,argparse

def verify_final():
 s=readj(E/'selection/final_selection.json')
 for p,h in s['files'].items():assert sha(E/p)==h,('frozen exploration input changed',p)
 rt=readj(E/'selection/inference_runtime_v2.json');assert sha(S/'inference_v2.py')==rt['inference_v2_sha256'];assert sha(E/'selection/final_selection.json')==rt['unchanged_final_selection_sha256']
 return s
class Predictor:
 def __init__(self,split):
  self.split=split;self.features={};self.scores={};self.annotations=None;self.raw=None
  if split=='val':self.ids=dev()['val_ids']
  else:
   a=np.load(BASE/'features/alignment.npz',allow_pickle=False);self.indices=np.flatnonzero(a['split']==2);self.ids=a['protein_id'][self.indices]
 def annotation_rows(self):
  if self.annotations is None:
   ids=set(self.ids);ann={}
   for f in ['positive_features_2019.tsv','background_features_2019.tsv']:
    for r in rows(ROOT/f):
     if r['ac'] in ids:ann[r['ac']]={k:r[k] for k in ['ac','go','keywords','function_cc','organism','len','seq_sha256']}
   assert set(ann)==ids;self.annotations=ann
  return self.annotations
 def raw_embedding(self):
  if self.raw is None:
   out=np.empty((len(self.ids),1280),np.float64);ix={str(a):i for i,a in enumerate(self.ids)};seen=set()
   for r in readj(OUT/'embeddings/manifest.json')['shards']:
    p=OUT/'embeddings'/r['file'];assert sha(p)==r['sha256']
    with np.load(p,allow_pickle=False) as z:
     for j,a in enumerate(z['protein_id']):
      if a in ix:out[ix[a]]=z['embedding'][j];seen.add(a)
   assert seen==set(self.ids);self.raw=out
  return self.raw
 def get_feature(self,name):
  if name in self.features:return self.features[name]
  if self.split=='val':out=feature(name,'val')
  else:
   fd=readj(E/f'features/{name}/definition.json');kind=fd['type']
   if kind=='baseline':
    if name=='B':out=np.load(BASE/'features/B.npy',allow_pickle=False,mmap_mode='r')[self.indices]
    else:out=sparse.load_npz(BASE/f'features/{name}.npz')[self.indices]
   elif kind=='annotation':
    ann=self.annotation_rows();vi={t:i for i,t in enumerate(fd['vocabulary'])};rr=[];cc=[]
    for i,a in enumerate(self.ids):
     tok=set(re.findall(r'GO:\d{7}',ann[a]['go']))|{'KW:'+v.strip() for v in ann[a]['keywords'].split(';') if v.strip()};cols=sorted(vi[t] for t in tok if t in vi);rr.extend([i]*len(cols));cc.extend(cols)
    out=normalize(sparse.csr_matrix((np.ones(len(rr)),(rr,cc)),shape=(len(self.ids),len(vi))),norm='l2')
   elif kind=='text':
    with open(E/f'features/{name}/transform.pkl','rb') as f:transform=pickle.load(f)
    ann=self.annotation_rows();out=transform.transform([ann[a][fd['field']] for a in self.ids]).tocsr()
   elif kind=='concat':out=sparse.hstack([sparse.csr_matrix(self.get_feature(n)) for n in fd['blocks']],format='csr')/np.sqrt(len(fd['blocks']))
   elif kind=='nystroem':
    with open(E/f'features/{name}/transform.pkl','rb') as f:transform=pickle.load(f)
    out=transform.transform(self.get_feature(fd['source']))
   elif kind=='embedding_variant':
    raw=self.raw_embedding()
    if fd['variant']=='raw_row_l2':out=normalize(raw,norm='l2')
    else:
     with np.load(BASE/'features/B_preprocessing.npz',allow_pickle=False) as z:out=(raw-z['mean'])/z['std']/np.sqrt(1280)
   elif kind=='embedding_length':
    ann=self.annotation_rows();length=np.log1p([int(ann[a]['len']) for a in self.ids]);column=0.25*(length-fd['mean'])/fd['std'];out=np.column_stack([self.get_feature(fd['base']),column])
   else:raise ValueError(kind)
  assert np.isfinite(out.data if sparse.issparse(out) else out).all();self.features[name]=out;return out
 def predict(self,name):
  if name in self.scores:return self.scores[name]
  summary=readj(E/f'candidates/{name}/summary.json');cfg=summary['config'];algorithm=cfg['algorithm'];start=time.monotonic()
  if algorithm=='ensemble':
   if summary.get('fixed_seed_mean',False):out=np.mean(np.stack([self.predict(n) for n in cfg['members']],axis=0),axis=0)
   else:out=sum(w*self.predict(n) for n,w in zip(cfg['members'],cfg['weights']))
  elif algorithm=='class_selector':out=np.column_stack([self.predict(n)[:,j] for j,n in enumerate(cfg['members_by_class'])])
  elif algorithm=='mlp':
   x=self.get_feature(cfg['feature']);x=np.asarray(x.toarray() if sparse.issparse(x) else x,dtype=np.float32)
   ip=E/f'inference_v2/{self.split}/inputs/{cfg["feature"]}.npy'
   if not ip.exists():npy(ip,x)
   else:assert np.array_equal(np.load(ip,mmap_mode='r',allow_pickle=False),x)
   op=E/f'inference_v2/{self.split}/{name}.npy';env=dict(os.environ);env['CUDA_VISIBLE_DEVICES']='0,1,2,3'
   subprocess.run(['/disk1/shengan_group/zyh/envs/seqscreen_esm/bin/python','-B',str(S/'infer_gpu.py'),'--name',name,'--input',str(ip),'--output',str(op)],check=True,env=env,cwd='/disk1/shengan_group/zyh')
   out=np.load(op,allow_pickle=False)
  elif algorithm in ['logistic','group_logistic','extra_trees']:
   x=self.get_feature(cfg['feature']);out=np.empty((len(self.ids),32),np.float64)
   if algorithm=='extra_trees':x=x.astype(np.float32)
   for j in range(32):
    p=E/f'experiments/{name}/class_{j:02d}'
    if algorithm=='extra_trees':
     with open(p.with_suffix('.pkl'),'rb') as f:model=pickle.load(f)
     out[:,j]=model['constant'] if isinstance(model,dict) else model.predict_proba(x)[:,list(model.classes_).index(1)]
    else:
     with np.load(p.with_suffix('.npz'),allow_pickle=False) as z:out[:,j]=float(z['constant']) if float(z['constant'])>=0 else expit(x@z['coef']+float(z['intercept']))
  else:raise ValueError(algorithm)
  assert out.shape==(len(self.ids),32) and np.isfinite(out).all() and ((out>=0)&(out<=1)).all();self.scores[name]=out
  log('inference_node',split=self.split,name=name,seconds=time.monotonic()-start);return out

def main():
 p=argparse.ArgumentParser();p.add_argument('split',choices=['val','test']);args=p.parse_args();selection=verify_final()
 if args.split=='test':assert (E/'revision/revision_evaluation_event.json').exists(),'revision event must be registered before test inference'
 out=E/f'inference_v2/{args.split}/final_predictions.npz';ct={'selection_sha256':sha(E/'selection/final_selection.json'),'split':args.split,'inference_code_sha256':sha(S/'inference_v2.py'),'infer_gpu_code_sha256':sha(S/'infer_gpu.py')}
 if verified(out.with_suffix('.DONE.json'),ct):log('final_prediction_resume_verified',split=args.split);return
 pred=Predictor(args.split);score=pred.predict(selection['chosen_candidate']);thresholds=np.array(selection['thresholds']);classes=np.array(selection['class_order'])
 npz(out,protein_id=pred.ids,class_order=classes,scores=score,predictions=(score>=thresholds).astype(np.uint8),thresholds=thresholds);files=[out]
 if args.split=='val':
  with np.load(E/f"candidates/{selection['chosen_candidate']}/validation.npz",allow_pickle=False) as z:
   assert np.array_equal(pred.ids,z['protein_id']);error=float(np.max(np.abs(score-z['scores'])));assert np.array_equal(score,z['scores']);assert np.array_equal(score>=thresholds,z['scores']>=z['thresholds'])
  jout(E/'validation/frozen_model_recompute_v2.json',{'passed':True,'split':'val','max_abs_diff':error,'protein_count':len(pred.ids),'classes':32,'model_sha256':sha(E/'selection/final_selection.json'),'at':now(),'test_used':False})
  files.append(E/'validation/frozen_model_recompute_v2.json')
 certificate(out.with_suffix('.DONE.json'),files,ct)
 log('final_prediction_complete',split=args.split)
if __name__=='__main__':main()

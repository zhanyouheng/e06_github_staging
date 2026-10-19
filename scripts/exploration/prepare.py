from common import *
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from collections import Counter
import re,sys,importlib

def main():
 no_test_guard();start=time.monotonic();context={'PLAN_sha256':sha(E/'PLAN.md'),'code':{n:sha(S/n) for n in ['prepare.py','common.py']}}
 if verified(E/'data/DONE.json',context):log('dev_resume_verified');return
 # Byte-level verification only: no parsing of old test predictions or metrics.
 original=readj(OUT/'qc/input_checksums_before.json');old={str(ROOT/k):v['sha256'] for k,v in original.items()}
 for line in (BASE/'SHA256SUMS').read_text().splitlines():h,p=line.split('  ',1);old[str(OUT/p)]=h
 for p in [BASE/'manifest.json',BASE/'DONE.json',BASE/'SHA256SUMS',OUT/'freeze/manifest.json',OUT/'freeze/SHA256SUMS']:
  old[str(p)]=sha(p)
 for r in readj(OUT/'freeze/manifest.json')['frozen_files']:old[str(OUT/r['path'])]=r['sha256']
 em=readj(OUT/'embeddings/manifest.json')
 for r in em['shards']:
  old[str(OUT/'embeddings'/r['file'])]=r['sha256'];old[str(OUT/'embeddings'/r['done_file'])]=r['done_sha256']
 for p in (OUT/'embeddings').glob('*'):
  if p.is_file():old[str(p)]=sha(p)
 for p,h in old.items():assert sha(p)==h,p
 jout(E/'integrity/old_artifacts_before.json',old)
 # Enforce read-only mode, without modifying a single content byte.
 for p in BASE.rglob('*'):
  if p.is_file():p.chmod(0o444)
 for p in (OUT/'embeddings').rglob('*'):
  if p.is_file():p.chmod(0o444)
 a=np.load(BASE/'features/alignment.npz',allow_pickle=False);lab=np.load(BASE/'features/development_labels.npz',allow_pickle=False)
 tr=a['split']==0;va=a['split']==1;assert np.array_equal(lab['train_ids'],a['protein_id'][tr]) and np.array_equal(lab['val_ids'],a['protein_id'][va])
 assert len(set(lab['train_ids'])&set(lab['val_ids']))==0
 npz(E/'data/dev.npz',train_ids=lab['train_ids'],val_ids=lab['val_ids'],train_y=lab['train_y'],val_y=lab['val_y'],class_order=a['class_order'],
     train_groups=a['component_id'][tr],val_groups=a['component_id'][va],val_annotation_usable=a['annotation_usable'][va])
 ids=lab['train_ids'].tolist()+lab['val_ids'].tolist();keep=set(ids);source={}
 for f in ['positive_features_2019.tsv','background_features_2019.tsv']:
  for r in rows(ROOT/f):
   if r['ac'] in keep:source[r['ac']]={k:r[k] for k in ['ac','go','keywords','function_cc','organism','len','seq_sha256']}
 assert set(source)==keep and len(source)==25027
 tsv(E/'data/annotations_dev.tsv',['ac','go','keywords','function_cc','organism','len','seq_sha256'],[source[i] for i in ids])
 defs={};stats=[]
 for name in ['A','B','C']:
  x=np.load(BASE/'features/B.npy',mmap_mode='r',allow_pickle=False) if name=='B' else sparse.load_npz(BASE/f'features/{name}.npz')
  store_feature(name,x[tr],x[va],{'type':'baseline','name':name,'source':str(BASE/f'features/{name}'+('.npy' if name=='B' else '.npz')) if False else str(BASE/'features'),
     'train_fitted_only':True,'baseline_features_DONE_sha256':sha(BASE/'features/DONE.json')})
  if name in ['B','C']:
   for s,mask in [('train',tr),('val',va)]:npy(E/f'features/{name}/{s}_gpu.npy',np.asarray(x[mask].toarray() if sparse.issparse(x) else x[mask],dtype=np.float32))
  del x
 # Raw historical annotations; namespaces are unchanged from the baseline.
 tokens=[set(re.findall(r'GO:\d{7}',source[i]['go']))|{'KW:'+k.strip() for k in source[i]['keywords'].split(';') if k.strip()} for i in ids]
 nt=len(lab['train_ids']);df=Counter(t for terms in tokens[:nt] for t in terms)
 for cutoff in [2,5]:
  name=f'A{cutoff}';vocab=sorted(t for t,n in df.items() if n>=cutoff);vi={t:j for j,t in enumerate(vocab)};rr=[];cc=[]
  for i,terms in enumerate(tokens):
   cols=sorted(vi[t] for t in terms if t in vi);rr.extend([i]*len(cols));cc.extend(cols)
  x=normalize(sparse.csr_matrix((np.ones(len(rr)),(rr,cc)),shape=(len(ids),len(vocab))),norm='l2')
  definition={'type':'annotation','min_df':cutoff,'vocabulary':vocab,'binary':True,'row_l2':True,'GO':'direct IDs only','namespace':['GO:','KW:']}
  store_feature(name,x[:nt],x[nt:],definition)
  for s,xx in [('train',x[:nt]),('val',x[nt:])]:
   if s=='train':xxtr=sparse.hstack([xx,sparse.csr_matrix(feature('B',s))],format='csr')/np.sqrt(2)
   else:xxva=sparse.hstack([xx,sparse.csr_matrix(feature('B',s))],format='csr')/np.sqrt(2)
  store_feature(name+'B',xxtr,xxva,{'type':'concat','blocks':[name,'B'],'divide_sqrt_blocks':True})
  stats.append({'feature':name,'dimensions':len(vocab),'raw_train_tokens':len(df)})
 for name,field,maximum in [('F','function_cc',30000),('O','organism',5000)]:
  texts=[source[i][field] for i in ids];vector=TfidfVectorizer(ngram_range=(1,2),min_df=3,max_features=maximum,sublinear_tf=True,norm='l2',lowercase=True,token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z0-9_-]+\b',dtype=np.float64)
  xt=vector.fit_transform(texts[:nt]).tocsr();xv=vector.transform(texts[nt:]).tocsr();store_feature(name,xt,xv,{'type':'text','field':field,'ngram':[1,2],'min_df':3,'max_features':maximum,'sublinear_tf':True,'train_fitted_only':True},vector)
  stats.append({'feature':name,'dimensions':xt.shape[1],'raw_empty_train':sum(not t.strip() for t in texts[:nt]),'raw_empty_val':sum(not t.strip() for t in texts[nt:])})
 for name,blocks in [('FA',['F','A']),('FAB',['F','A','B']),('FO',['F','O']),('FABO',['F','A','B','O'])]:
  xx={s:sparse.hstack([sparse.csr_matrix(feature(b,s)) for b in blocks],format='csr')/np.sqrt(len(blocks)) for s in ['train','val']}
  store_feature(name,xx['train'],xx['val'],{'type':'concat','blocks':blocks,'divide_sqrt_blocks':True})
 jout(E/'data/feature_statistics.json',{'features':stats,'train_proteins':nt,'val_proteins':len(lab['val_ids']),'test_proteins_materialized':0,'excluded':0,'PNU':{s:{'P':int((lab[s+'_y']==1).sum()),'N':int((lab[s+'_y']==0).sum()),'U':int((lab[s+'_y']==-1).sum())} for s in ['train','val']},'seconds':time.monotonic()-start})
 env={'python':sys.version,'environment':sys.prefix,'packages':{n:importlib.import_module(n).__version__ for n in ['numpy','scipy','sklearn','numba','joblib','threadpoolctl']}}
 jout(E/'env/cpu_versions.json',env)
 certificate(E/'data/DONE.json',[E/'data/dev.npz',E/'data/annotations_dev.tsv',E/'data/feature_statistics.json',E/'env/cpu_versions.json',E/'integrity/old_artifacts_before.json'],context)
 log('development_features_ready',seconds=time.monotonic()-start)
if __name__=='__main__':main()

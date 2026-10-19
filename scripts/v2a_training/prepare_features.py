from common import *
import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize
from collections import Counter

def main():
    assert verified(T/'preflight'),'preflight gate'
    d=np.load(T/'data/development.npz',allow_pickle=False);nt=int((d['scope']=='train').sum());raw=np.load(T/'data/embedding_raw.npy',mmap_mode='r');tokens=json.loads((T/'data/tokens.json').read_text());ids=d['protein_id']
    f=d['train_fold'];inner=d['inner_holdout'];sets={'full':np.arange(nt)}
    for k in range(5):sets['outer'+str(k)]=np.flatnonzero(f!=k);sets['inner'+str(k)]=np.flatnonzero((f!=k)&~inner)
    meta=read_tsv(T/'data/metadata.tsv');score=np.zeros((len(ids),6),np.float32)
    for i,r in enumerate(meta):
        assert r['ac']==ids[i]
        v=float(r['annotation_score']) if r['annotation_score'] not in ('','NA','nan') else np.nan
        score[i,int(v)-1 if np.isfinite(v) and 1<=v<=5 else 5]=1
    npy(T/'features/annotation_score.npy',score)
    for name,fit in sets.items():
        dest=T/'features'/name
        if verified(dest):print('verified',name,flush=True);continue
        start=time.time();df=Counter(t for i in fit for t in tokens[i]);mean=raw[fit].astype(np.float64).mean(0);std=raw[fit].astype(np.float64).std(0);zero=std==0;std[zero]=1
        b=normalize(((raw.astype(np.float64)-mean)/std).astype(np.float32));npz(dest/'B_parameters.npz',mean=mean,std=std,zero_variance=zero,fit_ids=ids[fit]);files=[dest/'B_parameters.npz']
        for route,mindf in [('C',10),('A2B',2)]:
            vocab=sorted(t for t,n in df.items() if n>=mindf);vi={t:j for j,t in enumerate(vocab)};rr=[];cc=[]
            for i,row in enumerate(tokens):
                cols=sorted(vi[t] for t in row if t in vi);rr.extend([i]*len(cols));cc.extend(cols)
            a=normalize(sparse.csr_matrix((np.ones(len(rr),np.float32),(rr,cc)),shape=(len(ids),len(vocab)))).toarray()
            x=np.concatenate([a,b],axis=1)/np.float32(np.sqrt(2));assert np.isfinite(x).all();npy(dest/(route+'.npy'),x);js(dest/(route+'_vocab.json'),{'min_df':mindf,'vocabulary':vocab,'fit_ids_sha256':hashlib.sha256('\n'.join(ids[fit]).encode()).hexdigest(),'vocabulary_fit_rows':len(fit),'normalization':'binary row L2; concatenated with train standardized row L2 B /sqrt(2)'})
            files.extend([dest/(route+'.npy'),dest/(route+'_vocab.json')])
        npz(dest/'fit_rows.npz',row_index=fit,protein_id=ids[fit]);files.append(dest/'fit_rows.npz');done(dest,files,seconds=time.time()-start);print('prepared',name,'fit rows',len(fit),round(time.time()-start,2),flush=True)
    done(T/'features',[T/'features/annotation_score.npy']+[T/'features'/s/'DONE.json' for s in sets],preprocessing='fit on each actual training side only')
if __name__=='__main__':main()

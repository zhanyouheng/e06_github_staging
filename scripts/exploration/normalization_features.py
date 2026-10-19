from common import *
from sklearn.preprocessing import normalize
from scipy import sparse

def main():
 no_test_guard();d=dev();ids=d['train_ids'].tolist()+d['val_ids'].tolist();ix={a:i for i,a in enumerate(ids)};nt=len(d['train_ids']);raw=np.empty((len(ids),1280),np.float64);seen=set()
 for r in readj(OUT/'embeddings/manifest.json')['shards']:
  p=OUT/'embeddings'/r['file'];assert sha(p)==r['sha256']
  with np.load(p,allow_pickle=False) as z:
   for j,a in enumerate(z['protein_id']):
    if a in ix:raw[ix[a]]=z['embedding'][j];seen.add(a)
 assert seen==set(ids)
 with np.load(BASE/'features/B_preprocessing.npz',allow_pickle=False) as b:mean=b['mean'];std=b['std'];assert np.array_equal(d['train_ids'],b['train_ids'])
 bstd=(raw-mean)/std/np.sqrt(1280);bcos=normalize(raw,norm='l2')
 store_feature('Bstd',bstd[:nt],bstd[nt:],{'type':'embedding_variant','variant':'standardized_no_row_norm','divide_sqrt_dimensions':True,'baseline_preprocessor_sha256':sha(BASE/'features/B_preprocessing.npz')})
 store_feature('Bcos',bcos[:nt],bcos[nt:],{'type':'embedding_variant','variant':'raw_row_l2','source':'frozen float32 embeddings'})
 ann={r['ac']:r for r in rows(E/'data/annotations_dev.tsv')};length=np.log1p([int(ann[a]['len']) for a in ids]);mu=float(length[:nt].mean());sigma=float(length[:nt].std());column=0.25*(length-mu)/(sigma or 1.)
 x=np.column_stack([bstd,column]);store_feature('BstdL',x[:nt],x[nt:],{'type':'embedding_length','base':'Bstd','length_transform':'0.25*(log1p(length)-train_mean)/train_std','mean':mu,'std':sigma or 1.})
 for name,b in [('Cstd','Bstd'),('Ccos','Bcos')]:
  x={s:sparse.hstack([feature('A',s),sparse.csr_matrix(feature(b,s))],format='csr')/np.sqrt(2) for s in ['train','val']};store_feature(name,x['train'],x['val'],{'type':'concat','blocks':['A',b],'divide_sqrt_blocks':True})
 log('normalization_features_ready')
if __name__=='__main__':main()

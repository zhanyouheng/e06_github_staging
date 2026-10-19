from common import *
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
X=None;V=None
def fit_free(j):
 p=E/f'references/E04_006_C/class_{j:02d}';src=OLD/f'experiments/E04_006_C/class_{j:02d}.pkl';ctx={'source_sha256':sha(src),'dev_sha256':sha(E/'data/dev.npz')}
 if verified(p.with_suffix('.DONE.json'),ctx):return
 with open(src,'rb') as z:m=pickle.load(z)
 tr=np.full(X.shape[0],m['constant']) if isinstance(m,dict) else m.predict_proba(X)[:,list(m.classes_).index(1)]
 va=np.full(V.shape[0],m['constant']) if isinstance(m,dict) else m.predict_proba(V)[:,list(m.classes_).index(1)]
 npz(p.with_suffix('.npz'),train_scores=tr,val_scores=va)
 certificate(p.with_suffix('.DONE.json'),[p.with_suffix('.npz')],ctx)
def main():
 global X,V
 no_test_guard();d=dev();X=feature('C','train').astype(np.float32);V=feature('C','val').astype(np.float32)
 with ProcessPoolExecutor(max_workers=16,mp_context=mp.get_context('fork')) as pool:list(pool.map(fit_free,range(32)))
 arr=[np.load(E/f'references/E04_006_C/class_{j:02d}.npz',allow_pickle=False) for j in range(32)]
 tr=np.column_stack([z['train_scores'] for z in arr]);va=np.column_stack([z['val_scores'] for z in arr])
 old=np.load(E/'references/E04_006_C/validation.npz',allow_pickle=False)
 assert np.array_equal(old['protein_id'],d['val_ids']);assert np.array_equal(old['scores'],va)
 p=E/'references/E04_006_C/train_scores.npz';npz(p,protein_id=d['train_ids'],class_order=d['class_order'],scores=tr)
 jout(E/'references/E04_006_C/train_inference.json',{'at':now(),'training':False,'splits':['train','val'],'validation_max_abs_diff':0.,'model_sha256':{str(OLD/f'experiments/E04_006_C/class_{j:02d}.pkl'):sha(OLD/f'experiments/E04_006_C/class_{j:02d}.pkl') for j in range(32)}})
 certificate(E/'references/E04_006_C/train_DONE.json',[p,E/'references/E04_006_C/train_inference.json'],{'script_sha256':sha(S/'legacy_trees.py')})
 log('legacy_tree_train_scores_complete',validation_exact=True)
if __name__=='__main__':main()

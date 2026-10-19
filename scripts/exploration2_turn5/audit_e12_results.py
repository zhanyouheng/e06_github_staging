"""Independent completed E12 certificate, support, ID and metric audit."""
from access_guard import install
install()
from common import *
from sklearn.metrics import average_precision_score
from scipy import sparse

def main():
    d=dev();results=[];y=d['train_y'];vy=d['val_y']
    for cert in sorted((E/'models/neural').glob('E12_*_s*/DONE.json')):
        assert verified(cert)
        p=cert.parent;meta=readj(p/'history.json');ctx=readj(cert)['context']
        cfg=meta['config'];assert ctx['config']==cfg
        assert ctx['code_sha256']==sha(S/'neural.py')
        assert ctx['dev_sha256']==sha(E/'data/dev.npz')
        support=meta['training_support']
        for state,num in [('P',1),('N',0),('U',-1)]:assert support[state]==(y==num).sum(0).tolist()
        assert meta['test_used'] is False and meta['independent_val_bitwise_equal'] is True
        assert meta['deterministic_algorithms'] is True and meta['tf32'] is False
        for split,ids in [('train',d['train_ids']),('validation',d['val_ids'])]:
            f=p/('train_scores.npz' if split=='train' else 'validation.npz')
            with np.load(f,allow_pickle=False) as z:
                assert np.array_equal(z['protein_id'],ids) and np.array_equal(z['class_order'],d['class_order'])
                s=z['scores'];assert s.shape==(len(ids),32) and np.isfinite(s).all()
                assert np.all((s>=0)&(s<=1))
                if split=='validation':vs=s
        aps=[]
        for j in range(32):
            k=vy[:,j]!=-1;aps.append(average_precision_score(vy[k,j],vs[k,j]))
        score=float(np.mean(aps));assert abs(score-meta['validation_macro_AP'])<1e-12
        hist=[r for r in meta['history'] if 'validation_macro_AP' in r]
        chosen=[r for r in hist if r['epoch']==meta['best_epoch']]
        assert len(chosen)==1 and abs(chosen[0]['validation_macro_AP']-score)<1e-12
        assert max(r['validation_macro_AP'] for r in hist)-score<=1e-6
        results.append({'name':cfg['name'],'seed':cfg['seed'],'validation_macro_AP':score,'best_epoch':meta['best_epoch'],'support_verified':True,'DONE_verified':True})
    features=[]
    for name in ['C','A2B']:
        for split in ['train','val']:
            ref=feature(name,split)
            ref=np.asarray(ref.toarray() if sparse.issparse(ref) else ref,dtype=np.float32)
            saved=np.load(OLD/f'features/{name}/{split}_gpu.npy',allow_pickle=False,mmap_mode='r')
            assert np.array_equal(ref,saved),(name,split)
            features.append({'feature':name,'split':split,'shape':list(ref.shape),'canonical_float32_equal':True,'gpu_array_sha256':sha(OLD/f'features/{name}/{split}_gpu.npy')})
    planned=readj(E/'configurations/E12_initial.json')
    extra=E/'configurations/E12_extra_seeds.json'
    if extra.exists():planned+=readj(extra)
    completed={r['name'] for r in results};missing=[c['name'] for c in planned if c['name'] not in completed]
    out={'at':now(),'passed_for_completed_models':True,'completed_model_count':len(results),'planned_model_count':len(planned),'missing_configurations':missing,'models':results,'feature_checks':features,'test_set_read':False,'neural_sha256':sha(S/'neural.py')}
    jout(E/'audit/E12_result_audit.json',out)
    print(json.dumps({'completed':len(results),'planned':len(planned),'missing':missing,'passed_for_completed':True}))

if __name__=='__main__':main()

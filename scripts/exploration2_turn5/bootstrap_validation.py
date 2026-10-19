"""Paired component bootstrap of frozen development predictions only.

Register the fixed protocol with --register. Run --candidate NAME only after the
root has selected an eligible development scheme. This module has no test route.
"""
from access_guard import install
install()
from common import *
from bootstrap_core import pack,run_block
from metrics import evaluate,class_sets
from sklearn.metrics import average_precision_score,f1_score
import argparse

BOOTSTRAP_SEED=20261010
BASELINE='E06_mix_098'
B=E/'bootstrap_validation'
CONFIG={
    'stage':'E15','scope':'train/validation development only; no test evaluation',
    'seed':BOOTSTRAP_SEED,'replicates':2000,'block_size':100,
    'sampling_unit':'frozen validation component','component_order':'lexically sorted np.unique(val_groups)',
    'sampling':'2000 independent draws of 1398 component indices with replacement; count multiplicities weight all member proteins',
    'shared_draws':['full_val','usable_annotations'],'metrics':['macro_AP','macro_F1'],
    'comparison':'selected_candidate_minus_E06_mix_098','fixed_K':'each subset original development labels; never shrink K per draw',
    'AP_invalid':'any original K_AP class lacks weighted positive or negative support',
    'F1_invalid':'any original K_F1 class lacks weighted positive support',
    'CI':'95% percentile; ordinary CI NA if any of 2000 rounds invalid; valid-only conditional interval separately labeled',
    'thresholds':'existing frozen candidate and baseline validation-selected class vectors; no threshold refit in bootstrap',
    'limits':'Both model selection and threshold fitting reused validation data. These intervals describe resampling variability of fixed selected development predictions and do not constitute independent generalization or selection-adjusted uncertainty.',
    'source_labels':'exploration2/data/dev.npz: val_y only','test_access':False,
}

def register():
    p=E/'configurations/bootstrap_validation.json'
    if p.exists():assert readj(p)==CONFIG
    else:jout(p,CONFIG)
    return p

def weighted_independent(y,scores,thresholds,protein_weights,kap,kfp):
    result=np.full((2,2),np.nan)
    for m in range(2):
        aps=[];f1s=[];apvalid=True;f1valid=True
        for j in range(32):
            k=y[:,j]!=-1;yy=y[k,j];w=protein_weights[k]
            P=float(w[yy==1].sum());N=float(w[yy==0].sum())
            if j in kap:
                if P==0 or N==0:apvalid=False
                else:aps.append(float(average_precision_score(yy,scores[m,k,j],sample_weight=w)))
            if j in kfp:
                if P==0:f1valid=False
                else:f1s.append(float(f1_score(yy,scores[m,k,j]>=thresholds[m,j],sample_weight=w,zero_division=0)))
        if apvalid and len(kap):result[m,0]=float(np.mean(aps))
        if f1valid and len(kfp):result[m,1]=float(np.mean(f1s))
    return result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--register',action='store_true')
    parser.add_argument('--candidate')
    args=parser.parse_args()
    cp=register()
    if args.register:
        assert args.candidate is None
        log('validation_bootstrap_protocol_registered',seed=BOOTSTRAP_SEED);return
    assert args.candidate,'An explicitly selected eligible --candidate is required'
    candidate=args.candidate
    assert '/' not in candidate and candidate!='..'
    cand=E/f'candidates/{candidate}';assert verified(cand/'DONE.json')
    sm=readj(cand/'summary.json')
    assert sm['selection_eligible'] is True and sm['validation_macro_AP']>.902357
    proposed=readj(E/'selection/proposed_selection.json')
    assert proposed['chosen_candidate']==candidate,'bootstrap is restricted to the selected proposal'
    if (E/'selection/final_selection.json').exists():
        final=readj(E/'selection/final_selection.json')
        assert final['chosen_candidate']==candidate
    paths=[cand/'validation.npz',E/f'references/{BASELINE}/validation.npz']
    ctx={'candidate':candidate,'candidate_certificate_sha256':sha(cand/'DONE.json'),
         'validation_files_sha256':[sha(p) for p in paths],
         'dev_sha256':sha(E/'data/dev.npz'),'protocol_sha256':sha(cp),
         'script_sha256':sha(Path(__file__)),'core_sha256':sha(S/'bootstrap_core.py'),
         'metrics_sha256':sha(S/'metrics.py')}
    if verified(B/'DONE.json',ctx):
        log('validation_bootstrap_resume_verified',candidate=candidate);return
    start=time.monotonic();d=dev();y=d['val_y'];groups=d['val_groups'];ids=d['val_ids'];classes=d['class_order'].tolist()
    assert y.shape==(6257,32) and not any('test' in k.lower() for k in d)
    models=[candidate,BASELINE];scores=[];thresholds=[]
    for p in paths:
        with np.load(p,allow_pickle=False) as z:
            assert np.array_equal(z['protein_id'],ids) and np.array_equal(z['class_order'],d['class_order'])
            assert z['scores'].shape==y.shape and z['thresholds'].shape==(32,)
            scores.append(z['scores'].astype(np.float64));thresholds.append(z['thresholds'].astype(np.float64))
    scores=np.stack(scores);thresholds=np.stack(thresholds)
    order,inverse=np.unique(groups,return_inverse=True);inverse=inverse.astype(np.int32)
    assert len(order)==1398
    sample_path=B/'draws.npz';sample_cert=B/'draws.DONE.json'
    draw_ctx={'seed':BOOTSTRAP_SEED,'protocol_sha256':sha(cp),'component_order':order.tolist(),'replicates':2000}
    rng=np.random.default_rng(BOOTSTRAP_SEED)
    sampled=rng.integers(0,len(order),size=(2000,len(order)),dtype=np.int32)
    counts=np.stack([np.bincount(r,minlength=len(order)) for r in sampled]).astype(np.uint16)
    assert np.all(counts.sum(1)==len(order))
    if not verified(sample_cert,draw_ctx):
        npz(sample_path,seed=np.array(BOOTSTRAP_SEED),component_order=order,sampled_component_indices=sampled,multiplicities=counts)
        certificate(sample_cert,[sample_path],draw_ctx)
    with np.load(sample_path,allow_pickle=False) as z:
        assert np.array_equal(z['component_order'],order)
        assert np.array_equal(z['sampled_component_indices'],sampled) and np.array_equal(z['multiplicities'],counts)
    intervals=[];supports={};audit=[];points={};point_rows=[];per_class_rows=[];files=[cp,sample_path,sample_cert]
    for subset,mask in [('full_val',np.ones(len(y),bool)),('usable_annotations',d['val_annotation_usable'])]:
        yy=y[mask];ss=scores[:,mask];gg=groups[mask];ii=inverse[mask]
        packed=pack(yy,ss,thresholds,ii);kap,kfp=class_sets(yy)
        point=[]
        for m in range(2):
            per,summary=evaluate(yy,ss[m],thresholds[m],classes,gg)
            point.append(summary)
            point_rows.append({'subset':subset,'model':models[m],**{k:json.dumps(v) if isinstance(v,list) else v for k,v in summary.items()}})
            per_class_rows.extend({'subset':subset,'model':models[m],**r} for r in per)
        points[subset]={models[m]:point[m] for m in range(2)}
        values=[];apbad=[];f1bad=[]
        for begin in range(0,2000,100):
            end=begin+100;p=B/f'{subset}/replicates_{begin:04d}_{end-1:04d}.npz';done=p.with_suffix('.DONE.json')
            context={**ctx,'subset':subset,'begin':begin,'end_exclusive':end,'draws_sha256':sha(sample_path)}
            if not verified(done,context):
                v,ba,bf=run_block(counts[begin:end],*packed)
                assert np.array_equal(np.isnan(v[:,0,:]),np.isnan(v[:,1,:]))
                npz(p,replicate=np.arange(begin,end),models=np.array(models),metrics=np.array(['macro_AP','macro_F1']),values=v,selected_minus_baseline=v[:,0,:]-v[:,1,:],invalid_AP_classes=ba,invalid_F1_classes=bf)
                certificate(done,[p],context)
                log('validation_bootstrap_block',subset=subset,completed=end,candidate=candidate)
            with np.load(p,allow_pickle=False) as z:
                values.append(z['values']);apbad.append(z['invalid_AP_classes']);f1bad.append(z['invalid_F1_classes'])
            files.extend([p,done])
        values=np.concatenate(values);ba=np.concatenate(apbad);bf=np.concatenate(f1bad)
        assert values.shape==(2000,2,2)
        supports[subset]={'proteins':int(mask.sum()),'components':len(set(gg)),
            'sampling_component_universe':1398,'P':int((yy==1).sum()),'N':int((yy==0).sum()),'U':int((yy==-1).sum()),
            'K_AP':len(kap),'K_F1':len(kfp),'K_AP_classes':[classes[j] for j in kap],'K_F1_classes':[classes[j] for j in kfp],
            'invalid_AP_by_class':{classes[j]:int(ba[:,j].sum()) for j in range(32) if ba[:,j].any()},
            'invalid_F1_by_class':{classes[j]:int(bf[:,j].sum()) for j in range(32) if bf[:,j].any()}}
        for rep in range(8):
            expected=weighted_independent(yy,ss,thresholds,counts[rep,ii].astype(np.float64),kap,kfp)
            assert np.allclose(expected,values[rep],rtol=0,atol=1e-12,equal_nan=True),(subset,rep,expected,values[rep])
            weighted_P=((yy==1)*counts[rep,ii,None]).sum(0)
            weighted_N=((yy==0)*counts[rep,ii,None]).sum(0)
            expected_ba=np.array([j in kap and (weighted_P[j]==0 or weighted_N[j]==0) for j in range(32)])
            expected_bf=np.array([j in kfp and weighted_P[j]==0 for j in range(32)])
            assert np.array_equal(expected_ba,ba[rep]) and np.array_equal(expected_bf,bf[rep])
            finite=np.isfinite(expected)
            audit.append({'subset':subset,'replicate':rep,'models':2,'metrics':2,'max_abs_diff':float(np.max(np.abs(expected[finite]-values[rep][finite]))) if finite.any() else None,'invalid_support_matches':True})
        for j,metric in enumerate(['macro_AP','macro_F1']):
            delta=values[:,0,j]-values[:,1,j];valid=np.isfinite(delta);n=int(valid.sum())
            conditional=np.percentile(delta[valid],[2.5,97.5]).tolist() if n else None
            intervals.append({'subset':subset,'candidate':candidate,'comparator':BASELINE,'metric':metric,
                'observed_delta':float(point[0][metric]-point[1][metric]),'candidate_point':point[0][metric],'baseline_point':point[1][metric],
                'attempts':2000,'valid_replicates':n,'invalid_replicates':2000-n,'valid_fraction':n/2000,
                'CI_status':'complete_ordinary_percentile' if n==2000 else 'NA_invalid_replicates',
                'CI_low':conditional[0] if n==2000 else None,'CI_high':conditional[1] if n==2000 else None,
                'conditional_valid_only_low':conditional[0] if conditional else None,'conditional_valid_only_high':conditional[1] if conditional else None,
                'involved_classes':sorted(supports[subset]['invalid_AP_by_class' if j==0 else 'invalid_F1_by_class']),
                'selection_adjusted':False})
        target=B/f'{subset}/paired_deltas.npz'
        npz(target,replicate=np.arange(2000),models=np.array(models),metrics=np.array(['macro_AP','macro_F1']),values=values,delta=values[:,0,:]-values[:,1,:])
        files.append(target)
    jout(B/'intervals.json',intervals)
    tsv(B/'intervals.tsv',list(intervals[0]),[{**r,'involved_classes':json.dumps(r['involved_classes'])} for r in intervals])
    jout(B/'support_summary.json',supports);jout(B/'point_estimates.json',points)
    tsv(B/'point_metrics.tsv',list(point_rows[0]),point_rows)
    tsv(B/'per_class.tsv',list(per_class_rows[0]),per_class_rows)
    jout(E/'audit/E15_bootstrap_checks.json',{'at':now(),'passed':True,'candidate':candidate,'checks':audit,'macro_values_checked':len(audit)*4,'sample_indices_regenerated_exactly':True,'test_read':False,'context':ctx})
    jout(B/'runtime.json',{'at':now(),'seconds':time.monotonic()-start,'pid':os.getpid(),'seed':BOOTSTRAP_SEED,'attempts':2000,'bootstrap_threads':int(os.environ.get('NUMBA_NUM_THREADS','1')),'test_read':False,'limits':CONFIG['limits']})
    files.extend(B/n for n in ['intervals.json','intervals.tsv','support_summary.json','point_estimates.json','point_metrics.tsv','per_class.tsv','runtime.json'])
    files.append(E/'audit/E15_bootstrap_checks.json')
    certificate(B/'DONE.json',files,ctx)
    log('validation_bootstrap_complete',candidate=candidate,intervals=intervals)

if __name__=='__main__':main()

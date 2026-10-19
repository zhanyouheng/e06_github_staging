"""Independent reconstruction of the completed E13/E14 development controller.

No controller selection helpers are reused: grid generation, sklearn AP, tie
breaking, global/class choices and shrinking are independently implemented here.
"""
from access_guard import install
install()
from common import *
from sklearn.metrics import average_precision_score
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import itertools,multiprocessing

AP_SCORE=None
AP_MASKS=None
AP_LABELS=None

def aps_one(i):
    out=np.full((6,32),np.nan)
    for side in range(6):
        for j in range(32):
            out[side,j]=average_precision_score(AP_LABELS[side][j],AP_SCORE[i,AP_MASKS[side][j],j])
    return i,out

def full_ap(y,score):
    return np.array([average_precision_score(y[y[:,j]!=-1,j],score[y[:,j]!=-1,j]) for j in range(32)])

def main():
    global AP_SCORE,AP_MASKS,AP_LABELS
    start=time.monotonic();d=dev();y=d['val_y'];ids=d['val_ids'];cl=d['class_order'];n=len(ids)
    jout(E/'audit/controller_review.pid.json',{'at':now(),'pid':os.getpid(),'script':str(Path(__file__))})
    assert y.shape==(6257,32) and set(np.unique(y))=={-1,0,1}
    assert sha(E/'data/dev.npz')==sha(OLD/'data/dev.npz')
    assert not set(d['train_ids'])&set(ids) and not set(d['train_groups'])&set(d['val_groups'])
    with np.load(E/'data/validation_selector_folds.npz',allow_pickle=False) as z:
        fold=z['fold'];assert np.array_equal(z['protein_id'],ids) and np.array_equal(z['component_id'],d['val_groups'])
    assert set(fold)==set(range(5))
    assert all(len(set(fold[d['val_groups']==g]))==1 for g in np.unique(d['val_groups']))
    names=['E06_mix_098','E04_006_C','E05_ensemble_C','E09_ensemble_A2B','E11_ensemble_extra_trees','E11_ensemble_random_forest','E12_ensemble_C','E12_ensemble_A2B']
    baseline_name=names[0];pool=readj(E/'fusion/pool.json');assert pool['members']==names
    assert pool['test_used'] is False and pool['plan_addendum_sha256']==sha(E/'PLAN_addendum_v1.md')
    source_summary={};source_arrays={};pool_rows=[]
    def summary(name):
        if name in source_summary:return source_summary[name]
        options=[E/f'candidates/{name}/summary.json',E/f'references/{name}/summary.json',OLD/f'candidates/{name}/summary.json']
        path=next(p for p in options if p.exists());q=readj(path);source_summary[name]=q;return q
    def leaves(name):
        cfg=summary(name)['config']
        if 'members' not in cfg:return {name}
        return set().union(*(leaves(n) for n in cfg['members']))
    with np.load(E/'fusion/pool_scores.npz',allow_pickle=False) as z:
        assert np.array_equal(z['protein_id'],ids) and z['members'].tolist()==names
        saved_prob=z['probability'];saved_cdf=z['cdf']
    for i,name in enumerate(names):
        info=pool['source_files'][name]
        vp=E/info['validation'];tp=E/info['train']
        assert sha(vp)==info['validation_sha256'] and sha(tp)==info['train_sha256']
        with np.load(vp,allow_pickle=False) as z:
            assert np.array_equal(z['protein_id'],ids) and np.array_equal(z['class_order'],cl)
            vs=z['scores'].astype(np.float64)
        with np.load(tp,allow_pickle=False) as z:
            assert np.array_equal(z['protein_id'],d['train_ids']) and np.array_equal(z['class_order'],cl)
            ts=z['scores'].astype(np.float64)
        assert vs.shape==(6257,32) and ts.shape==(18770,32)
        assert np.isfinite(vs).all() and np.isfinite(ts).all()
        assert np.array_equal(vs,saved_prob[i])
        train_sorted=np.sort(ts,axis=0)
        quant=np.empty_like(vs)
        for j in range(32):
            lo=np.searchsorted(train_sorted[:,j],vs[:,j],side='left')
            hi=np.searchsorted(train_sorted[:,j],vs[:,j],side='right')
            quant[:,j]=(lo+hi)/(2*len(ts))
        with np.load(E/f'fusion/cdf/{name}.npz',allow_pickle=False) as z:
            assert np.array_equal(z['sorted_train_scores'],train_sorted)
            assert np.array_equal(z['val_scores'],quant)
        assert np.array_equal(saved_cdf[i],quant)
        source_arrays[name]={'probability':vs,'cdf':quant}
        pool_rows.append({'name':name,'train_proteins':len(ts),'val_proteins':len(vs),'validation_macro_AP':float(full_ap(y,vs).mean()),'CDF_train_reference_exact':True,'pool_score_exact':True})
    # Rebuild the prescribed 243 points per space in registration order.
    expected=[];new=names[4:]
    for space in ['probability','cdf']:
        def add(members,weights,kind):
            i=len(expected);expected.append({'name':f'E13_grid_{i:03d}','space':space,'members':members,'weights':weights,'kind':kind,'complexity':len(set().union(*(leaves(z) for z in members))),'order':i,'seed':20261007})
        for member in names:add([member],[1.],'single')
        for a,b in itertools.combinations(names,2):
            if a in new or b in new:
                for tenth in range(1,10):add([a,b],[tenth/10,1-tenth/10],'pair')
        weights=sorted(set(itertools.permutations((.2,.2,.6)))|set(itertools.permutations((.2,.4,.4))))
        for a,b in itertools.combinations(new,2):
            for weight in weights:add([baseline_name,a,b],list(weight),'triple')
        add(names,[1/8]*8,'fixed_uniform')
    specs=readj(E/'configurations/E13_grid.json');assert specs==expected and len(specs)==486
    stored=np.load(E/'fusion/grid_scores.npy',mmap_mode='r',allow_pickle=False)
    assert stored.shape==(486,n,32)
    for i,cfg in enumerate(expected):
        score=sum(w*source_arrays[name][cfg['space']] for name,w in zip(cfg['members'],cfg['weights']))
        assert np.array_equal(score,stored[i]),('grid scores differ',i,float(np.max(np.abs(score-stored[i]))))
    AP_SCORE=stored;AP_MASKS=[];AP_LABELS=[]
    for side in range(6):
        include=np.ones(n,bool) if side==0 else fold!=side-1
        masks=[np.flatnonzero(include&(y[:,j]!=-1)) for j in range(32)]
        labels=[y[masks[j],j] for j in range(32)]
        assert all(np.any(v==0) and np.any(v==1) for v in labels)
        AP_MASKS.append(masks);AP_LABELS.append(labels)
    source_context={'script_sha256':sha(Path(__file__)),'dev_sha256':sha(E/'data/dev.npz'),'pool_sha256':sha(E/'fusion/pool_scores.npz'),'grid_sha256':sha(E/'fusion/grid_scores.npy'),'specs_sha256':sha(E/'configurations/E13_grid.json')}
    independent=np.empty((486,6,32))
    with ProcessPoolExecutor(max_workers=24,mp_context=multiprocessing.get_context('fork')) as executor:
        for begin in range(0,486,48):
            end=min(begin+48,486);p=E/f'audit/controller_AP_blocks/{begin:03d}_{end-1:03d}.npz';cert=p.with_suffix('.DONE.json')
            context={**source_context,'begin':begin,'end_exclusive':end}
            if not verified(cert,context):
                block=np.empty((end-begin,6,32))
                for i,ap in executor.map(aps_one,range(begin,end)):block[i-begin]=ap
                npz(p,AP=block,grid_index=np.arange(begin,end));certificate(cert,[p],context)
            with np.load(p,allow_pickle=False) as z:independent[begin:end]=z['AP']
            log('controller_independent_AP_block',completed=end,total=486)
    with np.load(E/'fusion/grid_AP.npz',allow_pickle=False) as z:
        grid_error=float(np.max(np.abs(independent-z['AP'])))
        assert grid_error<1e-12
        assert np.max(np.abs(independent.mean(2)-z['macro_AP']))<1e-12
    macro=independent.mean(2);baseline=source_arrays[baseline_name]['probability'];base_ap=float(full_ap(y,baseline).mean())
    assert abs(base_ap-.9023565789519359)<1e-12
    def choose(values,indices):
        valid=[i for i in indices if np.isfinite(values[i])]
        if not valid:return None
        best=max(values[i] for i in valid)
        tie=[i for i in valid if best-values[i]<=1e-6]
        return sorted(tie,key=lambda i:(expected[i]['complexity'],expected[i]['order']))[0]
    rows_out=[];trace_rows=[]
    def compare_rule(name,full,oof,extra):
        with np.load(E/f'candidates/{name}/validation.npz',allow_pickle=False) as z:
            assert np.array_equal(z['protein_id'],ids) and np.array_equal(z['class_order'],cl)
            assert np.array_equal(z['scores'],full),('full mismatch',name)
        with np.load(E/f'fusion/crossfit/{name}.npz',allow_pickle=False) as z:
            assert np.array_equal(z['protein_id'],ids) and np.array_equal(z['fold'],fold)
            assert np.array_equal(z['scores'],oof),('OOF mismatch',name)
        q=summary(name);full_ap_value=float(full_ap(y,full).mean());oof_ap_value=float(full_ap(y,oof).mean())
        eligible=oof_ap_value>=base_ap+.002
        assert abs(full_ap_value-q['validation_macro_AP'])<1e-12
        assert abs(oof_ap_value-q['crossfit_macro_AP'])<1e-12
        assert abs(oof_ap_value-base_ap-q['crossfit_delta'])<1e-12
        assert eligible==q['selection_eligible']
        check=readj(E/f'fusion/crossfit/{name}.json')
        assert check['selection_eligible']==eligible and abs(check['reference_macro_AP']-base_ap)<1e-12
        rows_out.append({'name':name,'full_val_AP':full_ap_value,'OOF_AP':oof_ap_value,'OOF_delta_vs_E06':oof_ap_value-base_ap,'selection_eligible':eligible,'beats_0.902357':full_ap_value>.902357,'full_scores_exact':True,'OOF_scores_exact':True,**extra})
        return check
    for space in ['probability','cdf']:
        indices=[i for i,c in enumerate(expected) if c['space']==space]
        for kind in ['learned_grid','fixed_uniform']:
            name=f'E13_{space}_{kind}'
            if kind=='fixed_uniform':
                chosen=next(i for i in indices if expected[i]['kind']=='fixed_uniform');fold_choices=[chosen]*5
            else:
                chosen=choose(macro[:,0],indices);fold_choices=[choose(macro[:,k+1],indices) for k in range(5)]
            full=np.array(stored[chosen]);oof=np.empty_like(full)
            for k,i in enumerate(fold_choices):
                held=fold==k;oof[held]=baseline[held] if i is None else stored[i,held]
                trace_rows.append({'rule':name,'fold':k,'class_index':'all','selected_grid':i,'global_grid':i,'lambda':1.})
            check=compare_rule(name,full,oof,{'rule_type':kind})
            assert check['full_configuration']==expected[chosen]
            for k,entry in enumerate(check['fold_choices']):
                assert entry['fold']==k and entry['selection_grid_index']==fold_choices[k]
                assert entry['selected_configuration']==(None if fold_choices[k] is None else expected[fold_choices[k]])
    probability=[i for i,c in enumerate(expected) if c['space']=='probability']
    base_index=next(i for i in probability if expected[i]['members']==[baseline_name])
    def class_choose(side,shrink):
        include=np.ones(n,bool) if side==0 else fold!=side-1
        glob=choose(macro[:,side],probability);glob=base_index if glob is None else glob
        output=np.empty((n,32));indices=[];lambdas=[]
        for j in range(32):
            known=include&(y[:,j]!=-1)
            if not np.any(y[known,j]==1) or not np.any(y[known,j]==0):
                best=base_index;lam=1.;output[:,j]=stored[best,:,j]
            else:
                best=choose(independent[:,side,j],probability);best=base_index if best is None else best;lam=1.
                if shrink:
                    options=[0.,.25,.5,.75,1.]
                    ap=[average_precision_score(y[known,j],v*stored[best,known,j]+(1-v)*stored[glob,known,j]) for v in options]
                    bestap=max(ap);lam=next(v for v,a in zip(options,ap) if bestap-a<=1e-6)
                output[:,j]=lam*stored[best,:,j]+(1-lam)*stored[glob,:,j] if shrink else stored[best,:,j]
            indices.append(best);lambdas.append(lam)
        return output,{'global_grid_index':glob,'class_grid_indices':indices,'shrinkage_lambdas':lambdas}
    for shrink in [False,True]:
        name='E14_class_'+('shrunk' if shrink else 'free')
        full,fullcfg=class_choose(0,shrink);oof=np.empty_like(full);traces=[]
        for k in range(5):
            score,cfg=class_choose(k+1,shrink);held=fold==k;oof[held]=score[held];traces.append(cfg)
            for j in range(32):trace_rows.append({'rule':name,'fold':k,'class_index':j,'selected_grid':cfg['class_grid_indices'][j],'global_grid':cfg['global_grid_index'],'lambda':cfg['shrinkage_lambdas'][j]})
        check=compare_rule(name,full,oof,{'rule_type':'shrunk' if shrink else 'free'})
        assert check['full_configuration']==fullcfg
        for k,trace in enumerate(check['fold_choices']):
            for key in traces[k]:assert trace[key]==traces[k][key],(name,k,key)
    # Recompute all candidate APs; no grid point is admitted as a standalone eligible scheme.
    all_records=[];stage_counts=Counter()
    for p in sorted((E/'candidates').glob('*/summary.json')):
        q=readj(p);name=q['name'];stage=q['config']['stage'];stage_counts[stage]+=1
        with np.load(p.parent/'validation.npz',allow_pickle=False) as z:
            assert np.array_equal(z['protein_id'],ids) and np.array_equal(z['class_order'],cl)
            ap=float(full_ap(y,z['scores']).mean())
        assert abs(ap-q['validation_macro_AP'])<1e-12
        if stage in ['E11','E12']:
            assert q['selection_eligible']==(name in names[4:])
            if q['selection_eligible']:
                assert len(q['config']['members'])==3
                assert [summary(m)['config']['seed'] for m in q['config']['members']]==[20261007,20261008,20261009]
        else:assert name in {r['name'] for r in rows_out}
        all_records.append({'name':name,'stage':stage,'validation_macro_AP':ap,'selection_eligible':q['selection_eligible'],'beats_0.902357':ap>.902357})
    assert dict(stage_counts)=={'E11':36,'E12':30,'E13':4,'E14':2}
    leaderboard=list(rows(E/'selection/validation_leaderboard.tsv'))
    assert {r['name'] for r in leaderboard}=={r['name'] for r in all_records}
    expected_order=sorted(all_records,key=lambda r:(-summary(r['name'])['validation_macro_AP'],r['name']))
    assert [r['name'] for r in leaderboard]==[r['name'] for r in expected_order]
    for a,b in zip(leaderboard,expected_order):
        assert abs(float(a['validation_macro_AP'])-b['validation_macro_AP'])<1e-12
        assert a['selection_eligible']==str(b['selection_eligible'])
        assert a['beats_user_threshold']==str(b['beats_0.902357'])
    qualifiers=[r for r in all_records if r['selection_eligible'] and r['beats_0.902357']]
    assert not qualifiers
    proposed=readj(E/'selection/proposed_selection.json')
    assert proposed['chosen_candidate'] is None and proposed['chosen_evidence'] is None
    assert proposed['qualifying_improvements']==0 and proposed['eligible_candidates']==4
    assert proposed['status']=='no_eligible_improvement' and proposed['candidate_records']==72
    assert proposed['leaderboard_sha256']==sha(E/'selection/validation_leaderboard.tsv')
    assert proposed['selection_uses_test'] is False and proposed['new_test_evaluation_authorized'] is False
    assert not (E/'selection/final_selection.json').exists() and not (E/'bootstrap_validation/DONE.json').exists()
    tsv(E/'audit/controller_pool_review.tsv',list(pool_rows[0]),pool_rows)
    tsv(E/'audit/controller_rule_review.tsv',list(rows_out[0]),rows_out)
    tsv(E/'audit/controller_fold_choice_review.tsv',list(trace_rows[0]),trace_rows)
    tsv(E/'audit/controller_candidate_AP_review.tsv',list(all_records[0]),all_records)
    result={'at':now(),'passed':True,'seconds':time.monotonic()-start,'pool_members_verified':8,'train_CDFs_reconstructed_exactly':8,'grid_points_rebuilt_exactly':486,'grid_AP_values_checked_vs_sklearn':486*6*32,'grid_AP_max_abs_diff':grid_error,'original_val_macro_classes':32,
            'global_rules_reconstructed':4,'class_rules_reconstructed':2,'fold_selection_traces_verified':len(trace_rows),'candidate_APs_recomputed':72,'baseline_validation_AP':base_ap,'eligibility_margin':.002,'application_threshold':.902357,'eligible_candidates':4,'qualifying_improvements':0,'all_six_fusion_rules_ineligible':True,'proposed_null_correct':True,'test_labels_read':False,'test_inference_run':False,
            'unknown_cells_excluded_from_all_AP':int((y==-1).sum()),'validation_known_P':int((y==1).sum()),'validation_known_N':int((y==0).sum()),'source_context':source_context,'limits':'The component cross-check restricts additional fusion-selection optimism; base architectures and early stopping used validation, so this is not an independent generalization estimate.'}
    jout(E/'audit/controller_independent_review.json',result)
    certificate(E/'audit/controller_independent_review.DONE.json',[E/'audit'/n for n in ['controller_independent_review.json','controller_pool_review.tsv','controller_rule_review.tsv','controller_fold_choice_review.tsv','controller_candidate_AP_review.tsv']],source_context)
    log('controller_actual_independent_review_complete',**result)

if __name__=='__main__':main()

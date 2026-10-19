"""Independent development-only diagnostic; no test paths are opened."""
from common import *
from itertools import combinations
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

NAMES = ['E06_mix_098', 'E04_006_C', 'E05_ensemble_C', 'E09_ensemble_A2B',
         'E01_002_A', 'E01_012_B', 'E01_022_C']

def macro(y, s):
    ap=[]
    for j in range(y.shape[1]):
        k=y[:,j]>=0
        if np.any(y[:,j]==1) and np.any(y[:,j]==0):
            ap.append(float(average_precision_score(y[k,j],s[k,j])))
    return float(np.mean(ap)) if ap else None,len(ap)

def main():
    out=E/'audit/E10';out.mkdir(parents=True,exist_ok=True)
    paths=[OLD/'data/dev.npz',OLD/'data/annotations_dev.tsv',OLD/'selection/validation_selector_folds.npz']
    paths += [OLD/f'candidates/{n}/{f}' for n in NAMES for f in ['summary.json','validation.npz']]
    context={'script_sha256':sha(Path(__file__)), 'seed':SEED, 'sources':{str(p.relative_to(OLD)):sha(p) for p in paths}}
    if verified(out/'DONE.json',context):return
    with np.load(paths[0],allow_pickle=False) as z:d={k:z[k] for k in z.files}
    assert set(d)=={'train_ids','val_ids','train_y','val_y','class_order','train_groups','val_groups','val_annotation_usable'}
    assert d['train_y'].shape==(18770,32) and d['val_y'].shape==(6257,32)
    assert not set(d['train_ids'])&set(d['val_ids'])
    assert not set(d['train_groups'])&set(d['val_groups'])
    assert set(np.unique(d['train_y']))=={-1,0,1} and set(np.unique(d['val_y']))=={-1,0,1}
    with np.load(paths[2],allow_pickle=False) as z:
        assert np.array_equal(z['protein_id'],d['val_ids']) and np.array_equal(z['component_id'],d['val_groups'])
        fold=z['fold']
    assert set(fold)==set(range(5))
    assert all(len(set(fold[d['val_groups']==g]))==1 for g in np.unique(d['val_groups']))
    ann={r['ac']:r for r in rows(paths[1])}
    assert set(ann)==set(d['train_ids'])|set(d['val_ids'])
    lengths=np.array([int(ann[a]['len']) for a in d['val_ids']])
    y=d['val_y'];classes=d['class_order'];summaries={};scores={};pred={};records=[]
    for n in NAMES:
        q=readj(OLD/f'candidates/{n}/summary.json');summaries[n]=q
        with np.load(OLD/f'candidates/{n}/validation.npz',allow_pickle=False) as z:
            assert np.array_equal(z['protein_id'],d['val_ids']) and np.array_equal(z['class_order'],classes)
            scores[n]=z['scores'];pred[n]=z['scores']>=z['thresholds']
        assert np.isfinite(scores[n]).all() and scores[n].shape==y.shape
        ap,k=macro(y,scores[n]);assert abs(ap-q['validation_macro_AP'])<1e-12
        records.append({'name':n,'macro_AP':ap,'K_AP':k,'proteins':len(y),'unknown_predicted_positive':int((pred[n]&(y==-1)).sum()),'all_predicted_positive':int(pred[n].sum())})
    assert abs(records[0]['macro_AP']-.9023565789519359)<1e-12
    tsv(out/'model_summary.tsv',list(records[0]),records)
    per=[];foldrows=[]
    for j,c in enumerate(classes):
        row={'class_id':str(c)}
        for sp in ['train','val']:
            yy=d[sp+'_y'][:,j];gg=d[sp+'_groups'];row.update({sp+'_P':int((yy==1).sum()),sp+'_N':int((yy==0).sum()),sp+'_U':int((yy==-1).sum()),sp+'_positive_components':len(set(gg[yy==1]))})
        for n in NAMES:
            k=y[:,j]>=0;row[n+'_AP']=float(average_precision_score(y[k,j],scores[n][k,j]))
        row['B_minus_A_AP']=row['E01_012_B_AP']-row['E01_002_A_AP'];per.append(row)
        for f in range(5):
            for portion,mask in [('selection',fold!=f),('heldout',fold==f)]:
                yy=y[mask,j];gg=d['val_groups'][mask]
                foldrows.append({'class_id':str(c),'fold':f,'portion':portion,'P':int((yy==1).sum()),'N':int((yy==0).sum()),'U':int((yy==-1).sum()),'positive_components':len(set(gg[yy==1])),'AP_defined':bool((yy==1).any() and (yy==0).any())})
    tsv(out/'per_class_support_AP.tsv',list(per[0]),per)
    tsv(out/'fold_support.tsv',list(foldrows[0]),foldrows)
    pairrows=[]
    # Same sampled known P/N pairs across every model pair, generated only from dev labels.
    pairs={}
    for j in range(32):
        rng=np.random.default_rng(SEED+j);p=np.flatnonzero(y[:,j]==1);n=np.flatnonzero(y[:,j]==0)
        pairs[j]=(rng.choice(p,4096,replace=True),rng.choice(n,4096,replace=True))
    for a,b in combinations(NAMES,2):
        for j,c in enumerate(classes):
            k=y[:,j]>=0;truth=y[k,j]==1;ea=pred[a][k,j]!=truth;eb=pred[b][k,j]!=truth
            ip,ineg=pairs[j];da=scores[a][ip,j]-scores[a][ineg,j];db=scores[b][ip,j]-scores[b][ineg,j]
            rho=float(spearmanr(scores[a][k,j],scores[b][k,j]).statistic)
            pairrows.append({'model_a':a,'model_b':b,'class_id':str(c),'known_cells':int(k.sum()),'spearman_known_scores':rho if np.isfinite(rho) else None,
              'a_errors':int(ea.sum()),'b_errors':int(eb.sum()),'both_errors':int((ea&eb).sum()),'only_a_error':int((ea&~eb).sum()),'only_b_error':int((~ea&eb).sum()),
              'sampled_PN_pairs':4096,'a_pair_inversions':int((da<0).sum()),'b_pair_inversions':int((db<0).sum()),'a_pair_ties':int((da==0).sum()),'b_pair_ties':int((db==0).sum()),
              'a_inversion_b_correct':int(((da<0)&(db>0)).sum()),'b_inversion_a_correct':int(((db<0)&(da>0)).sum())})
    tsv(out/'pairwise_complementarity.tsv',list(pairrows[0]),pairrows)
    strata=[]
    for level,mask in [('<=256',lengths<=256),('257-512',(lengths>256)&(lengths<=512)),('513-1022',(lengths>512)&(lengths<=1022)),('>1022',lengths>1022),('usable_annotation',d['val_annotation_usable']),('annotation_A_zero',~d['val_annotation_usable'])]:
        for n in NAMES:
            ap,k=macro(y[mask],scores[n][mask]);strata.append({'stratum':level,'model':n,'proteins':int(mask.sum()),'P':int((y[mask]==1).sum()),'N':int((y[mask]==0).sum()),'U':int((y[mask]==-1).sum()),'K_AP':k,'macro_AP':ap})
    tsv(out/'strata.tsv',list(strata[0]),strata)
    suppcorr=spearmanr([r['train_P'] for r in per],[r['B_minus_A_AP'] for r in per])
    jout(out/'summary.json',{'at':now(),'baseline_macro_AP':records[0]['macro_AP'],'baseline_AP_matches_to_1e12':True,'B_better_A_classes':[r['class_id'] for r in per if r['B_minus_A_AP']>0],
       'spearman_train_P_vs_B_minus_A_AP':float(suppcorr.statistic),'train_proteins':18770,'val_proteins':6257,'train_components':len(set(d['train_groups'])),'val_components':len(set(d['val_groups'])),
       'test_read':False,'old_inputs_are_explicit_dev_only':True,'label_states_unchanged':True,'fold_count':5,'fold_component_integrity':True,'heldout_fold_class_AP_undefined_rows':sum(not r['AP_defined'] for r in foldrows if r['portion']=='heldout'),
       'notes':['Complementarity and support correlations are descriptive, not causal.','Per-model binary errors use each archived model\'s existing validation-selected thresholds; AP/ranking comparisons do not use thresholds.','Length strata use their own defined class sets; compare paired models within each stratum, not raw macro values between strata.','PN pair samples use known positive and known negative cells only; unknown cells are excluded.','Fold diagnostics do not independently validate base models already selected/early-stopped on validation data.']})
    files=sorted(p for p in out.iterdir() if p.is_file() and p.name!='DONE.json')
    certificate(out/'DONE.json',files,context,{'stage':'E10','complete':True,'test_read':False})
    print(json.dumps(readj(out/'summary.json'),ensure_ascii=False))

if __name__=='__main__':main()

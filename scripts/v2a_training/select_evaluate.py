from common import *
from metrics import *
import numpy as np

COST={'M00':7,'M01':1,'M02':7,'M03':7,'M04':3}
def collect():
    d=dict(np.load(T/'data/development.npz',allow_pickle=False));ids=d['protein_id'];nt=len(d['train_fold']);scores={};oofs={};fail=[];recipes={}
    for family in ['M00','M01','M02','M03','M04']:
        try:
            if family=='M01':
                c=json.loads((T/'selection/M01_train_OOF_C.json').read_text())['selected_C'];root=T/'models/cpu'/family/('C'+str(c));ss=[]
                for j in range(32):
                    f=root/'full'/('class%02d'%j);assert verified(f);z=np.load(f/'predictions.npz');assert np.array_equal(z['protein_id'],ids[nt:]);ss.append(z['score'])
                score=np.column_stack(ss);z=np.load(root/'oof.npz');oof=z['scores'];recipe={'type':'logistic','C':c,'root':str(root.relative_to(T))}
            else:
                nn={};no={};paths={}
                for route in (['A2B'] if family=='M04' else ['A2B','C']):
                    seed_s=[];seed_o=[];paths[route]=[]
                    for seed in [20261006,20261007,20261008]:
                        f=T/'models/neural'/f'{family}_{route}_s{seed}';assert verified(f)
                        z=np.load(f/'full/predictions.npz');assert np.array_equal(z['protein_id'],ids[nt:]);seed_s.append(z['scores'].astype(float));z=np.load(f/'oof.npz');assert np.array_equal(z['protein_id'],ids[:nt]);seed_o.append(z['scores'].astype(float));paths[route].append(str(f.relative_to(T)))
                    nn[route]=np.mean(seed_s,axis=0);no[route]=np.mean(seed_o,axis=0)
                if family=='M04':score=nn['A2B'];oof=no['A2B'];recipe={'type':'fixed_three_seed_mean','neural':paths}
                else:
                    root=T/'models/cpu'/family/'trees';ss=[]
                    for j in range(32):
                        f=root/'full'/('class%02d'%j);assert verified(f);z=np.load(f/'predictions.npz');assert np.array_equal(z['protein_id'],ids[nt:]);ss.append(z['score'])
                    et=np.column_stack(ss);eo=np.load(root/'oof.npz')['scores'];score=.2*nn['A2B']+.1*nn['C']+.7*et;oof=.2*no['A2B']+.1*no['C']+.7*eo;recipe={'type':'E06_refit','weights':{'A2B_mlp':.2,'C_mlp':.1,'C_ExtraTrees':.7},'neural':paths,'trees':str(root.relative_to(T))}
            assert score.shape==(len(ids)-nt,32) and oof.shape==(nt,32) and np.isfinite(score).all() and np.isfinite(oof).all()
            scores[family]=score;oofs[family]=oof;recipes[family]=recipe
            p=T/'candidates'/family;npz(p/'validation.npz',protein_id=ids[nt:],class_order=d['class_order'],scores=score);npz(p/'oof.npz',protein_id=ids[:nt],class_order=d['class_order'],scores=oof);js(p/'recipe.json',recipe);done(p,[p/'validation.npz',p/'oof.npz',p/'recipe.json'])
        except (AssertionError,FileNotFoundError,KeyError) as e:fail.append({'family':family,'status':'failed_or_incomplete','reason':str(e) or 'required model certificate absent/invalid'})
    assert 'M00' in scores,'M00 is required; no substitute baseline allowed'
    tsv(T/'results/failed_families.tsv',fail,['family','status','reason']);return d,scores,recipes

class Selector:
    def __init__(self,y,scores,fold):
        self.y=y;self.scores=scores;self.fold=fold;self.base=list(scores);self.grid=[]
        self.grid.append({'name':'M00','weights':{'M00':1.},'cost':COST['M00']})
        for f in self.base:
            if f=='M00':continue
            for w in [.25,.5,.75,1.]:
                ww={f:w}
                if w<1:ww['M00']=1-w
                self.grid.append({'name':f'M00_{f}_{w:.2f}','weights':ww,'cost':sum(COST[x] for x in ww)})
        self.gscores=[sum(v*scores[k] for k,v in r['weights'].items()) for r in self.grid]
        self.cache={}
    def build(self,fit):
        key=tuple(int(v) for v in np.flatnonzero(np.bincount(self.fold[fit],minlength=5)))
        if key in self.cache:return self.cache[key]
        aa=np.array([aps(self.y[fit],s[fit]) for s in self.gscores]);fixed=((self.y[fit]==1).sum(0)>0)&((self.y[fit]==0).sum(0)>0);assert np.isfinite(aa[:,fixed]).all();vals=aa[:,fixed].mean(axis=1)
        order=sorted(range(len(vals)),key=lambda i:(-vals[i],self.grid[i]['cost'],self.grid[i]['name']));g=order[0];global_s=self.gscores[g];weights=[];class_s=np.empty_like(global_s);jsel=[]
        for j in range(32):
            if not ((self.y[fit,j]==1).any() and (self.y[fit,j]==0).any()):
                class_s[:,j]=self.scores['M00'][:,j];weights.append({'M00':1.});jsel.append('fallback_M00');continue
            best=sorted(range(len(vals)),key=lambda i:(-aa[i,j],self.grid[i]['cost'],self.grid[i]['name']))[0]
            class_s[:,j]=.5*self.gscores[best][:,j]+.5*global_s[:,j];cw={}
            for gg in [g,best]:
                for n,v in self.grid[gg]['weights'].items():cw[n]=cw.get(n,0)+.5*v
            weights.append(cw);jsel.append(self.grid[best]['name'])
        recipes={'M05':{'global_grid':self.grid[g],'fit_folds':list(key)},'M06':{'class_weights':weights,'class_grid':jsel,'global_shrink_target':self.grid[g],'shrinkage':.5,'fit_folds':list(key)}}
        val={**self.scores,'M05':global_s,'M06':class_s};self.cache[key]=(val,recipes,vals.tolist());return self.cache[key]
    def crossfit(self,scope):
        vals={n:np.full_like(self.scores['M00'],np.nan) for n in [*self.base,'M05','M06']};details=[]
        for k in sorted(set(self.fold[scope])):
            held=scope&(self.fold==k);fit=scope&(self.fold!=k);pred,recipe,gridap=self.build(fit)
            for n in vals:vals[n][held]=pred[n][held]
            details.append({'holdout_fold':int(k),'fit_folds':sorted(set(int(v) for v in self.fold[fit])),'recipes':recipe,'grid_macro_AP':gridap})
        return vals,details
    def choose(self,scope):
        pred,recipe,gridap=self.build(scope);cross,details=self.crossfit(scope);base=macro_ap(self.y[scope],self.scores['M00'][scope]);rows=[]
        for n in pred:
            full=macro_ap(self.y[scope],pred[n][scope]);cf=macro_ap(self.y[scope],cross[n][scope]);eligible=n!='M00' and cf>=base+.002 and full>base
            if n in COST:cost=COST[n]
            elif n=='M05':cost=recipe[n]['global_grid']['cost']
            else:cost=sum(COST[f] for f in set(k for row in recipe[n]['class_weights'] for k in row))
            rows.append({'candidate':n,'full_macro_AP':full,'crossfit_macro_AP':cf,'crossfit_delta_vs_M00':cf-base,'eligible':eligible,'primitive_cost':cost})
        elig=[r for r in rows if r['eligible']]
        if not elig:chosen='M00'
        else:
            top=max(r['full_macro_AP'] for r in elig);ties=[r for r in elig if r['full_macro_AP']>=top-1e-6];chosen=sorted(ties,key=lambda r:(r['primitive_cost'],r['candidate']))[0]['candidate']
        return chosen,pred,recipe,rows,cross,details

def evaluate(d,scores,recipes):
    nt=len(d['train_fold']);ids=d['protein_id'][nt:];y=d['y'][nt:];classes=d['class_order'];fold=d['val_fold'];allrows=np.ones(len(y),bool);sel=Selector(y,scores,fold)
    chosen,full,fr,leader,cf,fold_details=sel.choose(allrows)
    # Nested family qualification: outer held fold plays no role even in eligibility filtering.
    nested=np.empty_like(scores['M00']);nested_pred=np.zeros_like(y,dtype=bool);outer=[]
    for k in range(5):
        fit=fold!=k;held=~fit;n,pp,rr,ll,cc,dd=sel.choose(fit);nested[held]=pp[n][held];th=thresholds(y[fit],pp[n][fit]);nested_pred[held]=pp[n][held]>=th
        outer.append({'holdout_fold':k,'chosen_on_other_four':n,'fit_candidates':ll,'fit_recipe':rr.get(n,recipes.get(n)),'thresholds':th.tolist(),'inner_crossfit_details':dd})
    nested_ap=macro_ap(y,nested);base=macro_ap(y,scores['M00']);familygate=nested_ap>=base+.002
    selected=chosen if familygate else 'M00'
    js(T/'selection/fusion_grid.json',{'grid':sel.grid,'count':len(sel.grid),'variants':['V0'],'not_applicable':['V1']});js(T/'selection/fold_selection.json',{'crossfit':fold_details,'nested_family_crossfit':outer})
    tsv(T/'results/validation_leaderboard.tsv',leader)
    npz(T/'selection/family_crossfit.npz',protein_id=ids,class_order=classes,scores=nested,prediction=nested_pred)
    oldpath=O/'exploration/candidates/E06_mix_098/validation.npz';cert=json.loads((oldpath.parent/'DONE.json').read_text());recorded=[v for k,v in cert['files'].items() if k.endswith('E06_mix_098/validation.npz')];assert len(recorded)==1 and sha(oldpath)==recorded[0]
    old=np.load(oldpath,allow_pickle=False);assert np.array_equal(old['protein_id'],ids) and np.array_equal(old['class_order'],classes)
    full['E06_archived']=old['scores'];cf['E06_archived']=old['scores'];main=[];per=[];strata=[];upsrc=[];sens=[]
    metadata=read_tsv(T/'data/metadata.tsv')[nt:];tail={r['ac']:r for r in read_tsv(O/'tail_diagnosis/data/metadata_dev.tsv')};background=np.array([r['origin']=='background_sample' for r in metadata]);sources=d['source_names'][d['source_code'][nt:]]
    valgroups=d['groups'][nt:];kap=((y==1).sum(0)>0)&((y==0).sum(0)>0);kf=(y==1).sum(0)>0
    js(T/'data/evaluation_class_sets.json',{'K_AP':list(classes[kap]),'K_F1':list(classes[kf]),'class_order':list(classes),'fitted_on_labels_only_once':True,'reference_label_sha256':sha(T/'data/development.npz')})
    strata_masks={'all':allrows,'usable_annotation':d['val_annotation_usable']}
    length=np.array([int(r['length']) for r in metadata]);score_meta=np.array([r['annotation_score'] if r['annotation_score'] else 'missing' for r in metadata])
    for name,m in [('length_1_256',length<=256),('length_257_512',(length>256)&(length<=512)),('length_513_1022',(length>512)&(length<=1022)),('length_1023_plus',length>1022)]:strata_masks[name]=m
    for val in sorted(set(score_meta)):strata_masks['annotation_score_'+val]=score_meta==val
    for col in ['origin','fragment']:
        vals=np.array([r[col] for r in metadata])
        for val in sorted(set(vals)):strata_masks[col+'_'+val]=vals==val
    for col in ['kingdom','database','annotation_status','source_status','mapping_difference']:
        vals=np.array([tail[a][col] for a in ids])
        for val in sorted(set(vals)):strata_masks[col+'_'+val]=vals==val
    for name,s in full.items():
        th=old['thresholds'] if name=='E06_archived' else thresholds(y,s);pred=s>=th
        # For adaptive selectors, fit the entire predictor rule on each selection side before thresholds.
        cp=np.zeros_like(pred);ct=[]
        for k in range(5):
            fit=fold!=k;held=~fit
            if name=='E06_archived':kt=old['thresholds'];ps=s
            else:
                built,_,_=sel.build(fit);ps=built[name];kt=thresholds(y[fit],ps[fit])
            cp[held]=ps[held]>=kt;ct.append(kt)
        sm,rows=summary(y,s,pred,classes,background);cs,_=summary(y,cf[name],cp,classes,background);sm.update(candidate=name,crossfit_macro_AP=cs['macro_AP'],crossfit_macro_F1=cs['macro_F1'],selected=name==selected,variant='historical_anchor' if name=='E06_archived' else 'V0')
        sm['all_output_precision_reference_bound_low']=sm['TP']/sm['all_positive'] if sm['all_positive'] else None
        sm['all_output_precision_reference_bound_high']=(sm['TP']+sm['unknown_positive'])/sm['all_positive'] if sm['all_positive'] else None
        sm.update(proteins=len(y),components=len(set(valgroups)),P=int((y==1).sum()),N=int((y==0).sum()),U=int((y<0).sum()),K_AP_fraction=f'{int(kap.sum())}/32',K_F1_fraction=f'{int(kf.sum())}/32',all_negative_background_components=len(set(valgroups[(y==0).all(1)&background])))
        main.append(sm)
        for j,r in enumerate(rows):
            per.append({'candidate':name,**r,'threshold':float(th[j]),'val_positive_components':len(set(valgroups[y[:,j]==1])),'train_P':int((d['y'][:nt,j]==1).sum()),'train_N':int((d['y'][:nt,j]==0).sum()),'train_U':int((d['y'][:nt,j]<0).sum()),'train_positive_components':len(set(d['groups'][:nt][d['y'][:nt,j]==1])),'AP_NA_reason':'no_positive' if r['P']==0 else 'no_negative' if r['N']==0 else '', 'F1_NA_reason':'no_positive' if r['P']==0 else ''})
        dest=T/'predictions'/name;npz(dest/'validation.npz',protein_id=ids,class_order=classes,scores=s,prediction=pred,thresholds=th,crossfit_scores=cf[name],crossfit_prediction=cp,crossfit_thresholds=np.array(ct));js(dest/'metrics.json',sm);done(dest,[dest/'validation.npz',dest/'metrics.json'])
        for st,m in strata_masks.items():
            if not m.any():continue
            ss,_=summary(y[m],s[m],pred[m],classes,background[m]);strata.append({'candidate':name,'stratum':st,'proteins':int(m.sum()),'components':len(set(valgroups[m])),'P':int((y[m]==1).sum()),'N':int((y[m]==0).sum()),'U':int((y[m]<0).sum()),**ss})
        for src in sorted(set(sources.ravel())):
            mask=sources==src;U=mask&(y<0);pos=pred&mask;known=mask&(y>=0)
            upsrc.append({'candidate':name,'source':src,'cells':int(mask.sum()),'P':int((mask&(y==1)).sum()),'N':int((mask&(y==0)).sum()),'U':int(U.sum()),'unknown_positive':int((pred&U).sum()),'unknown_alarm_rate':float((pred&U).sum()/U.sum()) if U.any() else None,'source_unknown_share_all_prediction':float((pred&U).sum()/pred.sum()) if pred.any() else None,'known_precision':float((pos&(y==1)).sum()/(pos&known).sum()) if (pos&known).any() else None})
        ss,_=summary(d['before_ruleA_y'][nt:],s,pred,classes,background);ss['candidate']=name;ss['sensitivity']='same fixed scores and thresholds; before approved 18-cell ruleA priority';sens.append(ss)
        j=list(classes).index('virulence_activity');without=np.arange(32)!=j;ss,_=summary(y[:,without],s[:,without],pred[:,without],classes[without],background);ss['candidate']=name;ss['sensitivity']='31-class excluding VA, descriptive only, not a selection metric';sens.append(ss)
    tsv(T/'results/metrics_summary.tsv',main);tsv(T/'results/per_class.tsv',per);tsv(T/'results/strata.tsv',strata);tsv(T/'results/unknown_by_source.tsv',upsrc);tsv(T/'results/sensitivity.tsv',sens)
    # Arithmetic label-coverage audit only; the archived anchor is not a selectable candidate.
    oldpred=old['scores']>=old['thresholds'];oldy=d['old_y'][nt:]
    assert int((oldpred&(oldy<0)).sum())==14243 and int((oldpred&(oldy==1)).sum())==4713
    coverage=[]
    for frame,yy,pp in [('E06_old_reference',oldy,oldpred),('E06_v2a_reference',y,oldpred),('M00_v2a_reference',y,full['M00']>=thresholds(y,full['M00'])),('selected_v2a_reference',y,full[selected]>=thresholds(y,full[selected]))]:
        u=yy<0;up=int((pp&u).sum());total=int(pp.sum());coverage.append({'frame':frame,'U_cells':int(u.sum()),'U_positive':up,'all_positive':total,'U_alarm_rate':up/int(u.sum()) if u.any() else None,'U_share_all_positive':up/total if total else None,'known_TP':int((pp&(yy==1)).sum()),'known_FP':int((pp&(yy==0)).sum())})
    tsv(T/'results/coverage_vs_score_change.tsv',coverage)
    tsv(T/'results/archived_E06_old_U_positive_transitions.tsv',[{'new_state':label,'old_U_positive_cells':int((oldpred&(oldy<0)&(y==value)).sum())} for value,label in [(1,'P'),(0,'N'),(-1,'U')]])
    # Explicit evidence-tier cell evaluations; source is never an input predictor.
    tier=[]
    for src in sorted(set(sources[y==1])):
        pos=sources==src
        for name in ['E06_archived','M00',selected]:
            ss=full[name];th=old['thresholds'] if name=='E06_archived' else thresholds(y,ss);pred=ss>=th;pp=(y==1)&pos
            tier.append({'candidate':name,'positive_evidence_source':src,'positive_cells':int(pp.sum()),'recalled_positive_cells':int((pp&pred).sum()),'recall':float((pp&pred).sum()/pp.sum()) if pp.any() else None,'AP':'not reported: post-hoc source masking changes its target universe'})
    tsv(T/'results/evidence_tiers.tsv',tier)
    smap={r['candidate']:r for r in main};decision={'at':now(),'status':'eligible_improvement' if selected!='M00' else 'no_eligible_improvement_retain_M00','selected':selected,'best_fullval_eligible_family_before_family_gate':chosen,'family_selection_crossfit_macro_AP':nested_ap,'family_selection_crossfit_delta':nested_ap-base,'family_gate':familygate,'candidate_threshold':.002,'selected_metrics':smap[selected],'reference_metrics':smap['M00'],'archived_E06_metrics_same_mask':smap['E06_archived'],'recipe':fr.get(selected,recipes.get(selected)),'old_E06_validation_sha256':sha(oldpath),'variants':{'V0':'executed','V1':'not_applicable_no_blindcheck_0'},'test_loaded':False,'rule_A_sensitivity_not_used_for_selection':True}
    js(T/'selection/final_selection.json',decision);tsv(T/'results/variant_register.tsv',[{'family':f,'variant':'V1','status':'not_applicable','reason':'eligible no_blindcheck=0; identical targets to V0; no duplicate fits'} for f in ['M00','M01','M02','M03','M04','M05','M06']])
    done(T/'selection',[p for root in ['selection','results','predictions'] for p in (T/root).rglob('*') if p.is_file() and p.name!='DONE.json'],selected=selected)
    print(json.dumps(decision,ensure_ascii=False),flush=True)

def main():
    for name in ['gpu_worker_0_finished.json','gpu_worker_1_finished.json','cpu_finished.json']:assert (T/'logs'/name).exists(),name+' still pending'
    assert verified(T/'preflight') and verified(T/'features')
    d,s,r=collect();evaluate(d,s,r)
if __name__=='__main__':main()

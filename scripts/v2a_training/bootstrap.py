from common import *
import numpy as np
from concurrent.futures import ProcessPoolExecutor,as_completed

DATA=None
def initialize():
    global DATA
    d=np.load(T/'data/development.npz');nt=len(d['train_fold']);y=d['y'][nt:];g=d['groups'][nt:];groups,gi=np.unique(g,return_inverse=True);selection=json.loads((T/'selection/final_selection.json').read_text());selected=selection['selected'];names=list(dict.fromkeys(['M00',selected,'E06_archived']));prep={};pn=(y==1);nn=(y==0);kap=pn.sum(0)>0;kap&=nn.sum(0)>0;kf=pn.sum(0)>0
    for name in names:
        z=np.load(T/'predictions'/name/'validation.npz');s=z['scores'];pred=z['prediction'];caches=[]
        for j in range(32):
            known=np.flatnonzero(y[:,j]>=0);order=np.argsort(-s[known,j],kind='stable');idx=known[order];v=s[idx,j];end=np.r_[np.flatnonzero(v[:-1]!=v[1:]),len(v)-1];caches.append((idx,(y[idx,j]==1).astype(float),end))
        prep[name]=(caches,pred)
    DATA=(y,gi,groups,kap,kf,prep,selected)

def run_chunk(start):
    dest=T/'bootstrap/chunks'/f'{start:04d}'
    if verified(dest):return start
    y,gi,groups,kap,kf,prep,selected=DATA;draw=np.load(T/'bootstrap/sampling_indices.npy',mmap_mode='r')[start:min(start+100,2000)];names=list(prep);ap=np.full((len(draw),len(names)),np.nan);f1=ap.copy();invalidAP=np.zeros((len(draw),32),bool);invalidF=np.zeros_like(invalidAP)
    for r,sampled in enumerate(draw):
        wc=np.bincount(sampled,minlength=len(groups));w=wc[gi].astype(float);P=w@(y==1);N=w@(y==0);invalidAP[r]=kap&((P==0)|(N==0));invalidF[r]=kf&(P==0)
        for k,name in enumerate(names):
            caches,pred=prep[name];vals=np.full(32,np.nan)
            for j,(idx,target,end) in enumerate(caches):
                if not(P[j]>0 and N[j]>0):continue
                ww=w[idx];tp=np.cumsum(ww*target)[end];tot=np.cumsum(ww)[end];precision=np.divide(tp,tot,out=np.zeros_like(tp),where=tot>0);vals[j]=np.sum(np.diff(np.r_[0,tp])/P[j]*precision)
            if not invalidAP[r].any():ap[r,k]=np.mean(vals[kap])
            tp=w@((y==1)&pred);fp=w@((y==0)&pred)
            if not invalidF[r].any():f1[r,k]=np.mean((2*tp/(P+tp+fp))[kf])
    npz(dest/'replicates.npz',replicate=np.arange(start,start+len(draw)),names=np.array(names),macro_AP=ap,macro_F1=f1,invalid_AP_class=invalidAP,invalid_F1_class=invalidF);done(dest,[dest/'replicates.npz']);return start

def main():
    assert verified(T/'selection');initialize();y,gi,groups,kap,kf,prep,selected=DATA
    p=T/'bootstrap/sampling_indices.npy'
    if not p.exists():npy(p,np.random.default_rng(SEED).integers(0,len(groups),size=(2000,len(groups)),dtype=np.int32));npz(T/'bootstrap/sampling_frame.npz',component_id=groups,protein_component_index=gi)
    else:
        expected=np.random.default_rng(SEED).integers(0,len(groups),size=(2000,len(groups)),dtype=np.int32);np.testing.assert_array_equal(np.load(p),expected)
    with ProcessPoolExecutor(max_workers=16,initializer=initialize) as pool:
        for f in as_completed([pool.submit(run_chunk,s) for s in range(0,2000,100)]):print('bootstrap chunk',f.result(),flush=True)
    zz=[np.load(T/'bootstrap/chunks'/f'{s:04d}'/'replicates.npz') for s in range(0,2000,100)];names=list(zz[0]['names']);ap=np.concatenate([z['macro_AP'] for z in zz]);f1=np.concatenate([z['macro_F1'] for z in zz]);iap=np.concatenate([z['invalid_AP_class'] for z in zz]);iff=np.concatenate([z['invalid_F1_class'] for z in zz]);classes=np.load(T/'data/development.npz')['class_order'];npz(T/'bootstrap/replicates.npz',names=np.array(names),macro_AP=ap,macro_F1=f1,invalid_AP_class=iap,invalid_F1_class=iff)
    summary=[];selection=json.loads((T/'selection/final_selection.json').read_text());metrics={n:json.loads((T/'predictions'/n/'metrics.json').read_text()) for n in names}
    for a,b,purpose in [(selected,'M00','primary_final_minus_M00'),('M00','E06_archived','supervision_and_training_change'),(selected,'E06_archived','supplement_final_minus_historical_anchor')]:
        for label,arr,invalid in [('macro_AP',ap,iap),('macro_F1',f1,iff)]:
            delta=arr[:,names.index(a)]-arr[:,names.index(b)];valid=np.isfinite(delta);conditional=np.quantile(delta[valid],[.025,.975]).tolist() if valid.any() else None;all_valid=bool(valid.all());badnames=[str(classes[j]) for j in np.flatnonzero(invalid.any(0))]
            summary.append({'comparison':a+' - '+b,'purpose':purpose,'metric':label,'point_delta':metrics[a][label]-metrics[b][label],'CI_status':'valid_percentile' if all_valid else 'NA_any_invalid_round','main_CI_low':conditional[0] if all_valid else None,'main_CI_high':conditional[1] if all_valid else None,'conditional_CI_low':conditional[0] if conditional else None,'conditional_CI_high':conditional[1] if conditional else None,'valid_rounds':int(valid.sum()),'valid_fraction':float(valid.mean()),'invalid_classes':';'.join(badnames),'rounds':2000,'seed':SEED})
    tsv(T/'bootstrap/intervals.tsv',summary);tsv(T/'bootstrap/invalid_class_counts.tsv',[{'class':c,'AP_invalid_rounds':int(iap[:,j].sum()),'F1_invalid_rounds':int(iff[:,j].sum())} for j,c in enumerate(classes)])
    js(T/'bootstrap/protocol.json',{'at':now(),'seed':SEED,'rounds':2000,'component_count':len(groups),'sampling_index_sha256':sha(p),'shared_draws':True,'multiplicity_weighted':True,'thresholds_and_fitted_models_fixed':True,'macro_classes_fixed':True,'interval_is_conditional_on_development_selection':True,'any_invalid_round_main_CI_NA':True,'V1_V0_comparison':'not_applicable_identical_targets'})
    done(T/'bootstrap',[p,T/'bootstrap/sampling_frame.npz',T/'bootstrap/replicates.npz',T/'bootstrap/intervals.tsv',T/'bootstrap/invalid_class_counts.tsv',T/'bootstrap/protocol.json']+[T/'bootstrap/chunks'/f'{s:04d}'/'DONE.json' for s in range(0,2000,100)])
if __name__=='__main__':main()

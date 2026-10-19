"""Descriptive arithmetic on already frozen validation predictions; no selection."""
from common import *
from metrics import perclass
import numpy as np
def main():
    d=np.load(T/'data/development.npz');nt=len(d['train_fold']);y=d['y'][nt:];classes=d['class_order'];rows=[]
    for p in sorted((T/'predictions').glob('*/validation.npz')):
        if p.parent.name=='E06_archived':continue
        z=np.load(p);full=perclass(y,z['scores'],z['prediction'],classes);cf=perclass(y,z['crossfit_scores'],z['crossfit_prediction'],classes)
        for j,(a,b) in enumerate(zip(full,cf)):
            rows.append({'candidate':p.parent.name,'class':str(classes[j]),'P':a['P'],'N':a['N'],'U':a['U'],'AP_full':a['AP'],'AP_crossfit':b['AP'],'F1_full':a['F1'],'F1_crossfit':b['F1'],'F1_full_minus_crossfit':a['F1']-b['F1'] if a['F1'] is not None else None,'TP_full':a['TP'],'TP_crossfit':b['TP'],'FP_full':a['FP'],'FP_crossfit':b['FP'],'threshold_full':float(z['thresholds'][j]),'threshold_crossfit_min':float(z['crossfit_thresholds'][:,j].min()),'threshold_crossfit_max':float(z['crossfit_thresholds'][:,j].max()),'interpretation':'M00..M04 fixed score model: threshold estimation only; M05/M06: both selector and thresholds are crossfitted'})
    tsv(T/'results/crossfit_per_class.tsv',rows);js(T/'audit/crossfit_description.json',{'at':now(),'purpose':'report-only decomposition of the observed F1 selection gap','new_model_fit':False,'new_threshold_selection':False,'selection_changed':False,'inputs':'already archived scores/predictions/thresholds from selection phase'})
if __name__=='__main__':main()

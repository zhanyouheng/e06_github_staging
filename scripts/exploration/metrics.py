import numpy as np
from sklearn.metrics import average_precision_score
from numba import njit

def class_sets(y):return np.flatnonzero(((y==1).sum(0)>0)&((y==0).sum(0)>0)),np.flatnonzero((y==1).sum(0)>0)
def evaluate(y,s,threshold,classes,groups,support=None):
    assert y.shape==s.shape and y.shape[1]==len(classes) and np.isfinite(s).all()
    pred=s>=threshold;ka,kf=class_sets(y);records=[]
    for j,c in enumerate(classes):
        pos=y[:,j]==1;neg=y[:,j]==0;unk=y[:,j]==-1;p=int(pos.sum());n=int(neg.sum());u=int(unk.sum())
        tp=int((pred[:,j]&pos).sum());fp=int((pred[:,j]&neg).sum());fn=p-tp;tn=n-fp
        precision=tp/(tp+fp) if tp+fp else (0.0 if p+n else None)
        records.append(dict(class_id=c,P=p,N=n,U=u,positive_components=len(set(groups[pos])),AP=float(average_precision_score(y[pos|neg,j],s[pos|neg,j])) if p and n else None,
          precision=precision,recall=tp/p if p else None,F1=2*tp/(2*tp+fp+fn) if p else None,FPR=fp/n if n else None,
          TP=tp,FP=fp,TN=tn,FN=fn,AP_NA_reason='no_known_positives' if not p else 'no_known_negatives' if not n else '',
          recall_F1_NA_reason='no_known_positives' if not p else '',FPR_NA_reason='no_known_negatives' if not n else '',
          precision_NA_reason='all_unknown' if not p+n else '',training_support_status=support[j] if support else 'not_applicable',
          unknown_predicted_positive=int((pred[:,j]&unk).sum())))
    tp=sum(x['TP'] for x in records);fp=sum(x['FP'] for x in records);fn=sum(x['FN'] for x in records)
    allneg=np.all(y==0,axis=1);b0=int(allneg.sum());err=int(pred[allneg].any(axis=1).sum());predpos=int(pred.sum());up=int((pred&(y==-1)).sum())
    out=dict(proteins=len(y),components=len(set(groups)),P=int((y==1).sum()),N=int((y==0).sum()),U=int((y==-1).sum()),K_AP=len(ka),K_F1=len(kf),class_denominator=32,
      K_AP_classes=[classes[j] for j in ka],K_F1_classes=[classes[j] for j in kf],
      macro_AP=float(np.mean([records[j]['AP'] for j in ka])) if len(ka) else None,
      macro_F1=float(np.mean([records[j]['F1'] for j in kf])) if len(kf) else None,
      micro_precision=tp/(tp+fp) if tp+fp else (0.0 if int((y!=-1).sum()) else None),micro_recall=tp/(tp+fn) if tp+fn else None,
      micro_F1=2*tp/(2*tp+fp+fn) if tp+fn else None,TP=tp,FP=fp,FN=fn,TN=sum(x['TN'] for x in records),
      fully_negative_background_proteins=b0,fully_negative_background_components=len(set(groups[allneg])),any_label_false_positives=err,any_label_FPR=err/b0 if b0 else None,
      all_predicted_positive=predpos,unknown_predicted_positive=up,unknown_positive_fraction=up/predpos if predpos else None)
    return records,out

@njit(cache=True)
def threshold_scan(values,labels,columns,p_counts,kf,nclass):
    tp=np.zeros(nclass,np.int64);fp=np.zeros(nclass,np.int64)
    best=-1.;best_t=np.nextafter(values[0],np.inf);n=values.size;i=0;candidates=1
    # The all-negative endpoint has F1=0 for every class in K_F1.
    best=0.
    while i<n:
        v=values[i]
        while i<n and values[i]==v:
            j=columns[i]
            if labels[i]==1:tp[j]+=1
            elif labels[i]==0:fp[j]+=1
            i+=1
        total=0.
        for j in kf:total+=2.*tp[j]/(p_counts[j]+tp[j]+fp[j])
        score=total/kf.size
        if score>best:best=score;best_t=v
        candidates+=1
    return best_t,best,candidates

def choose_threshold(y,s):
    _,kf=class_sets(y);assert len(kf),'No validation macro F1 classes'
    flat=s.ravel();order=np.argsort(-flat,kind='stable');cols=np.tile(np.arange(s.shape[1]),len(s))
    t,f,n=threshold_scan(flat[order],y.ravel()[order],cols[order],(y==1).sum(0),kf,s.shape[1])
    return {'threshold':float(t),'validation_macro_F1':float(f),'candidate_count':int(n),'K_F1':len(kf),'all_negative_endpoint':float(np.nextafter(s.max(),np.inf))}

@njit(cache=True)
def weighted_ap_sorted(binary,score,weight):
    p=0.;n=0.
    for i in range(len(binary)):
        if binary[i]==1:p+=weight[i]
        else:n+=weight[i]
    if p==0 or n==0:return np.nan
    ctp=0.;ctotal=0.;result=0.;i=0
    while i<len(binary):
        v=score[i];gp=0.;gn=0.
        while i<len(binary) and score[i]==v:
            if binary[i]==1:gp+=weight[i]
            else:gn+=weight[i]
            i+=1
        ctp+=gp;ctotal+=gp+gn
        if gp>0:result+=(gp/p)*(ctp/ctotal)
    return result


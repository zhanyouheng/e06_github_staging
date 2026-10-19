import numpy as np

def aps(y,s,weights=None):
    result=np.full(y.shape[1],np.nan)
    for j in range(y.shape[1]):
        m=y[:,j]>=0;t=(y[m,j]==1);v=s[m,j];w=np.ones(len(v),float) if weights is None else np.asarray(weights)[m].astype(float)
        keep=w>0;t=t[keep];v=v[keep];w=w[keep]
        P=np.sum(w*t);N=np.sum(w*(~t))
        if not(P>0 and N>0):continue
        order=np.argsort(-v,kind='stable');v=v[order];t=t[order];w=w[order];ends=np.r_[np.flatnonzero(v[:-1]!=v[1:]),len(v)-1]
        tp=np.cumsum(w*t)[ends];total=np.cumsum(w)[ends];result[j]=np.sum(np.diff(np.r_[0,tp])/P*tp/total)
    return result

def macro_ap(y,s):
    a=aps(y,s);fixed=((y==1).sum(0)>0)&((y==0).sum(0)>0)
    assert np.isfinite(a[fixed]).all()
    return float(np.mean(a[fixed])) if fixed.any() else None

def thresholds(y,s):
    out=np.empty(y.shape[1],np.float64)
    for j in range(y.shape[1]):
        m=y[:,j]>=0;t=y[m,j]==1;v=s[m,j]
        endpoint=np.nextafter(float(np.max(s[:,j])),np.inf)
        if not len(v) or not t.any():out[j]=endpoint;continue
        order=np.argsort(-v,kind='stable');v=v[order];t=t[order];ends=np.r_[np.flatnonzero(v[:-1]!=v[1:]),len(v)-1];tp=np.cumsum(t)[ends];f1=2*tp/(t.sum()+ends+1)
        # Descending scores; argmax selects highest threshold among ties.
        out[j]=float(v[ends[int(np.argmax(f1))]])
    return out

def perclass(y,s,pred,classes,weights=None):
    a=aps(y,s,weights);w=np.ones(len(y),float) if weights is None else np.asarray(weights,dtype=float);rows=[]
    for j,c in enumerate(classes):
        P=float(w@(y[:,j]==1));N=float(w@(y[:,j]==0));U=float(w@(y[:,j]<0));tp=float(w@((y[:,j]==1)&pred[:,j]));fp=float(w@((y[:,j]==0)&pred[:,j]));fn=P-tp;tn=N-fp;up=float(w@((y[:,j]<0)&pred[:,j]));allp=tp+fp+up
        rows.append({'class':str(c),'P':P,'N':N,'U':U,'AP':float(a[j]) if np.isfinite(a[j]) else None,'precision':tp/(tp+fp) if tp+fp else (0. if P else None),'recall':tp/P if P else None,'F1':2*tp/(2*tp+fp+fn) if P else None,'FPR':fp/N if N else None,'TP':tp,'FP':fp,'TN':tn,'FN':fn,'unknown_positive':up,'unknown_alarm_rate':up/U if U else None,'unknown_share_all_positive':up/allp if allp else None})
    return rows

def summary(y,s,pred,classes,background=None,weights=None):
    rows=perclass(y,s,pred,classes,weights);w=np.ones(len(y),float) if weights is None else np.asarray(weights,float)
    def avg(key):
        v=[r[key] for r in rows if r[key] is not None and (key!='precision' or r['P']>0)];return float(np.mean(v)) if v else None
    tp=sum(r['TP'] for r in rows);fp=sum(r['FP'] for r in rows);fn=sum(r['FN'] for r in rows);tn=sum(r['TN'] for r in rows);u=sum(r['U'] for r in rows);up=sum(r['unknown_positive'] for r in rows);total=tp+fp+up
    b0=(y==0).all(1)&(np.ones(len(y),bool) if background is None else background);bd=float(w@b0);bp=float(w@(b0&pred.any(1)))
    microa=aps(y.reshape(-1,1),s.reshape(-1,1),np.repeat(w,y.shape[1]))[0]
    return dict(macro_AP=avg('AP'),macro_F1=avg('F1'),macro_precision=avg('precision'),macro_recall=avg('recall'),macro_FPR=avg('FPR'),micro_AP=float(microa) if np.isfinite(microa) else None,micro_precision=tp/(tp+fp) if tp+fp else 0.,micro_recall=tp/(tp+fn) if tp+fn else None,micro_F1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,micro_FPR=fp/(fp+tn) if fp+tn else None,K_AP=sum(r['AP'] is not None for r in rows),K_F1=sum(r['F1'] is not None for r in rows),TP=tp,FP=fp,TN=tn,FN=fn,unknown_cells=u,unknown_positive=up,unknown_alarm_rate=up/u if u else None,unknown_share_all_positive=up/total if total else None,all_positive=total,all_negative_background_denominator=bd,any_label_FPR=bp/bd if bd else None),rows

import numpy as np
from numba import njit,prange
from metrics import class_sets


@njit(cache=True,parallel=True)
def run_block(counts,group,label,tie_end,pred,n_known,ka,kf):
    R=counts.shape[0];M=group.shape[0];C=group.shape[1]
    result=np.full((R,M,2),np.nan);bad_ap=np.zeros((R,C),np.bool_);bad_f1=np.zeros((R,C),np.bool_)
    for r in prange(R):
        pp=np.zeros(C);nn=np.zeros(C)
        for c in range(C):
            for i in range(n_known[c]):
                w=counts[r,group[0,c,i]]
                if label[0,c,i]:pp[c]+=w
                else:nn[c]+=w
            bad_ap[r,c]=ka[c] and (pp[c]==0 or nn[c]==0)
            bad_f1[r,c]=kf[c] and pp[c]==0
        valid_ap=True;valid_f1=True;nk_ap=0;nk_f1=0
        for c in range(C):
            if bad_ap[r,c]:valid_ap=False
            if bad_f1[r,c]:valid_f1=False
            if ka[c]:nk_ap+=1
            if kf[c]:nk_f1+=1
        for m in range(M):
            ap_total=0.;f1_total=0.
            for c in range(C):
                if not ka[c] and not kf[c]:continue
                ctp=0.;ctotal=0.;gp=0.;gn=0.;ap_numerator=0.;tp=0.;fp=0.
                for i in range(n_known[c]):
                    w=counts[r,group[m,c,i]]
                    if label[m,c,i]:
                        gp+=w
                        if pred[m,c,i]:tp+=w
                    else:
                        gn+=w
                        if pred[m,c,i]:fp+=w
                    if tie_end[m,c,i]:
                        ctp+=gp;ctotal+=gp+gn
                        if gp>0:ap_numerator+=gp*ctp/ctotal
                        gp=0.;gn=0.
                if ka[c] and pp[c]>0:ap_total+=ap_numerator/pp[c]
                if kf[c] and pp[c]>0:f1_total+=2*tp/(pp[c]+tp+fp)
            if valid_ap and nk_ap:result[r,m,0]=ap_total/nk_ap
            if valid_f1 and nk_f1:result[r,m,1]=f1_total/nk_f1
    return result,bad_ap,bad_f1

def pack(y,scores,thresholds,component_index):
    M=len(scores);C=y.shape[1];n_known=(y!=-1).sum(0).astype(np.int64);L=int(n_known.max(initial=0))
    group=np.zeros((M,C,L),np.int32);label=np.zeros((M,C,L),np.bool_);ends=np.zeros((M,C,L),np.bool_);pred=np.zeros((M,C,L),np.bool_)
    for m,s in enumerate(scores):
        for c in range(C):
            ind=np.flatnonzero(y[:,c]!=-1);ind=ind[np.argsort(-s[ind,c],kind='stable')];n=len(ind)
            if not n:continue
            vals=s[ind,c];group[m,c,:n]=component_index[ind];label[m,c,:n]=y[ind,c]==1
            ends[m,c,:n]=np.r_[vals[:-1]!=vals[1:],True];pred[m,c,:n]=vals>=(thresholds[m,c] if thresholds.ndim==2 else thresholds[m])
    kap,kfp=class_sets(y);ka=np.zeros(C,np.bool_);kf=np.zeros(C,np.bool_);ka[kap]=True;kf[kfp]=True
    return group,label,ends,pred,n_known,ka,kf

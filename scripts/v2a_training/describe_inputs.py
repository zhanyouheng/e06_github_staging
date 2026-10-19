from common import *
import numpy as np
from collections import Counter
def main():
    d=np.load(T/'data/development.npz');nt=len(d['train_fold']);ids=d['protein_id'];tokens=json.loads((T/'data/tokens.json').read_text());old={r['ac']:r for r in read_tsv(O/'tail_diagnosis/data/metadata_dev.tsv')};rows=[];support=[]
    for route in ['C','A2B']:
        vocab=set(json.loads((T/'features/full'/(route+'_vocab.json')).read_text())['vocabulary'])
        for scope,idx in [('train',range(nt)),('val',range(nt,len(ids)))]:
            raw=sum(len(tokens[i]) for i in idx);oov=sum(len(set(tokens[i])-vocab) for i in idx)
            rows.append({'annotation_route':'A' if route=='C' else 'A2','scope':scope,'proteins':len(idx),'vocab_dimensions':len(vocab),'raw_protein_token_occurrences':raw,'OOV_occurrences':oov,'OOV_fraction':oov/raw if raw else None,'raw_no_annotation_proteins':sum(not tokens[i] for i in idx),'all_pruned_or_OOV_proteins':sum(bool(tokens[i]) and not(set(tokens[i])&vocab) for i in idx),'nonempty_retained_proteins':sum(bool(set(tokens[i])&vocab) for i in idx),'extraction_status_counts':json.dumps(Counter(old[ids[i]]['extraction_status'] for i in idx),ensure_ascii=False)})
    for j,c in enumerate(d['class_order']):
        for scope,m in [('train',d['scope']=='train'),('val',d['scope']=='val')]:
            yy=d['y'][m,j];gg=d['groups'][m];P=int((yy==1).sum());N=int((yy==0).sum());support.append({'class':str(c),'scope':scope,'proteins':int(m.sum()),'components':len(set(gg)),'P':P,'N':N,'U':int((yy<0).sum()),'positive_components':len(set(gg[yy==1])),'alpha_if_training':min(10,np.sqrt(N/max(P,1))) if scope=='train' else ''})
    tsv(T/'results/annotation_OOV.tsv',rows);tsv(T/'results/label_support.tsv',support)
    folds=[]
    for k in range(5):
        for role,m in [('outer_train',d['train_fold']!=k),('outer_hold',d['train_fold']==k),('early_train',(d['train_fold']!=k)&~d['inner_holdout']),('early_hold',(d['train_fold']!=k)&d['inner_holdout'])]:
            yy=d['y'][:nt][m];folds.append({'outer_fold':k,'role':role,'proteins':int(m.sum()),'components':len(set(d['groups'][:nt][m])),'P':int((yy==1).sum()),'N':int((yy==0).sum()),'U':int((yy<0).sum()),'K_AP':int((((yy==1).sum(0)>0)&((yy==0).sum(0)>0)).sum())})
    tsv(T/'results/fold_support.tsv',folds)
    js(T/'audit/input_descriptives.json',{'at':now(),'source_features':'frozen 2019 GO/keywords plus frozen embedding; M02 only current annotation_score','scope':'old development only','counts':'protein-token incidence, duplicate tokens within protein counted once','test_used':False,'seed':SEED})
if __name__=='__main__':main()

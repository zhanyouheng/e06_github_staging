from common import *
import numpy as np, re, sys, platform
from collections import Counter,defaultdict
from scipy import sparse
from sklearn.preprocessing import normalize
import sklearn,scipy

def counts(y):return {'P':int((y==1).sum()),'N':int((y==0).sum()),'U':int((y==-1).sum())}
def main():
    assert verified(T/'preregistration')
    if verified(T/'preflight'):print('preflight verified');return
    start=time.time();fail=[];checks=[]
    def check(name,ok,detail):
        checks.append({'check':name,'passed':bool(ok),'detail':detail})
        if not ok:fail.append(name)
    d=np.load(O/'tail_diagnosis/data/development.npz',allow_pickle=False)
    ids=np.concatenate([d['train_ids'],d['val_ids']]);nt=len(d['train_ids']);classes=d['class_order'];ix={a:i for i,a in enumerate(ids)};cx={c:j for j,c in enumerate(classes)};allowed=set(ids)
    groups=np.concatenate([d['train_groups'],d['val_groups']]);scope=np.array(['train']*nt+['val']*(len(ids)-nt))
    oldy=np.concatenate([d['train_y'],d['val_y']]);draft=np.load(V/'data/labels_dev_draft.npz',allow_pickle=False)
    new=set(draft['protein_id'][draft['scope']=='new_pending']);draft_ix={a:i for i,a in enumerate(draft['protein_id'])};di=np.array([draft_ix[a] for a in ids])
    check('universe_no_702',len(ids)==25027 and len(allowed)==25027 and not(allowed&new),{'train':nt,'val':len(ids)-nt,'new_excluded':len(new)})
    check('frozen_components_disjoint',not(set(groups[:nt])&set(groups[nt:])),{'train_components':len(set(groups[:nt])),'val_components':len(set(groups[nt:]))})
    y=np.full((len(ids),32),-9,dtype=np.int8);source=np.full(y.shape,'',dtype=object);hits=Counter();sides={}
    rawcounts=Counter()
    # Prefix-only uniqueness scan does not parse non-dev labels.
    with open(N/'positive_labels_full_v2026.tsv') as f:
        next(f)
        for line in f:rawcounts[line.split('\t',1)[0]]+=1
    for fn,side in [('positive_labels_full_v2026.tsv','positive'),('background_labels_final_v4.tsv','background')]:
        for r in projected_tsv(N/fn,allowed):
            a=r['ac'];c=r['class'];i=ix[a];j=cx[c];hits[(a,c)]+=1;sides.setdefault(a,set()).add(side)
            y[i,j]={'pos':1,'neg':0,'unknown':-1}[r['label']]
            source[i,j]=r.get('source','background_v4_'+r['label'])
    posids={a for a,v in sides.items() if v=={'positive'}};bgids={a for a,v in sides.items() if v=={'background'}}
    check('unique_dev_blocks',len(posids|bgids)==len(ids) and len(hits)==len(ids)*32 and all(v==1 for v in hits.values()) and all(rawcounts[a]==32 for a in posids),{'positive_dev':len(posids),'background_dev':len(bgids),'positive_global_unique_32row_ids':sum(v==32 for v in rawcounts.values()),'dev_duplicate_cells':sum(v!=1 for v in hits.values())})
    check('dev_not_in_disputed_blocks',all(rawcounts[a] in (0,32) for a in ids),{'dev_ids_in_64row_block':[a for a in ids if rawcounts[a] not in (0,32)]})
    contaminated=[]
    for r in projected_tsv(N/'final_labels_v2026.tsv',allowed):
        provenance=' '.join(str(r.get(k,'')) for k in ['label_source','blind_engine','user_review']).lower()
        if r['final_status']=='train_candidate' and ('review_2026' in provenance or 'augmented' in provenance):contaminated.append({k:r.get(k,'') for k in ['ac','class','final_status','label_source']})
    badsource=[{'ac':ids[i],'class':classes[j],'source':source[i,j]} for i,j in zip(*np.where(y==1)) if 'review_2026' in source[i,j].lower() or 'augmented' in source[i,j].lower()]
    check('no_B02_positive_provenance',not(contaminated or badsource),{'final_review_contamination':len(contaminated),'matrix_contamination':len(badsource)})
    tsv(T/'preflight/contaminated_cells.tsv',contaminated+badsource,['ac','class','final_status','label_source','source'])
    check('base_matches_turn12_cellwise',np.array_equal(y,draft['before_sameseq_state'][di]),{'unequal_cells':int((y!=draft['before_sameseq_state'][di]).sum())})
    base=y.copy();base_source=source.copy()
    meta={r['ac']:r for r in projected_tsv(V/'data/metadata_dev_draft.tsv',allowed)}
    hashmembers=defaultdict(list)
    for a in ids:hashmembers[meta[a]['sequence_hash']].append(a)
    actions=[]
    for r in read_tsv(V/'data/sameseq_adjudications_dev.tsv'):
        j=cx[r['class']]
        for a in hashmembers.get(r['sequence_hash'],[]):
            i=ix[a];before=int(y[i,j]);out=r['label'];after=1 if out=='pos' else -1 if out=='unknown' else (-1 if before==1 else before)
            if a=='P9WK61' and r['class']=='nonviral_adhesion':after=-1
            y[i,j]=after;source[i,j]='sameseq_'+out+':'+source[i,j]
            actions.append({'ac':a,'class':r['class'],'scope':scope[i],'before':before,'after':after,'changed':before!=after,'label':out})
    sameseq=y.copy();va=cx['virulence_activity'];bg_changed={r['ac'] for r in read_tsv(V/'reconciliation/v3_to_v4_background_dev.tsv')};ruleactions=[]
    for i,a in enumerate(ids):
        mandatory=base_source[i,va]=='excluded_va_mask' or a in bg_changed
        otherP=np.any(np.delete(y[i],va)==1)
        if mandatory or (y[i,va]==1 and otherP):
            before=int(y[i,va]);y[i,va]=-1;source[i,va]='rule_A_mask:'+source[i,va]
            if before!=-1:ruleactions.append({'ac':a,'scope':scope[i],'class':'virulence_activity','before':before,'after':-1,'reason':'rule_A_priority_approved_turn13'})
    if 'P9WK61' in ix:y[ix['P9WK61'],cx['nonviral_adhesion']]=-1
    check('final_matches_turn12_cellwise',np.array_equal(y,draft['state'][di]),{'unequal_cells':int((y!=draft['state'][di]).sum())})
    eligible=(y==-1)&(source=='no_blindcheck')
    check('V1_empty',not eligible.any(),{'eligible_cells':int(eligible.sum())})
    flow=[];changes=[]
    stages=[('frozen_v1_development',oldy),('v2026_proposed_base',base),('after_sameseq',sameseq),('ruleA_v2a',y)]
    for name,z in stages:
        for split,m in [('all',np.ones(len(ids),bool)),('train',scope=='train'),('val',scope=='val')]:flow.append({'stage':name,'split':split,'proteins':int(m.sum()),'cells':int(m.sum()*32),**counts(z[m])})
    for (bn,b),(an,a) in zip(stages[:-1],stages[1:]):
        for i,j in zip(*np.where(a!=b)):
            changes.append({'from_stage':bn,'to_stage':an,'ac':ids[i],'scope':scope[i],'class':classes[j],'before':int(b[i,j]),'after':int(a[i,j]),'source':str(source[i,j] if an=='ruleA_v2a' else base_source[i,j])})
    tsv(T/'data/flow.tsv',flow);tsv(T/'data/cell_changes.tsv',changes);tsv(T/'data/sameseq_actions.tsv',actions);tsv(T/'data/ruleA_priority.tsv',ruleactions)
    conflicts=[]
    for h,aa in hashmembers.items():
        if len(aa)<2:continue
        ii=[ix[a] for a in aa]
        checkgroups=len(set(groups[ii]))==1
        if not checkgroups:fail.append('same_sequence_component_violation')
        for j,c in enumerate(classes):
            pp=[a for a in aa if y[ix[a],j]==1];nn=[a for a in aa if y[ix[a],j]==0]
            if pp and nn:conflicts.append({'sequence_hash':h,'class':c,'P_ids':';'.join(pp),'N_ids':';'.join(nn),'disposition':'retained_unchanged_per_authorization'})
    tsv(T/'data/retained_PN_conflicts.tsv',conflicts)
    rep={g:min(ids[groups==g]) for g in np.unique(groups[:nt])}
    fold=np.array([int(hashlib.sha256((str(SEED)+':'+rep[g]).encode()).hexdigest()[:16],16)%5 for g in groups[:nt]],dtype=np.int8)
    inner=np.array([int(hashlib.sha256((str(SEED)+':early_stop:'+rep[g]).encode()).hexdigest()[:16],16)%10==0 for g in groups[:nt]])
    vf=np.load(V/'preregistration/validation_folds.npz',allow_pickle=False);vi={a:i for i,a in enumerate(vf['protein_id'])};vfold=vf['fold'][[vi[a] for a in ids[nt:]]]
    check('fold_integrity',all(len(set(fold[groups[:nt]==g]))==1 for g in set(groups[:nt])) and all(len(set(vfold[groups[nt:]==g]))==1 for g in set(groups[nt:])),{'train_fold_counts':np.bincount(fold,minlength=5).tolist(),'val_fold_counts':np.bincount(vfold,minlength=5).tolist()})
    check('inner_early_stop_nonempty',all(np.any((fold!=k)&inner) and np.any((fold!=k)&~inner) for k in range(5)),{})
    sn=sorted(set(source.ravel()));si={s:j for j,s in enumerate(sn)};sc=np.array([[si[s] for s in row] for row in source],dtype=np.int16)
    npz(T/'data/development.npz',protein_id=ids,class_order=classes,y=y,old_y=oldy,base_y=base,before_ruleA_y=sameseq,source_code=sc,source_names=np.array(sn),groups=groups,scope=scope,train_fold=fold,inner_holdout=inner,val_fold=vfold,val_annotation_usable=d['val_annotation_usable'],no_blindcheck_eligible=eligible)
    tsv(T/'data/metadata.tsv',[meta[a] for a in ids]);tsv(T/'data/folds.tsv',[{'ac':a,'scope':scope[i],'component_id':groups[i],'outer_fold':int(fold[i] if i<nt else vfold[i-nt]),'early_stop_holdout':bool(inner[i]) if i<nt else ''} for i,a in enumerate(ids)])
    # Frozen feature certificates: read only dev-specific feature arrays.
    cert=[]
    for route in ['A','A2','B']:
        dd=json.loads((O/f'exploration/features/{route}/DONE.json').read_text())
        for rel,h in dd['files'].items():
            p=O/'exploration'/rel;actual=sha(p);cert.append({'file':str(p.relative_to(O)),'expected':h,'actual':actual,'pass':actual==h});check('feature_sha_'+rel,actual==h,{})
    annotations={r['ac']:r for r in read_tsv(O/'exploration/data/annotations_dev.tsv')}
    check('annotations_exact_ids',set(annotations)==allowed and all(re.fullmatch('[0-9a-f]{16}',annotations[a]['seq_sha256']) and annotations[a]['seq_sha256']==meta[a]['sequence_hash'][:16] for a in ids),{'schema':'source seq_sha256 stores 16-character prefix; full 64-character hashes separately verified against frozen embedding index/shards'})
    tokens=[sorted(set(re.findall(r'GO:\d{7}',annotations[a]['go']))|{'KW:'+k.strip() for k in annotations[a]['keywords'].split(';') if k.strip()}) for a in ids]
    js(T/'data/tokens.json',tokens)
    df=Counter(t for row in tokens[:nt] for t in row)
    for route,minimum in [('A',10),('A2',2)]:
        vocab=sorted(t for t,n in df.items() if n>=minimum);lookup={t:i for i,t in enumerate(vocab)};rr=[];cc=[]
        for i,row in enumerate(tokens):
            c=sorted(lookup[t] for t in row if t in lookup);rr.extend([i]*len(c));cc.extend(c)
        a=normalize(sparse.csr_matrix((np.ones(len(rr),np.float32),(rr,cc)),shape=(len(ids),len(vocab))))
        archive=sparse.vstack([sparse.load_npz(O/f'exploration/features/{route}/train.npz'),sparse.load_npz(O/f'exploration/features/{route}/val.npz')]).tocsr();delta=a-archive;err=float(np.max(np.abs(delta.data))) if delta.nnz else 0.
        check('annotation_feature_reconstruction_'+route,a.shape==archive.shape and err<=1e-7,{'shape':list(a.shape),'max_abs':err})
    raw=np.empty((len(ids),1280),np.float32);seen=set();byshard=defaultdict(list)
    for r in projected_tsv(O/'embeddings/index.tsv',allowed,'protein_id'):byshard[r['shard_file']].append(r)
    for shard,rows in sorted(byshard.items()):
        path=O/'embeddings'/shard;h=sha(path);check('embedding_hash_'+shard,h==rows[0]['shard_checksum'],{})
        z=np.load(path,allow_pickle=False)
        for r in rows:
            a=r['protein_id'];j=int(r['row_index']);assert str(z['protein_id'][j])==a and str(z['sequence_hash'][j])==meta[a]['sequence_hash']
            raw[ix[a]]=z['embedding'][j];seen.add(a)
    check('embedding_alignment',seen==allowed and np.isfinite(raw).all(),{'rows':len(seen),'dimensions':1280,'all_finite':bool(np.isfinite(raw).all())});npy(T/'data/embedding_raw.npy',raw)
    pre=np.load(O/'training/features/B_preprocessing.npz',allow_pickle=False)
    print('B_preprocessing keys',pre.files,flush=True)
    mean=raw[:nt].astype(np.float64).mean(0);std=raw[:nt].astype(np.float64).std(0);std[std==0]=1
    b=normalize(((raw.astype(np.float64)-mean)/std).astype(np.float32))
    archive=np.concatenate([np.load(O/'exploration/features/B/train.npy'),np.load(O/'exploration/features/B/val.npy')]);err=float(np.max(np.abs(b-archive)))
    check('B_reconstruction',err<=1e-6,{'max_abs':err})
    tsv(T/'preflight/historical_feature_checksums.tsv',cert)
    old=json.loads((T/'audit/history_before.json').read_text());after=history_snapshot();modified=[k for k,v in old.items() if after.get(k)!=v]
    check('history_readonly',not modified,{'files_lstat_checked':len(old),'modified':modified})
    js(T/'env/python.json',{'at':now(),'executable':sys.executable,'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__})
    summary={'at':now(),'passed':not fail,'failures':fail,'checks':checks,'scope_counts':dict(Counter(scope)),'PNU':counts(y),'V1':'not_applicable','retained_PN_conflict_cells':len(conflicts),'retained_PN_hashes':len({r['sequence_hash'] for r in conflicts}),'ruleA_priority_demotions_dev':len(ruleactions),'sameseq_touched_dev':len(actions),'sameseq_changed_dev':sum(r['changed'] for r in actions),'seconds':time.time()-start,'test_loaded':False}
    js(T/'preflight/result.json',summary)
    if fail:
        print('BLOCKED',fail,flush=True);return
    files=[p for folder in ['data','preflight'] for p in (T/folder).rglob('*') if p.is_file() and p.name!='DONE.json']
    done(T/'preflight',files,passed=True);print(json.dumps({k:v for k,v in summary.items() if k!='checks'},ensure_ascii=False),flush=True)
if __name__=='__main__':main()

from common import *
from metrics import evaluate
from scipy.special import expit
from scipy.stats import spearmanr

def main():
 no_test_guard();d=dev();ann={r['ac']:r for r in rows(E/'data/annotations_dev.tsv')};cl=d['class_order'].tolist();records=[];summary=[];scores={}
 for route in ['A','B','C']:
  x=feature(route,'train')
  for w in ['weighted','unweighted']:
   name=route+'_'+w;sel=readj(BASE/f'selection/{name}.json');C=sel['selected_C'];z=np.load(BASE/f'validation/{name}/C{C:g}.npz',allow_pickle=False);assert np.array_equal(z['protein_id'],d['val_ids']);scores[name]=z['scores']
   with np.load(BASE/f'models/selected/{name}.npz',allow_pickle=False) as m:
    train_score=np.column_stack([np.full(x.shape[0],m['constant'][j]) if m['constant'][j]>=0 else expit(x@m['coef'][j]+m['intercept'][j]) for j in range(32)])
   for split,ss in [('train',train_score),('val',z['scores'])]:
    per,sm=evaluate(d[split+'_y'],ss,sel['threshold'],cl,d[split+'_groups']);summary.append({'model':name,'split':split,**{k:v for k,v in sm.items() if not isinstance(v,list)}})
    records.extend({'model':name,'split':split,**r} for r in per)
 tsv(E/'diagnostics/baseline_per_class.tsv',list(records[0]),records);tsv(E/'diagnostics/baseline_train_val.tsv',list(summary[0]),summary)
 delta=[]
 for w in ['weighted','unweighted']:
  a={r['class_id']:r for r in records if r['model']=='A_'+w and r['split']=='val'};b={r['class_id']:r for r in records if r['model']=='B_'+w and r['split']=='val'}
  for j,c in enumerate(cl):delta.append({'scheme':w,'class_id':c,'train_P':int((d['train_y'][:,j]==1).sum()),'val_P':int((d['val_y'][:,j]==1).sum()),'train_positive_components':len(set(d['train_groups'][d['train_y'][:,j]==1])),
    'val_positive_components':len(set(d['val_groups'][d['val_y'][:,j]==1])),'A_AP':a[c]['AP'],'B_AP':b[c]['AP'],'B_minus_A_AP':b[c]['AP']-a[c]['AP'],'B_better_than_A':b[c]['AP']>a[c]['AP']})
 tsv(E/'diagnostics/B_vs_A_by_class.tsv',list(delta[0]),delta)
 lengths=np.array([int(ann[a]['len']) for a in d['val_ids']]);strata=[('length','<=256',lengths<=256),('length','257-512',(lengths>256)&(lengths<=512)),('length','513-1022',(lengths>512)&(lengths<=1022)),('length','>1022',lengths>1022),
 ('annotation','usable',d['val_annotation_usable']),('annotation','A_zero',~d['val_annotation_usable'])]
 rows_out=[]
 for kind,level,mask in strata:
  for name,ss in scores.items():
   sel=readj(BASE/f'selection/{name}.json');_,sm=evaluate(d['val_y'][mask],ss[mask],sel['threshold'],cl,d['val_groups'][mask]);rows_out.append({'type':kind,'stratum':level,'model':name,**{k:v for k,v in sm.items() if not isinstance(v,list)}})
 tsv(E/'diagnostics/baseline_val_strata.tsv',list(rows_out[0]),rows_out)
 support=[]
 for scheme in ['weighted','unweighted']:
  r=[x for x in delta if x['scheme']==scheme];corr=spearmanr([x['train_P'] for x in r],[x['B_minus_A_AP'] for x in r])
  support.append({'scheme':scheme,'B_better_classes':[x['class_id'] for x in r if x['B_better_than_A']],'spearman_train_P_vs_B_minus_A_AP':float(corr.statistic),'p_value_descriptive_only':float(corr.pvalue)})
 jout(E/'diagnostics/summary.json',{'scope':'train and validation only','source_overlap':'historical function/GO/keyword inputs and later label construction may share functional evidence; measured differences cannot establish causal mechanisms',
   'limited_C_grid':'all baseline routes selected grid endpoint C10 on val; extended regularization is preregistered', 'support_correlations':support,'embedding_pooling':'frozen final-layer residue mean and overlap-averaged windows, no changes this turn','test_scores_read':False})
 log('diagnostics_complete')
if __name__=='__main__':main()

"""Read-only comparison of registered old/new neural seeds, without fitting."""
from access_guard import install
install()
from common import *
import torch

def main():
    rows=[];sources={}
    for feature_name,oldprefix,newprefix in [('C','E05_005_C','E12_002_C'),('A2B','E09_000_A2B','E12_014_A2B')]:
        for seed in [20261007,20261008]:
            oldname=f'{oldprefix}_s{seed}';newname=f'{newprefix}_s{seed}'
            oldbase=OLD/f'gpu/{oldname}';newbase=E/f'models/neural/{newname}'
            oh=readj(oldbase/'history.json');nh=readj(newbase/'history.json')
            with np.load(OLD/f'candidates/{oldname}/validation.npz',allow_pickle=False) as z:
                oldscore=z['scores'];oldids=z['protein_id'];oldclasses=z['class_order']
            with np.load(E/f'candidates/{newname}/validation.npz',allow_pickle=False) as z:
                newscore=z['scores'];assert np.array_equal(oldids,z['protein_id']);assert np.array_equal(oldclasses,z['class_order'])
            oldstate=torch.load(oldbase/'best.pt',map_location='cpu',weights_only=False)['model']
            newstate=torch.load(newbase/'best.pt',map_location='cpu',weights_only=False)['model']
            mapping={'layers.1.weight':'body.1.weight','layers.1.bias':'body.1.bias',
                     'layers.3.weight':'body.3.weight','layers.3.bias':'body.3.bias',
                     'layers.5.weight':'head.weight','layers.5.bias':'head.bias'}
            assert set(oldstate)==set(mapping) and set(newstate)==set(mapping.values())
            diffs={oldkey:float((oldstate[oldkey]-newstate[newkey]).abs().max()) for oldkey,newkey in mapping.items()}
            rows.append(dict(feature=feature_name,seed=seed,old_member=oldname,new_member=newname,
                old_config=oh['config'],new_config=nh['config'],old_best_epoch=oh['best_epoch'],new_best_epoch=nh['best_epoch'],
                old_validation_macro_AP=oh['validation_macro_AP'],new_validation_macro_AP=nh['validation_macro_AP'],
                validation_bitwise_equal=bool(np.array_equal(oldscore,newscore)),
                validation_max_abs_diff=float(np.max(np.abs(oldscore-newscore))),
                learned_weights_bitwise_equal=all(torch.equal(oldstate[k],newstate[v]) for k,v in mapping.items()),
                learned_weights_max_abs_diff=max(diffs.values()),parameter_name_mapping=mapping,
                old_dropout=oh['config']['dropout'],new_dropout=nh['config']['dropout'],
                common_semantics={'hidden':[512],'loss':'masked per-class known-normalized weighted BCE',
                  'ranking_loss_weight':nh['config']['rank_weight'],'learning_rate':.001,'weight_decay':.0001,
                  'batch_size':1024,'max_epochs':120,'validation_checks':'epoch 1 and every 5, patience 6',
                  'training_dtype':'FP32','tf32':False},test_used=False))
            for f in [oldbase/'history.json',newbase/'history.json',oldbase/'best.pt',newbase/'best.pt',
                      OLD/f'candidates/{oldname}/validation.npz',E/f'candidates/{newname}/validation.npz']:
                sources[str(f)]=sha(f)
    aliases=[dict(old=r['old_member'],new=r['new_member']) for r in rows if r['learned_weights_bitwise_equal'] and r['validation_bitwise_equal']]
    result=dict(at=now(),comparisons=rows,identical_model_aliases=aliases,source_sha256=sources,
        frozen_seed_sets={'old':[20261006,20261007,20261008],'new':[20261007,20261008,20261009]},
        independence_statement='New three-seed means are not statistically independent of old means: two seed values overlap; C members with identical learned tensors are duplicate models under different IDs. Distinct IDs and equal-seed repetitions must not be described as independent evidence.',
        graph_count_interpretation='The preregistered complexity counts unique model IDs, not independent random experiments. Identical aliases are disclosed; no seeds, models, thresholds or selections are changed by this audit.',
        A2B_note='Old A2B family uses dropout 0.2; new A2B family uses dropout 0.5. Shared seed values do not imply the same fitted model.',
        training_performed=False,test_used=False)
    jout(E/'audit/neural_seed_overlap.json',result)
    log('neural_seed_overlap_audit_complete',identical_model_aliases=aliases)

if __name__=='__main__':main()

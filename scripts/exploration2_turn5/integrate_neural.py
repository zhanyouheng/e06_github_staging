"""CPU-only E12 candidate registration and predetermined seed means."""
from common import *
from registry import register_candidate
from sklearn.metrics import average_precision_score
import argparse


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('mode', choices=['initial','extra']); args=parser.parse_args()
    no_test_guard(); d=dev(); available=[]
    for cp in sorted((E/'models/neural').glob('*/DONE.json')):
        assert verified(cp)
        meta=readj(cp.parent/'history.json'); cfg=meta['config']; name=cfg['name']
        with np.load(cp.parent/'validation.npz', allow_pickle=False) as z:
            assert np.array_equal(z['protein_id'], d['val_ids'])
            assert np.array_equal(z['class_order'], d['class_order'])
            score=z['scores'].astype(np.float64)
        aps=[]
        for j in range(32):
            known=d['val_y'][:,j]!=-1
            y=d['val_y'][known,j]
            if np.any(y==1) and np.any(y==0):
                aps.append(average_precision_score(y,score[known,j]))
        assert abs(np.mean(aps)-meta['validation_macro_AP'])<1e-12
        files=[cp.parent/f for f in ['best.pt','validation.npz','train_scores.npz','history.json','DONE.json']]
        if not verified(E/f'candidates/{name}/DONE.json'):
            register_candidate(name,cfg,score,files,dict(selection_eligible=False,
                reason='single-seed architecture/seed diagnostic; final E12 candidates require all three fixed seeds',
                fit_seconds=meta['seconds'],independent_val_bitwise_equal=meta['independent_val_bitwise_equal']))
        available.append(meta)
    if args.mode=='initial':
        initial=readj(E/'configurations/E12_initial.json')
        assert len(initial)==24
        complete={m['config']['name'] for m in available}
        missing=[cfg['name'] for cfg in initial if cfg['name'] not in complete]
        assert not missing, missing
        selected={}; configs=[]
        for f in ['C','A2B']:
            records=[m for m in available if m['config']['feature']==f and m['seed']==SEED]
            max_ap=max(m['validation_macro_AP'] for m in records)
            tied=[m for m in records if m['validation_macro_AP']>=max_ap-1e-6]
            chosen=min(tied,key=lambda m:m['config']['order'])
            selected[f]=dict(first_seed_winner=chosen['config']['name'], architecture=chosen['config'],
                             validation_macro_AP=chosen['validation_macro_AP'], tie_policy='within 1e-6 lower preregistered order',
                             selected_using='train and validation only', seeds=[SEED,SEED+1,SEED+2])
            for seed in [SEED+1,SEED+2]:
                configs.append({**chosen['config'],'name':chosen['config']['name'].replace(f'_s{SEED}',f'_s{seed}'),
                                'seed':seed,'seeds':[seed]})
        jout(E/'selection/E12_architectures.json',selected)
        jout(E/'configurations/E12_extra_seeds.json',configs)
    else:
        selected=readj(E/'selection/E12_architectures.json')
        for f in ['C','A2B']:
            base=selected[f]['first_seed_winner']
            members=[base.replace(f'_s{SEED}',f'_s{s}') for s in [SEED,SEED+1,SEED+2]]
            scores=[]; trains=[]; files=[]
            for name in members:
                p=E/f'models/neural/{name}'
                assert verified(p/'DONE.json') and verified(E/f'candidates/{name}/DONE.json')
                files.extend([p/'DONE.json',E/f'candidates/{name}/DONE.json'])
                with np.load(p/'validation.npz',allow_pickle=False) as z:scores.append(z['scores'].astype(np.float64))
                with np.load(p/'train_scores.npz',allow_pickle=False) as z:
                    assert np.array_equal(z['protein_id'],d['train_ids'])
                    trains.append(z['scores'].astype(np.float64))
            name=f'E12_ensemble_{f}'
            score=np.mean(np.stack(scores),axis=0)
            train_score=np.mean(np.stack(trains),axis=0)
            p=E/f'models/neural/{name}'
            npz(p/'train_scores.npz',protein_id=d['train_ids'],class_order=d['class_order'],scores=train_score)
            files.append(p/'train_scores.npz')
            cfg=dict(name=name,stage='E12',algorithm='ensemble',members=members,weights=[1/3]*3,
                     feature=f,seed=SEED,seeds=[SEED,SEED+1,SEED+2],order=100 if f=='C' else 101,
                     aggregation='np.mean(np.stack(member_float64_scores),axis=0)')
            if not verified(E/f'candidates/{name}/DONE.json'):
                summary=register_candidate(name,cfg,score,files,dict(selection_eligible=True,fixed_seed_mean=True,
                    primitive_members=members,train_scores=str((p/'train_scores.npz').relative_to(E)),
                    architecture_choice=selected[f],independent_val_bitwise_equal=True))
    log('neural_integration_complete',mode=args.mode)


if __name__=='__main__':main()

from common import *

def main():
    no_test_guard()
    d=dev(); initial=readj(E/'configurations/E12_initial.json'); extra=readj(E/'configurations/E12_extra_seeds.json')
    records=[]; files=[]
    for cfg in initial+extra:
        p=E/f"models/neural/{cfg['name']}"
        assert verified(p/'DONE.json') and verified(E/f"candidates/{cfg['name']}/DONE.json")
        m=readj(p/'history.json')
        assert m['independent_val_bitwise_equal'] and not m['test_used']
        assert m['training_support']['P']==(d['train_y']==1).sum(0).tolist()
        assert m['training_support']['N']==(d['train_y']==0).sum(0).tolist()
        assert m['training_support']['U']==(d['train_y']==-1).sum(0).tolist()
        for cp in (p/'checkpoints').glob('*.DONE.json'):assert verified(cp)
        row=dict(name=cfg['name'],feature=cfg['feature'],output=cfg['output'],dropout=cfg['dropout'],
                 rank_weight=cfg['rank_weight'],seed=cfg['seed'],validation_macro_AP=m['validation_macro_AP'],
                 best_epoch=m['best_epoch'],completed_epochs=len(m['history']),fit_seconds=m['seconds'],
                 peak_allocated_bytes=m['peak_allocated_bytes'],selection_eligible=False,status='complete')
        records.append(row);files.extend([p/'DONE.json',E/f"candidates/{cfg['name']}/DONE.json"])
    tsv(E/'diagnostics/E12_results.tsv',list(records[0]),records)
    comparisons=[]
    for f,old in [('C','E05_ensemble_C'),('A2B','E09_ensemble_A2B')]:
        new=readj(E/f'candidates/E12_ensemble_{f}/summary.json')
        previous=readj(OLD/f'candidates/{old}/summary.json')
        delta=new['validation_macro_AP']-previous['validation_macro_AP']
        comparisons.append(dict(feature=f,new_candidate=new['name'],old_candidate=old,
            new_validation_macro_AP=new['validation_macro_AP'],old_validation_macro_AP=previous['validation_macro_AP'],
            delta=delta,substantial_improvement_threshold=0.003,substantial_improvement_met=delta>=0.003,
            fixed_three_seeds=[SEED,SEED+1,SEED+2],selection_eligible=True))
        files.append(E/f'candidates/E12_ensemble_{f}/DONE.json')
    # Paired first-seed architecture effects; no additional candidates or model selection.
    lookup={(r['feature'],r['output'],r['dropout'],r['rank_weight']):r for r in records if r['seed']==SEED}
    effects=[]
    for f in ['C','A2B']:
        for output in ['dense','rank8','rank16']:
            for dropout in [.2,.5]:
                plain=lookup[f,output,dropout,0.]
                rank=lookup[f,output,dropout,.1]
                effects.append(dict(feature=f,output=output,dropout=dropout,
                    BCE_AP=plain['validation_macro_AP'],BCE_plus_ranking_AP=rank['validation_macro_AP'],
                    ranking_delta=rank['validation_macro_AP']-plain['validation_macro_AP']))
    tsv(E/'diagnostics/E12_ranking_effects.tsv',list(effects[0]),effects)
    m=dict(at=now(),stage='E12',initial_configurations=24,extra_seed_configurations=4,completed_fits=28,
           eligible_candidates=2,failed_fits=0,comparisons=comparisons,
           summed_fit_seconds=sum(r['fit_seconds'] for r in records),
           total_epochs=sum(r['completed_epochs'] for r in records),
           architectures=readj(E/'selection/E12_architectures.json'),
           negative_results='rank8/rank16 and the additional pairwise loss did not win either input architecture selection; neither seed mean exceeded its corresponding historical mean by the preregistered +0.003 threshold',
           independent_val_recompute_bitwise_equal_fits=28,
           full_checkpoint_next_batch_recovery_check='validation/E12_resume_check.json',
           unknown_mask='all losses and AP exclude unknown; pair sampling uses only known positive-negative pairs in the same class',
           training_support_PNU={'P':int((d['train_y']==1).sum()),'N':int((d['train_y']==0).sum()),'U':int((d['train_y']==-1).sum())},
           test_used=False)
    jout(E/'diagnostics/E12_summary.json',m)
    files += [E/'diagnostics/E12_results.tsv',E/'diagnostics/E12_ranking_effects.tsv',E/'diagnostics/E12_summary.json',
              E/'validation/E12_unit_checks.json',E/'validation/E12_input_verification.json',E/'validation/E12_resume_check.json',
              E/'configurations/E12_initial.json',E/'configurations/E12_extra_seeds.json',E/'selection/E12_architectures.json']
    certificate(E/'models/neural/DONE.json',files,dict(stage='E12',plan_sha256=sha(E/'PLAN.md'),seed=SEED),
                dict(completed_fits=28,eligible_candidates=2,test_used=False))
    log('neural_branch_verified_complete',completed_fits=28,eligible_candidates=2)

if __name__=='__main__':main()

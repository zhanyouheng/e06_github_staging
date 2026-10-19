from common import *
from inference_v2 import verify_final
def main():
 selection=verify_final();assert readj(E/'validation/frozen_model_recompute_v2.json')['passed'];assert readj(E/'validation/synthetic_tests.json')['passed']
 assert verified(E/'inference_v2/val/final_predictions.DONE.json')
 context={'selection_sha256':sha(E/'selection/final_selection.json'),'runtime_supplement_sha256':sha(E/'selection/inference_runtime_v2.json'),'scripts':{n:sha(S/n) for n in ['inference.py','inference_v2.py','infer_gpu.py','evaluate_revision.py','bootstrap_revision.py','bootstrap_core.py','metrics.py']}}
 p=E/'revision/revision_evaluation_event.json'
 if p.exists():assert readj(p)['context']==context;log('revision_event_resume_verified');return
 event={'event_type':'修订评价事件','event_id':'turn4_revision_1','overall_test_view_number':2,'new_selected_schemes':1,'registered_at_utc':now(),'registered_at_local':datetime.now().astimezone().isoformat(),
   'reason':'用户授权的探索阶段已结束；全部选择只使用train/val，现对唯一冻结方案进行一次新的完整测试评价。',
   'chosen_candidate':selection['chosen_candidate'],'validation_macro_AP':selection['validation_macro_AP'],'validation_macro_F1':selection['validation_macro_F1'],
   'relationship_to_main_experiment':'Turn3 A/B/C主实验已完成并由用户复核。该测试集已被查看，本次是第二次评价，不能称首次盲测。新方法属于探索优化，不能替代原表示对照的预注册结论。',
   'main_experiment_manifest_sha256':sha(BASE/'manifest.json'),'model_frozen_at_utc':selection['frozen_at_utc'],'context':context,
   'selection_test_use':False,'post_test_tuning_forbidden':True,'primary_comparison':'selected minus archived C_weighted macro AP',
   'supplementary_comparisons':'selected minus six archived A/B/C weighted/unweighted methods; F1 threshold policies differ',
   'bootstrap':'reuse frozen Turn3 2000 component draws, seed20261008; fixed K_AP/K_F1; main CI NA if any invalid replicate',
   'evaluation_subsets':['full_test','usable_annotations_defined_by_main_train_vocabulary'],'threshold_policy':'frozen per-class thresholds only; no test threshold selection'}
 jout(p,event);p.chmod(0o444);log('REVISION_EVENT_REGISTERED',event_id=event['event_id'],test_view_number=2,candidate=event['chosen_candidate'])
if __name__=='__main__':main()

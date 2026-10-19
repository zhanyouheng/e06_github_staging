"""E11 train/val-only tree exploration, per-class atomic resume checkpoints."""
from common import *
from registry import register_candidate
from scipy import sparse
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse, multiprocessing as mp, traceback, warnings

FEATURES = {}
D = None
CONTEXT = None
SEEDS = [20261007, 20261008, 20261009]


def configurations():
    cfgs = []
    for feature_name in ['C', 'A2B', 'A5B']:
        for max_features in [0.05, 0.2, 0.5]:
            for leaf in [1, 3]:
                cfgs.append(dict(algorithm='extra_trees', feature=feature_name,
                                 max_features=max_features, min_samples_leaf=leaf,
                                 weight='weighted'))
        for leaf in [1, 3]:
            cfgs.append(dict(algorithm='extra_trees', feature=feature_name,
                             max_features=0.2, min_samples_leaf=leaf,
                             weight='unweighted'))
    for feature_name in ['C', 'A2B', 'A5B']:
        for max_features in ['sqrt', 0.2]:
            cfgs.append(dict(algorithm='random_forest', feature=feature_name,
                             max_features=max_features, min_samples_leaf=1,
                             weight='weighted'))
    for i, cfg in enumerate(cfgs):
        cfg.update(stage='E11', order=i, n_estimators=256, seed=SEEDS[0],
                   name=f'E11_{i:03d}_{cfg["algorithm"]}_{cfg["feature"]}_s{SEEDS[0]}')
    assert len(cfgs) == 30
    return cfgs


def context(cfg, j):
    return dict(CONTEXT, config=cfg, class_index=j)


def predict(model, x):
    if isinstance(model, dict):
        return np.full(x.shape[0], model['constant'], dtype=np.float64)
    return model.predict_proba(x)[:, list(model.classes_).index(1)]


def fit_one(task):
    cfg, j = task
    p = E / 'models/trees' / cfg['name'] / f'class_{j:02d}'
    cert = p.with_suffix('.DONE.json')
    ct = context(cfg, j)
    if verified(cert, ct):
        return readj(p.with_suffix('.json'))
    start = time.monotonic()
    y = D['train_y'][:, j]
    known = y != -1
    z = y[known]
    positive = int((z == 1).sum())
    negative = int((z == 0).sum())
    support = ('normal' if positive and negative else 'no_known_labels' if not positive + negative
               else 'no_positive' if not positive else 'no_negative')
    alpha = min(10., float(np.sqrt(negative / positive))) if positive and negative and cfg['weight'] == 'weighted' else 1.
    r = dict(name=cfg['name'], class_index=j, class_name=str(D['class_order'][j]),
             P=positive, N=negative, U=int((y == -1).sum()), alpha=alpha,
             training_support_status=support, pid=os.getpid(), seed=cfg['seed'] + j,
             warnings=[], retries=[], started_at=now())
    jout(p.with_suffix('.RUNNING.json'), r)
    try:
        x, val = FEATURES[cfg['feature']]
        if support == 'normal':
            cls = ExtraTreesClassifier if cfg['algorithm'] == 'extra_trees' else RandomForestClassifier
            model = cls(n_estimators=256, criterion='gini', max_depth=None,
                        min_samples_split=2, min_samples_leaf=cfg['min_samples_leaf'],
                        max_features=cfg['max_features'], n_jobs=1,
                        bootstrap=cfg['algorithm'] == 'random_forest',
                        class_weight=None, random_state=cfg['seed'] + j)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                model.fit(x[known], z, sample_weight=np.where(z == 1, alpha, 1.))
            r['warnings'] = [str(w.message) for w in caught]
            if caught:
                raise RuntimeError('unexpected fit warnings: ' + repr(r['warnings']))
        else:
            model = {'constant': float(positive > 0)}
        score = predict(model, val)
        assert score.shape == (len(D['val_ids']),) and np.isfinite(score).all()
        assert np.all((score >= 0) & (score <= 1))
        mf = p.with_suffix('.pkl')
        sf = p.with_name(p.name + '_val.npy')
        pkl(mf, model)
        npy(sf, score)
        r.update(success=True, seconds=time.monotonic() - start, completed_at=now(),
                 model_files=[str(mf.relative_to(E)), str(sf.relative_to(E))])
        jout(p.with_suffix('.json'), r)
        certificate(cert, [mf, sf, p.with_suffix('.json')], ct)
        log('tree_class_complete', name=cfg['name'], class_index=j, seconds=r['seconds'])
        return r
    except Exception as exc:
        r.update(success=False, seconds=time.monotonic() - start, failed_at=now(),
                 error=repr(exc), traceback=traceback.format_exc())
        # Timestamped failures are retained even when a later resume succeeds.
        fail = p.with_name(p.name + f'.FAILURE.{os.getpid()}.{time.time_ns()}.json')
        jout(fail, r)
        log('tree_class_failure', name=cfg['name'], class_index=j, error=repr(exc))
        return r


def register_single(cfg):
    parent = E / 'models/trees' / cfg['name']
    files = []
    rr = []
    for j in range(32):
        p = parent / f'class_{j:02d}'
        if not verified(p.with_suffix('.DONE.json'), context(cfg, j)):
            return None
        rr.append(readj(p.with_suffix('.json')))
        files.extend([p.with_suffix('.pkl'), p.with_suffix('.json'), p.with_suffix('.DONE.json'),
                      p.with_name(p.name + '_val.npy')])
    score = np.column_stack([np.load(parent / f'class_{j:02d}_val.npy', allow_pickle=False) for j in range(32)])
    return register_candidate(cfg['name'], cfg, score, files,
                              {'selection_eligible': False,
                               'eligibility_reason': 'single-seed diagnostic; only complete fixed three-seed ensemble can be selected',
                               'class_fit_seconds_sum': float(sum(r['seconds'] for r in rr)),
                               'fit_class_count': 32, 'primitive_members': 1})


def run_configurations(configs, workers):
    result = {cfg['name']: [] for cfg in configs}
    # Fork inherits immutable dense feature arrays; individual tasks subset known labels.
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('fork')) as pool:
        futures = {pool.submit(fit_one, (cfg, j)): (cfg['name'], j) for cfg in configs for j in range(32)}
        for f in as_completed(futures):
            name, j = futures[f]
            try:
                r = f.result()
            except Exception as exc:
                r = dict(name=name, class_index=j, success=False, error=repr(exc),
                         traceback=traceback.format_exc(), at=now())
                jout(E / 'models/trees' / name / f'pool_failure_class_{j:02d}_{time.time_ns()}.json', r)
            result[name].append(r)
            if len(result[name]) == 32:
                if all(r['success'] for r in result[name]):
                    register_single(next(c for c in configs if c['name'] == name))
                else:
                    jout(E / 'models/trees' / name / 'INCOMPLETE.json', result[name])
    return result


def verify_inference_one(task):
    cfg, j = task
    p = E / 'models/trees' / cfg['name'] / f'class_{j:02d}'
    ct = dict(context(cfg, j), purpose='independent reload val equality and full train scores')
    cert = p.with_suffix('.INFERENCE_DONE.json')
    if verified(cert, ct):
        return readj(p.with_suffix('.INFERENCE.json'))
    assert verified(p.with_suffix('.DONE.json'), context(cfg, j))
    start = time.monotonic()
    with open(p.with_suffix('.pkl'), 'rb') as f:
        model = pickle.load(f)
    x, val = FEATURES[cfg['feature']]
    recomputed = predict(model, val)
    stored = np.load(p.with_name(p.name + '_val.npy'), allow_pickle=False)
    assert np.array_equal(stored, recomputed), (cfg['name'], j, np.max(np.abs(recomputed - stored)))
    train = predict(model, x)
    assert np.isfinite(train).all() and train.shape == (len(D['train_ids']),)
    f = p.with_name(p.name + '_train.npy')
    npy(f, train)
    r = dict(name=cfg['name'], class_index=j, val_max_abs_diff=0., val_exact=True,
             train_rows=len(train), seconds=time.monotonic() - start, at=now(), pid=os.getpid())
    jout(p.with_suffix('.INFERENCE.json'), r)
    certificate(cert, [f, p.with_suffix('.INFERENCE.json')], ct)
    return r


def finish_ensemble(algorithm, configs, workers):
    assert len(configs) == 3 and [c['seed'] for c in configs] == SEEDS
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('fork')) as pool:
        rr = list(pool.map(verify_inference_one, [(cfg, j) for cfg in configs for j in range(32)]))
    val = []
    train = []
    files = []
    for cfg in configs:
        p = E / 'models/trees' / cfg['name']
        val.append(np.column_stack([np.load(p / f'class_{j:02d}_val.npy', allow_pickle=False) for j in range(32)]))
        train.append(np.column_stack([np.load(p / f'class_{j:02d}_train.npy', allow_pickle=False) for j in range(32)]))
        for j in range(32):
            for suffix in ['.pkl', '.json', '.DONE.json', '_val.npy', '_train.npy', '.INFERENCE.json', '.INFERENCE_DONE.json']:
                files.append(p / f'class_{j:02d}{suffix}')
    vs = np.mean(np.stack(val), axis=0)
    ts = np.mean(np.stack(train), axis=0)
    name = f'E11_ensemble_{algorithm}'
    cfg = dict(name=name, stage='E11', algorithm='fixed_seed_mean', family=algorithm,
               feature=configs[0]['feature'], seeds=SEEDS,
               members=[c['name'] for c in configs], member_configs=configs,
               weights=[1/3, 1/3, 1/3], seed_selection='none; fixed three seeds', order=30 if algorithm == 'extra_trees' else 31)
    target = E / 'models/trees' / name
    npz(target / 'train_scores.npz', protein_id=D['train_ids'], class_order=D['class_order'], scores=ts)
    jout(target / 'independent_recompute.json', dict(passed=True, max_abs_diff=0., classes=rr,
                                                    sequence_splits=['train', 'val'], test_used=False))
    files.extend([target / 'train_scores.npz', target / 'independent_recompute.json'])
    return register_candidate(name, cfg, vs, files,
                              dict(selection_eligible=True, eligibility_reason='complete three-seed family mean',
                                   primitive_members=3, train_scores_path=str((target/'train_scores.npz').relative_to(E)),
                                   independent_val_max_abs_diff=0.))


def main():
    global FEATURES, D, CONTEXT
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=40)
    args = parser.parse_args()
    assert 1 <= args.workers <= 40
    no_test_guard()
    started = time.monotonic()
    D = dev()
    assert D['train_y'].shape == (18770, 32) and D['val_y'].shape == (6257, 32)
    features = {}
    for name in ['C', 'A2B', 'A5B']:
        arrays = []
        features[name] = {}
        for split in ['train', 'val']:
            x = feature(name, split)
            x = x.toarray() if sparse.issparse(x) else np.asarray(x)
            x = np.ascontiguousarray(x, dtype=np.float32)
            assert np.isfinite(x).all()
            x.flags.writeable = False
            arrays.append(x)
            source = OLD / 'features' / name / (split + '.npz')
            if not source.exists():
                source = source.with_suffix('.npy')
            features[name][split] = {'path': str(source), 'sha256': sha(source), 'shape': list(x.shape)}
        FEATURES[name] = tuple(arrays)
    CONTEXT = dict(code_sha256=sha(S/'trees.py'), dev_sha256=sha(E/'data/dev.npz'),
                   input_features=features, model_input_dtype='float32')
    configs = configurations()
    jout(E/'configurations/E11_initial.json', configs)
    log('tree_stage_start', initial_configurations=len(configs), workers=args.workers, context=CONTEXT)
    result = run_configurations(configs, args.workers)
    initial = []
    incomplete = []
    for cfg in configs:
        path = E/'candidates'/cfg['name']/'summary.json'
        if all(r['success'] for r in result[cfg['name']]) and path.exists():
            initial.append(readj(path))
        else:
            incomplete.append(cfg['name'])
    if incomplete:
        jout(E/'models/trees/INCOMPLETE.json', {'initial_incomplete': incomplete, 'at': now(),
                                               'reason': 'all architecture candidates must complete before family winner selection'})
        raise RuntimeError(f'E11 incomplete candidates: {incomplete}')
    winners = {}
    extras = []
    selected_cfgs = {}
    for algorithm in ['extra_trees', 'random_forest']:
        subset = [s for s in initial if s['config']['algorithm'] == algorithm]
        best_ap = max(s['validation_macro_AP'] for s in subset)
        best = min([s for s in subset if best_ap-s['validation_macro_AP'] <= 1e-6], key=lambda s:s['config']['order'])
        winners[algorithm] = dict(name=best['name'], validation_macro_AP=best['validation_macro_AP'],
                                  tie_rule='within 1e-6 lower predeclared order', seed=SEEDS[0])
        base_cfg = best['config']
        selected_cfgs[algorithm] = [base_cfg]
        for seed in SEEDS[1:]:
            cfg = dict(base_cfg, seed=seed, name=base_cfg['name'].rsplit('_s', 1)[0] + f'_s{seed}')
            extras.append(cfg)
            selected_cfgs[algorithm].append(cfg)
    jout(E/'configurations/E11_family_architecture_selection.json', {'at': now(), 'winners': winners, 'test_used': False})
    jout(E/'configurations/E11_extra_seeds.json', extras)
    result_extra = run_configurations(extras, args.workers)
    assert all(r['success'] for rs in result_extra.values() for r in rs), 'incomplete extra seed fits; resume same command'
    ensembles = [finish_ensemble(a, selected_cfgs[a], args.workers) for a in ['extra_trees', 'random_forest']]
    summary = {'at': now(), 'seconds': time.monotonic()-started, 'initial_count': len(configs),
               'extra_seed_count': len(extras), 'total_class_fits': (len(configs)+len(extras))*32,
               'winners': winners, 'ensembles': [{'name':s['name'], 'validation_macro_AP':s['validation_macro_AP']} for s in ensembles],
               'failures': len(list((E/'models/trees').glob('**/*.FAILURE.*.json'))), 'test_used': False}
    jout(E/'models/trees/E11_summary.json', summary)
    certificate(E/'models/trees/DONE.json', [E/'models/trees/E11_summary.json'] +
                [E/'candidates'/s['name']/'DONE.json' for s in ensembles], CONTEXT)
    log('tree_stage_complete', **summary)


if __name__ == '__main__':
    main()

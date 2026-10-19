"""E12 masked BCE/ranking MLPs; train/val only, transactional epoch recovery."""
from common import *
import argparse, random, traceback
import torch
from torch import nn


class Network(nn.Module):
    def __init__(self, dimensions, output, dropout):
        super().__init__()
        self.body = nn.Sequential(nn.LayerNorm(dimensions, elementwise_affine=False),
                                  nn.Linear(dimensions, 512), nn.GELU(), nn.LayerNorm(512), nn.Dropout(dropout))
        self.head = nn.Linear(512, 32) if output == 'dense' else nn.Sequential(
            nn.Linear(512, int(output.removeprefix('rank')), bias=False),
            nn.Linear(int(output.removeprefix('rank')), 32))
    def forward(self, x):
        return self.head(self.body(x))


def ap_numpy(y, score):
    out = []
    for j in range(32):
        known = y[:, j] != -1
        target, values = y[known, j], score[known, j]
        P, N = int((target == 1).sum()), int((target == 0).sum())
        if P == 0 or N == 0:
            continue
        order = np.argsort(-values, kind='stable')
        target, values = target[order], values[order]
        ends = np.r_[np.flatnonzero(values[:-1] != values[1:]), len(values)-1]
        tp = np.cumsum(target == 1)[ends]
        out.append(float(np.sum(np.diff(np.r_[0, tp]) / P * tp / (ends+1))))
    return float(np.mean(out))


def save_torch(path, state):
    q = tmp(path)
    torch.save(state, q)
    z = torch.load(q, map_location='cpu', weights_only=False)
    assert z['epoch'] == state['epoch']
    assert set(z['model']) == set(state['model'])
    for key in z['model']:
        assert torch.equal(z['model'][key], state['model'][key].cpu())
    commit(q, path)


def configurations():
    configs = []
    for feature_name in ['C', 'A2B']:
        for output in ['dense', 'rank8', 'rank16']:
            for dropout in [0.2, 0.5]:
                for rank_weight in [0., 0.1]:
                    i = len(configs)
                    configs.append(dict(name=f'E12_{i:03d}_{feature_name}_s{SEED}', stage='E12',
                        algorithm='masked_mlp', feature=feature_name, hidden=[512], output=output,
                        dropout=dropout, rank_weight=rank_weight, max_pairs_per_class=64,
                        seed=SEED, seeds=[SEED], weight='weighted', order=i))
    return configs


def predict(net, x, P, N):
    net.eval()
    with torch.inference_mode():
        score = torch.cat([net(x[b:b+2048]).sigmoid().cpu() for b in range(0, len(x), 2048)]).numpy()
    score[:, P == 0] = 0.
    score[:, (N == 0) & (P > 0)] = 1.
    assert np.isfinite(score).all()
    return score


def pairwise(logits, labels, generator):
    pos_rows, neg_rows, classes, weights = [], [], [], []
    eligible = 0
    for j in range(32):
        pos = np.flatnonzero(labels[:, j] == 1)
        neg = np.flatnonzero(labels[:, j] == 0)
        if not len(pos) or not len(neg):
            continue
        eligible += 1
        pairs = min(64, len(pos)*len(neg))
        p = generator.choice(pos, size=pairs, replace=True)
        n = generator.choice(neg, size=pairs, replace=True)
        assert np.all(labels[p, j] == 1) and np.all(labels[n, j] == 0)
        pos_rows.extend(p.tolist()); neg_rows.extend(n.tolist())
        classes.extend([j]*pairs); weights.extend([1./pairs]*pairs)
    if not eligible:
        return logits.sum()*0., 0
    device = logits.device
    p = torch.as_tensor(pos_rows, dtype=torch.long, device=device)
    n = torch.as_tensor(neg_rows, dtype=torch.long, device=device)
    c = torch.as_tensor(classes, dtype=torch.long, device=device)
    w = torch.as_tensor(weights, dtype=torch.float32, device=device)
    loss = (nn.functional.softplus(logits[n, c]-logits[p, c])*w).sum()/eligible
    return loss, len(classes)


def configure(seed, gpu):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(f'cuda:{gpu}')
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    return device


def run(cfg, gpu):
    no_test_guard()
    name = cfg['name']; p = E/'models'/'neural'/name
    context = dict(config=cfg, code_sha256=sha(S/'neural.py'), dev_sha256=sha(E/'data/dev.npz'),
                   train_input_sha256=sha(OLD/f"features/{cfg['feature']}/train_gpu.npy"),
                   val_input_sha256=sha(OLD/f"features/{cfg['feature']}/val_gpu.npy"))
    if verified(p/'DONE.json', context):
        log('neural_resume_verified', name=name)
        return
    device = configure(cfg['seed'], gpu)
    d = dev(); y, vy = d['train_y'], d['val_y']
    P, N = (y == 1).sum(0), (y == 0).sum(0)
    x = np.load(OLD/f"features/{cfg['feature']}/train_gpu.npy", allow_pickle=False)
    v = np.load(OLD/f"features/{cfg['feature']}/val_gpu.npy", allow_pickle=False)
    assert x.dtype == v.dtype == np.float32 and np.isfinite(x).all() and np.isfinite(v).all()
    xt, vt = torch.as_tensor(x, device=device), torch.as_tensor(v, device=device)
    yt = torch.as_tensor((y == 1).astype(np.float32), device=device)
    mask = torch.as_tensor((y != -1).astype(np.float32), device=device)
    alpha = np.minimum(10., np.sqrt(N/np.maximum(P, 1))).astype(np.float32)
    pw = torch.as_tensor(alpha, device=device); denom = mask.sum(0).clamp_min(1)
    net = Network(x.shape[1], cfg['output'], cfg['dropout']).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=0.001, weight_decay=0.0001)
    gen = torch.Generator(device='cpu').manual_seed(cfg['seed'])
    pairgen = np.random.default_rng(cfg['seed'])
    start_epoch, history, best, bad, best_epoch = 0, [], -1., 0, 0
    best_model, best_scores = None, None
    start, prior_elapsed, resumes = time.monotonic(), 0., []
    checkpoints = sorted((p/'checkpoints').glob('epoch_*.DONE.json'), reverse=True)
    for cp in checkpoints:
        if verified(cp, context):
            path = cp.with_name(cp.name.replace('.DONE.json', '.pt'))
            state = torch.load(path, map_location='cpu', weights_only=False)
            net.load_state_dict(state['model']); opt.load_state_dict(state['optimizer'])
            gen.set_state(state['generator_state']); pairgen.bit_generator.state = state['pair_generator_state']
            torch.set_rng_state(state['cpu_rng']); torch.cuda.set_rng_state(state['cuda_rng'], device)
            random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng'])
            start_epoch, history = state['epoch'], state['history']
            best, bad, best_epoch = state['best'], state['bad'], state['best_epoch']
            best_model, best_scores = state['best_model'], state['best_scores']
            prior_elapsed = state['elapsed_seconds']
            resumes = state.get('resumes', []) + [dict(at=now(), from_epoch=start_epoch, pid=os.getpid())]
            log('neural_epoch_resumed', name=name, epoch=start_epoch)
            break
    for epoch in range(start_epoch+1, 121):
        if bad >= 6:
            break
        net.train(); order = torch.randperm(len(x), generator=gen)
        loss_sum, bce_sum, rank_sum, pairs_total = 0., 0., 0., 0
        for begin in range(0, len(x), 1024):
            cpu_idx = order[begin:begin+1024]
            idx = cpu_idx.to(device)
            opt.zero_grad(set_to_none=True)
            logits = net(xt[idx])
            raw = nn.functional.binary_cross_entropy_with_logits(logits, yt[idx], pos_weight=pw, reduction='none')
            bce = ((raw*mask[idx]).sum(0)/denom*(len(x)/len(idx))).mean()
            rank, count = pairwise(logits, y[cpu_idx.numpy()], pairgen) if cfg['rank_weight'] else (logits.sum()*0., 0)
            loss = bce + cfg['rank_weight']*rank
            assert torch.isfinite(loss)
            loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 5.); opt.step()
            factor = len(idx)/len(x)
            loss_sum += float(loss.detach())*factor
            bce_sum += float(bce.detach())*factor
            rank_sum += float(rank.detach())*factor
            pairs_total += count
        rec = dict(epoch=epoch, loss=loss_sum, bce_loss=bce_sum, rank_loss=rank_sum,
                   sampled_pairs=pairs_total, seconds=prior_elapsed+time.monotonic()-start)
        if epoch == 1 or epoch % 5 == 0 or epoch == 120:
            scores = predict(net, vt, P, N)
            ap = ap_numpy(vy, scores); rec['validation_macro_AP'] = ap
            if ap > best+1e-6:
                best, best_epoch, bad = ap, epoch, 0
                best_model = {k: t.detach().cpu().clone() for k, t in net.state_dict().items()}
                best_scores = scores.copy()
            else:
                bad += 1
            log('neural_validation', name=name, gpu=gpu, epoch=epoch, best_epoch=best_epoch, AP=ap, best_AP=best, bad_checks=bad)
        history.append(rec)
        state = dict(epoch=epoch, model={k: t.detach().cpu() for k, t in net.state_dict().items()},
                     optimizer=opt.state_dict(), generator_state=gen.get_state(),
                     pair_generator_state=pairgen.bit_generator.state,
                     cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(device),
                     python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                     history=history, best=best, best_epoch=best_epoch, bad=bad,
                     best_model=best_model, best_scores=best_scores,
                     elapsed_seconds=prior_elapsed+time.monotonic()-start, config=cfg, resumes=resumes)
        ck = p/f'checkpoints/epoch_{epoch:04d}.pt'; cert = ck.with_name(ck.stem+'.DONE.json')
        save_torch(ck, state); certificate(cert, [ck], context)
        # Retain two fully validated epochs; a kill before the new certificate leaves the previous intact.
        for obsolete in sorted((p/'checkpoints').glob('epoch_*.DONE.json'))[:-2]:
            oldck = obsolete.with_name(obsolete.name.replace('.DONE.json', '.pt'))
            obsolete.unlink()
            oldck.unlink()
    assert best_model is not None
    best_state = dict(epoch=best_epoch, model=best_model, config=cfg, input_dimensions=x.shape[1], validation_macro_AP=best)
    save_torch(p/'best.pt', best_state)
    # Independent fresh model construction and disk reload, using identical inference batching.
    audit_net = Network(x.shape[1], cfg['output'], cfg['dropout']).to(device)
    audit_net.load_state_dict(torch.load(p/'best.pt', map_location='cpu', weights_only=False)['model'])
    scores = predict(audit_net, vt, P, N)
    assert np.array_equal(scores, best_scores), float(np.max(np.abs(scores-best_scores)))
    train_scores = predict(audit_net, xt, P, N)
    npz(p/'validation.npz', protein_id=d['val_ids'], class_order=d['class_order'], scores=scores)
    npz(p/'train_scores.npz', protein_id=d['train_ids'], class_order=d['class_order'], scores=train_scores)
    support = ['no_known_labels' if P[j]+N[j] == 0 else 'no_positive_constant_0' if P[j] == 0 else 'no_negative_constant_1' if N[j] == 0 else 'both' for j in range(32)]
    meta = dict(config=cfg, history=history, best_epoch=best_epoch, validation_macro_AP=best,
                seconds=prior_elapsed+time.monotonic()-start, gpu=gpu, pid=os.getpid(), seed=cfg['seed'],
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device), peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                torch=torch.__version__, cuda=torch.version.cuda, deterministic_algorithms=True, tf32=False,
                training_dtype='float32', inference_dtype='float32', early_stop_patience_checks=6,
                max_epochs=120, batch_size=1024, inference_batch_size=2048,
                loss='mean over 32 classes of known-count-normalized weighted BCE plus configured known-only pairwise softplus',
                ranking_sampling='per class per mini-batch min(64,P_batch*N_batch) iid pairs with replacement; per eligible class mean; unknown excluded',
                input_normalization='per-row LayerNorm without affine data fitting',
                output_structure=cfg['output'], optimizer=dict(name='AdamW', lr=0.001, weight_decay=0.0001, clip_grad_norm=5.),
                training_support=dict(P=P.tolist(), N=N.tolist(), U=(y == -1).sum(0).tolist(), status=support),
                independent_val_max_abs_diff=0., independent_val_bitwise_equal=True,
                epoch_checkpoints='last two complete atomic model/optimizer/RNG states; every epoch checkpointed',
                resumes=resumes, test_used=False)
    jout(p/'history.json', meta)
    certificate(p/'DONE.json', [p/'best.pt', p/'validation.npz', p/'train_scores.npz', p/'history.json'], context)
    log('neural_complete', name=name, best_AP=best, best_epoch=best_epoch, seconds=meta['seconds'])
    del audit_net, net, opt, xt, vt, yt, mask
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=int, default=0)
    parser.add_argument('--config-file', type=Path)
    parser.add_argument('--write-config', action='store_true')
    args = parser.parse_args()
    if args.write_config:
        no_test_guard(); jout(E/'configurations/E12_initial.json', configurations()); return
    assert torch.cuda.device_count() == 4
    configs = readj(args.config_file) if args.config_file else configurations()
    failed = []
    for i, cfg in enumerate(configs):
        if i % 4 != args.worker:
            continue
        try:
            run(cfg, args.worker)
        except Exception as exc:
            failure = dict(at=now(), config=cfg, pid=os.getpid(), gpu=args.worker, exception_type=type(exc).__name__,
                           error=str(exc), traceback=traceback.format_exc(), status='failed_not_silently_skipped', test_used=False)
            jout(E/f"models/neural/{cfg['name']}/failure.json", failure)
            log('neural_failed', name=cfg['name'], error=str(exc))
            failed.append(cfg['name']); torch.cuda.empty_cache()
    log('neural_worker_finished', worker=args.worker, failed=failed)
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

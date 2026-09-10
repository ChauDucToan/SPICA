#!/usr/bin/env python3
"""Fixed-init MP(mu_T) then MP(q) gradient calibration; no optimizer or training."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import itertools
import json
import os
from pathlib import Path
import traceback

# The existing diagnostic sets offline/W&B-disabled and single-thread defaults.
from diagnose_coupled_sigreg import (
    _seed42, _fresh_output, _file_sha256 as _sha256, _artifact_record as _artifact,
    _module_hash, _move_batch, _last_block_layout,
    _global_rng_state, _all_float32,
)

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL = ROOT / 'outputs/fusion_mp_execution_20260909T115500Z/runs/F2_MP'
INITIALIZATION = HISTORICAL / 'initialization.json'
BATCHES, BATCH_SIZE, RHO = 4, 32, 0.1
COMPONENTS = ('base', 't', 'q', 'text_ce')


def _json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


@contextmanager
def _capture(model):
    """Capture one shared training graph; restore hooks even on failure."""
    captured = {'model_calls': 0, 'photo_calls': 0}

    def hook(_module, _inputs, output):
        captured['model_calls'] += 1
        captured['output'] = output

    handle = model.register_forward_hook(hook)
    sentinel = object()
    old = model.__dict__.get('encode_photo', sentinel)
    original = model.encode_photo

    def encode(photos):
        captured['photo_calls'] += 1
        captured['bank'] = original(photos)
        return captured['bank']

    model.encode_photo = encode
    try:
        yield captured
    finally:
        handle.remove()
        if old is sentinel:
            model.__dict__.pop('encode_photo', None)
        else:
            model.__dict__['encode_photo'] = old


def _losses(model, batch):
    import torch
    from spica.coupled_predictive_losses import coupled_region_loss, _main_photo_multi_positive

    with _capture(model) as captured:
        terms = coupled_region_loss(
            model, *(batch[k] for k in ('clean', 'corrupted', 'photos', 'positive_indices', 'negative_indices', 'labels', 'photo_labels', 'photo_ids')),
            lambda_sig=0., sigreg=None, main_photo_objective='multi_positive_supervised_contrastive',
        )
    assert captured['model_calls'] == captured['photo_calls'] == 1
    output, bank, labels = captured['output'], captured['bank'], batch['labels']
    assert output.mu_t is not None

    def mp(query):
        n = len(labels)
        return .5 * sum(_main_photo_multi_positive(x, bank, batch['photo_labels'], labels) for x in (query[:n], query[n:]))

    result = {'base': terms['total'], 't': mp(output.mu_t), 'q': mp(output.q),
              'text_ce': .5 * sum(terms[f'{v}_ce_i'] + .25 * terms[f'{v}_ce_t'] + .25 * terms[f'{v}_ce_pool'] for v in ('clean', 'masked'))}
    assert all(torch.isfinite(x).item() for x in result.values())
    return result


def _scopes(model):
    last, _ = _last_block_layout(model)
    names = {id(p): n for n, p in model.named_parameters()}
    scopes = {'student_last_block': [(names[id(p)], p) for p in last]}
    for scope in ('predictor', 'pooled_head'):
        scopes[scope] = [(n, p) for n, p in model.named_parameters() if n.startswith(scope + '.') and p.requires_grad]
    scopes['prompts'] = [(names[id(p)], p) for group in model.optimizer_parameter_groups() if group['name'] == 'prompts' for p in group['params']]
    ids = [id(p) for rows in scopes.values() for _, p in rows]
    assert all(scopes.values()) and len(ids) == len(set(ids))
    return scopes


def _gradient(loss, scopes, retain):
    import torch

    values = iter(torch.autograd.grad(loss, [p for rows in scopes.values() for _, p in rows], retain_graph=retain, allow_unused=True))
    vectors, unused = {}, {}
    for scope, rows in scopes.items():
        chunks, missing = [], []
        for name, p in rows:
            g = next(values)
            if g is None:
                missing.append(name)
                g = torch.zeros_like(p)
            assert g.dtype == torch.float32 and g.shape == p.shape and torch.isfinite(g).all()
            chunks.append(g.detach().reshape(-1))
        vectors[scope] = torch.cat(chunks).cpu()
        unused[scope] = missing
    return vectors, unused


def _norm(x, positive=False):
    import torch

    n = float(torch.linalg.vector_norm(x.double()))
    if not __import__('math').isfinite(n) or (positive and n <= 0):
        raise FloatingPointError('nonfinite or zero calibration denominator/reference gradient')
    return n


def _cos(a, b):
    na, nb = _norm(a), _norm(b)
    return float(a.double().dot(b.double()) / (na * nb)) if na and nb else None


def _select(rows):
    """Sequential selection at ONE fixed state; stage2 is exact gradient algebra."""
    import numpy as np

    assert len(rows) == BATCHES
    last, pool = 'student_last_block', 'pooled_head'
    lt = RHO * float(np.median([_norm(g['base'][last], True) / _norm(g['t'][last], True) for g in rows]))
    candidates = {}
    for scope in (last, pool):
        candidates[scope] = RHO * float(np.median([
            _norm(g['base'][scope].double() + lt * g['t'][scope].double(), True) / _norm(g['q'][scope], True) for g in rows
        ]))
    lq = min(candidates.values())
    assert lt > 0 and lq > 0 and np.isfinite([lt, *candidates.values()]).all()
    return {'rho_t': RHO, 'rho_q': RHO, 'lambda_t': lt, 'lambda_q': lq,
            'lambda_q_candidates': candidates, 'binding_scope': min(candidates, key=candidates.get),
            'formula': 'lambdaT=.1 median(norm(base_last)/norm(T_last)); B1=base+lambdaT*T; lambdaQ=.1 min_scopes median(norm(B1_scope)/norm(Q_scope)), scopes=last,pool',
            'scope': 'initialization-only, median-based target; NOT a per-batch cap, mAP optimum or effective AdamW-update ratio'}


def _summarize(gradients, selection):
    lt, lq = selection['lambda_t'], selection['lambda_q']
    result = {}
    for scope in gradients['base']:
        b, t, q, ce = [gradients[k][scope].double() for k in COMPONENTS]
        b1, final = b + lt * t, b + lt * t + lq * q
        result[scope] = {
            'norms': {k: _norm(x) for k, x in zip((*COMPONENTS, 'base_plus_t', 'final'), (b, t, q, ce, b1, final), strict=True)},
            'realized_t_over_base': lt * _norm(t) / _norm(b, True),
            'realized_q_over_base_plus_t': lq * _norm(q) / _norm(b1, True),
            'final_norm_over_base': _norm(final) / _norm(b, True),
            'cosines': {k: _cos(a, c) for k, a, c in (('t_base', t, b), ('t_text_ce', t, ce), ('q_base_plus_t', q, b1), ('q_text_ce', q, ce), ('t_q', t, q), ('final_base', final, b))},
        }
    return result


def _first128(path):
    with path.open() as f:
        rows = [json.loads(x) for x in itertools.islice(f, 128)]
    assert len(rows) == 128
    return rows


def run(output):
    import torch
    from spica.data.coupled_training import load_protocol_data, make_train_loader, prepare_batch, verify_clip_cache, positive_pool_identity
    from spica.models.clip import load_frozen_clip
    from spica.models.coupled_predictive import CoupledPredictiveModel
    from spica.train_coupled_predictive import _initialization_hashes, _tensor_hash
    from spica.provenance import capture_provenance
    from run_coupled_campaign import _copy_source_archive

    assert torch.cuda.is_available(), 'CUDA required; no CPU fallback'
    config = {'diagnostic': 'fusion_mp_tq_init_v1', 'seed': 42, 'batches': 4, 'batch_size': 32, 'rho_t': .1, 'rho_q': .1,
              'baseline': 'F2_MP full total', 'sequence': 'add lambdaT*MP(muT), then lambdaQ*MP(q)',
              'tau': .07, 'positive_pool': 'full', 'optimizer_updates': 0, 'wandb': 'disabled'}
    provenance = capture_provenance(ROOT, resolved_config=config)
    _copy_source_archive(output, {'diagnostic': config['diagnostic']}, provenance)
    protocol, clip = load_protocol_data(), verify_clip_cache()
    _seed42()
    classmap = {int(i): str(protocol['names'][int(i)]) for i in protocol['split'].train_class_ids}
    assert len(classmap) == 84
    bundle = load_frozen_clip(model_name='ViT-B-32-quickgelu', pretrained=str(clip['path']), device=torch.device('cuda'))
    model = CoupledPredictiveModel(bundle.encoder, bundle.tokenizer, classmap, architecture='predictive_fusion_v2').to('cuda')
    model.train(True)
    _all_float32(model, 'model')
    init = _initialization_hashes(model)
    assert init == json.loads(INITIALIZATION.read_text())
    before = _module_hash(model)
    teacher_before = _module_hash(model.original_clip)
    assert not model.original_clip.training and all(not p.requires_grad for p in model.original_clip.parameters())
    scopes = _scopes(model)
    assert len(scopes['student_last_block']) == 12 and sum(p.numel() for _, p in scopes['student_last_block']) == 7087872
    layout = {s: [{'name': n, 'shape': list(p.shape), 'numel': p.numel()} for n, p in rows] for s, rows in scopes.items()}
    _json(output / 'gradient_layout.json', layout)
    traces, masks = _first128(HISTORICAL / 'observation_trace.jsonl'), _first128(HISTORICAL / 'mask_metadata.jsonl')
    loader = make_train_loader(protocol, bundle.transform, positive_pool='full')
    assert torch.equal(loader.generator.get_state(), torch.Generator().manual_seed(42).get_state())
    iterator, records, raw_artifacts = iter(loader), [], []
    torch.cuda.reset_peak_memory_stats()
    for i in range(BATCHES):
        raw = next(iterator)
        batch = prepare_batch(raw, data_root=Path(str(protocol['data'].root)), step=i, classids=model.classids.cpu().tolist())
        expected = [{k: v for k, v in x.items() if k not in ('step', 'actual_lr')} for x in traces[i*32:(i+1)*32]]
        actual = [{k: v for k, v in x.items() if k != 'step'} for x in batch['trace']]
        assert actual == expected and batch['mask_metadata']['rows'] == masks[i*32:(i+1)*32]
        assert batch['clean'].shape[0] == 32 and len(batch['photos']) <= 64
        records.append({'batch': i, 'trace': batch['trace'], 'mask_metadata': batch['mask_metadata'], 'photo_ids': list(batch['photo_ids']),
                        'tensors': {k: {'sha256': _tensor_hash(v), 'shape': list(v.shape)} for k, v in batch.items() if isinstance(v, torch.Tensor)},
                        'labels': batch['labels'].tolist(), 'photo_labels': batch['photo_labels'].tolist()})
        batch = _move_batch(batch, torch.device('cuda'))
        rng = _global_rng_state()
        losses = _losses(model, batch)
        gradients, unused = {}, {}
        for j, k in enumerate(COMPONENTS):
            gradients[k], unused[k] = _gradient(losses[k], scopes, retain=j < 3)
        assert all(not x for x in unused['base'].values())
        assert unused['t']['pooled_head'] == [n for n, _ in scopes['pooled_head']]
        assert unused['q']['predictor'] == [n for n, _ in scopes['predictor']]
        assert _norm(gradients['t']['pooled_head']) == _norm(gradients['q']['predictor']) == 0
        assert rng == _global_rng_state() and all(p.grad is None for p in model.parameters())
        assert _module_hash(model) == before
        name = f'raw_gradients_batch{i:02d}.pt'
        torch.save({'batch': i, 'losses': {k: float(v.detach().cpu()) for k, v in losses.items()}, 'gradients': gradients, 'unused': unused, 'layout': layout}, output / name)
        raw_artifacts.append(_artifact(output, name))
        del losses, gradients, unused, batch, raw
    # CPU algebra on immutable raw FP32 gradients, not another optimization step.
    raw_rows = [torch.load(output / a['path'], map_location='cpu', weights_only=True) for a in raw_artifacts]
    selection = _select([r['gradients'] for r in raw_rows])
    measured = [{'batch': r['batch'], 'losses': r['losses'], 'scopes': _summarize(r['gradients'], selection)} for r in raw_rows]
    assert _module_hash(model) == before and _module_hash(model.original_clip) == teacher_before
    assert capture_provenance(ROOT)['source_snapshot']['sha256'] == provenance['source_snapshot']['sha256']
    _json(output / 'data_records.json', records)
    _json(output / 'lambda_selection.json', selection)
    _json(output / 'measured_rows.json', measured)
    result = {'status': 'MEASURED_PENDING_REVIEW', 'verified': False, 'config': config, 'selection': selection, 'rows': measured,
              'source_snapshot_hash': provenance['source_snapshot']['sha256'], 'clip_identity': clip,
              'data_identity': {'split': protocol['split_identity'], 'manifest': protocol['manifest_identity'], 'pool': positive_pool_identity(protocol, 'full')},
              'historical_bindings': {str(p.relative_to(ROOT)): _sha256(p) for p in (INITIALIZATION, HISTORICAL/'observation_trace.jsonl', HISTORICAL/'mask_metadata.jsonl')},
              'initialization_hashes': init, 'model_hash_before': before, 'model_hash_after': _module_hash(model),
              'teacher_hash_before': teacher_before, 'teacher_hash_after': _module_hash(model.original_clip),
              'entire_model_unchanged': True, 'parameter_grad_all_none': True, 'optimizer_updates': 0,
              'memory': {'peak_allocated': torch.cuda.max_memory_allocated(), 'peak_reserved': torch.cuda.max_memory_reserved()},
              'artifacts': raw_artifacts + [_artifact(output, n) for n in ('data_records.json', 'gradient_layout.json', 'lambda_selection.json', 'measured_rows.json')],
              'finished_utc': datetime.now(timezone.utc).isoformat()}
    _json(output / 'diagnostic_result.json', result)
    return result


def _self_check(output):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    from check_fusion_qmp_cpu import tiny_model, batch as fixture
    from spica.coupled_predictive_losses import coupled_region_loss

    args = fixture()
    batch = dict(zip(('clean', 'corrupted', 'photos', 'positive_indices', 'negative_indices', 'labels', 'photo_labels', 'photo_ids'), args, strict=True))
    model, control = tiny_model(42), tiny_model(42)
    before = _module_hash(model)
    losses = _losses(model, batch)
    old = coupled_region_loss(control, *args, lambda_sig=0., sigreg=None, main_photo_objective='multi_positive_supervised_contrastive')['total']
    assert torch.equal(losses['base'], old)
    scopes, control_scopes = _scopes(model), _scopes(control)
    grads, unused = {}, {}
    for k in COMPONENTS:
        grads[k], unused[k] = _gradient(losses[k], scopes, retain=True)
    old_grad, _ = _gradient(old, control_scopes, retain=False)
    assert all(torch.equal(grads['base'][s], old_grad[s]) for s in scopes)
    assert unused['t']['pooled_head'] == [n for n, _ in scopes['pooled_head']]
    assert unused['q']['predictor'] == [n for n, _ in scopes['predictor']]
    assert _norm(grads['t']['predictor'], True) > 0 and _norm(grads['q']['pooled_head'], True) > 0
    selected = _select([grads]*4)
    final, _ = _gradient(losses['base'] + selected['lambda_t']*losses['t'] + selected['lambda_q']*losses['q'], scopes, retain=False)
    for s in scopes:
        combined = grads['base'][s].double() + selected['lambda_t']*grads['t'][s].double() + selected['lambda_q']*grads['q'][s].double()
        assert _norm(final[s].double()-combined) / _norm(combined, True) < 1e-5
    assert all(p.grad is None for p in model.parameters()) and _module_hash(model) == before
    # Explicit full-bank oracle checks the new head losses, not only the MP helper.
    out, bank = model(torch.cat((args[0], args[1]))), model.encode_photo(args[2])
    labels = args[5].repeat(2)
    positive = args[6][None, :] == labels[:, None]
    for k, query in [('t', out.mu_t), ('q', out.q)]:
        logits = torch.nn.functional.normalize(query.double(), dim=-1) @ torch.nn.functional.normalize(bank.double(), dim=-1).T / .07
        oracle = -(logits.log_softmax(-1) * positive).sum(-1).div(positive.sum(-1)).mean()
        torch.testing.assert_close(losses[k].detach().double(), oracle, rtol=1e-6, atol=1e-7)
    toy = {'base': {'student_last_block': torch.tensor([3.,4.]), 'pooled_head': torch.tensor([1.,0.])},
           't': {'student_last_block': torch.tensor([1.,0.]), 'pooled_head': torch.zeros(2)},
           'q': {'student_last_block': torch.tensor([0.,2.]), 'pooled_head': torch.tensor([10.,0.])}}
    answer = _select([toy]*4)
    assert abs(answer['lambda_t']-.5) < 1e-12 and abs(answer['lambda_q']-.01) < 1e-12 and answer['binding_scope']=='pooled_head'
    try:
        _norm(torch.zeros(2), True)
    except FloatingPointError:
        pass
    else:
        raise AssertionError('zero calibration reference accepted')
    assert 'encode_photo' not in model.__dict__
    result = {'status': 'PASS', 'summary': {'pass': 4, 'fail': 0, 'skip': 0}, 'checks': ['base scalar/gradient exact', 'new-head MP oracle and routing', 'sequential selection and weighted-gradient linearity', 'zero rejection/no updates/patch cleanup'],
              'source_sha256': _sha256(Path(__file__)), 'pretrained': False, 'device': 'cpu', 'optimizer_updates': 0}
    _json(output / 'receipt.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    output = None
    try:
        output = _fresh_output(args.output_dir)
        result = _self_check(output) if args.self_check else run(output)
        print(json.dumps({'status': result['status'], 'output': str(output), 'selection': result.get('selection')}))
        return 0
    except Exception as error:
        if output is not None:
            _json(output / ('receipt.json' if args.self_check else 'diagnostic_result.json'), {'status': 'FAIL', 'error': str(error), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    raise SystemExit(main())

"""Three no-update tests: head routing, global context shuffle, and pool bias.

Clean pseudo-validation only for retrieval; the first 32 historical train batches
(clean + region-masked) for the paired negative-pool counterfactual. No selection,
optimizer, backward, training-run mutation, or official-unseen evaluation.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from coupled_diagnostic_metrics import feature_geometry, paired_rank_report, self_check
from spica.data.coupled_training import load_protocol_data, make_train_loader, prepare_batch, verify_clip_cache
from spica.data.datasets import RetrievalEvalDataset
from spica.evaluation.coupled_predictive import CoupledPredictiveAdapter
from spica.evaluation.embeddings import EncodedRetrievalSet
from spica.evaluation.frozen_prompt import encode_prompted_loader, evaluate_prompted_all_denominators
from spica.models.clip import load_frozen_clip
from spica.models.coupled_predictive import CoupledPredictiveModel
from spica.provenance import capture_provenance
from spica.train_coupled_predictive import _seed, _state_hash

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('R0', 'R1', 'R1_SIG')
SEED = 845701
BATCHES = 32


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def archived_verifier(campaign):
    path = campaign / 'source_snapshot/files/scripts/verify_coupled_campaign.py'
    spec = importlib.util.spec_from_file_location('archived_coupled_verifier', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tensor_digest(x):
    return hashlib.sha256(memoryview(x.detach().cpu().contiguous().numpy())).hexdigest()


def numpy_features(output):
    return {name: value.detach().cpu().numpy() for name in ('g', 'q', 'mu_i', 'mu_t')
            if (value := getattr(output, name)) is not None}


def encode_queries(model, entries, transform):
    arrays = defaultdict(list)
    contexts = []
    hook = None
    if model.predictor is not None:
        hook = model.predictor.register_forward_pre_hook(
            lambda module, args: contexts.append(args[0].detach().cpu()))
    paths, labels = [], []
    loader = DataLoader(RetrievalEvalDataset(entries, transform), batch_size=256,
                        shuffle=False, num_workers=4)
    try:
        with torch.no_grad():
            for batch in loader:
                values = numpy_features(model(batch['image'].to('cuda')))
                for key, value in values.items():
                    arrays[key].append(value)
                paths.extend(batch['path'])
                labels.extend(batch['label'].tolist())
    finally:
        if hook is not None:
            hook.remove()
    return ({k: np.concatenate(v) for k, v in arrays.items()},
            torch.cat(contexts) if contexts else None, tuple(paths), np.asarray(labels))


def evaluate(features, labels, paths, gallery, destination, canonical):
    queries = EncodedRetrievalSet(torch.from_numpy(features), torch.from_numpy(labels), paths)
    values = evaluate_prompted_all_denominators(queries, gallery, query_chunk_size=256,
                                               device=torch.device('cuda'))
    prefix = values['prefix_positive']
    top = prefix.top_indices.numpy()
    relevance = gallery.labels.numpy()[top] == labels[:, None]
    p200 = relevance.astype(np.float32).mean(axis=1)
    p200_rounding_delta = abs(float(p200.astype(np.float64).mean()) - prefix.metrics.precision_at_k[200])
    # CPU count/200 and CUDA float32 reciprocal multiplication can differ by one ULP.
    assert p200_rounding_delta <= np.finfo(np.float32).eps, p200_rounding_delta
    np.savez(destination, top_indices=top, top_scores=prefix.top_scores.numpy(),
             full_ap=prefix.average_precision_per_query.numpy(), p200=p200,
             **{f'ap200_{name}': val.average_precision_at_k_per_query[200].numpy()
                for name, val in values.items()})
    counts = np.bincount(top[:, 0], minlength=len(gallery.paths))
    exposure = np.bincount(top.ravel(), minlength=len(gallery.paths))
    metrics = {
        'full_mAP': prefix.metrics.mean_average_precision,
        'P@1': prefix.metrics.precision_at_k[1],
        'P@200': prefix.metrics.precision_at_k[200],
        **{f'mAP@200_{name}': val.metrics.mean_average_precision_at_k[200]
           for name, val in values.items()},
    }
    return {'metrics': metrics, 'arrays': str(destination),
            'p200_cpu_array_mean_vs_gpu_scalar_abs_delta': p200_rounding_delta,
            'p200_float32_tolerance': float(np.finfo(np.float32).eps),
            'hubness': {'unique_top1_photos': int((counts > 0).sum()),
                        'most_popular_top1_share': float(counts.max() / len(labels)),
                        'photos_in_every_top200': int((exposure == len(labels)).sum()),
                        'unique_top200_photos': int((exposure > 0).sum()),
                        'canonical_top200_exposure_share': float(canonical[top].mean())}}, top


def shuffled_context_outputs(model, features, contexts, permutation):
    """Actually rerun the head on globally permuted contexts, not within-batch labels."""
    outputs = defaultdict(list)
    with torch.no_grad():
        for start in range(0, len(permutation), 256):
            indices = permutation[start:start + 256]
            if contexts is None:
                g = torch.from_numpy(features['g'][indices]).to('cuda')
            else:
                h = contexts[indices].to('cuda')
                g = h.mean(dim=1)
                mu_i, mu_t = model.predictor(h, model.photo_model.photo_prompt, model.text_bank.context)
                outputs['mu_i'].append(mu_i.cpu().numpy())
                outputs['mu_t'].append(mu_t.cpu().numpy())
            outputs['q'].append(F.normalize(model.pooled_head(g), dim=-1).cpu().numpy())
    result = {k: np.concatenate(v) for k, v in outputs.items()}
    parity = {k: float(np.max(np.abs(value - features[k][permutation]))) for k, value in result.items()}
    assert max(parity.values()) < 2e-6, parity
    return result, parity


def train_pool_test(model, protocol, transform, output):
    """Keep query/positive/full-negative fixed; replace only negative photo, same class."""
    _seed(42)
    iterator = iter(make_train_loader(protocol, transform))
    unique = {str(e.path.resolve()): e for e in protocol['pairing']['mapping'].values()}
    by_class = defaultdict(list)
    for key in sorted(unique):
        by_class[int(unique[key].label)].append(unique[key])
    assert len(unique) == 8400 and all(len(v) == 100 for v in by_class.values())
    canonical_rng = random.Random(SEED)
    arrays = defaultdict(list)
    records, mask_records = [], []
    root = Path(protocol['data'].root).resolve()
    with torch.no_grad():
        for step in range(BATCHES):
            batch = prepare_batch(next(iterator), data_root=root, step=step,
                                  classids=model.classids.cpu().tolist())
            chosen = [canonical_rng.choice(by_class[int(row['negative_label'])]) for row in batch['trace']]
            for row, entry in zip(batch['trace'], chosen):
                assert int(entry.label) == int(row['negative_label']) != int(row['label'])
                assert str((root / row['positive_photo_path']).resolve()) in unique
                records.append({**row, 'canonical_negative_path': str(entry.path.resolve()),
                                'full_negative_is_canonical': str((root / row['negative_photo_path']).resolve()) in unique})
            mask_records.extend(batch['mask_metadata']['rows'])
            dataset = RetrievalEvalDataset(chosen, transform)
            canonical_images = torch.stack([dataset[i]['image'] for i in range(len(dataset))]).to('cuda')
            nc = model.encode_photo(canonical_images)
            live = model.encode_photo(batch['photos'].to('cuda'))
            arrays['positive'].append(live[batch['positive_indices'].to('cuda')].cpu().numpy())
            arrays['negative_full'].append(live[batch['negative_indices'].flatten().to('cuda')].cpu().numpy())
            arrays['negative_canonical'].append(nc.cpu().numpy())
            values = numpy_features(model(torch.cat((batch['clean'], batch['corrupted'])).to('cuda')))
            for key, value in values.items():
                arrays[f'clean_{key}'].append(value[:32])
                arrays[f'masked_{key}'].append(value[32:])
    del iterator
    arrays = {k: np.concatenate(v) for k, v in arrays.items()}
    labels = np.asarray([row['label'] for row in records])
    permutation = np.random.default_rng(SEED).permutation(len(labels))
    arrays['labels'] = labels
    arrays['permutation'] = permutation
    np.savez(output / 'train_features.npz', **arrays)
    write_json(output / 'train_records.json', records)
    write_json(output / 'train_masks.json', mask_records)
    reports, geometries = {}, {}
    for view in ('clean', 'masked'):
        for head in ('g', 'q', 'mu_i', 'mu_t'):
            key = f'{view}_{head}'
            if key not in arrays:
                continue
            x = arrays[key]
            geometries[key] = feature_geometry(x, labels)
            if head == 'g':
                continue
            variants = {'intact': x, 'shuffled': x[permutation],
                        'constant_centroid': np.broadcast_to(x.mean(axis=0, dtype=np.float64), x.shape)}
            for variant, query in variants.items():
                reports[f'{key}/{variant}'] = paired_rank_report(query, arrays['positive'],
                                                               arrays['negative_full'], arrays['negative_canonical'])
    write_json(output / 'train_paired_ranking.json', reports)
    write_json(output / 'train_geometry.json', geometries)
    return {'sample_count': len(labels), 'batch_count': BATCHES,
            'class_count': len(set(labels.tolist())),
            'full_negative_canonical_fraction': float(np.mean([r['full_negative_is_canonical'] for r in records])),
            'permutation_same_class_fraction': float(np.mean(labels == labels[permutation])),
            'constant_definition': 'per-head/per-view empirical mean of these 1024 train query embeddings, normalized for scoring; no fitting/optimization',
            'report': str(output / 'train_paired_ranking.json'),
            'geometry': str(output / 'train_geometry.json')}, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign-root', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--cpu-self-check', action='store_true')
    args = parser.parse_args()
    if args.cpu_self_check:
        checks = self_check()
        counts = np.arange(201, dtype=np.float32)
        differences = np.abs(counts / np.float32(200) - counts * np.float32(1 / 200))
        assert 0 < differences.max() <= np.finfo(np.float32).eps
        checks['checks'].append('P200_float32_division_vs_reciprocal_rounding')
        print(json.dumps(checks))
        return
    if args.campaign_root is None or args.output is None:
        parser.error('--campaign-root and --output are required')
    campaign, output = args.campaign_root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = {'status': 'RUNNING', 'scope': 'clean pseudo-validation; train-only clean/corrupted paired-pool diagnostic',
              'step': 3600, 'arms': {}, 'optimizer_updates': 0, 'backward_calls': 0,
              'shuffle_seed': SEED, 'started_utc': datetime.now(timezone.utc).isoformat()}
    try:
        assert torch.cuda.is_available(), 'CUDA unavailable; no fallback'
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['WANDB_MODE'] = 'disabled'
        index = json.loads((campaign / 'source_snapshot/index.json').read_text())
        for row in index['manifest']:
            if row['path'].startswith(('src/', 'configs/')) or row['path'] in ('uv.lock', 'pyproject.toml'):
                assert sha(ROOT / row['path']) == row['sha256'], row['path']
        source = capture_provenance(ROOT)
        write_json(output / 'diagnostic_provenance.json', source)
        for row in source['source_snapshot']['manifest']:
            dest = output / 'source_snapshot/files' / row['path']
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / row['path'], dest)
            assert sha(dest) == row['sha256']
        write_json(output / 'source_snapshot/index.json', source['source_snapshot'])
        result['training_source_sha256'] = index['sha256']
        result['diagnostic_source_sha256'] = source['source_snapshot']['sha256']
        verifier = archived_verifier(campaign)
        clip = verify_clip_cache()
        protocol = load_protocol_data()
        split = protocol['split']
        result['clip_identity'] = clip
        result['split_identity'] = protocol['split_identity']
        permutation = np.random.default_rng(SEED).permutation(len(split.validation_sketch_entries))
        np.save(output / 'validation_permutation.npy', permutation)
        all_records = []
        for arm in ARMS:
            arm_started = time.monotonic()
            arm_out = output / arm
            arm_out.mkdir()
            result['current_arm'] = arm
            write_json(output / 'runtime.json', result)
            print(f'{arm}: loading checkpoint3600', flush=True)
            run = campaign / 'runs' / arm
            config = json.loads((run / 'resolved_config.json').read_text())
            checkpoint = run / 'checkpoint_step3600.pt'
            run_result = json.loads((run / 'run_result.json').read_text())
            expected = next(row for row in run_result['checkpoints'] if row['step'] == 3600)
            assert sha(checkpoint) == expected['sha256']
            _seed(42)
            bundle = load_frozen_clip(model_name='ViT-B-32-quickgelu', pretrained=clip['path'], device=torch.device('cuda'))
            classmap = {int(k): protocol['names'][int(k)] for k in config['train_class_ids']}
            model = CoupledPredictiveModel(bundle.encoder, bundle.tokenizer, classmap,
                                           architecture=config['architecture']).to('cuda').eval()
            verifier.load_replay_state(model, checkpoint, arm, 3600, index['sha256'])
            before = _state_hash(model)
            torch.cuda.reset_peak_memory_stats()
            print(f'{arm}: encoding clean query contexts and checkpoint gallery', flush=True)
            features, contexts, paths, labels = encode_queries(model, split.validation_sketch_entries, bundle.transform)
            loader = DataLoader(RetrievalEvalDataset(split.validation_photo_entries, bundle.transform),
                                batch_size=256, shuffle=False, num_workers=4)
            gallery = encode_prompted_loader(CoupledPredictiveAdapter(model, query='q'), loader, photo=True)
            canonical_base = Path(protocol['data'].root).resolve() / '256x256/photo/tx_000000000000_ready'
            canonical = np.asarray([Path(p).parent.parent == canonical_base for p in gallery.paths])
            assert canonical.sum() == 2000
            np.savez(arm_out / 'validation_features.npz', **features, labels=labels,
                     gallery=gallery.embeddings.numpy(), gallery_labels=gallery.labels.numpy(), canonical=canonical)
            write_json(arm_out / 'identities.json', {'query_paths': paths, 'gallery_paths': gallery.paths,
                       'permutation_same_class_fraction': float(np.mean(labels == labels[permutation])),
                       'permutation_fixed_points': int(np.sum(permutation == np.arange(len(labels))))})
            geometry = {k: feature_geometry(v, labels) for k, v in features.items()}
            geometry['gallery'] = feature_geometry(gallery.embeddings.numpy(), gallery.labels.numpy())
            write_json(arm_out / 'validation_geometry.json', geometry)
            context_sha256 = tensor_digest(contexts) if contexts is not None else None
            shuffled, parity = shuffled_context_outputs(model, features, contexts, permutation)
            del contexts
            gc.collect()
            np.savez(arm_out / 'shuffled_features.npz', **shuffled)
            retrieval = {}
            for head in ('q', 'mu_i', 'mu_t'):
                if head not in shuffled:
                    continue
                print(f'{arm}: retrieval {head} intact/shuffled', flush=True)
                intact, top = evaluate(features[head], labels, paths, gallery,
                                       arm_out / f'{head}_intact.npz', canonical)
                swapped, swap_top = evaluate(shuffled[head], labels, paths, gallery,
                                            arm_out / f'{head}_shuffled.npz', canonical)
                cosine = F.cosine_similarity(torch.from_numpy(features[head]), torch.from_numpy(shuffled[head]), dim=-1).numpy()
                overlap = np.asarray([len(set(a).intersection(b)) / 200 for a, b in zip(top.tolist(), swap_top.tolist())])
                retrieval[head] = {'intact': intact, 'shuffled': swapped,
                                   'original_vs_shuffled_mean_cosine': float(cosine.mean()),
                                   'original_vs_shuffled_top200_overlap': float(overlap.mean())}
                np.savez(arm_out / f'{head}_shuffle_comparison.npz', cosine=cosine, top200_overlap=overlap)
                del top, swap_top
            main_head = 'q' if arm == 'R0' else 'mu_i'
            print(f'{arm}: comparing main head to historical raw probe', flush=True)
            with (run / 'probe_step3600.json').open() as f:
                historical = json.load(f)
            assert list(paths) == historical['identities']['query_ids']
            assert list(gallery.paths) == historical['identities']['gallery_ids']
            assert labels.tolist() == historical['identities']['query_labels']
            raw = historical['clean']
            replay_delta = {k: abs(v - raw[k]) for k, v in retrieval[main_head]['intact']['metrics'].items() if k in raw}
            assert max(replay_delta.values()) < 1e-7, replay_delta
            saved = np.load(arm_out / f'{main_head}_intact.npz')
            assert np.array_equal(saved['top_indices'], np.asarray(raw['top_indices']))
            assert np.allclose(saved['full_ap'], raw['average_precision_per_query'], atol=1e-7, rtol=0)
            assert np.allclose(saved['p200'], raw['P@200_per_query'], atol=1e-7, rtol=0)
            del historical, raw, saved
            gc.collect()
            write_json(arm_out / 'retrieval.json', retrieval)
            print(f'{arm}: 32 no-update train batches / paired negative pools', flush=True)
            train_report, records = train_pool_test(model, protocol, bundle.transform, arm_out)
            if all_records:
                assert records == all_records
            else:
                all_records = records
                with (campaign / 'runs/R0/observation_trace.jsonl').open() as f:
                    for row in records:
                        old = json.loads(next(f))
                        for key in ('sketch_relative', 'positive_photo_path', 'negative_photo_path', 'label', 'negative_label'):
                            assert row[key] == old[key], (key, row[key], old[key])
            after = _state_hash(model)
            assert before == after, 'model state changed in no-update diagnostic'
            assert all(p.grad is None for p in model.parameters())
            result['arms'][arm] = {'checkpoint': str(checkpoint), 'checkpoint_sha256': expected['sha256'],
                                   'state_hash_before': before, 'state_hash_after': after,
                                   'main_head_replay_scalar_deltas': replay_delta,
                                   'main_head_replay_top_indices_exact': True,
                                   'context_shuffle_permutation_parity_max_abs': parity,
                                   'clean_context_tensor_sha256': context_sha256,
                                   'retrieval': retrieval, 'train_pool': train_report,
                                   'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                                   'elapsed_seconds': time.monotonic() - arm_started}
            write_json(output / 'runtime.json', result)
            del features, gallery, shuffled, model, bundle, loader
            gc.collect()
            torch.cuda.empty_cache()
        result.update(status='COMPLETE', current_arm=None, elapsed_seconds=time.monotonic() - started,
                      train_triplets_cross_arm_equal=True, train_triplets_match_historical_first_1024=True)
        write_json(output / 'summary.json', result)
        write_json(output / 'runtime.json', result)
        print(json.dumps({'status': 'COMPLETE', 'output': str(output)}), flush=True)
    except Exception:
        result.update(status='FAIL', traceback=traceback.format_exc())
        write_json(output / 'failure.json', result)
        write_json(output / 'runtime.json', result)
        raise


if __name__ == '__main__':
    main()

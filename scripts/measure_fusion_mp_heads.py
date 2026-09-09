"""F2_MP@3600 no-update head/cosine measurement; reuse verified control caches."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader

from diagnose_coupled_retrieval import archived_verifier, sha, write_json
from measure_fusion_heads import encode, measure, read
from coupled_diagnostic_metrics import self_check
from spica.data.coupled_training import load_protocol_data, verify_clip_cache
from spica.data.datasets import RetrievalEvalDataset
from spica.evaluation.frozen_prompt import encode_prompted_loader
from spica.models.clip import load_frozen_clip
from spica.models.coupled_predictive import CoupledPredictiveModel
from spica.provenance import source_snapshot
from spica.train_coupled_predictive import _state_hash

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / 'outputs/fusion_mp_execution_20260909T115500Z'
CACHE = ROOT / 'outputs/fusion_head_measurements_20260909T084200Z'


def spherical_angles(features):
    q, i, t = (np.asarray(features[k], dtype=np.float64) for k in ('q', 'mu_i', 'mu_t'))
    q, i, t = (x / np.linalg.norm(x, axis=1, keepdims=True) for x in (q, i, t))
    midpoint = i + t
    norms = np.linalg.norm(midpoint, axis=1, keepdims=True)
    assert (norms > 1e-12).all(), 'antipodal endpoints do not define unique minor-arc midpoint'
    midpoint /= norms
    def angle(x, y):
        return np.degrees(np.arccos(np.clip((x * y).sum(1), -1, 1)))
    values = {'q_mu_i': angle(q, i), 'q_mu_t': angle(q, t), 'mu_i_mu_t': angle(i, t),
              'q_to_midpoint': angle(q, midpoint)}
    values['arc_length_excess'] = values['q_mu_i'] + values['q_mu_t'] - values['mu_i_mu_t']
    assert (values['arc_length_excess'] >= -1e-7).all()
    return values, {k: {'mean_degrees': float(v.mean()), 'std_degrees': float(v.std()),
                       'p05_degrees': float(np.quantile(v, .05)), 'p95_degrees': float(np.quantile(v, .95))}
                    for k,v in values.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--cpu-self-check', action='store_true')
    args = parser.parse_args()
    if args.cpu_self_check:
        a = np.array([[1., 0, 0]])
        b = np.array([[.5, np.sqrt(3)/2, 0]])
        values, _ = spherical_angles({'q': a+b, 'mu_i':a, 'mu_t':b})
        assert np.allclose(values['mu_i_mu_t'],60) and np.allclose(values['q_mu_i'],30)
        assert np.allclose(values['q_to_midpoint'],0,atol=1e-6)
        print(json.dumps(self_check()))
        return
    if args.output is None:
        parser.error('--output required')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {'status':'RUNNING', 'step':3600, 'optimizer_updates':0, 'backward_calls':0,
              'scope':'clean pseudo-validation; main mu_i fixed before measurement', 'arms':{}}
    start = time.monotonic()
    try:
        assert torch.cuda.is_available(), 'no CUDA fallback'
        runtime = read(CAMPAIGN/'runtime.json')
        assert runtime['status']=='ARM_FINISHED_UNVERIFIED' and runtime['exit_code']==0 and runtime['child_pid'] is None
        train_index = read(CAMPAIGN/'source_snapshot/index.json')
        for row in train_index['manifest']:
            if row['path'].startswith(('src/','configs/')) or row['path'] in ('uv.lock','pyproject.toml'):
                assert sha(ROOT/row['path'])==row['sha256'], row['path']
        snapshot = source_snapshot(ROOT)
        for row in snapshot['manifest']:
            dest = output/'source_snapshot/files'/row['path']
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/row['path'],dest)
            assert sha(dest)==row['sha256']
        write_json(output/'source_snapshot/index.json',snapshot)
        report.update(source_snapshot_sha256=snapshot['sha256'], training_source_sha256=train_index['sha256'])
        clip = verify_clip_cache()
        protocol = load_protocol_data()
        split = protocol['split']
        run = CAMPAIGN/'runs/F2_MP'
        result = read(run/'run_result.json')
        config = read(run/'resolved_config.json')
        checkpoint = run/'checkpoint_step3600.pt'
        expected = next(x['sha256'] for x in result['checkpoints'] if x['step']==3600)
        assert sha(checkpoint)==expected and result['source_snapshot_hash']==train_index['sha256']
        bundle = load_frozen_clip(model_name='ViT-B-32-quickgelu',pretrained=clip['path'],device=torch.device('cuda'))
        model = CoupledPredictiveModel(bundle.encoder,bundle.tokenizer,{int(k):v for k,v in config['class_names'].items()},architecture=config['architecture']).to('cuda').eval()
        archived_verifier(CAMPAIGN).load_replay_state(model,checkpoint,'F2_MP',3600,train_index['sha256'])
        before = _state_hash(model)
        print('F2_MP3600 encoding query heads and live photo gallery',flush=True)
        features, paths, labels = encode(model,split.validation_sketch_entries,bundle.transform)
        gallery = encode_prompted_loader(model,DataLoader(RetrievalEvalDataset(split.validation_photo_entries,bundle.transform),batch_size=256,shuffle=False,num_workers=4),photo=True)
        assert paths==tuple(str(e.path) for e in split.validation_sketch_entries)
        assert gallery.paths==tuple(str(e.path) for e in split.validation_photo_entries)
        assert np.array_equal(labels,np.asarray([e.label for e in split.validation_sketch_entries]))
        measured = measure('F2_MP',3600,features,paths,labels,gallery,output/'F2_MP')
        old = read(run/'probe_step3600.json')
        assert old['identities']['query_ids']==list(paths) and old['identities']['gallery_ids']==list(gallery.paths)
        delta = {k: abs(v-old['clean'][k]) for k,v in measured['retrieval']['mu_i']['metrics'].items() if k in old['clean']}
        assert len(delta)==5 and max(delta.values())<1e-7,delta
        with np.load(output/'F2_MP/mu_i_retrieval.npz') as raw:
            assert np.array_equal(raw['top_indices'],np.asarray(old['clean']['top_indices']))
            assert np.allclose(raw['full_ap'],old['clean']['average_precision_per_query'],rtol=0,atol=1e-7)
            assert np.allclose(raw['p200'],old['clean']['P@200_per_query'],rtol=0,atol=1e-7)
        values, angle_summary = spherical_angles(features)
        np.savez(output/'F2_MP/angles.npz',**values)
        assert before==_state_hash(model) and all(p.grad is None for p in model.parameters())
        measured.update(angles=angle_summary,checkpoint_sha256=expected,model_state_before=before,
                        model_state_after=_state_hash(model),main_head_replay_deltas=delta,
                        main_head_top_indices_exact=True,encoding='new_CUDA_no_update')
        report['arms']['F2_MP']=measured
        # Reuse already measured control vectors. No control model inference or retraining.
        prior = read(CACHE/'summary.json')
        parent = read(CACHE/'parent_verification.json')
        independent = read(CACHE/'independent_cpu/receipt.json')
        assert parent['status']==independent['status']=='PASS' and prior['status']=='COMPLETE'
        assert sha(CACHE/'summary.json')==parent['receipt_sha256']['head_summary']
        assert sha(CACHE/'independent_cpu/receipt.json')==parent['receipt_sha256']['independent_cpu']
        report['control_evidence']={'summary':str(CACHE/'summary.json'),'summary_sha256':sha(CACHE/'summary.json'),
                                    'independent_receipt_sha256':sha(CACHE/'independent_cpu/receipt.json'),'files':{}}
        for arm in ('F2','R0','S0','S0_best_clean'):
            for name in ('features.npz','identities.json'):
                path=CACHE/arm/name
                digest=sha(path)
                assert digest==independent['file_hash_manifest'][str(path.relative_to(ROOT))]
                report['control_evidence']['files'][str(path)]=digest
            ids=read(CACHE/arm/'identities.json')
            assert ids['query_paths']==list(paths) and ids['gallery_paths']==list(gallery.paths)
            item=prior['arms'][arm]
            with np.load(CACHE/arm/'features.npz') as raw:
                assert np.array_equal(raw['labels'],labels) and np.array_equal(raw['gallery_labels'],gallery.labels.numpy())
                if arm=='F2':
                    _, item['angles']=spherical_angles(raw)
            item['encoding']='reused_SHA_verified_cache;no_new_control_encoder'
            report['arms'][arm]=item
        report.update(status='COMPLETE',elapsed_seconds=time.monotonic()-start,clip_identity=clip)
        write_json(output/'summary.json',report)
        print(json.dumps({'status':'COMPLETE','output':str(output),'MP':measured['retrieval']['mu_i']['metrics'],'angles':angle_summary}),flush=True)
    except Exception:
        report.update(status='FAIL',traceback=traceback.format_exc())
        write_json(output/'failure.json',report)
        raise


if __name__=='__main__':
    main()

"""Fixed-batch gradient measurements from pilot checkpoints; never selects lambda."""
import hashlib
import json
from pathlib import Path

from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader

import spica.train_alignment as train
from spica.models.checkpoint import load_prompt_checkpoint

OUT = Path(__file__).resolve().parent
RUNS = OUT / 'fixed_attempt/pilot'
calibration = json.loads((OUT / 'fixed_attempt/calibration/calibration.json').read_text())
args = OmegaConf.create(calibration['calibration_config'])
train._seed(int(args.seed))
device = torch.device('cuda')
data = train.load_data_config(train._path(args.data_config))
split, names, split_identity, _ = train._load_split(data, args)
clip = train.load_frozen_clip(model_name=args.model_name, pretrained=args.pretrained, device=device)
model = train.FrozenPromptModel(clip.encoder.model.visual, prompt_length=args.visual_prompt_length).to(device)
model.eval()
train_names = {key: names[key] for key in split.train_class_ids}
text_bank = train.SoftPromptTextBank(clip.encoder, clip.tokenizer, train_names, prompt_length=args.soft_prompt_length).to(device)
hard_text = train.encode_class_text_bank(clip.encoder, clip.tokenizer, train_names, prompt_template=args.prompt_template)
hard_values, hard_labels = hard_text.embeddings.to(device), hard_text.labels.to(device)
assert train._state_hash(model) == calibration['initial_model_state_hash']
assert train._state_hash(text_bank) == calibration['initial_text_bank_state_hash']
assert split_identity['sha256'] == calibration['split_identity_hash']
dataset = train.MultiPositiveRetrievalTrainDataset(split.train_sketch_entries, split.train_photo_entries, clip.transform, clip.transform, num_positive_photos=args.num_positive_photos)
sampler = train.MatchedClassBatchSampler([e.label for e in split.train_sketch_entries], classes_per_batch=args.classes_per_batch, samples_per_class=args.sketches_per_class, seed=args.seed, batches_per_epoch=args.batches_per_epoch)
generator = torch.Generator().manual_seed(args.seed)
loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0, generator=generator)
fixed_rng = train.capture_rng_state(generator)
iterator = iter(loader)
batches = [next(iterator) for _ in range(args.calibration_batches)]
identity = train._calibration_batch_identity(batches, 0)
assert identity['sha256'] == calibration['calibration']['fixed_batch_identity']['sha256']
frozen_clip_hash = train._state_hash(clip.encoder.model)
assert all(not p.requires_grad for p in clip.encoder.model.parameters())
assert not hard_values.requires_grad
parameters = (model.sketch_prompt, model.photo_prompt, text_bank.context)
names = ('sketch_prompt', 'photo_prompt', 'soft_text_context')
records = []
for arm in ('R', 'MD', 'MS'):
    result = json.loads((RUNS / arm / 'run_result.json').read_text())
    args = OmegaConf.create(result['resolved_config'])
    weight = float(args.lambda_alignment_mean)
    for step in (0, 500, 1800):
        row = next(h for h in result['history'] if h['training_global_step'] == step)
        path = Path(row['checkpoint'])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == row['checkpoint_sha256']
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        load_info = load_prompt_checkpoint(model, checkpoint, expected_config=result['resolved_config'])
        text_bank.load_state_dict(checkpoint['soft_prompt_state_dict'])
        assert train._state_hash(model) == checkpoint['model_state_hash']
        # Sentinel existing gradients/mixed flags exercise restoration on real CLIP.
        model.eval()
        text_bank.train()
        for p in parameters:
            p.grad = torch.ones_like(p)
        before_model = train._clone_module_state(model)
        before_text = train._clone_module_state(text_bank)
        before_flags = train._training_flags(model, text_bank)
        before_grads = train._parameter_grads(model, text_bank)
        before_rng = train.capture_rng_state(generator)
        if arm == 'R' and step == 0:
            train.restore_rng_state(fixed_rng, generator)
            verified = train._calibrate_mean_alignment(model, text_bank, hard_values, hard_labels, loader, sampler, generator, args, device)
            assert verified['status'] == 'VALID'
            assert verified['calibration']['state_restoration_verified']
            assert verified['calibration']['lambda_alignment_mean'] == calibration['calibration']['lambda_alignment_mean']
            assert verified['calibration']['fixed_batch_identity']['sha256'] == identity['sha256']
            assert all(torch.equal(p.grad, torch.ones_like(p)) for p in parameters)
            (OUT / 'raw_metrics/calibration_real_existing_gradient_restoration.json').write_text(json.dumps({'real_model': True, 'existing_gradients_nonzero': True, 'mixed_train_eval_flags': True, 'same_initialization_and_fixed_batches': True, 'payload': verified}, indent=2)+'\n')
            train.restore_rng_state(before_rng, generator)
        measured = []
        for batch in batches:
            forward_rng = train.capture_rng_state()
            policies = {}
            detached_sketch = None
            for policy in ('detached', 'symmetric'):
                train.restore_rng_state(forward_rng)
                model.train()
                rank, cls, _, align = train._batch_objectives(model, text_bank, hard_values, hard_labels, batch, args, device, alignment_mean_weight=1.0, alignment_covariance_weight=0.0, alignment_target_gradient=policy)
                objectives = (args.lambda_rank * rank, args.lambda_cls * cls, align.mean)
                gradients = [torch.autograd.grad(loss, parameters, retain_graph=i < 2, allow_unused=True) for i, loss in enumerate(objectives)]
                dense = [[torch.zeros_like(p) if g is None else g.detach() for p, g in zip(parameters, group, strict=True)] for group in gradients]
                base = [a+b for a,b in zip(dense[0], dense[1], strict=True)]
                groups = {}
                for i, name in enumerate(names):
                    bn = float(base[i].norm())
                    an = float(dense[2][i].norm())
                    ratio, reason = train._weighted_ratio_or_reason(an, bn, weight)
                    cosine, cosine_reason = train._cosine_or_reason(base[i], dense[2][i], bn, an)
                    groups[name] = dict(rank_gradient_norm=float(dense[0][i].norm()), cls_gradient_norm=float(dense[1][i].norm()), base_gradient_norm=bn, mean_gradient_norm=an, weighted_mean_to_base_ratio=ratio, ratio_reason=reason, cosine_with_base=cosine, cosine_reason=cosine_reason)
                policies[policy] = dict(raw_rank_loss=float(rank), raw_cls_loss=float(cls), raw_mean_loss=float(align.mean), weighted_rank_loss=float(objectives[0]), weighted_cls_loss=float(objectives[1]), weighted_mean_loss=weight*float(align.mean), gradients=groups)
                if policy == 'detached':
                    assert groups['photo_prompt']['mean_gradient_norm'] == 0.0
                    detached_sketch = dense[2][0].clone()
                else:
                    diff = float((detached_sketch-dense[2][0]).abs().max())
                    assert diff <= 1e-6
                    policies['sketch_gradient_max_abs_difference'] = diff
            assert policies['detached']['raw_mean_loss'] == policies['symmetric']['raw_mean_loss']
            measured.append(policies)
        train._restore_training_flags(before_flags)
        train._restore_parameter_grads(before_grads)
        train.restore_rng_state(before_rng, generator)
        assert train._module_state_matches(model, before_model)
        assert train._module_state_matches(text_bank, before_text)
        assert train._rng_matches(before_rng, train.capture_rng_state(generator))
        assert all(m.training == flag for m, flag in before_flags)
        assert all((p.grad is None and g is None) or (p.grad is not None and g is not None and torch.equal(p.grad, g)) for p, g in before_grads)
        torch.cuda.synchronize()
        assert train._state_hash(clip.encoder.model) == frozen_clip_hash
        record = dict(arm=arm, step=step, checkpoint=str(path.resolve()), checkpoint_sha256=digest, checkpoint_loading=load_info, fixed_batch_identity_sha256=identity['sha256'], run_lambda=weight, run_policy=args.alignment_target_gradient, diagnostic_only=True, lambda_selection_performed=False, state_restoration_verified=True, frozen_full_clip_state_sha256=frozen_clip_hash, frozen_full_clip_unchanged=True, hard_text_anchor_requires_grad=False, batches=measured)
        records.append(record)
        destination = OUT / 'raw_metrics' / f'offline_gradients_{arm}_{step}.json'
        destination.write_text(json.dumps(record, indent=2)+'\n')
        print(arm, step, 'PASS', flush=True)
(OUT / 'raw_metrics/offline_gradients_index.json').write_text(json.dumps([{'arm':r['arm'],'step':r['step'],'checkpoint_sha256':r['checkpoint_sha256']} for r in records], indent=2)+'\n')

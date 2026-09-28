"""Four train-only batches at fixed initialization, zero optimizer updates.

Choose lambda so median(lambda * ||g_KD|| / ||g_retrieval||) = 0.1
on all Student parameters. This is not an optimum, per-batch cap, or Adam ratio.
Raw evidence stays MEASURED_PENDING_REVIEW; a separate CPU review produces the
receipt accepted by --kd-calibration. No W&B or official image evaluation.
"""

import argparse
import hashlib
import json
import math
import random
import shutil
import statistics
from pathlib import Path

import torch
from torchvision import transforms

import spica.train_student as t


def select_lambda(ratios):
    if len(ratios) != 4 or any(not math.isfinite(x) or x <= 0 for x in ratios):
        raise ValueError("Need four finite positive KD/retrieval gradient ratios")
    return 0.1 / statistics.median(ratios)


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def measure(args):
    output = args.output.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    data = t.load_data_config(t.ROOT / 'configs/data/tuberlin_220_30.yaml')
    source = t.source_files()
    manifests = {str(path): t.sha256(path) for split in (data.train, data.test)
                 for path in (split.sketch_manifest, split.photo_manifest, split.class_map)}
    output.mkdir(parents=True)
    for name in source:
        dest = output / 'source' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(t.ROOT / name, dest)
    t.write_json(output / 'runtime.json', {'status': 'MEASURING_ZERO_UPDATES'})
    try:
        random.seed(t.SEED)
        torch.manual_seed(t.SEED)
        torch.cuda.manual_seed_all(t.SEED)
        bundle = t.load_frozen_clip(pretrained=str(args.clip_weights.absolute()), device='cuda')
        photo = bundle.encoder
        student = t.SketchStudent().to('cuda').train()
        normalize = bundle.transform.transforms[-1]
        if not isinstance(normalize, transforms.Normalize):
            raise TypeError('Expected pinned CLIP normalization')
        teacher = t.SketchTeacher(photo.model.visual, normalize.mean, normalize.std,
                                  visual_prompt=True).to('cuda').eval()
        sketch_transform = transforms.Compose(t.build_clip_eval_transform().transforms[:-1])
        loader = t.build_retrieval_train_loader(data, sketch_transform, bundle.transform,
                    batch_size=t.BATCH_SIZE, num_workers=4, pin_memory=True, drop_last=False)
        before = {'student': t.state_sha(student), 'teacher': t.state_sha(teacher), 'photo': t.state_sha(photo)}
        parameters = dict(student.named_parameters())
        rows = []
        artifacts = {}
        stream = iter(loader)
        for index in range(4):
            batch = next(stream)
            sketches = batch['sketch'].cuda(non_blocking=True)
            labels = batch['label'].cuda()
            photos = torch.cat((batch['positive_photo'], batch['negative_photo'])).cuda()
            photo_labels = torch.cat((batch['label'], batch['negative_label'])).cuda()
            with torch.no_grad():
                targets = photo(photos)
            # Preserve the actual teacher attention path (grad enabled), but detach KD.
            teacher_query = teacher(sketches).detach()
            query = student(sketches)
            retrieval = t.multi_positive_loss(query, targets, labels, photo_labels)
            kd = t.cosine_distillation(query, teacher_query)
            gr = torch.autograd.grad(retrieval, tuple(parameters.values()), retain_graph=True)
            gk = torch.autograd.grad(kd, tuple(parameters.values()))
            assert all(torch.isfinite(g).all() for g in gr + gk)
            gr = {n: g.detach().cpu() for n, g in zip(parameters, gr)}
            gk = {n: g.detach().cpu() for n, g in zip(parameters, gk)}
            nr = math.sqrt(sum(g.double().square().sum().item() for g in gr.values()))
            nk = math.sqrt(sum(g.double().square().sum().item() for g in gk.values()))
            dot = sum((gr[n].double() * gk[n].double()).sum().item() for n in gr)
            row = {'batch': index, 'retrieval_loss': retrieval.item(), 'kd_loss': kd.item(),
                   'retrieval_grad_norm': nr, 'kd_grad_norm': nk, 'ratio_kd_to_retrieval': nk / nr,
                   'gradient_cosine': dot / (nr * nk)}
            rows.append(row)
            payload = {'retrieval_gradients': gr, 'kd_gradients': gk,
                       'query': query.detach().cpu(), 'teacher': teacher_query.cpu(),
                       'photos': targets.cpu(), 'labels': labels.cpu(), 'photo_labels': photo_labels.cpu(),
                       'sketch_input_sha256': tensor_sha(sketches), 'photo_input_sha256': tensor_sha(photos),
                       'paths': {k: list(batch[k]) for k in ('sketch_path', 'positive_photo_path', 'negative_photo_path')},
                       'source_image_sha256': {str(p): t.sha256(p) for key in
                          ('sketch_path', 'positive_photo_path', 'negative_photo_path') for p in batch[key]}}
            file = output / f'batch_{index}.pt'
            torch.save(payload, file)
            artifacts[file.name] = t.sha256(file)
            print(json.dumps(row), flush=True)
            del query, retrieval, kd, gr, gk, payload
        after = {'student': t.state_sha(student), 'teacher': t.state_sha(teacher), 'photo': t.state_sha(photo)}
        assert before == after
        assert all(p.grad is None for m in (student, teacher, photo) for p in m.parameters())
        assert t.source_files() == source
        assert all(t.sha256(p) == h for p, h in manifests.items())
        coefficient = select_lambda([row['ratio_kd_to_retrieval'] for row in rows])
        result = {'status': 'MEASURED_PENDING_REVIEW', 'source_sha256': source,
                  'manifest_sha256': manifests, 'clip_sha256': t.sha256(args.clip_weights),
                  'clip_weights': str(args.clip_weights.absolute()), 'seed': t.SEED,
                  'batch_size': t.BATCH_SIZE, 'visual_prompt_tokens': 4, 'batches': rows,
                  'rule': t.KD_CALIBRATION_RULE,
                  'data_config_sha256': t.sha256(t.ROOT / 'configs/data/tuberlin_220_30.yaml'),
                  'lambda_kd': coefficient, 'weighted_ratios': [coefficient*r['ratio_kd_to_retrieval'] for r in rows],
                  'state_before': before, 'state_after': after, 'optimizer_updates': 0,
                  'test_images_opened': 0, 'artifacts_sha256': artifacts}
        t.write_json(output / 'measurement.json', result)
        t.write_json(output / 'runtime.json', {'status': 'MEASURED_PENDING_REVIEW', 'optimizer_updates': 0})
        print('lambda_KD', coefficient, 'MEASURED_PENDING_REVIEW', flush=True)
    except BaseException as error:
        t.write_json(output / 'runtime.json', {'status': 'FAILED_NO_RETRY', 'error': repr(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clip-weights', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        ratios = [0.1, 0.2, 0.5, 0.8]
        coefficient = select_lambda(ratios)
        assert math.isclose(statistics.median(coefficient*r for r in ratios), 0.1)
        for bad in ([0.1], [0, 1, 2, 3], [float('nan'), 1, 2, 3], [float('inf'), 1, 2, 3]):
            try:
                select_lambda(bad)
            except ValueError:
                pass
            else:
                raise AssertionError(bad)
        print('PASS: median weighted gradient ratio=0.1; invalid input rejected')
    else:
        if args.clip_weights is None or args.output is None:
            parser.error('--clip-weights and --output are required for measurement')
        measure(args)


if __name__ == '__main__':
    main()

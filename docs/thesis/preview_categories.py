"""CPU-only official-test examples; no model, retrieval, or paired-instance claim.
Run from repo root: CUDA_VISIBLE_DEVICES='' .venv/bin/python docs/thesis/preview_categories.py
Writes a fresh previews directory, never modifies source images.
"""
import csv
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys
from collections import Counter

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from spica.data.coupled_benchmark import load_benchmark_protocol
from spica.data.transforms import build_clip_eval_transform

OUT = Path(__file__).resolve().parent / 'dataset_statistics' / 'previews'
DATASETS = ['sketchy_104_21', 'tuberlin_220_30', 'quickdraw_80_30']


def extremes(counts, names):
    largest = min(counts, key=lambda label: (-counts[label], names[label]))
    smallest = min((label for label in counts if label != largest), key=lambda label: (counts[label], names[label]))
    return largest, smallest


def main():
    # Equal-count categories must still produce two distinct, deterministic classes.
    assert extremes({0: 10, 1: 10, 2: 10}, {0: 'b', 1: 'a', 2: 'c'}) == (1, 0)
    assert extremes({0: 3, 1: 12, 2: 1}, {0: 'b', 1: 'a', 2: 'c'}) == (1, 2)
    if OUT.exists():
        raise FileExistsError(f'Keep existing previews; fresh destination required: {OUT}')
    geometry = build_clip_eval_transform().transforms[:2]
    font = ImageFont.load_default(size=20)
    title_font = ImageFont.load_default(size=26)
    rows, selections, canvases = [], [], {}
    with (OUT.parent / 'tables/class_counts_all.csv').open() as f:
        counted = {(r['dataset'], r['split'], r['modality'], int(r['label'])): int(r['count']) for r in csv.DictReader(f)}
    for dataset in DATASETS:
        protocol = load_benchmark_protocol(ROOT / 'configs/data' / f'{dataset}.yaml')
        split = protocol.test
        photos = Counter(e.label for e in split.photo_entries)
        sketches = Counter(e.label for e in split.sketch_entries)
        for modality, counts in [('photo', photos), ('sketch', sketches)]:
            assert all(counted[(dataset, 'test', modality, label)] == count for label, count in counts.items())
        labels = extremes(photos, split.class_names)
        # Each class: sketch column (2 rows) + photo columns (4 x 2).
        canvas = Image.new('RGB', (1216, 1230), '#f1f3f5')
        draw = ImageDraw.Draw(canvas)
        draw.text((16, 14), dataset + ' | official test', fill='#172b4d', font=title_font)
        draw.text((16, 48), 'Category examples, NOT paired instances or retrieval results', fill='#334155', font=font)
        for block, (selection, label) in enumerate(zip(['MOST PHOTOS', 'FEWEST PHOTOS'], labels)):
            base = 92 + block * 554
            name = split.class_names[label]
            draw.text((16, base), f'{selection}: {name} | {photos[label]} photos, {sketches[label]} sketches', fill='#172b4d', font=title_font)
            selections.append({'dataset': dataset, 'split': 'test', 'selection': selection, 'label': label,
                               'class_name': name, 'photo_count': photos[label], 'sketch_count': sketches[label],
                               'protocol_identity_sha256': protocol.identity['sha256']})
            for modality, amount in [('sketch', 2), ('photo', 8)]:
                entries = sorted((e for e in getattr(split, f'{modality}_entries') if e.label == label), key=lambda e: e.path.relative_to(protocol.root).as_posix())
                assert len(entries) >= amount
                entries = entries[:amount]
                assert len({e.path for e in entries}) == amount
                for index, entry in enumerate(entries):
                    raw = entry.path.read_bytes()
                    with Image.open(BytesIO(raw)) as image:
                        rgb = image.convert('RGB')
                    tile = geometry[1](geometry[0](rgb))
                    assert tile.size == (224, 224)
                    yrow, xcol = (index, 0) if modality == 'sketch' else (index // 4, 1 + index % 4)
                    x, y = 16 + xcol * 240, base + 64 + yrow * 258
                    canvas.paste(tile, (x, y))
                    draw.text((x, y - 26), f'{modality} {index + 1}', fill='#1d4ed8' if modality == 'sketch' else '#9a3412', font=font)
                    rows.append({'dataset': dataset, 'split': 'test', 'selection': selection, 'label': label,
                                 'class_name': name, 'modality': modality, 'sample_index': index + 1,
                                 'source_path': entry.path.relative_to(ROOT).as_posix(),
                                 'raw_file_sha256': hashlib.sha256(raw).hexdigest(),
                                 'transformed_rgb_sha256': hashlib.sha256(tile.tobytes()).hexdigest(),
                                 'tile_x': x, 'tile_y': y})
        draw.text((16, 1200), 'RGB | bicubic short-side resize + center crop 224x224 | lexicographic paths', fill='#334155', font=ImageFont.load_default(size=16))
        canvases[dataset] = canvas
    assert len(rows) == 60 and sum(r['modality'] == 'sketch' for r in rows) == 12
    assert len({r['source_path'] for r in rows}) == 60
    OUT.mkdir()
    for dataset, canvas in canvases.items():
        canvas.save(OUT / f'{dataset}.png')
    for filename, data in [('image_manifest.csv', rows), ('selected_classes.csv', selections)]:
        with (OUT / filename).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)
    receipt = {'status': 'PASS_PREVIEW_ONLY', 'images_decoded': 60, 'sketches': 12, 'photos': 48,
               'selection': 'official test class max/min photo counts; ties by class name; first lexicographic paths',
               'geometry': 'production RGB bicubic short-side resize and center-crop224',
               'no_model_or_retrieval_or_source_image_changes': True,
               'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'figures_sha256': {d: hashlib.sha256((OUT / f'{d}.png').read_bytes()).hexdigest() for d in DATASETS}}
    (OUT / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    (OUT / 'README.md').write_text('# Preview category official test\n\nMỗi dataset chọn class có nhiều photo nhất và ít photo nhất; mỗi class gồm **2 sketch + 8 photo**. Cột đầu là sketch, bốn cột còn lại là photo.\n\nẢnh chọn theo thứ tự đường dẫn, không chọn tay theo độ đẹp. Tất cả ở224×224 theo preprocessing của mô hình; ảnh nguồn không bị sửa. Đây là ví dụ cùng category, **không phải cặp instance hay kết quả retrieval**. Mẫu nhỏ này không đại diện cho toàn bộ phân bố category.\n\n' + '\n\n'.join(f'## {d}\n\n![{d}]({d}.png)' for d in DATASETS) + '\n\n[selected_classes.csv](selected_classes.csv) lưu counts/protocol; [image_manifest.csv](image_manifest.csv) lưu path/SHA/vị trí từng ảnh. [Receipt](receipt.json). Trước khi xuất bản lại ảnh trong đồ án/công khai, kiểm tra yêu cầu trích nguồn và quyền sử dụng của dataset.\n')
    print('PASS: 3 sheets, 6 distinct classes, 12 sketches + 48 photos; source images unchanged')


if __name__ == '__main__':
    main()

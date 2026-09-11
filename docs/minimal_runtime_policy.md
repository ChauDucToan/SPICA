# Quy tắc runtime chung — các run mới

User chốt 2026-09-11: tối ưu **code/config/I/O**, không đổi model, loss, train pool hoặc chạy training mới. Không xóa artifact lịch sử.

## 1. W&B

- Mặc định **metrics + config**, không tự upload model, checkpoint, table hoặc source archive.
- Tắt system statistics, machine metadata, code/Git capture, requirements và console capture qua `WandbExperiment` chung.
- Chỉ sáu đường theo `step_train`:
  - `cleaned/mAP@200`, `cleaned/mAP@all`, `cleaned/P@200`.
  - `masked/mAP@200`, `masked/mAP@all`, `masked/P@200`.
- `mAP@200 = mean(sum(P(k)*rel(k), k<=200) / min(R,200))`.
- `P@all = mean(R / gallery_size)` không phụ thuộc ranking: chỉ ghi summary, tách scope probe/full.
- Loss/LR/gradient vẫn lưu local để chẩn đoán, không tạo thêm biểu đồ W&B.
- W&B vẫn có `_step`, `_timestamp`, `_runtime` và telemetry nội bộ SDK; tắt system collection **không** có nghĩa xóa mọi metadata nội bộ W&B.
- Generic table/artifact API giữ tương thích nhưng mặc định bỏ qua yêu cầu upload. Chỉ `allow_artifacts=True` tường minh mới cho phép; không caller hiện tại nào tự bật. Không coi lời gọi bị bỏ qua là bằng chứng đã upload.
- Legacy training scalar namespaces bị lọc. Gõ sai metric trong `cleaned/` hoặc `masked/` bị từ chối. Không chỉnh sửa lịch sử metric của run cũ.

## 2. Probe: mỗi 5 optimizer updates, không phải validation

Đã tích hợp trong `src/spica/train_coupled_benchmark.py` cho run không phải smoke:

- 32 sketch của32lớp train đầu theo label; một query/lớp, chọn theo path ổn định.
- 256photos:8ảnh/lớp tương ứng. Gallery đủ relevant, IDs cố định.
- 32clean +9×32masked queries: fractions .25/.5/.75, seeds101/202/303, cùng production raster mask.
- Cache input tensors trênCPU một lần. **Tính lại features query/gallery bằng model hiện tại mỗi lần probe**; không tái dùng gallery embedding cũ khi prompts đã đổi.
- q-only, không predictor; galleryforward chia batch64. Dùng evaluator metric chung, không implementationAP riêng.
- Giữ RNG Python/NumPy/Torch/CUDA, module train/eval flags và gradients. Không lấybatch từ trainingiterator, không đổi train sampler.
- Scope bắt buộc `seen_train_probe_not_validation`; không chọn checkpoint/early stopping hoặc suy ra unseen generalization.
- `masked` là mean9conditions. Không ghi feature/ranking/checkpoint mỗi lần probe; chỉ `probe_metrics.jsonl` và sáu scalar.
- Với galleryprobe này, `P@all=8/256=.03125`, và P200 tối đa8/200=.04. **Không so trực tiếp P200/probe với full-benchmark P200.**
- Chu kỳ đúng5/10/15/...; finalstep không chia hết5 không giả thành một probe mới. Fullofficial chỉ chạy sau checkpointfinal.

Probe 320queryviews+256photos mỗi5updates vẫn có chi phí forward. Chưa benchmark GPU latency/overhead cho runtime này; chỉ xác nhận CPU đúnglogic. Không gọi nó miễn phí.

## 3. Full evaluation và W&B summary

`scripts/train.py` dùng Hydra làm entrypoint mỏng, gọi **trainer hiện có**, rồi evaluator hiện có khi train hoàn tất. Smoke không tự chạy officialeval.

Final evaluation giữ raw local summary/NPZ/features hiện có, không upload tensor. Chỉ các run `tracking_policy=minimal_v1`, `wandb_mode=online` mới publish vào **summary của đúng training run**:

- `final/cleaned/{mAP@200,mAP@all,P@200,P@all}`.
- `final/masked/{mAP@200,mAP@all,P@200,P@all}`.
- step/checkpointSHA/scopefinal.

Không trộn fullofficial vào sáu đường probe. Trước cập nhật online kiểm tra run source/arm/dataset/horizon. Lượt CPU hiện chỉ mock final API, chưa xác nhận upload online thật. Publication lỗi giữ raw evaluation và báo lỗi, không retrain/re-evaluate tự động.

## 4. Lưu model local

- Production chỉ một `checkpoint_latest.pt`, ghi đè atomic mỗi100updates và cuối; manifest `checkpoint_latest.json` giữ step/SHA.
- Cadence checkpoint **khác** cadence probe. Không model mới mỗi5steps, không checkpointstep0 cho production.
- Chỉ lưu trainable model state, optimizer/scheduler/RNG + config và backbone cacheID/SHA. Eval dựng lại frozen state từ cachedCLIP đã SHA-check rồi kiểm tra toànmodelstatehash.
- Student visual của model hiện tại trainable rộng; modelstate và AdamWmoments vẫn đáng kể. Không thể nén thành vài prompt mà vẫn giữ model/optimizer hiện tại.
- Checkpoint coupled cũ **đã loại original_clip**; gain chính lần này là không upload, giảm số bản và bỏ frozenstate còn dư, không tuyên bố trước đây mọi checkpoint chứa toàn originalCLIP.
- Ghi thất bại giữ checkpoint cũ; tempfile thất bại được dọn. Checkpoint/JSON là hai lầnatomicreplace riêng; nếu crash giữa hai lần, SHA mismatch sẽ reject, không im lặng phục hồi.
- Có state để phục hồi model/optimizer/scheduler, **chưa thêm resume CLI hoặc exact data-stream resume**: worker/iterator sampler state chưa được lưu. `exact_data_stream_resume=false`.
- Smoke tiếp tục step0+step2 để phục vụ gate; không đổi artifact lịch sử.

## 5. Config chung, không framework mới

`configs/runtime/minimal.yaml` chỉ có:

```yaml
probe_every: 5
checkpoint_every: 100
```

Hydra compose cùnggroup qua `configs/train_coupled_benchmark.yaml`. Userconfig chỉ dataset/device/W&B/λSREF/smoke; model/loss/schedule dùng protocol hiện có, không sao chép thành hàng loạt Hydra flags. Config chi tiết/identity suy ra nằm local trong `resolved_config.json`, không buộc user nhập hoặc upload toàn bộ lên W&B.

Xem config, **không chạy train**:

```bash
PYTHONPATH=src .venv/bin/python scripts/train.py --cfg job
```

Khi có cấp phép training mới, entrypoint `scripts/train.py` sẽ train rồi finaleval. Dataset mặc địnhTU, λSREF mặc địnhnull nghĩa là MP-Q gốc; **không tự chọn λ từ run cũ**. YAML cho phép đổi cadence để tái sử dụng nhưng default đã khóa5/100. CLIargparse cũ còn dùng được.

## 6. Giới hạn validation và phạm vi refactor

Giữ validation tại input/config, dataset boundary, checkpointload, finite loss/gradient trước optimizer update và kiểm tra kết thúc. Không thêm source/model SHA mỗiprobe; metric dùng evaluator chung. Không bỏ failure safety chỉ để giảm dòngcode.

Chưa migrate tất cả trainer/launcher historical sang cadence mới hoặc xóa toàn bộ config lịch sử. SharedW&Bpolicy áp dụng các caller dùng `WandbExperiment`; probe/rollingcheckpoint/finalsummary mới tích hợp ở coupled officialTU/QuickDraw lane. Không dùng launcher/gate source-hash cũ để train source mới: cần freshgates nếu được duyệt chạy thật.

## CPU checks

Root `outputs/minimal_runtime_cpu_20260911/`:

- 19pytestPASS: tiny10update actualtrainer, probe5/10, rolling4/8/10 trongtest, đúngonecheckpoint, safeweights-onlymodel+optimizer+schedulerrestore, productionevalmodelrestore trainable-only, unsupportedformatreject, atomicfailurepreservesoldfile, Hydra/defaultvalidation, metric/RNG/mode/mask contracts, legacycheckpointtests.
- `sref_regression/receipt.json`:6groupsPASS, archivedloss/gradient parity vàtiny2updates/SREFmetadata.
- `check_offline.py`, `wandb_offline/receipt.json`: **W&B0.28.1 backend thật ởoffline**, đọcprotobufjournal:1historyrecord/sáumetric+axis,0statsrecords,0artifactrecords, khôngmachine metadata/requirementsfile. SDKtelemetry/timing còn tồn tại.
- Ruff/diffcheck/Hydra `--cfg job` PASS. Không GPU, pretrainedforward, officialimageevaluation, networkupload, newrun/seed hoặc model/loss change.

Independent review không sửa files: một số đề xuất cho rằng pre-existing `.gitignore`/V1verifierdiff phảirevert là sai phạm vi; parent giữ nguyên. Parent đã sửa unsupportedcheckpointformat, finalrunidentity, typo metric rejection và atomicfailure trước commit. Không revert unrelatedwork.

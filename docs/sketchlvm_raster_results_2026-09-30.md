# SketchLVM paper-category + che nét theo raster — kết quả

## Hoàn tất

Fresh **60 epochs / 14.460 updates / 924.000 lượt sketch**, seed42/B64; train và final test exit0, CPU báo cáo exit0. Runner 2026-09-30 14:00:36→15:06:34 UTC (train + test + báo cáo, không phải benchmark latency). Không còn tiến trình campaign/GPU compute tại lúc kiểm tra. Không retry, resume, checkpoint selection, seed/arm bổ sung hoặc push.

- Protocol đã duyệt: [SketchLVM raster](sketchlvm_raster_protocol_2026-09-30.md).
- Bổ sung theo yêu cầu user: [manifest upstream + kiểm tra lại FP64](sketchlvm_manifest_fp64_recheck_2026-09-30.md). Cả6file khớp public ZSE-SBIR release;70/class không phải local subsampling. FAIL1/128 vẫn giữ, tail-swap FP32 scalar khớp nhưng native CUDA tail rank chưa được lưu. User chọn giữ run, không train trùng.
- Root: `outputs/sketchlvm_tu_60epochs_raster_20260930/`; launch receipt và execution JSON ở cạnh root.
- Training source HEAD: **`96f161e3cced096d38845cc289de0a356c8a5057`**, archive 51 source/config/docs (bao gồm user-dirty AGENTS.md đã giữ nguyên), không đồng nhất HEAD với clean worktree.
- Final checkpoint `sketchlvm_latest.pt`: SHA256 **`25f6da5f01ab81daeafb4190000f0c6701615070bd06992de15b4bba5121c48c`**.
- Config SHA256 `8c324f7f675e9f87c3a670b7b6eaa750d37bb619d3a15e25ec966c56a370b6a7`.
- Final raster summary SHA256 `97852a494a7f88615f6a41b122408df6b71ded550dee2ebc32065c031c715ce0`.
- W&B [jk42jjtr](https://wandb.ai/a-cctest05187-erd/sbir/runs/jk42jjtr), **finished**; read-only API xác nhận một final row, ba clean metrics + progress_percent=100 khớp local ở tolerance1e-12; không download artifact/remote write.

Đây là **paper-category reimplementation theo các lựa chọn user**, không phải exact reproduction của paper/public basic model. CLIP B32QuickGELU frozen base; sketch/photo có ba shallow prompts độc lập, append sau CLS+49 positioned patches trước native preLN, tất cả visual LayerNorm affines độc lập. **84.480 trainable parameters / 106 tensors**. Photo gallery dùng **photo prompts/LN đã train**. Loss cosine hinge margin.3 + .5×(sketch CE + positive-photo CE), negative photo chỉ triplet; frozen220seen text prototypes/native pretrained logit scale. Adam1e-5, không schedule/clip/augmentation. Existing Student/teacher models và artifacts không bị thay đổi.

## Full official TU-Berlin 220/30

2.400 queries (80/class), 27.989 gallery photos, 30 unseen fine-tuning classes; một gallery dùng chung mọi điều kiện. Labels chỉ phục vụ metric/metadata, không vào encoder hoặc unseen text bank. AP@200 denominator=min(R,200), P@200 denominator200. Bảng dưới dùng **%**; không diễn giải mAP như classification accuracy.

| Điều kiện | Full mAP | AP@200 | P@200 | Full-AP retention so clean |
|---|---:|---:|---:|---:|
| Clean | **51,6094%** | 57,6045% | 64,1121% | 100% |
| Xóa raster25% | **33,8355%** | 34,9183% | 42,8950% | 65,5607% |
| Xóa raster50% | **16,4955%** | 13,4910% | 20,4596% | 31,9623% |
| Xóa raster75% | **5,8392%** | 2,0002% | 5,5515% | 11,3143% |

| Raster | Mean Δ full AP (điểm %) | Median Δ (điểm %) | Query tăng / giảm / hòa |
|---|---:|---:|---:|
| 25% | −17,7739 | −7,9470 | 522 / 1878 / 0 |
| 50% | −35,1138 | −28,4241 | 227 / 2173 / 0 |
| 75% | −45,7701 | −45,9141 | 120 / 2280 / 0 |

Win/loss/tie threshold1e-6, median interpolated quantile.5. **Cả30/30 class mean full AP giảm ở từng mức25/50/75%**, mặc dù một số query riêng lẻ cải thiện. Mức75%: 95% queries giảm; horse −88,6907 điểm%, tractor −83,9055, bus −80,1362. Không lấy các query cải thiện làm checkpoint/model selection.

**Kết luận hạn chế:** cấu hình này nhạy mạnh với erasure raster; tại75% chỉ giữ11,31% clean full mAP. Không chứng minh prompt chỉ “đoán hình trung bình”, bottleneck nằm ở patch projection, Student xử lý geometry tốt hơn, hay model mất toàn bộ thông tin nét. Không so trực tiếp masked curves với các mask/systems khác chưa được đo cùng protocol.

## Mask thực tế

Resize bicubic/center crop224, **trước normalization**. Ink=mean RGB<.9; xóa first ceil(f*N) pixels theo row-major, trắng cả3channels, không di chuyển/thay đổi pixels còn lại. Đây là count-threshold proxy, không semantic part mask hoặc độ dày nét thật. Count fraction khác nhẹ requested fraction do ceil; “ink mass”=sum(1−mean RGB) trên threshold-qualified pixels.

Không có query blank theo N=0. MeanN=1153,0163 (min245, max4261). `blank_k` trong raw report nghĩa **zero erasure**, không blank sketch: clean có2400 k=0.

| Raster | Mean k | Mean fraction pixels thực xóa | Mean mass fraction | Min–max mass fraction |
|---|---:|---:|---:|---:|
| 25% | 288,6279 | 25,0378% | 25,0385% | 11,1594–34,7175% |
| 50% | 576,7579 | 50,0253% | 49,9602% | 33,5391–64,5193% |
| 75% | 865,1433 | 75,0381% | 74,9449% | 65,0173–86,4276% |

Cùng count fraction không bảo đảm cùng ink mass/semantic importance. Raw per-query/per-class/features/rankings và pixel/hash traces ở `raster_test/`; toàn bộ120class rows trong independent report.

## Kiểm chứng và FAIL được giữ

1. CUDA gate `outputs/sketchlvm_gate_20260930/`: actual trainer3updates64/64/prefix40, 106Adam states/212moments và whole-model/RNG restore exact; frozen base unchanged; independent saved-gradient/CE-logsumexp + Adam CPU replay; 128real train mask image/pixel/hash replays. Zero workers/prefix40 không chứng nhận production four-worker/epoch-tail stream. Miniature evaluator bypass final horizon/data counts, không official image test.
2. **Raw gate `CPU_AUDIT_RAW_FAIL` vẫn nguyên:** 1/128 full AP sai khác2.2660297567e-5 > declared2e-5. AP200/P200 row này khớp; top200 identical, CPU candidate tail ranks208/209 có cosine gap1.6078e-7. Native full tail ranking chưa được lưu, nên không gọi causal GPU crossing đã chứng minh. Không nâng tolerance, không encoder/GPU retry. User **explicitly chọn tiếp tục60 và công bố FAIL**; adjacent launch receipt bind approval + raw receipt SHAs. Predecision parent receipt `PRODUCTION_BLOCKED.../production_started:false` là trạng thái trước lựa chọn này, không mô tả campaign hiện tại.
3. Auto CPU report: `outputs/sketchlvm_report_20260930/{report.md,summary.csv,receipt.json}`.
4. Independent final review: `outputs/sketchlvm_final_review_20260930/{review.py,review.log,report.md,receipt.json}`; **9.600 saved top200 rows** AP200/P200 NumPy independent PASS max3.33e-16, all120class means/deltas PASS, 12aggregate means max2.78e-17; query/gallery path+label orders và51archive/6manifest SHA bindings. Full AP chỉ saved-scalar means, **không second full-gallery sort**. 9.600mask traces chỉ count/hash consistency, **không reopen official image/pixel replay**. Invariant model/RNG hashes là producer evidence, không independent encoder replay.
5. Parent `parent_verify.py`/`parent_receipt.json`: actual safeCPU final checkpoint60/14460/924000, config equality/hash, 106finiteadapters/84480params, 106finiteAdam states/212moments atstep14460, externalCLIPSHA/source snapshot exact trước post-report docs, actual before-after invariant hashes; scoped PASS **với raw gate FAIL retained**. Không final-model restore/GPU replay.
6. `online_readonly.py`/`online_receipt.json`: finished W&B/four final scalars scoped PASS.

**Training/evaluator raw status giữ `FINISHED_UNVERIFIED`; không fully verified.** Official test đã quan sát trước nhiều lần, one seed, không blind holdout/SOTA/significance hoặc guaranteed unseen CLIP-pretraining categories. Report commit sau finish không phải training snapshot. Không tự train thêm, tune, seed/search, rerun evaluator, cleanup hoặc push.

## Code và CLI evaluator riêng

- Model: `src/spica/models/sketchlvm.py`.
- Fixed60 trainer: `python -m spica.train_sketchlvm --clip-weights PATH --output FRESH --wandb-mode online`.
- Separate evaluator: `python -m spica.evaluate_sketchlvm_raster --run RUN --output FRESH` (yêu cầu CUDA, final60 checkpoint và archived-source bindings). Đây là API đã implement/test; **không yêu cầu tự chạy lại** run đã hoàn tất.

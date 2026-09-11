# TU-Berlin: MP-Q + sketch frozen-reference — kết quả cuối 1189

## Kết luận

**Ablation SREF cải thiện cả clean và masked so với MP-Q trên TU-Berlin ở cùng bước1189.** Clean full mAP `.4396614905633032 → .4486155837250408` (+`.008954093161737564`, **+0.895409 điểm phần trăm**, +2.0366% tương đối). Masked macro `.32638461547718317 → .3339411972534705` (+`.007556581776287319`, **+0.755658 điểm**, +2.3152% tương đối). Cả50 ô (5metric × clean+9mask) đều tăng; không phải mọi query/lớp đều tốt hơn.

Đây là kết quả tích cực cho một cấu hình/seed TU, không chứng minh quên kiến thức, optimalλ, semantic-part understanding hay hiệu quả trên Sketchy/QuickDraw. Official baseline đã được xem trước khi thiết kế ablation: **không phải blind holdout**. λ chỉ dùng calibration train-only; không chọn checkpoint/loss/horizon theo officialtest. Không tự promote default hay chạy thêm arm/seed.

## Run → source → checkpoint → metric

- [Protocol và gates](coupled_sketch_ref_tuberlin_campaign_2026-09-11.md); [baseline TU/QuickDraw](coupled_official_benchmarks_results_2026-09-11.md).
- Root `outputs/coupled_sketch_ref_execution_20260911T061500Z/` (tên plannedroot không phải giờ bắt đầu).
- Training HEAD `1fba48da3ed6d795108fb89db27783ffb69c7496`; source597files SHA `b420abacf2a44c29312348edf58ae7aedc23a4aa7f1bf9868e313a37177ecd29`.
- W&B [`k48u33m7`](https://wandb.ai/a-cctest05187-erd/spica/runs/k48u33m7), finished;120historyrows/119lossbearingrows.
- Final `runs/tuberlin_220_30/checkpoint_step1189.pt`, SHA `c7dfcc6a816ff43a1b43b23ac598137b5ea93f2d60ca6f84e0e137df9fc9a7f6`.
- Fixed baseline TU1189: `outputs/coupled_benchmark_execution_20260910T164500Z/`, checkpoint SHA `a39a0a6a6107566afa43af44410f25d40ab8b36d28d75ac6a4bf0c7901752460`, W&B`18wp9wty`. Không re-encode/retrain baseline trong lượt báo cáo.
- Start06:05:35Z, finish06:21:54Z, 2026-09-11: toàn runner **16phút19giây**, bao gồm setup/hash/save/train/eval. TrainCOMPLETE1189; train/eval exit0. Không còn process campaign khi chốt báo cáo.
- Rawruntime giữ `TRAIN_AND_OFFICIAL_EVAL_FINISHED_UNVERIFIED`; `results.json` giữ rawUNVERIFIED. Báo cáo và commit sau không phải training snapshot.

Arm `F2_MP_Q_SREF_OFFICIAL`: giữ mainMP(μ_I), inferenceq; thêm λ=`.12384240801936452` × instanceInfoNCEτ.07, **frozen masked-sketch teacher rows → clean studentq columns**. Reuse corruptedview/originalCLIP; không thêm studentforward, không sameclassnegativefilter. Teacher chỉ dùng trainingloss, q-only inference khôngteacher/label/predictor. Initialization,38048traintraces/masks và1190LRrows byteexact baseline; seed42,B32,1189updates,warmup59. Train220classes/15400sketches/176081photos; test30classes/2400queries/27989photos. Mỗi testclass80queries.

## Retrieval: full gallery

Các số trong bảng là [0,1], Δ là SREF−baseline. Mỗi severity là mean3maskseeds101/202/303; all9 là mean9conditions, không bao gồm clean.

| Điều kiện | Baseline fullmAP | SREF fullmAP | ΔfullmAP | Baseline P200 | SREF P200 |
|---|---:|---:|---:|---:|---:|
| Clean | .439661 | .448616 | +.008954 | .502854 | .513290 |
| Xóa25% | .396831 | .406076 | +.009245 | .447978 | .459834 |
| Xóa50% | .340742 | .348500 | +.007758 | .380323 | .389618 |
| Xóa75% | .241581 | .247248 | +.005667 | .261378 | .269090 |
| Mean9mask | .326385 | .333941 | +.007557 | .363226 | .372847 |

| Metric | Clean baseline | Clean SREF | Masked baseline | Masked SREF |
|---|---:|---:|---:|---:|
| FullmAP | .439661 | .448616 | .326385 | .333941 |
| P200 | .502854 | .513290 | .363226 | .372847 |
| AP200 prefix-positive | .529811 | .539585 | .384066 | .393677 |
| AP200 all-relevant | .119978 | .121685 | .086477 | .088277 |
| AP200 min(relevant,200) | .421831 | .432374 | .289237 | .297448 |

AP200 denominator khác nhau; không thay thế lẫn nhau hoặc tự so với con số paper. FullmAP dùng tất cảgallery, cùngclass là relevant, querymacro; mọiquery có relevantphotos.

| Mask | Seed101 fullmAP | Seed202 fullmAP | Seed303 fullmAP |
|---|---:|---:|---:|
| 25% | .406432 | .407127 | .404668 |
| 50% | .348133 | .349962 | .347405 |
| 75% | .246928 | .247718 | .247098 |

Tỷ lệ masked/clean baseline→SREF:25% **90.258→90.517%**,50% **77.501→77.683%**,75% **54.947→55.114%**. Tăng retention rất nhỏ; bằng chứng hiện tại chủ yếu là retrieval tốt hơn ở cả clean/masked, chưa phải thay đổi mạnh độ bền khi xóa. Cácmask/IDs/order byteexact baseline; mỗiarm dùng galleryembeddings của chính prompts đã học, không ép features giữa haiarm giống nhau.

## Phân bố theo query và lớp

Query comparisons dùng AP SREF−baseline, severity lấymean3seeds/query trước. Tăng/giảm dưới đây dùng ngưỡng±.001; vùngnhỏ bao gồm hai biên, ba nhóm không chồng lấn.

| Điều kiện | Tăng>.001 | Giảm<−.001 | Nhỏ | MedianΔ | Tăng>.05 | Giảm<−.05 |
|---|---:|---:|---:|---:|---:|---:|
| Clean | 1250 | 1050 | 100 | +.002225 | 441 | 272 |
| 25% | 1250 | 1049 | 101 | +.002244 | 450 | 267 |
| 50% | 1246 | 1027 | 127 | +.001974 | 386 | 248 |
| 75% | 1203 | 1043 | 154 | +.001049 | 325 | 239 |
| Mean9mask | 1252 | 1032 | 116 | +.002370 | 357 | 212 |

Không áp ngưỡng: clean1293/2400queries tăng (**53.875%**),1107giảm; maskedmacro1307tăng (**54.458%**),1093giảm. CleanmeanΔ+.008954 lớn hơnmedian+.002225; không chỉ fewwins nhưng gain không đều. CleanΔp10/p90 −.058604/+.082965. CleanAPmedian `.376161→.392376`;p10 `.047147→.049392`;p90 `.939950→.937096` (tail tốt nhất không tăng toàn bộ). Query khônghit trongtop200 giảm **144→121** (6.00→5.04%); không đồng nghĩa fullAP0.

**19/30 lớp clean tăng**, **16/30 lớp maskedmacro tăng**. Một số lớp có thay đổi lớn:

| Lớp | Clean baseline→SREF | Δclean | Δmaskedmacro |
|---|---:|---:|---:|
| penguin | .710857→.814066 | +.103209 | +.100241 |
| shoe | .177468→.247337 | +.069868 | +.035084 |
| rollerblades | .359509→.417739 | +.058230 | +.039774 |
| lighter | .459180→.514535 | +.055354 | +.057284 |
| table | .732096→.782396 | +.050300 | +.049144 |
| laptop | .452560→.361795 | −.090765 | −.070825 |
| suitcase | .686125→.652656 | −.033469 | −.023828 |
| bus | .904972→.872436 | −.032536 | −.018064 |
| teacup | .896981→.869878 | −.027103 | −.028443 |

Ví dụpenguin78/80cleanqueries tăng, laptop71/80giảm. Đây là heterogeneity giữa lớp, không tự suy ra teacher bỏđặcđiểmsemantic nào. Toàn30classes và5scopes trongCSV dưới đây.

## Training và chi phí

Mean **10loss-bearing loggedrows đầu (10–100)** và **10cuối (1100–1189)**, không dùng step0. Cửa sổ đầu khác báo cáo baseline cũ chỉ dùng9rows10–90; bảng này đã tính lại cùngwindow cho cảhaiarm.

| Loss | Baseline đầu→cuối | SREF đầu→cuối |
|---|---:|---:|
| Total | 11.314060→3.845047 | 11.686439→4.117129 |
| Raw sketchref | khôngđo | 3.176244→2.273730 |
| CleanMP(μ_I) | 3.651795→1.200868 | 3.658448→1.204634 |
| MaskedMP(μ_I) | 3.850308→1.847385 | 3.854509→1.835838 |
| CleanqCE | 4.921543→.694083 | 4.855518→.708458 |
| MaskedqCE | 5.071273→1.744556 | 5.021274→1.752224 |
| Photoanchor | .061315→.146250 | .061304→.146721 |
| Textanchor | .026483→.325065 | .027643→.325596 |

TotalSREF có thêmterm nên không lấy totalcao hơn làm dấuhiệutrainkém. RawreferenceNCE giảm, nhưng baseline không log/referenceNCE: không gọi đó là measuredteacherdriftimprovement vsbaseline. CleanqCE cuối hơi cao hơnbaseline dù retrievalunseen tăng; seenclassificationloss không quyết định trực tiếp retrieval. Không kết luận overfit/forgetting/collapse hoặc gradientbalance kéo dài sauinitialization.

W&B cuối `_runtime`≈720.224s vsbaseline≈701s (loggingruntime, không phải pairedlatencybenchmark). Officialeval **135.761787s** vsbaseline136.694796s, không chứng minh inference nhanh hơn. Không thêmteacherforward lúcinference.

Trainpeak allocated **12,481,166,848bytes** (~11.624GiB), reserved13,061,062,656bytes (~12.164GiB); baselineallocated12,481,166,336bytes — peakgầnnhưnhau trongcác runs này, không đồng nghĩa extra frozenforward miễnphí. Evalpeakallocated1,829,593,600bytes/reserved2,237,661,184bytes.

## Verification và đường dẫn evidence

Mọi path sau tương đối rootSREF:

- Producer `results.json`, `evaluation/tuberlin_220_30/summary.json`,10condition summaries,FP32query/galleryfeatures,perqueryNPZ,maskJSONL; predictor0 vàmodelstatebefore/afterunchanged tại lúcproducer đánhgiá.
- `analysis_readonly_20260911/statistics/analyze_sref_tuberlin.py` + `statistics.json`: independentCPU **24000rows/arm**, savedtop200relevance/P200/3AP200 allqueryPASS, maxerror **1.4006055626403224e−7 ≤2e−6**. FullAP chỉ mean từsavedvectors, **không fullgallerysort hoặc encoderreplay**. GalleryIDs/labels/queryorder/semanticnames/masks khớpbaseline.
- `analysis_readonly_20260911/statistics/tuberlin_220_30_classwise.csv`: đủ30lớp/clean,25,50,75,all9; đây là statsartifact local, không nằm trongtrainingarchive.
- `analysis_readonly_20260911/audit_sref_training.py`, `audit_raw.json`, `receipt.json`, `online/sref_tuberlin_220_30.json`: onlineworker scopemetadata/2618selectedlossgradientLRscalarchecks. Worker prose từng vượt scope archive/ckpt nhưng script chưa làm, parent không tiếp nhận claim đó. Receiptworkerhardcodedstatus được parent kiểm tra tất cảbooleanchecks, không chỉ đọcPASS.
- `analysis_readonly_20260911/verify_parent.py`, `parent.log`, `parent_receipt.json`: **SCOPED_PARENT_BINDINGS_STATS_PASS**. ParentstreamedfinalcheckpointSHA; full597archivefiles+aggregate; train/valmanifest/classmapSHAs; initialization+38048traces/masks+1190LRrowsbyteexactbaseline; **3934allavailable loss/gradient/LRscalars** onlineexact/120rows; allsavedfeature/NPZ/summary/maskproducerSHAbindings cảhaiarm;120classmeans clean/masked botharms; querymediancounts checked; frozenstate/source/checkpoint→eval summary chainPASS.
- Parent corrected worker prose25%mean delta: **+.009244909833392337**, không+.009296 (số đó chỉseed101). Originalworker evidence được giữ. Oneinteractive inspectionKeyError`memory` doevalschema dùngpeak_gpu_memory_*; không model/evalfailure, không chạy lạievaluation.

Không finalcheckpointmodel/optimizerrestore, independentGPUencoderreplay, fullgallerysort, onlineartifactdownload hoặc significance/multiseed trong lượt này. Historical2update restoration/trainprobe vẫn là gate riêng, không substitute finalverification. Không rewrite rawUNVERIFIED thànhVERIFIED. Không thêmloss/train/defaultswitch/seed/QuickDraw/SIG/λsearch hoặc push.

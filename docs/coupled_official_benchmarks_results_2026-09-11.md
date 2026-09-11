# MP-Q official TU-Berlin / QuickDraw — kết quả cuối và thống kê

## 1. Kết luận và trạng thái

Cả hai train và official evaluation đã COMPLETE, exit 0; campaign chạy từ 2026-09-10 16:33:26 đến 20:07:21 UTC. Không còn process campaign khi kiểm tra báo cáo. **Clean full-gallery mAP: TU-Berlin 0.4396614905633032; QuickDraw 0.15478658944216914.** Masked macro 9 điều kiện: **0.32638461547718317 / 0.12753774058770742** (giá trị precision authoritative nằm trong summary gốc; các bảng dưới làm tròn 6 chữ số).

Raw runtime giữ `TRAIN_AND_OFFICIAL_EVAL_FINISHED_UNVERIFIED`; không thay bằng VERIFIED. Read-only online/provenance audit PASS, saved-ranking statistics PASS và parent bindings PASS chỉ trong phạm vi mô tả ở §8. Không train lại, GPU replay, chỉnh loss/horizon hoặc chọn checkpoint theo kết quả test.

## 2. Protocol đã khóa trước official test

[Protocol + prelaunch gates](coupled_official_benchmarks_campaign_2026-09-10.md).

| Thuộc tính | TU-Berlin | QuickDraw |
|---|---:|---:|
| Official seen / unseen classes | 220 / 30 | 80 / 30 |
| Train sketches | 15,400 | 236,080 |
| Full seen photo pool | 176,081 | 149,428 |
| Official test queries | 2,400 | 92,291 |
| Official gallery photos | 27,989 | 54,151 |
| Updates / warmup | 1,189 / 59 | 18,229 / 911 |
| Sketch observations = updates × 32 | 38,048 | 583,328 |
| Observations / train sketch | 2.4706493506 | 2.4708912233 |
| Training seed | 42 | 42 |

Fresh CLIP initialization riêng mỗi dataset, không lấy Sketchy-trained checkpoint sang fine-tune. **MP-Q = main MP(μ_I), inference q**, không phải QMP/main MP(q). Giữ F2_MP loss và optimizer: B32, một positive + một negative/sketch, tối đa 64 unique live photos/batch; clean/masked views lấy mean; tau .07; không SIG/PCE/TQMP. Original CLIP frozen; sketch student và prompts/readout thích nghi theo protocol. q-only inference không gọi predictor, không đưa true class label vào query forward.

Ngân sách là số quan sát tương đương ~2.47 lượt, không bảo đảm mỗi sketch được thấy đúng số lần ấy; hai views không nhân đôi số epochs. Official unseen không dùng cho loss, λ, schedule hoặc checkpoint selection. Chỉ final checkpoint, không best aliases. Ba seed 101/202/303 bên dưới là **mask seeds của cùng một training seed**, không phải ba training runs.

## 3. Kết quả retrieval

Full mAP là metric chính, query-weighted. AP200 prefix chia cho số positive trong top200; AP200 all-rel chia cho toàn bộ relevant trong gallery; AP200 min-k chia cho min(total relevant, 200). Các định nghĩa khác nhau, không gọi chung là mAP@200 để đối chiếu paper chưa xác nhận denominator.

### TU-Berlin

| Điều kiện | Full mAP | P@200 | AP200 prefix | AP200 all-rel | AP200 min-k |
|---|---:|---:|---:|---:|---:|
| Clean | .439661 | .502854 | .529811 | .119978 | .421831 |
| 25%, seed101 | .397135 | .448923 | .471415 | .108385 | .369577 |
| 25%, seed202 | .397625 | .448302 | .472565 | .108927 | .370134 |
| 25%, seed303 | .395732 | .446710 | .469684 | .106985 | .366538 |
| 50%, seed101 | .340898 | .380600 | .400966 | .091810 | .303946 |
| 50%, seed202 | .340903 | .380665 | .401751 | .092757 | .306087 |
| 50%, seed303 | .340426 | .379704 | .400490 | .091198 | .302968 |
| 75%, seed101 | .240580 | .260829 | .281007 | .059160 | .193875 |
| 75%, seed202 | .241691 | .261323 | .279452 | .059320 | .194260 |
| 75%, seed303 | .242472 | .261981 | .279268 | .059753 | .195751 |
| Mean 25% | .396831 | .447978 | .471221 | .108099 | .368749 |
| Mean 50% | .340742 | .380323 | .401069 | .091922 | .304333 |
| Mean 75% | .241581 | .261378 | .279909 | .059411 | .194629 |
| Mean all 9 masks | .326385 | .363226 | .384066 | .086477 | .289237 |

### QuickDraw

| Điều kiện | Full mAP | P@200 | AP200 prefix | AP200 all-rel | AP200 min-k |
|---|---:|---:|---:|---:|---:|
| Clean | .154787 | .150367 | .160758 | .008878 | .076657 |
| 25%, seed101 | .143314 | .135710 | .147533 | .007766 | .066975 |
| 25%, seed202 | .143019 | .134952 | .146380 | .007668 | .066194 |
| 25%, seed303 | .143428 | .135480 | .146974 | .007738 | .066735 |
| 50%, seed101 | .129923 | .119991 | .132896 | .006613 | .056992 |
| 50%, seed202 | .130173 | .119946 | .132680 | .006582 | .056783 |
| 50%, seed303 | .129641 | .119376 | .132083 | .006583 | .056769 |
| 75%, seed101 | .109613 | .099003 | .114535 | .005221 | .044611 |
| 75%, seed202 | .109566 | .098680 | .114411 | .005195 | .044361 |
| 75%, seed303 | .109163 | .098116 | .113768 | .005166 | .044022 |
| Mean 25% | .143254 | .135381 | .146962 | .007724 | .066635 |
| Mean 50% | .129912 | .119771 | .132553 | .006593 | .056848 |
| Mean 75% | .109447 | .098600 | .114238 | .005194 | .044331 |
| Mean all 9 masks | .127538 | .117917 | .131251 | .006503 | .055938 |

## 4. Mức suy giảm và phân bố query

Retention dưới đây = masked macro / clean macro, không phải phần trăm thông tin semantic được giữ lại.

| Mask yêu cầu | TU Δ full mAP | TU retention | QuickDraw Δ full mAP | QuickDraw retention |
|---|---:|---:|---:|---:|
| 25% | −.042831 | 90.26% | −.011533 | 92.55% |
| 50% | −.098919 | 77.50% | −.024874 | 83.93% |
| 75% | −.198080 | 54.95% | −.045339 | 70.71% |

TU giảm 19.808 điểm phần trăm tại 75%, QuickDraw giảm 4.534 điểm phần trăm. QuickDraw có nền clean thấp hơn; tỷ lệ giữ điểm lớn hơn **không đủ chứng minh robustness tốt hơn**, và chưa có baseline đối chứng cùng official protocol để kết luận gain.

| Clean per-query AP | TU | QuickDraw |
|---|---:|---:|
| p10 | .047147 | .028856 |
| p25 | .107782 | .043937 |
| Median | .376161 | .087724 |
| p75 | .764067 | .201835 |
| p90 | .939950 | .379780 |
| Không có positive trong top200 | 6.00% | 19.60% |

Ở 75%, no-hit top200 tăng lên khoảng 23.24% TU / 27.05% QuickDraw, lấy mean ba mask seeds. Full AP không có giá trị 0 không đồng nghĩa mọi query tốt: gallery đầy đủ vẫn chứa positive ở các rank sâu.

So masked AP trung bình ba seed của **từng query** với clean, ngưỡng tăng/giảm ±.001:

| Mask | TU tăng / giảm / nhỏ | QuickDraw tăng / giảm / nhỏ |
|---|---:|---:|
| 25% | 725 / 1,605 / 70 | 39,246 / 46,633 / 6,412 |
| 50% | 568 / 1,787 / 45 | 36,562 / 51,866 / 3,863 |
| 75% | 391 / 1,983 / 26 | 32,079 / 57,559 / 2,653 |

Ở 75%, AP giảm hơn .05 trên 1,548/2,400 TU queries và 29,363/92,291 QuickDraw queries; đồng thời 142 / 9,994 queries tăng hơn .05. Mean che giấu hai hướng biến động; tăng khi xóa không chứng minh đã xóa một semantic distractor.

Mask là xóa raster bằng vùng vuông chọn quanh ink, không xóa stroke/semantic part có annotation. Mean ink fraction thực tế ~25.75/50.80/75.68% TU và ~25.59/50.65/75.53% QuickDraw. TU tất cả 21,600 masked rows status ok; QuickDraw 830,618 ok + **1 target_unreachable** ở 75%, seed303; giữ query này trong thống kê, không bỏ để nâng điểm. Không có blank_input.

## 5. Chênh lệch giữa classes

Mỗi TU class có 80 queries. QuickDraw có 29 classes × 3,001 và cake 5,262. Vì vậy TU class-equal macro = query macro; QuickDraw clean class-equal full mAP .156581, query macro .154787: weighting có ảnh hưởng nhỏ, không giải thích phần lớn điểm thấp.

| Tập | 5 classes clean cao nhất (full mAP) | 5 classes clean thấp nhất |
|---|---|---|
| TU | tractor .912210; bus .904972; horse .898340; teacup .896981; ant .818259 | fan .078566; parachute .097368; pizza .120754; snowboard .134008; canoe .149919 |
| QuickDraw | shark .402946; tiger .370392; rhinoceros .356245; raccoon .331492; cow .320964 | campfire .035329; hamburger .044470; cactus .055753; palm tree .060571; banana .061530 |

TU giảm mạnh nhất clean→75%: teacup .896981→.455927; suitcase .686125→.264382; telephone .502038→.135178. QuickDraw: tiger .370392→.178667; rhinoceros .356245→.179858; shark .402946→.274659. Đây là xếp hạng mô tả hậu nghiệm, không causal diagnosis hoặc tập chọn để tune.

Đủ 30 classes/tập với clean/P200/ba AP200 và mỗi mức mask nằm trong CSV/JSON ở §8.

## 6. Training losses: học seen classes, không phải unseen learning curve

Mean các logged batches, không phải mọi batch. **Parent sửa cách diễn đạt worker:** first window gồm step0 không có loss, nên TU có **9 loss rows, steps10–90**, QuickDraw **59 loss rows, steps10–590**; không phải 10/60 loss observations. Last windows: TU10rows1100–1189, QuickDraw60rows17640–18229. Giữ nguyên worker evidence, parent ghi rõ trong receipt riêng.

| Loss raw | TU đầu → cuối | QuickDraw đầu → cuối |
|---|---:|---:|
| Total | 11.552664 → 3.845047 | 8.368871 → 2.908870 |
| Clean MP(μ_I) | 3.735347 → 1.200868 | 3.359734 → 1.371869 |
| Masked MP(μ_I) | 3.915126 → 1.847385 | 3.587391 → 1.871925 |
| Clean CE(μ_I) | 4.787092 → .724870 | 2.740062 → .305610 |
| Masked CE(μ_I) | 4.997199 → 1.748331 | 3.217985 → 1.011436 |
| Clean CE(q) | 5.048010 → .694083 | 2.961169 → .296891 |
| Masked CE(q) | 5.164775 → 1.744556 | 3.367249 → .992357 |

Loss giảm rõ trên seen training batches; masked khó hơn clean ở cuối. Không suy ra đã hội tụ retrieval/unseen, không chứng minh overfit/collapse/forgetting hoặc undertraining. Official retrieval chỉ đo final một lần nên không có test learning curve để biết peak sớm/muộn. CE của hai dataset dùng số train classes khác nhau, không so raw CE để kết luận dataset nào học tốt hơn.

## 7. Runtime, VRAM, checkpoint

| Producer / logging clock | TU | QuickDraw |
|---|---:|---:|
| W&B last `_runtime`, xấp xỉ | 701s (~11m41s) | 8,187s (~2h16m27s) |
| Full clean+9 evaluator elapsed | 136.695s (~2m17s) | 3,500.881s (~58m21s) |
| Train peak allocated / reserved, bytes | 12,481,166,336 / 13,061,062,656 | 8,481,327,104 / 8,975,810,560 |
| Eval peak allocated / reserved, bytes | 1,829,593,600 / 2,237,661,184 | 1,830,006,272 / 2,401,239,040 |

W&B clock không phải chứng nhận wall-time toàn process; evaluator elapsed gồm quy trình đánh giá/lưu artifacts, không phải benchmark latency mỗi query. Khác biệt VRAM không tự chứng minh nguyên nhân nếu chưa profiling.

- TU final1189 SHA256 `a39a0a6a6107566afa43af44410f25d40ab8b36d28d75ac6a4bf0c7901752460`.
- QuickDraw final18229 SHA256 `4ba565fc53b3efaef6f641f0e6e1585d2bc8c2f48919b0d31f35917194fb373c`.
- W&B TU [18wp9wty](https://wandb.ai/a-cctest05187-erd/spica/runs/18wp9wty), QuickDraw [jn8b9lmg](https://wandb.ai/a-cctest05187-erd/spica/runs/jn8b9lmg), đều finished. Official retrieval metrics local, **không có trong training W&B history**.

## 8. Evidence và giới hạn verification

Root: `outputs/coupled_benchmark_execution_20260910T164500Z/`.
Training HEAD `16f9d6b87a159fe5b23385630607e1d99908dff9`; archive **592files**, SHA256 `1d7532cf571cb775984f452902592a4b88d0d56c0b901ee3483721e389e467cb`. Báo cáo/commit này là postprocessing, không phải training snapshot.

- `runtime.json`, `results.json`, `runs/<dataset>/resolved_config.json`, `run_result.json`, `parent_final_gate.json`, checkpoints và histories.
- `evaluation/<dataset>/summary.json`: metrics precision authoritative; từng condition có full-AP vector, top200 và features hash-bound.
- `analysis_readonly_20260911/statistics/report.md`: đầy đủ phân bố AP, top200 no-hit, 5metrics×10conditions, masks, class highlights.
- `analysis_readonly_20260911/statistics/{tuberlin_220_30,quickdraw_80_30}_classwise.{csv,json}`: toàn bộ 30 classes/tập.
- `analysis_readonly_20260911/statistics/{statistics,artifact_hashes,manifest}.json`: precision và input bindings.
- `analysis_readonly_20260911/verification/receipt.json`, `summary/stats.json`, `raw_online/<dataset>.json`, `scripts/verify_stats.py`.
- `analysis_readonly_20260911/{verify_parent.py,parent_receipt.json,parent.log}`: parent PASS, 24,000 + 922,910 saved query-condition rows, online scalar checks **3,815 + 58,343**.

Verified scope:

1. Archived592files/aggregate, current530criticalfiles khớp tại thời điểm audit trước docs commit; checkpoint byte SHA khớp train/eval; init+first64traces/masks+3LRrows khớp smoke. Tất cả LR rows (1,190/18,230) arithmetic replay exact. Parent đếm thực tế full trace/mask files.
2. W&B120/1,824 logged rows finished, local/online loss-gradient-LR scalar exact. Parent kiểm tra lại từ raw online đã lưu, không download online checkpoint artifacts.
3. Independent saved-top200 relevance/formulas của P200 và ba AP200 trên **946,910 query-condition rows**, predeclared tolerance2e−6 PASS. Full mAP là tính lại mean **saved full-AP**, không tính lại full-gallery ranks.
4. Parent ràng buộc current manifests/class maps với training protocol SHA; official query/gallery path+label order exact; condition JSON→NPZ/query features/mask metadata và gallery features/labels/metadata SHA khớp producer; all60class full-AP means exact. Hash features không phải model replay.
5. Producer ghi q-only/predictor0 và state trước/sau eval không đổi; parent ràng buộc eval state với trained final state hash. Không deserialization hoặc independent model replay trong lượt này.

Không full-gallery second sort, encoder replay, online artifact download, multiseed training hoặc confidence/significance certificate. Không coi audited statistics là whole campaign VERIFIED. Không dùng raw chênh lệch TU/QuickDraw/Sketchy để claim gain: classes, gallery, splits khác nhau. Official split không đồng nghĩa exact reproduction của SketchLVM/SeCo; cần match đầy đủ metric/training protocol nếu so paper. Không thay default, tăng epochs hay mở thêm ablation từ kết quả này.

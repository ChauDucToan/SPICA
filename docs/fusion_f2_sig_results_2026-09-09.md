# F2 / F2+SIG — kết quả và đo các head, 2026-09-09

## Kết luận

- F2 và F2+SIG đều COMPLETE3600 updates, exit0. **F2+SIG FAIL promotion; giữ F2 trong cặp này.** Không thay S0 cho mục tiêu clean prefixAP200/P200.
- F2 tại best_clean vượt R0 trên cả ba clean metrics và ba masked-macro metrics. So S0: full mAP tốt hơn rõ, nhưng clean P200/prefixAP200 vẫn thấp hơn. Đây là so sánh các package/protocol khác nhau, không phải ablation nhân quả.
- Không còn dấu hiệu common-direction cực mạnh của mu_i như R1 cũ: mean cosine giữa query khác nhau F2/F2+SIG = .052470/.049615, thay vì R1 .951620. Hai head **của cùng query** lại rất giống nhau: cosine(mu_i,mu_t)=.985552/.990207. Hai phát biểu không mâu thuẫn.
- SIG làm thay đổi g rõ, nhưng không cải thiện clean retrieval. Không suy từ cosine thấp sang Gaussianity, tránh collapse, hoặc lambda tối ưu. Không đổi main head sang q/mu_t hậu nghiệm.

## Phạm vi và định nghĩa

Pseudo validation seed3407: 20 classes,10,963 sketch queries/13,999 gallery photos, cùng identity/order. Official unseen21classes không được dùng. Mọi inference head nhận sketch, không nhận label; label chỉ dùng sau forward để tính relevance/cosine cùng lớp.
`AP200` trong các bảng là **prefix-positive AP200**, không phải full mAP: `sum_{k<=200}(P(k)*rel(k))/max(1,R200)`. P200=`R200/200`; fullAP lấy tất cả ranks và chia tổng relevant trong gallery. Cả AP200 all-relevant và min(R,200) được lưu full precision trong JSON. Không coi .7364 của S0 là full mAP.
Masked macro = trung bình9điều kiện deletion25/50/75% × seeds101/202/303. Best_clean chọn strict maximum clean prefixAP200 trên steps600..3600, earliest tie. Masked tại best_clean được đọc **ở chính checkpoint đó**, không lấy peak riêng.

## 1. Best_clean của từng run

| Model | Step | Clean full mAP | Clean P200 | Clean AP200 | Masked full mAP | Masked P200 | Masked AP200 |
|---|---:|---:|---:|---:|---:|---:|---:|
| S0 | 1800 | 0.265979747 | 0.670306017 | 0.736406897 | 0.142767055 | 0.305037544 | 0.339448008 |
| R0 | 1200 | 0.353229363 | 0.456360019 | 0.515218386 | 0.240382332 | 0.310890615 | 0.357617838 |
| F2 | 2400 | 0.469791105 | 0.516573007 | 0.535347134 | 0.330985585 | 0.348336164 | 0.364819193 |
| F2_SIG | 2400 | 0.466939062 | 0.514188168 | 0.532509286 | 0.330678133 | 0.347674139 | 0.363652975 |

SIG−F2 ở best_clean (cùng2400): clean full−.002852043, P200−.002384840, prefixAP200−.002837848; masked prefix−.001166219. Cả4promotion criteria FAIL. Negative promotion không phải training failure.

## 2. Cùng step3600 — main retrieval head

| Model | Head | Clean full mAP | Clean P200 | Clean AP200 | Masked full mAP | Masked P200 | Masked AP200 |
|---|---|---:|---:|---:|---:|---:|---:|
| S0 | native prompted CLIP | 0.186983950 | 0.449897828 | 0.699090253 | 0.106077072 | 0.225162055 | 0.329966024 |
| R0 | q | 0.305843784 | 0.423951464 | 0.507648816 | 0.212324934 | 0.297152689 | 0.359103953 |
| F2 | mu_i | 0.471365186 | 0.514762828 | 0.533986756 | 0.334816536 | 0.349045722 | 0.366276100 |
| F2_SIG | mu_i | 0.466831463 | 0.512195100 | 0.530756497 | 0.335667118 | 0.351639902 | 0.368607541 |

Tại3600, SIG−F2: clean full−.004533723/P200−.002567729/prefix−.003230259; masked macro full+.000850582/P200+.002594180/prefix+.002331441. Trade-off nhỏ ở latest không lật được promotion tại best_clean. Best_masked của cảhai là3600.

## 3. Đo mới tất cả head trên clean validation

Các hàng dưới đều step3600, trừ S0_best_clean được ghi riêng1800. q/mu_t của F2 chỉ diagnostic, không selection/fusion. μ_i/μ_t/q là vector512D chuẩn hóa L2 (norm≈1); raw g là768D chưa chuẩn hóa. R0/S0 không có predictor μ_i/μ_t.

| Model | Step | Head | full mAP | P200 | AP200 | Cosine giữa query | Cosine cùng lớp | Cosine khác lớp | Covariance effective rank |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| F2 | 3600 | q | 0.477240893 | 0.514694872 | 0.533542940 | 0.042814 | 0.470351 | 0.020049 | 55.571709 |
| F2 | 3600 | mu_i | 0.471365186 | 0.514762828 | 0.533986756 | 0.052470 | 0.499032 | 0.028692 | 34.670816 |
| F2 | 3600 | mu_t | 0.468688950 | 0.512800774 | 0.532487645 | 0.083045 | 0.515355 | 0.060026 | 34.621732 |
| F2_SIG | 3600 | q | 0.474606769 | 0.514586781 | 0.532423223 | 0.043643 | 0.479833 | 0.020417 | 50.823956 |
| F2_SIG | 3600 | mu_i | 0.466831463 | 0.512195100 | 0.530756497 | 0.049615 | 0.498817 | 0.025696 | 34.203507 |
| F2_SIG | 3600 | mu_t | 0.464324729 | 0.510163722 | 0.529272441 | 0.073273 | 0.511319 | 0.049949 | 34.172320 |
| R0 | 3600 | q | 0.305843784 | 0.423951464 | 0.507648816 | 0.135356 | 0.530846 | 0.114298 | 53.198247 |
| S0 | 3600 | embedding | 0.186983950 | 0.449897828 | 0.699090253 | 0.232360 | 0.628202 | 0.211282 | 49.418780 |
| S0_best_clean | 1800 | embedding | 0.265979747 | 0.670306017 | 0.736406897 | 0.245008 | 0.631084 | 0.224450 | 49.974869 |

Mean cosine giữa query dùng tất cả cặp khác nhau, loại self; same/between class là pair-weighted, không class-macro. Effective rank dùng entropy eigenvalue của **centered covariance trên unit features**; không khẳng định numerical positive eigenvalues là exact rank đáng tin.

### Cosine các head của cùng sketch

| Model@3600 | q–mu_i | q–mu_t | mu_i–mu_t |
|---|---:|---:|---:|
| F2 | 0.771425477 | 0.760784002 | 0.985552014 |
| F2_SIG | 0.784927123 | 0.776837825 | 0.990207471 |

Hai μ gần đồng hướng theo từng sketch và retrieval gần nhau → ít khác biệt về output direction trong phép đo này. **Không đồng nghĩa collapse giữa mọi sketch**, cũng không chứng minh text prompt hữu ích hay fusion SA là nguyên nhân cải thiện; chưa can thiệp prompt để đo causal dependence.

### Query–photo cosine theo class (all-gallery, query macro)

Class mean của photo **không** được renormalize, vì mục tiêu là mean pairwise cosine thật sự. Không so trực tiếp với historical metric dùng cosine tới unit class centroid.

| Model | Step | Head | Cùng lớp | Khác lớp | Chênh lệch |
|---|---:|---|---:|---:|---:|
| F2 | 3600 | mu_i | 0.183115940 | 0.049694157 | 0.133421783 |
| F2 | 3600 | mu_t | 0.185446701 | 0.054321247 | 0.131125454 |
| F2 | 3600 | q | 0.137256395 | -0.002491324 | 0.139747719 |
| F2_SIG | 3600 | mu_i | 0.181170950 | 0.046817021 | 0.134353929 |
| F2_SIG | 3600 | mu_t | 0.182431060 | 0.049886663 | 0.132544396 |
| F2_SIG | 3600 | q | 0.144227033 | 0.000031739 | 0.144195293 |
| R0 | 3600 | q | 0.064712741 | -0.060514717 | 0.125227458 |
| S0 | 3600 | embedding | -0.007384386 | -0.103786944 | 0.096402558 |
| S0_best_clean | 1800 | embedding | 0.075457152 | -0.052539711 | 0.127996862 |

### g và photo gallery

| Model@3600 | g mean pair cosine | g raw centered trace | g covariance effective rank | Gallery mean pair cosine | Gallery effective rank |
|---|---:|---:|---:|---:|---:|
| R0 | 0.350187 | 53.076659 | 93.507997 | 0.471715 | 100.881175 |
| F2 | 0.420053 | 56.390062 | 100.716959 | 0.448447 | 109.741193 |
| F2_SIG | 0.113097 | 110.239977 | 83.741461 | 0.446990 | 109.662105 |

SIG: g mean cosine .420→.113 và raw trace56.39→110.24, nhưng entropy effective rank100.72→83.74. Không phải mọi chỉ số đa dạng đều tốt hơn.

## 4. Hubness và canonical-photo exposure

| Model | Step | Main head | Số photo top1 khác nhau | Top1 phổ biến nhất | Union top200 | Canonical trong top200 |
|---|---:|---|---:|---:|---:|---:|
| F2 | 3600 | mu_i | 1358 | 2.673% | 13317 | 15.318% |
| F2_SIG | 3600 | mu_i | 1332 | 2.928% | 13267 | 15.784% |
| R0 | 3600 | q | 494 | 5.254% | 5138 | 77.434% |
| S0 | 3600 | embedding | 842 | 2.581% | 2584 | 91.548% |
| S0_best_clean | 1800 | embedding | 1241 | 1.368% | 4340 | 53.854% |

Canonical gallery2000/13999=14.29%, xác định bằng đúng canonical branch/full path, không basename. F2 exposure15.32% gần pool fraction, khác R0@3600 77.43% và R1 cũ99.47%. Đây là giảm canonical domination quan sát được, **không phải counterfactual chứng minh đã loại mọi pool shortcut**. Không có photo nằm trong top200 của mọi query ở cả5model/step đã đo.

## 5. Evidence và giới hạn xác minh

- Training HEAD `649a2491eb4882b534a70897e5455bc36f6616c9`; source SHA `9dc0c3df2b730dddf3726587de007ad9df6eccb3d06ccf7a7a0767a0f5e5ec80`,560archivedfiles. F2 λ0; F2+SIG λ0.008392757138899188. Init,115200observation/mask records và3601LRrows/arm byte-identical.
- W&B finished [F2](https://wandb.ai/a-cctest05187-erd/spica/runs/1wxvk2lk) / [F2+SIG](https://wandb.ai/a-cctest05187-erd/spica/runs/kl6q1zf1);361unsampledhistoryrows/arm,980retrievalscalarchecks,6selectedaliases/4downloadedversions checkpoint/configSHAverified. [Online receipt](../outputs/fusion_execution_20260909T064000Z/statistics_online_20260909/verification_receipt.json). R0/S0 histories cũng lấy read-only.
- New GPU diagnostic `outputs/fusion_head_measurements_20260909T084200Z`:0optimizer updates/backward; model hashes trước/sau không đổi; F2/F2+SIG@3600 vàS0@3600/1800 mainclean replay cả5scalar delta0. F2/F2+SIG top200exact. R0 dùng saved features đã được audit độc lập trước đó; lần này verify bytes/hash + rerank, **không re-encode R0**.
- Diagnostic source SHA `56c162142c282cb24f8f8d0d73c7ce9ecd0c69846338b76d13de4caf51e06d8a`; archive chứa script mới. Dùng archived checkpoint loader, không gọi whole-campaign verifier dang dở. S0 inference source component bytes được so với archived semantic source.
- Independent CPU từ saved features:5arm/step,9heads; cosine/geometry float64≤1e-10; top200relevance/P200/3AP200 tất cảqueries, maxarraydelta1.44126851898e-07; fullAP chỉ16evenly-spacedqueries/head (144query-headcases), maxdelta3.28440032682e-06 do numerical/tie differences. Không phải full CPU second evaluator, không independent second encoder.
- [Independent receipt](../outputs/fusion_head_measurements_20260909T084200Z/independent_cpu/receipt.json), [parent verification](../outputs/fusion_head_measurements_20260909T084200Z/parent_verification.json):PASS; parent rehashed595receipt-boundfiles vàcrosschecked140clean/macrohistoryscalars, downloadedaliases, step selections, mainhead replay/W&B.
- Root training runtime vẫn `ARMS_FINISHED_UNVERIFIED`, không sửa raw status để gộp các tier verification. Không replay toàn bộ selected checkpoints/masked features; masked thống kê lấy historical rawprobes và W&B, không phải phép đo mới của q/mu_t.
- R0 vàS0 là external descriptive references, khác architecture, sampler, loss, số parameter, clean/masked training và scheduler. F2 thay cảfusion+photo sampling+semantic supervision: không quy cải thiện riêng cho self-attention. SIG−F2 matched package nhưng mộtseed không cho multi-seed significance.
- Không thêm training, seed, official unseen, tuning, head fusion, prompt/shuffle ablation; không ghi vào historical W&B runs hoặc tạo report run.

## 6. Tái lập no-update measurement

```bash
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH=/run/opengl-driver/lib OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \
  .venv/bin/python scripts/measure_fusion_heads.py --cpu-self-check
# GPU only with authorization, always a fresh output root:
LD_LIBRARY_PATH=/run/opengl-driver/lib HF_HUB_OFFLINE=1 WANDB_MODE=disabled \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \
  .venv/bin/python scripts/measure_fusion_heads.py --output outputs/<fresh-root>
```

Full-precision report: [JSON](fusion_f2_sig_results_2026-09-09.json). Raw μ/q/g/gallery vectors, per-query ranks/AP/cosine và IDs nằm trong measurement root; không đưa tensors lên Git.

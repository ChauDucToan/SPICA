# F2_MP — kết quả, μ/cosine và so sánh F2/R0/S0 (2026-09-09)

## Kết luận

- **Mục tiêu primary đạt mức tăng số học nhỏ:** clean full mAP@3600 tăng `.471365186 → .474080179` (+.002714993, +0.2715 điểm phần trăm). Masked full mAP `.334816536 → .342202940` (+.007386404). Một seed, không có khẳng định statistical significance hay superiority toàn diện.
- Đổi lại, clean P200/prefixAP200 giảm `.514763/.533987 → .489339/.503828`; masked P200/prefix cũng giảm. Không gọi MP thắng mọi metric, không áp điều kiện promotion cũ của F2_SIG cho objective primary mới.
- Diagnostic **q@3600 full mAP .491669182**, tốt hơn μ_I .474080179 và μ_T .477787247. Không chọn lại head hậu nghiệm; **μ_I vẫn main inference**. q của F2 cũ .477240893; q của MP cũng tăng P200/prefixAP200, nhưng không phải kết quả primary pre-registered.
- μ_I–μ_T cùng sketch cosine .985638, góc trung bình9.463762°; q tới trung điểm cung40.483321°. **Hai μ gần nhau, q chưa nằm giữa chúng**. μ_I giữa các query cosine .060382, không có kiểu common-direction .95 của R1 cũ.

## Phạm vi / metric

Clean pseudo-validation20classes,10,963queries/13,999gallery; cùng paths/order/labels các model. Không dùng official unseen21classes. MP train3600updates seed42/pseudo3407, kiến trúc F2, one-positive/one-negative sampling không đổi; bank≤64 **unique photos tổng cộng**, không64positives/sketch. Multi-positive loss lấy mọi photo cùng lớp trong bank, negative mọi lớp khác; tau.07/coef1 cố định, không SIG. Loss scale không được gradient-match với F2; không quy thay đổi riêng cho số negatives.
`AP200` trong bảng là **prefix-positive AP200**, không phải full mAP: `sum(P(k)*rel(k), k<=200)/max(1,R200)`. P200=R200/200. All-relevant và min(R,200) AP200 cũng lưu đầy đủ trong JSON. Masked macro là9điều kiện deletion25/50/75% × seeds101/202/303.

## 1. Primary — cùng step3600, main head

| Model | Clean full mAP | Clean P200 | Clean AP200 | Masked full mAP | Masked P200 | Masked AP200 |
|---|---:|---:|---:|---:|---:|---:|
| S0 | 0.186983950 | 0.449897828 | 0.699090253 | 0.106077072 | 0.225162055 | 0.329966024 |
| R0 | 0.305843784 | 0.423951464 | 0.507648816 | 0.212324934 | 0.297152689 | 0.359103953 |
| F2 | 0.471365186 | 0.514762828 | 0.533986756 | 0.334816536 | 0.349045722 | 0.366276100 |
| F2_MP | 0.474080179 | 0.489338675 | 0.503827883 | 0.342202940 | 0.339020536 | 0.354761560 |

| MP−F2@3600 | full mAP | P200 | AP200 |
|---|---:|---:|---:|
| clean | +0.002714993 | -0.025424153 | -0.030158873 |
| masked_macro | +0.007386404 | -0.010025186 | -0.011514540 |

## 2. Best_clean — auxiliary prefix selection, không primary full-mAP selection

Alias best_clean vẫn chọn prefixAP200 strict maximum ở600..3600/earliest tie. MP chọn600 dù full mAP tại3600 tốt hơn. Không đổi nghĩa alias. Mọi masked score dưới đây đọc ở đúng checkpoint best_clean của run đó.

| Model | Step | Clean full mAP | P200 | AP200 | Masked full mAP | Masked P200 | Masked AP200 |
|---|---:|---:|---:|---:|---:|---:|---:|
| S0 | 1800 | 0.265979747 | 0.670306017 | 0.736406897 | 0.142767055 | 0.305037544 | 0.339448008 |
| R0 | 1200 | 0.353229363 | 0.456360019 | 0.515218386 | 0.240382332 | 0.310890615 | 0.357617838 |
| F2 | 2400 | 0.469791105 | 0.516573007 | 0.535347134 | 0.330985585 | 0.348336164 | 0.364819193 |
| F2_MP | 600 | 0.456619823 | 0.487114831 | 0.506398984 | 0.318921066 | 0.321320901 | 0.339657610 |

MP best_masked=latest3600. Các so sánh R0/S0 là descriptive, khác architecture/training views/sampler/loss/scheduler; không causal ablation. MP vs F2 có cùng init/trace/mask/LR, nhưng là so sánh loss replacement với cả scale của nó.

## 3. F2_MP trajectory

| Step | Clean full mAP | P200 | AP200 | Masked full mAP | Masked P200 | Masked AP200 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.068337333 | 0.049547568 | 0.061372494 | 0.068842781 | 0.051617307 | 0.061775861 |
| 600 | 0.456619823 | 0.487114831 | 0.506398984 | 0.318921066 | 0.321320901 | 0.339657610 |
| 1200 | 0.463692344 | 0.485500766 | 0.500673022 | 0.330139324 | 0.330867811 | 0.348284597 |
| 1800 | 0.457230809 | 0.477532600 | 0.492046955 | 0.326154185 | 0.325922946 | 0.341783089 |
| 2400 | 0.474074183 | 0.489489637 | 0.502182800 | 0.339961722 | 0.337244208 | 0.352001176 |
| 3000 | 0.472641273 | 0.487349712 | 0.501679591 | 0.341546817 | 0.337950675 | 0.353329402 |
| 3600 | 0.474080179 | 0.489338675 | 0.503827883 | 0.342202940 | 0.339020536 | 0.354761560 |

| MP@3600 deletion | full mAP | P200 | AP200 |
|---|---:|---:|---:|
| 0.25 | 0.414381242 | 0.420463216 | 0.435222950 |
| 0.5 | 0.347021630 | 0.343479119 | 0.358918845 |
| 0.75 | 0.265205947 | 0.253119274 | 0.270142886 |

## 4. Head measurements — cùng step3600

F2_MP thực sự re-encode CUDA/no-update; F2/R0/S0 reuse features và metrics đã verified trước đó, kiểm tra SHA/IDs/labels khớp. Không claim re-encode các controls lần này. S0@1800 best_clean cũng được giữ trong JSON riêng, không trộn bảng3600.

| Model/head | full mAP | P200 | AP200 | Query-pair cosine | Same-class cosine | Between-class cosine | Covariance effective rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| F2_MP/q | 0.491669182 | 0.520355275 | 0.535433599 | 0.043809 | 0.476222 | 0.020784 | 53.315444 |
| F2_MP/mu_i | 0.474080179 | 0.489338675 | 0.503827883 | 0.060382 | 0.494670 | 0.037257 | 35.856312 |
| F2_MP/mu_t | 0.477787247 | 0.496813362 | 0.511757036 | 0.073745 | 0.502019 | 0.050940 | 35.863636 |
| F2/q | 0.477240893 | 0.514694872 | 0.533542940 | 0.042814 | 0.470351 | 0.020049 | 55.571709 |
| F2/mu_i | 0.471365186 | 0.514762828 | 0.533986756 | 0.052470 | 0.499032 | 0.028692 | 34.670816 |
| F2/mu_t | 0.468688950 | 0.512800774 | 0.532487645 | 0.083045 | 0.515355 | 0.060026 | 34.621732 |
| R0/q | 0.305843784 | 0.423951464 | 0.507648816 | 0.135356 | 0.530846 | 0.114298 | 53.198247 |
| S0/embedding | 0.186983950 | 0.449897828 | 0.699090253 | 0.232360 | 0.628202 | 0.211282 | 49.418780 |

Cosine query-pair = all non-self pairs; within/between class pair-weighted. Covariance effective rank là entropy rank trên **centered unit-feature covariance**, không numerical exact rank. Retrieval vectors512D norm≈1; raw g768D chưa normalize.

### Cùng sketch: cosine và góc

| Model@3600 | cos(q,μ_I) | cos(q,μ_T) | cos(μ_I,μ_T) | angle(q,μ_I) | angle(q,μ_T) | angle(μ_I,μ_T) | angle(q,midpoint) |
|---|---:|---:|---:|---:|---:|---:|---:|
| F2 | 0.771425477 | 0.760784002 | 0.985552014 | 39.228123° | 40.184183° | 9.527231° | 39.467545° |
| F2_MP | 0.757317380 | 0.752502315 | 0.985638078 | 40.501868° | 40.923095° | 9.463762° | 40.483321° |

Góc là mean của arccos từng query (không arccos mean cosine). Midpoint minor arc=`normalize(μ_I+μ_T)`. MP arc-length excess `angle(q,μ_I)+angle(q,μ_T)-angle(μ_I,μ_T)`≈71.9612°, không gần0. Gần cách đều hai μ không có nghĩa nằm giữa. Hình học này không chứng minh usefulness/causal role của photo/text prompts.

### Query–gallery cosine (mọi photo theo class, query macro)

| Model/head@3600 | Photo cùng lớp | Photo khác lớp | Margin |
|---|---:|---:|---:|
| F2_MP/mu_i | 0.186445131 | 0.007687036 | 0.178758095 |
| F2_MP/mu_t | 0.193352539 | 0.016087599 | 0.177264940 |
| F2_MP/q | 0.178675285 | -0.001985536 | 0.180660821 |
| F2/mu_i | 0.183115940 | 0.049694157 | 0.133421783 |
| F2/mu_t | 0.185446701 | 0.054321247 | 0.131125454 |
| F2/q | 0.137256395 | -0.002491324 | 0.139747719 |
| R0/q | 0.064712741 | -0.060514717 | 0.125227458 |
| S0/embedding | -0.007384386 | -0.103786944 | 0.096402558 |

Photo class means không renormalize; đây là mean pairwise cosine, khác cosine tới unit class centroid. MP μ_I class margin .178758 > F2 .133422, chủ yếu khác lớp cosine thấp hơn; margin trung bình tăng vẫn không bảo đảm top200 tốt hơn.

## 5. Hubness / photo exposure

| Model@3600/main | Unique top1 photos | Max top1 share | Union top200 | Canonical top200 exposure |
|---|---:|---:|---:|---:|
| S0/embedding | 842 | 2.581% | 2584 | 91.548% |
| R0/q | 494 | 5.254% | 5138 | 77.434% |
| F2/mu_i | 1358 | 2.673% | 13317 | 15.318% |
| F2_MP/mu_i | 1328 | 2.408% | 13257 | 13.650% |

Canonical gallery2000/13999=14.29%, exact full canonical paths, không basename. MP exposure13.65%, F2 15.32%; không canonical domination kiểu R0/S0@3600. Không có photo trong top200 của mọi query. Đây là descriptive ranking statistic, không chứng minh hết mọi shortcut.

## 6. Verification / provenance

- [W&B F2_MP y40hu06b](https://wandb.ai/a-cctest05187-erd/spica/runs/y40hu06b), finished/exit0/3600. Training HEAD `e2eb3413171f06a068be59aa8e9ad4f95d14d3f4`;565sourcefiles archive SHA `ab064ced3b11c95f142ac94b336072d88312f5e14d776a386416bbf3cf70f29e` verified from archived bytes, not current docs/gitignore.
- Read-only online receipt `outputs/fusion_mp_execution_20260909T115500Z/statistics_online_20260909/verification_receipt.json`:361MP unsampled rows,490retrievalscalarcomparisons across7probes;3aliases/2downloadedversions checkpoint+configSHAverified. Init equal F2;115200trace/maskrows and3601LRrows byte-identical; frozen original unchanged. `max_positives:64` in worker metadata is a loose bound, **actual contract is ≤64totalbankphotos, not64positives per query**.
- Latest/best_masked checkpoint SHA `58fb5fb2aa0f822d1df8bee9192a0bda5eec1580b4ce70ed9ae243fdcf9fb14e`; best_clean@600 SHA `eb68573e18f44f21ffc0a5f0f99665022b537cae7c407eef2e5ea854da37cb11`.
- New CUDA root `outputs/fusion_mp_head_measurements_20260909T135400Z`:36.49seconds,0optimizer updates/backward;3heads full clean retrieval. Main μ_I all5scalar replay delta0/top200exact/perqueryfullAP/P200 match; model hashes unchanged. Diagnostic source `a335a2c3e9c16f4997af4ef2d5423a986547f633d724e36126b55d4773d2524e` separately archived. Labels only for metrics, not model input.
- Independent CPU receipt under `independent_cpu/`:all3heads saved top200 relevance/P200/3AP200/scalars, float64 geometry/cosine/angles,128-query brute fixture, source/checkpoint/control cacheSHA/IDs PASS. FullAP second sort only16deterministicqueries/head (48cases), maxdelta5.977865971e-6; not full independent second evaluator or encoder. State equality audit is from saved before/after hashes; not a second state capture.
- Parent `outputs/fusion_mp_head_measurements_20260909T135400Z/parent_verification.json`:19boundfiles rehashed,70clean/macrohistoryscalars crosschecked,3selectedaliases hashes verified, primarydelta recomputed; statusPASS.
- Training runtime remains **ARM_FINISHED_UNVERIFIED**; these evidence tiers are separate, not rewriting raw status. Masked scores from training rawprobes+W&B; no new masked q/μ_T evaluation or all-selectedcheckpoint replay. No new head policy, training, seed, official unseen, lambda search or W&B report run. Pre-existing `.gitignore`, unfinished V1 verifier diff and `outputsnewgate/` preserved.

Full precision: [JSON](fusion_mp_results_2026-09-09.json). Runnable measurement: `scripts/measure_fusion_mp_heads.py --cpu-self-check`; actual GPU measurement requires fresh `--output` and authorization.

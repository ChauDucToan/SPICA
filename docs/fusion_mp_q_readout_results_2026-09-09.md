# F2_MP-Q — fixed3600 clean/masked readout check (2026-09-09)

## Kết luận

**PASS bước kiểm tra readout:** q tốt hơn μ_I trên cả5metrics của clean và từng9maskedconditions, cùng checkpoint3600/gallery/query/masks. Đây là bằng chứng ủng hộ q-retrieval cho hướng ưu tiên full mAP, không cần thêm loss hay train lại để có kết quả này.
Không phải q thắng mọi query/lớp: clean fullAP tăng ở6147/10963queries (56.07%), giảm4816;12/20classes tăng,8giảm. Chưa multi-seed hoặc official-unseen confirmation; không sửa mainhead/aliases/history của run đã train.

## Protocol đã chốt trước phần masked mới

- User duyệt kiểm tra F2_MP-Q trước khi quyết định phiên bản tiếp theo. Fixed **step3600**, không tìm best checkpoint, không head fusion. Clean q đã được quan sát trước khi chọn hướng này; không gọi toàn phép thử là blind/held-out confirmatory experiment.
- Reuse F2_MP checkpoint, không thay trọng số/loss/sampler. q=`normalize(pooled_head(mean(context_tokens(sketch))))`; bypass predictor thật sự. Photo gallery vẫn frozen-original CLIP với learned C_I. Text bank/predictor không cần forward trong query retrieval.
- Pseudo-validation20classes,10963queries/13999gallery. Clean + deletion25/50/75% × seeds101/202/303. Labels chỉ cho metric; không đưa vào query encoder.
- μ_I control là rawprobe3600 đã có onlineverification, không re-encode μ_I masked lần này. Query/gallery IDs+labels và98667maskrecords/statuscounts so từng điều kiện exact.

## 1. Clean và masked macro

| Condition | Head | full mAP | P200 | Prefix AP200 |
|---|---|---:|---:|---:|
| Clean | mu_i | 0.474080179 | 0.489338675 | 0.503827883 |
| Clean | q | 0.491669182 | 0.520355275 | 0.535433599 |
| Clean | delta | +0.017589003 | +0.031016600 | +0.031605716 |
| Masked macro9 | mu_i | 0.342202940 | 0.339020536 | 0.354761560 |
| Masked macro9 | q | 0.355752598 | 0.358522295 | 0.373294285 |
| Masked macro9 | delta | +0.013549658 | +0.019501758 | +0.018532724 |

Prefix-positive AP200=`sum(P(k)*rel(k),k<=200)/max(1,R200)`, không full mAP. Cả AP200 all-relevant/min(R,200) cũng tăng trong cả10cases và lưu full precision JSON. Primary full mAP tăng clean+.017589003, maskedmacro+.013549658.

## 2. Theo mức deletion (mean3seeds)

| Deletion | Head | full mAP | P200 | Prefix AP200 |
|---|---|---:|---:|---:|
| 0.25 | mu_i | 0.414381242 | 0.420463216 | 0.435222950 |
| 0.25 | q | 0.428360607 | 0.443773137 | 0.458930433 |
| 0.25 | delta | +0.013979365 | +0.023309921 | +0.023707484 |
| 0.5 | mu_i | 0.347021630 | 0.343479119 | 0.358918845 |
| 0.5 | q | 0.361215708 | 0.363628409 | 0.378035389 |
| 0.5 | delta | +0.014194078 | +0.020149290 | +0.019116544 |
| 0.75 | mu_i | 0.265205947 | 0.253119274 | 0.270142886 |
| 0.75 | q | 0.277681479 | 0.268165338 | 0.282917031 |
| 0.75 | delta | +0.012475531 | +0.015046064 | +0.012774145 |

## 3. Từng condition và phân bố lợi ích

| Condition | Δ full mAP | Δ P200 | Δ Prefix AP200 | Số lớp fullAP tăng/20 | Query fullAP win/loss/tie |
|---|---:|---:|---:|---:|---|
| clean | +0.017589003 | +0.031016600 | +0.031605716 | 12 | 6147/4816/0 |
| mask_25_101 | +0.013665707 | +0.023213992 | +0.023432251 | 11 | 6051/4912/0 |
| mask_25_202 | +0.014259140 | +0.023136003 | +0.024006727 | 11 | 6129/4834/0 |
| mask_25_303 | +0.014013247 | +0.023579768 | +0.023683474 | 11 | 6125/4838/0 |
| mask_50_101 | +0.013525014 | +0.020067500 | +0.019422418 | 11 | 6134/4829/0 |
| mask_50_202 | +0.013813264 | +0.018730730 | +0.017447801 | 12 | 6199/4764/0 |
| mask_50_303 | +0.015243956 | +0.021649640 | +0.020479413 | 12 | 6206/4757/0 |
| mask_75_101 | +0.012103298 | +0.014510170 | +0.012476391 | 12 | 6074/4889/0 |
| mask_75_202 | +0.012654893 | +0.015099425 | +0.013119702 | 12 | 6174/4789/0 |
| mask_75_303 | +0.012668403 | +0.015528596 | +0.012726344 | 12 | 6076/4887/0 |

Query win/loss tie tolerance1e-7; class metrics lấy meanAP trên queries của class. Không significance test/CI.

| Clean class | Queries | μ_I fullAP | q fullAP | Δ |
|---|---:|---:|---:|---:|
| bell | 536 | 0.355786627 | 0.461940204 | +0.106153577 |
| piano | 567 | 0.598797021 | 0.681593479 | +0.082796458 |
| spider | 570 | 0.881768509 | 0.954630416 | +0.072861907 |
| pretzel | 602 | 0.579920917 | 0.629136329 | +0.049215412 |
| pistol | 502 | 0.756247779 | 0.803309114 | +0.047061335 |
| hedgehog | 553 | 0.472055625 | 0.517773690 | +0.045718065 |
| strawberry | 550 | 0.411988003 | 0.454862726 | +0.042874723 |
| zebra | 558 | 0.377257951 | 0.410901174 | +0.033643223 |
| fish | 480 | 0.263377558 | 0.294208755 | +0.030831197 |
| tiger | 693 | 0.763221392 | 0.791074226 | +0.027852834 |
| shoe | 496 | 0.091872662 | 0.096072690 | +0.004200029 |
| apple | 501 | 0.293802173 | 0.296620051 | +0.002817878 |
| trumpet | 496 | 0.575211309 | 0.569367088 | -0.005844221 |
| hamburger | 474 | 0.579932226 | 0.562173296 | -0.017758930 |
| ray | 468 | 0.376274898 | 0.355764530 | -0.020510367 |
| airplane | 659 | 0.660949179 | 0.638432752 | -0.022516427 |
| spoon | 485 | 0.155743210 | 0.132420264 | -0.023322946 |
| axe | 552 | 0.263744342 | 0.238783770 | -0.024960572 |
| camel | 640 | 0.538157149 | 0.498755826 | -0.039401323 |
| hammer | 581 | 0.261823401 | 0.218843815 | -0.042979586 |

## 4. So sánh ngoài checkpoint (descriptive)

| Model/main@3600 | Clean full mAP | P200 | Prefix AP200 | Masked full mAP | P200 | Prefix AP200 |
|---|---:|---:|---:|---:|---:|---:|
| S0 | 0.186983950 | 0.449897828 | 0.699090253 | 0.106077072 | 0.225162055 | 0.329966024 |
| R0 | 0.305843784 | 0.423951464 | 0.507648816 | 0.212324934 | 0.297152689 | 0.359103953 |
| F2 | 0.471365186 | 0.514762828 | 0.533986756 | 0.334816536 | 0.349045722 | 0.366276100 |
| F2_MP/μ_I | 0.474080179 | 0.489338675 | 0.503827883 | 0.342202940 | 0.339020536 | 0.354761560 |
| F2_MP-Q/q | 0.491669182 | 0.520355275 | 0.535433599 | 0.355752598 | 0.358522295 | 0.373294285 |

Controls khác training/sampler/architecture/scheduler không causal. Không so q@3600 với best_clean của baseline để gọi fixed-step gain. S0@best_clean1800 vẫn cleanP200 .670306/prefix .736407; đó là comparison khác.

## 5. Verification / inference contract

- Tiny CPU q-only vs full-model q exact; predictor calls0. Real pretrained/real-data parity trên32clean và32severe masked inputs exact0. Full q evaluation predictor hook đếm **0calls**; không chỉ gọi full model rồi bỏ μ output.
- Main model state hashes trước/sau unchanged, no optimizer/backward. Predictor vẫn được load để audit/parity; việc không forward nó không đồng nghĩa đã đóng gói model tối giản hay benchmark latency/VRAM deployment.
- Clean q cả5scalar/top200/embedding exact với measurement trước; photo gallery embedding exact. Model loader kiểm tra checkpoint/model/frozen-original hashes với archived source. Không dùng unfinished whole-campaign verifier.
- IndependentCPU từ saved features:10conditions full saved top200 relevance/P200/3AP200/scalars, IDs/labels/98667maskrecords, class/query deltas; fullAP second full-gallery sort chỉ16queries/condition=160cases, maxdelta **1.62725e-6** (float32near ties), không full all-query second evaluator. No independent second encoder; state/call counts audit từ serialized parent assertions.
- Independent receipt `outputs/fusion_mp_q_evaluation_20260909T145000Z/independent_cpu/receipt.json`, parent `outputs/fusion_mp_q_evaluation_20260909T145000Z/parent_verification.json`:PASS. q features all10conditions/gallery và fullperquery probe saved; không đưa tensor lênGit.
- Checkpoint SHA `58fb5fb2aa0f822d1df8bee9192a0bda5eec1580b4ce70ed9ae243fdcf9fb14e`; training HEAD `e2eb341`, training source `ab064ced3b11c95f142ac94b336072d88312f5e14d776a386416bbf3cf70f29e`. New q-only diagnostic source `6c5f39041205c5b60b8220857e2accc651ff11e8e58f264834c9f775df7b15aa` (archive); elapsed111.924s gồm parity/encodes/ranking/serialization, không inference latency benchmark.
- Root `outputs/fusion_mp_q_evaluation_20260909T145000Z`; script `scripts/evaluate_fusion_mp_q.py`. CPU `--cpu-self-check`; GPU `--output <fresh-root>` chỉ khi được userduyệt. Existing run [y40hu06b](https://wandb.ai/a-cctest05187-erd/spica/runs/y40hu06b) là **μ_I training results**, không log q mới vào run/aliases cũ, không tạo W&B training run mới.
- Không mới train/selection search/seed/official unseen. Runtime training vẫn ARM_FINISHED_UNVERIFIED, không rewrite historical evidence. Trước mắt chỉ xác minh q là readout candidate; không tự thay mặc định trainer hoặc production inference.

Full precision: [JSON](fusion_mp_q_readout_results_2026-09-09.json).

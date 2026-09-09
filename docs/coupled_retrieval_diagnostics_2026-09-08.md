# Coupled retrieval: head routing, context shuffle, and negative-pool diagnostic

## Kết luận chính

**Không có bằng chứng toàn sketch student/predictor đã collapse. Vấn đề tập trung ở photo output `mu_i`: các vector có hướng chung rất mạnh, và sampled photo-ranking objective có thể đạt loss thấp nhờ phân biệt photo pools, không cần nhiều thông tin riêng của query.** Đây là kết quả diagnostic ở checkpoint3600, không phải một campaign mới hoặc chứng minh nguyên nhân duy nhất của training trajectory.

- R1 cùng checkpoint/gallery: `mu_i` full mAP **0.064978**, pooled `q` **0.136524**, text-side `mu_t` **0.411763**. `mu_t` không nhận query label/target embedding ở inference; chỉ global learned context như kiến trúc đã chốt.
- R1 `mu_i` cosine trung bình giữa các query **0.951620**; `mu_t` **0.053603**. Shuffle context làm `mu_i` còn mean cosine0.951671 với output gốc, nhưng vẫn giảm AP200/P200: head không hoàn toàn bỏ qua sketch.
- R1 clean photo-ranking: negative full-pool loss **0.438463** → canonical negative cùng class với negative cũ **0.765608**. Chỉ thay negative photo; query, positive, checkpoint, negative class giữ nguyên.
- Constant-query centroid control của R1 vẫn đạt full-pool loss **0.450063**, gần intact0.438463; khi canonical-matched, constant loss **0.799805**, gần equal-score softplus(0.2)=0.798139.
- SIGReg thay đổi rõ geometry của `g`, nhưng không loại bỏ hướng chung/shortcut ở `mu_i`. Không kết luận SIGReg vô hiệu ở mọi vị trí hoặc scale.
- **Không đổi inference policy/promotion:** mainline R1/R1_SIG vẫn dùng `mu_i`. `q`/`mu_t` ở đây là counterfactual diagnostic, không chọn head hậu nghiệm rồi tuyên bố campaign thắng.

## 1. Protocol và phạm vi

User duyệt ba phép kiểm tra no-update. Root hoàn tất:

`outputs/coupled_retrieval_diagnostics_20260908T191840Z/`

- R0/R1/R1_SIG cùng **checkpoint3600**, không chọn lại best step.
- Retrieval: toàn bộ **10,963 clean pseudo-validation sketches**, **13,999 gallery photos**,20 validation classes; không official21 unseen classes và không9masked retrieval probes trong diagnostic này.
- Mỗi arm encode gallery bằng photo prompts **của chính checkpoint đó**. Trong một arm, tất cả heads/controls dùng đúng cùng gallery.
- `q`, `mu_i`, `mu_t` được lấy từ cùng sketch forward; `mu_t` được ghi thêm để phân biệt lỗi toàn predictor với lỗi photo branch.
- Main-head clean replay: R0/q, R1/mu_i, R1_SIG/mu_i khớp **cả5scalar metrics delta0.0 và toàn bộ top200 indices** với historical raw probe3600.
- Shuffle: một permutation toàn10963queries, NumPy seed **845701**, không shuffle chỉ trong minibatch đã xếp theo class. Đầu ra được tính lại bằng **predictor trên patch contexts đã hoán vị**; pooled `q` được tính lại từ pooled context. Query labels/paths giữ nguyên, donor labels không vào forward.
- Permutation có **4.998632% same-class donors**, được báo cáo chứ không giả cross-class tuyệt đối. Global label-blind permutation là một control duy nhất, không phải seed sweep. Một permutation giữ multiset outputs; global hubness không đổi theo construction, nên quan trọng là per-query overlap, output cosine và relevance metrics.
- Pool counterfactual: **32×32=1,024** train sketches, đủ84classes; đúng first1024 historical query/positive/full-negative triplets. Clean và region-corrupted views đều được kiểm tra. Cùng triplets, canonical replacement photos và mask metadata ở cả3arms.
- Positive pool:8,400canonical photos/100per train class. Full negatives theo sampler gốc từ58,950photos, khác query class. Canonical negative thay thế lấy **đúng class của full negative cũ**, uniform trong100canonical photos/class, private Python RNG seed845701. Không biến same-class relevant thành negative.
- **15.234375%** original full-negative observations trong mẫu đã thuộc canonical pool; không loại chúng để phóng đại khác biệt. Positives và negatives có thể trùng ảnh giữa những queries khác class; không ép pools disjoint.
- Train control permutation cũng global, same-class fraction1.464844%. Constant control là normalized empirical mean của1024embeddings cùng head/view, dùng chung cho mọi query; không tối ưu một constant vector bằng labels/photos.
- Model eval, no-grad; **0optimizer updates,0backward calls**, không tạo optimizer. Model state hashes trước/sau giống nhau; original checkpoint files không bị ghi. Không áp SIGReg loss hoặc resample SIGReg trong diagnostic.

Training source: HEAD `0a27fa2`, SHA256 `07a3e8e750189bbc622a9edd71d075de6abb43ad50fc8ee1ae70971e29586cbb`.

Diagnostic source SHA256: `dd3c21ef1ad467590c4283354ac977515c539371e934f4340d84086c0572720a`; source archive đi kèm. Runtime-critical src/config/uv.lock/pyproject được so với training archive trước inference. Dùng **archived verifier checkpoint loader**, không dùng bản sửa verifier còn dở trong working tree. Commit/report sau diagnostic không phải executed snapshot.

## 2. Test1 — cùng checkpoint, khác output head

Mọi số0–1. AP200 trong bảng là **prefix-positive**; hai denominator khác có trong JSON/raw NPZ.

|Arm|Head|Clean full mAP|Clean P200|Clean prefix AP200|Vai trò|
|---|---|---:|---:|---:|---|
|R0|q|0.305844|0.423951|0.507649|Main inference|
|R1|q|0.136524|0.319074|0.495089|Auxiliary diagnostic|
|R1|mu_i|0.064978|0.121007|0.190061|Main inference|
|R1|mu_t|0.411763|0.472243|0.499689|Auxiliary diagnostic|
|R1_SIG|q|0.139354|0.324632|0.497926|Auxiliary diagnostic|
|R1_SIG|mu_i|0.066530|0.127221|0.203757|Main inference|
|R1_SIG|mu_t|0.408038|0.470537|0.499863|Auxiliary diagnostic|

Diễn giải:

1. `q` của R1 tốt hơn `mu_i` rất rõ nhưng vẫn không phục hồi full mAP/P200 của R0/q. Vì vậy vấn đề không chỉ là chọn nhầm head; training package/shared gallery cũng khác R0.
2. Cùng predictor architecture, `mu_t` chuyển thông tin sketch sang photo retrieval tốt. Không thể nói toàn predictor không học được hoặc gallery R1 hoàn toàn hỏng.
3. `mu_t` full mAP/P200 cao hơn R0/q@3600 nhưng prefix AP200 thấp hơn. Đây không phải chiến thắng tất cả metrics; cũng không được đổi official head sau khi nhìn validation.
4. S0 cũ@best_clean1800 vẫn prefix0.736407/P2000.670306/full mAP0.265980. External S0 không phải matched control của diagnostic và không bị thay thế bởi phát hiện mu_t.

## 3. Test2 — hoán vị context, giữ labels/galleries

|Arm/head|Full mAP intact→shuffle|P200 intact→shuffle|Prefix AP200 intact→shuffle|Cosine output gốc/shuffle|Mean top200 overlap|
|---|---|---|---|---:|---:|
|R0/q|0.305844→0.072088|0.423951→0.051128|0.507649→0.058513|0.136456|0.099530|
|R1/q|0.136524→0.057255|0.319074→0.050604|0.495089→0.063598|0.289742|0.147843|
|R1/mu_i|0.064978→0.051995|0.121007→0.050329|0.190061→0.066859|0.951671|0.653694|
|R1/mu_t|0.411763→0.082903|0.472243→0.050549|0.499689→0.057985|0.054689|0.039078|
|R1_SIG/q|0.139354→0.057593|0.324632→0.050781|0.497926→0.063890|0.266489|0.143673|
|R1_SIG/mu_i|0.066530→0.052103|0.127221→0.050443|0.203757→0.066794|0.942587|0.625599|
|R1_SIG/mu_t|0.408038→0.082775|0.470537→0.050828|0.499863→0.058352|0.053708|0.042839|

Actual shuffled predictor output khớp intact output indexed bởi permutation: max absolute1.1920929e-7; q exact0. Đây kiểm tra đúng intervention, không phải chất lượng retrieval.

**Photo branch vẫn dùng một ít thông tin sketch:** P200/AP200 giảm khi shuffle. Tuy nhiên phần chung rất lớn: R1/mu_i giữ65.37% top200 khi thay context bằng sketch khác, so R0/q9.95% và R1/mu_t3.91%. Không gọi complete query-independence.

## 4. Test3a — geometry trên clean validation

Pair cosines loại self-pairs, tính đúng trọng số số cặp, không lấy ngẫu nhiên một minibatch. Covariance tính sau unit-normalization: `C=(U-mean(U))^T(U-mean(U))/N`. Entropy effective rank là `exp(-sum p log p)`, **p=eigenvalue/trace**, không phải singular-value entropy convention. Trace và common direction cần đọc cùng rank; centered rank đơn lẻ không phát hiện năng lượng lớn nằm trong mean.

|Arm/head|Mean pair cosine|Within-class cosine|Between-class cosine|trace(C)|Covariance entropy effective rank|
|---|---:|---:|---:|---:|---:|
|R0/g|0.350187|0.610054|0.336350|0.649753|93.508|
|R0/q|0.135356|0.530846|0.114298|0.864565|53.198|
|R1/g|0.416299|0.644025|0.404174|0.583647|97.583|
|R1/q|0.288811|0.602407|0.272113|0.711124|56.179|
|R1/mu_i|0.951620|0.974617|0.950395|0.048376|33.500|
|R1/mu_t|0.053603|0.485587|0.030601|0.946311|37.420|
|R1_SIG/g|0.122275|0.489132|0.102741|0.877645|75.943|
|R1_SIG/q|0.265243|0.597303|0.247561|0.734690|51.797|
|R1_SIG/mu_i|0.942534|0.970338|0.941054|0.057461|32.933|
|R1_SIG/mu_t|0.052926|0.492467|0.029522|0.946987|36.438|

- R1 `mu_i`: khoảng95.16% unit-feature second moment nằm trong mean direction (`||mean(U)||²=1-trace(C)`). Residual variance/effective rank khác0, nên **anisotropic/partial degeneration**, không chứng minh mọi vector bằng nhau.
- Within-minus-between cosine R1/mu_i≈0.02422, R1/mu_t≈0.45499. Photo branch còn class signal nhưng yếu hơn text branch trên các đo này.
- `g` không collapse. SIG làm raw centered covariance trace của g tăng **53.72498→113.37116**, unit mean pair cosine giảm0.41630→0.12228. Tuy vậy entropy effective rank **97.58→75.94**, nên không nói mọi diversity metric đều cải thiện hoặc distribution đã Gaussian.
- `mu_i` chỉ cải thiện modest; mean cosine0.95162→0.94253. Regularize g không đủ bảo đảm predictor output tránh hướng chung.
- Prompted gallery covariance entropy rank R0/R1/SIG **100.881/43.898/44.437**, participation ratio **37.021/8.409/8.545**. Geometry gallery có thay đổi, nhưng mu_t retrieval tốt bác bỏ diễn giải gallery hoàn toàn không còn hữu ích.

## 5. Test3b — matched negative-photo pools

Loss per observation: `softplus(0.2+s_negative-s_positive)`, cosine-normalized vectors. Đây là **float64 diagnostic recomputation tại fixed checkpoint eval**, không phải tuyên bố bit-identical minibatch losses của historical training. Không so tổng loss giữa models có objective khác nhau.

### Main retrieval heads; cả clean và corrupted

|Arm/head|Query control|Clean full-neg loss|Clean canonical-neg loss|Masked full-neg loss|Masked canonical-neg loss|
|---|---|---:|---:|---:|---:|
|R0/q|Intact|0.591295|0.633749|0.613141|0.664034|
|R0/q|Shuffled|0.746634|0.796168|0.739932|0.797405|
|R0/q|Constant centroid|0.635430|0.798059|0.638447|0.798297|
|R1/mu_i|Intact|0.438463|0.765608|0.441810|0.773527|
|R1/mu_i|Shuffled|0.460844|0.799341|0.458859|0.799648|
|R1/mu_i|Constant centroid|0.450063|0.799805|0.450125|0.799830|
|R1_SIG/mu_i|Intact|0.439829|0.764219|0.443294|0.771614|
|R1_SIG/mu_i|Shuffled|0.462952|0.799149|0.461485|0.799438|
|R1_SIG/mu_i|Constant centroid|0.451629|0.799732|0.451742|0.799760|

### Clean intact margin `s_positive - s_negative`

|Arm/head|Full negatives|Canonical matched negatives|
|---|---:|---:|
|R0/q|0.418963|0.325936|
|R1/mu_i|0.849899|0.064306|
|R1_SIG/mu_i|0.845211|0.066859|

R1 canonical-negative loss tăng0.327146; paired descriptive SE0.005115. SIG tăng0.324390/SE0.005079; R0 chỉ tăng0.042453/SE0.001080. SE chỉ mô tả1024sampled observations có ảnh/classes lặp, không significance qua seeds/classes.

**Control quyết định:** constant R1/mu_i có full-negative margin0.821886, gần intact0.849899; canonical-negative margin chỉ0.001316. Nghĩa là tại checkpoint đã học, một hướng query chung đã tách được phần lớn positive-vs-full-negative score gap. Khi loại khác biệt pool bằng negative cùng canonical distribution, khả năng phân biệt class của mu_i yếu hơn nhiều.

Text-side R1/mu_t clean intact loss **0.686543→0.692285** khi đổi full→canonical negatives, chênh0.005743; dù ranking loss cao hơn mu_i với full negatives, nó có validation retrieval tốt hơn. Điều này minh họa objective sampled ranking ban đầu không phải surrogate đủ đáng tin cho category retrieval khi positive/negative pools lệch phân phối.

### Điều đã kiểm tra và điều chưa chứng minh

- Đã can thiệp có kiểm soát **negative identity** tại cùng negative class, model và query; kết hợp với constant/shuffled-query controls cho bằng chứng trực tiếp rằng fixed model tận dụng query-independent pool discrimination.
- Chưa retrain với matched sampling: không thể tuyên bố đây là nguyên nhân duy nhất, hay sửa sampler chắc chắn phục hồi clean AP200/unseen transfer.
- R0 cũng có pool bias (constant loss0.635<0.798), nhưng còn nhiều lợi ích từ đúng query khi cả hai pools matched. Sampler giống nhau không có nghĩa mọi architecture/loss route chống shortcut như nhau.

## 6. Diễn giải tổng hợp và hướng sửa chưa thực hiện

1. **Không phải toàn student/predictor không học:** g/q/mu_t còn thông tin; mu_t từ cùng predictor đạt mAP0.4118.
2. **Không phải chỉ thiếu SIG hoặc train quá ngắn:** mu_i ưu tiên common direction, đạt sampled ranking loss thấp ngay cả với constant query. Thêm Gaussian regularity ở g không trực tiếp sửa tín hiệu supervision này.
3. **Lỗ hổng objective:** R0/q nhận ranking+text CE trực tiếp cùng một representation. R1 tách photo ranking ở mu_i và semantic CE ở mu_t; shared parameters cung cấp đường gián tiếp, không đảm bảo mu_i giữ class discrimination. Photo prompts live cho phép đồng thích nghi của query/photo spaces.
4. **Ưu tiên nếu user duyệt bước tiếp:** khóa/khớp distribution positive-negative (không loại hard different-class negatives), và kiểm tra semantic supervision trực tiếp trên representation thật sự dùng retrieval. Mọi đổi sampler/loss/head cần campaign mới có matched control; không sửa raw results/current checkpoints.
5. **R0 vs S0 vẫn là câu hỏi khác:** thay native CLIP head, fine-tune student, masks và text-bank learning cùng thay đổi. Ba diagnostics này phân biệt failure ở R1; không cô lập nguyên nhân clean AP200 gap S0→R0.

## 7. Kiểm chứng, roundoff và evidence

- Main-head CUDA replay3arms, fixed3600: all5scalar deltas0.0; top200 indices exact. Không phải9selected-alias/full9mask replays.
- **Independent CPU verifier:** full-gallery stable sorts từ saved FP32 embeddings cho **14cases** (7heads×intact/shuffle), tự tính full AP/P200/3AP200; tự tính geometry và paired losses bằng Torch CPU float64, kiểm tra1024historicaltriplets/canonicalclassmembership/masks/checkpoint+source hashes.
- CPU/GPU scores gần tie có thể sort khác: tổng1,687query-case rows/3,392top-index elements khác giữa14cases. **Max scalar difference4.5927280e-7** (R1/mu_i intact P200), không claim bitwise CPU/GPU ranking equality. Max per-query P200 difference0.005000017, fullAP4.8354554e-5, prefixAP2000.001111106; đây là observed cross-device differences, không chỉ một phép làm tròn scalar mean.
- Independent receipt dùng scalar tolerance8×float32eps cho re-sorted aggregates và ghi actual deltas; không coi đó là bound lý thuyết cho mọi possible near-tie distribution. Positive-eigenvalue rank/ill-conditioned condition ratio không dùng để kết luận collapse.
- **Parent CPU check:** tự reconstruct P200/3AP200 từ saved top200+labels cho14cases; max per-query delta **1.3765468e-7** do FP32/FP64 arithmetic. Main-head clean/masked paired losses+controls tự recompute, tolerance1e-12; PASS.
- First diagnostic root `outputs/coupled_retrieval_diagnostics_20260908T191524Z/` **FAIL**, preserved. Cause: P200 array mean CPU vs evaluator GPU scalar guard1e-12; measured R0 delta9.435215997e-9. New script explicitly allows float32 epsilon1.1920929e-7 for that one comparison and records deltas. Không sửa original scalar hay per-query arrays để ép PASS.
- Successful diagnostic wall time **118.508s**, including setup/CPU/inference, excluding independent CPU audit; per-arm GPU peak allocated1.82–1.83GB. No throughput/convergence claim.
- Original campaign `verification/summary.json` vẫn giữ **FAIL: KeyError 'R0'**; diagnostic verification không tự biến toàn campaign thành VERIFIED. Three training arms đã COMPLETE3600, W&B histories đã được đọc riêng; không có thay đổi W&B runs/aliases hoặc diagnostic report run mới.

Evidence dưới successful root:

- `summary.json`, `runtime.json`, `diagnostic_provenance.json`, `source_snapshot/`.
- `<arm>/validation_features.npz`, `shuffled_features.npz`, `identities.json`, `validation_geometry.json`.
- `<arm>/<head>_{intact,shuffled}.npz`, `<head>_shuffle_comparison.npz`, `retrieval.json`.
- `<arm>/train_features.npz`, `train_records.json`, `train_masks.json`, `train_paired_ranking.json`, `train_geometry.json`.
- `independent_cpu/receipt.json`, `verify_independent_cpu.py`, `audit.log`.
- `parent_verification.json`, `verify_parent.py`.
- Compact raw-precision report: [JSON](coupled_retrieval_diagnostics_2026-09-08.json).

Standalone CPU self-check (không tạo V1 pytest suite):

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv/bin/python scripts/diagnose_coupled_retrieval.py --cpu-self-check
```

Executed command/environment (historical; không chạy lại nếu chưa được yêu cầu):

```bash
LD_LIBRARY_PATH=/run/opengl-driver/lib HF_HUB_OFFLINE=1 WANDB_MODE=disabled \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \
.venv/bin/python scripts/diagnose_coupled_retrieval.py \
  --campaign-root outputs/coupled_predictive_execution_20260908T153000Z \
  --output outputs/coupled_retrieval_diagnostics_20260908T191840Z
```

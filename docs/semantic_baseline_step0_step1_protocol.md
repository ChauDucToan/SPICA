# Semantic baseline — Bước 0 và thiết kế Bước 1

Ngày chốt thiết kế: **2026-09-07**. Progress checkpoint trước tài liệu này: `4cd4a152d016904d330ab098822856a5e0159c53`.

## 1. Trạng thái, phạm vi và quyền thực thi

- **Bước 0: protocol phát triển nội bộ đã chốt; audit khả năng reproduce SeCo đã thực hiện.** SeCo paper-equivalent benchmark vẫn BLOCKED bởi evaluator/code/checkpoint chưa xác minh được, không phải bởi SPICA chưa chạy C/M.
- **Bước 1: DESIGN ONLY / NOT IMPLEMENTED / NOT TRAINED.** Đây không phải config Hydra runnable và không phải giấy phép launch GPU.
- User yêu cầu commit tiến độ hiện tại và commit mỗi phần công việc tiếp theo khi hoàn tất. Không tự push. Commit đầu lưu source/config/tests/reports C/M, không biến commit bàn giao thành training snapshot.
- Quyền của lần này: audit read-only, thiết kế, tài liệu, kiểm tra CPU, local commits. Không training mới, không diagnostic real-CLIP/GPU, official unseen, multi-seed, lambda search, liên hệ tác giả hoặc upload W&B mới.
- Giữ nguyên datasets, historical outputs/checkpoints/config/source snapshots. `outputsnewgate/` là intermediate CPU smoke, vẫn local/untracked; không xóa/di chuyển hoặc dùng làm chứng nhận campaign mới.

### Quyết định user đã xác nhận trong lần này

1. **Promotion Bước 1:** tăng clean `mAP@200_prefix_positive`, đồng thời clean P@200 và full mAP **không giảm**, so `best_clean` với `best_clean` trên pseudo-validation. Masked metrics chỉ báo cáo trong giai đoạn chọn clean architecture này.
2. **S2:** `lambda_anchor=1.0`, không search. Anchor là mean theo toàn bộ pseudo-train classes của `1-cosine(t_learned,t_fixed)`. Đây là hệ số pilot định trước, **không calibrated, không optimal, không phải hệ số được bê từ paper KgCoOp**.
3. Tiếp tục quy tắc ngân sách/selection đã được duyệt: 3600 updates/run, probes mỗi600, không early stopping, ba selections riêng, prefix-positive selection và log đầy đủ denominators. Đây là thiết kế cho run tương lai, không phải lệnh thực thi.

## 2. Điểm xuất phát đã kiểm chứng

Đọc [kết quả C/M3600](masked_view_3600_results_2026-09-07.md), [pilot1800](masked_view_pilot_report_2026-09-07.md) và [gradient diagnostic](masked_view_gradient_diagnostic_and_direction2_2026-09-07.md).

- C/M3600 đã hoàn thành từ đầu, seed42/pseudo3407, tất cả original CLIP weights frozen, chỉ sketch/photo prompts học; source training SHA256 `e60c74aa84fe09ee2216248408f2e145460a7b421294a991f97a5a00291309af`.
- C/M best_masked: prefix AP200 `0.343001/0.402876`, P200 `0.301417/0.270547`, full mAP `0.141752/0.117118`. Đây là trade-off giữa metrics, không universal dominance.
- C best_clean@1800 là checkpoint tốt nhất theo rule trong các probes đã quan sát, không baseline tối ưu toàn cục. Ngân sách1800/3600 không chứng minh hội tụ.
- Photo full/masked rank gradients dương48/48 trên tập diagnostic; severe sketch conflict tập trung hơn ở CE. Không suy ra causal explanation của unseen retrieval hoặc tự giảm/bỏ masked CE.
- KD, soft-prompt campaign mới, text anchor, multimodal coupling, patch-JEPA: **NOT RUN**.
- Proposal3600 cũ có các đoạn trạng thái pre-execution; giữ nguyên như tài liệu lịch sử. Báo cáo results3600 mới là nguồn trạng thái hoàn tất. Tài liệu này supersede thứ tự nghiên cứu hướng2, không sửa kết quả hoặc giả định run cũ.

## 3. Bước 0 — hợp đồng dữ liệu và benchmark

### 3.1 Mục tiêu

**Zero-shot category-level sketch–photo retrieval dưới raster information deletion, giữ clean retrieval.** Unseen là không dùng class đó trong downstream training; không khẳng định CLIP chưa gặp concept trong pretraining. Không đổi category relevance thành paired-instance relevance.

Hai trục độc lập: (a) unseen-category generalization, (b) robustness khi mất mực. Không gọi raster deletion là true-stroke removal, temporal prefix hoặc early-sketch retrieval.

### 3.2 Split và identity cố định

Nguồn: `configs/data/sketchy_104_21.yaml`, root `datasets/Sketchy`, các manifests trong `zeroshot0/`. Chỉ đọc, không sao chép/sửa ảnh hoặc manifests.

| Phần | Classes | Sketches | Photos | Quyền sử dụng |
|---|---:|---:|---:|---|
| Pseudo-train |84|46,624|58,950|Training, text bank, train-only diagnostic|
| Pseudo-validation |20|10,963|13,999|Probes/selection/so sánh phát triển; không gradient|
| Official unseen |21|Không load cho bước này|Không load cho bước này|Không selection, không text-anchor bank, không training hoặc evaluation|

125 classes gốc =104 downstream-train +21 official unseen; pseudo3407 chia104 thành84+20. Pseudo-validation class IDs theo train map: `7,13,16,27,28,31,33,34,39,42,45,51,52,53,60,65,75,86,90,99`.

- Pseudo split identity: `3e02604d2ed315aa254d4264ec440a7e50233c7c9b175be224519feafda88425`.
- Training/validation ordered-entry hashes, class-list hashes, file hashes được lưu tại [audit identities](semantic_baseline_step0_audit.json); phải recompute bằng code hiện hữu trước launch, không chỉ tin counts hoặc file path.
- Positive sampling giữ **same-class canonical pool8,400 photos**, từ pairing manifest `outputs/pairing_preparation_20260906_145238/sketchy_pseudo_train_pairing.json`, SHA256 `545f67663682ed5fb79397c775848b90e206579647e605cba24cb6d4dcf8104c`.
- Negatives từ pseudo-train photo pool58,950, loại class của query theo sampler hiện hữu. Không đổi sang exact-instance pairing, hard-negative mining hoặc class-balanced batching.
- Train sampler shuffle giữ nguyên; evaluation query/gallery ordering phải cố định và hash. Gallery chỉ là13,999 pseudo-validation photos, không trộn train gallery hoặc official unseen.

### 3.3 Metrics và selection

Tái sử dụng evaluator hiện hữu, query-macro average và stable sort. Với `rel(k)` theo category, `P(k)` là precision tại rank k, `R` là tổng positives trong gallery, `R200` là positives trong prefix200 và `S200=sum_{k<=200} P(k)*rel(k)`:

| Metric | Định nghĩa |
|---|---|
| P@200 |`R200/200` (gallery thật lớn hơn200)|
| mAP@200_prefix_positive |query mean của `S200/max(1,R200)`|
| mAP@200_all_relevant |query mean của `S200/R`|
| mAP@200_min_relevant_k |query mean của `S200/min(R,200)`|
| full_mAP |AP trên toàn gallery, denominator R, rồi query mean|

Query không có positive gallery là lỗi protocol theo contract evaluator, không âm thầm loại hoặc cho0. Không đổi implementation/defaults historical.

Mỗi run chạy đủ3600 optimizer updates, evaluate/save tại `0,600,1200,1800,2400,3000,3600`:

- `latest`:3600.
- `best_clean`: argmax clean prefix-positive mAP200, chỉ steps600–3600, tie giữ step sớm nhất.
- `best_masked`: cùng rule trên masked macro9 prefix-positive mAP200.
- P200/full mAP và denominators phụ phải lấy ở **chính checkpoint đó**. So same-selection-type; latest–latest vẫn báo riêng.
- Probe0 chỉ reference, không tham gia best selection. Không early stop hoặc kéo dài run khi curve xấu/tốt.

**Promotion rule đã duyệt:** gọi `A(X), P(X), F(X)` là ba clean metrics prefix AP200/P200/full mAP tại `X.best_clean`. Candidate X đủ điều kiện thay S0 nếu và chỉ nếu:

`A(X)>A(S0) AND P(X)>=P(S0) AND F(X)>=F(S0)`.

Dùng raw scalar đầy đủ precision, không so rounded table. Không numerical slack cho promotion; replay tolerance chỉ dành kiểm tra implementation, không biến metric giảm thành không giảm. Không thêm post-hoc threshold, không dùng masked scores để chọn soft/text architecture.

- S1 và S2 đều so S0; attribution riêng của text anchor phải báo S2–S1 tại cùng selection rule, kể cả S2 thắng S0 nhưng thua S1.
- Secondary tie-breaker định trước trong thiết kế này (ngoài promotion gate user đã duyệt): nếu cả hai pass, chọn clean prefix AP200 cao hơn; nếu bằng nhau, ưu tiên P200 rồi full mAP; nếu cả ba bằng nhau, ưu tiên S1 vì ít loss hơn. Nếu không candidate nào pass, giữ S0, không tự mở search.
- Báo các chênh lệch latest và best_masked như secondary, không thay bằng peak khác.
- Đây là gate chọn baseline cho development ở một training seed/split, **không phải statistical significance hoặc chứng nhận SOTA/robust improvement**. Pseudo-validation đã dùng nhiều lần nên có nguy cơ development overfitting; official test vẫn giữ riêng.

### 3.4 Mask evaluation — giống nhau cho mọi arm, không mask train ở Bước 1

Giữ `ink_centered_square_v1`, threshold `<0.9`, requested fractions `[0.25,0.50,0.75]`, evaluation seeds `[101,202,303]`. 9 conditions ×10,963 queries =98,667 query-mask records/probe. Cả ba là **mask seeds**, không phải training seeds.

- Raster được xóa trước encoder; preserve ít nhất một ink pixel theo policy hiện hữu. Ghi requested/realized fraction, ink counts, bbox, status, input/output hashes.
- Gallery giữ clean. Không dùng unmasked query, teacher hoặc class label như inference input.
- So cùng mask hashes giữa arms và probes; không tạo mask dễ hơn cho candidate. Blank/unreachable phải báo và kiểm tra, không âm thầm bỏ query.
- Bước 1 train `full_full`, **không tạo hoặc dùng train masks**. Phải có two-identical-full-view path không gọi `_masked_training_views()`: C lịch sử vẫn tạo masks/metadata rồi bỏ masked tensor khi dùng full_full, nên không được copy nguyên routing đó. Chỉ evaluation tạo9 masks. Train mask seed4242/fractions được bảo lưu cho Bước 3 tương lai, không được kể như robustness training đã xảy ra ở S0/S1/S2.

### 3.5 SeCo reproducibility gate và research evidence

Primary SeCo: [arXiv2608.03120v1](https://arxiv.org/html/2608.03120v1). Repo inspected tại `e70aead311777a884c350f5bc96a2aa85a708d85`: [SeCoSBIR](https://github.com/huutuan1705/SeCoSBIR/tree/e70aead311777a884c350f5bc96a2aa85a708d85), public tree chỉ có `index.html` tại thời điểm audit2026-09-07. Snapshot responses/hash nằm trong audit JSON. Không suy ra họ chưa làm thí nghiệm.

**BLOCKED trước claim reproduce/vượt SeCo official:**

1. Chưa có training/evaluator/checkpoint release có thể kiểm chứng từ tree được inspect.
2. §5 nói top200 nhưng đồng thời báo mAP@all; chưa có công thức AP200/denominator/truncation/zero-positive policy đủ rõ. SPICA prefix-positive chưa paper-equivalent.
3. §4.4 liệt kê LayerNorm trainable dù nói CLIP frozen. SPICA giữ toàn bộ original CLIP frozen. Bản giữ LN frozen phải ghi `constraint-matched adaptation`, không gọi exact reproduction.
4. Cần class/file lists cho từng Sketchy split, augmentations, optimizer/scheduler chi tiết, seed, prompt initialization, checkpoint/validation selection, và cách tạo visual prompts ở inference không cần oracle query label.

Đây cũng là danh sách câu hỏi để xin tác giả khi user cho phép; lần này **không gửi liên hệ**. Các blockers không ngăn thiết kế/campaign development nội bộ, nhưng chặn kết luận paper-equivalent.

Kết quả paper SeCo: text-guided adaptation + visual frozen-reference InfoNCE, không explicit text-prototype anchor trong Eq17; augmentation đi vào frozen reference, clean input vào adapted branch. Paper chưa báo raster mask protocol của mình.

KgCoOp: [paper2303.13283](https://arxiv.org/abs/2303.13283), [official trainer pin c43ccc32c8739f64b165cfa205fcc6c0f5bb5b17](https://github.com/htyao89/KgCoOp/blob/c43ccc32c8739f64b165cfa205fcc6c0f5bb5b17/KgCoOp/trainers/kgcoop.py). Code normalized class features, `1-mean(cosine)` và CE + weighted anchor. Paper dùng weight8.0; **SPICA chọn1.0 theo quyết định pilot riêng**, không tuyên bố replicate KgCoOp.

Các bài học khác chỉ dùng định hướng, không chép nguyên architecture/loss:

- [PromptSRC pin bb95c77b634d63488f2cad81ff4a72d53bdd06d5](https://github.com/muzairkhattak/PromptSRC/tree/bb95c77b634d63488f2cad81ff4a72d53bdd06d5): image/text feature agreement, posterior KL, textual template diversity, parameter aggregation. Gaussian ở aggregation weights, không latent Gaussian prior.
- [MaPLe](https://arxiv.org/abs/2210.03117), [SpLIP](https://arxiv.org/html/2407.04207v1): multimodal coupling không tự bảo đảm chống quên; SpLIP còn train LN/jigsaw, không âm thầm đưa các thành phần đó vào SPICA.
- [ProGrad](https://arxiv.org/html/2205.14865v4), [CoCoOp](https://arxiv.org/abs/2203.05557): gradient control/input-conditioned prompts là các hướng khác, chưa phải candidate Bước 1.
- Classification base-to-new results không tự chuyển thành SBIR mAP evidence. Anti-collapse, semantic generalization và robustness trước information loss là các câu hỏi khác nhau; JEPA/AC-MTM/TC-JEPA không tự giải quyết cả ba.

## 4. Bước 1 — thiết kế thí nghiệm S0/S1/S2

Proposed campaign identifier: `frozen_prompt_semantic_text_step1_2026-09-07`; roles `semantic_text_S0`, `semantic_text_S1`, `semantic_text_S2`. Chưa đăng ký chúng trong code hoặc tạo configs runnable.

### 4.1 Chỉ thay text bank/anchor

| Thành phần | S0 | S1 | S2 |
|---|---|---|---|
| Sketch/photo visual prompts |3 shallow tokens mỗi modality|Giống S0|Giống S0|
| Text context |Fixed `a photo of a`|4 learnable tokens shared across classes|Giống S1|
| Text anchor weight |0|0|**1**|
| Trainable parameter count dự kiến |4,608|6,656|6,656|
| Query training |full + full|full + full|full + full|
| Rank + hard-label CE |Có, query CE only|Giống S0|Giống S0|
| Photo CE/coupling/adapters/deep prompts |Không|Không|Không|
| KD/teacher visual/SIGReg/JEPA |Không|Không|Không|

- Original visual/text towers, token/positional embeddings, LN, native projections và logit scale frozen; chỉ `P_s,P_p` và S1/S2 context `C` học.
- Photo prompt trainable trong cả3 arms; rank cập nhật photo prompt. Trong kiến trúc **độc lập** Bước 1, query CE không có gradient trực tiếp vào photo prompt; text-context CE không biến photo encoder thành shared coupling.
- Soft context chung class-agnostic, không per-class learnable vectors, không dynamic/query-conditioned prompts. Hard-label target vẫn là class ID; soft context không có nghĩa label smoothing.
- Đây là test tối thiểu trước architecture shared context→visual projection. Projection W/private-token redesign/photo CE chỉ thuộc Bước 2 tương lai và cần contrast riêng.

### 4.2 Khởi tạo và text-bank contract

- Cả3 arms **from scratch** từ cùng pretrained CLIP và cùng seed42. Không warm-start S1/S2 từ C.best_clean hoặc S0.best_clean.
- CLIP: `ViT-B-32-quickgelu`, `openai`, normalized512-D outputs, existing224×224 transform. Pretrained safetensors identity SHA256 `e6d1bd7789aa45192b3bf90570a789b478bae1b74ebcce7eddd908e83a2b7c31`.
- P_s/P_p dùng initialization hiện hữu N(0,0.02), mỗi tensor3×768. Assert initial tensors equal giữa arms; text setup không được đổi sampler RNG hoặc visual initialization. Lưu RNG states trước/sau setup/probes.
- C của S1/S2 là4×512, khởi tạo từ frozen token embeddings của đúng prefix `a photo of a`.
- T0: normalized frozen text features cho **toàn bộ84 pseudo-train classes**, hard template `a photo of a {}`; sorted class IDs, underscores→spaces, không tự thêm dấu chấm, không thay tên class.
- S1/S2: `[SOT, C, class-name tokens, EOT, padding]`, CLIP tokenizer/positional layout/EOT pooling giống hard path. Verify prefix đúng4 tokens và metadata/embedding parity trước update; nếu tokenizer/layout không khớp thì fail readiness, không âm thầm đổi template hoặc thay reference.
- Compute T0 dưới no_grad và giữ detached. Helper `encode_class_text_bank()` hiện có `@torch.inference_mode()`; trước khi đưa bank vào graph CE/anchor phải **ra ngoài inference_mode rồi clone**, ví dụ `T0 = hard_bank.embeddings.detach().clone()`, để T0 là normal detached tensor có thể được autograd save-for-backward. Detach inference tensor đơn thuần chưa đủ. Compute learned T(C) có autograd qua frozen text transformer tới C; **không** bọc learned text forward bằng no_grad hoặc dùng cached detached trainable bank qua updates.
- Initial text-bank parity gate: `torch.testing.assert_close(T(C_init).float(),T0.float(),atol=1e-6,rtol=1e-6)`, cùng class order; ghi actual max error. Không đòi byte-equal text vectors giữa hai execution paths; phải byte-equal S1/S2 context init và P_s/P_p.
- Lưu tokenizer/model identity, ordered names/IDs, token IDs/EOT positions, T0 tensor/hash, initial context/learned-bank hash. Hash là byte identity; tolerance parity không đổi hash thành bằng nhau.
- Anchor trên train class bank, không chứa pseudo-validation/official names. Prototypes không là input retrieval; checkpoint replay dùng ảnh để tạo query/gallery, không oracle class.

### 4.3 Loss chính xác

Với normalized query `q_v`, positive/negative photo `p+`,`p-`, label y và text bank T:

`L_rank,v = mean_batch softplus(0.2 - q_v·p+ + q_v·p-)`

`L_CE,v = mean_batch CE(q_v @ T.T / 0.07, y)`

Hai full views giống input và dùng cùng sketch prompt:

`L_task = 0.5*(L_rank,1 + L_rank,2) + 0.5*(L_CE,1 + L_CE,2)`.

S2 thêm:

`L_anchor = mean_{c in all 84 pseudo-train classes}(1 - cosine(T(C)[c], stopgrad(T0[c])))`

`L_total = L_task + lambda_anchor * L_anchor`.

- Lambda rank/CE đều1.0; S0/S1 anchor0, S2 anchor1.0. T0 không EMA, không learned teacher; giữ cố định xuyên run.
- Anchor lấy **một lần mỗi optimizer update**, không nhân theo2 views, batch size hoặc tần suất class xuất hiện. Learned bank forward một lần/update, reuse graph cho cả2 CE và anchor; photos positive/negative encode một lần rồi reuse.
- S1 cũng có thể log detached anchor/drift để so sánh, nhưng không cộng vào objective; không để logging thay đổi RNG/gradients.
- Không clipping/label smoothing/temperature learning/gradient surgery hoặc weighting theo mask severity mới. Nonfinite features/loss/gradient hoặc norm0 là lỗi cần dừng an toàn và giữ artifacts.
- Khi T(C_init)≈T0, anchor≈0 và gradient≈0 là **đúng**. Không dùng ratio với g_anchor≈0 để calibration; không epsilon-divide, không perturb prototype giả để tạo hệ số. S2 không thể bị fail chỉ vì anchor chưa có gradient ở step0; task gradients vào C phải hữu hạn và được kiểm tra.
- Text anchoring là soft regularizer, không bảo đảm unseen generalization. CLIP semantic prior có thể không tối ưu task; cả S1/S2 có thể thua S0.

### 4.4 Optimizer, batch và ngân sách thiết kế

- AdamW, active visual/text prompt groups LR `1e-3`, weight decay `1e-4`; betas `(0.9,0.999)`, eps `1e-8`, amsgrad false theo defaults hiện hữu. Ghi resolved optimizer implementation/flags thực tế; không nâng dependency hoặc thay fused/foreach policy để tối ưu tốc độ trong contrast này.
- Scheduler constant `LambdaLR(lambda _:1.0)` như C/M reference. Không warmup/cosine hoặc per-arm schedule mới. S0 không có active text group; S1/S2 cùng group layout và hyperparameters.
- Batch32 original sketches/update, two-view query forward64, positives/negatives reuse. Workers4, shuffle true, drop_last true, eval batch256, query_chunk256, log_every10.
- 3600 updates/arm =115,200 original sketch observations và230,400 query-view forwards/arm, chưa tính probes. Chạy hết, không early stopping. Planned total3 arms10,800 updates, **chưa được launch**.
- Cả3 có identical original sample/positive/negative trace khi cùng seeds/setup; không đổi sampling để phục vụ text bank hoặc anchor. Full/full control giữ parity với C3600, không dùng một-view codepath chỉ vì toán học tương đương.
- Budget inherited từ protocol để kiểm soát contrast, không tối ưu SGD/AdamW hoặc hội tụ. SeCo10 epochs/SGD không tương đương budget này.
- Khi có quyền chạy: tuần tự S0→S1→S2, fresh directories/run IDs, không đổi source/config giữa arms. Nếu gate fail, giữ evidence và xin quyết định, không tự sửa giữa campaign rồi tiếp tục như matched.

### 4.5 Tracking và provenance

Tái sử dụng W&B destination đã duyệt `a-cctest05187-erd/spica`, Hydra/provenance/checkpoint infrastructure; **không init online cho design/CPU tests**. Config implementation ban đầu disabled; chỉ bật online khi launch mới được cho phép.

Giữ metric namespace3600 hiện hữu và thêm scalars có tên riêng cho text-anchor loss/weighted loss, bank drift, text-context gradient norm và trainable count. Không gửi T0/arrays/raw paths qua scalar logger. Không upload dataset images/full dataset directory. Checkpoint có provenance/path metadata phải được nói rõ, không gọi đã sanitized toàn bộ.

Mỗi run lưu config/Hydra/command, seed/data manifest hashes, source archive/index/hash, dependency/uv.lock hash, initial/final CLIP invariance, trace, raw clean+9mask probe arrays, immutable step checkpoints và latest/best aliases. S1/S2 thêm context state, fixed T0/initial bank identity, anchor formula/reduction/weight. Lưu optimizer/scheduler/RNG, gallery cache identity theo actual photo prompt checkpoint.

Source implementation mới phải có snapshot mới. `4cd4a15` và commit protocol này không là training snapshots. Existing source hash hàm `source_snapshot()` có thể bao gồm các source-like files nonignored trong `outputsnewgate/`; trước launch phải inventory và archive đúng những gì được hash, không nói working tree clean hoặc chỉ HEAD đủ tái lập. Không xóa intermediate để làm đẹp hash; thay đổi exclusion policy nếu cần là việc riêng có review/version, không nằm trong design này.

## 5. Kế hoạch implementation/test cho lần tiếp theo — chưa thực hiện

### 5.1 Reuse và chỗ cần tích hợp

| File hiện hữu | Công việc tối thiểu dự kiến |
|---|---|
| `src/spica/evaluation/text_bank.py` |Reuse SoftPromptTextBank; kiểm tra prefix/layout parity, lưu T0/token identity; không viết text encoder mới|
| `src/spica/train_frozen_prompt.py` |New campaign validation; full_full path cho S roles; learned bank once/update; anchor once; initial/freeze/gradient checks; lưu text metadata|
| `src/spica/frozen_prompt_artifacts.py` |New roles/treatments/probes/selections, không relabel historical campaigns|
| `src/spica/evaluation/masked_view.py` |Extend standalone selection replay cho campaign mới, validate visual + text/provenance state; giữ default1800 historical|
| `src/spica/tracking/wandb.py` |Reuse finite-scalar logger/probes; bổ sung keys nếu cần, không logger mới|
| New experiment configs/tests |3 new configs khi implementation được duyệt; tests theo patterns có sẵn, không dependencies mới|

Trainer hiện có text_mode=soft ở một số historical paths nhưng C/M campaign mới khóa hard; đổi config string đơn thuần **không đủ**. Two-view/probe/evaluator routing hiện phụ thuộc campaign. New clean path cần2 full forwards nhưng không tạo train-mask records; historical C/M mask construction vẫn giữ nguyên. Không nới hard-only validation của historical C/M để giả vờ nó là S1.

Standalone retrieval không cần text forward, nhưng vẫn phải validate checkpoint text-state integrity để full run reproducible. Ưu tiên extend evaluator hiện hữu, không tạo evaluator thứ ba hoặc framework mới.

### 5.2 Acceptance gates trước bất kỳ GPU training

1. **CPU loss/autograd check:** S2 anchor đúng classmean, once/update; T0 normal detached tensor (bao gồm test helper trả inference tensor rồi clone ngoài context); C/P_s/P_p nhận đúng gradients; original CLIP không nhận gradients. S1 và S2(weight0) phải có identical update với same fixture/init/RNG. Đổi labels không được đổi class-bank anchor riêng.
2. **Initialization parity:** real tokenizer + frozen text tower CPU khi có quyền implementation/preflight; S0/S1 banks gần nhau trong tolerance khai báo, S1/S2 C identical, allarms Ps/Pp identical. Không gọi đây là full campaign đã chạy.
3. **Historical compatibility:** toàn suite hiện có; tiny CPU production C/M archived/current parity theo patterns cũ. New S0 bỏ việc tạo train masks không dùng; gate phải chứng minh sample/optimizer trajectory không đổi, không đòi mask trace S0 bằng historical C. Future real S0 tại0/1800/3600 đối chiếu prompt tensors/metrics với historical C khi contract đủ tương đương; checkpoint file SHA có thể khác bởi metadata.
4. **Integration fixture:** new roles S0/S1/S2, đủ sạch+9 masked conditions, best selection excludes0/earliest ties, latest/best replay, source/T0/context hashes, RNG preservation, frozen bytes; fake W&B backend, không online upload.
5. **Boundary mutation checks:** lỗi class ordering, mismatched T0/model/split/checkpoint, missing context state, anchor lambda/reduction drift, wrong selection, teacher/text inference leakage, nonfinite metrics phải bị reject. Đổi query labels không được đổi masks/model inputs/embeddings hoặc fixed training text-bank construction; labels được dùng làm relevance bookkeeping nên có thể đổi metrics và checkpoint selection dựa trên metrics. Selection không được đọc oracle labels qua một đường riêng ngoài evaluator. Spy check new S roles không gọi training masker nhưng vẫn gọi đủ9 evaluation conditions.
6. **Real GPU preflight/training:** vẫn chờ user cho phép riêng; no-update preflight phải finite/frozen/source-bound trước launch. CPU fixture PASS không chứng nhận real GPU training đã diễn ra.

### 5.3 Report sau Bước 1 tương lai

Mỗi arm báo3 rows latest/best_clean/best_masked, step/checkpointSHA/source, clean+masked P200/3AP200/fullAP, runtime/peak GPU và bank drift. Report riêng S1–S0, S2–S0, S2–S1 theo cùng selection. Áp dụng promotion rule §3.3; không tự cộng hoặc bỏ gate khi kết quả xấu.

## 6. Roadmap sau thiết kế này

1. **Tiếp theo:** xin quyền implementation Bước 1, viết code/config/tests, CPU gate, review, commit. Không launch tự động sau commit.
2. Xin quyền preflight/training S0/S1/S2 với protocol này; chạy và review/report/commit kết quả khi hoàn tất.
3. Bước 2: photo CE rồi multimodal coupling dưới controls riêng, chỉ sau khi chọn được baseline; không tự gọi architecture nhiều modules là mạnh hơn.
4. Bước 3: full/full vs full/masked trên baseline mới, matched init/data/photo policy/budget.
5. Bước 4–5: no-update KD diagnostic rồi controlled full-to-masked retrieval KD; fixed clean teacher khác original frozen-CLIP reference, không gộp hai mục đích. Teacher mới chọn theo rule, không mặc định C1800 luôn là teacher cuối cùng.
6. Bước 6: official SeCo comparison/multi-seed/unseen-mask confirmation khi được duyệt và benchmark blockers đã rõ. Chỉ claim thắng masked khi cùng mask condition và có mask-trained baseline đủ mạnh.
7. Bước 7: patch-JEPA khi có bằng chứng warrant, không Gaussian/teacher/predictor/EMA/loss search đồng loạt.

## 7. Verification của lần bàn giao này

Progress commit `4cd4a15`:51 files source/config/tests/reports, không checkpoint tensor/raw historical outputs. Chỉ sửa trailing whitespace một dòng trong CPU pairing script khi staged diff-check phát hiện; không đổi training logic ở bước commit.

Parent chạy trong existing `.venv`, không upgrade dependencies, GPU bị ẩn:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=src .venv/bin/python -m pytest -q
.venv/bin/ruff check src scripts tests
git diff --cached --check
```

Kết quả trước progress commit: **244 tests passed**, Ruff PASS; staged diff-check PASS sau whitespace fix. Sau hoàn tất thiết kế: parent rerun **244 passed (8.28s)**, Ruff/diff PASS; xác minh8 primary response receipts,8 local identities, pinned SeCo contents và local document links. Independent reviewer (`cx/gpt-5.6-luna`, max) **PASS** sau bổ sung T0 inference-tensor handling và clean full_full no-training-mask routing. Đây là verification code hiện hữu và thiết kế, không tests cho S2 chưa implement. Audit nguồn/dataset và design review được lưu riêng trong [audit JSON](semantic_baseline_step0_audit.json). Không push, không model update mới, không W&B session mới.

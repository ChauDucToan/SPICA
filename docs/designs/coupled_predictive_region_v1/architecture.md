# Coupled predictive sketch retrieval — thiết kế V1

**2026-09-08 · DESIGN + CODE MINH HỌA · CHƯA INTEGRATE/TRAIN.**

**Quyết định mới nhất:** [region-first protocol](region_first_protocol.md) chốt region deletion, clean/corrupted50–50, starting loss groups, optimizer/budget; SIGReg diagnostic details và execution gates vẫn pending. Bản V1 trước đó giữ nguyên ở `../coupled_predictive_design_20260908T021035Z/`.

Code đi kèm: [model_sketch.py](model_sketch.py), [masking_sketch.py](masking_sketch.py). Không có Hydra config, trainer, optimizer step, pretrained-model load, GPU experiment hoặc W&B run mới. User đã chọn tài liệu + code khung, không phải implementation production.

## 0. Quyết định và ranh giới với campaign đang chạy

User chốt: **cách 2** — sketch encoder riêng được fine-tune; semantic guidance truyền qua predictor/loss, không đưa hint token vào sketch encoder. Photo/text original CLIP frozen, nhưng soft prompts học được và dùng chung với predictor. Giữ nhánh SIGReg, pooled retrieval phụ, và prediction photo/text; negatives phải giám sát trực tiếp embedding inference. Hard class-name tokens giữ nguyên, context text học được; không dùng oracle query label ở inference.

Đây là workstream khác S0/S1/S2: S0–S2 chỉ học prompts, còn V1 fine-tune một bản sketch visual tower riêng. Không so V1 với S0 rồi quy toàn bộ khác biệt cho predictor khi trainability khác nhau.

Campaign S0–S2 đã kết thúc cả ba arm theo runner tại thời điểm lưu bản tracked này; retrieval/artifact verification còn pending. Execution root: `outputs/semantic_text_execution_20260908T011006Z/`; training source HEAD `6a4d8fc09c1ba15d8c68bbd67ade9f0a2f344ab1`, SHA256 `ca4671cc50010a654f217ddb46714a71063ba4e63a25d6cdd634c9298873d9d1`. Bản nháp gốc ở `outputs/` bị source inventory loại trừ có chủ đích. Bản tracked này được lưu **sau khi cả ba arm kết thúc**; source hash hiện tại sẽ đổi do post-training documentation, không đổi archived training snapshot và không yêu cầu retrain. Không sửa runner, training source hoặc historical artifacts.

## 1. Câu hỏi nghiên cứu

> Shared CLIP–predictor prompts có giúp một sketch encoder học representation phục vụ category retrieval từ raster bị mất nét, trong khi giữ clean retrieval và generalization sang lớp chưa train?

Ba vấn đề phải đánh giá riêng: information-deletion robustness; unseen-category generalization; noisy/missing-label robustness. Khung supervised V1 chưa giải quyết noise/missing labels. Không suy kết quả từ loss giảm, attention map đẹp, Gaussian-like latent hoặc tên JEPA.

## 2. Kiến trúc và inference

```text
                          ORIGINAL CLIP: all original weights frozen
Photo I ─────────────────→ image tower + C_I ───────────────→ Z_I (live)
        └────────────────→ image tower, NO prompt ──────────→ Z_I^0 (fixed)
Class-name tokens c ─────→ text tower + C_T ────────────────→ Z_T(c) (live)
Hard template(c) ────────→ text tower, NO learned context ─→ Z_T^0(c) (fixed)
                                     │
                          C_I and C_T shared by identity
                                     │
Sketch S → RasterMask(S) → E_s → H_s  │
                           │       │ │
                           │       ├─→ mean_pool → g_s ──→ SIGReg [port]
                           │       │                 └──→ W_pool → L2 → q_s
                           │       │                                   └─ rank + CE (aux)
                           │       └─→ small shared predictor ← C_I,C_T
                           │                     ├─ μ_I → photo rank + positive alignment
                           │                     └─ μ_T → text CE + positive alignment
                           └── gradients from all sketch branches

Reference anchors: Z_I ↔ stopgrad(Z_I^0), Z_T ↔ stopgrad(Z_T^0)
Inference: S → E_s → predictor(H_s,C_I,C_T) → μ_I
Gallery: I_j → prompted CLIP image tower(C_I*) → cached Z_Ij
Score: cosine(μ_I, Z_Ij). No labels, class-name bank, target photo or clean-sketch hint in query forward.
```

Checkpoint inference phải gắn chung student encoder, predictor, W_pool, C_I và C_T cùng source/config identity; không ghép student/predictor từ step khác. `C_I*` là prompt tại checkpoint đã chọn. Gallery phải rebuild/rekey theo prompt/checkpoint; **không** dùng unprompted gallery cho model đã rank với prompted photos, và không cache live embeddings xuyên optimizer updates khi C_I còn học.

Pooled `q_s` được giữ theo mô tả ba nhánh ban đầu, để encoder có retrieval supervision không đi qua predictor. Nó là auxiliary/control, **không fusion hoặc đổi main embedding hậu nghiệm**. Main embedding thiết kế là `μ_I`; báo q_s riêng. Text context C_T có thể tồn tại như tham số toàn cục lúc inference, nhưng không chạy text tower/tên lớp cho từng query. Trong khối minh họa không có self-attention giữa query tokens nên photo output không phụ thuộc text query; có thể bỏ tính nhánh text sau này nếu parity được kiểm chứng.

## 3. Backbone, token và parameter ownership

| Thành phần | V1 đề xuất | Trạng thái gradient |
|---|---|---|
| Original CLIP image/text towers | ViT-B/32 family hiện có, output512 | Frozen mọi original weight, LN, embeddings, native projections, logit scale; eval mode |
| E_s | Deep-copy visual tower từ cùng pretrained CLIP; KHÔNG alias original visual | Fine-tune phần tạo context: conv/CLS/position/transformer; không học original teacher |
| H_s | Patch tokens sau transformer, trước final pooling/LN/projection; loại CLS khỏi output | `[B,49,768]` với224px/B32; position đã thêm trước attention |
| Student ln_post/proj | Không dùng trên context path | Frozen/unused; không tính như active trainable parameters |
| C_I | Reuse `FrozenPromptModel.photo_prompt`, dự kiến3×768 | Một `nn.Parameter`, học qua photo branch và predictor |
| C_T | Reuse `SoftPromptTextBank.context`,4×512 | Một `nn.Parameter`, shared qua mọi train classes và predictor |
| Predictor | **Đề xuất** width256,4heads,1cross-attention block + residual/MLP/LN, shared output512 | Trainable; 2 modality-specific input projections, shared attention/MLP/output |
| W_pool | Linear768→512 | Trainable; rồi L2 normalize |
| SIGReg | Exact implementation đã pin/review ở bước implementation sau | Không learnable teacher; gradient vào g_s/E_s |

`make_student_visual()` thể hiện deep-copy; `context_tokens()` thể hiện đường OpenCLIP cụ thể. Đây là adapter minh họa private API, chưa chạy thật trên pretrained tower. Không gọi `TrainableSketchContextEncoder.forward()` hiện có rồi pretend đó là patch tokens: helper đó trả pooled `[B,D]`. `TrainableSketchHiddenEncoder` cũng trả pooled state, chưa phải `[B,N,d]`.

Predictor nhận trực tiếp **cùng objects** `photo_model.photo_prompt` và `text_bank.context`, không tạo bản copy trainable thứ hai. Các projections chỉ đổi chiều token. Khi tích hợp, một model owner đăng ký prompts, predictor chỉ sở hữu projections/attention/MLP; deduplicate optimizer parameters bằng identity và xác minh không có duplicate group. Giữ reference tower eval ngay cả khi outer student `.train()`.

Prompt initialization dự kiến reuse photo N(0,.02), text exact prefix `a photo of a` từ frozen tokenizer/embedding; kiểm chứng parity trước update. Đây là đề xuất implementation, chưa có real gate cho V1.

## 4. Raster corruption

Raster masking xảy ra **trước mọi sketch-encoder attention**. Không mask latent sau attention rồi gọi là đã giấu thông tin. H_s có thể chứa attention giữa các nét còn lại; không được chứa forward từ ảnh sketch sạch của cùng query làm input hint.

Hai corruption cần phân biệt:

1. `ink_centered_square_v1` hiện có: region deletion; giữ làm evaluator chung với controls hiện tại.
2. Thinning/segment-like raster dropout mới: erosion trên **binary ink mask** (ink=1), không erosion trực tiếp ảnh grayscale trắng nền làm nét đen dày lên. Không gọi là true-stroke dropout nếu không có stroke trajectories.

`thinning_preview()` đã có code CPU minh họa binary ink erosion và metadata, nhưng **không nối vào region trial**, không topology-preserving và severity chưa calibrated. Cần actual preview/contract trước experiment thinning. Pixel ngẫu nhiên dễ tạo nhiễu không giống mất đoạn nét; không mặc định equivalent với region deletion.

Train đã chốt clean/corrupted losses50–50; region requested fractions25/50/75%,base seed4242,masked view1 reuse historical policy như [protocol](region_first_protocol.md). Inference không thêm corruption nhân tạo, nhưng query thực tế có thể đã thiếu nét. So same raster hashes giữa candidate/control, gallery clean. Không gọi25/50/75% là thinning severity đã calibrated.

## 5. Objective chính xác trong code minh họa

Tất cả q_s, μ_I, μ_T, Z_I, Z_T dùng cosine trên unit vectors; g_s không normalize hay batch-standardize. `unit()` fail nếu nonfinite/empty/zero norm.

### 5.1 Photo negatives: trực tiếp lên μ_I

Với một positive cùng lớp và K negatives khác lớp:

`rank(q) = mean_{b,k} softplus(m + cos(q_b,Z^-_bk) - cos(q_b,Z^+_b))`.

- `rank_i = rank(μ_I)` là task loss của output inference.
- `rank_pool = rank(q_s)` là auxiliary loss.
- Không detach photo embeddings trong rank: C_I phải nhận gradient qua prompted photo encoder. Không copy nguyên `jepa_ranking_loss()` cũ vì helper đó detach cả positive/negative.
- `m=.2` trong minh họa reuse công thức hiện tại, chưa phải margin tối ưu. Không thêm negative-only sigmoid bias tự do hoặc claim softplus luôn có gradient nhẹ hơn hinge.
- Sampling/mining chưa implement. K trong tensor chỉ mô tả tập **valid** negatives; cùng class dù khác instance không là negative. Unknown label không được biến thành different-class bằng sentinel.

### 5.2 Text negatives: hard-label CE giữ nguyên

`ce_t = CE(μ_T @ T(C_T).T / τ, y)`;
`ce_pool = CE(q_s @ T(C_T).T / τ, y)`.

Dùng `jepa_text_classification_loss(..., detach_text=False)`: text soft context vẫn học, original text tower frozen. Bank complete sorted **84 pseudo-train classes**; không anchor/training với pseudo-validation hoặc official class names. Prototype khác lớp trong denominator đã là text negatives, không thêm text pairwise rank trùng lặp ở V1. τ=.07 chỉ là minh họa inherited, chưa tuning.

### 5.3 Positive prediction: fixed-kappa vMF = weighted cosine

`align_i = mean_b [1-cos(μ_I,b, stopgrad(Z_I,b^+))]`;
`align_t = mean_b [1-cos(μ_T,b, stopgrad(T(C_T)[y_b]))]`.

Reuse `text_anchor_loss()` vì primitive này là cosine theo hàng với target detached và validation. Tên helper không có nghĩa photo row là class prototype. Nếu κ cố định, vMF NLL khác `κ*(1-cos)` bởi hằng số; không implement Bessel hoặc pretend đây là uncertainty modeling. **Không học κ trong V1**, hệ số cosine nằm trong lambda explicit.

Detach là **quyết định gradient-route đề xuất trong bản thiết kế**: regression không kéo target trực tiếp để giảm loss. C_I/C_T vẫn thay đổi từ rank/CE, predictor và anchors; target update sau vẫn dịch chuyển. Không gọi chúng là fixed teacher chỉ vì có stop-gradient.

### 5.4 Preserve original CLIP prior

`anchor_i = mean_{unique sampled photo IDs} [1-cos(Z_I(I), stopgrad(Z_I^0(I)))]`;
`anchor_t = mean_{all 84 train classes} [1-cos(T(C_T)[c], stopgrad(T0[c]))]`.

Image reference dùng chính ảnh clean đã encode ở prompted photo side (positive và sampled negatives, deduplicate identities), không dùng sketch label để chọn một ảnh khác. Text anchor lấy cả bank once/update, không theo tần suất nhãn trong batch hoặc nhân theo số views. Frozen hard-bank helper có inference-mode tensor: clone ra normal detached tensor **ngoài** inference_mode trước backward (`clone_fixed_text_bank` hiện có).

Anchor không tự sửa sketch label/correspondence sai; bản thân bank/image reference độc lập khỏi nhãn sketch nhưng supervised association vẫn có thể sai. Anchor mạnh quá hạn chế thích nghi hữu ích; không bảo đảm semantic prior CLIP tối ưu task.

### 5.5 SIGReg và tổng loss

`sigreg = verified_SIGReg(g_s)` trên sample axis B; không coi49 patch tokens hay hai view tương quan như49×/2× số mẫu độc lập. Thực hiện theo contract view-axis của implementation được pin. Không dùng `SignatureRegularizer` cũ: nó batch-standardize và chỉ là SIGReg-style, không exact LeJEPA.

`L = Σ_j λ_j L_j`, với j gồm:
`rank_i, ce_t, rank_pool, ce_pool, align_i, align_t, anchor_i, anchor_t, sigreg`.

`weighted_total()` là primitive không tự chọn hệ số. `region_objective()` đã encode nhóm task1,pool0.5,alignment0.1,reference1 được user duyệt, clean/corrupted50–50 và reference anchors once/update; hệ số thành phần chính xác xem [protocol §3](region_first_protocol.md). Unit weights ở demo V1 chỉ là tensor wiring; demo region mới kiểm tra tổng3.6 và từng gradient coefficient. Optimizer starting groups/effective batch32/budget3600 đã được đồng ý; schedule details/SIGReg rho/pin/lambda thực đo và production gates vẫn pending, **không launch**. Không nhầm nhóm reference1 ở đây với mỗi anchor=1 của một campaign khác.

SIGReg là port cố ý chưa implement: `sigreg_term(g)` raise `NotImplementedError`, không âm thầm trả0 hoặc thay bằng covariance/MSE. Bản minh họa không có complete full-objective forward.

## 6. Luồng gradient và chống shortcut

| Loss | E_s/predictor | C_I | C_T | Original CLIP/reference |
|---|---|---|---|---|
| rank_i | Cả hai | Predictor-query path + live photo rank path | Không cần cho μ_I trong block minh họa | Không update original weights |
| ce_t | Cả hai | Không | Predictor-query path + live text-bank path | Không update original weights |
| rank_pool | E_s + W_pool; không predictor | Live photo path | Không | Không |
| ce_pool | E_s + W_pool; không predictor | Không | Live text path | Không |
| align_i / align_t | E_s + predictor | Query path cho photo | Query path cho text | Regression targets detached |
| anchor_i / anchor_t | Không | Photo anchor | Text anchor | References detached |
| SIGReg | E_s qua g_s; không predictor/W_pool | Không | Không | Không |

Các tham số predictor chung có thể tích lũy gradient cả hai modality; bảng nói trực tiếp theo graph từng forward, không phủ nhận ảnh hưởng của update về sau.

Predictor input chỉ H_s và global prompt parameters. Không nhận class IDs, class-name tokens, positive photo features, target text features, clean-sketch latent hoặc per-query teacher hint. Shape check global prompt2D ngăn một số nhầm lẫn, nhưng chính orchestration/parameter ownership mới chứng minh cùng một prompt cho mọi query; shape2D một mình không là bằng chứng chống leakage.

Kiểm tra cần có trước claim hiệu quả:
- Empty/constant context và same-/different-class context swap: predictor có phụ thuộc sketch không? Đo retrieval, không chỉ norm output đổi.
- Pooled-only matched-capacity/control: predictor thêm lợi ích thật không?
- Gradient norms của E_s/predictor/prompts và reference drift; xem geometry là diagnostic, không causal proof.
- Query embeddings bất biến khi chỉ thay bookkeeping label/positive photo input ở loss side.
- Trainable student không alias original CLIP, original bytes không đổi; inference gallery hash khớp C_I checkpoint.

CPU demo chỉ kiểm tra graph algebra trên tensors random; context swap ở initialization không chứng minh model đã train không shortcut.

## 7. Missing/noisy labels: được ghi nhận, chưa hứa giải quyết

- Query inference không cần nhãn: đã đáp ứng bởi architecture/API.
- Unseen-class generalization: phải đo trên classes không train; pretrained CLIP có thể đã gặp concept, không gọi là concept-unseen pretraining.
- Training có trusted photo correspondence nhưng thiếu category label: cosine photo alignment có thể có cơ sở; **không tự coi mọi ảnh khác là negative**. Text CE cần nhãn đáng tin hoặc một weak-supervision design riêng.
- Label/correspondence đều thiếu hoặc sai: không thể tự suy positive đúng. Teacher soft scores cũng có thể sai; thresholds/pseudo-labeling/noise curriculum không được thêm ngầm.
- `supervised_terms()` chỉ dành trusted labeled rows và explicit valid negative labels; reject sentinel/missing IDs. Nó kiểm tra consistency của metadata, không phát hiện mọi semantic mislabel.

Noisy/missing-label experiments cần protocol riêng: corruption chỉ trên train annotations, clean validation ground truth giữ nguyên, same corruption masks/seeds, pairing provenance rõ. Không trộn thành một metric rồi claim cả3 khả năng.

## 8. Liên hệ TC-JEPA, LeJEPA và SeCo — đúng giới hạn

Primary đã đọc: [TC-JEPA v1](https://arxiv.org/html/2605.03245v1), §3, §5.3, Appendix C.1/D; [bản extract](../../../outputs/coupled_predictive_design_20260908T021035Z/tc_jepa_extract.txt) và HTML giữ ở root V1 trước.

- TC-JEPA dùng predictor **narrow ViT width384**, depth6 cho ViT-B/16 hoặc12 cho L/H; không phải chứng cứ một layer luôn đủ. Text conditioner dùng cross-attention nhẹ ở nhiều layers, và ablation single-layer conditioning kém hơn multi-layer trong setup của họ.
- TC-JEPA dùng pretrained T5 caption tokens, EMA visual targets và patch-feature prediction; conditioner/predictor bị discard tại inference. V1 dùng global shared CLIP prompts, photo/text global targets và **giữ predictor làm query path**. Cross-attention direction cũng khác: V1 prompt queries đọc visual context; TC-JEPA predicted patch queries đọc word tokens. Không gọi exact TC-JEPA implementation.
- TC-JEPA hỗ trợ động cơ conditioning predictor thay bulky encoder; không chứng minh V1 sẽ tốt hơn hoặc predictor càng nhỏ càng tốt. Width256/one block là lựa chọn tối thiểu để thử, không giá trị lấy từ paper hoặc optimality claim.
- [LeJEPA](https://arxiv.org/html/2511.08544v3) là nguồn SIGReg, không bảo đảm geometry/zero-shot cho hybrid supervised CLIP này. Exact official implementation pin/review còn pending.
- [SeCo-SBIR](https://arxiv.org/html/2608.03120v1) gợi ý text-guided adaptation và frozen visual reference. Explicit classmean text cosine anchor ở đây không được gắn nhãn là exact SeCo loss; xem audit repo/metric blockers hiện có trong `docs/semantic_baseline_step0_step1_protocol.md`.

Không claim vượt SketchLVM/SeCo từ diagram. So paper cần cùng split/gallery/relevance/AP denominator/trainability/compute/selection, và strong mask-trained control.

## 9. Đối chiếu với code hiện có và code khung

| Architecture | `model_sketch.py` | Reuse / integration còn thiếu |
|---|---|---|
| Trainable sketch riêng | `make_student_visual`, `context_tokens` | Deep-copy policy theo `models/clip.py`; patch-token adapter mới cần gate, không sửa pooled API historical |
| Shared C_I | Đối số `photo_prompt` trong predictor | `FrozenPromptModel.photo_prompt`, `encode_photo` |
| Shared C_T + hard class words | Đối số `text_context`; bank vào loss, không vào predictor | `SoftPromptTextBank.context/forward`; tokenizer/EOT/parity existing |
| Predictor nhỏ | `SmallPromptPredictor` | PyTorch MHA/Linear/LN, không dependency mới |
| Pool + sphere | `sketch_branches` → g/q/μ_I/μ_T | W_pool riêng; SIGReg lấy g trước L2 |
| Pairwise photo category ranking | `softplus_rank`, `supervised_terms` | Công thức live-photo từ trainer hiện tại, không detached old JEPA rank helper |
| Text negatives/hard labels | `supervised_terms` | `jepa_text_classification_loss(detach_text=False)` |
| Cosine/fixed-vMF alignment | `supervised_terms` | `text_anchor_loss` primitive rowwise cosine/stopgrad |
| Photo/text frozen reference | `preservation_terms` | Original CLIP `encode_image`; hard bank `encode_class_text_bank` + normal clone |
| SIGReg | `sigreg_term` | Port BLOCKED, không dùng standardized legacy helper |
| Approved region weights/views | `region_objective` + `weighted_total` | Group coefficients đã chốt, không Hydra/trainer; SIGReg coefficient phải explicit |
| Sanity check | `demo` | CPU tensors only, no dataset/CLIP/SIGReg/optimizer |

Luồng integration dự kiến (pseudocode, **không phải trainer runnable**):

```python
# original_clip: frozen + eval; student_visual: independent trainable copy
# photo_model.photo_prompt and text_bank.context are sole owners of C_I/C_T.
masked = region_mask(sketch)                   # approved historical policy, before attention
h = context_tokens(student_visual, masked)     # no hint/label/photo input
b = sketch_branches(h, photo_model.photo_prompt, text_bank.context,
                    predictor, pooled_head)
photo_live = photo_model.encode_photo(unique_training_photos)  # autograd to C_I
text_live = text_bank()                         # once/update, autograd to C_T
# Original references computed without prompts, detached normal tensors.
terms = supervised_terms(b, positive_live, negative_live, text_live,
                         class_ids, labels, positive_labels, negative_labels)
terms.update(preservation_terms(photo_live, photo_reference, text_live, text_reference))
terms['sigreg'] = sigreg_term(b['g'], verified_sigreg)
# Single-view component sketch only; the actual region design combines BOTH
# views with region_objective(), reference anchors once: see region_first_protocol.md.
loss = weighted_total(terms, explicit_weights)  # not a runnable update/trainer
```

Positive/negative gathers must use the exact encoded identities/order; references correspond to those same images. Frozen original image/text outputs may be cached with transform/model/class identity; prompted photo/text outputs are live graphs during training.

## 10. Gates và quyết định còn lại trước implementation/train

1. Region/clean mixing, starting groups/weights/budget đã được duyệt riêng cho design mới. Khóa schedule details, SIGReg diagnostic specifics và final trial manifest trước execution; thinning nằm ngoài first trial.
2. Pin/review faithful SIGReg, validate sample/view axes, không default legacy implementation.
3. Real tokenizer/CLIP patch-token path and independent student/frozen teacher gates; measure memory, không suy preflight6.71GB của S1 đủ cho fine-tuned E_s.
4. Label/negative trust contract, no cross-split leakage, once/update text/anchor reuse, optimizer ownership and graph checks.
5. CPU integration + no-update GPU only when separately authorized. No proposal here licenses GPU training.
6. Proposed ablations: fine-tuned pooled baseline → predictor package, preservation-off và SIGReg-on controls riêng → thinning → noisy/missing labels. Starting trials có approved preservation ON; xem exact comparison scope tại region_first_protocol.md. Đây là kế hoạch, không scheduled runs.
7. Use existing clean/macro9 P200/three AP200/full mAP reports and same-selection comparisons. Exact new selection/promotion gate must be agreed before launch; no peak mixing or choosing q/μ/fusion post hoc.

## 11. Trạng thái kiểm tra

V1 parent CPU tensor self-check/review đã PASS; predictor đề xuất có **1,183,744 parameters** (không gồm E_s, shared prompts hoặc W_pool). Bản region-first mới có checks riêng cho loss coefficients/views và parked thinning; xem log/verification receipt mới, không dùng chứng nhận V1 để thay cho code mới. Đây không phải số đo retrieval hoặc benchmark predictor size.

Chạy `model_sketch.py` bằng CPU để kiểm tra shapes/unit norms, gradients vào context/shared prompts/photo positives/negatives/text bank, detached regression/reference targets, rejected same-class negatives, SIGReg fail-closed. File log/verification receipt ghi kết quả thật riêng. Không gọi đây là model readiness hoặc retrieval effectiveness.

Đã chuyển bản thiết kế sang tracked docs sau khi runner báo S0–S2 COMPLETED. Không push; không launch V1. Raw draft/check logs vẫn giữ tại `outputs/coupled_predictive_region_design_20260908T025016Z/`.

# SPICA — báo cáo kiểm chứng an toàn tại repo (2026-09-06)

## Phạm vi và kết luận ngắn

- **HEAD khi bắt đầu:** `efa4bd1d489d059d2a0dd4086a181ec96a784f2f`; working tree sạch; không sửa source/model/training loss và không chạy GPU campaign mới.
- Đã kiểm tra code, lineage/evidence local, dữ liệu/manifest, replay CPU nhỏ và nguồn primary cần thiết. Báo cáo này là artifact mới; việc thêm tài liệu làm source hash hiện tại khác snapshot training, **không phải lý do để gán lại hoặc retrain run cũ**.
- Kết luận hiện tại: inference của các frozen-prompt run là **sketch-input-only**, nhưng output gallery có thể đổi do photo prompt; corrected alignment pilot **R > MD > MS** là kết quả âm đã được kiểm chứng đúng lineage; các đề xuất JEPA/mask/relational KD và true-pairing vẫn là giả thuyết hoặc cần quyết định thiết kế và thực nghiệm mới.
- Không có bằng chứng repo-local có lineage cho tuyên bố “wrong text làm retrieval giảm 99%”. Không dùng nó làm fact.

## 1. Frozen prompt: inference, số liệu và causal scope

### Inference contract đã kiểm chứng

Evidence trong `outputs/frozen_prompt_v2_selection_verified_2026-09-04.json`, các `run_result.json`/`probe_step5400.json` của campaign frozen-prompt, và code:

- `src/spica/models/frozen_prompt.py:139-183`: `forward()` nhận sketch image và dùng `sketch_prompt`; `encode_photo()` dùng `photo_prompt`; cả hai đều đi qua cùng frozen visual tower. Prompt được chèn **trước transformer** tại `src/spica/models/frozen_prompt.py:147-167`, không phải mask nội dung.
- `src/spica/evaluate_frozen_prompt.py:82-90,107-124`: query loader chỉ cung cấp sketch; gallery được encode riêng; kết quả ghi `text_used_for_inference: false`.
- `src/spica/train_frozen_prompt.py` và artifact ghi `photo_required: false`, `text_required: false`, `oracle_class_required: false` cho query.
- FP1S có `train_sketch_prompt=true`, `train_photo_prompt=false`; gallery là frozen photo CLIP adapter. FP2 có hard text CE loss-only, cả sketch/photo prompt trainable. FP3 có soft text prompt trainable, nhưng **CLIP text tower parameters vẫn frozen**; soft context là tham số riêng của text bank. Đây là phân biệt giữa frozen weights và learned outputs.

Do đó, “sketch-only” đúng ở **input query/inference**, không có nghĩa toàn bộ hệ thống có cùng output như vanilla CLIP: FP2/FP3 có photo prompt trong gallery, còn FP1S không cập nhật photo prompt. Các chênh lệch giữa FP1S/FP2/FP3 vì vậy không tự là causal effect của một yếu tố duy nhất nếu treatment, text supervision và checkpoint horizon không được ghép đúng.

### Truy số và protocol

Bảng selection local truy được bằng:

| Role | Checkpoint/step | pseudo-unseen full mAP | Evidence |
|---|---:|---:|---|
| FP1S | `frozen_prompt_step5400.pt`, 5400 | `0.6333361297917539` | `outputs/frozen_prompt_v2_selection_verified_2026-09-04.json`, checkpoint SHA256 `d5632548...bc6a8a` |
| FP2 | `frozen_prompt_step5400.pt`, 5400 | `0.6465140474522029` | cùng file, checkpoint SHA256 `ef89e096...be48e8` |
| FP3 | `frozen_prompt_step5400.pt`, 5400 | `0.6727519489712296` | cùng file, checkpoint SHA256 `a3c44c3d...e8602` |

Selection metric là `full_pseudo_unseen_mAP`; official unseen không dùng để selection. Đây là các cấu hình khác nhau, không được diễn giải thành bằng chứng causal của photo prompt/text hoặc “đúng/sai text”. `0.672752` cũng không phải kết quả corrected alignment R/MD/MS và không được so trực tiếp với official SketchLVM `.723`.

Repo search không tìm thấy artifact có trường/manifest/reproducer cho “wrong text”, “99%”, hoặc bảng no/correct/wrong-text cùng query IDs và checkpoint matching. Vì vậy claim đó là **NOT VERIFIED**.

## 2. Generic Photo View, sampling và giả định toán học

### Điều code thực sự làm

- `src/spica/models/jepa.py:121-137`, `photo_semantic_target()` nhận `[B,M,D]`, normalize từng frozen photo embedding, lấy mean rồi normalize lại và `detach()`:
  \[
  t_s=\operatorname{norm}\left(\frac1M\sum_m\operatorname{norm}(V_m)\right).
  \]
  Đây là normalized class-conditional multi-positive target, không phải một photo instance được ghép với sketch.
- `src/spica/data/datasets.py:99-124,189-220`: positive photo được lấy ngẫu nhiên từ `_photos_by_label[label]`; với `M` ảnh thì `random.sample` nếu đủ, nếu thiếu thì lấy lặp. Negative chỉ conditional trên label khác. Không có trường pair ID.
- `configs/data/sketchy_104_21.yaml` dùng Sketchy `zeroshot0`; local file lists có path sketch/photo khác cấu trúc. Kiểm tra CPU ngày 2026-09-06: train manifest đọc được `57,587` sketches, `72,949` photos, `104` lớp mỗi modality; không có sketch stem nào trùng photo stem trong phép heuristic (0/57,587). Đây không chứng minh không có pairing trong nguồn gốc, chỉ chứng minh **repo sampler không mang pairing explicit** và filename không đủ làm bằng chứng.

### Phạm vi của claim \(f^*=E[V\mid S]\)

Với MSE và target vector chưa normalize, Bayes optimum là \(f^*(s)=E[V\mid S=s]\). Nếu objective chỉ so cosine trên hướng, optimum hướng là `norm(E[V|S=s])` khi kỳ vọng khác zero. Trong code, biến target thực tế là \(T=\operatorname{norm}(M^{-1}\sum_m\operatorname{norm}(V_m))\). Do normalization phi tuyến, nghiệm cosine chính xác là hướng của \(E[T\mid S]\), không mặc định bằng hướng của \(E[V\mid S]\). Sampling **cùng class** vẫn cho \(E[T\mid S]=\sum_y P(y\mid S)E[T\mid Y=y]\) dưới giả định độc lập conditional; vì vậy kết luận giới hạn supervision ở class vẫn giữ nguyên. Nó không chứng minh instance-level target, vì sketch không được ghép với ảnh cùng instance.

Cần phân biệt:

- **Category-level:** label đủ để lấy nhiều photo positives; đây là contract hiện tại.
- **Instance-level/FG-SBIR:** cần mapping sketch–photo thật và negative cùng category khác instance; repo hiện tại chưa có mapping được xác minh.

## 3. Corrected alignment seed42/split3407

Đây là phần đã VERIFIED trước đó, không retrain lại và không gắn source hash tài liệu hiện tại vào run cũ.

| Arm | Config treatment | mAP@1800 | Delta vs R | Status |
|---|---|---:|---:|---|
| R | rank + CE, covariance 0 | `0.6715752530134864` | — | `VALID_CONTROL` |
| MD | mean text/log, detached | `0.6671100875995982` | `-0.004465165413888` | `MATCHED` |
| MS | mean text/log, symmetric | `0.6541532269372080` | `-0.017422026076278` | `MATCHED` |

Lineage: `outputs/alignment_verification_20260906_093823_82c6461/corrected_pilot_summary.json`, `matching_validation.json`, `artifact_inventory.md`, `resolved_configs/{R,MD,MS}.json`, raw histories/checkpoints trong `fixed_attempt/pilot/`, calibration SHA256 `f038ba8797...df67fd`, source snapshot training `a30cb43ecc...46ff3b9`, inherited reviewed snapshot `82c6461`. Cả ba dùng seed `42`, pseudo split seed `3407`, horizon `1800`, `ViT-B-32-quickgelu/openai`, batch/sampler matched, và covariance weight bằng 0. MD/MS dùng `lambda_alignment_mean=0.2603831284137216`, còn R dùng `lambda_alignment_mean=0.0`; matching mode là `corrected_v2`.

Evidence replay cho biết: 82 targeted tests + 13 pairing mutation checks, Ruff, CUDA preflight và raw AP replay đã pass trong bundle. Replay CPU mới trong phiên này cũng pass (xem §7). Mean gap giảm (R `.796686`, MD `.733997`, MS `.625390`) nhưng mAP giảm; điều này bác bỏ suy luận “moment agreement đủ để cải thiện retrieval”, không bác bỏ mọi dạng alignment khác. Bootstrap là uncertainty của 10,963 queries trên một seed/split, không phải multi-seed hay official-unseen evidence.

## 4. Frozen CLIP, gradient path và mask

### Weights, outputs và gradient

- `src/spica/models/clip.py:15-47`: `FrozenClipEncoder` gọi `requires_grad_(False)`, eval mode, image/text output normalized.
- `src/spica/models/frozen_prompt.py:57-64`: visual tower bị freeze; chỉ prompt và tùy chọn LayerNorm được phép train. `src/spica/train_frozen_prompt.py` có optimizer groups, byte identity snapshot và `_assert_clip_policy()` để fail closed nếu CLIP-owned parameter ngoài allow-list đổi.
- `src/spica/train_jepa.py:778-807`: photo embeddings đi qua `_encode_photo_targets()` trong `torch.no_grad()`; target detach; text CE mặc định detach hard bank. Với soft text bank, chỉ `soft_prompt.context` trainable, text encoder weights không train.
- `src/spica/train_jepa.py:847-865` và các kiểm tra tương ứng: parameter trainable phải có finite gradient, frozen parameter không được nhận gradient. `SketchPhotoJepa.forward()` chỉ nhận raw sketch (`src/spica/models/jepa.py:73-98`); text/photo không vào predictor.
- FP3/frozen-prompt không phải “CLIP dùng sketch với separate trainable sketch encoder” theo nghĩa full backbone: đó là frozen CLIP visual tower + sketch prompt. JEPA family có `TrainableSketchContextEncoder`; mode có thể `frozen`, `partial`, `full` (`src/spica/models/clip.py:89-219`). Đây là một implementation choice cần giữ rõ khi thiết kế experiment.

Photo-side CE là hằng số nếu toàn bộ photo-side input và photo encoder output đều frozen. Nhưng nếu photo prompt là trainable và photo-side CE được tính qua output đó thì CE không hằng số đối với prompt; ở corrected alignment, calibration đo riêng detached/symmetric photo gradient và ghi photo ratio, không được giả định bằng zero.

### Masking

Không tìm thấy content-aware sketch masking trong code hiện tại. Prompt insertion trước self-attention không phải masking. Xóa pixel trước encoder và loại/thay token trước self-attention là hai cơ chế masking hợp lệ nhưng không tương đương. Ngược lại, nếu chỉ mask embedding sau khi encoder đã nhìn sketch đầy đủ, token giữ lại có thể chứa thông tin vùng bị che do self-attention trước đó. Chưa có thực nghiệm clean/partial được kiểm chứng ở đây; trạng thái eval của frozen teacher không tự chứng minh preprocessing/augmentation tạo target ổn định.

## 5. Relational KD: kiểm chứng toán học, chưa phải implementation

Đề xuất có thể viết, với fixed photo anchors \(a_j\),
\[
 r^V=\operatorname{softmax}(v^Ta/\tau),\qquad
 r^S=\operatorname{softmax}(q^Ta/\tau),
\]
\[
 L_{KD}=KL(r^V\|r^S),\qquad
 L_{rank}=\operatorname{softplus}(m+q^Tv^- - q^Tv^+),
\]
 cộng text CE hiện tại. Với anchor/teacher fixed, gradient theo student logits là `rS-rV`, và gradient Euclidean theo query đơn vị là \(\nabla_q L_{KD}=\tau^{-1}\sum_j(r^S_j-r^V_j)a_j\). Nếu \(q=u/\|u\|\), gradient theo output chưa normalize là \(\nabla_u L_{KD}=(I-qq^T)\nabla_q L_{KD}/\|u\|\) với \(u\ne0\); không có gradient về fixed anchors. Ranking softplus có gradient logistic theo score gap. Vì vậy toán học không có lỗi hiển nhiên, nhưng còn các điều kiện cần đo:

1. **Target/pairing:** một sketch có thể có nhiều positives; teacher distribution phải dùng true pair, category multi-positive hoặc một quy ước rõ ràng. Current dataset chỉ chứng minh same-class.
2. **Relevance:** photo CLIP anchor có thể giữ texture/background/chụp cảnh thay vì shape sketch; KD có thể truyền nuisance, không tự động là semantic teacher tốt.
3. **Normalization/Jacobian:** cần log grad norm riêng của KD/rank/CE và tổng, tránh lambda được chọn theo raw loss khác scale.
4. **Anchor rank/capacity:** softmax không phân biệt các logits lệch nhau cùng một hằng số. Do đó KD chỉ nhận dạng query qua span của các hiệu \(a_j-a_1\); nếu span thiếu chiều, nhiều query khác nhau có cùng phân phối anchor. Nhiệt độ/số anchors cần kiểm chứng, không có bảo đảm KD tốt hơn regression.
5. **Metric:** category mAP cho phép nhiều positives, còn FG retrieval cần instance relevance; không được gọi KD ưu việt hơn MSE chỉ từ form loss.

Text tự suy từ cùng sketch không thêm conditional information nếu không có context ngoài sketch. Context user có thể thêm thông tin, nhưng không bảo đảm tự phát hiện text sai; no/correct/wrong text phải là test factor riêng. Chưa có code relational KD, nên status là **hypothesis, NOT RUN**.

## 6. SignatureRegularizer và LeJEPA claim

`src/spica/models/jepa.py:240-311` hiện có tên “SIGReg-style”, nhưng **không phải implementation faithful của SIGReg/LeJEPA primary paper**:

- Current code center/standardize từng chiều trong batch (`:297-299`), nên mất mean và scale penalty. CPU probe ngày 2026-09-06 cho `reg(x) == reg(x*a+b)` với affine positive per-dimension trong sai số máy (`0.0`), xác nhận invariance này.
- Current code dùng fixed CPU random projections, 16 tần số dương `frequency_max/num_frequencies ... frequency_max`, không tích phân đối xứng/weighted Epps–Pulley như pseudocode primary.
- LeJEPA arXiv:2511.08544 §4.2–§5 mô tả Epps–Pulley characteristic-function matching, target Gaussian, raw embeddings, random directions và SIGReg + prediction; abstract cũng nhấn mạnh không dùng stop-gradient/teacher-student trong recipe LeJEPA. Code SPICA vẫn có stop-gradient cho frozen photo target vì đó là design khác.

Do đó standardization hiện tại có thể là regularizer shape-only hữu ích, nhưng không được gọi là “đúng LeJEPA SIGReg” hoặc suy ra target isotropic Gaussian đã được enforced. Target cố định giúp chặn một dạng joint collapse, nhưng không bảo đảm student retrieval tốt; cần geometry + retrieval thực nghiệm.

## 7. CPU checks đã chạy trong phiên này

Môi trường dùng `.venv`, không đổi dependency.

| Command | Exit/result |
|---|---|
| `source .venv/bin/activate && CUDA_VISIBLE_DEVICES='' pytest -q` | **0; 183 passed in 9.46s** |
| CPU autograd probe cho `photo_semantic_target`, prediction/ranking loss, hard-text detach, `SignatureRegularizer` parameter/gradient | **0; PASS**; target detached, photo target không nhận grad, hard text không nhận grad, regularizer không có parameter |
| `CUDA_VISIBLE_DEVICES='' python scripts/summarize_alignment.py outputs/alignment_verification_20260906_093823_82c6461/fixed_attempt/pilot --horizon 1800 --output /tmp/spica_reviewer_replay_20260906.md` | **0**; replay report tạo được; các config/source/checkpoint/raw metric đọc được |
| Manifest/data audit trên `configs/data/sketchy_104_21.yaml` | **0**; 57,587 sketch train, 72,949 photo train, 104 lớp mỗi modality; sampler chỉ conditional label; không có explicit pairing |
| Affine probe cho SignatureRegularizer | **0**; `sigreg_affine_abs_diff=0.0`, `parameter_count=0` |

Parent kiểm tra độc lập sau subagent: `CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q` → **183 passed in 2.97s**, exit 0; `git diff --check` pass. Parent đối chiếu lại Table 1 và phần LayerNorm của SketchLVM primary, cùng config R để sửa ghi chú lambda control. Các probe ad-hoc trong bảng không có script/raw log được lưu trong báo cáo, nên chưa phải bằng chứng tái lập độc lập đầy đủ.

Không chạy training, official unseen, multi-seed, covariance, 5400-step extension, lambda search, hoặc tải model/GPU mới.

## 8. SketchLVM benchmark và metric protocol

Primary paper [Sain et al., arXiv:2303.13440v3](https://arxiv.org/abs/2303.13440) §6, Table 1 báo cáo cho **Sketchy extended, TU-Berlin và QuickDraw**:

| Dataset | Paper metric | SketchLVM/Ours table value |
|---|---|---:|
| Sketchy extended | mAP@200 | `.723` |
| TU-Berlin | mAP@all | `.651` |
| QuickDraw | mAP@all | `.202` |

Cùng bảng ghi P@200 `.725`, P@100 `.732`, P@200 `.388` tương ứng. Paper mô tả Sketchy extended có 104 train/21 test và thêm ảnh ImageNet; TU-Berlin 220/30; QuickDraw 80/30. Paper dùng category label relevance; fine-grained section dùng protocol khác (Top-1/Top-5 instance accuracy), không trộn hai metric.

Repo:

- `src/spica/evaluation/metrics.py:73-118` có ba AP@K denominator: `prefix_positive` (default historical), `all_relevant`, `min_relevant_k`.
- `src/spica/evaluation/frozen_prompt.py:105-122` gọi `mAP@200` với `prefix_positive`; full mAP vẫn là macro AP toàn gallery. Vì vậy cần ghi rõ metric name/denominator, không gọi mọi số `mAP@200` là paper mAP@all.
- `configs/data/sketchy_104_21.yaml`, `tuberlin_220_30.yaml`, `quickdraw_80_30.yaml` giữ split manifests; `src/spica/evaluate_frozen_prompt.py` và embedding caches giữ thứ tự query/gallery. Local corrected pilot là pseudo split 20 validation classes, không phải official 21-class benchmark.
- Chưa chạy lại official unseen và chưa có checkpoint selection mới theo bảng paper trong phiên này. Vì vậy `.672752` là pseudo-unseen full mAP của frozen-prompt FP3, không được so numerical claim với `.723` SketchLVM official.

## 9. Năm bước readiness

Mọi run mới nên dùng cùng pseudo split `3407`, seed được phê duyệt, sampler/batch cố định, cùng preprocessing/gallery cache, cùng loss budget và cùng checkpoint rule: fixed step trước (đề xuất 1800 cho pilot), không chọn peak candidate so với fixed-step control; sau đó mới mở seed/horizon nếu được duyệt. Official unseen chỉ là final diagnostic, không selection.

| Bước | Trạng thái | Evidence / điều kiện để gọi VERIFIED |
|---|---|---|
| 1. Baseline sketch-only với fixed CLIP gallery | **VERIFIED một phần / protocol-ready** | FP0 và frozen-prompt artifacts chứng minh raw sketch query, text không inference, fixed CLIP path có thể replay. Cần chốt một baseline chính thức duy nhất (vanilla FP0 hay fixed-photo FP1S) và lưu exact gallery/cache + checkpoint rule trước khi dùng làm control mới. |
| 2. Same-class target vs true-paired target | **PARTIAL / BLOCKED_PENDING_USER_APPROVAL** | Same-class sampler và `photo_semantic_target()` đã audit. True-pair mapping không tồn tại trong current dataset contract; cần user chọn nguồn mapping/định nghĩa instance và negative cùng lớp trước khi code/train. |
| 3. Rank+CE vs +relational prediction | **NOT_RUN / BLOCKED_PENDING_USER_APPROVAL** | Chưa có relational KD implementation hay target anchor manifest. Cần chốt fixed photo anchors, pairing/multi-positive rule, temperature, grad normalization và same-budget comparison. |
| 4. No-mask vs pre-encoder mask, clean + partial mAP | **NOT_RUN / BLOCKED_PENDING_USER_APPROVAL** | Chưa có mask operator/placement. Cần chốt mask trước patch embedding/encoder, content-aware policy, stochastic transforms, clean/partial corruption protocol và whether sketch backbone remains frozen/prompt-only. |
| 5. Visual-only vs text context: no/correct/wrong | **PARTIAL / BLOCKED_PENDING_USER_APPROVAL** | Existing FP1/FP1S/FP2/FP3 chứng minh loss-only text và no-text inference, nhưng chưa có controlled context-at-inference matrix, wrong-text lineage hay 99% result. Cần chốt text context source, ambiguity policy và whether wrong text is adversarial or naturally sampled. |

### Quyết định user sau audit

User chọn **chỉ đóng băng weights CLIP**: cho phép photo prompt trainable thay đổi gallery embeddings. Mọi weights CLIP, gồm LayerNorm, projections và logit scale, vẫn phải giữ nguyên. Vì vậy gallery CLIP gốc cố định là một đối chứng tùy chọn, **không phải ràng buộc bắt buộc của user**; các dòng readiness phía trên mô tả đề xuất strict-output ban đầu, chưa phải protocol đã chốt.

Pairing có thể chỉ dùng làm supervision phụ trong training; không cần đổi relevance đánh giá từ category-level sang FG. Quyết định frozen weights không tự cấp phép chạy cả campaign GPU; cần chốt config và ngân sách trước khi chạy.

## 10. Primary sources đã đọc

- [SketchLVM / CLIP for All Things ZS-SBIR, arXiv:2303.13440v3](https://arxiv.org/html/2303.13440v3): visual prompts **và cập nhật LayerNorm**, text CE, Sketchy/TU-Berlin/QuickDraw split và Table 1 metrics. Vì vậy không phải baseline hoàn toàn frozen theo ràng buộc mới.
- [LeJEPA, arXiv:2511.08544v3](https://arxiv.org/html/2511.08544v3): isotropic Gaussian motivation, SIGReg/Epps–Pulley và distinction from moment-only regularization.
- [SeCo-SBIR, arXiv:2608.03120v1](https://arxiv.org/abs/2608.03120): paper tồn tại, submitted 2026-08-04; abstract mô tả text-guided coupling, frozen reference consistency và multi-objective SBIR. Đây là related work, không phải evidence cho SPICA.
- [CoCoOp, arXiv:2203.05557v2](https://arxiv.org/abs/2203.05557v2): conditional prompt và nguy cơ static prompt overfit base classes.
- [I-JEPA, arXiv:2301.08243v3](https://arxiv.org/abs/2301.08243v3): masking scale/informative context là design choice cho semantic prediction.
- [DINO-WM, arXiv:2411.04983v2](https://arxiv.org/abs/2411.04983v2) và [V-JEPA 2, arXiv:2506.09985v1](https://arxiv.org/abs/2506.09985v1): predictive latent representation trong world/video setting; không chứng minh category SBIR target hiện tại.
- [Sketch Less for More, arXiv:2002.10310v4](https://arxiv.org/abs/2002.10310): partial/early sketch motivation ở FG-SBIR; không phải bằng chứng cho mask implementation của SPICA.

## Files changed

- `docs/spica_verification_2026-09-06.md` — báo cáo mới; không historical artifact nào bị overwrite/xóa/di chuyển.

## Verified limitations

Chưa có training mới cho JEPA proposal, relational KD, masking, true pairing, wrong-text matrix, official unseen hoặc multi-seed. Không có kết luận “vượt SketchLVM”, không có mAP promise, và không promotion mainline từ các số pseudo-unseen.

## 11. CPU/config preparation: hard-text photo-prompt pilot

This appendix records preparation only. **No new training or GPU pilot was run**, and none of the five readiness experiments is complete.

### Preparation contract

`configs/experiments/frozen_prompt_pilot_hard_text_photo_prompt.yaml` reuses the existing `frozen_prompt_final_FP2` treatment fields (rank + hard text CE, trainable sketch/photo prompts, frozen CLIP-owned weights), but it is deliberately a **CPU/config smoke gate**, not a new primary/final campaign run:

- `run_kind: smoke`, `experiment_campaign: frozen_prompt_final_smoke_2026-09-04`;
- isolated manifest: `outputs/experiments/frozen_prompt_hard_text_photo_prompt_pilot_cpu_gate/experiment_manifest.json`;
- one update (`max_steps: 1`) only, with `allow_short_run: true`;
- DataLoader sample-order randomization remains `shuffle=True` in the trainer. The removed proposal was **patch-shuffling loss**, not DataLoader random shuffle; this pilot has no patch-shuffling loss implementation or claim;
- full CLIP weights remain frozen; sketch/photo prompt tensors are the configured optimizer targets. **`train_frozen_prompt.py:1446-1457` computes its own softplus rank loss without detaching photo embeddings**, so both prompts receive gradients. The separate `jepa_ranking_loss` helper detaches targets but is not used by this trainer; it must not be used to diagnose this baseline's gradient path. No trainer/loss change is required for photo-prompt gradients. A random toy text bank in the CPU probe is not a full text-tower test.

The role/treatment is reused so the existing validator can audit the intended loss and parameter policy. The campaign string is a **reused legacy smoke identifier**, not a newly registered campaign; the manifest path and output namespace are isolated. This preparation does not write to `experiment_manifest_frozen_prompt_final_2026-09-04.json` and must not be reported as part of the completed historical campaign. `ensure_manifest()` creates/checks the isolated smoke manifest and `manifest_entry_identity()` records its role entry. The historical final manifest remains untouched. The config validator is not `--cfg` resolution: `_validate()` was executed by the probe below.

Exact CPU/config checks:

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES='' .venv/bin/python scripts/check_frozen_prompt_pilot_cpu.py
# PASS: CPU rank+hard-CE updates both prompts; frozen visual weights unchanged
PYTHONPATH=src CUDA_VISIBLE_DEVICES='' .venv/bin/python -m spica.train_frozen_prompt \
  --config-name train_frozen_prompt \
  +experiments=frozen_prompt_pilot_hard_text_photo_prompt device=cpu --cfg job --resolve
# PASS; resolved smoke campaign, isolated manifest, run_kind=smoke, max_steps=1, no shuffle override
```

The GPU command is intentionally **NOT EXECUTED**:

```bash
PYTHONPATH=src .venv/bin/python -m spica.train_frozen_prompt \
  --config-name train_frozen_prompt \
  +experiments=frozen_prompt_pilot_hard_text_photo_prompt device=cuda
```

Do not copy the earlier `CUDA_VISIBLE_DEVICES='' ... device=cuda` form as a runnable GPU command: an empty CUDA visibility mask makes CUDA unavailable and the trainer correctly rejects `device=cuda`. No 1800-step pilot was called because this preparation config is smoke-only and the user did not authorize training.

The CPU probe mirrors the frozen-prompt trainer's inline softplus rank expression and calls its existing text-CE, optimizer, gradient assertions and `_validate()` on separate toy sketch/positive-photo/negative-photo tensors. Both prompts receive nonzero gradients and update; frozen visual parameters remain unchanged. It is a toy gradient/config check, not an end-to-end trainer execution or a full CLIP text-tower/logit-scale test. The text bank is a random fixed tensor. The 1800-step primary pilot still needs a distinct campaign identity: the existing validator hardcodes historical primary campaign names, so this preparation stops at the supported isolated smoke gate instead of silently registering a new primary campaign.

### Pairing-candidate audit

`scripts/audit_sketchy_pairing_candidates.py` now fails closed per photo stem: duplicate photo-manifest rows are reported as `AMBIGUOUS_PHOTO_STEM` and are not selected as a unique photo. Multiple sketches mapping to one photo candidate are reported separately as `candidate_keys_with_multiple_sketches` / `candidate_max_sketches_per_photo_candidate`; this is expected many-to-one candidate coverage, not ambiguous photo candidates. Label agreement and path existence remain candidate-only checks; the report status remains `CANDIDATE_ONLY_NOT_VERIFIED`.

The script runs a stdlib `TemporaryDirectory` self-check covering suffix mapping, duplicate photo stems, label mismatch, and missing paths before auditing real manifests. It emits summary counts and examples only, not a large candidate-key list. Exact check:

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES='' .venv/bin/python scripts/audit_sketchy_pairing_candidates.py \
  --output /tmp/spica_sketchy_pairing_candidates.json
# PASS self-check; real-manifest report written to /tmp/...; candidate-only, not verified
```

### Parent review: kết quả tái lập cuối cùng

Sau khi sửa probe dùng đúng loss của frozen-prompt trainer (không dùng helper JEPA detached), parent chạy lại tất cả lệnh CPU ở trên: probe **PASS**, Hydra resolve **exit 0**, pairing self-check/audit **exit 0**, full pytest **183 passed in 3.26s**, Ruff **PASS**, `git diff --check` **PASS**. Raw audit tạm: `/tmp/spica_pairing_parent_review.json`; resolved config tạm: `/tmp/spica_pilot_config_parent_review.yaml`. Hai script trong repo là đường tái lập nếu file tạm không còn.

Audit pairing fail-closed phát hiện giới hạn quan trọng mà phép dict-by-stem ban đầu đã bỏ qua:

| Manifest | Sketch có photo stem ứng viên | Sketch có nhiều photo rows cùng stem | Sketch có đúng một photo row và path tồn tại |
|---|---:|---:|---:|
| train | 57,587 | 57,581 | 6 |
| zero/heldout | 12,694 | 12,694 | 0 |

Ví dụ stem `n04379243_22674` có cả `256x256/photo/tx_000000000000_ready/table/n04379243_22674.jpg` và `EXTEND_image_sketchy_ready/table/n04379243_22674.jpg` trong train manifest. Không thể chọn photo row bằng dict ghi đè rồi gọi mapping unique. Cần xác minh nguồn ảnh gốc/extended, nội dung và quy tắc chọn pairing; chưa kết luận các bản này là cùng ảnh hay khác instance. Zero label mismatch chỉ áp dụng tập match unique, không xác nhận các trường hợp ambiguous. Key overlap train/heldout bằng 0; không chạy official metrics.

### Readiness status

This work changes preparation files and documentation only. It does not establish a completed pilot, comparison, checkpoint, mAP result, true-pair mapping, or promotion decision. GPU training, official unseen evaluation, multi-seed, covariance, 5400-step extension, lambda search, masking, relational KD, and wrong-text matrix remain **NOT RUN**.

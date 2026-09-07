# Hướng 1: gradient diagnostic và chuẩn bị hướng 2 — 2026-09-07

## Trạng thái và kết luận

**Chẩn đoán GPU đã hoàn tất, exit0; không optimizer, không cập nhật weights, không retrain.** CPU preflight dùng CLIP/dữ liệu thật PASS; toàn bộ **230 tests PASS**, Ruff và `git diff --check` PASS. Reviewer độc lập không còn blocker; parent tính lại norm/dot/cosine từ raw gradient NPZ.

Kết quả sửa lại giả thuyết trước chẩn đoán:

1. **Chưa thấy full/masked rank đối nghịch trên photo prompt:** cosine dương ở **48/48** phép đo. Không nên gọi gallery drift là hậu quả đã chứng minh của gradient conflict ở photo.
2. **Rank và hard-text CE thường đối nghịch trên sketch prompt**, kể cả ở control C. Với M@1800 clean, cosine trung bình **−0.7481**, CE/rank norm ratio trung bình **2.2264**, cả4 batch cosine âm.
3. **Xung đột giữa full/masked sketch gradients tập trung ở severe masking:** M@1800, mask75%, cosine tổng trung bình **−0.0897**, âm ở3/4 batch, masked/full norm ratio **1.8088**. Trong cùng điều kiện, rank full/masked vẫn cùng hướng mạnh (**0.9521**), còn CE full/masked có cosine trung bình **−0.0433**.
4. Các phép đo **không chứng minh causal explanation cho unseen mAP giảm**. Đây là raw gradients ở các trạng thái checkpoint, trên128 pseudo-train sketches, không phải AdamW updates hay ablation teacher.

**Hướng2 mới ở mức proposal/preparation, chưa triển khai teacher/KD, chưa chọn lambda hoặc chạy training.**

## Protocol đã thực thi

- Training pilot gốc: `outputs/masked_view_gpu_pilot_20260907T011358Z/`.
- Checkpoints C/M tại **250 và1800**. Step250 là điểm tham chiếu hậu kiểm từ trajectory đã biết, không dùng để thay fixed-step1800 result.
- Bốn batch32 đầu tiên trong historical training trace: `global_step=0,1,2,3`, tương ứng update1–4; **128 sketch paths khác nhau**. Replay cùng query/label/positive/negative cho cả hai arms và cả hai checkpoints.
- Chỉ pseudo-train classes; không dùng pseudo-validation hoặc official unseen để đo gradient.
- Positive từ canonical same-class pool; negative được xác minh thuộc full pseudo-train photo pool và khác label. Positive historical SHA được kiểm tra. Historical trace không có SHA cho mọi negative; diagnostic lưu SHA hiện tại của negative, **không tuyên bố có historical negative-byte identity khi hash cũ là null**.
- CLIP eval transform hiện hữu, deterministic; full sketch pixel hashes tái tạo đúng historical training input. Query/photo tensors được cache dùng chung; hashes của batch tensors được lưu và đối chiếu.
- `ink_centered_square_v1`, threshold0.9, fractions25/50/75%, mask seed4242 + root-relative sketch path, `view=0`; không thêm step vào seed để giữ vùng che nhất quán giữa checkpoints. Đây là **fixed diagnostic mask plan**, không replay toàn bộ lịch ngẫu nhiên của training.
- Tổng cộng **48 measurement pairs** =2 arms ×2 checkpoints ×4 batches ×3 fractions. Có384 logical masked samples;1536 mask records khi lặp qua arms/checkpoints, tất cả status `ok` và metadata khớp. Bốn batches không phải bốn training seeds; ba fractions dùng cùng sketch, không độc lập.
- Model eval mode, original CLIP visual/text/LayerNorm/projections/logit scale frozen. Chỉ sketch/photo prompts có `requires_grad`; mỗi prompt2304 parameters, tổng4608.
- Photo positive/negative encode một lần cho mỗi batch; reuse graph cho full và masked losses. Query views xử lý riêng bằng cùng model; frozen ViT không có batch-coupled training statistics. CPU check so forward batch2 với stacked batch4; không tuyên bố đã test forward CPU batch32/64.

### Loss và gradient đo

Giữ nguyên production loss, không dùng helper ranking có photo detach:

\[
L_{r,v}=\operatorname{mean}\operatorname{softplus}(0.2-q_v^\top p^++q_v^\top p^-),
\quad L_{c,v}=\operatorname{CE}(q_vT^\top/0.07,y),
\]

với embeddings normalized, hard text bank của pseudo-train classes, template `a photo of a {}`.

\[
g_v=\lambda_r g_{r,v}+\lambda_c g_{c,v},\quad
\lambda_r=\lambda_c=1,\quad
 g_{mix}=0.5(g_f+g_m).
\]

`torch.autograd.grad` tách bốn thành phần `full_rank/full_ce/masked_rank/masked_ce` theo sketch/photo; không `.backward()` accumulation, không optimizer. CE→photo là **structurally unused (`None`)**, lưu vector0 và cosine undefined/null; không gán cosine0 để giả vờ có gradient.

Ở checkpoint C, masked/mixed gradient là **phép đo counterfactual**; C lịch sử chỉ train full+full. Không gọi nó là update đã xảy ra ở C.

## 1. Full–masked gradients theo từng prompt @1800

Các số là **trung bình metric của4 batch**, không phải cosine giữa hai vector gradient trung bình. Ratio là trung bình norm(masked)/norm(full); đây không phải ratio của hai mean norms.

| Arm | Mask target | Sketch total cosine | Sketch norm ratio | Sketch cosine âm | Photo rank cosine | Photo norm ratio |
|---|---:|---:|---:|---:|---:|---:|
| C |25%|0.6086|1.0969|0/4|0.8855|1.0831|
| C |50%|0.2955|1.5576|0/4|0.6809|1.4292|
| C |75%|0.3141|1.5782|0/4|0.5681|1.8363|
| M |25%|0.6105|1.0356|0/4|0.9389|0.9858|
| M |50%|0.2362|1.5430|0/4|0.8351|1.0972|
| M |75%|**−0.0897**|**1.8088**|**3/4**|**0.6224**|1.2465|

Photo total gradient bằng photo rank gradient vì CE không phụ thuộc photo prompt. Full/masked photo cosine dương ở cả48 phép đo, bao gồm cả step250. Điều này bác bỏ mô tả **“photo full và masked đang kéo ngược nhau trên các batch đã đo”**, không bác bỏ mọi khả năng gallery co-adaptation gây kém generalization.

### Sự thay đổi ở M từ step250 đến1800

| Mask | Sketch total cosine @250 | @1800 | Sketch norm ratio @250 | @1800 |
|---|---:|---:|---:|---:|
|25%|0.8215|0.6105|0.8782|1.0356|
|50%|0.5582|0.2362|0.7555|1.5430|
|75%|0.1344|−0.0897|0.7312|1.8088|

Masked gradient ở M về cuối trở nên lớn hơn tương đối và ít cùng hướng hơn. Chỉ có hai trạng thái checkpoint: chưa chứng minh tiến triển monotonic giữa chúng hoặc cơ chế update đã gây ra sự thay đổi.

## 2. Rank so với CE trên sketch prompt

| Checkpoint | View | Mean rank norm | Mean CE norm | Mean CE/rank ratio | Mean cosine | Cosine âm |
|---|---|---:|---:|---:|---:|---:|
| C@250 |full|2.5520|5.7563|2.2383|−0.1655|3/4|
| C@250 |masked|3.7515|5.9635|1.6302|0.2309|3/12|
| C@1800 |full|1.7917|6.3252|3.5632|−0.3569|4/4|
| C@1800 |masked|2.3405|7.7128|3.3964|−0.1179|10/12|
| M@250 |full|1.4518|7.9649|5.5730|−0.4998|4/4|
| M@250 |masked|1.6973|5.9129|3.6202|−0.2360|11/12|
| M@1800 |full|3.2063|7.0730|2.2264|**−0.7481**|4/4|
| M@1800 |masked|3.3279|7.1014|2.1891|−0.1240|8/12|

Full gradients chỉ được đếm một lần/batch, không nhân ba do lặp ở ba severities. Tổng full component conflicts là15/16 unique full batches/checkpoint combinations; masked là32/48. Cũng không coi bốn checkpoints dùng cùng samples là independent datasets.

CE có norm lớn hơn rank trong các trung bình trên, nhưng **không thể nói M thua vì CE/rank ratio lớn hơn C**: ở step1800, ratio của M thực tế nhỏ hơn C. Dấu hiệu nổi bật hơn là **góc đối nghịch mạnh hơn trên clean branch của M**.

### Kiểm tra hệ quả first-order, không nhầm với AdamW

Từ raw vectors, cosine giữa **full rank gradient và full total gradient trên sketch**:

- C@1800: trung bình **−0.0744**, âm2/4 batch.
- M@1800: trung bình **−0.4113**, âm4/4 batch.

Với một bước SGD rất nhỏ **chỉ trên sketch prompt**, đi theo `−g_total` trong các trường hợp cosine âm có xu hướng tăng local rank loss. Đây là cách định lượng trade-off rank–CE; không phải khẳng định actual total rank loss tăng, vì actual update còn có photo branch và AdamW moments/preconditioning.

Cẩn trọng thêm: ở M@1800, cosine giữa **full rank và mixed total** lại tăng từ−0.2452 ở25%, lên0.0764 ở50%, rồi0.2530 ở75%. Nghĩa là severe masking không đơn giản gây hại cho mọi thành phần rank: nó đổi hướng trade-off. Không được lấy một cosine âm full-total để kết luận mọi retrieval gradient đều bị phá.

## 3. Nguồn của severe full/masked conflict ở M@1800

| Mask | Sketch rank full↔masked cosine | Sketch CE full↔masked cosine |
|---|---:|---:|
|25%|0.9943|0.7808|
|50%|0.9828|0.3576|
|75%|**0.9521**|**−0.0433**|

Rank gradients giữa hai views vẫn gần cùng hướng, trong khi CE mất alignment khi thiếu nhiều thông tin. Đây là evidence trực tiếp mạnh hơn giả thuyết “mọi masked gradient đều xung đột với clean”.

Dù M@1800 mask75% có full/masked total cosine âm ở3/4 batch, **mixed-total projection lên full-total vẫn dương ở cả4 batch**; sketch mean projection0.3868, min0.0681. Vì vậy chưa được nói ngay cả local clean total loss cũng chắc tăng theo một bước SGD nhỏ. Góc âm giữa hai components không đồng nghĩa tổng sau average đảo chiều.

## 4. Chuẩn bị hướng 2 — proposal, NOTRUN

### Giả thuyết cập nhật

Một full-view teacher ổn định có thể cung cấp target representation cho masked query khi hard class evidence yếu. Tuy nhiên, thêm KD cũng thêm gradient có thể đối nghịch rank/CE. **Teacher không tự bảo đảm giữ clean mAP, và gradient conflict hiện có cũng xuất hiện ở C.**

### Đề xuất tối thiểu để xin duyệt trước implementation/training

- Fixed teacher lấy từ **C@1800**, original CLIP cùng identity, prompt teacher frozen, `eval()` + `no_grad()`; không EMA trong phép thử đầu.
- Mục tiêu hướng2 ưu tiên là **full-teaches-masked**: cùng một sketch full cho teacher, masked cho student. Không cần true sketch-photo pairing; không dùng ảnh validation/unseen làm target.
- Candidate KD tối thiểu: `mean(1 - cosine(student_masked, stopgrad(teacher_full)))` trên normalized output. Đây là **feature distillation**, không tự gọi là JEPA hoặc LeJEPA.
- Giữ hard text CE và ranking như yêu cầu. Không âm thầm bỏ CE, đổi sampling, giảm mask ratio hoặc thêm loss bảo vệ full cùng lúc.
- Clean-anchor `student_full → teacher_full` là một lựa chọn khác, trực tiếp kiểm tra bảo toàn clean; **không tự gộp hai KD placements**, vì sẽ khó biết thành phần nào giúp.

### Kiểm tra cơ chế trước khi chọn lambda/train

Một bước đo **không update** có thể tái sử dụng fixed batches/checkpoints để lấy `g_KD` và so với:

1. `g_rank,masked`, `g_CE,masked` theo từng severity.
2. `g_full,total`, đặc biệt mask75%.
3. Tổng task gradient và norm ratio, để phát hiện KD chỉ lấn át hoặc làm tăng conflict.

Nếu student khởi tạo bằng teacher và dùng clean-anchor, gradient KD tại initialization có thể bằng0: không chia norm để calibration tại trạng thái này. Masked target có khác biệt input nên là phép đo khác. Không tự chọn lambda dựa trên mAP hoặc mở search.

**Chưa chạy phép đo teacher này.** Kết quả hiện tại chỉ là direction1 decomposition; proposal không được trình bày như evidence teacher có ích.

### Gallery/frame và control

Để thử riêng teacher trong một retrieval frame ổn định, đề xuất **freeze photo prompt từ C@1800 ở cả candidate và control mới**, cùng warm-start C@1800, cùng budget mới. Original CLIP vẫn frozen toàn bộ. Đây là lựa chọn thiết kế để loại bớt gallery drift, **không phải kết luận dữ liệu đã chứng minh photo conflict**.

Nếu user muốn tiếp tục train photo prompt, cả hai arms mới đều train photo prompt và cần theo dõi teacher-query/current-gallery alignment. Không được khẳng định target frame mismatch tất yếu làm thí nghiệm vô nghĩa: đó là một biến động cần kiểm soát/đo.

Control mới phải giống candidate về initialization, photo trainability, batch/steps/seeds và full+masked supervision; chỉ khác KD weight0 so với candidate. Không dùng historical C/M để thay cho control có warm-start/freeze khác. C@1800 checkpoint không tự có nghĩa student đã chạy thêm1800steps; báo rõ pretraining stage và continuation stage riêng.

Các quyết định **chưa được duyệt**: photo-prompt policy, target placement, student initialization, KD coefficient/calibration và training budget. Chưa tạo config training runnable với các giá trị đoán.

### Điểm tích hợp tối thiểu khi được duyệt

- Tái sử dụng `FrozenPromptModel` và `load_prompt_checkpoint`; không cần format checkpoint mới hoặc teacher factory.
- Một teacher forward detached và một KD term trong trainer; logs phân biệt rank/CE/KD, checkpoint teacher identity và treatment mới.
- Candidate/control configs mới, provenance campaign mới; không chỉnh historical manifests/configs/checkpoints.
- CPU test teacher bất biến, original CLIP frozen, target detached, đúng view, cấu hình KD0 khớp control.
- Không LoRA, EMA, SIGReg, patch shuffle, covariance hoặc official-unseen evaluation tự động.

## Evidence và khả năng replay

Successful execution root:

`outputs/masked_view_gradient_diagnostic_20260907T040030Z/`

- `command.txt`, `execution.log`, `exit_code.txt`.
- `diagnostic/summary.json`: COMPLETE, lineage, configs/source hashes, before/after checkpoint/model hashes.
- `diagnostic/raw_gradients.npz`:12 arrays, mỗi array `(48,2304)`, gồm bốn component gradients và hai totals theo hai prompts.
- `diagnostic/raw_rows.jsonl`:48 records, scalar losses, norms/dots/cosines, input hashes, direct-autograd additivity errors.
- `diagnostic/fixed_trace_manifest.jsonl`, `batch_sample_manifest.jsonl`: samples, positive/negative identities, hashes, mask metadata.
- `diagnostic/input_run_result_C.json`, `input_run_result_M.json`: snapshots nguồn training, không sửa originals.
- `diagnostic/source_snapshot/`:19 source files dùng cho diagnostic, tách khỏi historical training source.
- `verify_and_summarize.py`, `verified_analysis.json`: parent CPU replay và bảng aggregate, không model/GPU.

Replay raw evidence:

```bash
.venv/bin/python outputs/masked_view_gradient_diagnostic_20260907T040030Z/verify_and_summarize.py
```

Chạy lại diagnostic, nếu cần, phải dùng output directory **mới**; exact command ở `command.txt`. Diagnostic GPU section khoảng28.5s, peak allocated7,440,801,792bytes (~6.93GiB), không bao gồm toàn bộ CPU lineage/preflight thời gian trước đó.

### Checkpoint lineage

| Arm/step | SHA256 |
|---|---|
| C@250 | `9e746231c6d8c417da25640e94c6729a33adff44430dc293d4b76d4e512857cc` |
| C@1800 | `0af31d564f4068a5dbd510acbee533c73d9a1d44c67213da07d26e4c702a5d4a` |
| M@250 | `93e034ce843e912b1823fb8d34e7d81e6012e12f33e8d8a9aedbee7d151be1a3` |
| M@1800 | `ba66c367f006edc560ce980d35d61fb220e23e571386f860db52f0f94c1d3bad` |

Tất cả hashes trước/sau khớp. Full model state hash trước/sau khớp; runtime assertion cũng kiểm tra original CLIP state không đổi và `.grad` của parameters không tích lũy. Không lưu hoặc cập nhật optimizer.

### Một lỗi kiểm tra số học đã được sửa

Attempt `outputs/masked_view_gradient_diagnostic_20260907T035820Z/` exit1 vì coordinate-wise `allclose` của gradient tổng với tổng component gradients quá chặt quanh tọa độ gần0. Không có optimizer trong attempt này; không sửa training/checkpoint.

Source trước sửa được giữ ở `outputs/masked_view_gradient_linearity_fix_20260907T035934Z/diagnose_before_fix.py`. Bản chạy thành công dùng **relative-L2 error ≤1e−5**, ghi cả relative error và max absolute error; regression test vẫn reject thành phần sai thực sự.

- Max relative-L2 error giữa **direct autograd total và tổng components**: **2.7735e−6**.
- Max absolute error: **8.5905e−6**.
- Không nhầm với sai số nhỏ hơn khi chỉ cộng lại arrays NPZ: đó là phép kiểm tra khác, không có thêm backward pass.

Failed artifacts giữ nguyên. Không retrain hoặc đổi loss để vượt check.

## Giới hạn cuối cùng

Bốn fixed training batches, một training seed/split, một diagnostic mask seed, hai checkpoint states; không confidence interval hoặc bằng chứng toàn training trajectory. Raw gradients không bao gồm Adam moments, learning rate, weight decay, Hessian hoặc held-out generalization. Mọi khẳng định về hướng2 vẫn cần phép thử có control được duyệt mới.

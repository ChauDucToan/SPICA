# Campaign 3600/600 và đề xuất hướng 2

## Quyết định đã được user chốt

- Mỗi run mới **3600 optimizer updates**, đánh giá/lưu checkpoint mỗi600 steps; kết thúc khi chạy hết, không early stopping.
- Probe0 là reference; checkpoint đủ các mốc `0,600,1200,1800,2400,3000,3600`.
- Lưu riêng `latest`, `best_clean`, `best_masked` cho mọi run. Best chỉ xét600–3600; điểm bằng nhau giữ checkpoint sớm hơn.
- `best_clean`: maximize clean **mAP@200_prefix_positive** trên pseudo-validation.
- `best_masked`: maximize macro **mAP@200_prefix_positive** trên9 điều kiện: fractions25/50/75% × mask seeds101/202/303.
- P@200 phải lấy tại chính checkpoint được báo cáo; không lấy peak P@200 ghép với peak mAP@200 từ step khác.
- Log cả P@200, full mAP và ba AP@200 denominator conventions riêng. Không đổi historical metrics hoặc chọn checkpoint từ official test.
- User đồng ý dùng `prefix_positive` để selection trong lúc benchmark-paper equivalence chưa được xác minh.

**3600 là ngân sách mới, không phải tuyên bố baseline đã hội tụ hoặc đạt nghiệm tối ưu.**

## Convention SketchLVM: phần xác minh được và chưa được

Primary sources đã đọc:

- [SketchLVM evaluator, commit ca33be98cb9f986813f611c997ceaa474b0bc96e](https://github.com/aneeshan95/Sketch_LVM/blob/ca33be98cb9f986813f611c997ceaa474b0bc96e/src/model_LN_prompt.py)
- [Environment của chính repo](https://github.com/aneeshan95/Sketch_LVM/blob/ca33be98cb9f986813f611c997ceaa474b0bc96e/environment.yml): torchmetrics0.9.3.
- [Implementation retrieval_average_precision của dependency](https://github.com/Lightning-AI/torchmetrics/blob/ff61c482e5157b43e647565fa0020a4ead6e9d61/torchmetrics/functional/retrieval/average_precision.py).

Evaluator public gọi `retrieval_average_precision(distance, target)` trên toàn gallery, không cắt200; dependency tính AP trên toàn vector. Điều này **không xác định công thức mAP@200 dùng cho bảng paper**. Vì vậy không gọi `prefix_positive` là official-equivalent.

Với \(S_{200}=\sum_{k\le200}P(k)rel(k)\), \(R\) là toàn bộ positives và \(R_{200}\) là positives trong top200:

| Tên ghi log | Mẫu số |
|---|---|
| `mAP@200_prefix_positive` | \(\max(1,R_{200})\) |
| `mAP@200_all_relevant` | \(R\) |
| `mAP@200_min_relevant_k` | \(\min(R,200)\) |
| `full_mAP` | AP toàn gallery chia \(R\) |

P@200 = positives trong top200/200. Các metric được average theo query, không macro theo class. Ranking stable giữ thứ tự gallery khi score bằng nhau. Query không có positive gallery bị chặn theo contract SPICA, không âm thầm tính0.

Một ranking pass tạo đủ các denominator, không thêm encoder forward chỉ để tính metric khác. Pseudo split3407 không phải official104/21 test; chưa so số mới trực tiếp với paper.

## Giai đoạn 1: reference và hướng 1 trong ngân sách mới

Campaign mới: `frozen_prompt_masked_view_3600_2026-09-07`.

| Arm | Query views train | Initialization | Trainability |
|---|---|---|---|
| C3600 | full + full | from scratch, cùng pretrained CLIP | sketch/photo prompts |
| M3600 | full + masked | giống C3600 | giống C3600 |

Configs:

- `configs/experiments/masked_view_3600_C.yaml`
- `configs/experiments/masked_view_3600_M.yaml`

Cả hai giữ canonical same-class positive pool, full negative pool, rank + hard text CE, equal view weights0.5, seed42/pseudo split3407, batch32/workers4. Original CLIP visual/text/LayerNorm/projections/logit scale frozen toàn bộ. Không teacher/KD/LoRA/SIGReg.

Không continuation từ historical1800 vì lịch probes/source protocol mới phải so từ đầu cho hai arms. Không relabel run1800 thành3600. Mọi checkpoint cũ giữ nguyên.

## Giai đoạn 2: proposal A–D, chưa triển khai/chạy

### A. Teacher/reference

**Đề xuất teacher = C3600/best_clean**, vì teacher nhận full sketch và mục tiêu checkpoint đã được user chốt trước. Luôn báo thêm C3600/latest để biết teacher tốt nhất theo validation nằm ở step nào và có suy giảm cuối training không.

Không gọi teacher là optimal hoặc gold. Nếu latest vẫn cải thiện, report rõ budget-limited; không tự kéo dài training hoặc chọn checkpoint ngoài ngân sách.

### B. KD gradient diagnostic trước training

Teacher frozen, `eval()`/`no_grad()`; teacher nhận full sketch, student nhận masked sketch của cùng sample:

\[
L_{KD}=\mathrm{mean}(1-\cos(z_{student,masked},\operatorname{stopgrad}(z_{teacher,full}))).
\]

Tái sử dụng fixed pseudo-train batches. Đo KD norm/cosine theo sketch prompt so với masked rank, masked CE và full task gradient. KD không trực tiếp tạo photo gradient trong công thức này.

Đo trên student khởi tạo bằng teacher và có thể checkpoint M3600 như một trạng thái khảo sát khác, nhưng phải ghi rõ state; không trộn thành một calibration. Không optimizer update. Đây là feature distillation, không tự gọi là JEPA/LeJEPA.

### C. Hệ số và photo policy

Đề xuất một calibration train-only để khởi tạo hệ số, **không search theo validation mAP**:

\[
\lambda_{KD}=\rho\;\operatorname{median}_b
\frac{\|g_{task,b}\|}{\|g_{KD,b}\|},\qquad \rho=0.1\;\text{(đề xuất, chưa duyệt/chạy)}.
\]

`g_task` phải được định nghĩa là gradient của loss full/masked average thực sự dùng ở stage2; báo norm ratio và angle từng severity. KD gradient0/không finite thì calibration fail, không chia epsilon để tạo lambda rất lớn. Giá trị0.1 chỉ là điểm khởi đầu kiểm soát độ lớn, không nghiệm tối ưu.

**Đề xuất tiếp tục train photo prompt ở cả control và candidate đầu tiên**, phù hợp baseline hiện hữu. Direction1 diagnostic không tìm thấy full/masked photo-rank conflict; chưa có lý do đủ mạnh để đổi photo trainability cùng lúc. Fixed-photo teacher experiment vẫn là lựa chọn riêng nếu muốn loại gallery drift, nhưng phải có control riêng cùng frozen-photo policy.

### D. Control/candidate mới

Sau khi duyệt B/C:

| Run stage2 | Initialization | Rank + hard CE | KD | Photo prompt |
|---|---|---|---|---|
| D2-control | cùng C3600/best_clean | full+masked |0|trainable|
| D2-candidate | giống control | giống control |lambda đã chốt|trainable|

Mỗi run stage2 đề xuất3600 **additional updates**, cùng optimizer/scheduler mới và batch/probe/selection protocol600; không một arm resume optimizer còn arm kia reset.

Báo `source_checkpoint_step`, `additional_updates` và effective inherited update count. Nếu teacher best tại step2400 thì student state kế thừa2400 updates rồi thêm3600; chi phí xây teacher vẫn là cả run3600 đã hoàn tất. Không gọi tất cả là một experiment từ đầu chỉ tốn3600.

Chưa tạo teacher training configs hoặc chạy KD diagnostic. Photo policy, calibration ratio, warm-start/continuation stage2 vẫn là proposal cần duyệt; C3600/M3600 infrastructure không tự cấp phép stage2.

## W&B/Hydra và tái sử dụng code

Tái sử dụng `src/spica/tracking/wandb.py`, không logger framework mới. Schema scalar chung:

```text
step_train
clean/P@200
clean/mAP@200_prefix_positive
clean/mAP@200_all_relevant
clean/mAP@200_min_relevant_k
clean/full_mAP
masked/macro/<same metrics>
masked/fraction_025/<same metrics>
masked/fraction_050/<same metrics>
masked/fraction_075/<same metrics>
masked/fraction_025/seed_101/<same metrics>
... đủ9 conditions
```

Mỗi experiment có W&B run riêng, cùng campaign group, role/name riêng. Artifact name chứa run ID để tránh alias của rerun đè nhầm run khác. Các aliases `step600`, `step1200`, …, `latest`, `best_clean`, `best_masked` chỉ đến checkpoint tương ứng; local giữ mọi step file bất biến và ba named copies cập nhật atomically.

Local lưu đầy đủ resolved config, `.hydra/{config,hydra,overrides}.yaml` khi chạy Hydra CLI, command, source snapshot/index/SHA, uv.lock trong snapshot, checkpoint prompts + optimizer/scheduler/RNG/data/source identities. Direct CPU test gọi `run()` không tự tạo Hydra CLI files; test đó chỉ chứng minh resolved config/source được lưu.

W&B config scalar được whitelist; không log raw image/mask dataset thành scalar. Checkpoint artifact vẫn chứa provenance/path metadata để tái lập, nên **không xem whitelist config là đã xóa mọi local path khỏi checkpoint binary**. Không upload toàn output directory hoặc ảnh dataset.

Online mode/entity/project còn chờ user xác nhận; configs hiện `tracking.mode: disabled` để CPU gate không vô tình gửi dữ liệu. Không tuyên bố đã có W&B dashboard, upload hoặc backend alias verification.

Reused modules: CLIP loader/transform, `FrozenPromptModel`, checkpoint loader, data/pairing/masking, text bank/CE, retrieval evaluator, provenance và tracking. New3600 role/config/protocol tách khỏi historical1800. Metrics refactor dùng chung sort, giữ API và defaults historical.

## Báo cáo sau khi run hoàn tất

Mỗi row là một checkpoint cụ thể:

| Experiment | Loại | Step | Clean P@200 | Clean mAP@200_prefix | Masked macro P@200 | Masked macro mAP@200_prefix | SHA/W&B |
|---|---|---:|---:|---:|---:|---:|---|
| C3600 | latest |3600|…|…|…|…|…|
| C3600 | best_clean |…|…|…|…|…|…|
| C3600 | best_masked |…|…|…|…|…|…|
| M3600 / future variant | cùng3 loại |…|…|…|…|…|…|

So latest–latest, best_clean–best_clean, best_masked–best_masked. Không peak candidate so latest control. Log/report các denominator phụ riêng, không ghép số để tạo chiến thắng.

## Readiness và evidence hiện có

Final gate: `outputs/masked_view_3600_final_gate_20260907T072146Z/gate_summary.json` — **CPU_INTEGRATION_VERIFIED, 244 tests PASS**, Ruff/diff check PASS. Parent kiểm tra SHA của named checkpoint copies/source snapshot/uv.lock và6 replays (3 selections ×2 arms), clean replay delta0. W&B URLs trong fixture là fake, không phải dashboard đã tạo. Chưa chạy real GPU3600 hoặc online upload.

- CPU tiny-CLIP production training2 updates cho C/M mới; real tracking wrapper với fake backend, cả9 masked conditions mỗi probe.
- Independent replay `latest/best_clean/best_masked` bằng evaluator standalone trên CPU fixture; checkpointSHA và clean metric khớp. Chưa phải real GPU3600.
- Historical compatibility: `outputs/masked_view_3600_historical_parity_20260907T064255Z/parity_summary.json` — archived1800 package vs current, cảC/M trên2 CPU updates: observations/losses/metrics và12 prompt tensors khớp chính xác, CLIP weights bất biến.
- Independent metric parity audit:52 synthetic fixtures, old archive vs refactor, default fullAP/P@K/AP@K/topindices/topscores bit-exact.
- Historical parity nhỏ không chứng minh mọi GPU run tái tạo bitwise; source mới được archive riêng, không thay training snapshot cũ.
- `outputsnewgate/` là smoke artifact trung gian đã có trước final integration, giữ nguyên; không coi nó là chứng nhận source cuối.

Commands đề xuất, **chưa chạy GPU/online**:

```bash
HF_HUB_OFFLINE=1 PYTHONPATH=src .venv/bin/python -m spica.train_frozen_prompt \
  +experiments=masked_view_3600_C device=cuda \
  tracking.mode=online tracking.project=<approved_project> tracking.entity=<approved_entity>
# Sau khi C hoàn tất mới chạy M với cùng overrides môi trường/protocol.
```

Không dùng lệnh này trước khi xác nhận W&B destination và real-data/GPU preflight. Replay checkpoint mới vào output directory mới:

```bash
PYTHONPATH=src .venv/bin/python scripts/evaluate_masked_view.py \
  --run-result <run>/run_result.json --selection best_clean \
  --output-dir <fresh_directory> --device cuda
```

Nếu dùng `--checkpoint-step` thì không đồng thời dùng `--selection`; historical default vẫn1800. `best_clean/best_masked/latest` là lựa chọn explicit cho campaign3600.

## Khép hướng1

Pilot1800 và gradient diagnostic đã hoàn tất; kết luận giữ trong ngân sách đó. Vòng3600 chỉ khép khi C/M train đủ3600, tất cả probes600 và latest/best replay hợp lệ, metrics/source/checkpoint/W&B evidence đầy đủ. Nếu online lỗi, local evidence giữ lại và trạng thái upload chưa hoàn tất, không tự nhận đã upload. Không early stop vì validation giảm, không tự mở thêm variants/seeds/search để tìm kết quả tốt.

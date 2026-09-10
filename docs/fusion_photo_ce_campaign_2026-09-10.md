# F2_MP_PCE — calibration, gates và một arm chạy ngầm

## Phạm vi user duyệt

User: **“Chọn lambda và train ngầm đi”**, sau đó “continue”. Một arm mới từ khởi tạo, 3600 updates, online W&B; không resume baseline, không A/C/D, không validation-based lambda search, không retry tự động. Đây là protocol/prelaunch evidence, **chưa phải kết quả training**.

- Arm `F2_MP_PCE`, method `coupled_predictive_fusion_mp_photo_ce_v1`.
- Giữ toàn bộ F2_MP, thêm `lambda_photo_ce * CE(actual_unique_live_photos, learned_train_text_bank)`; tau .07, mean unique bank **một lần/update**, cả photo/text không detach.
- Giữ model, sampler B32/1 positive + 1 negative/sketch/≤64 unique photos/full58,950 pool, clean+masked views, optimizer, LR/warmup180/cosine3600, seed42. Không SIG.
- Training main μ_I; inference **q qua QOnlyAdapter**, không predictor. Historical F2_MP default μ_I không đổi.
- Primary fixed **clean/full_mAP@3600 vs F2_MP-Q q .49166918150172645**; masked full secondary `.3557525980363653`. Prefix-selected best aliases auxiliary, không lấy candidate peak so fixed control.

## λ được đo và khóa

`scripts/diagnose_fusion_photo_ce.py`, root `outputs/fusion_photo_ce_diagnostic_20260910T122000Z/`.

Bốn batch đầu B32 tại cùng initialization, **0 optimizer updates**, dùng gradient của full F2_MP base và raw photo CE trên cùng graph. Photo CE không trực tiếp tác động sketch student nên đo hai scope riêng: `photo_model.photo_prompt`, `text_bank.context`.

`lambda = .1 * min_scope median_batch(||g_base|| / ||g_photo_ce||)`.

| Scope | Candidate λ | Realized weighted CE/base gradient range |
|---|---:|---:|
| Photo prompt | **0.03291338002672159** | 4.75–17.57% |
| Text context | 0.0390486811814284 | 6.89–9.81% |

Chọn **λ_photo_ce = 0.03291338002672159**, photo scope binds. Đây là quy tắc median-ratio initial gradient, không per-batch 10% cap, không đảm bảo median realized đúng 10% với 4 batch, không effective AdamW update ratio hoặc mAP optimum. Photo/base cosine dương 4/4; text/base âm 3/4 (khoảng −.493 đến +.329). Không kết luận xung đột tồn tại suốt training hoặc CE sẽ cải thiện retrieval.

- Real cached CLIP, 84 classes, initialization exact historical MP;128 traces/masks exact; model/teacher unchanged, `.grad` None và RNG unchanged trong từng phép đo.
- Raw `diagnostic_result.json` giữ **MEASURED_PENDING_REVIEW**, `verified:false`.
- Independent CPU `independent_cpu/verify_cpu.py`: 2194 checks;582 archived source files,9 current components, cached CLIP SHA/data manifest,128 real mask/input reconstructions,16 raw prompt gradient tensors/FP64 algebra. Safe `weights_only=True`; không independent GPU replay.
- Parent `parent_review.py` / `parent_receipt.json`: đóng hai giới hạn independent: worker tính candidates nhưng gán sẵn selected/binding; parent thực sự lấy min từ raw. Tiny independent CE là algebra-only; parent so **actual production CE và photo/text prompt gradients** với logsumexp oracle. Không sửa original evidence.
- Reviewed `diagnostic_verified.json` SHA **`5c2edbd2e080c5bf16d223d5cb09b390f2e48e42e520c28a087672948c79cb74`**; accepted cùng parent receipt trong gate, scope calibration-only.
- Diagnostic snapshot HEAD39d4b50 + new diagnostic/launcher; later docs/commit không phải measurement snapshot. Archive authoritative. Parent sửa lỗi cosine trên 2D prompt tensors trước GPU và thêm runnable self-check; không GPU rerun.

## CPU và actual GPU gates

- `outputs/fusion_photo_ce_calibration_cpu_20260910T120900Z/receipt.json`: diagnostic self-check PASS (base scalar/gradient, CE, 2D cosine, selection, zero rejection).
- `outputs/fusion_photo_ce_runner_review_20260910T121000Z/receipt.json`:6 CPU contract checks PASS (valid/tampered/missing/unsafe/unreviewed receipts, child command); synthetic λ.37 không production coefficient. `cpu_gate/receipt.json`: existing PCE checker4PASS.
- Prior implementation MP/QMP/TQMP standalone regressions5+6+5PASS remain historical unchanged source evidence, not newly rerun full test suite.
- `outputs/fusion_photo_ce_optimizer_gate_20260910T123500Z/F2_MP_PCE/`: **actual CUDA2updates, W&B disabled**, COMPLETE2. No additional updates during verifier.
- `verify_gpu.py` / `gpu_gate.json`:179 optimizer states,358 finite/nonzero moments, actual optimizer restore tensor/group equality; model initial tensors,64 traces/masks/LRs exact historical MP; frozen CLIP unchanged.
- Step2 checkpoint SHA **`96923e5150b5763cf522638431bbeff1fcd98a4500ff52768eb63af8037e8a7e`**. Smoke checkpoint not production candidate.
- First real batch base11.131014823913574, photoCE3.197282075881958, weighted total11.236248016357422; production/diagnostic component and total deltas0.
- q32 full/bypass exact; full production clean +9mask probe10963queries/13999photos, predictor0 calls, model hash before/after exact. Serialized metric sanity gate, not second independent all-query full sorting.
- Peak allocated8,587,645,952B; reserved9,114,222,592B for smoke (not production peak guarantee).
- Parent corrected unexecuted worker gate scripts: serialized metadata location/string, W&B disabled, before-state placement, artifact relative paths and CPU receipt schema. Preserved `*_worker_unexecuted.py`; no failed GPU attempt or extra optimizer run.

## Launch binding and monitoring

`scripts/run_fusion_multipositive.py --arm F2_MP_PCE --diagnostic <reviewed receipt>` validates reviewed+raw/artifacts/current components and passes **only `--lambda-photo-ce`** to trainer, not `--diagnostic` (PCE trainer forbids that route). λ remains explicit, no production default. Trainer λ source correctly remains `explicit_argument_no_calibration_claim`; calibration identity lives in SHA-bound runner manifest and archived receipts, not falsely marked as trainer-internal calibration.

`assemble_launch_gate.py`: CPU/GPU/source/data/init/calibration/qbaseline + read-only W&B baseline connectivity;15components, immutable evidence. Gate SHA **`b1373dea950945e6e9ee049909e44fca44c003c8c2ddb10dc1ff88ed19993056`**.

Fresh planned root: **`outputs/fusion_photo_ce_execution_20260910T125500Z/`**. Adjacent `.launch.json`, `.runner.log`, runtime and actual PIDs authoritative; do not launch a duplicate. Runner inert without `--launch`, fresh-root +24GiB guard, source snapshot and no retry/resume. Detached runner wrapped in idle inhibition; **avoid manual sleep/reboot**, idle inhibition does not block those actions.

Freeze source/config/docs after launch until finish. Completion becomes `ARM_FINISHED_UNVERIFIED` plus local fixed-step comparison, not online artifact/full-replay VERIFIED or promotion. Preserve unrelated `.gitignore`, unfinished V1 verifier diff, `outputsnewgate/`; no push.

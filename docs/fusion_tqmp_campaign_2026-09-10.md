# F2_TQMP — authorized single3600 arm (2026-09-10)

User: **“Bắt đầu train đi”**, after accepting sequential MP(μ_T)→MP(q) initialization-gradient calibration. One new arm from common seed42 initialization; no baseline retraining, resume, retry, seeds, official-unseen, or coefficient search.

## Locked objective / readout

Arm **F2_TQMP**, campaign/method **coupled_predictive_fusion_tqmp_v1**, architecture unchanged `predictive_fusion_v2`.

`L = L_F2_MP + 0.17877235601108843 MP(mu_T) + 0.023774345199536452 MP(q)`.

Retain main MP(μ_I)1, CE_i1, CE_t.25, q paired ranking.25/CE.25, align_i/t.05, anchorsI/T.5. Mean clean/masked50–50. All MP terms use average-per-positive negative log probability over **same live unique photo bank**, tau.07. No SIG. New logged `clean/masked_rank_t_mp` and `clean/masked_rank_q_mp` coexist with `rank_i`; no removal/replacement of μ_I term as in failed F2_QMP.

Main inference **q**, existing QOnlyAdapter (Hmean→pooled_head→normalize), predictor skipped during probes. Training retains all heads; metadata `training_main_query=mu_i` identifies coefficient1 ranking head, **not** evaluation readout. Existing F2/F2_MP defaults unchanged; TQMP explicit route only.

Sampler/model/data/LR unchanged: B32,1positive+1negative/sketch,≤64unique photos total, full58950photo pool, seed42/pseudo3407,84train/20pseudo-validation classes. No64positives/sketch, no official-unseen21classes. Same AdamW groups, warmup180+cosine3600, probes0/600/.../3600.

Primary fixed3600 **clean full mAP vs F2_MP-Q q=.49166918150172645**; masked q full mAP=.3557525980363653 secondary. Baseline separate no-update q-readout receipt SHA `f210017b24f93fb659315daa820c1265c698aeb9a52112803543db63cd452afa`, MPcheckpoint3600 SHA `58fb5fb2aa0f822d1df8bee9192a0bda5eec1580b4ce70ed9ae243fdcf9fb14e`. Historical W&B y40hu06b is training identity with μ_I metrics, not source of q scalar. No old-run metric mutation. best_clean/best_masked retain **prefix-positive AP200**, auxiliary on new q readout. No peakcandidate vsfixedcontrol.

## Calibration binding

[Calibration report](fusion_mp_tq_diagnostic_2026-09-10.md), root `outputs/fusion_mp_tq_diagnostic_20260910T001200Z/`. Verified receipt **diagnostic_verified.json**, SHA **d3021b6ecd7a1b0d89b1da5c62761442728f6f095a1c9a4ace572ca9525cd68b**. Raw source573files SHA `42485ce4bfc86dc296cba91d997a3d9a810c1859a82d87eb9566bc7d1705b53f`.

Trainer requires --diagnostic for TQMP, rejects wrong campaign/receipt before device/data, validates approved SHA plus14bound files (including raw gradients/independent/parent receipts), exact lambdas, initialization and split/manifest/full-pool identities. No lambda CLI override. Training source intentionally adds route/code/docs; no false requirement full training-source SHA equal earlier diagnostic SHA. CPU loss/loader parity and actual GPU loss replay bind the semantics instead.

These coefficients are initialization-scale measurements only:10% median-ratio target, not per-batch cap, optimal mAP, or same AdamW update ratio. λ_q pooled cap binds; no promise that norm/directional behavior persists through training.

## Gates before production

- New `scripts/check_fusion_tqmp_cpu.py`; parent `outputs/fusion_tqmp_cpu_parent_20260910T013500Z/` **5PASS/0FAIL/0SKIP**. Independent `outputs/fusion_tqmp_independent_20260910T013500Z/`: new5 +legacyMP5 +legacyQMP6 = **16PASS/0FAIL/0SKIP**, source-bound review. Scalar/gradient oracle B+λT·T+λq·Q, λzero base exact, other terms exact, correct t→predictor/notpool and q→pool/notpredictor, two tiny AdamW steps, model reload/frozen unchanged. CPU reload is model-only; actual optimizer reload verified below. make_train_loader/prepare_batch AST unchanged, legacy objective gradients/schema preserved.
- Real CUDA2updates COMPLETE at **`outputs/fusion_tqmp_optimizer_gate_20260910T014500Z/F2_TQMP/`**. Init tensors/hash exactMP smoke;64traces/masks/LR bytes exact.179optimizer states/358finite nonzero moments, optimizer restore exact, frozen original unchanged. Metadata/checkpoint/result all exact q/lambdas/diagnostic identity.
- Additional no-update initial realbatch check: all4diagnostic components B/T/Q/textCE replay **delta0**, production total vs weighted diagnostic **delta0**, state unchanged. This tests implementation against actual calibration, not only a toy graph.
- Reloaded TQMP@2 full clean+9masked probe10963queries/13999gallery PASS via historical archived serialized-metric helper (not unfinished current V1 verifier). q full-vs-bypass real32exact, predictor0calls during probe, modelstate unchanged. No production-performance claim.
- Peak2update memoryallocated8,587,647,488bytes/reserved9,116,319,744; includes optimizer, not deployment measurement.
- Read-only online baseline connectivity PASS. No synthetic train result; historical QMP artifact visibility issue remains separate, not repaired or claimed verified here.
- Combined **launch_gate.json SHA a4c069c60c93fbc8d68e1740df69aaf2aa2bef8c43c68f37635b6fd5a5773e5f**,14components/9evidence files, under optimizer-gate root. Gates ≠ production training or performance improvement.

## Launch and safety

Reuse `scripts/run_fusion_multipositive.py --arm F2_TQMP --diagnostic ...`, default historical arm unchanged. Inert without--launch; output must be fresh child ofoutputs and≥24GiBfree. Gate/source/component/readout/calibration hashes verified. Archive source, baseline q receipt, calibration verified+raw receipts; pass archived verified receipt to child. One3600arm online W&B; no retry/resume, automatic extra replays or old full-campaign verifier.

Fresh plannedroot **`outputs/fusion_tqmp_execution_20260910T014500Z/`**. Adjacent launch receipt/log and actualruntime/PIDs authoritative; do not duplicate. Freeze source/config/docs whileactive. Completion remains **ARM_FINISHED_UNVERIFIED** + localfixedstep comparison, not online/artifact/full-selected-replay certification.

Idle inhibition available; manual sleep/reboot can still interrupt training. No machine policy changes. Preserve pre-existing `.gitignore` (`sda/`), unfinished V1verifier diff and outputsnewgate/; never remove historical tensors to free space.

```bash
LD_LIBRARY_PATH=/run/opengl-driver/lib HF_HUB_OFFLINE=1 WANDB_MODE=online \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \
systemd-inhibit --what=idle --mode=block --who=spica --why='Authorized F2_TQMP3600' \
.venv/bin/python scripts/run_fusion_multipositive.py --arm F2_TQMP --launch \
 --diagnostic outputs/fusion_mp_tq_diagnostic_20260910T001200Z/diagnostic_verified.json \
 --gate outputs/fusion_tqmp_optimizer_gate_20260910T014500Z/launch_gate.json \
 --output outputs/fusion_tqmp_execution_20260910T014500Z
```

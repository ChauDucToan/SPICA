# F2_QMP — transfer MP supervision to q, no MP for μ_T (2026-09-09)

## Authorization / locked intervention

User accepted **implement + checks + one3600-update training arm**, then `continue`. Explicit new arm `F2_QMP`, method/campaign `coupled_predictive_fusion_qmp_v1`, same `predictive_fusion_v2` architecture. Train from scratch seed42/pseudo split3407, **q main retrieval**. Not a retraining of the historical F2_MP control, not a continuation/resume of its checkpoint.

Only transfer the main photo term:

\[
L_{QMP}=L_{F2\_MP}-L_{MP}(\mu_I,B_{photo})+L_{MP}(q,B_{photo}).
\]

Per view: `MP(q)*1 + CE(mu_i)*1 + CE(mu_t)*.25 + paired_rank(q)*.25 + CE(q)*.25 + align_i*.05 + align_t*.05`. Average clean/masked50–50, anchorsI/T .5 each. **No MP(mu_i), no MP(mu_t)**. μ_I/μ_T heads remain trainable through CE/alignment; μ_T is NOT removed. q paired rank remains intentionally alongside MP(q). MP averages negative log-probability for each same-class positive over unique live batch photo bank; tau.07 fixed, no temperature/weight search or equal-gradient-scale claim. New logging `clean_rank_q_mp`/`masked_rank_q_mp`, not misleading `rank_i`.

Data/model/init/optimizer/scheduler unchanged from F2_MP: B32, one sampled positive+negative/sketch, up to64unique photos total, full58950photo pool, same masks and seeds,84trainclass text bank. No64positives/sketch, no sampler change, no official-unseen21classes. Original CLIP frozen; prompts/student/predictor train as before. SIG0/no diagnostic accepted.

`QOnlyAdapter` promoted to evaluation helper for **explicit F2_QMP evaluation only**. Hmean→pooledhead→normalize; zero predictor forwards during q probe. Historical F2/F2_MP default μ_I adapter and checkpoint schemas retained. Train still uses full model/all heads.

## Fixed baseline / metrics

Primary **clean/full_mAP at3600**, vs **F2_MP-Q readout q** `.49166918150172645`; masked full mAP `.3557525980363653` secondary. Baseline training W&B `y40hu06b` has historical μ_I metrics, **not** these q metrics. SHA-bound q-only baseline receipt: `outputs/fusion_mp_q_evaluation_20260909T145000Z/summary.json`, SHA `f210017b24f93fb659315daa820c1265c698aeb9a52112803543db63cd452afa`; checkpoint3600 SHA `58fb5fb2aa0f822d1df8bee9192a0bda5eec1580b4ce70ed9ae243fdcf9fb14e`. Trainer verifies before device/data; launcher archives receipt independently.

Report all5clean/masked metrics and fractions/conditions every600 including0; no peakcandidate vsfixedcontrol. `best_clean`/`best_masked` aliases retain prefix-positive AP200 selection (on q for this arm), **auxiliary**, not full-mAP primary selection. No q/μ head search. One seed only, no significance/superiority promise.

## Gates

- Worker intermediate CPU /tmp/spica-qmp-test* roots preserved; final accepted receipts below, not temporary prototypes.
- Independent strengthened checker `scripts/check_fusion_qmp_cpu.py`, root `outputs/fusion_qmp_cpu_independent_20260909T180400Z/`: **6PASS/0FAIL/0SKIP**. Parent rerun `outputs/fusion_qmp_cpu_parent_20260909T181500Z/` same. Source-bound actual production `_view_terms` q/bank gradient nonzero, MP gradient to μ_I/μ_T absent; total equals oldMP−MP(muI)+MP(q) within float32 2e-6; other terms exact. Legacy F2 AND F2_MP archived scalar/gradient exact; loader/packing AST archived exact; training forward hooks one branch each;47tinyactive gradients finite/nonzero; synthetic production-group AdamW/checkpoint/moments restore exact with correct frozen-original seed. New q-only output exact and predictor0calls; trainer q selection/olddefault routing checked. No pretrained CPU claims.
- Real CUDA **2updates COMPLETE**, root `outputs/fusion_qmp_optimizer_gate_20260909T183300Z/F2_QMP/`. Initial model tensors/init exact F2_MP smoke;64trace/mask rows and LR bytes exact.179optimizer states/358finite nonzero moments; optimizer restore exact, frozen original unchanged. Peakallocated8,587,645,440bytes/reserved9,114,222,592.
- Reloaded QMP@2 clean+9masked full probe10963queries/13999gallery checked with archived serialized-metric helper. Production `_make_eval_probe(...,'F2_QMP')` routes q with predictor **0calls**; real32clean q full-model/bypass exact; state unchanged. Gate is not3600 result.
- Online connectivity READ_ONLY to finished historical training run; q baseline verified separately by receiptSHA. No synthetic q training result or W&B mutation.
- Combined launch gate `outputs/fusion_qmp_optimizer_gate_20260909T183300Z/launch_gate.json`, SHA **`16ec9239bd487fe51a5b1ea80ce9bddb7584913ae930cc5e6ad4075cf0abb4c5`**,12sourcecomponents/5evidencefiles (CPU parent+independent, GPU, connectivity, baseline readout). Actual GPU proof `gpu_gate.json`, script `verify_gpu.py`, full probe saved separately.

## Execution and boundaries

Reuse `scripts/run_fusion_multipositive.py` with **explicit `--arm F2_QMP`**, default historical `F2_MP` unchanged. Inert without--launch; fresh output,24GiB free gate, current/archive source components match; correct qbaseline comparison and archived readout; no retry/resume or extra arms. Completion **ARM_FINISHED_UNVERIFIED** + local fixed-step comparison only, not automatic full replay/W&Bverification. CPU gates and smoke are separate from production.

Fresh plannedroot **`outputs/fusion_qmp_execution_20260909T184000Z/`**; adjacent launch receipt/PIDs/runtime authoritative. Freeze source/config/docs whileactive, avoid duplicate launches. Idle inhibitor works; explicit sleep inhibition denied by machine policy, avoid manual sleep/reboot.

```bash
LD_LIBRARY_PATH=/run/opengl-driver/lib HF_HUB_OFFLINE=1 WANDB_MODE=online \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \
systemd-inhibit --what=idle --mode=block --who=spica --why='Authorized F2_QMP3600' \
.venv/bin/python scripts/run_fusion_multipositive.py --arm F2_QMP --launch \
  --output outputs/fusion_qmp_execution_20260909T184000Z \
  --gate outputs/fusion_qmp_optimizer_gate_20260909T183300Z/launch_gate.json
```

Preserve `.gitignore` pre-existing dirty (`sda/`), unfinished `scripts/verify_coupled_campaign.py` diff and outputsnewgate/. No old V1 verifier repair, push, dependencies change, baseline training, MPμ_T, newseed/unseen/search or automatic replays. New documentation/commit changes aggregate snapshot; actual launch archive determines training source, not prior gate aggregate.

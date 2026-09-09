# F2_MP — authorized single-arm full-mAP ablation (2026-09-09)

## Authorization and locked intervention

User selected **full-gallery mAP**, then requested implement/start training and `continue`. One fresh arm **F2_MP**, campaign/method `coupled_predictive_fusion_mp_v1`, **3600 updates**, seed42/pseudo split3407, online W&B. No SIG, baseline retraining, extra seed, official-unseen evaluation, lambda/temperature search, automatic retry/resume.

Same F2 architecture `predictive_fusion_v2`, independent sketch student/shared prompts, frozen original CLIP, inference **mu_i**, warmup180+cosine, optimizer/groups/LRs and all auxiliary coefficients. Only `rank_i` changes from paired softplus to:

\[
L_{MP}=-\frac1B\sum_i\frac1{|P_i|}\sum_{p\in P_i}
\log\frac{\exp(\hat\mu_{I,i}^{\top}\hat z_p/0.07)}{\sum_{a\in\mathcal B_{photo}}\exp(\hat\mu_{I,i}^{\top}\hat z_a/0.07)}.
\]

`P_i` is every same-class photo in the **existing unique live batch photo bank**. Different-class photos are negatives; no cosine filtering, hard mining, symmetric photo→sketch loss or memory bank. Each query has its own positive in the bank. Average *negative log probability of every positive*, not negative log summed positive probability. A disproportionately high-scoring positive can receive a downweighting gradient toward the uniform positive target; not a promise to increase every individual positive logit.

**Sampler unchanged**: B32 queries, one sampled positive and one negative each, up to64 unique photos total. Full58,950-photo pool; canonical8400photos audit only. Multiple positives come from label matches within that existing bank, **not64 positives per sketch**. Same photo encoding/reference/text/query call counts and mask RNG. Worker initially introduced64positives/sketch; parent rejected/removed this before any GPU execution. Preserved intermediate CPU receipt `outputs/fusion_multipositive_cpu_20260909T112953Z/` is **not** the accepted launch gate.

Per view: `MP(mu_i)+CE(mu_i)+.25*CE(mu_t)+.25*(paired_rank(q)+CE(q))+.05*(align_i+align_t)`; average clean/masked50–50; add `.5*anchor_i+.5*anchor_t`. Main loss coefficient1, temperature.07 fixed, **not calibrated for equal gradient magnitude to F2**. This tests the loss replacement including its scale, not an equal-gradient contrastive ablation. Original paired softplus defaults remain exact for historical arms.

## Locked comparison

Primary: **clean/full_mAP at step3600**, against historical F2 run [1wxvk2lk](https://wandb.ai/a-cctest05187-erd/spica/runs/1wxvk2lk), **0.4713651857643906**. Source baseline SHA `9dc0c3df2b730dddf3726587de007ad9df6eccb3d06ccf7a7a0767a0f5e5ec80`, training HEAD `649a249`. All other5-metric clean/masked trajectories remain available every600 steps, including step0. P200/prefixAP200 and masked full mAP report trade-offs; no guarantee all improve or multi-seed significance.

Legacy `best_clean`/`best_masked` aliases still select **prefix-positive AP200**; auxiliary only for this experiment. No relabeling historical selection or choosing candidate peak against fixed control. `latest` at3600 is the primary comparison checkpoint. `main_photo_objective`, temperature, method/version, loss/sampling identity and primary-comparison definition are serialized; no inference switch to q/mu_t.

## Gates and evidence

- Parent CPU: `outputs/fusion_mp_cpu_parent_20260909T113600Z/receipt.json`, **5PASS/0FAIL/0SKIP**, verifiedtrue. Formula/gradients, class membership, bank permutation, paired F2 archive exact forward/loss/gradient, unchanged other terms/coefficients, tiny active47gradient tensors, synthetic AdamW/checkpoint, routes and full-pool1positive metadata.
- Independent CPU/review: `outputs/fusion_mp_independent_20260909T114000Z/{receipt,independent_review}.json`, **PASS**. Explicit NumPy per-positive oracle; loader/prepare_batch AST exactly HEAD and archived F2; dedup/photo labels/positive existence; invalid route/stale gate rejection; inert runner. No GPU/network from independent worker.
- Actual pretrained/real-data CUDA two updates: `outputs/fusion_mp_optimizer_gate_20260909T114000Z/F2_MP/`, COMPLETE2. Parent verification `gpu_gate.json`: initial model tensors exactly historical F2 smoke;64traces/masks/LRs byte-identical.179optimizer states /358finite nonzero moments, optimizer reload exact, frozen original unchanged, checkpoint/source hashes verified. Peak allocated **8,587,645,440bytes**, reserved9,114,222,592.
- Reloaded MP@2 full no-update probe:10,963queries/13,999gallery, clean+9maskedconditions, serialized per-query metrics checked with archived verifier helper. Original/current state hash unchanged. This is readiness, **not3600 results**.
- W&B current connectivity read-only verified against historical F2 latest scalar in `online_connectivity.json`. Existing F2 online contract already361rows/6aliases verified; current MP serialization tested CPU, production run URL only after actual launch. No synthetic MP result invented.
- Combined gate `outputs/fusion_mp_optimizer_gate_20260909T114000Z/launch_gate.json`, SHA **`c9d4538309a2c8f63e84fd6d8299fcd17a2adb8d33ba4b712d51d3014a0598c3`**, binds11components and5CPU/GPU/online receipts. Assembly initially assumed independent-review receipt had CPU-checker `summary`, raisedKeyError; original script/failure preserved, assembly-only fix, **no retraining**.

## Execution / fail-stop

New launcher `scripts/run_fusion_multipositive.py` is inert without `--launch`, validates fresh output under `outputs/`, ≥24GiB free for one arm, gates/current+archived component hashes and baseline checkpoint hash. Exact training source archived before child; trainer rejects source changes through completion. Finish status **ARM_FINISHED_UNVERIFIED**, local fixed-step comparison only; training failures and postprocessing failures distinguished. Does not invoke unfinished V1 campaign verifier or auto-replay.

Planned root **`outputs/fusion_mp_execution_20260909T115500Z/`**; adjacent launch receipt + runtime/PIDs authoritative. Do not launch again if root exists. `systemd-inhibit --what=idle --mode=block` available; blocking explicit sleep denied by system policy. Idle inhibitor does **not** prevent manual sleep/reboot: user must avoid those. No machine policy edits or process kills.

```bash
LD_LIBRARY_PATH=/run/opengl-driver/lib HF_HUB_OFFLINE=1 WANDB_MODE=online \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \
systemd-inhibit --what=idle --mode=block --who=spica --why='Authorized F2_MP3600 training' \
.venv/bin/python scripts/run_fusion_multipositive.py --launch \
  --output outputs/fusion_mp_execution_20260909T115500Z \
  --gate outputs/fusion_mp_optimizer_gate_20260909T114000Z/launch_gate.json
```

Freeze source/config/docs during actual campaign. Preserve unfinished `scripts/verify_coupled_campaign.py` diff and `outputsnewgate/`, all historical data/checkpoints/artifacts. Post-gate documentation/commit may change aggregate source hash; component bindings remain authoritative, actual launch archive is training source.

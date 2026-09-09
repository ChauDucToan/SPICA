# F2 → F2+SIG: authorized 3600-update campaign

## Permission and launch contract

User explicitly authorized **GPU preflight and training F2 and F2+SIG**. This supersedes the implementation-only scope in [F2 CPU readiness](coupled_predictive_fusion_v2_implementation_2026-09-09.md), but does not authorize other arms, seeds, official-unseen evaluation, lambda searches, or automatic retries/resume.

- Sequential arms **F2 → F2_SIG**, each from scratch3600updates, online W&B `a-cctest05187-erd/spica`.
- Planned fresh root **`outputs/fusion_execution_20260909T064000Z/`**; actual process start is recorded in its launch receipt/runtime. Root name may be preallocated, not a measurement of start time.
- Runner `scripts/run_fusion_campaign.py` is inert without `--launch`; source/archive/receipt checks and40GiB prelaunch/15GiB per-arm disk gates. No duplicate run, no fallback batch/AMP, no retry/resume, no deletion of historical outputs.
- Freeze tracked/nonignored source/config/docs after launch until both arms finish. Root includes executed source archive, provenance, copied gate/diagnostic receipts, runtime PIDs/commands/exits, per-arm checkpoints/probes/W&B identifiers.
- Both arms complete → `ARMS_FINISHED_UNVERIFIED` and local selection summary. This runner does **not** invoke the unfinished historical V1 verifier or automatically relabel the campaign VERIFIED. W&B unsampled histories/selected artifact downloads can be verified after completion; no extra replay campaign launched here.

## Matched architecture, sampling, budget and selection

Both arms use exactly `predictive_fusion_v2`: sketch student, shared prompts, cross-attention first, contextual joint self-attention+residual, FFN and output split; **main inference mu_i**. q/mu_t remain auxiliary. Same full positive/negative photo universe58950, class-correct same-class positives/different-class negatives, train84/pseudo-val20/official21excluded, B32, clean/corrupted50–50, seed42, workers4, FP32/noAMP. Original CLIP weights frozen, live prompts train.

Loss per view, before clean/corrupted mean:

`rank_i + ce_i + .25*ce_t + .25*(rank_pool+ce_pool) + .05*(align_i+align_t)`

Plus `.5*anchor_i + .5*anchor_t` once. F2 has no SIG; F2_SIG alone adds

`lambda_sig * .5*(SIGReg(g_clean[B32]) + SIGReg(g_corrupted[B32]))`.

g is unnormalized student patch mean. SIGReg private CPU RNG starts from42 independently in training; it does not continue the diagnostic RNG. No model/init/sampler differences are introduced by SIG. F2 initialization identity now explicitly includes the whole predictor state, in addition to common student/pool/prompts/T0.

Optimizer AdamW, student LR1e-5, predictor/pool1e-4, prompts1e-4; matrix WD.01, vector/bias/LN0, promptsWD1e-4; betas(.9,.999),eps1e-8.180update warmup + cosine, scheduler final state0 at3600. Same actual group-LR trajectory.115200original observations/230400views per arm.

Probes0/600/.../3600;10963queries/13999gallery, clean plus9conditions (deletion25/50/75% × seeds101/202/303). Five explicitly named metrics: full mAP, P200, AP200 prefix-positive/all-relevant/min(R,200). Positive-step selections only; earliest tie:

- latest3600
- best_clean: clean prefix AP200
- best_masked: macro9 masked prefix AP200

Promotion F2_SIG vs F2 at their respective best_clean: clean prefix strict increase, P200/full mAP nondecrease, masked macro prefix strict increase at those same selected checkpoints. Negative result is not verification failure. Do not choose q/mu_t/fusion output after seeing results. F2 vs historical R1 is a changed architecture+sampling+supervision package, not an isolated attention ablation; S0 is not a matched control here.

## New F2 SIGReg gradient scale

No-update real pretrained/data diagnostic:

`outputs/fusion_sigreg_diagnostic_20260909T055900Z/`

Architecture `predictive_fusion_v2`, active positive pool `full`. Main task numerator = mean views **rank_i+ce_i**, excludes auxiliary/ref terms. Denominator = separate-view SIGReg average. Scope last student transformer block,12tensors/7087872coordinates. Four fixed train batches32, mask steps0..3, no optimizer/model updates; raw FP32 gradient arrays saved, norm/dot/cos recomputed CPU FP64 independently.

|Batch|Task/SIG gradient norm ratio|Gradient cosine|
|---|---:|---:|
|0|0.06750762602762986|−0.03614131106990464|
|1|0.08573461568557375|−0.10874451988646933|
|2|0.09345955456275831|−0.040254556888027204|
|3|0.08212052709241|0.028760460418608955|

**Fixed lambda =0.1 × median ratio = `0.008392757138899188`.** CV0.11462264; max/min1.38442958. Modest observed spread, no post-hoc numerical spread threshold; mixed gradient cosines do not establish conflict-free updates. This is not a calibrated optimum, validation-tuned lambda, guaranteed10% AdamW update, or Gaussianity/retrieval-quality proof.

- Raw `diagnostic_result.json` remains `MEASURED_PENDING_REVIEW`; SHA256 `e37115390768c800dbd36c235a40e850f0dc53ca2bcc70befa0a9b7f9c935bc6`.
- Separate verified receipt `diagnostic_verified.json` PASS, SHA256 **`e695ad3566d5c2571c7715529f98fbe7a4c53993934dadbd5d469e8a2376c38b`**, binds raw receipt path/bytes and independent audit. Original raw measurement is not rewritten.
- Independent audit `review/independent_review.json`, SHA256 `32161b74013de6e83d4be53ff784a59cc9792b20876b29fff5253392fd5cc43e`: raw gradients/layout/artifacts/RNG/batch/source checks.118/128positive samples are outside canonical pool; correct labels/data scope checked.
- Source archive559files, aggregate `1918dea904810cecc6efcfd20f20cccf086632f42ce06e67ad6363b250c85b42`. This is the diagnostic/gate source before this doc/commit, **not necessarily the eventual training aggregate**. Trainer binds immutable measurement, component hashes, new F2 architecture/fullpool/task identity/class84/whole-predictor initialization, not current aggregate==diagnostic aggregate.
- No-update peak allocated7,852,782,080bytes/reserved8,231,321,600bytes, no optimizer states in this measurement.

F2_SIG requires this new receipt; V1 receipt alone is rejected. F2 control still rejects any diagnostic argument. F2_SIG actual lambda is serialized in config, checkpoint, W&B config and loss coefficient identity.

## Real optimizer/evaluator and online gates

`outputs/fusion_optimizer_gate_20260909T060500Z/`

Both arms completed **two actual CUDA optimizer updates in separate smoke roots**, not warm-started production training:

|Arm|Trainable optimizer tensors|Peak allocated bytes|Peak reserved bytes|
|---|---:|---:|---:|
|F2|179|8,587,645,440|9,114,222,592|
|F2_SIG|179|8,587,843,584|9,120,514,048|

- Both checkpoints2 SHA-verified and full model state reconstructed from verified CLIP cache+saved tensors.
- Initial model state bytes identical, including fusion predictor; same64observation/mask rows and actual group-LR trajectories.
-179optimizer states,358finite/nonzero Adam moment tensors per arm; optimizer step2, scheduler epoch2. SIG private RNG advanced only in SIG arm.
- Frozen original state unchanged. SIG enabled only in F2_SIG with new fixed lambda.
- Full production F2@2 evaluator:10963queries/13999gallery, clean+9masks, serialized metric/top-index/per-query schema verified. **These smoke metrics are not main campaign results.**
- Initial `verify_gate.py` incorrectly treated SIGReg `_extra_state` dict as a Tensor; verification failed after producing the F2 full probe. Preserved `gpu_gate_failure.json`, `verification.log`, `verify_gate_before_rng_schema_fix.py`. Repair reads `_extra_state['generator_state']`; `verification_v2.log`/`gpu_gate.json` PASS. Reused the already-produced full probe; **no training rerun** or raw checkpoint/probe edits.
- Independent final CPU gate review `independent_final/receipt.json` PASS, SHA256 `b5a82c8ed7a1bf57cce9d9af1cb06785a2e4aaad6791d3fbd189af9a85e0e2af`. Checks actual saved checkpoint moments/initial states/source bindings/RNG plus runner wrong-receipt/missing-components/inert/negative-promotion fixtures; no extra GPU.
- Final combined launch receipt `launch_gate.json` SHA256 **`aff94fad19fd39635209ea3b24f580f95e3f62e0d4d8aecd5386a90a63765800`**, binds diagnostic/gpu/independent/online receipts and12current source components.

Online synthetic contract gate (no dataset uploads/no training metrics):

`outputs/fusion_online_gate_20260909T060800Z/receipt.json`, run [ky9rbcy2](https://wandb.ai/a-cctest05187-erd/spica/runs/ky9rbcy2).

PASS: actual online history steps0/10/600, merged scalars/conditions, selected aliases logged latest-last, downloaded artifact hashes. This is explicitly **synthetic preflight**, not an F2 trained run.

CPU source gate `outputs/fusion_campaign_cpu_20260909T061000Z/receipt.json`:13PASS/0FAIL/0SKIP, includes F2_SIG routing and refusal of an old V1 diagnostic. Ruff and diff check PASS. Historical/V1 pytest suite was not recreated or rerun.

## Execution command and boundaries

Runner starts in a detached session with stdout/stderr redirected to `<root>.runner.log`; command below is the approved launch shape, not permission to launch it twice:

```bash
LD_LIBRARY_PATH=/run/opengl-driver/lib HF_HUB_OFFLINE=1 WANDB_MODE=online \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \
.venv/bin/python scripts/run_fusion_campaign.py --launch \
  --output-dir outputs/fusion_execution_20260909T064000Z \
  --diagnostic outputs/fusion_sigreg_diagnostic_20260909T055900Z/diagnostic_verified.json \
  --gate outputs/fusion_optimizer_gate_20260909T060500Z/launch_gate.json
```

Check live PID and root `runtime.json` before acting. A gate PASS/launch is not completion, convergence, promotion or successful full campaign verification. Keep machine awake; do not reboot/sleep. On process failure preserve evidence and stop—no automatic resume/restart or source edits while the other arm may still be active.

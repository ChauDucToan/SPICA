# SPICA corrected alignment verification — completed locally

## Outcome

**End-to-end verification completed:** real CUDA CLIP calibration, three 50-step smoke runs, R/MD/MS pilots to exactly1800, strict-v2 matching, paired-query bootstrap, checkpoint/state checks and offline gradient/geometry diagnostics. **Scientific result: R > MD > MS; negative for this pilot configuration.**

| Check | Result | Evidence |
|---|---|---|
| Reviewed snapshot | HEAD `82c6461df7931c2c9cc4d8815aab9fe16517e399`; initially clean | `execution_fix.patch`, source archive |
| Targeted tests | 82 passed / 0 failed before and after fix; exit0 | `logs/targeted_tests*.log` |
| Additional pairing risks | 13 mutation checks passed; exit0 | `pairing_negative_checks.json` |
| Ruff | Relevant source/tests and evidence scripts pass | `logs/ruff_after_fix.log`, `logs/ruff_evidence_scripts.log` |
| CUDA | Actual forward/backward/synchronize pass | `gpu_preflight.md`, raw logs |
| Real calibration | VALID; lambda **0.2603831284137216** | `calibration_diagnostic_corrected.json` |
| Smoke | R/MD/MS 50 updates, all VALID; all three pairings MATCHED | `smoke_corrected_report.json` |
| Pilot | R/MD/MS 1800 updates; all VALID | `corrected_pilot_report.json` |
| Matching | R–MD, R–MS, MD–MS all **MATCHED**, mode **corrected_v2** | `matching_validation.json` |
| Checkpoints | All15 exact0/100/500/1000/1800 checkpoints hashed and loaded; step/metric/config/source bound | `artifact_inventory.json`, report hash checks |
| RNG/init | R/MD/MS checkpoint RNG states agree at all5 steps; step0 visual prompts and soft context byte-equal | `checkpoint_state_validation.json` |
| Offline diagnostics | Gradients and geometry at0/500/1800 for all3 arms, exit0 | `raw_metrics/`, `logs/postprocessing_sequence.log` |
| Raw-number replay | AP arithmetic means reproduce all15 scheduled mAP values within1e-12 | `logs/finalize_summary.log`, `finalize_summary.py` |

Primary and secondary numerical tables: [`corrected_pilot_summary.md`](corrected_pilot_summary.md). Every row has run/config/source/checkpoint/step/raw-history links in its companion JSON.

## Concrete execution bug and recheck

The initial HEAD smoke failed **before any update**, at `probe(0)`: `CategoryRetrievalEvaluation` is not subscriptable. Three newly introduced mAP accesses incorrectly used `["full_mAP"]`. The only product-code change uses the existing `.metrics.mean_average_precision` field at those three locations in `src/spica/train_alignment.py`. No model/loss/optimizer/refactor/dependency changes.

Reproducer invocation and traceback are retained in `logs/smoke_R.log`. The same real trainer/probe path then completed R/MD/MS smoke and all pilot evaluation points. Runnable commands remain in `run_fixed_campaign.sh`; to repeat without overwriting evidence, give its OUT a new directory. `tests_and_validation.md` lists every command, exit/count and the initial command/check-script mistakes rather than hiding them.

## Calibration A–G on real frozen CLIP

- **A:** Detached photo alignment norms are `[0,0,0,0]`.
- **B:** Separately measured symmetric photo norms are `[11.275469780,12.502554893,11.539653778,11.372909546]`; the graph reaches photo prompts. No assumption that every batch must be nonzero was used.
- **C:** Detached/symmetric sketch gradients agree with maximum absolute difference **0**, tolerance **1e-6**. Offline diagnostics also check equal mean forward values and sketch gradients on the same four batches at each selected checkpoint.
- **D:** Lambda is exactly `0.1 * median(base_sketch_norm / detached_mean_sketch_norm)` = **0.2603831284137216**. Both MD/MS use the same canonical artifact and SHA256. No post-hoc lambda replacement or tuning.
- **E:** Undefined detached photo cosines are JSON `null` with `alignment_gradient_zero`. Other ratios/cosines carry reasons where undefined; no fabricated zero cosine.
- **F:** Parameters, buffers, RNG, sampler epoch, existing gradients and module train/eval flags are verified restored. The extra real-model invocation deliberately supplies nonzero existing prompt/context gradients and mixed train/eval flags: `raw_metrics/calibration_real_existing_gradient_restoration.json`. Frozen full CLIP state and fixed hard-text anchoring are also checked during offline diagnostics. Soft-text context receives the base CE graph, not the hard-anchor alignment graph.
- **G:** Calibration initialization hashes, split hash, source hash and four fixed-batch identities match MD/MS. Training verifies replay of all four batches; checkpoint metadata records replay count/hash. R’s identical initialization and checkpoint RNG are verified against both variants.

The symmetric weighted photo alignment/base ratios at initialization range **6.115650434–11.255014255**, despite calibration targeting sketch ratio0.1. This is a measured branch imbalance, not proof of the cause of lower mAP. It was not used to retune lambda.

## Why old completed runs were not silently reused as corrected-v2 evidence

Local historical artifacts **did exist**: all old R/MD/MS histories, result JSONs, probes and checkpoint files through1800 were found and checked. Missing GitHub files were never treated as proof of an unrun experiment. Search was limited to configured/referenced project roots; dataset and checkpoint paths are local, not missing external mounts.

The old calibration has the invalid legacy `weighted_photo_gradient_ratios_if_symmetric` field (detached zeros), no separately measured symmetric diagnostics, and no fixed-batch replay/text-initialization evidence sufficient for current policy. The old report uses **historical matching**, even where its filename says v2. Old raw metrics remain descriptively usable, but the comparisons are **UNVERIFIED for corrected-v2**, not newly MATCHED.

Old pilots share source snapshot `c19e3646c415b643441b2a3b038af4a71aea734f439b423139b0d293287e43f4`, recorded dirty base `6a7a846`, rather than HEAD82c6461. Source comparison found exact committed bytes for203/212 recorded files at successor c85be91; the remaining9 are explicitly not fully reconstructed here. Current changes include trainer/calibration/checkpoint evidence, not only reporting. Retraining was justified by missing calibration/replay evidence and a newly executed trainer failure—not simply HEAD changing. Old MD/MS were never relabeled as using a new calibration, even though the newly calculated lambda happened to agree numerically.

All historical artifacts and failed attempts remain in place. New matched runs use source snapshot **`a30cb43ecc0dfebafc47d3a214c6c61803d03148f1ebc901b4d9bab1546ff3b9`** (HEAD plus the three-access execution fix). Exact source bytes and manifests are archived locally. Historical source limits are isolated in `historical_source_comparison.json` and inventory entries, not waived by the new report.

## Scientific interpretation and limits

At1800: **R0.671575253, MD0.667110088, MS0.654153227**. Both mean-only variants lose to R; symmetric is also worse than detached. Corrected pseudo-validation mean gap is nevertheless smaller: R0.796686, MD0.733997, MS0.625390 on the fixed geometry subset. **Moment agreement is not sufficient for retrieval improvement here.** No mainline promotion and no numerical-novelty claim.

Paired bootstrap verifies identical query IDs/order and gallery identity, then resamples10,963 queries10,000 times. This quantifies **queries of one seed/split**, not multi-seed robustness, class/split uncertainty or official-unseen generalization. Geometry diagnostics use fixed16/class subsets; efficiency is amortized runtime including scheduled probes, not isolated kernel timing. Full per-step sample identities are not logged; fixed-prefix replay and matching RNG at all scheduled checkpoints are the retained sequence evidence.

Historical source reconstruction remains limited as stated above. Resume-state correctness remains **unverified/not supported**; no run was resumed and no optimizer was silently reset. No covariance campaign, 5400-step extension, extra seed/lambda, official unseen, new dependencies, GPU rental, process kill, system change, commit or push was performed.

## Handoff

Local result directory: `outputs/alignment_verification_20260906_093823_82c6461/`.
Review archive: `review_bundle.tar.gz` with reports, calibration, matching evidence, resolved configs, raw histories/metrics, logs, patch and exact small source archive. Large backbone/checkpoint tensors are excluded; their original paths and SHA256s remain in inventory/config evidence. `review_bundle_files.json` binds bundled files by SHA256. **Local only—not published.**

No execution blocker remains. CPU-only exact report replay (new output name):

```bash
CUDA_VISIBLE_DEVICES="" direnv exec . uv run --frozen --no-sync python scripts/summarize_alignment.py \
  outputs/alignment_verification_20260906_093823_82c6461/fixed_attempt/pilot \
  --horizon 1800 \
  --output outputs/alignment_verification_20260906_093823_82c6461/reviewer_replay.md
```

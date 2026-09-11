# TU-Berlin MP-Q + sketch frozen-reference InfoNCE — authorized campaign

## Locked question and authorization

User requested implementing direction2 on TU-Berlin, explicitly chose **official220/30, one fixed ablation**, then **existing masked teacher → clean student**. Keep historical MP-Q baseline1189/warmup59/B32/seed42; fresh CLIP initialization, no baseline retraining. Official baseline was already inspected before this design: **not a blind holdout claim**. No coefficient/model/step selection on official results; only fixed final checkpoint clean+9masks. No QuickDraw arm, newseed, retry/resume, otherloss/readout/patch/prompt/augmentation changes.

Historical baseline root `outputs/coupled_benchmark_execution_20260910T164500Z/`, TU final1189 SHA `a39a0a6a6107566afa43af44410f25d40ab8b36d28d75ac6a4bf0c7901752460`; cleanfull `.4396614905633032`, maskedmacrofull `.32638461547718317`. Compare fixed1189 to fixed1189, primary cleanfull. Maskedmacro and five metrics/condition are secondary; no peak selection.

## Exact treatment

New arm `F2_MP_Q_SREF_OFFICIAL`, method `coupled_predictive_mp_sketch_ref_official_v1`:

```
L = historical_F2_MP_total + lambda_sketch_ref * L_sketch_ref
r = stopgrad(normalize(original_CLIP(existing_corrupted_sketch)))
q = existing clean student q = normalize(pooled_head(mean(H)))
L_sketch_ref = cross_entropy(normalize(r) @ normalize(q).T / .07, arange(B))
```

Teacher rows, student columns; one-way instanceNCE, no same-class negative filtering, no symmetric term. Original frozen CLIP already exists, no extra teacher copy. Original photo reference forward/batch preserved; explicit SREF adds one B32 frozen sketch forward, no additional student forward. Existing combined clean+masked student forward and all MP/CE/alignment/anchors unchanged. Teacher eval/no_grad; gradients go only to sketch student and pooled head through clean q. Predictor/photo/text prompts get no **direct SREF** gradient, though retain other losses. No true labels/targets/teacher latents enter student forward.

This is SeCo-inspired, not SeCo reproduction: teacher uses existing raster region deletion, not a new erase/crop/flip augmentation pipeline. Same `region_pair` seed4242+update and severity choice .25/.5/.75; test conditions remain fixed seeds101/202/303×threefractions. It is not masked student→clean teacher.

Loss API `lambda_sketch_ref=None` exact legacy keys/scalars/gradients/forward counts; explicit0 records raw SREF, keeps legacy total/gradients. Strict finite nonnegative coefficient; reject bool/overflow/NaN/Inf/negative/wrongarchitecture/QMP/SIG/PCE combination. Trainer presence of `--lambda-sketch-ref` selects newTU-onlyarm; absent preserves legacy TU/QuickDraw routes. No productiondefaultlambda. Explicit evaluator `--arm F2_MP_Q_SREF_OFFICIAL`, oldarm remains default. Original retrieval implementation/model unchanged; CLI binds new identities and emits `sketch_ref_identity.json` beside immutable summary.

## Train-only calibration — complete, no optimizer updates

Root `outputs/coupled_sketch_ref_diagnostic_20260911T042000Z/`.
Four initial realB32, both last student block12tensors/7,087,872coords and pooledhead2tensors/393,728coords:

`lambda = .1 * min_scope median_batch(||grad_full_F2_MP|| / ||grad_raw_SREF||)`.

| Scope | Candidate |
|---|---:|
| Last student block | .4935710836205779 |
| Pooled head (binding) | **.12384240801936452** |

Selected **λ=.12384240801936452**. Initial realized weightedSREF/base norm ~2.32–2.92% lastblock and9.34–11.96% pool. This is initialization median-ratio heuristic, not a perbatch10%cap, mAPoptimum, persistentgradientbalance, or effectiveAdamWratio.

Raw status `MEASURED_PENDING_REVIEW`, verifiedfalse retained. Four raw CPUFP32 perparam+flat gradient files, actualteacher/cleanq/logits, FP64norm/cosines, 128trace/mask records andtensorhashes. One combined studentforward/two originalvisual calls perbatch, noadditional oracleforward. Fullmodel/teacher/RNG unchanged, gradNone, source596files archive bound, noofficialimageevaluation.

Independent `independent_cpu/verify.py` + `receipt.json` PASS `CPU_RAW_CALIBRATION_AND_CONTRACTS_ONLY`: independently recomputed FP64norms/median/min, maxCEdelta2.514e−7≤2e−6; 17components/archive596files/cache/manifests/init; real4workerCPU128rows trace/mask/inputhashreplay,1152trainopens from prefetch,0testopens. This is not independentGPUgradient/encoderreplay. Independent tiny archived comparison uses diagnostic archive (not original baseline) and its standalone weighted-gradient example is algebra-only; **parent actual archivedbaseline47grad parity and production weighted-gradient oracle** cover those acceptance requirements. Original independent evidence unchanged.

Parent `outputs/coupled_sketch_ref_parent_review_20260911T030000Z/assemble_diagnostic.py` recomputes min independently and creates bound `diagnostic_verified.json` PASS scope `SKETCH_REF_INITIALIZATION_CALIBRATION_ONLY`. Raw status not rewritten. Independent prototype gate-script SHA names preserved unexecuted worker script, not final optimizer verifier; calibration acceptance does not certify that prototype.

## CPU and actual CUDA gates

Parent `outputs/coupled_sketch_ref_parent_cpu_20260911T041000Z/`:
- Six CPUcheck groups PASS; actual old archivedF2_MP keys/scalars/all47grad exact; explicitNone/0 and correct1vs2teacherforwards/inputbinding.
- Independent logsumexp formula, production weightedgradientlinearity, detachedteacher/ref-onlyrouting; invalidroutes and metadata.
- Tiny mockedactualtrainer2updates,47states safeactualmodel/optimizer/schedulerrestore; source `SYNTHETIC_UNVERIFIED`, notproductionprovenance.
- Diagnostic CPUselfcheck/evaluator CPUselfcheck/runner inert PASS; Ruff anddiffcheck PASS.

Actual CUDA root `outputs/coupled_sketch_ref_gate_20260911T043000Z/`:
- `run_smoke.py`: realtrainer2updates at selectedλ/production1189+59schedule/W&Bdisabled, COMPLETEexit0.
- First actualbatch baseline12.669638633728027, rawSREF3.4790847301483154, total13.100497245788574; diagnostic/production rawSREF andweightedtotal delta0.
- Exactbaselineinit, first64traces/masks and3LRrows; originalCLIPunchanged,179optimizerstates358finite/nonzero moments.
- Peakallocated12,481,166,848bytes/reserved13,061,062,656bytes.
- Existing evaluator `--train-probe --arm F2_MP_Q_SREF_OFFICIAL`: actualcachedCLIP32trainqueries/256trainphotos, clean+9masks; qfull/bypassexactfirst32, predictor0 duringprobe, stateunchanged, noofficialtestimages.
- Final `verify_saved.py` CPU-only PASS: actualsavedmodel+optimizer+schedulerrestoreexact; allinitialcheckpointentries exacthistoricalTU; source/archive components; firstbatchreceipt; savedprobe320rows independentFP64sort, APtol2e−5/P2002e−6, noexactrankingclaim.

Intermediate worker code preserved at `outputs/coupled_sketch_ref_parent_review_20260911T030000Z/worker_originals/`: caught unconditionalcombinedteacher breakinglegacy, missingmatrix diagnosticcontract, wrongresultarmconstant, rawreceiptselfhashrequirement/oldsmokeroute/duplicatedevaluator, corrected **before realGPUmeasurement**. `verify_saved_worker_unexecuted.py` preserves wrong final-campaign-oriented verifier; not executed/accepted. Parent replaced it with actual two-update gate verifier. No failedGPU/retrain to forcePASS.

## Execution controls

New `scripts/run_coupled_sketch_ref.py` inert without `--launch`; revieweddiagnostic and launchgate required, freshdirectchildoutputs≥24GiB, exactcomponentsets/artifacts/evidence checked beforedevice; actualSREFsmoke component hashes bind productioncode even iflaterdocs change fullsource. Sourcearchive captured at launch; freeze source/config/docs until finaleval finishes. Noautomaticretry/resume. Online W&B training, official retrieval savedlocal separately. Final raw `TRAIN_AND_OFFICIAL_EVAL_FINISHED_UNVERIFIED`, localfixedcomparison pendingindependentreview; notpromotion/certification.

Planned freshroot **`outputs/coupled_sketch_ref_execution_20260911T061500Z/`**. Adjacent launchreceipt/runtime/process authoritative; do not duplicate. Idle inhibit only, avoidmanualsleep/reboot. No dependencyupgrade, push or historicalartifactcleanup. Any later docs commit is not the diagnostic/training snapshot; use archivedsource andhashes.

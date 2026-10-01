# Repository Guidelines

## Agent-Specific Instructions

## CNN–Mamba Student branch — user-authorized implementation (2026-09-28)

**SketchLVM public inputs fresh60 + masking COMPLETE (2026-10-01):** [gap/masking report](docs/sketchlvm_public_inputs_gap_and_masking_2026-10-01.md). Trainroot `outputs/sketchlvm_tu_60epochs_public_inputs_clean_20260930/`, HEAD1019e40/54archivefiles, seed42B64/60epochs14460updates924000obs, train+cleanexit0/finish00:59:48Z, checkpointSHA5491c1aa...; cleanfull/AP200/P200/P100 .522722511/.585810748/.64786875/.667625 vsoldfull.516093792, +.662872pp; still−12.827749pp vs paper65.1, notexactreproduction/causeproof. W&Bv197ddsx, no independentonlineaudit here. User then explicitly selected NEW52.27checkpoint for masking; onezero-update CUDA eval `outputs/sketchlvm_public_inputs_masking_20261001/`, exit0/finish12:01:09Z; originalcachedgallery/clean reused, restoredstatehashmatchesoriginalclean, model/RNG/modes unchanged,0W&B/retries. Samepublicinputsraster25/50/75 meanRGB<.9 ceilfirstrowmajorink whitenallchannels beforeImageNetnorm; full .335551397/.157877453/.056603119, retention64.193/30.203/10.829%, all30classdeclineeachmask,75%2288/2400APdecline. Newmasked belowoldsaved by.280314/.707789/.178925pp; all9600old/newpixel/bitmaptraceSHAs equal, sources224square(noquerypadborder), hashcomparisonnotimagereplay. CPUreview `outputs/sketchlvm_public_inputs_masking_review_20261001/` parent7200savedrankAP200/P200/P100+120classmeans/18artifact/54archive/6manifestPASSscoped; fullAPsavedmeans/pairedONLY, noencoder/image/fullsortreplay. Worker `outputs/sketchlvm_clean_gap_analysis_20261001/` lower-median convention preserved, parentinterpolatedglobal+.229612pp+perclassoverlay recorded; means/WLT1461/939/0 unaffected. RawFINISHED_UNVERIFIED and originalFP64FAILSHA14c8ca... unchanged. Publicbasic sharedLN/tripletonly/evalsampledgallery !=verifiedpaperTUcategorycode; temp/pool/evaluatorcorrespondence unresolved, paper204489photos vslocal204070notprovenperformancecause. Noactivecampaign/autotrain/retry/newseed/search/replay/push/cleanup. Preserve sibling/env/historicalartifacts/user-editeddelegationwording; postreportcommitnottrainingsnapshot.

**Latest SketchLVM fresh60 CLEAN-only authorization (2026-09-30):** User explicitly chose “Rà khớp cấu hình gốc rồi train fresh60 clean-only”; [protocol](docs/sketchlvm_paper_clean_protocol_2026-09-30.md). One fresh seed42B64/Adam1e-5/60epochs14460updates924000obs, explicit `--paper-clean`, new `sketchlvm_paper_category_public_inputs_v2`; preserve old default/run/FP64FAIL. Source-supported input correction: PIL ImageOps.pad224square/defaultBICUBIC+black then Resize224square/defaultbilinear, ImageNet mean/std for BOTH visual branches; keep paper separate3prompts/LNs/84480param106tensors/.3coshinge+.5sk/posphotoCE/nativefrozen textscale. Public dataset code is Sketchy, publicbasic modelsharedLN/tripletonly; NOT exactauthorTUreproduction/guaranteed65.1. Onlyfinalofficial clean2400/27989, P100 fromtop200 plus full/AP200/P200, sbir4finalmetrics/progress100, epochlosseslocal; NO raster/intermediate test/selection/search/resume/retry/extraarmseed/push. Planned gate `outputs/sketchlvm_paper_clean_gate_20260930/` actual3CUDA64/64/prefix40+32/256TRAIN clean + wholemodel106states212moments/RNGrestore, CPUloss/Adam/publicinput/savedrank checks, **no newFP64fullsortcertification**; originalrawFAIL remains unchanged. Plannedroot **`outputs/sketchlvm_tu_60epochs_public_inputs_clean_20260930/`**, adjacentlaunch/executionreceipt/runtime/PIDs authoritative, inspectbeforelaunch/no duplicates; prelaunch note is NOT completion. **Freeze all nonignored source/config/docs during gates+production**, preserve sibling/env/historicalartifacts/user-edited delegation wording. Parent/subagent delegation remains originalprojectcx/gpt-5.6-luna/max. RawfinishFINISHED_UNVERIFIED, no exactpaper/metricguarantee/fullyverified claim.

**SketchLVM fresh60 + raster COMPLETE (2026-09-30):** [results](docs/sketchlvm_raster_results_2026-09-30.md). Root `outputs/sketchlvm_tu_60epochs_raster_20260930/`, trainingHEAD96f161e/51archivedfiles+userdirtyAGENTS (notcleanHEAD), checkpointSHA25f6da5...; seed42B64/60epochs14460updates924000obs, train/eval+reportexit0, finish15:06:34Z. Three independent prompts/LN per sketch/photo,84480trainable/106tensors, nativeCLS; cosinehinge.3+.5*(sketchCE+posphotoCE), frozen220text/nativeCLIPscale, **trainedphoto gallery**. Official2400/27989: clean full/AP200/P200 .516093792/.576044967/.641120833; rowmajor25/50/75% countink erasure full .338354534/.164955346/.058392373; retention65.5607/31.9623/11.3143%, all30classmeanfullAPdeclineeachmask. MaskRGBmean<.9 after224 beforeNorm,ceil(f*N),Nmean1153.016/no blanks, preserveothers; notsemanticparts/truepenwidth/average-shape proof. Gateactual3CUDA64/64/prefix40/restore106states212moments+loss/Adam/maskCPU PASSscoped but **FP64 gate rawFAIL retained** 1/128 fullAP2.26603e-5>2e-5. UserexplicitchoiceproceedwithFAIL boundadjacentlaunchreceipt; predecisionparentBLOCKEDfalse historical, neveroverride/tolraise/retry. FinalrawFINISHED_UNVERIFIED unchanged. CPUauto `outputs/sketchlvm_report_20260930/`, independent `outputs/sketchlvm_final_review_20260930/` all9600savedrankAP200/P200 max3.33e-16/all120classmeans/51archive+6manifests/pairedstatsPASS; fullAPsavedmeansONLY, masktraceconsistencyONLY(noimage-reopen/encoder/fullsort); parentactualfinalsafecheckpoint+all106finiteadap/212moments/sourceBeforeDocsSHA PASS. W&Bjk42jjtrfinished,readonly1row4scalars exact1e-12/0artifactdownloads/writes. No activeGPU/campaign; no autoretrain/replay/search/newseed/push/cleanup. Oneexploratoryseed/officialtestpreviouslyseen/notexactpaper/SOTA/geometrycausal/significance/fullcertification. Postreportcommitnottrainingsnapshot; preserveuserAGENTSdiff/sibling env/source/historicalartifacts.

**SketchLVM paper-category user-authorized fresh60 + raster evaluator (2026-09-30):** [protocol](docs/sketchlvm_raster_protocol_2026-09-30.md). Separate new model/trainer/evaluator, preserve existing runs/models/sibling environment. Frozen CLIP B32QuickGELU + separate sketch/photo three shallow prompts appended AFTER native CLS+49 positioned patches BEFORE native preLN, separate all visual LN affines; 84,480 trainable/106 tensors. Cosine hinge margin .3 + .5*(sketchCE+positivephotoCE), no negativeCE; frozen original220textprototypes/nativepretrainedlogit scale, trained photo branch gallery. Fresh seed42/B64/Adam1e-5/no scheduleclipaugmentation, TU220/30 60epochs/14460updates/924000obs, lastbatch40. Final-only clean+rowmajor25/50/75% ceil(f*N) meanRGB<.9 pixel erasure after224resizecrop before normalization; preserve others, record counts/mass/hashes. No unseen labels/text at encoder, no intermediate test/selection/resume/retry/newseed/search/push. Gate `outputs/sketchlvm_gate_20260930/` actual3CUDA/train-only32/256 fourconditions + CPU audits; inspect receipts before launch, no prelaunch completion claim. Planned production **`outputs/sketchlvm_tu_60epochs_raster_20260930/`**, adjacentreceipt/runtime/PIDs authoritative/no duplicates. Freeze all nonignored source/config/docs during gates+production; final sbir three clean scalars only, epoch losses local. Raw finish UNVERIFIED until reviewed; previously observed official test, not exact reproduction/geometry/significance proof. Preserve user-dirty historical AGENTS line. Delegation follows originalproject cx/gpt-5.6-luna/max; no model fallback.

**Official retrieval replay user-authorized + COMPLETE:** User explicitly chose fullofficial retrieval ONLY, no centroid/fewshot/newtrain. [report](docs/student_official_retrieval_2026-09-28.md). Fresh `outputs/student_official_replay_20260928/`, adjacentreceipt PID5370historicalnowfinished, evaluator ignored `outputs/student_official_replay_gate_20260928/evaluate.py`, evalsourceHEADeb611aa/archive18ce611final10checkpoint3809e4... sharedCLIPe6d1bd... . All2400query/27989gallery/30classes, no inferenceclasslabels/textbank/backward/selection/W&B. All3features+perqueryAP/AP200/top200arrays exactold,6metricsdelta0, state/RNG unchanged. Studentfull/AP200/P200 .272204838/.216423019/.291170826; Teacher .566515100/.567195003/.622287491, +.294310262fullAP,2016/2400queries(84%) and30/30classmeanAPteacherwins. StandardTRAIN220→TEST30 categories, NOT necessarilyunseenCLIPpretraining orblindholdout; NOT geometry/causal/latency/significanceproof. IndependentCPU4800FP64fullsort `outputs/student_official_replay_review_20260928/` raw **CPU_AUDIT_RAW_FAIL retained** (Student10AP200,Teacher2fullAP+12AP200 exceed2e-5tol; aggregateswithin2e-6/P200countsallrowexact). ParentallCSV/classmeans/savedrankAP200/P200/max3.33e-16 +actualSHA PASSscopedwithFAILretained. OriginaltrainingFINISHED_UNVERIFIED/freshOFFICIAL_RETRIEVAL_FINISHED_UNVERIFIED unchanged, no retry/tolraise/newexperiment/push. No productionactive; preservehistoricalevidence/sibling/sourceenv/userdirtyAGENTSline.

**Seen-training diagnostic COMPLETE (scoped, no encoder replay):** [report](docs/student_seen_semantics_2026-09-28.md). GPUencode root `outputs/student_semantics_diagnostic_20260928/`, commit4230a2a, archive18ce611 checkpointSHA3809e4.../config7e49c8..., state/RNG exact/no grads/no newofficialtestimages. CentroidStudent .52694805 vsTeacher .79675325; gallery1756fullAP .27872224/.58144794. Allseenencodertrain data, NOT geometry superiority/generalizationproof. IndependentCPU all6160FP64sortrows/centroid/reconstructedRandom42split/hashPASS; audit `outputs/student_semantics_review_20260928/` raw **FAIL preserved** because perqueryPearson/overlapcolumns swapped. Onlywriter signature fixed + distinctmappingCPUcheck; no encoder rerun. Correctedoverlay `outputs/student_semantics_corrected_20260928/per_query.csv` +receipt **PASS_CORRECTED_AGREEMENT_OVERLAY_ONLY**, independent3080agreementrows maxPearson2.554e-7/overlap2.861e-8, allotherfields/rawunchanged. OriginaltrainingFINISHED_UNVERIFIED stays; no push/further experiment. Standaloneanalysis/no testsuite; jaxtypinghints no numericalchange. Preserve unrelatedAGENTSuserline and siblingsource/environment.


**Latest authorized frozen seen-train diagnostic + jaxtyping:** User requests readable standalone analysis (NOpytest/testsuite) and shape type hints, approved all15400seenTRAIN sketches + up to8photo/class1756gallery (actualmanifest squirrel only4; user chose no duplicate padding). Latestdeepcheckpoint18ce611 final10 Studentfull .2722048384 teacher .5665151002, no encoderupdates/newofficialtest/classifiertraining/W&B. Deterministicseed42 perclass56centroid/14query (3080query), normalized nearestsketchcentroid classification vs samequery→sharedfrozenphoto retrieval; cosine/score/rankagreement perquery/class. Allimages previously trained encoder, **NOT independentholdout or geometryqualityproof**; AP usesactual4or8positives, P200 cap.04 exceptsquirrel.02. New `src/spica/diagnose_student_semantics.py`, archivedmodelsource and checkpoint/config/CLIP SHA strictsafe weights_onlyrestore, trainimageallowlist, no grads/state/RNGchange. Jaxtyping0.3.11+wadler-lindig0.1.7 installed ONLYStudent .venv, existinglockpackages unchanged; readabletensor-shape hints no runtimechecker/numericalchanges. Planned root **`outputs/student_semantics_diagnostic_20260928/`**; fresh only/PIDs/GPU inspect; freeze source/config/docs duringencoding, preserve priorruns/sibling env/dirtyAGENTSline. SavedfeatureCPUreview scope, not independentencoderreplay. No retries/newtraining/seeds/push. Historical cx/gpt-6-luna line is preexistinguserdiff; parent delegation still governed by originalprojectcx/gpt-5.6-luna/max.


**Latest deep-visual-prompt authorization:** after whitepatchmask λ1.2 completed rawUNVERIFIED Studentfull .2688652228 vs priorunmasked .2662939949; teacherfull .5693351723 vs .5672210936. User requests soft prompts **in all12 visual layers**. Fresh10epochs/4820update TU220/30 seed42B32, keep teacher LoRAQ/V+exactwhite keymask all12, frozenphoto, Student model/loss/KDλ1.2/schedule; change only teacher prompting: replace4shallow tokens propagated acrosslayers with12 independent sets of4 tokens (learned768D normal.02, normalized using original nativepreLN, injected afterCLS before original49patches at eachlayer; discarded aftereachlayer; nativeCLS readout, positions unchanged). NOT SeCo text-guided prompts, no text loss. New CLI `--deep-visual-prompt` requires --lora-teacher --visual-prompt --mask-white-patches --lambda-kd1.2; legacy numerical behavior unchanged. Teacher331776trainable (294912LoRA+36864prompts), Student2251782. Gate **`outputs/student_deepvpt_gate_20260928/`**, planned production **`outputs/student_deepvpt_teacher_10epochs_20260928/`**; CPU real architecture/CUDA3updates+all12mask/gradient/optimizerrestore/FP64train-onlyretrieval required. Adjacentreceipt/runtime/PIDs authoritative/no duplicates. Freeze source/docs/config during gates+production. No selection/resume/retries/other arms/seeds/push. Previously seen official test, rawfinish UNVERIFIED; preserve old runs/sibling env/files.

**Latest user-authorized whitepatch mask ablation:** λ1.2 joint10 finished rawUNVERIFIED Studentfull .2662939949 teacher .5672210936. User now requested mask completely white patches all layers. Fresh10/4820/seed42B32, **retain λKD1.2, LoRA+4VPT and all other losses/schedules/photo/Student**. New `--mask-white-patches` teacher-only train+eval: RGB raw after224resizecrop, patch32 blank iff EVERY channel/pixel exactly1, no tolerance/morphology/no sorting; original positions preserved, CLS+4prompts alwaysvalid even allwhite. Additive−inf key mask B*heads,L,L to all12nativeattentionlayers; padded query outputs never read by validkeys; nativeCLS/LN/proj preserved. Dense mask no speedup promise. Legacy false exactprevious; reject oldKDcalibration with mask because changedforward. Gate **`outputs/student_whitepatch_gate_20260928/`**, plannedproduction **`outputs/student_whitepatch_teacher_10epochs_20260928/`** inspectruntime+receipts+PIDs/no duplicate/retry/resume/push/extraarm. Must gate12maskpaths, realtrimmed-tokenparity/blankcase/KDisolation/frozenstate+jointrestore before launch. Freeze source/config/docs during gates+production. Officialtestpreviouslyseen; no bestselection, sixsbirfinalscalars only; preserve historical/sibling env/artifacts.

**Latest user authorization fixed λKD=1.2:** Fresh10epochs/4820updates, retain LoRA+4VPT/Student/photoCLIP/data/seed42/B32/alllosses and finalonly sbir evaluation, change λ ONLY via new `--lambda-kd 1.2` (notcalibrated, mutuallyexclusive --kd-calibration). Prior calibratedλ .163140544 run finished rawUNVERIFIED Studentfull .259736746, teacher .567221094; teacher finaladapters/features/gallery byteexact priorλ1, so no teacher improvement claim. New gate **`outputs/student_lora_vpt_kd12_gate_20260928/`**, planned production **`outputs/student_lora_vpt_kd12_teacher_10epochs_20260928/`**, runtime/receipt/PIDs authoritative/no duplicate/retry/resume/extraarms/push. Freeze source/config/docs while gates/productionactive. User also requested SeCo/SketchLVM research and proposed whitepatch→PAD; analysis only in ignored `outputs/clip_sketch_research_20260928/`, **NO patch sorting/masking in this λ1.2 run**. Keep original positions with patchtokens; reordering alone does not focus CLSattention; real blankkeymask must affect allblocks, no zero/PADembedding shortcut or speedup claim. Any patchmask training requires separate explicit decision/gates. Preserve all prior runs/siblingfiles/env.

**Latest authorization KD calibration then fresh10:** User selected measure4trainB32 at fixed initialization, zero optimizerupdates/no officialimages, set λKD so median4(λ||gKD||/||gretrieval||)=.1 over all Student parameters; specifically λ=.1/median4(KDnorm/retrievalnorm), not median inverse. Retain LoRA+4shallowVPT teacher, same Student/seed42/data/4820updates/241warmup; teacher learns frombatch1 as before. λ fixed after scopedCPUreview, not optimum/perbatchcap/Adamratio; no lambda search or officialtest selection. New `scripts/calibrate_student_kd.py`, trainer explicit `--kd-calibration REVIEWED_RECEIPT`; absent receipt exact oldλ1. Planned diagnostic **`outputs/student_lora_vpt_kd_diagnostic_20260928/`**, gate **`outputs/student_lora_vpt_kd_gate_20260928/`**, production **`outputs/student_lora_vpt_kd_teacher_10epochs_20260928/`**. Validate raw gradients/selection/source and actual weighted trainer CUDA/checkpoint before launch; inspect runtime+receipts+PIDs/no duplicates. Freeze source/config/docs once measurement starts through production. Prior VPTλ1 finished rawUNVERIFIED Studentfull .2637826476/teacherfull .5672210936, so weaker than noVPTλ1 .2669895063/.5733337582; this experiment changes λ ONLY, no removal of VPT. Preserve all historical evidence/sibling env/source, no retry/resume/newseed/extraarm/push. Future rawfinish UNVERIFIED until reviewed.

**Latest VPT authorization:** User chose **keep λKD=1** (no calibration) to isolate adding teacher visual prompts. Fresh10epochs/4820steps, same LoRAQ/V r8alpha8/LR1e-4, StudentLR3e-4, B32seed42, photoCLIPfrozen, simultaneous teacher/Student frombatch1. New `--lora-teacher --visual-prompt`: 4 shallow learned768D tokens afterCLS/afteroriginalpositions/beforepreLN, normalstd.02, no new position embeddings, no deep reinsertion; native CLSpooling. PromptLR1e-4/wd.01 same teacher group, total teacher297984trainable (294912LoRA+3072prompt), λKD1unchanged. No-prompt mode remains exact historical teacher. Prior joint10 atHEAD136b917 finished rawUNVERIFIED: Studentfull .2669895063, teacherfull .5733337582; same frozen gallery bytes as Student-only10. Do not call VPT gain before result or LoRA gain over unmeasured originalCLIP. Planned root **`outputs/student_lora_vpt_teacher_10epochs_20260928/`**, fresh gate **`outputs/student_lora_vpt_gate_20260928/`**, inspect receipts/runtime/PIDs/no duplicates. FreshactualCUDAgate required, preserve all old gates/artifacts; finalonly Student+teacher in sbir, no selection/resume/retry/extra arms/push. Freeze source/config/docs whileactive. Official test previously seen, notblindholdout. No lambda change authorized for this run.

**Latest authorization: fresh joint teacher/Student10epochs**, user approved fixed λKD=1 (not calibrated), CLIP sketch teacher only LoRA Q/V all12 blocks r8/alpha8/dropout0/LR1e-4; Student unchanged LR3e-4, photoCLIP frozen. Teacher retrieval and Student retrieval+cosineKD with detached teacher target, both learn from first batch, no separate teacher warm-up. Shared 5%LRwarmup/cosine over4820steps, AdamWwd.01, clip1 independently Student/teacher, seed42/B32; same geometry, no augmentation/text/EMA. New `--lora-teacher` mode preserves old Student-only100 default. Fresh init (not resume100), finalonly both readouts, Student primary + teacher diagnostic, sbir six separated scalars; no selection/retry/search/extra arms. Old100 completed rawUNVERIFIED fullmAP .2607088325; old10 .2553224267; MPQ .4396614906. Previously seen official test exploratory. Planned root **`outputs/student_lora_teacher_10epochs_20260928/`**, gate `outputs/student_lora_gate_20260928/`, inspect runtime+receipt+PIDs before launch; freeze source/config/docs during gates/production. Teacher adapter checkpoint requires pinned original CLIP and normalization from archived config/code; no base pretrained upload. Gates required actual joint trainer CUDA, frozen base/teacher gradient isolation + adapter/optimizer restore. Preserve historical artifacts/sibling source/env; no automatic push. Subagents must use cx/gpt-5.6-luna/max, no silent fallback.

**New authorization after the completed 10-epoch run:** Fresh **100 epochs from initialization** on official TU220/30; keep model/seed42/B32/loss/data, replan cosine/5% warmup for 48,200 updates (482/epoch,1,540,000 observations). User explicitly chose fresh100, not resume90. 10-epoch source `a09f097`, run `outputs/student_tu_10epochs_20260928_v2/` rawFINISHED_UNVERIFIED, final clean fullmAP .2553224267 vs fixed MP-Q .4396614906; seen official result prompted this follow-up, **not blind holdout**. New `src/spica/train_student.py` `EPOCHS=100`, W&B `sbir`, final three clean metrics only, no test/selection midrun. Planned root `outputs/student_tu_100epochs_20260928/`, inspect runtime+adjacent launch receipt/PIDs, no duplicates. Preserve all previous failed/successful outputs, no resume/retries/new arm/seed, no push unless separately requested. **Freeze all nonignored source/config/docs throughout active production**. Previous 10-epoch status below is historical and does not describe the new 100-epoch run.

This worktree is `feature/cnn-mamba-student`, branched directly from `main` at `b21154e`, NOT from `experiments`. The user explicitly authorized implementation here, superseding the historical read-only restriction and historical Mo-vMF implementation order below for this branch only. Never edit/check out the sibling `spica/` worktree: its existing campaign/source freeze and unrelated dirty files must remain intact.

- Student: `src/spica/models/sketch_student.py`, 2,251,782 randomly initialized parameters. CNN, two shared-weight four-direction Mamba-2 hybrid blocks, residual gates, foreground pooling, normalized512-D output. No custom validators or tests inside the model, no custom production selective-scan kernel. This is not an exact VMamba implementation.
- User approved official `mamba-ssm` in an isolated `.venv`; pinned Git source and compatible `apache-tvm-ffi==0.1.9` in `pyproject.toml`/`uv.lock`. Existing main dependency versions, including Torch2.13/CUDA13, unchanged. Do not alter the sibling environment. No subagent used for this implementation; do not misreport it as cx/gpt-6-luna work.
- `src/spica/train_student.py`: 10 full officialTU220/30 epochs, seed42/B32/drop_last=False, frozen CLIP photo targets, multi-positive contrastive only; no text KD, sketch-teacher loss, masking or consistency. 15400sketches/482batches per epoch →4820updates/154000observations. Final clean evaluation only, no validation/checkpoint selection/retry/resume.
- User-selected primary control: official MP-Q1189 clean fullmAP `.4396614905633032`. Supplied summary SHA `afb927f7142aafe98bf57c742a034d8478bdb2586880a53f6a3c5c15b2dfba42`; all six train/test manifest/class-map SHAs checked against control. Different photo branch, loss package and budget: not a matched backbone-only ablation, not significance, not a blind holdout.
- Minimal standalone check `scripts/check_student.py` is outside the model, not a test suite. CPU full224 forward/backward/scan layout/loss/weights round-trip passed using FP64 reference SSD with stable chunk carry; initial upstream reference tail/overflow failures retained under `outputs/implementation_20260928/`. This does NOT certify production CUDA kernels. No pretrained/data encoding or optimizer updates in this check.
- **User authorized train10epochs + final test + W&B project `sbir` (2026-09-28)**. Sibling triplet campaign finished09:37UTC; no concurrent GPU work. Synthetic actual CUDA forward/backward/state roundtrip PASS; initial Triton `/sbin/ldconfig` failure on NixOS preserved, solved with `TRITON_LIBCUDA_PATH=/run/opengl-driver/lib` (no dependency/model change). Fresh real-train smoke/restore evidence at `outputs/student_gate_20260928/` required before launch. Initial production `outputs/student_tu_10epochs_20260928/` failed at CLIP loading before optimizer/W&B/updates: trainer `.resolve()` stripped `.safetensors` cache symlink suffix; old external smoke did not resolve and missed the bug. Failure preserved. **User explicitly authorized fix + fresh launch**, new planned root **`outputs/student_tu_10epochs_20260928_v2/`**. Preserve suffix with expanduser/absolute, no unsafe pickle fallback. Gate must exercise actual trainer startup. Adjacent launch receipt + runtime + live PIDs authoritative; inspect before launch, no duplicates/further retries. All source/docs frozen during production; final clean three scalars only in `sbir`, no epoch loss curves. Local runtime updates step1/every50/epoch. Raw finish stays UNVERIFIED, not automatic promotion. No extra seed/arm/search/push.

Treat this file as the durable record of repository conventions, current architectural decisions, and implementation order. Sections below describe the historical main branch unless explicitly retained above.

## Primary Research Protocol and Architecture

### Primary protocol: standard ZS-SBIR

The primary SPICA result must follow standard zero-shot sketch-based image retrieval:

- Train on seen classes and evaluate on disjoint unseen classes.
- At inference, the model receives only a query sketch, a photo gallery, and learned model parameters.
- Rank gallery photos from sketch-conditioned scores. Ground-truth query labels are evaluator-only metadata and must never enter the retrieval model.
- The primary method must not require the unseen-class text bank at inference. This keeps comparisons with SketchLVM/CLIP-AT, SpLIP, SeCo-SBIR, and visual ZS-SBIR baselines protocol-compatible.
- Learned class-agnostic prompts are allowed at inference because they are model parameters rather than query-label information.
- Match dataset split, gallery construction, and metric definitions (`mAP@all`, `mAP@200`, `P@K`) before comparing reported numbers across papers.

### Primary architecture: sketch-conditioned Mo-vMF retrieval

Keep conditional mixture-of-von-Mises-Fisher prediction on the frozen CLIP photo hypersphere as the primary research contribution. For a sketch `s`, the model predicts:

```text
{pi_k(s), mu_k(s), kappa_k(s)} for k = 1, ..., K
```

with mixture weights summing to one, unit component directions, and positive concentrations. Frozen or consistently prompted CLIP photo embeddings `z_p` are unit vectors. Rank a gallery photo with mixture log-likelihood:

```text
score(s, p) = log sum_k [
    pi_k(s) C_D(kappa_k(s)) exp(kappa_k(s) mu_k(s)^T z_p)
]
```

The standard inference path is therefore:

```text
query sketch
-> visual/context encoder
-> sketch-conditioned Mo-vMF head
-> mixture likelihood against frozen photo embeddings
-> gallery ranking
```

This path must not consume true class text or a selected unseen-class prompt. Text may be used during training as semantic supervision, but the trained primary model must be executable from the sketch alone.

### Role of text

For the primary protocol, valid text uses include:

- Seen-class CLIP text embeddings as training-only semantic teachers or auxiliary classification targets.
- Distillation from seen-class text into a sketch-conditioned latent that no longer needs text at inference.
- Class-agnostic learned textual or visual prompt tokens shared across queries.
- Training-only regularization of global semantic components or mixture parameters.

Do not make ground-truth unseen class text part of standard inference. If an experiment uses all unseen class names to classify a sketch and condition retrieval, label it explicitly as **unseen-vocabulary-assisted ZS-SBIR** and report it separately from the primary standard protocol.

### Status of text-fusion diagnostics

The existing hard, soft-posterior, and oracle text-fusion experiments are diagnostics rather than the primary architecture:

- Oracle text uses the true query category and is a label-leaking upper bound, never a valid primary result.
- Hard predicted text requires an unseen-class vocabulary and reduces category-level retrieval largely to zero-shot classification.
- Soft-posterior fusion forms one deterministic mean text vector and therefore collapses distinct semantic hypotheses; it is not a mixture model.
- On Sketchy 104/21 with frozen OpenAI CLIP ViT-B/32 QuickGELU, observed exploratory test results were: sketch-only `mAP=0.247028`, hard predicted text `mAP=0.698833`, and oracle text `mAP=0.919814`.
- The soft-posterior sweep converged to hard argmax as temperature approached zero (`mAP=0.697817` at `temperature=0.001`, `alpha=1`). Increasing temperature reduced mAP, and every tested curve preferred `alpha=1`. Static global sketch/text averaging therefore did not improve the hard predicted-text baseline.
- Per-query branch-complementarity diagnostic (Sketchy 104/21, same setup): text classification accuracy 0.727036. On correct-text queries (9229) sketch mAP=0.291116 vs text mAP=0.924725; on incorrect-text queries (3465) sketch mAP=0.129601 vs text mAP=0.097170. Sketch rescue rate on incorrect queries was 0.594805 with mean AP gain 0.032431. Oracle per-query routing (max of the two branch APs) reached mAP=0.714054, a maximum possible gain of 0.015221 over predicted text.
- Score-margin bucketing (quartiles of top-1/top-2 class-score margin): text accuracy is monotone in margin (0.387/0.620/0.909/0.992 from low to high) and text mAP beats sketch mAP in every bucket (0.4076 vs 0.1421 in the lowest-margin quartile; 0.9389 vs 0.4164 in the highest). A sketch-route rule breaks even only when it isolates failed predictions at roughly 95% precision; no margin bucket reaches that, and entropy adds no new information because it is monotone in margin through softmax.

These findings reject deterministic global soft averaging for the primary path. They do not reject conditional Mo-vMF: averaging class text embeddings produces one direction, whereas Mo-vMF retains separate photo-space components and scores them with mixture likelihood. Do not conflate uncertainty over class names with one-to-many variation among valid photos.

### Agreement Gate status

Agreement Gate between the sketch-only branch and the predicted-text branch is **rejected** based on the diagnostics above. Complementarity is real but too small and too diluted: the expected gain on failed predictions (+0.032 AP) is dominated by the cost of mis-routing successful ones (-0.634 AP), and the per-query oracle ceiling of +0.0152 mAP leaves no room for a practical router. Do not build a cross-attention or learned gate for this branch pair.

If gating is reconsidered later, it must satisfy all of:

- Its inference inputs are available from the sketch or from class-agnostic learned parameters (a gate consuming the unseen text bank belongs to the separately named vocabulary-assisted protocol).
- A per-query diagnostic shows the intended branches are complementary in the relevant operating region, not only in aggregate.
- The expected gain justifies the added model complexity relative to the conditional Mo-vMF head.

### Implementation order from the current state

1. Preserve the existing sketch-only standard evaluation as the protocol anchor.
2. Treat the text-fusion diagnostics (soft/hard/oracle, branch complementarity, margin buckets) as closed; do not extend them unless a new signal appears.
3. Agreement Gate is rejected; revisit only under the conditions listed above.
4. Stabilize the training dataset, transforms, and positive/negative sampling contracts.
5. Implement a sketch-only conditional Mo-vMF head and mixture-likelihood retrieval before adding text supervision.
6. Add seen-class text as training-only auxiliary supervision and verify that standard inference remains sketch-only.
7. Evaluate category-level ZS-SBIR, GZS-SBIR, and a fine-grained or instance-sensitive protocol so that one-to-many modeling is tested meaningfully.
8. Treat unseen-vocabulary-assisted fusion/gating as an optional secondary experiment.
9. Compare against deterministic point prediction and relevant probabilistic embedding baselines such as PCME/PCME++; do not claim first use without a fresh literature verification.

## Data Architecture

Keep the data package compact for the current project scope:

```text
src/spica/data/
├── __init__.py
├── manifest.py
├── datasets.py
└── transforms.py
```

- `manifest.py` reads image manifests and class maps into typed records. It does not open images or depend on PyTorch datasets.
- `datasets.py` contains both `RetrievalEvalDataset` and `RetrievalTrainDataset`. Do not create separate `eval_dataset.py` or `train_dataset.py` files unless this module becomes substantially large.
- `transforms.py` contains sketch/photo preprocessing for training and evaluation. Dataset classes receive transforms through their constructors rather than constructing transforms internally.
- Use ZSE-SBIR's conceptual separation between stochastic training samples and deterministic query/gallery evaluation, but do not copy its implementation structure.
- Keep the YAML files and their manifests as the source of truth for dataset protocols; do not hard-code unseen classes in Python.

Follow this implementation order:

1. Implement and verify `manifest.py`.
2. Add `RetrievalEvalDataset` to `datasets.py`.
3. Add minimal evaluation transforms to `transforms.py`.
4. Verify separate sketch-query and photo-gallery datasets for all four protocols.
5. Add `RetrievalTrainDataset` with class-aware positive and negative sampling.
6. Add the DataLoader or data-builder layer only after the dataset contracts are stable.

## Project Structure & Module Organization

Spica is a Python 3.12 project using a `src` layout. Put reusable package code in `src/spica/`; `src/spica/__init__.py` defines the package boundary. Dataset-specific utilities currently live under `data/Sketchy-std/`, including argument parsing in `args/`. Large local datasets belong in `datasets/`, which is intentionally ignored by Git. Store generated models as `.pt`, `.pth`, or `.ckpt` files and generated artifacts under `outputs/`; these are also ignored. Add automated tests under `tests/`, mirroring package paths where practical (for example, `tests/test_dataset.py`).

## Build, Test, and Development Commands

- `uv sync --dev` creates or updates the environment from `pyproject.toml` and `uv.lock`.
- `uv run ruff check .` runs the configured Python linter.
- `uv run ruff format .` formats Python files consistently.
- `uv run --with pytest pytest` runs tests discovered under `tests/`; pytest is configured but is not yet a persistent development dependency.
- `uv build` creates source and wheel distributions using `uv_build`.

There is no application entry point yet. Run an individual module or script with `uv run python path/to/script.py` and pass dataset paths explicitly rather than relying on machine-specific defaults.

## Coding Style & Naming Conventions

Use four spaces for indentation and follow Ruff's default formatting. Prefer type hints for public functions and concise docstrings for non-obvious behavior. Use `snake_case` for modules, functions, variables, and CLI flags; use `PascalCase` for classes such as `Sketchy`; use `UPPER_SNAKE_CASE` for constants. Use `pathlib.Path` for filesystem operations and keep reusable logic out of command-line parsing modules.

## Testing Guidelines

Write pytest tests named `test_*.py` with functions named `test_*`. Cover dataset splitting, path discovery, transforms, and returned tuple shapes. Use temporary directories and small generated images instead of checking datasets or model checkpoints into Git. No coverage threshold is currently enforced; new behavior and bug fixes should include focused regression tests.

## Commit & Pull Request Guidelines

The repository has no commit history yet, so no established convention exists. Use short, imperative subjects such as `Add Sketchy dataset split tests`, and keep each commit focused. Pull requests should explain the change, list validation commands, link relevant issues, and call out data or configuration assumptions. Include screenshots or sample outputs only when visual behavior changes; never commit credentials, local dataset paths, generated outputs, or large model files.

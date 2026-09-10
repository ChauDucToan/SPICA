# MP-Q — official TU-Berlin và QuickDraw campaign

## User authorization / locked protocol

User yêu cầu train + thống kê TU-Berlin/QuickDraw, chọn qua ba câu hỏi:
1. **F2_MP, inference q (MP-Q)**, không PCE/QMP/TQMP/SIG.
2. **Official TU220/30 và QuickDraw80/30**, unseen chỉ scoring sau checkpoint cuối; không chọn checkpoint/λ/horizon theo unseen.
3. **Matched observation exposure≈2.47** thay vì3600updates/tập.

|Dataset|Seen sketch/photo|Unseen query/gallery|Updates|Warmup|Actual observation passes|
|---|---:|---:|---:|---:|---:|
|TU-Berlin220/30|15400 /176081|2400 /27989|1189|59|2.4706493506|
|QuickDraw80/30|236080 /149428|92291 /54151|18229|911|2.4708912233|

Seed42, B32, `drop_last=True`, shuffle generator42,4persistentworkers + same legacy `_worker_seed`, AdamW groups unchanged, warmup `round(.05*horizon)` then same cosine form. Passes=updates×32/train sketches, not exact unique visits; clean+masked views do not double epoch count. Same init CLIP ViT-B/32 quickgelu cache SHA `e6d1bd7789aa45192b3bf90570a789b478bae1b74ebcce7eddd908e83a2b7c31`; **each dataset fresh model, no Sketchy checkpoint transfer**.

Method `coupled_predictive_mp_official_v1`, arm `F2_MP_Q_OFFICIAL`. **Main loss MP(μ_I), not MP(q)**: historical F2_MP full total, CE_i1/CE_t.25/pairedqrank.25/qCE.25/align.05/anchors.5, MPtau.07, noSIG. Full seen photo pool,1positive+1negative/sketch,≤64unique live photos, average-per-positive MP. Train text bank220/80classes only; inference q bypass predictor with no labels/text targets supplied to query forward.

No cross-dataset raw mAP comparison interpreted as intrinsic superiority; dataset/gallery/split difficulties differ. Official unseen means excluded from this fine-tuning, not proven absent from CLIP pretraining. No paper exact-reproduction/metric-equivalence claim.

## Data trust boundary

`src/spica/data/coupled_benchmark.py` separate adapter. Existing Sketchy trainer/model/loss/data/evaluator files untouched. New adapter binds exact configs, six manifest/class-map byte hashes/dataset, actualcounts/classcoverage, all referenced paths exist, duplicate/path escape/cross-split path overlap checks.

**Both official splits relabel train/test numeric IDs starting0**. Class disjointness is checked by actual class names and paths, not numeric IDs. No invented Sketchy pairing manifest. Root `outputs/coupled_benchmark_data_audit_20260910/`:metadata/path-stat audit +bounded training-image decode and8synthetic rejection checks. Not full dataset pixel-SHA/decode certification; official images are not opened in these gates.

## Implementation / artifacts

- `src/spica/train_coupled_benchmark.py`: standalone locked dataset route; reuses historical model/loss/trainer helpers. Only checkpoint0 andfinal, no best aliases or pseudo probes, no official image loaders. Trainhistory0/every10/final(nonmultiple10), LR everyupdate,32observation/mask records perupdate, fixed source+frozen guards.
- `src/spica/evaluation/coupled_benchmark.py`, `scripts/evaluate_coupled_benchmark.py`: COMPLETE final-only checkpoint/source/config/data gates, safe scoped `weights_only=True`, q-only, one gallery encode, clean+fixed9deletions(.25/.5/.75×101/202/303), chunk256full stable sorting. No true test labels in query forward.
- Full-gallery mAP primary; P200 + all three AP200 denominators auxiliary. Querymacro, maskedmacro9conditions, fractionmacros. FP32 query/gallery `.npy`, compact AP/P200/top200int32 `.npz`, mask metadata JSONL/condition, hashes and compact JSON summary. Avoid giant top200 JSON/fullNxG score matrix. QuickDraw ~50billion query-gallery score pairs across10views; evaluation cost substantial, not certified by tiny timing.
- `scripts/run_coupled_benchmarks.py`: inert absent`--launch`, freshroot≥24GiB, fixedorder **TU train→TU final evaluation→QuickDraw train→QuickDraw final evaluation**, online training W&B, local statistics. Stops on any failure, no retry/resume/selection. Before each official evaluation, checks actual final checkpoint and initial64traces/masks/3LRrows against respective smoke. W&B training run contains training data, final official statistics in separate local evaluator artifacts/results, not retroactive old Sketchy runs.

## Gates complete, not production results

`outputs/coupled_benchmark_parent_cpu_20260910T162000Z/`: final safe tiny CPU trainer and evaluator checks, inert CLI. ArchivedF2MP scalar and all47active gradients exact; actual model/optimizer restore, mocked2updates64traces/masks, schedule. Tiny source intentionally `SYNTHETIC_UNVERIFIED`; not real source identity. Original worker CPU checker had an executed unsafe `weights_only=False` on its own synthetic checkpoint and overstated gradient/restore checks. Original script retained at `outputs/coupled_benchmark_independent_cpu_20260910/check_coupled_benchmark_cpu.original.py`; replaced safe checks before actualGPU. Current checker has fixed ignored output root: do not rerun into it and overwrite receipts; parent captured accepted final stdout separately. Earlier safe checker receipt was refreshed by its fixed-output rerun, not claimed immutable original independent evidence.

Real root **`outputs/coupled_benchmark_gate_20260910T162500Z/`**:
- Each dataset actual CUDA2updates, **same production horizon/warmup**, W&B disabled. MainMPμI/noofficialimages.
- Peakallocated TU12,481,166,336B /Q8,481,327,104B; reserved13,061,062,656 /8,975,810,560B. No total production peak guarantee.
- Checkpoint2 SHA TU `dbf6333675f9e8972bfd20bdbb0c785d6fed9c8c2e2ef898e9246c16944e71f7`, Q `05d6d55868f71e7c6076bae1690f83b740badc3d45b7a18b2465c3a9dca6929e`.
- Each train-only probe32classes/32queries/256gallery, guaranteed positives,10conditions, qfull/bypass exact, predictor0,stateunchanged. Probe elapsed1.92s/2.69s describes tiny fixture, not fulltest latency.
- `verify_saved.py` / `verify_saved.json`: CPU actual realmodel+optimizer restore179states358finite/nonzero moments/dataset; reconstructed64train traces/masks with test-image-open guard;183common initialstate tensors exact acrossdatasets (class-specific T0 exempted). Savedprobe640rows independently FP64fullsorted against256gallery, all5metrics/AParrays and savedranking relevance PASS at predeclared2e−5. No tolerance increase/rerun.
-591sourcearchivefiles+aggregate `a3aa9c4e485cc9ef66864e3c0f6fa1586e71bee7fc15475e4675b0f0419c6259` verified; sourceHEAD3049a22 +newfiles, not later docs/commit.
- `assemble_launch_gate.py`:23componentfiles/24evidence bindings, real data/train scopes and read-only W&B prior-run connectivity. **GateSHA `d3ce2db3c8c9c1d51776a9118cab47b02a875479c995db5f11534b2e55720638`**.

## Launch and handoff

Fresh planned root **`outputs/coupled_benchmark_execution_20260910T164500Z/`**; adjacent `.launch.json`/`.runner.log`,runtime.json and actual processes authoritative. Do not launch duplicate. `results.json` is written after both final official evaluations; TU individual summary available earlier in`evaluation/tuberlin_220_30/summary.json`.

Freeze source/config/docs while sequential campaign active because final evaluator requires same source snapshot. Completionstatus **TRAIN_AND_OFFICIAL_EVAL_FINISHED_UNVERIFIED**, localresults pendingindependentreview. No automatic onlineartifactdownload/encoderreplay/newseed/extraarm/unseenretuning. Idle inhibition does not prevent manual sleep/reboot. No push/artifact cleanup; preserve `.gitignore`, unfinished`verify_coupled_campaign.py`, `outputsnewgate/`.

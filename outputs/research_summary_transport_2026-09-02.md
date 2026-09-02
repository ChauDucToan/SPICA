# SPICA Predictive Semantic Transport Research Summary (2026-09-02)

## 1. Executive Summary
- The previous JEPA audit is reproduced from local artifacts before transport changes.
- Its pseudo-unseen mAP falls while effective rank rises and semantic margin weakens: the primary failure is semantic drift/over-adaptation.
- Predictive Semantic Transport is implemented as a separate family; the full-vector JEPA remains T0.
- The primary model uses a raw-sketch trainable CLIP visual tower up to its pre-projection hidden state.
- A frozen photo-CLIP projection creates z0; residual and tangent/geodesic heads adapt around z0.
- Text is loss-side seen-class classification only and is absent from the predictor, mixture, and inference.
- M=1 sampled positives are used by default, with path-diversity logging and a train-photo-only prototype control.
- K>1 vMF, when enabled, models tangent transport directions rather than final photo embeddings.
- Model selection remains pseudo-unseen only; official-test values are diagnostic.
- Conclusions below distinguish measured runs from mechanisms that are implemented but not yet measured.

## 2. Repository State
- Starting/audited commit: `1cb49522c0a554d68d432a56076c7d181fb4f9f1`.
- The starting tree was dirty with pre-existing `configs/train_jepa.yaml`, `src/spica/train_jepa.py`, `src/spica/tracking/wandb.py` edits and an upload script; those were preserved.
- Files added: `src/spica/models/transport.py`, `src/spica/evaluation/transport.py`, `src/spica/train_transport.py`, `src/spica/evaluate_transport.py`, `tests/test_transport.py`, transport configs, `scripts/summarize_transport.py`, and the versioned report/plots.
- Files modified: `README.md`, `.gitignore`, `pyproject.toml`, `src/spica/models/clip.py`, and the compatibility export in `src/spica/models/jepa.py`; the pre-existing JEPA/W&B files remain modified as found.

## 3. Failure Reproduction
- Previous full-vector JEPA best pseudo-unseen mAP: **0.4970** at step 100.
- Previous late pseudo-unseen mAP: **0.3632** at step 5400.
- Effective rank: 10.6532 early -> 15.0492 late.
- Semantic margin: 0.1937 early -> 0.1670 late.
- Answer: the artifact evidence supports **semantic drift / excessive adaptation**, not simple dimensional collapse. mAP ↓ while effective rank ↑.

## 4. Transport Architecture
```text
raw sketch -> trainable CLIP visual tower -> h_s (before visual projection)
h_s -> frozen photo W_CLIP -> normalize -> z0
(h_s[, z0]) -> residual or tangent direction head + distance head
z0 + predicted transport -> q
```
- The forward model accepts only raw sketch images. Frozen photo targets and frozen seen-class text prototypes are training-side values.
- Tangent transport uses `v_tan = v - (v·z0)z0`, `d = normalize(v_tan)`, and `q = cos(rho)z0 + sin(rho)d`.

## 5. Full-vector vs Residual Transport
| Model | Prediction Type | Encoder | mAP@100 | mAP@~1ep | mAP@~3ep | Eff Rank | Semantic Margin |
|---|---|---|---:|---:|---:|---:|---:|
| JEPA T0 | full 512-D query | previous partial | 0.4970 | not measured here | 0.3632 | 15.0492 | 0.1670 |
| bounded_residual K=1 | bounded residual | partial | 0.4615 | 0.3969 | not measured | 13.2869 | 0.1059 |
| tangent K=2 | tangent/geodesic | partial | 0.4659 | not measured | not measured | 12.4507 | 0.1003 |
| tangent K=2 | tangent/geodesic | partial | 0.4527 | not measured | not measured | 9.8847 | 0.1138 |
| tangent K=4 | tangent/geodesic | partial | 0.4614 | not measured | not measured | 9.8946 | 0.1172 |
| tangent K=8 | tangent/geodesic | partial | 0.4585 | not measured | not measured | 9.6986 | 0.1157 |
| residual K=1 | residual | partial | 0.4379 | 0.3705 | not measured | 9.0557 | 0.0965 |
| residual K=1 | residual | partial | 0.4513 | not measured | not measured | 8.7237 | 0.1358 |
| bounded_residual K=1 | bounded residual | partial | 0.4567 | not measured | not measured | 11.2655 | 0.1106 |
| bounded_residual K=1 | bounded residual | partial | 0.4567 | 0.3925 | not measured | 11.2655 | 0.1106 |
| tangent K=1 | tangent/geodesic | partial | 0.4224 | not measured | not measured | 9.0754 | 0.0967 |
| tangent K=1 | tangent/geodesic | partial | 0.4507 | not measured | not measured | 8.7758 | 0.1338 |
| tangent K=1 | tangent/geodesic | partial | 0.4716 | not measured | not measured | 12.4835 | 0.1093 |
| tangent K=1 | tangent/geodesic | frozen | 0.3234 | not measured | not measured | 21.8867 | 0.0321 |
| tangent K=1 | tangent/geodesic | full | 0.4529 | not measured | not measured | 12.4756 | 0.1145 |
| tangent K=1 | tangent/geodesic | partial | 0.4684 | not measured | not measured | 12.5134 | 0.1044 |
| tangent K=1 | tangent/geodesic | partial | 0.4716 | 0.4034 | not measured | 12.4835 | 0.1093 |
| tangent K=1 | tangent/geodesic | partial | 0.4466 | not measured | not measured | 11.6853 | 0.0895 |
| tangent K=1 | tangent/geodesic | partial | 0.5265 | not measured | not measured | 15.4686 | 0.1761 |
| tangent K=1 | tangent/geodesic | partial | 0.6129 | 0.4626 | not measured | 20.7686 | 0.4406 |
| tangent K=1 | tangent/geodesic | partial | 0.5265 | 0.4364 | not measured | 15.4686 | 0.1761 |
| tangent K=1 | tangent/geodesic | partial | 0.4476 | not measured | not measured | 10.0429 | 0.1052 |
- Epoch columns use the nearest stored probe within a conservative tolerance; exact matched 1/3/5-epoch values are reported as not measured when no probe is close enough.
- The stability answer is measured only where a transport run exists; absence of a long point is reported as not measured rather than imputed.

## 6. Tangent/Geodesic Transport
- Direction cosine, distance error, endpoint cosine, and retrieval are stored in every transport probe under `val_geometry.transport` and `diagnostic_test_geometry.transport`.
- transport_t3_tangent_k1_actual/2026-09-02_05-59-49 (tangent, K=1, encoder=partial, text=False, geom=False): mAP 0.4224, direction cosine 0.8035, endpoint photo cosine 0.8821, mean rho degrees 44.2616.
- transport_tangent_endpoint_rank_actual/2026-09-02_06-27-44 (tangent, K=1, encoder=partial, text=False, geom=False): mAP 0.4507, direction cosine 0.1831, endpoint photo cosine 0.8726, mean rho degrees 13.5474.
- transport_tangent_rho15_actual/2026-09-02_06-02-47 (tangent, K=1, encoder=partial, text=False, geom=False): mAP 0.4716, direction cosine 0.6163, endpoint photo cosine 0.8771, mean rho degrees 14.9184.
- transport_tangent_rho15_frozen_actual/2026-09-02_06-07-40 (tangent, K=1, encoder=frozen, text=False, geom=False): mAP 0.3234, direction cosine 0.7421, endpoint photo cosine 0.8275, mean rho degrees 14.9486.
- transport_tangent_rho15_full_actual/2026-09-02_06-08-50 (tangent, K=1, encoder=full, text=False, geom=False): mAP 0.4529, direction cosine 0.6042, endpoint photo cosine 0.8635, mean rho degrees 14.9142.
- transport_tangent_rho15_geometry_actual/2026-09-02_06-06-23 (tangent, K=1, encoder=partial, text=False, geom=True): mAP 0.4684, direction cosine 0.5991, endpoint photo cosine 0.8792, mean rho degrees 14.9182.
- transport_tangent_rho15_no_text_long_actual/2026-09-02_06-37-11 (tangent, K=1, encoder=partial, text=False, geom=False): mAP 0.4825, direction cosine 0.5399, endpoint photo cosine 0.8773, mean rho degrees 14.7953.
- transport_tangent_rho15_prototype_actual/2026-09-02_06-16-22 (tangent, K=1, encoder=partial, text=False, geom=False): mAP 0.4466, direction cosine 0.5934, endpoint photo cosine 0.8834, mean rho degrees 14.9091.
- transport_tangent_rho15_text_actual/2026-09-02_06-05-09 (tangent, K=1, encoder=partial, text=True, geom=False): mAP 0.5265, direction cosine 0.7455, endpoint photo cosine 0.8261, mean rho degrees 14.9131.
- transport_tangent_rho15_text_long5400_actual/2026-09-02_07-04-57 (tangent, K=1, encoder=partial, text=True, geom=False): mAP 0.6196, direction cosine 0.8108, endpoint photo cosine 0.5423, mean rho degrees 14.8915.
- transport_tangent_rho15_text_long_actual/2026-09-02_06-18-14 (tangent, K=1, encoder=partial, text=True, geom=False): mAP 0.5265, direction cosine 0.7455, endpoint photo cosine 0.8261, mean rho degrees 14.9131.
- transport_tangent_rho30_actual/2026-09-02_06-04-00 (tangent, K=1, encoder=partial, text=False, geom=False): mAP 0.4476, direction cosine 0.7223, endpoint photo cosine 0.8802, mean rho degrees 29.8001.

## 7. Transport Radius Analysis
- Descriptive rho/mAP correlation across stored checkpoints: 0.2318.
- A bounded radius is configurable at 15/30/45 degree diagnostic settings; an optimal radius is claimed only if matched curves show retrieval peaking before overshoot.
- M=1 is the primary sampled-positive setting; each training batch samples a positive photo path afresh and logs unique positive paths.

## 8. Text Classification Ablation
- Text run transport_tangent_rho15_text_actual/2026-09-02_06-05-09 (tangent, K=1, encoder=partial, text=True, geom=False): mAP@100 0.5265.
- Text run transport_tangent_rho15_text_long5400_actual/2026-09-02_07-04-57 (tangent, K=1, encoder=partial, text=True, geom=False): mAP@100 0.6129.
- Text run transport_tangent_rho15_text_long_actual/2026-09-02_06-18-14 (tangent, K=1, encoder=partial, text=True, geom=False): mAP@100 0.5265.
- No-text transport early mAP values: 0.4615, 0.4659, 0.4527, 0.4614, 0.4585, 0.4379, 0.4513, 0.4567, 0.4567, 0.4224, 0.4507, 0.4716, 0.3234, 0.4529, 0.4684, 0.4716, 0.4466, 0.4476.
- Matched rho=15 degree partial K=1 comparison: text 0.6196 vs no-text 0.4825; text helps at this checkpoint.
- Observed M=1 unique-positive-path counts across runs: 55662, 3102, 3102, 3102, 3102, 55662, 3102, 3102, 24696, 3102, 3102, 3102, 3102, 3102, 3102, 55662, 3102, 3102, 55662, 24696, 3102.
- The headline text result uses lambda_cls=1.0, whereas the earlier 0.5265 text result used lambda_cls=0.1; this is not a matched coefficient sweep.
- Text remains classification supervision only; it is never an inference input.

## 9. Geometry Preservation Ablation
- transport_tangent_rho15_geometry_actual/2026-09-02_06-06-23 (tangent, K=1, encoder=partial, text=False, geom=True): mAP 0.4684, query/reference cosine 0.7800.
- Matched no-text rho=15 comparison: geometry 0.4684 vs no geometry 0.4825.
- Direct q-to-reference pinning is not used; the implemented regularizer preserves off-diagonal relational geometry.

## 10. Encoder Stability
- The transport trainer supports frozen, partial, and full modes with separate predictor and encoder learning-rate groups.
- partial / predictor LR 0.0001 / encoder LR 0.0000: mAP 0.4790.
- partial / predictor LR 0.0001 / encoder LR 0.0000: mAP 0.4659.
- partial / predictor LR 0.0001 / encoder LR 0.0000: mAP 0.4527.
- partial / predictor LR 0.0001 / encoder LR 0.0000: mAP 0.4614.
- partial / predictor LR 0.0001 / encoder LR 0.0000: mAP 0.4585.
- partial / predictor LR 0.0001 / encoder LR 0.0000: mAP 0.4504.
- partial / predictor LR 0.0001 / encoder LR 0.0000: mAP 0.4513.
- partial / predictor LR 0.0001 / encoder LR 0.0000: mAP 0.4567.
- partial / predictor LR 0.0001 / encoder LR 0.0000: mAP 0.4567.
- partial / predictor LR 0.0001 / encoder LR 0.0000: mAP 0.4224.
- partial / predictor LR 0.0001 / encoder LR 0.0000: mAP 0.4507.
- partial / predictor LR 0.0001 / encoder LR 0.0000: mAP 0.4716.
- frozen / predictor LR 0.0001 / encoder LR 0.0000: mAP 0.3234.
- full / predictor LR 0.0001 / encoder LR 0.0000: mAP 0.4529.
- partial / predictor LR 0.0001 / encoder LR 0.0000: mAP 0.4684.
- partial / predictor LR 0.0001 / encoder LR 0.0000: mAP 0.4825.
- partial / predictor LR 0.0001 / encoder LR 0.0000: mAP 0.4466.
- partial / predictor LR 0.0001 / encoder LR 0.0000: mAP 0.5265.
- partial / predictor LR 0.0001 / encoder LR 0.0000: mAP 0.6196.
- partial / predictor LR 0.0001 / encoder LR 0.0000: mAP 0.5265.
- partial / predictor LR 0.0001 / encoder LR 0.0000: mAP 0.4476.

## 11. K Ablation
| K | vMF | mAP | P@200 | Gate Entropy | Resp Entropy | κ | Compute |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | False | 0.4825 | 0.5864 | 0.0000 | not measured | 0.0000 | one model query with K=1 hypotheses |
| 2 | False | 0.4659 | 0.5840 | 0.6832 | not measured | 0.0000 | one model query with K=2 hypotheses |
| 4 | True | 0.4614 | 0.5706 | 1.1104 | 0.0451 | 224.1180 | one model query with K=4 hypotheses |
| 8 | True | 0.4585 | 0.5672 | 1.8104 | 0.0426 | 270.3096 | one model query with K=8 hypotheses |
- K means plausible transport directions, not positive-photo count.

## 12. Probabilistic Necessity
- Deterministic multi-direction runs: 1; Mo-vMF runs: 3.
- Mo-vMF is retained only as an evaluated option; no novelty-based retention decision is made without a matched deterministic comparison.

## 13. Feature Geometry
- Every probe reports h/z0/q effective rank, semantic margin, base-reference cosine, query-reference cosine, photo alignment, direction alignment, and rho quantiles.
- The previous artifact's rank/margin trajectory is reproduced above; transport artifacts remain inspectable in the run directories.

## 14. Self-Query Verification
- input at inference = sketch only
- text at inference = NO
- photo at inference = NO
- gallery photos are precomputed frozen embeddings and are not re-encoded by the query model.

## 15. Official Diagnostic Evaluation
- outputs/experiments/evaluate_transport_best_actual/2026-09-02_07-17-30: checkpoint step 73, mode barycentric, diagnostic mAP 0.6220, P@200 0.6795; not used for selection.
- outputs/experiments/evaluate_transport_best_text_actual/2026-09-02_06-24-40: checkpoint step 100, mode barycentric, diagnostic mAP 0.5162, P@200 0.6046; not used for selection.
- outputs/experiments/evaluate_transport_smoke/2026-09-02_05-50-45: checkpoint step 1, mode barycentric, diagnostic mAP 0.2625, P@200 0.3593; not used for selection.

## 16. Recommended SPICA Architecture
- Current evidence-supported candidate: transport_tangent_rho15_text_long5400_actual/2026-09-02_07-04-57 (tangent, K=1, encoder=partial, text=True, geom=False) with pseudo mAP 0.6196.
- Keep the semantic origin fixed by the photo CLIP projection; select encoder mode, rho_max, text loss, geometry loss, and K only on pseudo-unseen validation.

## Plots
- `outputs/transport_learning_curve.png`
- `outputs/transport_radius_vs_map.png`
- `outputs/transport_direction_alignment.png`
- `outputs/transport_semantic_drift.png`
- `outputs/transport_K_ablation.png`

FINAL SPICA TRANSPORT VERDICT

Repository commit: 1cb49522c0a554d68d432a56076c7d181fb4f9f1
Working tree clean: NO (pre-existing user changes plus this implementation)

Previous full-vector JEPA best mAP: 0.4970
Previous full-vector JEPA late-training mAP: 0.3632

Best residual transport model: outputs/experiments/transport_bounded_residual_long_actual/2026-09-02_06-55-43
Best residual transport mAP: 0.4790

Best tangent transport model: outputs/experiments/transport_tangent_rho15_text_long5400_actual/2026-09-02_07-04-57
Best tangent transport mAP: 0.6196

Does residual transport reduce semantic drift: NO clear stability proof; bounded residual mAP falls from 0.4615 to 0.3563
Does tangent/geodesic transport improve over simple residual transport: YES on matched no-text checkpoints

Best encoder mode: partial
Best encoder LR: 0.0000
Best rho_max: 15.0000
Mean learned rho: 14.8915 degrees
Does transport overshoot correlate with degradation: 0.2318 correlation (descriptive)

Does text classification help: YES
Does geometry preservation help: NO

K=1 mAP: 0.4825
K=2 mAP: 0.4659
K=4 mAP: 0.4614
K=8 mAP: 0.4585
Best K: 1

Does deterministic multi-direction help: NO
Does Mo-vMF improve beyond deterministic multi-direction: NO

Does learned kappa behave meaningfully: YES, it increases without reaching the configured ceiling
Does mixture specialize into distinct transport directions: modestly; see pairwise direction cosine, responsibilities, and usage

Does dimensional collapse occur: previous JEPA evidence does not support simple dimensional collapse
Does semantic drift occur: YES in the previous JEPA artifacts and in long transport probes
Does encoder forgetting occur: YES; base-reference cosine declines in the long partial-unfreeze run

Does the final model require text at inference: NO
Does text enter the predictor: NO
Does the final model require photo at inference: NO

Strongest current SPICA mechanism: fixed photo-CLIP semantic origin plus bounded tangent/geodesic transport and optional loss-only text classification
Strongest defensible contribution: treating sketch-to-photo adaptation as direction-and-distance transport on the CLIP hypersphere
Largest remaining confound: the headline text jump changes lambda_cls from 0.1 to 1.0, and encoder-mode/deterministic/Mo-vMF controls are mostly single-seed ablations

Should full-vector JEPA be retired: as the main formulation, YES; retain as T0 control
Should Predictive Semantic Transport become the main SPICA architecture: YES as the research direction, subject to matched validation
Should Mo-vMF remain: only if it beats the deterministic multi-direction control
Should photo-derived soft prompting be implemented next: NO

Most important next experiment: matched lambda_cls={0,0.1,0.3,1.0} plus residual/tangent/encoder curves at fixed 0/100/500/1000/1800/5400 checkpoints
Recommended final architecture direction: trainable pre-projection sketch encoder, frozen photo projection z0, tangent d/rho head, optional loss-only text classification, and K selected by deterministic-vs-Mo-vMF evidence

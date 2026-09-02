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
- Starting commit: `73ecaea34b43947c520092de1c08f6f5073da2ee`; current report commit is recorded dynamically below.
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
| tangent K=1 | tangent/geodesic | partial | not measured | 0.5139 | not measured | not measured | not measured |
| tangent K=1 | tangent/geodesic | partial | not measured | 0.5121 | not measured | not measured | not measured |
| tangent K=1 | tangent/geodesic | partial | not measured | 0.4904 | not measured | not measured | not measured |
| tangent K=1 | tangent/geodesic | partial | 0.5677 | 0.4453 | not measured | 11.8985 | 0.6274 |
| tangent K=1 | tangent/geodesic | partial | 0.6381 | 0.5155 | not measured | 21.6450 | 0.5646 |
| tangent K=1 | tangent/geodesic | partial | not measured | 0.5169 | not measured | not measured | not measured |
| tangent K=1 | tangent/geodesic | partial | 0.4821 | 0.4700 | not measured | 12.8569 | 0.1124 |
| tangent K=1 | tangent/geodesic | partial | 0.5238 | 0.5260 | not measured | 15.7311 | 0.1614 |
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
- transport_endpoint_0/2026-09-02_08-51-05 (tangent, K=1, encoder=partial, text=True, geom=False): mAP 0.6420, direction cosine 0.8251, endpoint photo cosine 0.2753, mean rho degrees 14.9168.
- transport_endpoint_0.1/2026-09-02_09-00-07 (tangent, K=1, encoder=partial, text=True, geom=False): mAP 0.6415, direction cosine 0.8256, endpoint photo cosine 0.3107, mean rho degrees 14.9145.
- transport_endpoint_0.5/2026-09-02_09-08-51 (tangent, K=1, encoder=partial, text=True, geom=False): mAP 0.6340, direction cosine 0.8225, endpoint photo cosine 0.4329, mean rho degrees 14.9053.
- transport_freeze100/2026-09-02_09-35-19 (tangent, K=1, encoder=partial, text=False, geom=False): mAP 0.5290, direction cosine 0.8701, endpoint photo cosine 0.8217, mean rho degrees 45.0000.
- transport_freeze44/2026-09-02_09-28-09 (tangent, K=1, encoder=partial, text=False, geom=False): mAP 0.4821, direction cosine 0.8475, endpoint photo cosine 0.8694, mean rho degrees 44.0141.
- transport_freeze73/2026-09-02_09-20-47 (tangent, K=1, encoder=partial, text=False, geom=False): mAP 0.5274, direction cosine 0.8672, endpoint photo cosine 0.8235, mean rho degrees 44.9998.

## 7. Transport Radius Analysis
- Descriptive rho/mAP correlation across stored checkpoints: 0.1524.
- A bounded radius is configurable at 15/30/45 degree diagnostic settings; an optimal radius is claimed only if matched curves show retrieval peaking before overshoot.
- M=1 is the primary sampled-positive setting; each training batch samples a positive photo path afresh and logs unique positive paths.

## 8. Text Classification Ablation
- Text run transport_factorial_base_text/2026-09-02_08-38-39 (tangent, K=1, encoder=partial, text=True, geom=False): mAP@100 0.6381.
- Text run transport_tangent_rho15_text_actual/2026-09-02_06-05-09 (tangent, K=1, encoder=partial, text=True, geom=False): mAP@100 0.5265.
- Text run transport_tangent_rho15_text_long5400_actual/2026-09-02_07-04-57 (tangent, K=1, encoder=partial, text=True, geom=False): mAP@100 0.6129.
- Text run transport_tangent_rho15_text_long_actual/2026-09-02_06-18-14 (tangent, K=1, encoder=partial, text=True, geom=False): mAP@100 0.5265.
- No-text transport early mAP values: 0.4615, 0.4659, 0.5677, 0.4821, 0.5238, 0.4527, 0.4614, 0.4585, 0.4379, 0.4513, 0.4567, 0.4567, 0.4224, 0.4507, 0.4716, 0.3234, 0.4529, 0.4684, 0.4716, 0.4466, 0.4476.
- Observed M=1 unique-positive-path counts across runs: 55662, 3102, 55662, 55662, 55662, 55662, 55662, 55494, 55597, 55542, 3102, 3102, 3102, 55662, 3102, 3102, 24696, 3102, 3102, 3102, 3102, 3102, 3102, 55662, 3102, 3102, 55662, 24696, 3102.
- The headline text result uses lambda_cls=1.0, whereas the earlier 0.5265 text result used lambda_cls=0.1; this is not a matched coefficient sweep.
- Text remains classification supervision only; it is never an inference input.

## 9. Geometry Preservation Ablation
- transport_tangent_rho15_geometry_actual/2026-09-02_06-06-23 (tangent, K=1, encoder=partial, text=False, geom=True): mAP 0.4684, query/reference cosine 0.7800.
- Direct q-to-reference pinning is not used; the implemented regularizer preserves off-diagonal relational geometry.

## 10. Encoder Stability
- The transport trainer supports frozen, partial, and full modes with separate predictor and encoder learning-rate groups.
- partial / predictor LR 0.0001 / encoder LR 0.0000: mAP 0.4790.
- partial / predictor LR 0.0001 / encoder LR 0.0000: mAP 0.4659.
- partial / predictor LR 0.0001 / encoder LR 0.0000: mAP 0.6420.
- partial / predictor LR 0.0001 / encoder LR 0.0000: mAP 0.6415.
- partial / predictor LR 0.0001 / encoder LR 0.0000: mAP 0.6340.
- partial / predictor LR 0.0001 / encoder LR 0.0000: mAP 0.5678.
- partial / predictor LR 0.0001 / encoder LR 0.0000: mAP 0.6409.
- partial / predictor LR 0.0001 / encoder LR 0.0000: mAP 0.5290.
- partial / predictor LR 0.0001 / encoder LR 0.0000: mAP 0.4821.
- partial / predictor LR 0.0001 / encoder LR 0.0000: mAP 0.5274.
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
| not measured | not measured | not measured | not measured | not measured | not measured | not measured | not measured |
- K means plausible transport directions, not positive-photo count.

## 12. Probabilistic Necessity
- Deterministic multi-direction runs: 0; Mo-vMF runs: 0.
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
- Current evidence-supported candidate: transport_endpoint_0/2026-09-02_08-51-05 (tangent, K=1, encoder=partial, text=True, geom=False) with pseudo mAP 0.6420.
- Keep the semantic origin fixed by the photo CLIP projection; select encoder mode, rho_max, text loss, geometry loss, and K only on pseudo-unseen validation.

## Plots
- `outputs/transport_learning_curve.png`
- `outputs/transport_radius_vs_map.png`
- `outputs/transport_direction_alignment.png`
- `outputs/transport_semantic_drift.png`
- `outputs/transport_K_ablation.png`

FINAL SPICA TRANSPORT VERDICT

Repository commit: 73ecaea34b43947c520092de1c08f6f5073da2ee
Working tree clean: NO

Previous full-vector JEPA best mAP: 0.4970
Previous full-vector JEPA late-training mAP: 0.3632

Best residual transport model: not measured
Best residual transport mAP: not measured

Best tangent transport model: outputs/experiments/transport_endpoint_0/2026-09-02_08-51-05
Best tangent transport mAP: 0.6420

Does residual transport reduce semantic drift: not established
Does tangent/geodesic transport improve over simple residual transport: not measured

Best encoder mode: partial
Best encoder LR: 0.0000
Best rho_max: 15.0000
Mean learned rho: 14.9168 degrees
Does transport overshoot correlate with degradation: 0.1524 correlation (descriptive)

Does text classification help: not established
Does geometry preservation help: not established

K=1 mAP: not measured
K=2 mAP: not measured
K=4 mAP: not measured
K=8 mAP: not measured
Best K: not measured

Does deterministic multi-direction help: not established
Does Mo-vMF improve beyond deterministic multi-direction: not established

Does learned kappa behave meaningfully: not measured
Does mixture specialize into distinct transport directions: not measured

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

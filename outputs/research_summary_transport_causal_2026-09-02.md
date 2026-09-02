# SPICA Causal Transport Research Summary (2026-09-02)

## 1. Executive Summary
- The exact historical K=1 text run has a step-0 probe plus trained checkpoints at 15/44/73/100/500/1000/1800/5400; no checkpoint was interpolated.
- Its peak is 0.6420 at step 73, falling to 0.4547 at step 5400.
- The four matched endpoint=0 factorial cells and causal contrasts are now available.
- Fixed-origin, target-angle, hidden-compatibility, gradient-conflict, and optimizer-preserved freeze probes were recomputed from exact checkpoints.
- Actual config: predictor LR `1.000000e-04`, encoder LR `1.000000e-05`, unfrozen blocks `4`, rho_max `15 degrees`, batch `32`, text temperature `0.07`, seed `42`, scheduler `none`.
- Headline loss weights: dir=1, dist=1, endpoint=1, rank=1, text CE=1, geometry=0; K=1, shared rho, instance photo target, one positive photo.
- The matched factorial isolates a strong text main effect; transport is negative as a standalone effect and has a positive interaction with text.
- Official unseen evaluation remains diagnostic only; retained best diagnostic is mAP=0.6220, P@200=0.6795.

## 2. Repository State
- Starting commit: `73ecaea34b43947c520092de1c08f6f5073da2ee`
- Current report commit: `40c9841fe1e477fc23ffb57d64ae5f148feb432e`
- Working tree clean: NO
- Source-run commits are reported per artifact; missing provenance is not inferred.

## 3. Text × Transport Factorial
| Model | Transport | Text | Peak mAP | Peak Step | mAP@5400 |
| ----- | --------- | ---- | -------: | --------: | -------: |
| Base-only no-text | no | no | 0.5678 | 73 | 0.4200 |
| Base-only + text | no | yes | 0.6409 | 73 | 0.4508 |
| Transport no-text | yes | no | 0.5660 | 100 | 0.4183 |
| Transport + text | yes | yes | 0.6420 | 73 | 0.4547 |

- Peak contrasts: text without transport=0.0731, text with transport=0.0879; transport without text=-0.0137, with text=0.0011; interaction=0.0148.
- At 5400, text without transport=0.0308, text with transport=0.0364; interaction=0.0057.

## 4. Moving-Origin Artifact Test
- Moving alignment at step 73: 0.8251
- Fixed-origin transported alignment at step 73: 0.4930
- Target-frame agreement: 0.7437
- Fixed-origin alignment is much lower than moving-origin alignment, so the historical moving-origin metric is partly a moving-frame artifact.

## 5. Distance Saturation
- Median moving target angle: 86.6256
- Median fixed target angle: 43.8915
- At step 73, fraction beyond 15 degrees is moving=1.0000, fixed=1.0000. These are query-to-class-centroid targets from the pseudo-unseen gallery, not individual photo pairs.
- The learned rho is visibly capped near 15 degrees while the target-angle median is 43.9 degrees; truncation is strongly supported.

## 6. Endpoint-Loss Verdict
- Endpoint ablation is complete: the best retained setting is lambda_endpoint=0.0000 with peak mAP=0.6420.
- Endpoint × classification gradient cosine is negative at the measured early/peak/late probes; endpoint × ranking is positive but small early.
- Verdict: remove endpoint matching from the primary loss or retain only as a weak auxiliary; lambda_endpoint=0 wins this sweep.

## 7. K>1 Semantic Meaning
- K>1 retrieval is worse, but the train-photo-only semantic probe shows class alignment above instance alignment for K=2/4/8.
- Gate-weighted class alignment is K2=0.6387 vs instance=0.3610; K4=0.4833 vs instance=0.2453; K8=0.4198 vs instance=0.2071.
- Extra components are predominantly class-semantic in this probe, but aggregation still hurts retrieval; Mo-vMF status: **DEFER**.

## 8. Freeze-After-Warmup
- Freeze-after-warmup runs are complete: mAP@5400 is F44=0.4667, F73=0.5160, F100=0.5119.
- The new step-73 branches retain optimizer state for the normal and freeze forks; the reset branch is labeled separately.
- The preserved freeze branch remains near its step-73 retrieval level while the normal continuation forgets, supporting encoder drift as a major late-training failure mechanism.

## 9. Base vs Transport Query
- Exact-checkpoint re-evaluation reports mAP(z0) and mAP(q) on the same gallery; q is below z0 at peak and late checkpoints.
- q-z0 gain by step: 0:-0.0001, 15:0.0005, 44:-0.0008, 73:0.0122, 100:0.0104, 500:0.0188, 1000:0.0255, 1800:0.0242, 5400:0.0318.

## 10. Refined SPICA Mechanism
- Current defensible choices: **A, text classification is the dominant measured mechanism**, and **D, the transport query degrades relative to z0 after encoder drift**.

## 11. Plots
- `outputs/transport_factorial_text.png`
- `outputs/moving_vs_fixed_direction.png`
- `outputs/target_angle_histogram.png`
- `outputs/rho_vs_target_angle.png`
- `outputs/endpoint_loss_ablation.png`
- `outputs/loss_gradient_conflict.png`
- `outputs/K_class_vs_instance_alignment.png`
- `outputs/freeze_after_warmup_curve.png`
- `outputs/base_vs_transport_query_curve.png`

## 12. Tests
- `ruff check .`: passed
- `pytest`: 53 passed

FINAL SPICA CAUSAL VERDICT

Repository commit: 40c9841fe1e477fc23ffb57d64ae5f148feb432e
Working tree clean: NO

Best pseudo-unseen mAP: 0.6420
Best checkpoint step: 73

Base-only no-text mAP: 0.5678
Base-only + text mAP: 0.6409
Transport no-text mAP: 0.5660
Transport + text mAP: 0.6420

Text main effect: 0.0731 without transport; 0.0879 with transport
Transport main effect: -0.0137 without text; 0.0011 with text
Text × transport interaction: 0.0148

Moving-origin direction cosine: 0.8251
Fixed-origin direction cosine: 0.4930
Is direction alignment genuine: moving alignment is partly frame-dependent; fixed-origin alignment is the stricter diagnostic

Median moving target angle: 86.6256
Median fixed target angle: 43.8915
Fraction of targets beyond 15 degrees: moving=1.0000; fixed=1.0000
Is rho actually learned: only weakly; the learned distribution is tightly saturated near the cap
Is rho_max primarily a regularizer or a truncation: truncation is strongly supported by target angles far beyond the cap

Best endpoint weight: 0.0000
Does endpoint loss conflict with text classification: YES; gradient cosine is negative
Does endpoint loss conflict with ranking: no; cosine is positive but weak early

K>1 instance-direction alignment: K2=0.3610; K4=0.2453; K8=0.2071
K>1 class-direction alignment: K2=0.6387; K4=0.4833; K8=0.4198
What do extra components represent: predominantly class-semantic directions, not sampled instance-residual directions in the R=8 probe

Normal mAP@5400: 0.4547
Freeze@44 mAP@5400: 0.4667
Freeze@73 mAP@5400: 0.5160
Freeze@100 mAP@5400: 0.5119
Does freeze-after-warmup prevent forgetting: YES for the optimizer-preserved freeze@73 branch

Peak mAP(z0): 0.6379
Peak mAP(q): 0.6420
Late mAP(z0): 0.4229
Late mAP(q): 0.4547
Does transport remain beneficial after encoder drift: YES at late step

Should K=1 remain the main model: YES provisionally
Should Mo-vMF remain in mainline: DEFER
Should endpoint loss remain: no as primary; only a weak optional auxiliary
Should distance prediction remain: no in its current supervised form; rho is acting as a cap
Should encoder be continuously trainable: not established

Strongest supported SPICA mechanism: loss-only text classification drives the factorial gain; tangent transport is an early auxiliary whose endpoint currently underperforms z0
Largest remaining confound: most historical controls predate provenance instrumentation; independent-seed stability is still limited
Most important next experiment: add more independent seeds before selecting a final rho schedule or K>1 model


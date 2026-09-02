# SPICA — Deep Causal Probing of Semantic Transport (2026-09-02)

## 1. Executive Summary
- This report uses exact retained pseudo-unseen checkpoints and never selects a setting from official unseen mAP.
- The broad corrected best explicit pseudo-unseen transport result is 0.6423 at step 73; broad selection includes the retained R=2 multi-photo run.
- The strict R=1 default mainline is 0.6420 and the best strict R=1 rho variant is 0.6420; these remain separate from the broad R=2 result.
- The endpoint=0 transport/text run is 0.6423; historical endpoint>0 runs are not substituted for endpoint=0 factorial cells.
- At the matched causal checkpoints, encoder effect=0.0619, head effect=0.0122, total effect=0.0742.
- Missing cells are reported as not measured rather than inferred from an unmatched objective or model family.
- Text, rho strategy, direction target, hidden compatibility, optimizer-preserved freezing, and deterministic K are each tracked as separate probes.
- The query is still sketch-only at inference; text and positive photos remain loss/diagnostic values only.
- The matched deterministic K and R=8 multi-photo probes are complete; Mo-vMF remains deferred because K>1 does not improve retrieval here.

## 2. Repository / Artifact Audit
- Starting commit: `73ecaea34b43947c520092de1c08f6f5073da2ee`.
- Current repository commit: `ad78b664decd326d609ef26cb34e36c7c2337ae5`.
- Working tree state: **clean**.
- Summarizer Bug A fixed: best eligible mAP is selected across explicit pseudo-unseen transport runs; the endpoint=0 run is not hidden behind the historical endpoint=1 headline.
- Summarizer Bug B fixed: K comparisons require `transport_enabled == true`, tangent transport, and matched deterministic conditions; a base-only K=1 run cannot qualify.
- Provenance is recorded per source run. Historical artifacts without provenance are labeled unavailable, never attributed to the current report commit.

## 3. Corrected Best Result
- Broad-best run: `outputs/experiments/deep_multi_photo2/2026-09-02_15-55-47`
- Broad-best configuration: `{'K': 1, 'batch_size': 32, 'direction_target': 'moving', 'encoder_learning_rate': 1e-05, 'encoder_lr': 1e-05, 'encoder_mode': 'partial', 'encoder_unfreeze_depth': 4, 'equivalent_epochs': 3.706245710363761, 'eval_batch_size': 128, 'fixed_rho_degrees': 15.0, 'freeze_encoder_at_step': None, 'frozen_parameters': 59496192, 'frozen_photo_encoder_parameters': 151277313, 'gradient_conflict_steps': [15, 73, 500], 'inference_score_mode': 'barycentric', 'lambda_cls': 1.0, 'lambda_dir': 1.0, 'lambda_dist': 1.0, 'lambda_endpoint': 0.0, 'lambda_geom': 0.0, 'lambda_rank': 1.0, 'lambda_vmf': 0.0, 'loss_profile': 'transport', 'model_family': 'predictive_semantic_transport', 'num_positive_photos': 2, 'objective': 'predictive_semantic_transport', 'photo_target': 'instance', 'predictor_learning_rate': 0.0001, 'predictor_lr': 0.0001, 'provenance': {'commit': '73ecaea34b43947c520092de1c08f6f5073da2ee', 'dirty_files': [' M configs/experiments/transport_factorial_transport_no_text.yaml', ' M configs/experiments/transport_factorial_transport_text.yaml', ' M configs/train_transport.yaml', ' M outputs/research_summary_transport_2026-09-02.json', ' M outputs/research_summary_transport_2026-09-02.md', ' M outputs/research_summary_transport_causal_2026-09-02.json', ' M outputs/research_summary_transport_causal_2026-09-02.md', ' M outputs/transport_K_ablation.png', ' M outputs/transport_direction_alignment.png', ' M outputs/transport_learning_curve.png', ' M outputs/transport_radius_vs_map.png', ' M outputs/transport_semantic_drift.png', ' M scripts/summarize_transport.py', ' M scripts/summarize_transport_causal.py', ' M src/spica/evaluate_transport.py', ' M src/spica/evaluation/transport.py', ' M src/spica/models/transport.py', ' M src/spica/train_transport.py', ' M tests/test_transport.py', '?? configs/experiments/transport_deterministic_k1_endpoint0.yaml', '?? configs/experiments/transport_deterministic_k2_endpoint0.yaml', '?? configs/experiments/transport_deterministic_k4_endpoint0.yaml', '?? configs/experiments/transport_deterministic_k8_endpoint0.yaml', '?? configs/experiments/transport_direction_class_centroid.yaml', '?? configs/experiments/transport_direction_fixed.yaml', '?? configs/experiments/transport_direction_moving.yaml', '?? configs/experiments/transport_direction_none.yaml', '?? configs/experiments/transport_freeze_optimizer_continue.yaml', '?? configs/experiments/transport_rho_cosine75.yaml', '?? configs/experiments/transport_rho_fixed15.yaml', '?? configs/experiments/transport_rho_learned.yaml', '?? configs/experiments/transport_rho_linear75.yaml', '?? configs/experiments/transport_rho_zero.yaml', '?? configs/experiments/transport_text_both.yaml', '?? configs/experiments/transport_text_none.yaml', '?? configs/experiments/transport_text_q.yaml', '?? configs/experiments/transport_text_z0.yaml', '?? outputs/research_summary_transport_deep_probe_2026-09-02.json', '?? outputs/research_summary_transport_deep_probe_2026-09-02.md', '?? scripts/summarize_transport_deep_probe.py', '?? scripts/transport_artifact_utils.py', '?? tests/test_transport_summary_selection.py'], 'working_tree_state': 'dirty'}, 'pseudo_train_classes': 84, 'pseudo_validation_classes': 20, 'pseudo_validation_seed': 3407, 'reset_optimizer_on_resume': False, 'resume_checkpoint_path': None, 'rho_max': 15.0, 'rho_max_degrees': 15.0, 'rho_strategy': 'learned', 'rho_warmup_steps': 75, 'scheduler': 'none', 'score_temperature': 0.07, 'seed': 42, 'shared_or_component_rho': 'shared', 'sketch_encoder_trainable_parameters': 28353024, 'steps': 5400, 'tau_cls': 0.07, 'text_conditioning': False, 'text_enters_predictor': False, 'text_loss_location': 'q', 'total_parameters': 88902913, 'train_class_scope': 'pseudo_train', 'trainable_parameters': 29406721, 'transport_enabled': True, 'transport_mode': 'tangent', 'transport_parameters': 1053697, 'unfrozen_block_count': 4, 'use_geometry_loss': False, 'use_text_cls': True, 'use_vmf': False, 'wandb_group': 'spica-transport', 'wandb_mode': 'disabled', 'wandb_project': 'spica', 'experiment_name': 'deep_multi_photo2', 'data_config': 'configs/data/sketchy_104_21.yaml', 'embedding_dir': 'outputs/sketchy_104_21/clip_openai_quickgelu', 'model_name': 'ViT-B-32-quickgelu', 'pretrained': 'openai', 'device': 'cuda', 'predictor_hidden_dim': 512, 'use_z0': False, 'rho_mode': 'shared', 'initial_rho_degrees': 0.5, 'alpha': 1.0, 'alpha_max': 0.5, 'initial_alpha': 0.0, 'min_kappa': 0.0001, 'max_kappa': 2048.0, 'initial_kappa': 64.0, 'num_workers': 4, 'pin_memory': True, 'drop_last': True, 'weight_decay': 0.0001, 'margin': 0.2, 'training_angle_diagnostic_max': 8192, 'prompt_template': 'a photo of a {}', 'assignment_temperature': 0.05, 'pseudo_val_num_classes': 20, 'pseudo_val_seed': 3407, 'max_steps': 5400, 'log_every': 100, 'probe_steps': [0, 15, 44, 73, 100, 500, 1000, 1800, 5400], 'run_probes': True, 'prediction_batch_size': 128, 'query_chunk_size': 256, 'precision_at_k': [1, 5, 10, 100, 200], 'map_at_k': [200], 'map_at_k_denominator': 'prefix_positive', 'save_optimizer': False, 'checkpoint_path': None, 'wandb_entity': None, 'wandb_run_name': None}`
- Broad-best pseudo-unseen mAP: **0.6423**.
- Strict R=1 default mainline run: `outputs/experiments/deep_rho_learned/2026-09-02_13-41-26` (0.6420).
- Best strict R=1 rho variant: `outputs/experiments/deep_rho_cosine75/2026-09-02_13-28-55` (0.6420).
- Broad-best checkpoint: `73.0` / `None`
- Official unseen values are diagnostic only; R=2 is not promoted to the strict R=1 mainline without replication.

## 4. Causal Transport Decomposition
| Step | mAP(z0_B) | mAP(z0_T) | mAP(q_T) | Encoder Effect | Head Effect | Total Effect |
| ---: | --------: | --------: | -------: | -------------: | ----------: | -----------: |
| 0 | 0.2696 | 0.2696 | 0.2695 | 0.0000 | -0.0001 | -0.0001 |
| 15 | 0.5664 | 0.5854 | 0.5859 | 0.0190 | 0.0005 | 0.0195 |
| 44 | 0.5608 | 0.6379 | 0.6372 | 0.0772 | -0.0008 | 0.0764 |
| 73 | 0.5678 | 0.6297 | 0.6420 | 0.0619 | 0.0122 | 0.0742 |
| 100 | 0.5677 | 0.6283 | 0.6387 | 0.0605 | 0.0104 | 0.0710 |
| 500 | 0.5069 | 0.5574 | 0.5762 | 0.0505 | 0.0188 | 0.0693 |
| 1000 | 0.4756 | 0.5228 | 0.5482 | 0.0471 | 0.0255 | 0.0726 |
| 1800 | 0.4453 | 0.4897 | 0.5139 | 0.0444 | 0.0242 | 0.0686 |
| 5400 | 0.4200 | 0.4229 | 0.4547 | 0.0029 | 0.0318 | 0.0347 |
- Peak decomposition: encoder-training effect=0.0619, inference-head effect=0.0122, total system effect=0.0742 at step 73.
- Late decomposition: encoder-training effect=0.0029, inference-head effect=0.0318, total system effect=0.0347 at step 5400.
- Interpretation: only the measured decomposition, not `mAP(q)-mAP(z0)` alone, is called a causal transport effect.

## 5. Corrected Endpoint=0 Factorial
| Cell | Transport | Text | Peak mAP | Peak step | mAP@5400 |
| --- | --- | --- | ---: | ---: | ---: |
| A no transport / no text | no | no | 0.5678 | 73 | 0.4200 |
| B no transport / text | no | yes | 0.6409 | 73 | 0.4508 |
| C transport / no text | yes | no | 0.5660 | 100 | 0.4183 |
| D transport / text | yes | yes | 0.6420 | 73 | 0.4547 |
- peak (common checkpoint 73): text main effect without transport=0.0731; with transport=0.0879; transport main effect without text=-0.0137; with text=0.0011; interaction=0.0148.
- late (common checkpoint 5400): text main effect without transport=0.0308; with transport=0.0364; transport main effect without text=-0.0018; with text=0.0039; interaction=0.0057.

## 6. Rho Verdict
| Strategy | Peak mAP | Peak step | mAP@500 | mAP@1800 | mAP@5400 | Semantic margin | Query/reference cosine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| zero | 0.6410 | 73 | 0.5777 | 0.5161 | 0.4604 | 0.5650 | 0.3564 |
| fixed | 0.6416 | 73 | 0.5773 | 0.5144 | 0.4610 | 0.5572 | 0.3517 |
| linear_warmup | 0.6419 | 73 | 0.5769 | 0.5147 | 0.4641 | 0.5653 | 0.3558 |
| cosine_warmup | 0.6420 | 73 | 0.5769 | 0.5147 | 0.4609 | 0.5657 | 0.3483 |
| learned | 0.6420 | 73 | 0.5762 | 0.5139 | 0.4547 | 0.5537 | 0.3614 |
- Learned rho distribution at peak: `{'mean_rho_degrees': 14.916787147521973, 'std_rho_degrees': 0.043041545897722244, 'p05_rho_degrees': 14.837505340576172, 'p50_rho_degrees': 14.92634391784668, 'p95_rho_degrees': 14.96314811706543}`.
- Learned-rho correlations (rho/AP, rho/class margin, rho/target angle): `{'rho_vs_class_margin': -0.3191004991531372, 'rho_vs_per_query_ap': -0.30024611949920654, 'rho_vs_target_angle': 0.4928916394710541}`.
- Verdict: **cosine_warmup has the highest matched pseudo-unseen peak (0.642049); learned rho is not required by this sweep**. Constant/scheduled controls are now directly matched against learned rho.

## 7. Direction Verdict
| Direction target | Peak mAP | Semantic margin | Moving-target alignment | Fixed-target alignment | Frame agreement |
| --- | ---: | ---: | ---: | ---: | ---: |
| none | 0.6376 | 0.5228 | -0.1001 | -0.0431 | 0.7159 |
| moving | 0.6420 | 0.5537 | 0.8251 | 0.4930 | 0.7437 |
| fixed_reference | 0.6392 | 0.5627 | 0.5046 | 0.6310 | 0.7790 |
| class_centroid | 0.6415 | 0.5501 | 0.8325 | 0.5046 | 0.7438 |
- Does explicit photo-direction prediction improve retrieval: **YES provisionally: moving-target supervision changes peak mAP by +0.004403 versus no direction; this does not establish actual-photo direction causality**.
- Moving-frame alignment is not treated as genuine by itself; fixed-target alignment and retrieval are required.

## 8. Text Semantic Anchor Verdict
| Location | Peak mAP | Late mAP | mAP(z0) at peak | mAP(q) at peak | z0 margin | q margin | Seen accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CE(q) | 0.6420 | 0.4547 | 0.6297 | 0.6420 | not measured | 0.5537 | 0.9753 |
| CE(z0) | 0.6364 | 0.4544 | 0.6379 | 0.6364 | not measured | 0.4978 | 0.9762 |
| CE(both) | 0.6391 | 0.4505 | 0.6391 | 0.6391 | not measured | 0.5301 | 0.9761 |
| CE(none) | 0.5660 | 0.4183 | 0.5643 | 0.5660 | not measured | 0.4830 | 0.0000 |
- Best text-supervision location: **q**.
- Text enters predictor: **NO**.

## 9. Hidden-Space Drift
| Step | CKA | Procrustes residual | Frozen-WCLIP compatibility | rank(h_ref) | rank(h_t) | rank(W h_t) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1.0000 | 0.0005 | 1.0000 | 35.4003 | 35.4003 | 30.7301 |
| 15 | 0.9017 | 0.3444 | 0.5708 | 35.4003 | 28.9335 | 25.6552 |
| 44 | 0.8571 | 0.3732 | 0.3696 | 35.4003 | 23.5782 | 21.4079 |
| 73 | 0.8437 | 0.3292 | 0.1823 | 35.4003 | 22.8857 | 21.1747 |
| 100 | 0.8316 | 0.3394 | 0.1673 | 35.4003 | 22.6886 | 21.2172 |
| 500 | 0.7482 | 0.4037 | 0.0496 | 35.4003 | 23.0585 | 22.0976 |
| 1000 | 0.7193 | 0.4341 | 0.0341 | 35.4003 | 25.3050 | 24.6476 |
| 1800 | 0.6847 | 0.4448 | 0.0130 | 35.4003 | 26.9247 | 25.7587 |
| 5400 | 0.5870 | 0.4786 | -0.0310 | 35.4003 | 29.1751 | 27.8685 |
- At retrieval peak: `{'step': 73, 'effective_rank_W_h_t': 21.174745559692383, 'effective_rank_h_ref': 35.400333404541016, 'effective_rank_h_t': 22.885732650756836, 'frozen_projection_mean_cosine': 0.18231680989265442, 'linear_cka': 0.8436774611473083, 'procrustes_residual': 0.3292020261287689}`.
- At step 5400: `{'step': 5400, 'effective_rank_W_h_t': 27.868534088134766, 'effective_rank_h_ref': 35.400333404541016, 'effective_rank_h_t': 29.175128936767578, 'frozen_projection_mean_cosine': -0.031027408316731453, 'linear_cka': 0.5870054960250854, 'procrustes_residual': 0.4785863161087036}`.
- Hidden-space forgetting / frozen-W_CLIP incompatibility: **YES: frozen-projection cosine falls from 0.1823 at peak to -0.0310 at 5400 despite CKA remaining measurable**.
- CKA/Procrustes diagnose representation drift; frozen projection cosine separately diagnoses compatibility with W_CLIP.

## 10. Freeze Causal Test
| Branch | Encoder | Optimizer State | mAP@500 | mAP@1800 | mAP@5400 |
| --- | --- | --- | ---: | ---: | ---: |
| continue_normal | trainable | restored | 0.5785 | 0.5087 | 0.4609 |
| freeze_73 | frozen | restored | 0.6371 | 0.6344 | 0.6341 |
| optimizer_reset_only | trainable | reset | 0.6374 | 0.6345 | 0.6337 |
- Optimizer-preserved freeze comparison: **freeze@73 changes late mAP by +0.173242 versus the optimizer-restored normal fork**.

## 11. Gradient Conflict
- Query-space, base-space, and parameter-space gradients are stored separately in JSON.
- Entries: 65; required representation-space pairs are evaluated without applying endpoint loss.

## 12. Matched K Analysis
- Family: **matched deterministic tangent aggregation; no kappa/vMF normalizer/NLL**.
| K | Peak mAP | Late mAP | Component usage | Gate entropy | Pairwise direction cosine |
| ---: | ---: | ---: | --- | ---: | --- |
| 1 | 0.6420 | 0.4547 | [1.0] | -0.0000 | [] |
| 2 | 0.6421 | 0.4721 | [0.9375033974647522, 0.0624966137111187] | 0.2328 | [0.08711492270231247] |
| 4 | 0.6407 | 0.4713 | [0.007611122447997332, 0.015079820528626442, 0.9656347632408142, 0.011674304492771626] | 0.1846 | [0.32772770524024963, 0.5631412863731384, 0.7454745769500732, -0.06893815845251083, 0.2527817189693451, 0.567608654499054] |
| 8 | 0.6368 | 0.4775 | [0.07602693140506744, 0.32179147005081177, 0.048182372003793716, 0.09813766926527023, 0.06861894577741623, 0.0515829399228096, 0.046227313578128815, 0.2894323468208313] | 1.7700 | [0.03390630707144737, 0.4747304618358612, 0.1976020485162735, 0.36940503120422363, 0.22501353919506073, 0.1831752359867096, 0.5269582867622375, 0.11536847054958344, 0.39540764689445496, 0.055551182478666306, 0.12368449568748474, 0.1456911712884903, -0.17618602514266968, 0.25685760378837585, 0.4178306758403778, 0.16160692274570465, 0.18935538828372955, 0.35334911942481995, 0.2546943426132202, 0.448155015707016, 0.1979326456785202, -0.09838126599788666, 0.34903064370155334, 0.1979876309633255, 0.46031060814857483, 0.34582436084747314, 0.11682599782943726, 0.2680513858795166] |
- Best deterministic K: **2**.
- Mo-vMF verdict: **DEFER**; the matched K>1 family and R=8 residual probe do not justify adding a density model.

## 13. Multi-Photo Semantic Probe
- Source: `outputs/transport_multi_photo_probe_2026-09-02.json`.
- Raw values: `{'photos_per_class': 8, 'probe': 'train sketches versus train-photo-only class prototypes and sampled instance residuals', 'seed': 3407, 'split': {'seed': 3407, 'train_class_ids': [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 29, 30, 32, 35, 36, 37, 38, 40, 41, 43, 44, 46, 47, 48, 49, 50, 54, 55, 56, 57, 58, 59, 61, 62, 63, 64, 66, 67, 68, 69, 70, 71, 72, 73, 74, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 87, 88, 89, 91, 92, 93, 94, 95, 96, 97, 98, 100, 101, 102, 103], 'train_photos': 58950, 'train_sketches': 46624, 'validation_class_ids': [7, 13, 16, 27, 28, 31, 33, 34, 39, 42, 45, 51, 52, 53, 60, 65, 75, 86, 90, 99]}, 'values': {'1': {'alignment': {'class_alignment_by_component': [0.867877185344696], 'class_alignment_gate_weighted': 0.867877185344696, 'class_alignment_max': 0.867877185344696, 'instance_residual_alignment_by_component': [-0.0036427516024559736], 'instance_residual_alignment_gate_weighted': 0.07534753531217575, 'instance_residual_alignment_max': 0.07534753531217575, 'instance_residual_alignment_max_by_component': [0.07534753531217575], 'instance_residual_alignment_mean': -0.0036427516024559736, 'photos_per_class': 8, 'seed': 3407}, 'checkpoint': 'outputs/experiments/deep_rho_learned/2026-09-02_13-41-26/checkpoints/transport_step73.pt', 'checkpoint_step': 73, 'run': 'outputs/experiments/deep_rho_learned/2026-09-02_13-41-26'}, '2': {'alignment': {'class_alignment_by_component': [0.07820858806371689, 0.911555826663971], 'class_alignment_gate_weighted': 0.1358644813299179, 'class_alignment_max': 0.911555826663971, 'instance_residual_alignment_by_component': [0.0008768975385464728, -0.004028778523206711], 'instance_residual_alignment_gate_weighted': 0.061882100999355316, 'instance_residual_alignment_max': 0.08645246922969818, 'instance_residual_alignment_max_by_component': [0.0610383078455925, 0.07459887117147446], 'instance_residual_alignment_mean': -0.0015759312082082033, 'photos_per_class': 8, 'seed': 3407}, 'checkpoint': 'outputs/experiments/deep_deterministic_k2/2026-09-02_14-57-35/checkpoints/transport_step73.pt', 'checkpoint_step': 73, 'run': 'outputs/experiments/deep_deterministic_k2/2026-09-02_14-57-35'}, '4': {'alignment': {'class_alignment_by_component': [0.3241286873817444, 0.9124507308006287, -0.07265489548444748, 0.2603572607040405], 'class_alignment_gate_weighted': -0.04689515382051468, 'class_alignment_max': 0.9124507308006287, 'instance_residual_alignment_by_component': [0.0002017904625972733, -0.0038212526123970747, 0.0009616282186470926, -0.0007192743360064924], 'instance_residual_alignment_gate_weighted': 0.06225363537669182, 'instance_residual_alignment_max': 0.12265148013830185, 'instance_residual_alignment_max_by_component': [0.10263849794864655, 0.07309524714946747, 0.061390470713377, 0.08250012248754501], 'instance_residual_alignment_mean': -0.00084427569527179, 'photos_per_class': 8, 'seed': 3407}, 'checkpoint': 'outputs/experiments/deep_deterministic_k4/2026-09-02_15-09-50/checkpoints/transport_step73.pt', 'checkpoint_step': 73, 'run': 'outputs/experiments/deep_deterministic_k4/2026-09-02_15-09-50'}, '8': {'alignment': {'class_alignment_by_component': [0.39481183886528015, -0.17594116926193237, 0.2626510262489319, -0.07058464735746384, 0.3832350969314575, 0.10728634893894196, 0.20783716440200806, 0.9065879583358765], 'class_alignment_gate_weighted': -0.17177419364452362, 'class_alignment_max': 0.9065879583358765, 'instance_residual_alignment_by_component': [-0.001018512761220336, -5.795827746624127e-05, -0.0012923640897497535, 0.0008401184459216893, -0.0013384779449552298, 0.00025092667783610523, 0.00031333649531006813, -0.0031325877644121647], 'instance_residual_alignment_gate_weighted': 0.0692773312330246, 'instance_residual_alignment_max': 0.18837611377239227, 'instance_residual_alignment_max_by_component': [0.1338246464729309, 0.06925623118877411, 0.1300130933523178, 0.061419323086738586, 0.12131715565919876, 0.10697273164987564, 0.10332801938056946, 0.07046222686767578], 'instance_residual_alignment_mean': -0.0006794397486373782, 'photos_per_class': 8, 'seed': 3407}, 'checkpoint': 'outputs/experiments/deep_deterministic_k8/2026-09-02_15-21-47/checkpoints/transport_step73.pt', 'checkpoint_step': 73, 'run': 'outputs/experiments/deep_deterministic_k8/2026-09-02_15-21-47'}}}`.
- The intended interpretation compares class alignment with gate-weighted and max instance-residual alignment using at least R=8 train photos per class.

## 14. Stability
| Run | Peak | Peak step | Late | Retention ratio | Absolute decay |
| --- | ---: | ---: | ---: | ---: | ---: |
| deep_seed123_rho_cosine75 | 0.6443 | 44 | 0.4584 | 0.7116 | 0.1858 |
| deep_seed123_K2 | 0.6432 | 44 | 0.4746 | 0.7379 | 0.1686 |
| deep_seed123_baseline | 0.6432 | 44 | 0.4617 | 0.7179 | 0.1815 |
| deep_multi_photo2 | 0.6423 | 73 | 0.4549 | 0.7082 | 0.1874 |
| deep_seed123_direction_none | 0.6423 | 44 | 0.4687 | 0.7298 | 0.1736 |
| deep_deterministic_k2 | 0.6421 | 73 | 0.4721 | 0.7354 | 0.1699 |
| deep_rho_cosine75 | 0.6420 | 73 | 0.4609 | 0.7179 | 0.1811 |
| deep_freeze_source73 | 0.6420 | 73 | not measured | not measured | not measured |
| deep_rho_learned | 0.6420 | 73 | 0.4547 | 0.7083 | 0.1873 |
| transport_endpoint_0 | 0.6420 | 73 | 0.4547 | 0.7083 | 0.1873 |
| deep_multi_photo8 | 0.6420 | 73 | 0.4507 | 0.7021 | 0.1913 |
| deep_rho_linear75 | 0.6419 | 73 | 0.4641 | 0.7229 | 0.1778 |
| deep_rho_fixed15 | 0.6416 | 73 | 0.4610 | 0.7186 | 0.1806 |
| deep_direction_class_centroid | 0.6415 | 73 | 0.4638 | 0.7230 | 0.1777 |
| transport_endpoint_0.1 | 0.6415 | 73 | 0.4637 | 0.7228 | 0.1778 |
| deep_multi_photo4 | 0.6414 | 73 | 0.4549 | 0.7092 | 0.1865 |
| deep_rho_zero | 0.6410 | 73 | 0.4604 | 0.7184 | 0.1805 |
| transport_factorial_base_text | 0.6409 | 73 | 0.4508 | 0.7034 | 0.1901 |
| deep_deterministic_k4 | 0.6407 | 73 | 0.4713 | 0.7356 | 0.1694 |
| deep_direction_fixed | 0.6392 | 73 | 0.4969 | 0.7773 | 0.1423 |
| deep_text_both | 0.6391 | 73 | 0.4505 | 0.7049 | 0.1886 |
| deep_direction_none | 0.6376 | 44 | 0.4908 | 0.7699 | 0.1467 |
| deep_freeze_reset | 0.6374 | 500 | 0.6337 | 0.9943 | 0.0036 |
| deep_freeze_preserve | 0.6371 | 500 | 0.6341 | 0.9952 | 0.0030 |
| deep_deterministic_k8 | 0.6368 | 44 | 0.4775 | 0.7499 | 0.1593 |
| deep_text_z0 | 0.6364 | 44 | 0.4544 | 0.7140 | 0.1820 |
| transport_endpoint_0.5 | 0.6340 | 73 | 0.4327 | 0.6826 | 0.2012 |
| transport_tangent_rho15_text_long5400_actual | 0.6196 | 73 | 0.4219 | 0.6810 | 0.1976 |
| deep_freeze_normal | 0.5785 | 500 | 0.4609 | 0.7966 | 0.1176 |
| deep_seed123_text_none | 0.5748 | 73 | 0.4244 | 0.7384 | 0.1504 |
- Peak and mAP@5400 are both reported; early-stopping selection is not called long-run stability.

## 15. Independent-Seed Replication
- Primary causal/probe sections remain restricted to seed 42; independent runs are not mixed into those contrasts.
| Seed | Control | Run | Peak mAP | Peak step | Late mAP |
| ---: | --- | --- | ---: | ---: | ---: |
| 123 | baseline_q_moving | `outputs/experiments/deep_seed123_baseline/2026-09-02_18-33-08` | 0.6432 | 44 | 0.4617 |
| 123 | rho_cosine_q_moving | `outputs/experiments/deep_seed123_rho_cosine75/2026-09-02_18-45-36` | 0.6443 | 44 | 0.4584 |
| 123 | K2_q_moving | `outputs/experiments/deep_seed123_K2/2026-09-02_18-57-59` | 0.6432 | 44 | 0.4746 |
| 123 | direction_none_q | `outputs/experiments/deep_seed123_direction_none/2026-09-02_19-10-44` | 0.6423 | 44 | 0.4687 |
| 123 | text_none_moving | `outputs/experiments/deep_seed123_text_none/2026-09-02_19-22-37` | 0.5748 | 73 | 0.4244 |
- Same-seed matched deltas (at the left run's peak checkpoint):
  - seed 123, rho_cosine_minus_learned: 0.0011 at step 44
  - seed 123, K2_minus_K1: 0.0001 at step 44
  - seed 123, moving_minus_no_direction: 0.0009 at step 44
  - seed 123, text_q_minus_no_text: 0.0779 at step 44

## 16. Refined SPICA Mechanism
- Evidence supports a trainable semantic-origin adaptation plus a bounded task-specific retrieval displacement; the completed matched probes do not support exact photo reconstruction as the default target.
- Text is a loss-only semantic adaptation signal; no text enters the predictor or inference.
- Exact photo endpoint matching is not treated as a retrieval objective when endpoint=0 wins the matched sweep.
- Direction and distance prediction are retained as hypotheses, not protected architectural commitments.

## Plots
- `outputs/causal_transport_decomposition.png`
- `outputs/endpoint0_factorial.png`
- `outputs/rho_schedule_ablation.png`
- `outputs/training_target_angle_histogram.png`
- `outputs/direction_supervision_ablation.png`
- `outputs/text_anchor_location.png`
- `outputs/hidden_space_compatibility.png`
- `outputs/freeze_optimizer_control.png`
- `outputs/query_gradient_conflict.png`
- `outputs/matched_K_ablation.png`
- `outputs/K_class_vs_instance_residual.png`
- `outputs/stability_retention.png`
- `outputs/seed_replication_controls.png`

## Raw Artifact Coverage
- Transport run artifacts inspected: 57.
- Official test used for selection: **NO**.

FINAL SPICA DEEP-PROBE VERDICT

Repository commit: ad78b664decd326d609ef26cb34e36c7c2337ae5
Working tree clean: YES

Broad corrected best pseudo-unseen mAP: 0.6423
Broad corrected best configuration: {'K': 1, 'batch_size': 32, 'direction_target': 'moving', 'encoder_learning_rate': 1e-05, 'encoder_lr': 1e-05, 'encoder_mode': 'partial', 'encoder_unfreeze_depth': 4, 'equivalent_epochs': 3.706245710363761, 'eval_batch_size': 128, 'fixed_rho_degrees': 15.0, 'freeze_encoder_at_step': None, 'frozen_parameters': 59496192, 'frozen_photo_encoder_parameters': 151277313, 'gradient_conflict_steps': [15, 73, 500], 'inference_score_mode': 'barycentric', 'lambda_cls': 1.0, 'lambda_dir': 1.0, 'lambda_dist': 1.0, 'lambda_endpoint': 0.0, 'lambda_geom': 0.0, 'lambda_rank': 1.0, 'lambda_vmf': 0.0, 'loss_profile': 'transport', 'model_family': 'predictive_semantic_transport', 'num_positive_photos': 2, 'objective': 'predictive_semantic_transport', 'photo_target': 'instance', 'predictor_learning_rate': 0.0001, 'predictor_lr': 0.0001, 'provenance': {'commit': '73ecaea34b43947c520092de1c08f6f5073da2ee', 'dirty_files': [' M configs/experiments/transport_factorial_transport_no_text.yaml', ' M configs/experiments/transport_factorial_transport_text.yaml', ' M configs/train_transport.yaml', ' M outputs/research_summary_transport_2026-09-02.json', ' M outputs/research_summary_transport_2026-09-02.md', ' M outputs/research_summary_transport_causal_2026-09-02.json', ' M outputs/research_summary_transport_causal_2026-09-02.md', ' M outputs/transport_K_ablation.png', ' M outputs/transport_direction_alignment.png', ' M outputs/transport_learning_curve.png', ' M outputs/transport_radius_vs_map.png', ' M outputs/transport_semantic_drift.png', ' M scripts/summarize_transport.py', ' M scripts/summarize_transport_causal.py', ' M src/spica/evaluate_transport.py', ' M src/spica/evaluation/transport.py', ' M src/spica/models/transport.py', ' M src/spica/train_transport.py', ' M tests/test_transport.py', '?? configs/experiments/transport_deterministic_k1_endpoint0.yaml', '?? configs/experiments/transport_deterministic_k2_endpoint0.yaml', '?? configs/experiments/transport_deterministic_k4_endpoint0.yaml', '?? configs/experiments/transport_deterministic_k8_endpoint0.yaml', '?? configs/experiments/transport_direction_class_centroid.yaml', '?? configs/experiments/transport_direction_fixed.yaml', '?? configs/experiments/transport_direction_moving.yaml', '?? configs/experiments/transport_direction_none.yaml', '?? configs/experiments/transport_freeze_optimizer_continue.yaml', '?? configs/experiments/transport_rho_cosine75.yaml', '?? configs/experiments/transport_rho_fixed15.yaml', '?? configs/experiments/transport_rho_learned.yaml', '?? configs/experiments/transport_rho_linear75.yaml', '?? configs/experiments/transport_rho_zero.yaml', '?? configs/experiments/transport_text_both.yaml', '?? configs/experiments/transport_text_none.yaml', '?? configs/experiments/transport_text_q.yaml', '?? configs/experiments/transport_text_z0.yaml', '?? outputs/research_summary_transport_deep_probe_2026-09-02.json', '?? outputs/research_summary_transport_deep_probe_2026-09-02.md', '?? scripts/summarize_transport_deep_probe.py', '?? scripts/transport_artifact_utils.py', '?? tests/test_transport_summary_selection.py'], 'working_tree_state': 'dirty'}, 'pseudo_train_classes': 84, 'pseudo_validation_classes': 20, 'pseudo_validation_seed': 3407, 'reset_optimizer_on_resume': False, 'resume_checkpoint_path': None, 'rho_max': 15.0, 'rho_max_degrees': 15.0, 'rho_strategy': 'learned', 'rho_warmup_steps': 75, 'scheduler': 'none', 'score_temperature': 0.07, 'seed': 42, 'shared_or_component_rho': 'shared', 'sketch_encoder_trainable_parameters': 28353024, 'steps': 5400, 'tau_cls': 0.07, 'text_conditioning': False, 'text_enters_predictor': False, 'text_loss_location': 'q', 'total_parameters': 88902913, 'train_class_scope': 'pseudo_train', 'trainable_parameters': 29406721, 'transport_enabled': True, 'transport_mode': 'tangent', 'transport_parameters': 1053697, 'unfrozen_block_count': 4, 'use_geometry_loss': False, 'use_text_cls': True, 'use_vmf': False, 'wandb_group': 'spica-transport', 'wandb_mode': 'disabled', 'wandb_project': 'spica', 'experiment_name': 'deep_multi_photo2', 'data_config': 'configs/data/sketchy_104_21.yaml', 'embedding_dir': 'outputs/sketchy_104_21/clip_openai_quickgelu', 'model_name': 'ViT-B-32-quickgelu', 'pretrained': 'openai', 'device': 'cuda', 'predictor_hidden_dim': 512, 'use_z0': False, 'rho_mode': 'shared', 'initial_rho_degrees': 0.5, 'alpha': 1.0, 'alpha_max': 0.5, 'initial_alpha': 0.0, 'min_kappa': 0.0001, 'max_kappa': 2048.0, 'initial_kappa': 64.0, 'num_workers': 4, 'pin_memory': True, 'drop_last': True, 'weight_decay': 0.0001, 'margin': 0.2, 'training_angle_diagnostic_max': 8192, 'prompt_template': 'a photo of a {}', 'assignment_temperature': 0.05, 'pseudo_val_num_classes': 20, 'pseudo_val_seed': 3407, 'max_steps': 5400, 'log_every': 100, 'probe_steps': [0, 15, 44, 73, 100, 500, 1000, 1800, 5400], 'run_probes': True, 'prediction_batch_size': 128, 'query_chunk_size': 256, 'precision_at_k': [1, 5, 10, 100, 200], 'map_at_k': [200], 'map_at_k_denominator': 'prefix_positive', 'save_optimizer': False, 'checkpoint_path': None, 'wandb_entity': None, 'wandb_run_name': None}
Strict R=1 default mainline mAP: 0.6420
Best strict R=1 rho-variant mAP: 0.6420
Broad-best checkpoint: /home/oslamelon/Desktop/Projects/spica/outputs/experiments/deep_multi_photo2/2026-09-02_15-55-47/checkpoints/transport_step73.pt

BASE / TRANSPORT CAUSAL DECOMPOSITION
Base-trained z0 peak mAP: 0.5678
Transport-trained z0 peak mAP: 0.6297
Transport q peak mAP: 0.6420
Encoder-training effect: 0.0619
Inference-head effect: 0.0122
Total transport-system effect: 0.0742

Base-trained z0 late mAP: 0.4200
Transport-trained z0 late mAP: 0.4229
Transport q late mAP: 0.4547

ENDPOINT
Best endpoint weight: 0.0000
Should endpoint loss remain: NO as a primary loss; endpoint=0 is the matched factorial condition

RHO
Learned-rho peak mAP: 0.6420
Fixed-15 peak mAP: 0.6416
Linear-warmup peak mAP: 0.6419
Cosine-warmup peak mAP: 0.6420
Zero-rho peak mAP: 0.6410
Does rho encode query-dependent distance: PARTLY, but it saturates near the 15-degree cap
Should distance head remain: NO by default; scheduled/zero controls match or nearly match learned rho
Best interpretation of rho: cosine_warmup has the highest matched pseudo-unseen peak (0.642049); learned rho is not required by this sweep

DIRECTION
No-direction mAP: 0.6376
Moving-direction mAP: 0.6420
Fixed-direction mAP: 0.6392
Class-centroid-direction mAP if available: 0.6415
Does explicit direction prediction help: YES provisionally: moving-target supervision changes peak mAP by +0.004403 versus no direction; this does not establish actual-photo direction causality
Is moving-frame alignment genuine but co-adapted: moving alignment exceeds fixed-origin alignment, so the moving metric is partly frame-dependent
Best direction supervision: moving

TEXT
CE(q) mAP: 0.6420
CE(z0) mAP: 0.6364
CE(both) mAP: 0.6391
No-text mAP: 0.5660
Best text-supervision location: q
Does text enter predictor: NO

HIDDEN FEATURE SPACE
CKA at peak: 0.8437
CKA late: 0.5870
Procrustes residual at peak: 0.3292
Procrustes residual late: 0.4786
Frozen-WCLIP compatibility at peak: 0.1823
Frozen-WCLIP compatibility late: -0.0310
Is encoder hidden space forgetting CLIP: YES; CKA/Procrustes drift is visible
Is frozen WCLIP becoming incompatible: YES; frozen projection compatibility declines late

FREEZE
Continue-normal mAP@5400: 0.4609
Freeze@73 mAP@5400: 0.6341
Optimizer-reset-only mAP@5400: 0.6337
Does encoder freezing causally help: freeze@73 changes late mAP by +0.173242 versus the optimizer-restored normal fork
Is optimizer reset a confound: YES unless optimizer state is restored

K
Deterministic K1 mAP: 0.6420
Deterministic K2 mAP: 0.6421
Deterministic K4 mAP: 0.6407
Deterministic K8 mAP: 0.6368
Best deterministic K: 2

K>1 class alignment: 0.1359
K>1 instance-residual alignment: 0.0619
What do extra directions represent: mostly class-semantic rather than instance-residual directions in the R=8 train-photo probe

Mo-vMF verdict: DEFER

STABILITY
Best peak mAP: 0.6443
Best late mAP: 0.4584
Retention ratio: 0.7116
Absolute decay: 0.1858

INDEPENDENT-SEED REPLICATION
Independent seeds measured: 123
Replication deltas are reported at matched checkpoints in the Independent-Seed Replication table above; these runs do not replace the seed-42 mainline.

Strongest supported SPICA mechanism: semantic-origin adaptation with a bounded task-specific displacement; exact direction/distance reconstruction is not assumed
Largest remaining confound: single independent seed; direction and K deltas are small relative to the seed-42 sweep
Most important next experiment: add more independent seeds before selecting a final rho schedule or K>1 model
Should SPICA predict the actual photo direction: NO requirement; moving-frame alignment is not sufficient evidence
Should SPICA predict actual photo distance: NO requirement; learned rho behaves largely as a trust-region cap
Should SPICA instead learn a bounded task-optimal retrieval displacement: YES as the working hypothesis

Recommended mainline architecture: CLIP-initialized sketch encoder -> frozen photo-compatible z0 -> minimal bounded sketch-only retrieval displacement; no text/photo at inference
Recommended mainline selection: retain strict R=1 until the R=2 multi-photo advantage survives independent seeds
Recommended mainline loss: rank loss plus loss-only semantic CE at the best validated anchor, endpoint loss disabled pending matched causal confirmation

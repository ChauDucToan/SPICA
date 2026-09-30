# Student–teacher: frozen seen-training diagnostic

## Kết quả

| Chỉ số | CNN–Mamba Student | CLIP LoRA + deep VPT teacher |
|---|---:|---:|
| Nearest sketch-centroid top-1 | **52.6948%** | **79.6753%** |
| Sketch → photo full mAP | **27.8722%** | **58.1448%** |
| mAP@200, denominator min(R,200) | 27.6305% | 58.0516% |
| P@200 | 3.4182% | 3.7857% |

Teacher cao hơn Student **26.9805 điểm %** ở centroid classification và **30.2726 điểm %** full retrieval AP. Student thắng8/220lớp ở centroid (12lớp hòa) và8/220lớp ở fullAP. Số đúng top1 được recompute chính xác từ feature CPU.

**Kết luận giới hạn:** Student không chỉ kém khi nối sketch với photo; nó cũng kém teacher ở khả năng phân tách nhãn giữa sketch trên bộ seen-training này. Do đó chưa có bằng chứng cho giả thuyết “Student xử lý sketch tốt hơn, chỉ thiếu semantic alignment”. Teacher có pretraining là giải thích khả dĩ, nhưng diagnostic không cô lập nguyên nhân đó. Nearest-centroid theo category vẫn không phải phép đo thuần hình học/nét vẽ; không chứng minh geometry processing, hiệu suất tính toán hay khả năng khái quát của model nào.

Agreement trên cùng3080queries: matched embedding cosine mean **.5287349**, photo-score Pearson mean **.6792067**, top200 overlap **54.0026%**, stable ordinal rank correlation mean **.6485357**. Ordinal correlation dùng tie-break theo gallery order, **không** phải tie-corrected Spearman.

## Phạm vi đã được user duyệt

- Frozen final10epoch deep-prompt checkpoint, không optimizer/backward/cập nhật encoder/classifier. Không ảnh official test mới, không W&B/new training/seed sweep.
- All15400TU220/30 **seen-train sketches**. Sort path từng class rồi localRandom42:56centroid +14query/class, tổng12320centroid/3080queries. Centroid là mean feature rồi normalize, so cosine nearest-centroid. Mỗi encoder có centroid bank trong own feature space, cùng nhãn và split.
- Photo gallery shared frozen CLIP, tối đa8photo train/class:219×8 + squirrel4 = **1756**. Manifest thực tế squirrel chỉ có4photo; user chọn giữ4, không lặp ảnh. AP dùng actual positives4/8. Queries14/class nên query-macro và class-macro tương đương, sai khác FP32 nhỏ.
- **Mọi sketch đã được dùng train encoder**, kể cả3080queries. Split chỉ tách centroid/query, không tạo independent encoder holdout. Photo gallery sampled1756, không so trực tiếp với official27989gallery.
- P200 tối đa4% cho219class,2% cho squirrel; AP informative hơn ở gallery này.

## Provenance và evidence

Implementation/launch commit **4230a2a** trên `feature/cnn-mamba-student`. Dependencies `jaxtyping==0.3.11`, `wadler-lindig==0.1.7`, isolatedStudent `.venv`; previous lock package/source unchanged. Models/loss/trainer chỉ thêm shape aliases/annotations và formatting; executable class/function AST identical khi bỏ annotations. Không runtime type-checker/test suite/validator trong Student model.

- Frozen train source **18ce61164fac4dde28cd14f94fe9cae5ae6c205b**; archived Student/teacher code loaded, all44archivedfiles hash-checked.
- Source run: `outputs/student_deepvpt_teacher_10epochs_20260928/`, original raw **FINISHED_UNVERIFIED** giữ nguyên.
- Checkpoint SHA256: `3809e4a5286d4a341565bd4c70d15cc5c267376fbe7ef4d82fc464694173d1f1`.
- Config SHA256: `7e49c84ccfb19fc1ec1da808a98e6fe52bc015082de56bf685458ed505edfd27`.
- Original CLIP SHA256: `e6d1bd7789aa45192b3bf90570a789b478bae1b74ebcce7eddd908e83a2b7c31`.
- Raw diagnostic: `outputs/student_semantics_diagnostic_20260928/`; `results.json`, saved Student/teacher/gallery features, `split_manifest.csv`, `per_class.csv`; adjacent launchreceipt/log. Exactdiagnostic sourcearchive in `diagnostic_source/`, training sourcearchive remains untouched. SHA bindings recorded in `results.json`.
- State/RNG before/after exact; all3models eval, no gradients. Actual image decode allowlist enforces onlyselectedseen-training entries. CUDA TF32 matmulfalse/cudnntrue; no bit-reproducible kernel claim. Safe `weights_only=True` checkpoint loading, strictStudent state + exact60teacher trainable adapterkeys.
- Readiness check/reviewer and AST/dependency audits: ignored `outputs/student_semantics_readiness_20260928/`; standalone check, not pytest.

## CSV incident and narrow repair

Independent CPU NumPyFP64 audit `outputs/student_semantics_review_20260928/` independently reconstructs manifests/Random42split, verifies feature/path/class/hash bindings, recomputes two centroid predictions and **all6160query-arm fullgallery sorts** plus perclass/aggregate/agreement. FullAP maxdelta Student **5.4646e-6**, teacher **7.7023e-8** versus raw, within predeclared2e-5; AP200 maxdelta3.33e-16; exact classification counts. All metadata/artifact/source/checkpoint hashes passed.

**Original audit remains FAIL:** per-query CSV writer parameter ordering swapped Pearson and overlap column values. Aggregate metrics/class stats/features were correct. Source signature fixed, tiny runnable CPU mapping check added to ignored readiness script; no rerun of images/encoders.

Use corrected **`outputs/student_semantics_corrected_20260928/per_query.csv`**, not the raw CSV, for those two agreement columns. Original raw CSV SHA `39d8ca19f192f44446c579502a892e073fb184e094aa7292bc7ae868226fc1bd` unchanged. Overlay swaps only two columns, every other field unchanged. Corrected SHA `0fe4af9dfa84bd0e03f32b75833a7ff09db79e8d80d48fb1b53a5d4686468f3c`.

Parent independent CPUFP64 recomputation of3080agreement rows passed: maxPearsondelta2.554e-7, overlap2.861e-8. Scoped receipt **PASS_CORRECTED_AGREEMENT_OVERLAY_ONLY** in correctedroot; original FAIL and production FINISHED_UNVERIFIED are not rewritten. No independent encoder replay/full certification; no further experiment/push authorized.

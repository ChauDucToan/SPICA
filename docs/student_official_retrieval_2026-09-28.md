# Frozen Student/teacher — official TU-Berlin retrieval

## Phạm vi

User chọn **chỉ retrieval chuẩn trên toàn bộ official test**, không centroid/few-shot classification. Checkpoint final10epoch deepVPT đã được chọn trước phép đo; giữ Student, teacher và frozen CLIP photo encoder ở eval. Không train/backward, optimizer, checkpoint selection, unseen text bank hay W&B mới.

- TU220/30: fine-tune trên220class, đánh giá30class khác tên hoàn toàn.
- **2400sketchqueries**,80/class; **27989photos**,401–2922photo/class, dùng toàn bộ officialgallery, không sample8/class.
- Với mỗi sketch, encoder chỉ nhận pixel; normalize embedding rồi rank toàn bộphoto theo cosine. Query/gallery labels chỉ dùng trong evaluator: cùngcategory là relevant, không phải cùnginstance/đúngnétvẽ.
- Cùngquery/gallery/order/preprocessing cho Student và teacher. Teacher giữ LoRAQ/V +4deepVPT/layer +exactwhitekeymask all12layers; original CLIP photo encoder không LoRA/prompts/mask.
- 30class không xuất hiện trong fine-tuning encoder của campaign, **không bảo đảm chúng chưa xuất hiện trong CLIP pretraining**. Officialtest đã được xem ở nhiều exploratory experiment; không gọi là blindholdout.

## Kết quả replay

| Metric | Student | Teacher |
|---|---:|---:|
| Full mAP | **27.2205%** | **56.6515%** |
| mAP@200, min(R,200) | 21.6423% | 56.7195% |
| P@200 | 29.1171% | 62.2287% |

Teacher hơn **29.4310điểm % fullmAP**. Teacher thắng2016/2400queries (**84%**), Student thắng384 (**16%**), khônghòa khi dùng |ΔAP|≤1e-6. Teacher có meanAPcao hơn ở **30/30class**; không có nghĩa teacher thắng mọiquery. Counts được xác nhận trên cả producerAP và independentFP64AP.

## Bài test có ý nghĩa gì?

Đây là **category-level zero-shot sketch-based image retrieval**: model đã học220loại đối tượng, nhưng khi gặp sketch thuộc30loại khác, nó phải tìm photo cùngloại trong một gallery lớn. Không đưa nhãn đúng hoặc unseen text prototype vào model để giúp tìm. Bài test đo chất lượng **xếp hạng sketch–photo ở category chưa dùng fine-tune**, kết hợp khả năng nhận dạng nét sketch với việc đặt embedding của nó gần photo đúng.

- **Full mAP:** với từngquery, tính Average Precision từ vị trí cácphoto relevant trên toàn bộ27989rank, rồi trung bình2400queries. AP thưởng cho việc relevantphoto xuất hiện sớm. **mAP27.22% không có nghĩa chỉ27.22%queries “được nhận dạng đúng”**, cũng không phải recall27.22%.
- **mAP@200:** chỉ cộng precision tại relevantpositions trong200rank đầu; chia min(R,200), R là sốphoto relevant thật của class. Ởgallerynày R≥401, denominatorluôn200. Chỉ số này quan tâm chất lượng vùngđầu và không bằng fullAP.
- **P@200:** tỉ lệphoto relevant trong200kết quả đầu, không quan tâm hoán vị bêntrong200. Student trung bình **58.23/200photo đúngclass**, teacher **124.46/200**. Đây không phải top1 classification accuracy.

## Điều gì có thể và không thể kết luận?

1. **Khoảng cách teacher–Student vẫn lớn trên class không dùng fine-tuning**, không chỉ trênseen-training diagnostic. Trong cấu hình/checkpoint hiện tại, teacher là hệ thống retrieval mạnh hơn rõ về numericalmetrics.
2. Kết hợp với centroidseen-train Student52.69%/teacher79.68%, dữ liệu **không ủng hộ** khẳng định “Student phân tích sketch tốt hơn, chỉ thiếu semantic transfer”. Student đang kém cả same-modality classseparation ởseenclasses lẫn crossmodal retrieval ởofficialunseenclasses.
3. **Không tách được nguyên nhân.** Teacher có CLIPpretraining, backbone/head/capacity khác; Student randominit2251782params học retrieval+cosineKD. Không phải matchedbackbone/pretraining-only ablation, nên không thể nói Mamba vốn kém, CNN xử lý nét kém, teacher dạy sai, hoặc KD/masking là nguyên nhân duy nhất. Có thể Student chưa chuyển đủ kiến thức/quan hệxếp hạng, nhưng phép đo không chứng minh cơ chế đó.
4. **Không phải geometryprobe**: relevant được định nghĩa bằng category, không bằng nét/pose/topology/instance. Không đo latency/robustness/corruption/generalization sangdatasetkhác; không significance/multiseed/SOTAclaim.
5. **Không so trực tiếp seen-train mAP27.87% với official27.22% để suy ra ít overfit**. Classset, queryset, gallery1756vs27989 và positivecount4/8vs401–2922 đều khác; đó không phải train/testgap dưới cùng distribution/gallery.

## Reproducibility và audit limits

Evaluation sourceHEAD **eb611aa859f284c8ecc103d39c4cde2b78b884e1**; producer operational script `outputs/student_official_replay_gate_20260928/evaluate.py` tái dùng loading/encoding/metrics hiện có. Source/docs freeze từlaunch đếnfinish. Adjacent launchreceipt PID5370 historical (đãfinish; không reusePID), không retrain/resume/automaticretry/push.

Training sourceHEAD **18ce61164fac4dde28cd14f94fe9cae5ae6c205b**, archivedmodelsource từ `outputs/student_deepvpt_teacher_10epochs_20260928/source/`.

- CheckpointSHA: `3809e4a5286d4a341565bd4c70d15cc5c267376fbe7ef4d82fc464694173d1f1`.
- ConfigSHA: `7e49c84ccfb19fc1ec1da808a98e6fe52bc015082de56bf685458ed505edfd27`.
- OriginalCLIPSHA: `e6d1bd7789aa45192b3bf90570a789b478bae1b74ebcce7eddd908e83a2b7c31`.
- Freshroot **`outputs/student_official_replay_20260928/`**: sourcearchive/evaluatorcopy/inputmanifest30389paths+producerimageSHA, Student/teacher/gallery features, perqueryAP/rank tensors, CSV/classCSV, `summary.json`/runtime. SummarySHA `90c635cffc72d9f433d2067824ea19cde533ad4c6071030c0b9b3a52d1151485`.
- **Allthree freshfeature tensors bitexact old**, samepaths/labels; sixmetricsdelta0; allperqueryAP/AP200/top200arrays alsoexactold. Fullmodelstatehashes/RNG unchanged, all_eval/no gradients. This reproduces the prior result, not a new statistically independent run.

Independent workerCPUaudit **`outputs/student_official_replay_review_20260928/`** performs4800NumPyFP64fullgallerysorts, verifies manifests/classes/source/checkpoint/config/CLIP/featureSHA/order and exactoldnewfeatures. FP64meanfullAP Student `.272204842106`, teacher `.566515153564`; allaggregate metrics within2e-6, P200 relevantcounts exactallrows. Imagebytes were not reopened inCPUaudit; imageSHAvalues remain producer evidence.

**Strictperquery audit FAIL retained**, tolerance2e-5unchanged: Student10AP200rows max3.88138e-5; Teacher2fullAProws max4.97406e-5 and12AP200rows max1.08696e-4 exceedtolerance. SomeFP32/FP64rankingorders differ; no claimexactranking/fullyverified. TeacherAP200aggregate discrepancy only1.54564e-7 (=.00001546điểm %), so roundedtable/order ofsystemmeans is stable under this independent computation. No rerun/toleranceraise to forcePASS.

Parent `parent_verify.py`/`parent_receipt.json` independently close **allCSV/classmeans/savedtop200AP200/P200** and actualSHA/oldnewperqueryarraychecks: AP200/CSV max3.33e-16,classmean5.55e-17; confirms2016/384querywins and30classwins. Scopedreceipt `SCOPED_BINDINGS_SAVED_RANK_METRICS_PASS_FP64_FAILURE_RETAINED` does not override workerFAIL.

Rawtraining **FINISHED_UNVERIFIED** and fresh **OFFICIAL_RETRIEVAL_FINISHED_UNVERIFIED** remain unchanged. No further experiments or push authorized.

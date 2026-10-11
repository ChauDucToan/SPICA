# Dạy model học chi tiết sketch: supervision trước, kiến trúc sau

Ngày: **2026-10-04**. Đây là phân tích và đề xuất nghiên cứu, **không phải authorization triển khai hoặc chạy experiment mới**.

## 1. Kết luận chính

> Muốn model phân biệt một chi tiết, cần dữ liệu và objective mà bỏ qua chi tiết đó sẽ làm dự đoán sai. Chỉ thêm CNN, Mamba, attention, prompts hoặc caption không bảo đảm điều này.

Lộ trình ít thay đổi nhất cho dự án:

1. xác định relevance: cùng **category**, đúng **instance**, hay đúng **part/attribute**;
2. kiểm tra positive/negative supervision có khớp relevance không;
3. nếu chỉ có category labels, bắt đầu bằng local self-supervision trên sketch, **không biến ảnh cùng lớp thành negatives**;
4. chỉ thêm instance loss và same-category hard negatives khi có correspondence hợp lệ;
5. giữ global retrieval làm anchor, đo detail bằng một diagnostic riêng trước khi tăng độ phức tạp.

Nền tảng: [điểm yếu VLM](vlm_weaknesses_before_after_clip_2026-10-01.md) và [CLIP–BLIP–BLIP-2](clip_blip_blip2_analysis_2026-10-04.md).

## 2. “Chi tiết” phải có định nghĩa có thể kiểm tra

| Loại thông tin | Ví dụ minh họa | Supervision/đánh giá tương ứng |
|---|---|---|
| Category semantics | Ghế thay vì xe | Class labels, category retrieval |
| Local shape | Chân cong thay vì chân thẳng | Pair/attribute labels xác nhận |
| Part configuration | Có/không tay vịn, số nan hoa | Part/count labels hoặc counterfactuals hợp lệ |
| Topology | Nét giao nhau hay chỉ chạm; đường kín hay hở | Synthetic geometric labels có ground truth |
| Instance identity | Đúng chiếc ghế được sketch mô tả | Sketch–photo instance pairing |
| Style robustness | Cùng vật thể, nét đậm/mảnh hoặc người vẽ khác | Identity-preserving views và đánh giá độc lập |
| Partial evidence | Thiếu một phần sketch | Partial-query protocol; không mặc định label vẫn đủ xác định instance |

Các ví dụ là mục tiêu đề xuất, không phải khẳng định TU-Berlin manifest hiện có những nhãn đó. Một model có thể rất tốt ở category retrieval nhưng chưa phân biệt instance; ngược lại, sensitivity với một nét bị xóa không tự chứng minh nét đó được hiểu đúng.

## 3. Repository hiện đã có gì, thiếu supervision gì?

### 3.1 Không phải thiếu local computation hoàn toàn

`src/spica/models/sketch_student.py` đã có:

- CNN xử lý ink từ độ phân giải 224×224 trước khi downsample;
- feature grids qua 112×112, 56×56 và 28×28;
- hai hybrid blocks, mỗi block kết hợp local convolution với shared-weight four-direction Mamba-2 scans;
- phục hồi tọa độ XY sau scans;
- foreground-weighted pooling, projection và normalized 512-D output.

Model là random-init CNN–Mamba, không phải VMamba SS2D nguyên bản. Trần tiềm năng cần kiểm tra là **supervision/readout cuối có sử dụng detail không**, chứ không thể suy từ tên architecture rằng model đã giỏi hoặc không có local features.

SketchLVM hiện có separate visual branches, shallow prompts và trainable LayerNorms trên CLIP nền. Khả năng adaptation này khác với học lại toàn bộ visual encoder; song không có bằng chứng ở đây rằng số prompt hay frozen weights là nguyên nhân chính của gap.

### 3.2 Category positive không phải instance positive

`src/spica/data/datasets.py` lấy positive photo ngẫu nhiên trong cùng class. `src/spica/train_student.py::multi_positive_loss` đặt positive mask bằng:

```text
positive(s, p) = [class(s) == class(p)]
```

SketchLVM trainer cũng dùng category-level triplet và classification targets. Không có contract hiện tại nói rằng photo positive là đúng vật thể mà sketch mô tả.

Ví dụ: sketch ghế có tay vịn và ảnh ghế không tay vịn vẫn có thể là **category positive hợp lệ**. Loss category không trực tiếp thưởng việc đẩy hai kiểu tay vịn ra xa nhau. Điều này không làm loss “sai”: nó phù hợp category-level ZS-SBIR, nhưng không đủ để tuyên bố instance/detail matching.

**Không làm:** thêm ảnh khác cùng class vào negatives của loss category. Đó là false negatives theo protocol hiện tại. Nếu có instance labels, dùng một objective/head detail riêng và đánh giá trade-off với category objective.

### 3.3 Bằng chứng địa phương cần giữ đúng phạm vi

- [Seen-training diagnostic](student_seen_semantics_2026-09-28.md): Student/teacher centroid top-1 **52.6948%/79.6753%**, full retrieval mAP **27.8722%/58.1448%**. Tất cả sketches đã được encoder thấy khi train. Chưa có bằng chứng Student xử lý geometry tốt hơn teacher; pretraining là một khả năng giải thích, không phải nguyên nhân đã được cô lập.
- [SketchLVM public-input report](sketchlvm_public_inputs_gap_and_masking_2026-10-01.md): clean full mAP **52.272251%**, chưa khớp paper **65.10%** và chưa xác định nguyên nhân chính. Masking row-major ink làm AP giảm, nhưng không phải semantic-part intervention hoặc chứng minh model chỉ nhớ average shape.
- Original runs vẫn **FINISHED_UNVERIFIED**; scoped CPU checks không thay raw failure/status lịch sử. Tài liệu này không thêm replay, training hay numerical certification.

## 4. Ưu tiên 1 — dữ liệu đúng với mục tiêu

### 4.1 Nếu chỉ có class labels như contract TU hiện tại

Có thể học:

- category retrieval bằng loss hiện có;
- sketch-only local consistency giữa hai views của **cùng ảnh**;
- reconstruction trên sketch train;
- synthetic geometric tasks có ground truth tự tạo, nếu dữ liệu/protocol được duyệt.

Không tự tạo instance pairing bằng cách “ảnh cùng lớp có CLIP score cao nhất”. Đây chỉ là pseudo-pair và có thể chọn ảnh đúng nghĩa nhưng sai hình dạng. Nếu dùng sau này, phải ghi độ tin cậy và đo độ chính xác trên mẫu được kiểm tra; không gọi là ground truth.

### 4.2 Nếu muốn instance-level detail

Cần manifest có ít nhất:

```text
sketch_path, positive_photo_id, instance_id, class_id, split
```

Một sketch có thể có nhiều positive photos nếu cùng instance hoặc relevance annotation cho phép. Phải giữ các sketches/photos cùng instance trong cùng split khi mục tiêu là unseen-instance generalization; split theo file có thể làm rò rỉ identity.

Không suy correspondence từ category name. Không mặc định mọi bộ Sketchy/local copy đã giữ nguyên pair metadata: phải xác minh mapping trên release thực tế. Tài liệu này chưa xác minh original Sketchy pairing và không lấy số liệu dataset từ search snippets.

### 4.3 Annotation nhỏ nhưng có giá trị

Nếu chưa thể xây bộ instance lớn, nên ưu tiên một bộ diagnostic nhỏ có part/attribute labels được người đánh giá xác nhận thay vì chỉ thêm captions dài không kiểm tra. Chọn thuộc tính **thực sự nhìn thấy trong sketch**, chẳng hạn số bộ phận hoặc hình dạng contour; không buộc model suy màu/texture vốn không được vẽ.

Mọi nhãn sketch mới dùng cho train/selection phải thuộc phần train/development được duyệt. Không xem thêm official test để chọn dạng augmentation, loss, negatives hoặc coefficients.

## 5. Ưu tiên 2 — giữ detail bằng supervision tối thiểu

### 5.1 Local consistency trên cùng sketch: lựa chọn đầu tiên nếu thiếu pair labels

Với hai transformations nhẹ `a`, `b` của một sketch train, tọa độ correspondence được biết từ transformations. So sánh local descriptors ở cùng vị trí vật thể sau mapping:

```text
L_local = mean_(x in visible_overlap)
          [1 - cosine(h_a(a(x)), stop_gradient(h_b(b(x))))]
```

Đây là **công thức đề xuất**, chưa triển khai. Chỉ dùng overlap còn nhìn thấy ở cả hai views; không ép patch bị xóa hoàn toàn phải giống patch gốc như thể evidence không thay đổi.

Trần của cách này:

- cosine consistency đơn độc có nghiệm collapse: mọi descriptors bằng nhau;
- global retrieval loss không chứng minh local head không collapse, đặc biệt nếu head tách riêng;
- teacher/self-distillation hoặc spatially discriminative targets phải có biện pháp chống collapse và kiểm tra descriptor variance/correspondence accuracy;
- neighboring/repeated parts không được tùy tiện coi là negatives khác nghĩa.

DINO là nguồn tham khảo cho self-distillation và visual features [R3], **không phải bằng chứng công thức trên tự đủ hoặc DINO bảo đảm sketch topology**. Không viết mới một DINO framework chỉ để thử ý tưởng; tái dùng feature grid hiện có trước.

### 5.2 Sketch reconstruction: chọn target tránh shortcut nền trắng

MAE chứng minh masked pixel reconstruction là một hướng pretraining hiệu quả cho ảnh [R2]. Khi áp dụng vào raster sketch, nền trắng chiếm phần lớn diện tích; loss trung bình toàn ảnh có thể bị tối ưu bằng việc dự đoán trắng.

Đề xuất nếu được duyệt:

- mask có kiểm soát lượng foreground và background;
- đo reconstruction error riêng trên ink và background;
- có thể dùng foreground-weighted loss, ví dụ với target ink `k(x)`:

```text
w(x) = 1 + α · k(x), α > 0
L_rec = Σ_(x in masked) w(x) · |predicted_ink(x) - k(x)|
        / Σ_(x in masked) w(x)
```

Weights lấy từ train target, không từ query label lúc inference. Kiểm tra baseline “luôn trắng” và không chọn α bằng official test. Reconstruction là auxiliary task, không phải metric thay thế retrieval; kết quả MAE với 75% random patches trên ảnh **không cấp phép** mặc định xóa 75% ink sketch.

### 5.3 Instance contrastive/ranking: chỉ khi có pairing hợp lệ

Với `P(s)` là positives đã xác nhận, và candidates gồm ảnh đúng lẫn cùng-category wrong instances:

```text
L_instance = -log [Σ_(p in P(s)) exp(score_detail(s,p)/τ)
                  / Σ_(p in candidates) exp(score_detail(s,p)/τ)]
```

Điều kiện quan trọng:

- cùng-category negative phải **sai instance/attribute theo nhãn**, không chỉ là file khác;
- loại duplicate photos, alternate views và ambiguous matches khỏi negatives;
- mine negatives chỉ trong train pool được duyệt;
- bắt đầu bằng negatives vừa khó, không chọn duy nhất ảnh khó nhất chưa kiểm tra vì dễ chứa false negatives.

ARO cho thấy composition-aware hard negatives có ích trên benchmark của nó [R4]. Đó là động lực thử nghiệm, không phải bảo đảm cách mining bất kỳ sẽ cải thiện FG-SBIR.

### 5.4 Không gộp mọi objective ngay từ đầu

Viết mục tiêu theo khả năng dữ liệu:

```text
Chỉ category labels:
    L = L_category + λ_local · L_local
    hoặc thử L_rec như một alternative riêng

Có verified instance labels:
    L = L_category + λ_instance · L_instance
    thêm local auxiliary chỉ khi một phép đo cho thấy cần
```

Không đưa coefficients “tối ưu” khi chưa đo. Nếu cần cân bằng, đo loss/gradient trên train-development đã duyệt, ghi protocol trước và tránh official-test tuning. Global cosine KD truyền semantic anchor, không tự cung cấp part correspondence; local KD chỉ hợp lý khi teacher descriptors và matching đã được kiểm tra.

## 6. Ưu tiên 3 — tận dụng representation hiện có trước khi đổi backbone

### 6.1 Đường tối thiểu cho Student

**Đề xuất:** dùng feature grid 28×28 trước foreground pooling cho auxiliary detail supervision. Giữ output 512-D và đường retrieval hiện tại làm baseline; không cần ngay một backbone lớn hơn, transformer decoder, graph hoặc nhiều teacher.

Nếu phép đo cho thấy grid 28×28 không giữ detail cần thiết, mới cân nhắc một skip từ feature map 56×56 hoặc thay đổi downsampling. Chưa có bằng chứng rằng chỉ tăng resolution sẽ sửa semantic/instance ambiguity; ghi rõ compute và pretraining budget khi so sánh.

Foreground masking có thể giảm ảnh hưởng nền, nhưng:

- trắng thật, padding và anti-aliased pixels là những khái niệm khác nhau;
- dense attention key masking không mặc nhiên tăng tốc;
- bỏ/sắp lại tokens phải giữ vị trí gốc khi phương pháp dùng spatial structure;
- raster scan không phải thứ tự pen strokes; không suy stroke order từ row-major pixels.

### 6.2 Photo representation cũng phải đủ phân biệt

Nếu hai photos khác nhau ở detail nhưng gần như cùng embedding, chỉ làm sketch encoder nhạy hơn chưa chắc đổi được ranking. Cần kiểm tra photo descriptors trên một train/development set có nhãn phù hợp trước khi quy mọi lỗi cho query encoder.

CLIP global photo vectors có thể giữ làm semantic anchor. Nếu local visual matching thực sự cần thiết, dùng photo feature maps/tokens và adaptation riêng; đó là thay đổi representation và cần báo cáo, không còn là matched sketch-backbone-only ablation.

### 6.3 Global-first, local-second chỉ thêm khi phép đo yêu cầu

Một phương án gallery lớn:

```text
sketch -> global score -> top-K photo candidates
       -> local/detail matching trên candidates -> rerank
```

Không bắt buộc correspondence cùng tọa độ XY giữa sketch và photo vì free-hand sketches có thể lệch hình và bố cục. Spatial-semantic FG-SBIR đã nghiên cứu kết hợp coarse/fine information và robustness với misalignment [R1, PDF pp.1–3]. Không cần sao chép toàn bộ architecture/loss của paper để thử local features.

Trước khi thêm reranker phải đo **candidate recall@K** và local matching quality. Part/token similarity thuần max có thể match lặp một photo part cho nhiều sketch parts; foreground, geometry và ambiguity cần được kiểm tra. Chưa có latency/memory benchmark thì không tuyên bố nhanh hơn.

### 6.4 Giữ primary protocol và contribution

Local features có thể là đầu vào cho sketch-conditioned Mo-vMF head theo định hướng repository; mixture giữ one-to-many hypotheses nhưng **không tự tạo nhãn detail**. Không đổi primary method thành unseen-text classifier hoặc tái mở Agreement Gate đã bị đóng trong dự án.

Inference vẫn chỉ dùng sketch, gallery và learned parameters. Category/instance/part labels chỉ dùng ở train/evaluator theo protocol. CLIP/BLIP/BLIP-2 teacher có thể training-only; caption teacher không phải unseen-text bank ở inference.

## 7. Augmentation: học cái gì nên bất biến, cái gì phải nhạy?

| Biến đổi đề xuất | Mục tiêu có thể hợp lệ | Rủi ro cần kiểm tra |
|---|---|---|
| Dịch chuyển/scale nhẹ giữ foreground | Bất biến vị trí/kích thước canvas | Crop làm mất part hoặc thay tỷ lệ |
| Thay độ đậm nét nhẹ | Style robustness | Mất junction/đường kín nếu quá mạnh |
| Hai crops có overlap xác định | Local correspondence | Không dùng vùng chỉ xuất hiện ở một view làm positive |
| Drop nét/mask nhẹ | Partial-query robustness | Detail identity có thể đã mất; không mặc định instance vẫn xác định |
| Flip/rotation | Chỉ nếu relevance cho phép | Thay handedness, orientation hoặc quan hệ trái/phải |
| Edge maps từ photo train | Một auxiliary view/transfer signal | Photo edges không phải free-hand sketch; texture edges có thể là shortcut |

**Nguyên tắc:** bất biến với style không đồng nghĩa bất biến với mọi thay đổi cấu trúc. Nếu xóa tay vịn là thay đổi discriminative attribute, ép embedding hoàn toàn không đổi có thể dạy model bỏ qua chính detail cần học.

Counterfactual phải có nhãn và tính hợp lệ được kiểm tra. Không sửa một nét rồi tự kết luận ảnh gốc là negative khi sketch mới còn mơ hồ hoặc chưa có photo counterpart đúng.

## 8. Đánh giá để không nhầm “nhạy với nét” thành “hiểu nét”

### 8.1 Tách các câu hỏi

| Phép đo đề xuất | Trả lời được | Không trả lời được một mình |
|---|---|---|
| Category mAP@all/AP@200/P@K | Category retrieval theo relevance đã khai báo | Đúng instance/part |
| Verified instance Recall@K/rank | Tìm đúng object instance | Generalization khi split đã lộ instance |
| Attribute/counterfactual preference | Score đổi đúng hướng khi detail đổi | Retrieval toàn gallery |
| Synthetic topology accuracy | Có nhận ra quan hệ hình học có nhãn | Transfer sang human sketches |
| Style perturbation stability | Robust với nuisance hợp lệ | Model có dùng discriminative detail không |
| Local correspondence + variance | Descriptor matching và collapse | Global semantic transfer |

Đối với within-category diagnostic, class label có thể dùng để lọc candidates **trong evaluator**; phải gọi đây là diagnostic có điều kiện, không biến class filtering thành đường inference standard SBIR.

### 8.2 Detail intervention cần control

Nếu có part labels, có thể so:

1. original sketch;
2. bỏ discriminative part;
3. bỏ vùng control có ink mass tương đương nhưng không phải part đó;
4. đổi style nhẹ nhưng giữ cấu trúc.

Theo dõi paired score margins giữa verified correct và hard-negative photos, ranks và per-query metrics. AP giảm ở (2) chưa đủ: cần xem có giảm nhiều hơn control và đúng loại nhầm lẫn mong đợi không. Matching ink mass không loại hết confound về vị trí/hình dạng; không gọi đây là causal proof toàn diện.

Masking row-major 25/50/75% đã có trong dự án là stress test lịch sử, **không thay cho semantic-part control**. Không sửa định nghĩa, replay hoặc nâng mức chứng nhận các kết quả đó.

### 8.3 Protocol và numerical hygiene

Nếu experiment được duyệt sau này:

- chọn hyperparameters bằng train-development; split development phải tách khỏi encoder training nếu muốn đo generalization thật;
- giữ official unseen test final-only; những kết quả đã xem không còn là blind holdout cho các quyết định tiếp theo;
- ghi manifest/pair bindings, checkpoint, code, preprocessing, gallery và relevance;
- khai báo AP denominator, cutoff, precision cutoff, ties và gallery order;
- report per-query/per-class results, không chỉ một mean;
- nhiều seeds/uncertainty nếu muốn claim ổn định hoặc significance, chỉ sau authorization riêng;
- không bỏ qua FAIL hoặc tự tăng tolerance để một kết quả thành PASS.

## 9. Thứ tự thử nghiệm nhỏ nhất — chỉ là kế hoạch

**Bước A: supervision/readout audit.** Xác định class-only hay verified pairs; tạo một diagnostic detail bằng train/development phù hợp. Không chạy thêm official test để tìm ý tưởng.

**Bước B: một thay đổi.** Nếu class-only: category baseline → cùng encoder/budget + một local auxiliary được chống collapse, hoặc reconstruction trong một arm thay thế. Nếu verified pairs: ưu tiên instance loss + labeled same-category negatives trước khi đổi backbone.

**Bước C: chỉ tăng local capacity khi có bằng chứng.** Feature grid hiện có không đủ mới thêm một higher-resolution skip hoặc adaptation của photo representation. Không đồng thời thay backbone, teacher, loss và preprocessing.

**Bước D: reranking hoặc thêm teacher lớn khi còn bottleneck cụ thể.** Giữ protocol sketch-only; đo candidate recall, memory và latency. BLIP/BLIP-2 là lựa chọn cần kiểm chứng, không lời giải mặc định.

Giữ fixed seed/budget/preprocessing/gallery khi có thể; dữ liệu supervision thêm hoặc pretraining mới là confound phải ghi rõ. Một arm thăm dò không chứng minh lợi ích bền vững. **Chưa có bước nào được triển khai/chạy bởi tài liệu này.**

## 10. Nguồn và giới hạn bằng chứng

- **[R1] Song et al., Deep Spatial-Semantic Attention for Fine-Grained Sketch-Based Image Retrieval**, ICCV 2017: https://openaccess.thecvf.com/content_ICCV_2017/papers/Song_Deep_Spatial-Semantic_Attention_ICCV_2017_paper.pdf — đọc PDF pp.1–6; dùng cho bài toán instance retrieval, local detail và cross-domain misalignment, không lấy kết quả của paper làm dự báo gain cho Student.
- **[R2] He et al., Masked Autoencoders Are Scalable Vision Learners**, 2021/2022: https://arxiv.org/abs/2111.06377v3 — nguồn gốc cho masked reconstruction; foreground-aware sketch formulation phía trên là đề xuất, không phải method được paper xác nhận.
- **[R3] Caron et al., Emerging Properties in Self-Supervised Vision Transformers (DINO)**, 2021: https://arxiv.org/abs/2104.14294v2 — nguồn gốc cho self-distillation và tính chất visual features; không chứng minh sketch part correspondence.
- **[R4] Yuksekgonul et al., ARO / bags-of-words**, ICLR 2023: https://arxiv.org/abs/2210.01936v3 — evidence về compositional shortcuts và hard negatives trong task được nghiên cứu.
- CLIP, BLIP, BLIP-2: method/retrieval/limitations trong các PDF gốc được dẫn ở [tài liệu so sánh](clip_blip_blip2_analysis_2026-10-04.md).
- Bằng chứng local: [Student seen semantics](student_seen_semantics_2026-09-28.md), [SketchLVM clean gap/masking](sketchlvm_public_inputs_gap_and_masking_2026-10-01.md); source contracts tại `src/spica/data/datasets.py`, `src/spica/models/sketch_student.py`, `src/spica/train_student.py`, `src/spica/train_sketchlvm.py`.

MAE/DINO/ARO được kiểm tra trang nguồn gốc để xác nhận mục tiêu và phạm vi; không tuyên bố đã rà toàn bộ implementation hoặc có benchmark sketch mới. Nguồn Sketchy gốc chưa truy cập đầy đủ, nên không dựa vào pairing/số liệu chưa xác minh. Không có code/dependency/model/artifact changes, training, encoder replay hoặc claim performance improvement trong công việc này.

# CLIP, BLIP và BLIP-2: kiến trúc, objective và ý nghĩa với sketch

Ngày: **2026-10-04**. Phân tích các paper gốc, không phải bảng xếp hạng các checkpoint hiện hành.

## 1. Tóm tắt quyết định

- **CLIP:** lựa chọn đơn giản cho global embeddings và semantic anchor; objective gốc không trực tiếp giám sát local correspondence.
- **BLIP:** kết hợp alignment, fusion và generation; có retrieval bằng candidate selection + ITM reranking, không cần sinh caption để retrieval.
- **BLIP-2:** dùng Q-Former để tái sử dụng vision encoder và LLM. **Nhánh image–text retrieval không cần LLM**, và paper fine-tune cả vision encoder lẫn Q-Former ở nhánh đó.
- Với sketch detail, câu hỏi quyết định là **dữ liệu/loss có yêu cầu phân biệt chi tiết đó không**, không chỉ “model nào mới hơn”.

Đọc kèm [tổng quan điểm yếu VLM](vlm_weaknesses_before_after_clip_2026-10-01.md) và [đề xuất học sketch detail](sketch_detail_learning_recommendations_2026-10-04.md).

## 2. CLIP: global alignment bằng dual encoder

### 2.1 Kiến trúc

```text
image -> image encoder -> linear projection -> normalized image embedding
text  -> text encoder  -> linear projection -> normalized text embedding
                         similarity + learned logit scale
```

Paper gốc thử cả modified ResNet và ViT cho image encoder, dùng Transformer cho text. ViT dùng global CLS readout; ResNet dùng attention pooling. Vì vậy “CLIP” không đồng nghĩa với duy nhất ViT-B/32 hay một patch size [C, PDF pp.4–5].

Text representation được lấy tại end-of-text token. Không có cross-attention giữa image tokens và text tokens trong dual-encoder forward gốc. Image/text encoder được huấn luyện, không phải mặc định frozen trong pretraining; downstream có thể freeze hoặc fine-tune tùy phương pháp.

### 2.2 Objective

Với batch gồm N cặp đúng `(I_i, T_i)`, gọi `u_i`, `v_j` là embeddings đơn vị:

```text
S_ij = β · u_iᵀ v_j,        β > 0 là learned logit scale
L_image = -(1/N) Σ_i log [exp(S_ii) / Σ_j exp(S_ij)]
L_text  = -(1/N) Σ_j log [exp(S_jj) / Σ_i exp(S_ij)]
L_CLIP  = (L_image + L_text) / 2
```

Mỗi cặp trên đường chéo là positive; các ghép khác trong batch là negatives theo objective gốc. Hai caption/ảnh khác nhau có thể vẫn tương đương về nghĩa, nên batch construction này có rủi ro false negatives; đó là suy luận từ nhãn pairing, không phải định lượng lỗi trong run hiện tại.

Paper dùng khoảng **400 triệu cặp image–text web**, train các encoder từ đầu và không dùng objective sinh caption trong mô hình CLIP cuối [C, PDF pp.3–5].

### 2.3 Inference và khả năng cache

- Classification: encode text prompts cho candidate classes, so similarity với ảnh.
- Image–text retrieval: encode hai bên độc lập, rank bằng similarity.
- Sketch–photo retrieval dùng image encoder cho cả hai domain là một **ứng dụng/adaptation**, không phải objective image–text gốc.

Có thể cache gallery vectors và dùng matrix multiplication/vector search. Không cần text bank trong đường sketch–photo retrieval nếu phương pháp định nghĩa query và gallery đều bằng visual features.

### 2.4 Điểm mạnh và trần supervision

**Điểm mạnh:** pretraining rộng, embeddings dùng lại thuận tiện, inference không cần fusion từng cặp. CLIP không phải chỉ học tên lớp: caption web có thể chứa thuộc tính và quan hệ phong phú.

**Giới hạn thiết kế:** loss nhìn vào global vectors. Nó không cung cấp dense labels như “patch này là chân trái của chiếc ghế, tương ứng với stroke kia”. Một chi tiết có thể có trong patch representation nhưng không được global similarity ưu tiên.

**Bằng chứng trong paper:** zero-shot CLIP còn yếu ở một số fine-grained tasks, counting và distribution shifts [C, PDF pp.19–20]. Không suy rộng thành “CLIP không biết hình học” hoặc “mọi fine-grained task đều thất bại”.

**Riêng ViT-B/32:** ảnh 224×224 tạo 49 patch tokens cộng CLS. Patch embedding vẫn nhận các pixel bên trong từng patch; không có cơ sở nói tất cả nét nhỏ hơn 32 pixel bị xóa. Muốn xác định bottleneck phải kiểm tra input raster, intermediate features và task cần detail.

## 3. BLIP: một framework cho understanding và generation

### 3.1 Multimodal Mixture of Encoder–Decoder (MED)

BLIP dùng visual Transformer và text Transformer có ba chế độ [B, §3.1–3.2, PDF p.3]:

| Chế độ | Image–text tương tác | Đầu ra / mục tiêu |
|---|---|---|
| Unimodal encoders | Hai bên encode độc lập | Embeddings cho ITC |
| Image-grounded text encoder | Text cross-attend vào image features; bidirectional self-attention | Pair representation cho ITM |
| Image-grounded text decoder | Cross-attention vào ảnh; causal self-attention | Sinh text bằng LM |

Text encoder/decoder chia sẻ các tham số ngoại trừ self-attention layers trong thiết kế paper. Không nên nhầm việc chia sẻ weights giữa các chế độ với “mọi downstream task chạy cùng một forward”.

### 3.2 Ba objective và chúng thực sự dạy gì

**ITC — Image–Text Contrastive:** đưa image/text representations về không gian tương thích. BLIP dùng momentum encoder và soft targets để xử lý các potential positives trong những cặp được coi là negatives.

**ITM — Image–Text Matching:** binary classification xem cặp có khớp không sau fusion. Negatives có contrastive similarity cao được ưu tiên chọn. Điều này buộc model kiểm tra cặp khó hơn, không chỉ phân biệt hai caption hoàn toàn khác chủ đề.

**LM — Language Modeling:** autoregressive cross-entropy cho caption dựa trên ảnh. Đây là objective sinh text, không phải masked-language modeling thông thường.

Tổ hợp này cung cấp nhiều kiểu tín hiệu hơn CLIP gốc. Tuy nhiên, **ITM là nhãn cấp cặp; LM là nhãn cấp chuỗi text**. Nếu caption không nói kiểu tay vịn, loss không trực tiếp yêu cầu nhận diện kiểu tay vịn.

### 3.3 CapFilt: sửa dữ liệu thay vì chỉ sửa model

Luồng của paper [B, §3.3, PDF p.4]:

```text
pretrained MED
  -> captioner: fine-tune LM trên COCO, sinh captions cho web images
  -> filter: fine-tune ITC/ITM trên COCO, lọc caption web và caption tổng hợp
filtered web pairs + filtered synthetic pairs + human-annotated pairs
  -> pretrain một model mới
```

Captioner và filter được fine-tune riêng. Paper khảo sát việc chia sẻ tham số và cho rằng confirmation bias làm giảm hiệu quả lọc [B, PDF p.6]. BLIP báo cáo các cấu hình dữ liệu **14M và 129M images**; không gộp mọi variant thành một ngân sách duy nhất.

**Hàm ý:** chất lượng dữ liệu là một trục cải thiện thực sự. Nhưng captioner/filter đều có sai số; không coi caption sinh ra là ground truth cho thuộc tính sketch chưa được kiểm tra.

### 3.4 Retrieval không phải caption generation

BLIP paper fine-tune retrieval bằng ITC và ITM. Inference [B, §5.1, PDF p.6]:

1. dùng image–text feature similarity chọn top-k;
2. rerank các cặp đã chọn bằng ITM.

Paper đặt k=256 trên COCO và k=128 trên Flickr30K. Đây là setting của paper, **không phải đề xuất mặc định k cho gallery sketch của dự án**.

ITC embeddings vẫn có thể phục vụ candidate generation; ITM chỉ chạy trên subset. Kết quả của hệ hai tầng phụ thuộc cả candidate recall và reranker: candidate đúng bị loại ở tầng đầu thì tầng sau không thể cứu.

### 3.5 Ý nghĩa với sketch

BLIP có thể cung cấp visual features, caption teacher hoặc ý tưởng global-then-local reranking. Nhưng ITM gốc nhận **ảnh + text**, không nhận **sketch + photo**. Muốn chuyển thành visual-pair reranker phải thiết kế/adapt và kiểm tra riêng; không thể chỉ đưa hai ảnh vào head ITM rồi tuyên bố đúng task.

## 4. BLIP-2: tái sử dụng frozen models qua Q-Former

### 4.1 Q-Former là gì?

Q-Former là trainable Querying Transformer khởi tạo từ BERT-base; cross-attention layers được khởi tạo mới. Learned queries tương tác với visual features qua cross-attention mỗi hai blocks và tương tác với nhau qua self-attention [B2, §3.1, PDF pp.2–3].

Setting của paper:

- **32 learned queries**, hidden dimension **768**;
- khoảng **188M tham số Q-Former**;
- query outputs có kích thước `32×768`, thay vì toàn bộ image feature sequence, ví dụ `257×1024` cho ViT-L/14 được paper nêu.

“Lightweight” là tương đối so với backbone/LLM nhiều tỷ tham số. Q-Former này **không nhỏ theo cùng nghĩa** với Student khoảng 2.25M tham số hoặc prompt/LN adaptation của dự án.

Queries là learned model parameters, không phải 32 câu hỏi do người dùng nhập, không mặc định là 32 object slots hay 32 parts có nhãn.

### 4.2 Stage 1: representation learning

```text
image -> frozen vision encoder -> visual token sequence
                                -> Q-Former learned queries -> query outputs
text ---------------------------> Q-Former text branch
```

Ba objectives dùng attention masks khác nhau [B2, §3.2, PDF p.3]:

| Objective | Query–text mask | Cách dùng representation |
|---|---|---|
| ITC | Queries và text không nhìn nhau | Similarity với text CLS; lấy **max** qua query similarities |
| ITM | Bidirectional query–text interaction | Classifier trên query outputs; **average logits** qua queries |
| ITG | Queries không nhìn text; text nhìn queries và các text token trước | Sinh text từ thông tin queries đã lấy ở ảnh |

ITG là **image-grounded text generation của stage 1**, không phải yêu cầu phải có LLM bên ngoài ngay ở stage này. Paper dùng in-batch negatives cho ITC và hard negative mining cho ITM.

### 4.3 Stage 2: nối với frozen LLM

```text
image -> frozen vision encoder -> trainable Q-Former
                               -> trainable linear projection
                               -> visual soft prompts -> frozen LLM -> text
```

Paper thử CLIP ViT-L/14 và EVA-CLIP ViT-g/14, dùng features từ lớp áp chót; LLM gồm OPT hoặc FlanT5 [B2, §3.3–3.4, PDF p.4].

- OPT: conditional language modeling.
- FlanT5: prefix language modeling.

Vision encoder và LLM frozen trong pretraining này; gradients vẫn đi qua LLM để học connector, không đồng nghĩa không tốn forward/backward computation cho đường nối.

Paper dùng khoảng **129M images**, với caption bootstrapping/ranking. Dữ liệu lớn và pretraining sẵn của backbone/LLM vẫn là nguồn năng lực; ít trainable parameters không có nghĩa hệ thống đã học mọi thứ từ ít dữ liệu.

### 4.4 Sửa một hiểu nhầm quan trọng: retrieval không dùng LLM

BLIP-2 paper viết rõ [B2, §4.4, PDF p.8]:

```text
first-stage pretrained model, không có LLM
  -> fine-tune image encoder + Q-Former bằng ITC, ITM, ITG trên COCO
  -> image–text similarity chọn k=128 candidates
  -> ITM reranking
```

Vì thế phải phân biệt:

| Phát biểu | Đúng hay sai? |
|---|---|
| BLIP-2 luôn cần LLM để retrieval | **Sai** |
| Vision encoder luôn frozen ở mọi downstream setting | **Sai** |
| BLIP-2 hỗ trợ image–text retrieval | **Đúng**, có thí nghiệm trong paper |
| Kết quả đó tự chứng minh BLIP-2 tốt cho sketch–photo retrieval | **Sai**, khác modality/task/protocol |
| Q-Former compression chắc chắn làm hỏng detail | **Chưa chứng minh**; compression cũng có thể chọn thông tin hữu ích |

### 4.5 Điểm mạnh và giới hạn có thể kiểm tra

**Điểm mạnh:** có bước alignment trước khi nối LLM; tái sử dụng các unimodal models; representation stage dùng được không cần generation; nhiều query outputs có thể cung cấp biểu diễn phong phú hơn một vector.

**Giới hạn kiến trúc có điều kiện:** nếu frozen visual features không chứa thông tin cần thiết thì connector không thể đọc lại pixel để tạo bằng chứng thực. Nhưng nếu thông tin còn trong token sequence, Q-Former có thể học khai thác nó. Không được đồng nhất “global CLS yếu” với “mọi intermediate visual token đều yếu”.

**Giới hạn generation do paper nêu:** knowledge/reasoning lỗi, in-context VQA không cải thiện trong setting đã thử, và kế thừa risks/bias của LLM [B2, §5, PDF p.8]. Không biến các hạn chế năm 2023 thành tuyên bố mọi LVLM hiện nay có cùng kết quả.

## 5. So sánh đúng trục

| Trục | CLIP gốc | BLIP | BLIP-2 |
|---|---|---|---|
| Representation chính | Global image/text vectors | Global vectors + fused pair features | Query outputs + text features |
| Pretraining objectives | Symmetric contrastive | ITC + ITM + LM | Stage 1: ITC + ITM + ITG; stage 2: LLM generation |
| Image encoder frozen? | Không trong pretraining gốc | Không phải thiết kế frozen-base như BLIP-2 | Có trong pretraining; retrieval paper fine-tune lại |
| Sinh text trực tiếp? | Không | Có decoder | Stage 1 có ITG; generation với LLM ở stage 2 |
| Retrieval không cần generation? | Có | Có | Có, không cần LLM |
| Cache/candidate retrieval | Một vector cho mỗi input ở đường global | ITC vectors; ITM trên subset | Query/text features; ITM trên subset |
| Dense sketch–photo correspondence trong objective gốc? | Không | Không | Không |

Không so trainable parameter count mà bỏ qua tổng backbone size, pretrained compute, token count, resolution và downstream fine-tuning. Không lấy COCO/Flickr Recall@K so trực tiếp với TU-Berlin mAP@all/P@100.

## 6. Áp dụng có điều kiện vào repository

### 6.1 Hiện trạng khác ba paper gốc

- `src/spica/models/sketch_student.py`: random-init CNN + hai hybrid blocks với shared-weight four-direction Mamba-2 scans; cuối cùng foreground-weighted pooling thành normalized 512-D vector. Không phải VMamba SS2D nguyên bản.
- `src/spica/train_student.py::multi_positive_loss`: positive mask theo **class labels**; mọi photo cùng lớp trong tập đưa vào loss đều là positives.
- `src/spica/data/datasets.py`: positive photo được chọn ngẫu nhiên theo class, không có instance correspondence trong contract hiện tại.
- `src/spica/models/sketchlvm.py` và trainer riêng: adaptation hai visual branches bằng shallow prompts/LN, category supervision; không phải BLIP MED hoặc BLIP-2 Q-Former.

Do đó không thể kết luận “Student thiếu một BLIP-2 connector” hay “thêm caption chắc sẽ sửa detail”. Trước hết phải hỏi supervision hiện tại có thưởng việc phân biệt detail đó không.

### 6.2 Phân vai đề xuất, chưa triển khai

| Vai trò | Lựa chọn khả dĩ | Điều kiện |
|---|---|---|
| Semantic/photo-space anchor | CLIP | Đo riêng độ phân biệt của photo features; không giả định hoàn hảo |
| Trainable sketch/detail encoder | Encoder hiện có với local supervision được bổ sung | Chỉ sau khi xác định dữ liệu/relevance đúng |
| Caption/attribute teacher | BLIP hoặc BLIP-2 | Training-only; kiểm tra mô tả trên ảnh/sketch; không tin pseudo-label tuyệt đối |
| Candidate reranking | Local visual-pair matching | Phải sketch-conditioned, không dùng true labels/unseen text; không gọi head ITM gốc là visual-pair head |
| Giữ one-to-many uncertainty | Sketch-conditioned Mo-vMF theo protocol repository | Local features có thể nuôi head; mixture không tự tạo supervision chi tiết |

Đây là danh sách vai trò, **không đề xuất lắp tất cả thành một hệ thống**. Với dữ liệu chỉ có category labels, ưu tiên xác định phép đo và self-supervision trong sketch trước khi thêm teacher lớn.

### 6.3 Kết quả lịch sử không xác định nguyên nhân

[Seen-training diagnostic](student_seen_semantics_2026-09-28.md) cho thấy Student kém teacher ở cả centroid classification và retrieval; không có bằng chứng Student đã xử lý geometry tốt hơn rồi chỉ thiếu semantic alignment.

[SketchLVM gap/masking report](sketchlvm_public_inputs_gap_and_masking_2026-10-01.md) chưa xác định nguyên nhân chính của gap với paper. Xóa ink làm AP giảm không chứng minh patch size, Q-Former hay một loại VLM là lời giải. Các run/audit raw statuses được giữ nguyên; tài liệu này không nâng mức chứng nhận.

## 7. Nguồn đã kiểm tra

- **[C] Radford et al., CLIP (2021):** https://arxiv.org/abs/2103.00020 — PDF pp.3–5 (method), 7–8 (prompts), 19–20 (limitations).
- **[B] Li et al., BLIP (2022), v2:** https://arxiv.org/abs/2201.12086v2 — PDF pp.3–4 (MED/loss/CapFilt), 5–6 (ablations/retrieval), 7–8 (generation/downstream).
- **[B2] Li et al., BLIP-2 (2023), v3:** https://arxiv.org/abs/2301.12597v3 — PDF pp.2–4 (Q-Former/two stages), 8 (retrieval/limitations).

Số trang là trang PDF, không phải số trang proceedings. Các nhận định ứng dụng vào sketch được đánh dấu là suy luận/đề xuất; không có benchmark CLIP–BLIP–BLIP-2 cùng protocol sketch được chạy trong công việc này. **Không sửa code, train, evaluate hoặc đổi dependencies.**

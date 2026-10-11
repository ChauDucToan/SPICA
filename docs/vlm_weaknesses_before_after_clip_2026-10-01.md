# Điểm yếu của VLM trước và sau CLIP

Khởi tạo: 2026-10-01. Rà soát và bổ sung nguồn: **2026-10-04**.

## 1. Phạm vi và cách đọc kết luận

VLM được dùng theo nghĩa rộng: image–text representation learning, retrieval, captioning, VQA và mô hình ngôn ngữ có đầu vào ảnh. Đây là tổng quan các thiết kế đại diện, **không phải khảo sát đầy đủ mọi VLM đến năm 2026**.

Phân biệt ba loại phát biểu:

- **Thiết kế đã công bố:** kiến trúc, objective và chế độ train/inference trong nguồn gốc.
- **Bằng chứng thực nghiệm:** kết quả trên đúng mô hình và benchmark được bài báo kiểm tra.
- **Suy luận/đề xuất:** rủi ro khi áp dụng vào sketch; chưa phải nguyên nhân đã được chứng minh trong repository.

“Trước CLIP” không đồng nghĩa “mọi mô hình đều dùng detector”. Có ít nhất hai nhóm: CNN–RNN cho caption/VQA và region-based VLP. VirTex là một ví dụ trước CLIP học CNN bằng caption mà không cần pipeline region detector [1,7]. ALBEF xuất hiện cùng năm 2021 nhưng **sau CLIP**, là ví dụ align-before-fuse không cần bounding-box annotation [9]. BLIP và BLIP-2 không phải bản nâng cấp tuần tự thay thế mọi chức năng của CLIP.

Tài liệu liên quan:

- [Phân tích riêng CLIP, BLIP và BLIP-2](clip_blip_blip2_analysis_2026-10-04.md).
- [Đề xuất học chi tiết sketch và thiết kế phép đo](sketch_detail_learning_recommendations_2026-10-04.md).

## 2. Trước CLIP: vấn đề xuất hiện ở đâu?

### 2.1 CNN–RNN captioning và VQA: ngôn ngữ có thể che việc nhìn chưa đủ

Show and Tell kết hợp thị giác với recurrent language generation, tối đa hóa likelihood của caption đích [1]. Tín hiệu này khác với kiểm tra từng nét hoặc bộ phận: câu “một chiếc ghế” có thể đúng dù model chưa phân biệt hình chân ghế.

**Bằng chứng về shortcut:** công trình xây dựng VQA v2 dùng các cặp ảnh tương tự nhưng có đáp án khác nhau cho cùng câu hỏi. Các VQA model được khảo sát giảm hiệu năng trên dữ liệu cân bằng, cho thấy chúng đã khai thác language priors [14]. Đây là bằng chứng cho những mô hình/dữ liệu được kiểm tra, không phải kết luận mọi VQA model bỏ qua ảnh.

**Hàm ý cho sketch:** caption trôi chảy hoặc trả lời đúng tên lớp chưa chứng minh model đọc được số nan hoa, chỗ nối nét hoặc hình dạng một bộ phận.

### 2.2 Region-based VLP: detector tạo một điểm nghẽn trước fusion

Các thiết kế đại diện [2–6]:

```text
ảnh -> detector / region features -> visual tokens
văn bản -> word tokens
visual tokens + word tokens -> multimodal encoder -> task head
```

- ViLBERT: hai stream với co-attention.
- LXMERT: language, object-relationship và cross-modality encoders.
- VisualBERT: joint modeling của text và image-region features.
- UNITER: nhiều pretraining objectives, gồm word–region alignment.
- Oscar: object tags làm điểm neo để liên kết ảnh và ngôn ngữ.

**Rủi ro suy ra từ thiết kế:** nếu bộ trích region bỏ sót một vùng hoặc không biểu diễn chi tiết trong vùng, fusion phía sau không được trực tiếp nhìn lại pixel để sửa. Taxonomy, bounding boxes và pseudo-labels của detector có thể giới hạn thông tin đầu vào. Điều này đáng lưu ý với sketch chứa nét mảnh, contour và junction, nhưng **chưa phải phép đo detector trên sketch của dự án**.

Không nên khẳng định “region features không thể biểu diễn hình học”. Chúng có thể chứa thông tin vị trí và hình dạng; giới hạn là thông tin nào được giữ, độ phân giải nào và supervision nào được dùng.

### 2.3 Local objectives có ích nhưng không tương đương dense correspondence

Word–region alignment và multimodal attention là các bước tiến thực sự. Tuy nhiên:

- caption không nhất thiết mô tả mọi bộ phận;
- object label từ detector không phải nhãn correspondence nét-với-bộ-phận;
- pair matching đúng không đòi hỏi từng từ phải được grounding chính xác;
- attention map chỉ là một phép quan sát, không tự nó chứng minh quan hệ nhân quả hay correspondence đúng.

Đây là giới hạn về **độ đầy đủ của supervision**, không phải khẳng định mọi mô hình fusion đều thiếu local alignment.

### 2.4 Chi phí và giao diện transfer

Detector, region storage và multimodal fusion tạo thêm công đoạn so với dual-encoder embedding. Nếu phải chạy fusion cho từng cặp query–candidate, chi phí tăng theo số cặp; không được suy rộng thành mọi region-based model đều chậm hơn trên mọi task.

Nhiều model trước CLIP được đánh giá qua task-specific fine-tuning. Chúng không có sẵn cùng giao diện zero-shot classification bằng text prompts như CLIP. **Không đồng nghĩa chúng hoàn toàn không có khả năng zero-shot transfer.**

## 3. CLIP thay đổi điều gì, và không giải quyết điều gì?

### 3.1 Bước tiến chính: semantic alignment có thể mở rộng

CLIP huấn luyện image encoder và text encoder bằng symmetric contrastive loss trên khoảng 400 triệu cặp image–text. Embeddings được chuẩn hóa; similarity giữa ảnh và text dùng làm điểm số [8, PDF pp.3–5].

```text
ảnh -> image embedding ----+
                          +-> similarity
text/prompt -> text embedding
```

Gallery embeddings có thể tính trước. Zero-shot classification được xây từ các tên lớp/prompt cung cấp lúc inference. “Open vocabulary” ở đây vẫn là **so sánh với những mô tả ứng viên được đưa vào**, không phải model tự sinh mọi nhãn có thể có.

### 3.2 Global contrastive loss không trực tiếp giám sát từng chi tiết

**Thiết kế:** loss gốc so sánh embedding toàn ảnh với embedding toàn câu, không có nhãn patch–word correspondence trong objective CLIP [8].

**Suy luận:** nếu caption và negatives chỉ cần phân biệt “ghế” với “xe”, model có thể giảm loss mà chưa học phân biệt hai kiểu chân ghế. Điều này không chứng minh patch tokens của CLIP không chứa chi tiết; thông tin có thể tồn tại nhưng chưa được readout/loss tận dụng đủ.

Với ViT-B/32 ở ảnh 224×224, có lưới 7×7 patch tokens trước CLS readout. **Một patch 32×32 không tự động xóa toàn bộ nét bên trong:** patch embedding vẫn nhận pixel trong patch. Patch size, resize và global readout là những điểm cần kiểm tra, không phải bằng chứng sẵn có rằng chúng gây ra gap của SketchLVM.

### 3.3 Fine-grained, counting và OOD: có bằng chứng, nhưng phải giữ đúng phạm vi

CLIP paper tự ghi nhận zero-shot performance yếu ở một số bài toán fine-grained như loại xe, hoa và máy bay; counting và một số task ngoài phân phối cũng khó [8, PDF pp.19–20]. Điều này **không** có nghĩa CLIP thất bại trên mọi fine-grained dataset hoặc fine-tuned CLIP không thể cải thiện.

Hai nguồn kiểm tra compositionality bổ sung:

- **Winoground:** ghép hai ảnh với hai caption chứa cùng tập từ nhưng thứ tự khác; các model được khảo sát gặp khó khăn rõ rệt [13].
- **ARO:** kiểm tra attribution, relation và order; chỉ ra retrieval benchmark phổ biến có thể được giải bằng shortcuts mà không cần dùng đầy đủ compositional structure [12].

Những kết quả này giúp giải thích vì sao **retrieval score cao không đủ chứng minh grounding/compositionality**. Chúng không trực tiếp đo khả năng học stroke của CNN–Mamba hay SketchLVM hiện tại.

### 3.4 Prompt, domain và dữ liệu

CLIP paper phân tích prompt engineering/ensembling và ghi nhận giới hạn OOD, bias từ web data [8, PDF pp.7–8,19–20]. Với sketch cần kiểm tra riêng:

- preprocessing có giữ foreground và nét mảnh không;
- query sketch và photo gallery có dùng đúng normalization của model không;
- mô tả text có phù hợp domain không, nếu task thực sự dùng text;
- model có dựa vào style/background thay vì cấu trúc phân biệt không.

Trong standard ZS-SBIR của repository, **không đưa nhãn query hoặc unseen-class text bank vào retrieval model**. Phân tích prompt của CLIP classification không phải lý do đổi giao thức inference SBIR.

### 3.5 CLIP không sinh câu trả lời

CLIP gốc là dual encoder, không có autoregressive language decoder. Lỗi ranking/classification của CLIP khác với “hallucination” trong câu sinh của một LVLM. Nối CLIP với LLM có thể thêm khả năng diễn đạt, nhưng không tự động sửa thông tin thị giác mà encoder chưa giữ được.

## 4. BLIP và BLIP-2: cải thiện thật, giới hạn còn lại

### 4.1 BLIP không chỉ là captioner

BLIP kết hợp ba objective [10, PDF pp.3–4]:

1. **ITC:** align image và text embeddings.
2. **ITM:** cross-modal matching, với hard negatives.
3. **LM:** sinh text dựa trên ảnh.

CapFilt tạo caption tổng hợp và lọc cả caption web lẫn caption tổng hợp trước khi pretrain model mới. Bài báo còn khảo sát confirmation bias khi captioner/filter chia sẻ tham số [10, PDF p.6].

**Giới hạn suy ra:** caption tổng hợp và filter có thể bỏ qua hoặc chấp nhận sai một chi tiết. Caption dài hơn không tự biến thành dense part supervision. ITM khuyến khích kiểm tra cặp ảnh–text kỹ hơn, nhưng không có bảo đảm toán học rằng mọi stroke được grounding đúng.

**Retrieval:** BLIP có đường embedding similarity để lấy candidates, sau đó ITM reranking; paper dùng cách này để tránh fusion trên toàn gallery [10, §5.1, PDF p.6]. Vì vậy không đúng khi mô tả BLIP là luôn phải sinh caption hoặc luôn phải fusion mọi ảnh trong gallery.

### 4.2 BLIP-2 có hai chế độ cần phân biệt

**Pretraining/generation:** frozen image encoder → trainable Q-Former → projection → frozen LLM. Paper dùng 32 learned queries, mỗi query 768 chiều; đây là information bottleneck có chủ đích để giữ thông tin liên quan tới text [11, PDF pp.3–4].

**Image–text retrieval:** paper dùng model sau stage 1 **không có LLM**, fine-tune **cả image encoder và Q-Former**, rồi similarity candidate selection + ITM reranking [11, §4.4, PDF p.8].

Do đó:

- sai: “BLIP-2 chỉ làm generation, không phù hợp retrieval”;
- sai: “image encoder của BLIP-2 luôn frozen ở mọi downstream task”;
- đúng: đường **generation qua LLM** không cần thiết nếu chỉ muốn retrieval;
- chưa được chứng minh ở đây: image–text retrieval của BLIP-2 sẽ vượt CLIP/SketchLVM ở sketch–photo retrieval.

### 4.3 Q-Former là khả năng chọn thông tin, không chỉ là mất thông tin

Query compression có thể bỏ phần không cần cho task và **giúp** alignment; paper báo cáo lợi ích của ITG cho image–text retrieval [11, PDF p.8]. Số queries hữu hạn không tự chứng minh chi tiết quan trọng bị mất.

Rủi ro cần kiểm tra khi chuyển sang sketch là: supervision text không yêu cầu giữ một chi tiết nào đó, hoặc visual features nền đã không biểu diễn nó. Đây là **giả thuyết về transfer**, không phải kết quả FG-SBIR đã được đo trên BLIP-2 trong tài liệu này.

### 4.4 Generation và visual faithfulness là hai thước đo khác nhau

BLIP-2 paper nêu hạn chế về generation, reasoning/knowledge và việc kế thừa rủi ro từ frozen LLM [11, §5, PDF p.8]. POPE đo object hallucination ở các LVLM được kiểm tra, cho thấy đối tượng thường xuất hiện/co-occur trong dữ liệu instruction có thể bị sinh ra dù không có trong ảnh [15].

*Vision language models are blind* kiểm tra các VLM sinh ngôn ngữ hiện đại trên những task như giao nhau của đường tròn/đường thẳng và đếm hình [16]. Đây là cảnh báo cần test thị giác mức thấp riêng, **không phải thí nghiệm trực tiếp trên CLIP/BLIP/Student của repository**.

Không đồng nhất:

```text
caption trôi chảy ≠ grounding đúng
category retrieval tốt ≠ instance matching tốt
robust với xóa nét ≠ phân biệt đúng nét quan trọng
```

## 5. Bảng so sánh có điều kiện

| Hướng | Lợi thế thiết kế | Điểm cần kiểm tra khi học sketch detail |
|---|---|---|
| CNN–RNN caption/VQA | Sinh ngôn ngữ, chuyển thông tin thị giác thành câu | Language prior; caption/answer không yêu cầu mọi nét |
| Region-based VLP | Object/region tokens và cross-modal interactions | Detector coverage; resolution; pseudo-label và correspondence |
| CLIP | Global embedding, text interface, gallery caching thuận tiện | Global objective/readout; domain transfer; negatives có thực sự đòi hỏi detail |
| BLIP | ITC + ITM + LM; CapFilt; retrieval hai tầng | Caption/filter errors; pair matching chưa đủ dense grounding |
| BLIP-2 | Tái sử dụng vision/LLM; Q-Former chọn thông tin; retrieval không cần LLM | Frozen-base transfer trong chế độ tương ứng; query bottleneck; cần fine-tuning/protocol đúng task |

Không xếp hạng “mô hình nào hiểu sketch tốt nhất” khi chưa có benchmark cùng dữ liệu, ngân sách, gallery và định nghĩa relevance.

## 6. Kết luận cho dự án

Ưu tiên không phải “thay CLIP bằng một VLM lớn hơn”, mà là:

1. xác định detail nào phải ảnh hưởng ranking;
2. có positive/negative supervision tương ứng;
3. giữ local features và đo được việc dùng chúng;
4. giữ category-level và instance-level evaluation tách biệt.

CLIP có thể làm semantic/photo-space anchor. BLIP/BLIP-2 có thể cung cấp embeddings, reranking hoặc training-only teacher; dùng caption chỉ khi thông tin được kiểm tra. Không mặc định loại chúng khỏi retrieval, cũng không mặc định chúng giải quyết domain gap sketch.

**Chỉ nghiên cứu/tài liệu:** không thay model, loss, protocol hoặc chạy training/evaluation mới. Các kết quả lịch sử và raw audit statuses được giữ nguyên.

## 7. Nguồn và mức kiểm tra

CLIP, BLIP và BLIP-2 được đối chiếu các phần method, retrieval và limitations trong PDF gốc; số trang ở trên là trang PDF. Các tiền thân dùng trang/abstract gốc để mô tả thiết kế tổng quát, không tuyên bố rà soát toàn bộ thí nghiệm. ARO, Winoground và *VLMs are blind* dùng nguồn gốc cho mục đích/nhận định benchmark; không đưa số hiệu năng chi tiết chưa kiểm tra. POPE được đọc thêm PDF pp.1–7.

1. Vinyals et al., **Show and Tell: A Neural Image Caption Generator**, 2014/2015: https://arxiv.org/abs/1411.4555
2. Lu et al., **ViLBERT**, 2019: https://arxiv.org/abs/1908.02265
3. Tan and Bansal, **LXMERT**, 2019: https://arxiv.org/abs/1908.07490
4. Li et al., **VisualBERT**, 2019: https://arxiv.org/abs/1908.03557
5. Chen et al., **UNITER**, 2019/2020: https://arxiv.org/abs/1909.11740
6. Li et al., **Oscar**, 2020: https://arxiv.org/abs/2004.06165
7. Desai and Johnson, **VirTex**, 2020/2021: https://arxiv.org/abs/2006.06666
8. Radford et al., **Learning Transferable Visual Models From Natural Language Supervision (CLIP)**, 2021: https://arxiv.org/abs/2103.00020
9. Li et al., **Align before Fuse (ALBEF)**, 2021: https://arxiv.org/abs/2107.07651v2
10. Li et al., **BLIP**, 2022: https://arxiv.org/abs/2201.12086v2
11. Li et al., **BLIP-2**, 2023: https://arxiv.org/abs/2301.12597v3
12. Yuksekgonul et al., **When and why vision-language models behave like bags-of-words, and what to do about it? (ARO)**, ICLR 2023: https://arxiv.org/abs/2210.01936v3
13. Thrush et al., **Winoground**, CVPR 2022: https://arxiv.org/abs/2204.03162v1
14. Goyal et al., **Making the V in VQA Matter**, 2016/2017: https://arxiv.org/abs/1612.00837
15. Li et al., **Evaluating Object Hallucination in Large Vision-Language Models (POPE)**, EMNLP 2023: https://arxiv.org/abs/2305.10355
16. Rahmanzadehgervi et al., **Vision language models are blind**, bản khảo sát v1 năm 2024: https://arxiv.org/abs/2407.06581v1

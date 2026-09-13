# 2. Phương pháp và giao thức đã thực thi

## 2.1 Bài toán

Đầu vào là phác thảo raster; đầu ra là danh sách ảnh xếp hạng theo cosine similarity. Relevance là **cùng lớp ngữ nghĩa**, không phải đúng một instance ghép cặp. Theo split identity archived trong `resolved_config.json` (`protocol_identity.semantic_class_name_overlap=[]`) và các data gate của campaign, các tập lớp train và test không giao nhau theo tên ngữ nghĩa. Audit chuẩn bị tài liệu không mở lại toàn bộ manifest ảnh để kiểm tra độc lập tính chất này. ID số của mỗi split có thể bắt đầu lại từ0, vì vậy không dùng phép giao ID số để suy ra leakage. Zero-shot ở đây là không dùng các lớp test để thích nghi mô hình downstream; không chứng minh các khái niệm đó vắng trong tiền huấn luyện CLIP.

Không có nhãn thật của query test, caption riêng của query hoặc ảnh đúng được đưa vào query encoder. Điều kiện xóa tác động query; gallery ảnh giữ nguyên.

## 2.2 Kiến trúc MP-Q

Backbone CLIP ViT-B/32 quickgelu: ảnh224×224, patch32,49patch token, width768, embedding512. Sketch student là bản khởi tạo từ CLIP độc lập và được fine-tune; CLIP tham chiếu ảnh/văn bản giữ frozen. Prompt ảnh/văn bản, predictor và pooled head được học. Không mô tả toàn hệ thống là chỉ prompt-tuning: phần lớn sketch visual encoder cũng thay đổi.

```text
                          TRAIN ONLY: predictor
shared photo/text prompts ───┐
                             v
sketch clean/corrupted → student → H[49,768] → projection[49,256]
                          │                   │
                          │        prompt-query CrossAttention
                          │                   ↓
                          │        joint SelfAttention + residual
                          │                   ↓
                          │               FFN + output
                          │                μ_I, μ_T → train losses
                          ↓
                    mean patch tokens → pooled head → normalize → q
                                                                  │
photo gallery → frozen CLIP + learned photo prompts → normalize → cosine rank
```

Predictor nhận prompt ảnh3×768 và text context4×512, đưa về width256, cross-attend lên sketch rồi self-attend joint photo/text tokens (4heads), FFN và output512. μ_I/μ_T là normalized mean của hai nhóm token đầu ra. Không phải self-attention trên prompt trước khi đọc sketch.

**MP-Q:** training main objective qua μ_I, inference chỉ q = normalize(pooled_head(mean(H))). Predictor không chạy ở query inference. Không nhầm với F2_QMP (chuyển main MP sang q), cũng không suy ra predictor vô dụng trong training chỉ vì nó được bỏ qua lúc inference. Các prompt ảnh ở gallery lấy từ cùng checkpoint.

Nguồn triển khai: `src/spica/models/coupled_predictive.py`, `QOnlyAdapter` trong evaluator, và [thiết kế F2](../coupled_predictive_fusion_v2_implementation_2026-09-09.md). Tài liệu F2 historical dùng μ_I inference; **sơ đồ MP-Q ở đây mới là readout của official runs được báo cáo**.

## 2.3 Sampling và loss

Mỗi batch32sketch: một ảnh dương cùng lớp, một ảnh âm thuộc lớp khác cho mỗi sketch; deduplicate thành ≤64live photos. Ảnh dương được lấy từ toàn bộ pool ảnh train cùng lớp. Mục tiêu nhiều-positive còn coi mọi ảnh trong live bank cùng lớp với query là positive; **không phải64positive cho mỗi sketch, không phải full dataset gallery mỗi bước**. Cấu hình lưu `negative_rule=uniform_photo_from_other_train_class`; chi tiết cách lấy lớp rồi lấy ảnh xem sampler archived để tránh hiểu nhầm phân phối uniform toàn ảnh.

Với s_ij = cosine(μ_I(i),p_j)/τ, τ=0,07 và P_i là ảnh cùng lớp trong live bank:

\[
L_{MP}=-\frac1B\sum_i\frac1{|P_i|}\sum_{j\in P_i}\log\frac{\exp(s_{ij})}{\sum_k\exp(s_{ik})}.
\]

Đây là trung bình log-probability theo từng positive, không phải log của tổng positive probability.

Với mỗi view v (clean/corrupted), loss:

\[
L_v=L_{MP}(\mu_I)+CE(\mu_I,T)+0.25CE(\mu_T,T)
+0.25[rank(q,p^+,p^-)+CE(q,T)]
+0.05[align_I+align_T].
\]

\[
L=0.5(L_{clean}+L_{corrupted})+0.5anchor_I+0.5anchor_T.
\]

`rank(q)` dùng softplus(0.2 + cos(q,p−) − cos(q,p+)). Text CE dùng learned train-class text bank và τ0,07. Align dùng cosine với target detach; anchors giữ ảnh/text gần CLIP reference frozen. Anchors tính một lần ngoài view average. Official MP-Q không SIGReg, không photo-CE bổ sung, không sketch-reference InfoNCE. Không dùng λ của các ablation lịch sử như thể là mặc định official.

AdamW: student LR1e−5, predictor/pooled/prompts LR1e−4; decay matrix0,01, bias/LN/vector0, prompt0,0001; betas0,9/0,999, eps1e−8. Warmup sau đó cosine; chi tiết7groups và hệ số loss trong `training_configs.json` là authoritative.

## 2.4 Dữ liệu và ngân sách

| Dataset | Train classes | Train sketches/photos | Test classes | Test queries/gallery | Updates/warmup |
|---|---:|---:|---:|---:|---:|
| Sketchy |104|57.587 /72.949|21|12.694 /12.553|4446 /222|
| TU-Berlin |220|15.400 /176.081|30|2.400 /27.989|1189 /59|
| QuickDraw |80|236.080 /149.428|30|92.291 /54.151|18229 /911|

Seed42, batch32, khoảng2,47 lượt quan sát sketch (updates×32/count); không đảm bảo mỗi ảnh được thấy đúng số lần do sampling/iterator. Không gọi đây là số epoch hội tụ tối ưu. Tham chiếu ngân sách là3600×32/46624, làm tròn update theo quy mô mỗi bộ.

Train dùng clean và corrupted view, xóa vùng raster với mức25/50/75%. Không mô tả đây là xóa stroke vector, semantic-part deletion hay một phép crop thông thường. Test gồm clean và9combination mức25/50/75% × mask seeds101/202/303. Severity là mục tiêu theo chính sách mực; giữ `mask_status_counts`, không âm thầm bỏ `target_unreachable`.

Official test tại ceil(horizon×i/5), i=1..5;5lần full clean+9. Do ceil số nguyên, tiến độ thực tế có thể là20,0168%, không tròn20%. Tập test đã được quan sát nhiều lần: **không blind holdout**, không chọn checkpoint theo test. Bảng chính dùng100%fixed-final; số tốt nhất trước đó chỉ có thể ghi là mô tả đường cong, không primary.

## 2.5 Định nghĩa metric — bắt buộc ghi mẫu số

Cho rel(k)∈{0,1}, R là số ảnh relevant trong toàn gallery, P(k)=Σ_{j≤k}rel(j)/k:

- P@200 = Σ_{k≤200}rel(k)/200.
- Full AP = Σ_{k≤G}P(k)rel(k)/R; full mAP là trung bình AP theo query.
- AP200 **minR** = Σ_{k≤200}P(k)rel(k)/min(R,200). Đây là `test/*/mAP@200` được log.
- AP200 all-relevant = cùng tử số/R.
- AP200 prefix-positive = cùng tử số/max(1,R200), với R200=Σ_{k≤200}rel(k).

Các công thức AP200 không thay thế nhau. Zero-positive convention được xử lý trong evaluator; official protocols ở đây có ảnh cùng lớp trong gallery. Masked macro là trung bình9metric điều kiện, mỗi metric đã query-mean; không phải training-seed mean. P@all chỉ summary, phụ thuộc tỷ lệ lớp trong gallery, không phản ánh chất lượng thứ hạng.

Đối chiếu paper cần cùng split, gallery, relevance, metric denominator và cutoff. Đặc biệt P@200 của TU không được gọi là P@100; bộ tài liệu chưa thêm P@100. Không dùng pseudo Sketchy84/20 làm control trực tiếp cho official104/21.

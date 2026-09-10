# Báo cáo chi tiết: kích thước vật thể và độ ổn định của tín hiệu local

Ngày 2026-09-10. **Đã hoàn tất đo GPU không cập nhật weights; kiểm tra độc lập full-AP có 2 dòng vượt ngưỡng, giữ FAIL.** Không gọi toàn diagnostic VERIFIED.

## 1. Kết luận điều hành

1. **Không xác nhận giả thuyết “chỉ cần làm vật thể lớn hơn là sẽ tốt”.** Đưa hình về giữa tăng full mAP trung bình trên mẫu100query; phóng to thêm không tăng tiếp ở MP-Q và giảm ở TQMP.
2. **Có độ nhạy với vị trí/kích thước, không có nghĩa thiếu toàn bộ thông tin local.** Center-only giữ nguyên mọi pixel không trắng, không nội suy, nhưng một số query đổi AP rất mạnh. Tác động trung bình dương không phổ quát: median center-only delta âm ở cả hai model.
3. **Ảnh rất nhỏ là một trường hợp cần quan tâm, không phải lời giải chung.** Bảy sketch có ink-bbox≤64×64: zoom giúp MP-Q trung bình+.08318 nhưng TQMP−.00321; từng query thắng/thua rất khác nhau. Cỡ nhóm7 quá nhỏ cho kết luận rộng.
4. **Đa số dịch cửa sổ xóa2pixel chỉ thay đổi AP nhỏ, nhưng có đuôi phản ứng lớn.** Median |AP_neighbor−AP_base|≈.00346/.00361; percentile95≈.05457/.05817. Có các hiệu ứng đổi dấu rõ, ngay cả trong tập có lượng/vị trí nét bị xóa tương đối gần nhau.
5. **Không có bằng chứng TQMP đã sửa được độ ổn định local.** TQMP có241/3066 đổi dấu so216/3066 MP-Q, nhưng đây là mô tả hai mô hình trọn gói, không kiểm định thống kê hay tác dụng riêng của text/MP(μ_T).
6. **Không tự đổi preprocessing, inference, loss hoặc kéo dài training.** Candidate cần kiểm tra là khả năng bất biến với vị trí/scale và cách sử dụng local ổn định, không chỉ thêm local attention hoặc thêm MP loss.

## 2. Phạm vi và định danh

User chọn kiểm tra **cả scale và độ tin cậy local**, trên đúng100sketch và hai checkpoint đã dùng ở diagnostic trước.

Root: `outputs/scale_local_reliability_20260910T064000Z/`.

|Đối tượng|Định danh|
|---|---|
|Source HEAD trước đo|`5c91efc8b53a6d12de374fbba7099b55b2ceb10f`|
|MP-Q|F2_MP checkpoint3600, q readout; SHA `58fb5fb2aa0f822d1df8bee9192a0bda5eec1580b4ce70ed9ae243fdcf9fb14e`|
|TQMP|F2_TQMP checkpoint3600, q readout; SHA `e914f6d38bed502a0357c221dc79131cb9b2f17e05860ec371a77b03f1965dfc`|
|MP training source|`ab064ced3b11c95f142ac94b336072d88312f5e14d776a386416bbf3cf70f29e`|
|TQMP training source|`b8eb37c7446aae4227f0560f7bc70c53970d17aab46681227b8fb8b13b8ba4ca`|
|Mẫu|100pseudo-validationqueries,20classes×5, giữ nguyên selectionSHA của lượt trước|
|Gallery|13,999photoIDs/labels giống nhau; **mỗi checkpoint dùng gallery embeddings riêng của nó**|
|Updates/backward|0/0; gradientsNone; hashes weights trước/sau bằng nhau|
|Inference|ProductionQOnlyAdapter; predictor0calls trong probe, full-vs-bypass delta0|

Không thêm official-unseen/query/seed/checkpoint, không tìm λ/zoom, không training, không kết quả W&B mới. Không phải benchmark paper hay full10,963-query validation. Mẫu này đã được quan sát ở diagnostic trước, **không phải blind holdout**.

Dùng gallery cache SHA-bound từ lượt trước, không re-encode gallery trong lượt này. Loader checkpoint từ archived verifier, `weights_only=True` + NumPy safe globals; không dùng unfinishedworktreeV1verifier. Model inference source byte-matched với cả hai trainingarchives. Source/input/component hashes tại `measurement/protocol_lock.json`; không gọi đây là toàn bộ dependency/environment archive.

## 3. Thiết kế kiểm tra được khóa trước GPU

### 3.1. Tách vị trí khỏi phóng to

Mỗi query có ba ảnh:

1. **Original:** ảnh dataset224×224 nguyên trạng.
2. **Centered:** tìm bounding box tất cả pixel có bất kỳ kênhRGB<255; dịch nguyênpixel để support về giữa canvas trắng224×224. Giữ cả nét mờ/antialias; không mất pixel không trắng, không nội suy.
3. **Zoomed:** từ ảnh centered, crop vuông chứa toàn bộ support, cạnh
   `L = min(224, ceil(1.25 × max(support_width, support_height)))`, rồi resize224 bằng PIL bicubic. Chỉ cắt vùng trắng trước nội suy. L=224 thì không resize, chính là ảnhcentered.

-100×3=300scaleviews.
-83query được zoom,17có hệ số1.
-Median zoom≈1.32549, mean1.56565, max5.20930.
-Đây là **một rule cố định**, không dò scale để lấy tốt nhất.
-Zoom giữ toàn bộ nét và bố cục tương đối, nhưng thay độ dày nét/tần số không gian/antialias. Vì thế zoom−center là tác động của **crop+resampling package**, không hoàn toàn là độ phân giải độc lập và không tạo thêm chi tiết nguồn.
-Các ảnh đã224×224; không claim đo được mất mát ở bản gốc trước khi datasetfiles được tạo.

### 3.2. Độ ổn định của vùng xóa

Giữ cả772local16×16cells từ lượt trước, không chọn theo kết quả. Mỗi cell có5vị trí:

`(0,0), (+2,0), (−2,0), (0,+2), (0,−2)`.

Tọa độ ở original224canvas, không ở ảnhzoomed. Neighbor vượt biên được đánh unavailable, không clamp/đổi vị trí.

-772base +3066validneighbors=3838localviews.
-22neighborout-of-bounds giữ metadata, không inference.
-111validneighbor xóa0thresholdedink; giữ flags, không coi như đã xóa một semanticpart.
-233neighbor tạo ảnh pixel-identical với base: đối chứng numerical.
-Predefinedmatchedsubset: Jaccard của tập inkpixels bị xóa≥.75 và tỉ lệ sốinkpixels trong[.8,1.2]:1622neighbors, trong đó1389thực sự khác ảnhpixel.
-Thresholdedink vẫn RGBmean<.9. Matchedsubset **không có semanticpartannotation**, không đảm bảo xóa cùng bộ phận hoặc cùng contrast.

Tổng4160metadata rows, **4138available rows/arm**. Cùng input/mask/rowIDs cho hai model. GPU batch256 cố định, paddinglastbatch bằng originalquery0; giảm một khác biệt batchshape đã thấy trước nhưng **không bảo đảm bit-exact mọi duplicate ở vị trí batch khác nhau**.

### 3.3. Metric và dấu tác động

- FullAP của mỗi view: toàn gallery, mọi photo cùng class là positive.
- Scale mean:100query bằng trọng số,5/class.
- Local unsigned/signed summaries ghi rõ neighborweighted hay family→querymacro; các hàng xóa trên cùng query không phải mẫu độc lập để lập significance/CI.
- Sign flip đáng kể: base vàneighbor đều |AP−AP_original|≥.001, nhưng dấu trái nhau. Không đếm roundoff gần0 là semanticflip.

## 4. Kết quả scale

### 4.1. Tất cả100query

Bảng sau dùng savedGPUmetrics, tổng hợp FP64 bởi parent. Các meanFP64 independent khác dưới2e−6; individualfullAPgate vẫn FAIL như mục7.

|Model/view|Full AP samplemacro|P200|PrefixAP200|All-relevantAP200|Min(R,200)AP200|
|---|---:|---:|---:|---:|---:|
|MP-Q original|.437277409|.460850001|.477216466|.103155370|.361043415|
|MP-Q centered|.456492016|.478500000|.491558469|.108874816|.381061844|
|MP-Q zoomed|.456004060|.476550000|.494124348|.109599436|.383596641|
|TQMP original|.429704938|.441499999|.455362622|.097379084|.340826106|
|TQMP centered|.444139741|.454649999|.467969368|.103607734|.362626462|
|TQMP zoomed|.437668767|.446600001|.464576533|.101425881|.354989697|

|FullAP delta|MP-Q|TQMP|
|---|---:|---:|
|Centered−original|+.019214607|+.014434804|
|Zoomed−original|+.018726651|+.007963829|
|Zoomed−centered|−.000487956|−.006470974|

**Không phải một cải thiện phổ quát.**

- Centered−original, |Δ|≥.001: MP47win/48loss/5small; TQ44win/54loss/2small.
- MedianΔcenter: MP−.000445, TQ−.001940. Mean dương chịu tác động của những ca tăng lớn.
- Zoomed−original, |Δ|≥.001: MP55win/43loss/2small; TQ48win/47loss/5small.
- Centered tăng classmean ở11/20classes mỗimodel; zoomed ở13/20MP và10/20TQ.
- Tác động cực trị center-only: MP−.50873(q52hedgehog) đến+.77155(q78trumpet); TQ−.43839(q52) đến+.83646(q78). Đây là ví dụposthoc về độ nhạy vị trí, không tập đánh giá được chọn lại.

**Diễn giải:** thấy lợi ích trung bình của phép đưa hình về giữa trên mẫu này. Không đủ để tự đổi default hoặc gọi centering tối ưu. Lợi ích zoom sooriginal chứa cả centering; nếu bỏ centeredcontrol sẽ quy nhầm toàn bộ gain cho việc chi tiết lớn hơn.

### 4.2. Bảy vật thể nhỏ

Nhóm đã định trước theo sourceinkbboxmaxside≤64: indices50,58,62,71,73,75,77. Cảnonwhitebbox lẫninkbbox cho cùng7query trong mẫu này.

|Nhóm|MPcenter−original|MPzoom−original|TQcenter−original|TQzoom−original|
|---|---:|---:|---:|---:|
|Small7|−.001769168|+.083179346|−.004429765|−.003212053|
|Other93|+.020794031|+.013875373|+.015854717|+.008805025|

|Query/class|Zoom×|MP original→zoom|TQ original→zoom|
|---|---:|---:|---:|
|50 hedgehog|2.800|.915472→.853150|.943236→.748089|
|58 strawberry|2.835|.151444→.193891|.176474→.113972|
|62 pistol|3.446|.141209→.780832|.148234→.152713|
|71 bell|5.091|.092895→.340341|.068765→.375749|
|73 bell|3.111|.612695→.230768|.559057→.336070|
|75 trumpet|5.209|.069340→.098664|.058347→.070526|
|77 trumpet|4.480|.099041→.166705|.050438→.184947|

MP5/7 tăng, TQ4/7 tăng; mean có thể bị ca lớn chi phối. Cùng mộtphépzoom, query62 tăng rất mạnh ởMP nhưng gần nhưkhông đổi ởTQ. Query73 là phản ví dụ rõ cho “nhỏ thì zoom luôn giúp”.

[Hình đủ7query, không chọn theo outcome](../outputs/scale_local_reliability_20260910T064000Z/parent/tiny7_scale.png).

Nét phóng lớn có thể chỉ là nét nguồn vốn nhòe được nội suy lớn lên. “Cho vật thể chiếm nhiều patch hơn” không tự động bằng “khôi phục chi tiết bị thiếu”.

## 5. Kết quả local reliability

### 5.1. Dịch2pixel có đổi mạnh không?

|Thước đo trên3066neighbors|MP-Q|TQMP|
|---|---:|---:|
|Median abs(AP_neighbor−AP_base)|.003458917|.003608469|
|Percentile95 abs(AP_neighbor−AP_base)|.054566223|.058170568|
|Maximum abs(AP_neighbor−AP_base)|.528040081|.559154958|
|Số abs(AP_neighbor−AP_base)>.05|175/3066|193/3066|
|Signflips có abs(cảhaiΔ)≥.001|216/3066 (7.05%)|241/3066 (7.86%)|
|Query có ít nhất1signflip|69/100|75/100|
|Familyrange median, gồm base+neighbors|.014569491|.014631741|
|Familyrange percentile95, gồm base+neighbors|.133979699|.148451923|

Trong2391/2455neighbors có cảbase/neighborhiệuứng≥.001, sốflip vẫn216/241. Các mẫu số khác nhau trả lời câu hỏi khác nhau; không gọi7.05% là xác suất sai của model.

**Đa số thay đổi nhỏ, một phần có nhạy mạnh.** Không nên nói “toàn bộ local đều bất ổn”. Ngược lại, chỉ nhìn mean signed nhỏ sẽ che mất đuôi lớn và các biến động trái dấu.

### 5.2. Giữ lượng/vị trí nét bị xóa tương đối giống nhau

|Matched Jaccard/count, ảnhpixel khác:1389neighbors|MP-Q|TQMP|
|---|---:|---:|
|Median abs(AP_neighbor−AP_base)|.004248738|.004268318|
|Percentile95 abs(AP_neighbor−AP_base)|.049327123|.055186129|
|Signflips rõ|78/1389 (5.62%)|93/1389 (6.70%)|
|Query có signflip|40/100|51/100|
|Số abs(AP_neighbor−AP_base)>.05|66/1389|80/1389|

Giới hạn: Jaccard/count không xác nhận semantic equivalence. Một nhánh nối hoặc chỗ khép đường viền có thể chỉ vài pixel nhưng đổi cấu trúc. Kết quả ủng hộ **độ nhạy với cách xóa local**, không chứng minh model đang bám feature vô nghĩa hoặc đã hiểu sai semanticpart.

Có1522/3066neighborcửa sổ chạm sốinputpatchkhácbase(1→2),1544vẫn1patch. Đây là đếm patch mà **hình chữ nhật mask** chạm, không nhất thiết patch thực sự có pixelđổi. Geometry vàinkcontent cùng đổi nên chưa thể quy nguyên nhân riêng cho patchboundary. Không chạy thêm patchsize/backbone ablation.

### 5.3. Hai ví dụ nổi bật lượt trước có phải may rủi một cửa sổ?

Giữ nguyên query/vùng đã được nêu ở báo cáo trước; liệt kê đủbốn hướngjitter, không lựa hướng đẹp nhất.

**MP-Q q61 pistol:** originalAP.787144; xóa base.236850. JitterAP:

- Right2px: .612948
- Left2px: .194294
- Down2px: .260761
- Up2px: .262524

Cảbốn vẫn làm kém original, nhưng độ lớn khác mạnh. **Hiệu ứng xấu có tính nhất quán về dấu trong vùng lân cận này; cường độ rất nhạy với mép cửa sổ.** TQMP cùngquery cũng giảm ở cảbốn hướng sooriginal(.636072), APjitter≈.346378–.437573.

**MP-Q q54 hedgehog:** originalAP.308198; base.810579. BốnneighborAP .722659/.797017/.805336/.796892: **đều cải thiện rõ**. TQMP cũng tăng ở cảbốnneighbor từoriginal.299162 đến≈.524170–.717298.

Vì vậy hai ví dụ đó không chỉ là một vị trí ngẫu nhiên làmđổi dấu; vẫn chưa biết việc xóa loại bỏ cuegây nhiễu hay tạo pattern thuận lợi khác. Không có annotation bộphận để suy ra model “hiểu gai/mắt/cò súng” từ những số này.

[Hình original/base/đủ4jitter củaMP-Q](../outputs/scale_local_reliability_20260910T064000Z/parent/prior_examples_jitter.png).

## 6. Điều gì thực sự được hiểu rõ hơn?

### 6.1. Meanpool không làm model tự nhiên bất biến vị trí

Ta có `q(x)=normalize(W mean(H(x))+b)`. Mean bất biến với việc **hoán vị một tập feature cố định**. Nhưng dịch ảnh2pixel không phải chỉhoánvịfeatures:

- Pixel phân bố lại trong các patch32×32.
- Positionalembeddings và tương tác attention thay đổi.
- Các finalpatchtokens không còn là permutation của bộcũ.

Do đó không thể suy từmeanpool ra `q(translate(x))=q(x)`. Center-only change đã cho thấy invariance này không có trên nhiều query. Chưa tách riêng ảnh hưởng positionalencoding, patchalignment hay những tham sốstudent/head.

### 6.2. Recipe train hiện tại có deletion, không có scale/translation augmentation riêng

Đọc lại `src/spica/train_coupled_predictive.py`: loader dùng `bundle.transform` của CLIP evaluation. `src/spica/data/coupled_views.py::region_pair` tạo cleanclone vàmaskedview ở vị trí cũ, chọnmứcxóa25/50/75%. Không có bước randomscale/randomtranslation riêng trong đường này.

Đây là **sự khác biệt cụ thể của recipe**, không phải kết luận rằng chỉ thêm augmentation sẽ giải quyết được. Masking dạy đối phó với thiếuvùng; nó không tương đương dạy ổn định với việc toàn hình chuyển chỗ hoặc đổi scale.

### 6.3. Cập nhật các nghi ngờ A–D

**A. Pretrainedreadout:** vẫn có sự thay thế CLS/LN/projection bằng meanpatch+newlinear, nhưng hiện không thể nói thiếuCLS là nguyên nhân. Dữ liệu mới cho thấy q cólocalresponse và position/scale sensitivity; không có controllednativeCLS training/readout ablation. Broadstudent adaptation có thể làm thay đổi invariance đãpretrain, nhưng chưa đo quan hệteacherđểkếtluận.

**B. Photo→textCE:** là thiếu ràngbuộcsemantic trực tiếp cho actualgalleryphoto, khác vớiCE_i của sketchhead. Chưa có canthiệpB; dữ liệu này không chứng minh bổ sungphotoCE sẽ sửa nhạyvịtrí. Deeptextcoupling và CE là hai cơchế khác nhau, không nhập làm một.

**C. Sketchreferenceconsistency:** có thể là hướng giữsemantic/invariance, nhưng phải chọn biếnđổi mà nội dungclasscòn giữ. Épconsistency với bấtkỳ deletion nào có thể làm model bỏ qua chi tiết hữu ích. OriginalCLIPteacher cũng có hạn chế. Diagnostic này không xác nhận forgetting, Gaussianity haycollapse.

**D. Budget:**3600updates≈2.47originalsketchobservationpasses. Chưa đo learningcurve theo cùnggeometricdiagnostics; không biết sensitivitydo chưa học đủhay doobjective/parameterization. Tăngsteps không tự sửa một loại invariance khôngđược giám sát trongrecipe.

**Trọng tâm mới có bằng chứng tốt hơn:** phân biệt (i) cóthể nhìn thấy tín hiệu local, (ii) sử dụng tín hiệu đó ổn định, (iii) tín hiệu có ý nghĩa semantic và chuyển giao được. Lượttrước ủnghộ(i), lượtnày tìm thấy hạnchế của(ii), còn(iii) chưađủevidence.

## 7. Kiểm chứng và giới hạn số học — không ép PASS

### 7.1. Producer và input gates

- `prepare.py`, `inputs/receipt.json`: sourceSHA/identity/same100queries,300scaleimages và3838validlocalmasks;22unavailable giữ nguyênmetadata.
- `independent_gate/verify.py`, `receipt.json`: **PASS**,100sourceSHA/300exactscaleconstruction/alljittergeometry/supportpreservation/productiontransformparity, source/readout/archivedsafeloader reviewed trướcGPU.
- `measure.py`, `measurement/summary.json`: **COMPLETE_NO_UPDATE**, cảhaiweights trước/sau exact, gradientsNone,0backward/updates, predictorprobe0calls. CachedgallerySHA bound. GPUmeasurementelapsed≈15.75s, không deploymentbenchmark/totalanalysistimeclaim.

### 7.2. Independent fullsort gate **FAIL**, giữ nguyên

`independent_results/verify_and_analyze.py`, `receipt.json`, `summary.json`, `report.md`:

-All8276availablemodel-viewrows sorted fullgallery bằngFP64 normalizedsavedq; khôngsecondencoder.
-PredeclaredfullAPmaxdelta2e−5 bị vượt ở **1row/arm**:
  - MP max2.212899022180359e−5.
  - TQ max2.8367853356414674e−5.
-Top200rankrows khôngexact73/4138MP,64/4138TQ; retainedas numerical/ranking discrepancies, chưa quy riêng cho exactties.
-Savedtop200 relevance→P200/3AP200 re-evaluation **PASS**≤2e−6; actualmax≈1.284e−7.
-Khôngnới ngưỡng, sửaraw, re-encode/train lại để épPASS. Khônggọi toànrunVERIFIED. IndependentFAIL là numericalconformance gate, không chứng minh training thấtbại.

### 7.3. Identical-image controls vẫn có sai khác nhỏ

233neighborso vớibase cho pixelidenticalimages đãxácnhận. SavedGPUAPmaxdiff3.06964e−5MP/4.13507e−5TQ;0meaningfulsignflip. Maxqcoordinatedelta7.67112e−5/1.12228e−4, cosine distances≤1.34923e−7/2.46131e−7. Vì vậy batchsize256cốđịnh **không** tạo bảođảmexactduplicateởmọivịtrí. Không xác định kernelcause từ lượt này.

Các biếnđộng lớn0.05–0.5 lớn hơnnoise quan sát nhiều, nhưng các win/loss gần0 không nên diễn giải. Bảng chính dùng threshold.001 cho thayđổicóýnghĩa mô tả; đây không phải statisticalsignificance threshold.

### 7.4. Parent checks và hiệu chỉnh cách gọi một số trường independent

`analyze_parent.py`, `parent/summary.json` có status **SCOPED_PARENT_CHECKS_PASS_INDEPENDENT_FULL_AP_GATE_FAIL**:

-Bound nguồn/input/producerfiles/stateassertions, đọc pixelidenticalcontrols thực sự.
-All8276savedtop200 P200/3AP200 formulas PASS.
-Recomputed savedGPUscale/localaggregation bằngFP64; independentscale5metricmeans khác tối đa<2e−6.
-Giữ independentFAIL, không coi nó được parentPASS thay thế.

Hai cách gọi cần đọc đúng trong independentJSON:
-`faint_nonwhite_outliers=100` chỉ kiểm tra nonwhitecount>thresholdedinkcount: cho thấy antialias/nétmờ hiện diện, **không chứng minh100ảnh có nhiễu/outlier**. Parent/report không dùng tênđó đểkếtluậnnhiễu.
-`family_full_ap_range` của independent tính range của neighbors, không gồm base. Bảngmục5 dùng parentrange gồm **base+mọineighborhợplệ**, không tráo định nghĩa.
-Independentquerymacroempty-subsetfill0 chỉ ảnhhưởng subsetexact-imagecontrols thiếuở30queries. Parent tính subsetmean trênqueriescóeligibledata và ghi denominator70; không dùngmissing=0 đểkhẳngđịnh robustness.

## 8. Đề xuất, không phải thay đổi đã triển khai

1. **Không tự thêm zoom/centering default:** cảhai có ca thua lớn; cùngmẫu100đãquan sát nhiều lần. Cần lockpolicy và kiểm tra tập pseudo-validation rộng hơn khi được duyệt, không tốiưutrênofficialtest.
2. Nếu muốn thay training, ưu tiên một **controlledgeometry-invariance intervention** (vídụ scale/translation hợp lý, giữnét và ngữcảnh), giữ model/losscontrol để biết nguyênnhân. Chưa được thực hiện hoặc chọn hyperparameters từ kếtquảnày.
3. Nếu muốn đánhgiá “hiểu local”, cần một task/annotation hoặc counterfactualsemantic đángtin. Không ép mọi crop/regionpartial mang nhãn toànảnh, không suy từattention/sensitivity ra understanding.
4. Sau đó mới so controlledreadout/teacherconsistency/photoCE từng yếu tố; không cộng đồngthời rồi quy gain cho text.

**Kết luận cuối:** model không thiếu hoàn toàn khả năng đọc chi tiết. Bằng chứng hiện tại rõ hơn về **sự nhạy với hình học đầu vào và độ ổn định không đồng đều của tín hiệu local**, chứ chưa chỉ ra một thành phần thiếu duy nhất giải thích khoảng cách benchmark.

## 9. Đường dẫn bằng chứng

- Protocol: `outputs/scale_local_reliability_20260910T064000Z/inputs/protocol.json`, `measurement/protocol_lock.json`.
- Rawinputs/rowIDs/masks: `inputs/scale_images.npy`, `local_masks.npz`, `row_manifest.json`.
- Rawfeatures/ranking: `measurement/MP_Q/` và`measurement/TQMP/`.
- Independentinputgate: `independent_gate/receipt.json` **PASS**.
- Independentretrieval: `independent_results/receipt.json` **FAIL preserved**.
- Parentprecision/denominators/classdeltas/querydeltas/examples: `parent/summary.json`, `MP_Q_local_records.json`, `TQMP_local_records.json`.
- Figures: `parent/tiny7_scale.png`, `parent/prior_examples_jitter.png`.
- Previousreport: [local information](local_information_diagnostic_2026-09-10.md); priorA–Dbenchmarkaudit: [TQMPresults/gap](fusion_tqmp_results_and_benchmark_gap_2026-09-10.md).

No production source edits. Subsequent documentationcommit is not either training snapshot or the premeasurementcomponentlock.

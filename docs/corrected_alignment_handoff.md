# SPICA corrected alignment — bàn giao cho ChatGPT Web

## Đọc nhanh

Campaign đã chạy thật và kiểm chứng xong trên RTX5070Ti/CUDA: **negative result, không promote mainline**. Đây là kết quả mới seed42/pseudo split3407, không lấy số từ report historical để điền vào.

| Arm | mAP@1800 | Δ vs R | Matching |
|---|---:|---:|---|
| R (control) | 0.671575253 | — | VALID_CONTROL |
| MD (mean-only, detached) | 0.667110088 | −0.004465165 | MATCHED |
| MS (mean-only, symmetric) | 0.654153227 | −0.017422026 | MATCHED |

**MS − MD = −0.012956861.** Cả ba pairing dùng strict **corrected_v2**. Covariance weight=0; MD/MS dùng chung λ=`0.2603831284137216` từ calibration CLIP thật, target sketch-gradient ratio=0.1.

**82 targeted tests + 13 pairing mutation checks PASS; Ruff PASS; CUDA forward/backward/synchronize PASS.** Một lỗi chỉ xuất hiện khi chạy smoke: trainer truy cập `CategoryRetrievalEvaluation` như dictionary. Sửa đúng ba chỗ đọc mAP, rồi chạy lại calibration source-bound, smoke50 và pilot1800 tuần tự. Không đổi model/loss/dependencies.

Corrected mean gap@1800 giảm từ R0.796686 xuống MD0.733997/MS0.625390 nhưng mAP cũng giảm: **moment agreement chưa đủ cho retrieval**. Bootstrap là uncertainty trên10,963 queries của **một seed/split**, không phải multi-seed evidence. Không chạy official unseen, covariance expansion,5400 steps hoặc lambda search.

## Thứ tự đọc đề nghị

Tất cả evidence dưới `outputs/alignment_verification_20260906_093823_82c6461/`:

1. [Verification summary](../outputs/alignment_verification_20260906_093823_82c6461/verification_summary.md) — kết quả kiểm chứng A–G, bug/reproducer, nguồn gốc và giới hạn.
2. [Pilot summary](../outputs/alignment_verification_20260906_093823_82c6461/corrected_pilot_summary.md) — bảng primary/secondary, bootstrap, geometry và runtime.
3. [Tests and validation](../outputs/alignment_verification_20260906_093823_82c6461/tests_and_validation.md) và [GPU preflight](../outputs/alignment_verification_20260906_093823_82c6461/gpu_preflight.md) — exact commands, exit codes; có raw logs đi kèm.
4. [Calibration JSON](../outputs/alignment_verification_20260906_093823_82c6461/calibration_diagnostic_corrected.json) — detached/symmetric đo riêng, fixed batches, initialization/source identity và restoration.
5. [Matching validation JSON](../outputs/alignment_verification_20260906_093823_82c6461/matching_validation.json) — actual R–MD/R–MS/MD–MS; không chỉ synthetic tests.
6. [Summary JSON](../outputs/alignment_verification_20260906_093823_82c6461/corrected_pilot_summary.json), [resolved configs](../outputs/alignment_verification_20260906_093823_82c6461/resolved_configs/) và [artifact inventory](../outputs/alignment_verification_20260906_093823_82c6461/artifact_inventory.json) — lineage đến run ID/source/checkpoint/step/hash.
7. [Campaign status](../outputs/alignment_verification_20260906_093823_82c6461/campaign_status.json) — completed và phần còn unverified.

Nếu giao diện GitHub không render JSON/log, dùng nút **Raw**. Các file text quan trọng được commit riêng để đọc mà không cần giải nén.

## Raw evidence đầy đủ

Tải [review_bundle.tar.gz](../outputs/alignment_verification_20260906_093823_82c6461/review_bundle.tar.gz) (~5.1MB), giải nén hoặc upload vào ChatGPT có khả năng đọc file:

- Raw `run_result.json`, training histories, bootstrap và offline gradient/geometry metrics.
- Reports, calibration, matching evidence, configs, logs và source archive/patch.
- [Danh sách file và SHA256](../outputs/alignment_verification_20260906_093823_82c6461/review_bundle_files.json); [SHA256 archive](../outputs/alignment_verification_20260906_093823_82c6461/review_bundle.sha256).
- Không chứa backbone/checkpoint tensor lớn. Đường dẫn local và SHA256 vẫn được giữ để xác minh sau. Các `run_result.json` ~25MB/arm nằm trong bundle, không commit lặp lại dưới dạng text khổng lồ.

**Bundle là snapshot bằng chứng bất biến trước commit bàn giao.** `published:false` và câu “local only” phản ánh thời điểm kiểm chứng; commit local không đồng nghĩa đã push/publish. Đường dẫn tuyệt đối trong raw evidence là máy đã chạy, không phải đường dẫn có thể truy cập từ ChatGPT Web.

## Provenance và lịch sử

Training source: reviewed HEAD `82c6461` + [execution fix](../outputs/alignment_verification_20260906_093823_82c6461/execution_fix.patch), SHA256 snapshot `a30cb43ecc0dfebafc47d3a214c6c61803d03148f1ebc901b4d9bab1546ff3b9`. File AGENTS/bàn giao và commit hiện tại được thêm **sau training**, không relabel nguồn của run.

Pilot historical có đủ raw artifacts, nhưng calibration cũ không đo symmetric gradient thật và thiếu evidence replay cần cho corrected-v2. Nó không được tự nâng thành MATCHED, không được gắn calibration mới hồi tố. Các artifacts đó được giữ nguyên; lý do retrain không đơn thuần là HEAD thay đổi.

## Prompt gợi ý cho ChatGPT Web

> Đọc AGENTS.md và docs/corrected_alignment_handoff.md trên branch experiments, rồi lần theo verification_summary, pilot summary và matching/calibration evidence. Tóm tắt điều đã được xác nhận và giới hạn khoa học. Phân biệt historical matching với corrected_v2 và training source với commit bàn giao. Không đề xuất retrain chỉ vì HEAD khác; không gọi kết quả một seed/split là robust improvement.

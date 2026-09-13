# 6. Tài liệu tham khảo và cách dùng

Metadata tên/tác giả/arXiv dưới đây được đối chiếu trực tiếp trang abstract của arXiv trong lượt chuẩn bị13/09/2026. Việc đọc abstract **không phải audit evaluator/recipe toàn văn**. `references.bib` dùng bản arXiv để không bịa volume/page/DOI conference.

1. Radford et al. (2021), **Learning Transferable Visual Models From Natural Language Supervision**. [arXiv:2103.00020](https://arxiv.org/abs/2103.00020). Dùng cho cơ sở CLIP và tiền huấn luyện ảnh–ngôn ngữ.
2. Sain et al. (2023), **CLIP for All Things Zero-Shot Sketch-Based Image Retrieval, Fine-Grained or Not**. [arXiv:2303.13440v3](https://arxiv.org/abs/2303.13440v3). Metadata ghi acceptedCVPR2023. Dùng cho prompt learning trong category/fine-grained ZS-SBIR; tên gọi nội bộ SketchLVM không thay cho tên bài đầy đủ khi trích dẫn.
3. Long Hoang Dang et al. (2026), **SeCo-SBIR: Semantically Consistent Prompt Learning for Zero-Shot Sketch-Based Image Retrieval**. [arXiv:2608.03120v1](https://arxiv.org/abs/2608.03120v1). Abstract mô tả text-guided multimodal prompting và asymmetric consistency với frozen reference. Không từ đó gọi TU-SREF nội bộ là reproduction đầy đủ SeCo.

## Nguồn nội bộ

- [Official MP-Q protocol](../minimal_official_campaign_2026-09-12.md).
- [20% restart](../percent20_official_campaign_2026-09-12.md) và [remaining TU/Q](../percent20_remaining_campaign_2026-09-12.md).
- [F2 kiến trúc](../coupled_predictive_fusion_v2_implementation_2026-09-09.md), [MP loss](../fusion_multipositive_campaign_2026-09-09.md), [q readout](../fusion_mp_q_readout_results_2026-09-09.md).
- [Baseline official TU/Q](../coupled_official_benchmarks_results_2026-09-11.md), [SREF TU](../coupled_sketch_ref_tuberlin_results_2026-09-11.md).

Nội bộ là hồ sơ phương pháp/thực nghiệm của đồ án, không công bố peer-reviewed. Dùng source snapshots/receipts để truy số liệu, dùng papers để ghi nhận công trình liên quan.

## Chưa đủ cho một bibliography đồ án hoàn chỉnh

Cần bổ sung tài liệu gốc Sketchy, TU-Berlin, QuickDraw và protocol/splits; paper Transformer/ViT, contrastive/supervised contrastive learning, AdamW và retrieval metrics nếu có giải thích tương ứng. Chưa xác minh metadata các nguồn này trong bộ chuẩn bị, nên không tự điền trích dẫn giả hoặc lấy số SOTA chưa đối chiếu. Đọc toàn văn trước khi xây dựng bảng so sánh định lượng với paper; đặc biệt cutoff TU P@100 và AP200 denominator.

# SketchLVM — manifest benchmark upstream và kiểm tra lại FP64

## Quyết định của user

User yêu cầu train dùng official và kiểm tra kỹ FAIL FP64, sau đó chọn **đối chiếu manifest benchmark gốc**, rồi **giữ run60 đã hoàn tất, không train trùng** khi xác nhận run cũ đã đọc nguyên manifest upstream. Không mở campaign mới, không sửa dataset/config/model/evaluator, không GPU/encoder replay, không đổi tolerance/raw status hoặc push. Kết quả training giữ nguyên trong [báo cáo60epochs](sketchlvm_raster_results_2026-09-30.md).

## 70 sketch/class không phải local subsampling

Đã đối chiếu cả6manifest/class-map với repository chính thức **ZSE-SBIR**, pinned commit `88f87f769208f9de5b08f2d7f0569a8087ee5b4b`. Git blob SHA1 và byte length của từng file local khớp metadata release; SHA256 local được lưu trong receipt. Loader upstream `get_all_train_file()` đọc trực tiếp `png_ready_filelist_train.txt`, không glob toàn bộ80sketch/class.

| File | Rows | Classes | Rows/class |
|---|---:|---:|---:|
| `png_ready_filelist_train.txt` | **15.400** | **220** | **70** |
| `png_ready_filelist_zero.txt` | **2.400** | **30** | **80** |
| `ImageResized_ready_filelist_train.txt` | 176.081 | 220 | Không đều |
| `ImageResized_ready_filelist_zero.txt` | 27.989 | 30 | Không đều |
| `cname_cid.txt` | 220 | 220 | 1 |
| `cname_cid_zero.txt` | 30 | 30 | 1 |

Nguồn primary:

- [Repository/release](https://github.com/buptLinfy/ZSE-SBIR/tree/88f87f769208f9de5b08f2d7f0569a8087ee5b4b).
- [Published TU manifests](https://github.com/buptLinfy/ZSE-SBIR/tree/88f87f769208f9de5b08f2d7f0569a8087ee5b4b/datasets/TUBerlin/zeroshot).
- [Loader đọc manifest](https://github.com/buptLinfy/ZSE-SBIR/blob/88f87f769208f9de5b08f2d7f0569a8087ee5b4b/data_utils/utils.py).

Physical pool có80PNG/class, bao gồm2.200seen sketches ngoài manifest. Không có validation manifest local; không tự gọi10sketch/class đó là validation. Thêm chúng sẽ thay đổi pool manifest đã công bố, nên không thực hiện theo lựa chọn của user.

**Đính chính diễn đạt trước:** run này không tự chọn70sketch/class; nó dùng nguyên manifest benchmark công khai. Tuy nhiên, chứng minh khớp **ZSE-SBIR release** không chứng minh tác giả **SketchLVM** đã dùng đúng cùng pool/sampler/preprocessing. Không nâng thành exact paper reproduction, và chưa xác định vì sao mAP thấp hơn số paper.

Evidence: `outputs/sketchlvm_official_manifest_review_after_user/{verify.py,upstream_directory.json,upstream_loader.txt,receipt.json,verify_with_system_ca.log,parent_bindings.json}`. Parent bindings đối chiếu cả6SHA với actual production config và enumerate seen filenames17600/2200unlisted + local TXT paths (không image decode). Lần chạy metadata đầu bị lỗi CA trong Python, log `verify.log` giữ nguyên; lần sau parent truyền system CA environment, không bỏ xác minh TLS/dependency change. Receipt/stdout không độc lập capture CA environment của process. Chỉ đọc metadata/source, không download dataset hay ảnh.

## FP64: kiểm tra độc lập128dòng, FAIL vẫn còn

CPU-only trên saved train-gate features/ranks. Worker dùng NumPy FP64 normalization/full sort; parent kiểm tra độc lập toàn bộ128query-condition rows và full AP bằng `math.fsum`. PT/NPZ arrays và CSV path/label/order được đối chiếu. Full AP denominator=R; AP200 denominator=min(R,200); P200 denominator200.

- **Đúng1/128dòng FAIL** ở raster25%, query28, `chair/4022.png`.
- Full AP producer CUDA đã lưu: **0.9361964464187622**; đây là scalar từ saved tensor, không independent native CUDA replay.
- Independent normalized FP64 full AP: **0.9362191067163292**.
- Absolute delta **2.266029756703425e-5 > tolerance2e-5**, không nới tolerance.
- Tất cả saved top200 AP200/P200 formulas khớp; parent max error **1.1102230246251565e-16 / 0**.
- Original `cpu_audit.json` SHA trước/sau đều **`14c8ca068bac0cb549d132d65d9761bb75e5d19c7f6a30eec022f2c90b3b4b40`**.
- `parent_bindings.json` bind saved failing tensor→summary artifact SHA→original gate receipt và provenance SHA. CUDA execution là producer/source evidence; không independent replay hoặc bằng chứng native tail rank.

### Tail near-tie: bằng chứng mạnh, chưa chứng minh CUDA order

Trong FP64, positive gallery195 đứng rank208, negative gallery86 đứng rank209. Cosine gap **1.607775225176411e-7**. Query cóR=64, đã có62positives trước rank208.

Giả sử positive lùi208→209, còn negative tiến209→208:

```text
ΔAP = 63/64 × (1/209 − 1/208)
    = −2.264388571948661e−5
```

AP FP64 của thứ tự giả định này là0.9361964628306098, cách saved CUDA scalar chỉ1.6411847548e-8. Khi tính AP của thứ tự giả định bằng đúng FP32 arithmetic, scalar **khớp saved CUDA AP chính xác**:0.9361964464187622.

Điều này hỗ trợ rất mạnh giả thuyết numerical near-tie ở tail. **Nhưng historical native CUDA full scores/rank không được lưu**, nên scalar match không chứng minh rank CUDA thực sự hoặc nguyên nhân kernel/reduction/precision. Các FP32CPU/raw-dot variants cũng không phải replay CUDA. Không relabel rawFAIL thànhPASS và không suy ra encoder/model đã được fully verified.

Worker prose bị đảo hướng rank trong `report.json` và có câu rank không chắc trong `report.md`; các numeric pair ranks ở `results.json` đúng. Giữ worker originals, parent receipt/report ghi diễn giải đã sửa. Receipt gốc của gate/producer không bị sửa.

Evidence: `outputs/sketchlvm_fp64_recheck_after_user/parent_verify.py`, `parent_verify.log`, `parent_receipt.json`, `parent_report.md`; worker `recheck.py`, input hashes/results/receipt/logs cùng root được giữ nguyên. Parent status **`PARENT_CPU_RECHECK_CONFIRMED_ORIGINAL_FP64_FAIL_RETAINED`**. Không chạy independent final official full-gallery sort/image replay trong lượt này.

Nếu sau này mở gate mới: lưu native full-gallery scores/ranks và numeric provenance để đối chiếu với FP64 reference. Đây là yêu cầu evidence tương lai, không permission mở thêm gate/train/replay. Final producer vẫn **`FINISHED_UNVERIFIED`**.

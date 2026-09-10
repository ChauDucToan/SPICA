# F2_MP_PCE — kết quả và kiểm tra read-only

## Kết luận

**Đã hoàn thành 3600/3600 updates, exit0. Không thay MP-Q bằng PCE:** clean full mAP thấp hơn `.000751405`; masked full tăng rất nhỏ `.000240082`. Một seed, không chứng minh khác biệt thống kê hoặc photo-CE luôn vô ích. Với λ và budget này, chưa có lợi ích rõ ràng trên mục tiêu chính.

## Identity và hoàn thành

- Root `outputs/fusion_photo_ce_execution_20260910T125500Z/`; arm `F2_MP_PCE`, λ `.03291338002672159`, tau `.07`.
- Training main μ_I, inference q; giữ F2_MP + photo→train-text CE, một lần/unique live bank/update. Không SIG.
- Training HEAD `f0f1f532167789c3ae829b56075a1aab85d35bf2`; source583files SHA `93add4c28fff102a25f2dcfc7378d27656e212964d4108c2f31835ba6b4061c6`.
- Runtime12:50:28→13:29:43UTC, **39 phút15 giây**, `child_pid:null`, exit0. Không còn trainer/runner active lúc kiểm tra.
- W&B [kjw8pg7s](https://wandb.ai/a-cctest05187-erd/spica/runs/kjw8pg7s), state `finished`.
- latest/best_clean/best_masked đều3600, cùng checkpoint SHA **`9e0a4e8e93a50fc306a3273be57d8c1bc1ef609e25f373836e84aad6127b629f`**. Best aliases vẫn chọn prefix AP200, không thay selection rule.
- Raw runtime **ARM_FINISHED_UNVERIFIED** giữ nguyên. Report này không đổi nó thành whole-campaign VERIFIED.

## Fixed-step q@3600

| Metric | MP-Q reference | PCE | PCE−MP-Q |
|---|---:|---:|---:|
| Clean full mAP | .491669182 | .490917777 | −.000751405 |
| Clean P@200 | .520355275 | .517288140 | −.003067135 |
| Clean prefix AP200 | .535433599 | .532160915 | −.003272684 |
| Masked macro full mAP | .355752598 | .355992680 | +.000240082 |
| Masked macro P@200 | .358522295 | .357722737 | −.000799558 |
| Masked macro prefix AP200 | .373294285 | .372109138 | −.001185147 |

Clean full giảm **0.07514 điểm phần trăm**; masked full tăng **0.02401 điểm phần trăm**. Không phải .07514% relative. Cả5 clean metric giảm. Masked không đồng loạt tốt hơn.

So TQMP@3600: clean full `.485342514`→PCE`.490917777` (+.005575262); masked full `.354266677`→`.355992680` (+.001726004). PCE tốt hơn TQMP trên các metric/conditions đã lưu, nhưng **chưa vượt MP-Q**, không attribution text-only qua hai recipe khác nhau.

### Δ full mAP theo từng mask, PCE−MP-Q

| Deletion | seed101 | seed202 | seed303 |
|---|---:|---:|---:|
|25%|−.000174474|+.000248896|−.000981787|
|50%|+.000807678|−.000539621|+.000414113|
|75%|+.000451183|+.001393056|+.000541694|

6/9 điều kiện tăng,3/9 giảm; không khẳng định mọi query/class đều hưởng lợi.

### PCE trajectory

| Step | Clean full | Masked macro full |
|---|---:|---:|
|0|.088526295|.087114666|
|600|.415563415|.298307338|
|1200|.474843053|.342112827|
|1800|.466514184|.337567391|
|2400|.485347767|.348603748|
|3000|.490005973|.355796014|
|3600|.490917777|.355992680|

Không suy ra cần train lâu hơn hoặc tự mở rộng horizon từ đường này.

## Loss: constraint hoạt động, không đảm bảo retrieval gain

Trung bình60 logged batches3010–3600:

| Loss | MP | PCE |
|---|---:|---:|
|Clean q CE (`clean_ce_pool`)|.140866151|.140641365|
|Clean μ_I MP|.902104214|.897984564|
|Photo reference anchor|.116945676|.129338277|
|Text reference anchor|.141499481|.155376106|
|Actual photo CE|không log|1.460875332|

PCE weighted photo CE≈.0481. Tại initialization4batches photoCE≈3.17–3.20; late logged mean≈1.46, nhưng khác batch/state. **Không có matched final MP photo-CE measurement**, nên không nói photo-CE PCE chắc chắn tốt hơn MP, hoặc photo/text drift là nguyên nhân retrieval thay đổi. qCE gần như không đổi. Không suy raw gradient thành effective AdamW update.

Independent summary có field `qCE_clean_ce_i` đặt tên nhầm: đó là **μ_I CE**, không q. Parent giữ original summary và ghi correction ở `parent_receipt.json`; bảng trên dùng đúng `clean_ce_pool`.

## Kiểm chứng đã thực hiện

`analysis_readonly_20260910/verify_readonly.py`, `summary.json`, `receipt.json`, `wandb_history_kjw8pg7s.jsonl`:

- Online361unsampledrows0..3600; **490retrieval +11,887loss/gradient/LR scalars exact local/online**.
- Streaming SHA actual selected checkpoint;583archivedfiles +aggregate verified.
- Initialization,115200observationtraces/masks và LRhistory byte-identical historical F2_MP. Frozen before/after hashes do producer ghi bằng nhau; không independent tensor replay.
- Parent `verify_parent.py` / `parent_receipt.json`: rebind verified calibration SHA/raw/init/data/λ, fixed qbaseline SHA, saved final probe vs selected metrics/deltas, correct loss interpretation; hashes script/report/raw evidence.
- **Không GPU/encoder/checkpoint deserialization/retrieval replay, không independent all-query sorting, không online artifact download.** Online scalar verification không thay các tầng đó.
- Không train thêm, không đổi default/head/λ, không promote. Preserve unrelated dirty files/historical failures; local report commit không phải training snapshot.

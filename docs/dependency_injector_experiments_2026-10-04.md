# DI framework cho thí nghiệm: dependency-injector

Người dùng chọn **dependency-injector**, không chỉ DI thủ công bằng callable. Phiên bản pin: **4.49.1**.

## Container thực sự

[`TrainingContainer`](../src/spica/containers.py) kế thừa `containers.DeclarativeContainer`:

- `student`: `providers.Factory`, tạo encoder mới khi resolve.
- `retrieval_loss`: `providers.Callable`, thực thi loss trên mỗi batch.
- [`train_student.py`](../src/spica/train_student.py) nhận container và dùng chính providers để tạo Student/tính loss. CLI và API callable cũ cũng được compose vào container; không có nhánh chạy bỏ qua framework.
- Override/reset/context manager dùng API của **thư viện**, không tự viết container/registry. Không cần wiring decorator vì trainer là composition root và resolve providers trực tiếp.

Ví dụ đổi Student mà không sửa trainer:

```python
from dependency_injector import providers
from spica.containers import create_training_container
from spica.models.sketch_student import SketchStudent
from spica.models.vmamba_student import build_vmamba_tiny
from spica.train_student import multi_positive_loss, train

container = create_training_container(SketchStudent, multi_positive_loss)
container.student.override(providers.Factory(build_vmamba_tiny))
train(args, container=container)  # args không chứa hai flag component bên dưới
```

Loss khác hoặc constructor có tham số:

```python
# Các implementation do bạn cung cấp, giữ contract B×512 và loss bốn tensor.
with container.student.override(providers.Factory(MyEncoder, width=96)):
    with container.retrieval_loss.override(providers.Callable(my_loss, margin=0.3)):
        train(args, container=container)  # output mới; thí nghiệm phải được phê duyệt riêng
```

`Factory` cho encoder trainable giúp không chia sẻ model giữa các run; không override bằng một model đã train hoặc `Singleton` rồi gọi đó là fresh initialization. Hyperparameter/preset nên khai báo trong `experiment_metadata`; callable name và source archive không tự chứng minh provenance của code/config ngoài repository.

CLI cũ vẫn dùng được và bên trong đi qua framework:

```bash
.venv/bin/python -m spica.train_student \
  --student-factory spica.models.vmamba_student:build_vmamba_tiny \
  --epochs 60 \
  --clip-weights "$CLIP_WEIGHTS" --mpq-summary "$MPQ_SUMMARY" \
  --output "$FRESH_OUTPUT"
```

Không trộn `container=` với `student_factory=`, `retrieval_loss=` hoặc hai flag component. Dependency nội bộ theo type hints; không tự thêm `isinstance`/`raise` để kiểm tra lại container, provider, model. Thư viện xử lý callable/provider sai khi cấu hình hoặc resolve. Giữ guard CLI, tránh ghi đè output, SHA/protocol và nonfinite loss. Defaults CNN–Mamba100/joint10, loss, optimizer/grouping, LR schedule và evaluator không đổi. Config ghi framework/version và target đang override.

## Kiểm tra và snapshot

```bash
PYTHONPATH=src OMP_NUM_THREADS=2 \
.venv/bin/python scripts/check_training_container.py
```

Standalone check nhỏ: Factory tạo instances riêng, override/reset, loss/gradient bằng đúng lời gọi gốc trên tensor CPU. Không image/CUDA/training/W&B. Fixture mở rộng đã chạy riêng trong ignored `outputs/dependency_injector_preparation_20261004/full_framework_check.py`: container/API/CLI/default routes đều qua trainer, hai synthetic CPU steps mỗi route và weights cuối giống hệt nhau. Không CLIP thật hay official retrieval; metric fixture không có ý nghĩa benchmark. Giữ fixture mở rộng ngoài code chính để không làm phình implementation. Đây không phải certification CUDA hay run mới.

Bản sửa được chuẩn bị trong ignored `outputs/dependency_injector_preparation_20261004/`, dependency cài `--target` tách biệt trong khi run VMamba60 còn active. Chỉ áp dụng vào source/`.venv` **sau khi child và supervisor của run đó đã kết thúc**. Thêm duy nhất dependency-injector; các phiên bản cũ không đổi. Không `uv sync` toàn môi trường, không sửa sibling, không resume/retrain/evaluate lại.

**Run VMamba60 đã launch trước bản sửa này dùng DI thủ công**, không được gọi là run dùng framework. Snapshot source80 và raw artifacts/status của nó được giữ nguyên; source sau sửa không phải training snapshot. Actual trainer/config của run đó dùng **FP32**, không BF16: các prefix timing BF16 trước đó không phải benchmark cùng precision với production. Log/checkpoint/launch/execution receipt là authority; protocol/timing cũ được giữ, với correction receipt riêng trong ignored outputs. Không đổi precision giữa run và không nâng FINISHED_UNVERIFIED/historical FAIL thành chứng nhận.

Nguồn API đã đọc: [Factory](https://python-dependency-injector.ets-labs.org/providers/factory.html), [Callable](https://python-dependency-injector.ets-labs.org/providers/callable.html), [provider overriding](https://python-dependency-injector.ets-labs.org/providers/overriding.html).

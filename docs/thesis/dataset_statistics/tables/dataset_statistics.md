# Thống kê sketch ở224×224

Ba bảng tương ứng ba dataset; mỗi thống kê tính giữa các ảnh, không phải trung bình category. SD là độ lệch chuẩn tổng thể (ddof=0). Ảnh trắng được tính cho mật độ, chỉ loại khỏi thống kê độ dày.

**Độ dày là proxy đường kính EDT tại các điểm cực đại cục bộ, không phải bề rộng nét chính xác. Ví dụ nét raster1px cho proxy2px; junction và lưới pixel gây sai lệch.**

## sketchy_104_21

| Split | Đại lượng | Số ảnh hợp lệ | Trung bình | Trung vị | Độ lệch chuẩn |
|---|---|---:|---:|---:|---:|
| train | Số pixel mực (px) | 57587 | 2630.917360 | 2352.000000 | 1427.179166 |
| train | Mật độ mực (%) | 57587 | 5.243378 | 4.687500 | 2.844346 |
| train | Độ dày proxy EDT (px) | 57587 | 4.067157 | 4.001874 | 0.316316 |
| test | Số pixel mực (px) | 12694 | 2629.297857 | 2291.000000 | 1513.042195 |
| test | Mật độ mực (%) | 12694 | 5.240150 | 4.565928 | 3.015470 |
| test | Độ dày proxy EDT (px) | 12694 | 4.056315 | 4.010601 | 0.283425 |

Ảnh trắng: train=0; test=0

## tuberlin_220_30

| Split | Đại lượng | Số ảnh hợp lệ | Trung bình | Trung vị | Độ lệch chuẩn |
|---|---|---:|---:|---:|---:|
| train | Số pixel mực (px) | 15400 | 1172.198831 | 1083.000000 | 500.492497 |
| train | Mật độ mực (%) | 15400 | 2.336174 | 2.158402 | 0.997474 |
| train | Độ dày proxy EDT (px) | 15400 | 2.010093 | 2.004464 | 0.021388 |
| test | Số pixel mực (px) | 2400 | 1153.016250 | 1056.000000 | 474.936257 |
| test | Mật độ mực (%) | 2400 | 2.297944 | 2.104592 | 0.946541 |
| test | Độ dày proxy EDT (px) | 2400 | 2.009940 | 2.004072 | 0.029328 |

Ảnh trắng: train=0; test=0

## quickdraw_80_30

| Split | Đại lượng | Số ảnh hợp lệ | Trung bình | Trung vị | Độ lệch chuẩn |
|---|---|---:|---:|---:|---:|
| train | Số pixel mực (px) | 236080 | 1657.027774 | 1535.000000 | 775.466216 |
| train | Mật độ mực (%) | 236080 | 3.302431 | 3.059232 | 1.545492 |
| train | Độ dày proxy EDT (px) | 236080 | 2.044844 | 2.035803 | 0.051649 |
| test | Số pixel mực (px) | 92291 | 1687.930427 | 1544.000000 | 819.590615 |
| test | Mật độ mực (%) | 92291 | 3.364020 | 3.077168 | 1.633432 |
| test | Độ dày proxy EDT (px) | 92291 | 2.047157 | 2.036695 | 0.059989 |

Ảnh trắng: train=0; test=0


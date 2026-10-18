# ScanQA Object Selector — Baseline Experiment Report

Ngày ghi nhận: **2026-10-18 (Asia/Saigon)**

## 1. Mục tiêu

Huấn luyện một object selector trên ScanQA để xếp hạng các ScanNet object liên
quan đến câu hỏi. `object_id` chỉ được giữ làm metadata để xuất kết quả, không
được đưa vào model. Model dự kiến có thể dùng làm selector trước QA và làm
baseline chuyển giao sang SQA3D.

## 2. Dữ liệu

| Split | Câu hỏi | Scenes | GT object/câu trung bình | Min | Max |
|---|---:|---:|---:|---:|---:|
| ScanQA train | 25.563 | 562 | 1,3793 | 1 | 21 |
| ScanQA val | 4.675 | 71 | 2,1345 | 1 | 25 |

Model sử dụng GT ScanNet instances làm tập object ứng viên. Mỗi scene được tạo
từ ba file:

```text
<scene>_vh_clean_2.ply
<scene>_vh_clean_2.0.010000.segs.json
<scene>.aggregation.json
```

Feature đầu vào gồm:

- frozen MiniLM embedding của câu hỏi và nhãn object;
- 31 geometry/shape/density features;
- 79 RGB/HSV appearance features;
- 17 scene-context features;
- 128 local XYZ+RGB points cho mỗi object;
- center/size theo mét cho relative-geometry attention.

## 3. Thiết lập model và training

| Thuộc tính | Giá trị |
|---|---:|
| GPU | NVIDIA RTX 3090 |
| Trainable parameters | 3.247.454 |
| Hidden dimension | 256 |
| Transformer layers | 2 |
| Attention heads | 4 |
| Dropout | 0,1 |
| Maximum predicted count | 25 |
| Batch size | 64 |
| Learning rate | 0,0003 |
| Weight decay | 0,01 |
| Epoch tối đa | 20 |
| Early-stopping patience | 5 |
| AMP | Bật |
| Seed | 42 |

Loss:

```text
balanced object BCE
+ 0,25 × count cross-entropy
+ 0,10 × hardest-pair ranking loss (margin 0,5)
```

Checkpoint được chọn theo `val_dynamic_f1`.

## 4. Run và checkpoint

```text
Run:             20261018_153724
Best epoch:      10
Last epoch:      15
Early stopping:  Có, sau 5 epoch không cải thiện Dynamic F1
Best Dynamic F1: 0,416056
```

Thời gian thực đo:

| Thông tin | Giá trị |
|---|---:|
| Epoch 10 | 12,55 giây |
| Trung bình/epoch | 12,62 giây |
| Tổng 15 epoch | 189,35 giây (~3 phút 9 giây) |

Checkpoint:

```text
../20261018_153724/best.pt
SHA-256: DCA3AA1FC2300510FA11F8DE70DBAFECBD31572398D9E5592A63357530C45036
```

## 5. Định nghĩa GT oracle

GT oracle không phải một model. Đây là trần lý thuyết theo số GT object của mỗi
câu hỏi: toàn bộ GT object được xếp trước object không liên quan. Precision
oracle thấp dần khi K tăng vì mỗi câu val chỉ có trung bình 2,13 GT object nhưng
vẫn phải chọn K object.

## 6. So sánh GT oracle, CAPrunner và model mới

### Top-10

| Metric | GT oracle | CAPrunner | Model mới | Mới − CAPrunner |
|---|---:|---:|---:|---:|
| Precision@10 | 21,18% | 15,37% | **18,13%** | **+2,76** |
| Recall@10 | 99,90% | 73,68% | **87,13%** | **+13,45** |
| F1@10 | 33,20% | 24,04% | **28,46%** | **+4,42** |
| Hit@10 | 100,00% | 87,02% | **95,49%** | **+8,47** |
| Recall-All@10 | 99,51% | 59,02% | **76,96%** | **+17,95** |

### Recall@K

| K | GT oracle | CAPrunner | Model mới | Mới − CAPrunner |
|---:|---:|---:|---:|---:|
| 1 | 61,81% | 27,74% | **35,05%** | +7,31 |
| 3 | 96,31% | 49,75% | **62,58%** | +12,84 |
| 5 | 98,97% | 58,69% | **74,08%** | +15,39 |
| 7 | 99,57% | 65,01% | **80,87%** | +15,86 |
| 10 | 99,90% | 73,68% | **87,13%** | +13,45 |

### Recall-All@K

| K | GT oracle | CAPrunner | Model mới | Mới − CAPrunner |
|---:|---:|---:|---:|---:|
| 1 | 33,82% | 14,72% | **19,70%** | +4,98 |
| 3 | 90,35% | 33,35% | **45,93%** | +12,58 |
| 5 | 97,07% | 41,99% | **59,23%** | +17,24 |
| 7 | 98,55% | 48,60% | **67,94%** | +19,34 |
| 10 | 99,51% | 59,02% | **76,96%** | +17,95 |

### Precision, F1 và Hit của model mới

| K | Precision | Recall | F1 | Hit | Recall-All |
|---:|---:|---:|---:|---:|---:|
| 1 | 56,53% | 35,05% | 40,97% | 56,53% | 19,70% |
| 3 | 38,25% | 62,58% | **44,63%** | 80,62% | 45,93% |
| 5 | 28,67% | 74,08% | 38,95% | 88,47% | 59,23% |
| 7 | 23,15% | 80,87% | 34,01% | 92,39% | 67,94% |
| 10 | 18,13% | 87,13% | 28,46% | 95,49% | 76,96% |

Top-3 có macro F1 cao nhất. Top-5 là cấu hình cân bằng để đưa object vào QA.
Top-10 phù hợp khi ưu tiên không bỏ sót object.

## 7. Dynamic-count result

| Metric | Kết quả |
|---|---:|
| Dynamic Precision | 56,47% |
| Dynamic Recall | 35,87% |
| Dynamic F1 | 41,61% |
| Dynamic Hit | 57,07% |
| Dynamic Recall-All | 20,34% |
| Predicted count trung bình | 1,0558 |
| GT count trung bình | 2,1345 |

Count head đang dự đoán thiếu object. Vì vậy chưa nên dùng dynamic count trong
pipeline chính; dùng Top-5 hoặc Top-7 ổn định hơn.

## 8. Loss tại best epoch

| Loss | Train | Val |
|---|---:|---:|
| Total | 0,4838 | 1,0135 |
| Object | 0,2626 | 0,3841 |
| Count | 0,6252 | 1,9589 |
| Ranking | 0,6486 | 1,3962 |

Khoảng cách lớn nhất nằm ở count loss và ranking loss. Đây là dấu hiệu của
count-distribution shift và generalization gap, không phải lỗi runtime.

## 9. Kết luận hiện tại

- Model mới vượt CAPrunner ở toàn bộ metric Top-K chung.
- Mức tăng mạnh nhất tại Top-10 là Recall-All: **+17,95 điểm phần trăm**.
- CAPrunner cũ dùng GT object IDs trong bước chọn pose, tạo oracle-pose leakage;
  model mới không đưa ID vào input mà vẫn tốt hơn.
- Cấu hình sử dụng đề xuất: **Top-5** cho cân bằng, **Top-7/10** khi ưu tiên recall.
- CAPrunner có mAP@10 = 48,70%; model mới chưa tính mAP nên không so sánh metric
  này. mAP là metric bổ sung, không phải metric chính cho mục tiêu lấy đủ object.

## 10. Giới hạn và kế hoạch experiment tiếp theo

1. Chia ScanQA train theo scene thành train/dev; giữ ScanQA val để đánh giá cuối.
2. Oversample câu hỏi nhiều object với `weight = min(gt_count, 5)`.
3. Thay 25-way count classification bằng ordinal count prediction.
4. Phạt underprediction mạnh hơn overprediction.
5. Chọn checkpoint bằng `0,5 × F1@3 + 0,5 × Recall-All@5`.
6. Chạy ba seed 42, 43 và 44; báo cáo mean ± standard deviation.
7. Với SQA3D, bổ sung situation text, camera position và rotation cho các quan hệ
   left/right/front/behind phụ thuộc góc nhìn.

Mục tiêu experiment tiếp theo:

| Metric | Baseline hiện tại | Mục tiêu |
|---|---:|---:|
| Recall@5 | 74,08% | ≥75% |
| Recall-All@5 | 59,23% | ≥62% |
| Recall@10 | 87,13% | ≥87% |
| Dynamic F1 | 41,61% | ≥43% |
| Predicted count | 1,06 | 1,7–2,2 |

## 11. Nguồn kết quả

```text
Checkpoint/config/history:
  ../20261018_153724/

CAPrunner metrics:
  ../outputs/capruner_scanqa_val_metrics/summary.json
  ../outputs/capruner_scanqa_val_metrics/metrics_at_k.csv

ScanQA annotations:
  ../data/scanqa/ScanQA_v1.0_train.json
  ../data/scanqa/ScanQA_v1.0_val.json
```

### Ghi chú phương pháp

- Kết quả model mới lấy từ `val_metrics` lưu trong `best.pt` epoch 10.
- ScanQA val đã được dùng cho early stopping, nên trong báo cáo khoa học cần ghi
  rõ đây là development-style evaluation hoặc chuyển sang train/dev split.
- Metric được macro-average theo câu hỏi, trừ khi tên metric ghi rõ micro.
- Kết quả sử dụng GT ScanNet instances; chưa phải đánh giá end-to-end với object
  detector proposals.

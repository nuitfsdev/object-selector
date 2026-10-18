# Clean train/dev calibration experiment

Quy trình này không dùng ScanQA val để chọn epoch, threshold hoặc giới hạn số
object. Official val chỉ được chạy một lần sau khi policy đã được khóa.

## Dữ liệu

`split_train_dev.py` chia 562 ScanQA train scenes thành:

```text
506 train-fit scenes: 22.978 câu hỏi
 56 dev scenes:        2.585 câu hỏi
```

Hai tập scene không giao nhau. Split seed 42 được tìm hoàn toàn từ ScanQA train
để giữ mean GT count và tỷ lệ câu multi-object gần nhau:

| Split | Mean GT count | Multi-object rate |
|---|---:|---:|
| Train-fit | 1,3796 | 22,77% |
| Dev | 1,3764 | 22,79% |

Các file split được ghi vào:

```text
../data/scanqa/splits/seed42/
```

## Policy được khóa trước official val

1. Train model trên train-fit.
2. Early stopping trên dev bằng:

```text
0,5 × F1@3 + 0,5 × Recall-All@5
```

3. Trên dev, quét threshold từ 0,05 đến 0,95 với bước 0,01.
4. Chọn threshold có macro F1 cao nhất trong các threshold đạt:

```text
Recall-All >= 75%
Average selected objects <= 10
```

5. Nếu không có threshold thỏa cả hai điều kiện, script ghi
   `constraints_met=false` và chọn F1 cao nhất để chẩn đoán; không nên xem đó là
   policy chính thức nếu chưa xem lại constraint.
6. Policy inference:

```text
score >= selected threshold
empty set -> fallback Top-10
more than 10 -> retain highest-scoring 10
```

7. Khóa policy rồi đánh giá official ScanQA val đúng một lần.

## Chạy trên RTX 3090 VPS

Nếu `.venv` và full caches vẫn còn:

```bash
cd /workspace/scanqa_object_selector
tmux new -s selector-clean
bash run_clean_experiment.sh
```

Nếu cache đã mất:

```bash
bash vps_setup.sh
bash prepare_vps_data.sh --accept-scannet-tos
bash run_clean_experiment.sh
```

Detach tmux bằng `Ctrl+B`, sau đó `D`. Quay lại bằng:

```bash
tmux attach -t selector-clean
```

## Outputs

```text
runs_clean/<timestamp>/
  best.pt
  last.pt
  history.json
  config.json
  threshold_calibration/
    selected_policy.json
    threshold_curve.csv
  final_val/
    summary.json
    predictions.jsonl
```

`selected_policy.json` lưu threshold và metric dev. `final_val/summary.json` là
kết quả cuối trên official val. Không chạy lại calibration sau khi xem
`final_val`.

## Baseline threshold 0,5 hiện có

Báo cáo exploratory của checkpoint cũ nằm tại:

```text
../outputs/selector_threshold_eval/threshold_0.5_fallback_top10/REPORT.md
```

Kết quả này dùng threshold 0,5 được định trước nhưng checkpoint cũ đã dùng val
cho early stopping, nên chỉ dùng làm baseline phân tích.

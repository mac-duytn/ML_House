# Dự đoán giá nhà King County (House Price Prediction)

Dự án Machine Learning dạng **hồi quy** dự đoán giá bán nhà ở King County (Seattle, Mỹ) từ các đặc điểm như diện tích, chất lượng xây dựng và vị trí. Dữ liệu gồm 21.613 giao dịch (05/2014 – 05/2015) từ [Kaggle – House Sales in King County](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction).

Báo cáo đầy đủ (giới thiệu, mô tả dữ liệu, EDA, tiền xử lý, mô hình, kết quả, hạn chế) nằm trong hai notebook:
- [notebooks/1.0-EDA.ipynb](notebooks/1.0-EDA.ipynb) – mục 1–3: giới thiệu bài toán, mô tả dữ liệu, EDA.
- [notebooks/2.0-model-prototyping.ipynb](notebooks/2.0-model-prototyping.ipynb) – mục 4–6: tiền xử lý, huấn luyện, phân tích kết quả.

## Kết quả (tập test, 3.241 căn nhà)

| Mô hình | MAE (USD) | RMSE (USD) | R² | MAPE (%) |
|---|---:|---:|---:|---:|
| Linear Regression (baseline) | 76,383 | 137,822 | 0.878 | 13.90% |
| Ridge | 76,378 | 137,719 | 0.879 | 13.90% |
| Lasso | 76,593 | 135,907 | 0.882 | 13.97% |
| Decision Tree | 92,803 | 181,386 | 0.789 | 16.16% |
| Random Forest | 72,116 | 139,435 | 0.876 | 12.74% |
| **XGBoost (Tuned)** | **64,478** | **130,863** | **0.890** | **11.55%** |
| KNN | 91,951 | 185,837 | 0.779 | 15.47% |
| MLP | 84,835 | 164,763 | 0.826 | 15.34% |
| *OLS cũ (tham chiếu)* | *86,606* | *—* | *0.656* | *—* |

Mô hình tốt nhất (chọn theo RMSE trên tập validation): `XGBoost` (Val RMSE: 109,247 USD; Test R²: 0.8903; Test MAPE: 11.55%). Biến quan trọng nhất: `lat` > `sqft_living` > `grade` > `long`.

## Cài đặt

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Hướng dẫn sử dụng

```bash
# Tiền xử lý, chia 70/15/15, huấn luyện + tinh chỉnh các mô hình, lưu mô hình & biểu đồ
python src/train.py

# Dự đoán từ mô hình đã lưu (CSV cùng cột với dữ liệu thô)
python src/predict.py data/unprocessed/kc_house_data.csv predictions.csv

# Mở notebook
jupyter notebook notebooks/
```

`train.py` ghi ra: `models/best_model.joblib`, `data/processed/{train,val,test}.csv`, `reports/results.csv`, `reports/feature_importance.csv`, `reports/figures/*.png`.

## Cấu trúc dự án

```
ML_house-price-prediction/
├── README.md
├── requirements.txt
├── data/
│   ├── unprocessed/          # Dữ liệu thô (kc_house_data.csv)
│   └── processed/            # train/val/test sau tiền xử lý
├── notebooks/
│   ├── 1.0-EDA.ipynb
│   └── 2.0-model-prototyping.ipynb
├── src/
│   ├── data_processing.py    # Làm sạch, tạo biến, chia dữ liệu, preprocessor
│   ├── train.py              # Huấn luyện, GridSearchCV, đánh giá, lưu mô hình
│   └── predict.py            # Dự đoán từ mô hình đã lưu
├── models/                   # best_model.joblib
└── reports/
    ├── results.csv, feature_importance.csv, report_numbers.json
    └── figures/              # Biểu đồ EDA và kết quả
```

## Giấy phép

Chưa chỉ định giấy phép cho mã nguồn. Dữ liệu thuộc bộ dữ liệu công khai trên Kaggle, tuân theo điều khoản của nguồn gốc.

## Tác giả / Liên hệ

Điền thông tin tác giả tại đây.

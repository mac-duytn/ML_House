"""Dự đoán giá nhà từ mô hình đã lưu.

Chạy:  python src/predict.py path/to/houses.csv [output.csv]
File đầu vào có cùng các cột như dữ liệu thô (kc_house_data.csv), có hoặc không có cột price.
"""
import sys

import joblib
import numpy as np
import pandas as pd

from data_processing import FEATURE_COLS, MODELS_DIR, add_features, clean


def predict(df: pd.DataFrame, model_path=MODELS_DIR / "best_model.joblib") -> pd.Series:
    model = joblib.load(model_path)
    X = add_features(clean(df))[FEATURE_COLS]
    return pd.Series(np.expm1(model.predict(X)), name="predicted_price")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    preds = predict(pd.read_csv(sys.argv[1]))
    if len(sys.argv) > 2:
        preds.to_csv(sys.argv[2], index=False)
    else:
        print(preds.round(0).to_string())

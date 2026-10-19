"""Các hàm tiền xử lý dữ liệu cho bài toán dự đoán giá nhà King County."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "unprocessed" / "kc_house_data.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"

TARGET = "price"
RANDOM_STATE = 42
N_CLUSTERS = 8  # chọn bằng Elbow + Silhouette, xem train.py

# Biến phân loại không có thứ tự -> One-Hot Encoding
NOMINAL_COLS = ["zipcode"]
# Biến nhị phân -> giữ nguyên
BINARY_COLS = ["waterfront", "was_renovated", "has_basement"]
# Biến số (gồm cả biến thứ tự view/condition/grade đã mã hóa số sẵn) -> StandardScaler
NUMERIC_COLS = [
    "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "view",
    "condition", "grade", "sqft_above", "sqft_basement", "lat", "long",
    "sqft_living15", "sqft_lot15", "house_age", "sale_year", "sale_month",
]
GEO_COLS = ["lat", "long"]
FEATURE_COLS = NUMERIC_COLS + BINARY_COLS + NOMINAL_COLS

# Ngoại lai: chỉ tính ngưỡng trên train, áp dụng cho biến diện tích nhà và số phòng ngủ (không dùng price)
OUTLIER_IQR_K = 5
OUTLIER_COLS = ["sqft_living", "sqft_above", "sqft_basement"]
MAX_BEDROOMS = 10


class KMeansLabel(BaseEstimator, TransformerMixin):
    """Gán nhãn cụm K-Means cho cặp (lat, long). Chỉ fit trên train."""

    def __init__(self, n_clusters=N_CLUSTERS, random_state=RANDOM_STATE):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def fit(self, X, y=None):
        self.km_ = KMeans(self.n_clusters, n_init=10, random_state=self.random_state).fit(X)
        return self

    def transform(self, X):
        return self.km_.predict(X).reshape(-1, 1)


class ClipToTrain(BaseEstimator, TransformerMixin):
    """Cắt (winsorize) biến số về khoảng [min, max] của train để mô hình không ngoại suy
    vô hạn khi gặp giá trị chưa từng thấy (vd. lỗi nhập liệu 33 phòng ngủ ở val/test)."""

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_, self.hi_ = X.min(axis=0), X.max(axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)


def load_raw(path=RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Điền giá trị thiếu bằng quan hệ xác định (không dùng thống kê của tập dữ liệu)."""
    df = df.copy()
    missing = df["sqft_above"].isna()
    df.loc[missing, "sqft_above"] = df.loc[missing, "sqft_living"] - df.loc[missing, "sqft_basement"]
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tách ngày bán, tạo tuổi nhà và các cờ nhị phân."""
    df = df.copy()
    sale_date = pd.to_datetime(df["date"], format="%Y%m%dT%H%M%S")
    df["sale_year"] = sale_date.dt.year
    df["sale_month"] = sale_date.dt.month
    df["house_age"] = df["sale_year"] - df["yr_built"]
    df["was_renovated"] = (df["yr_renovated"] > 0).astype(int)
    df["has_basement"] = (df["sqft_basement"] > 0).astype(int)
    return df


def split_data(df: pd.DataFrame, val_size=0.15, test_size=0.15, seed=RANDOM_STATE):
    """Chia train/val/test = 70/15/15. Nhóm theo id để một căn nhà bán nhiều lần
    không nằm đồng thời ở train và test (tránh rò rỉ dữ liệu)."""
    outer = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_idx, test_idx = next(outer.split(df, groups=df["id"]))
    trainval, test = df.iloc[trainval_idx], df.iloc[test_idx]
    inner = GroupShuffleSplit(n_splits=1, test_size=val_size / (1 - test_size), random_state=seed)
    train_idx, val_idx = next(inner.split(trainval, groups=trainval["id"]))
    return trainval.iloc[train_idx], trainval.iloc[val_idx], test


def remove_train_outliers(train: pd.DataFrame):
    """Loại ngoại lai CHỈ trên train. Ngưỡng = Q3 + 5*IQR (tính trên train) của các biến
    diện tích nhà, và bedrooms >= 10. Tuyệt đối không dựa vào price."""
    mask = train["bedrooms"] < MAX_BEDROOMS
    thresholds = {}
    for c in OUTLIER_COLS:
        q1, q3 = train[c].quantile([0.25, 0.75])
        thresholds[c] = float(q3 + OUTLIER_IQR_K * (q3 - q1))
        mask &= train[c] <= thresholds[c]
    return train[mask], thresholds, int((~mask).sum())


def build_preprocessor(n_clusters=N_CLUSTERS) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", Pipeline([("clip", ClipToTrain()), ("scale", StandardScaler())]), NUMERIC_COLS),
            ("bin", "passthrough", BINARY_COLS),
            ("zip", OneHotEncoder(handle_unknown="ignore", sparse_output=False), NOMINAL_COLS),
            (
                "geo",
                Pipeline([
                    ("scale", StandardScaler()),
                    ("km", KMeansLabel(n_clusters)),
                    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]),
                GEO_COLS,
            ),
        ]
    )


def prepare_datasets():
    """Đọc dữ liệu thô -> làm sạch -> tạo biến -> chia tập -> loại outlier (chỉ train).
    Trả về train (đã loại outlier), val, test và thông tin outlier. Lưu CSV vào data/processed/."""
    df = add_features(clean(load_raw()))
    train_full, val, test = split_data(df)
    train, thresholds, n_removed = remove_train_outliers(train_full)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", train), ("val", val), ("test", test)):
        part.to_csv(PROCESSED_DIR / f"{name}.csv", index=False)
    info = {"n_before": len(train_full), "n_removed": n_removed, "thresholds": thresholds}
    return train, val, test, info


def xy(df: pd.DataFrame):
    """Tách X và y. Mô hình học trên log(price) vì giá lệch phải mạnh."""
    return df[FEATURE_COLS], np.log1p(df[TARGET])

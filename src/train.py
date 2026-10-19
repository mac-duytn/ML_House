"""Huấn luyện, tinh chỉnh và so sánh các mô hình hồi quy giá nhà.

Chạy:  python src/train.py
Kết quả: models/best_model.joblib, reports/results.csv, reports/report_numbers.json, reports/figures/*.png
"""
import json
import time

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, silhouette_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

import data_processing as dp
from data_processing import FIGURES_DIR, MODELS_DIR, RANDOM_STATE, ROOT, build_preprocessor, xy

# Baseline OLS của bài cũ (số liệu do người dùng cung cấp)
OLD_OLS = {"R2": 0.6565, "MAE": 86606}

# (tên, mô hình, lưới siêu tham số) - tiền tố "model__" trỏ tới bước model trong Pipeline
CANDIDATES = [
    ("Linear Regression", LinearRegression(), {}),
    ("Ridge", Ridge(), {"model__alpha": [0.1, 1, 10, 100]}),
    ("Lasso", Lasso(max_iter=20000), {"model__alpha": [1e-4, 1e-3, 1e-2]}),
    (
        "Decision Tree",
        DecisionTreeRegressor(random_state=RANDOM_STATE),
        {"model__max_depth": [8, 12, 16], "model__min_samples_leaf": [5, 10, 20]},
    ),
    (
        "Random Forest",
        RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        {"model__n_estimators": [200], "model__max_depth": [None, 20], "model__min_samples_leaf": [1, 3]},
    ),
    (
        "XGBoost",
        XGBRegressor(random_state=RANDOM_STATE, tree_method="hist", n_jobs=-1),
        {
            "model__n_estimators": [400, 800],
            "model__learning_rate": [0.03, 0.06],
            "model__max_depth": [4, 6],
        },
    ),
    (
        "KNN",
        KNeighborsRegressor(n_jobs=-1),
        {"model__n_neighbors": [5, 10, 15], "model__weights": ["uniform", "distance"]},
    ),
    (
        "MLP",
        MLPRegressor(early_stopping=True, max_iter=300, random_state=RANDOM_STATE),
        {"model__hidden_layer_sizes": [(64, 32), (128, 64)], "model__alpha": [1e-3, 1e-2]},
    ),
]


def evaluate(model, X, y_log):
    """Tính metric trên thang đo gốc (USD), không phải log."""
    y_true, y_pred = np.expm1(y_log), np.expm1(model.predict(X))
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": r2_score(y_true, y_pred),
        "MAPE": float(np.mean(np.abs(y_true - y_pred) / y_true) * 100),
    }


def eda_numbers(raw):
    """Các con số thật dùng trong phần EDA của báo cáo."""
    df = dp.add_features(dp.clean(raw))
    corr = df.drop(columns=["id", "date"]).corr()["price"].drop("price").sort_values(ascending=False)
    q1, q3 = df["price"].quantile([0.25, 0.75])
    return {
        "n_rows": int(raw.shape[0]), "n_cols": int(raw.shape[1]),
        "n_dup_id": int(raw["id"].duplicated().sum()),
        "missing": {k: int(v) for k, v in raw.isna().sum().items() if v > 0},
        "price_skew": float(df["price"].skew()), "logprice_skew": float(np.log1p(df["price"]).skew()),
        "price_mean": float(df["price"].mean()), "price_median": float(df["price"].median()),
        "price_min": float(df["price"].min()), "price_max": float(df["price"].max()),
        "price_iqr_outlier_pct": float((df["price"] > q3 + 1.5 * (q3 - q1)).mean() * 100),
        "corr": corr.round(3).to_dict(),
        "waterfront_pct": float(df["waterfront"].mean() * 100),
        "median_by_waterfront": df.groupby("waterfront")["price"].median().to_dict(),
        "median_by_grade": df.groupby("grade")["price"].median().to_dict(),
        "median_by_view": df.groupby("view")["price"].median().to_dict(),
        "zip_median_min": [int(df.groupby("zipcode")["price"].median().idxmin()), float(df.groupby("zipcode")["price"].median().min())],
        "zip_median_max": [int(df.groupby("zipcode")["price"].median().idxmax()), float(df.groupby("zipcode")["price"].median().max())],
        "n_zip": int(df["zipcode"].nunique()),
        "date_min": str(raw["date"].min())[:8], "date_max": str(raw["date"].max())[:8],
        "bedrooms_max_row": raw.loc[raw["bedrooms"].idxmax(), ["bedrooms", "sqft_living"]].astype(int).to_dict(),
        "sqft_lot_skew": float(df["sqft_lot"].skew()), "sqft_living_skew": float(df["sqft_living"].skew()),
    }


def choose_k(train, ks=range(2, 13)):
    """Elbow (inertia) và Silhouette cho K-Means trên (lat, long) đã chuẩn hóa, chỉ dùng train."""
    Z = StandardScaler().fit_transform(train[dp.GEO_COLS])
    rng = np.random.RandomState(RANDOM_STATE)
    sample = rng.choice(len(Z), size=5000, replace=False)
    inertia, sil = {}, {}
    for k in ks:
        km = KMeans(k, n_init=10, random_state=RANDOM_STATE).fit(Z)
        inertia[k] = float(km.inertia_)
        sil[k] = float(silhouette_score(Z[sample], km.labels_[sample]))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(list(inertia), list(inertia.values()), "o-"); axes[0].set(xlabel="k", ylabel="Inertia", title="Elbow")
    axes[1].plot(list(sil), list(sil.values()), "o-"); axes[1].set(xlabel="k", ylabel="Silhouette", title="Silhouette")
    plt.tight_layout(); fig.savefig(FIGURES_DIR / "kmeans_elbow_silhouette.png", dpi=130); plt.close(fig)
    # Chọn k có silhouette cao nhất trong khoảng 4-12 (k quá nhỏ chỉ tách "Seattle / ngoại ô", ít thông tin)
    best_k = max((k for k in ks if k >= 4), key=lambda k: sil[k])
    return best_k, inertia, sil


def baseline_diagnostics(train):
    """Kiểm định thống kê rút gọn cho OLS baseline (log-giá, biến số gốc): F, VIF, JB, Breusch-Pagan."""
    X = train[dp.NUMERIC_COLS + dp.BINARY_COLS].astype(float).values
    y = np.log1p(train[dp.TARGET]).values
    n, p = X.shape
    Xc = np.column_stack([np.ones(n), X])
    beta = np.linalg.lstsq(Xc, y, rcond=None)[0]
    resid = y - Xc @ beta
    r2 = 1 - resid.var() / y.var()
    F = (r2 / p) / ((1 - r2) / (n - p - 1))
    jb, jb_p = stats.jarque_bera(resid)
    aux = np.linalg.lstsq(Xc, resid ** 2, rcond=None)[0]
    r2_aux = 1 - ((resid ** 2 - Xc @ aux).var() / (resid ** 2).var())
    bp = n * r2_aux
    cols = dp.NUMERIC_COLS + dp.BINARY_COLS
    vif = {}
    Xs = (X - X.mean(0)) / X.std(0)
    for j, c in enumerate(cols):
        others = np.delete(Xs, j, axis=1)
        b = np.linalg.lstsq(others, Xs[:, j], rcond=None)[0]
        r2j = 1 - ((Xs[:, j] - others @ b).var() / Xs[:, j].var())
        vif[c] = float(1 / max(1 - r2j, 1e-12))
    return {
        "R2_log": float(r2), "F": float(F), "F_p": float(stats.f.sf(F, p, n - p - 1)),
        "JB": float(jb), "JB_p": float(jb_p), "resid_skew": float(stats.skew(resid)),
        "resid_kurtosis": float(stats.kurtosis(resid) + 3),
        "BP": float(bp), "BP_p": float(stats.chi2.sf(bp, p)),
        "VIF_top": dict(sorted(vif.items(), key=lambda kv: -kv[1])[:5]),
    }


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    nums = {"old_ols": OLD_OLS}

    raw = dp.load_raw()
    nums["eda"] = eda_numbers(raw)

    train, val, test, info = dp.prepare_datasets()
    nums["split"] = {"train": len(train), "val": len(val), "test": len(test),
                     "total": len(train) + info["n_removed"] + len(val) + len(test), **info}
    X_tr, y_tr = xy(train)
    X_val, y_val = xy(val)
    X_te, y_te = xy(test)
    groups = train["id"].values
    print(nums["split"])

    k, inertia, sil = choose_k(train)
    nums["kmeans"] = {"k": k, "inertia": inertia, "silhouette": sil}
    print("K-Means k =", k)
    nums["baseline_diag"] = baseline_diagnostics(train)

    cv = GroupKFold(n_splits=5)
    rows, fitted = [], {}
    for name, model, grid in CANDIDATES:
        t0 = time.time()
        pipe = Pipeline([("prep", build_preprocessor(k)), ("model", model)])
        if grid:
            search = GridSearchCV(pipe, grid, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=1)
            search.fit(X_tr, y_tr, groups=groups)
            pipe, best_params = search.best_estimator_, search.best_params_
            cv_rmse_log = -search.best_score_
            cv_std = float(search.cv_results_["std_test_score"][search.best_index_])
        else:
            from sklearn.model_selection import cross_val_score
            scores = cross_val_score(pipe, X_tr, y_tr, cv=cv, groups=groups, scoring="neg_root_mean_squared_error")
            cv_rmse_log, cv_std = -scores.mean(), float(scores.std())
            best_params = {}
            pipe.fit(X_tr, y_tr)
        fitted[name] = pipe
        m = {s: evaluate(pipe, X, y) for s, (X, y) in
             {"Train": (X_tr, y_tr), "Val": (X_val, y_val), "Test": (X_te, y_te)}.items()}
        row = {"Model": name, "CV_RMSE_log": cv_rmse_log, "CV_std": cv_std}
        for s, d in m.items():
            row.update({f"{s}_{kk}": v for kk, v in d.items()})
        row["Best_params"] = json.dumps({a.replace("model__", ""): b for a, b in best_params.items()}, default=str)
        row["Seconds"] = time.time() - t0
        rows.append(row)
        print(f"{name:18s} CV(log) {cv_rmse_log:.4f} | train R2={m['Train']['R2']:.4f} val R2={m['Val']['R2']:.4f} "
              f"test R2={m['Test']['R2']:.4f} MAE={m['Test']['MAE']:,.0f}  {best_params}  [{row['Seconds']:.0f}s]", flush=True)

    results = pd.DataFrame(rows)
    results.to_csv(ROOT / "reports" / "results.csv", index=False)

    best_name = results.sort_values("Val_RMSE").iloc[0]["Model"]
    best = fitted[best_name]
    nums["best_model"] = best_name
    print("Best model (by validation RMSE):", best_name)
    joblib.dump(best, MODELS_DIR / "best_model.joblib")

    # --- Biểu đồ so sánh R² train/test ---
    fig, ax = plt.subplots(figsize=(9, 4))
    results.set_index("Model")[["Train_R2", "Test_R2"]].plot.bar(ax=ax)
    ax.set_ylim(0.6, 1.02); ax.set_ylabel("R²"); ax.set_title("R² train và test theo mô hình")
    plt.xticks(rotation=25); plt.tight_layout()
    fig.savefig(FIGURES_DIR / "model_comparison.png", dpi=130); plt.close(fig)

    # --- Thực tế vs dự đoán, và sai số ở nhà đắt ---
    y_true, y_pred = np.expm1(y_te).values, np.expm1(best.predict(X_te))
    ape = np.abs(y_pred - y_true) / y_true * 100
    signed = (y_pred - y_true) / y_true * 100
    thr = float(np.quantile(y_true, 0.9))
    hi = y_true >= thr
    nums["error_analysis"] = {
        "overall_mape": float(ape.mean()), "within10": float((ape < 10).mean() * 100),
        "within20": float((ape < 20).mean() * 100),
        "top10_threshold": thr, "top10_n": int(hi.sum()),
        "top10_mape": float(ape[hi].mean()), "top10_mae": float(np.abs(y_pred - y_true)[hi].mean()),
        "top10_bias_pct": float(signed[hi].mean()), "top10_under_share": float((signed[hi] < 0).mean() * 100),
        "rest_mape": float(ape[~hi].mean()), "rest_mae": float(np.abs(y_pred - y_true)[~hi].mean()),
        "rest_bias_pct": float(signed[~hi].mean()),
        "max_price_test": float(y_true.max()),
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(y_true[~hi], y_pred[~hi], s=6, alpha=0.35, label="90% nhà còn lại")
    axes[0].scatter(y_true[hi], y_pred[hi], s=8, alpha=0.6, color="tab:orange", label="10% nhà đắt nhất")
    lim = [0, max(y_true.max(), y_pred.max())]
    axes[0].plot(lim, lim, "r--", label="dự đoán hoàn hảo")
    axes[0].set(xlabel="Giá thực tế (USD)", ylabel="Giá dự đoán (USD)", title=f"{best_name}: thực tế vs dự đoán (test)")
    axes[0].legend()
    axes[1].hist(signed[~hi], bins=60, range=(-60, 60), alpha=0.6, label="90% còn lại")
    axes[1].hist(signed[hi], bins=60, range=(-60, 60), alpha=0.7, color="tab:orange", label="10% đắt nhất")
    axes[1].set(xlabel="Sai số tương đối có dấu (%) = (dự đoán − thực tế)/thực tế", ylabel="Số căn nhà", title="Phân phối sai số tương đối")
    axes[1].legend()
    plt.tight_layout(); fig.savefig(FIGURES_DIR / "best_model_predictions.png", dpi=130); plt.close(fig)

    # --- Feature importance (permutation, trên biến gốc, tập test, đo bằng giảm R² của log-giá) ---
    perm = permutation_importance(best, X_te, y_te, n_repeats=5, random_state=RANDOM_STATE, n_jobs=1)
    imp = pd.Series(perm.importances_mean, index=X_te.columns).sort_values()
    imp.to_csv(ROOT / "reports" / "feature_importance.csv", header=["importance"])
    nums["importance_top"] = imp.sort_values(ascending=False).head(10).round(4).to_dict()
    fig, ax = plt.subplots(figsize=(7, 6))
    imp.tail(15).plot.barh(ax=ax)
    ax.set(xlabel="Giảm R² (log-giá) khi xáo trộn biến", title=f"Top 15 Feature Importance - {best_name}")
    plt.tight_layout(); fig.savefig(FIGURES_DIR / "feature_importance.png", dpi=130); plt.close(fig)

    nums["results"] = results.drop(columns=["Best_params"]).round(5).to_dict(orient="records")
    nums["best_params"] = {r["Model"]: json.loads(r["Best_params"]) for r in rows}
    (ROOT / "reports" / "report_numbers.json").write_text(json.dumps(nums, indent=2, ensure_ascii=False, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()

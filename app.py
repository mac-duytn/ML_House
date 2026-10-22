import os
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Thêm thư mục src vào sys.path để import data_processing
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))

from data_processing import FEATURE_COLS, MODELS_DIR, add_features, clean

# Thiết lập trang Streamlit
st.set_page_config(
    page_title="Dự Đoán Giá Nhà King County",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tải mô hình
@st.cache_resource
def load_model():
    model_path = ROOT / "models" / "best_model.joblib"
    if not model_path.exists():
        st.error(f"Không tìm thấy file mô hình tại {model_path}!")
        return None
    return joblib.load(model_path)

model = load_model()

# Header ứng dụng
st.title("🏡 Hệ Thống Dự Đoán Giá Nhà King County (Seattle, USA)")
st.markdown("""
Ứng dụng sử dụng mô hình học máy **XGBoost Regressor** (đã tinh chỉnh siêu tham số và kiểm định chéo) 
để định giá bất động sản tại quận King, tiểu bang Washington dựa trên diện tích, vị trí và chất lượng công trình.
""")

st.divider()

# Danh sách zipcode phổ biến
ZIPCODES = [
    98001, 98002, 98003, 98004, 98005, 98006, 98007, 98008, 98010, 98011,
    98014, 98019, 98022, 98023, 98024, 98027, 98028, 98029, 98030, 98031,
    98032, 98033, 98034, 98038, 98039, 98040, 98042, 98045, 98052, 98053,
    98055, 98056, 98058, 98059, 98065, 98070, 98072, 98074, 98075, 98077,
    98092, 98102, 98103, 98105, 98106, 98107, 98108, 98109, 98112, 98115,
    98116, 98117, 98118, 98119, 98122, 98125, 98126, 98133, 98136, 98144,
    98146, 98148, 98155, 98166, 98168, 98177, 98178, 98188, 98198, 98199
]

# Tọa độ trung bình tham khảo theo zipcode
DEFAULT_COORDS = {
    98004: (47.618, -122.203), # Bellevue
    98039: (47.625, -122.234), # Medina (Khu nhà giàu)
    98103: (47.673, -122.343), # Seattle - Green Lake
    98105: (47.660, -122.285), # Seattle - University District
    98052: (47.680, -122.122), # Redmond (Microsoft)
}

# Form nhập liệu 3 cột
col1, col2, col3 = st.columns([1, 1, 1], gap="medium")

with col1:
    st.subheader("📐 Quy Mô & Diện Tích")
    sqft_living = st.number_input("Diện tích sinh hoạt (sqft):", min_value=300, max_value=12000, value=2000, step=50, help="1 sqft ≈ 0,0929 m²")
    sqft_lot = st.number_input("Diện tích khuôn viên đất (sqft):", min_value=500, max_value=1000000, value=7500, step=250)
    has_bsmt = st.checkbox("Có tầng hầm?", value=False)
    if has_bsmt:
        sqft_basement = st.number_input("Diện tích tầng hầm (sqft):", min_value=50, max_value=4000, value=500, step=50)
    else:
        sqft_basement = 0
    sqft_above = sqft_living - sqft_basement

    st.caption(f"Diện tích tầng trên mặt đất: **{sqft_above:,} sqft**")

with col2:
    st.subheader("🛋 Cấu Trúc & Chất Lượng")
    bedrooms = st.slider("Số phòng ngủ:", min_value=1, max_value=10, value=3, step=1)
    bathrooms = st.slider("Số phòng tắm:", min_value=0.75, max_value=7.0, value=2.25, step=0.25)
    floors = st.selectbox("Số tầng:", options=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5], index=1)
    grade = st.slider("Cấp bậc thiết kế & thi công (Grade 1-13):", min_value=1, max_value=13, value=7, help="7-8: Bình thường/Khá; 10+: Biệt thự cao cấp")
    condition = st.select_slider("Tình trạng bảo trì (1-5):", options=[1, 2, 3, 4, 5], value=3, format_func=lambda x: f"{x} - {'Kém' if x<=2 else 'Trung bình' if x==3 else 'Rất tốt'}")

with col3:
    st.subheader("📍 Vị Trí & Tiện Ích")
    zipcode = st.selectbox("Mã bưu điện (Zipcode):", options=ZIPCODES, index=ZIPCODES.index(98103))
    
    # Gợi ý tọa độ theo zipcode nếu có
    default_lat, default_long = DEFAULT_COORDS.get(zipcode, (47.56, -122.21))
    lat = st.number_input("Vĩ độ (Latitude):", min_value=47.1, max_value=47.8, value=float(default_lat), format="%.4f")
    long_val = st.number_input("Kinh độ (Longitude):", min_value=-122.6, max_value=-121.3, value=float(default_long), format="%.4f")
    
    yr_built = st.number_input("Năm xây dựng:", min_value=1900, max_value=2015, value=1985, step=1)
    is_reno = st.checkbox("Đã từng cải tạo / trùng tu?", value=False)
    yr_renovated = st.number_input("Năm cải tạo:", min_value=yr_built, max_value=2015, value=2005) if is_reno else 0
    
    waterfront = 1 if st.checkbox("Giáp bờ nước (Waterfront)?", value=False) else 0
    view = st.slider("Chất lượng tầm nhìn (View 0-4):", min_value=0, max_value=4, value=0)

st.divider()

# Nút bấm dự đoán
predict_btn = st.button("🚀 Bấm Để Dự Đoán Giá Nhà", type="primary", use_container_width=True)

if predict_btn:
    if model is None:
        st.error("Chưa tải được mô hình!")
    else:
        # Chuẩn bị dữ liệu đầu vào giống định dạng gốc
        raw_dict = {
            "id": [1],
            "date": ["20150101T000000"],
            "bedrooms": [bedrooms],
            "bathrooms": [bathrooms],
            "sqft_living": [sqft_living],
            "sqft_lot": [sqft_lot],
            "floors": [floors],
            "waterfront": [waterfront],
            "view": [view],
            "condition": [condition],
            "grade": [grade],
            "sqft_above": [sqft_above],
            "sqft_basement": [sqft_basement],
            "yr_built": [yr_built],
            "yr_renovated": [yr_renovated],
            "zipcode": [zipcode],
            "lat": [lat],
            "long": [long_val],
            "sqft_living15": [sqft_living],
            "sqft_lot15": [sqft_lot],
        }
        input_df = pd.DataFrame(raw_dict)
        
        # Tiền xử lý dữ liệu qua pipeline
        processed_df = add_features(clean(input_df))[FEATURE_COLS]
        
        # Dự đoán giá (thang log) và đổi ngược lại USD
        pred_log = model.predict(processed_df)
        pred_price = np.expm1(pred_log)[0]
        
        # Hiển thị kết quả nổi bật
        st.success("### 🎉 KẾT QUẢ DỰ ĐOÁN")
        res_col1, res_col2, res_col3 = st.columns([1.5, 1, 1])
        
        with res_col1:
            st.metric(
                label="Giá bán ước tính:",
                value=f"${pred_price:,.0f} USD",
                delta=f"≈ {pred_price * 25400 / 1e9:,.2f} tỷ VNĐ"
            )
            st.caption("*(Tỷ giá quy đổi ước tính 1 USD ≈ 25.400 VNĐ)*")
            
        with res_col2:
            lower_bound = pred_price * (1 - 0.1155)
            upper_bound = pred_price * (1 + 0.1155)
            st.metric(
                label="Khoảng giá dự kiến (± 11.55% MAPE):",
                value=f"${lower_bound:,.0f} - ${upper_bound:,.0f}"
            )
            st.caption("Dựa trên biên độ sai số kiểm nghiệm trên tập Test của mô hình XGBoost.")

        with res_col3:
            unit_price = pred_price / sqft_living
            st.metric(
                label="Đơn giá trên diện tích:",
                value=f"${unit_price:,.0f} / sqft",
                delta=f"≈ {unit_price * 10.764:,.0f} USD/m²"
            )

        # Hiển thị bản đồ vị trí
        st.subheader("🗺 Vị Trí Bất Động Sản Trên Bản Đồ")
        map_df = pd.DataFrame({"lat": [lat], "lon": [long_val]})
        st.map(map_df, zoom=12)

st.sidebar.title("ℹ️ Về Dự Án")
st.sidebar.info("""
- **Đồ án:** Phân tích Dữ liệu & Học máy
- **Đề tài:** Dự đoán giá nhà King County (USA)
- **Mô hình tốt nhất:** XGBoost Regressor (R² = 0.8903, MAPE = 11.55%)
- **Tập dữ liệu:** 21.613 giao dịch nhà ở (05/2014 – 05/2015)
- **Tác giả:** Duytnd
""")
st.sidebar.markdown("---")
st.sidebar.markdown("[GitHub Repository](https://github.com/Duytnd/ML_House_PP)")


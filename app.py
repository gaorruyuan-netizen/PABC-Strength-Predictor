import streamlit as st
import joblib
import numpy as np
import pandas as pd

# =========================================================
# 页面配置
# =========================================================
st.set_page_config(
    page_title="PABC Compressive Strength Predictor",
    page_icon="🧱",
    layout="wide"
)

# ===== 页面背景改为白色 + 论文风格配色 =====
st.markdown(
    """
    <style>
    .stApp {
        background-color: #ffffff;
    }
    h1, h2, h3 {
        color: #003366;
        font-weight: 700;
    }
    .stButton>button {
        background-color: #003366;
        color: white;
        border-radius: 6px;
        height: 3em;
        font-size: 16px;
    }
    section[data-testid="stSidebar"] {
        background-color: #f5f5f5;
    }
    </style>
    """,
    unsafe_allow_html=True
)

MODEL_PATH = "best_model.pkl"
FEATURE_COLS = ['Cement', 'Sand', 'Water', 'SA', 'EP', 'BF', 'HRWR', 'DP', 'T']
EPS = 1e-8

# =========================================================
# 加载模型
# =========================================================
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# =========================================================
# 特征工程
# =========================================================
def transform_temperature(T):
    return T / 800.0


def add_comprehensive_ratio_features(X, feature_cols):
    X = np.array(X, dtype=float)

    Cement, Sand, Water, SA, EP, BF, HRWR, DP, T = X[0]

    Binder = Cement + SA + DP + HRWR
    TotalMass = Water + Cement + Sand + SA + EP + BF + HRWR + DP

    ratio_features = [
        Water / (Cement + EPS),
        SA / (Cement + EPS),
        DP / (Cement + EPS),
        HRWR / (Cement + EPS),
        EP / (Sand + EPS),
        Water / (Binder + EPS),
        Sand / (Binder + EPS),
        TotalMass,
        Water / (TotalMass + EPS),

        # BF相关特征（始终存在，保证24维）
        BF / (Binder + EPS),
        Sand / (Sand + BF + EPS),
        (Sand + BF) / (Binder + EPS)
    ]

    ratio_features_array = np.array(ratio_features).reshape(1, -1)
    X_extended = np.hstack([X, ratio_features_array])

    return X_extended


def add_temperature_nonlinear_features(X):
    T = X[:, 8]

    T2 = T ** 2
    logT = np.log1p(T)
    invT = 1.0 / (T + EPS)

    X_extended = np.hstack([X, T2.reshape(-1, 1), logT.reshape(-1, 1), invT.reshape(-1, 1)])
    return X_extended


def preprocess_input(cement, sand, water, sa, ep, bf, hrwr, dp, t):
    X = np.array([[cement, sand, water, sa, ep, bf, hrwr, dp, t]], dtype=float)

    # 温度归一化
    X[0, 8] = transform_temperature(X[0, 8])

    # 比例特征
    X = add_comprehensive_ratio_features(X, FEATURE_COLS)

    # 温度非线性特征
    X = add_temperature_nonlinear_features(X)

    return X


def predict_strength(cement, sand, water, sa, ep, bf, hrwr, dp, t):
    X = preprocess_input(cement, sand, water, sa, ep, bf, hrwr, dp, t)
    y_pred = model.predict(X)[0]
    return float(y_pred), X


# =========================================================
# 页面标题
# =========================================================
st.title("PABC Compressive Strength Prediction System")
st.markdown(
    """
This web app predicts the **compressive strength of PABC** using a trained **LightGBM model**.
Please input the material proportions and temperature below.
"""
)

# =========================================================
# 侧边栏
# =========================================================
with st.sidebar:
    st.header("Model Information")
    st.write("**Model:** LightGBM")
    st.write("**Target:** Compressive Strength")
    st.write("**Input variables:** 9 raw features")
    st.write("**Temperature transform:** T / 800")
    st.write("**Extra engineering:** ratio + nonlinear temperature features")

    st.header("Notes")
    st.write("- All mixture quantities use the same unit as your training data.")
    st.write("- Temperature unit: °C")
    st.write("- Prediction output unit: MPa")

# =========================================================
# 主界面输入
# =========================================================
col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("Input Parameters")

    c1, c2, c3 = st.columns(3)
    with c1:
        cement = st.number_input("Cement (g)", min_value=0.0, value=250.0)
        sa = st.number_input("SA (g)", min_value=0.0, value=20.0)
        hrwr = st.number_input("HRWR (g)", min_value=0.0, value=5.0)

    with c2:
        sand = st.number_input("Sand (g)", min_value=0.0, value=180.0)
        ep = st.number_input("EP (g)", min_value=0.0, value=15.0)
        dp = st.number_input("DP (g)", min_value=0.0, value=10.0)

    with c3:
        water = st.number_input("Water (g)", min_value=0.0, value=125.0)
        bf = st.number_input("BF (g)", min_value=0.0, value=0.0)
        t = st.number_input("Temperature (°C)", min_value=0.0, value=20.0)

    predict_button = st.button("Predict Compressive Strength", use_container_width=True)

with col2:
    st.subheader("Engineering Description")
    st.info(
        "The deployed app uses the trained LightGBM model together with the same "
        "feature-engineering strategy adopted during model training, ensuring "
        "consistency between offline model development and online prediction."
    )

# =========================================================
# 预测结果
# =========================================================
if predict_button:
    try:
        pred, X_processed = predict_strength(
            cement, sand, water, sa, ep, bf, hrwr, dp, t
        )

        st.success(f"Predicted Compressive Strength: {pred:.3f} MPa")

        with st.expander("Show processed feature vector"):
            df = pd.DataFrame(X_processed)
            st.dataframe(df, use_container_width=True)

        raw_input_df = pd.DataFrame([{
            "Cement": cement,
            "Sand": sand,
            "Water": water,
            "SA": sa,
            "EP": ep,
            "BF": bf,
            "HRWR": hrwr,
            "DP": dp,
            "T (°C)": t
        }])

        st.subheader("Input Summary")
        st.dataframe(raw_input_df, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")


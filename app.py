import streamlit as st
import joblib
import numpy as np
import pandas as pd

# =========================================================
# 基本配置
# =========================================================
st.set_page_config(
    page_title="PABC Compressive Strength Predictor",
    page_icon="🧱",
    layout="wide"
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
# 特征工程：与 CS.py 保持一致
# =========================================================
def transform_temperature(T: np.ndarray, method: str = "divide") -> np.ndarray:
    """
    对温度做物理变换
    训练代码中使用的是 T' = T / 800
    """
    T = np.asarray(T, dtype=float)
    if method == "divide":
        return T / 800.0
    elif method == "log":
        return np.log1p(T)
    else:
        raise ValueError(f"未知的温度变换方法: {method}")


def add_comprehensive_ratio_features(X: np.ndarray, feature_cols) -> np.ndarray:
    """
    添加比例特征，与训练代码保持一致
    """
    X = np.array(X, dtype=float)

    col_indices = {col: feature_cols.index(col) for col in feature_cols}

    Cement = X[:, col_indices['Cement']]
    Sand   = X[:, col_indices['Sand']]
    Water  = X[:, col_indices['Water']]
    SA     = X[:, col_indices['SA']]
    EP     = X[:, col_indices['EP']]
    BF     = X[:, col_indices['BF']]
    HRWR   = X[:, col_indices['HRWR']]
    DP     = X[:, col_indices['DP']]

    Binder = Cement + SA + DP + HRWR
    TotalMass = Water + Cement + Sand + SA + EP + BF + HRWR + DP

    ratio_features = []

    # 1) 基础比例
    ratio_features.append(Water / (Cement + EPS))   # W/C

    # 2) 胶凝体系比例
    ratio_features.append(SA / (Cement + EPS))      # SA/Cement
    ratio_features.append(DP / (Cement + EPS))      # DP/Cement
    ratio_features.append(HRWR / (Cement + EPS))    # HRWR/Cement
    ratio_features.append(EP / (Sand + EPS))        # EP/Sand

    # 3) Binder相关
    ratio_features.append(Water / (Binder + EPS))   # Water/Binder
    ratio_features.append(Sand / (Binder + EPS))    # Sand/Binder

    # 4) 总量相关
    ratio_features.append(TotalMass)                # TotalMass
    ratio_features.append(Water / (TotalMass + EPS))# WaterFraction

    # 5) BF相关（训练代码逻辑：只要有任一样本 BF > 0 就追加这三列）
    if np.any(BF > EPS):
        ratio_features.append(BF / (Binder + EPS))            # CoarseAgg/Binder
        ratio_features.append(Sand / (Sand + BF + EPS))       # Fine/TotalAgg
        ratio_features.append((Sand + BF) / (Binder + EPS))   # Agg/Binder

    ratio_features_array = np.column_stack(ratio_features)
    X_extended = np.hstack([X, ratio_features_array])

    return X_extended


def add_temperature_nonlinear_features(
    X: np.ndarray,
    temp_col_idx: int,
    model_name: str = "LGBM"
):
    """
    温度非线性特征扩展
    在你的训练代码里，LGBM 走的是“非RF分支”：
    添加 T'、T'^2、logT、invT
    其中 T' 已经在原始特征里，所以这里只是把 T'^2、logT、invT
    插入到 T' 后面
    """
    X = np.array(X, dtype=float)

    T_prime = X[:, temp_col_idx]
    T_prime_squared = T_prime ** 2
    logT = np.log1p(T_prime)
    invT = 1.0 / (T_prime + EPS)

    X_extended = np.hstack([
        X[:, :temp_col_idx + 1],               # 到 T' 为止（包含 T'）
        T_prime_squared.reshape(-1, 1),        # T'^2
        logT.reshape(-1, 1),                   # logT
        invT.reshape(-1, 1),                   # invT
        X[:, temp_col_idx + 1:]                # 后面的其他特征
    ])

    new_temp_col_idx = temp_col_idx
    return X_extended, new_temp_col_idx


def preprocess_input(
    cement, sand, water, sa, ep, bf, hrwr, dp, t
) -> np.ndarray:
    """
    完整预处理流程：
    原始输入 -> T/800 -> 比例特征 -> 温度非线性特征
    """
    X = np.array([[cement, sand, water, sa, ep, bf, hrwr, dp, t]], dtype=float)

    # 1) 温度物理变换：T -> T/800
    temp_col_idx = FEATURE_COLS.index('T')
    X[:, temp_col_idx] = transform_temperature(X[:, temp_col_idx], method='divide')

    # 2) 比例特征
    X = add_comprehensive_ratio_features(X, FEATURE_COLS)

    # 3) 温度非线性特征
    X, _ = add_temperature_nonlinear_features(
        X,
        temp_col_idx=temp_col_idx,
        model_name="LGBM"
    )

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
# 侧边栏说明
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
        cement = st.number_input("Cement (g)", min_value=0.0, value=250.0, step=1.0)
        sa = st.number_input("SA (g)", min_value=0.0, value=20.0, step=1.0)
        hrwr = st.number_input("HRWR (g)", min_value=0.0, value=5.0, step=0.1)

    with c2:
        sand = st.number_input("Sand (g)", min_value=0.0, value=180.0, step=1.0)
        ep = st.number_input("EP (g)", min_value=0.0, value=15.0, step=1.0)
        dp = st.number_input("DP (g)", min_value=0.0, value=10.0, step=1.0)

    with c3:
        water = st.number_input("Water (g)", min_value=0.0, value=125.0, step=1.0)
        bf = st.number_input("BF (g)", min_value=0.0, value=0.0, step=1.0)
        t = st.number_input("Temperature (°C)", min_value=0.0, value=20.0, step=1.0)

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

        st.success(f"Predicted Compressive Strength: **{pred:.3f} MPa**")

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
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image

# ==========================================
# 1. Page Configuration & Setup
# ==========================================
st.set_page_config(page_title="STF Risk Predictor", page_icon="🏥", layout="wide")

st.title("Multimodal Prediction System for high STF-based Risk")
st.markdown("Integration of non-linear radiomics (Random Forest) and classical clinical regression (Logistic Regression) for a precise, two-stage STF risk assessment.")
st.markdown("---")

# ==========================================
# 2. Tabs Layout (Split Prediction Tool and Flowchart)
# ==========================================
tab1, tab2 = st.tabs(["🚀 Prediction Tool", "📖 Methodological Framework"])

# ----------------- TAB 2: Full Screen Image -----------------
with tab2:
    st.subheader("Methodological Framework for Image Super-Resolution and STF Quantification")
    current_cwd = os.getcwd() # 通常是 GitHub 仓库根目录 /mount/src/stf-risk-predictor/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        path_a = os.path.join(script_dir, 'Figure.png')
        if os.path.exists(path_a):
            st.image(path_a, use_column_width=True)
        else:
     # 猜测二：图片在 GitHub 仓库的最外面一层根目录 (Figure.png)
        path_b = os.path.join(current_cwd, 'Figure.png')
        if os.path.exists(path_b):
            st.image(path_b, use_column_width=True)
        else:
     # 都不在，抛出具体的缺失信息
            st.error("⚠️ 找不到 Figure.png。我已经找了以下两个地方：")
            st.code(f"1. {path_a}\n2. {path_b}")
            st.warning("👉 请检查 GitHub 仓库：\n1. 图片是否已经成功 Push 上去了？\n2. 名字是否确切为大写的 'Figure.png'？")
    except Exception as e:
        st.warning("⚠️ Place 'Figure.png' in the same folder to view the flowchart here.")

# ----------------- TAB 1: Main Prediction UI -----------------
with tab1:
    # --- Load Pre-trained Models ---
    @st.cache(allow_output_mutation=True)
    def load_models():
        rf = joblib.load('STF-Deploy/rf_model.pkl')       # Stage 1: Radiomics Model
        lr = joblib.load('STF-Deploy/combined_model.pkl') # Stage 2: Clinical Combined Model
        return rf, lr

    try:
        rf_model, lr_model = load_models()
    except Exception as e:
        st.error(f"⚠️ Failed to load models. Please check if .pkl files exist. Error: {e}")
        st.stop()

# --- 3 Columns Layout for Perfect Visual Balance ---
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🖼️ Radiomics (Part 1)")
        st.markdown("*(Stage I Radiomics Features)*")
        st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True) 
        
        feat_1 = st.number_input("1. exp_ngtdm_Busyness", value=0.0)
        feat_2 = st.number_input("2. log_glcm_ClusterShade", value=0.0)
        feat_3 = st.number_input("3. log_glcm_DifferenceAverage", value=0.0)
        feat_4 = st.number_input("4. square_glcm_Imc1", value=0.0)

    with col2:
        st.subheader("🖼️ Radiomics (Part 2)")
        st.markdown("*(Stage I Radiomics Features)*")
        st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
        
        feat_5 = st.number_input("5. sqrt_firstorder_Skewness", value=0.0)
        feat_6 = st.number_input("6. sqrt_glrlm_ShortRunLowGrayLevel", value=0.0) 
        feat_7 = st.number_input("7. wavelet_LHL_firstorder_Skewness", value=0.0)
        feat_8 = st.number_input("8. wavelet_LLL_ngtdm_Strength", value=0.0)

    with col3:
        st.subheader("📋 Clinical Signatures")
        st.markdown("*(Stage II Comprehensive Assessment)*")
        st.markdown("<div style='height: 0px;'></div>", unsafe_allow_html=True) 
        
        sex_input = st.selectbox("Sex", options=["Male", "Female"])
        bmi_input = st.selectbox("Body Mass Index (BMI)", options=["≤ 25 kg/m²", "> 25 kg/m²"])
        nlr_input = st.selectbox("NLR", options=["≤ 3.0", "> 3.0"])
        age_input = st.selectbox("Age", options=["≤ 65 years", "> 65 years"])

    st.markdown("---")

    # --- Center the Button ---
    _, col_btn, _ = st.columns([1, 2, 1])
    
    with col_btn:
        run_prediction = st.button("🚀 Calculate STF-based Risk Probability")

    # --- Backend Processing & Prediction ---
    if run_prediction:
        with st.spinner("Processing multimodal data..."):
            
            # FIXED: Data Preprocessing (Matching the exact text strings)
            sex_val = 1 if sex_input == "Male" else 2
            bmi_val = 0 if "≤" in bmi_input else 1
            nlr_val = 0 if "≤" in nlr_input else 1
            age_val = 0 if "≤" in age_input else 1
            
            # Stage I Calculate Radscore (8 selected radiomics features)
            radiomics_data = np.array([[feat_1, feat_2, feat_3, feat_4, feat_5, feat_6, feat_7, feat_8]])
            rad_score_prob = rf_model.predict_proba(radiomics_data)[0][1]
            
            # Stage II Final STF Probability
            lr_input_df = pd.DataFrame({
                'Sex': [sex_val],
                'BMI': [bmi_val],
                'NLR': [nlr_val],
                'Age': [age_val],
                'Radscore': [rad_score_prob]
            })
            
            final_prob = lr_model.predict_proba(lr_input_df)[0][1] * 100
            
            # --- Dashboard Results Display ---
            st.markdown("---")
            st.subheader("📊 Predictive Assessment Results")
            
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                st.metric(label="Stage I: Radiomics Risk (Radscore)", value=f"{rad_score_prob * 100:.2f}%")
                st.caption("Probability derived strictly from non-linear high-dimensional imaging features.")
                
            with res_col2:
                st.metric(label="Stage II: Final STF Risk", value=f"{final_prob:.2f}%", delta="Combined Clinical Evaluation", delta_color="off")
            
            # Final Risk Stratification Alert
            if final_prob > 50:
                st.error(f"⚠️ **High Risk of STF detected.** (Probability: {final_prob:.2f}%)")
                st.progress(int(final_prob))
            else:
                st.success(f"✅ **Low Risk of STF.** (Probability: {final_prob:.2f}%)")
                st.progress(int(final_prob))

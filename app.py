import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# ============================================================
# ตั้งค่าหน้าเว็บ
# ============================================================
st.set_page_config(
    page_title="❤️ Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS สำหรับความสวยงาม
st.markdown("""
<style>
    /* พื้นหลังหลัก */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* หัวข้อหลัก */
    .main-title {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: white;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        padding: 1rem;
    }
    
    .sub-title {
        font-size: 1.3rem;
        text-align: center;
        color: #f0f0f0;
        margin-bottom: 2rem;
    }
    
    /* กล่องผลลัพธ์ */
    .result-box-safe {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    
    .result-box-danger {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    
    /* ปุ่มทำนาย */
    .stButton>button {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        font-size: 1.2rem;
        font-weight: bold;
        padding: 0.8rem 2rem;
        border-radius: 50px;
        border: none;
        width: 100%;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2c3e50 0%, #4a6278 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Metric cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 1rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# โหลดโมเดล
# ============================================================
@st.cache_resource
def load_model():
    try:
        model = joblib.load('heart_disease_model.pkl')
        features = joblib.load('feature_names.pkl')
        return model, features
    except Exception as e:
        st.error(f"❌ ไม่สามารถโหลดโมเดลได้: {e}")
        return None, None

model, feature_names = load_model()

# ============================================================
# Header
# ============================================================
st.markdown('<div class="main-title">❤️ Heart Disease Predictor</div>', 
            unsafe_allow_html=True)
st.markdown('<div class="sub-title">🔬 ระบบทำนายความเสี่ยงโรคหัวใจด้วย AI (Decision Tree)</div>', 
            unsafe_allow_html=True)

# ============================================================
# Sidebar - Input Form
# ============================================================
with st.sidebar:
    st.markdown("## 📝 กรอกข้อมูลสุขภาพ")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("🎂 อายุ (ปี)", min_value=20, max_value=100, value=50, step=1)
        sex = st.selectbox("⚧ เพศ", ["ชาย", "หญิง"])
        sex_val = 1 if sex == "ชาย" else 0
        
        chest_pain = st.selectbox(
            "💔 ประเภทอาการเจ็บหน้าอก",
            options=[1, 2, 3, 4],
            format_func=lambda x: {
                1: "Typical Angina",
                2: "Atypical Angina", 
                3: "Non-anginal Pain",
                4: "Asymptomatic"
            }[x]
        )
        
        resting_bp = st.number_input(
            "🩸 ความดันโลหิตขณะพัก (mm Hg)", 
            min_value=80, max_value=220, value=130, step=1
        )
    
    with col2:
        cholesterol = st.number_input(
            "🧪 คอเลสเตอรอล (mg/dl)", 
            min_value=100, max_value=600, value=200, step=1
        )
        
        fasting_bs = st.selectbox(
            "🍬 น้ำตาลในเลือดขณะอดอาหาร > 120 mg/dl?",
            ["ไม่", "ใช่"]
        )
        fasting_bs_val = 1 if fasting_bs == "ใช่" else 0
        
        resting_ecg = st.selectbox(
            "📈 ผล ECG ขณะพัก",
            options=[1, 2, 3],
            format_func=lambda x: {
                1: "Normal",
                2: "ST-T wave abnormality",
                3: "Left ventricular hypertrophy"
            }[x]
        )
        
        max_hr = st.number_input(
            "💓 อัตราการเต้นหัวใจสูงสุด (bpm)", 
            min_value=50, max_value=220, value=140, step=1
        )
    
    exercise_angina = st.selectbox(
        "🏃 มีอาการเจ็บหน้าอกขณะออกกำลังกาย?",
        ["ไม่", "ใช่"]
    )
    exercise_angina_val = 1 if exercise_angina == "ใช่" else 0
    
    col3, col4 = st.columns(2)
    with col3:
        oldpeak = st.number_input(
            "📉 ST Depression (Oldpeak)", 
            min_value=-3.0, max_value=7.0, value=1.0, step=0.1,
            format="%.1f"
        )
    
    with col4:
        st_slope = st.selectbox(
            "📐 ST Slope",
            options=[1, 2, 3],
            format_func=lambda x: {
                1: "Upsloping",
                2: "Flat",
                3: "Downsloping"
            }[x]
        )
    
    st.markdown("---")
    predict_button = st.button("🔮 ทำนายผล", use_container_width=True)

# ============================================================
# Main Content
# ============================================================
if predict_button:
    if model is None:
        st.error("❌ โมเดลยังไม่ถูกโหลด")
    else:
        # สร้าง DataFrame จากข้อมูล input
        input_data = pd.DataFrame({
            'Age': [age],
            'Sex': [sex_val],
            'ChestPainType': [chest_pain],
            'RestingBP': [resting_bp],
            'Cholesterol': [cholesterol],
            'FastingBS': [fasting_bs_val],
            'RestingECG': [resting_ecg],
            'MaxHR': [max_hr],
            'ExerciseAngina': [exercise_angina_val],
            'Oldpeak': [oldpeak],
            'ST_Slope': [st_slope]
        })
        
        # ทำนาย
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        
        st.markdown("---")
        
        # แสดงผลลัพธ์
        if prediction == 1:
            st.markdown(f"""
            <div class="result-box-danger">
                <h1 style="color:white; margin:0;">⚠️ ผลการทำนาย</h1>
                <h2 style="color:white;">มีความเสี่ยงเป็นโรคหัวใจ</h2>
                <h1 style="color:white; font-size:4rem; margin:1rem 0;">
                    {probability[1]*100:.1f}%
                </h1>
                <p style="color:white; font-size:1.2rem;">
                    ความน่าจะเป็นที่จะเป็นโรคหัวใจ
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-box-safe">
                <h1 style="color:white; margin:0;">✅ ผลการทำนาย</h1>
                <h2 style="color:white;">ไม่มีความเสี่ยงเป็นโรคหัวใจ</h2>
                <h1 style="color:white; font-size:4rem; margin:1rem 0;">
                    {probability[0]*100:.1f}%
                </h1>
                <p style="color:white; font-size:1.2rem;">
                    ความน่าจะเป็นที่จะไม่เป็นโรคหัวใจ
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # แสดงรายละเอียดเพิ่มเติม
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.metric("🎯 ความเสี่ยงโรคหัวใจ", f"{probability[1]*100:.1f}%")
        
        with col_b:
            st.metric("💚 ความปลอดภัย", f"{probability[0]*100:.1f}%")
        
        with col_c:
            risk_level = "สูง 🔴" if probability[1] > 0.7 else \
                        "ปานกลาง 🟡" if probability[1] > 0.4 else "ต่ำ 🟢"
            st.metric("⚡ ระดับความเสี่ยง", risk_level)
        
        # Gauge Chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability[1] * 100,
            title={'text': "ความเสี่ยงโรคหัวใจ (%)", 'font': {'size': 24}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 40], 'color': '#a8e6cf'},
                    {'range': [40, 70], 'color': '#ffd3b6'},
                    {'range': [70, 100], 'color': '#ff8b94'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': probability[1] * 100
                }
            }
        ))
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
        
        # แสดงข้อมูล input ที่กรอก
        with st.expander("📋 ดูข้อมูลที่คุณกรอก"):
            st.dataframe(input_data.T.rename(columns={0: 'ค่า'}), 
                        use_container_width=True)

# ============================================================
# Footer
# ============================================================
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:white; padding:1rem;">
    <p>⚕️ <b>คำเตือน:</b> ผลลัพธ์นี้เป็นเพียงการคาดการณ์จากโมเดล Machine Learning 
    ไม่ใช่การวินิจฉัยทางการแพทย์ กรุณาปรึกษาแพทย์ผู้เชี่ยวชาญ</p>
    <p>🛠️ พัฒนาด้วย Python, Scikit-learn และ Streamlit</p>
</div>
""", unsafe_allow_html=True)
# ============================================================
# Heart Disease Prediction Web App - Streamlit
# รันด้วยคำสั่ง: streamlit run app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px

# ตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="🫀 Heart Disease Predictor",
    page_icon="❤️",
    layout="wide"
)

# โหลดโมเดล
@st.cache_resource
def load_model():
    with open('heart_disease_model.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

try:
    model_package = load_model()
    model = model_package['model']
    scaler = model_package['scaler']
    features = model_package['features']
except FileNotFoundError:
    st.error("❌ ไม่พบไฟล์ heart_disease_model.pkl กรุณาวางไฟล์ในโฟลเดอร์เดียวกัน")
    st.stop()

# Header
st.title("🫀 ระบบทำนายความเสี่ยงโรคหัวใจ")
st.markdown("""
<p style='font-size: 18px;'>
ระบบนี้ใช้โมเดล <b>Decision Tree</b> ในการประเมินความเสี่ยงโรคหัวใจ 
จากข้อมูลสุขภาพของคุณ
</p>
""")

st.markdown("---")

# สร้าง Sidebar สำหรับ Input
st.sidebar.header("📝 กรอกข้อมูลสุขภาพ")

# Input Fields
age = st.sidebar.slider("🎂 อายุ (ปี)", 20, 90, 50)
sex = st.sidebar.radio("⚧ เพศ", ["ชาย", "หญิง"], index=0)
chest_pain = st.sidebar.selectbox(
    "💔 ประเภทอาการเจ็บหน้าอก",
    options=[1, 2, 3, 4],
    format_func=lambda x: {
        1: "Typical Angina (เจ็บแน่นแบบ典型)",
        2: "Atypical Angina (เจ็บแน่นแบบไม่典型)",
        3: "Non-Anginal Pain (ไม่ใช่เจ็บแน่น)",
        4: "Asymptomatic (ไม่มีอาการ)"
    }[x]
)
resting_bp = st.sidebar.number_input("🩸 ความดันโลหิตขณะพัก (mm Hg)", 80, 200, 120)
cholesterol = st.sidebar.number_input("🧪 โคเลสเตอรอล (mg/dl)", 100, 600, 200)
fasting_bs = st.sidebar.radio("🍬 น้ำตาลในเลือดขณะอดอาหาร > 120 mg/dl?", 
                              ["ไม่ใช่ (0)", "ใช่ (1)"], index=0)
fasting_bs_val = 1 if "ใช่" in fasting_bs else 0

resting_ecg = st.sidebar.selectbox(
    "📈 ผล ECG ขณะพัก",
    options=[0, 1, 2],
    format_func=lambda x: {
        0: "Normal (ปกติ)",
        1: "ST-T wave abnormality",
        2: "Left ventricular hypertrophy"
    }[x]
)
max_hr = st.sidebar.slider("💓 อัตราการเต้นหัวใจสูงสุด", 60, 210, 140)
exercise_angina = st.sidebar.radio("🏃 มีอาการเจ็บหน้าอกขณะออกกำลังกาย?", 
                                   ["ไม่ใช่ (0)", "ใช่ (1)"], index=0)
exercise_angina_val = 1 if "ใช่" in exercise_angina else 0
oldpeak = st.sidebar.slider("📉 Oldpeak (ST depression)", 0.0, 6.0, 1.0, 0.1)
st_slope = st.sidebar.selectbox(
    "📊 ST Slope",
    options=[1, 2, 3],
    format_func=lambda x: {1: "Upsloping", 2: "Flat", 3: "Downsloping"}[x]
)

# แปลงเพศเป็นตัวเลข
sex_val = 1 if sex == "ชาย" else 0

# สร้าง DataFrame สำหรับทำนาย
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

# ปุ่มทำนาย
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button("🔮 ทำนายผล", use_container_width=True, type="primary")

# แสดงผล
if predict_button:
    # ทำนาย
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    
    st.markdown("---")
    
    # แสดงผลลัพธ์
    if prediction == 1:
        st.error("## ⚠️ ผลทำนาย: มีความเสี่ยงเป็นโรคหัวใจ")
        st.markdown(f"**โอกาสเกิดโรค:** `{probability[1]*100:.2f}%`")
    else:
        st.success("## ✅ ผลทำนาย: ไม่มีความเสี่ยงเป็นโรคหัวใจ")
        st.markdown(f"**โอกาสไม่เกิดโรค:** `{probability[0]*100:.2f}%`")
    
    # แสดง Probability Gauge
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 ความน่าจะเป็น")
        prob_df = pd.DataFrame({
            'สถานะ': ['ไม่เป็นโรค', 'เป็นโรค'],
            'ความน่าจะเป็น (%)': [probability[0]*100, probability[1]*100]
        })
        fig = px.pie(prob_df, values='ความน่าจะเป็น (%)', names='สถานะ',
                     color='สถานะ',
                     color_discrete_map={'ไม่เป็นโรค': '#2ecc71', 'เป็นโรค': '#e74c3c'},
                     hole=0.4)
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📋 ข้อมูลที่คุณกรอก")
        input_display = input_data.copy()
        input_display['Sex'] = sex
        input_display['ChestPainType'] = {1:"Typical Angina", 2:"Atypical Angina", 
                                          3:"Non-Anginal", 4:"Asymptomatic"}[chest_pain]
        input_display['FastingBS'] = fasting_bs_val
        input_display['ExerciseAngina'] = exercise_angina_val
        input_display['RestingECG'] = {0:"Normal", 1:"ST-T abnormality", 
                                       2:"LV hypertrophy"}[resting_ecg]
        input_display['ST_Slope'] = {1:"Upsloping", 2:"Flat", 3:"Downsloping"}[st_slope]
        st.dataframe(input_display.T.rename(columns={0: 'ค่า'}), use_container_width=True)
    
    # Feature Importance
    st.markdown("---")
    st.subheader("🔍 ปัจจัยที่มีผลต่อการทำนายมากที่สุด")
    importances = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=True).tail(7)
    
    fig2 = px.bar(importances, x='Importance', y='Feature', 
                  orientation='h', color='Importance',
                  color_continuous_scale='RdYlGn_r')
    fig2.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
<p>⚕️ <b>คำเตือน:</b> ผลการทำนายนี้เป็นเพียงการประเมินเบื้องต้น 
ไม่สามารถใช้แทนการวินิจฉัยจากแพทย์ได้</p>
<p>🤖 Powered by Decision Tree & Streamlit</p>
</div>
""")
import pipeline
import streamlit as st
import pickle
import pandas as pd
import os

st.set_page_config(page_title="Student Placement Prediction", page_icon="🎓")

BASE_DIR = os.path.dirname(__file__)

model_clf = pickle.load(open(os.path.join(BASE_DIR, "model_classification.pkl"), "rb"))
model_reg = pickle.load(open(os.path.join(BASE_DIR, "model_regression.pkl"), "rb"))

st.title("🎓 Student Placement Prediction")
st.write("Isi data mahasiswa lalu klik Prediksi")
st.divider()

with st.form("form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        ssc = st.number_input("SSC %", 0, 100, 85)
        hsc = st.number_input("HSC %", 0, 100, 83)
        degree = st.number_input("Degree %", 0, 100, 83)
        cgpa = st.number_input("CGPA", 0.0, 10.0, 7.92)
        entrance = st.number_input("Entrance Score", 0, 100, 91)
        tech = st.number_input("Technical Skill", 0, 100,93)

    with col2:
        soft = st.number_input("Soft Skill", 0, 100, 84)
        intern = st.number_input("Internship", 0, 10, 1)
        project = st.number_input("Projects", 0, 10, 1)
        exp = st.number_input("Experience (months)", 0, 60, 10)
        cert = st.number_input("Certifications", 0, 10, 2)
        attend = st.number_input("Attendance %", 0, 100, 81)
        backlog = st.number_input("Backlogs", 0, 10, 2)
        extra = st.selectbox("Extracurricular", ["Yes", "No"])

    submit = st.form_submit_button("Prediksi")

if submit:
    try:
        input_df = pd.DataFrame([{
            "gender": gender,
            "ssc_percentage": ssc,
            "hsc_percentage": hsc,
            "degree_percentage": degree,
            "cgpa": cgpa,
            "entrance_exam_score": entrance,
            "technical_skill_score": tech,
            "soft_skill_score": soft,
            "internship_count": intern,
            "live_projects": project,
            "work_experience_months": exp,
            "certifications": cert,
            "attendance_percentage": attend,
            "backlogs": backlog,
            "extracurricular_activities": extra
        }])

        pred_clf = model_clf.predict(input_df)[0]

        st.divider()
        st.subheader("Hasil Prediksi")

        if pred_clf == 1:
            st.success("✅ Placed — Mahasiswa ini diprediksi akan mendapatkan pekerjaan.")
            pred_reg = model_reg.predict(input_df)[0]
            st.info(f"💰 Estimasi Gaji: **{pred_reg:.2f} LPA**")
        else:
            st.error("❌ Not Placed — Mahasiswa ini diprediksi tidak mendapatkan pekerjaan.")

    except Exception as e:
        st.error("Terjadi error saat melakukan prediksi.")
        st.exception(e)

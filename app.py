import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix
from sklearn.preprocessing import label_binarize

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model dan preprocessor
@st.cache_resource
def load_model():
    model_path = os.path.join(BASE_DIR, 'model', 'rf_model.pkl')
    scaler_path = os.path.join(BASE_DIR, 'model', 'scaler.pkl')
    feature_cols_path = os.path.join(BASE_DIR, 'model', 'feature_cols.pkl')
    label_encoder_path = os.path.join(BASE_DIR, 'model', 'label_encoder.pkl')
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(feature_cols_path, 'rb') as f:
        feature_cols = pickle.load(f)
    with open(label_encoder_path, 'rb') as f:
        le_status = pickle.load(f)
    return model, scaler, feature_cols, le_status

# Load dataset untuk referensi
@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(BASE_DIR, 'students_performance.csv'), sep=';')
    return df

# Set page config
st.set_page_config(
    page_title="Jaya Jaya Institut - Sistem Prediksi Dropout Mahasiswa",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model
model, scaler, feature_cols, le_status = load_model()
data = load_data()

# Title
st.title("🎓 Jaya Jaya Institut - Sistem Prediksi Dropout Mahasiswa")
st.markdown("Aplikasi ini membantu mengidentifikasi mahasiswa yang berisiko dropout sehingga mereka dapat menerima bimbingan khusus.")

# Sidebar
st.sidebar.header("Navigasi")
page = st.sidebar.radio("Pilih Halaman", ["Beranda", "Prediksi Individu", "Analisis Data", "Dasbor"])

# Home Page
if page == "Beranda":
    st.header("🏠 Beranda")
    st.markdown("""
    ### Tentang Aplikasi
    
    Aplikasi ini dibuat untuk membantu Jaya Jaya Institut mengidentifikasi mahasiswa yang berisiko dropout.
    
    ### Fitur Utama
    
    - **Prediksi Risiko Dropout**: Memprediksi apakah seorang mahasiswa berisiko dropout berdasarkan berbagai faktor
    - **Analisis Data**: Memvisualisasikan data dan faktor-faktor yang mempengaruhi dropout
    - **Dasbor**: Memantau performa mahasiswa secara real-time
    
    ### Cara Menggunakan
    
    1. Pilih halaman dari sidebar
    2. Masukkan data mahasiswa untuk prediksi atau lihat analisis data
    3. Gunakan dasbor untuk memantau performa mahasiswa
    
    ### Tentang Model
    
    Model ini menggunakan Random Forest Classifier dengan **akurasi 76.84%** dan **AUC 0.885**.
    
    Faktor utama yang mempengaruhi prediksi:
    - Unit kurikuler (semester 1 dan 2) - disetujui, nilai, evaluasi
    - Nilai masuk (Admission grade)
    - Usia saat pendaftaran
    - Status biaya kuliah
    - Status penerima beasiswa
    """)
    
    # Statistik singkat
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Mahasiswa", len(data))
    
    with col2:
        graduate_count = (data['Status'] == 'Graduate').sum()
        st.metric("Mahasiswa Lulus", graduate_count)
    
    with col3:
        dropout_percentage = (data['Status'] == 'Dropout').sum()
        st.metric("Mahasiswa Dropout", dropout_percentage)
    
    with col4:
        enrolled_percentage = (data['Status'] == 'Enrolled').sum()
        st.metric("Mahasiswa Aktif", enrolled_percentage)

# Individual Prediction Page
elif page == "Prediksi Individu":
    st.header("👤 Prediksi Risiko Dropout Individu")
    st.markdown("Masukkan data mahasiswa untuk memprediksi risiko dropout mereka.")
    
    # Form input
    col1, col2 = st.columns(2)
    
    with col1:
        marital_status = st.selectbox("Status Pernikahan", [1, 2])
        application_mode = st.selectbox("Mode Pendaftaran", list(range(1, 43)))
        application_order = st.selectbox("Urutan Pendaftaran", list(range(1, 8)))
        course = st.selectbox("Program Studi", [171, 9254, 9070, 9773, 8014, 9991, 9500, 9774, 9238, 9670, 9853, 9556, 9085, 9147, 9118, 9130, 9222, 9938, 9056, 9075, 9089, 9108, 9136, 9140, 9170, 9190, 9225, 9257, 9303, 9309, 9313, 9315, 9318, 9325, 9330, 9335, 9340, 9342, 9346, 9348, 9350])
        daytime_evening_attendance = st.selectbox("Kehadiran Siang/Malam", [1, 0])
        previous_qualification = st.selectbox("Kualifikasi Sebelumnya", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42])
    
    with col2:
        nationality = st.selectbox("Kewarganegaraan", list(range(1, 44)))
        mothers_qualification = st.selectbox("Kualifikasi Ibu", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43])
        fathers_qualification = st.selectbox("Kualifikasi Ayah", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43])
        mothers_occupation = st.selectbox("Pekerjaan Ibu", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        fathers_occupation = st.selectbox("Pekerjaan Ayah", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        admission_grade = st.number_input("Nilai Masuk", min_value=0.0, max_value=200.0, value=120.0)
    
    col3, col4 = st.columns(2)
    
    with col3:
        displaced = st.selectbox("Pindah (Displaced)", [1, 0])
        educational_special_needs = st.selectbox("Kebutuhan Khusus Pendidikan", [1, 0])
        debtor = st.selectbox("Memiliki Hutang (Debtor)", [1, 0])
        tuition_fees_up_to_date = st.selectbox("Biaya Kuliah Lunas", [1, 0])
        gender = st.selectbox("Jenis Kelamin", [1, 0])
        scholarship_holder = st.selectbox("Penerima Beasiswa", [1, 0])
    
    with col4:
        age_at_enrollment = st.number_input("Usia saat Pendaftaran", min_value=16, max_value=70, value=20)
        international = st.selectbox("Internasional", [1, 0])
        curricular_units_1st_sem_credited = st.number_input("Sem 1 - Dikreditkan", min_value=0, max_value=20, value=0)
        curricular_units_1st_sem_enrolled = st.number_input("Sem 1 - Diambil", min_value=0, max_value=20, value=6)
        curricular_units_1st_sem_evaluations = st.number_input("Sem 1 - Evaluasi", min_value=0, max_value=20, value=0)
        curricular_units_1st_sem_approved = st.number_input("Sem 1 - Disetujui", min_value=0, max_value=20, value=0)
    
    col5, col6 = st.columns(2)
    
    with col5:
        curricular_units_1st_sem_grade = st.number_input("Sem 1 - Nilai", min_value=0.0, max_value=20.0, value=0.0)
        curricular_units_1st_sem_without_evaluations = st.number_input("Sem 1 - Tanpa Evaluasi", min_value=0, max_value=20, value=0)
        curricular_units_2nd_sem_credited = st.number_input("Sem 2 - Dikreditkan", min_value=0, max_value=20, value=0)
        curricular_units_2nd_sem_enrolled = st.number_input("Sem 2 - Diambil", min_value=0, max_value=20, value=6)
        curricular_units_2nd_sem_evaluations = st.number_input("Sem 2 - Evaluasi", min_value=0, max_value=20, value=0)
        curricular_units_2nd_sem_approved = st.number_input("Sem 2 - Disetujui", min_value=0, max_value=20, value=0)
    
    with col6:
        curricular_units_2nd_sem_grade = st.number_input("Sem 2 - Nilai", min_value=0.0, max_value=20.0, value=0.0)
        curricular_units_2nd_sem_without_evaluations = st.number_input("Sem 2 - Tanpa Evaluasi", min_value=0, max_value=20, value=0)
        unemployment_rate = st.number_input("Tingkat Pengangguran", min_value=0.0, max_value=20.0, value=10.8)
        inflation_rate = st.number_input("Tingkat Inflasi", min_value=-5.0, max_value=10.0, value=1.4)
        gdp = st.number_input("GDP", min_value=-5.0, max_value=10.0, value=1.74)
    
    # Prepare features
    features = {
        'Marital_status_encoded': marital_status - 1,
        'Application_mode_encoded': application_mode - 1,
        'Course_encoded': course - 1,
        'Daytime_evening_attendance_encoded': daytime_evening_attendance,
        'Previous_qualification_encoded': previous_qualification - 1,
        'Nacionality_encoded': nationality - 1,
        'Mothers_qualification_encoded': mothers_qualification - 1,
        'Fathers_qualification_encoded': fathers_qualification - 1,
        'Mothers_occupation_encoded': mothers_occupation,
        'Fathers_occupation_encoded': fathers_occupation,
        'Admission_grade': admission_grade,
        'Displaced_encoded': displaced,
        'Educational_special_needs_encoded': educational_special_needs,
        'Debtor_encoded': debtor,
        'Tuition_fees_up_to_date_encoded': tuition_fees_up_to_date,
        'Gender_encoded': gender,
        'Scholarship_holder_encoded': scholarship_holder,
        'Age_at_enrollment': age_at_enrollment,
        'International_encoded': international,
        'Curricular_units_1st_sem_credited': curricular_units_1st_sem_credited,
        'Curricular_units_1st_sem_enrolled': curricular_units_1st_sem_enrolled,
        'Curricular_units_1st_sem_evaluations': curricular_units_1st_sem_evaluations,
        'Curricular_units_1st_sem_approved': curricular_units_1st_sem_approved,
        'Curricular_units_1st_sem_grade': curricular_units_1st_sem_grade,
        'Curricular_units_1st_sem_without_evaluations': curricular_units_1st_sem_without_evaluations,
        'Curricular_units_2nd_sem_credited': curricular_units_2nd_sem_credited,
        'Curricular_units_2nd_sem_enrolled': curricular_units_2nd_sem_enrolled,
        'Curricular_units_2nd_sem_evaluations': curricular_units_2nd_sem_evaluations,
        'Curricular_units_2nd_sem_approved': curricular_units_2nd_sem_approved,
        'Curricular_units_2nd_sem_grade': curricular_units_2nd_sem_grade,
        'Curricular_units_2nd_sem_without_evaluations': curricular_units_2nd_sem_without_evaluations,
        'Unemployment_rate': unemployment_rate,
        'Inflation_rate': inflation_rate,
        'GDP': gdp
    }
    
    feature_df = pd.DataFrame([features])
    
    # Scale features
    feature_scaled = scaler.transform(feature_df)
    
    # Predict
    if st.button("Prediksi Risiko Dropout"):
        prediction = model.predict(feature_scaled)[0]
        probability = model.predict_proba(feature_scaled)[0]
        
        # Display result
        status_names = ['Dropout', 'Enrolled', 'Graduate']
        if prediction == 0:
            st.error("⚠️ MAHASISWA BERISIKO DROPOUT!")
            st.warning(f"Probabilitas dropout: {probability[0]*100:.1f}%")
            st.info("Rekomendasi: Berikan bimbingan khusus dan intervensi pendidikan segera")
        elif prediction == 1:
            st.warning("⚠️ MAHASISWA AKTIF - PANTAU PERKEMBANGAN")
            st.info(f"Probabilitas aktif: {probability[1]*100:.1f}%")
            st.info("Rekomendasi: Lanjutkan pemantauan dan dukungan rutin")
        else:
            st.success("✅ MAHASISWA KEMUNGKINAN BESAR LULUS")
            st.success(f"Probabilitas lulus: {probability[2]*100:.1f}%")
            st.info("Rekomendasi: Lanjutkan pemantauan dan dorong penyelesaian studi")
        
        # Display feature importance for this prediction
        st.subheader("Faktor yang Mempengaruhi Prediksi")
        feature_importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        st.dataframe(feature_importance, use_container_width=True)

# Data Analysis Page
elif page == "Analisis Data":
    st.header("📊 Analisis Data")
    
    # Overview
    st.subheader("Ringkasan Data")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Mahasiswa", len(data))
    
    with col2:
        st.metric("Rata-rata Usia", f"{data['Age_at_enrollment'].mean():.1f}")
    
    with col3:
        status_counts = data['Status'].value_counts()
        st.metric("Mahasiswa Berdasarkan Status", int(status_counts.sum()))
    
    # Visualizations
    st.subheader("Distribusi Status Mahasiswa")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Pie chart
    status_counts = data['Status'].value_counts()
    axes[0].pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', 
                colors=['#e74c3c', '#3498db', '#2ecc71'], startangle=90)
    axes[0].set_title('Distribusi Status Mahasiswa')
    
    # Bar chart
    status_counts.plot(kind='bar', ax=axes[1], color=['#e74c3c', '#3498db', '#2ecc71'])
    axes[1].set_xlabel('Status')
    axes[1].set_ylabel('Jumlah Mahasiswa')
    axes[1].set_xticklabels(status_counts.index, rotation=0)
    
    st.pyplot(fig)
    
    # Factor analysis
    st.subheader("Analisis Faktor Risiko")
    
    factor = st.selectbox("Pilih Faktor", ['Marital_status', 'Gender', 'Scholarship_holder', 'Displaced', 'Debtor'])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    status_by_factor = data.groupby(factor)['Status'].value_counts(normalize=True).unstack() * 100
    status_by_factor.plot(kind='bar', ax=ax, color=['#e74c3c', '#3498db', '#2ecc71'])
    ax.set_title(f'Persentase Status Mahasiswa Berdasarkan {factor.replace("_", " ").title()}')
    ax.set_xlabel(factor.replace("_", " ").title())
    ax.set_ylabel('Persentase (%)')
    ax.set_ylim(0, 100)
    ax.legend(title='Status')
    
    st.pyplot(fig)
    
    # Admission grade distribution
    st.subheader("Distribusi Nilai Masuk Berdasarkan Status")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    for status in data['Status'].unique():
        subset = data[data['Status'] == status]
        ax.hist(subset['Admission_grade'], alpha=0.5, label=status, bins=30)
    ax.set_xlabel('Nilai Masuk')
    ax.set_ylabel('Frekuensi')
    ax.set_title('Distribusi Nilai Masuk Berdasarkan Status')
    ax.legend()
    
    st.pyplot(fig)

# Dashboard Page
elif page == "Dasbor":
    st.header("📈 Dasbor Pemantauan Mahasiswa")
    
    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Mahasiswa", len(data))
    
    with col2:
        graduate_count = (data['Status'] == 'Graduate').sum()
        st.metric("Lulus", graduate_count)
    
    with col3:
        dropout_count = (data['Status'] == 'Dropout').sum()
        st.metric("Dropout", dropout_count)
    
    with col4:
        enrolled_count = (data['Status'] == 'Enrolled').sum()
        st.metric("Aktif", enrolled_count)
    
    # At risk students
    st.subheader("📋 Mahasiswa Berisiko Dropout")
    
    at_risk = data[data['Status'] == 'Dropout'].copy()
    
    if len(at_risk) > 0:
        # Display top 10 at-risk students
        st.dataframe(
            at_risk[['Marital_status', 'Application_mode', 'Course', 'Gender', 'Age_at_enrollment', 
                    'Admission_grade', 'Scholarship_holder']].head(10),
            use_container_width=True
        )
        
        # Download button
        csv = at_risk.to_csv(index=False)
        st.download_button(
            label="Unduh Daftar Mahasiswa Berisiko",
            data=csv,
            file_name="mahasiswa_berisiko_dropout.csv",
            mime="text/csv"
        )
    else:
        st.success("Tidak ada mahasiswa berisiko dropout.")
    
    # Key insights
    st.subheader("💡 Wawasan Utama")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Faktor Risiko")
        st.markdown("""
        - **Nilai Masuk Rendah**: Mahasiswa dengan nilai masuk lebih rendah memiliki risiko lebih tinggi
        - **Bukan Penerima Beasiswa**: Mahasiswa tanpa beasiswa menghadapi lebih banyak tantangan finansial
        - **Mahasiswa Lebih Tua**: Mahasiswa yang lebih tua mungkin memiliki lebih banyak komitmen eksternal
        - **Beban Kuliah Tinggi**: Mahasiswa dengan terlalu banyak unit kurikuler mungkin kesulitan
        """)
    
    with col2:
        st.markdown("### Rekomendasi")
        st.markdown("""
        1. Berikan dukungan finansial kepada mahasiswa berisiko
        2. Tawarkan program bimbingan dan mentoring akademik
        3. Pantau mahasiswa dengan nilai masuk rendah secara ketat
        4. Dorong partisipasi dalam program beasiswa
        5. Berikan konseling dan bimbingan karir
        6. Buat jaringan dukungan untuk mahasiswa yang lebih tua
        """)

# Custom CSS
st.markdown("""
<style>
    .stMetric [data-testid="stMetricValue"] {
        font-size: 24px;
    }
    .stMetric [data-testid="stMetricLabel"] {
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

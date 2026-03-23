import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model dan preprocessor
@st.cache_resource
def load_model():
    with open(os.path.join(BASE_DIR, 'model', 'rf_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'model', 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'model', 'feature_cols.pkl'), 'rb') as f:
        feature_cols = pickle.load(f)
    return model, scaler, feature_cols

# Load dataset untuk referensi
@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(BASE_DIR, 'students_performance.csv'))
    df['average_score'] = (df['math score'] + df['reading score'] + df['writing score']) / 3
    return df

# Set page config
st.set_page_config(
    page_title="Jaya Jaya Institut - Dropout Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model
model, scaler, feature_cols = load_model()
data = load_data()

# Title
st.title("🎓 Jaya Jaya Institut - Sistem Prediksi Risiko Dropout")
st.markdown("Aplikasi ini membantu mengidentifikasi siswa yang berisiko dropout sehingga dapat diberikan bimbingan khusus.")

# Sidebar
st.sidebar.header("Navigasi")
page = st.sidebar.radio("Pilih Halaman", ["Home", "Prediksi Individu", "Analisis Data", "Dashboard"])

# Home Page
if page == "Home":
    st.header("🏠 Beranda")
    st.markdown("""
    ### Tentang Aplikasi
    
    Aplikasi ini dibuat untuk membantu Jaya Jaya Institut dalam mengidentifikasi siswa yang berisiko dropout.
    
    ### Fitur Utama
    
    - **Prediksi Risiko Dropout**: Memprediksi apakah seorang siswa berisiko dropout berdasarkan berbagai faktor
    - **Analisis Data**: Visualisasi data dan faktor-faktor yang mempengaruhi dropout
    - **Dashboard**: Monitor performa siswa secara real-time
    
    ### Cara Menggunakan
    
    1. Pilih halaman dari sidebar
    2. Masukkan data siswa untuk prediksi atau lihat analisis data
    3. Gunakan dashboard untuk monitoring performa siswa
    
    ### Tentang Model
    
    Model ini menggunakan Random Forest Classifier dengan akurasi **85.50%** dan AUC **0.923**.
    
    Faktor-faktor utama yang mempengaruhi prediksi:
    - Nilai matematika, membaca, dan menulis
    - Jenis program makan siang (lunch)
    - Keterlibatan dalam kursus persiapan ujian
    - Tingkat pendidikan orang tua
    """)
    
    # Statistik singkat
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Siswa", len(data))
    
    with col2:
        dropout_count = (data['average_score'] < (data['average_score'].median() - 15)).sum()
        st.metric("Siswa Berisiko Dropout", dropout_count)
    
    with col3:
        dropout_percentage = (dropout_count / len(data)) * 100
        st.metric("Persentase Risiko", f"{dropout_percentage:.1f}%")
    
    with col4:
        avg_score = data['average_score'].mean()
        st.metric("Rata-rata Nilai", f"{avg_score:.1f}")

# Individual Prediction Page
elif page == "Prediksi Individu":
    st.header("👤 Prediksi Risiko Dropout Individu")
    st.markdown("Masukkan data siswa untuk memprediksi risiko dropout mereka.")
    
    # Form input
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Jenis Kelamin", ["male", "female"])
        race = st.selectbox("Kelompok Etnis/Ras", ["group A", "group B", "group C", "group D", "group E"])
        parental_edu = st.selectbox("Tingkat Pendidikan Orang Tua", [
            "bachelor's degree", "some college", "master's degree", 
            "associate's degree", "high school", "some high school"
        ])
    
    with col2:
        lunch = st.selectbox("Program Makan Siang", ["standard", "free/reduced"])
        test_prep = st.selectbox("Kursus Persiapan Ujian", ["none", "completed"])
        
        math_score = st.number_input("Nilai Matematika", min_value=0, max_value=100, value=70)
        reading_score = st.number_input("Nilai Membaca", min_value=0, max_value=100, value=70)
        writing_score = st.number_input("Nilai Menulis", min_value=0, max_value=100, value=70)
    
    # Prepare features - use the actual column names from the model
    features = {
        'gender_encoded': 1 if gender == 'female' else 0,
        'parental level of education_encoded': ['bachelor\'s degree', 'some college', 'master\'s degree', 
                                                'associate\'s degree', 'high school', 'some high school'].index(parental_edu),
        'lunch_encoded': 1 if lunch == 'standard' else 0,
        'test preparation course_encoded': 1 if test_prep == 'completed' else 0,
        'math score': math_score,
        'reading score': reading_score,
        'writing score': writing_score,
        'race_group B': 1 if race == 'group B' else 0,
        'race_group C': 1 if race == 'group C' else 0,
        'race_group D': 1 if race == 'group D' else 0,
        'race_group E': 1 if race == 'group E' else 0
    }
    
    feature_df = pd.DataFrame([features])
    
    # Scale features
    feature_scaled = scaler.transform(feature_df)
    
    # Predict
    if st.button("Prediksi Risiko Dropout"):
        prediction = model.predict(feature_scaled)[0]
        probability = model.predict_proba(feature_scaled)[0]
        
        # Display result
        if prediction == 1:
            st.error("⚠️ SISWA BERISIKO DROPOUT!")
            st.warning(f"Probabilitas risiko dropout: {probability[1]*100:.1f}%")
            st.info("Rekomendasi: Segera berikan bimbingan khusus dan intervensi pendidikan")
        else:
            st.success("✅ SISWA TIDAK BERISIKO DROPOUT")
            st.success(f"Probabilitas risiko dropout: {probability[1]*100:.1f}%")
            st.info("Rekomendasi: Lanjutkan pendampingan rutin")
        
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
    
    # Add calculated column
    data['dropout_risk'] = (data['average_score'] < (data['average_score'].median() - 15)).astype(int)
    
    # Overview
    st.subheader("Overview Data")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Siswa", len(data))
    
    with col2:
        st.metric("Rata-rata Nilai", f"{data['average_score'].mean():.1f}")
    
    with col3:
        st.metric("Siswa Berisiko Dropout", data['dropout_risk'].sum())
    
    # Visualizations
    st.subheader("Distribusi Risiko Dropout")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Pie chart
    dropout_counts = data['dropout_risk'].value_counts()
    axes[0].pie(dropout_counts, labels=['Tidak Berisiko', 'Berisiko'], autopct='%1.1f%%', 
                colors=['#2ecc71', '#e74c3c'], startangle=90)
    axes[0].set_title('Distribusi Risiko Dropout')
    
    # Bar chart
    dropout_counts.plot(kind='bar', ax=axes[1], color=['#2ecc71', '#e74c3c'])
    axes[1].set_xlabel('Risiko Dropout')
    axes[1].set_ylabel('Jumlah Siswa')
    axes[1].set_xticklabels(['Tidak Berisiko (0)', 'Berisiko (1)'], rotation=0)
    
    st.pyplot(fig)
    
    # Factor analysis
    st.subheader("Analisis Faktor Risiko")
    
    factor = st.selectbox("Pilih Faktor", ['gender', 'lunch', 'test preparation course'])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    dropout_by_factor = data.groupby(factor)['dropout_risk'].mean() * 100
    dropout_by_factor.plot(kind='bar', ax=ax, color=['#3498db', '#e74c3c'])
    ax.set_title(f'Persentase Risiko Dropout Berdasarkan {factor.replace("_", " ").title()}')
    ax.set_xlabel(factor.replace("_", " ").title())
    ax.set_ylabel('Persentase Risiko Dropout (%)')
    ax.set_ylim(0, 100)
    
    st.pyplot(fig)
    
    # Score distribution
    st.subheader("Distribusi Nilai")
    
    subject = st.selectbox("Pilih Mata Pelajaran", ['math score', 'reading score', 'writing score'])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=data, x='dropout_risk', y=subject, ax=ax)
    ax.set_title(f'Nilai {subject.split()[0].capitalize()} Berdasarkan Risiko Dropout')
    ax.set_xlabel('Risiko Dropout (0=Tidak, 1=Ya)')
    ax.set_ylabel('Nilai')
    
    st.pyplot(fig)

# Dashboard Page
elif page == "Dashboard":
    st.header("📈 Dashboard Monitoring Siswa")
    
    # Add calculated columns
    data['dropout_risk'] = (data['average_score'] < (data['average_score'].median() - 15)).astype(int)
    
    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Siswa", len(data))
    
    with col2:
        st.metric("Siswa Berisiko", data['dropout_risk'].sum())
    
    with col3:
        st.metric("Persentase Risiko", f"{data['dropout_risk'].mean()*100:.1f}%")
    
    with col4:
        st.metric("Rata-rata Nilai", f"{data['average_score'].mean():.1f}")
    
    # Siswa berisiko
    st.subheader("📋 Daftar Siswa Berisiko Dropout")
    
    at_risk = data[data['dropout_risk'] == 1].copy()
    
    if len(at_risk) > 0:
        # Display top 10 at-risk students
        st.dataframe(
            at_risk[['gender', 'race/ethnicity', 'parental level of education', 
                    'lunch', 'math score', 'reading score', 'writing score', 'average_score']].head(10),
            use_container_width=True
        )
        
        # Download button
        csv = at_risk.to_csv(index=False)
        st.download_button(
            label="Download Daftar Siswa Berisiko",
            data=csv,
            file_name="siswa_berisiko_dropout.csv",
            mime="text/csv"
        )
    else:
        st.success("Tidak ada siswa yang berisiko dropout pada threshold saat ini.")
    
    # Key insights
    st.subheader("💡 Insight Penting")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Faktor Risiko Utama")
        st.markdown("""
        - **Nilai Rendah**: Siswa dengan nilai di bawah median - 15 berisiko tinggi
        - **Lunch Program**: Siswa dengan free/reduced lunch memiliki risiko lebih tinggi
        - **Tidak Ikut Kursus**: Siswa yang tidak mengikuti test preparation course
        - **Pendidikan Orang Tua**: Siswa dengan orang tua berpendidikan rendah
        """)
    
    with col2:
        st.markdown("### Rekomendasi")
        st.markdown("""
        1. Berikan bimbingan khusus kepada siswa berisiko
        2. Dorong partisipasi dalam program tutoring
        3. Libatkan orang tua dalam proses pendampingan
        4. Monitoring nilai secara berkala
        5. Perkuat program support untuk siswa kurang mampu
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

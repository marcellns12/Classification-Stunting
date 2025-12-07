import streamlit as st
import pandas as pd
import numpy as np
import joblib
# PENTING: Impor semua kelas yang digunakan di Pipeline agar loading berhasil
from imblearn.pipeline import ImbPipeline 
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import xgboost as xgb 

# --------------------------------------------------------------------------
# 1. FUNGSI UNTUK MEMUAT MODEL (Hanya dimuat sekali)
# --------------------------------------------------------------------------
@st.cache_resource
def load_pipeline():
    try:
        # Memuat seluruh pipeline
        pipeline = joblib.load('stacking_pipeline_final.pkl')
        return pipeline
    except FileNotFoundError:
        st.error("File model 'stacking_pipeline_final.pkl' tidak ditemukan. Pastikan file berada di direktori yang sama.")
        return None

pipeline = load_pipeline()

# --------------------------------------------------------------------------
# 2. DEFINISI ANTARMUKA STREAMLIT
# --------------------------------------------------------------------------

st.title("Aplikasi Prediksi Status Gizi (Stunting) - Stacking Model")
st.markdown("Masukkan data balita. Model akan memprediksi Status Gizi (0: Normal, 1: Stunting)")

if pipeline is not None:
    
    st.header("Input Data Balita")

    # Input Fitur Dasar
    col1, col2 = st.columns(2)
    with col1:
        umur = st.number_input("1. Umur (bulan)", min_value=1, max_value=72, value=12)
        berat_kg = st.number_input("2. Berat Badan (kg)", min_value=1.0, max_value=30.0, value=7.0, format="%.2f")

    with col2:
        jenis_kelamin = st.selectbox("3. Jenis Kelamin", ['Laki-laki', 'Perempuan'])
    
    # --------------------------------------------------------------------------
    # 3. FEATURE ENGINEERING (HARUS SAMA PERSIS DENGAN SAAT PELATIHAN)
    # --------------------------------------------------------------------------
    # Ini harus dilakukan PADA DATA BARU sebelum diprediksi
    rasio = berat_kg / umur 
    berat_kuadrat = berat_kg ** 2
    
    st.caption(f"Fitur Turunan Otomatis: Rasio_Berat_Umur ({rasio:.4f}), Berat_Kuadrat ({berat_kuadrat:.2f})")
    
    # --------------------------------------------------------------------------
    # 4. PREDIKSI
    # --------------------------------------------------------------------------
    
    if st.button("Prediksi Status Gizi"):
        # Susun input ke DataFrame dengan NAMA KOLOM dan URUTAN yang sama persis
        data_input = {
            'Umur (bulan)': [umur],
            'Berat Badan (kg)': [berat_kg],
            'Rasio_Berat_Umur': [rasio], # Fitur Engineered
            'Berat_Kuadrat': [berat_kuadrat], # Fitur Engineered
            'Jenis Kelamin': [jenis_kelamin]
        }
        input_df = pd.DataFrame(data_input)
        
        try:
            # Panggil predict pada pipeline. Preprocessor akan berjalan otomatis!
            prediction = pipeline.predict(input_df)
            
            # Interpretasi Hasil
            status = "Stunting" if prediction[0] == 1 else "Normal"
            
            st.subheader("Hasil Prediksi:")
            if status == "Stunting":
                st.error(f"Status Gizi Balita Diprediksi: **{status}** ❌")
                st.warning("Diperlukan perhatian dan intervensi gizi lebih lanjut!")
            else:
                st.success(f"Status Gizi Balita Diprediksi: **{status}** ✅")
                st.info("Kondisi Gizi Balita Terlihat Baik.")
                
        except Exception as e:
            st.error(f"Terjadi kesalahan saat prediksi: {e}")

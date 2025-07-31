import streamlit as st
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# ========== Dummy Pipeline ==========
# NOTE: Ganti bagian ini dengan pipeline asli dari notebook jika ingin akurasi asli
# Untuk demo, kita buat pipeline sederhana dengan fitur numerik dan kategorikal

# Fitur
num_features = ['Umur', 'HariSejakPembelian', 'KunjunganWeb']
cat_features = ['TingkatPenghasilan', 'TingkatKampanye']

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, num_features),
    ('cat', categorical_transformer, cat_features)
])

# Model dummy
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

# Data dummy untuk fit awal (karena belum load model asli)
df_dummy = pd.DataFrame({
    'Umur': [25, 28, 42],
    'HariSejakPembelian': [10, 20, 15],
    'KunjunganWeb': [5, 2, 3],
    'TingkatPenghasilan': ['Tinggi', 'Sedang', 'Rendah'],
    'TingkatKampanye': ['Produk Baru', 'Diskon', 'Email'],
    'Response': [1, 0, 0]
})
model.fit(df_dummy.drop('Response', axis=1), df_dummy['Response'])

# ========== UI Streamlit ==========
st.title("Prediksi Respon Kampanye Pemasaran")

with st.form("form"):
    col1, col2 = st.columns(2)

    with col1:
        umur = st.number_input("Umur", min_value=1, max_value=100, step=1)
        hari = st.number_input("Hari Sejak Pembelian Terakhir", min_value=0)

    with col2:
        penghasilan = st.selectbox("Tingkat Penghasilan", ["Rendah", "Sedang", "Tinggi"])
        kunjungan = st.number_input("Jumlah Kunjungan Web Bulan Lalu", min_value=0)

    kampanye = st.selectbox("Tingkat Kampanye", ["Email", "Diskon", "Produk Baru"])
    submitted = st.form_submit_button("Prediksi Respon")

if submitted:
    df_input = pd.DataFrame.from_dict({
        'Umur': [umur],
        'HariSejakPembelian': [hari],
        'KunjunganWeb': [kunjungan],
        'TingkatPenghasilan': [penghasilan],
        'TingkatKampanye': [kampanye]
    })
    hasil = model.predict_proba(df_input)[0][1]
    pred_label = "Yes" if hasil >= 0.5 else "No"
    st.success(f"Prediksi: {int(hasil * 100)}% ({pred_label})")

# ========== Tabel Hasil Dummy ==========
st.subheader("Prediksi Baru-Baru Ini")
data = {
    "Pelanggan": [
        "Umur: 25, Penghasilan: Tinggi",
        "Umur: 28, Penghasilan: Sedang",
        "Umur: 42, Penghasilan: Rendah"
    ],
    "Kampanye": ["Produk Baru", "Diskon", "Email"],
    "Prediksi": ["78% (Yes)", "32% (No)", "45% (No)"],
    "Actual": ["Yes", "No", "No"]
}
st.table(pd.DataFrame(data))

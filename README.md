# Twitch Streamer Analytics & Partner Prediction

Aplikasi Streamlit untuk Tubes Kelompok 2 (Top Streamers on Twitch). Aplikasi ini punya tiga halaman: dashboard EDA, pencarian status channel dari data asli, dan prediksi status Partner menggunakan model Random Forest yang sudah di-tuning (hasil dari notebook `tubes_kelompok2.py`).

## Isi folder

- `app.py` — kode utama aplikasi Streamlit.
- `best_twitch_partner_model.joblib` — model Random Forest terbaik (hasil GridSearchCV).
- `twitch_scaler.joblib` — StandardScaler yang dipakai saat training.
- `feature_columns.joblib` — daftar nama kolom fitur setelah encoding, dipakai supaya urutan fitur saat prediksi konsisten dengan saat training.
- `requirements.txt` — daftar library yang dibutuhkan.
- `train_model.py` — script yang dipakai untuk menghasilkan ulang ketiga file `.joblib` di atas jika perlu retrain.

File model wajib ada di folder yang sama dengan `app.py` agar halaman "Prediksi Status Partner" berfungsi.

## Menjalankan secara lokal

1. Install dependency: `pip install -r requirements.txt`
2. Jalankan aplikasi: `streamlit run app.py`
3. Browser akan terbuka otomatis ke `http://localhost:8501`

## Deploy ke Streamlit Community Cloud

1. Buat repository GitHub baru, lalu upload semua file di folder ini (termasuk ketiga file `.joblib`, jangan diabaikan lewat `.gitignore`).
2. Buka [share.streamlit.io](https://share.streamlit.io), login dengan akun GitHub.
3. Klik "New app", pilih repository dan branch yang sudah dibuat, lalu set "Main file path" ke `app.py`.
4. Klik "Deploy". Streamlit Cloud akan otomatis install dari `requirements.txt` dan menjalankan aplikasi.

## Catatan tentang model

Dataset aslinya sangat tidak seimbang (978 channel Partnered vs hanya 22 yang tidak), sehingga walaupun sudah ditangani dengan SMOTE dan `class_weight='balanced'`, model masih jauh lebih akurat saat memprediksi kelas "Partner" dibanding kelas "Bukan Partner". Anggap hasil prediksi sebagai estimasi kasar, bukan keputusan final.

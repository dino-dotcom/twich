import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# KONFIGURASI HALAMAN
# =========================================================
st.set_page_config(
    page_title="Twitch Streamer Analytics",
    page_icon="🎮",
    layout="wide"
)

DATA_URL = "https://raw.githubusercontent.com/DachsteinSilalahi/Dataset_kelompok2/refs/heads/main/twitchdata-update.csv"

MODEL_PATH = "best_twitch_partner_model.joblib"
SCALER_PATH = "twitch_scaler.joblib"
FEATURES_PATH = "feature_columns.joblib"


# =========================================================
# LOAD DATA & MODEL (CACHED)
# =========================================================
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL)
    return df


@st.cache_resource
def load_model_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_columns = joblib.load(FEATURES_PATH)
    return model, scaler, feature_columns


df = load_data()

try:
    model, scaler, feature_columns = load_model_artifacts()
    model_loaded = True
except FileNotFoundError:
    model, scaler, feature_columns = None, None, None
    model_loaded = False


# =========================================================
# SIDEBAR NAVIGASI
# =========================================================
st.sidebar.title("🎮 Twitch Analytics")
st.sidebar.caption("Tubes Kelompok 2 — Top Streamers on Twitch")

page = st.sidebar.radio(
    "Pilih Halaman",
    ["📊 Dashboard EDA", "🔍 Cek Status Channel", "🤖 Prediksi Status Partner"]
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Anggota Kelompok 2**\n"
    "- Falisha Salsabila\n"
    "- Aldrin Rivaldo Kumakauw\n"
    "- Muhammad Iqbal\n"
    "- Dino Kurniawan\n"
    "- Dachstein Silalahi"
)

if not model_loaded:
    st.sidebar.warning(
        "File model (.joblib) belum ditemukan di folder yang sama dengan "
        "app.py. Halaman prediksi tidak akan berfungsi sampai file "
        "`best_twitch_partner_model.joblib`, `twitch_scaler.joblib`, dan "
        "`feature_columns.joblib` tersedia."
    )


# =========================================================
# HALAMAN 1: DASHBOARD EDA
# =========================================================
if page == "📊 Dashboard EDA":
    st.title("📊 Dashboard Analisis Streamer Twitch")
    st.write(
        "Eksplorasi data dari dataset Top Streamers on Twitch "
        f"({df.shape[0]} channel, {df.shape[1]} kolom)."
    )

    # --- Metrik ringkas ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Channel", f"{df.shape[0]:,}")
    col2.metric("Channel Partnered", f"{int(df['Partnered'].sum()):,}")
    col3.metric("Jumlah Bahasa", df['Language'].nunique())
    col4.metric("Rata-rata Followers", f"{df['Followers'].mean():,.0f}")

    st.markdown("---")

    # --- Top N berdasarkan metrik pilihan ---
    st.subheader("Top 10 Streamer Berdasarkan Metrik")
    metric_options = [
        "Watch time(Minutes)", "Stream time(minutes)", "Peak viewers",
        "Average viewers", "Followers", "Followers gained", "Views gained"
    ]
    selected_metric = st.selectbox("Pilih metrik", metric_options)

    top10 = df.sort_values(by=selected_metric, ascending=False).head(10)

    chart_col, table_col = st.columns([2, 1])
    with chart_col:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(
            data=top10, x=selected_metric, y="Channel",
            hue="Channel", legend=False, palette="viridis", ax=ax
        )
        ax.set_title(f"Top 10 Channel — {selected_metric}")
        st.pyplot(fig)
    with table_col:
        st.dataframe(
            top10[["Channel", selected_metric]].reset_index(drop=True),
            use_container_width=True
        )

    st.markdown("---")

    # --- Distribusi Partnered, Mature, dan Bahasa ---
    dist_col1, dist_col2 = st.columns(2)

    with dist_col1:
        st.subheader("Distribusi Status Partnered")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.countplot(data=df, x="Partnered", hue="Partnered", legend=False, palette="magma", ax=ax)
        ax.set_xlabel("Partnered")
        ax.set_ylabel("Jumlah Channel")
        st.pyplot(fig)

    with dist_col2:
        st.subheader("Distribusi Konten Mature")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.countplot(data=df, x="Mature", hue="Mature", legend=False, palette="magma", ax=ax)
        ax.set_xlabel("Mature")
        ax.set_ylabel("Jumlah Channel")
        st.pyplot(fig)

    st.subheader("5 Bahasa dengan Channel Terbanyak")
    top5_lang = df["Language"].value_counts().head(5)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=top5_lang.values, y=top5_lang.index, hue=top5_lang.index, legend=False, palette="crest", ax=ax)
    ax.set_xlabel("Jumlah Channel")
    ax.set_ylabel("Bahasa")
    st.pyplot(fig)

    st.markdown("---")

    st.subheader("Korelasi Followers vs Followers Gained")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x="Followers", y="Followers gained", hue="Partnered", alpha=0.7, ax=ax)
    st.pyplot(fig)
    corr = df["Followers"].corr(df["Followers gained"])
    st.caption(f"Korelasi Pearson antara Followers dan Followers Gained: **{corr:.4f}**")


# =========================================================
# HALAMAN 2: CEK STATUS CHANNEL (DATA ASLI)
# =========================================================
elif page == "🔍 Cek Status Channel":
    st.title("🔍 Cek Status Channel")
    st.write(
        "Pilih channel dari dataset untuk melihat status Partnered, "
        "Mature, dan metrik lengkapnya."
    )

    selected_channel = st.selectbox(
        "Pilih Channel Streamer",
        options=["Pilih Streamer"] + sorted(df["Channel"].tolist())
    )

    if selected_channel != "Pilih Streamer":
        streamer_data = df[df["Channel"] == selected_channel].iloc[0]

        st.subheader(f"Detail untuk Channel: {selected_channel}")

        col1, col2 = st.columns(2)
        with col1:
            if streamer_data["Partnered"]:
                st.success("Status Partnered: Partner ✅")
            else:
                st.warning("Status Partnered: Bukan Partner ❌")
        with col2:
            if streamer_data["Mature"]:
                st.error("Status Mature: Konten Dewasa 🔒")
            else:
                st.info("Status Mature: Konten Umum 🟢")

        st.markdown("---")
        st.subheader("Metrik Streamer")

        m1, m2, m3 = st.columns(3)
        m1.metric("Watch time (menit)", f"{streamer_data['Watch time(Minutes)']:,}")
        m2.metric("Stream time (menit)", f"{streamer_data['Stream time(minutes)']:,}")
        m3.metric("Peak viewers", f"{streamer_data['Peak viewers']:,}")

        m4, m5, m6 = st.columns(3)
        m4.metric("Average viewers", f"{streamer_data['Average viewers']:,}")
        m5.metric("Followers", f"{streamer_data['Followers']:,}")
        m6.metric("Followers gained", f"{streamer_data['Followers gained']:,}")

        st.metric("Views gained", f"{streamer_data['Views gained']:,}")
        st.write(f"**Bahasa:** {streamer_data['Language']}")


# =========================================================
# HALAMAN 3: PREDIKSI STATUS PARTNER (MODEL ML)
# =========================================================
elif page == "🤖 Prediksi Status Partner":
    st.title("🤖 Prediksi Status Partner")
    st.write(
        "Masukkan metrik sebuah channel (baru/hipotetis) untuk "
        "memprediksi kemungkinan statusnya menjadi **Partner** Twitch, "
        "menggunakan model Random Forest terbaik dari hasil tuning."
    )

    if not model_loaded:
        st.error(
            "Model belum tersedia. Pastikan file "
            "`best_twitch_partner_model.joblib`, `twitch_scaler.joblib`, "
            "dan `feature_columns.joblib` berada di folder yang sama "
            "dengan app.py."
        )
    else:
        language_options = sorted([
            col.replace("Language_", "")
            for col in feature_columns if col.startswith("Language_")
        ])
        # Tambahkan kembali kategori baseline yang di-drop saat one-hot encoding
        all_languages = sorted(set(language_options + ["Arabic"]))

        with st.form("prediction_form"):
            st.subheader("Input Metrik Channel")

            c1, c2 = st.columns(2)
            with c1:
                watch_time = st.number_input(
                    "Watch time (Minutes)", min_value=0, value=200_000_000, step=1_000_000
                )
                stream_time = st.number_input(
                    "Stream time (minutes)", min_value=0, value=100_000, step=1_000
                )
                peak_viewers = st.number_input(
                    "Peak viewers", min_value=0, value=10_000, step=100
                )
                avg_viewers = st.number_input(
                    "Average viewers", min_value=0, value=2_000, step=50
                )
            with c2:
                followers = st.number_input(
                    "Followers", min_value=0, value=500_000, step=1_000
                )
                followers_gained = st.number_input(
                    "Followers gained", min_value=0, value=100_000, step=1_000
                )
                views_gained = st.number_input(
                    "Views gained", min_value=0, value=5_000_000, step=10_000
                )
                language = st.selectbox("Bahasa Utama", all_languages, index=all_languages.index("English"))
                mature = st.checkbox("Konten Mature?")

            submitted = st.form_submit_button("Prediksi Sekarang")

        if submitted:
            input_row = {col: 0 for col in feature_columns}
            input_row["Watch time(Minutes)"] = watch_time
            input_row["Stream time(minutes)"] = stream_time
            input_row["Peak viewers"] = peak_viewers
            input_row["Average viewers"] = avg_viewers
            input_row["Followers"] = followers
            input_row["Followers gained"] = followers_gained
            input_row["Views gained"] = views_gained

            lang_col = f"Language_{language}"
            if lang_col in input_row:
                input_row[lang_col] = 1
            if mature and "Mature_True" in input_row:
                input_row["Mature_True"] = 1

            input_df = pd.DataFrame([input_row])[feature_columns]
            input_scaled = scaler.transform(input_df)

            prediction = model.predict(input_scaled)[0]
            proba = model.predict_proba(input_scaled)[0]
            classes = list(model.classes_)
            prob_partner = proba[classes.index(True)] if True in classes else proba[1]

            st.markdown("---")
            st.subheader("Hasil Prediksi")

            if prediction:
                st.success(f"✅ Channel ini diprediksi akan menjadi **PARTNER**")
            else:
                st.warning(f"❌ Channel ini diprediksi **BELUM** menjadi Partner")

            st.metric("Probabilitas Menjadi Partner", f"{prob_partner * 100:.2f}%")
            st.progress(float(prob_partner))

            st.caption(
                "Catatan: prediksi ini bersifat estimasi berdasarkan pola "
                "data historis dan tidak menjamin keputusan resmi dari Twitch."
            )

        st.markdown("---")
        st.subheader("Fitur Paling Berpengaruh (Top 10)")
        importances = pd.DataFrame({
            "Feature": feature_columns,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False).head(10)

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=importances, x="Importance", y="Feature", hue="Feature", legend=False, palette="viridis", ax=ax)
        st.pyplot(fig)

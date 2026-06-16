import streamlit as st
import pandas as pd
import plotly.express as px

# ======================================

# KONFIGURASI

# ======================================

st.set_page_config(
page_title="Dashboard Analisis Twitch",
page_icon="🎮",
layout="wide"
)

# ======================================

# LOAD DATA

# =====================================
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/DachsteinSilalahi/Dataset_kelompok2/main/twitchdata-update.csv"
    return pd.read_csv(url)

df = load_data()

# ======================================

# SIDEBAR

# ======================================

menu = st.sidebar.radio(
"Pilih Menu",
[
"Home",
"Dataset",
"EDA",
"Visualisasi",
"Korelasi",
"Machine Learning",
"Kesimpulan"
]
)

# ======================================

# HOME

# ======================================

if menu == "Home":

```
st.title("🎮 Dashboard Analisis Twitch Streamers")

col1,col2,col3,col4 = st.columns(4)

col1.metric(
    "Jumlah Streamer",
    len(df)
)

col2.metric(
    "Jumlah Kolom",
    df.shape[1]
)

col3.metric(
    "Jumlah Bahasa",
    df["Language"].nunique()
)

col4.metric(
    "Partnered",
    df["Partnered"].sum()
)

st.markdown("---")

st.write("""
Dashboard ini dibuat untuk analisis data Top Streamers on Twitch.
Dashboard mencakup:
- Exploratory Data Analysis
- Visualisasi Data
- Analisis Korelasi
- Machine Learning Classification
""")
```

# ======================================

# DATASET

# ======================================

elif menu == "Dataset":

```
st.title("📂 Dataset")

st.dataframe(df)

st.subheader("Tipe Data")

st.write(df.dtypes)
```

# ======================================

# EDA

# ======================================

elif menu == "EDA":

```
st.title("📊 Exploratory Data Analysis")

st.subheader("Statistik Deskriptif")

st.dataframe(df.describe())

st.subheader("Missing Values")

st.dataframe(
    pd.DataFrame(
        df.isnull().sum(),
        columns=["Missing Value"]
    )
)
```

# ======================================

# VISUALISASI

# ======================================

elif menu == "Visualisasi":

```
st.title("📈 Visualisasi")

tab1, tab2, tab3, tab4 = st.tabs([
    "Followers",
    "Watch Time",
    "Peak Viewers",
    "Language"
])

with tab1:

    top10 = df.nlargest(10,"Followers")

    fig = px.bar(
        top10,
        x="Channel",
        y="Followers",
        title="Top 10 Followers"
    )

    st.plotly_chart(fig)

with tab2:

    top10 = df.nlargest(
        10,
        "Watch time(Minutes)"
    )

    fig = px.bar(
        top10,
        x="Channel",
        y="Watch time(Minutes)",
        title="Top 10 Watch Time"
    )

    st.plotly_chart(fig)

with tab3:

    top10 = df.nlargest(
        10,
        "Peak viewers"
    )

    fig = px.bar(
        top10,
        x="Channel",
        y="Peak viewers",
        title="Top 10 Peak Viewers"
    )

    st.plotly_chart(fig)

with tab4:

    lang = df["Language"].value_counts().head(10)

    fig = px.pie(
        values=lang.values,
        names=lang.index,
        title="Top 10 Languages"
    )

    st.plotly_chart(fig)
```

# ======================================

# KORELASI

# ======================================

elif menu == "Korelasi":

```
st.title("🔗 Analisis Korelasi")

pilihan = st.selectbox(
    "Pilih Korelasi",
    [
        "Followers vs Followers gained",
        "Watch time vs Followers gained",
        "Peak viewers vs Followers gained",
        "Average viewers vs Followers gained",
        "Views gained vs Followers gained",
        "Stream time vs Followers gained"
    ]
)

mapping = {
    "Followers vs Followers gained":
        ("Followers","Followers gained"),

    "Watch time vs Followers gained":
        ("Watch time(Minutes)","Followers gained"),

    "Peak viewers vs Followers gained":
        ("Peak viewers","Followers gained"),

    "Average viewers vs Followers gained":
        ("Average viewers","Followers gained"),

    "Views gained vs Followers gained":
        ("Views gained","Followers gained"),

    "Stream time vs Followers gained":
        ("Stream time(minutes)","Followers gained")
}

x_col,y_col = mapping[pilihan]

corr = df[x_col].corr(df[y_col])

st.metric(
    "Nilai Korelasi",
    round(corr,4)
)

fig = px.scatter(
    df,
    x=x_col,
    y=y_col,
    trendline="ols"
)

st.plotly_chart(fig)
```

# ======================================

# MACHINE LEARNING

# ======================================

elif menu == "Machine Learning":

```
st.title("🤖 Machine Learning")

hasil = pd.DataFrame({
    "Model":[
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ],
    "Keterangan":[
        "Baseline Model",
        "Rule Based",
        "Best Model"
    ]
})

st.dataframe(hasil)

st.success(
    "Random Forest dipilih sebagai model terbaik berdasarkan hasil evaluasi kelompok."
)
```

# ======================================

# KESIMPULAN

# ======================================

elif menu == "Kesimpulan":

```
st.title("📝 Kesimpulan")

st.write("""
1. Dataset memiliki 1000 streamer Twitch.

2. Bahasa Inggris merupakan bahasa yang paling dominan.

3. Mayoritas streamer telah berstatus Partnered.

4. Followers memiliki korelasi kuat dengan Followers Gained.

5. Watch Time memiliki hubungan positif terhadap pertumbuhan followers.

6. Random Forest menjadi model terbaik untuk klasifikasi status Partnered.
""")
```

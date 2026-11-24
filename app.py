
import streamlit as st
import pandas as pd
import plotly.express as px
import json

from datetime import datetime, timezone
from pymongo import MongoClient
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from ucimlrepo import fetch_ucirepo

st.set_page_config(
    page_title="AI Data Analysis",
    page_icon="📊",
    layout="wide"
)

st.title("📊 AI Data Analysis Dashboard")
st.write(
    "Aplikasi analisis data menggunakan Streamlit, "
    "Machine Learning, visualisasi, dan MongoDB Atlas."
)

# =====================================
# 1. KONEKSI MONGODB ATLAS
# =====================================
@st.cache_resource
def connect_mongodb():
    uri = st.secrets["MONGO_URI"]
    client = MongoClient(
        uri,
        serverSelectionTimeoutMS=10000
    )
    client.admin.command("ping")
    return client


client = None
db = None
mongo_status = "Belum terhubung"

try:
    client = connect_mongodb()
    db = client["tugas_ai_db"]
    mongo_status = "Terhubung"
except Exception as e:
    st.warning(
        "MongoDB belum terhubung. Periksa secrets.toml "
        "dan konfigurasi MongoDB Atlas."
    )
    with st.expander("Detail error koneksi"):
        st.code(str(e))

jumlah_tersimpan = 0

if db is not None:
    try:
        jumlah_tersimpan = db["hasil_clustering"].count_documents({})
    except Exception:
        jumlah_tersimpan = 0

# =====================================
# 2. RINGKASAN DASHBOARD
# =====================================
col1, col2, col3, col4 = st.columns(4)

col1.metric("Status Aplikasi", "Aktif")
col2.metric("Dataset", "UCI Iris")
col3.metric("Database", mongo_status)
col4.metric("Hasil Tersimpan", jumlah_tersimpan)

if client is not None:
    st.success("Berhasil terhubung ke MongoDB Atlas!")

st.divider()

# =====================================
# 3. MEMUAT DATASET
# =====================================
st.header("1. Dataset UCI Machine Learning")

if st.button("Muat Dataset Iris", type="primary"):
    try:
        with st.spinner("Mengambil dataset dari UCI..."):
            iris = fetch_ucirepo(id=53)

            X = iris.data.features.copy()
            X = X.apply(pd.to_numeric, errors="coerce")
            X = X.dropna()

            st.session_state["iris_X"] = X
            st.session_state.pop("hasil_clustering", None)
            st.session_state.pop("silhouette", None)

        st.success("Dataset Iris berhasil dimuat!")

    except Exception as e:
        st.error("Dataset gagal dimuat.")
        st.code(str(e))

if "iris_X" in st.session_state:
    X = st.session_state["iris_X"]

    a, b, c = st.columns(3)
    a.metric("Jumlah Data", len(X))
    b.metric("Jumlah Fitur", X.shape[1])
    c.metric("Data Kosong", int(X.isna().sum().sum()))

    with st.expander("Lihat preview dataset", expanded=True):
        st.dataframe(X.head(10), use_container_width=True)

    with st.expander("Statistik deskriptif"):
        st.dataframe(X.describe(), use_container_width=True)

    st.subheader("Korelasi Antarfitur")
    corr = X.corr()
    fig_corr = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        title="Matriks Korelasi Fitur Iris",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1
    )
    st.plotly_chart(fig_corr, use_container_width=True)

else:
    st.info("Klik 'Muat Dataset Iris' untuk memulai analisis.")

    
# =====================================
# EVALUASI JUMLAH CLUSTER TERBAIK
# =====================================
st.divider()
st.header("3. Evaluasi Jumlah Cluster Terbaik")

if "iris_X" in st.session_state:
    X_eval = st.session_state["iris_X"]

    if st.button("Analisis Jumlah Cluster Terbaik"):
        with st.spinner("Menghitung evaluasi cluster..."):
            scaler_eval = StandardScaler()
            X_scaled_eval = scaler_eval.fit_transform(X_eval)

            hasil_evaluasi = []

            for k in range(2, 7):
                model_eval = KMeans(
                    n_clusters=k,
                    random_state=42,
                    n_init=10
                )

                labels_eval = model_eval.fit_predict(X_scaled_eval)

                hasil_evaluasi.append({
                    "Jumlah Cluster": k,
                    "Inertia": float(model_eval.inertia_),
                    "Silhouette Score": float(
                        silhouette_score(X_scaled_eval, labels_eval)
                    )
                })

            evaluasi_df = pd.DataFrame(hasil_evaluasi)

            st.session_state["evaluasi_cluster"] = evaluasi_df

    if "evaluasi_cluster" in st.session_state:
        evaluasi_df = st.session_state["evaluasi_cluster"]

        st.subheader("Perbandingan Kualitas Cluster")
        st.dataframe(evaluasi_df, use_container_width=True)

        fig_elbow = px.line(
            evaluasi_df,
            x="Jumlah Cluster",
            y="Inertia",
            markers=True,
            title="Elbow Method"
        )
        st.plotly_chart(fig_elbow, use_container_width=True)

        fig_silhouette = px.line(
            evaluasi_df,
            x="Jumlah Cluster",
            y="Silhouette Score",
            markers=True,
            title="Perbandingan Silhouette Score"
        )
        st.plotly_chart(fig_silhouette, use_container_width=True)

        terbaik = evaluasi_df.loc[
            evaluasi_df["Silhouette Score"].idxmax()
        ]

        st.info(
            f"Jumlah cluster dengan Silhouette Score tertinggi "
            f"pada pengujian ini adalah "
            f"{int(terbaik['Jumlah Cluster'])}, dengan skor "
            f"{terbaik['Silhouette Score']:.3f}."
        )
else:
    st.info("Muat dataset Iris terlebih dahulu.")


# =====================================
# 4. K-MEANS CLUSTERING
# =====================================
st.divider()
st.header("2. Analisis K-Means Clustering")

if "iris_X" in st.session_state:
    X = st.session_state["iris_X"]

    jumlah_cluster = st.slider(
        "Pilih jumlah cluster",
        min_value=2,
        max_value=6,
        value=3
    )

    if st.button("Jalankan K-Means", type="primary"):
        try:
            with st.spinner("Melakukan clustering dan evaluasi..."):
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                model = KMeans(
                    n_clusters=jumlah_cluster,
                    random_state=42,
                    n_init=10
                )

                labels = model.fit_predict(X_scaled)

                hasil = X.copy()
                hasil["Cluster"] = labels.astype(str)

                score = silhouette_score(X_scaled, labels)

                st.session_state["hasil_clustering"] = hasil
                st.session_state["jumlah_cluster"] = jumlah_cluster
                st.session_state["silhouette"] = float(score)

            st.success("K-Means berhasil dijalankan!")

        except Exception as e:
            st.error("Analisis clustering gagal.")
            st.code(str(e))

if "hasil_clustering" in st.session_state:
    hasil = st.session_state["hasil_clustering"]
    fitur = list(st.session_state["iris_X"].columns)

    st.subheader("Evaluasi Hasil Clustering")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Jumlah Cluster",
        st.session_state["jumlah_cluster"]
    )
    col2.metric("Jumlah Data", len(hasil))
    col3.metric(
        "Silhouette Score",
        f'{st.session_state["silhouette"]:.3f}'
    )

    st.caption(
        "Silhouette Score berkisar dari -1 sampai 1. "
        "Nilai yang lebih tinggi umumnya menunjukkan "
        "cluster yang lebih terpisah dengan baik."
    )

    # Pilihan fitur untuk grafik
    st.subheader("Visualisasi Cluster")

    col_x, col_y = st.columns(2)

    with col_x:
        fitur_x = st.selectbox(
            "Fitur sumbu X",
            fitur,
            index=0
        )

    with col_y:
        fitur_y = st.selectbox(
            "Fitur sumbu Y",
            fitur,
            index=min(1, len(fitur) - 1)
        )

    fig = px.scatter(
        hasil,
        x=fitur_x,
        y=fitur_y,
        color="Cluster",
        title="Sebaran Data Berdasarkan Cluster",
        hover_data=fitur,
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Distribusi anggota cluster
    st.subheader("Distribusi Anggota Cluster")

    jumlah_data = (
        hasil["Cluster"]
        .value_counts()
        .sort_index()
        .rename_axis("Cluster")
        .reset_index(name="Jumlah Data")
    )

    fig_bar = px.bar(
        jumlah_data,
        x="Cluster",
        y="Jumlah Data",
        color="Cluster",
        text="Jumlah Data",
        title="Jumlah Anggota Setiap Cluster",
        template="plotly_white"
    )

    st.plotly_chart(fig_bar, use_container_width=True)
    st.dataframe(jumlah_data, use_container_width=True)

    # Profil karakteristik cluster
    st.subheader("Profil Rata-rata Setiap Cluster")

    profil = hasil.groupby("Cluster")[fitur].mean().round(3)
    st.dataframe(profil, use_container_width=True)

    # Tabel dan unduhan
    st.subheader("Tabel Hasil Clustering")

    st.dataframe(hasil, use_container_width=True)

    csv_data = hasil.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Unduh Hasil Clustering (CSV)",
        data=csv_data,
        file_name="hasil_clustering_iris.csv",
        mime="text/csv"
    )

    # =====================================
    # 5. SIMPAN KE MONGODB
    # =====================================
    st.divider()
    st.subheader("3. Penyimpanan MongoDB Atlas")

    if client is not None:
        if st.button("Simpan Hasil ke MongoDB"):
            try:
                collection = db["hasil_clustering"]

                # Mengubah nilai pandas/numpy menjadi
                # tipe data Python yang kompatibel BSON.
                records = json.loads(
                    hasil.to_json(orient="records")
                )

                dokumen = {
                    "dataset": "UCI Iris",
                    "algoritma": "K-Means",
                    "jumlah_cluster": int(
                        st.session_state["jumlah_cluster"]
                    ),
                    "silhouette_score": float(
                        st.session_state["silhouette"]
                    ),
                    "jumlah_data": len(hasil),
                    "fitur": list(hasil.columns),
                    "hasil": records,
                    "waktu": datetime.now(timezone.utc)
                }

                result = collection.insert_one(dokumen)

                st.success(
                    "Hasil berhasil disimpan ke MongoDB Atlas!"
                )
                st.write("ID dokumen:", str(result.inserted_id))
                st.rerun()

            except Exception as e:
                st.error("Gagal menyimpan hasil ke MongoDB.")
                st.code(str(e))
    else:
        st.info("Hubungkan MongoDB untuk menyimpan hasil analisis.")

# =====================================
# 6. INFORMASI PROYEK
# =====================================
st.divider()
st.header("Informasi Proyek")

st.write("**Framework:** Streamlit")
st.write("**Bahasa pemrograman:** Python")
st.write("**Dataset:** UCI Iris")
st.write("**Database:** MongoDB Atlas")
st.write("**Algoritma:** K-Means Clustering")
st.write("**Evaluasi:** Silhouette Score")
st.write("**Visualisasi:** Plotly")
st.write("**Ekspor data:** CSV")

st.caption(
    "Terakhir dijalankan: "
    f"{datetime.now().strftime('%d-%m-%Y %H:%M:%S')}"
)

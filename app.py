# app.py
import streamlit as st
from src.utils import load_and_preprocess
from src.pca_module import run_pca, plot_pca_variance, plot_pca_2d, plot_pca_3d
from src.tsne_module import run_tsne, plot_tsne_2d, plot_tsne_3d
from src.evaluation import run_clustering_comparison, plot_silhouette_scores

st.set_page_config(page_title="Grupo 6: PCA & t-SNE", layout="wide")
st.title("Reducción de Dimensionalidad: PCA y t-SNE")

# --- Carga de datos ---
st.header("1. Cargar Dataset")
uploaded_file = st.file_uploader("Sube un archivo CSV", type=["csv"])
if uploaded_file is None:
    st.info("Usando dataset de ejemplo integrado.")
    X = None  # Se reemplazará por un dataset de ejemplo en utils si es necesario
else:
    X = load_and_preprocess(uploaded_file)

if X is None:
    st.stop()

# --- PCA ---
st.header("2. Análisis con PCA")
n_components_pca = st.slider("Componentes principales (PCA)", 2, min(10, X.shape[1]), 2)
pca_result = run_pca(X, n_components_pca)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Varianza explicada acumulada")
    fig_var = plot_pca_variance(pca_result['explained_variance_ratio'])
    st.pyplot(fig_var)
with col2:
    if n_components_pca >= 3:
        st.subheader("Visualización 3D (PCA)")
        fig_pca_3d = plot_pca_3d(pca_result['transformed'])
        st.plotly_chart(fig_pca_3d)
    else:
        st.subheader("Visualización 2D (PCA)")
        fig_pca_2d = plot_pca_2d(pca_result['transformed'])
        st.pyplot(fig_pca_2d)

# --- t-SNE ---
st.header("3. Análisis con t-SNE")
perplexity = st.slider("Perplejidad (t-SNE)", 5, 50, 30)
n_iter = st.slider("Iteraciones (t-SNE)", 250, 1000, 500)
tsne_result = run_tsne(X, perplexity=perplexity, n_iter=n_iter)

st.subheader("Visualización 2D (t-SNE)")
fig_tsne = plot_tsne_2d(tsne_result['transformed'])
st.plotly_chart(fig_tsne)

# --- Comparación con clustering ---
st.header("4. Comparación con Clustering")
use_pca_for_clustering = st.checkbox("Usar embeddings de PCA para clustering", value=True)
use_tsne_for_clustering = st.checkbox("Usar embeddings de t-SNE para clustering", value=True)

if use_pca_for_clustering or use_tsne_for_clustering:
    clustering_results = run_clustering_comparison(
        X_original=X,
        X_pca=pca_result['transformed'] if use_pca_for_clustering else None,
        X_tsne=tsne_result['transformed'] if use_tsne_for_clustering else None
    )
    st.subheader("Puntuaciones de Silhouette")
    fig_sil = plot_silhouette_scores(clustering_results)
    st.pyplot(fig_sil)
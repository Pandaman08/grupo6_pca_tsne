# app.py
import streamlit as st
import pandas as pd
import numpy as np
from src.utils import load_and_preprocess
from src.pca_module import run_pca, plot_pca_variance, plot_pca_2d, plot_pca_3d
from src.tsne_module import run_tsne, plot_tsne_2d, plot_tsne_3d 
from src.evaluation import (
    run_clustering_comparison, 
    plot_silhouette_scores,
    validate_clustering_pipeline
)

st.set_page_config(page_title="Grupo 6: PCA & t-SNE", layout="wide")
st.title("Reducción de Dimensionalidad: PCA y t-SNE")

# --- Carga de datos ---
st.header("1. Cargar Dataset")
uploaded_file = st.file_uploader("Sube un archivo CSV", type=["csv"])

if uploaded_file is None:
    st.info("Usando dataset de ejemplo (Iris).")
    X, y = load_and_preprocess(None)  # ✅ Recibir ambos
else:
    X, y = load_and_preprocess(uploaded_file)  # ✅ Recibir ambos

if X is None:
    st.stop()

st.success(f"Dataset: {X.shape[0]} muestras, {X.shape[1]} características")

# --- PCA ---
st.header("2. Análisis con PCA")
n_components_pca = st.slider("Componentes principales (PCA)", 2, min(10, X.shape[1]), 2)
pca_result = run_pca(X, n_components_pca)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Varianza explicada")
    fig_var = plot_pca_variance(pca_result['explained_variance_ratio'])
    st.pyplot(fig_var)
    
    # Tabla de varianza
    variance_df = pd.DataFrame({
        'PC': [f'PC{i+1}' for i in range(len(pca_result['explained_variance_ratio']))],
        'Varianza %': pca_result['explained_variance_ratio'] * 100,
        'Acumulada %': np.cumsum(pca_result['explained_variance_ratio']) * 100
    })
    st.dataframe(variance_df)

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

show_tsne_3d = st.checkbox("Mostrar t-SNE en 3D")
if show_tsne_3d:
    tsne_3d = run_tsne(X, perplexity=perplexity, n_iter=n_iter, n_components=3)
    fig_tsne_3d = plot_tsne_3d(tsne_3d['transformed'])
    st.plotly_chart(fig_tsne_3d)
    
# --- Comparación con clustering ---
st.header("4. Comparación con Clustering")
use_pca = st.checkbox("Usar PCA para clustering", value=True)
use_tsne = st.checkbox("Usar t-SNE para clustering", value=True)

if use_pca or use_tsne:
    clustering_results = run_clustering_comparison(
        X_original=X,
        X_pca=pca_result['transformed'] if use_pca else None,
        X_tsne=tsne_result['transformed'] if use_tsne else None
    )
    st.subheader("Silhouette Scores")
    fig_sil = plot_silhouette_scores(clustering_results)
    st.pyplot(fig_sil)

# --- Validación Cruzada ---
st.header("5. Validación Cruzada")
if st.button("Ejecutar validación cruzada"):
    cv_results = validate_clustering_pipeline(X, n_components=n_components_pca)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Score Promedio", f"{cv_results['mean']:.3f}")
    with col2:
        st.metric("Desviación Estándar", f"{cv_results['std']:.3f}")
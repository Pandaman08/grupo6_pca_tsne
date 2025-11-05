# pca_module.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px

def run_pca(X, n_components):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    return {
        'transformed': X_pca,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'components': pca.components_
    }

def plot_pca_variance(explained_variance_ratio):
    cumsum = np.cumsum(explained_variance_ratio)
    fig, ax = plt.subplots()
    ax.plot(range(1, len(cumsum)+1), cumsum, marker='o')
    ax.set_xlabel('Número de componentes')
    ax.set_ylabel('Varianza explicada acumulada')
    ax.grid(True)
    return fig

def plot_pca_2d(X_pca):
    df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    fig = px.scatter(
        df, x='PC1', y='PC2',
        title='Proyección PCA (2D)',
        opacity=0.8,
        width=700, height=500
    )
    fig.update_traces(marker=dict(size=6))
    fig.update_layout(
        xaxis_title='PC1',
        yaxis_title='PC2',
        template='plotly_white'
    )
    return fig

def plot_pca_3d(X_pca):
    import plotly.graph_objects as go
    fig = go.Figure(data=[go.Scatter3d(
        x=X_pca[:, 0], y=X_pca[:, 1], z=X_pca[:, 2],
        mode='markers', marker=dict(size=4, opacity=0.8)
    )])
    fig.update_layout(scene=dict(
        xaxis_title='PC1',
        yaxis_title='PC2',
        zaxis_title='PC3'
    ))
    return fig

def plot_pca_loadings(components, feature_names, n_components=2):
    loadings = components[:n_components].T 
    df_loadings = pd.DataFrame(
        loadings,
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=feature_names
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(df_loadings, annot=True, cmap='RdBu_r', center=0, ax=ax)
    ax.set_title('Loadings: Contribución de cada variable a los componentes')
    plt.tight_layout()
    return fig


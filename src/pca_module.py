import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def run_pca(X, n_components):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    return {
        'transformed': X_pca,
        'explained_variance_ratio': pca.explained_variance_ratio_
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
    fig, ax = plt.subplots()
    ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
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
import numpy as np
import plotly.graph_objects as go
from sklearn.manifold import TSNE

def run_tsne(X, perplexity=30, n_iter=500, n_components=2):
    """
    Ejecuta t-SNE. n_components puede ser 2 o 3.
    Devuelve {'transformed': X_tsne} donde X_tsne.shape = (n_samples, n_components)
    """
    X_arr = np.asarray(X)
    if X_arr.ndim != 2:
        raise ValueError("X debe ser una matriz 2D de forma (n_samples, n_features).")
    if n_components not in (2, 3):
        raise ValueError("n_components debe ser 2 o 3.")
    tsne = TSNE(n_components=n_components, perplexity=perplexity, n_iter=n_iter, random_state=42)
    X_tsne = tsne.fit_transform(X_arr)
    return {'transformed': X_tsne}

def plot_tsne_2d(X_tsne, labels=None):
    fig = go.Figure(data=go.Scatter(
        x=X_tsne[:, 0],
        y=X_tsne[:, 1],
        mode='markers',
        marker=dict(size=5, opacity=0.8, color=labels)
    ))
    fig.update_layout(title="t-SNE 2D", xaxis_title='t-SNE 1', yaxis_title='t-SNE 2')
    return fig

def plot_tsne_3d(X_tsne, labels=None):
    # espera X_tsne con 3 columnas
    fig = go.Figure(data=go.Scatter3d(
        x=X_tsne[:, 0],
        y=X_tsne[:, 1],
        z=X_tsne[:, 2],
        mode='markers',
        marker=dict(size=4, opacity=0.8, color=labels)
    ))
    fig.update_layout(title="t-SNE 3D", scene=dict(
        xaxis_title='t-SNE 1', yaxis_title='t-SNE 2', zaxis_title='t-SNE 3'
    ))
    return fig

import plotly.graph_objects as go
from sklearn.manifold import TSNE

def run_tsne(X, perplexity=30, n_iter=500):
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    X_tsne = tsne.fit_transform(X)
    return {'transformed': X_tsne}

def plot_tsne_2d(X_tsne):
    fig = go.Figure(data=go.Scatter(
        x=X_tsne[:, 0], y=X_tsne[:, 1],
        mode='markers', marker=dict(size=4, opacity=0.8)
    ))
    fig.update_layout(xaxis_title='t-SNE 1', yaxis_title='t-SNE 2')
    return fig
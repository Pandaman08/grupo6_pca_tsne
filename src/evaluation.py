import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

def run_clustering_comparison(X_original, X_pca=None, X_tsne=None):
    results = {}
    n_clusters = min(10, X_original.shape[0] // 2)
    n_clusters = max(2, n_clusters)

    # Original
    kmeans_orig = KMeans(n_clusters=n_clusters, random_state=42).fit(X_original)
    score_orig = silhouette_score(X_original, kmeans_orig.labels_)
    results['Original'] = score_orig

    # PCA
    if X_pca is not None:
        kmeans_pca = KMeans(n_clusters=n_clusters, random_state=42).fit(X_pca)
        score_pca = silhouette_score(X_pca, kmeans_pca.labels_)
        results['PCA'] = score_pca

    # t-SNE
    if X_tsne is not None:
        kmeans_tsne = KMeans(n_clusters=n_clusters, random_state=42).fit(X_tsne)
        score_tsne = silhouette_score(X_tsne, kmeans_tsne.labels_)
        results['t-SNE'] = score_tsne

    return results

def plot_silhouette_scores(scores_dict):
    methods = list(scores_dict.keys())
    scores = list(scores_dict.values())
    fig, ax = plt.subplots()
    ax.bar(methods, scores, color=['gray', 'blue', 'red'][:len(methods)])
    ax.set_ylabel('Silhouette Score')
    ax.set_ylim(0, 1)
    return fig
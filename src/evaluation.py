# evaluation.py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

def run_clustering_comparison(X_original, X_pca=None, X_tsne=None):
    results = {}
    n_clusters = min(10, X_original.shape[0] // 2)
    n_clusters = max(2, n_clusters)

    # Original
    kmeans_orig = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans_orig.fit(X_original)
    results['Original'] = silhouette_score(X_original, kmeans_orig.labels_)

    # PCA
    if X_pca is not None:
        kmeans_pca = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans_pca.fit(X_pca)
        results['PCA'] = silhouette_score(X_pca, kmeans_pca.labels_)

    # t-SNE
    if X_tsne is not None:
        kmeans_tsne = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans_tsne.fit(X_tsne)
        results['t-SNE'] = silhouette_score(X_tsne, kmeans_tsne.labels_)

    return results

def plot_silhouette_scores(scores_dict):
    methods = list(scores_dict.keys())
    scores = list(scores_dict.values())
    
    fig, ax = plt.subplots()
    ax.bar(methods, scores, color=['gray', 'blue', 'red'][:len(methods)])
    ax.set_ylabel('Silhouette Score')
    ax.set_ylim(0, 1)
    ax.set_title('Comparación de Silhouette Scores')
    plt.tight_layout()
    return fig

def validate_clustering_pipeline(X, n_components=2, n_clusters=3):
    """Validación cruzada calculando silhouette en cada fold"""
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in kfold.split(X):
        X_train = X[train_idx]
        X_val = X[val_idx]
        
        # PCA
        pca = PCA(n_components=n_components, random_state=42)
        X_train_pca = pca.fit_transform(X_train)
        X_val_pca = pca.transform(X_val)
        
        # KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(X_train_pca)
        val_labels = kmeans.predict(X_val_pca)
        
        # Silhouette score
        score = silhouette_score(X_val_pca, val_labels)
        scores.append(score)
    
    scores = np.array(scores)
    return {
        'scores': scores,
        'mean': scores.mean(),
        'std': scores.std()
    }  # ✅ SIN COMA